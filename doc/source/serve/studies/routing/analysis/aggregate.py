"""
Aggregate metrics from raw experiment logs using Ray Data for parallel processing.

Supports reading from both local files and S3.

Computes:
- Latency percentiles (p50, p90, p95, p99, max)
- Routing delay percentiles (time from Parent send to Child receive)
- Throughput metrics (RPS, goodput, error rate)
- Fairness metrics (CV, Jain's Index, Min/Max ratio)

Performance: Uses Ray Data for parallel reading and Ray tasks for parallel aggregation.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import ray


@dataclass
class FairnessMetrics:
    """Fairness metrics for request distribution across replicas."""

    coefficient_of_variation: float  # CV = σ / μ (lower is better, 0 = perfect)
    jains_index: float  # Range [1/n, 1], 1 = perfectly fair
    min_max_ratio: float  # Range [0, 1], 1 = perfectly fair
    replica_count: int  # Number of unique replicas that received requests
    expected_replicas: int  # Expected number of replicas from config
    unique_replica_pct: float  # Percentage of expected replicas used (0-100)
    min_requests: int
    max_requests: int
    mean_requests: float
    std_requests: float


@dataclass
class LatencyMetrics:
    """Latency percentiles in milliseconds."""

    p50: float
    p90: float
    p95: float
    p99: float
    max: float
    min: float
    mean: float


@dataclass
class ThroughputMetrics:
    """Throughput metrics."""

    num_concurrent: int
    achieved_rps: float
    goodput: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float


@dataclass
class UtilizationMetrics:
    """Replica utilization metrics.
    
    Measures how much of a replica's processing capacity was actually used.
    Low utilization + high routing delay indicates router-side bottlenecks.
    """

    mean: float  # Mean utilization across replicas (0-1)
    min: float  # Minimum utilization (identifies starved replicas)
    max: float  # Maximum utilization
    std: float  # Standard deviation
    replica_count: int  # Number of replicas measured


@dataclass
class AggregatedMetrics:
    """Complete aggregated metrics for an experiment run."""

    run_id: str
    config: dict
    latency: LatencyMetrics
    # Forward path timing
    client_to_parent_delay: LatencyMetrics  # Client → Parent
    parent_to_child_delay: LatencyMetrics  # Parent → Child
    # Return path timing
    child_to_parent_delay: LatencyMetrics  # Child → Parent
    parent_to_client_delay: LatencyMetrics  # Parent → Client
    # Work time
    simulated_latency: LatencyMetrics  # Actual work time in Child (sleep duration)
    throughput: ThroughputMetrics
    child_fairness: FairnessMetrics  # Fairness across child replicas
    parent_fairness: FairnessMetrics  # Fairness across parent replicas
    child_utilization: UtilizationMetrics  # Utilization of child replicas


# ============================================================================
# S3 Helper Functions
# ============================================================================

def parse_s3_path(s3_path: str) -> tuple:
    """
    Parse S3 path into bucket and key.
    
    Args:
        s3_path: S3 path (e.g., "s3://bucket/prefix/file.csv")
    
    Returns:
        Tuple of (bucket, key)
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""
    return bucket, key


def read_json_from_s3(s3_path: str) -> dict:
    """
    Read JSON file from S3.
    
    Args:
        s3_path: S3 path to JSON file.
    
    Returns:
        Parsed JSON as dictionary.
    """
    import boto3
    
    bucket, key = parse_s3_path(s3_path)
    s3_client = boto3.client("s3")
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    
    return json.loads(content)


def list_s3_files(s3_prefix: str, suffix: str = "") -> List[str]:
    """
    List files in S3 under a prefix.
    
    Args:
        s3_prefix: S3 path prefix (e.g., "s3://bucket/prefix/")
        suffix: Optional suffix to filter files (e.g., ".csv")
    
    Returns:
        List of full S3 paths.
    """
    import boto3
    
    bucket, prefix = parse_s3_path(s3_prefix)
    s3_client = boto3.client("s3")
    
    files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not suffix or key.endswith(suffix):
                files.append(f"s3://{bucket}/{key}")
    
    return sorted(files)


def write_json_to_s3(data: dict, s3_path: str) -> None:
    """
    Write JSON data to S3.
    
    Args:
        data: Dictionary to write as JSON.
        s3_path: S3 path for output file.
    """
    import boto3
    
    bucket, key = parse_s3_path(s3_path)
    s3_client = boto3.client("s3")
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    
    print(f"Wrote JSON to {s3_path}")


def write_csv_to_s3(df: pd.DataFrame, s3_path: str) -> None:
    """
    Write pandas DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame to write.
        s3_path: S3 path for output file.
    """
    import boto3
    
    bucket, key = parse_s3_path(s3_path)
    s3_client = boto3.client("s3")
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    
    print(f"Wrote CSV to {s3_path}")


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_fairness_metrics(
    request_counts: List[int], expected_replicas: int = 0
) -> FairnessMetrics:
    """
    Compute fairness metrics from per-replica request counts.

    Args:
        request_counts: List of request counts per replica (only replicas that received requests).
        expected_replicas: Expected number of replicas from config (0 if unknown).

    Returns:
        FairnessMetrics dataclass with all computed metrics.
    """
    if not request_counts or all(c == 0 for c in request_counts):
        return FairnessMetrics(
            coefficient_of_variation=float("inf"),
            jains_index=0.0,
            min_max_ratio=0.0,
            replica_count=len(request_counts),
            expected_replicas=expected_replicas,
            unique_replica_pct=0.0,
            min_requests=0,
            max_requests=0,
            mean_requests=0.0,
            std_requests=0.0,
        )

    n = len(request_counts)  # Number of unique replicas that received requests
    counts = np.array(request_counts, dtype=float)

    mean = np.mean(counts)
    std = np.std(counts)
    min_val = int(np.min(counts))
    max_val = int(np.max(counts))

    # Coefficient of Variation: CV = σ / μ
    # Lower is better, 0 = perfect fairness
    cv = std / mean if mean > 0 else float("inf")

    # Jain's Fairness Index: J = (Σxᵢ)² / (n × Σxᵢ²)
    # Range [1/n, 1], 1 = perfectly fair
    sum_x = np.sum(counts)
    sum_x_squared = np.sum(counts**2)
    jains = (sum_x**2) / (n * sum_x_squared) if sum_x_squared > 0 else 0.0

    # Min/Max Ratio
    # Range [0, 1], 1 = perfectly fair
    min_max = min_val / max_val if max_val > 0 else 0.0

    # Unique replica percentage: what % of expected replicas received requests
    # 100% means all replicas were utilized
    unique_pct = (n / expected_replicas * 100) if expected_replicas > 0 else 100.0

    return FairnessMetrics(
        coefficient_of_variation=cv,
        jains_index=jains,
        min_max_ratio=min_max,
        replica_count=n,
        expected_replicas=expected_replicas,
        unique_replica_pct=unique_pct,
        min_requests=min_val,
        max_requests=max_val,
        mean_requests=float(mean),
        std_requests=float(std),
    )


def compute_utilization_metrics(
    df: pd.DataFrame,
    replica_id_col: str,
    work_time_col: str,
    duration_s: float,
    max_ongoing_requests: int = 5,
) -> UtilizationMetrics:
    """
    Compute replica utilization metrics.
    
    Utilization = total_work_time / (duration × max_ongoing_requests)
    
    Args:
        df: DataFrame with request results.
        replica_id_col: Column name for replica ID.
        work_time_col: Column name for work time in milliseconds.
        duration_s: Duration of the experiment in seconds.
        max_ongoing_requests: Max concurrent requests per replica.
    
    Returns:
        UtilizationMetrics dataclass.
    """
    if replica_id_col not in df.columns or work_time_col not in df.columns:
        return UtilizationMetrics(
            mean=0.0, min=0.0, max=0.0, std=0.0, replica_count=0
        )
    
    # Filter out NaN values
    valid = df[[replica_id_col, work_time_col]].dropna()
    if len(valid) == 0:
        return UtilizationMetrics(
            mean=0.0, min=0.0, max=0.0, std=0.0, replica_count=0
        )
    
    # Sum work time per replica (in ms)
    work_per_replica = valid.groupby(replica_id_col)[work_time_col].sum()
    
    # Max capacity per replica = duration_s * 1000ms * max_ongoing_requests
    max_capacity_ms = duration_s * 1000 * max_ongoing_requests
    
    # Utilization = work_time / max_capacity
    utilizations = work_per_replica / max_capacity_ms
    
    # Clip to [0, 1] in case of measurement errors
    utilizations = utilizations.clip(0, 1)
    
    return UtilizationMetrics(
        mean=float(utilizations.mean()),
        min=float(utilizations.min()),
        max=float(utilizations.max()),
        std=float(utilizations.std()),
        replica_count=len(utilizations),
    )


def compute_latency_metrics(latencies_ms: List[float]) -> LatencyMetrics:
    """
    Compute latency percentiles from raw latency data.

    Args:
        latencies_ms: List of latencies in milliseconds.

    Returns:
        LatencyMetrics dataclass with all percentiles.
    """
    if not latencies_ms:
        return LatencyMetrics(
            p50=0.0, p90=0.0, p95=0.0, p99=0.0,
            max=0.0, min=0.0, mean=0.0,
        )

    arr = np.array(latencies_ms)
    return LatencyMetrics(
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(np.max(arr)),
        min=float(np.min(arr)),
        mean=float(np.mean(arr)),
    )


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_from_dataframe(
    df: pd.DataFrame,
    run_id: str,
    config: dict,
) -> AggregatedMetrics:
    """
    Aggregate metrics from a pandas DataFrame of request results.
    
    Args:
        df: DataFrame with request results.
        run_id: Run identifier.
        config: Experiment configuration dictionary.
    
    Returns:
        AggregatedMetrics for this data.
    """
    # Filter to successful requests for latency metrics
    successful = df[df["success"] == True]

    # Compute latency metrics
    latency = compute_latency_metrics(successful["latency_ms"].tolist())

    # Compute client → parent delay metrics (Client send → Parent receive)
    # Filter out any negative values (clock skew) and NaN
    if "client_to_parent_delay_ms" in successful.columns:
        client_to_parent_delays = successful["client_to_parent_delay_ms"].dropna()
        client_to_parent_delays = client_to_parent_delays[client_to_parent_delays >= 0].tolist()
    else:
        client_to_parent_delays = []
    client_to_parent_delay = compute_latency_metrics(client_to_parent_delays)

    # Compute parent → child delay metrics (Parent send → Child receive)
    # Filter out any negative values (clock skew) and NaN
    if "parent_to_child_delay_ms" in successful.columns:
        parent_to_child_delays = successful["parent_to_child_delay_ms"].dropna()
        parent_to_child_delays = parent_to_child_delays[parent_to_child_delays >= 0].tolist()
    else:
        parent_to_child_delays = []
    parent_to_child_delay = compute_latency_metrics(parent_to_child_delays)

    # Compute child → parent delay metrics (Child send → Parent receive)
    if "child_to_parent_delay_ms" in successful.columns:
        child_to_parent_delays = successful["child_to_parent_delay_ms"].dropna()
        child_to_parent_delays = child_to_parent_delays[child_to_parent_delays >= 0].tolist()
    else:
        child_to_parent_delays = []
    child_to_parent_delay = compute_latency_metrics(child_to_parent_delays)

    # Compute parent → client delay metrics (Parent send → Client receive)
    if "parent_to_client_delay_ms" in successful.columns:
        parent_to_client_delays = successful["parent_to_client_delay_ms"].dropna()
        parent_to_client_delays = parent_to_client_delays[parent_to_client_delays >= 0].tolist()
    else:
        parent_to_client_delays = []
    parent_to_client_delay = compute_latency_metrics(parent_to_client_delays)

    # Compute simulated latency metrics (actual work time in Child)
    simulated_latencies = successful["simulated_latency_ms"].dropna().tolist()
    simulated_latency = compute_latency_metrics(simulated_latencies)

    # Compute throughput metrics
    total = len(df)
    success_count = len(successful)
    fail_count = total - success_count

    # Compute actual duration from timestamps
    duration_s = config.get("duration_s", 60.0)
    if "start_time" in df.columns and len(df) > 0:
        actual_duration = df["end_time"].max() - df["start_time"].min()
        if actual_duration > 0:
            duration_s = actual_duration

    throughput = ThroughputMetrics(
        num_concurrent=config.get("num_concurrent", 0),
        achieved_rps=total / duration_s if duration_s > 0 else 0,
        goodput=success_count / duration_s if duration_s > 0 else 0,
        total_requests=total,
        successful_requests=success_count,
        failed_requests=fail_count,
        error_rate=fail_count / total if total > 0 else 0,
    )

    # Compute fairness metrics for child replicas
    child_counts = (
        successful.groupby("child_replica_id").size().tolist()
        if "child_replica_id" in successful.columns
        else []
    )
    expected_child_replicas = config.get("child_replicas", 0)
    child_fairness = compute_fairness_metrics(child_counts, expected_child_replicas)

    # Compute fairness metrics for parent replicas
    parent_counts = (
        successful.groupby("parent_replica_id").size().tolist()
        if "parent_replica_id" in successful.columns
        else []
    )
    expected_parent_replicas = config.get("parent_replicas", 0)
    parent_fairness = compute_fairness_metrics(parent_counts, expected_parent_replicas)

    # Compute child replica utilization
    # utilization = total_work_time / (duration × max_ongoing_requests)
    child_utilization = compute_utilization_metrics(
        successful,
        replica_id_col="child_replica_id",
        work_time_col="simulated_latency_ms",
        duration_s=duration_s,
        max_ongoing_requests=config.get("max_ongoing_requests", 5),
    )

    return AggregatedMetrics(
        run_id=run_id,
        config=config,
        latency=latency,
        client_to_parent_delay=client_to_parent_delay,
        parent_to_child_delay=parent_to_child_delay,
        child_to_parent_delay=child_to_parent_delay,
        parent_to_client_delay=parent_to_client_delay,
        simulated_latency=simulated_latency,
        throughput=throughput,
        child_fairness=child_fairness,
        parent_fairness=parent_fairness,
        child_utilization=child_utilization,
    )


# ============================================================================
# Ray Data Parallel Aggregation
# ============================================================================

def _get_run_info_from_summary(summary_path: str) -> Tuple[str, str, dict]:
    """
    Extract run info from a summary.json file.
    
    Args:
        summary_path: S3 path to summary.json file.
    
    Returns:
        Tuple of (run_path, run_id, config).
    """
    summary = read_json_from_s3(summary_path)
    run_config = summary.get("run_config", {})
    run_id = run_config.get("run_id", "unknown")
    config = run_config.get("config", {})
    
    # Add timing info to config
    config["duration_s"] = run_config.get("duration_s", 60.0)
    config["warmup_s"] = run_config.get("warmup_s", 10.0)
    config["num_concurrent"] = run_config.get("config", {}).get("num_concurrent", 0)
    
    run_path = summary_path.rsplit("/", 1)[0] + "/"
    return run_path, run_id, config


@ray.remote
def _aggregate_single_run_ray(
    run_path: str,
    run_id: str,
    config: dict,
) -> Optional[AggregatedMetrics]:
    """
    Ray task to aggregate a single run using Ray Data for parallel CSV reading.
    
    Args:
        run_path: S3 path to run directory.
        run_id: Run identifier.
        config: Experiment configuration.
    
    Returns:
        AggregatedMetrics or None if aggregation fails.
    """
    try:
        # List CSV files for this run
        csv_files = list_s3_files(run_path, suffix=".csv")
        
        if not csv_files:
            print(f"No CSV files found for run {run_id}")
            return None
        
        # Use Ray Data for parallel CSV reading
        ds = ray.data.read_csv(csv_files)
        
        # Convert to pandas for aggregation (the data fits in memory per run)
        combined_df = ds.to_pandas()
        
        return aggregate_from_dataframe(combined_df, run_id, config)
    except Exception as e:
        print(f"ERROR aggregating {run_id}: {e}")
        return None


def aggregate_runs_parallel_from_s3(
    s3_prefix: str,
    max_concurrent_runs: int = 16,
) -> List[AggregatedMetrics]:
    """
    Aggregate metrics for all runs under an S3 prefix using Ray for parallelism.
    
    Uses Ray tasks to process multiple runs concurrently, and Ray Data within
    each run for parallel CSV reading.
    
    Args:
        s3_prefix: S3 path prefix (e.g., "s3://bucket/routing-study/")
        max_concurrent_runs: Maximum number of runs to process concurrently.
    
    Returns:
        List of AggregatedMetrics for all successful runs.
    """
    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Ensure path ends with /
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    
    # Find all summary.json files (one per run)
    print(f"Listing runs under {s3_prefix}...")
    summary_files = list_s3_files(s3_prefix, suffix="summary.json")
    
    if not summary_files:
        print(f"No runs found under {s3_prefix}")
        return []
    
    print(f"Found {len(summary_files)} runs to aggregate")
    
    # Extract run info from all summaries
    print("Reading run configurations...")
    run_infos = []
    for summary_path in summary_files:
        try:
            run_path, run_id, config = _get_run_info_from_summary(summary_path)
            run_infos.append((run_path, run_id, config))
        except Exception as e:
            print(f"ERROR reading summary {summary_path}: {e}")
    
    print(f"Processing {len(run_infos)} runs in parallel...")
    
    # Submit all runs as Ray tasks
    futures = []
    for run_path, run_id, config in run_infos:
        future = _aggregate_single_run_ray.remote(run_path, run_id, config)
        futures.append(future)
    
    # Collect results with progress reporting
    aggregated = []
    completed = 0
    total = len(futures)
    
    while futures:
        # Wait for next batch of results
        ready, futures = ray.wait(futures, num_returns=min(max_concurrent_runs, len(futures)))
        results = ray.get(ready)
        
        for metrics in results:
            completed += 1
            if metrics is not None:
                aggregated.append(metrics)
                print(f"[{completed}/{total}] Aggregated {metrics.run_id}")
            else:
                print(f"[{completed}/{total}] Skipped (failed)")
    
    print(f"\nSuccessfully aggregated {len(aggregated)} of {total} runs")
    return aggregated


def aggregate_run_from_s3(s3_run_path: str) -> AggregatedMetrics:
    """
    Aggregate metrics for a single experiment run from S3 using Ray Data.
    
    Reads all task_*.csv files and summary.json from the run path.
    Uses Ray Data for parallel CSV reading.
    
    Args:
        s3_run_path: S3 path to run directory (e.g., "s3://bucket/prefix/exp_id/run_id/")
    
    Returns:
        AggregatedMetrics for this run.
    """
    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Ensure path ends with /
    if not s3_run_path.endswith("/"):
        s3_run_path += "/"
    
    # Read summary.json for config
    summary_path = f"{s3_run_path}summary.json"
    summary = read_json_from_s3(summary_path)
    
    run_config = summary.get("run_config", {})
    run_id = run_config.get("run_id", "unknown")
    config = run_config.get("config", {})
    
    # Add timing info to config
    config["duration_s"] = run_config.get("duration_s", 60.0)
    config["warmup_s"] = run_config.get("warmup_s", 10.0)
    config["num_concurrent"] = run_config.get("config", {}).get("num_concurrent", 0)
    
    # List CSV files
    task_files = list_s3_files(s3_run_path, suffix=".csv")
    
    if not task_files:
        raise ValueError(f"No task CSV files found at {s3_run_path}")
    
    print(f"Reading {len(task_files)} task files from {s3_run_path} using Ray Data")
    
    # Use Ray Data for parallel CSV reading
    ds = ray.data.read_csv(task_files)
    
    # Convert to pandas for aggregation
    combined_df = ds.to_pandas()
    print(f"Combined {len(combined_df)} total requests")
    
    return aggregate_from_dataframe(combined_df, run_id, config)


def aggregate_experiment_prefix_from_s3(
    s3_prefix: str,
) -> List[AggregatedMetrics]:
    """
    Aggregate metrics for all experiments under a prefix in S3.
    
    This is the main entry point for parallel aggregation using Ray.
    
    Args:
        s3_prefix: S3 path prefix (e.g., "s3://bucket/routing-study/")
    
    Returns:
        List of AggregatedMetrics for all runs.
    """
    return aggregate_runs_parallel_from_s3(s3_prefix)


# Legacy function for backward compatibility
def aggregate_all_runs_from_s3(
    s3_experiment_path: str,
) -> List[AggregatedMetrics]:
    """
    Aggregate metrics for all runs under an experiment path in S3.
    
    This is a legacy wrapper that now uses parallel processing.
    
    Args:
        s3_experiment_path: S3 path to experiment directory 
            (e.g., "s3://bucket/prefix/experiment_id/")
    
    Returns:
        List of AggregatedMetrics for all runs.
    """
    return aggregate_runs_parallel_from_s3(s3_experiment_path)


# ============================================================================
# Output Functions
# ============================================================================

def metrics_to_dict(metrics: AggregatedMetrics) -> dict:
    """Convert AggregatedMetrics to dictionary for JSON serialization."""
    return {
        "run_id": metrics.run_id,
        "config": metrics.config,
        "latency_ms": {
            "p50": metrics.latency.p50,
            "p90": metrics.latency.p90,
            "p95": metrics.latency.p95,
            "p99": metrics.latency.p99,
            "max": metrics.latency.max,
            "min": metrics.latency.min,
            "mean": metrics.latency.mean,
        },
        "client_to_parent_delay_ms": {
            "p50": metrics.client_to_parent_delay.p50,
            "p90": metrics.client_to_parent_delay.p90,
            "p95": metrics.client_to_parent_delay.p95,
            "p99": metrics.client_to_parent_delay.p99,
            "max": metrics.client_to_parent_delay.max,
            "min": metrics.client_to_parent_delay.min,
            "mean": metrics.client_to_parent_delay.mean,
        },
        "parent_to_child_delay_ms": {
            "p50": metrics.parent_to_child_delay.p50,
            "p90": metrics.parent_to_child_delay.p90,
            "p95": metrics.parent_to_child_delay.p95,
            "p99": metrics.parent_to_child_delay.p99,
            "max": metrics.parent_to_child_delay.max,
            "min": metrics.parent_to_child_delay.min,
            "mean": metrics.parent_to_child_delay.mean,
        },
        "child_to_parent_delay_ms": {
            "p50": metrics.child_to_parent_delay.p50,
            "p90": metrics.child_to_parent_delay.p90,
            "p95": metrics.child_to_parent_delay.p95,
            "p99": metrics.child_to_parent_delay.p99,
            "max": metrics.child_to_parent_delay.max,
            "min": metrics.child_to_parent_delay.min,
            "mean": metrics.child_to_parent_delay.mean,
        },
        "parent_to_client_delay_ms": {
            "p50": metrics.parent_to_client_delay.p50,
            "p90": metrics.parent_to_client_delay.p90,
            "p95": metrics.parent_to_client_delay.p95,
            "p99": metrics.parent_to_client_delay.p99,
            "max": metrics.parent_to_client_delay.max,
            "min": metrics.parent_to_client_delay.min,
            "mean": metrics.parent_to_client_delay.mean,
        },
        "simulated_latency_ms": {
            "p50": metrics.simulated_latency.p50,
            "p90": metrics.simulated_latency.p90,
            "p95": metrics.simulated_latency.p95,
            "p99": metrics.simulated_latency.p99,
            "max": metrics.simulated_latency.max,
            "min": metrics.simulated_latency.min,
            "mean": metrics.simulated_latency.mean,
        },
        "throughput": {
            "num_concurrent": metrics.throughput.num_concurrent,
            "achieved_rps": metrics.throughput.achieved_rps,
            "goodput": metrics.throughput.goodput,
            "total_requests": metrics.throughput.total_requests,
            "successful_requests": metrics.throughput.successful_requests,
            "failed_requests": metrics.throughput.failed_requests,
            "error_rate": metrics.throughput.error_rate,
        },
        "child_fairness": {
            "cv": metrics.child_fairness.coefficient_of_variation,
            "jains_index": metrics.child_fairness.jains_index,
            "min_max_ratio": metrics.child_fairness.min_max_ratio,
            "replica_count": metrics.child_fairness.replica_count,
            "expected_replicas": metrics.child_fairness.expected_replicas,
            "unique_replica_pct": metrics.child_fairness.unique_replica_pct,
            "min_requests": metrics.child_fairness.min_requests,
            "max_requests": metrics.child_fairness.max_requests,
            "mean_requests": metrics.child_fairness.mean_requests,
            "std_requests": metrics.child_fairness.std_requests,
        },
        "parent_fairness": {
            "cv": metrics.parent_fairness.coefficient_of_variation,
            "jains_index": metrics.parent_fairness.jains_index,
            "min_max_ratio": metrics.parent_fairness.min_max_ratio,
            "replica_count": metrics.parent_fairness.replica_count,
            "expected_replicas": metrics.parent_fairness.expected_replicas,
            "unique_replica_pct": metrics.parent_fairness.unique_replica_pct,
            "min_requests": metrics.parent_fairness.min_requests,
            "max_requests": metrics.parent_fairness.max_requests,
            "mean_requests": metrics.parent_fairness.mean_requests,
            "std_requests": metrics.parent_fairness.std_requests,
        },
        "child_utilization": {
            "mean": metrics.child_utilization.mean,
            "min": metrics.child_utilization.min,
            "max": metrics.child_utilization.max,
            "std": metrics.child_utilization.std,
            "replica_count": metrics.child_utilization.replica_count,
        },
    }


def create_summary_dataframe(
    metrics_list: List[AggregatedMetrics],
) -> pd.DataFrame:
    """
    Create pandas DataFrame from aggregated metrics for analysis.

    Returns DataFrame with flattened columns for easy filtering and plotting.
    """
    rows = []
    for m in metrics_list:
        row = {
            "run_id": m.run_id,
            "algorithm": m.config.get("algorithm"),
            "scale": m.config.get("scale"),
            "ratio": m.config.get("ratio"),
            "topology": m.config.get("topology"),
            "locality": m.config.get("locality"),
            "load_level": m.config.get("load_level"),
            "parent_replicas": m.config.get("parent_replicas"),
            "child_replicas": m.config.get("child_replicas"),
            "num_concurrent": m.config.get("num_concurrent"),
            # Latency
            "latency_p50": m.latency.p50,
            "latency_p90": m.latency.p90,
            "latency_p95": m.latency.p95,
            "latency_p99": m.latency.p99,
            "latency_max": m.latency.max,
            "latency_mean": m.latency.mean,
            # Client -> Parent delay
            "client_parent_delay_p50": m.client_to_parent_delay.p50,
            "client_parent_delay_p90": m.client_to_parent_delay.p90,
            "client_parent_delay_p99": m.client_to_parent_delay.p99,
            "client_parent_delay_mean": m.client_to_parent_delay.mean,
            # Parent -> Child delay
            "parent_child_delay_p50": m.parent_to_child_delay.p50,
            "parent_child_delay_p90": m.parent_to_child_delay.p90,
            "parent_child_delay_p99": m.parent_to_child_delay.p99,
            "parent_child_delay_mean": m.parent_to_child_delay.mean,
            # Child -> Parent delay (return path)
            "child_parent_delay_p50": m.child_to_parent_delay.p50,
            "child_parent_delay_p90": m.child_to_parent_delay.p90,
            "child_parent_delay_p99": m.child_to_parent_delay.p99,
            "child_parent_delay_mean": m.child_to_parent_delay.mean,
            # Parent -> Client delay (return path)
            "parent_client_delay_p50": m.parent_to_client_delay.p50,
            "parent_client_delay_p90": m.parent_to_client_delay.p90,
            "parent_client_delay_p99": m.parent_to_client_delay.p99,
            "parent_client_delay_mean": m.parent_to_client_delay.mean,
            # Simulated latency (actual work time)
            "simulated_latency_p50": m.simulated_latency.p50,
            "simulated_latency_p90": m.simulated_latency.p90,
            "simulated_latency_p99": m.simulated_latency.p99,
            "simulated_latency_mean": m.simulated_latency.mean,
            # Throughput
            "achieved_rps": m.throughput.achieved_rps,
            "goodput": m.throughput.goodput,
            "error_rate": m.throughput.error_rate,
            # Child fairness
            "child_cv": m.child_fairness.coefficient_of_variation,
            "child_jains": m.child_fairness.jains_index,
            "child_min_max": m.child_fairness.min_max_ratio,
            "child_unique_pct": m.child_fairness.unique_replica_pct,
            # Parent fairness
            "parent_cv": m.parent_fairness.coefficient_of_variation,
            "parent_jains": m.parent_fairness.jains_index,
            "parent_min_max": m.parent_fairness.min_max_ratio,
            "parent_unique_pct": m.parent_fairness.unique_replica_pct,
            # Child utilization
            "child_util_mean": m.child_utilization.mean,
            "child_util_min": m.child_utilization.min,
            "child_util_max": m.child_utilization.max,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_aggregated_metrics(
    metrics_list: List[AggregatedMetrics],
    output_path: str,
) -> None:
    """
    Save aggregated metrics to JSON file (local or S3).
    
    Args:
        metrics_list: List of AggregatedMetrics.
        output_path: Output path (local path or s3:// path).
    """
    data = [metrics_to_dict(m) for m in metrics_list]
    
    if output_path.startswith("s3://"):
        write_json_to_s3(data, output_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved aggregated metrics to {output_path}")


def save_summary_csv(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save summary DataFrame to CSV (local or S3).
    
    Args:
        df: Summary DataFrame.
        output_path: Output path (local path or s3:// path).
    """
    if output_path.startswith("s3://"):
        write_csv_to_s3(df, output_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved summary CSV to {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from S3 or local files using Ray Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all runs from a specific execution
  python -m analysis.aggregate s3://bucket/routing-study/20260110_123456/

  # Aggregate all runs across all executions
  python -m analysis.aggregate s3://bucket/routing-study/

  # Aggregate with custom output paths
  python -m analysis.aggregate s3://bucket/routing-study/20260110_123456/ \\
      --output /tmp/metrics.json --csv-output /tmp/summary.csv
      
  # Control parallelism
  python -m analysis.aggregate s3://bucket/routing-study/ --max-concurrent 32
""",
    )
    parser.add_argument(
        "source",
        type=str,
        help="S3 path to execution (s3://bucket/prefix/execution_id/) or local directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for aggregated metrics (JSON). Can be local or S3 path.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Output path for summary CSV. Can be local or S3 path.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum number of runs to process concurrently (default: 16).",
    )

    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        print(f"Ray initialized with {ray.cluster_resources().get('CPU', 0)} CPUs")

    # Aggregate based on source type
    if args.source.startswith("s3://"):
        print(f"Aggregating from S3: {args.source}")
        metrics = aggregate_runs_parallel_from_s3(
            args.source,
            max_concurrent_runs=args.max_concurrent,
        )
    else:
        # Legacy local file support (still uses Ray Data for reading)
        results_dir = Path(args.source)
        raw_dir = results_dir / "raw"
        manifests_dir = results_dir / "manifests"
        
        csv_files = sorted(raw_dir.glob("*.csv"))
        
        if csv_files:
            # Use Ray Data to read all CSVs in parallel
            print(f"Reading {len(csv_files)} CSV files using Ray Data")
            
            metrics = []
            for csv_path in csv_files:
                run_id = csv_path.stem
                manifest_path = manifests_dir / f"{run_id}.json"
                
                if not manifest_path.exists():
                    print(f"WARNING: No manifest for {run_id}, skipping")
                    continue
                
                try:
                    # Use Ray Data for reading
                    ds = ray.data.read_csv(str(csv_path))
                    df = ds.to_pandas()
                    
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    
                    m = aggregate_from_dataframe(
                        df, 
                        manifest["run_id"],
                        manifest["config"],
                    )
                    metrics.append(m)
                    print(f"Aggregated {run_id}")
                except Exception as e:
                    print(f"ERROR aggregating {run_id}: {e}")
        else:
            metrics = []

    if not metrics:
        print("No results found to aggregate")
        exit(1)

    print(f"\nAggregated {len(metrics)} runs")

    # Determine output paths
    if args.source.startswith("s3://"):
        default_json = f"{args.source.rstrip('/')}/aggregated/all_metrics.json"
        default_csv = f"{args.source.rstrip('/')}/aggregated/summary.csv"
    else:
        default_json = str(Path(args.source) / "all_metrics.json")
        default_csv = str(Path(args.source) / "summary.csv")
    
    json_output = args.output or default_json
    csv_output = args.csv_output or default_csv

    # Save outputs
    save_aggregated_metrics(metrics, json_output)
    
    df = create_summary_dataframe(metrics)
    save_summary_csv(df, csv_output)

    # Print summary
    print("\nSample metrics:")
    print(df.head())
