"""
Aggregate metrics from raw experiment logs.

Computes:
- Latency percentiles (p50, p90, p95, p99, max)
- Routing delay percentiles (time from Parent send to Child receive)
- Throughput metrics (RPS, goodput, error rate)
- Fairness metrics (CV, Jain's Index, Min/Max ratio)
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# Default paths - use /tmp to avoid packaging large results with Ray runtime_env
RESULTS_DIR = Path("/tmp/routing_results")


@dataclass
class FairnessMetrics:
    """Fairness metrics for request distribution across replicas."""

    coefficient_of_variation: float  # CV = σ / μ (lower is better, 0 = perfect)
    jains_index: float  # Range [1/n, 1], 1 = perfectly fair
    min_max_ratio: float  # Range [0, 1], 1 = perfectly fair
    replica_count: int
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
class AggregatedMetrics:
    """Complete aggregated metrics for an experiment run."""

    run_id: str
    config: dict
    latency: LatencyMetrics
    routing_delay: LatencyMetrics  # Time from Parent send to Child receive
    simulated_latency: LatencyMetrics  # Actual work time in Child (sleep duration)
    throughput: ThroughputMetrics
    child_fairness: FairnessMetrics  # Fairness across child replicas
    parent_fairness: FairnessMetrics  # Fairness across parent replicas


def compute_fairness_metrics(request_counts: List[int]) -> FairnessMetrics:
    """
    Compute fairness metrics from per-replica request counts.

    Args:
        request_counts: List of request counts per replica.

    Returns:
        FairnessMetrics dataclass with all computed metrics.
    """
    if not request_counts or all(c == 0 for c in request_counts):
        return FairnessMetrics(
            coefficient_of_variation=float("inf"),
            jains_index=0.0,
            min_max_ratio=0.0,
            replica_count=len(request_counts),
            min_requests=0,
            max_requests=0,
            mean_requests=0.0,
            std_requests=0.0,
        )

    n = len(request_counts)
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

    return FairnessMetrics(
        coefficient_of_variation=cv,
        jains_index=jains,
        min_max_ratio=min_max,
        replica_count=n,
        min_requests=min_val,
        max_requests=max_val,
        mean_requests=float(mean),
        std_requests=float(std),
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


def load_raw_results(csv_path: Path) -> pd.DataFrame:
    """Load raw results from CSV file."""
    return pd.read_csv(csv_path)


def load_manifest(manifest_path: Path) -> dict:
    """Load manifest JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def aggregate_run(
    raw_csv_path: Path,
    manifest_path: Path,
) -> AggregatedMetrics:
    """
    Aggregate metrics for a single experiment run.

    Args:
        raw_csv_path: Path to raw results CSV.
        manifest_path: Path to manifest JSON.

    Returns:
        AggregatedMetrics for this run.
    """
    # Load data
    df = load_raw_results(raw_csv_path)
    manifest = load_manifest(manifest_path)

    run_id = manifest["run_id"]
    config = manifest["config"]

    # Filter to successful requests for latency metrics
    successful = df[df["success"] == True]

    # Compute latency metrics
    latency = compute_latency_metrics(successful["latency_ms"].tolist())

    # Compute routing delay metrics (Parent send → Child receive)
    # Filter out any negative values (clock skew) and NaN
    routing_delays = successful["routing_delay_ms"].dropna()
    routing_delays = routing_delays[routing_delays >= 0].tolist()
    routing_delay = compute_latency_metrics(routing_delays)

    # Compute simulated latency metrics (actual work time in Child)
    simulated_latencies = successful["simulated_latency_ms"].dropna().tolist()
    simulated_latency = compute_latency_metrics(simulated_latencies)

    # Compute throughput metrics
    total = len(df)
    success_count = len(successful)
    fail_count = total - success_count
    duration_s = config.get("steady_state_duration_s", 60.0)  # Default from manifest

    # If we have timestamps, compute actual duration
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
    child_fairness = compute_fairness_metrics(child_counts)

    # Compute fairness metrics for parent replicas
    parent_counts = (
        successful.groupby("parent_replica_id").size().tolist()
        if "parent_replica_id" in successful.columns
        else []
    )
    parent_fairness = compute_fairness_metrics(parent_counts)

    return AggregatedMetrics(
        run_id=run_id,
        config=config,
        latency=latency,
        routing_delay=routing_delay,
        simulated_latency=simulated_latency,
        throughput=throughput,
        child_fairness=child_fairness,
        parent_fairness=parent_fairness,
    )


def aggregate_all_runs(
    results_dir: Path = RESULTS_DIR,
) -> List[AggregatedMetrics]:
    """
    Aggregate metrics for all runs in results directory.

    Args:
        results_dir: Base results directory.

    Returns:
        List of AggregatedMetrics for all runs.
    """
    raw_dir = results_dir / "raw"
    manifests_dir = results_dir / "manifests"

    aggregated = []

    for csv_path in sorted(raw_dir.glob("*.csv")):
        run_id = csv_path.stem
        manifest_path = manifests_dir / f"{run_id}.json"

        if not manifest_path.exists():
            print(f"WARNING: No manifest for {run_id}, skipping")
            continue

        try:
            metrics = aggregate_run(csv_path, manifest_path)
            aggregated.append(metrics)
            print(f"Aggregated {run_id}")
        except Exception as e:
            print(f"ERROR aggregating {run_id}: {e}")

    return aggregated


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
        "routing_delay_ms": {
            "p50": metrics.routing_delay.p50,
            "p90": metrics.routing_delay.p90,
            "p95": metrics.routing_delay.p95,
            "p99": metrics.routing_delay.p99,
            "max": metrics.routing_delay.max,
            "min": metrics.routing_delay.min,
            "mean": metrics.routing_delay.mean,
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
            "min_requests": metrics.parent_fairness.min_requests,
            "max_requests": metrics.parent_fairness.max_requests,
            "mean_requests": metrics.parent_fairness.mean_requests,
            "std_requests": metrics.parent_fairness.std_requests,
        },
    }


def save_aggregated_metrics(
    metrics_list: List[AggregatedMetrics],
    output_path: Path,
) -> None:
    """Save aggregated metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [metrics_to_dict(m) for m in metrics_list]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved aggregated metrics to {output_path}")


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
            # Routing delay (Parent -> Child)
            "routing_delay_p50": m.routing_delay.p50,
            "routing_delay_p90": m.routing_delay.p90,
            "routing_delay_p99": m.routing_delay.p99,
            "routing_delay_mean": m.routing_delay.mean,
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
            # Parent fairness
            "parent_cv": m.parent_fairness.coefficient_of_variation,
            "parent_jains": m.parent_fairness.jains_index,
            "parent_min_max": m.parent_fairness.min_max_ratio,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Results directory. Default: {RESULTS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for aggregated metrics (JSON)",
    )

    args = parser.parse_args()

    # Aggregate all runs
    metrics = aggregate_all_runs(args.results_dir)

    if not metrics:
        print("No results found to aggregate")
        exit(1)

    # Save to JSON
    output_path = args.output or (args.results_dir / "all_metrics.json")
    save_aggregated_metrics(metrics, output_path)

    # Create and save summary DataFrame
    df = create_summary_dataframe(metrics)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Print summary
    print(f"\nAggregated {len(metrics)} runs")
    print("\nSample metrics:")
    print(df.head())

