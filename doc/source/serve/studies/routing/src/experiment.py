"""
Experiment runner for routing algorithm study.

Orchestrates a single experiment run:
1. Configure environment (topology)
2. Deploy Serve application
3. Run load generator
4. Collect results
5. Write manifest
6. Shutdown
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ray
from ray import serve

from src.app import build_app
from src.configurations import ExperimentConfig, Topology
from src.loadgen import (
    LoadTestResult,
    run_load_test_sync,
    save_results_to_csv,
)


# Default paths - use /tmp to avoid packaging large results with Ray runtime_env
RESULTS_DIR = Path("/tmp/routing_results")
RAW_DIR = RESULTS_DIR / "raw"
MANIFESTS_DIR = RESULTS_DIR / "manifests"


def generate_run_id() -> str:
    """Generate unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


def write_manifest(
    run_id: str,
    config: ExperimentConfig,
    repetition: int,
    output_dir: Path = MANIFESTS_DIR,
) -> Path:
    """
    Write manifest file for an experiment run.

    Args:
        run_id: Unique identifier for this run.
        config: Experiment configuration.
        repetition: Repetition number (1-indexed).
        output_dir: Directory to write manifest to.

    Returns:
        Path to the written manifest file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repetition": repetition,
        "config": config.to_dict(),
    }

    manifest_path = output_dir / f"{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    return manifest_path


def write_results_summary(
    run_id: str,
    config: ExperimentConfig,
    results: LoadTestResult,
    output_dir: Path = RESULTS_DIR / "aggregated",
) -> Path:
    """
    Write summary metrics for an experiment run.

    Args:
        run_id: Unique identifier for this run.
        config: Experiment configuration.
        results: Load test results.
        output_dir: Directory to write summary to.

    Returns:
        Path to the written summary file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute latency percentiles from successful requests
    successful_latencies = [
        r.latency_ms for r in results.steady_state_results if r.success
    ]

    if successful_latencies:
        successful_latencies.sort()
        n = len(successful_latencies)
        latency_stats = {
            "p50": successful_latencies[int(n * 0.50)],
            "p90": successful_latencies[int(n * 0.90)],
            "p95": successful_latencies[int(n * 0.95)],
            "p99": successful_latencies[int(n * 0.99)] if n >= 100 else successful_latencies[-1],
            "max": successful_latencies[-1],
            "min": successful_latencies[0],
            "mean": sum(successful_latencies) / n,
        }
    else:
        latency_stats = {}

    summary = {
        "run_id": run_id,
        "config": config.to_dict(),
        "throughput": {
            "target_rps": results.target_rps,
            "offered_rps": results.offered_rps,
            "achieved_rps": results.achieved_rps,
            "goodput": results.goodput,
            "total_requests": results.total_requests,
            "successful_requests": results.successful_requests,
            "failed_requests": results.failed_requests,
            "error_rate": results.error_rate,
        },
        "latency_ms": latency_stats,
    }

    summary_path = output_dir / f"{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {summary_path}")
    return summary_path


async def wait_for_serve_ready(
    url: str = "http://localhost:8000",
    timeout_s: float = 120.0,
    check_interval_s: float = 2.0,
) -> bool:
    """
    Wait for Serve application to be ready.

    Args:
        url: URL to health check.
        timeout_s: Maximum time to wait.
        check_interval_s: Interval between checks.

    Returns:
        True if ready, False if timeout.
    """
    import aiohttp

    start_time = time.time()
    print(f"Waiting for Serve to be ready at {url}...")

    while time.time() - start_time < timeout_s:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        print(f"Serve is ready (took {time.time() - start_time:.1f}s)")
                        return True
                    else:
                        print(f"  Health check returned status {response.status}")
        except Exception as e:
            print(f"  Health check failed: {type(e).__name__}: {e}")

        await asyncio.sleep(check_interval_s)

    print(f"Timeout waiting for Serve to be ready after {timeout_s}s")
    return False


async def run_experiment(
    config: ExperimentConfig,
    repetition: int = 1,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    results_dir: Path = RESULTS_DIR,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Run a single experiment with the given configuration.

    Args:
        config: Experiment configuration.
        repetition: Repetition number (1-indexed).
        warmup_duration_s: Duration of warmup period.
        steady_state_duration_s: Duration of steady state measurement.
        results_dir: Base directory for results.
        run_id: Optional run ID (generated if not provided).

    Returns:
        Run ID if successful, None if failed.
    """
    run_id = run_id or generate_run_id()

    print(f"\n{'='*60}")
    print(f"Starting experiment: {run_id}")
    print(f"Configuration: {config}")
    print(f"Repetition: {repetition}")
    print(f"{'='*60}\n")

    try:
        # Step 1: Set environment for topology
        if config.topology == Topology.PACKED:
            os.environ["RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY"] = "1"
        else:
            os.environ["RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY"] = "0"

        print(f"Set RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY={os.environ['RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY']}")

        # Step 2: Build and deploy application
        print(f"\nDeploying application with {config.parent_replicas} parent, "
              f"{config.child_replicas} child replicas...")

        app = build_app(
            parent_replicas=config.parent_replicas,
            child_replicas=config.child_replicas,
            prefer_local_routing=config.locality,
            request_router_class=config.algorithm.get_router_class(),
        )
        serve.run(app)

        # Step 3: Wait for application to be ready
        ready = await wait_for_serve_ready()
        if not ready:
            print("ERROR: Serve application failed to become ready")
            return None

        # Step 4: Write manifest before running load test
        write_manifest(
            run_id=run_id,
            config=config,
            repetition=repetition,
            output_dir=results_dir / "manifests",
        )

        # Step 5: Run load test (uses multi-process for high RPS automatically)
        print(f"\nRunning load test at {config.target_rps} RPS...")
        results = run_load_test_sync(
            target_rps=config.target_rps,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
        )

        # Step 6: Save raw results
        raw_path = results_dir / "raw" / f"{run_id}.csv"
        save_results_to_csv(results, raw_path)

        # Step 7: Save summary
        write_results_summary(
            run_id=run_id,
            config=config,
            results=results,
            output_dir=results_dir / "aggregated",
        )

        # Step 8: Print summary
        print(f"\n{'='*60}")
        print(f"Experiment {run_id} complete!")
        print(f"{'='*60}")
        print(f"Target RPS: {config.target_rps}")
        print(f"Offered RPS: {results.offered_rps:.1f}")
        print(f"Achieved RPS: {results.achieved_rps:.1f}")
        print(f"Goodput: {results.goodput:.1f}")
        print(f"Error rate: {results.error_rate:.2%}")
        print(f"Total requests: {results.total_requests}")
        
        # Compute and print latency stats from successful requests
        successful = [r for r in results.steady_state_results if r.success]
        if successful:
            latencies = [r.latency_ms for r in successful]
            avg_latency = sum(latencies) / len(latencies)
            print(f"Avg latency: {avg_latency:.2f}ms")
            
            routing_delays = [r.routing_delay_ms for r in successful if r.routing_delay_ms is not None]
            if routing_delays:
                avg_routing_delay = sum(routing_delays) / len(routing_delays)
                print(f"Avg routing delay: {avg_routing_delay:.2f}ms")
        
        print(f"{'='*60}\n")

        return run_id

    except Exception as e:
        print(f"ERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Step 9: Shutdown Serve and Ray for clean state between experiments
        print("Shutting down Serve...")
        serve.shutdown()
        print("Shutting down Ray...")
        ray.shutdown()
        print("Shutdown complete")


def run_experiment_sync(
    config: ExperimentConfig,
    repetition: int = 1,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    results_dir: Path = RESULTS_DIR,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """Synchronous wrapper for run_experiment."""
    return asyncio.run(
        run_experiment(
            config=config,
            repetition=repetition,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            results_dir=results_dir,
            run_id=run_id,
        )
    )


if __name__ == "__main__":
    # Quick test
    from src.configurations import generate_quick_configs

    config = generate_quick_configs()[0]
    run_experiment_sync(
        config=config,
        warmup_duration_s=5.0,
        steady_state_duration_s=10.0,
    )

