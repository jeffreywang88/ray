"""
Experiment runner for routing algorithm study.

Orchestrates a single experiment run:
1. Configure environment (topology)
2. Deploy Serve application
3. Run load generator via Ray tasks
4. Write summary to S3
5. Shutdown
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ray
from ray import serve

from src.app import build_app
from src.configurations import (
    ExperimentConfig,
    ExperimentRunConfig,
    Topology,
)
from src.loadgen import (
    LoadTestSummary,
    run_load_test,
)


def delete_s3_prefix(s3_path: str) -> int:
    """
    Delete all objects under an S3 prefix.
    
    Args:
        s3_path: S3 path prefix (e.g., "s3://bucket/prefix/")
    
    Returns:
        Number of objects deleted.
    """
    import boto3
    
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    # Parse S3 path
    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    s3_client = boto3.client("s3")
    
    # List and delete all objects under the prefix
    deleted_count = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        
        # Delete in batches of 1000 (S3 limit)
        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": delete_keys}
        )
        deleted_count += len(delete_keys)
    
    return deleted_count


def write_summary_to_s3(
    run_config: ExperimentRunConfig,
    load_summary: LoadTestSummary,
) -> str:
    """
    Write experiment summary to S3.
    
    Args:
        run_config: Experiment run configuration.
        load_summary: Load test summary from tasks.
    
    Returns:
        S3 path to the summary file.
    """
    import boto3
    
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config.to_dict(),
        "load_summary": load_summary.to_dict(),
    }
    
    s3_path = f"{run_config.s3_output_path}/summary.json"
    
    # Parse S3 path
    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1]
    
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(summary, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    
    print(f"Wrote summary to {s3_path}")
    return s3_path


async def wait_for_serve_ready(
    app_name: str = "routing-study",
    timeout_s: float = 120.0,
    check_interval_s: float = 2.0,
) -> bool:
    """
    Wait for Serve application to be ready by trying to get the app handle.

    Args:
        app_name: Name of the app to check.
        timeout_s: Maximum time to wait.
        check_interval_s: Interval between checks.

    Returns:
        True if ready, False if timeout.
    """
    start_time = time.time()
    print(f"Waiting for Serve app '{app_name}' to be ready...")

    while time.time() - start_time < timeout_s:
        try:
            # Try to get the app handle - this will work when the app is ready
            handle = serve.get_app_handle(app_name)
            
            # Try calling the health endpoint to verify it's actually working
            # This ensures replicas are ready, not just registered
            response = handle.health.remote()
            result = await response
            
            print(f"Serve is ready (took {time.time() - start_time:.1f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            # Show more detail for debugging
            error_msg = str(e)[:100] if str(e) else type(e).__name__
            print(f"  [{elapsed:.1f}s] Not ready: {error_msg}")

        await asyncio.sleep(check_interval_s)

    print(f"Timeout waiting for Serve to be ready after {timeout_s}s")
    return False


async def run_experiment(
    run_config: ExperimentRunConfig,
    clear_s3: bool = False,
) -> Optional[LoadTestSummary]:
    """
    Run a single experiment with the given configuration.

    Args:
        run_config: Full experiment run configuration including S3 settings.
        clear_s3: If True, delete existing files at the S3 output path before running.

    Returns:
        LoadTestSummary if successful, None if failed.
    """
    config = run_config.config
    
    print(f"\n{'='*60}")
    print(f"Starting experiment: {run_config.experiment_id}")
    print(f"Run ID: {run_config.run_id}")
    print(f"Configuration: {config}")
    print(f"Repetition: {run_config.repetition}")
    print(f"S3 Output: {run_config.s3_output_path}")
    print(f"{'='*60}\n")

    try:
        # Step 0a: Clear existing S3 files if requested (clears ALL runs for this experiment)
        if clear_s3:
            print(f"Clearing existing files at {run_config.s3_experiment_path}...")
            deleted = delete_s3_prefix(run_config.s3_experiment_path)
            print(f"  Deleted {deleted} objects")
        
        # Step 0b: Set environment for faster shutdown (proxy drain period)
        os.environ["RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S"] = "0"
        
        os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "1800"
        os.environ["RAY_SERVE_DISABLE_SHUTTING_DOWN_INGRESS_REPLICAS_FORCEFULLY"] = "1"
        
        # Step 1: Set environment for topology
        if config.topology == Topology.PACKED:
            os.environ["RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY"] = "1"
        else:
            os.environ["RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY"] = "0"

        print(f"Set RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY="
              f"{os.environ['RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY']}")

        # Step 2: Build and deploy application
        print(f"\nDeploying application with {config.parent_replicas} parent, "
              f"{config.child_replicas} child replicas...")

        app = build_app(
            parent_replicas=config.parent_replicas,
            child_replicas=config.child_replicas,
            prefer_local_routing=config.locality,
            request_router_class=config.algorithm.get_router_class(),
        )
        serve.run(app, name="routing-study")
        
        # Print Ray log directory for debugging
        try:
            log_dir = ray._private.worker._global_node.get_logs_dir_path()
            print(f"Ray logs: {log_dir}")
        except Exception:
            pass  # Not critical if this fails

        # Step 3: Wait for application to be ready
        ready = await wait_for_serve_ready(app_name="routing-study")
        if not ready:
            print("ERROR: Serve application failed to become ready")
            return None

        # Step 4: Run load test using Ray tasks
        print(f"\nRunning load test:")
        print(f"  Concurrent users: {run_config.num_concurrent}")
        print(f"  Tasks: {run_config.num_tasks}")
        print(f"  Warmup: {run_config.warmup_s}s")
        print(f"  Duration: {run_config.duration_s}s")
        
        load_summary = run_load_test(
            parent_app_name="routing-study",
            num_concurrent=run_config.num_concurrent,
            duration_s=run_config.duration_s,
            warmup_s=run_config.warmup_s,
            s3_output_path=run_config.s3_output_path,
        )

        # Step 5: Write summary to S3
        write_summary_to_s3(run_config, load_summary)

        # Step 6: Print summary
        print(f"\n{'='*60}")
        print(f"Experiment {run_config.run_id} complete!")
        print(f"{'='*60}")
        print(f"Concurrent users: {run_config.num_concurrent}")
        print(f"Total requests: {load_summary.total_requests}")
        print(f"Successful: {load_summary.total_successful}")
        print(f"Failed: {load_summary.total_failed}")
        print(f"Error rate: {load_summary.error_rate:.2%}")
        print(f"Achieved RPS: {load_summary.achieved_rps:.1f}")
        print(f"Results: {run_config.s3_output_path}")
        print(f"{'='*60}\n")

        return load_summary

    except Exception as e:
        print(f"ERROR: Experiment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Step 7: Shutdown Serve for clean state between experiments
        print("Shutting down Serve...")
        try:
            await asyncio.wait_for(serve.shutdown_async(), timeout=60.0)
            print("Serve shutdown complete")
        except asyncio.TimeoutError:
            print("WARNING: Serve shutdown timed out after 60s, continuing...")
        except Exception as e:
            print(f"WARNING: Serve shutdown error: {e}")
        
        print("Disconnecting Ray...")
        try:
            ray.shutdown()
        except Exception as e:
            print(f"WARNING: Ray shutdown error: {e}")
        print("Shutdown complete")
        await asyncio.sleep(1)


def run_experiment_sync(
    run_config: ExperimentRunConfig,
    clear_s3: bool = False,
) -> Optional[LoadTestSummary]:
    """Synchronous wrapper for run_experiment."""
    return asyncio.run(run_experiment(run_config, clear_s3=clear_s3))


# ============================================================================
# Convenience functions for creating run configs
# ============================================================================

def create_run_config(
    config: ExperimentConfig,
    s3_bucket: str,
    s3_prefix: str = "routing-study",
    repetition: int = 1,
    warmup_s: float = 10.0,
    duration_s: float = 60.0,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ExperimentRunConfig:
    """
    Create an ExperimentRunConfig from an ExperimentConfig.
    
    Args:
        config: Base experiment configuration.
        s3_bucket: S3 bucket for results.
        s3_prefix: S3 prefix path.
        repetition: Repetition number.
        warmup_s: Warmup duration.
        duration_s: Steady-state duration.
        experiment_id: Optional experiment ID (generated if not provided).
        run_id: Optional run ID (generated if not provided).
    
    Returns:
        Complete ExperimentRunConfig.
    """
    return ExperimentRunConfig(
        config=config,
        experiment_id=experiment_id or ExperimentRunConfig.generate_experiment_id(config),
        run_id=run_id or ExperimentRunConfig.generate_run_id(),
        repetition=repetition,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        warmup_s=warmup_s,
        duration_s=duration_s,
    )


if __name__ == "__main__":
    # Quick test - requires S3 bucket to be set
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiment <s3_bucket>")
        print("Example: python -m src.experiment my-routing-study-bucket")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    
    from src.configurations import generate_quick_configs
    
    config = generate_quick_configs()[0]
    run_config = create_run_config(
        config=config,
        s3_bucket=s3_bucket,
        warmup_s=5.0,
        duration_s=10.0,
    )
    
    run_experiment_sync(run_config)
