#!/usr/bin/env python
"""
Main CLI for running routing algorithm study experiments.

Usage examples:
    # Run quick test (3 configs, 1 repetition)
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type quick

    # Run prioritized subset (36 configs at Large scale, 75% load)
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type prioritized --repetitions 3

    # Run full matrix (135 configs)
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type full --repetitions 3

    # Run specific algorithm only
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type quick --algorithm pow2

    # Custom durations for testing
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type quick --warmup 5 --duration 10

    # Resume from a specific config index
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type prioritized --start-index 15

    # Dry run to see what would be executed
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type full --dry-run

    # Clear S3 files before each experiment run (useful for re-running)
    python scripts/run_experiments.py --s3-bucket my-bucket --run-type quick  
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configurations import (
    Algorithm,
    ExperimentConfig,
    ExperimentRunConfig,
    get_configs_by_run_type,
    print_config_summary,
)
from src.experiment import run_experiment_sync, create_run_config


# Default S3 prefix for results
DEFAULT_S3_PREFIX = "routing-study"


def filter_configs(
    configs: List[ExperimentConfig],
    algorithm: Optional[str] = None,
    scale: Optional[str] = None,
    load_level: Optional[float] = None,
) -> List[ExperimentConfig]:
    """
    Filter configurations by specified criteria.

    Args:
        configs: List of configurations to filter.
        algorithm: Filter by algorithm name (pow2, random, round_robin).
        scale: Filter by scale (small, medium, large, xlarge).
        load_level: Filter by load level (0.5, 0.75, 1.0).

    Returns:
        Filtered list of configurations.
    """
    filtered = configs

    if algorithm:
        filtered = [c for c in filtered if c.algorithm.value == algorithm]

    if scale:
        filtered = [c for c in filtered if c.scale.value == scale]

    if load_level is not None:
        filtered = [c for c in filtered if c.load_level.value == load_level]

    return filtered


def save_experiment_plan(
    configs: List[ExperimentConfig],
    repetitions: int,
    s3_bucket: str,
    s3_prefix: str,
) -> None:
    """Save experiment plan to S3 for reference."""
    import boto3
    
    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_configs": len(configs),
        "repetitions": repetitions,
        "total_runs": len(configs) * repetitions,
        "estimated_duration_hours": (len(configs) * repetitions * 1.5) / 60,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "configs": [c.to_dict() for c in configs],
    }

    s3_path = f"s3://{s3_bucket}/{s3_prefix}/experiment_plan.json"
    
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/experiment_plan.json",
        Body=json.dumps(plan, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Saved experiment plan to {s3_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run routing algorithm study experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # S3 configuration (required)
    parser.add_argument(
        "--s3-bucket",
        type=str,
        required=True,
        help="S3 bucket for storing results (required)",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=DEFAULT_S3_PREFIX,
        help=f"S3 prefix path within bucket. Default: {DEFAULT_S3_PREFIX}",
    )

    # Run type selection
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["full", "prioritized", "quick"],
        default="quick",
        help=(
            "Type of experiment run: "
            "'quick' (3 configs for testing), "
            "'prioritized' (36 configs at Large scale, 75%% load), "
            "'full' (all 135 configs). Default: quick"
        ),
    )

    # Repetitions
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per configuration. Default: 1",
    )

    # Filtering options
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["pow2", "random", "round_robin"],
        help="Run only configurations with this algorithm",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["small", "medium", "large", "xlarge"],
        help="Run only configurations with this scale",
    )
    parser.add_argument(
        "--load-level",
        type=float,
        choices=[0.5, 0.75, 1.0],
        help="Run only configurations with this load level",
    )

    # Duration options
    parser.add_argument(
        "--warmup",
        type=float,
        default=10.0,
        help="Warmup duration in seconds. Default: 10",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Steady state duration in seconds. Default: 60",
    )

    # Resume support
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this configuration index (0-indexed). Useful for resuming.",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running experiments",
    )

    # Clear S3 before each run
    parser.add_argument(
        "--clear-s3",
        action="store_true",
        help="Delete existing files at S3 output path before each experiment run",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Get and filter configurations
    configs = get_configs_by_run_type(args.run_type)
    configs = filter_configs(
        configs,
        algorithm=args.algorithm,
        scale=args.scale,
        load_level=args.load_level,
    )

    if not configs:
        print("ERROR: No configurations match the specified filters")
        sys.exit(1)

    # Calculate totals
    total_configs = len(configs)
    total_runs = total_configs * args.repetitions
    estimated_duration_mins = total_runs * (args.warmup + args.duration + 30) / 60

    # Print summary
    print("\n" + "=" * 60)
    print("ROUTING ALGORITHM STUDY - EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"\nRun type: {args.run_type}")
    print(f"Configurations: {total_configs}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Total runs: {total_runs}")
    print(f"Warmup: {args.warmup}s, Duration: {args.duration}s")
    print(f"Estimated duration: {estimated_duration_mins:.1f} minutes ({estimated_duration_mins/60:.1f} hours)")
    print(f"S3 output: s3://{args.s3_bucket}/{args.s3_prefix}/")
    if args.clear_s3:
        print(f"Clear S3: ENABLED (will delete existing files before each run)")

    if args.start_index > 0:
        remaining = total_configs - args.start_index
        print(f"Starting from index: {args.start_index} ({remaining} configs remaining)")

    if args.verbose:
        print_config_summary(configs)

    print("=" * 60 + "\n")

    # Dry run mode
    if args.dry_run:
        print("DRY RUN - No experiments will be executed\n")
        
        print("\nConfigurations to run:")
        for i, config in enumerate(configs[args.start_index:], start=args.start_index):
            print(f"  [{i}] {config}")
            print(f"       Replicas: {config.parent_replicas} parent, {config.child_replicas} child")
            print(f"       Concurrent users: {config.num_concurrent}, Tasks: {config.num_tasks}")
        return

    # Save experiment plan to S3
    save_experiment_plan(
        configs=configs,
        repetitions=args.repetitions,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )

    # Run experiments
    successful_runs = 0
    failed_runs = 0
    start_time = time.time()

    for config_idx, config in enumerate(configs[args.start_index:], start=args.start_index):
        for rep in range(1, args.repetitions + 1):
            run_number = (config_idx - args.start_index) * args.repetitions + rep
            total_remaining = total_runs - run_number + 1

            print(f"\n{'#' * 60}")
            print(f"# Run {run_number}/{total_runs} (config {config_idx}/{total_configs}, rep {rep}/{args.repetitions})")
            print(f"# Remaining: {total_remaining} runs")
            print(f"{'#' * 60}")

            # Create run config with S3 settings
            run_config = create_run_config(
                config=config,
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                repetition=rep,
                warmup_s=args.warmup,
                duration_s=args.duration,
            )

            result = run_experiment_sync(run_config, clear_s3=args.clear_s3)

            if result:
                successful_runs += 1
                print(f"Results saved to: {run_config.s3_output_path}")
            else:
                failed_runs += 1
                print(f"WARNING: Run failed for config {config_idx}, rep {rep}")

            # Progress update
            elapsed = time.time() - start_time
            runs_completed = successful_runs + failed_runs
            if runs_completed > 0:
                avg_time_per_run = elapsed / runs_completed
                remaining_runs = total_runs - runs_completed
                eta_mins = (avg_time_per_run * remaining_runs) / 60
                print(f"\nProgress: {runs_completed}/{total_runs} runs complete, "
                      f"ETA: {eta_mins:.1f} minutes")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    if successful_runs + failed_runs > 0:
        print(f"Success rate: {successful_runs/(successful_runs+failed_runs)*100:.1f}%")
    print(f"Results saved to: s3://{args.s3_bucket}/{args.s3_prefix}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
