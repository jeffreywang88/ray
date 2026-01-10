"""
Closed-loop load generator for routing algorithm study.

Uses Ray tasks to generate load via DeploymentHandle, with results written directly to S3.
Each Ray task handles up to MAX_USERS_PER_TASK concurrent users via asyncio.
"""

import asyncio
import csv
import io
import math
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional

import ray
from ray import serve

from src.configurations import MAX_USERS_PER_TASK


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: str
    start_time: float
    end_time: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    parent_replica_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    child_replica_id: Optional[str] = None
    child_node_id: Optional[str] = None
    simulated_latency_ms: Optional[float] = None
    client_to_parent_delay_ms: Optional[float] = None  # Client → Parent routing
    routing_delay_ms: Optional[float] = None  # Parent → Child routing


@dataclass
class TaskSummary:
    """Lightweight summary returned by each task (actual data goes to S3)."""
    
    task_id: int
    num_requests: int
    num_successful: int
    num_failed: int
    s3_file_path: str
    start_time: float
    end_time: float


@dataclass 
class LoadTestSummary:
    """Aggregated summary from all tasks."""
    
    task_summaries: List[TaskSummary]
    num_concurrent: int
    num_tasks: int
    warmup_s: float
    duration_s: float
    
    @property
    def total_requests(self) -> int:
        return sum(t.num_requests for t in self.task_summaries)
    
    @property
    def total_successful(self) -> int:
        return sum(t.num_successful for t in self.task_summaries)
    
    @property
    def total_failed(self) -> int:
        return sum(t.num_failed for t in self.task_summaries)
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_failed / self.total_requests
    
    @property
    def achieved_rps(self) -> float:
        """Approximate RPS based on task timings."""
        if not self.task_summaries:
            return 0.0
        first_start = min(t.start_time for t in self.task_summaries)
        last_end = max(t.end_time for t in self.task_summaries)
        duration = last_end - first_start
        if duration <= 0:
            return 0.0
        return self.total_requests / duration
    
    def to_dict(self) -> dict:
        return {
            "num_concurrent": self.num_concurrent,
            "num_tasks": self.num_tasks,
            "warmup_s": self.warmup_s,
            "duration_s": self.duration_s,
            "total_requests": self.total_requests,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "error_rate": self.error_rate,
            "achieved_rps": self.achieved_rps,
            "task_summaries": [asdict(t) for t in self.task_summaries],
        }


def write_results_to_s3(results: List[RequestResult], s3_path: str) -> None:
    """
    Write results to S3 as CSV.
    
    Args:
        results: List of RequestResult to write.
        s3_path: S3 path (e.g., "s3://bucket/prefix/task_0000.csv").
    """
    import boto3
    
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""
    
    # Create CSV content in memory
    fieldnames = [
        "request_id", "start_time", "end_time", "latency_ms", "success", "error",
        "parent_replica_id", "parent_node_id", "child_replica_id", "child_node_id",
        "simulated_latency_ms", "client_to_parent_delay_ms", "routing_delay_ms",
    ]
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        writer.writerow({
            "request_id": result.request_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "latency_ms": result.latency_ms,
            "success": result.success,
            "error": result.error,
            "parent_replica_id": result.parent_replica_id,
            "parent_node_id": result.parent_node_id,
            "child_replica_id": result.child_replica_id,
            "child_node_id": result.child_node_id,
            "simulated_latency_ms": result.simulated_latency_ms,
            "client_to_parent_delay_ms": result.client_to_parent_delay_ms,
            "routing_delay_ms": result.routing_delay_ms,
        })
    
    # Upload to S3
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=output.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    
    print(f"Wrote {len(results)} results to {s3_path}")


@ray.remote
def run_load_generator_task(
    task_id: int,
    parent_app_name: str,
    num_users: int,
    duration_s: float,
    warmup_s: float,
    s3_output_path: str,
) -> TaskSummary:
    """
    Ray task that generates closed-loop traffic with multiple concurrent users.
    
    Each task handles up to MAX_USERS_PER_TASK concurrent users via asyncio.
    Results are written directly to S3, avoiding data transfer back to caller.
    
    Args:
        task_id: Unique identifier for this task.
        parent_app_name: Name of the parent Serve app to call.
        num_users: Number of concurrent users for this task.
        duration_s: Duration of steady-state measurement.
        warmup_s: Duration of warmup period (results discarded).
        s3_output_path: S3 path prefix for results.
    
    Returns:
        TaskSummary with counts and S3 file path.
    """
    
    # Progress logging interval in seconds
    PROGRESS_INTERVAL_S = 10.0
    
    async def run():
        # Get DeploymentHandle for this task
        parent_handle = serve.get_app_handle(parent_app_name)
        
        results: List[RequestResult] = []
        results_lock = asyncio.Lock()
        
        task_start_time = time.perf_counter()
        warmup_end = task_start_time + warmup_s
        end_time = task_start_time + warmup_s + duration_s
        
        # Track last progress log time
        last_progress_time = task_start_time
        last_progress_count = 0
        
        async def user_loop(user_id: int):
            """Single user making closed-loop requests."""
            nonlocal last_progress_time, last_progress_count
            
            while time.perf_counter() < end_time:
                request_id = str(uuid.uuid4())
                request_start = time.perf_counter()
                
                try:
                    # Use wall-clock time for cross-process timing
                    client_send_time = time.time()
                    response = await parent_handle.remote(client_send_time)
                    request_end = time.perf_counter()
                    
                    # Only record after warmup period
                    if request_start >= warmup_end:
                        async with results_lock:
                            results.append(RequestResult(
                                request_id=request_id,
                                start_time=request_start,
                                end_time=request_end,
                                latency_ms=(request_end - request_start) * 1000,
                                parent_replica_id=response.get("parent_replica_id"),
                                parent_node_id=response.get("parent_node_id"),
                                child_replica_id=response.get("child_replica_id"),
                                child_node_id=response.get("child_node_id"),
                                simulated_latency_ms=response.get("simulated_latency_ms"),
                                client_to_parent_delay_ms=response.get("client_to_parent_delay_ms"),
                                routing_delay_ms=response.get("routing_delay_ms"),
                                success=True,
                            ))
                            
                            # Log progress periodically (only from user 0 to avoid spam)
                            if user_id == 0:
                                now = time.perf_counter()
                                if now - last_progress_time >= PROGRESS_INTERVAL_S:
                                    elapsed = now - task_start_time
                                    count = len(results)
                                    delta_count = count - last_progress_count
                                    delta_time = now - last_progress_time
                                    rps = delta_count / delta_time if delta_time > 0 else 0
                                    
                                    phase = "warmup" if now < warmup_end else "steady"
                                    remaining = end_time - now
                                    
                                    print(f"[Task {task_id}] {elapsed:.0f}s elapsed, "
                                          f"{phase}, {count} reqs, ~{rps:.0f} rps, "
                                          f"{remaining:.0f}s remaining")
                                    
                                    last_progress_time = now
                                    last_progress_count = count
                                    
                except Exception as e:
                    request_end = time.perf_counter()
                    if request_start >= warmup_end:
                        async with results_lock:
                            results.append(RequestResult(
                                request_id=request_id,
                                start_time=request_start,
                                end_time=request_end,
                                latency_ms=(request_end - request_start) * 1000,
                                success=False,
                                error=str(e),
                            ))
        
        print(f"[Task {task_id}] Starting with {num_users} users, "
              f"{warmup_s}s warmup + {duration_s}s steady state")
        
        # Run all users concurrently within this task
        await asyncio.gather(*[user_loop(i) for i in range(num_users)])
        
        task_end_time = time.perf_counter()
        total_duration = task_end_time - task_start_time
        avg_rps = len(results) / duration_s if duration_s > 0 else 0
        print(f"[Task {task_id}] Complete: {len(results)} requests in {total_duration:.1f}s "
              f"(~{avg_rps:.0f} rps avg)")
        
        return results, task_start_time, task_end_time
    
    results, task_start_time, task_end_time = asyncio.run(run())
    
    # Write results directly to S3
    s3_file_path = f"{s3_output_path}/task_{task_id:04d}.csv"
    write_results_to_s3(results, s3_file_path)
    
    # Return lightweight summary only
    return TaskSummary(
        task_id=task_id,
        num_requests=len(results),
        num_successful=sum(1 for r in results if r.success),
        num_failed=sum(1 for r in results if not r.success),
        s3_file_path=s3_file_path,
        start_time=task_start_time,
        end_time=task_end_time,
    )


def run_load_test(
    parent_app_name: str,
    num_concurrent: int,
    duration_s: float,
    warmup_s: float,
    s3_output_path: str,
) -> LoadTestSummary:
    """
    Run a closed-loop load test using Ray tasks.
    
    Distributes concurrent users across Ray tasks, with each task handling
    up to MAX_USERS_PER_TASK users. Results are written directly to S3.
    
    Tasks are scheduled on nodes with the "load_test" custom resource.
    Each task requests 1/num_tasks of the resource to ensure all tasks
    fit on the designated load generator node.
    
    Args:
        parent_app_name: Name of the parent Serve app to call.
        num_concurrent: Total number of concurrent users.
        duration_s: Duration of steady-state measurement.
        warmup_s: Duration of warmup period.
        s3_output_path: S3 path prefix for results.
    
    Returns:
        LoadTestSummary with aggregated counts and task details.
    """
    # Calculate number of tasks needed
    num_tasks = max(1, math.ceil(num_concurrent / MAX_USERS_PER_TASK))
    users_per_task = num_concurrent // num_tasks
    remainder = num_concurrent % num_tasks
    
    # Each task requests 1/num_tasks of the "load_test" resource
    # This ensures all tasks fit on a node with load_test: 1
    load_test_resource = 1.0 / num_tasks
    
    print(f"Load test configuration:")
    print(f"  Total concurrent users: {num_concurrent}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Users per task: ~{users_per_task}")
    print(f"  Resource per task: load_test={load_test_resource:.4f}")
    print(f"  S3 output: {s3_output_path}")
    
    # Spawn Ray tasks with custom resource for node affinity
    task_refs = []
    for i in range(num_tasks):
        # Distribute remainder across first tasks
        task_users = users_per_task + (1 if i < remainder else 0)
        task_refs.append(
            run_load_generator_task.options(
                num_cpus=1,
                resources={"load_test": load_test_resource},
            ).remote(
                task_id=i,
                parent_app_name=parent_app_name,
                num_users=task_users,
                duration_s=duration_s,
                warmup_s=warmup_s,
                s3_output_path=s3_output_path,
            )
        )
        print(f"  Started task {i}: {task_users} users")
    
    # Wait for all tasks to complete
    print("Waiting for tasks to complete...")
    task_summaries = ray.get(task_refs)
    
    print(f"All {num_tasks} tasks complete")
    
    return LoadTestSummary(
        task_summaries=task_summaries,
        num_concurrent=num_concurrent,
        num_tasks=num_tasks,
        warmup_s=warmup_s,
        duration_s=duration_s,
    )
