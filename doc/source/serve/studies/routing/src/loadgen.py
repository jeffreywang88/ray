"""
Closed-loop load generator for routing algorithm study.

Implements closed-loop load generation where a fixed number of concurrent "users"
each wait for a response before sending the next request. This naturally limits
load and prevents queue buildup.

Supports multi-process mode with max 100 concurrent requests per process.
"""

import asyncio
import csv
import math
import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp


# Maximum concurrent requests per worker process
MAX_CONCURRENT_PER_WORKER = 128


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
    routing_delay_ms: Optional[float] = None


@dataclass
class LoadTestResult:
    """Aggregated results from a load test."""

    results: List[RequestResult]
    warmup_duration_s: float
    steady_state_duration_s: float
    num_concurrent: int  # Total concurrent users

    @property
    def steady_state_results(self) -> List[RequestResult]:
        """Results from steady state period only (excludes warmup)."""
        if not self.results:
            return []
        start_time = min(r.start_time for r in self.results)
        warmup_end = start_time + self.warmup_duration_s
        return [r for r in self.results if r.start_time >= warmup_end]

    @property
    def total_requests(self) -> int:
        return len(self.steady_state_results)

    @property
    def successful_requests(self) -> int:
        return sum(1 for r in self.steady_state_results if r.success)

    @property
    def failed_requests(self) -> int:
        return sum(1 for r in self.steady_state_results if not r.success)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def offered_rps(self) -> float:
        """Rate at which requests were completed (same as achieved for closed-loop)."""
        return self.achieved_rps

    @property
    def achieved_rps(self) -> float:
        """True throughput: completed requests / actual time span."""
        results = self.steady_state_results
        if not results:
            return 0.0
        first_start = min(r.start_time for r in results)
        last_end = max(r.end_time for r in results)
        actual_duration = last_end - first_start
        if actual_duration == 0:
            return 0.0
        return len(results) / actual_duration

    @property
    def goodput(self) -> float:
        """Successful requests per second (true throughput of successes)."""
        results = self.steady_state_results
        if not results:
            return 0.0
        first_start = min(r.start_time for r in results)
        last_end = max(r.end_time for r in results)
        actual_duration = last_end - first_start
        if actual_duration == 0:
            return 0.0
        return self.successful_requests / actual_duration

    # Keep target_rps for backwards compatibility
    @property
    def target_rps(self) -> int:
        """Backwards compatibility - returns num_concurrent."""
        return self.num_concurrent


def _worker_run_load_test(
    worker_id: int,
    num_concurrent: int,
    warmup_duration_s: float,
    steady_state_duration_s: float,
    target_url: str,
    start_barrier_time: float,
    result_queue: mp.Queue,
) -> None:
    """
    Worker function that runs in a separate process.
    
    Args:
        worker_id: Unique identifier for this worker.
        num_concurrent: Number of concurrent users for this worker.
        warmup_duration_s: Warmup duration.
        steady_state_duration_s: Steady state duration.
        target_url: URL to send requests to.
        start_barrier_time: Wall-clock time when all workers should start.
        result_queue: Queue to put results into.
    """
    # Wait until the barrier time to synchronize all workers
    wait_time = start_barrier_time - time.time()
    if wait_time > 0:
        time.sleep(wait_time)
    
    async def run():
        generator = ClosedLoopLoadGenerator(
            target_url=target_url,
            request_timeout_s=30.0,
        )
        return await generator.run(
            num_concurrent=num_concurrent,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            progress_interval_s=10.0,
            worker_id=worker_id,
        )
    
    result = asyncio.run(run())
    
    # Convert results to serializable format for queue
    serialized_results = [
        {
            "request_id": r.request_id,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "error": r.error,
            "parent_replica_id": r.parent_replica_id,
            "parent_node_id": r.parent_node_id,
            "child_replica_id": r.child_replica_id,
            "child_node_id": r.child_node_id,
            "simulated_latency_ms": r.simulated_latency_ms,
            "routing_delay_ms": r.routing_delay_ms,
        }
        for r in result.results
    ]
    
    result_queue.put((worker_id, serialized_results))


class MultiProcessLoadGenerator:
    """
    Multi-process closed-loop load generator.
    
    Distributes concurrent users across worker processes,
    with max MAX_CONCURRENT_PER_WORKER users per process.
    """
    
    def __init__(
        self,
        target_url: str = "http://localhost:8000",
        max_concurrent_per_worker: int = MAX_CONCURRENT_PER_WORKER,
    ):
        self.target_url = target_url
        self.max_concurrent_per_worker = max_concurrent_per_worker
    
    def run(
        self,
        num_concurrent: int,
        warmup_duration_s: float = 10.0,
        steady_state_duration_s: float = 60.0,
    ) -> LoadTestResult:
        """
        Run closed-loop load test using multiple worker processes.
        
        Args:
            num_concurrent: Total number of concurrent users.
            warmup_duration_s: Warmup duration.
            steady_state_duration_s: Steady state duration.
        
        Returns:
            Combined LoadTestResult from all workers.
        """
        # Calculate number of workers needed
        num_workers = max(1, math.ceil(num_concurrent / self.max_concurrent_per_worker))
        users_per_worker = num_concurrent // num_workers
        remainder = num_concurrent % num_workers
        
        print(f"Multi-process closed-loop load generator:")
        print(f"  Total concurrent users: {num_concurrent}")
        print(f"  Workers: {num_workers}")
        print(f"  ~{users_per_worker} users per worker")
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Schedule start time (give workers time to spawn)
        start_barrier_time = time.time() + 2.0
        
        # Start worker processes
        workers = []
        for i in range(num_workers):
            # Distribute remainder across first workers
            worker_users = users_per_worker + (1 if i < remainder else 0)
            
            p = mp.Process(
                target=_worker_run_load_test,
                args=(
                    i,
                    worker_users,
                    warmup_duration_s,
                    steady_state_duration_s,
                    self.target_url,
                    start_barrier_time,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)
            print(f"  Started worker {i}: {worker_users} concurrent users")
        
        # Collect results from all workers
        all_results: List[RequestResult] = []
        for _ in range(num_workers):
            worker_id, serialized_results = result_queue.get()
            print(f"  Worker {worker_id} complete: {len(serialized_results)} results")
            
            # Deserialize results
            for r in serialized_results:
                all_results.append(RequestResult(**r))
        
        # Wait for all workers to finish
        for p in workers:
            p.join()
        
        print(f"All workers complete. Total results: {len(all_results)}")
        
        return LoadTestResult(
            results=all_results,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            num_concurrent=num_concurrent,
        )


class ClosedLoopLoadGenerator:
    """
    Closed-loop load generator with fixed concurrency.

    Each "user" runs in a loop: send request → wait for response → repeat.
    This naturally limits load based on system capacity and prevents queue buildup.
    """

    def __init__(
        self,
        target_url: str = "http://localhost:8000",
        request_timeout_s: float = 30.0,
    ):
        """
        Initialize load generator.

        Args:
            target_url: URL to send requests to.
            request_timeout_s: Timeout for individual requests.
        """
        self.target_url = target_url
        self.request_timeout_s = request_timeout_s

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
    ) -> RequestResult:
        """Send a single request and record result."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            async with session.get(
                self.target_url,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
            ) as response:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    return RequestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        latency_ms=latency_ms,
                        success=True,
                        parent_replica_id=data.get("parent_replica_id"),
                        parent_node_id=data.get("parent_node_id"),
                        child_replica_id=data.get("child_replica_id"),
                        child_node_id=data.get("child_node_id"),
                        simulated_latency_ms=data.get("simulated_latency_ms"),
                        routing_delay_ms=data.get("routing_delay_ms"),
                    )
                else:
                    return RequestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        latency_ms=latency_ms,
                        success=False,
                        error=f"HTTP {response.status}",
                    )
        except asyncio.TimeoutError:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                success=False,
                error="Timeout",
            )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                success=False,
                error=str(e),
            )

    async def _user_loop(
        self,
        user_id: int,
        session: aiohttp.ClientSession,
        end_time: float,
        results: List[RequestResult],
        results_lock: asyncio.Lock,
    ) -> None:
        """
        Simulate a single user making requests in a loop.
        
        Each user waits for response before sending next request (closed-loop).
        """
        while time.time() < end_time:
            result = await self._send_request(session)
            async with results_lock:
                results.append(result)

    async def run(
        self,
        num_concurrent: int,
        warmup_duration_s: float = 10.0,
        steady_state_duration_s: float = 60.0,
        progress_interval_s: float = 5.0,
        worker_id: Optional[int] = None,
    ) -> LoadTestResult:
        """
        Run closed-loop load test with fixed concurrency.

        Args:
            num_concurrent: Number of concurrent users (each sends requests in a loop).
            warmup_duration_s: Duration of warmup period (data discarded).
            steady_state_duration_s: Duration of steady state measurement.
            progress_interval_s: Interval for progress logging.
            worker_id: Optional worker ID for multi-process mode logging.

        Returns:
            LoadTestResult containing all request results.
        """
        total_duration_s = warmup_duration_s + steady_state_duration_s
        log_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

        # Configure connection pool
        connector = aiohttp.TCPConnector(
            limit=num_concurrent + 10,  # Slightly more than concurrent users
            keepalive_timeout=30,
            force_close=False,
        )

        results: List[RequestResult] = []
        results_lock = asyncio.Lock()

        async with aiohttp.ClientSession(connector=connector) as session:
            start_time = time.time()
            end_time = start_time + total_duration_s

            print(f"{log_prefix}Starting closed-loop load test:")
            print(f"{log_prefix}  Concurrent users: {num_concurrent}")
            print(f"{log_prefix}  Duration: {total_duration_s}s "
                  f"({warmup_duration_s}s warmup + {steady_state_duration_s}s steady state)")

            # Start progress logging task
            async def log_progress():
                last_count = 0
                last_time = start_time
                while time.time() < end_time:
                    await asyncio.sleep(progress_interval_s)
                    current_time = time.time()
                    elapsed = current_time - start_time
                    async with results_lock:
                        current_count = len(results)
                    
                    # Calculate RPS for this interval
                    interval_requests = current_count - last_count
                    interval_duration = current_time - last_time
                    interval_rps = interval_requests / interval_duration if interval_duration > 0 else 0
                    
                    in_warmup = elapsed < warmup_duration_s
                    phase = "warmup" if in_warmup else "steady"
                    print(f"{log_prefix}  [{phase}] {elapsed:.1f}s: {current_count} requests "
                          f"(RPS: {interval_rps:.1f})")
                    
                    last_count = current_count
                    last_time = current_time

            # Start all user tasks
            progress_task = asyncio.create_task(log_progress())
            user_tasks = [
                asyncio.create_task(
                    self._user_loop(i, session, end_time, results, results_lock)
                )
                for i in range(num_concurrent)
            ]

            # Wait for all users to complete
            await asyncio.gather(*user_tasks)
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        print(f"{log_prefix}Load test complete: {len(results)} total requests")

        return LoadTestResult(
            results=results,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            num_concurrent=num_concurrent,
        )


def save_results_to_csv(results: LoadTestResult, output_path: Path) -> None:
    """
    Save load test results to CSV file.

    Args:
        results: LoadTestResult to save.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "request_id",
        "start_time",
        "end_time",
        "latency_ms",
        "success",
        "error",
        "parent_replica_id",
        "parent_node_id",
        "child_replica_id",
        "child_node_id",
        "simulated_latency_ms",
        "routing_delay_ms",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results.steady_state_results:
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
                "routing_delay_ms": result.routing_delay_ms,
            })

    print(f"Saved {len(results.steady_state_results)} results to {output_path}")


def run_load_test_sync(
    num_concurrent: int,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    target_url: str = "http://localhost:8000",
) -> LoadTestResult:
    """
    Run a closed-loop load test using multi-process mode.

    Always uses multi-process mode to avoid event loop conflicts when called
    from async contexts (e.g., run_experiment).

    Args:
        num_concurrent: Number of concurrent users.
        warmup_duration_s: Duration of warmup period.
        steady_state_duration_s: Duration of steady state measurement.
        target_url: URL to send requests to.

    Returns:
        LoadTestResult containing all request results.
    """
    # Always use multi-process mode to avoid asyncio.run() conflicts
    # when called from within an existing event loop
    generator = MultiProcessLoadGenerator(target_url=target_url)
    return generator.run(
        num_concurrent=num_concurrent,
        warmup_duration_s=warmup_duration_s,
        steady_state_duration_s=steady_state_duration_s,
    )


async def run_load_test(
    num_concurrent: int,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    target_url: str = "http://localhost:8000",
) -> LoadTestResult:
    """
    Async convenience function to run a closed-loop load test (single-process only).

    For high concurrency (>100), use run_load_test_sync() instead which supports
    multi-process mode.

    Args:
        num_concurrent: Number of concurrent users.
        warmup_duration_s: Duration of warmup period.
        steady_state_duration_s: Duration of steady state measurement.
        target_url: URL to send requests to.

    Returns:
        LoadTestResult containing all request results.
    """
    generator = ClosedLoopLoadGenerator(target_url=target_url)
    return await generator.run(
        num_concurrent=num_concurrent,
        warmup_duration_s=warmup_duration_s,
        steady_state_duration_s=steady_state_duration_s,
    )


if __name__ == "__main__":
    # Quick test with low concurrency
    async def main():
        results = await run_load_test(
            num_concurrent=10,
            warmup_duration_s=2.0,
            steady_state_duration_s=5.0,
        )
        print(f"\nResults summary:")
        print(f"  Concurrent users: {results.num_concurrent}")
        print(f"  Total requests: {results.total_requests}")
        print(f"  Successful: {results.successful_requests}")
        print(f"  Failed: {results.failed_requests}")
        print(f"  Error rate: {results.error_rate:.2%}")
        print(f"  Achieved RPS: {results.achieved_rps:.1f}")
        print(f"  Goodput: {results.goodput:.1f}")

    asyncio.run(main())
