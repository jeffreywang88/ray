"""
Poisson load generator for routing algorithm study.

Implements open-loop load generation where requests are sent at a target rate
independent of response times. Uses Poisson arrivals to model independent user requests.

Supports multi-process mode for high RPS (>250) to avoid asyncio event loop saturation.
"""

import asyncio
import csv
import math
import multiprocessing as mp
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp


# Maximum RPS per worker process to keep event loop responsive
MAX_RPS_PER_WORKER = 800


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
    target_rps: int

    @property
    def steady_state_results(self) -> List[RequestResult]:
        """Results from steady state period only (excludes warmup)."""
        if not self.results:
            return []
        start_time = self.results[0].start_time
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
        """Rate at which requests were sent (offered load)."""
        if self.steady_state_duration_s == 0:
            return 0.0
        return self.total_requests / self.steady_state_duration_s

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


def _worker_run_load_test(
    worker_id: int,
    target_rps: int,
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
        target_rps: Target RPS for this worker.
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
        generator = PoissonLoadGenerator(
            target_url=target_url,
            max_connections=100,  # Lower per-worker
            max_in_flight=500,    # Lower per-worker
        )
        return await generator.run(
            target_rps=target_rps,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            progress_interval_s=10.0,  # Less frequent logging per worker
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
    Multi-process load generator that distributes load across worker processes.
    
    Each worker handles at most MAX_RPS_PER_WORKER to keep asyncio responsive.
    """
    
    def __init__(
        self,
        target_url: str = "http://localhost:8000",
        max_rps_per_worker: int = MAX_RPS_PER_WORKER,
    ):
        self.target_url = target_url
        self.max_rps_per_worker = max_rps_per_worker
    
    def run(
        self,
        target_rps: int,
        warmup_duration_s: float = 10.0,
        steady_state_duration_s: float = 60.0,
    ) -> LoadTestResult:
        """
        Run load test using multiple worker processes.
        
        Args:
            target_rps: Total target RPS across all workers.
            warmup_duration_s: Warmup duration.
            steady_state_duration_s: Steady state duration.
        
        Returns:
            Combined LoadTestResult from all workers.
        """
        # Calculate number of workers needed
        num_workers = max(1, math.ceil(target_rps / self.max_rps_per_worker))
        rps_per_worker = target_rps // num_workers
        remainder = target_rps % num_workers
        
        print(f"Multi-process load generator: {num_workers} workers, ~{rps_per_worker} RPS each")
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Schedule start time (give workers time to spawn)
        start_barrier_time = time.time() + 2.0
        
        # Start worker processes
        workers = []
        for i in range(num_workers):
            # Distribute remainder across first workers
            worker_rps = rps_per_worker + (1 if i < remainder else 0)
            
            p = mp.Process(
                target=_worker_run_load_test,
                args=(
                    i,
                    worker_rps,
                    warmup_duration_s,
                    steady_state_duration_s,
                    self.target_url,
                    start_barrier_time,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)
            print(f"  Started worker {i}: {worker_rps} RPS")
        
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
            target_rps=target_rps,
        )


class PoissonLoadGenerator:
    """
    Open-loop load generator with Poisson arrivals.

    Sends requests at a target rate independent of response times.
    This is more realistic than closed-loop (fixed concurrency) because:
    - Real traffic doesn't wait for responses before sending new requests
    - Allows queue buildup to stress the routing algorithm
    - Poisson arrivals model independent user requests
    """

    def __init__(
        self,
        target_url: str = "http://localhost:8000",
        request_timeout_s: float = 30.0,
        max_connections: int = 0,  # 0 = unlimited
        max_in_flight: int = 5000,  # Limit concurrent requests to prevent memory issues
    ):
        """
        Initialize load generator.

        Args:
            target_url: URL to send requests to.
            request_timeout_s: Timeout for individual requests.
            max_connections: Maximum concurrent connections (0 = unlimited).
            max_in_flight: Maximum concurrent in-flight requests.
        """
        self.target_url = target_url
        self.request_timeout_s = request_timeout_s
        self.max_connections = max_connections
        self.max_in_flight = max_in_flight

    async def _send_request_with_semaphore(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> RequestResult:
        """Send request with semaphore to limit concurrency."""
        async with semaphore:
            return await self._send_request(session)

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

    async def run(
        self,
        target_rps: int,
        warmup_duration_s: float = 10.0,
        steady_state_duration_s: float = 60.0,
        progress_interval_s: float = 5.0,
        worker_id: Optional[int] = None,
    ) -> LoadTestResult:
        """
        Run load test with Poisson arrivals.

        Args:
            target_rps: Target requests per second.
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
            limit=self.max_connections,
            keepalive_timeout=30,
            force_close=False,
        )

        results: List[RequestResult] = []
        tasks: List[asyncio.Task] = []
        
        # Semaphore to limit in-flight requests (prevents memory exhaustion)
        semaphore = asyncio.Semaphore(self.max_in_flight)

        async with aiohttp.ClientSession(connector=connector) as session:
            start_time = time.time()
            end_time = start_time + total_duration_s
            last_progress = start_time
            requests_sent = 0

            print(f"{log_prefix}Starting load test: {target_rps} RPS for {total_duration_s}s "
                  f"({warmup_duration_s}s warmup + {steady_state_duration_s}s steady state)")

            # Pre-generate all arrival times to avoid per-request overhead
            # This gives us accurate Poisson arrivals without asyncio.sleep precision issues
            arrival_times = []
            t = start_time
            while t < end_time:
                inter_arrival_s = random.expovariate(target_rps)
                t += inter_arrival_s
                if t < end_time:
                    arrival_times.append(t)

            print(f"{log_prefix}  Pre-generated {len(arrival_times)} arrival times")

            arrival_idx = 0
            while arrival_idx < len(arrival_times):
                current_time = time.time()

                # Send all requests whose arrival time has passed
                while arrival_idx < len(arrival_times) and arrival_times[arrival_idx] <= current_time:
                    task = asyncio.create_task(self._send_request_with_semaphore(session, semaphore))
                    tasks.append(task)
                    requests_sent += 1
                    arrival_idx += 1

                # Progress logging
                if current_time - last_progress >= progress_interval_s:
                    elapsed = current_time - start_time
                    actual_rps = requests_sent / elapsed if elapsed > 0 else 0
                    in_warmup = elapsed < warmup_duration_s
                    phase = "warmup" if in_warmup else "steady"
                    print(f"{log_prefix}  [{phase}] {elapsed:.1f}s: {requests_sent} requests sent "
                          f"(actual RPS: {actual_rps:.1f})")
                    last_progress = current_time

                # Sleep until next arrival (or short sleep if we're behind)
                if arrival_idx < len(arrival_times):
                    sleep_time = max(0.0001, arrival_times[arrival_idx] - time.time())
                    # Cap sleep to 10ms to stay responsive
                    sleep_time = min(sleep_time, 0.001)
                    await asyncio.sleep(sleep_time)

            # Wait for all in-flight requests to complete
            print(f"{log_prefix}Waiting for {len(tasks)} in-flight requests to complete...")
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed:
                if isinstance(result, RequestResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    # Log but don't fail
                    print(f"{log_prefix}Request exception: {result}")

        print(f"{log_prefix}Load test complete: {len(results)} total requests")

        return LoadTestResult(
            results=results,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
            target_rps=target_rps,
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
    target_rps: int,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    target_url: str = "http://127.0.0.1:8000",
) -> LoadTestResult:
    """
    Run a load test, automatically using multi-process mode for high RPS.

    Uses multi-process mode when target_rps > MAX_RPS_PER_WORKER to avoid
    asyncio event loop saturation.

    Args:
        target_rps: Target requests per second.
        warmup_duration_s: Duration of warmup period.
        steady_state_duration_s: Duration of steady state measurement.
        target_url: URL to send requests to.

    Returns:
        LoadTestResult containing all request results.
    """
    if target_rps > MAX_RPS_PER_WORKER:
        # Use multi-process mode for high RPS
        generator = MultiProcessLoadGenerator(target_url=target_url)
        return generator.run(
            target_rps=target_rps,
            warmup_duration_s=warmup_duration_s,
            steady_state_duration_s=steady_state_duration_s,
        )
    else:
        # Use single-process mode for low RPS
        async def _run():
            generator = PoissonLoadGenerator(target_url=target_url)
            return await generator.run(
                target_rps=target_rps,
                warmup_duration_s=warmup_duration_s,
                steady_state_duration_s=steady_state_duration_s,
            )
        return asyncio.run(_run())


async def run_load_test(
    target_rps: int,
    warmup_duration_s: float = 10.0,
    steady_state_duration_s: float = 60.0,
    target_url: str = "http://localhost:8000",
) -> LoadTestResult:
    """
    Async convenience function to run a load test (single-process only).

    For high RPS (>250), use run_load_test_sync() instead which supports
    multi-process mode.

    Args:
        target_rps: Target requests per second.
        warmup_duration_s: Duration of warmup period.
        steady_state_duration_s: Duration of steady state measurement.
        target_url: URL to send requests to.

    Returns:
        LoadTestResult containing all request results.
    """
    generator = PoissonLoadGenerator(target_url=target_url)
    return await generator.run(
        target_rps=target_rps,
        warmup_duration_s=warmup_duration_s,
        steady_state_duration_s=steady_state_duration_s,
    )


if __name__ == "__main__":
    # Quick test with low RPS
    async def main():
        results = await run_load_test(
            target_rps=10,
            warmup_duration_s=2.0,
            steady_state_duration_s=5.0,
        )
        print(f"\nResults summary:")
        print(f"  Total requests: {results.total_requests}")
        print(f"  Successful: {results.successful_requests}")
        print(f"  Failed: {results.failed_requests}")
        print(f"  Error rate: {results.error_rate:.2%}")
        print(f"  Offered RPS: {results.offered_rps:.1f}")
        print(f"  Achieved RPS: {results.achieved_rps:.1f}")
        print(f"  Goodput: {results.goodput:.1f}")

    asyncio.run(main())

