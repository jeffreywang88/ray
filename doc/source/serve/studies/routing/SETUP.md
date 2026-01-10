# Ray Serve Power-of-Two Routing Algorithm Study

## Overview

This study evaluates the effectiveness of Ray Serve's Power-of-Two Choices (Pow2) request routing algorithm. We compare Pow2 against baseline algorithms (Random, Round-Robin) across various cluster configurations to understand routing performance, fairness, and scalability characteristics.

---

## Goals

### Primary Goals

1. **Validate Pow2 Effectiveness**: Confirm that the Power-of-Two Choices algorithm provides measurable benefits over simpler alternatives (Random, Round-Robin).

2. **Characterize Fairness**: Quantify how evenly requests are distributed across replicas under different configurations.

3. **Understand Scalability**: Measure how routing performance degrades (or improves) as replica count increases.

4. **Evaluate Topology Impact**: Understand how replica placement (packed vs spread) and locality preferences affect routing decisions.

### Secondary Goals

- Identify configuration sweet spots for production deployments
- Establish baseline metrics for future routing algorithm improvements
- Document any unexpected behaviors or edge cases

---

## Experiment Setup

### Cluster Configuration

| Node Type | Spec | Count | Purpose |
|-----------|------|-------|---------|
| Head Node | 8 CPU, 32 GB | 1 | Ray head, cluster management |
| Deployment Workers | 8 CPU, 32 GB | 20-200 (autoscaling) | Parent/Child deployment replicas (1 CPU each) |
| Load Generator | 48 CPU, 192 GB | 1 (fixed) | IngressDeployment + load test tasks |

**Why separate node types?**
- **Deployment workers (8 CPU):** Each node fits 8 replicas (1 CPU each). Autoscaling adjusts capacity based on experiment scale.
- **Load generator (48 CPU):** Single large node ensures all load test tasks run co-located with the `IngressDeployment`. With `num_tasks + 1` CPUs reserved, even the largest experiments (2048 users = 16 tasks = 17 CPUs) fit comfortably.

### Serve Application Architecture

```
                                          ┌──────────────────┐     ┌──────────────────┐
┌───────────────────────────┐             │ ParentDeployment │────▶│ ChildDeployment  │
│  Ray Task 1 (≤128 users)  │──handle───▶│    (1 CPU)       │     │    (1 CPU)       │
└───────────────────────────┘             └──────────────────┘     └──────────────────┘
┌───────────────────────────┐                    │                        │
│  Ray Task 2 (≤128 users)  │──handle───▶        ▼                        ▼
└───────────────────────────┘             Records: replica_id,      Records: replica_id,
             ...                          node_id, timestamp        node_id, request_count
┌───────────────────────────┐
│  Ray Task N (≤128 users)  │──handle───▶
└───────────────────────────┘
             │
             └──── Orchestrated by IngressDeployment
```

**Architecture Components:**

1. **IngressDeployment**: A single-replica Serve deployment that orchestrates the experiment. It receives experiment configuration and spawns Ray tasks to generate load.

2. **Ray Tasks (Load Generators)**: Each task handles up to 128 concurrent "users" via asyncio. Users within a task share a DeploymentHandle and run closed-loop requests concurrently.

3. **ParentDeployment**: Receives requests from load generator tasks and forwards to ChildDeployment.

4. **ChildDeployment**: Performs simulated work with variable latency.

**Why this architecture?**
- Tests the actual DeploymentHandle routing (the core subject of this study)
- Removes external components (no HAProxy, no HTTP layer)
- Load generators scale naturally across the cluster with Ray tasks
- Minimizes task scheduling overhead (128 users per task vs 1 user per task)
- Eliminates single-entry-point bottleneck

### Deployment Configuration

```python
@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=5,      # Default, limits concurrency per replica
    max_queued_requests=-1,      # Unlimited queuing (default)
)
```

**Rationale:**
- `1 CPU` per replica provides consistent resource allocation
- `max_ongoing_requests=5` (default) ensures routing decisions matter
- At 512 replicas × 2 deployments = 1024 CPUs required for XLarge scale

### Workload

```python
# Parent calls Child, Child performs minimal work with variable latency
async def child_handler(parent_replica_id, parent_node_id):
    # Exponential distribution with mean=10ms to mimic real-world variance
    # - Most requests complete quickly
    # - Occasional slow requests (tail latency)
    latency_s = random.expovariate(1 / 0.01)  # mean = 10ms
    latency_s = min(latency_s, 0.1)  # cap at 100ms to avoid extreme outliers
    await asyncio.sleep(latency_s)
    return {
        "child_replica_id": self.replica_id,
        "child_node_id": self.node_id,
        "parent_replica_id": parent_replica_id,
        "parent_node_id": parent_node_id,
    }
```

**Latency Distribution Properties:**
- **Distribution:** Exponential (memoryless, realistic for many workloads)
- **Mean:** 10ms
- **Median:** ~7ms (ln(2) × mean)
- **p99:** ~46ms (ln(100) × mean)
- **Cap:** 100ms (prevents runaway outliers from skewing results)

**Theoretical Throughput per Replica:**
- Mean latency: 10ms (exponential distribution)
- Concurrency: 5 (`max_ongoing_requests`)
- Max RPS per replica: ~500 req/s (based on mean)
- Actual throughput varies due to latency variance
- **Expected:** ~300-400 req/s per replica under realistic conditions

**Throughput Bottleneck by Ratio:**

The bottleneck is always the deployment with fewer replicas relative to load:

| Ratio | Bottleneck | Example (Medium Scale) | Max Throughput |
|-------|------------|------------------------|----------------|
| 1:1 | Either (balanced) | 32 Parent, 32 Child | 16,000 req/s |
| 1:2 | Parent | 32 Parent, 64 Child | 16,000 req/s (parent-limited) |
| 2:1 | Child | 64 Parent, 32 Child | 16,000 req/s (child-limited) |

**Note:** In 2:1 configurations, the child deployment becomes the bottleneck. Each parent sends requests to child, so effective child load = parent_requests × fan-out. With 64 parents each at 250 req/s = 16,000 req/s hitting 32 child replicas = 500 req/s per child (at capacity).

---

## Independent Variables (What We Change)

| Variable | Values | Tested At | Primary Question |
|----------|--------|-----------|------------------|
| Algorithm | Pow2, Random, RoundRobin | All scales | Does Pow2 provide measurable benefit? |
| Scale | Small → XLarge | — | How does routing scale? |
| Parent:Child Ratio | 1:1, 1:2, 2:1 | Large only | How does asymmetry affect routing? |
| Topology | Packed, Spread | Large only | Does placement strategy matter? |
| Locality | Preferred, None | Large only | Does locality preference help? |
| Load Level | 50%, 75%, 100% | All scales | How does load affect behavior? |

### 0. Load Generation Strategy

**Approach: Closed-Loop with Fixed Concurrency via Ray Tasks**

We use closed-loop load generation where a fixed number of concurrent "users" each wait for a response before sending the next request. Each user runs as an independent Ray task that calls ParentDeployment via DeploymentHandle.

**Benefits of this approach:**
- Tests the actual DeploymentHandle routing path (subject of this study)
- Naturally limits load to match system capacity
- Prevents queue buildup and timeouts at high scale
- Load generators distributed across cluster (no single bottleneck)
- Minimizes Ray task overhead (128 users per task via asyncio)
- Each task has its own handle, simulating independent callers

**Experiment Configuration:**

```python
@dataclass
class ExperimentConfig:
    # Experiment identification
    experiment_id: str
    run_id: str                  # Unique ID for this run (e.g., UUID or timestamp)
    
    # S3 output configuration
    s3_bucket: str               # S3 bucket for storing results
    s3_prefix: str = "results"   # Prefix path within bucket
    
    # Load parameters
    num_concurrent_users: int    # Number of Ray tasks generating load
    duration_s: float = 60.0     # Steady-state measurement duration
    warmup_s: float = 10.0       # Warmup period (results discarded)
    
    # Deployment configuration (for reference/logging)
    algorithm: str               # "pow2", "random", "round_robin"
    parent_replicas: int
    child_replicas: int
    topology: str                # "packed" or "spread"
    locality_preference: bool
    
    @property
    def s3_output_path(self) -> str:
        """S3 path for this experiment run's results."""
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.experiment_id}/{self.run_id}"
```

**IngressDeployment Design:**

The `IngressDeployment` is scheduled on the dedicated load generator node (48 CPU) using a custom resource label. It requests enough CPUs to run all load generator tasks plus itself, ensuring tasks co-locate on the same node.

Each task writes its results directly to S3, avoiding large data transfers back to the ingress.

```python
MAX_USERS_PER_TASK = 128  # Maximum concurrent users per Ray task

def get_ingress_num_cpus(num_concurrent_users: int) -> int:
    """Calculate CPUs needed: 1 for ingress + 1 per task."""
    num_tasks = math.ceil(num_concurrent_users / MAX_USERS_PER_TASK)
    return num_tasks + 1

# Scheduled on load generator node via custom resource
# Node labeled with: ray.io/node-type: loadgen
@serve.deployment(num_replicas=1)
class IngressDeployment:
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentRunSummary:
        """Orchestrate a load test experiment."""
        # Calculate number of tasks needed
        num_tasks = math.ceil(config.num_concurrent_users / MAX_USERS_PER_TASK)
        users_per_task = config.num_concurrent_users // num_tasks
        remainder = config.num_concurrent_users % num_tasks
        
        # Spawn Ray tasks, distributing users across tasks
        # Each task writes directly to S3 (no data returned)
        task_refs = []
        for i in range(num_tasks):
            task_users = users_per_task + (1 if i < remainder else 0)
            task_refs.append(
                run_load_generator_task.options(num_cpus=1).remote(
                    task_id=i,
                    parent_app_name="parent",
                    num_users=task_users,
                    duration_s=config.duration_s,
                    warmup_s=config.warmup_s,
                    s3_output_path=config.s3_output_path,
                )
            )
        
        # Wait for all tasks to complete (they write to S3, return only summary)
        task_summaries = ray.get(task_refs)
        
        # Write experiment config and aggregated summary to S3
        run_summary = ExperimentRunSummary(
            config=config,
            num_tasks=num_tasks,
            task_summaries=task_summaries,
            total_requests=sum(t.num_requests for t in task_summaries),
            total_successful=sum(t.num_successful for t in task_summaries),
            total_failed=sum(t.num_failed for t in task_summaries),
        )
        write_json_to_s3(run_summary, f"{config.s3_output_path}/summary.json")
        
        return run_summary
```

**Ray Task Load Generator:**

Each Ray task runs an async event loop with multiple concurrent "users", similar to the existing `ClosedLoopLoadGenerator` design. Tasks write results directly to S3, returning only a lightweight summary.

```python
@dataclass
class TaskSummary:
    """Lightweight summary returned by each task (actual data goes to S3)."""
    task_id: int
    num_requests: int
    num_successful: int
    num_failed: int
    s3_file_path: str

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
    Results are written directly to S3, avoiding data transfer back to ingress.
    """
    
    async def run():
        # Get DeploymentHandle for this task
        parent_handle = serve.get_app_handle(parent_app_name)
        
        results: List[RequestResult] = []
        results_lock = asyncio.Lock()
        
        start_time = time.time()
        warmup_end = start_time + warmup_s
        end_time = start_time + warmup_s + duration_s
        
        async def user_loop(user_id: int):
            """Single user making closed-loop requests."""
            while time.time() < end_time:
                request_start = time.time()
                try:
                    response = await parent_handle.remote()
                    request_end = time.time()
                    
                    # Only record after warmup period
                    if request_start >= warmup_end:
                        async with results_lock:
                            results.append(RequestResult(
                                start_time=request_start,
                                end_time=request_end,
                                latency_ms=(request_end - request_start) * 1000,
                                parent_replica_id=response["parent_replica_id"],
                                parent_node_id=response["parent_node_id"],
                                child_replica_id=response["child_replica_id"],
                                child_node_id=response["child_node_id"],
                                success=True,
                            ))
                except Exception as e:
                    if time.time() >= warmup_end:
                        async with results_lock:
                            results.append(RequestResult(
                                start_time=request_start,
                                end_time=time.time(),
                                latency_ms=(time.time() - request_start) * 1000,
                                success=False,
                                error=str(e),
                            ))
        
        # Run all users concurrently within this task
        await asyncio.gather(*[user_loop(i) for i in range(num_users)])
        return results
    
    results = asyncio.run(run())
    
    # Write results directly to S3 (CSV format for efficiency)
    s3_file_path = f"{s3_output_path}/task_{task_id:04d}.csv"
    write_results_to_s3(results, s3_file_path)
    
    # Return lightweight summary only
    return TaskSummary(
        task_id=task_id,
        num_requests=len(results),
        num_successful=sum(1 for r in results if r.success),
        num_failed=sum(1 for r in results if not r.success),
        s3_file_path=s3_file_path,
    )
```

**Calculating Concurrent Users by Load Level:**

```
Concurrent Users = Load% × Bottleneck Replicas × CONCURRENT_PER_REPLICA

CONCURRENT_PER_REPLICA = 4  (must be <= max_ongoing_requests to avoid timeouts)
```

| Scale | Ratio | Bottleneck Replicas | 50% Load | 75% Load | 100% Load |
|-------|-------|---------------------|----------|----------|-----------|
| Small | 1:1 | 8 | 16 | 24 | 32 |
| Medium | 1:1 | 32 | 64 | 96 | 128 |
| Large | 1:1 | 128 | 256 | 384 | 512 |
| Large | 1:2 | 128 (parent) | 256 | 384 | 512 |
| Large | 2:1 | 128 (child) | 256 | 384 | 512 |
| XLarge | 1:1 | 512 | 1024 | 1536 | 2048 |

**Load Generation Infrastructure:**

| Component | Specification |
|-----------|--------------|
| Load Generator | Ray tasks spawned by IngressDeployment |
| Request Method | DeploymentHandle.remote() |
| Max Users per Task | 128 (uses asyncio for concurrency within task) |
| IngressDeployment CPUs | `num_tasks + 1` (ensures tasks co-locate with ingress) |
| Task Scheduling | `num_cpus=1` per task |
| Request Timeout | 30s (capture extreme tail latency) |
| Warmup Handling | Discard first 10s of measurements per task |

**Note:** Closed-loop prevents overloading the system. With `max_ongoing_requests=5` per replica and `CONCURRENT_PER_REPLICA=4`, we stay under server capacity and avoid request timeouts.

### 1. Routing Algorithm

| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **Power-of-Two (Pow2)** | Randomly sample 2 replicas, pick one with shorter queue | Default `PowerOfTwoChoicesRequestRouter` |
| **Random** | Uniformly random replica selection | Custom `RandomRequestRouter` |
| **Round-Robin** | Cycle through replicas in order | Custom `RoundRobinRequestRouter` |

### 2. Replica Count and Parent:Child Ratio

We test four scale levels, each with three Parent:Child ratios (1:1, 1:2, 2:1).

**Rationale for ratios:**
- **1:1** — Baseline symmetric configuration
- **1:2** — Parent fans out to more child replicas (common in inference pipelines)
- **2:1** — Multiple parents share fewer child replicas (resource-constrained child)

| Scale | Ratio | Parent | Child | Total Replicas | Deployment CPUs | Worker Nodes (8 CPU) |
|-------|-------|--------|-------|----------------|-----------------|----------------------|
| Small | 1:1 | 8 | 8 | 16 | 16 | 2 |
| Medium | 1:1 | 32 | 32 | 64 | 64 | 8 |
| Large | 1:1 | 128 | 128 | 256 | 256 | 32 |
| Large | 1:2 | 128 | 256 | 384 | 384 | 48 |
| Large | 2:1 | 256 | 128 | 384 | 384 | 48 |
| XLarge | 1:1 | 512 | 512 | 1024 | 1024 | 128 |

**Notes:** 
- Ratio (1:2, 2:1), Topology, and Locality variations only tested at Large scale
- Small, Medium, XLarge use fixed config: 1:1 ratio, Packed topology, No locality preference
- This focuses detailed analysis at Large scale while still capturing scaling trends
- Load generator node (48 CPU) handles all scales: max 17 CPUs needed (2048 users at XLarge)

### 3. Replica Topology

| Setting | Env Var | Effect |
|---------|---------|--------|
| **Packed** | `RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY=1` | Replicas bin-packed onto fewer nodes |
| **Spread** | `RAY_SERVE_USE_PACK_SCHEDULING_STRATEGY=0` | Replicas spread across nodes |

### 4. Locality Preference

| Setting | Control | Effect |
|---------|---------|--------|
| **Local Node Preferred** | `DeploymentHandle.options(prefer_local_routing=True)` | Prefer replicas on same node |
| **No Locality Preference** | `DeploymentHandle.options(prefer_local_routing=False)` | No node preference |

### 5. Load Level

| Level | Description | Target |
|-------|-------------|--------|
| Low | Well below capacity | 25% of max concurrent users |
| Medium | Moderate load | 50% of max concurrent users |
| High | Near capacity | 75% of max concurrent users |
| Saturated | At capacity | 100% of max concurrent users |

**Note:** Max concurrent users = `bottleneck_replicas × CONCURRENT_PER_REPLICA` where `CONCURRENT_PER_REPLICA = 4`.

---

## Dependent Variables (What We Measure)

### 1. Latency Metrics

| Metric | Description | Collection Method |
|--------|-------------|-------------------|
| **E2E Latency p50** | Median request latency | Task-side timestamps |
| **E2E Latency p90** | 90th percentile latency | Task-side timestamps |
| **E2E Latency p95** | 95th percentile latency | Task-side timestamps |
| **E2E Latency p99** | 99th percentile latency (tail) | Task-side timestamps |
| **E2E Latency max** | Maximum observed latency | Task-side timestamps |

### 2. Throughput Metrics

| Metric | Description | Collection Method |
|--------|-------------|-------------------|
| **Concurrent Users** | Number of simultaneous users | Configuration parameter |
| **Achieved RPS** | Actual requests per second | Completed requests / actual duration |
| **Goodput** | Successful requests per second | Successful requests / actual duration |
| **Error Rate** | Percentage of failed requests | Task-side |

### 3. Fairness Metrics

All fairness metrics are computed from the distribution of request counts across Child replicas.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Coefficient of Variation** | `CV = σ / μ` | Normalized std dev; 0 = perfect, lower is better |
| **Jain's Fairness Index** | `J = (Σxᵢ)² / (n × Σxᵢ²)` | Range [1/n, 1]; 1 = perfectly fair |
| **Min/Max Ratio** | `min(x) / max(x)` | Range [0, 1]; 1 = perfectly fair |

**Primary metric:** Jain's Fairness Index (single number, widely used in literature)
**Secondary metrics:** CV (for spread), Min/Max (for worst-case)

### 4. Utilization Metrics

Replica utilization measures how much of a replica's processing capacity was actually used during the experiment. Low utilization despite pending requests indicates router-side bottlenecks (e.g., head-of-line blocking from queue length probing).

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Replica Utilization** | `total_work_time / (duration × max_ongoing_requests)` | Range [0, 1]; 1 = fully utilized |
| **Mean Utilization** | Mean across all replicas | Overall system efficiency |
| **Min Utilization** | Minimum across replicas | Identifies starved replicas |

**Calculation:**
```python
# For each child replica:
total_work_time_ms = sum(simulated_latency_ms for all requests handled)
max_capacity_ms = duration_s * 1000 * max_ongoing_requests  # e.g., 60s * 1000 * 5 = 300,000ms
utilization = total_work_time_ms / max_capacity_ms
```

**Example:**
- Duration: 60s, `max_ongoing_requests=5` → max capacity = 300,000ms of work per replica
- If a replica did 150,000ms of simulated work → 50% utilization
- 50% utilization means the replica was idle half the time (capacity wasted)

**Interpretation:**
- **High utilization + low routing delay** = system is efficient
- **Low utilization + high routing delay** = router is the bottleneck (requests queued at router while replicas idle)
- **Low utilization + low routing delay** = insufficient load (not enough concurrent users)

---

## Experiment Methodology

### Phase 1: Warm-up

**Duration:** 10 seconds

**Purpose:**
- Populate queue length caches
- Allow DeploymentHandle routing to stabilize
- Warm up replica actor pools

**Precondition:** Application must be fully deployed and ready before starting (verified via health check).

**Traffic Pattern:** All Ray task users start simultaneously and begin making requests via DeploymentHandle.

**Data:** Each task discards results recorded before `warmup_end` timestamp. This happens within each task, not as a post-processing step.

### Phase 2: Steady-state measurement

**Duration:** 60 seconds

**Purpose:**
- Collect primary metrics under stable conditions
- Sufficient for statistical significance with high request rates

**Traffic Pattern:** Fixed number of Ray task users, each making closed-loop requests via DeploymentHandle.

**Data:** Primary analysis dataset. All tasks return their collected `RequestResult` lists to the `IngressDeployment` for aggregation.

### Repetitions

Each configuration combination is run **3 times** to account for variance. Results are reported as mean ± standard deviation across runs.

### Running an Experiment

**Step 1: Deploy IngressDeployment on load generator node**

The `IngressDeployment` is deployed on the dedicated 48 CPU load generator node. Use node affinity to ensure placement:

```python
import math
from ray import serve

MAX_USERS_PER_TASK = 128

def deploy_ingress(num_concurrent_users: int):
    """Deploy IngressDeployment on load generator node with CPUs for all tasks."""
    num_tasks = math.ceil(num_concurrent_users / MAX_USERS_PER_TASK)
    num_cpus = num_tasks + 1  # +1 for the ingress itself
    
    ingress = IngressDeployment.options(
        ray_actor_options={
            "num_cpus": num_cpus,
            # Schedule on the 48 CPU load generator node
            "resources": {"loadgen": 1},
        }
    ).bind()
    serve.run(ingress, name="ingress")

# Example: 384 users → 3 tasks → 4 CPUs (fits easily on 48 CPU node)
deploy_ingress(num_concurrent_users=384)

# Example: 2048 users → 16 tasks → 17 CPUs (still fits on 48 CPU node)
deploy_ingress(num_concurrent_users=2048)
```

**Note:** The 48 CPU load generator node should be configured with custom resource `loadgen: 48` to enable affinity scheduling.

**Step 2: Run the experiment**

```python
import ray
import uuid
from ray import serve

# Connect to the running Serve application
ingress_handle = serve.get_app_handle("ingress")

# Configure and run the experiment
config = ExperimentConfig(
    experiment_id="pow2_large_75pct",
    run_id=str(uuid.uuid4())[:8],  # Unique run ID
    s3_bucket="my-routing-study-bucket",
    s3_prefix="experiments",
    num_concurrent_users=384,  # 75% load at Large scale
    duration_s=60.0,
    warmup_s=10.0,
    algorithm="pow2",
    parent_replicas=128,
    child_replicas=128,
    topology="packed",
    locality_preference=False,
)

# Run the experiment (blocks until complete)
# Raw data is written directly to S3 by tasks
summary: ExperimentRunSummary = ray.get(ingress_handle.run_experiment.remote(config))

print(f"Experiment complete!")
print(f"  Total requests: {summary.total_requests}")
print(f"  Success rate: {summary.total_successful / summary.total_requests:.2%}")
print(f"  Data location: {config.s3_output_path}")

# Compute detailed metrics from S3 data (can be done later/offline)
metrics = compute_metrics(config.s3_output_path)
```

---

## Experiment Matrix

### Full Factorial Design

**Large scale (full variation):**

| Dimension | Values | Count |
|-----------|--------|-------|
| Algorithm | Pow2, Random, RoundRobin | 3 |
| Ratio | 1:1, 1:2, 2:1 | 3 |
| Topology | Packed, Spread | 2 |
| Locality | Preferred, None | 2 |
| Load Level | 50%, 75%, 100% | 3 |

Large configs: 3 × 3 × 2 × 2 × 3 = **108 configurations**

**Other scales (fixed: 1:1 ratio, Packed topology, No locality preference):**

| Dimension | Values | Count |
|-----------|--------|-------|
| Algorithm | Pow2, Random, RoundRobin | 3 |
| Scale | Small, Medium, XLarge | 3 |
| Load Level | 50%, 75%, 100% | 3 |

Other configs: 3 × 3 × 3 = **27 configurations**

**Total Combinations:** 108 + 27 = **135 configurations**

**With 3 repetitions:** 405 experiment runs

### Prioritized Subset (Initial Study)

Focus on Large scale at 75% load to compare algorithms across all variations:

| Dimension | Values | Count |
|-----------|--------|-------|
| Algorithm | Pow2, Random, RoundRobin | 3 |
| Ratio | 1:1, 1:2, 2:1 | 3 |
| Topology | Packed, Spread | 2 |
| Locality | Preferred, None | 2 |
| Load Level | 75% | 1 |

**Reduced Combinations:** 3 × 3 × 2 × 2 × 1 = **36 configurations**

**With 3 repetitions:** 108 experiment runs

---

## Data Collection

### S3 Storage Structure

Each experiment run writes data to S3 with the following structure:

```
s3://{bucket}/{prefix}/{experiment_id}/{run_id}/
├── summary.json          # Experiment config + aggregated summary
├── task_0000.csv         # Raw results from task 0
├── task_0001.csv         # Raw results from task 1
├── task_0002.csv         # Raw results from task 2
└── ...
```

**Why S3?**
- Raw result data can be large (millions of requests at high scale)
- Avoids data transfer overhead between tasks and ingress
- Tasks write in parallel for efficient I/O
- Data persists immediately (no loss if aggregation fails)
- Easy to analyze with tools that support S3 (pandas, Spark, etc.)

### Per-Request Data (CSV Files)

Each task writes its results to a CSV file in S3:

```python
@dataclass
class RequestResult:
    start_time: float        # Unix timestamp when request started
    end_time: float          # Unix timestamp when request completed
    latency_ms: float        # End-to-end latency in milliseconds
    parent_replica_id: str   # ID of parent replica that handled the request
    parent_node_id: str      # Node where parent replica runs
    child_replica_id: str    # ID of child replica that handled the request
    child_node_id: str       # Node where child replica runs
    success: bool            # Whether the request succeeded
    error: Optional[str]     # Error message if request failed
```

### Experiment Run Summary (JSON)

The `IngressDeployment` writes a summary file after all tasks complete:

```python
@dataclass
class ExperimentRunSummary:
    config: ExperimentConfig
    num_tasks: int
    task_summaries: List[TaskSummary]  # Summary from each task
    
    # Aggregated counts (computed from task summaries)
    total_requests: int
    total_successful: int
    total_failed: int
    
    # S3 paths for raw data files
    @property
    def data_files(self) -> List[str]:
        return [t.s3_file_path for t in self.task_summaries]
```

### Post-Experiment Aggregation

Detailed metrics (latency percentiles, fairness) are computed offline from the S3 data:

```python
def compute_metrics(s3_output_path: str) -> ExperimentMetrics:
    """Load all task CSVs from S3 and compute metrics."""
    # Load all task results
    all_results = []
    for csv_path in list_s3_files(f"{s3_output_path}/task_*.csv"):
        df = pd.read_csv(csv_path)
        all_results.append(df)
    
    combined = pd.concat(all_results)
    
    return ExperimentMetrics(
        # Latency percentiles
        latency_p50=combined["latency_ms"].quantile(0.50),
        latency_p90=combined["latency_ms"].quantile(0.90),
        latency_p95=combined["latency_ms"].quantile(0.95),
        latency_p99=combined["latency_ms"].quantile(0.99),
        latency_max=combined["latency_ms"].max(),
        
        # Throughput
        achieved_rps=len(combined) / (combined["end_time"].max() - combined["start_time"].min()),
        
        # Fairness metrics from child replica distribution
        child_replica_counts=combined["child_replica_id"].value_counts().to_dict(),
        jains_fairness_index=compute_jains_index(combined["child_replica_id"]),
        coefficient_of_variation=compute_cv(combined["child_replica_id"]),
        min_max_ratio=compute_min_max_ratio(combined["child_replica_id"]),
    )
```

---

## Hypothesis Testing

### Hypothesis 1: Pow2 Improves Tail Latency
**Expectation:** Pow2 will show lower p99 latency than Random/RoundRobin due to load-aware selection.

### Hypothesis 2: Pow2 Improves Fairness
**Expectation:** Pow2 will show higher Jain's Index and lower Gini coefficient than Random.

### Hypothesis 5: RoundRobin Has Best Fairness
**Expectation:** RoundRobin will show near-perfect fairness (Jain's ≈ 1.0) but potentially worse tail latency under load.

---

## Limitations and Caveats

1. **Synthetic Workload:** Exponential latency distribution (mean 10ms) is a simplification. Real inference workloads may have bimodal distributions (cache hit/miss), heavier tails, or correlated latencies.

2. **No Autoscaling:** Replica counts are fixed; real deployments may autoscale.

3. **Queue Length Cache Always On:** Can't disable the cache in non-Ray-Client contexts; this is the production configuration.

4. **No Multiplexing:** Study doesn't cover multiplexed model routing which has different behavior.

5. **Network Homogeneity:** All nodes are same type in same AZ; cross-AZ routing not tested.

6. **Ray Task Overhead:** Load generator tasks have scheduling overhead. With 128 users per task, this is minimized but may still introduce some variance at very high scale (thousands of users requiring dozens of tasks).

---

## Output Artifacts

All artifacts are stored in S3 under the configured bucket and prefix.

**Per-Run Artifacts (in `s3://{bucket}/{prefix}/{experiment_id}/{run_id}/`):**
1. **Raw Data:** CSV files with per-request logs (`task_XXXX.csv`)
2. **Run Summary:** JSON file with config and task summaries (`summary.json`)

**Aggregated Artifacts (generated by post-processing):**
1. **Aggregated Metrics:** JSON files with computed metrics per configuration
2. **Visualizations:**
   - Latency CDFs by algorithm
   - Fairness metrics bar charts
   - Heatmaps of metrics across configuration space
3. **Summary Report:** Key findings and recommendations

---


**Run Time Estimate:**
- Per configuration: ~1.5 min (10s warm-up + 60s steady + overhead)
- Prioritized subset (108 runs): ~2.7 hours
- Full matrix (405 runs): ~10 hours

---

## References

- [Power of Two Choices in Randomized Load Balancing](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) - Mitzenmacher et al.
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [Jain's Fairness Index](https://en.wikipedia.org/wiki/Fairness_measure)
