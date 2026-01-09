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

| Component | Specification |
|-----------|--------------|
| Cloud Provider | AWS |
| Worker Node Type | m7i.8xlarge (32 vCPU, 128 GiB RAM) |
| Head Node | Same as worker (m7i.8xlarge) |
| Autoscaling | 1 node minimum, scale up as needed |
| Max Nodes | 12 (384 vCPU total capacity) |

### Serve Application Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   HAProxy   │────▶│ ParentDeployment │────▶│ ChildDeployment  │
│ (head node) │     │   (0.25 CPU)     │     │   (0.25 CPU)     │
└─────────────┘     └──────────────────┘     └──────────────────┘
                           │                        │
                           ▼                        ▼
                    Records: replica_id,      Records: replica_id,
                    node_id, timestamp        node_id, request_count
```

**Why HAProxy Mode?**
- Isolates the study to replica-to-replica routing
- Removes Serve's native proxy from the critical path
- Single entry point simplifies traffic generation, need to watch out for bottlenecks here.

### Deployment Configuration

```python
@serve.deployment(
    ray_actor_options={"num_cpus": 0.25},
    max_ongoing_requests=5,      # Default, limits concurrency per replica
    max_queued_requests=-1,      # Unlimited queuing (default)
)
```

**Rationale:**
- `0.25 CPU` per replica avoids CPU contention with trivial workload
- `max_ongoing_requests=5` (default) ensures routing decisions matter
- At 512 replicas × 2 deployments = 256 CPUs required

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
- Concurrency: 5
- Max RPS per replica: ~500 req/s (based on mean)
- Actual throughput varies due to latency variance
- **Measured:** ~300 req/s per replica with 2 replicas ([Locust load test](locust_2_replicas.png))

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

**Approach: Closed-Loop with Fixed Concurrency**

We use closed-loop load generation where a fixed number of concurrent "users" each wait for a response before sending the next request. This approach:
- Naturally limits load to match system capacity
- Prevents queue buildup and timeouts at high scale
- Provides stable, predictable throughput measurements
- Matches behavior of tools like Locust for easier validation

```python
async def closed_loop_user(
    session: aiohttp.ClientSession,
    end_time: float,
    results: List[RequestResult],
):
    """Simulate a single user making requests in a loop."""
    while time.time() < end_time:
        result = await send_request(session)
        results.append(result)
        # Immediately send next request (closed-loop)
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
| Load Generator Location | Head node (32 vCPU m7i.8xlarge has capacity) |
| Client Library | aiohttp with connection pooling |
| Multi-Process | Yes, max 500 concurrent users per process |
| Connection Pool | Keepalive enabled |
| Request Timeout | 30s (capture extreme tail latency) |
| Warmup Handling | Discard first 10s of measurements |

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

| Scale | Ratio | Parent Replicas | Child Replicas | Total Replicas | CPUs Required |
|-------|-------|-----------------|----------------|----------------|---------------|
| Small | 1:1 | 8 | 8 | 16 | 4 |
| Medium | 1:1 | 32 | 32 | 64 | 16 |
| Large | 1:1 | 128 | 128 | 256 | 64 |
| Large | 1:2 | 128 | 256 | 384 | 96 |
| Large | 2:1 | 256 | 128 | 384 | 96 |
| XLarge | 1:1 | 512 | 512 | 1024 | 256 |

**Note:** 
- Ratio (1:2, 2:1), Topology, and Locality variations only tested at Large scale
- Small, Medium, XLarge use fixed config: 1:1 ratio, Packed topology, No locality preference
- This focuses detailed analysis at Large scale while still capturing scaling trends

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
| **E2E Latency p50** | Median request latency | Client-side timestamps |
| **E2E Latency p90** | 90th percentile latency | Client-side timestamps |
| **E2E Latency p95** | 95th percentile latency | Client-side timestamps |
| **E2E Latency p99** | 99th percentile latency (tail) | Client-side timestamps |
| **E2E Latency max** | Maximum observed latency | Client-side timestamps |

### 2. Throughput Metrics

| Metric | Description | Collection Method |
|--------|-------------|-------------------|
| **Concurrent Users** | Number of simultaneous users | Configuration parameter |
| **Achieved RPS** | Actual requests per second | Completed requests / actual duration |
| **Goodput** | Successful requests per second | Successful requests / actual duration |
| **Error Rate** | Percentage of failed requests | Client-side |

### 3. Fairness Metrics

All fairness metrics are computed from the distribution of request counts across Child replicas.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Coefficient of Variation** | `CV = σ / μ` | Normalized std dev; 0 = perfect, lower is better |
| **Jain's Fairness Index** | `J = (Σxᵢ)² / (n × Σxᵢ²)` | Range [1/n, 1]; 1 = perfectly fair |
| **Min/Max Ratio** | `min(x) / max(x)` | Range [0, 1]; 1 = perfectly fair |

**Primary metric:** Jain's Fairness Index (single number, widely used in literature)
**Secondary metrics:** CV (for spread), Min/Max (for worst-case)

---

## Experiment Methodology

### Phase 1: Warm-Up

**Duration:** 10 seconds

**Purpose:**
- Populate queue length caches
- Establish baseline connections
- Allow routing tasks to stabilize

**Precondition:** Application must be fully deployed and ready before starting (verified via health check).

**Traffic Pattern:** All concurrent users start immediately

**Data:** Discarded (not included in analysis)

### Phase 2: Steady State Measurement

**Duration:** 60 seconds

**Purpose:**
- Collect primary metrics under stable conditions
- Sufficient for statistical significance with high request rates

**Traffic Pattern:** Fixed number of concurrent users, each sending requests continuously

**Data:** Primary analysis dataset

### Repetitions

Each configuration combination is run **3 times** to account for variance. Results are reported as mean ± standard deviation across runs.

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

### Client-Side Logging

Each request logs:
```json
{
  "request_id": "uuid",
  "start_time": 1234567890.123,
  "end_time": 1234567890.456,
  "latency_ms": 333.0,
  "parent_replica_id": "parent-abc123",
  "parent_node_id": "node-xyz",
  "child_replica_id": "child-def456",
  "child_node_id": "node-xyz",
  "success": true,
  "error": null
}
```

### Aggregation

Post-experiment aggregation computes from client-side logs:
1. Per-replica request counts (for fairness metrics)
2. Latency percentiles (p50, p90, p95, p99, max)
3. Throughput (total requests / duration)
4. All fairness metrics from per-replica counts

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

2. **Single Entry Point:** All traffic through one HAProxy instance may become a bottleneck at high scale.

3. **No Autoscaling:** Replica counts are fixed; real deployments may autoscale.

4. **Queue Length Cache Always On:** Cannot disable cache in non-Ray-Client contexts; this is the production configuration.

5. **No Multiplexing:** Study doesn't cover multiplexed model routing which has different behavior.

6. **Network Homogeneity:** All nodes are same type in same AZ; cross-AZ routing not tested.

---

## Output Artifacts

1. **Raw Data:** CSV files with per-request logs
2. **Aggregated Metrics:** JSON files with computed metrics per configuration
3. **Visualizations:**
   - Latency CDFs by algorithm
   - Fairness metrics bar charts
   - Heatmaps of metrics across configuration space
4. **Summary Report:** Key findings and recommendations

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
