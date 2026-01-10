# Routing Algorithm Improvements

This document captures performance issues discovered during the routing algorithm study and proposes improvements for Ray Serve's request routing.

## Executive Summary

The current routing implementation has significant overhead that impacts request latency:

1. **Queue length probing is in the request path** - Requests with cache misses on both chosen candidate replicas wait for queue length probes before being routed
2. **Thundering herd on probes** - Multiple handles probe the same replicas simultaneously
3. **Unnecessary probing for simple routers** - Random and RoundRobin don't need queue lengths but still probe

## Problem 1: Queue Length Probing in Request Path

### Current Behavior

```
Request → choose_replicas() → _select_from_candidate_replicas() → probe if needed → route
```

**Cache is enabled by default** (`use_replica_queue_len_cache=True` outside Ray Client).

**Probing is triggered when:**
1. Cache miss or entry expired (TTL: 10s via `RAY_SERVE_QUEUE_LENGTH_CACHE_TIMEOUT_S`)
2. Cached `queue_len >= max_ongoing_requests` (forces re-probe to check if capacity freed)

**The problem:** Probing blocks the request path with a 1s timeout per replica. Under load, when many replicas are at capacity, constant re-probing causes latency spikes.

### Impact

- **Added latency**: request pays the probe latency cost
- **Timeout accumulation**: Under load, probes queue up and timeout
- **Observed in study**: `client_to_parent_delay_ms` p50 of 24ms when probing is slow

The latency breakdown shows routing overhead (Client→Parent, Parent→Child delays) often exceeds the actual simulated work time, especially under high load:

![Latency Breakdown](figures/latency_breakdown.png)

## Problem 2: Thundering Herd on Probes

### Current Behavior

Each `DeploymentHandle` has its own router that independently probes replicas. While pow2 only probes 2 replicas per request, high concurrency creates many simultaneous probes:

```
Load Generator Task 1 ─── Handle ─── Router ───┐
Load Generator Task 2 ─── Handle ─── Router ───┼──→ Each probes 2 replicas per request
Load Generator Task 3 ─── Handle ─── Router ───┤    (but many concurrent requests)
...                                            │
Load Generator Task 27 ── Handle ─── Router ───┘
```

With 27 tasks, high concurrency per task, and cache misses:
- Each request probes 2 replicas (pow2 algorithm)
- But with ~500 concurrent requests across all tasks, probes overlap
- Cache entries expire (default timeout), triggering fresh probes
- Under load, this still creates a probe storm as caches churn

Additionally, each probe on cache miss spawns an async task in the router's event loop:
- These probe tasks compete with request handling for event loop time
- Slows down the forward path (client → parent → child)
- Slows down the backward path (child → parent → client)
- Creates a feedback loop: slower responses → more concurrent requests → more probes

### Observed Impact

```
WARNING: Failed to get queue length from Replica(...) within 1.0s
[repeated 11815x across cluster]
```

### Proposed Solution: Centralized Queue Length Service with Background Probing

A two-tier architecture that addresses both problems:

1. **Central Service (Ray Actor)**: One per deployment, probes replicas in background
2. **Process-Level Singleton Cache**: Shared by all routers in same process, pulls from central service

```
                    ┌─────────────────────────────────────┐
                    │   Queue Length Service (Ray Actor)  │
                    │   (One per deployment)              │
                    │                                     │
                    │   - Background probing loop         │
                    │   - Probes each replica periodically│
                    │   - Caches all queue lengths        │
                    └─────────────────────────────────────┘
                           │                    ↑
                      Probe replicas       Pull (periodic)
                           ↓                    │
┌─────────────┐    ┌─────────────────────────────────────────────────┐
│   Replica   │    │              Process (Task/Actor)               │
│   Replica   │    │  ┌─────────────────────────────────────────┐   │
│   Replica   │    │  │  Singleton Cache (process-level)        │   │
└─────────────┘    │  │  - Shared by all routers in process     │   │
                   │  │  - Pulls from central service (async)   │   │
                   │  │  - Zero probe latency for routers       │   │
                   │  └─────────────────────────────────────────┘   │
                   │         ↑              ↑              ↑        │
                   │    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐   │
                   │    │ Router  │    │ Router  │    │ Router  │   │
                   │    │(Handle) │    │(Handle) │    │(Handle) │   │
                   │    └─────────┘    └─────────┘    └─────────┘   │
                   └─────────────────────────────────────────────────┘
```

### Key Design Points

1. **Background probing** - Probing happens in central service, completely out of request path
2. **Process-level batching** - All routers in a process share one cache, no per-router probing
3. **One actor per deployment** - Central service scales with deployments, not with handles
4. **Pull model** - Singleton cache pulls from central actor periodically (in background)

### Existing Router Optimizations (to preserve)

Each router has an important optimization for keeping the cache accurate without constant probing:

**Optimistic Queue Length Tracking**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Router chooses replica based on cached queue length                 │
│                                                                         │
│  2. on_send_request(replica_id)     → Cache[replica] += 1              │
│     Called immediately when sending request                             │
│                                                                         │
│  3. on_new_queue_len_info(replica_id, info) → Cache[replica] = actual  │
│     Called when replica responds with real queue length                 │
└─────────────────────────────────────────────────────────────────────────┘
```

This prevents the "thundering herd to one replica" problem:
- Without: 10 rapid requests all see queue_len=0, all route to same replica
- With: 1st sees 0, 2nd sees 1, 3rd sees 2... requests spread out

**Code locations:**
- `on_send_request()` - `request_router.py` lines 687-693
- `on_new_queue_len_info()` - `request_router.py` lines 675-685  
- Called from `router.py` lines 796 and 817

**With centralized service, routers will:**
- Continue incrementing shared cache on `on_send_request()`
- Receive accurate updates from central service instead of per-router probes
- Maintain locality routing, pow2 selection, and other existing optimizations

### Scaling Analysis: One Actor Per Deployment

**Variables:**
- R = number of replicas
- P = number of processes with routers (tasks/actors making requests)
- I = probe interval (how often central actor probes replicas)
- Q = query interval (how often singleton caches pull from central actor)

**Central Actor Workload:**

| Operation | Rate | Example (R=512, P=100, I=Q=100ms) |
|-----------|------|-----------------------------------|
| Probe replicas | R / I | 512 / 0.1s = **5,120 RPCs/sec** |
| Serve cache queries | P / Q | 100 / 0.1s = **1,000 RPCs/sec** |
| **Total** | (R + P) / I | **6,120 RPCs/sec** |

**Comparison to Current (per-router probing):**

| Metric | Current (per-router) | Centralized |
|--------|---------------------|-------------|
| Probe RPCs/sec | P × 2 × (requests/sec) | R / I |
| Example: 100 processes, 1000 RPS each, 512 replicas | 100 × 2 × 1000 = **200,000** | 512 / 0.1 = **5,120** |
| **Reduction** | - | **~40x fewer probes** |

**Actor Capacity:**
- A Ray actor can handle **10,000-50,000+ simple RPCs/sec** (depending on payload size)
- Queue length response is tiny (~100 bytes for 512 replicas)
- At 6,120 RPCs/sec, the actor is well under capacity

**Scaling Limits:**
- **R < 5,000 replicas**: Single actor handles easily
- **R > 5,000 replicas**: Consider sharding by replica ID range or hierarchical aggregation
- **P > 1,000 processes**: Consider per-node aggregator actors

**Conclusion:** One actor per deployment scales comfortably to ~1,000 replicas and ~500 processes. Beyond that, add per-node aggregators as a second tier.

### Benefits

- **Zero probe latency in request path** - Routers read from local cache instantly
- **O(R) probes instead of O(H × R)** - where R=replicas, H=handles
- **No event loop contention** - No probe tasks competing with request handling
- **Consistent view** - All routers see the same queue length data
- **Graceful degradation** - If cache is stale, fallback to random selection

---
