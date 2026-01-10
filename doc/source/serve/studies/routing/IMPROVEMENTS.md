# Routing Algorithm Improvements

This document captures performance issues discovered during the routing algorithm study and proposes improvements for Ray Serve's request routing.

## Executive Summary

The current routing implementation has significant overhead that impacts request latency:

1. **Queue length probing is in the request path** - Every request waits for queue length probes before being routed
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

- **Added latency**: Every request pays the probe latency cost
- **Timeout accumulation**: Under load, probes queue up and timeout
- **Observed in study**: `client_to_parent_delay_ms` p50 of 24ms when probing is slow

### Proposed Solution: Background Probing

Move queue length probing to a background task that continuously updates a cache:

### Benefits

- **Zero probe latency in request path** - Requests route immediately using cached data
- **Bounded staleness** - Cache refreshed every N ms (configurable)
- **Graceful degradation** - If cache is stale, fallback to random selection

---

## Problem 2: Thundering Herd on Probes

### Current Behavior

Each `DeploymentHandle` has its own router that independently probes replicas:

```
Load Generator Task 1 ─── Handle ─── Router ───┐
Load Generator Task 2 ─── Handle ─── Router ───┼──→ Probe all 512 replicas
Load Generator Task 3 ─── Handle ─── Router ───┤
...                                            │
Load Generator Task 27 ── Handle ─── Router ───┘
```

With 27 tasks and 512 replicas:
- Each task probes 512 replicas
- Total: 27 × 512 = 13,824 probe RPCs per probe cycle
- Under load, this creates a probe storm

### Observed Impact

```
WARNING: Failed to get queue length from Replica(...) within 1.0s
[repeated 11815x across cluster]
```

### Proposed Solution: Centralized Queue Length Service

Create a centralized service that aggregates queue lengths:

```
┌─────────────────────────────────────────────────────────┐
│                  Queue Length Aggregator                │
│  (One per node or one per cluster)                      │
│                                                         │
│  - Probes each replica once per interval                │
│  - Broadcasts updates to all local routers              │
│  - Replicas push updates on significant changes         │
└─────────────────────────────────────────────────────────┘
         ↑                    ↓
    Push updates         Broadcast cache
         │                    │
    ┌────┴────┐          ┌────┴────┐
    │ Replica │          │ Router  │
    │ Replica │          │ Router  │
    │ Replica │          │ Router  │
    └─────────┘          └─────────┘
```

### Alternative: Replica-Push Model

Instead of routers pulling queue lengths, replicas push updates:

```python
class ReplicaActor:
    async def _push_queue_length_loop(self):
        """Push queue length updates when they change significantly."""
        last_pushed = 0
        while True:
            current = self._get_queue_length()
            # Push if queue length changed by more than threshold
            if abs(current - last_pushed) > self._push_threshold:
                await self._broadcast_queue_length(current)
                last_pushed = current
            await asyncio.sleep(0.1)
```

### Benefits

- **O(R) probes instead of O(H × R)** where R=replicas, H=handles
- **Consistent view** - All routers see the same queue length data
- **Reduced network load** - Fewer probe RPCs

---

## Implementation Priority

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| Background probing for Pow2 | High | Medium | P0 |
| Centralized queue length service | Medium | High | P1 |

## Metrics to Track

After implementing improvements, measure:

1. **client_to_parent_delay_ms** - Should drop significantly
2. **Probe RPC count** - Should reduce by orders of magnitude  
3. **Queue length probe timeout rate** - Should approach zero
4. **Request latency percentiles** - p50, p90, p99 should improve

## Related Code Locations

- `ray/python/ray/serve/_private/request_router/request_router.py`
  - `_probe_queue_lens()` - Queue length probing logic
  - `_select_from_candidate_replicas()` - Capacity checking
  - `RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S` - Probe timeout config

- `ray/python/ray/serve/_private/replica.py`
  - Queue length reporting
  - `max_ongoing_requests` handling

## References

- [Power of Two Choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) - Original algorithm paper
- [Ray Serve Architecture](https://docs.ray.io/en/latest/serve/architecture.html)
