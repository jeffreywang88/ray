"""Opt-in per-request latency tracing of the ingress router, for benchmarks.

Enabled by ``RAY_SERVE_LLM_KV_TRACE_DIR``: each router process then appends
one JSON line per routed request to ``<dir>/router-<pid>.jsonl`` (Unix-time
stamps), times every task it submits to the renderer's thread pool (queue wait,
run time, thread CPU time), and records its event loop's scheduling lag every
10 s. Off by default; nothing here runs when the variable is unset.
"""

import asyncio
import contextvars
import json
import os
import time
from typing import Any, Dict, List, Optional

TRACE_DIR = os.environ.get("RAY_SERVE_LLM_KV_TRACE_DIR") or None

# The trace record of the request being routed in the current task.
current: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "kv_route_trace", default=None
)


class TraceWriter:
    def __init__(self, name: str):
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{name}-{os.getpid()}.jsonl")
        self._f = open(path, "a", buffering=1)

    def write(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")


def writer(name: str) -> Optional[TraceWriter]:
    return TraceWriter(name) if TRACE_DIR else None


def _fn_name(fn) -> str:
    fn = getattr(fn, "func", fn)  # functools.partial from make_async
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))


def wrap_executor(executor) -> None:
    """Records each task's queue wait, run time and thread CPU time on the
    trace record of the request that submitted it."""
    submit = executor.submit

    def traced_submit(fn, /, *args, **kwargs):
        record = current.get()
        if record is None:
            return submit(fn, *args, **kwargs)
        queued = time.perf_counter()

        def run():
            started = time.perf_counter()
            cpu = time.thread_time()
            try:
                return fn(*args, **kwargs)
            finally:
                pool: List = record.setdefault("pool", [])
                pool.append(
                    (
                        _fn_name(fn),
                        round((started - queued) * 1e3, 3),
                        round((time.perf_counter() - started) * 1e3, 3),
                        round((time.thread_time() - cpu) * 1e3, 3),
                    )
                )

        return submit(run)

    executor.submit = traced_submit


async def monitor_loop_lag(
    out: TraceWriter, interval_s: float = 0.02, window_s: float = 10.0
) -> None:
    """Writes the event loop's scheduling lag (how late a sleep wakes up)."""
    lags: List[float] = []
    window_start = time.time()
    while True:
        t = time.perf_counter()
        await asyncio.sleep(interval_s)
        lags.append((time.perf_counter() - t - interval_s) * 1e3)
        now = time.time()
        if now - window_start >= window_s:
            lags.sort()
            n = len(lags)
            out.write(
                {
                    "kind": "loop_lag",
                    "t": now,
                    "n": n,
                    "mean_ms": round(sum(lags) / n, 3),
                    "p50_ms": round(lags[n // 2], 3),
                    "p99_ms": round(lags[min(n - 1, int(n * 0.99))], 3),
                    "max_ms": round(lags[-1], 3),
                }
            )
            lags = []
            window_start = now
