"""Benchmark-only instrumentation of the KV-aware ingress path.

Installed from ``sitecustomize`` when ``BENCH_INSTRUMENT=1`` (set only on the
LLMRouter deployment). Patches are applied right after the target modules are
imported, so Ray's source stays untouched and baseline and optimized trees get
the same probes. ``LLMRouter.bench_stats()`` returns a snapshot.

Probes (cumulative counters unless noted; the monitor takes deltas):
- res_arrived: peer reservations received by ``on_reservations_created``.
- res_processed / res_applied: reservations reaching ``_apply_reservations`` /
  booked through ``create_reservation``; skipped = processed - applied.
- lag: wall time from the selecting ingress's ``select_worker`` to the moment a
  peer books the reservation (stamped into the broadcast as ``_t_sel``).
- lifecycle_events / lifecycle_batches.
- select / tokenize / pick latency samples (seconds), reset on each read.
- loop_cpu_s: CPU time of the event-loop thread (main thread).
- sem_held / sem_waiters / sem_max: the Serve replica's max_ongoing_requests semaphore (slots
  held, requests waiting for a slot, limit). Every request into the replica, HTTP or handle
  call, takes a slot first, before any router code runs.
- loop_lag: how late a 50 ms sleep on the router's event loop wakes up (seconds). CPU time
  cannot show a loop that waits for the GIL or works through a long queue of ready callbacks.
- load_view: tracked in-flight requests per worker id.
"""

import asyncio
import importlib.abc
import os
import sys
import threading
import time

_TRACKER = "ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker"
_ROUTER = "ray.llm._internal.serve.core.ingress.router"
_TOKENIZER = "ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer"

# BENCH_TRACE=1: per-request wall-clock stamps through the route handler, returned to
# HAProxy as an "x-bt-r" request header that reaches the engine (mock_server logs it).
# Stamps (time.time()): r0 route entry, r1 body read, r2 tokenize call, r3/r4 worker
# thread start/end, r5 tokenize resumed on the loop, r6 _pick_replica entry, r7
# KVTokenTracker.select_worker entry, r8/r9 select_and_reserve call/resume, r10
# select_worker return, r11 _pick_replica resumed, r12 route return.
TRACE = os.environ.get("BENCH_TRACE") == "1"
_TRACE_KEYS = ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12")
_TRACE_VAR = None
_TRACE_BY_TOKENS = {}
if TRACE:
    import contextvars

    _TRACE_VAR = contextvars.ContextVar("bench_trace", default=None)


def _stamp(trace, key):
    if trace is not None and key not in trace:
        trace[key] = time.time()


class _TracingExecutor:
    """Wraps the renderer's executor: stamps worker start/end into the submitting request's trace."""

    def __init__(self, executor):
        self._executor = executor

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def submit(self, fn, *args, **kwargs):
        trace = _TRACE_VAR.get()

        def run():
            _stamp(trace, "r3")
            try:
                return fn(*args, **kwargs)
            finally:
                if trace is not None:
                    trace["r4"] = time.time()
                    trace["hops"] = trace.get("hops", 0) + 1

        return self._executor.submit(run)

STATS = {
    "res_reported": 0,
    "res_arrived": 0,
    "res_arrival_batches": 0,
    "res_processed": 0,
    "res_applied": 0,
    "lifecycle_events": 0,
    "lifecycle_batches": 0,
    "selects": 0,
    "routes": 0,
}
SAMPLES = {"lag": [], "select": [], "tokenize": [], "pick": [], "lifecycle": [], "loop_lag": [], "xwake": []}
_MAX_SAMPLES = 200_000
_MAIN_THREAD_ID = None
_LAG_PROBE = None
_LAG_PERIOD_S = 0.05


async def _loop_lag_probe():
    loop = asyncio.get_running_loop()
    while True:
        t = loop.time()
        await asyncio.sleep(_LAG_PERIOD_S)
        _sample("loop_lag", loop.time() - t - _LAG_PERIOD_S)


def _xwake_probe(loop):
    """BENCH_TRACE only: every 10 ms post a callback to the loop from another thread and record
    how late it runs. This is the wake-up cost a worker or Rust thread pays to hand a result back."""
    def run():
        while True:
            time.sleep(0.01)
            t = time.perf_counter()
            try:
                loop.call_soon_threadsafe(lambda t=t: _sample("xwake", time.perf_counter() - t))
            except RuntimeError:
                return

    threading.Thread(target=run, name="bench-xwake", daemon=True).start()


def _sample(name, value):
    bucket = SAMPLES[name]
    if len(bucket) < _MAX_SAMPLES:
        bucket.append(value)


def _summ(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": sum(values) / n,
        "p50": values[n // 2],
        "p90": values[min(n - 1, int(n * 0.9))],
        "p99": values[min(n - 1, int(n * 0.99))],
        "max": values[-1],
    }


def _thread_cpu(thread_id):
    try:
        clock = time.pthread_getcpuclockid(thread_id)
        return time.clock_gettime(clock)
    except Exception:
        return None


def _replica_slots():
    """Slot use of this replica's max_ongoing_requests semaphore.

    Reached through Ray's actor instance (ReplicaActor._replica_impl._semaphore): Serve
    gc.freeze()s replicas after startup, so a gc.get_objects() scan cannot see it.
    """
    try:
        import ray

        for actor in ray._private.worker.global_worker.actors.values():
            sem = getattr(getattr(actor, "_replica_impl", None), "_semaphore", None)
            if sem is not None:
                return {
                    "sem_held": sem._value,
                    "sem_waiters": sum(1 for w in (sem._waiters or ()) if not w.cancelled()),
                    "sem_max": sem.get_max_value(),
                }
    except Exception:
        pass
    return {}


def snapshot(tracker):
    samples = {k: _summ(v) for k, v in SAMPLES.items()}
    for v in SAMPLES.values():
        v.clear()
    out = dict(STATS)
    out["time"] = time.time()
    out["samples"] = samples
    out["loop_cpu_s"] = _thread_cpu(_MAIN_THREAD_ID) if _MAIN_THREAD_ID else None
    out["process_cpu_s"] = time.process_time()
    out.update(_replica_slots())
    out["switch_interval_s"] = sys.getswitchinterval()
    if tracker is not None:
        # Mesh-only trees dropped the Ray-RPC replication state; fall back to the
        # Serve replica rank so per-router rows stay distinct.
        rank = getattr(tracker, "_ingress_replica_rank", None)
        if rank is None:
            try:
                from ray import serve

                rank = serve.get_replica_context().rank.rank
            except Exception:
                pass
        out["rank"] = rank
        out["tracked_requests"] = len(tracker._requests)
        completed = getattr(tracker, "_completed_request_ids", None)
        out["completed_ids"] = len(completed) if completed is not None else None
        updates = getattr(tracker, "_reservation_updates", None)
        out["update_queue_batches"] = updates.qsize() if updates is not None else None
        fwd = getattr(tracker, "_reservation_forwarder", None)
        out["outgoing_queue"] = fwd._reservations.qsize() if fwd is not None else None
        out["load_view"] = {
            str(w): len(ids) for w, ids in tracker._request_ids_by_worker.items()
        }
        out["num_workers"] = len(tracker._replica_id_by_worker)
        # The selection service's own per-worker view: own + peers' requests in
        # either replication mode. This is what selection scores against.
        svc_view = {}
        try:
            for model in tracker._svc.loads():
                for load in model.get("loads", []):
                    svc_view[str(load["worker_id"])] = load.get("active_requests", 0)
        except Exception:
            pass
        out["svc_load_view"] = svc_view
        now = time.monotonic()
        stale = []
        for rid, st in tracker._requests.items():
            age = now - st.created_at
            if age < STALE_AGE_S:
                break
            stale.append([rid, str(st.worker_id), round(age, 1), st.prefill_completed])
        out["stale"] = stale[:500]
        out["stale_count"] = len(stale)
    return out


STALE_AGE_S = float(os.environ.get("BENCH_STALE_AGE_S", "600"))


class _SvcProxy:
    """Forwards to the Dynamo SelectionService, timing the calls we care about."""

    def __init__(self, svc):
        self._svc = svc
        self.pending_tsel = {}
        # Only where the service has it: the tracker feature-detects this method.
        if hasattr(svc, "select_and_reserve_blocking"):
            self.select_and_reserve_blocking = self._select_and_reserve_blocking

    def _select_and_reserve_blocking(self, request):
        trace = _TRACE_VAR.get() if TRACE else None
        _stamp(trace, "r8")
        t = time.perf_counter()
        try:
            return self._svc.select_and_reserve_blocking(request)
        finally:
            _sample("select", time.perf_counter() - t)
            STATS["selects"] += 1
            _stamp(trace, "r9")

    def __getattr__(self, name):
        return getattr(self._svc, name)

    async def create_reservation(self, request):
        t_sel = self.pending_tsel.pop(request["selection_id"], None)
        if t_sel is not None:
            _sample("lag", time.time() - t_sel)
        STATS["res_applied"] += 1
        return await self._svc.create_reservation(request)

    def select_and_reserve(self, request):
        # A plain method returning the service's own future: an ``async def`` wrapper
        # would be a Task, adding two event-loop iterations to every selection.
        trace = _TRACE_VAR.get() if TRACE else None
        _stamp(trace, "r8")
        t = time.perf_counter()
        fut = self._svc.select_and_reserve(request)

        def done(_):
            _sample("select", time.perf_counter() - t)
            STATS["selects"] += 1
            _stamp(trace, "r9")

        fut.add_done_callback(done)
        return fut


def _patch_tracker(mod):
    cls = mod.KVTokenTracker
    fwd_cls = mod.ReservationBroadcastForwarder

    orig_create = cls._create_selection_service

    def _create_selection_service(self):
        # The tracker is built on the replica's asyncio loop thread, which is
        # not the process main thread in a Serve replica.
        global _MAIN_THREAD_ID, _LAG_PROBE
        _MAIN_THREAD_ID = threading.get_ident()
        if _LAG_PROBE is None:
            try:
                _LAG_PROBE = asyncio.get_running_loop().create_task(_loop_lag_probe())
                if TRACE:
                    _xwake_probe(asyncio.get_running_loop())
            except RuntimeError:  # not on a running loop: no probe
                pass
        orig_create(self)
        if self._svc is not None:
            self._svc = _SvcProxy(self._svc)
        mod._BENCH_TRACKER = self

    cls._create_selection_service = _create_selection_service

    orig_report = fwd_cls.report

    def report(self, reservation):
        reservation["_t_sel"] = time.time()
        STATS["res_reported"] += 1
        return orig_report(self, reservation)

    fwd_cls.report = report

    orig_created = cls.on_reservations_created

    async def on_reservations_created(self, reservations):
        own = self._ingress_replica_rank
        STATS["res_arrived"] += sum(
            1 for r in reservations if r["source_ingress_replica_rank"] != own
        )
        STATS["res_arrival_batches"] += 1
        return await orig_created(self, reservations)

    cls.on_reservations_created = on_reservations_created

    orig_apply = cls._apply_reservations

    async def _apply_reservations(self, reservations):
        STATS["res_processed"] += len(reservations)
        if isinstance(self._svc, _SvcProxy):
            for r in reservations:
                t_sel = r.get("_t_sel")
                if t_sel is not None:
                    self._svc.pending_tsel[r["request_id"]] = t_sel
        try:
            return await orig_apply(self, reservations)
        finally:
            if isinstance(self._svc, _SvcProxy):
                for r in reservations:
                    self._svc.pending_tsel.pop(r["request_id"], None)

    cls._apply_reservations = _apply_reservations

    orig_lifecycle = cls.on_lifecycle_events

    async def on_lifecycle_events(self, events):
        t = time.perf_counter()
        STATS["lifecycle_batches"] += 1
        STATS["lifecycle_events"] += len(events)
        try:
            return await orig_lifecycle(self, events)
        finally:
            _sample("lifecycle", time.perf_counter() - t)

    cls.on_lifecycle_events = on_lifecycle_events

    if TRACE:
        orig_select = cls.select_worker

        async def select_worker(self, request_id, token_ids, *args, **kwargs):
            # The routing task may serve another request than the one whose context it
            # inherited, so find the trace by the token list the route passed down.
            trace = _TRACE_BY_TOKENS.get(id(token_ids))
            _stamp(trace, "r7")
            token = _TRACE_VAR.set(trace)
            try:
                return await orig_select(self, request_id, token_ids, *args, **kwargs)
            finally:
                _TRACE_VAR.reset(token)
                _stamp(trace, "r10")

        cls.select_worker = select_worker


def _patch_router(mod):
    cls = mod.LLMRouter
    orig_pick = cls._pick_replica

    async def _pick_replica(self, *args, **kwargs):
        trace = _TRACE_VAR.get() if TRACE else None
        _stamp(trace, "r6")
        t = time.perf_counter()
        STATS["routes"] += 1
        try:
            return await orig_pick(self, *args, **kwargs)
        finally:
            _sample("pick", time.perf_counter() - t)
            _stamp(trace, "r11")

    cls._pick_replica = _pick_replica

    if TRACE:
        orig_route = cls.route
        orig_parse = mod._parse_request_body
        opt_field = mod.RAY_SERVE_INGRESS_REQUEST_ROUTER_OPT_HEADERS_FIELD

        def _parse_request_body(body):
            _stamp(_TRACE_VAR.get(), "r1")
            return orig_parse(body)

        mod._parse_request_body = _parse_request_body
        # serve.ingress copies the route method with its own globals dict.
        route_globals = getattr(orig_route, "__globals__", {})
        if route_globals.get("_parse_request_body") is orig_parse:
            route_globals["_parse_request_body"] = _parse_request_body

        async def route(self, request):
            trace = {"r0": time.time()}
            token = _TRACE_VAR.set(trace)
            try:
                response = await orig_route(self, request)
            finally:
                _TRACE_VAR.reset(token)
                for k in [k for k, v in _TRACE_BY_TOKENS.items() if v is trace]:
                    del _TRACE_BY_TOKENS[k]
            trace["r12"] = time.time()
            if isinstance(response, dict):
                stamps = ",".join(
                    "%.6f" % trace[k] if k in trace else "" for k in _TRACE_KEYS
                )
                response.setdefault(opt_field, {})["x-bt-r"] = (
                    f"{stamps};{trace.get('hops', 0)}"
                )
            return response

        cls.route = route

    def bench_stats(self):
        return snapshot(self._kv_token_tracker)

    cls.bench_stats = bench_stats


def _patch_tokenizer(mod):
    cls = mod.Tokenizer
    orig = cls.tokenize

    async def tokenize(self, *args, **kwargs):
        trace = _TRACE_VAR.get() if TRACE else None
        _stamp(trace, "r2")
        t = time.perf_counter()
        result = None
        try:
            result = await orig(self, *args, **kwargs)
            return result
        finally:
            _sample("tokenize", time.perf_counter() - t)
            if trace is not None:
                trace["r5"] = time.time()
                if result is not None:
                    _TRACE_BY_TOKENS[id(result)] = trace

    cls.tokenize = tokenize

    if TRACE:
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            impl = self._renderer.renderer
            if getattr(impl, "_executor", None) is not None:
                impl._executor = _TracingExecutor(impl._executor)

        cls.__init__ = __init__


_TOKENIZER_POOL = "ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer_pool"


def _patch_tokenizer_pool(mod):
    """Tokenization in worker processes: r2 submitted, r5 resumed (no worker-side stamps)."""
    cls = mod.TokenizerProcessPool
    orig = cls.tokenize

    async def tokenize(self, payload, body, request_path):
        trace = _TRACE_VAR.get() if TRACE else None
        _stamp(trace, "r2")
        t = time.perf_counter()
        result = None
        try:
            result = await orig(self, payload, body, request_path)
            return result
        finally:
            _sample("tokenize", time.perf_counter() - t)
            if trace is not None:
                trace["r5"] = time.time()
                if result is not None:
                    _TRACE_BY_TOKENS[id(result)] = trace

    cls.tokenize = tokenize


def _patch_router_stats(mod):
    """BENCH_INSTRUMENT=stats: only the read-only ``bench_stats`` the monitor polls
    (load views, tracked requests, process CPU), no per-request probes."""

    def bench_stats(self):
        return snapshot(self._kv_token_tracker)

    mod.LLMRouter.bench_stats = bench_stats


_PATCHES = {_TRACKER: _patch_tracker, _ROUTER: _patch_router, _TOKENIZER: _patch_tokenizer,
            _TOKENIZER_POOL: _patch_tokenizer_pool}
if os.environ.get("BENCH_INSTRUMENT") == "stats":
    _PATCHES = {_ROUTER: _patch_router_stats}


class _PatchingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is None:
                continue
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _patch=patch):
                _orig(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec
        return None


def install():
    if not any(isinstance(f, _PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchingFinder())
    # Experiment knob (router processes only): GIL switch interval, default 0.005 s.
    if os.environ.get("BENCH_SWITCH_INTERVAL_S"):
        sys.setswitchinterval(float(os.environ["BENCH_SWITCH_INTERVAL_S"]))


if os.environ.get("BENCH_INSTRUMENT") in ("1", "stats"):
    install()
