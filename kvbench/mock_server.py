"""Lightweight LLMServer stand-in backed by Dynamo's live Mocker.

It plays the engine-replica side of KV-aware direct streaming exactly like
``LLMServer`` + ``VLLMEngine`` do, without importing vLLM or torch (1.4 GiB RSS
per replica), so 1024 replicas fit on one host:

- advertises ``kv_event_metadata`` and ``kv_token_metadata`` in routing stats;
- publishes vLLM-format KV events on a ZMQ PUB + replay ROUTER (Mocker, Rust);
- consumes router-forwarded prompt token ids from the ZMQ token channel;
- reports ``on_prefill_complete`` / ``on_request_completed`` lifecycle events to
  every LLMRouter replica, batched exactly like ``LifecycleEventForwarder``;
- serves OpenAI ``/v1/chat/completions`` on its direct-ingress HTTP port, with
  ``usage.prompt_tokens_details.cached_tokens`` from the Mocker scheduler.

Output after the first token is coalesced (Rust side) into one SSE chunk per
``flush_interval_s`` so the host does not pay Python work per token.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import ray
from ray import serve
from ray.llm._internal.serve.routing_policies.kv_aware import token_channel
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_TOKEN_KEY_HEADER,
    KV_TOKEN_METADATA_KEY,
)

try:  # Only the broadcast copy below uses it; mesh-only trees dropped it.
    from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
        LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
    )
except ImportError:
    LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S = 3
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    get_llm_router_handle,
    get_worker_id,
)

logger = logging.getLogger("ray.serve")

# A word the Qwen3 tokenizer encodes as exactly one token, so clients that count
# output tokens by re-tokenizing the text see the same count as usage reports.
_TOKEN_TEXT = " the"

# BENCH_TRACE=1: one JSON line per request to $BENCH_TRACE_DIR/eng_<pid>.jsonl with the
# HAProxy (x-bt-hq/h0/h1) and router (x-bt-r) stamps it carried, plus e0 handler entry,
# e1 tokens resolved (before submit), e2 first generated tokens, e3 first content frame.
_TRACE = os.environ.get("BENCH_TRACE") == "1"
_TRACE_DIR = os.environ.get("BENCH_TRACE_DIR", "/tmp")


class _LocalLifecycleEventForwarder:
    """Verbatim copy of ``kv_aware.vllm.token_tracking.LifecycleEventForwarder``
    for Ray trees where it lives only in that vLLM-importing module."""

    def __init__(self, handle, worker_id: int):
        self.handle = handle
        self.worker_id = worker_id
        self._events: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None

    def report(self, method_name: str, *args) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._events.put_nowait((method_name, args))

    async def _deliver(self) -> None:
        while True:
            batch = [await self._events.get()]
            while not self._events.empty():
                batch.append(self._events.get_nowait())
            try:
                results = await self.handle.broadcast(
                    "on_lifecycle_events", batch
                ).results_async(
                    timeout_s=LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(
                        "KV lifecycle events dropped on %d/%d ingress replicas: %s",
                        len(errors),
                        len(results),
                        errors[0],
                    )
            except Exception as e:
                logger.warning("Dropping KV lifecycle events: %s", e)
            finally:
                for _ in batch:
                    self._events.task_done()


try:  # Ray trees reporting to the selecting router's replica over a pinned handle.
    from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
        KV_LIFECYCLE_OWNER_HEADER as KV_LIFECYCLE_ENDPOINT_HEADER,
    )
    from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
        LifecycleEventForwarder,
    )
    _FORWARDER_TAKES_WORKER_ID = False
except ImportError:
    _FORWARDER_TAKES_WORKER_ID = True
    try:  # Ray trees with the engine-agnostic lifecycle module (replica-mesh ZMQ mode).
        from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
            KV_LIFECYCLE_ENDPOINT_HEADER,
        )
        from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
            LifecycleEventForwarder,
        )
    except ImportError:
        KV_LIFECYCLE_ENDPOINT_HEADER = None
        LifecycleEventForwarder = _LocalLifecycleEventForwarder


def _report(forwarder, owner_endpoint, method_name, *args) -> None:
    if hasattr(forwarder, "report_to"):
        forwarder.report_to(owner_endpoint, method_name, *args)
    else:
        forwarder.report(method_name, *args)


def _serve_request_id() -> str:
    context = serve.context._serve_request_context.get()
    return context.request_id if context is not None else ""


def _fallback_token_ids(body: Dict[str, Any]) -> List[int]:
    """Stand-in prompt when the token channel missed: never matches router hashes,
    so a miss shows up as lost KV hits instead of being hidden."""
    text = json.dumps(body.get("messages", ""))
    return [hash(text[i : i + 4]) & 0x7FFFFFFF for i in range(0, len(text), 4)]


class MockerServer:
    async def __init__(self, cfg: Dict[str, Any]):
        from dynamo._core import MockerWorker

        self._cfg = cfg
        context = serve.get_replica_context()
        # Global rank: ports stay unique even when several Ray nodes share one host.
        rank = context.rank.rank
        ip = ray.util.get_node_ip_address()
        kv_port = cfg["kv_port_base"] + rank
        replay_port = cfg["replay_port_base"] + rank
        token_port = cfg["token_port_base"] + rank
        self._worker = MockerWorker(cfg["mocker"], kv_port, replay_port)
        engine = cfg["mocker"]["engine"]
        self._flush_interval_s = cfg.get("flush_interval_s", 0.05)
        self._output_token_scale = cfg.get("output_token_scale", 1.0)
        self._default_max_tokens = cfg.get("default_max_tokens", 256)

        self._token_store = token_channel.TokenStore()
        self._token_receiver = token_channel.TokenReceiver(
            bind_endpoint=f"tcp://*:{token_port}", store=self._token_store
        )
        await self._token_receiver.start()

        # Experiment knob: advertise a KV-event address nothing publishes on, so routers still register
        # this engine (and route by load) but ingest no KV events. Isolates the routers' ingestion cost.
        advertised_kv_port = kv_port + 40000 if cfg.get("no_kv_events_to_router") else kv_port
        self._routing_stats = {
            "kv_event_metadata": {
                "endpoint": f"tcp://{ip}:{advertised_kv_port}",
                "replay_endpoint": f"tcp://{ip}:{replay_port}",
                "block_size": engine["block_size"],
                "max_num_batched_tokens": engine["max_num_batched_tokens"],
                "dp_rank": 0,
            },
            KV_TOKEN_METADATA_KEY: {"endpoint": f"tcp://{ip}:{token_port}"},
        }
        self._worker_id = get_worker_id(context.replica_id.unique_id)
        self._forwarder = (
            LifecycleEventForwarder(get_llm_router_handle(), self._worker_id)
            if _FORWARDER_TAKES_WORKER_ID
            else LifecycleEventForwarder(get_llm_router_handle())
        )

        self._inflight: Dict[str, float] = {}
        self._trace: Dict[str, Dict[str, Any]] = {}
        self._trace_buf: List[Dict[str, Any]] = []
        # Bounded record of request ids seen here, for leak diagnosis:
        # id -> [started, prefill_reported, completed_reported].
        self._seen: "OrderedDict[str, List[bool]]" = OrderedDict()
        self._stats = {
            "requests": 0,
            "completed": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "token_channel_hits": 0,
            "token_channel_misses": 0,
            "token_key_missing": 0,
            "token_store_misses": 0,
            # Engine-side TTFT: arrival at this replica -> first generated tokens (sum, count).
            # AgentX's TTFT minus this is the time spent before the engine (HAProxy + router).
            "engine_ttft_s": 0.0,
            "engine_ttft_n": 0,
        }
        self._app = Starlette(
            routes=[
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/v1/models", self._models, methods=["GET"]),
                Route("/health", self._health, methods=["GET"]),
            ]
        )

    async def __serve_build_asgi_app__(self):
        return self._app

    async def record_routing_stats(self) -> Dict[str, Any]:
        return self._routing_stats

    async def check_health(self) -> None:
        return None

    def _trace_flush(self) -> None:
        if not self._trace_buf:
            return
        lines = "".join(json.dumps(r) + "\n" for r in self._trace_buf)
        self._trace_buf.clear()
        with open(os.path.join(_TRACE_DIR, f"eng_{os.getpid()}.jsonl"), "a") as f:
            f.write(lines)

    def get_stats(self) -> Dict[str, Any]:
        if _TRACE:
            self._trace_flush()
        stored, removed = self._worker.kv_event_counts()
        return {
            "worker_id": self._worker_id,
            "time": time.time(),
            "inflight": len(self._inflight),
            "inflight_ids": list(self._inflight),
            "engine_active": self._worker.active_request_count(),
            "kv_stored_blocks": stored,
            "kv_removed_blocks": removed,
            **self._stats,
        }

    async def _models(self, request: Request):
        model = self._cfg["model_id"]
        return JSONResponse(
            {"object": "list", "data": [{"id": model, "object": "model"}]}
        )

    async def _health(self, request: Request):
        return JSONResponse({"status": "ok"})

    async def _chat(self, request: Request):
        e0 = time.time() if _TRACE else None
        body = json.loads(await request.body())
        request_id = _serve_request_id() or uuid.uuid4().hex
        owner_endpoint = (
            request.headers.get(KV_LIFECYCLE_ENDPOINT_HEADER)
            if KV_LIFECYCLE_ENDPOINT_HEADER
            else None
        )
        token_ids = None
        token_key = request.headers.get(KV_TOKEN_KEY_HEADER)
        if token_key:
            entry = self._token_store.pop(token_key)
            if entry is not None:
                token_ids = token_channel.decode_prompt_token_ids(entry.payload)
            else:
                self._stats["token_store_misses"] += 1
        else:
            self._stats["token_key_missing"] += 1
        tok_hit = bool(token_ids)
        if token_ids:
            self._stats["token_channel_hits"] += 1
        else:
            self._stats["token_channel_misses"] += 1
            token_ids = _fallback_token_ids(body)

        requested = (
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or self._default_max_tokens
        )
        max_tokens = max(1, int(round(requested * self._output_token_scale)))
        self._stats["requests"] += 1
        self._inflight[request_id] = time.time()
        if _TRACE:
            h = request.headers
            self._trace[request_id] = {
                "rid": h.get("x-request-id"), "hq": h.get("x-bt-hq"), "h0": h.get("x-bt-h0"),
                "h1": h.get("x-bt-h1"), "r": h.get("x-bt-r"), "e0": e0,
                "e1": self._inflight[request_id], "tok": tok_hit, "plen": len(token_ids),
            }
        self._seen[request_id] = [True, False, False]
        if len(self._seen) > 200_000:
            self._seen.popitem(last=False)
        try:
            live = await self._worker.submit(token_ids, max_tokens)
        except Exception:
            self._inflight.pop(request_id, None)
            self._stats["errors"] += 1
            _report(self._forwarder, owner_endpoint, "on_request_completed", request_id)
            raise

        model = body.get("model", self._cfg["model_id"])
        created = int(time.time())
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                self._stream(
                    live,
                    request_id,
                    model,
                    created,
                    len(token_ids),
                    include_usage,
                    owner_endpoint,
                ),
                media_type="text/event-stream",
            )
        completion, cached = await self._drain(
            live, request_id, len(token_ids), owner_endpoint
        )
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _TOKEN_TEXT * completion,
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": self._usage(len(token_ids), completion, cached),
            }
        )

    def _usage(self, prompt: int, completion: int, cached: int) -> Dict[str, Any]:
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        }

    async def _chunks(self, live, request_id: str, owner_endpoint=None):
        """Yield (new_tokens, cached_tokens) and drive lifecycle events."""
        first = True
        cached = 0
        try:
            while True:
                n, c, done = await live.next_chunk(self._flush_interval_s)
                if c is not None:
                    cached = c
                if n and first:
                    first = False
                    arrived = self._inflight.get(request_id)
                    if arrived is not None:
                        self._stats["engine_ttft_s"] += time.time() - arrived
                        self._stats["engine_ttft_n"] += 1
                    if _TRACE and request_id in self._trace:
                        self._trace[request_id]["e2"] = time.time()
                    _report(
                        self._forwarder,
                        owner_endpoint,
                        "on_prefill_complete",
                        request_id,
                    )
                    self._mark(request_id, 1)
                if n:
                    yield n, cached
                if done:
                    break
            self._stats["completed"] += 1
        finally:
            self._inflight.pop(request_id, None)
            if _TRACE:
                self._trace.pop(request_id, None)
            _report(self._forwarder, owner_endpoint, "on_request_completed", request_id)
            self._mark(request_id, 2)

    def _mark(self, request_id: str, index: int) -> None:
        entry = self._seen.get(request_id)
        if entry is not None:
            entry[index] = True

    def seen(self, request_ids: List[str]) -> Dict[str, List[bool]]:
        return {r: self._seen[r] for r in request_ids if r in self._seen}

    def _record_completion(self, prompt: int, completion: int, cached: int) -> None:
        self._stats["prompt_tokens"] += prompt
        self._stats["completion_tokens"] += completion
        self._stats["cached_tokens"] += cached

    async def _drain(self, live, request_id: str, prompt: int, owner_endpoint=None):
        completion, cached = 0, 0
        async for n, cached in self._chunks(live, request_id, owner_endpoint):
            completion += n
        self._record_completion(prompt, completion, cached)
        return completion, cached

    async def _stream(
        self, live, request_id, model, created, prompt, include_usage, owner_endpoint
    ):
        def frame(delta: Dict[str, Any], finish: Optional[str] = None) -> bytes:
            return (
                "data: "
                + json.dumps(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                )
                + "\n\n"
            ).encode()

        completion, cached = 0, 0
        yield frame({"role": "assistant", "content": ""})
        async for n, cached in self._chunks(live, request_id, owner_endpoint):
            completion += n
            if _TRACE:
                rec = self._trace.pop(request_id, None)
                if rec is not None:
                    rec["e3"] = time.time()
                    rec["cached"] = cached
                    self._trace_buf.append(rec)
                    if len(self._trace_buf) >= 200:
                        self._trace_flush()
            yield frame({"content": _TOKEN_TEXT * n})
        self._record_completion(prompt, completion, cached)
        yield frame({}, finish="length")
        if include_usage:
            usage_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": self._usage(prompt, completion, cached),
            }
            yield ("data: " + json.dumps(usage_chunk) + "\n\n").encode()
        yield b"data: [DONE]\n\n"
