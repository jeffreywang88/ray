"""Serve LLM app whose engine replicas are Dynamo Mocker engines.

``MockerServer`` stands in for ``LLMServer`` under KV-aware direct streaming: it
publishes KV events, reads the prompt token ids the router forwards, reports
request lifecycle events to every ``LLMRouter`` replica and serves
``/v1/chat/completions``. The ingress router is the real ``LLMRouter``.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import ray
from ray import serve
from ray.llm._internal.serve.core.ingress.builder import (
    _build_openai_ingress_request_router,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware import token_channel
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_TOKEN_KEY_HEADER,
    KV_TOKEN_METADATA_KEY,
    LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_router import (
    KVAwareRouter,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    get_llm_router_handle,
)
from ray.serve.config import RequestRouterConfig
from ray.serve.deployment import Application
from ray.serve.llm import LLMConfig

logger = get_logger(__name__)

ENGINE_DEPLOYMENT = "LLMServer:mock"
# One token per chunk of streamed text; clients count tokens from `usage`.
TOKEN_TEXT = " the"
DEFAULT_MAX_TOKENS = 256
KV_PORT_BASE = 23000
REPLAY_PORT_OFFSET = 100
TOKEN_PORT_OFFSET = 200


class LifecycleEventForwarder:
    """Copy of ``kv_aware.vllm.token_tracking.LifecycleEventForwarder``, which
    imports vLLM. Batches lifecycle events and broadcasts them in order to every
    LLMRouter replica without blocking generation."""

    def __init__(self, handle):
        self._handle = handle
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
                results = await self._handle.broadcast(
                    "on_lifecycle_events", batch
                ).results_async(
                    timeout_s=LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(
                        "KV lifecycle events dropped on %d/%d routers: %s",
                        len(errors),
                        len(results),
                        errors[0],
                    )
            except Exception as e:
                logger.warning("Dropping KV lifecycle events: %s", e)
            finally:
                for _ in batch:
                    self._events.task_done()


class MockerServer:
    async def __init__(self, cfg: Dict[str, Any]):
        from dynamo._core import MockerWorker

        self._model_id = cfg["model_id"]
        self._stream_interval_s = cfg["stream_interval_s"]
        context = serve.get_replica_context()
        rank = context.rank.local_rank
        ip = ray.util.get_node_ip_address()
        kv_port = KV_PORT_BASE + rank
        replay_port = kv_port + REPLAY_PORT_OFFSET
        token_port = kv_port + TOKEN_PORT_OFFSET
        engine = cfg["mocker"]["engine"]
        self._worker = MockerWorker(cfg["mocker"], kv_port, replay_port)
        self._token_store = token_channel.TokenStore()
        self._token_receiver = token_channel.TokenReceiver(
            bind_endpoint=f"tcp://*:{token_port}", store=self._token_store
        )
        await self._token_receiver.start()
        self._routing_stats = {
            "kv_event_metadata": {
                "endpoint": f"tcp://{ip}:{kv_port}",
                "replay_endpoint": f"tcp://{ip}:{replay_port}",
                "block_size": engine["block_size"],
                "max_num_batched_tokens": engine["max_num_batched_tokens"],
                "dp_rank": 0,
            },
            KV_TOKEN_METADATA_KEY: {"endpoint": f"tcp://{ip}:{token_port}"},
        }
        self._forwarder = LifecycleEventForwarder(get_llm_router_handle())
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

    async def _models(self, request: Request):
        return JSONResponse(
            {"object": "list", "data": [{"id": self._model_id, "object": "model"}]}
        )

    async def _health(self, request: Request):
        return JSONResponse({"status": "ok"})

    def _prompt_token_ids(self, request: Request, body: Dict[str, Any]) -> List[int]:
        key = request.headers.get(KV_TOKEN_KEY_HEADER)
        entry = self._token_store.pop(key) if key else None
        if entry is not None:
            return token_channel.decode_prompt_token_ids(entry.payload)
        # Never matches the router's block hashes, so a missed token-channel
        # message shows up as lost KV hits instead of being hidden.
        text = json.dumps(body.get("messages", ""))
        return [hash(text[i : i + 4]) & 0x7FFFFFFF for i in range(0, len(text), 4)]

    async def _chat(self, request: Request):
        body = json.loads(await request.body())
        context = serve.context._serve_request_context.get()
        request_id = (context.request_id if context else "") or uuid.uuid4().hex
        token_ids = self._prompt_token_ids(request, body)
        max_tokens = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or DEFAULT_MAX_TOKENS
        )
        try:
            live = await self._worker.submit(token_ids, max(1, max_tokens))
        except Exception:
            self._forwarder.report("on_request_completed", request_id)
            raise
        model = body.get("model", self._model_id)
        if body.get("stream"):
            include_usage = bool(
                (body.get("stream_options") or {}).get("include_usage")
            )
            return StreamingResponse(
                self._stream(live, request_id, model, len(token_ids), include_usage),
                media_type="text/event-stream",
            )
        completion, cached = 0, 0
        async for n, cached in self._chunks(live, request_id):
            completion += n
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": TOKEN_TEXT * completion,
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": _usage(len(token_ids), completion, cached),
            }
        )

    async def _chunks(self, live, request_id: str):
        """Yields (new_tokens, cached_tokens) and reports lifecycle events."""
        first = True
        cached = 0
        try:
            while True:
                n, c, done = await live.next_chunk(self._stream_interval_s)
                if c is not None:
                    cached = c
                if n and first:
                    first = False
                    self._forwarder.report("on_prefill_complete", request_id)
                if n:
                    yield n, cached
                if done:
                    break
        finally:
            self._forwarder.report("on_request_completed", request_id)

    async def _stream(self, live, request_id, model, prompt_tokens, include_usage):
        header = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }

        def frame(payload: Dict[str, Any]) -> bytes:
            return f"data: {json.dumps({**header, **payload})}\n\n".encode()

        def delta(content: Dict[str, Any], finish: Optional[str] = None) -> bytes:
            return frame(
                {"choices": [{"index": 0, "delta": content, "finish_reason": finish}]}
            )

        completion, cached = 0, 0
        yield delta({"role": "assistant", "content": ""})
        async for n, cached in self._chunks(live, request_id):
            completion += n
            yield delta({"content": TOKEN_TEXT * n})
        yield delta({}, finish="length")
        if include_usage:
            yield frame(
                {"choices": [], "usage": _usage(prompt_tokens, completion, cached)}
            )
        yield b"data: [DONE]\n\n"


def _usage(prompt: int, completion: int, cached: int) -> Dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def build_app(
    *,
    model_id: str,
    tokenizer: str,
    num_engines: int,
    engine_cpus: float,
    node_ids: List[str],
    mocker_engine: Dict[str, Any],
    stream_interval_s: float,
    max_model_len: int,
    env_vars: Dict[str, str],
) -> Application:
    """Builds the app: ``num_engines`` Mocker replicas pinned to ``node_ids``
    behind ``KVAwareRouter``, with the ``LLMRouter`` ingress request router."""
    from dynamo._core import _normalize_mocker_config

    router_config = RequestRouterConfig(request_router_class=KVAwareRouter)
    cfg = {
        "model_id": model_id,
        "mocker": _normalize_mocker_config({"engine": mocker_engine}),
        "stream_interval_s": stream_interval_s,
    }
    server = serve.deployment(
        serve.ingress()(MockerServer),
        name=ENGINE_DEPLOYMENT,
        num_replicas=num_engines,
        max_ongoing_requests=4096,
        request_router_config=router_config,
        logging_config={"enable_access_log": False, "log_level": "WARNING"},
        ray_actor_options={
            "num_cpus": engine_cpus,
            "label_selector": {"ray.io/node-id": f"in({','.join(node_ids)})"},
            "runtime_env": {
                "env_vars": {
                    **env_vars,
                    "DYN_RUNTIME_NUM_WORKER_THREADS": "1",
                    "DYN_RUNTIME_MAX_BLOCKING_THREADS": "4",
                    "DYN_LOG": "warn",
                    "DYN_MOCKER_KV_COALESCE_MS": "10",
                }
            },
        },
    ).bind(cfg)
    llm_config = LLMConfig(
        model_loading_config={"model_id": model_id, "model_source": tokenizer},
        deployment_config={"request_router_config": router_config},
        engine_kwargs={
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "renderer_num_workers": 4,
        },
        runtime_env={"env_vars": {**env_vars, "DYN_RUNTIME_NUM_WORKER_THREADS": "8"}},
    )
    router = _build_openai_ingress_request_router(server=server, llm_config=llm_config)
    return server._with_ingress_request_router(router)
