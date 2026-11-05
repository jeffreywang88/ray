"""Serve LLM app whose engine replicas are Dynamo Mocker engines.

``MockerServer`` stands in for ``LLMServer`` + vLLM under KV-aware direct
streaming. The ingress router is the real ``LLMRouter``. Everything the router
observes matches a vLLM replica:

- KV events: the Mocker publishes vLLM-format events on ZMQ, advertised in
  ``record_routing_stats``.
- Prompt: the engine runs the token ids the router hashed. They normally come
  over the token channel. When they don't (HAProxy truncated the body so the
  router could not tokenize it, the router's push to a not-yet-connected
  engine failed, or the payload had not arrived yet), the engine renders and
  tokenizes the request with the router's own vLLM renderer, as vLLM's OpenAI
  server does. A stand-in prompt there would miss every KV block the router
  indexed for the session and seed the index with blocks no request shares.
- Lifecycle events: reported with the installed Ray's protocol (see
  ``LifecycleReporter``).

Prompt sources are counted in ``kvbench_mocker_prompts{source}``. Each
response names its engine replica in ``x-kvbench-engine``;
``kvbench_mocker_session_requests{source}`` counts requests by whether the
engine served their session before, and ``kvbench_mocker_ttft_ms`` is the
engine's own TTFT (request arrival at the engine to first token).
"""

import asyncio
import collections
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
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_router import (
    KVAwareRouter,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    get_worker_id,
)
from ray.serve._private.constants import SERVE_SESSION_ID
from ray.serve.config import RequestRouterConfig
from ray.serve.deployment import Application
from ray.serve.llm import LLMConfig
from ray.util import metrics

logger = get_logger(__name__)

ENGINE_DEPLOYMENT = "LLMServer:mock"
CHAT_PATH = "/v1/chat/completions"
# One token per chunk of streamed text; clients count tokens from `usage`.
TOKEN_TEXT = " the"
DEFAULT_MAX_TOKENS = 256
KV_PORT_BASE = 23000
REPLAY_PORT_OFFSET = 100
TOKEN_PORT_OFFSET = 200

# Where a request's prompt token ids came from (the `source` tag).
PROMPTS_METRIC = "kvbench_mocker_prompts"
SOURCE_CHANNEL = "token_channel"
# The router staged no tokens: it could not tokenize (e.g. a truncated body)
# or its push failed (e.g. the first push to an engine, before ZMQ connects).
SOURCE_NO_KEY = "engine_no_key"
# The router staged tokens, but they were not in the store when the HTTP
# request arrived.
SOURCE_STORE_MISS = "engine_store_miss"

# Requests by whether this engine served their session before (the `source`
# tag): with session pinning, a session's turns after its first stay "repeat".
SESSIONS_METRIC = "kvbench_mocker_session_requests"
SESSION_NEW = "new"
SESSION_REPEAT = "repeat"
SESSION_NONE = "none"
# Sessions an engine remembers (oldest dropped first); a 25-minute run sees a
# few thousand per engine.
MAX_SESSIONS = 200_000
# Engine-side TTFT: from the request reaching the engine to its first token,
# i.e. without the router's tokenization and routing.
TTFT_METRIC = "kvbench_mocker_ttft_ms"
TTFT_BOUNDARIES_MS = (
    [5, 10, *range(20, 100, 10), *range(100, 300, 25), *range(300, 500, 50)]
    + [*range(500, 1000, 100), *range(1000, 2000, 250), 2000, 2500, 3000, 4000]
    + [5000, 7500, 10000, 15000, 20000, 30000, 60000]
)
# Response header naming the engine replica that served the request.
ENGINE_HEADER = "x-kvbench-engine"


class LifecycleReporter:
    """Reports lifecycle events with the protocol of the installed Ray.

    - Replica mesh (ray-project/ray#67791): the selecting router names its
      endpoint in a request header and the engine pushes that request's events
      to it over ZMQ. A request without the header (HAProxy's fallback path)
      reports nothing.
    - Otherwise (Ray master): the engine broadcasts every request's events to
      all LLMRouter replicas over Serve RPC.
    """

    def __init__(self, replica_unique_id: str):
        try:
            from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
                KV_LIFECYCLE_OWNER_HEADER,
            )
            from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
                LifecycleEventForwarder,
            )
        except ImportError:
            # Only Ray without the replica mesh has the router handle helper.
            from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (  # noqa: E501
                get_llm_router_handle,
            )
            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.token_tracking import (  # noqa: E501
                LifecycleEventForwarder,
            )

            self.protocol = "broadcast"
            self._owner_header = None
            self._forwarder = LifecycleEventForwarder(
                get_llm_router_handle(), get_worker_id(replica_unique_id)
            )
        else:
            self.protocol = "owner_zmq"
            self._owner_header = KV_LIFECYCLE_OWNER_HEADER
            self._forwarder = LifecycleEventForwarder()

    def owner(self, request: Request) -> Optional[str]:
        """The request's report target; ``None`` means don't report."""
        if self._owner_header is None:
            return ""
        return request.headers.get(self._owner_header)

    def report(self, owner: Optional[str], hook: str, request_id: str) -> None:
        if owner is None:
            return
        if self._owner_header is None:
            self._forwarder.report(hook, request_id)
        else:
            self._forwarder.report(owner, hook, request_id)


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
        self._engine_id = context.replica_id.unique_id
        self._lifecycle = LifecycleReporter(self._engine_id)
        # The router's own renderer, built from the same LLMConfig, so a
        # fallback tokenization yields exactly the ids the router hashes.
        from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (  # noqa: E501
            TokenizeError,
            Tokenizer,
        )

        self._tokenize_error = TokenizeError
        self._tokenizer = await asyncio.to_thread(Tokenizer, cfg["llm_config"])
        self._prompts = metrics.Counter(
            PROMPTS_METRIC,
            description="Mocker requests by where their prompt token ids came from.",
            tag_keys=("source",),
        )
        self._sessions: "collections.OrderedDict[str, None]" = collections.OrderedDict()
        self._session_requests = metrics.Counter(
            SESSIONS_METRIC,
            description="Mocker requests by whether this engine served their "
            "session (the session-id header) before.",
            tag_keys=("source",),
        )
        self._ttft = metrics.Histogram(
            TTFT_METRIC,
            description="Mocker TTFT from the request reaching the engine to its "
            "first token.",
            boundaries=TTFT_BOUNDARIES_MS,
        )
        logger.info("Lifecycle protocol: %s", self._lifecycle.protocol)
        self._app = Starlette(
            routes=[
                Route(CHAT_PATH, self._chat, methods=["POST"]),
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

    async def _prompt_token_ids(
        self, request: Request, body: Dict[str, Any]
    ) -> Optional[List[int]]:
        """The router-forwarded prompt, else the engine's own tokenization,
        as ``inject_prompt_token_ids`` and vLLM's OpenAI server do."""
        key = request.headers.get(KV_TOKEN_KEY_HEADER)
        entry = self._token_store.pop(key) if key else None
        if entry is not None:
            token_ids = token_channel.decode_prompt_token_ids(entry.payload)
            if token_ids:
                self._prompts.inc(tags={"source": SOURCE_CHANNEL})
                return token_ids
        source = SOURCE_STORE_MISS if key else SOURCE_NO_KEY
        self._prompts.inc(tags={"source": source})
        return await self._tokenizer.tokenize(body, request_path=CHAT_PATH)

    def _count_session(self, request: Request) -> None:
        session_id = request.headers.get(SERVE_SESSION_ID)
        if not session_id:
            source = SESSION_NONE
        elif session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            source = SESSION_REPEAT
        else:
            self._sessions[session_id] = None
            if len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
            source = SESSION_NEW
        self._session_requests.inc(tags={"source": source})

    async def _chat(self, request: Request):
        arrived = time.monotonic()
        self._count_session(request)
        headers = {ENGINE_HEADER: self._engine_id}
        body = json.loads(await request.body())
        context = serve.context._serve_request_context.get()
        request_id = (context.request_id if context else "") or uuid.uuid4().hex
        owner = self._lifecycle.owner(request)
        error = (400, "BadRequestError", "unsupported request body")
        try:
            token_ids = await self._prompt_token_ids(request, body)
        except self._tokenize_error as e:
            token_ids, error = None, (e.status_code, e.type, e.message)
        if not token_ids:
            # The router booked this request, so free it like a failed engine call.
            self._lifecycle.report(owner, "on_request_completed", request_id)
            status, kind, message = error
            return JSONResponse(
                {"error": {"message": message, "type": kind, "code": status}},
                status_code=status,
                headers=headers,
            )
        max_tokens = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or DEFAULT_MAX_TOKENS
        )
        try:
            live = await self._worker.submit(token_ids, max(1, max_tokens))
        except Exception:
            self._lifecycle.report(owner, "on_request_completed", request_id)
            raise
        model = body.get("model", self._model_id)
        if body.get("stream"):
            include_usage = bool(
                (body.get("stream_options") or {}).get("include_usage")
            )
            return StreamingResponse(
                self._stream(
                    live,
                    request_id,
                    owner,
                    model,
                    len(token_ids),
                    include_usage,
                    arrived,
                ),
                media_type="text/event-stream",
                headers=headers,
            )
        completion, cached = 0, 0
        async for n, cached in self._chunks(live, request_id, owner, arrived):
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
            },
            headers=headers,
        )

    async def _chunks(
        self, live, request_id: str, owner: Optional[str], arrived: float
    ):
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
                    self._ttft.observe((time.monotonic() - arrived) * 1000)
                    self._lifecycle.report(owner, "on_prefill_complete", request_id)
                if n:
                    yield n, cached
                if done:
                    break
        finally:
            self._lifecycle.report(owner, "on_request_completed", request_id)

    async def _stream(
        self, live, request_id, owner, model, prompt_tokens, include_usage, arrived
    ):
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
        async for n, cached in self._chunks(live, request_id, owner, arrived):
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
    cfg = {
        "model_id": model_id,
        "mocker": _normalize_mocker_config({"engine": mocker_engine}),
        "stream_interval_s": stream_interval_s,
        # The engine tokenizes with the router's config when the token channel
        # misses, like vLLM.
        "llm_config": llm_config,
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
    router = _build_openai_ingress_request_router(server=server, llm_config=llm_config)
    return server._with_ingress_request_router(router)
