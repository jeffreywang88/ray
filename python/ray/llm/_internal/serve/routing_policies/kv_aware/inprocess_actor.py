"""KVAwareRouter4: the ``KVRouterActor`` lives IN-PROCESS in the LLMRouter
ingress replica (not a separate Ray deployment actor), so ``select_worker_chat``
is a local coroutine call instead of an actor RPC hop.

The single in-process instance is created by ``LLMRouter.__init__`` (which has the
``llm_config`` + the LLMServer deployment handle) and shared via this process
global with:
  - ``KVAwareRouter.choose_replicas`` (runs in the same ingress process; does the
    local select), and
  - ``LLMRouter``'s handle-callable ``on_lifecycle_events`` / ``get_prompt_tokens``
    methods, which the engine replicas RPC into (these calls were engine->actor
    RPCs before; only the on-critical-path select becomes local).
"""
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ray.llm._internal.serve.core.configs.llm_config import LLMConfig

_INPROCESS_KV_ROUTER = None


def set_inprocess_kv_router(actor) -> None:
    global _INPROCESS_KV_ROUTER
    _INPROCESS_KV_ROUTER = actor


def get_inprocess_kv_router() -> Optional[object]:
    return _INPROCESS_KV_ROUTER


# The LLMRouter ingress deployment name (serve.deployment(LLMRouter) with no name
# override -> class name). KVAwareRouter4's engine replicas RPC its handle methods
# (on_lifecycle_events / get_prompt_tokens) instead of a named KVRouterActor.
LLM_ROUTER_DEPLOYMENT_NAME = "LLMRouter"


def get_llm_router_handle():
    """Handle to the in-app LLMRouter deployment, for engine replicas to call its
    KVAwareRouter4 engine-facing methods. Resolved in the current Serve app.
    """
    from ray import serve

    app_name = serve.get_replica_context().app_name
    return serve.get_deployment_handle(
        LLM_ROUTER_DEPLOYMENT_NAME, app_name=app_name
    )


def build_inprocess_kv_router(llm_config: "LLMConfig", serve_deployment_id: Any):
    """Construct the in-process ``KVRouterActor`` from ``llm_config``.

    Mirrors the ``init_kwargs`` that ``utils._maybe_setup_kv_aware_routing``
    passes to the deployment-actor ``KVRouterActor`` (indexer_threads /
    model_source / fused_threads), plus the explicit LLMServer deployment id
    that ``_start_replica_tracking`` needs since there is no deployment-actor
    context in-process. Must be called from within the ingress replica's event
    loop (KVRouterActor.__init__ subscribes a LongPollClient to it).
    """
    from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
        DEFAULT_KV_INDEXER_THREADS,
        KV_FUSED_THREADS_KEY,
        KV_INDEXER_THREADS_KEY,
    )
    from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
        KVRouterActor,
    )

    model_source = llm_config.model_loading_config.model_source
    actor = KVRouterActor(
        indexer_threads=llm_config.experimental_configs.get(
            KV_INDEXER_THREADS_KEY, DEFAULT_KV_INDEXER_THREADS
        ),
        model_source=model_source if isinstance(model_source, str) else None,
        fused_threads=llm_config.experimental_configs.get(KV_FUSED_THREADS_KEY),
        serve_deployment_id=serve_deployment_id,
    )
    set_inprocess_kv_router(actor)
    return actor
