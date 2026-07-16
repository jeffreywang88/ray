"""KVAwareRouter4/5: the ``KVRouterActor`` lives IN-PROCESS in the LLMRouter
ingress replica (not a separate Ray deployment actor), so ``select_worker_chat``
is a local coroutine call instead of an actor RPC hop.

The in-process instance is created by ``LLMRouter.__init__`` (which has the
``llm_config`` + the LLMServer deployment handle) and shared via this process
global with:
  - ``KVAwareRouter.choose_replicas`` (runs in the same ingress process; does the
    local select), and
  - ``LLMRouter``'s handle-callable ``on_lifecycle_events`` method, which the
    engine replicas RPC into (this was an engine->actor RPC before; only the
    on-critical-path select becomes local).

KVAwareRouter5 runs N>1 ingress replicas, each with its OWN in-process selection
service. Consistency across them:
  - KV events (radix tree): each selection service independently subscribes to
    every engine's KV-event ZMQ endpoint (unchanged), so all N radix trees see
    all engines' cache state.
  - Active load (reservations): Dynamo's native replica-sync plane. Each
    selection service is created with a ``replica_sync_port`` and registers the
    other replicas as peers (``register_replica_peer``); a reservation booked on
    any replica propagates to all peers, and any peer can mutate/free it. So the
    engine's lifecycle events can land on ANY ingress replica (load-balanced) and
    still keep every selector's load view globally consistent. Peer endpoints are
    exchanged through a tiny detached registry actor (control-plane only).
"""
import socket
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ray.llm._internal.serve.core.configs.llm_config import LLMConfig

_INPROCESS_KV_ROUTER = None

# experimental_configs key for the ingress-router replica count (kv5 uses 4).
INGRESS_ROUTER_REPLICAS_KEY = "INGRESS_ROUTER_REPLICAS"


def set_inprocess_kv_router(actor) -> None:
    global _INPROCESS_KV_ROUTER
    _INPROCESS_KV_ROUTER = actor


def get_inprocess_kv_router() -> Optional[object]:
    return _INPROCESS_KV_ROUTER


# The LLMRouter ingress deployment name (serve.deployment(LLMRouter) with no name
# override -> class name). KVAwareRouter4/5's engine replicas RPC its handle method
# (on_lifecycle_events) instead of a named KVRouterActor.
LLM_ROUTER_DEPLOYMENT_NAME = "LLMRouter"


def get_llm_router_handle():
    """Handle to the in-app LLMRouter deployment, for engine replicas to call its
    KVAwareRouter4/5 engine-facing methods. Resolved in the current Serve app.
    With N ingress replicas the call load-balances to one of them; replica-sync
    keeps every selector's load view consistent regardless of which one books.
    """
    from ray import serve

    app_name = serve.get_replica_context().app_name
    return serve.get_deployment_handle(
        LLM_ROUTER_DEPLOYMENT_NAME, app_name=app_name
    )


def _pick_free_tcp_port() -> int:
    """Reserve an ephemeral TCP port for the selection service's replica-sync
    listener. Small TOCTOU window before the Rust listener binds it; acceptable
    for a single-node benchmark deploy.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def get_sync_registry(app_name: str):
    """Detached, app-scoped registry actor that exchanges replica-sync endpoints
    between the N in-process selection services (control-plane only, off the
    per-request path). Created on first use; shared by all ingress replicas.
    """
    cls = _ensure_registry_cls()
    name = f"__kv5_sync_registry_{app_name}"
    return cls.options(
        name=name,
        namespace="serve",
        lifetime="detached",
        num_cpus=0,
        get_if_exists=True,
    ).remote()


def _kv_sync_registry_cls():
    import ray

    @ray.remote
    class _KVSyncRegistry:
        """replica_id -> "tcp://ip:port" of each ingress replica's sync listener."""

        def __init__(self):
            self._peers: Dict[str, str] = {}

        def register(self, replica_id: str, endpoint: str) -> Dict[str, str]:
            self._peers[replica_id] = endpoint
            return dict(self._peers)

        def unregister(self, replica_id: str) -> None:
            self._peers.pop(replica_id, None)

        def peers(self) -> Dict[str, str]:
            return dict(self._peers)

    return _KVSyncRegistry


# Built lazily so importing this module doesn't require a Ray runtime.
_KVSyncRegistry = None


def _ensure_registry_cls():
    global _KVSyncRegistry
    if _KVSyncRegistry is None:
        _KVSyncRegistry = _kv_sync_registry_cls()
    return _KVSyncRegistry


def build_inprocess_kv_router(llm_config: "LLMConfig", serve_deployment_id: Any):
    """Construct the in-process ``KVRouterActor`` from ``llm_config``.

    Mirrors the ``init_kwargs`` that ``utils._maybe_setup_kv_aware_routing``
    passes to the deployment-actor ``KVRouterActor`` (indexer_threads /
    model_source / fused_threads), plus the explicit LLMServer deployment id
    that ``_start_replica_tracking`` needs since there is no deployment-actor
    context in-process. Must be called from within the ingress replica's event
    loop (KVRouterActor.__init__ subscribes a LongPollClient to it).

    KVAwareRouter5 (INGRESS_ROUTER_REPLICAS > 1): also reserve a replica_sync_port
    so the selection service joins Dynamo's replica-sync plane and its load view
    stays consistent with the other ingress replicas' selectors.
    """
    from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
        DEFAULT_KV_INDEXER_THREADS,
        KV_FUSED_THREADS_KEY,
        KV_INDEXER_THREADS_KEY,
    )
    from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
        KVRouterActor,
    )

    n_ingress = int(
        llm_config.experimental_configs.get(INGRESS_ROUTER_REPLICAS_KEY, 1)
    )
    replica_sync_port = _pick_free_tcp_port() if n_ingress > 1 else None

    model_source = llm_config.model_loading_config.model_source
    actor = KVRouterActor(
        indexer_threads=llm_config.experimental_configs.get(
            KV_INDEXER_THREADS_KEY, DEFAULT_KV_INDEXER_THREADS
        ),
        model_source=model_source if isinstance(model_source, str) else None,
        fused_threads=llm_config.experimental_configs.get(KV_FUSED_THREADS_KEY),
        serve_deployment_id=serve_deployment_id,
        replica_sync_port=replica_sync_port,
    )
    set_inprocess_kv_router(actor)
    return actor
