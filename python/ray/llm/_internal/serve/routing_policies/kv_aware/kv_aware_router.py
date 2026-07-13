import logging
import random
from typing import List, Optional

import ray
from ray.actor import ActorHandle
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    REQUEST_ROUTING_BODY_KWARG,
    REQUEST_TOKEN_IDS_KWARG,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KV_ROUTER_ACTOR_NAME,
    get_worker_id,
)
from ray.serve._private.constants import (
    SERVE_DEPLOYMENT_ACTOR_PREFIX,
    SERVE_LOGGER_NAME,
    SERVE_NAMESPACE,
)
from ray.serve._private.request_router.common import PendingRequest
from ray.serve._private.request_router.replica_wrapper import RunningReplica
from ray.serve._private.request_router.request_router import RequestRouter
from ray.serve.config import RequestRouterConfig

logger = logging.getLogger(SERVE_LOGGER_NAME)


def _get_expected_output_tokens(pending_request: PendingRequest) -> Optional[int]:
    """The request's output cap from the routing payload, if it carries one.

    The parsed request rides as the first positional routing arg (see
    ``LLMRouter._pick_replica``). Chat requests cap output with
    ``max_completion_tokens`` (``max_tokens`` is its deprecated alias);
    completion requests use ``max_tokens``. Captured by ``select`` so the
    cached selection's decode-load estimate reflects the request.
    """
    if not pending_request.args:
        return None
    payload = pending_request.args[0]
    for field in ("max_completion_tokens", "max_tokens"):
        value = getattr(payload, field, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


class KVAwareRouter(RequestRouter):
    """Routes each request to the candidate that best balances expected KV-cache
    overlap against the worker's current prefill/decode load.

    Scoring is delegated to the deployment-scoped ``KVRouterActor`` (which owns the
    Dynamo selection service and the global KV index); this per-handle router stays
    thin and simply maps candidate replicas to/from Dynamo worker ids.
    """

    def initialize_state(self):
        """Resolve the deployment's ``KVRouterActor``.

        The actor is attached to this deployment via ``DeploymentActorConfig``
        whenever the request router is a ``KVAwareRouter``, so it exists by the time
        requests route. We resolve its Serve-generated name and block on a cheap
        call to confirm it finished initializing, so the first routed request finds
        a ready scorer.
        """
        self._kv_router_actor = self._discover_kv_router_actor()
        # Synchronization barrier: Ray defers actor methods until __init__ completes,
        # so awaiting any method blocks until KVRouterActor is constructed.
        ray.get(self._kv_router_actor.ready.remote())

    def _discover_kv_router_actor(self) -> ActorHandle:
        """Handle to this deployment's ``KVRouterActor`` by its Serve-scoped name."""
        prefix = (
            f"{SERVE_DEPLOYMENT_ACTOR_PREFIX}"
            f"{self._deployment_id.app_name}::{self._deployment_id.name}::"
        )
        suffix = f"::{KV_ROUTER_ACTOR_NAME}"
        for entry in ray.util.list_named_actors(all_namespaces=True):
            name = entry.get("name") or ""
            if (
                entry.get("namespace") == SERVE_NAMESPACE
                and name.startswith(prefix)
                and name.endswith(suffix)
            ):
                return ray.get_actor(name, namespace=SERVE_NAMESPACE)
        raise RuntimeError(
            f"KVRouterActor for deployment {self._deployment_id} not found; it must "
            "be attached via DeploymentActorConfig when using KVAwareRouter."
        )

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        """Choose the candidate replica(s) to route ``pending_request`` to.

        Maps the candidate replicas to their Dynamo worker ids, asks the
        ``KVRouterActor`` to rank them, and routes to the chosen worker's
        replica. With direct streaming enabled, HAProxy then forwards the
        original request to that replica.

        Chat requests carry the raw body and take the fused
        ``select_worker_chat`` path: the actor renders the chat template,
        tokenizes, and selects in one Rust call, so prompt token ids never
        enter Python. Completion requests carry pre-tokenized ids
        (``select_worker``). Requests with neither have nothing to score on
        and route to a random candidate (batch prompts, truncated or
        unparseable bodies).

        Args:
            candidate_replicas: The replicas eligible to serve the request.
            pending_request: The request being routed.

        Returns:
            Ranked groups of replicas.
        """
        routing_body = (
            pending_request.kwargs.get(REQUEST_ROUTING_BODY_KWARG)
            if pending_request is not None
            else None
        )
        token_ids = (
            pending_request.kwargs.get(REQUEST_TOKEN_IDS_KWARG)
            if pending_request is not None
            else None
        )
        if not routing_body and not token_ids:
            return [[random.choice(candidate_replicas)]] if candidate_replicas else []

        worker_id_to_replica = {
            get_worker_id(replica.replica_id.unique_id): replica
            for replica in candidate_replicas
        }
        if routing_body:
            selection = await self._kv_router_actor.select_worker_chat.remote(
                pending_request.metadata.request_id,
                routing_body,
                list(worker_id_to_replica),
                _get_expected_output_tokens(pending_request),
            )
        else:
            selection = await self._kv_router_actor.select_worker.remote(
                pending_request.metadata.request_id,
                token_ids,
                list(worker_id_to_replica),
                _get_expected_output_tokens(pending_request),
            )
        return [[worker_id_to_replica[selection["worker_id"]]]]


def is_kv_aware(llm_config: LLMConfig) -> bool:
    """Whether ``llm_config`` selects a ``KVAwareRouter`` for replica selection."""
    request_router_config = llm_config.deployment_config.get("request_router_config")
    if isinstance(request_router_config, dict):
        request_router_config = RequestRouterConfig(**request_router_config)
    return isinstance(request_router_config, RequestRouterConfig) and issubclass(
        request_router_config.get_request_router_class(), KVAwareRouter
    )
