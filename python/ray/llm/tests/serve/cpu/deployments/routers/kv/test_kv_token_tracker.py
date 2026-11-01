"""Validation for the router-local KV selector: build + process-global
registration, ``KVAwareRouter`` binding to it, the ``KVTokenTracker`` ->
Dynamo selection-service booking path, and replica-mesh replication between
LLMRouter replicas.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

import ray
import ray.cloudpickle
from ray import serve
from ray._common.test_utils import async_wait_for_condition
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.ingress.router import LLMRouter
from ray.llm._internal.serve.routing_policies.kv_aware import kv_token_tracker
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    KVTokenTracker,
    build_kv_token_tracker,
    get_kv_token_tracker,
    get_worker_id,
    set_kv_token_tracker,
)
from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
    LifecycleEventForwarder,
)
from ray.serve._private.common import DeploymentID
from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve.config import RequestRouterConfig
from ray.serve.llm.request_router import KVAwareRouter


@pytest.fixture(autouse=True)
def reset_process_global():
    set_kv_token_tracker(None)
    yield
    set_kv_token_tracker(None)


class _SpyTracker:
    """Records constructor kwargs; skips the real selection service + LongPoll."""

    def __init__(
        self,
        *,
        indexer_threads,
        serve_deployment_id,
        replica_sync_port,
        router_deployment_id,
    ):
        self.indexer_threads = indexer_threads
        self.serve_deployment_id = serve_deployment_id
        self.replica_sync_port = replica_sync_port
        self.router_deployment_id = router_deployment_id


@pytest.fixture
def router_replica_context(monkeypatch):
    """The LLMRouter replica context ``build_kv_token_tracker`` reads."""
    context = SimpleNamespace(
        rank=SimpleNamespace(local_rank=1),
        replica_id=SimpleNamespace(deployment_id="router-dep-id"),
    )
    monkeypatch.setattr(kv_token_tracker.serve, "get_replica_context", lambda: context)


class _NoLongPollTracker(KVTokenTracker):
    """Real selection service without a controller membership subscription."""

    def _start_replica_tracking(self):
        pass


def _llm_config(experimental_configs=None) -> LLMConfig:
    return LLMConfig(
        model_loading_config={
            "model_id": "qwen3-0.6b",
            "model_source": "Qwen/Qwen3-0.6B",
        },
        accelerator_type=None,
        experimental_configs=experimental_configs or {},
    )


def test_build_registers_tracker(monkeypatch, router_replica_context):
    """build_kv_token_tracker builds the tracker with the explicit deployment
    id and registers it so the same-process router/ingress can reach it."""
    monkeypatch.setattr(kv_token_tracker, "KVTokenTracker", _SpyTracker)
    tracker = build_kv_token_tracker(_llm_config(), "dep-id")
    assert get_kv_token_tracker() is tracker
    assert tracker.serve_deployment_id == "dep-id"


def test_indexer_threads_from_config(monkeypatch, router_replica_context):
    """The KV_INDEXER_THREADS experimental config reaches the router-local tracker."""
    monkeypatch.setattr(kv_token_tracker, "KVTokenTracker", _SpyTracker)
    tracker = build_kv_token_tracker(_llm_config({"KV_INDEXER_THREADS": 8}), "dep-id")
    assert tracker.indexer_threads == 8


@pytest.mark.parametrize(
    "experimental_configs,port",
    [({}, 23456), ({"KV_REPLICA_SYNC_PORT_BASE": 20000}, 20001)],
)
def test_build_wires_replica_mesh(
    monkeypatch, router_replica_context, experimental_configs, port
):
    """Every router binds its replica-mesh port on a free port, or at base +
    node-local rank when pinned, and finds its peers through its own LLMRouter
    deployment."""
    monkeypatch.setattr(kv_token_tracker, "KVTokenTracker", _SpyTracker)
    monkeypatch.setattr(kv_token_tracker, "find_free_port", lambda: 23456)
    tracker = build_kv_token_tracker(_llm_config(experimental_configs), "dep-id")
    assert tracker.replica_sync_port == port
    assert tracker.router_deployment_id == "router-dep-id"


def test_router_binds_tracker():
    """KVAwareRouter picks up the tracker the LLMRouter registered."""
    sentinel = object()
    set_kv_token_tracker(sentinel)
    router = KVAwareRouter.__new__(KVAwareRouter)
    router.initialize_state()
    assert router._kv_token_tracker is sentinel


def test_router_degrades_without_tracker():
    """A process with no router-local tracker (proxy fallback router) degrades to
    load-balanced selection rather than erroring."""
    router = KVAwareRouter.__new__(KVAwareRouter)
    router._deployment_id = "dep"
    router.initialize_state()
    assert router._kv_token_tracker is None


def _worker_load(tracker: KVTokenTracker, worker_id: int, field: str) -> int:
    """A booked-load field for ``worker_id`` in the tracker's selection service."""
    for model in tracker._svc.loads():
        for load in model["loads"]:
            if load["worker_id"] == worker_id:
                return load[field]
    return 0


def _active_requests(tracker: KVTokenTracker, worker_id: int) -> int:
    """Active requests the tracker's selection service books on ``worker_id``."""
    return _worker_load(tracker, worker_id, "active_requests")


@pytest.mark.asyncio
async def test_lifecycle_booking_end_to_end():
    """End-to-end router-local path on the real Dynamo selection service:
    select_and_reserve books load immediately, and engine lifecycle events move
    prefill to decode and then free the request."""
    pytest.importorskip("dynamo.llm")
    worker_id = get_worker_id("replica-0")

    tracker = _NoLongPollTracker()
    try:
        tracker._register_block_size(16, "replica-0")
        await tracker._upsert_worker(
            worker_id,
            "replica-0",
            {
                "endpoint": "tcp://127.0.0.1:59999",
                "block_size": 16,
                "max_num_batched_tokens": 8192,
                "dp_rank": 0,
            },
        )
        selection = await tracker.select_worker(
            "req-e2e", [1, 2, 3, 4], [worker_id], 20
        )
        assert selection["worker_id"] == worker_id
        assert "req-e2e" in tracker._requests
        assert _active_requests(tracker, worker_id) == 1

        router = LLMRouter.__new__(LLMRouter)
        router._kv_token_tracker = tracker
        await router.on_lifecycle_events([("on_prefill_complete", ("req-e2e",))])
        assert _active_requests(tracker, worker_id) == 1
        await router.on_lifecycle_events([("on_request_completed", ("req-e2e",))])
        assert "req-e2e" not in tracker._requests
        assert _active_requests(tracker, worker_id) == 0
    finally:
        if tracker._svc is not None:
            await tracker._svc.delete_worker(worker_id)


async def _register_engine_worker(tracker: KVTokenTracker, worker_id: int) -> None:
    """Register one engine replica's KV-event endpoint with ``tracker``."""
    tracker._register_block_size(16, "engine-replica-0")
    await tracker._upsert_worker(
        worker_id,
        "engine-replica-0",
        {
            "endpoint": "tcp://127.0.0.1:59998",
            "block_size": 16,
            "max_num_batched_tokens": 8192,
            "dp_rank": 0,
        },
    )


class _PeerRecordingService:
    """Selection-service stand-in recording replica-mesh peer changes."""

    def __init__(self):
        self.calls = []

    async def register_replica_peer(self, endpoint):
        self.calls.append(("register", endpoint))
        return True

    async def deregister_replica_peer(self, endpoint):
        self.calls.append(("deregister", endpoint))
        return True


def _router_targets(endpoints_by_id):
    from ray.serve._private.common import (
        DeploymentTargetInfo,
        ReplicaID,
        RunningReplicaInfo,
    )

    deployment_id = DeploymentID(name="LLMRouter", app_name="app")
    return DeploymentTargetInfo(
        is_available=True,
        running_replicas=[
            RunningReplicaInfo(
                replica_id=ReplicaID(unique_id=uid, deployment_id=deployment_id),
                node_id="node",
                node_ip="10.0.0.1",
                availability_zone="az",
                actor_name=f"actor-{uid}",
                max_ongoing_requests=1000,
                routing_stats=(
                    {"kv_replica_sync_metadata": {"endpoint": endpoint}}
                    if endpoint
                    else {}
                ),
            )
            for uid, endpoint in endpoints_by_id.items()
        ],
    )


@pytest.mark.asyncio
async def test_router_targets_drive_replica_mesh_peers():
    """Peers follow the LLMRouter replicas that advertise a mesh endpoint,
    excluding this router itself."""
    tracker = _NoLongPollTracker()
    tracker._svc = _PeerRecordingService()
    tracker._own_replica_unique_id = "self"

    tracker._on_router_targets(
        _router_targets({"self": "tcp://a:1", "b": "tcp://b:1", "c": None})
    )
    await asyncio.gather(*tracker._pending_tasks)
    assert tracker._svc.calls == [("register", "tcp://b:1")]

    # "b" restarts on a new endpoint and "c" starts advertising one.
    tracker._svc.calls.clear()
    tracker._on_router_targets(
        _router_targets({"self": "tcp://a:1", "b": "tcp://b:2", "c": "tcp://c:1"})
    )
    await asyncio.gather(*tracker._pending_tasks)
    assert sorted(tracker._svc.calls) == [
        ("deregister", "tcp://b:1"),
        ("register", "tcp://b:2"),
        ("register", "tcp://c:1"),
    ]


@pytest.mark.asyncio
async def test_replica_mesh_replicates_owner_load_to_peer():
    """With replica-mesh replication the peer sees the owner's booking and its
    release without any reservation broadcast or lifecycle event of its own."""
    pytest.importorskip("dynamo.llm")
    from ray._common.network_utils import find_free_port

    worker_id = get_worker_id("engine-replica-0")
    ports = [find_free_port(), find_free_port()]
    owner = _NoLongPollTracker(replica_sync_port=ports[0])
    peer = _NoLongPollTracker(replica_sync_port=ports[1])
    trackers = (owner, peer)
    try:
        for tracker in trackers:
            await _register_engine_worker(tracker, worker_id)
        await owner._svc.register_replica_peer(f"tcp://127.0.0.1:{ports[1]}")
        await peer._svc.register_replica_peer(f"tcp://127.0.0.1:{ports[0]}")
        await asyncio.sleep(0.5)

        await owner.select_worker("req-mesh", list(range(64)), [worker_id], 32)
        await async_wait_for_condition(
            lambda: _active_requests(peer, worker_id) == 1, timeout=10
        )
        assert "req-mesh" not in peer._requests

        await owner.on_lifecycle_events(
            [
                ("on_prefill_complete", ("req-mesh",)),
                ("on_request_completed", ("req-mesh",)),
            ]
        )
        await async_wait_for_condition(
            lambda: _active_requests(peer, worker_id) == 0, timeout=10
        )
    finally:
        for tracker in trackers:
            if tracker._svc is not None:
                await tracker._svc.delete_worker(worker_id)


# ---- LongPoll replica-membership tracking -----------------------------------


@pytest.fixture(scope="module")
def serve_instance():
    if not ray.is_initialized():
        ray.init()
    yield
    serve.shutdown()


_REPLICATION_APP_NAME = "kv-replica-mesh-test"
_REPLICATION_WORKER_ID = get_worker_id("replicated-engine-worker")


class _RouterMeshTracker(KVTokenTracker):
    """Real selection service that follows only its own deployment's replicas,
    the LLMRouter replica-mesh peers, over LongPoll."""

    def _start_replica_tracking(self):
        controller = ray.get_actor(SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
        self._long_poll_client = LongPollClient(
            controller,
            {
                (
                    LongPollNamespace.DEPLOYMENT_TARGETS,
                    self._router_deployment_id,
                ): self._on_router_targets
            },
            call_in_event_loop=asyncio.get_running_loop(),
            client_id=f"{type(self).__name__}:{self._router_deployment_id}",
        )


@serve.deployment(
    num_replicas=2,
    request_router_config=RequestRouterConfig(request_routing_stats_period_s=0.5),
)
class ReplicaMeshDeployment:
    """Two real Serve replicas, each with an independent selection service,
    wired like LLMRouter replicas: peers found through routing stats and
    lifecycle events received on the selecting replica's ZMQ endpoint."""

    async def __init__(self):
        from ray._common.network_utils import find_free_port

        self._tracker = _RouterMeshTracker(
            replica_sync_port=find_free_port(),
            router_deployment_id=serve.get_replica_context().replica_id.deployment_id,
        )
        await self._tracker.start_lifecycle_receiver(find_free_port())
        self._tracker._register_block_size(16, "replicated-engine-worker")
        await self._tracker._upsert_worker(
            _REPLICATION_WORKER_ID,
            "replicated-engine-worker",
            {
                "endpoint": "tcp://127.0.0.1:59997",
                "block_size": 16,
                "max_num_batched_tokens": 8192,
                "dp_rank": 0,
            },
        )

    async def record_routing_stats(self) -> dict:
        return self._tracker.replica_sync_routing_stats()

    async def num_peers(self) -> int:
        return len(self._tracker._replica_sync_peers)

    async def select_worker(self, request_id, token_ids):
        await self._tracker.select_worker(
            request_id, token_ids, [_REPLICATION_WORKER_ID]
        )
        return self._tracker.lifecycle_endpoint

    async def on_lifecycle_events(self, events):
        await self._tracker.on_lifecycle_events(events)

    async def get_worker_load(self):
        return {
            field: _worker_load(self._tracker, _REPLICATION_WORKER_ID, field)
            for field in (
                "active_requests",
                "potential_prefill_tokens",
                "potential_decode_blocks",
            )
        }


async def _broadcast(handle, method_name, *args):
    return await handle.broadcast(method_name, *args).results_async()


@pytest.mark.asyncio
async def test_serve_replicas_converge_over_replica_mesh(serve_instance):
    """Routers find each other through routing stats, and the selecting
    router's booking, prefill completion and free reach its peer over the
    replica mesh after the engine reports only to that router, over ZMQ."""
    pytest.importorskip("dynamo.llm")
    module = sys.modules[__name__]
    ray.cloudpickle.register_pickle_by_value(module)
    request_id = "req-serve-mesh"
    forwarder = None
    try:
        handle = serve.run(
            ReplicaMeshDeployment.bind(),
            name=_REPLICATION_APP_NAME,
            route_prefix=None,
        )
        forwarder = LifecycleEventForwarder(handle)

        async def peers_connected():
            return await _broadcast(handle, "num_peers") == [1, 1]

        await async_wait_for_condition(
            peers_connected, timeout=60, retry_interval_ms=200
        )
        # The replica mesh SUB sockets connect in the background.
        await asyncio.sleep(1)

        async def loads_are(predicate):
            loads = await _broadcast(handle, "get_worker_load")
            return len(loads) == 2 and all(predicate(load) for load in loads)

        # A normal handle call lands on exactly one replica, which selects,
        # reserves and replicates the booking to its peer.
        owner = await handle.select_worker.remote(request_id, list(range(64)))
        await async_wait_for_condition(
            loads_are,
            predicate=lambda load: load["active_requests"] == 1
            and load["potential_prefill_tokens"] > 0,
            timeout=10,
        )

        forwarder.report_to(owner, "on_prefill_complete", request_id)
        await async_wait_for_condition(
            loads_are,
            predicate=lambda load: load["active_requests"] == 1
            and load["potential_prefill_tokens"] == 0,
            timeout=10,
        )

        forwarder.report_to(owner, "on_request_completed", request_id)
        await async_wait_for_condition(
            loads_are,
            predicate=lambda load: load["active_requests"] == 0,
            timeout=10,
        )
        # Both events went to the owner over ZMQ, none through the broadcast.
        assert forwarder._broadcast_task is None
    finally:
        if forwarder is not None:
            forwarder.close()
        serve.delete(_REPLICATION_APP_NAME)
        ray.cloudpickle.unregister_pickle_by_value(module)


@serve.deployment
class ReplicaTrackingDeployment:
    """Dummy deployment that advertises a per-replica KV-events endpoint via
    ``record_routing_stats`` as a real engine replica would, so a KVTokenTracker
    watching this deployment registers each replica as a worker."""

    async def __call__(self) -> str:
        return "ok"

    async def record_routing_stats(self) -> dict:
        rank = serve.get_replica_context().rank.local_rank
        return {
            "kv_event_metadata": {
                "endpoint": f"tcp://127.0.0.1:{25000 + rank}",
                "block_size": 16,
                "max_num_batched_tokens": 8192,
                "dp_rank": 0,
            }
        }


class _MembershipTracker(KVTokenTracker):
    """Real LongPoll replica tracking with the selection service disabled, so the
    DEPLOYMENT_TARGETS subscription and reconciliation are exercised without
    dynamo."""

    def _create_selection_service(self) -> None:
        self._svc = None


@pytest.mark.asyncio
async def test_longpoll_tracks_membership(serve_instance):
    """The tracker's LongPoll subscription reconciles its tracked workers to the
    deployment's live replicas across scale up and down -- the same membership
    tracking the KVTokenTracker does inside the LLMRouter ingress replica."""
    app_name = "kv-replica-tracking"

    def deploy(num_replicas):
        serve.run(
            ReplicaTrackingDeployment.options(num_replicas=num_replicas).bind(),
            name=app_name,
            route_prefix="/kv_track",
        )

    deploy(2)
    # Build the tracker the way the LLMRouter ingress would, pointed at the dummy
    # deployment; its LongPollClient binds to this test's event loop.
    tracker = _MembershipTracker(
        serve_deployment_id=DeploymentID(
            name="ReplicaTrackingDeployment", app_name=app_name
        )
    )
    try:
        await async_wait_for_condition(
            lambda: len(tracker._replica_id_by_worker) == 2, timeout=30
        )
        deploy(4)  # upscale: the new replicas are picked up over LongPoll
        await async_wait_for_condition(
            lambda: len(tracker._replica_id_by_worker) == 4, timeout=30
        )
        deploy(2)  # downscale: the departed replicas are dropped
        await async_wait_for_condition(
            lambda: len(tracker._replica_id_by_worker) == 2, timeout=30
        )
    finally:
        serve.delete(app_name, _blocking=True)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
