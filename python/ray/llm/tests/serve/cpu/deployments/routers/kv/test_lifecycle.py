"""Lifecycle-event delivery from engine replicas to LLMRouter replicas."""

import asyncio
import sys

import pytest

from ray.llm._internal.serve.routing_policies.kv_aware import lifecycle
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_LIFECYCLE_OWNER_HEADER,
)
from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
    LifecycleEventForwarder,
    LifecycleOwnerMiddleware,
    get_request_lifecycle_owner,
)
from ray.serve._private.common import DeploymentID
from ray.serve.exceptions import DeploymentUnavailableError

_DEPLOYMENT_ID = DeploymentID(name="LLMRouter", app_name="app")


class _Router:
    """One LLMRouter replica's ``on_lifecycle_events`` intake."""

    def __init__(self, fail=False):
        self.batches = []
        self._fail = fail

    async def on_lifecycle_events(self, batch):
        if self._fail:
            raise RuntimeError("booking failed")
        self.batches.append(batch)


class _PinnedHandle:
    """LLMRouter DeploymentHandle stand-in: ``options(replica_id=...)`` pins
    ``on_lifecycle_events`` calls to that router; a missing one is departed.
    ``broadcast`` calls every router."""

    is_initialized = True

    def __init__(self, routers, replica_id=None):
        self._routers = routers
        self._replica_id = replica_id
        self.on_lifecycle_events = self

    def options(self, *, replica_id):
        return _PinnedHandle(self._routers, replica_id)

    async def remote(self, batch):
        router = self._routers.get(self._replica_id)
        if router is None:
            raise DeploymentUnavailableError(_DEPLOYMENT_ID)
        await router.on_lifecycle_events(batch)

    def broadcast(self, method_name, batch):
        routers = list(self._routers.values())

        class _Response:
            async def results_async(self, *, timeout_s, return_exceptions):
                return await asyncio.gather(
                    *(getattr(r, method_name)(batch) for r in routers),
                    return_exceptions=return_exceptions,
                )

        return _Response()


def _events(batches):
    return [event for batch in batches for event in batch]


@pytest.mark.asyncio
async def test_report_to_delivers_to_owner_in_order():
    """Each owner receives exactly its own events, in report order."""
    routers = {"a": _Router(), "b": _Router()}
    forwarder = LifecycleEventForwarder(_PinnedHandle(routers))
    try:
        forwarder.report_to("a", "on_prefill_complete", "r1")
        forwarder.report_to("b", "on_prefill_complete", "r2")
        forwarder.report_to("a", "on_decode_progress", "r1", 32)
        forwarder.report_to("a", "on_request_completed", "r1")
        await forwarder.flush()
        assert _events(routers["a"].batches) == [
            ("on_prefill_complete", ("r1",)),
            ("on_decode_progress", ("r1", 32)),
            ("on_request_completed", ("r1",)),
        ]
        assert _events(routers["b"].batches) == [("on_prefill_complete", ("r2",))]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_failed_delivery_does_not_block_later_events():
    """A failed call is dropped; later events for that owner and events for
    other owners still arrive."""
    failing, healthy = _Router(fail=True), _Router()
    forwarder = LifecycleEventForwarder(
        _PinnedHandle({"failing": failing, "healthy": healthy})
    )
    try:
        forwarder.report_to("failing", "on_request_completed", "r1")
        forwarder.report_to("healthy", "on_request_completed", "r2")
        await forwarder.flush()
        failing._fail = False
        forwarder.report_to("failing", "on_request_completed", "r3")
        await forwarder.flush()
        assert _events(failing.batches) == [("on_request_completed", ("r3",))]
        assert _events(healthy.batches) == [("on_request_completed", ("r2",))]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_departed_owner_drops_its_queue():
    """Events for a router replica that is gone are dropped with its queue, so
    the forwarder holds no state for it."""
    routers = {"live": _Router()}
    forwarder = LifecycleEventForwarder(_PinnedHandle(routers))
    try:
        forwarder.report_to("gone", "on_prefill_complete", "r1")
        forwarder.report_to("gone", "on_request_completed", "r1")
        forwarder.report_to("live", "on_request_completed", "r2")
        await asyncio.wait_for(forwarder.flush(), timeout=5)
        assert "gone" not in forwarder._queues
        assert "gone" not in forwarder._tasks
        assert _events(routers["live"].batches) == [("on_request_completed", ("r2",))]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_events_without_owner_are_broadcast():
    """With no owner, every router receives the events, even when one fails."""
    routers = {"a": _Router(), "b": _Router(fail=True), "c": _Router()}
    forwarder = LifecycleEventForwarder(_PinnedHandle(routers))
    try:
        forwarder.report_to(None, "on_request_completed", "r1")
        await forwarder.flush()
        for name in ("a", "c"):
            assert _events(routers[name].batches) == [("on_request_completed", ("r1",))]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_idle_owner_releases_its_queue(monkeypatch):
    """An owner with no events for a while releases its queue and task, and a
    later event for it starts new ones."""
    monkeypatch.setattr(lifecycle, "_OWNER_IDLE_TIMEOUT_S", 0.05)
    routers = {"a": _Router()}
    forwarder = LifecycleEventForwarder(_PinnedHandle(routers))
    try:
        forwarder.report_to("a", "on_request_completed", "r1")
        await forwarder.flush()
        await asyncio.sleep(0.3)
        assert forwarder._queues == {} and forwarder._tasks == {}

        forwarder.report_to("a", "on_request_completed", "r2")
        await forwarder.flush()
        assert _events(routers["a"].batches) == [
            ("on_request_completed", ("r1",)),
            ("on_request_completed", ("r2",)),
        ]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_middleware_exposes_owner_to_the_request():
    seen = []

    async def app(scope, receive, send):
        seen.append(get_request_lifecycle_owner())

    middleware = LifecycleOwnerMiddleware(app)
    header = KV_LIFECYCLE_OWNER_HEADER.encode()
    await middleware(
        {"type": "http", "headers": [(header, b"LLMRouter#app#router-0")]}, None, None
    )
    await middleware({"type": "http", "headers": []}, None, None)
    assert seen == ["LLMRouter#app#router-0", None]
    assert get_request_lifecycle_owner() is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
