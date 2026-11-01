"""Lifecycle-event delivery from engine replicas to LLMRouter replicas."""

import asyncio
import sys

import pytest

from ray._common.network_utils import find_free_port
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_LIFECYCLE_OWNER_HEADER,
)
from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
    LifecycleEventForwarder,
    LifecycleOwnerMiddleware,
    LifecycleReceiver,
    decode_lifecycle_events,
    encode_lifecycle_events,
    get_request_lifecycle_owner,
)


class _BroadcastHandle:
    """LLMRouter DeploymentHandle stand-in recording broadcast batches."""

    def __init__(self):
        self.batches = []

    def broadcast(self, method_name, batch):
        handle = self

        class _Response:
            async def results_async(self, *, timeout_s, return_exceptions):
                handle.batches.append((method_name, list(batch)))
                return [None]

        return _Response()


async def _wait_for(predicate, timeout_s=5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.01)


def test_events_round_trip():
    events = [("on_prefill_complete", ("r1",)), ("on_decode_progress", ("r1", 32))]
    assert decode_lifecycle_events(encode_lifecycle_events(events)) == events


async def _start_receiver(endpoint):
    received = []

    async def handler(events):
        received.extend(events)

    receiver = LifecycleReceiver(bind_endpoint=endpoint, handler=handler)
    assert await receiver.start()
    return receiver, received


_EVENTS = [
    ("on_prefill_complete", ("r1",)),
    ("on_decode_progress", ("r1", 32)),
    ("on_request_completed", ("r1",)),
]


@pytest.mark.asyncio
async def test_report_to_delivers_to_owner_in_order():
    """Events for one owner arrive at its receiver in report order, including
    those reported before the PUSH socket finished connecting, without a
    broadcast."""
    endpoint = f"tcp://127.0.0.1:{find_free_port()}"
    receiver, received = await _start_receiver(endpoint)
    handle = _BroadcastHandle()
    forwarder = LifecycleEventForwarder(handle)
    try:
        for name, args in _EVENTS:
            forwarder.report_to(endpoint, name, *args)
        await _wait_for(lambda: len(received) == len(_EVENTS))
        assert received == _EVENTS
        assert handle.batches == []
    finally:
        forwarder.close()
        await receiver.close()


@pytest.mark.asyncio
async def test_report_to_queues_until_owner_listens():
    """Events reported before the owner's receiver binds are delivered once it
    does, rather than dropped."""
    endpoint = f"tcp://127.0.0.1:{find_free_port()}"
    forwarder = LifecycleEventForwarder(_BroadcastHandle())
    receiver = None
    try:
        for name, args in _EVENTS:
            forwarder.report_to(endpoint, name, *args)
        receiver, received = await _start_receiver(endpoint)
        await _wait_for(lambda: len(received) == len(_EVENTS))
        assert received == _EVENTS
    finally:
        forwarder.close()
        if receiver is not None:
            await receiver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "not-an-endpoint"])
async def test_events_without_a_reachable_owner_are_broadcast(owner):
    """With no owner, or one whose push cannot be queued, events go to every
    router replica."""
    handle = _BroadcastHandle()
    forwarder = LifecycleEventForwarder(handle)
    try:
        forwarder.report_to(owner, "on_request_completed", "r1")
        await forwarder.flush()
        assert handle.batches == [
            ("on_lifecycle_events", [("on_request_completed", ("r1",))])
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
        {"type": "http", "headers": [(header, b"tcp://10.0.0.1:9757")]}, None, None
    )
    await middleware({"type": "http", "headers": []}, None, None)
    assert seen == ["tcp://10.0.0.1:9757", None]
    assert get_request_lifecycle_owner() is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
