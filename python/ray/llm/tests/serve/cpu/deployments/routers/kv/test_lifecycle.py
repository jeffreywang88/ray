"""Lifecycle-event delivery from engine replicas to LLMRouter replicas."""

import asyncio
import sys

import pytest

from ray._common.network_utils import find_free_port
from ray.llm._internal.serve.routing_policies.kv_aware import token_channel
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_LIFECYCLE_ENDPOINT_HEADER,
)
from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
    LifecycleEndpointMiddleware,
    LifecycleEventForwarder,
    LifecycleReceiver,
    decode_lifecycle_events,
    encode_lifecycle_events,
    get_request_lifecycle_endpoint,
)


class _BroadcastRecorder:
    """DeploymentHandle stand-in recording broadcast lifecycle batches."""

    def __init__(self):
        self.batches = []

    def broadcast(self, method_name, batch):
        recorder = self

        class _Response:
            async def results_async(self, *, timeout_s=None, return_exceptions=False):
                recorder.batches.append((method_name, list(batch)))
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


@pytest.mark.asyncio
async def test_report_to_delivers_to_owner_in_order():
    """Events for one owner arrive at its receiver in report order, without a
    Serve broadcast."""
    endpoint = f"tcp://127.0.0.1:{find_free_port()}"
    received = []

    async def handler(events):
        received.extend(events)

    receiver = LifecycleReceiver(bind_endpoint=endpoint, handler=handler)
    assert await receiver.start()
    handle = _BroadcastRecorder()
    forwarder = LifecycleEventForwarder(handle, worker_id=1)
    try:
        # The PUSH socket connects in the background; until it does, a report
        # falls back to the broadcast path.
        forwarder._sender = token_channel.TokenSender()
        forwarder._sender.connect([endpoint])
        await asyncio.sleep(0.5)

        forwarder.report_to(endpoint, "on_prefill_complete", "r1")
        forwarder.report_to(endpoint, "on_request_completed", "r1")
        await _wait_for(lambda: len(received) == 2)
        assert received == [
            ("on_prefill_complete", ("r1",)),
            ("on_request_completed", ("r1",)),
        ]
        assert handle.batches == []
    finally:
        forwarder.close()
        await receiver.close()


@pytest.mark.asyncio
async def test_report_to_without_owner_broadcasts():
    handle = _BroadcastRecorder()
    forwarder = LifecycleEventForwarder(handle, worker_id=1)
    try:
        forwarder.report_to(None, "on_request_completed", "r1")
        await forwarder.flush()
        assert handle.batches == [
            ("on_lifecycle_events", [("on_request_completed", ("r1",))])
        ]
    finally:
        forwarder.close()


@pytest.mark.asyncio
async def test_middleware_exposes_owner_endpoint_to_the_request():
    seen = []

    async def app(scope, receive, send):
        seen.append(get_request_lifecycle_endpoint())

    middleware = LifecycleEndpointMiddleware(app)
    header = KV_LIFECYCLE_ENDPOINT_HEADER.encode()
    await middleware(
        {"type": "http", "headers": [(header, b"tcp://10.0.0.1:9757")]}, None, None
    )
    await middleware({"type": "http", "headers": []}, None, None)
    assert seen == ["tcp://10.0.0.1:9757", None]
    assert get_request_lifecycle_endpoint() is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
