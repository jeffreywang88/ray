"""Delivery of request lifecycle events from engine replicas to LLMRouter replicas.

Engine replicas report ``on_prefill_complete`` / ``on_request_completed`` so the
routers' selection services can release the load they booked for a request.

- ``LifecycleEventForwarder`` broadcasts each event to every LLMRouter replica
  over a Serve handle. With replica-mesh replication enabled it instead pushes the
  event to the router that selected the request (``report_to``), whose selection
  service replicates the change to its peers.
- ``LifecycleReceiver`` is that router's ZMQ PULL endpoint.

This module stays free of engine imports so any engine integration can use it.
"""

import asyncio
import contextlib
import contextvars
from typing import Any, Awaitable, Callable, List, Optional

import msgpack
import zmq
import zmq.asyncio as zmq_asyncio

from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware import token_channel
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_LIFECYCLE_ENDPOINT_HEADER,
    KV_TOKEN_ZMQ_RECEIVE_QUEUE_LIMIT,
    LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
)

logger = get_logger(__name__)

# Most messages the receiver drains into one handler call.
_MAX_DRAIN = 256
_LIFECYCLE_ENDPOINT_HEADER_BYTES = KV_LIFECYCLE_ENDPOINT_HEADER.encode("latin-1")

# The selecting router's lifecycle endpoint for the request being served.
_request_lifecycle_endpoint: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar("kv_request_lifecycle_endpoint", default=None)


def get_request_lifecycle_endpoint() -> Optional[str]:
    """Lifecycle endpoint of the router that selected the current request."""
    return _request_lifecycle_endpoint.get()


class LifecycleEndpointMiddleware:
    """ASGI middleware exposing the router-supplied lifecycle endpoint header
    to the engine through ``get_request_lifecycle_endpoint``."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            for name, value in scope.get("headers") or ():
                if name == _LIFECYCLE_ENDPOINT_HEADER_BYTES:
                    token = _request_lifecycle_endpoint.set(value.decode("latin-1"))
                    try:
                        await self.app(scope, receive, send)
                    finally:
                        _request_lifecycle_endpoint.reset(token)
                    return
        await self.app(scope, receive, send)


def encode_lifecycle_events(events: List[tuple]) -> bytes:
    return msgpack.packb([[hook, list(args)] for hook, args in events])


def decode_lifecycle_events(payload: bytes) -> List[tuple]:
    return [(hook, tuple(args)) for hook, args in msgpack.unpackb(payload)]


class LifecycleEventForwarder:
    """Ordered, non-blocking bridge from the engine to the LLMRouter replicas'
    KVTokenTrackers, which maintain per-replica token load statistics.

    ``report`` broadcasts to every ingress replica: it only enqueues locally, and
    a single delivery task per replica drains the queue to the trackers, awaiting
    one broadcast at a time so events arrive everywhere in the order they were
    reported. Events that pile up during a call are sent together in the next one.

    ``report_to`` sends to one router's lifecycle endpoint over ZMQ and falls back
    to ``report`` when the endpoint is unknown or the push cannot be queued.
    """

    def __init__(self, handle: Any, worker_id: int):
        self.handle = handle
        self.worker_id = worker_id
        self._events: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None
        self._sender: Optional[token_channel.TokenSender] = None

    def report(self, method_name: str, *args) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._events.put_nowait((method_name, args))

    def report_to(self, endpoint: Optional[str], method_name: str, *args) -> None:
        if endpoint:
            if self._sender is None:
                self._sender = token_channel.TokenSender()
            payload = encode_lifecycle_events([(method_name, args)])
            if self._sender.push(endpoint, "lifecycle", payload):
                return
        self.report(method_name, *args)

    async def _deliver(self) -> None:
        while True:
            # Wait for the next event, then drain whatever queued up behind it
            # into the same batch.
            batch = [await self._events.get()]
            while not self._events.empty():
                batch.append(self._events.get_nowait())
            try:
                # return_exceptions so one failed replica cannot mask delivery
                # to the rest; failures are logged and dropped below.
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
                # Best-effort delivery: any failure (no LLMRouter replica being
                # available during a redeploy, backpressure, a booking error) is
                # dropped so the engine's token stream is never disrupted; report()
                # restarts the delivery task on the next event.
                logger.warning("Dropping KV lifecycle events: %s", e)
            finally:
                for _ in batch:
                    self._events.task_done()

    async def flush(self) -> None:
        """Wait until every reported event has been delivered."""
        await self._events.join()

    def close(self) -> None:
        """Cancel the delivery task and close sockets on engine shutdown."""
        if self._delivery_task is not None:
            self._delivery_task.cancel()
            self._delivery_task = None
        if self._sender is not None:
            self._sender.close()
            self._sender = None


class LifecycleReceiver:
    """ZMQ PULL endpoint through which engine replicas report the lifecycle
    events of requests this router selected.

    Messages are applied in arrival order; whatever queued up behind a message
    is handed to ``handler`` in the same call.
    """

    def __init__(
        self,
        *,
        bind_endpoint: str,
        handler: Callable[[List[tuple]], Awaitable[None]],
        receive_queue_limit: int = KV_TOKEN_ZMQ_RECEIVE_QUEUE_LIMIT,
    ):
        self._bind_endpoint = bind_endpoint
        self._handler = handler
        self._receive_queue_limit = receive_queue_limit
        self._socket = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        context = zmq_asyncio.Context.instance()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, self._receive_queue_limit)
        try:
            socket.bind(self._bind_endpoint)
        except Exception:
            socket.close(linger=0)
            logger.exception(
                "Failed to bind KV lifecycle endpoint %s.", self._bind_endpoint
            )
            return False
        self._socket = socket
        self._task = asyncio.create_task(self._run())
        logger.info("KV lifecycle events listening on %s.", self._bind_endpoint)
        return True

    async def _run(self) -> None:
        socket = self._socket
        while True:
            try:
                messages = [await socket.recv_multipart()]
                while len(messages) < _MAX_DRAIN:
                    try:
                        messages.append(await socket.recv_multipart(flags=zmq.NOBLOCK))
                    except zmq.Again:
                        break
            except asyncio.CancelledError:
                raise
            except (zmq.ContextTerminated, zmq.ZMQError) as e:
                if socket.closed or getattr(e, "errno", None) in (
                    zmq.ETERM,
                    zmq.ENOTSOCK,
                ):
                    break
                logger.exception("Failed to receive KV lifecycle events.")
                await asyncio.sleep(1)
                continue
            events = []
            for parts in messages:
                if len(parts) != 2:
                    logger.warning(
                        "Dropping KV lifecycle message with %d frame(s).", len(parts)
                    )
                    continue
                try:
                    events.extend(decode_lifecycle_events(parts[1]))
                except Exception:
                    logger.exception("Dropping undecodable KV lifecycle message.")
            if events:
                try:
                    await self._handler(events)
                except Exception:
                    logger.exception("Failed to apply KV lifecycle events.")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
