"""Delivery of request lifecycle events from engine replicas to LLMRouter replicas.

Engine replicas report ``on_prefill_complete`` / ``on_request_completed`` so the
routers' selection services can release the load they booked for a request.

- The router that selected a request names its lifecycle endpoint in a request
  header, which ``LifecycleOwnerMiddleware`` exposes to the engine as the
  request's owner.
- ``LifecycleEventForwarder`` pushes each event to that owner over ZMQ. The
  owner's selection service replicates the change to its peers over Dynamo's
  replica mesh. Events with no owner are broadcast to every router replica.
- ``LifecycleReceiver`` is the owner's ZMQ PULL endpoint.

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
    KV_LIFECYCLE_OWNER_HEADER,
    KV_TOKEN_ZMQ_RECEIVE_QUEUE_LIMIT,
    LIFECYCLE_EVENT_DELIVERY_TIMEOUT_S,
)
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)

# Most messages the receiver drains into one handler call.
_MAX_DRAIN = 256
_LIFECYCLE_OWNER_HEADER_BYTES = KV_LIFECYCLE_OWNER_HEADER.encode("latin-1")

# Lifecycle endpoint of the LLMRouter replica that selected the request being served.
_request_lifecycle_owner: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar("kv_request_lifecycle_owner", default=None)


def get_request_lifecycle_owner() -> Optional[str]:
    """Lifecycle endpoint of the router that selected the current request."""
    return _request_lifecycle_owner.get()


class LifecycleOwnerMiddleware:
    """ASGI middleware exposing the router-supplied lifecycle owner header to
    the engine through ``get_request_lifecycle_owner``."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            for name, value in scope.get("headers") or ():
                if name == _LIFECYCLE_OWNER_HEADER_BYTES:
                    token = _request_lifecycle_owner.set(value.decode("latin-1"))
                    try:
                        await self.app(scope, receive, send)
                    finally:
                        _request_lifecycle_owner.reset(token)
                    return
        await self.app(scope, receive, send)


def encode_lifecycle_events(events: List[tuple]) -> bytes:
    return msgpack.packb([[hook, list(args)] for hook, args in events])


def decode_lifecycle_events(payload: bytes) -> List[tuple]:
    return [(hook, tuple(args)) for hook, args in msgpack.unpackb(payload)]


class LifecycleEventForwarder:
    """Ordered, non-blocking bridge from the engine to the KVTokenTracker of the
    LLMRouter replica that selected each request.

    ``report_to`` pushes to the owner's lifecycle endpoint over a ZMQ PUSH
    socket per owner, so a request's events arrive in the order they were
    reported; a socket queues events while it connects, so a router's first
    events are not lost. Events with no owner, or whose push cannot be queued,
    are broadcast to every router replica over a Serve handle: a single delivery
    task drains them, one broadcast at a time, in the order they were reported.
    """

    def __init__(self, handle: DeploymentHandle):
        self._handle = handle
        self._sender: Optional[token_channel.TokenSender] = None
        self._broadcasts: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None

    def report_to(self, owner: Optional[str], method_name: str, *args) -> None:
        if owner is not None:
            if self._sender is None:
                self._sender = token_channel.TokenSender(immediate=False)
            payload = encode_lifecycle_events([(method_name, args)])
            if self._sender.push(owner, "lifecycle", payload):
                return
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.get_running_loop().create_task(
                self._deliver_broadcasts()
            )
        self._broadcasts.put_nowait((method_name, args))

    async def _deliver_broadcasts(self) -> None:
        while True:
            # Wait for the next event, then drain whatever queued up behind it
            # into the same batch.
            batch = [await self._broadcasts.get()]
            while not self._broadcasts.empty():
                batch.append(self._broadcasts.get_nowait())
            try:
                # return_exceptions so one failed replica cannot mask delivery
                # to the rest; failures are logged and dropped below.
                results = await self._handle.broadcast(
                    "on_lifecycle_events", batch
                ).results_async(
                    timeout_s=LIFECYCLE_EVENT_DELIVERY_TIMEOUT_S,
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(
                        "KV lifecycle events dropped on %d/%d router replicas: %s",
                        len(errors),
                        len(results),
                        errors[0],
                    )
            except Exception as e:
                # Best-effort delivery: any failure (no LLMRouter replica being
                # available during a redeploy, a booking error) is dropped so the
                # engine's token stream is never disrupted.
                logger.warning("Dropping KV lifecycle events: %s", e)
            finally:
                for _ in batch:
                    self._broadcasts.task_done()

    async def flush(self) -> None:
        """Wait until every broadcast event has been delivered or dropped."""
        await self._broadcasts.join()

    def close(self) -> None:
        """Cancel the broadcast task and close sockets on engine shutdown."""
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            self._broadcast_task = None
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
