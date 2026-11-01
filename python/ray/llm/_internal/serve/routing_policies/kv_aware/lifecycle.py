"""Delivery of request lifecycle events from engine replicas to LLMRouter replicas.

Engine replicas report ``on_prefill_complete`` / ``on_request_completed`` so the
routers' selection services can release the load they booked for a request.

- The router that selected a request names itself in a request header, which
  ``LifecycleOwnerMiddleware`` exposes to the engine as the request's owner.
- ``LifecycleEventForwarder`` calls that owner's ``on_lifecycle_events`` over a
  Serve handle pinned to it. The owner's selection service replicates the
  change to its peers over Dynamo's replica mesh.

This module stays free of engine imports so any engine integration can use it.
"""

import asyncio
import contextvars
from typing import Any, Dict, Optional

from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_LIFECYCLE_OWNER_HEADER,
    LIFECYCLE_EVENT_DELIVERY_TIMEOUT_S,
)
from ray.serve.exceptions import DeploymentUnavailableError
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)

# An owner's delivery task exits after this long without events, so routers that
# left leave no state behind; a later event for the owner starts a new one.
_OWNER_IDLE_TIMEOUT_S = 60

_LIFECYCLE_OWNER_HEADER_BYTES = KV_LIFECYCLE_OWNER_HEADER.encode("latin-1")

# Full replica id of the LLMRouter replica that selected the request being served.
_request_lifecycle_owner: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar("kv_request_lifecycle_owner", default=None)


def get_request_lifecycle_owner() -> Optional[str]:
    """Replica id of the router that selected the current request."""
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


class LifecycleEventForwarder:
    """Ordered, non-blocking bridge from the engine to the KVTokenTracker of the
    LLMRouter replica that selected each request.

    ``report_to`` only enqueues locally, so generation never blocks on delivery.
    One delivery task per owner replica drains that owner's queue over a handle
    pinned to the replica, awaiting one call at a time so the owner applies
    events in the order they were reported. Events that pile up during a call
    are sent together in the next one.
    """

    def __init__(self, handle: DeploymentHandle):
        if not handle.is_initialized:
            # Pinned handles share this handle's router and its replica set.
            handle._init()
        self._handle = handle
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def report_to(self, owner: str, method_name: str, *args) -> None:
        queue = self._queues.get(owner)
        if queue is None:
            queue = self._queues[owner] = asyncio.Queue()
        task = self._tasks.get(owner)
        if task is None or task.done():
            self._tasks[owner] = asyncio.get_running_loop().create_task(
                self._deliver(owner, queue)
            )
        queue.put_nowait((method_name, args))

    async def _deliver(self, owner: str, queue: asyncio.Queue) -> None:
        handle = self._handle.options(replica_id=owner)
        while True:
            # Wait for the next event, then drain whatever queued up behind it
            # into the same batch.
            try:
                event = await asyncio.wait_for(queue.get(), _OWNER_IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                if not queue.empty():
                    continue
                del self._queues[owner]
                del self._tasks[owner]
                return
            batch = [event]
            while not queue.empty():
                batch.append(queue.get_nowait())
            try:
                await asyncio.wait_for(
                    handle.on_lifecycle_events.remote(batch),
                    LIFECYCLE_EVENT_DELIVERY_TIMEOUT_S,
                )
            except DeploymentUnavailableError:
                # The owner replica is gone, and with it the load it booked, so
                # drop its queue; a later event for it starts a new one.
                logger.info(
                    "Dropping KV lifecycle events for departed router %s.", owner
                )
                del self._queues[owner]
                del self._tasks[owner]
                for _ in range(len(batch) + queue.qsize()):
                    queue.task_done()
                return
            except Exception as e:
                # Best-effort delivery: any other failure (a timeout, a booking
                # error) is dropped so the engine's token stream is never
                # disrupted. The owner frees the request at its stale-request TTL.
                logger.warning(
                    "Dropping %d KV lifecycle events for router %s: %s",
                    len(batch),
                    owner,
                    e,
                )
            for _ in batch:
                queue.task_done()

    async def flush(self) -> None:
        """Wait until every reported event has been delivered or dropped."""
        await asyncio.gather(*(queue.join() for queue in list(self._queues.values())))

    def close(self) -> None:
        """Cancel the delivery tasks on engine shutdown."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._queues.clear()
