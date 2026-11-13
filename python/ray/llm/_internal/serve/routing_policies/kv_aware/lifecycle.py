from enum import Enum
from typing import Optional

import msgpack
import zmq

import ray
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    SELECTION_MODEL_NAME,
    SELECTION_ROUTING_GROUP,
)

logger = get_logger(__name__)

# The selection service's topic for request events.
_TOPIC = b"dynamo.slot-tracker.v1"


class LifecycleEvent(str, Enum):
    PREFILL_COMPLETE = "MarkPrefillCompleted"
    FINISHED = "Free"


class LifecyclePublisher:
    """Publishes this LLMServer's request lifecycle events to every LLMRouter.

    Each router's selection service subscribes to the advertised endpoint and
    applies the events to its booking of the request, so the load is released
    even if the router that selected the request is gone. Best effort: events
    sent before a router subscribes, or while it falls behind, are dropped.
    """

    def __init__(self):
        self._socket: Optional[zmq.Socket] = None
        self._event: dict = {}

    def start(self, worker_id: int, dp_rank: int, block_size: int) -> str:
        """Bind to a free port and return the endpoint to advertise."""
        socket = zmq.Context.instance().socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            port = socket.bind_to_random_port("tcp://*")
        except zmq.ZMQError:
            socket.close()
            raise
        self._socket = socket
        self._event = {
            "model_name": SELECTION_MODEL_NAME,
            "routing_group": SELECTION_ROUTING_GROUP,
            "block_size": block_size,
            "event": {
                "worker": {"worker_id": worker_id, "dp_rank": dp_rank},
                # The worker's own id marks the event as the worker's.
                "router_id": worker_id,
                "lora_name": None,
            },
        }
        endpoint = f"tcp://{ray.util.get_node_ip_address()}:{port}"
        logger.info("Publishing KV lifecycle events on %s.", endpoint)
        return endpoint

    def publish(self, request_id: str, event: LifecycleEvent) -> None:
        """Publish ``event`` for ``request_id`` without blocking."""
        if self._socket is None:
            return
        message = {
            **self._event,
            "event": {
                **self._event["event"],
                "request_id": request_id,
                "data": event.value,
            },
        }
        try:
            self._socket.send_multipart(
                [_TOPIC, msgpack.packb(message)], flags=zmq.NOBLOCK
            )
        except zmq.ZMQError as e:
            logger.warning("Dropping KV lifecycle event for %s: %s", request_id, e)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
