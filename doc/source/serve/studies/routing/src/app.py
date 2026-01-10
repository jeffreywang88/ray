"""
Ray Serve application for routing algorithm study.

Contains Parent and Child deployments that form a simple request chain:
HAProxy -> Parent -> Child

Parent deployment receives external requests and forwards to Child.
Child deployment performs simulated work with exponential latency distribution.
"""

import asyncio
import random
import time
from typing import Optional

import ray
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle

from src.configurations import CPU_PER_REPLICA


@serve.deployment(
    ray_actor_options={"num_cpus": CPU_PER_REPLICA, "resources": {"deployments": 1}},
    max_ongoing_requests=5,
    max_queued_requests=-1,
)
class ChildDeployment:
    """
    Child deployment that simulates work with variable latency.

    Latency follows exponential distribution with mean=10ms, capped at 100ms.
    Returns replica ID for routing analysis.
    """

    def __init__(self):
        self.replica_id = serve.get_replica_context().replica_id.unique_id
        self.node_id = ray.get_runtime_context().get_node_id()
        self.request_count = 0

    async def __call__(
        self, parent_replica_id: str, parent_node_id: str, parent_send_time: float
    ) -> dict:
        # Record when child receives the request - routing delay is the difference
        # Uses wall-clock time (time.time()) which is synchronized across nodes via NTP
        child_receive_time = time.time()
        routing_delay_ms = (child_receive_time - parent_send_time) * 1000
        
        self.request_count += 1

        # Exponential distribution with mean=10ms to mimic real-world variance
        # - Most requests complete quickly
        # - Occasional slow requests (tail latency)
        latency_s = random.expovariate(1 / 0.01)  # mean = 10ms
        latency_s = min(latency_s, 0.1)  # Cap at 100ms to avoid extreme outliers
        await asyncio.sleep(latency_s)

        # Record when child sends response (for return path timing)
        child_send_time = time.time()

        return {
            "child_replica_id": self.replica_id,
            "child_node_id": self.node_id,
            "parent_replica_id": parent_replica_id,
            "parent_node_id": parent_node_id,
            "simulated_latency_ms": latency_s * 1000,
            "parent_to_child_delay_ms": routing_delay_ms,
            "child_send_time": child_send_time,
        }


@serve.deployment(
    ray_actor_options={"num_cpus": CPU_PER_REPLICA, "resources": {"deployments": 1}},
    max_ongoing_requests=5,
    max_queued_requests=-1,
)
class ParentDeployment:
    """
    Parent deployment that receives external requests and forwards to Child.

    Acts as the entry point for the request chain.
    Returns combined routing information from both Parent and Child.
    """

    def __init__(
        self,
        child_handle: DeploymentHandle,
        prefer_local_routing: bool = False,
    ):
        self.child_handle = child_handle

        # Apply locality preference to the handle using _init()
        # Must be called before .remote() or .options()
        if prefer_local_routing:
            self.child_handle._init(_prefer_local_routing=True)

        self.replica_id = serve.get_replica_context().replica_id.unique_id
        self.node_id = ray.get_runtime_context().get_node_id()
        self.request_count = 0

    def health(self) -> dict:
        """Health check endpoint - returns replica info without calling child."""
        return {
            "status": "healthy",
            "replica_id": self.replica_id,
            "node_id": self.node_id,
            "request_count": self.request_count,
        }

    async def __call__(self, client_send_time: float = None) -> dict:
        """
        Handle incoming request.
        
        Args:
            client_send_time: Wall-clock time when client sent the request (for routing delay).
        
        Returns:
            Response dict with parent/child replica and timing info.
        """
        # Record when parent receives the request (wall-clock for cross-process timing)
        parent_receive_time = time.time()
        self.request_count += 1

        # Calculate client → parent routing delay
        client_to_parent_delay_ms = None
        if client_send_time is not None:
            client_to_parent_delay_ms = (parent_receive_time - client_send_time) * 1000

        # Record send time for parent → child routing delay measurement
        parent_send_time = time.time()
        
        # Forward to child and get response
        child_response = await self.child_handle.remote(
            self.replica_id, self.node_id, parent_send_time
        )

        # Record when parent receives child's response (return path timing)
        parent_receive_child_time = time.time()
        
        # Calculate child → parent return delay
        child_send_time = child_response.get("child_send_time")
        child_to_parent_delay_ms = None
        if child_send_time is not None:
            child_to_parent_delay_ms = (parent_receive_child_time - child_send_time) * 1000

        # Add timing info to response
        child_response["client_to_parent_delay_ms"] = client_to_parent_delay_ms
        child_response["child_to_parent_delay_ms"] = child_to_parent_delay_ms
        
        # Record when parent sends response to client (for parent → client delay)
        child_response["parent_send_response_time"] = time.time()
        
        return child_response


def build_app(
    parent_replicas: int,
    child_replicas: int,
    prefer_local_routing: bool = False,
    request_router_class: Optional[str] = None,
) -> serve.Application:
    """
    Build the Serve application with specified configuration.

    Args:
        parent_replicas: Number of Parent deployment replicas.
        child_replicas: Number of Child deployment replicas.
        prefer_local_routing: Whether to prefer routing to replicas on the same node.
        request_router_class: Import path to custom request router class.
            None uses default Pow2 router.
            Examples:
                - "src.routers.RandomRequestRouter"
                - "src.routers.RoundRobinRequestRouter"

    Returns:
        Configured Serve application ready for deployment.
    """
    # Configure Child deployment
    child_config = {
        "num_replicas": child_replicas,
    }
    if request_router_class:
        child_config["request_router_config"] = RequestRouterConfig(
            request_router_class=request_router_class,
        )

    child = ChildDeployment.options(**child_config)
    child_handle = child.bind()

    # Configure Parent deployment
    # Note: request_router_config only affects how OTHER deployments route TO this one,
    # so we don't need to set it on Parent (external requests come via HTTP, not handles)
    parent_config = {
        "num_replicas": parent_replicas,
    }

    # Pass locality preference to ParentDeployment constructor
    # The handle._init() call happens inside the deployment's __init__
    parent = ParentDeployment.options(**parent_config).bind(
        child_handle,
        prefer_local_routing,
    )

    return parent


# Convenience function for quick testing
def run_app(
    parent_replicas: int = 2,
    child_replicas: int = 2,
    prefer_local_routing: bool = False,
    request_router_class: Optional[str] = None,
    blocking: bool = False,
):
    """Deploy and run the application for testing."""
    app = build_app(
        parent_replicas=parent_replicas,
        child_replicas=child_replicas,
        prefer_local_routing=prefer_local_routing,
        request_router_class=request_router_class,
    )
    serve.run(app, blocking=blocking, name="routing-study")
    print(f"Application deployed with {parent_replicas} parent, {child_replicas} child replicas")
    print("Send requests to http://localhost:8000")


if __name__ == "__main__":
    run_app(blocking=True, parent_replicas=128, child_replicas=128)
