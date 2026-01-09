"""
Custom request routers for routing algorithm study.

Implements Random and Round-Robin routing algorithms to compare against
Ray Serve's default Power-of-Two Choices (Pow2) router.
"""

import random
from typing import List, Optional

from ray.serve.request_router import (
    FIFOMixin,
    PendingRequest,
    RequestRouter,
    ReplicaID,
    ReplicaResult,
    RunningReplica,
)


class RandomRequestRouter(FIFOMixin, RequestRouter):
    """
    Random request router that selects replicas uniformly at random.

    This router ignores queue lengths and other replica state, providing
    a baseline for comparison against load-aware algorithms like Pow2.

    Uses FIFOMixin to ensure requests are processed in order.
    """

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        """
        Select a random replica from candidates.

        Args:
            candidate_replicas: List of available replicas.
            pending_request: The pending request (unused for random selection).

        Returns:
            Single-element ranked list containing one randomly selected replica.
        """
        if not candidate_replicas:
            return [[]]

        selected = random.choice(candidate_replicas)
        return [[selected]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ):
        """Callback after request is routed (no-op for random router)."""
        pass


class RoundRobinRequestRouter(FIFOMixin, RequestRouter):
    """
    Round-Robin request router that cycles through replicas in order.

    This router maintains perfect fairness by visiting each replica in
    sequence. It ignores queue lengths but ensures equal distribution.

    Uses FIFOMixin to ensure requests are processed in order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter = 0
        self._replica_order: List[ReplicaID] = []

    def update_replicas(self, replicas: List[RunningReplica]):
        """
        Update the list of available replicas.

        Called by Serve when replicas are added or removed.
        Maintains stable ordering for consistent round-robin behavior.
        """
        super().update_replicas(replicas)

        # Build stable ordering based on replica IDs
        current_ids = {r.replica_id for r in replicas}

        # Keep existing order for replicas that still exist
        self._replica_order = [
            rid for rid in self._replica_order if rid in current_ids
        ]

        # Add new replicas at the end
        existing_ids = set(self._replica_order)
        for replica in replicas:
            if replica.replica_id not in existing_ids:
                self._replica_order.append(replica.replica_id)

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        """
        Select the next replica in round-robin order.

        Args:
            candidate_replicas: List of available replicas.
            pending_request: The pending request (unused for round-robin).

        Returns:
            Single-element ranked list containing the next replica in sequence.
        """
        if not candidate_replicas:
            return [[]]

        # Build lookup for quick access
        replica_map = {r.replica_id: r for r in candidate_replicas}

        # Find the next available replica in our ordering
        # (some replicas in our order might not be in current candidates)
        available_order = [
            rid for rid in self._replica_order if rid in replica_map
        ]

        if not available_order:
            # Fallback: just pick first candidate
            return [[candidate_replicas[0]]]

        # Select next in round-robin sequence
        index = self._counter % len(available_order)
        self._counter += 1

        selected_id = available_order[index]
        return [[replica_map[selected_id]]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ):
        """Callback after request is routed (no-op for round-robin router)."""
        pass

