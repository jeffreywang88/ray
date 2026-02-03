"""Gang scheduling configuration and context for Ray Serve deployments.

Gang scheduling ensures that groups of replicas (gangs) are scheduled together
atomically, which is essential for distributed training and inference workloads
that require tight coordination between replicas.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ray._common.pydantic_compat import BaseModel, Field
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.util.annotations import PublicAPI

logger = logging.getLogger(SERVE_LOGGER_NAME)


class GangPlacementStrategy(str, Enum):
    """Placement strategy for replicas within a gang."""

    PACK = "PACK"
    """Pack replicas on as few nodes as possible (best effort)."""

    SPREAD = "SPREAD"
    """Spread replicas across distinct nodes as evenly as possible (best effort)."""

    STRICT_PACK = "STRICT_PACK"
    """Pack all replicas on a single node. The gang is not allowed to span multiple nodes."""

    STRICT_SPREAD = "STRICT_SPREAD"
    """Place replicas across distinct nodes. Only one replica per node is allowed."""


class GangRuntimeFailurePolicy(str, Enum):
    """Policy for handling runtime failures of replicas in a gang."""

    RESTART_GANG = "RESTART_GANG"
    """Kill and restart entire gang atomically when any replica fails.
    Use for: Tightly coupled systems where partial gang is useless.
    Ensures consistency but higher recovery time."""

    RESTART_REPLICA = "RESTART_REPLICA"
    """Kill and restart individual replica when it fails.
    Use for: Systems that can tolerate partial gang availability.
    Faster recovery but may result in inconsistent state."""


@PublicAPI(stability="alpha")
class GangSchedulingConfig(BaseModel):
    """Configuration for gang scheduling of deployment replicas.

    Gang scheduling ensures that groups of replicas are scheduled together
    atomically, which is essential for distributed workloads that require
    coordination between replicas.

    Example:
        .. code-block:: python

            from ray import serve
            from ray.serve.gang_scheduling import GangSchedulingConfig, GangPlacementStrategy

            @serve.deployment(
                num_replicas=8,
                gang_scheduling_config=GangSchedulingConfig(
                    gang_size=4,
                    gang_placement_strategy=GangPlacementStrategy.STRICT_PACK,
                    runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_GANG
                )
            )
            class MyDeployment:
                pass
    """

    gang_size: int = Field(
        description=(
            "Number of replicas per gang. "
            "num_replicas must be a multiple of gang_size."
        ),
        ge=1,
    )

    gang_timeout_s: float = Field(
        default=300.0,
        description="Maximum time to wait for gang scheduling (seconds).",
        gt=0,
    )

    gang_placement_strategy: GangPlacementStrategy = Field(
        default=GangPlacementStrategy.PACK,
        description=(
            "Placement strategy for replicas within a gang. "
            "Options: PACK (pack with best effort, default), "
            "SPREAD (maximize availability), "
            "STRICT_PACK (pack on single node), "
            "STRICT_SPREAD (one per node)."
        ),
    )

    max_retries: int = Field(
        default=3,
        description="Maximum gang scheduling retry attempts.",
        ge=0,
    )

    runtime_failure_policy: GangRuntimeFailurePolicy = Field(
        default=GangRuntimeFailurePolicy.RESTART_GANG,
        description=(
            "What to do when a replica fails after gang is running. "
            "RESTART_GANG: kill and restart entire gang atomically. "
            "RESTART_REPLICA: kill and restart individual replica."
        ),
    )


@PublicAPI(stability="alpha")
@dataclass
class GangContext:
    """Context information for a replica that is part of a gang.

    This context provides information about the gang membership, including
    the replica's rank within the gang and the identities of all gang members.

    Attributes:
        gang_id: Unique identifier for this gang.
        rank: This replica's rank within the gang (0-indexed).
        world_size: Total number of replicas in this gang.
        member_replica_ids: List of replica IDs in this gang, ordered by rank.
    """

    gang_id: str  # Unique identifier for this gang
    rank: int  # This replica's rank within the gang (0-indexed)
    world_size: int  # Total number of replicas in this gang
    member_replica_ids: List[str]  # List of replica IDs in this gang, ordered by rank
