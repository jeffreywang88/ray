"""
Experiment configuration matrix for routing algorithm study.

Defines all 135 experiment configurations across:
- 3 algorithms (Pow2, Random, RoundRobin)
- 4 scales (Small, Medium, Large, XLarge)
- 3 ratios (1:1, 1:2, 2:1) - only at Large scale
- 2 topologies (Packed, Spread) - only at Large scale
- 2 locality settings (Preferred, None) - only at Large scale
- 3 load levels (50%, 75%, 100%)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, List, Optional
import hashlib


class Algorithm(Enum):
    """Routing algorithm options."""

    POW2 = "pow2"  # Default Power-of-Two Choices
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"

    def get_router_class(self) -> Optional[str]:
        """Return import path for custom router, or None for default Pow2."""
        if self == Algorithm.POW2:
            return None
        elif self == Algorithm.RANDOM:
            return "src.routers.RandomRequestRouter"
        elif self == Algorithm.ROUND_ROBIN:
            return "src.routers.RoundRobinRequestRouter"


class Scale(Enum):
    """Cluster scale options."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    # XLARGE = "xlarge" # disbale untill we convert script to anyscale service and then we can use anyscale SDK to launch service
    # and hit the AWS ALB endpoint. For now we are always hitting the local endpoint, which hits the same HAProxy endpoint.


class Ratio(Enum):
    """Parent:Child replica ratio options."""

    ONE_TO_ONE = "1:1"
    ONE_TO_TWO = "1:2"
    TWO_TO_ONE = "2:1"


class Topology(Enum):
    """Replica placement topology options."""

    PACKED = "packed"
    SPREAD = "spread"


class LoadLevel(Enum):
    """Load level as percentage of theoretical max throughput."""

    LOW = 0.50
    MEDIUM = 0.75
    HIGH = 1.00


# Scale definitions: base replica counts for 1:1 ratio
SCALE_REPLICAS = {
    Scale.SMALL: 8,
    Scale.MEDIUM: 32,
    Scale.LARGE: 128,
    # Scale.XLARGE: 512, # disable untill we convert script to anyscale service and then we can use anyscale SDK to launch service
    # and hit the AWS ALB endpoint. For now we are always hitting the local endpoint, which hits the same HAProxy endpoint.
}

# Realistic max RPS per replica
# Theoretical: 5 concurrent × (1000ms / 10ms) = 500 RPS
# But actual capacity is much lower due to:
# - Routing overhead (queue probes, network RTT)
# - Serialization/deserialization
# - asyncio scheduling
# - Parent→Child chain adds latency
# Empirically measured with Locust: ~120 RPS per replica at scale
RPS_PER_REPLICA = 300

# Concurrent users per replica for closed-loop load testing
# This controls how much load we put on each replica
CONCURRENT_PER_REPLICA = 8


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for a single experiment run.

    Immutable dataclass representing all parameters for one experiment.
    """

    algorithm: Algorithm
    scale: Scale
    ratio: Ratio
    topology: Topology
    locality: bool  # prefer_local_routing
    load_level: LoadLevel

    @property
    def parent_replicas(self) -> int:
        """Number of Parent deployment replicas."""
        base = SCALE_REPLICAS[self.scale]
        if self.ratio == Ratio.ONE_TO_TWO:
            return base
        elif self.ratio == Ratio.TWO_TO_ONE:
            return base * 2
        else:  # 1:1
            return base

    @property
    def child_replicas(self) -> int:
        """Number of Child deployment replicas."""
        base = SCALE_REPLICAS[self.scale]
        if self.ratio == Ratio.ONE_TO_TWO:
            return base * 2
        elif self.ratio == Ratio.TWO_TO_ONE:
            return base
        else:  # 1:1
            return base

    @property
    def bottleneck_replicas(self) -> int:
        """Number of replicas in the bottleneck deployment."""
        return min(self.parent_replicas, self.child_replicas)

    @property
    def theoretical_max_rps(self) -> int:
        """Theoretical maximum RPS based on bottleneck replicas."""
        return self.bottleneck_replicas * RPS_PER_REPLICA

    @property
    def target_rps(self) -> int:
        """Target RPS for this load level (estimated from concurrent users)."""
        return int(self.theoretical_max_rps * self.load_level.value)

    @property
    def num_concurrent(self) -> int:
        """Number of concurrent users for closed-loop load testing."""
        # Scale concurrent users based on bottleneck replicas and load level
        base_concurrent = self.bottleneck_replicas * CONCURRENT_PER_REPLICA
        return int(base_concurrent * self.load_level.value)

    @property
    def total_replicas(self) -> int:
        """Total number of replicas across both deployments."""
        return self.parent_replicas + self.child_replicas

    @property
    def cpus_required(self) -> float:
        """CPUs required for this configuration (0.25 per replica)."""
        return self.total_replicas * 0.25

    @property
    def config_hash(self) -> str:
        """Short hash identifying this configuration."""
        config_str = (
            f"{self.algorithm.value}_{self.scale.value}_{self.ratio.value}_"
            f"{self.topology.value}_{self.locality}_{self.load_level.value}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "algorithm": self.algorithm.value,
            "scale": self.scale.value,
            "ratio": self.ratio.value,
            "topology": self.topology.value,
            "locality": self.locality,
            "load_level": self.load_level.value,
            "parent_replicas": self.parent_replicas,
            "child_replicas": self.child_replicas,
            "num_concurrent": self.num_concurrent,
            "target_rps": self.target_rps,
            "theoretical_max_rps": self.theoretical_max_rps,
            "config_hash": self.config_hash,
        }

    def __str__(self) -> str:
        return (
            f"Config[{self.algorithm.value}, {self.scale.value}, {self.ratio.value}, "
            f"{self.topology.value}, locality={self.locality}, load={self.load_level.value}]"
        )


def generate_medium_scale_configs() -> Generator[ExperimentConfig, None, None]:
    """
    Generate all configurations for Medium scale (full variation).

    Medium scale tests all combinations:
    - 3 algorithms × 3 ratios × 2 topologies × 2 localities × 3 load levels = 108 configs
    """
    for algorithm in Algorithm:
        for ratio in Ratio:
            for topology in Topology:
                for locality in [False, True]:
                    for load_level in LoadLevel:
                        yield ExperimentConfig(
                            algorithm=algorithm,
                            scale=Scale.MEDIUM,
                            ratio=ratio,
                            topology=topology,
                            locality=locality,
                            load_level=load_level,
                        )


def generate_other_scale_configs() -> Generator[ExperimentConfig, None, None]:
    """
    Generate configurations for Small, Medium, XLarge scales.

    These scales use fixed settings: 1:1 ratio, Packed topology, No locality.
    - 3 algorithms × 3 scales × 3 load levels = 27 configs
    """
    for algorithm in Algorithm:
        for scale in [Scale.SMALL, Scale.LARGE]: #, Scale.XLARGE]:
            for load_level in LoadLevel:
                yield ExperimentConfig(
                    algorithm=algorithm,
                    scale=scale,
                    ratio=Ratio.ONE_TO_ONE,
                    topology=Topology.PACKED,
                    locality=False,
                    load_level=load_level,
                )


def generate_all_configs() -> List[ExperimentConfig]:
    """
    Generate all 135 experiment configurations.

    Returns:
        List of all configurations for the full experiment matrix.
    """
    configs = []
    configs.extend(generate_medium_scale_configs())
    configs.extend(generate_other_scale_configs())
    return configs


def generate_prioritized_configs() -> List[ExperimentConfig]:
    """
    Generate prioritized subset of 36 configurations.

    Focus on Large scale at 75% load to compare algorithms across all variations:
    - 3 algorithms × 3 ratios × 2 topologies × 2 localities × 1 load level = 36 configs
    """
    configs = []
    for algorithm in Algorithm:
        for ratio in Ratio:
            for topology in Topology:
                for locality in [False, True]:
                    configs.append(
                        ExperimentConfig(
                            algorithm=algorithm,
                            scale=Scale.LARGE,
                            ratio=ratio,
                            topology=topology,
                            locality=locality,
                            load_level=LoadLevel.MEDIUM,  # 75%
                        )
                    )
    return configs


def generate_quick_configs() -> List[ExperimentConfig]:
    """
    Generate minimal configuration set for quick testing.

    Tests each algorithm at Small scale, 1:1 ratio, 75% load.
    - 3 algorithms × 1 scale × 1 load level = 3 configs
    """
    return [
        ExperimentConfig(
            algorithm=algorithm,
            scale=Scale.SMALL,
            ratio=Ratio.ONE_TO_ONE,
            topology=Topology.PACKED,
            locality=False,
            load_level=LoadLevel.MEDIUM,
        )
        for algorithm in Algorithm
    ]


def get_configs_by_run_type(run_type: str) -> List[ExperimentConfig]:
    """
    Get configurations for a specific run type.

    Args:
        run_type: One of "full", "prioritized", or "quick".

    Returns:
        List of configurations for the specified run type.

    Raises:
        ValueError: If run_type is not recognized.
    """
    if run_type == "full":
        return generate_all_configs()
    elif run_type == "prioritized":
        return generate_prioritized_configs()
    elif run_type == "quick":
        return generate_quick_configs()
    else:
        raise ValueError(f"Unknown run type: {run_type}. Use 'full', 'prioritized', or 'quick'.")


def print_config_summary(configs: List[ExperimentConfig]) -> None:
    """Print summary of configuration list."""
    print(f"\nTotal configurations: {len(configs)}")
    print("\nBreakdown by algorithm:")
    for alg in Algorithm:
        count = sum(1 for c in configs if c.algorithm == alg)
        print(f"  {alg.value}: {count}")

    print("\nBreakdown by scale:")
    for scale in Scale:
        count = sum(1 for c in configs if c.scale == scale)
        if count > 0:
            print(f"  {scale.value}: {count}")

    print("\nSample configurations:")
    for config in configs[:3]:
        print(f"  {config}")
    if len(configs) > 3:
        print(f"  ... and {len(configs) - 3} more")


if __name__ == "__main__":
    print("=== Full Configuration Matrix ===")
    print_config_summary(generate_all_configs())

    print("\n=== Prioritized Subset ===")
    print_config_summary(generate_prioritized_configs())

    print("\n=== Quick Test ===")
    print_config_summary(generate_quick_configs())
