"""Cluster helpers: scale to a phase's nodes, check placement, sample node CPU."""

import asyncio
import collections
import logging
import time
from typing import Dict, List, Tuple

import numpy as np

import ray
from ray.autoscaler.sdk import request_resources
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors

logger = logging.getLogger(__name__)

# Custom resource of the load-generator worker group (see the compute config).
CLIENT_RESOURCE = "agentx_client"
INGRESS_PORT = 8000


def _alive_nodes() -> List[dict]:
    return [n for n in ray.nodes() if n["Alive"]]


def head_node_id() -> str:
    for n in _alive_nodes():
        if "node:__internal_head__" in n["Resources"]:
            return n["NodeID"]
    raise RuntimeError("no alive head node")


def _engine_nodes(node_cpus: float) -> List[dict]:
    head = head_node_id()
    nodes = [
        n
        for n in _alive_nodes()
        if n["NodeID"] != head and n["Resources"].get("CPU", 0) >= node_cpus
    ]
    return sorted(nodes, key=lambda n: n["NodeID"])


def _client_nodes() -> List[dict]:
    nodes = [n for n in _alive_nodes() if n["Resources"].get(CLIENT_RESOURCE)]
    return sorted(nodes, key=lambda n: n["NodeID"])


def scale(
    num_engine_nodes: int, node_cpus: float, num_client_nodes: int, timeout_s: float
) -> Tuple[List[dict], List[dict]]:
    """Waits for the engine and client nodes and returns the ones to use."""
    request_resources(
        bundles=[{"CPU": node_cpus}] * num_engine_nodes
        + [{CLIENT_RESOURCE: 1}] * num_client_nodes
    )
    start = time.monotonic()
    while True:
        engines, clients = _engine_nodes(node_cpus), _client_nodes()
        if len(engines) >= num_engine_nodes and len(clients) >= num_client_nodes:
            return engines[:num_engine_nodes], clients[:num_client_nodes]
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"{len(engines)}/{num_engine_nodes} engine nodes and "
                f"{len(clients)}/{num_client_nodes} client nodes after {timeout_s}s"
            )
        time.sleep(10)


def ingress_urls(nodes: List[dict]) -> List[str]:
    return [f"http://{n['NodeManagerAddress']}:{INGRESS_PORT}" for n in nodes]


def _actors_per_node(name_fragment: str) -> Dict[str, int]:
    actors = list_actors(filters=[("state", "=", "ALIVE")], limit=100_000)
    return collections.Counter(
        a.node_id for a in actors if a.name and name_fragment in a.name
    )


def wait_for_placement(
    node_ids: List[str],
    engines: int,
    engine_deployment: str,
    routers_per_node: int,
    timeout_s: float = 300,
) -> int:
    """Waits until the engines and routers are placed and returns the router count.

    The engines must be spread evenly over ``node_ids`` and every engine node must
    run ``routers_per_node`` LLMRouter replicas.
    """
    per_node = engines // len(node_ids)
    start = time.monotonic()
    while True:
        engine_counts = _actors_per_node(engine_deployment)
        router_counts = _actors_per_node("LLMRouter")
        problems = []
        if dict(engine_counts) != {n: per_node for n in node_ids}:
            problems.append(f"engines per node {dict(engine_counts)}, want {per_node}")
        short = [n for n in node_ids if router_counts.get(n, 0) != routers_per_node]
        if short:
            problems.append(
                f"{len(short)} engine nodes without {routers_per_node} routers"
            )
        if not problems:
            return sum(router_counts.values())
        if time.monotonic() - start > timeout_s:
            raise RuntimeError("placement check failed: " + "; ".join(problems))
        time.sleep(5)


@ray.remote(num_cpus=0)
class CpuSampler:
    def __init__(self, interval_s: float):
        self._interval_s = interval_s
        self._samples: List[float] = []
        self._running = True

    async def run(self) -> None:
        import psutil

        psutil.cpu_percent()
        while self._running:
            await asyncio.sleep(self._interval_s)
            self._samples.append(psutil.cpu_percent())

    async def stop(self) -> List[float]:
        self._running = False
        return self._samples


def start_cpu_samplers(node_ids: List[str], interval_s: float = 5.0) -> dict:
    samplers = {
        node_id: CpuSampler.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote(interval_s)
        for node_id in node_ids
    }
    for sampler in samplers.values():
        sampler.run.remote()
    return samplers


def stop_cpu_samplers(samplers: dict) -> Dict[str, Dict[str, float]]:
    """Returns the max and p95 CPU percent per node."""
    samples = ray.get([s.stop.remote() for s in samplers.values()])
    for sampler in samplers.values():
        ray.kill(sampler)
    return {
        node_id: {
            "max": float(np.max(s)) if s else 0.0,
            "p95": float(np.percentile(s, 95)) if s else 0.0,
        }
        for node_id, s in zip(samplers, samples)
    }
