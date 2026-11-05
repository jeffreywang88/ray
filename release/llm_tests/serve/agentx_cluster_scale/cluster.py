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


def _engine_nodes(node_cpus: float, exclude: Tuple[str, ...] = ()) -> List[dict]:
    head = head_node_id()
    nodes = [
        n
        for n in _alive_nodes()
        if n["NodeID"] != head
        and n["NodeID"] not in exclude
        and n["Resources"].get("CPU", 0) >= node_cpus
    ]
    return sorted(nodes, key=lambda n: n["NodeID"])


def _client_nodes(resource: str) -> List[dict]:
    nodes = [n for n in _alive_nodes() if n["Resources"].get(resource)]
    return sorted(nodes, key=lambda n: n["NodeID"])


def scale(
    num_engine_nodes: int,
    node_cpus: float,
    num_client_nodes: int,
    timeout_s: float,
    client_resource: str = CLIENT_RESOURCE,
    reserve_engine_nodes: int = 0,
    client_node_ids: List[str] = (),
    exclude_node_ids: List[str] = (),
) -> Tuple[List[dict], List[dict]]:
    """Waits for the engine and client nodes and returns the ones to use.

    ``reserve_engine_nodes`` keeps the largest phase's nodes requested, so the
    autoscaler starts them early and doesn't remove them between phases. Nodes
    without replicas get no proxy, so no routers, and don't affect a phase.
    With ``client_resource="CPU"``, client nodes are plain ``node_cpus``
    workers; the first ``num_client_nodes`` of them are clients, never engines.
    """
    shared = client_resource == "CPU"
    engine_bundles = [{"CPU": node_cpus}] * max(num_engine_nodes, reserve_engine_nodes)
    client_bundles = [{"CPU": node_cpus} if shared else {client_resource: 1}]
    # Excluded nodes take a node each without being usable.
    excluded = [{"CPU": node_cpus}] * len(exclude_node_ids)
    request_resources(
        bundles=engine_bundles + client_bundles * num_client_nodes + excluded
    )
    start = time.monotonic()
    while True:
        workers = _engine_nodes(node_cpus, tuple(exclude_node_ids))
        if client_node_ids:
            by_id = {n["NodeID"]: n for n in _alive_nodes()}
            clients = [by_id[i] for i in client_node_ids if i in by_id]
            engines = [n for n in workers if n["NodeID"] not in set(client_node_ids)]
            if len(clients) < len(client_node_ids):
                raise RuntimeError("some --client-node-ids are not alive")
        elif shared:
            clients, engines = workers[:num_client_nodes], workers[num_client_nodes:]
        else:
            engines, clients = workers, _client_nodes(client_resource)
        if len(engines) >= num_engine_nodes and len(clients) >= num_client_nodes:
            return engines[:num_engine_nodes], clients[:num_client_nodes]
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"{len(engines)}/{num_engine_nodes} engine nodes and "
                f"{len(clients)}/{num_client_nodes} client nodes after {timeout_s}s"
            )
        time.sleep(10)


def node_memory_gb(node_ids: List[str]) -> float:
    """The smallest total memory among the nodes, in GB."""
    import psutil

    @ray.remote(num_cpus=0)
    def total() -> float:
        return psutil.virtual_memory().total / 2**30

    return min(
        ray.get(
            [
                total.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(n, soft=False)
                ).remote()
                for n in node_ids
            ]
        )
    )


def ingress_urls(nodes: List[dict]) -> List[str]:
    return [f"http://{n['NodeManagerAddress']}:{INGRESS_PORT}" for n in nodes]


def _actors_per_node(name_fragment: str) -> Dict[str, int]:
    # The state API serves at most 10,000 rows by default.
    actors = list_actors(filters=[("state", "=", "ALIVE")], limit=10_000)
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
    """Samples the node's CPU and memory use (percent)."""

    def __init__(self, interval_s: float):
        self._interval_s = interval_s
        self._samples: List[Tuple[float, float, float]] = []
        self._running = True

    async def run(self) -> None:
        import psutil

        psutil.cpu_percent()
        while self._running:
            await asyncio.sleep(self._interval_s)
            self._samples.append(
                (time.time(), psutil.cpu_percent(), psutil.virtual_memory().percent)
            )

    async def stop(self) -> List[Tuple[float, float, float]]:
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


def stop_cpu_samplers(
    samplers: dict, window: Tuple[float, float]
) -> Dict[str, Dict[str, float]]:
    """Returns the max and p95 CPU percent and the max memory percent per node,
    over the samples inside ``window`` (the profiling phase, in epoch seconds),
    so AgentX's dataset build and warmup don't count."""
    samples = ray.get([s.stop.remote() for s in samplers.values()])
    for sampler in samplers.values():
        ray.kill(sampler)
    result = {}
    for node_id, node_samples in zip(samplers, samples):
        inside = [x for x in node_samples if window[0] <= x[0] <= window[1]]
        cpu = [c for _, c, _ in inside]
        mem = [m for _, _, m in inside]
        result[node_id] = {
            "max": float(np.max(cpu)) if cpu else 0.0,
            "p95": float(np.percentile(cpu, 95)) if cpu else 0.0,
            "mem_max": float(np.max(mem)) if mem else 0.0,
        }
    return result
