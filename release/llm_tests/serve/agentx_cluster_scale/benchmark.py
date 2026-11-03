"""Benchmark phases: one replica, one full node, then the whole cluster."""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple

import agentx
import calibration
import cluster
import metrics
from mock_engine import ENGINE_DEPLOYMENT, build_app

from ray import serve
from ray.autoscaler.sdk import request_resources

logger = logging.getLogger(__name__)

APP_NAME = "llm"
PERF_METRICS = (
    ("output_tok_s", "THROUGHPUT"),
    ("output_tok_s_per_replica", "THROUGHPUT"),
    ("p90_interactivity", "THROUGHPUT"),
    ("req_s", "THROUGHPUT"),
    ("mean_ttft_ms", "LATENCY"),
    ("p90_ttft_ms", "LATENCY"),
    ("tok_s_scaling_efficiency", "THROUGHPUT"),
    ("p90_interactivity_ratio", "THROUGHPUT"),
)


@dataclass
class Phase:
    name: str
    nodes: int
    engines: int


def plan_phases(args: argparse.Namespace) -> List[Phase]:
    per_node = args.chips_per_node // args.gpus_per_replica
    phases = {
        "replica": Phase("replica", 1, 1),
        "node": Phase("node", 1, per_node),
        "cluster": Phase("cluster", args.nodes, args.nodes * per_node),
    }
    return [phases[name] for name in args.phases]


def _wait_running(timeout_s: float) -> None:
    start = time.monotonic()
    while True:
        app = serve.status().applications.get(APP_NAME)
        if app is not None and app.status == "RUNNING":
            return
        if app is not None and app.status == "DEPLOY_FAILED":
            raise RuntimeError(f"deploy failed: {app.message}")
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"app not RUNNING after {timeout_s}s")
        time.sleep(2)


def run_phase(
    args: argparse.Namespace,
    env_vars: Dict[str, str],
    phase: Phase,
    speedups: Tuple[float, float],
    duration_s: float,
) -> Dict[str, Any]:
    engine_nodes, client_nodes = cluster.scale(
        phase.nodes, args.node_cpus, args.client_nodes, args.node_timeout_s
    )
    node_ids = [n["NodeID"] for n in engine_nodes]
    client_ids = [n["NodeID"] for n in client_nodes] or [cluster.head_node_id()]
    app = build_app(
        model_id=args.model,
        tokenizer=args.tokenizer or args.model,
        num_engines=phase.engines,
        engine_cpus=args.node_cpus * args.gpus_per_replica / args.chips_per_node,
        node_ids=node_ids,
        mocker_engine={
            "block_size": args.block_size,
            "num_gpu_blocks": args.kv_cache_tokens // args.block_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "speedup_ratio": speedups[0],
            "decode_speedup_ratio": speedups[1],
        },
        stream_interval_s=args.stream_interval_s,
        max_model_len=args.max_model_len,
        env_vars=env_vars,
    )
    load_config = agentx.LoadConfig(
        agentx_bin=args.agentx_bin,
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        duration_s=duration_s,
        seed=args.seed,
    )
    logger.info(
        "%s: %d engines on %d nodes, speedups %s",
        phase.name,
        phase.engines,
        phase.nodes,
        speedups,
    )
    serve.start(
        proxy_location="EveryNode",
        controller_options={"runtime_env": {"env_vars": env_vars}},
    )
    try:
        serve.run(app, name=APP_NAME, route_prefix="/", blocking=False)
        _wait_running(args.deploy_timeout_s)
        routers = cluster.wait_for_placement(
            node_ids, phase.engines, ENGINE_DEPLOYMENT, args.routers_per_node
        )
        samplers = cluster.start_cpu_samplers(node_ids + client_ids)
        load = agentx.run_load(
            load_config,
            cluster.ingress_urls(engine_nodes),
            args.concurrency_per_engine * phase.engines,
            client_ids,
        )
        cpu = cluster.stop_cpu_samplers(samplers)
    finally:
        serve.shutdown()
    result = metrics.summarize(load, phase.engines, args.gpus_per_replica)
    result.update(
        nodes=phase.nodes,
        engines=phase.engines,
        routers=routers,
        agentx_shards=load["shards"],
        speedup=speedups[0],
        decode_speedup=speedups[1],
        engine_node_cpu_max=max(cpu[n]["max"] for n in node_ids),
        client_node_cpu_max=max(cpu[n]["max"] for n in client_ids),
    )
    logger.info("%s: %s", phase.name, json.dumps(result))
    return result


def calibrate(
    args: argparse.Namespace, run: Callable[[Tuple[float, float]], Dict[str, Any]]
) -> Tuple[Tuple[float, float], Dict[str, Any], List[dict], bool]:
    target = calibration.Target(
        output_tok_s=args.target_tok_s_per_gpu * args.gpus_per_replica,
        p90_itl_ms=1000 / args.target_p90_interactivity,
    )
    speedups, points = (1.0, 1.0), []
    for _ in range(args.calibration_max_iters):
        result = run(speedups)
        points.append(
            calibration.Point(
                *speedups, result["output_tok_s_per_replica"], result["p90_itl_ms"]
            )
        )
        done = calibration.converged(points[-1], target, args.calibration_tolerance)
        if done or len(points) == args.calibration_max_iters:
            return speedups, result, [asdict(p) for p in points], done
        speedups = calibration.next_speedups(points, target)


def _issues(args, phases: Dict[str, Dict[str, Any]], converged: bool) -> List[str]:
    issues = [] if converged else ["calibration did not converge"]
    for name, m in phases.items():
        if m["failed_requests"]:
            issues.append(f"{name}: {m['failed_requests']} failed requests")
        for role in ("engine", "client"):
            cpu = m[f"{role}_node_cpu_max"]
            if cpu > args.max_cpu_pct:
                issues.append(f"{name}: {role} node CPU peaked at {cpu:.0f}%")
    return issues


def _perf_metrics(phases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "perf_metric_name": f"{name}_{key}",
            "perf_metric_value": round(float(m[key]), 4),
            "perf_metric_type": metric_type,
        }
        for name, m in phases.items()
        for key, metric_type in PERF_METRICS
        if key in m
    ]


def main(args: argparse.Namespace, env_vars: Dict[str, str]) -> int:
    def run(name: str, speedups: Tuple[float, float], duration_s: float):
        phase = next(p for p in plan_phases(args) if p.name == name)
        return run_phase(args, env_vars, phase, speedups, duration_s)

    serve.shutdown()
    phases: Dict[str, Dict[str, Any]] = {}
    history, converged = [], True
    try:
        if args.speedup is not None and args.decode_speedup is not None:
            speedups = (args.speedup, args.decode_speedup)
            phases["replica"] = run("replica", speedups, args.duration_s)
        else:
            cal_duration = args.calibration_duration_s or args.duration_s
            speedups, phases["replica"], history, converged = calibrate(
                args, lambda s: run("replica", s, cal_duration)
            )
            if cal_duration != args.duration_s:
                phases["replica"] = run("replica", speedups, args.duration_s)
        for phase in plan_phases(args):
            if phase.name != "replica":
                phases[phase.name] = run(phase.name, speedups, args.duration_s)
    finally:
        request_resources(bundles=[])

    replica = phases["replica"]
    for phase in plan_phases(args):
        m = phases[phase.name]
        m["theoretical_output_tok_s"] = (
            phase.engines * args.target_tok_s_per_gpu * args.gpus_per_replica
        )
        m["tok_s_scaling_efficiency"] = m["output_tok_s"] / (
            phase.engines * replica["output_tok_s"]
        )
        m["p90_interactivity_ratio"] = (
            m["p90_interactivity"] / replica["p90_interactivity"]
        )
    issues = _issues(args, phases, converged)
    result = {
        "perf_metrics": _perf_metrics(phases),
        "config": vars(args),
        "speedup": speedups[0],
        "decode_speedup": speedups[1],
        "calibration": history,
        "replica_mean_ttft_vs_target": replica["mean_ttft_ms"]
        / args.target_mean_ttft_ms
        - 1,
        "phases": phases,
        "issues": issues,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    for path in (os.path.join(args.output_dir, "result.json"), args.test_output_json):
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
    logger.info("result: %s", json.dumps({k: result[k] for k in ("phases", "issues")}))
    return 1 if issues else 0
