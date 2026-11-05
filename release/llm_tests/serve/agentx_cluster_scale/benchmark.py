"""Benchmark phases: one replica, one full node, then the whole cluster."""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import agentx
import calibration
import cluster
import metrics
import routing
from mock_engine import ENGINE_DEPLOYMENT, build_app

import ray
from ray import serve
from ray.autoscaler.sdk import request_resources
from ray.util.state import list_actors

logger = logging.getLogger(__name__)

APP_NAME = "llm"
# Above this, HAProxy is bypassing the router for a meaningful share of requests.
MAX_FALLBACKS_PER_S = 0.1
# Shards start profiling when their own warmup ends, so the headline metrics
# cover only the window when all of them profile; shorter than this, raise
# --duration-s.
MIN_CONCURRENT_WINDOW_S = 120
# Sessions in the pre-load session-pinning probe (4 per engine, up to this).
PROBE_MAX_SESSIONS = 64
PERF_METRICS = (
    ("output_tok_s", "THROUGHPUT"),
    ("output_tok_s_per_replica", "THROUGHPUT"),
    ("p90_interactivity", "THROUGHPUT"),
    ("req_s", "THROUGHPUT"),
    ("mean_ttft_ms", "LATENCY"),
    ("p90_ttft_ms", "LATENCY"),
    ("engine_p90_ttft_ms", "LATENCY"),
    ("route_p50_ms", "LATENCY"),
    ("route_p99_ms", "LATENCY"),
    # Share of routed requests past Serve's top latency bucket (1 s), whose
    # time the histogram can't resolve.
    ("route_share_over_top_bucket", "LATENCY"),
    ("tok_s_scaling_efficiency", "THROUGHPUT"),
    ("p90_interactivity_ratio", "THROUGHPUT"),
    ("p90_ttft_ratio", "LATENCY"),
    # Prompt tokens the engines served from cache: how well KV-aware routing
    # keeps each session on the engine holding its prefix.
    ("kv_hit_ratio", "THROUGHPUT"),
)


@dataclass
class Phase:
    name: str
    nodes: int
    engines: int


def plan_phases(args: argparse.Namespace) -> List[Phase]:
    """replica, node, then one cluster phase per --nodes size, in that order."""
    per_node = args.chips_per_node // args.gpus_per_replica
    phases = []
    if "replica" in args.phases:
        phases.append(Phase("replica", 1, 1))
    if "node" in args.phases:
        phases.append(Phase("node", 1, per_node))
    if "cluster" in args.phases:
        phases += [Phase(f"cluster_n{n}", n, n * per_node) for n in args.nodes]
    return phases


def _shutdown_serve(timeout_s: float = 300) -> None:
    """Shuts Serve down and waits until its actors are gone.

    serve.shutdown() can time out and kill the controller asynchronously; a
    phase that starts Serve before that lands loses its new controller.
    """
    try:
        serve.shutdown()
    except ray.exceptions.RayActorError:
        # The controller was already being killed; wait for its actors below.
        logger.warning("Serve controller died during shutdown", exc_info=True)
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        alive = [
            a
            for a in list_actors(filters=[("state", "=", "ALIVE")], limit=10_000)
            if a.class_name.startswith(
                ("ServeController", "ServeReplica", "ProxyActor", "HAProxyManager")
            )
        ]
        if not alive:
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Serve actors still alive {timeout_s}s after shutdown")
    # Let the replicas' ports (ZMQ, HAProxy) be released.
    time.sleep(10)


def _wait_settled(timeout_s: float, settle_s: float = 30) -> None:
    """Waits until every replica (engines and routers) is RUNNING, then lets
    the proxies settle.

    Each replica that becomes ready reloads HAProxy, and a reloaded-away HAProxy
    process force-closes its connections after Serve's hard-stop-after (120 s).
    AgentX's warmup primers run longer than that, so load must start only after
    the last reload.
    """
    start = time.monotonic()
    while True:
        app = serve.status().applications.get(APP_NAME)
        deployments = app.deployments.values() if app is not None else []
        if deployments and all(
            d.status == "HEALTHY"
            and all(
                str(state).endswith("RUNNING")
                for state, n in d.replica_states.items()
                if n
            )
            for d in deployments
        ):
            break
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"replicas not all RUNNING after {timeout_s}s")
        time.sleep(2)
    time.sleep(settle_s)


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
        phase.nodes,
        args.node_cpus,
        args.client_nodes,
        args.node_timeout_s,
        client_resource=args.client_resource,
        reserve_engine_nodes=max(
            args.reserve_nodes, *(p.nodes for p in plan_phases(args))
        ),
        client_node_ids=args.client_node_ids,
        exclude_node_ids=args.exclude_node_ids,
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
        session_ids=args.session_ids,
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
        _wait_settled(args.deploy_timeout_s)
        probe = routing.session_probe(
            cluster.ingress_urls(engine_nodes),
            args.model,
            sessions=min(PROBE_MAX_SESSIONS, 4 * phase.engines),
        )
        logger.info("%s: session probe %s", phase.name, json.dumps(probe))
        samplers = cluster.start_cpu_samplers(node_ids + client_ids)
        scraper = routing.RoutingScraper().start()
        load = agentx.run_load(
            load_config,
            cluster.ingress_urls(engine_nodes),
            args.concurrency_per_engine * phase.engines,
            client_ids,
            client_cpus=args.node_cpus,
            client_memory_gb=cluster.node_memory_gb(client_ids),
        )
        scraper.stop()
        cpu = cluster.stop_cpu_samplers(samplers, metrics.profiling_window(load))
    finally:
        _shutdown_serve()
    if args.output_dir:
        # Per-request records, for analysis beyond the summary.
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{phase.name}_records.npz"),
            shard_sizes=np.asarray(load["shard_sizes"]),
            shard_windows_ns=np.asarray(
                [w or [-1, -1] for w in load["shard_windows_ns"]], dtype=np.int64
            ),
            **load["columns"],
        )
    result = metrics.summarize(load, phase.engines, args.gpus_per_replica)
    result.update(scraper.summary(*metrics.measurement_window(load)))
    result.update(
        nodes=phase.nodes,
        engines=phase.engines,
        routers=routers,
        session_ids=args.session_ids,
        **{f"session_probe_{k}": v for k, v in probe.items()},
        agentx_shards=load["shards"],
        failed_shards=len(load["shard_errors"]),
        shard_errors=load["shard_errors"][:3],
        speedup=speedups[0],
        decode_speedup=speedups[1],
        engine_node_cpu_max=max(cpu[n]["max"] for n in node_ids),
        engine_node_cpu_p95_max=max(cpu[n]["p95"] for n in node_ids),
        client_node_cpu_max=max(cpu[n]["max"] for n in client_ids),
        engine_node_mem_pct_max=max(cpu[n]["mem_max"] for n in node_ids),
        client_node_mem_pct_max=max(cpu[n]["mem_max"] for n in client_ids),
    )
    logger.info("%s: %s", phase.name, json.dumps(result))
    return result


def calibrate(
    args: argparse.Namespace, run: Callable[[Tuple[float, float]], Dict[str, Any]]
) -> Tuple[Tuple[float, float], Dict[str, Any], List[dict], bool]:
    target = calibration.Target(
        p90_ttft_ms=args.target_p90_ttft_ms,
        p90_itl_ms=1000 / args.target_p90_interactivity,
    )
    fixed_speedup = args.speedup is not None
    start_speedup, start_decode = args.calibration_start
    speedups = (args.speedup if fixed_speedup else start_speedup, start_decode)
    # The TTFT the speedup is calibrated on: the client's, or the engine's own.
    ttft_key = f"{'engine_' if args.calibration_ttft == 'engine' else ''}p90_ttft_ms"
    prior = (
        calibration.ENGINE_TTFT_JACOBIAN
        if args.calibration_ttft == "engine"
        else calibration.INITIAL_JACOBIAN
    )
    points: List[calibration.Point] = []
    if args.calibration_history:
        # Resume a search: earlier points (calibration.jsonl lines) seed it.
        with open(args.calibration_history) as f:
            for line in filter(str.strip, f):
                d = json.loads(line)
                points.append(
                    calibration.Point(
                        **{k: d[k] for k in calibration.Point.__dataclass_fields__}
                    )
                )
        if len(points) >= args.calibration_max_iters:
            raise ValueError("--calibration-history already has max-iters points")
        speedups = calibration.next_speedups(points, target, fixed_speedup, prior)
        logger.info("Resuming calibration after %d points at %s", len(points), speedups)
    for _ in range(args.calibration_max_iters - len(points)):
        result = run(speedups)
        if result.get(ttft_key) is None:
            raise RuntimeError(f"calibration run measured no {ttft_key}")
        points.append(
            calibration.Point(
                *speedups,
                result[ttft_key],
                result["p90_itl_ms"],
                result["output_tok_s_per_replica"],
            )
        )
        if args.output_dir:
            # One line per point, so a run that dies keeps its points.
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "calibration.jsonl"), "a") as f:
                f.write(json.dumps({**asdict(points[-1]), "phase": result}) + "\n")
        done = calibration.converged(
            points[-1], target, args.calibration_tolerance, fixed_speedup
        )
        if done or len(points) == args.calibration_max_iters:
            return speedups, result, [asdict(p) for p in points], done
        speedups = calibration.next_speedups(points, target, fixed_speedup, prior)


def _issues(args, phases: Dict[str, Dict[str, Any]], converged: bool) -> List[str]:
    issues = [] if converged else ["calibration did not converge"]
    for name, m in phases.items():
        if m["failed_requests"]:
            issues.append(f"{name}: {m['failed_requests']} failed requests")
        if m.get("failed_shards"):
            issues.append(
                f"{name}: {m['failed_shards']} of {m['agentx_shards']} AgentX shards "
                f"failed; first: {m['shard_errors'][0][:300]}"
            )
        if (m["fallbacks_per_s"] or 0) > MAX_FALLBACKS_PER_S:
            issues.append(f"{name}: {m['fallbacks_per_s']:.2f} router fallbacks/s")
        if m["agentx_shards"] > 1 and m["shard_overlap_s"] < MIN_CONCURRENT_WINDOW_S:
            issues.append(
                f"{name}: AgentX shards profiled together for only "
                f"{m['shard_overlap_s']:.0f}s (started {m['shard_start_skew_s']:.0f}s "
                "apart); raise --duration-s"
            )
        if (
            m["session_probe_pinned_failed_requests"]
            or m["session_probe_control_failed_requests"]
        ):
            issues.append(f"{name}: session probe requests failed")
        if args.session_ids:
            if (m["session_probe_pinned_one_engine_share"] or 0) < 1:
                issues.append(
                    f"{name}: only {m['session_probe_pinned_one_engine_share']} of "
                    "probe sessions kept every turn on one engine"
                )
            if m["requests_without_session"]:
                issues.append(
                    f"{name}: {m['requests_without_session']:.0f} engine requests "
                    "carried no session id"
                )
        if m["truncated_bodies"]:
            # The router routed these without tokens, so without KV overlap.
            issues.append(
                f"{name}: HAProxy truncated {m['truncated_bodies']:.0f} request "
                "bodies; raise RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"
            )
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
        if m.get(key) is not None
    ]


def main(args: argparse.Namespace, env_vars: Dict[str, str]) -> int:
    def run(name: str, speedups: Tuple[float, float], duration_s: float):
        phase = next(p for p in plan_phases(args) if p.name == name)
        return run_phase(args, env_vars, phase, speedups, duration_s)

    _shutdown_serve()
    phases: Dict[str, Dict[str, Any]] = {}
    history, converged = [], True
    cal = args.calibration_phase
    try:
        if args.decode_speedup is not None:
            speedups = (args.speedup, args.decode_speedup)
        else:
            cal_duration = args.calibration_duration_s or args.duration_s
            speedups, phases[cal], history, converged = calibrate(
                args, lambda s: run(cal, s, cal_duration)
            )
            if cal_duration != args.duration_s:
                phases[cal] = run(cal, speedups, args.duration_s)
        for phase in plan_phases(args):
            if phase.name not in phases:
                phases[phase.name] = run(phase.name, speedups, args.duration_s)
    finally:
        if not args.keep_nodes:
            request_resources(bundles=[])

    # Scaling reference: the replica phase, else the single node (per engine).
    reference_name = next((n for n in ("replica", "node") if n in phases), None)
    reference = phases.get(reference_name)
    for phase in plan_phases(args):
        m = phases[phase.name]
        m["theoretical_output_tok_s"] = (
            phase.engines * args.target_tok_s_per_gpu * args.gpus_per_replica
        )
        if reference is not None:
            m["tok_s_scaling_efficiency"] = m["output_tok_s"] / (
                phase.engines * reference["output_tok_s_per_replica"]
            )
            m["p90_interactivity_ratio"] = (
                m["p90_interactivity"] / reference["p90_interactivity"]
            )
            m["p90_ttft_ratio"] = m["p90_ttft_ms"] / reference["p90_ttft_ms"]
    issues = _issues(args, phases, converged)
    calibrated = phases.get(cal)
    result = {
        "perf_metrics": _perf_metrics(phases),
        "config": vars(args),
        "speedup": speedups[0],
        "decode_speedup": speedups[1],
        "calibration": history,
        # The calibration phase against the InferenceX datapoint (relative error).
        "calibration_vs_target": None
        if calibrated is None
        else {
            "phase": cal,
            "output_tok_s_per_replica": calibrated["output_tok_s_per_replica"]
            / (args.target_tok_s_per_gpu * args.gpus_per_replica)
            - 1,
            "p90_interactivity": calibrated["p90_interactivity"]
            / args.target_p90_interactivity
            - 1,
            "mean_ttft_ms": calibrated["mean_ttft_ms"] / args.target_mean_ttft_ms - 1,
            "p90_ttft_ms": calibrated["p90_ttft_ms"] / args.target_p90_ttft_ms - 1,
            "engine_p90_ttft_ms": None
            if calibrated["engine_p90_ttft_ms"] is None
            else calibrated["engine_p90_ttft_ms"] / args.target_p90_ttft_ms - 1,
        },
        "reference_phase": reference_name,
        "phases": phases,
        "issues": issues,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    for path in (os.path.join(args.output_dir, "result.json"), args.test_output_json):
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
    logger.info("result: %s", json.dumps({k: result[k] for k in ("phases", "issues")}))
    return 1 if issues else 0
