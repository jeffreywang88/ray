"""Deploy KV-aware direct streaming with N Mocker-backed replicas and R LLMRouters.

Mirrors ``build_openai_app``'s direct-streaming branch: the data-plane deployment
uses ``KVAwareRouter`` and the ingress request router is the real ``LLMRouter``
built by ``_build_openai_ingress_request_router``. Only the data-plane class
differs (``MockerServer`` instead of ``LLMServer``).

Usage: python app.py --num-replicas N --engine-cpus C [--speedup S] ...

Cluster version: restarts Serve with a proxy (HAProxy) on every node, each with
RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE LLMRouter replicas beside it; each mock
engine takes a whole worker node (--engine-cpus).
The Serve env vars in SERVE_ENV_KEYS are passed from this driver's environment
to the controller (and the proxy it starts) and to every replica.
"""

import argparse
import json
import os
import time

import ray
from ray import serve
from ray.llm._internal.serve.core.ingress.builder import (
    _build_openai_ingress_request_router,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_router import (
    KVAwareRouter,
)
from ray.serve.config import RequestRouterConfig
from ray.serve.llm import LLMConfig

from mock_server import MockerServer

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

SERVE_ENV_KEYS = (
    "RAY_SERVE_ENABLE_HA_PROXY",
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
    "RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE",
    "RAY_SERVE_HAPROXY_MAXCONN",
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED",
    "RAY_SERVE_EVENT_LOOP_MONITORING_INTERVAL_S",
)


def serve_env():
    return {k: os.environ[k] for k in SERVE_ENV_KEYS if k in os.environ}


def harness_env():
    """Serve env vars plus the per-node harness dir (install_variant.py writes it)."""
    node_dir = os.environ["KVBENCH_NODE_DIR"]
    return {**serve_env(), "PYTHONPATH": f"{node_dir}:${{PYTHONPATH}}"}


def build(args) -> "serve.Application":
    from dynamo._core import _normalize_mocker_config

    engine = {
        "block_size": args.block_size,
        "num_gpu_blocks": args.num_gpu_blocks,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "speedup_ratio": args.speedup,
    }
    if args.decode_speedup is not None:
        engine["decode_speedup_ratio"] = args.decode_speedup
    mocker = _normalize_mocker_config({"engine": engine})
    cfg = {
        "model_id": MODEL,
        "mocker": mocker,
        "kv_port_base": args.kv_port_base,
        "replay_port_base": args.kv_port_base + 1500,
        "token_port_base": args.kv_port_base + 3000,
        "flush_interval_s": args.flush_interval_s,
        "output_token_scale": args.output_token_scale,
        "no_kv_events_to_router": args.no_kv_events_to_router,
    }
    router_config = RequestRouterConfig(
        request_router_class=KVAwareRouter,
        request_routing_stats_period_s=args.routing_stats_period_s,
    )
    server = serve.deployment(
        serve.ingress()(MockerServer),
        name="LLMServer:mock",
        num_replicas=args.num_replicas,
        max_ongoing_requests=args.max_ongoing_requests,
        request_router_config=router_config,
        health_check_period_s=10,
        # Harness only: the mock replicas' per-request access logs are not part of
        # the system under test and cost CPU across 1024 processes.
        logging_config={"enable_access_log": False, "log_level": "WARNING"},
        ray_actor_options={
            "num_cpus": args.engine_cpus,
            "runtime_env": {
                "env_vars": {
                    **harness_env(),
                    "DYN_RUNTIME_NUM_WORKER_THREADS": "1",
                    "DYN_RUNTIME_MAX_BLOCKING_THREADS": "4",
                    "DYN_LOG": "warn",
                    "BENCH_BLOCK_IMPORTS": "pandas,pyarrow",
                    **dict(kv.split("=", 1) for kv in args.replica_env),
                }
            },
        },
    ).bind(cfg)
    llm_config = LLMConfig(
        model_loading_config={"model_id": MODEL, "model_source": MODEL},
        deployment_config={"request_router_config": router_config},
        # Only the router's pre-routing tokenizer reads these (the mock engines don't).
        engine_kwargs={"max_model_len": args.max_model_len, **json.loads(args.engine_kwargs)},
        experimental_configs=json.loads(args.experimental_configs),
        runtime_env={
            "env_vars": {
                **harness_env(),
                "BENCH_INSTRUMENT": "1",
                **dict(kv.split("=", 1) for kv in args.router_env),
            }
        },
    )
    router = _build_openai_ingress_request_router(server=server, llm_config=llm_config)
    return server._with_ingress_request_router(router)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-replicas", type=int, required=True)
    p.add_argument("--deploy-timeout-s", type=float, default=1200)
    p.add_argument("--engine-cpus", type=float, required=True,
                   help="CPUs per mock engine: a worker node's CPU count gives one engine per node")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-gpu-blocks", type=int, default=65536)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--speedup", type=float, default=1.0)
    p.add_argument("--decode-speedup", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--max-ongoing-requests", type=int, default=256)
    p.add_argument("--kv-port-base", type=int, default=23000)
    p.add_argument("--flush-interval-s", type=float, default=0.05)
    p.add_argument("--output-token-scale", type=float, default=1.0)
    p.add_argument("--routing-stats-period-s", type=float, default=5.0)
    p.add_argument("--experimental-configs", default="{}")
    p.add_argument("--engine-kwargs", default="{}", help='e.g. {"renderer_num_workers": 4}')
    p.add_argument("--no-kv-events-to-router", action="store_true",
                   help="engines advertise a dead KV-event address: routers ingest no KV events")
    p.add_argument(
        "--replica-env",
        action="append",
        default=[],
        help="KEY=VAL env var for the mock replicas only (repeatable).",
    )
    p.add_argument(
        "--router-env",
        action="append",
        default=[],
        help="KEY=VAL env var for the LLMRouter processes only (repeatable).",
    )
    args = p.parse_args()
    ray.init(address="auto")
    # Clean state per run: a fresh controller and HAProxy import the installed variant.
    serve.shutdown()
    serve.start(
        proxy_location="EveryNode",
        controller_options={"runtime_env": {"env_vars": serve_env()}},
    )
    t = time.time()
    serve.run(build(args), name="llm", route_prefix="/", blocking=False)
    while True:
        app = serve.status().applications.get("llm")
        if app is not None and app.status == "RUNNING":
            break
        if app is not None and app.status == "DEPLOY_FAILED":
            raise RuntimeError(app.message)
        if time.time() - t > args.deploy_timeout_s:
            raise RuntimeError(f"not RUNNING after {args.deploy_timeout_s}s: {app}")
        time.sleep(2)
    print(f"deployed N={args.num_replicas} in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
