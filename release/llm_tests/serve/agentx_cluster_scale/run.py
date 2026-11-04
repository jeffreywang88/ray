"""Ray Serve LLM cluster-scale benchmark with Mocker engines and AgentX.

Calibrates one Mocker replica to an InferenceX datapoint, then measures one
full node and the whole cluster. See README.md. Defaults: DeepSeek-V4.1-Flash
FP4 on B200, DEP4 (4 GPUs per replica), concurrency 128.
"""

import argparse
import logging
import os
import sys

import ray

# HAProxy's routing body limit stays at Serve's default (8 MiB). The traces'
# prompts reach 990k tokens (median 142k), several MB of JSON. A body over the
# limit reaches the router truncated, so it is routed without KV overlap; the
# benchmark counts truncations and fails the run if there are any.
SERVE_ENV = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    "RAY_SERVE_HAPROXY_MAXCONN": "65536",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED": "1",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument_group("InferenceX datapoint")
    a.add_argument("--gpus-per-replica", type=int, default=4)
    a.add_argument("--concurrency-per-engine", type=int, default=128)
    a.add_argument("--target-tok-s-per-gpu", type=float, default=1406.97)
    a.add_argument("--target-p90-interactivity", type=float, default=107.5)
    a.add_argument(
        "--target-mean-ttft-ms",
        type=float,
        default=1048,
        help="Cross-check only: reported against the calibrated replica.",
    )
    a.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    a.add_argument("--tokenizer", default=None, help="Defaults to --model.")
    a.add_argument(
        "--kv-cache-tokens",
        type=int,
        default=4 * 41_178_655,
        help="KV cache tokens of the whole replica (vLLM logs it per DP rank).",
    )
    a.add_argument("--max-num-seqs", type=int, default=512)
    a.add_argument("--max-num-batched-tokens", type=int, default=8192)
    a.add_argument("--max-model-len", type=int, default=1_048_576)
    a.add_argument("--block-size", type=int, default=16)

    c = p.add_argument_group("cluster")
    c.add_argument("--nodes", type=int, default=128)
    c.add_argument("--chips-per-node", type=int, default=8)
    c.add_argument("--node-cpus", type=float, default=16)
    c.add_argument("--routers-per-node", type=int, default=2)
    c.add_argument(
        "--client-nodes",
        type=int,
        default=0,
        help="Nodes with the agentx_client resource; 0 runs AgentX on the head.",
    )

    m = p.add_argument_group("mocker")
    m.add_argument("--speedup", type=float, default=None)
    m.add_argument("--decode-speedup", type=float, default=None)
    m.add_argument("--stream-interval-s", type=float, default=0.0)
    m.add_argument("--calibration-tolerance", type=float, default=0.03)
    m.add_argument("--calibration-max-iters", type=int, default=8)
    m.add_argument("--calibration-duration-s", type=float, default=None)

    r = p.add_argument_group("run")
    r.add_argument(
        "--phases",
        type=lambda s: s.split(","),
        default=["replica", "node", "cluster"],
    )
    r.add_argument("--duration-s", type=float, default=900)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--max-cpu-pct", type=float, default=85)
    r.add_argument("--agentx-bin", default="agentx")
    r.add_argument("--deploy-timeout-s", type=float, default=1800)
    r.add_argument("--node-timeout-s", type=float, default=3600)
    r.add_argument("--output-dir", default="/tmp/agentx_cluster_scale")
    r.add_argument(
        "--test-output-json",
        default=os.environ.get("TEST_OUTPUT_JSON", "/tmp/release_test_out.json"),
    )
    args = p.parse_args(argv)
    if args.chips_per_node % args.gpus_per_replica:
        p.error("--chips-per-node must be a multiple of --gpus-per-replica")
    if "replica" not in args.phases:
        p.error("--phases must include replica: it is the scaling reference")
    if (args.speedup is None) != (args.decode_speedup is None):
        p.error("set both --speedup and --decode-speedup, or neither to calibrate")
    return args


def main() -> int:
    args = parse_args()
    env_vars = {
        **SERVE_ENV,
        "RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE": str(args.routers_per_node),
    }
    # Serve reads these at import time, so set them before importing it.
    os.environ.update(env_vars)
    import benchmark

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not ray.is_initialized():
        ray.init(address="auto")
    return benchmark.main(args, env_vars)


if __name__ == "__main__":
    sys.exit(main())
