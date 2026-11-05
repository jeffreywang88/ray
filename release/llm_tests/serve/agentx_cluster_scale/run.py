"""Ray Serve LLM cluster-scale benchmark with Mocker engines and AgentX.

Calibrates one Mocker replica to an InferenceX datapoint, then measures one
full node and clusters of each --nodes size. See README.md. Defaults:
DeepSeek-V4.1-Flash FP4 on B200, TP4 with MTP, concurrency 32 (InferenceX run
40269923234).
"""

import argparse
import logging
import os
import sys

import ray

SERVE_ENV = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    "RAY_SERVE_HAPROXY_MAXCONN": "65536",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED": "1",
}


def router_env(args: argparse.Namespace) -> dict:
    """HAProxy limits that let the router tokenize every prompt, up to the
    model's full context, as in production.

    - Body buffer: HAProxy forwards at most this much of a body to the router.
      A truncated body is routed without tokens, so without KV overlap.
    - Timeout: HAProxy's wait for the body and for each router call. Past it,
      HAProxy retries another router, then falls back to its own pick.
    """
    return {
        "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE": str(
            args.router_body_buffer_mib * 1024 * 1024
        ),
        "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_TIMEOUT_S": str(
            args.router_timeout_s
        ),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument_group("InferenceX datapoint")
    a.add_argument("--gpus-per-replica", type=int, default=4)
    a.add_argument("--concurrency-per-engine", type=int, default=32)
    a.add_argument("--target-tok-s-per-gpu", type=float, default=370.3)
    a.add_argument("--target-p90-interactivity", type=float, default=234.3)
    a.add_argument(
        "--target-mean-ttft-ms",
        type=float,
        default=166,
        help="Cross-check only: reported against the calibrated replica.",
    )
    a.add_argument(
        "--target-p90-ttft-ms",
        type=float,
        default=267,
        help="Calibration target for --speedup (prefill), with p90 ITL for the "
        "decode rate.",
    )
    a.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    a.add_argument("--tokenizer", default=None, help="Defaults to --model.")
    a.add_argument(
        "--kv-cache-tokens",
        type=int,
        default=41_870_178,
        help="KV cache tokens of the whole replica (vLLM logs it per DP rank).",
    )
    a.add_argument("--max-num-seqs", type=int, default=128)
    a.add_argument("--max-num-batched-tokens", type=int, default=8192)
    a.add_argument("--max-model-len", type=int, default=1_048_576)
    a.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="vLLM's KV block size for the model: its KV events and the routers' "
        "index use it. DeepSeek-V4.1 on B200 uses 128.",
    )

    c = p.add_argument_group("cluster")
    c.add_argument(
        "--nodes",
        type=lambda v: sorted({int(n) for n in v.split(",")}),
        default=[128],
        help="Comma-separated cluster sizes; the cluster phase runs once per size.",
    )
    c.add_argument("--chips-per-node", type=int, default=8)
    c.add_argument("--node-cpus", type=float, default=16)
    c.add_argument("--routers-per-node", type=int, default=2)
    c.add_argument(
        "--router-body-buffer-mib",
        type=int,
        default=16,
        help="HAProxy's routing body buffer. A 1M-token AgentX prompt is about "
        "4.2 MB of JSON (3.97 bytes per token, measured with the default "
        "tokenizer); 16 MiB leaves about 4x headroom.",
    )
    c.add_argument(
        "--router-timeout-s",
        type=int,
        default=30,
        help="HAProxy's limit on each router call. The router tokenizes a 1M-token "
        "prompt in about 4 s on an idle core (Serve's default limit is 5 s).",
    )
    c.add_argument(
        "--client-nodes",
        type=int,
        default=0,
        help="AgentX load-generator nodes, one shard each; 0 runs AgentX on the head.",
    )
    c.add_argument(
        "--exclude-node-ids",
        type=lambda v: [x for x in v.split(",") if x],
        default=[],
        help="Nodes to use for neither engines nor clients (e.g. ones running a "
        "different Ray build).",
    )
    c.add_argument(
        "--keep-nodes",
        action="store_true",
        help="Leave the autoscaler request in place at exit (a driver running "
        "several invocations back to back manages capacity). By default the run "
        "releases its nodes; idle ones go away at once.",
    )
    c.add_argument(
        "--client-node-ids",
        type=lambda v: [x for x in v.split(",") if x],
        default=[],
        help="Use these nodes as the load generators, in this order (shard i on node "
        "i mod K, with seed --seed + i mod K), instead of picking them; keeps AgentX's "
        "per-node dataset cache warm across runs. Needs --client-resource CPU.",
    )
    c.add_argument(
        "--reserve-nodes",
        type=int,
        default=0,
        help="Engine nodes to keep requested from the autoscaler for the whole run "
        "(at least the largest phase's), so later phases don't wait for nodes.",
    )
    c.add_argument(
        "--client-resource",
        default="agentx_client",
        help="Resource that marks client nodes. CPU uses plain --node-cpus workers "
        "(the same node type as the engines), kept apart from the engine nodes.",
    )

    m = p.add_argument_group("mocker")
    m.add_argument(
        "--speedup",
        type=float,
        default=None,
        help="Prefill (and decode) speed. Alone, it is fixed and only the decode "
        "rate is calibrated, on p90 ITL.",
    )
    m.add_argument(
        "--decode-speedup",
        type=float,
        default=None,
        help="Decode speed on top of --speedup. With --speedup, skips calibration.",
    )
    m.add_argument("--stream-interval-s", type=float, default=0.0)
    m.add_argument(
        "--calibration-start",
        type=lambda v: tuple(float(x) for x in v.split(",")),
        default=(1.0, 1.0),
        help="speedup,decode-speedup of the first calibration run.",
    )
    m.add_argument(
        "--calibration-phase",
        choices=["replica", "node"],
        default="replica",
        help="Phase the speedups are calibrated on: one replica, or one full node "
        "(its replicas behind the KV-aware routers).",
    )
    m.add_argument(
        "--calibration-ttft",
        choices=["client", "engine"],
        default="client",
        help="TTFT matched to --target-p90-ttft-ms: AgentX's, or the engine's own "
        "(arrival at the engine to first token), which leaves out the router's "
        "tokenization when that alone exceeds the target.",
    )
    m.add_argument(
        "--calibration-history",
        default=None,
        help="A calibration.jsonl from an earlier run with the same targets and "
        "--calibration-ttft: its points seed the search, which continues after them "
        "and counts them toward --calibration-max-iters.",
    )
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
    r.add_argument(
        "--session-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="AgentX sends X-Session-ID (its stable per-conversation id), so the "
        "KV-aware router pins each conversation's turns to one engine.",
    )
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
    if args.decode_speedup is not None and args.speedup is None:
        p.error(
            "--decode-speedup needs --speedup (alone, --speedup fixes prefill "
            "and calibrates the decode rate)"
        )
    if args.calibration_phase not in args.phases and args.decode_speedup is None:
        p.error(
            f"--phases must include {args.calibration_phase} to calibrate the speedups"
        )
    if len(args.calibration_start) != 2 or min(args.calibration_start) <= 0:
        p.error(
            "--calibration-start takes two positive numbers: speedup,decode-speedup"
        )
    if args.client_node_ids:
        if args.client_resource != "CPU":
            p.error("--client-node-ids needs --client-resource CPU")
        if args.client_nodes not in (0, len(args.client_node_ids)):
            p.error("--client-nodes must match the number of --client-node-ids")
        args.client_nodes = len(args.client_node_ids)
    unknown = set(args.phases) - {"replica", "node", "cluster"}
    if unknown:
        p.error(f"unknown phases {sorted(unknown)}")
    return args


def main() -> int:
    args = parse_args()
    env_vars = {
        **SERVE_ENV,
        **router_env(args),
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
