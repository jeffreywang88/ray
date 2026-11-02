"""Derive a token-scaled copy of a Weka trace corpus (AgentX `weka_trace` format).

Each trace hash id stands for one `block_size`-token prompt block. This rewrites
every trace to `block_size = k` and scales the declared prompt-side token counts
by `k / block_size`, so prompts shrink by that factor while every hash id, every
prefix-sharing relationship, every timestamp and every think time is unchanged.
Output lengths are kept unless `--output-scale` is given.

Usage: python scale_traces.py IN.jsonl OUT.jsonl --tokens-per-block 1
       python scale_traces.py --split IN.jsonl OUT_DIR   (one NNNN_<id12>.json per trace)
"""

import argparse
import json
import math


def scale_request(req, factor, output_scale):
    if "in" in req:
        req["in"] = max(1, math.ceil(req["in"] * factor))
    if "out" in req and output_scale != 1.0:
        req["out"] = max(1, round(req["out"] * output_scale))
    for key in ("tool_tokens", "system_tokens"):
        if key in req:
            req[key] = math.ceil(req[key] * factor)
    for nested in req.get("requests", []) or []:
        scale_request(nested, factor, output_scale)


def split(src, dst):
    import os

    os.makedirs(dst, exist_ok=True)
    with open(src) as fin:
        for i, line in enumerate(fin):
            line = line.rstrip("\n")
            name = f"{i:04d}_{json.loads(line)['id'][:12]}.json"
            with open(os.path.join(dst, name), "w") as f:
                f.write(line + "\n")


def main():
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "--split":
        return split(sys.argv[2], sys.argv[3])
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--tokens-per-block", type=int, required=True)
    p.add_argument("--output-scale", type=float, default=1.0)
    args = p.parse_args()
    n = 0
    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            trace = json.loads(line)
            factor = args.tokens_per_block / trace["block_size"]
            trace["block_size"] = args.tokens_per_block
            for key in ("tool_tokens", "system_tokens"):
                if key in trace:
                    trace[key] = math.ceil(trace[key] * factor)
            for req in trace["requests"]:
                scale_request(req, factor, args.output_scale)
            trace.pop("totals", None)
            fout.write(json.dumps(trace, separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote {n} traces to {args.dst}")


if __name__ == "__main__":
    main()
