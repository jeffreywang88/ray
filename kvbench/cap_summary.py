"""Summarize router-capacity runs (plan7): one row per run directory.

usage: python cap_summary.py [--from S] [--to S] [--json OUT] RUN_DIR [RUN_DIR ...]

Window is seconds from AgentX's profiling start. Columns:
- req/s: AgentX completed requests per second; per-engine = req/s / N.
- fallback %: engine arrivals without a router token key (HAProxy gave up on the router after a
  timeout or error and sent the request to its default server).
- tok miss %: routed requests whose prompt tokens were not in the engine's token store when the
  request arrived (the engine tokenizes locally; routing is unaffected).
- loop busy / ms per route: per-router asyncio-loop thread CPU (median over 10 s samples,
  all routers).
- rest util: CPU of everything except the routers / cores left for it. Above ~85 % the run
  is host-bound, not router-bound, and does not count.
"""

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
from window import load_slices, profile_start_from_log, window  # noqa: E402


def _med(values):
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else None


def _meta(run_dir):
    # meta.json embeds app args unescaped ({"KV_REPLICA_SYNC":true}), so read fields by regex.
    text = open(os.path.join(run_dir, "meta.json")).read()

    def num(name):
        m = re.search(rf'"{name}": "?(\d+)', text)
        return int(m.group(1)) if m else 0

    mode = "mesh" if "KV_REPLICA_SYNC" in text else "rpc"
    return num("R"), num("N"), num("sessions_per_worker"), num("router_cores"), mode


def summarize(run_dir, t_from, t_to):
    R, N, spw, k, mode = _meta(run_dir)
    w = window(run_dir, t_from, t_to)
    try:
        t0 = load_slices(run_dir)[0]["start"]
    except Exception:
        t0 = profile_start_from_log(run_dir)
    rows = [r for r in analyze.intervals(analyze.load(os.path.join(run_dir, "monitor.jsonl")))
            if t_from <= r["t"] - t0 <= (t_to if t_to is not None else 1e9)]
    per_router = [x for r in rows for x in r["routers"].values()]
    comps = {}
    for r in rows:
        for c, v in r["cpu_cores"].items():
            comps.setdefault(c, []).append(v)
    cpu = {c: _med(v) for c, v in comps.items()}
    rest = sum(v for c, v in cpu.items() if c != "router" and v is not None)
    rest_cores = 48 - R * k if k else 48
    arrivals = sum(r["requests_per_s"] * r["dt"] for r in rows)
    return {
        "run": os.path.relpath(run_dir, os.path.expanduser("~/results")),
        "mode": mode, "R": R, "N": N, "spw": spw,
        "req_s": w.get("req_s"),
        "req_s_engine": w["req_s"] / N if w.get("req_s") else None,
        "hit_ratio": w.get("hit_ratio"),
        "ttft_p50": w.get("ttft_p50"), "ttft_p99": w.get("ttft_p99"),
        "itl_p50": w.get("itl_p50"),
        "fallback_pct": 100 * w["token_key_missing"] / arrivals if arrivals else None,
        "tok_miss_pct": 100 * w["token_misses"] / arrivals if arrivals else None,
        "inflight_engine": w["inflight_median"] / N if w.get("inflight_median") else None,
        "loop_busy": _med([x["loop_cpu_frac"] for x in per_router]),
        "loop_ms_route": _med([x["loop_ms_per_route"] for x in per_router]),
        "router_proc_cores": _med([x["proc_cpu_cores"] for x in per_router]),
        "pick_p99_ms": _med([x["pick_p99_ms"] for x in per_router]),
        "cpu": cpu,
        "rest_util_pct": 100 * rest / rest_cores,
        "host_cpu_pct": w.get("host_cpu_pct_median"),
        "tokenize_p50_ms": _med([x["tokenize_p50_ms"] for x in per_router]),
        "engine_ttft_ms": _med([r.get("engine_ttft_ms") for r in rows]),
        "loop_lag_p50_ms": _med([x.get("loop_lag_p50_ms") for x in per_router]),
        "loop_lag_p99_ms": _med([x.get("loop_lag_p99_ms") for x in per_router]),
    }


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="t_from", type=float, default=60.0)
    p.add_argument("--to", dest="t_to", type=float, default=None)
    p.add_argument("--json")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    out = []
    print("| run | mode | R | N | spw | req/s | /engine | hit | TTFT p50/p99 ms | fallback % | tok miss % "
          "| in flight/engine | loop busy | loop ms/route | router cores | rest util % |")
    print("|---" * 16 + "|")
    for d in a.runs:
        try:
            s = summarize(d, a.t_from, a.t_to)
        except Exception as e:  # incomplete run: say so, keep going
            print(f"| {d} | error: {e} |")
            continue
        out.append(s)
        print(f"| {s['run']} | {s['mode']} | {s['R']} | {s['N']} | {s['spw']} | {fmt(s['req_s'])} "
              f"| {fmt(s['req_s_engine'], 2)} | {fmt(s['hit_ratio'], 3)} "
              f"| {fmt(s['ttft_p50'], 0)} / {fmt(s['ttft_p99'], 0)} | {fmt(s['fallback_pct'])} | {fmt(s['tok_miss_pct'])} "
              f"| {fmt(s['inflight_engine'])} | {fmt(s['loop_busy'], 2)} | {fmt(s['loop_ms_route'], 2)} "
              f"| {fmt(s['router_proc_cores'], 2)} | {fmt(s['rest_util_pct'], 0)} |")
        print("|   cpu cores: " + ", ".join(f"{c} {v:.1f}" for c, v in sorted(s["cpu"].items()) if v) + " |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
