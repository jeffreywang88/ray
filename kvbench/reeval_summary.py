"""Summarize the re-evaluation runs (results/reeval): median and range over repeats.

usage: python reeval_summary.py [RESULTS_DIR] [--json OUT] [--from S]

Runs are named <label>_rep<k>. For each label: throughput, TTFT, hit ratio, load-view mismatch,
router and HAProxy CPU per request, replication lag, token-store misses, and whether the run kept
up (TTFT p99 <= 1 s, no fallbacks, non-router cores <= 85 % busy). Prints the scaling-curve table
(curve_*), the commit-chain table (chain_* plus master / rpc at N=128) and any experiment rows.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
from cap_summary import summarize  # noqa: E402
from window import load_slices, profile_start_from_log  # noqa: E402

METRICS = ("req_s", "ttft_p50", "ttft_p99", "hit_ratio", "mismatch_pct", "router_ms_req",
           "haproxy_ms_req", "loop_ms_route", "repl_lag_ms", "tok_miss_pct", "fallback_pct",
           "rest_util_pct", "engine_ttft_ms", "host_cpu_pct",
           "router_proc_cores")


def _med(v):
    v = [x for x in v if x is not None and x == x]
    return statistics.median(v) if v else None


def run_metrics(d, t_from):
    s = summarize(d, t_from, None)
    try:
        t0 = load_slices(d)[0]["start"]
    except Exception:
        t0 = profile_start_from_log(d)
    rows = [r for r in analyze.intervals(analyze.load(os.path.join(d, "monitor.jsonl")))
            if r["t"] - t0 >= t_from]
    mism = [100 * m["abs_diff"] / m["actual"] for r in rows for m in r["mismatch"].values()
            if m.get("actual")]
    lag = [x["lag_p50"] for r in rows for x in r["routers"].values() if x.get("lag_p50") is not None]
    req = s["req_s"] or None
    s["mismatch_pct"] = _med(mism)
    s["repl_lag_ms"] = 1000 * _med(lag) if lag else None
    s["router_ms_req"] = 1000 * s["cpu"].get("router", 0) / req if req else None
    s["haproxy_ms_req"] = 1000 * s["cpu"].get("haproxy", 0) / req if req else None
    s["keeps_up"] = bool(s["ttft_p99"] is not None and s["ttft_p99"] <= 1000
                         and (s["fallback_pct"] or 0) == 0 and s["rest_util_pct"] <= 85)
    return s


def aggregate(root, t_from, prefix=""):
    by_label = {}
    for d in sorted(glob.glob(os.path.join(root, f"{prefix}*_rep*"))):
        m = re.match(r"(.+)_rep(\d+)$", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "monitor.jsonl")):
            continue
        try:
            s = run_metrics(d, t_from)
        except Exception as e:
            print(f"skip {d}: {e}", file=sys.stderr)
            continue
        if not s.get("req_s"):
            continue
        by_label.setdefault(m.group(1), []).append(s)
    out = {}
    for label, runs in by_label.items():
        agg = {"n": len(runs), "keeps_up": sum(r["keeps_up"] for r in runs), "R": runs[0]["R"],
               "N": runs[0]["N"], "mode": runs[0]["mode"]}
        for k in METRICS:
            vals = [r.get(k) for r in runs if r.get(k) is not None]
            agg[k] = {"med": _med(vals), "min": min(vals) if vals else None,
                      "max": max(vals) if vals else None}
        out[label] = agg
    return out


def cell(a, k, nd=0):
    v = a[k]
    if v["med"] is None:
        return "-"
    if a["n"] == 1 or v["min"] == v["max"]:
        return f"{v['med']:.{nd}f}"
    return f"{v['med']:.{nd}f} ({v['min']:.{nd}f}-{v['max']:.{nd}f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=os.path.expanduser("~/results/reeval"))
    p.add_argument("--from", dest="t_from", type=float, default=60.0)
    p.add_argument("--json")
    a = p.parse_args()
    agg = aggregate(a.root, a.t_from)
    if a.json:
        json.dump(agg, open(a.json, "w"), indent=1)

    print("## Scaling curves (R=2, 16 sessions/engine): median (min-max) over repeats\n")
    print("| variant | N | runs kept up | req/s | per engine | x linear | TTFT p50 / p99 ms | hit | mismatch % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for var in ("master", "rpc", "mesh"):
        base = agg.get(f"curve_{var}_N32")
        for N in (32, 64, 128, 192, 256):
            x = agg.get(f"curve_{var}_N{N}")
            if not x:
                continue
            lin = (x["req_s"]["med"] / (base["req_s"]["med"] * N / 32)) if base else None
            print(f"| {var} | {N} | {x['keeps_up']}/{x['n']} | {cell(x, 'req_s')} | {x['req_s']['med'] / N:.2f} "
                  f"| {lin:.2f} | {cell(x, 'ttft_p50')} / {cell(x, 'ttft_p99')} | {cell(x, 'hit_ratio', 3)} "
                  f"| {cell(x, 'mismatch_pct', 1)} |")

    print("\n## Commit chain at R=2, N=128 (16 sessions/engine)\n")
    print("| step | runs kept up | req/s | TTFT p50 / p99 ms | hit | mismatch % | router ms/req | HAProxy ms/req "
          "| loop ms/route | repl lag ms | tok miss % |")
    print("|---|" + "---|" * 10)
    order = [("master", "curve_master_N128")] + [(f"{i:02d}", f"chain_{i:02d}") for i in range(1, 7)] + [
        ("07 mesh", "chain_07mesh"), ("08 mesh", "chain_08mesh"), ("09 mesh", "curve_mesh_N128"),
        ("09 rpc", "curve_rpc_N128")]
    for name, label in order:
        x = agg.get(label)
        if not x:
            continue
        print(f"| {name} | {x['keeps_up']}/{x['n']} | {cell(x, 'req_s')} | {cell(x, 'ttft_p50')} / {cell(x, 'ttft_p99')} "
              f"| {cell(x, 'hit_ratio', 3)} | {cell(x, 'mismatch_pct', 1)} | {cell(x, 'router_ms_req', 1)} "
              f"| {cell(x, 'haproxy_ms_req', 2)} | {cell(x, 'loop_ms_route', 2)} | {cell(x, 'repl_lag_ms', 0)} "
              f"| {cell(x, 'tok_miss_pct', 1)} |")

    extra = sorted(k for k in agg if not k.startswith(("curve_", "chain_")))
    if extra:
        print("\n## Experiments\n")
        print("| label | runs kept up | req/s | TTFT p50 / p99 ms | hit | router ms/req |")
        print("|---|---|---|---|---|---|")
        for k in extra:
            x = agg[k]
            print(f"| {k} | {x['keeps_up']}/{x['n']} | {cell(x, 'req_s')} | {cell(x, 'ttft_p50')} / {cell(x, 'ttft_p99')} "
                  f"| {cell(x, 'hit_ratio', 3)} | {cell(x, 'router_ms_req', 1)} |")


if __name__ == "__main__":
    main()
