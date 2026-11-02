"""Windowed metrics for one run: AgentX 1 s timeslices + monitor.jsonl, aligned on
wall-clock time.

usage: python window.py RUN_DIR [--from S] [--to S] [--json]
S is seconds from the start of AgentX's profiling phase (first timeslice).

Latency percentiles over a window are taken from the request-weighted mixture of
the per-slice distributions (each slice's p1..p99 interpolated linearly), so they
are approximations; the full-window values from profile_export_aiperf.json are exact.
"""

import argparse
import bisect
import collections
import csv
import glob
import gzip
import json
import os
import statistics

QUANTILES = [(1, "p1"), (5, "p5"), (10, "p10"), (25, "p25"), (50, "p50"),
             (75, "p75"), (90, "p90"), (95, "p95"), (99, "p99")]


def load_slices(run_dir):
    path = glob.glob(os.path.join(run_dir, "agentx", "**", "profile_export_aiperf_timeslices.csv*"),
                     recursive=True)[0]
    slices = collections.defaultdict(lambda: {"metrics": collections.defaultdict(dict)})
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for row in csv.DictReader(f):
            s = slices[int(row["Timeslice"])]
            s["start"] = int(row["Start_NS"]) / 1e9
            s["end"] = int(row["End_NS"]) / 1e9
            try:
                s["metrics"][row["Metric"]][row["Stat"]] = float(row["Value"])
            except ValueError:
                pass
    return [slices[k] for k in sorted(slices)]


def mixture_quantiles(slices, metric, qs=(50, 99)):
    """Quantiles of the request-weighted mixture of per-slice distributions."""
    samples = []
    for s in slices:
        m = s["metrics"].get(metric)
        n = s["metrics"].get("Request Count", {}).get("avg", 0)
        if not m or not n or "p50" not in m:
            continue
        xs = [q for q, _ in QUANTILES if _ in m]
        ys = [m[k] for _, k in QUANTILES if k in m]
        # 99 evenly spaced points of this slice's quantile function, weight n each.
        for p in range(1, 100):
            i = bisect.bisect_left(xs, p)
            if i == 0:
                v = ys[0]
            elif i >= len(xs):
                v = ys[-1]
            else:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                v = y0 + (y1 - y0) * (p - x0) / (x1 - x0)
            samples.append((v, n))
    if not samples:
        return {q: None for q in qs}
    samples.sort()
    total = sum(w for _, w in samples)
    out, acc, j = {}, 0.0, 0
    targets = sorted(qs)
    for v, w in samples:
        acc += w
        while j < len(targets) and acc >= total * targets[j] / 100:
            out[targets[j]] = v
            j += 1
    for q in targets[j:]:
        out[q] = samples[-1][0]
    return out


def monitor_window(run_dir, t_from, t_to):
    recs = []
    with open(os.path.join(run_dir, "monitor.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if t_from <= r["time"] <= t_to:
                recs.append(r)
    if len(recs) < 2:
        return {}
    a, b = recs[0], recs[-1]
    dt = b["time"] - a["time"]

    def total(r, key):
        return sum(s.get(key, 0) for s in r["servers"])

    d_prompt = total(b, "prompt_tokens") - total(a, "prompt_tokens")
    d_cached = total(b, "cached_tokens") - total(a, "cached_tokens")
    router_cores = []
    for p, r in zip(recs, recs[1:]):
        step = r["time"] - p["time"]
        for x, y in zip(p["routers"], r["routers"]):
            if "process_cpu_s" in x and "process_cpu_s" in y:
                router_cores.append((y["process_cpu_s"] - x["process_cpu_s"]) / step)
    return {
        "hit_ratio": d_cached / d_prompt if d_prompt else None,
        "kv_removed_per_s": (total(b, "kv_removed_blocks") - total(a, "kv_removed_blocks")) / dt,
        "token_misses": total(b, "token_channel_misses") - total(a, "token_channel_misses"),
        "token_key_missing": total(b, "token_key_missing") - total(a, "token_key_missing"),
        "completed_per_s_engine": (total(b, "completed") - total(a, "completed")) / dt,
        "inflight_median": statistics.median(sum(s["inflight"] for s in r["servers"]) for r in recs),
        "host_cpu_pct_median": statistics.median(r.get("host_cpu_pct") or 0 for r in recs),
        "host_cpu_pct_max": max(r.get("host_cpu_pct") or 0 for r in recs),
        "router_cores_median": statistics.median(router_cores) if router_cores else None,
    }


def profile_start_from_log(run_dir):
    """Profiling-phase start (epoch s) from agentx.log, for runs AgentX aborted
    before exporting timeslices."""
    import datetime
    import re
    meta = open(os.path.join(run_dir, "meta.json")).read()
    day = re.search(r'"start": "(\d{4}-\d{2}-\d{2})', meta).group(1)
    for line in open(os.path.join(run_dir, "agentx.log"), errors="replace"):
        if "Phase profiling (profiling) started" in line:
            hms = line.split()[0]
            dt = datetime.datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S.%f")
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def window(run_dir, t_from=0.0, t_to=None):
    try:
        slices = load_slices(run_dir)
    except IndexError:
        slices = []
    if not slices:
        # AgentX aborted: monitor-only metrics, aligned on the profiling start in its log.
        t0 = profile_start_from_log(run_dir)
        if t0 is None:
            raise IndexError("no timeslices and no profiling start in agentx.log")
        lo, hi = t0 + t_from, t0 + (t_to if t_to is not None else 900)
        out = {"run": os.path.basename(os.path.normpath(run_dir)), "from_s": t_from,
               "to_s": t_to, "agentx_exported": False}
        out.update(monitor_window(run_dir, lo, hi))
        return out
    t0 = slices[0]["start"]
    lo = t0 + t_from
    hi = slices[-1]["end"] if t_to is None else t0 + t_to
    sel = [s for s in slices if s["start"] >= lo and s["end"] <= hi]
    n = sum(s["metrics"].get("Request Count", {}).get("avg", 0) for s in sel)
    span = (sel[-1]["end"] - sel[0]["start"]) if sel else 0
    ttft = mixture_quantiles(sel, "Time to First Token")
    itl = mixture_quantiles(sel, "Inter Token Latency")
    out = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "from_s": t_from,
        "to_s": round(hi - t0, 1),
        "requests": n,
        "req_s": n / span if span else None,
        "ttft_p50": ttft[50],
        "ttft_p99": ttft[99],
        "itl_p50": itl[50],
        "itl_p99": itl[99],
    }
    out.update(monitor_window(run_dir, lo, hi))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--from", dest="t_from", type=float, default=0.0)
    p.add_argument("--to", dest="t_to", type=float, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    w = window(args.run_dir, args.t_from, args.t_to)
    if args.json:
        print(json.dumps(w, default=str))
    else:
        print(" ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in w.items()))


if __name__ == "__main__":
    main()
