"""Replication-ceiling campaign: per-run metrics from Ray's own Prometheus metrics plus
client-side TTFT, and the median [min-max] tables across runs.

Steady window: from 60 s after the first profiling-phase request (AgentX runs a
max_tokens=1 warmup phase first) to the end of the benchmark duration.
Histograms are differenced between the first scrape inside the window and the last one;
quantiles interpolate within buckets like PromQL histogram_quantile. Serve and HAProxy
flush counters periodically, so windowed counter rates lag: router CPU per request uses
the exact client request rate in the window, and calls per routed request uses the whole
run's totals (first scrape to the last, after the load drained), where they reconcile.

  run <run_dir> [--from-s 60]   -> per-run JSON on stdout (prom_summary.json)
  table <root>                  -> markdown tables over <variant>_N<n>_rep<k> runs
"""

import argparse
import glob
import gzip
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse(lines):
    """Prometheus text lines -> [(name, labels, value)]."""
    out = []
    for ln in lines:
        m = _LINE.match(ln)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(3) or ""))
        try:
            out.append((m.group(1), labels, float(m.group(4))))
        except ValueError:
            pass
    return out


def load_scrapes(run_dir):
    path = os.path.join(run_dir, "prom.jsonl.gz")
    scrapes = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a torn last line
            scrapes.append((rec["t"], parse(rec["lines"])))
    return scrapes


def total(samples, name, **match):
    return sum(
        v for n, lb, v in samples if n == name and all(lb.get(k) == x for k, x in match.items())
    )


def buckets(samples, name, **match):
    """Cumulative bucket counts by upper bound, summed over all other labels."""
    out = {}
    for n, lb, v in samples:
        if n != name + "_bucket" or not all(lb.get(k) == x for k, x in match.items()):
            continue
        le = float("inf") if lb["le"] in ("+Inf", "Inf") else float(lb["le"])
        out[le] = out.get(le, 0.0) + v
    return out


def delta_buckets(a, b, name, **match):
    ba, bb = buckets(a, name, **match), buckets(b, name, **match)
    return {le: bb[le] - ba.get(le, 0.0) for le in bb}


def quantile(q, cum):
    """histogram_quantile over cumulative {le: count}."""
    if not cum:
        return None
    les = sorted(cum)
    count = cum[les[-1]]
    if count <= 0:
        return None
    rank = q * count
    prev_le, prev_c = 0.0, 0.0
    for le in les:
        c = cum[le]
        if c >= rank:
            if le == float("inf"):
                return prev_le  # beyond the last finite bucket
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return les[-1]


def share_above(cum, bound):
    count = cum.get(float("inf"))
    if not count:
        return None
    return 1.0 - cum.get(float(bound), 0.0) / count


def load_records(run_dir):
    compact = os.path.join(run_dir, "agentx", "ttft.csv.gz")
    if os.path.exists(compact):  # compact_runs.py: start_s,phase,ttft_ms per request
        recs = []
        with gzip.open(compact, "rt") as f:
            next(f)
            for line in f:
                start, phase, ttft = line.rstrip("\n").split(",")
                if phase == "profiling":
                    recs.append((float(start), float(ttft) if ttft else None))
        return recs
    paths = glob.glob(os.path.join(run_dir, "agentx", "**", "profile_export.jsonl*"), recursive=True)
    if not paths:
        return []
    op = gzip.open if paths[0].endswith(".gz") else open
    recs = []
    with op(paths[0], "rt") as f:
        for line in f:
            rec = json.loads(line)
            md = rec.get("metadata", {})
            start = md.get("request_start_ns")
            if start is None or md.get("benchmark_phase", "profiling") != "profiling":
                continue
            ttft = rec.get("metrics", {}).get("time_to_first_token")
            ttft_ms = None
            if ttft:
                val = ttft["value"] if isinstance(ttft, dict) else ttft
                unit = ttft.get("unit", "ms") if isinstance(ttft, dict) else "ms"
                ttft_ms = val * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[unit]
            recs.append((start / 1e9, ttft_ms))
    return recs


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def load_meta(run_dir):
    """run_point.sh writes app_args unescaped, so meta.json is not always valid JSON."""
    text = open(os.path.join(run_dir, "meta.json")).read()
    try:
        return json.loads(text)
    except ValueError:
        grab = lambda key: re.search(rf'"{key}": "?([^",]+)', text).group(1)  # noqa: E731
        return {"N": int(grab("N")), "duration_s": float(grab("duration_s")),
                "ray_commit": grab("ray_commit")}


def run_summary(run_dir, from_s=60.0):
    meta = load_meta(run_dir)
    dur = float(meta["duration_s"])
    recs = load_records(run_dir)
    t0 = min(c0 for c0, _ in recs)
    w0, w1 = t0 + from_s, t0 + dur
    win = [ttft for c0, ttft in recs if w0 <= c0 <= w1]
    ok = [x for x in win if x is not None]
    out = {
        "N": meta["N"],
        "routers": meta.get("R"),
        "ray_commit": meta["ray_commit"],
        "window_s": w1 - w0,
        "requests": len(win),
        "failed": len(win) - len(ok),
        "req_s": len(win) / (w1 - w0),
        "ttft_p50_ms": pct(ok, 0.50),
        "ttft_p99_ms": pct(ok, 0.99),
    }

    scrapes = load_scrapes(run_dir)
    inside = [(t, s) for t, s in scrapes if w0 <= t <= w1]
    if len(inside) >= 2:
        (ta, a), (tb, b) = inside[0], inside[-1]
        dt = tb - ta
        routed = total(b, "ray_serve_haproxy_ingress_router_requests_total") - total(
            a, "ray_serve_haproxy_ingress_router_requests_total"
        )
        lat = delta_buckets(a, b, "ray_serve_haproxy_ingress_router_latency_ms")
        # Ray LLM runs the router's code on the replica's main loop
        # (RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD=0 in its runtime env).
        lag = delta_buckets(
            a, b, "ray_serve_event_loop_scheduling_latency_ms",
            deployment="LLMRouter", loop_type="main",
        )
        fallbacks = total(b, "ray_serve_haproxy_ingress_router_fallbacks_total") - total(
            a, "ray_serve_haproxy_ingress_router_fallbacks_total"
        )
        cpu = [total(s, "ray_component_cpu_percentage") for _, s in inside]
        router_cores = statistics.mean(cpu) / 100.0
        first, last = scrapes[0][1], scrapes[-1][1]
        routed_run = total(last, "ray_serve_haproxy_ingress_router_requests_total") - total(
            first, "ray_serve_haproxy_ingress_router_requests_total"
        )
        calls_run = total(last, "ray_serve_deployment_request_counter_total", deployment="LLMRouter") - total(
            first, "ray_serve_deployment_request_counter_total", deployment="LLMRouter"
        )
        out.update({
            "scrape_window_s": dt,
            "routed_per_s": routed / dt if dt else None,
            "route_ms_p50": quantile(0.50, lat),
            "route_ms_p99": quantile(0.99, lat),
            "router_cores_prom": router_cores,
            "router_cpu_ms_per_req_prom": 1000 * router_cores / out["req_s"] if out["req_s"] else None,
            "loop_lag_ms_p50": quantile(0.50, lag),
            "loop_lag_ms_p99": quantile(0.99, lag),
            "loop_lag_share_gt5ms": share_above(lag, 5),
            "loop_lag_share_gt10ms": share_above(lag, 10),
            "loop_lag_samples": lag.get(float("inf")),
            "router_calls_per_routed_req": calls_run / routed_run if routed_run else None,
            "fallbacks_per_s": fallbacks / dt if dt else None,
        })

    # Harness-side context: KV hit ratio and load-view mismatch (monitor.jsonl), and
    # (psutil CPU is head-local and routers run on every node, so it is not reported.)
    try:
        import reeval_summary

        m = reeval_summary.run_metrics(run_dir, from_s)
        out["hit_ratio"] = m.get("hit_ratio")
        out["mismatch_pct"] = m.get("mismatch_pct")
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        out["harness_error"] = repr(e)
    try:
        import analyze

        rows = analyze.load(os.path.join(run_dir, "monitor.jsonl"))
        in_win = [rec for rec in rows if w0 <= rec["time"] <= w1]
        out["tracked_requests_max"] = max(
            (r.get("tracked_requests") or 0) for rec in in_win for r in rec.get("routers", [])
        )
        # A leak would leave bookings behind once the load is gone.
        out["tracked_requests_end"] = sum(
            (r.get("tracked_requests") or 0) for r in rows[-1].get("routers", [])
        )
    except Exception:
        pass
    return out


# ---- tables -----------------------------------------------------------------

_RUN = re.compile(r"^(?P<variant>.+)_N(?P<n>\d+)_rep(?P<rep>\d+)$")
VARIANTS = ["nomesh", "mesh_serve", "mesh_zmq"]
NAMES = {"nomesh": "no-mesh (Ray-RPC)", "mesh_serve": "mesh + Serve handle", "mesh_zmq": "mesh + ZMQ"}


def collect(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_N*_rep*"))):
        m = _RUN.match(os.path.basename(d))
        f = os.path.join(d, "prom_summary.json")
        if m and os.path.exists(f):
            runs.setdefault((m["variant"], int(m["n"])), []).append(json.load(open(f)))
    return runs


def cell(rs, key, nd=1, scale=1.0):
    xs = [r[key] * scale for r in rs if r.get(key) is not None]
    if not xs:
        return "-"
    med = statistics.median(xs)
    if len(xs) == 1:
        return f"{med:.{nd}f}"
    return f"{med:.{nd}f} [{min(xs):.{nd}f}-{max(xs):.{nd}f}]"


def table(root):
    runs = collect(root)
    ns = sorted({n for _, n in runs})
    variants = [v for v in VARIANTS if any((v, n) in runs for n in ns)]
    head = "| | " + " | ".join(f"N={n}" for n in ns) + " |\n|---|" + "---|" * len(ns)

    def rows(fmt):
        lines = [head]
        for v in variants:
            lines.append(f"| {NAMES.get(v, v)} | " + " | ".join(
                fmt(runs.get((v, n), [])) if runs.get((v, n)) else "-" for n in ns) + " |")
        return "\n".join(lines)

    nrep = lambda rs: f" (n={len(rs)})" if rs else ""  # noqa: E731
    sections = [
        ("Time spent routing, p50 / p99 (ms) - HAProxy around each router call",
         lambda rs: f"{cell(rs, 'route_ms_p50', 2)} / {cell(rs, 'route_ms_p99', 2)}"),
        ("Router CPU per routed request (ms) - Ray per-process CPU / routed req/s",
         lambda rs: cell(rs, "router_cpu_ms_per_req_prom", 2)),
        ("Router event-loop lag p50 / p99 (ms)",
         lambda rs: f"{cell(rs, 'loop_lag_ms_p50', 2)} / {cell(rs, 'loop_lag_ms_p99', 2)}"),
        ("Router event-loop lag, share of samples > 5 ms / > 10 ms (%)",
         lambda rs: f"{cell(rs, 'loop_lag_share_gt5ms', 1, 100)} / {cell(rs, 'loop_lag_share_gt10ms', 1, 100)}"),
        ("Steady-state TTFT p50 / p99 (ms) - client side",
         lambda rs: f"{cell(rs, 'ttft_p50_ms', 1)} / {cell(rs, 'ttft_p99_ms', 1)}"),
        ("Context: throughput (req/s) / router fallbacks per s / failed requests",
         lambda rs: f"{cell(rs, 'req_s', 0)} / {cell(rs, 'fallbacks_per_s', 2)} / {cell(rs, 'failed', 0)}"),
        ("Context: calls into routers per routed request",
         lambda rs: cell(rs, "router_calls_per_routed_req", 2)),
        ("Context: KV cache hit ratio / load-view mismatch (%)",
         lambda rs: f"{cell(rs, 'hit_ratio', 3)} / {cell(rs, 'mismatch_pct', 1)}"),
        ("Context: total LLMRouter replicas / runs",
         lambda rs: f"{cell(rs, 'routers', 0)}{nrep(rs)}"),
    ]
    for title, fmt in sections:
        print(f"\n**{title}**\n\n{rows(fmt)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    r.add_argument("--from-s", type=float, default=60.0)
    t = sub.add_parser("table")
    t.add_argument("root")
    a = p.parse_args()
    if a.cmd == "run":
        print(json.dumps(run_summary(a.run_dir, a.from_s), indent=1))
    else:
        table(a.root)


if __name__ == "__main__":
    main()
