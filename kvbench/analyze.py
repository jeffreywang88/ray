"""Summarize a run directory: monitor.jsonl (+ AgentX artifacts if present).

Prints a per-interval table and a steady-state summary (JSON) over the window
[--skip-s, end]. Also writes RUN_DIR/summary.json.

Usage: python analyze.py RUN_DIR [--skip-s 300] [--table]
"""

import argparse
import glob
import json
import os
import statistics


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def server_totals(rec):
    keys = ("requests", "completed", "prompt_tokens", "cached_tokens", "completion_tokens",
            "kv_stored_blocks", "kv_removed_blocks", "token_channel_hits", "token_key_missing",
            "token_store_misses", "inflight", "errors", "engine_ttft_s", "engine_ttft_n")
    out = {k: 0 for k in keys}
    for s in rec["servers"]:
        for k in keys:
            out[k] += s.get(k, 0)
    return out


def router_map(rec):
    return {r.get("rank"): r for r in rec["routers"]}


def mismatch(rec):
    """Sum over workers of |tracked in-flight - actual in-flight|, per router."""
    actual = {str(s["worker_id"]): s["inflight"] for s in rec["servers"]}
    total_actual = sum(actual.values())
    out = {}
    for rank, r in router_map(rec).items():
        view = r.get("svc_load_view") or r.get("load_view") or {}
        diff = sum(abs(view.get(w, 0) - a) for w, a in actual.items())
        diff += sum(v for w, v in view.items() if w not in actual)
        out[rank] = {"abs_diff": diff, "tracked": sum(view.values()), "actual": total_actual}
    return out


def intervals(recs):
    rows = []
    for prev, cur in zip(recs, recs[1:]):
        dt = cur["time"] - prev["time"]
        if dt <= 0:
            continue
        a, b = server_totals(prev), server_totals(cur)
        d = {k: b[k] - a[k] for k in b}
        row = {
            "t": cur["time"],
            "dt": dt,
            "completed_per_s": d["completed"] / dt,
            "requests_per_s": d["requests"] / dt,
            "hit_ratio": d["cached_tokens"] / d["prompt_tokens"] if d["prompt_tokens"] else None,
            "prompt_tok_per_s": d["prompt_tokens"] / dt,
            "kv_removed_per_s": d["kv_removed_blocks"] / dt,
            "kv_stored_per_s": d["kv_stored_blocks"] / dt,
            "token_miss": d["token_key_missing"] + d["token_store_misses"],
            "token_hits": d["token_channel_hits"],
            "inflight": b["inflight"],
            "engine_ttft_ms": 1000 * d["engine_ttft_s"] / d["engine_ttft_n"] if d["engine_ttft_n"] else None,
            "errors": d["errors"],
            "cpu_cores": {k: v / dt for k, v in cur.get("cpu_delta_s", {}).items()},
            "cpu_peak_proc_cores": cur.get("cpu_peak_proc_cores", {}),
            "host_cpu_pct": cur.get("host_cpu_pct"),
            "mem_available_gb": cur.get("mem_available_gb"),
            "routers": {},
            "mismatch": mismatch(cur),
        }
        pr, cr = router_map(prev), router_map(cur)
        for rank, r in cr.items():
            p = pr.get(rank)
            if p is None:
                continue
            arrived = r["res_arrived"] - p["res_arrived"]
            processed = r["res_processed"] - p["res_processed"]
            applied = r["res_applied"] - p["res_applied"]
            routes = r["routes"] - p["routes"]
            loop = (r["loop_cpu_s"] - p["loop_cpu_s"]) if r.get("loop_cpu_s") and p.get("loop_cpu_s") else None
            proc = r["process_cpu_s"] - p["process_cpu_s"]
            s = r.get("samples") or {}
            row["routers"][rank] = {
                "routes_per_s": routes / dt,
                "res_arrived_per_s": arrived / dt,
                "res_processed_per_s": processed / dt,
                "res_applied_per_s": applied / dt,
                "skip_pct": 100.0 * (processed - applied) / processed if processed else None,
                "backlog": r["res_arrived"] - r["res_processed"],
                "lag_p50": (s.get("lag") or {}).get("p50"),
                "lag_max": (s.get("lag") or {}).get("max"),
                "sem_held": r.get("sem_held"),
                "sem_waiters": r.get("sem_waiters"),
                "loop_lag_p50_ms": 1000 * (s.get("loop_lag") or {}).get("p50", float("nan")),
                "loop_lag_p99_ms": 1000 * (s.get("loop_lag") or {}).get("p99", float("nan")),
                "lifecycle_events_per_s": (r["lifecycle_events"] - p["lifecycle_events"]) / dt,
                "loop_cpu_frac": loop / dt if loop is not None else None,
                "loop_ms_per_route": 1000 * loop / routes if loop is not None and routes else None,
                "proc_cpu_cores": proc / dt,
                "select_p50_ms": 1000 * (s.get("select") or {}).get("p50", float("nan")),
                "select_p99_ms": 1000 * (s.get("select") or {}).get("p99", float("nan")),
                "pick_p50_ms": 1000 * (s.get("pick") or {}).get("p50", float("nan")),
                "pick_p99_ms": 1000 * (s.get("pick") or {}).get("p99", float("nan")),
                "tokenize_p50_ms": 1000 * (s.get("tokenize") or {}).get("p50", float("nan")),
                "tracked": r.get("tracked_requests"),
            }
        rows.append(row)
    return rows


def _med(values):
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else None


def agentx_summary(run_dir):
    paths = glob.glob(os.path.join(run_dir, "**", "profile_export_aiperf.json"), recursive=True)
    if not paths:
        return None
    with open(paths[0]) as f:
        data = json.load(f)
    out = {}
    for key in ("time_to_first_token", "inter_token_latency", "request_latency",
                "request_throughput", "output_token_throughput", "request_count"):
        m = data.get(key) or (data.get("records") or {}).get(key)
        if isinstance(m, dict):
            out[key] = {k: m.get(k) for k in ("avg", "p50", "p90", "p99", "unit") if k in m}
    return out


def summarize(rows, skip_s):
    if not rows:
        return {}
    t0 = rows[0]["t"]
    win = [r for r in rows if r["t"] - t0 >= skip_s] or rows
    dur = sum(r["dt"] for r in win)
    out = {
        "window_s": dur,
        "completed_per_s": sum(r["completed_per_s"] * r["dt"] for r in win) / dur,
        "hit_ratio": None,
        "kv_removed_per_s": sum(r["kv_removed_per_s"] * r["dt"] for r in win) / dur,
        "token_miss_total": sum(r["token_miss"] for r in win),
        "token_hits_total": sum(r["token_hits"] for r in win),
        "inflight_median": _med([r["inflight"] for r in win]),
        "host_cpu_pct_median": _med([r.get("host_cpu_pct") for r in win]),
        "mem_available_gb_min": min((r.get("mem_available_gb") or 1e9) for r in win),
        "cpu_cores_median": {},
        "routers": {},
    }
    prompt = sum(r["prompt_tok_per_s"] * r["dt"] for r in win)
    cached = sum((r["hit_ratio"] or 0) * r["prompt_tok_per_s"] * r["dt"] for r in win)
    out["hit_ratio"] = cached / prompt if prompt else None
    for comp in win[-1]["cpu_cores"]:
        out["cpu_cores_median"][comp] = _med([r["cpu_cores"].get(comp) for r in win])
    out["cpu_peak_proc_cores_max"] = {
        comp: max((r["cpu_peak_proc_cores"].get(comp, 0) for r in win), default=0)
        for comp in win[-1].get("cpu_peak_proc_cores", {})
    }
    ranks = sorted({k for r in win for k in r["routers"]}, key=lambda x: (x is None, x))
    for rank in ranks:
        rr = [r["routers"][rank] for r in win if rank in r["routers"]]
        out["routers"][rank] = {k: _med([x[k] for x in rr]) for k in rr[0]}
        out["routers"][rank]["backlog_last"] = rr[-1]["backlog"]
        out["routers"][rank]["mismatch_median"] = _med(
            [r["mismatch"].get(rank, {}).get("abs_diff") for r in win]
        )
    # Lag trend: compare first and last third of the window.
    lags = [_med([x.get("lag_p50") for x in r["routers"].values()]) for r in win]
    lags = [x for x in lags if x is not None]
    if len(lags) >= 6:
        k = len(lags) // 3
        out["lag_p50_first_third"] = _med(lags[:k])
        out["lag_p50_last_third"] = _med(lags[-k:])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--skip-s", type=float, default=300)
    p.add_argument("--table", action="store_true")
    args = p.parse_args()
    rows = intervals(load(os.path.join(args.run_dir, "monitor.jsonl")))
    if args.table:
        for r in rows:
            rr = r["routers"]
            lag = _med([x["lag_p50"] for x in rr.values()])
            skip = _med([x["skip_pct"] for x in rr.values()])
            loopms = _med([x["loop_ms_per_route"] for x in rr.values()])
            hr = r["hit_ratio"]
            print(
                f"{r['t'] - rows[0]['t']:7.0f}s done/s={r['completed_per_s']:7.1f} "
                f"hit={hr if hr is None else round(hr, 3)} inflight={r['inflight']:5d} "
                f"churn={r['kv_removed_per_s']:8.0f} miss={r['token_miss']:4d} "
                f"lag50={lag if lag is None else round(lag, 3)} skip%={skip if skip is None else round(skip, 1)} "
                f"loop_ms/req={loopms if loopms is None else round(loopms, 2)} "
                f"cpu={ {k: round(v, 1) for k, v in r['cpu_cores'].items()} }"
            )
    summary = summarize(rows, args.skip_s)
    summary["agentx"] = agentx_summary(args.run_dir)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
