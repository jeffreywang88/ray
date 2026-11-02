"""Shrink finished campaign runs: keep per-request start time, phase and TTFT
(agentx/ttft.csv.gz) instead of AgentX's full record export, gzip the timeslice CSV and
logs. Verifies the recomputed prom_summary.json matches before deleting anything.

usage: python compact_runs.py <run_dir>...
"""
import glob
import gzip
import json
import os
import subprocess
import sys

for run in sys.argv[1:]:
    paths = glob.glob(os.path.join(run, "agentx", "profile_export.jsonl*"))
    if not paths:
        continue
    out = os.path.join(run, "agentx", "ttft.csv.gz")
    with gzip.open(paths[0], "rt") as f, gzip.open(out, "wt") as w:
        w.write("start_s,phase,ttft_ms\n")
        for line in f:
            rec = json.loads(line)
            md = rec.get("metadata", {})
            if md.get("request_start_ns") is None:
                continue
            ttft = rec.get("metrics", {}).get("time_to_first_token")
            ms = ""
            if ttft:
                val = ttft["value"] if isinstance(ttft, dict) else ttft
                unit = ttft.get("unit", "ms") if isinstance(ttft, dict) else "ms"
                ms = repr(val * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[unit])
            w.write(f"{md['request_start_ns'] / 1e9!r},{md.get('benchmark_phase', 'profiling')},{ms}\n")
    old = json.load(open(os.path.join(run, "prom_summary.json")))
    # Recompute from the compact file alone (the full export is moved aside first).
    os.rename(paths[0], paths[0] + ".bak")
    new = json.loads(subprocess.check_output(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prom_tables.py"), "run", run]))
    keys = ("requests", "failed", "req_s", "ttft_p50_ms", "ttft_p99_ms", "router_cpu_ms_per_req_prom")
    if any(old.get(k) != new.get(k) for k in keys):
        os.rename(paths[0] + ".bak", paths[0])
        os.remove(out)
        print(f"MISMATCH {run}: kept full export", {k: (old.get(k), new.get(k)) for k in keys})
        continue
    os.remove(paths[0] + ".bak")
    for name in ("profile_export_aiperf_timeslices.csv",):
        p = os.path.join(run, "agentx", name)
        if os.path.exists(p):
            subprocess.check_call(["gzip", "-f", p])
    subprocess.call(["rm", "-rf", os.path.join(run, "agentx", "logs")])
    if os.path.exists(os.path.join(run, "agentx.log")):
        subprocess.check_call(["gzip", "-f", os.path.join(run, "agentx.log")])
    print(f"compacted {run}")
