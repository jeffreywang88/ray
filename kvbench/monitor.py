"""Sample router probes, replica stats, HAProxy and per-component CPU every
`--interval` seconds into a JSONL file. Values are cumulative; analyze.py takes
deltas. Stops on SIGTERM/SIGINT or after `--duration` seconds.

Usage: python monitor.py --out RUN_DIR/monitor.jsonl [--interval 10]
"""

import argparse
import csv
import io
import json
import signal
import time
import urllib.request

import psutil

import ray
from ray import serve

_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def _component(proc):
    try:
        name = proc.info["name"] or ""
        cmd = " ".join(proc.info["cmdline"] or [])
    except Exception:
        return None
    if name == "haproxy":
        return "haproxy"
    if "LLMRouter" in cmd or "kv_aware.vllm.tokenizer_worker" in cmd:
        # Tokenizer worker processes do the router's work.
        return "router"
    if "LLMServer:mock" in cmd:
        return "replica"
    if "agentx-venv" in cmd or "aiperf" in cmd:
        return "agentx"
    if "ServeController" in cmd:
        return "controller"
    if any(k in cmd for k in ("raylet", "gcs_server", "dashboard", "ray::", "log_monitor")):
        return "ray_other"
    return "other"


_PREV_CPU = {}
RSS_GB = {}


def cpu_by_component():
    """CPU seconds used per component since the previous call (per-pid deltas, so
    processes that exit or start between samples don't skew the totals), plus
    the busiest single process per component (cores) to spot CPU-bound ones."""
    global _PREV_CPU
    now = time.time()
    deltas, counts, peak, cur = {}, {}, {}, {}
    global RSS_GB
    RSS_GB = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_times", "memory_info"]):
        comp = _component(proc)
        t = proc.info["cpu_times"]
        if comp is None or t is None:
            continue
        total = t.user + t.system
        cur[proc.info["pid"]] = (now, total)
        mem = proc.info.get("memory_info")
        if mem is not None:
            RSS_GB[comp] = RSS_GB.get(comp, 0.0) + mem.rss / 2**30
        counts[comp] = counts.get(comp, 0) + 1
        prev = _PREV_CPU.get(proc.info["pid"])
        if prev is None:
            continue
        d = max(0.0, total - prev[1])
        deltas[comp] = deltas.get(comp, 0.0) + d
        cores = d / max(1e-6, now - prev[0])
        peak[comp] = max(peak.get(comp, 0.0), cores)
    _PREV_CPU = cur
    return deltas, counts, peak


def haproxy_stats():
    try:
        raw = urllib.request.urlopen("http://127.0.0.1:8404/stats;csv", timeout=5).read()
    except Exception as e:
        return {"error": str(e)}
    rows = list(csv.DictReader(io.StringIO(raw.decode().lstrip("# "))))
    out = {}
    for r in rows:
        if r.get("svname") == "BACKEND" or r.get("svname") == "FRONTEND":
            out[f'{r["pxname"]}/{r["svname"]}'] = {
                k: r.get(k) for k in ("scur", "stot", "qcur", "rate", "req_tot", "ttime", "qtime", "ctime", "rtime")
            }
    return out


def broadcast(handle, method, timeout_s=60):
    for _ in range(10):
        try:
            return handle.broadcast(method).results(timeout_s=timeout_s, return_exceptions=True)
        except Exception:
            time.sleep(1)
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    routers = serve.get_deployment_handle("LLMRouter", app_name="llm")
    servers = serve.get_deployment_handle("LLMServer:mock", app_name="llm")
    start = time.time()
    with open(args.out, "a") as f:
        while not _STOP:
            t0 = time.time()
            router_stats = [r for r in broadcast(routers, "bench_stats") if isinstance(r, dict)]
            server_stats = []
            for r in broadcast(servers, "get_stats"):
                if isinstance(r, dict):
                    r.pop("inflight_ids", None)
                    server_stats.append(r)
            cpu, nproc, peak = cpu_by_component()
            rec = {
                "time": t0,
                "collect_s": time.time() - t0,
                "routers": router_stats,
                "servers": server_stats,
                "haproxy": haproxy_stats(),
                "cpu_delta_s": cpu,
                "cpu_peak_proc_cores": peak,
                "rss_gb": RSS_GB,
                "nproc": nproc,
                "mem_available_gb": psutil.virtual_memory().available / 2**30,
                "host_cpu_pct": psutil.cpu_percent(interval=None),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if args.duration is not None and time.time() - start > args.duration:
                break
            time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
