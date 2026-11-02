"""Scrape Ray's Prometheus endpoint every --interval seconds into a gzipped JSONL file.

Keeps only the families the replication-ceiling tables read, so a 5-minute run stays
small: HAProxy ingress-router metrics, Serve event-loop lag, Serve per-replica request
counters, and Ray's per-process CPU for the LLMRouter replicas. Each line is
{"t": unix seconds, "lines": [prometheus text lines]}. Waits for the endpoint to
come up and exits on SIGTERM. Sends the cluster's auth token (Ray token auth guards the
endpoint), read from $RAY_AUTH_TOKEN or ~/.ray/auth_token, as a Prometheus server would.

Multi-node: pass one --url per node. Each round fetches every node in parallel and
writes their lines merged into one record, or nothing if any node failed, so summed
counters never drop because a node was missed.

usage: python promscrape.py --url http://NODE1:PORT/metrics [--url ...] --out RUN/prom.jsonl.gz
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
import signal
import time
import urllib.request

KEEP = (
    "ray_serve_haproxy_ingress_router_",
    "ray_serve_event_loop_scheduling_latency_ms",
    "ray_serve_event_loop_tasks",
    "ray_serve_deployment_request_counter",
    "ray_serve_deployment_processing_latency_ms",
    "ray_component_cpu_percentage",
)
_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def keep(line):
    if not line.startswith(KEEP):
        return False
    # Only the routers' processes from the per-process CPU family.
    return not line.startswith("ray_component_cpu_percentage") or "LLMRouter" in line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=5.0)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    token = os.environ.get("RAY_AUTH_TOKEN")
    token_file = os.path.expanduser("~/.ray/auth_token")
    if not token and os.path.exists(token_file):
        token = open(token_file).read().strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return [ln for ln in r.read().decode().splitlines() if keep(ln)]

    pool = ThreadPoolExecutor(max_workers=min(32, len(args.url)))
    with gzip.open(args.out, "at") as f:
        while not _STOP:
            t0 = time.time()
            try:
                lines = [ln for part in pool.map(fetch, args.url) for ln in part]
                f.write(json.dumps({"t": t0, "lines": lines}) + "\n")
                f.flush()
            except Exception:
                pass  # a node not up yet or restarting: skip the whole round
            time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
