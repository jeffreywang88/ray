"""Ingress routing latency and fallbacks from HAProxy's Serve metrics.

Needs RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED=1. Every node's metrics
endpoint is scraped periodically; a phase's numbers come from the difference
between the snapshots that bracket its profiling window.
"""

import collections
import math
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import ray

LATENCY_BUCKET = "ray_serve_haproxy_ingress_router_latency_ms_bucket"
FALLBACKS = "ray_serve_haproxy_ingress_router_fallbacks_total"
LE = re.compile(r'le="([^"]+)"')

Snapshot = Tuple[float, Dict[float, float], float]


def _auth_headers() -> Dict[str, str]:
    token = os.environ.get("RAY_AUTH_TOKEN")
    path = os.path.expanduser("~/.ray/auth_token")
    if not token and os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _scrape(url: str, headers: Dict[str, str]) -> Tuple[Dict[float, float], float]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return parse(response.read().decode())


def parse(text: str) -> Tuple[Dict[float, float], float]:
    """Latency buckets summed over all label sets, and the fallback count."""
    buckets: Dict[float, float] = collections.defaultdict(float)
    fallbacks = 0.0
    for line in text.splitlines():
        if line.startswith(LATENCY_BUCKET):
            buckets[float(LE.search(line).group(1))] += float(line.rsplit(None, 1)[1])
        elif line.startswith(FALLBACKS):
            fallbacks += float(line.rsplit(None, 1)[1])
    return buckets, fallbacks


def histogram_quantile(cumulative: Dict[float, float], q: float) -> float:
    """PromQL-style quantile; a rank in the +Inf bucket returns the last finite bound."""
    bounds = sorted(cumulative)
    rank = q * cumulative[bounds[-1]]
    prev_bound, prev_count = 0.0, 0.0
    for bound in bounds:
        count = cumulative[bound]
        if count >= rank:
            if math.isinf(bound):
                return prev_bound
            if count == prev_count:
                return bound
            return prev_bound + (bound - prev_bound) * (rank - prev_count) / (
                count - prev_count
            )
        prev_bound, prev_count = bound, count
    return prev_bound


class RoutingScraper:
    def __init__(self, interval_s: float = 10.0):
        self._interval_s = interval_s
        self._snapshots: List[Snapshot] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "RoutingScraper":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        headers = _auth_headers()
        with ThreadPoolExecutor(max_workers=32) as pool:
            while not self._stop.is_set():
                urls = [
                    f"http://{n['NodeManagerAddress']}:{n['MetricsExportPort']}/metrics"
                    for n in ray.nodes()
                    if n["Alive"]
                ]
                now = time.time()
                try:
                    scrapes = list(pool.map(lambda u: _scrape(u, headers), urls))
                except Exception:
                    # A round missing a node would undercount; drop it.
                    scrapes = None
                if scrapes:
                    buckets: Dict[float, float] = collections.defaultdict(float)
                    for node_buckets, _ in scrapes:
                        for bound, count in node_buckets.items():
                            buckets[bound] += count
                    self._snapshots.append(
                        (now, dict(buckets), sum(f for _, f in scrapes))
                    )
                self._stop.wait(self._interval_s)

    def summary(self, start_s: float, end_s: float) -> Dict[str, Optional[float]]:
        """Routing p50/p99 and fallbacks/s between the snapshots inside the window."""
        empty = {"route_p50_ms": None, "route_p99_ms": None, "fallbacks_per_s": None}
        inside = [s for s in self._snapshots if start_s <= s[0] <= end_s]
        if len(inside) < 2:
            return empty
        (t0, b0, f0), (t1, b1, f1) = inside[0], inside[-1]
        delta = {bound: count - b0.get(bound, 0.0) for bound, count in b1.items()}
        if delta.get(math.inf, 0.0) <= 0:
            return empty
        return {
            "route_p50_ms": histogram_quantile(delta, 0.5),
            "route_p99_ms": histogram_quantile(delta, 0.99),
            "fallbacks_per_s": (f1 - f0) / (t1 - t0),
        }
