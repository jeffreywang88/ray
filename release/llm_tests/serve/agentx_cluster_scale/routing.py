"""Ingress routing metrics from every node's Prometheus endpoint.

Needs RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED=1. Every node's metrics
endpoint is scraped periodically; a phase's numbers come from the difference
between the snapshots that bracket its profiling window:

- HAProxy's time in the ingress router (latency histogram), its router
  fallbacks, and bodies it truncated before forwarding them to the router.
- Where each Mocker request's prompt token ids came from (``mock_engine``):
  the router's token channel, or the engine's own tokenization.
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

from mock_engine import (
    PROMPTS_METRIC,
    SOURCE_CHANNEL,
    SOURCE_NO_KEY,
    SOURCE_STORE_MISS,
)

import ray

LATENCY_BUCKET = "ray_serve_haproxy_ingress_router_latency_ms_bucket"
# Counter name prefix -> snapshot key. A ``source`` tag is appended to the key.
COUNTERS = {
    "ray_serve_haproxy_ingress_router_fallbacks_total": "fallbacks",
    "ray_serve_haproxy_ingress_router_truncations_total": "truncations",
    f"ray_{PROMPTS_METRIC}_total": "prompts",
}
LE = re.compile(r'le="([^"]+)"')
SOURCE = re.compile(r'source="([^"]+)"')

# (time, latency buckets, counters)
Snapshot = Tuple[float, Dict[float, float], Dict[str, float]]


def _auth_headers() -> Dict[str, str]:
    token = os.environ.get("RAY_AUTH_TOKEN")
    path = os.path.expanduser("~/.ray/auth_token")
    if not token and os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _scrape(
    url: str, headers: Dict[str, str]
) -> Tuple[Dict[float, float], Dict[str, float]]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return parse(response.read().decode())


def parse(text: str) -> Tuple[Dict[float, float], Dict[str, float]]:
    """Latency buckets and counters, each summed over all label sets."""
    buckets: Dict[float, float] = collections.defaultdict(float)
    counters: Dict[str, float] = collections.defaultdict(float)
    for line in text.splitlines():
        if line.startswith(LATENCY_BUCKET):
            buckets[float(LE.search(line).group(1))] += float(line.rsplit(None, 1)[1])
            continue
        for prefix, key in COUNTERS.items():
            # The next character must end the name, so a longer metric
            # name that shares the prefix is not counted.
            if line.startswith(prefix) and line[len(prefix) : len(prefix) + 1] in (
                "{",
                " ",
            ):
                source = SOURCE.search(line)
                if source:
                    key = f"{key}:{source.group(1)}"
                counters[key] += float(line.rsplit(None, 1)[1])
                break
    return buckets, counters


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


def _delta(after: Dict, before: Dict) -> Dict:
    return {k: v - before.get(k, 0.0) for k, v in after.items()}


def prompt_sources(counters: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Mocker prompts by source, and the share the engine had to tokenize."""
    channel = counters.get(f"prompts:{SOURCE_CHANNEL}", 0.0)
    no_key = counters.get(f"prompts:{SOURCE_NO_KEY}", 0.0)
    store_miss = counters.get(f"prompts:{SOURCE_STORE_MISS}", 0.0)
    total = channel + no_key + store_miss
    return {
        "prompts_from_token_channel": channel,
        "prompts_engine_tokenized_no_key": no_key,
        "prompts_engine_tokenized_store_miss": store_miss,
        "engine_tokenized_share": (no_key + store_miss) / total if total else None,
    }


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
                    counters: Dict[str, float] = collections.defaultdict(float)
                    for node_buckets, node_counters in scrapes:
                        for bound, count in node_buckets.items():
                            buckets[bound] += count
                        for key, value in node_counters.items():
                            counters[key] += value
                    self._snapshots.append((now, dict(buckets), dict(counters)))
                self._stop.wait(self._interval_s)

    def summary(self, start_s: float, end_s: float) -> Dict[str, Optional[float]]:
        """Metrics between the first and last snapshots inside the window."""
        empty = {
            "route_p50_ms": None,
            "route_p99_ms": None,
            "route_share_over_top_bucket": None,
            "route_top_bucket_ms": None,
            "fallbacks_per_s": None,
            "truncated_bodies": None,
            **prompt_sources({}),
        }
        inside = [s for s in self._snapshots if start_s <= s[0] <= end_s]
        if len(inside) < 2:
            return empty
        (t0, b0, c0), (t1, b1, c1) = inside[0], inside[-1]
        buckets, counters = _delta(b1, b0), _delta(c1, c0)
        result = {
            **empty,
            "fallbacks_per_s": counters.get("fallbacks", 0.0) / (t1 - t0),
            "truncated_bodies": counters.get("truncations", 0.0),
            **prompt_sources(counters),
        }
        total = buckets.get(math.inf, 0.0)
        if total > 0:
            # Serve's histogram tops out at 1 s, and routing a prompt near 1M
            # tokens takes seconds (mostly tokenization). A quantile past the
            # top bucket is unknown, so it is None rather than the bucket bound.
            top = max(b for b in buckets if not math.isinf(b))
            over = 1 - buckets.get(top, 0.0) / total
            result["route_share_over_top_bucket"] = over
            result["route_top_bucket_ms"] = top
            for q, key in ((0.5, "route_p50_ms"), (0.99, "route_p99_ms")):
                if 1 - q >= over:
                    result[key] = histogram_quantile(buckets, q)
        return result
