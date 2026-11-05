"""Ingress routing metrics from every node's Prometheus endpoint.

Needs RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED=1. Every node's metrics
endpoint is scraped periodically; a phase's numbers come from the difference
between the snapshots that bracket its profiling window:

- HAProxy's time in the ingress router (latency histogram), its router
  fallbacks, and bodies it truncated before forwarding them to the router.
- Where each Mocker request's prompt token ids came from (``mock_engine``):
  the router's token channel, or the engine's own tokenization.
- The Mocker engines' own TTFT, and their requests by whether the engine had
  served the request's session before.

``session_probe`` checks session pinning end to end before the load starts.
"""

import collections
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from mock_engine import (
    CHAT_PATH,
    ENGINE_HEADER,
    PROMPTS_METRIC,
    SESSION_NEW,
    SESSION_NONE,
    SESSION_REPEAT,
    SESSIONS_METRIC,
    SOURCE_CHANNEL,
    SOURCE_NO_KEY,
    SOURCE_STORE_MISS,
    TTFT_METRIC,
)
from ray.serve._private.constants import SERVE_SESSION_ID

import ray

# Histogram bucket series -> snapshot key.
HISTOGRAMS = {
    "ray_serve_haproxy_ingress_router_latency_ms_bucket": "route",
    f"ray_{TTFT_METRIC}_bucket": "engine_ttft",
}
# Counter name prefix -> snapshot key. A ``source`` tag is appended to the key.
COUNTERS = {
    "ray_serve_haproxy_ingress_router_fallbacks_total": "fallbacks",
    "ray_serve_haproxy_ingress_router_truncations_total": "truncations",
    f"ray_{PROMPTS_METRIC}_total": "prompts",
    f"ray_{SESSIONS_METRIC}_total": "sessions",
}
LE = re.compile(r'le="([^"]+)"')
SOURCE = re.compile(r'source="([^"]+)"')

Buckets = Dict[str, Dict[float, float]]
# (time, histogram buckets by key, counters)
Snapshot = Tuple[float, Buckets, Dict[str, float]]


def _auth_headers() -> Dict[str, str]:
    token = os.environ.get("RAY_AUTH_TOKEN")
    path = os.path.expanduser("~/.ray/auth_token")
    if not token and os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _scrape(url: str, headers: Dict[str, str]) -> Tuple[Buckets, Dict[str, float]]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return parse(response.read().decode())


def _is_series(line: str, name: str) -> bool:
    # The next character must end the name, so a longer metric name that
    # shares the prefix is not counted.
    return line.startswith(name) and line[len(name) : len(name) + 1] in ("{", " ")


def parse(text: str) -> Tuple[Buckets, Dict[str, float]]:
    """Histogram buckets and counters, each summed over all label sets."""
    buckets: Buckets = collections.defaultdict(lambda: collections.defaultdict(float))
    counters: Dict[str, float] = collections.defaultdict(float)
    for line in text.splitlines():
        if not line.startswith("ray_"):
            continue
        histogram = next(
            (k for n, k in HISTOGRAMS.items() if _is_series(line, n)), None
        )
        if histogram is not None:
            bound = float(LE.search(line).group(1))
            buckets[histogram][bound] += float(line.rsplit(None, 1)[1])
            continue
        for prefix, key in COUNTERS.items():
            if _is_series(line, prefix):
                source = SOURCE.search(line)
                if source:
                    key = f"{key}:{source.group(1)}"
                counters[key] += float(line.rsplit(None, 1)[1])
                break
    return {k: dict(v) for k, v in buckets.items()}, counters


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


def session_requests(counters: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Engine requests by whether that engine served their session before.

    Under session pinning, a session reaches a new engine only on its first
    turn, so ``session_new_engine_share`` is the share of requests that start a
    session; a session moved to another engine adds a "new" there. The share
    depends only on the per-node workload, which doesn't change with N.
    """
    new = counters.get(f"sessions:{SESSION_NEW}", 0.0)
    repeat = counters.get(f"sessions:{SESSION_REPEAT}", 0.0)
    none = counters.get(f"sessions:{SESSION_NONE}", 0.0)
    return {
        "session_requests_new": new,
        "session_requests_repeat": repeat,
        "requests_without_session": none,
        "session_new_engine_share": new / (new + repeat) if new + repeat else None,
    }


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
                    buckets: Buckets = collections.defaultdict(
                        lambda: collections.defaultdict(float)
                    )
                    counters: Dict[str, float] = collections.defaultdict(float)
                    for node_buckets, node_counters in scrapes:
                        for key, histogram in node_buckets.items():
                            for bound, count in histogram.items():
                                buckets[key][bound] += count
                        for key, value in node_counters.items():
                            counters[key] += value
                    self._snapshots.append(
                        (now, {k: dict(v) for k, v in buckets.items()}, dict(counters))
                    )
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
            "engine_p50_ttft_ms": None,
            "engine_p90_ttft_ms": None,
            **prompt_sources({}),
            **session_requests({}),
        }
        inside = [s for s in self._snapshots if start_s <= s[0] <= end_s]
        if len(inside) < 2:
            return empty
        (t0, b0, c0), (t1, b1, c1) = inside[0], inside[-1]
        counters = _delta(c1, c0)
        result = {
            **empty,
            "fallbacks_per_s": counters.get("fallbacks", 0.0) / (t1 - t0),
            "truncated_bodies": counters.get("truncations", 0.0),
            **prompt_sources(counters),
            **session_requests(counters),
        }
        ttft = _delta(b1.get("engine_ttft", {}), b0.get("engine_ttft", {}))
        if ttft.get(math.inf, 0.0) > 0:
            for q, key in ((0.5, "engine_p50_ttft_ms"), (0.9, "engine_p90_ttft_ms")):
                result[key] = histogram_quantile(ttft, q)
        buckets = _delta(b1.get("route", {}), b0.get("route", {}))
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


def _probe_prompt(rng: random.Random, words: int) -> str:
    # Unrelated text on every turn: no prompt shares KV blocks with another, so
    # only the session id can keep a session's turns on one engine.
    return " ".join(f"w{rng.randrange(1_000_000)}" for _ in range(words))


def _probe_session(
    url: str, model: str, session_id: Optional[str], turns: int, seed: str
) -> List[Optional[str]]:
    """Sends ``turns`` sequential requests; returns the engine of each."""
    rng = random.Random(seed)
    engines = []
    for _ in range(turns):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _probe_prompt(rng, 200)}],
            "max_tokens": 16,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers[SERVE_SESSION_ID] = session_id
        request = urllib.request.Request(
            url.rstrip("/") + CHAT_PATH,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
                engines.append(response.headers.get(ENGINE_HEADER))
        except Exception:
            engines.append(None)
    return engines


def session_probe(
    urls: List[str], model: str, sessions: int, turns: int = 4
) -> Dict[str, Any]:
    """Checks that the routers pin a session's turns to one engine.

    Runs ``sessions`` concurrent sessions, round-robin over ``urls`` (one per
    node, as AgentX's shards do), each ``turns`` sequential requests with
    unrelated prompts: first without a session id (control), then with one.
    A session is pinned when every turn returned the same engine.
    """
    tag = f"probe-{int(time.time())}"
    result: Dict[str, Any] = {"sessions": sessions, "turns": turns}
    for name, with_ids in (("control", False), ("pinned", True)):
        with ThreadPoolExecutor(max_workers=sessions) as pool:
            runs = list(
                pool.map(
                    lambda i: _probe_session(
                        urls[i % len(urls)],
                        model,
                        f"{tag}-{i}" if with_ids else None,
                        turns,
                        seed=f"{tag}-{name}-{i}",
                    ),
                    range(sessions),
                )
            )
        answered = [r for r in runs if None not in r]
        result[f"{name}_failed_requests"] = sum(r.count(None) for r in runs)
        result[f"{name}_one_engine_share"] = (
            sum(len(set(r)) == 1 for r in answered) / len(answered)
            if answered
            else None
        )
        result[f"{name}_engines"] = len({e for r in answered for e in r})
    return result
