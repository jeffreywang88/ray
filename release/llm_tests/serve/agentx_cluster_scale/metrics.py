"""Run metrics with InferenceX's definitions (infx/results/agentic/request_metrics.py).

AgentX runs one shard per engine node, and a shard starts profiling when its own
warmup ends: under load, shards can start minutes apart, and a short profile can
end before another shard's starts. The headline metrics therefore cover only the
concurrent window, from the last shard's first request to the first shard's last
request, when every shard is profiling: requests that start in that window. With
one shard, that window is the whole profile.

Output tok/s is the output tokens of error-free requests over the window's
length. p90 interactivity is 1 / p90 of the per-request inter-token latency, in
tokens/s/user. Percentiles interpolate linearly between samples. The ``all_``
metrics use every profiling request, with tok/s as each shard's tokens over its
own window, summed.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if values.size else float("nan")


def _positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0)]


def _window_ns(load: Dict[str, Any]) -> Tuple[int, int]:
    starts, ends = load["columns"]["start_ns"], load["columns"]["end_ns"]
    return int(starts[starts >= 0].min()), int(ends[ends >= 0].max())


def profiling_window(load: Dict[str, Any]) -> Tuple[float, float]:
    """First request start and last request end, in seconds since the epoch."""
    start_ns, end_ns = _window_ns(load)
    return start_ns / 1e9, end_ns / 1e9


def _shard_windows(load: Dict[str, Any]) -> list:
    return [w for w in load.get("shard_windows_ns") or [_window_ns(load)] if w]


def concurrent_window_ns(load: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """When every shard with records was profiling; None if they never overlap."""
    windows = _shard_windows(load)
    start, end = max(w[0] for w in windows), min(w[1] for w in windows)
    return (start, end) if end > start else None


def measurement_window(load: Dict[str, Any]) -> Tuple[float, float]:
    """The concurrent window, else the whole profile, in seconds since the epoch."""
    start_ns, end_ns = concurrent_window_ns(load) or _window_ns(load)
    return start_ns / 1e9, end_ns / 1e9


def _latency(cols: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, float]:
    itl_s = _positive(cols["itl_ms"][mask]) / 1000
    ttft_ms = _positive(cols["ttft_ms"][mask])
    return {
        "mean_interactivity": float(1 / np.mean(itl_s)) if itl_s.size else float("nan"),
        "p50_interactivity": 1 / _percentile(itl_s, 50),
        "p90_interactivity": 1 / _percentile(itl_s, 90),
        "p90_itl_ms": 1000 * _percentile(itl_s, 90),
        "mean_ttft_ms": float(np.mean(ttft_ms)) if ttft_ms.size else float("nan"),
        "p50_ttft_ms": _percentile(ttft_ms, 50),
        "p90_ttft_ms": _percentile(ttft_ms, 90),
        "p99_ttft_ms": _percentile(ttft_ms, 99),
        "kv_hit_ratio": float(
            np.nansum(cols["cached"][mask]) / max(1.0, np.nansum(cols["isl"][mask]))
        ),
    }


def summarize(
    load: Dict[str, Any], num_replicas: int, gpus_per_replica: int
) -> Dict[str, float]:
    cols = load["columns"]
    start_ns, end_ns = _window_ns(load)
    windows = load.get("shard_windows_ns") or [[start_ns, end_ns]]
    shard_of = np.zeros(cols["osl"].size, dtype=np.int64)
    offset = 0
    for i, size in enumerate(load.get("shard_sizes") or [cols["osl"].size]):
        shard_of[offset : offset + size] = i
        offset += size
    # A shard without records (all failed) contributes no rate and no start.
    rate = np.array([0.0 if w is None else 1e9 / max(1, w[1] - w[0]) for w in windows])
    present = _shard_windows(load)
    starts = [w[0] for w in present]
    everything = np.ones(cols["osl"].size, dtype=bool)
    all_metrics = {
        "output_tok_s": float(np.nansum(cols["osl"] * rate[shard_of])),
        "req_s": float(rate[shard_of].sum()),
        **_latency(cols, everything),
    }
    # Headline: requests that start while every shard is profiling. If the
    # shards never overlap (an issue), fall back to all requests.
    window = concurrent_window_ns(load)
    if window is None:
        mask, window_s = everything, 0.0
        headline = dict(all_metrics)
        headline["input_tok_s"] = float(np.nansum(cols["isl"] * rate[shard_of]))
    else:
        mask = (cols["start_ns"] >= window[0]) & (cols["start_ns"] <= window[1])
        window_s = (window[1] - window[0]) / 1e9
        headline = {
            "output_tok_s": float(np.nansum(cols["osl"][mask]) / window_s),
            "req_s": float(mask.sum() / window_s),
            "input_tok_s": float(np.nansum(cols["isl"][mask]) / window_s),
            **_latency(cols, mask),
        }
    output_tok_s = headline.pop("output_tok_s")
    return {
        "requests": int(mask.sum()),
        "failed_requests": int(load["errors"]),
        "duration_s": (end_ns - start_ns) / 1e9,
        "shard_start_skew_s": (max(starts) - min(starts)) / 1e9,
        # How long every shard was profiling at once: the headline window.
        "shard_overlap_s": window_s,
        "output_tok_s": output_tok_s,
        "output_tok_s_per_replica": output_tok_s / num_replicas,
        "output_tok_s_per_gpu": output_tok_s / (num_replicas * gpus_per_replica),
        **headline,
        "all_requests": int(cols["osl"].size),
        **{f"all_{k}": v for k, v in all_metrics.items()},
    }
