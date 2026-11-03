"""Run metrics with InferenceX's definitions (infx/results/agentic/request_metrics.py).

Output tok/s is the output tokens of error-free profiling requests over
(last request end - first request start). p90 interactivity is 1 / p90 of the
per-request inter-token latency, in tokens/s/user. Percentiles interpolate
linearly between samples.
"""

from typing import Any, Dict, Tuple

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


def summarize(
    load: Dict[str, Any], num_replicas: int, gpus_per_replica: int
) -> Dict[str, float]:
    cols = load["columns"]
    start_ns, end_ns = _window_ns(load)
    duration_s = (end_ns - start_ns) / 1e9
    output_tok_s = np.nansum(cols["osl"]) / duration_s
    itl_s = _positive(cols["itl_ms"]) / 1000
    ttft_ms = _positive(cols["ttft_ms"])
    prompt_tokens = np.nansum(cols["isl"])
    return {
        "requests": int(cols["osl"].size),
        "failed_requests": int(load["errors"]),
        "duration_s": float(duration_s),
        "req_s": float(cols["osl"].size / duration_s),
        "input_tok_s": float(prompt_tokens / duration_s),
        "output_tok_s": float(output_tok_s),
        "output_tok_s_per_replica": float(output_tok_s / num_replicas),
        "output_tok_s_per_gpu": float(output_tok_s / (num_replicas * gpus_per_replica)),
        "mean_interactivity": float(1 / np.mean(itl_s)),
        "p50_interactivity": 1 / _percentile(itl_s, 50),
        "p90_interactivity": 1 / _percentile(itl_s, 90),
        "p90_itl_ms": 1000 * _percentile(itl_s, 90),
        "mean_ttft_ms": float(np.mean(ttft_ms)),
        "p50_ttft_ms": _percentile(ttft_ms, 50),
        "p90_ttft_ms": _percentile(ttft_ms, 90),
        "p99_ttft_ms": _percentile(ttft_ms, 99),
        "kv_hit_ratio": float(np.nansum(cols["cached"]) / prompt_tokens),
    }
