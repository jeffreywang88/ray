"""Picks the Mocker speedups that make one replica match an InferenceX datapoint.

``speedup`` scales prefill and decode; ``decode_speedup`` scales decode on top,
so the decode rate is ``speedup * decode_speedup``. Inter-token latency falls
with the decode rate, and output tok/s rises with both the decode rate and the
prefill rate (shorter TTFT). The two are coupled, so the search takes Broyden
secant steps in log space over (speedup, decode rate).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# d log(tok/s, p90 ITL) / d log(speedup, decode rate) before any measurement.
INITIAL_JACOBIAN = np.array([[0.3, 0.7], [0.0, -1.0]])
MAX_LOG_STEP = np.log(8.0)


@dataclass
class Target:
    output_tok_s: float
    p90_itl_ms: float


@dataclass
class Point:
    speedup: float
    decode_speedup: float
    output_tok_s: float
    p90_itl_ms: float

    def errors(self, target: Target) -> List[float]:
        return [
            self.output_tok_s / target.output_tok_s - 1,
            self.p90_itl_ms / target.p90_itl_ms - 1,
        ]


def converged(point: Point, target: Target, tolerance: float) -> bool:
    return all(abs(e) <= tolerance for e in point.errors(target))


def _knobs(p: Point) -> np.ndarray:
    return np.log([p.speedup, p.speedup * p.decode_speedup])


def _residual(p: Point, target: Target) -> np.ndarray:
    return np.log(
        [p.output_tok_s / target.output_tok_s, p.p90_itl_ms / target.p90_itl_ms]
    )


def next_speedups(history: List[Point], target: Target) -> Tuple[float, float]:
    """Returns the next (speedup, decode_speedup) after the points in ``history``."""
    jacobian = INITIAL_JACOBIAN.copy()
    for prev, cur in zip(history, history[1:]):
        dx = _knobs(cur) - _knobs(prev)
        df = _residual(cur, target) - _residual(prev, target)
        if dx @ dx > 1e-12:
            jacobian += np.outer(df - jacobian @ dx, dx) / (dx @ dx)
    step = -np.linalg.lstsq(jacobian, _residual(history[-1], target), rcond=None)[0]
    step *= min(1.0, MAX_LOG_STEP / np.abs(step).max(initial=1e-12))
    speedup, decode_rate = np.exp(_knobs(history[-1]) + step)
    return float(speedup), float(decode_rate / speedup)
