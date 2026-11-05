"""Picks the Mocker speedups that make one replica match an InferenceX datapoint.

The Mocker's two knobs map to the engine's two latencies:

- ``speedup`` scales prefill (and decode), so it sets TTFT;
- the decode rate ``speedup * decode_speedup`` sets the inter-token latency.

Output tok/s is not a target: AgentX replays recorded think times, so a closed
loop's request rate barely depends on engine speed, and a short window holds
fewer long outputs than InferenceX's hour. It is reported as a check.

Client TTFT also includes the router's work (rendering and tokenizing the
prompt), which the speedups don't touch. When that alone exceeds the target,
client TTFT can't be met; the search then stops at ``MAX_SPEEDUP`` and the
calibration doesn't converge. ``--calibration-ttft engine`` matches the
engine's own TTFT (request arrival at the engine to first token) instead.
With ``--speedup`` fixed, only the decode rate is searched, on p90 ITL.

The search takes Broyden secant steps in log space over (speedup, decode rate).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# d log(p90 TTFT, p90 ITL) / d log(speedup, decode rate) before any measurement.
# Routing time dilutes client TTFT's response to prefill speed; prefill chunks
# that share a step with decodes make ITL respond a little to speedup.
INITIAL_JACOBIAN = np.array([[-0.5, 0.0], [-0.1, -1.0]])
# The same for the engine's own TTFT, which is all engine time: running and
# queueing behind chunked prefills. Queueing makes it fall faster than 1 /
# prefill speed: on the default workload, p90 fell ~25x from speedup 1 to 5.
ENGINE_TTFT_JACOBIAN = np.array([[-2.0, 0.0], [-0.1, -1.0]])
MAX_LOG_STEP = np.log(8.0)
MIN_SPEEDUP, MAX_SPEEDUP = 0.05, 64.0


@dataclass
class Target:
    p90_ttft_ms: float
    p90_itl_ms: float


@dataclass
class Point:
    speedup: float
    decode_speedup: float
    p90_ttft_ms: float
    p90_itl_ms: float
    output_tok_s: float

    def errors(self, target: Target) -> List[float]:
        return [
            self.p90_ttft_ms / target.p90_ttft_ms - 1,
            self.p90_itl_ms / target.p90_itl_ms - 1,
        ]


def converged(
    point: Point, target: Target, tolerance: float, fixed_speedup: bool
) -> bool:
    ttft_err, itl_err = point.errors(target)
    return abs(itl_err) <= tolerance and (fixed_speedup or abs(ttft_err) <= tolerance)


def _knobs(p: Point) -> np.ndarray:
    return np.log([p.speedup, p.speedup * p.decode_speedup])


def _residual(p: Point, target: Target) -> np.ndarray:
    return np.log(
        [p.p90_ttft_ms / target.p90_ttft_ms, p.p90_itl_ms / target.p90_itl_ms]
    )


def next_speedups(
    history: List[Point],
    target: Target,
    fixed_speedup: bool = False,
    prior: np.ndarray = INITIAL_JACOBIAN,
) -> Tuple[float, float]:
    """Returns the next (speedup, decode_speedup) after the points in ``history``."""
    jacobian = prior.copy()
    for prev, cur in zip(history, history[1:]):
        dx = _knobs(cur) - _knobs(prev)
        df = _residual(cur, target) - _residual(prev, target)
        if dx @ dx > 1e-12:
            jacobian += np.outer(df - jacobian @ dx, dx) / (dx @ dx)
    residual = _residual(history[-1], target)
    if fixed_speedup:
        # ITL falls as the decode rate rises; a flat or rising measured slope is
        # noise, so keep at least a tenth of the prior slope.
        slope = min(jacobian[1, 1], 0.1 * prior[1, 1])
        step = np.array([0.0, -residual[1] / slope])
    else:
        step = -np.linalg.lstsq(jacobian, residual, rcond=None)[0]
    step *= min(1.0, MAX_LOG_STEP / np.abs(step).max(initial=1e-12))
    speedup, decode_rate = np.exp(_knobs(history[-1]) + step)
    if not fixed_speedup:
        speedup = float(np.clip(speedup, MIN_SPEEDUP, MAX_SPEEDUP))
    return float(speedup), float(decode_rate / speedup)
