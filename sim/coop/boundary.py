"""Tolerance boundaries for simulated cooperative tracking."""
from __future__ import annotations

import math
from typing import Sequence

from sim.s6_recovery import wilson_interval

FAIL_RATE_THRESHOLD = 0.5


def _end(hit: int | None, delays: Sequence[float]):
    if hit is None:
        return f">{float(delays[-1]):g}"
    if hit == 0:
        return "<0"
    return float(delays[hit - 1])


def tolerated(delays: Sequence[float], fail_counts: Sequence[int], n_seeds: int) -> dict:
    """Return the atlas-compatible simulated tolerance boundary."""
    if isinstance(n_seeds, bool) or int(n_seeds) != n_seeds or n_seeds < 1:
        raise ValueError("n_seeds must be a positive integer")
    if len(delays) == 0 or len(delays) != len(fail_counts):
        raise ValueError("delays and fail_counts must be nonempty and have equal length")
    if any(not math.isfinite(float(delay)) for delay in delays):
        raise ValueError("delays must be finite")
    if any(float(delays[i]) >= float(delays[i + 1]) for i in range(len(delays) - 1)):
        raise ValueError("delays must be strictly increasing")
    if any(isinstance(k, bool) or int(k) != k or k < 0 or k > n_seeds for k in fail_counts):
        raise ValueError("fail counts must be integers in [0, n_seeds]")
    rates = [float(k) / float(n_seeds) for k in fail_counts]
    intervals = [wilson_interval(int(k), int(n_seeds)) for k in fail_counts]
    low = [x[0] for x in intervals]
    high = [x[1] for x in intervals]
    failing = next((i for i, rate in enumerate(rates) if rate >= FAIL_RATE_THRESHOLD), None)
    upper_hit = next((i for i, value in enumerate(high) if value >= FAIL_RATE_THRESHOLD), None)
    lower_hit = next((i for i, value in enumerate(low) if value >= FAIL_RATE_THRESHOLD), None)
    result = {
        "value": None,
        "censored": None,
        "fails_at_zero": False,
        "interval_s": [_end(upper_hit, delays), _end(lower_hit, delays)],
    }
    if failing is None:
        result["censored"] = f">{float(delays[-1]):g}"
    elif failing == 0:
        result["fails_at_zero"] = True
    else:
        result["value"] = float(delays[failing - 1])
    return result
