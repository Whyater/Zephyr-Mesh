"""Shared small statistics helpers for the summary modules."""
from __future__ import annotations

from math import floor


def quantile(values: list[float], probability: float) -> float | None:
    """Linear-interpolation quantile (the same rule as NumPy's default).

    Returns ``None`` for an empty sample so callers can report "no data".
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]
