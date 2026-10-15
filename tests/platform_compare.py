"""Compare saved floating-point references across operating systems.

The saved numbers were recorded on macOS. Other platforms use different math
libraries, so results can differ in the last one or two digits. On macOS the
checks stay exact; elsewhere they allow a relative error of 1e-9 (or an absolute
error of 1e-12 near zero), far below anything the simulator reports.
"""
from __future__ import annotations

import math
import sys

import numpy as np

EXACT = sys.platform == "darwin"
RTOL = 1e-9
ATOL = 1e-12


def same_array(actual, expected) -> bool:
    """True when two arrays match: exactly on macOS, within tight tolerance elsewhere."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if EXACT:
        return bool(np.array_equal(actual, expected))
    return actual.shape == expected.shape and bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))


def same_tree(actual, expected) -> bool:
    """True when two JSON-like trees match, with the same float rule as same_array."""
    if EXACT:
        return actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_tree(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(same_tree(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, float) or isinstance(expected, float):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return actual == expected
        return math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL)
    return actual == expected
