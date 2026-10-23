"""Runner checks for EXP-036 (proposal): decision rules, bootstrap and byte reproduction."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.d_run_coop_ct import (
    CONDITIONS,
    DECISION_DELAY,
    _bootstrap_indices,
    _checks,
    _difference,
    _verdicts,
    main,
)

KEY = f"{DECISION_DELAY:g}"


def _boundary(kind, value=None):
    if kind == "zero":
        return {"value": None, "censored": None, "fails_at_zero": True}
    if kind == "censored":
        return {"value": None, "censored": ">0.5", "fails_at_zero": False}
    return {"value": value, "censored": None, "fails_at_zero": False}


def _ci(low, high):
    return {KEY: {"difference": (low + high) / 2.0, "ci_low": low, "ci_high": high}}


def _case(b, d, f, bd, df, fg):
    boundaries = {name: _boundary("value", 0.1) for name in CONDITIONS}
    boundaries.update({"B": b, "D": d, "F": f, "E": _boundary("zero")})
    comparisons = {"B-D": _ci(*bd), "D-F": _ci(*df), "F-G": _ci(*fg)}
    return _verdicts(boundaries, comparisons)


def test_constant_difference_has_a_zero_width_interval():
    indices = _bootstrap_indices(12)
    out = _difference(np.ones(12, dtype=bool), np.zeros(12, dtype=bool), indices)
    assert out == {"difference": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_bootstrap_keeps_pairs_together():
    """Left and right fail on the same seeds: every resample gives exactly 0. A bootstrap
    that resampled the two sides separately would give a nonzero spread."""
    flags = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    out = _difference(flags, flags.copy(), _bootstrap_indices(10))
    assert out == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_bootstrap_interval_matches_a_hand_count():
    """5 seeds where left fails and right passes, 15 where both pass: the mean difference
    is 5/20 = 0.25 and the 95 % interval must contain it and stay inside [0, 1]."""
    left = np.array([1] * 5 + [0] * 15, dtype=bool)
    out = _difference(left, np.zeros(20, dtype=bool), _bootstrap_indices(20))
    assert out["difference"] == 0.25
    assert 0.0 <= out["ci_low"] < 0.25 < out["ci_high"] <= 1.0
    # Binomial standard error sqrt(0.25 * 0.75 / 20) = 0.097, so the interval is about +-0.19.
    assert 0.3 < out["ci_high"] - out["ci_low"] < 0.5


@pytest.mark.parametrize("b,d,bd,expected", [
    (_boundary("value", 0.2), _boundary("censored"), (0.1, 0.5), "YES"),
    (_boundary("value", 0.2), _boundary("value", 0.3), (0.01, 0.2), "YES"),
    (_boundary("value", 0.2), _boundary("value", 0.2), (-0.1, 0.1), "NO"),
    (_boundary("value", 0.2), _boundary("value", 0.15), (-0.2, 0.0), "NO"),
    (_boundary("value", 0.2), _boundary("value", 0.2), (0.1, 0.3), "MIXED"),
    (_boundary("value", 0.2), _boundary("censored"), (-0.1, 0.1), "MIXED"),
    (_boundary("zero"), _boundary("value", 0.0), (0.2, 0.4), "YES"),
    (_boundary("censored"), _boundary("censored"), (0.1, 0.2), "MIXED"),
])
def test_q1_rule(b, d, bd, expected):
    out = _case(b, d, _boundary("value", 0.1), bd, (-0.1, 0.1), (-0.1, 0.1))
    assert out["q1_cooperation_beyond_filter"]["verdict"] == expected


@pytest.mark.parametrize("d,f,df,fg,expected,noise_only", [
    (_boundary("censored"), _boundary("value", 0.2), (0.1, 0.4), (-0.1, 0.1), "COLLAPSES", True),
    (_boundary("censored"), _boundary("value", 0.2), (0.1, 0.4), (-0.3, -0.05), "COLLAPSES", False),
    (_boundary("censored"), _boundary("censored"), (-0.05, 0.05), (0.0, 0.0), "SURVIVES", None),
    (_boundary("value", 0.3), _boundary("value", 0.4), (-0.3, -0.1), (0.0, 0.0), "SURVIVES", None),
    (_boundary("censored"), _boundary("value", 0.4), (-0.05, 0.05), (0.0, 0.0), "MIXED", None),
    (_boundary("value", 0.0), _boundary("zero"), (0.0, 0.2), (0.0, 0.0), "MIXED", None),
])
def test_q2_rule(d, f, df, fg, expected, noise_only):
    out = _case(_boundary("value", 0.2), d, f, (0.0, 0.1), df, fg)
    q2 = out["q2_gain_survives_shared_fading"]
    assert q2["verdict"] == expected
    assert q2["only_noise_averaging_left"] is noise_only


def test_checks_flag_a_shared_seed_whose_losses_differ():
    rows = [{"condition": "F", "steps": 120, "all_lost": 30, "o1_lost": 30},
            {"condition": "D", "steps": 120, "all_lost": 3, "o1_lost": 36}]
    covering = {"I-G": {"0": {"ci_low": -0.1, "ci_high": 0.1}, "0.3": {"ci_low": -0.1, "ci_high": 0.1}}}
    good = _checks(rows, covering, (0.0, 0.3))
    assert good["passed"] and good["shared_all_lost_fraction"] == 0.25
    assert good["independent_all_lost_fraction"] == 3 / 120
    rows[0]["all_lost"] = 29
    assert not _checks(rows, covering, (0.0, 0.3))["passed"]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pilot_is_byte_identical_with_one_or_two_processes(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    main(["--pilot", "--processes", "1", "--output", str(one)])
    main(["--pilot", "--processes", "2", "--output", str(two)])
    m1 = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((two / "manifest.json").read_text(encoding="utf-8"))
    volatile = set(m1["volatile_fields"])
    assert {k: v for k, v in m1.items() if k not in volatile} == {k: v for k, v in m2.items() if k not in volatile}
    for name in m1["outputs"]:
        assert _digest(one / name) == _digest(two / name)
    assert "sim/estimation/ct.py" in m1["code_sha256"]
