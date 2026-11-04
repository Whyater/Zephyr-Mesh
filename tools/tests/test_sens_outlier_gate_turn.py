"""Tests for the turn-aware outlier and gate study (tools/sens_outlier_gate_turn.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import sens_outlier_gate as parent
from tools import sens_outlier_gate_turn as turn

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("p95_error_m", "max_error_m", "rmse_m", "failed", "fused_count", "gated_count",
        "gated_outliers", "gated_clean", "forced_accepts")


def test_cv_arms_equal_the_parent_study_exactly():
    """Control: same packets, same outliers, so the two CV arms must match the parent bit for bit."""
    spec = parent.OutlierSpec("jump", 0.03)
    seeds = [11, 12]
    mine = turn.cell_runs("constant_turn", 0.2, spec, seeds, arms=turn.ARMS[:2])
    theirs = parent.cell_runs("constant_turn", 0.2, spec, seeds)
    assert [r["arm"] for r in mine] == [r["arm"] for r in theirs]
    for a, b in zip(mine, theirs):
        for key in KEYS:
            assert a[key] == b[key], (a["arm"], a["seed"], key)


def test_gated_turn_arms_equal_the_atlas_specs():
    """The gated CT and IMM arms are the ones the S6 atlas registered (EXP-012), not new tunings."""
    atlas = json.loads((ROOT / "sim" / "runs" / "s6-atlas" / "atlas.json").read_text())
    registered = {a["name"]: a for a in atlas["arms"] if a["name"] in ("ct_gated", "imm_gated", "recovery_gated")}
    assert set(registered) == {"ct_gated", "imm_gated", "recovery_gated"}
    for arm in turn.ARMS:
        if arm.name in registered:
            assert arm.as_dict() == registered[arm.name]


def test_each_gated_arm_differs_from_its_twin_only_by_the_gate():
    arms = {a.name: a.as_dict() for a in turn.ARMS}
    for plain, gated in turn.PAIRS:
        a, b = dict(arms[plain]), dict(arms[gated])
        assert b.pop("gate_probability") == 0.99 and a.pop("gate_probability") is None
        a.pop("name"), b.pop("name")
        assert a == b


def test_predicted_reject_rate_hand_values():
    """Independent: a consistent filter (mean NIS 3) rejects exactly 1 %; the line CV filter's
    mean NIS of 2.51 gives 0.36 %, worked out by hand as P(chi2_3 > 11.34 x 3 / 2.51 = 13.55)."""
    assert turn.predicted_clean_reject_rate(3.0) == pytest.approx(0.01, rel=1e-9)
    assert turn.predicted_clean_reject_rate(2.51) == pytest.approx(0.0036, abs=0.0002)
    assert turn.predicted_clean_reject_rate(3.50) == pytest.approx(0.021, abs=0.001)


def test_plan_is_the_preregistered_grid():
    jobs = turn.plan()
    assert len(jobs) == 2 * 3 * 3
    assert {s.kind for _, _, s in jobs} == {"jump"}
    assert sorted({s.probability for _, _, s in jobs}) == [0.0, 0.01, 0.03]


def test_refuses_to_write_under_sim_runs(tmp_path):
    with pytest.raises(SystemExit, match="refusing"):
        turn.main(["--output", str(ROOT / "sim" / "runs" / "nope"), "--seeds", "1"])
