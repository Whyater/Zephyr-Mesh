"""Tests for the EXP-032 runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.e_exp032_relay import (PREDICTION_CORRECTION_3, PREDICTION_TEXT,
                                  build_jobs, run_experiment)
from tests.reference_e.hand_checks import max_range_dual_slope_m


def test_grid_builder_matches_preregistration():
    # 6 distances x 6 fractions x 50 point-mass seeds + 2 x 2 x 20 physics seeds = 1880.
    jobs, seeds = build_jobs()
    assert len(seeds) == 50
    assert len(jobs) == 1880
    assert jobs[0]["labels"] == {"adapter": "point_mass", "distance_m": 300.0, "relay_fraction": "none"}
    assert any(j["labels"] == {"adapter": "point_mass", "distance_m": 1300.0, "relay_fraction": 0.75} for j in jobs)
    assert any(j["labels"] == {"adapter": "physics", "distance_m": 900.0, "relay_fraction": 0.5} for j in jobs)


def test_pilot_is_small_and_hand_check_is_reference():
    jobs, seeds = build_jobs(pilot=True)
    assert len(seeds) == 3
    assert len(jobs) == 14  # 4 point-mass cells x 3 + 1 physics cell x 2.
    breakpoint_m = 4.0 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)
    reference = max_range_dual_slope_m(13.5, -98.0, 40.2, 2.0, breakpoint_m, 4.0, 0.0)
    assert 517.0 < reference < 519.0


def test_tiny_run_writes_byte_stable_outputs(tmp_path):
    first = run_experiment(tmp_path / "one", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    second = run_experiment(tmp_path / "two", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    assert first["missing_metrics"]
    for name in ("relay.csv", "summary.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    summary = json.loads((tmp_path / "one" / "summary.json").read_text(encoding="utf-8"))
    assert summary["hand_check"]["reference_median_signal_range_m"] > 500.0
    assert (tmp_path / "one" / "delivery_vs_distance.png").exists()
    assert (tmp_path / "one" / "latency_vs_distance.png").exists()


def test_prediction_constants_match_preregistration():
    prereg = Path("/Users/akashi0281/.zm-worktrees/run-e-briefs/prereg.md")
    if not prereg.exists():
        pytest.skip("prereg.md is outside the repository and is unavailable")
    text = prereg.read_text(encoding="utf-8")
    assert PREDICTION_TEXT in text
    assert PREDICTION_CORRECTION_3 in text
