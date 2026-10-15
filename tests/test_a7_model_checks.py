"""Fast checks for the reproducible disturbance model hand checks."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import tools.a_make_disturbance_model_checks as checks

ROOT = Path(__file__).resolve().parents[1]


def test_ou_small_run_passes_three_standard_errors():
    result = checks.ou_checks(sigmas=(0.5,), n=20_000, seeds=(17,))
    cell = result["cells"][0]
    # n_eff = N dt / (2 T) = 20000 * 0.01 / 2 = 100 independent samples.
    assert cell["n_eff"] == pytest.approx(100.0)
    assert result["pass"]


def test_ground_effect_ratio_is_exact_at_one_two_and_four_radii():
    result = checks.ground_effect_checks()
    assert result["pass"]
    by_height = {(c["profile"], c["height_R"]): c["value"] for c in result["cells"]}
    # Written out by hand: R / 4z = 1/4, 1/8, 1/16.
    for profile in checks.PROFILES:
        assert by_height[(profile, 1.0)] == pytest.approx(16.0 / 15.0, abs=1e-15)
        assert by_height[(profile, 2.0)] == pytest.approx(64.0 / 63.0, abs=1e-15)
        assert by_height[(profile, 4.0)] == pytest.approx(256.0 / 255.0, abs=1e-15)


def test_downwash_far_wake_matches_momentum_theory_by_hand():
    result = checks.downwash_checks()
    assert result["pass"]
    cell = next(c for c in result["cells"] if c["profile"] == "crazyflie-class")
    # T = 0.028 * 9.81 N, A = 4 pi (0.0225 m)^2, v_h = sqrt(T / (2 rho A)) is about 4.198 m/s.
    v_h = math.sqrt(0.028 * 9.81 / (2.0 * 1.225 * 4.0 * math.pi * 0.0225**2))
    assert cell["far_wake_mps"] == pytest.approx(2.0 * v_h, rel=1e-3)


def test_isotropic_gust_force_ratio_is_reproducible_and_below_one_axis_formula():
    first = checks._gust_monte_carlo(0.5, 123, 200_000)
    second = checks._gust_monte_carlo(0.5, 123, 200_000)
    assert first == second
    # Cross-wind axes only see W times the gust, not 2 W, so the 3-D ratio sits well
    # below the one-axis sqrt(1 + 2 W^2 / sigma^2) = sqrt(19), about 4.36.
    assert first["one_axis_formula"] == pytest.approx(math.sqrt(19.0))
    assert 2.0 < first["rms_ratio_unweighted"] < 2.3
    assert 2.0 < first["rms_ratio_weighted"] < 2.4


def test_gust_trajectory_reproduces_committed_run_rms():
    committed = ROOT / "runs/a7-disturbance/per_run.csv"
    if not committed.is_file():
        pytest.skip("committed run not present")
    with committed.open(newline="", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle)
                   if r.get("part") == "b" and r["profile"] == "crazyflie-class" and r.get("seed") == "0"
                   and float(r["sigma_mps"]) == 0.25 and float(r["mean_wind_mps"]) == 1.5)
    result = checks._gust_trajectory(("crazyflie-class", 0.25, 1.5, 0))
    assert result["plain_rms_m"] == pytest.approx(float(row["rms_error_m"]), abs=1e-9)
    # The offset-removed RMS is smaller than the plain RMS because the steady push is taken out.
    assert result["rms_about_mean_m"] < result["plain_rms_m"]


def test_refuses_nonempty_output_directory(tmp_path: Path):
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        checks.main(["--output-dir", str(output)])
