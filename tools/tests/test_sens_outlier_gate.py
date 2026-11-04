"""Checks for the outlier sensitivity wrapper (tools/sens_outlier_gate.py).

The control tests come first: with no outliers and no gate the wrapper must
give the engine's own numbers bit for bit, and the registered S6 recovery
rows within the repo's platform rule. The hand checks after that do not reuse
the wrapper's arithmetic: the one-outlier response is compared with a
separate steady-state Kalman gain, and the episode chain with its long-run
share and mean length.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from sim.estimation.base import RECOVERY_V1
from sim.estimation.engine import deliver_cell, run_arm, run_arm_detailed, s6_cell
from sim.s6_recovery import seed_list
from sim.tests.platform_compare import same_text
from tools import sens_outlier_gate as sog

ROOT = Path(__file__).resolve().parents[2]
SEEDS = seed_list(50)


def test_control_matches_engine_bit_for_bit():
    for motion, delay in (("constant_velocity", 0.1), ("constant_turn", 0.2)):
        cell = s6_cell(motion, delay, 0.0, 0.05)
        rows = sog.cell_runs(motion, delay, sog.OutlierSpec("jump", 0.0), SEEDS[:4])
        for seed, row in zip(SEEDS[:4], [r for r in rows if r["arm"] == "recovery"]):
            ref = run_arm(deliver_cell(cell, seed), cell, RECOVERY_V1)
            assert row["p95_error_m"] == ref.p95_error_m
            assert row["max_error_m"] == ref.max_error_m
            assert row["rmse_m"] == ref.rmse_m
            assert row["fused_count"] == ref.fused_count
            assert row["outlier_count"] == 0


def test_control_matches_registered_s6_recovery_rows():
    path = ROOT / "sim/runs/s6-recovery/per_seed.csv"
    with path.open(newline="") as handle:
        registered = [r for r in csv.DictReader(handle)
                      if r["slice"] == "main" and r["arm"] == "recovery" and r["motion"] == "constant_velocity"
                      and r["delay_s"] == "0.1" and r["loss_probability"] == "0.0" and r["noise_std_m"] == "0.05"
                      and r["loss_model"] == "independent" and r["jitter_s"] == "0.0"]
    registered.sort(key=lambda r: int(r["seed_index"]))
    assert len(registered) == 50
    rows = [r for r in sog.cell_runs("constant_velocity", 0.1, sog.OutlierSpec("jump", 0.0), SEEDS[:6])
            if r["arm"] == "recovery"]
    for row, ref in zip(rows, registered):
        assert row["seed"] == int(ref["seed"])
        for key in ("p95_error_m", "max_error_m", "rmse_m", "mean_nis"):
            assert same_text(repr(row[key]), ref[key]), key
        assert row["fused_count"] == int(ref["fused_count"])
        assert str(row["failed"]).lower() == ref["failed"]


def test_raising_p_only_adds_outliers_and_offsets_have_the_set_size():
    n = 121
    low, _ = sog.outlier_frames(n, sog.OutlierSpec("jump", 0.01), 7)
    high, offsets = sog.outlier_frames(n, sog.OutlierSpec("jump", 0.30), 7)
    assert np.all(high[low])  # every 1 % outlier is also a 30 % outlier
    assert high.sum() > low.sum()
    assert np.allclose(np.linalg.norm(offsets[high], axis=1), 1.0)
    assert np.all(offsets[~high] == 0.0)
    hit, flip = sog.outlier_frames(n, sog.OutlierSpec("zflip", 0.30), 7)
    assert np.array_equal(hit, high)  # same coins for both kinds
    assert np.all(flip[hit] == [0.0, 0.0, 1.0])


def test_episode_chain_long_run_share_and_mean_length():
    """Hand rule: a two-state chain with exit chance e and entry chance s spends s / (s + e)
    of its frames inside and its runs last 1 / e frames on average."""
    n = 400_000
    hit, _ = sog.outlier_frames(n, sog.OutlierSpec("zflip", 0.05, episode_mean_frames=5.0), 3)
    share = hit.mean()
    edges = np.diff(np.r_[0, hit.astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert abs(share - 0.05) < 0.004  # runs are correlated, so the band is wider than binomial
    # Run lengths are geometric with exit chance e = 0.2: std sqrt(1 - e) / e = 4.47 frames.
    standard_error = np.sqrt(1 - 0.2) / 0.2 / np.sqrt(lengths.size)
    assert abs(lengths.mean() - 5.0) < 4 * standard_error


def _riccati_gain(T=0.1, q=0.1, r=0.05**2):
    F = np.array([[1.0, T], [0.0, 1.0]])
    Q = q * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
    P = np.eye(2)
    for _ in range(2000):
        P = F @ P @ F.T + Q
        k = P[:, 0] / (P[0, 0] + r)
        P = P - np.outer(k, P[0, :])
    return k


def test_one_outlier_response_follows_the_hand_gain():
    """The filter is linear, so the gap between a run with one 1 m outlier and the same run
    without it is the filter's response to a unit kick: K_pos at the fusion, then the
    1D recursion e <- (1 - K_pos)(e + T v), v <- v - K_vel (e + T v)."""
    k_pos, k_vel = _riccati_gain()
    assert k_pos == pytest.approx(0.4687, abs=1e-3)
    # Noise-free cell: the clean error is about 5e-6 m, so the 3D error gap is the kick response.
    cell = s6_cell("constant_velocity", 0.0, 0.0, 0.0)
    clean = deliver_cell(cell, SEEDS[0])
    target_seq = 60  # t = 6.0 s, long after the start-up transient
    deliveries = list(clean.deliveries)
    i = next(j for j, d in enumerate(deliveries) if d.seq == target_seq)
    z = np.asarray(deliveries[i].measurement_m) + [1.0, 0.0, 0.0]
    deliveries[i] = sog.dataclasses.replace(deliveries[i], measurement_m=tuple(z))
    kicked = sog.dataclasses.replace(clean, deliveries=deliveries)
    a = run_arm_detailed(clean, cell, RECOVERY_V1).outcome
    b = run_arm_detailed(kicked, cell, RECOVERY_V1).outcome
    assert float(np.max(a.errors_m[a.query_times_s > 2.0])) < 1e-4
    e, v, expected = k_pos, k_vel, []
    for _ in range(4):
        expected.append(e)
        e = e + 0.1 * v  # predict the gap one packet ahead
        nu = -e  # the clean packet sees the gap as an innovation
        e, v = e + k_pos * nu, v + k_vel * nu
    assert expected[:3] == pytest.approx([0.469, 0.326, 0.203], abs=2e-3)
    for step, want in enumerate(expected):
        qi = int(np.argmin(np.abs(b.query_times_s - (6.0 + 0.1 * step))))
        assert abs(b.errors_m[qi]) == pytest.approx(abs(want), abs=1e-3), (step, b.errors_m[qi], want)


def test_gate_rejects_isolated_outliers_and_threshold_is_the_chi2_point():
    assert chi2.ppf(sog.GATE_PROBABILITY, 3) == pytest.approx(11.3449, abs=1e-4)
    rows = sog.cell_runs("constant_velocity", 0.0, sog.OutlierSpec("jump", 0.03), SEEDS[:5])
    gated = [r for r in rows if r["arm"] == "recovery_gated"]
    assert sum(r["checked_outliers"] for r in gated) > 0
    assert sum(r["gated_outliers"] for r in gated) == sum(r["checked_outliers"] for r in gated)


def test_refuses_to_write_under_sim_runs(tmp_path):
    with pytest.raises(SystemExit, match="sim/runs"):
        sog.main(["--output", str(ROOT / "sim" / "runs" / "nope"), "--seeds", "1"])
