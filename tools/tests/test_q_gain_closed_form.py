"""Independent check: the tracker's steady gain against the closed-form alpha-beta relation.

``tools/cwna_gain_check.py`` was written separately from the simulator (it imports
nothing from ``sim``). For a constant-velocity filter with continuous white
acceleration noise q and measurement variance R, its steady gain [alpha, beta/T]
solves beta^2 = lambda^2 (1 - alpha) and alpha^2 + alpha beta - 2 beta + beta^2 / 6 = 0,
with lambda^2 = q T^3 / R.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.adaptive_r import make_tracker, run_recovery_arm
from sim.q_sweep import Q_GRID
from sim.s6_recovery import RecoveryConfig, deliver, seed_list
from sim.tracker import ConstantVelocityTracker
from tools.cwna_gain_check import riccati_gain, steady_gain

T = 0.1


def _gain_from_tracker(tracker: ConstantVelocityTracker, axis: int = 0) -> tuple[float, float]:
    """Gain the tracker would apply to the next measurement one period later, as (alpha, beta)."""
    P = np.asarray(tracker.predict_at(tracker.timestamp_s + T).covariance)
    s = P[axis, axis] + tracker.R[axis, axis]
    return P[axis, axis] / s, P[axis, axis + 3] / s * T


@pytest.mark.parametrize("q", Q_GRID)
@pytest.mark.parametrize("std", [0.02, 0.05, 0.10])
def test_tracker_converges_to_the_closed_form_gain(q, std):
    tracker = ConstantVelocityTracker(acceleration_spectral_density=q, measurement_std_m=std)
    for k in range(1, 801):
        tracker.update((0.0, 0.0, 0.0), k * T)
    alpha, beta = _gain_from_tracker(tracker)
    a_ref, b_ref = steady_gain(q, std**2, T)
    assert alpha == pytest.approx(a_ref, rel=1e-9, abs=1e-12)
    assert beta == pytest.approx(b_ref, rel=1e-9, abs=1e-12)


def test_the_two_reference_methods_agree():
    for q in Q_GRID:
        a1, b1 = steady_gain(q, 0.05**2, T)
        a2, b2 = riccati_gain(q, 0.05**2, T)
        assert abs(a1 - a2) < 1e-9 and abs(b1 - b2) < 1e-9


def test_gain_on_one_sweep_cell_matches_the_closed_form():
    # Straight line, 5 cm noise, 0.3 s delay, no loss, registered q = 0.1: the gain the
    # recovery arm ends the 12 s flight with is the steady gain.
    config = RecoveryConfig()
    seed = seed_list(1)[0]
    deliveries = deliver("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.05, seed=seed,
                         config=config)
    for q in (0.01, 0.1, 10.0):
        q_config = replace(config, tracker_acceleration_psd=q)
        tracker = make_tracker("constant_velocity", q_config, r_mode="told", measurement_std_m=0.05)
        run_recovery_arm(deliveries, "constant_velocity", q_config, tracker)
        alpha, beta = _gain_from_tracker(tracker)
        a_ref, b_ref = steady_gain(q, 0.05**2, T)
        # q = 0.01 settles slowest; 117 updates still leave it within 1e-6.
        assert alpha == pytest.approx(a_ref, rel=1e-6)
        assert beta == pytest.approx(b_ref, rel=1e-6)
