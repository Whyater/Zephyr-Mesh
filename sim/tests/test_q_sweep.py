"""Tests for the EXP-040 process-noise (q) sweep (proposal, simulated)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.q_sweep import Q_GRID, REGISTERED_Q, simulate_q_condition
from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list

CONFIG = RecoveryConfig()
SEED = seed_list(1)[0]


def test_grid_is_half_decades_and_contains_registered_q_exactly():
    assert len(Q_GRID) == 7
    assert REGISTERED_Q == 0.1 and REGISTERED_Q in Q_GRID
    assert Q_GRID[0] == 0.01 and Q_GRID[-1] == 10.0
    steps = np.diff(np.log10(Q_GRID))
    assert np.allclose(steps, 0.5, atol=1e-12)


@pytest.mark.parametrize("motion,delay,loss,model", [
    ("constant_velocity", 0.2, 0.0, "independent"),
    ("constant_turn", 0.3, 0.0, "independent"),
    ("constant_velocity", 0.0, 0.3, "burst"),
])
def test_registered_q_matches_the_registered_recovery_arm_bit_for_bit(motion, delay, loss, model):
    out = simulate_q_condition(motion, delay_s=delay, loss_probability=loss, noise_std_m=0.05, loss_model=model,
                               seed=SEED, config=CONFIG, q_values=(REGISTERED_Q,), told_stds={"told_5cm": 0.05})
    deliveries = deliver(motion, delay_s=delay, loss_probability=loss, noise_std_m=0.05, seed=SEED, config=CONFIG,
                         loss_model=model)
    reference = run_arms(deliveries, motion, CONFIG)["recovery"]
    got = out[(REGISTERED_Q, "told_5cm")]
    assert repr(got.p95_error_m) == repr(reference.p95_error_m)
    assert repr(got.max_error_m) == repr(reference.max_error_m)
    assert np.array_equal(got.errors_m, reference.errors_m)


def test_q_reaches_the_tracker_and_changes_the_answer():
    out = simulate_q_condition("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.05,
                               loss_model="independent", seed=SEED, config=CONFIG, q_values=(0.01, 0.1, 10.0),
                               told_stds={"told_5cm": 0.05})
    p = [out[(q, "told_5cm")].p95_error_m for q in (0.01, 0.1, 10.0)]
    assert len(set(p)) == 3
    # Same q through the plain recovery path with a replaced config gives the same answer.
    deliveries = deliver("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.05, seed=SEED,
                         config=CONFIG)
    ref = run_arms(deliveries, "constant_velocity", replace(CONFIG, tracker_acceleration_psd=0.01))["recovery"]
    assert repr(out[(0.01, "told_5cm")].p95_error_m) == repr(ref.p95_error_m)


def test_straight_line_without_delay_prefers_less_motion_noise_than_ten():
    # Limiting case: the target never accelerates, so q = 10 (velocity may wander about 3 m/s per
    # square-root second) must chase noise far more than q = 0.01.
    out = simulate_q_condition("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.10,
                               loss_model="independent", seed=SEED, config=CONFIG, q_values=(0.01, 10.0),
                               told_stds={"told_5cm": 0.05})
    assert out[(0.01, "told_5cm")].p95_error_m < 0.5 * out[(10.0, "told_5cm")].p95_error_m
