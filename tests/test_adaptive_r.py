"""Checks for the opt-in adaptive measurement-noise tracker (EXP-037 proposal).

The two convergence checks do not trust the tracker's own matrix code: the
expected long-run estimate is worked out here with a separately written
two-state (position, velocity) filter in plain Python floats.
"""
from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from sim.adaptive_r import (
    AdaptiveRTracker,
    make_tracker,
    run_recovery_arm,
    simulate_condition,
)
from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list
from sim.tracker import ConstantVelocityTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
DT = 0.1


def _straight_line_run(noise_std, q, *, steps, seed, window=20, floor_std=0.001, start_std=0.05):
    """Feed a 1 m/s straight-line target with Gaussian noise into the adaptive tracker."""
    rng = np.random.default_rng(seed)
    tracker = AdaptiveRTracker(position_m=(0.0, 0.0, 0.0), acceleration_spectral_density=q,
                               measurement_std_m=start_std, window=window, floor_std_m=floor_std)
    estimates = []
    for k in range(1, steps + 1):
        t = k * DT
        z = np.array([1.0 * t, 0.0, 0.0]) + rng.normal(0.0, noise_std, size=3)
        tracker.update(z, t)
        estimates.append(tracker.R[0, 0])
    return tracker, np.asarray(estimates)


def _expected_r_hat(true_variance, q, *, floor_variance=1e-6, iterations=60):
    """Long-run covariance-matching estimate on a straight line, from a hand-written 2-state filter.

    Per axis the filter tracks position and velocity. Its own error bars come
    from its process noise q and its current noise guess r. The true target has
    no acceleration, so the real error of that same filter (same gain) only
    comes from measurement noise. The estimate settles where
    r = (real predicted variance + true noise) - (filter's predicted variance).
    """
    def filter_steady(r):
        p00, p01, p11 = 1.0, 0.0, 1.0
        for _ in range(4000):
            # predict with the filter's own process noise
            m00 = p00 + 2 * DT * p01 + DT * DT * p11 + q * DT ** 3 / 3
            m01 = p01 + DT * p11 + q * DT ** 2 / 2
            m11 = p11 + q * DT
            s = m00 + r
            k0, k1 = m00 / s, m01 / s
            # Joseph form with H = [1, 0]
            p00 = (1 - k0) ** 2 * m00 + k0 * k0 * r
            p01 = (1 - k0) * (m01 - k1 * m00) + k0 * k1 * r
            p11 = m11 - 2 * k1 * m01 + k1 * k1 * m00 + k1 * k1 * r
        return m00, k0, k1

    def real_predicted_variance(k0, k1):
        e00, e01, e11 = 0.0, 0.0, 0.0
        for _ in range(4000):
            # the real target never accelerates, so no process noise here
            n00 = e00 + 2 * DT * e01 + DT * DT * e11
            n01 = e01 + DT * e11
            n11 = e11
            e00 = (1 - k0) ** 2 * n00 + k0 * k0 * true_variance
            e01 = (1 - k0) * (n01 - k1 * n00) + k0 * k1 * true_variance
            e11 = n11 - 2 * k1 * n01 + k1 * k1 * n00 + k1 * k1 * true_variance
        return n00

    def gap(r):
        filter_m00, k0, k1 = filter_steady(r)
        return real_predicted_variance(k0, k1) + true_variance - filter_m00 - r

    # gap(r) falls as r grows and changes sign once, so bisect for the root
    # (plain repeated substitution overshoots when the slope is steeper than -1).
    low, high = floor_variance, 1.0
    if gap(low) <= 0:
        return floor_variance
    for _ in range(iterations):
        middle = (low + high) / 2
        if gap(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


# ---- the tracker itself --------------------------------------------------------

def test_matches_the_plain_tracker_until_the_window_fills():
    plain = ConstantVelocityTracker(measurement_std_m=0.05)
    adaptive = AdaptiveRTracker(measurement_std_m=0.05, window=5)
    rng = np.random.default_rng(1)
    for k in range(1, 5):
        z = rng.normal(0.0, 0.1, size=3)
        plain.update(z, k * DT)
        adaptive.update(z, k * DT)
        assert np.array_equal(plain.x, adaptive.x)
        assert np.array_equal(plain.P, adaptive.P)
        assert adaptive.r_history == []
    # The fifth update fills the window, and (as in Mohamed 1999) that same
    # update is already fused with the new estimate.
    z = rng.normal(0.0, 0.1, size=3)
    plain.update(z, 0.5)
    adaptive.update(z, 0.5)
    assert adaptive.R[0, 0] != plain.R[0, 0]
    assert len(adaptive.r_history) == 1
    assert not np.array_equal(plain.x, adaptive.x)


@pytest.mark.parametrize("noise_std", [0.02, 0.05, 0.10])
def test_converges_to_the_true_noise_when_the_motion_model_is_right(noise_std):
    """Independent check 1: straight line, Gaussian noise, a filter that expects almost no acceleration.

    With q tiny the filter's own error bars are right, so the estimate should
    settle on the true variance. Over 2000 updates after a 1000-update settle,
    a 20-sample window (60 squared numbers) is about 18 % noisy one window at
    a time, but the long-run average must land within 5 % of the truth.
    """
    _, estimates = _straight_line_run(noise_std, 1e-6, steps=3000, seed=7)
    reference = _expected_r_hat(noise_std ** 2, 1e-6)
    assert reference == pytest.approx(noise_std ** 2, rel=0.03)  # the hand model agrees: almost no bias here
    assert np.mean(estimates[1000:]) == pytest.approx(noise_std ** 2, rel=0.05)


@pytest.mark.parametrize("noise_std", [0.02, 0.10])
def test_matches_the_hand_model_with_the_sweep_process_noise(noise_std):
    """Independent check 2: with the sweep's q = 0.1 the filter's error bars are too wide on a straight line.

    That makes the estimate come out low. The hand-written 2-state model
    predicts by how much; the tracker's long-run average must agree within 5 %.
    """
    runs = [_straight_line_run(noise_std, 0.1, steps=3000, seed=s)[1][1000:] for s in (11, 12, 13)]
    expected = _expected_r_hat(noise_std ** 2, 0.1)
    assert expected < noise_std ** 2  # a known low bias, not a match to the truth
    assert np.mean(np.concatenate(runs)) == pytest.approx(expected, rel=0.05)


def test_limiting_case_zero_noise_sits_on_the_floor():
    """Limiting case: a perfect sensor on a perfectly straight line gives the floor exactly."""
    tracker, estimates = _straight_line_run(0.0, 1e-6, steps=400, seed=3, floor_std=0.001)
    assert estimates[-1] == 0.001 ** 2
    assert tracker.estimated_std_m == pytest.approx(0.001)


def test_limiting_case_huge_starting_uncertainty_does_not_go_negative():
    """Early on the filter is very unsure, so innovations minus expectation is negative: floor, never below."""
    tracker = AdaptiveRTracker(position_variance_m2=100.0, velocity_variance_m2ps2=100.0,
                               measurement_std_m=0.05, window=2, floor_std_m=0.002)
    tracker.update((0.0, 0.0, 0.0), 0.1)
    tracker.update((0.1, 0.0, 0.0), 0.2)
    assert tracker.R[0, 0] == pytest.approx(0.002 ** 2)
    assert np.all(np.linalg.eigvalsh(tracker.R) > 0)


def _scored_mean_std(motion, noise_std, seed):
    config = RecoveryConfig()
    result = simulate_condition(motion, delay_s=0.0, loss_probability=0.0, noise_std_m=noise_std,
                                loss_model="independent", seed=seed, config=config,
                                arms={"adaptive": ("adaptive", 0.05)})
    return result["adaptive"].mean_scored_std_m


def test_known_failure_mode_a_turn_looks_like_noise():
    """The constant-velocity model lags a 1 m turn, and the estimator reads the lag as extra sensor noise.

    Same 5 cm noise and seeds: on a straight line the estimate sits a little
    under 5 cm (the q bias above); on the turn it sits above 5 cm.
    """
    seeds = seed_list(4)
    straight = np.mean([_scored_mean_std("constant_velocity", 0.05, s) for s in seeds])
    turn = np.mean([_scored_mean_std("constant_turn", 0.05, s) for s in seeds])
    assert straight < 0.05 < turn
    assert turn > 1.15 * straight


def test_limiting_case_perfect_sensor_on_a_turn_stays_on_the_floor():
    """With no noise the filter trusts each point, follows the turn closely, and nothing inflates."""
    assert _scored_mean_std("constant_turn", 0.0, seed_list(1)[0]) == pytest.approx(0.001)


@pytest.mark.parametrize("kwargs", [{"window": 1}, {"window": 2.5}, {"window": True},
                                    {"floor_std_m": 0.0}, {"floor_std_m": -1.0},
                                    {"floor_std_m": float("nan")}])
def test_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        AdaptiveRTracker(**kwargs)


def test_a_rejected_measurement_leaves_the_window_alone():
    tracker = AdaptiveRTracker(window=3)
    tracker.update((0.0, 0.0, 0.0), 0.2)
    before = (list(tracker._surprise), list(tracker._expected), tracker.R.copy())
    with pytest.raises(ValueError):
        tracker.update((0.0, 0.0, 0.0), 0.1)  # older than the filter clock
    with pytest.raises(ValueError):
        tracker.update((0.0, 0.0), 0.3)  # wrong shape
    assert (list(tracker._surprise), list(tracker._expected)) == before[:2]
    assert np.array_equal(tracker.R, before[2])


def test_an_overflowing_measurement_is_refused_without_touching_the_filter():
    tracker = AdaptiveRTracker(window=2)
    tracker.update((0.0, 0.0, 0.0), 0.1)
    state = (tracker.x.copy(), tracker.P.copy(), tracker.timestamp_s, tracker.update_count, tracker.R.copy(),
             list(tracker._surprise))
    with pytest.raises(ValueError):
        tracker.update((1e200, 0.0, 0.0), 0.2)  # finite, but its square overflows
    assert np.array_equal(tracker.x, state[0]) and np.array_equal(tracker.P, state[1])
    assert (tracker.timestamp_s, tracker.update_count) == state[2:4]
    assert np.array_equal(tracker.R, state[4]) and list(tracker._surprise) == state[5]


# ---- the runner reproduces the registered S6 recovery numbers -------------------

@pytest.mark.parametrize("motion,delay,loss,noise,loss_model,told", [
    ("constant_velocity", 0.3, 0.0, 0.10, "independent", 0.05),
    ("constant_velocity", 0.3, 0.0, 0.10, "independent", 0.10),
    ("constant_turn", 0.2, 0.3, 0.05, "burst", 0.05),
    ("constant_turn", 0.0, 0.3, 0.10, "independent", 0.10),
])
def test_told_arm_is_bit_identical_to_the_s6_recovery_arm(motion, delay, loss, noise, loss_model, told):
    config = RecoveryConfig()
    for seed in seed_list(3):
        deliveries = deliver(motion, delay_s=delay, loss_probability=loss, noise_std_m=noise, seed=seed,
                             config=config, loss_model=loss_model)
        reference = run_arms(deliveries, motion, config, tracker_measurement_std_m=told)["recovery"]
        ours = run_recovery_arm(deliveries, motion, config,
                                make_tracker(motion, config, r_mode="told", measurement_std_m=told))
        for field in ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count",
                      "fused_out_of_order_count", "failed"):
            assert getattr(ours, field) == getattr(reference, field), field
        assert np.array_equal(ours.query_times_s, reference.query_times_s)
        assert np.array_equal(ours.errors_m, reference.errors_m)


def test_told_arms_reproduce_the_registered_per_seed_rows():
    """Tie to the registered run: the told-5 cm and told-10 cm arms give the stored S6 numbers exactly."""
    wanted = {
        "main/constant_velocity/d0.300/l0.00/n0.100/independent/j0.000/r0.050": 0.05,
        "oracle/constant_velocity/d0.300/l0.00/n0.100/independent/j0.000/r0.100": 0.10,
    }
    stored = {}
    with (REPO_ROOT / "sim/runs/s6-recovery/per_seed.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["cell_id"] in wanted and row["arm"] == "recovery" and int(row["seed_index"]) < 2:
                stored[(row["cell_id"], int(row["seed_index"]))] = float(row["p95_error_m"])
    assert len(stored) == 4
    config = RecoveryConfig()
    seeds = seed_list(2)
    for (cell_id, index), p95 in stored.items():
        result = simulate_condition("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.10,
                                    loss_model="independent", seed=seeds[index], config=config,
                                    arms={"told": ("told", wanted[cell_id])})
        assert repr(result["told"].arm.p95_error_m) == repr(p95)


def test_adaptive_arm_records_its_noise_estimate():
    config = RecoveryConfig()
    result = simulate_condition("constant_velocity", delay_s=0.2, loss_probability=0.0, noise_std_m=0.10,
                                loss_model="independent", seed=seed_list(1)[0], config=config,
                                arms={"adaptive": ("adaptive", 0.05)})
    info = result["adaptive"]
    assert info.final_std_m is not None and 0.02 < info.final_std_m < 0.2
    assert info.mean_scored_std_m is not None and sqrt(1e-6) <= info.mean_scored_std_m


# ---- the pre-registered rule is applied by code --------------------------------

def _fake_cell(motion, noise, delay, loss, medians, labels, cost=0.0):
    arms = {arm: {"p95_median_m": medians[arm], "label": labels.get(arm, "passes")} for arm in medians}
    return {"condition_id": f"{motion}/{noise}/{delay}/{loss}", "motion": motion, "noise_std_m": noise,
            "delay_s": delay, "loss": loss, "arms": arms, "paired_adaptive_minus_told_5cm": {"median_m": cost}}


def test_rule_verdicts():
    from tools.run_s6_adaptive_r import evaluate_rules

    good_h1 = _fake_cell("constant_velocity", 0.10, 0.3, "none",
                         {"told_5cm": 0.30, "told_truth": 0.24, "adaptive": 0.25}, {"told_5cm": "fails"})
    small_gap = _fake_cell("constant_velocity", 0.10, 0.0, "none",
                           {"told_5cm": 0.19, "told_truth": 0.185, "adaptive": 0.30}, {})
    h2_ok = _fake_cell("constant_velocity", 0.05, 0.2, "none",
                       {"told_5cm": 0.12, "told_truth": 0.12, "adaptive": 0.125}, {}, cost=0.005)
    turn_flip = _fake_cell("constant_turn", 0.05, 0.2, "none",
                           {"told_5cm": 0.2, "told_truth": 0.2, "adaptive": 0.26}, {"adaptive": "fails"})
    assert evaluate_rules([good_h1, small_gap, h2_ok])["verdict"] == "worth proposing"
    assert evaluate_rules([good_h1, h2_ok, turn_flip])["verdict"] == "worth proposing only with a turn guard"
    bad_h1 = _fake_cell("constant_velocity", 0.10, 0.3, "none",
                        {"told_5cm": 0.30, "told_truth": 0.24, "adaptive": 0.29}, {"told_5cm": "fails"})
    assert evaluate_rules([bad_h1, h2_ok])["verdict"] == "not worth proposing as is"
    costly = _fake_cell("constant_velocity", 0.05, 0.2, "none",
                        {"told_5cm": 0.12, "told_truth": 0.12, "adaptive": 0.14}, {}, cost=0.02)
    assert evaluate_rules([good_h1, costly])["H2_no_harm"]["pass"] is False
