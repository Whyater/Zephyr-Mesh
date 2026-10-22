"""Tests for the experiment engine, the CV estimator adapter and the model factory.

The most important group checks that the engine's naive and recovery-v1 arms
reproduce the S6 recovery sweep (sim/s6_recovery.py) to 1e-12 on every S6 recovery cell type.
That is a regression check against existing code, not an independent physics
check. The retrodiction tests are independent: they compare against an
offline filter that simply sorts packets by capture time.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2

from sim import s6_recovery as s6
from sim.estimation import engine
from sim.estimation.base import NAIVE, RECOVERY_V1, ArmSpec, EstimatorSettings
from sim.estimation.cv import CVEstimator
from sim.estimation.factory import make_estimator

CONFIG = s6.RecoveryConfig()
SEEDS = s6.seed_list(3)
RETRO = ArmSpec("retro", model="cv", stamping="capture", late_policy="retrodict")


def _s6_cases():
    cases = []
    for motion in s6.MOTIONS:
        for delay in (0.0, 0.1, 0.3):
            for loss in (0.0, 0.3):
                for noise in (0.0, 0.05, 0.10):
                    cases.append((motion, delay, loss, noise, "independent", 0.0, None))
        cases.append((motion, 0.1, 0.3, 0.05, "burst", 0.0, None))
        cases.append((motion, 0.3, 0.3, 0.05, "burst", 0.0, None))
        cases.append((motion, 0.3, 0.0, 0.05, "independent", 0.15, None))
        cases.append((motion, 0.3, 0.3, 0.05, "independent", 0.15, None))
        cases.append((motion, 0.3, 0.3, 0.10, "independent", 0.0, 0.10))  # oracle R
    return cases


def _delivery_key(d):
    return (d.seq, d.capture_time_s, d.receive_time_s, d.measurement_m)


@pytest.mark.parametrize("case", _s6_cases(), ids=lambda c: "/".join(str(v) for v in c))
def test_naive_and_recovery_reproduce_s6(case):
    motion, delay, loss, noise, loss_model, jitter, r_std = case
    cell = engine.s6_cell(motion, delay, loss, noise, loss_model=loss_model, jitter_s=jitter)
    for seed in SEEDS:
        kwargs = dict(delay_s=delay, loss_probability=loss, noise_std_m=noise, seed=seed, config=CONFIG,
                      loss_model=loss_model, jitter_s=jitter)
        bundle = engine.deliver_cell(cell, seed, CONFIG)
        reference = s6.deliver(motion, **kwargs)
        assert [_delivery_key(d) for d in bundle.deliveries] == [_delivery_key(d) for d in reference]
        assert all(d.stamp_s == d.capture_time_s for d in bundle.deliveries)
        expected = s6.run_arms(reference, motion, CONFIG, tracker_measurement_std_m=r_std)
        for arm in (NAIVE, RECOVERY_V1):
            got = engine.run_arm(bundle, cell, replace(arm, r_std_m=r_std), CONFIG)
            want = expected[arm.name]
            for name in ("p95_error_m", "max_error_m", "rmse_m"):
                assert abs(getattr(got, name) - getattr(want, name)) <= 1e-12, (arm.name, name)
            if want.mean_nis is None:
                assert got.mean_nis is None
            else:
                assert abs(got.mean_nis - want.mean_nis) <= 1e-12
            for name in ("fused_count", "dropped_count", "fused_out_of_order_count", "failed"):
                assert getattr(got, name) == getattr(want, name), (arm.name, name)
            assert np.max(np.abs(got.errors_m - want.errors_m)) <= 1e-12


def test_seed_streams_keep_s6_link_and_sensor_seeds():
    for seed in SEEDS:
        streams = engine.seed_streams(seed)
        link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
        assert (streams["link_seed"], streams["sensor_seed"]) == (link_seed, sensor_seed)
        extras = [streams[k] for k in ("target", "clock", "sync_up", "sync_down", "outliers", "sender")]
        assert len(set(extras)) == len(extras)


# ---- CV adapter and factory -------------------------------------------------

def _fed_cv(r_std=None):
    est = CVEstimator(np.zeros(3), EstimatorSettings(), r_std_m=r_std)
    rng = np.random.default_rng(1)
    for k in range(20):
        est.update(np.array([0.1 * k, 0.02 * k, 0.0]) + rng.normal(0, 0.05, 3), 0.1 * k)
    return est


def test_cv_predict_positions_matches_predict_at_and_does_not_mutate():
    est = _fed_cv()
    before = copy.deepcopy(est)
    times = est.timestamp_s + np.linspace(0.0, 0.7, 36)
    batch = est.predict_positions(times)
    single = np.array([est.predict_at(t).position_m for t in times])
    assert np.max(np.abs(batch - single)) <= 1e-12
    assert np.array_equal(est.tracker.x, before.tracker.x) and np.array_equal(est.tracker.P, before.tracker.P)
    with pytest.raises(ValueError):
        est.predict_at(est.timestamp_s - 0.01)
    with pytest.raises(ValueError):
        est.update(np.zeros(3), est.timestamp_s)  # duplicate stamp


def test_cv_scales_reach_the_tracker():
    est = _fed_cv()
    est.q_scale, est.r_scale = 4.0, 9.0
    assert np.allclose(est.measurement_cov(), np.eye(3) * 0.05**2 * 9.0)
    est.update(np.array([2.0, 0.4, 0.0]), est.timestamp_s + 0.1)
    assert est.tracker.q_acc == pytest.approx(0.4)
    assert np.allclose(est.tracker.R, np.eye(3) * 0.05**2 * 9.0)


def test_cv_copy_is_independent():
    est = _fed_cv()
    twin = est.copy()
    twin.update(np.array([5.0, 5.0, 5.0]), est.timestamp_s + 0.1)
    assert twin.update_count == est.update_count + 1
    assert not np.array_equal(twin.tracker.x, est.tracker.x)


def test_factory_builds_every_model():
    settings = EstimatorSettings()
    for model in ("cv", "ca", "ct", "imm"):
        est = make_estimator(model, settings, np.zeros(3), q_factor=2.0, r_std_m=0.1)
        est.update(np.array([0.01, 0.0, 0.0]), 0.0)
        est.update(np.array([0.11, 0.0, 0.0]), 0.1)
        assert est.update_count == 2
        assert np.allclose(est.measurement_cov(), np.eye(3) * 0.01)
    ct = make_estimator("ct", settings, np.zeros(3), model_options=(("known_turn_rate", 1.0),))
    assert ct.update_count == 0
    with pytest.raises(ValueError):
        make_estimator("kalman", settings, np.zeros(3))
    with pytest.raises((ValueError, TypeError)):
        make_estimator("cv", settings, np.zeros(3), model_options=(("known_turn_rate", 1.0),))


# ---- retrodiction ------------------------------------------------------------

def _offline_cv(packets, r_std=None):
    est = CVEstimator(np.zeros(3), EstimatorSettings(), r_std_m=r_std)
    for packet in sorted(packets, key=lambda d: d.stamp_s):
        est.update(np.asarray(packet.measurement_m), packet.stamp_s)
    return est


@pytest.mark.parametrize("delay", [0.1, 0.3])
def test_retrodiction_without_jitter_equals_recovery(delay):
    cell = engine.s6_cell("constant_turn", delay, 0.3, 0.05)
    assert engine.retro_window_for(cell) == 0.0
    for seed in SEEDS:
        bundle = engine.deliver_cell(cell, seed, CONFIG)
        retro = engine.run_arm(bundle, cell, RETRO, CONFIG)
        recovery = engine.run_arm(bundle, cell, RECOVERY_V1, CONFIG)
        assert retro.retro_window_s == 0.0 and retro.refiltered_count == 0
        for name in ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count"):
            assert getattr(retro, name) == getattr(recovery, name)


def test_retrodiction_with_jitter_matches_offline_sorted_filter():
    cell = engine.s6_cell("constant_turn", 0.3, 0.0, 0.05, jitter_s=0.15)
    window = engine.retro_window_for(cell)
    assert 0.3 < window < 1.5
    total_refiltered = 0
    for seed in SEEDS:
        bundle = engine.deliver_cell(cell, seed, CONFIG)
        run = engine.run_arm_detailed(bundle, cell, RETRO, CONFIG)
        out = run.outcome
        total_refiltered += out.refiltered_count
        assert out.retro_window_s == window
        assert out.fused_count + out.dropped_count + out.gated_count == len(bundle.deliveries)
        offline = _offline_cv(run.fused)
        assert run.estimator.timestamp_s == offline.timestamp_s
        assert np.max(np.abs(run.estimator.tracker.x - offline.tracker.x)) <= 1e-9
        assert np.max(np.abs(run.estimator.tracker.P - offline.tracker.P)) <= 1e-9
        recovery = engine.run_arm(bundle, cell, RECOVERY_V1, CONFIG)
        assert out.dropped_count < recovery.dropped_count
    assert total_refiltered > 0


def test_retrodiction_errors_match_offline_filter_at_queries():
    """At each checked query t, the offline filter holds every packet received by t, sorted by capture time."""
    cell = engine.s6_cell("constant_velocity", 0.3, 0.0, 0.05, jitter_s=0.15)
    bundle = engine.deliver_cell(cell, SEEDS[0], CONFIG)
    out = engine.run_arm(bundle, cell, RETRO, CONFIG)
    assert out.dropped_count == 0 and out.refiltered_count > 0
    truth = bundle.target.position(out.query_times_s)
    for i in range(0, out.query_times_s.size, 7):
        t = out.query_times_s[i]
        received = [d for d in bundle.deliveries if d.receive_time_s <= t + engine.TIME_TOLERANCE_S]
        offline = _offline_cv(received)
        estimate = offline.predict_at(max(t, offline.timestamp_s)).position_m
        assert abs(np.linalg.norm(estimate - truth[i]) - out.errors_m[i]) <= 1e-9


def test_retrodiction_with_zero_window_equals_recovery():
    cell = engine.s6_cell("constant_turn", 0.3, 0.3, 0.05, jitter_s=0.15)
    arm = replace(RETRO, retro_window_s=0.0)
    for seed in SEEDS:
        bundle = engine.deliver_cell(cell, seed, CONFIG)
        retro = engine.run_arm(bundle, cell, arm, CONFIG)
        recovery = engine.run_arm(bundle, cell, RECOVERY_V1, CONFIG)
        assert retro.refiltered_count == 0
        for name in ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count"):
            assert getattr(retro, name) == getattr(recovery, name)


def test_retrodiction_works_with_other_models_and_adaptive():
    cell = engine.s6_cell("constant_turn", 0.3, 0.3, 0.05, jitter_s=0.15)
    bundle = engine.deliver_cell(cell, SEEDS[1], CONFIG)
    for arm in (replace(RETRO, name="ca", model="ca"), replace(RETRO, name="imm", model="imm"),
                replace(RETRO, name="adq", adaptive="q")):
        out = engine.run_arm(bundle, cell, arm, CONFIG)
        assert out.refiltered_count > 0 and np.isfinite(out.p95_error_m)


# ---- gate --------------------------------------------------------------------

def test_gate_rejects_outliers_and_keeps_clean_packets():
    """A 1 m jump against a 5 cm filter has NIS near (1 / 0.06)^2, about 280, far above
    chi2.ppf(0.99, 3) = 11.3, so nearly every outlier is rejected; a clean packet is
    rejected only about 1 % of the time."""
    cell = engine.CellSpec(target="cv", noise_model="gaussian_outliers", noise_std_m=0.05,
                           outlier_probability=0.05, outlier_m=1.0)
    arm = ArmSpec("gated", stamping="capture", gate_probability=0.99, r_std_m=0.05)
    counts = dict(gated_outliers=0, gated_clean=0, gate_checked_outliers=0, gate_checked_clean=0)
    for seed in s6.seed_list(10):
        bundle = engine.deliver_cell(cell, seed, CONFIG)
        out = engine.run_arm(bundle, cell, arm, CONFIG)
        assert out.extras["gate_checked_outliers"] == len(bundle.outlier_seqs)
        assert out.gated_count == out.extras["gated_outliers"] + out.extras["gated_clean"]
        for key in counts:
            counts[key] += out.extras[key]
    assert counts["gate_checked_outliers"] > 30
    assert counts["gated_outliers"] / counts["gate_checked_outliers"] > 0.95
    assert counts["gated_clean"] / counts["gate_checked_clean"] < 0.05
    assert chi2.ppf(0.99, 3) == pytest.approx(11.345, abs=1e-3)


def test_pure_gate_can_lock_out_and_the_guard_recovers():
    """Seed 2509691599 fuses one 1 m outlier during start-up, when the filter's
    position variance is still about 1 m^2, so the outlier looks believable.
    With a pure gate (gate_reset_after=0) every clean packet after that is
    rejected and the error grows without bound. The guard fuses a packet after
    3 rejections in a row and the filter comes back to the 5 cm noise level."""
    cell = engine.CellSpec(target="cv", noise_model="gaussian_outliers", noise_std_m=0.05,
                           outlier_probability=0.05, outlier_m=1.0)
    guarded = ArmSpec("gated", stamping="capture", gate_probability=0.99, r_std_m=0.05)
    pure = replace(guarded, gate_reset_after=0)
    bundle = engine.deliver_cell(cell, 2509691599, CONFIG)
    locked = engine.run_arm(bundle, cell, pure, CONFIG)
    recovered = engine.run_arm(bundle, cell, guarded, CONFIG)
    assert locked.extras["gate_forced_accepts"] == 0
    assert locked.extras["gated_clean"] > 100 and locked.p95_error_m > 1.0
    assert recovered.extras["gate_forced_accepts"] >= 1 and recovered.p95_error_m < 0.25


# ---- noise models, cells, run_cell -------------------------------------------

def test_student_t_noise_has_the_requested_std_and_is_seeded():
    cell = engine.CellSpec(target="cv", rate_hz=50.0, noise_model="student_t", noise_std_m=0.05, noise_dof=5.0)
    a = engine.deliver_cell(cell, SEEDS[0], CONFIG)
    b = engine.deliver_cell(cell, SEEDS[0], CONFIG)
    assert [_delivery_key(d) for d in a.deliveries] == [_delivery_key(d) for d in b.deliveries]
    z = np.array([d.measurement_m for d in a.deliveries])
    truth = a.target.position(np.array([d.capture_time_s for d in a.deliveries]))
    assert 0.04 < np.std(z - truth) < 0.06
    with pytest.raises(ValueError):
        engine.CellSpec(noise_model="student_t", noise_dof=2.0)


def test_cell_key_and_dict():
    a = engine.CellSpec(target="ct", target_params={"radius_m": 2.0}, delay_s=0.1)
    b = engine.CellSpec(target="ct", target_params=(("radius_m", 2.0),), delay_s=0.1)
    assert a == b and a.key() == b.key()
    assert engine.CellSpec(delay_s=0.1).key() != engine.CellSpec(delay_s=0.10000001).key()
    assert a.as_dict()["target_params"] == {"radius_m": 2.0}
    with pytest.raises(ValueError):
        engine.CellSpec(loss_model="gilbert")


def test_default_packet_bytes_match_policies():
    policies = pytest.importorskip("sim.estimation.policies")
    sizes = policies.packet_bytes(False)
    assert sizes["data"] == engine.DEFAULT_DATA_BYTES
    assert sizes["on_air"] == engine.DEFAULT_ON_AIR_BYTES


def test_run_cell_structure_and_repeatability():
    cell = engine.s6_cell("constant_turn", 0.3, 0.3, 0.05, jitter_s=0.15)
    arms = [NAIVE, RECOVERY_V1, RETRO]
    first = engine.run_cell(cell, arms, SEEDS, CONFIG)
    second = engine.run_cell(cell, arms, SEEDS, CONFIG)
    assert first == second
    assert set(first["arms"]) == {"naive", "recovery", "retro"}
    assert set(first["paired_vs_naive"]) == {"recovery", "retro"}
    assert set(first["paired_vs_recovery"]) == {"naive", "retro"}
    assert len(first["per_seed"]) == 3 * len(SEEDS)
    row = first["per_seed"][0]
    for key in ("cell_key", "arm", "seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m", "mean_nis",
                "fused_count", "dropped_count", "fused_out_of_order_count", "gated_count", "refiltered_count",
                "failed", "x_monitor_warn_fraction", "x_final_q_scale", "x_final_r_scale"):
        assert key in row
    assert first["key"] == cell.key() and row["cell_key"] == cell.key()
    # The engine's per-arm summaries equal the S6 recovery sweep's summaries for the same seeds.
    reference = [s6.simulate_run("constant_turn", delay_s=0.3, loss_probability=0.3, noise_std_m=0.05, seed=s,
                                 config=CONFIG, jitter_s=0.15) for s in SEEDS]
    want = s6.summarize_arm([r["recovery"] for r in reference])
    got = first["arms"]["recovery"]
    for key, value in want.items():
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, abs=1e-12), key
        else:
            assert got[key] == value, key
