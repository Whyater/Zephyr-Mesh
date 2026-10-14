"""Tests for the truth motions (sim/estimation/targets.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.estimation.kinematics import ct_step
from sim.estimation.targets import TARGET_NAMES, make_target
from sim.s6_recovery import RecoveryConfig, truth_position

TIMES = np.arange(0.0, 12.0 + 0.001, 0.002)


def test_cv_and_ct_match_s6_bit_for_bit():
    config = RecoveryConfig()
    t = np.arange(0.0, config.duration_s + config.query_dt_s / 2, config.query_dt_s)
    for name, s6_name in (("cv", "constant_velocity"), ("ct", "constant_turn")):
        assert np.array_equal(make_target(name).position(t), truth_position(s6_name, t, config))
    assert np.array_equal(make_target("cv").position(TIMES), truth_position("constant_velocity", TIMES, config))
    assert np.array_equal(make_target("ct").position(TIMES), truth_position("constant_turn", TIMES, config))


def test_names_defaults_and_start_position():
    assert set(TARGET_NAMES) == {"cv", "ct", "jink", "figure8", "start_stop", "mixed", "ca_line", "cwna"}
    for name in TARGET_NAMES:
        target = make_target(name, seed=3)
        assert target.name == name
        assert isinstance(target.params, dict) and target.params
        assert np.array_equal(target.start_position, np.zeros(3))
        assert np.array_equal(target.position([0.0]), np.zeros((1, 3)))
        assert target.position(TIMES[:5]).shape == (5, 3)
        assert target.velocity(TIMES[:5]).shape == (5, 3)
    assert make_target("ct", {"radius_m": 2.0}).params == {"speed_mps": 1.0, "radius_m": 2.0}


def test_bad_names_and_params_raise():
    with pytest.raises(ValueError):
        make_target("nope")
    with pytest.raises(ValueError):
        make_target("cv", {"radius_m": 1.0})
    with pytest.raises(ValueError):
        make_target("ct", {"radius_m": 0.0})
    with pytest.raises(ValueError):
        make_target("jink", {"turn_rate_std_radps": -1.0})
    with pytest.raises(ValueError):
        make_target("cv", {"speed_mps": float("nan")})
    with pytest.raises(ValueError):
        make_target("cv").position([-0.1])
    with pytest.raises(ValueError):
        make_target("cv").position([[0.0, 1.0]])


@pytest.mark.parametrize("name", ["cv", "ct", "figure8", "start_stop", "mixed", "ca_line"])
def test_velocity_matches_finite_difference_of_position(name):
    target = make_target(name)
    t = np.linspace(0.3, 11.7, 400)
    h = 1e-6
    numeric = (target.position(t + h) - target.position(t - h)) / (2 * h)
    analytic = target.velocity(t)
    # The central difference is wrong only at a corner of a piecewise motion.
    close = np.all(np.abs(numeric - analytic) < 1e-4, axis=1)
    assert close.mean() > 0.97


def test_ca_line_is_constant_acceleration():
    target = make_target("ca_line", {"accel_mps2": 1.5})
    t = np.array([0.0, 1.0, 2.0])
    assert np.allclose(target.position(t)[:, 0], 0.75 * t**2)
    assert np.allclose(target.velocity(t)[:, 0], 1.5 * t)


def test_figure8_closed_form_and_speed():
    target = make_target("figure8")
    assert np.allclose(target.position([math.pi / 1.0])[0], [1.5 * math.sin(0.5 * math.pi), 0.75 * math.sin(math.pi), 0.0])
    speed = np.linalg.norm(target.velocity(TIMES), axis=1)
    assert speed.max() < 2.0
    # Independent bound: speed <= sqrt((Ax W)^2 + (2 Ay W)^2).
    assert speed.max() <= math.hypot(1.5 * 0.5, 2 * 0.75 * 0.5) + 1e-12


def _phase_boundaries_start_stop(p):
    v, a = p["cruise_speed_mps"], p["accel_mps2"]
    t_acc = v / a
    period = 2 * t_acc + p["cruise_s"] + p["stop_s"]
    edges = np.cumsum([t_acc, p["cruise_s"], t_acc, p["stop_s"]])
    return np.array([c * period + e for c in range(3) for e in edges[:-1]] + [(c + 1) * period for c in range(3)])


def test_start_stop_continuity_and_limits():
    target = make_target("start_stop")
    for tb in _phase_boundaries_start_stop(target.params):
        eps = 1e-9
        before = target.position([tb - eps])[0], target.velocity([tb - eps])[0]
        after = target.position([tb + eps])[0], target.velocity([tb + eps])[0]
        at = target.position([tb])[0], target.velocity([tb])[0]
        for side in (before, after):
            assert np.max(np.abs(side[0] - at[0])) < 1e-8
            assert np.max(np.abs(side[1] - at[1])) < 1e-8
    t = np.arange(0.0, 12.0, 0.001)
    speed = target.velocity(t)[:, 0]
    assert speed.max() <= 1.0 + 1e-12 and speed.min() >= -1e-12
    assert np.isclose(speed.max(), 1.0)
    accel = np.diff(speed) / 0.001
    assert np.abs(accel).max() <= 2.0 + 1e-6
    assert np.isclose(np.abs(accel).max(), 2.0, atol=1e-6)
    # Distance over one period: v^2/a + v cruise_s.
    period = 2 * 0.5 + 1.5 + 1.0
    assert np.isclose(target.position([period])[0, 0], 1.0**2 / 2.0 + 1.5)
    assert np.isclose(target.position([3 * period])[0, 0], 3 * (1.0**2 / 2.0 + 1.5))
    assert np.all(np.diff(target.position(t)[:, 0]) >= -1e-12)


def test_start_stop_stands_still_at_stop():
    target = make_target("start_stop")
    stop_mid = 0.5 + 1.5 + 0.5 + 0.5  # inside the stand-still phase of the first cycle
    assert target.velocity([stop_mid])[0, 0] == 0.0


def test_mixed_continuity_and_geometry():
    target = make_target("mixed")
    T1, T2 = 4.0, math.pi  # straight 4 s, then pi rad at 1 rad/s
    for tb in (T1, T1 + T2):
        eps = 1e-9
        lo = np.concatenate([target.position([tb - eps])[0], target.velocity([tb - eps])[0]])
        hi = np.concatenate([target.position([tb + eps])[0], target.velocity([tb + eps])[0]])
        mid = np.concatenate([target.position([tb])[0], target.velocity([tb])[0]])
        assert np.max(np.abs(lo - mid)) < 1e-8 and np.max(np.abs(hi - mid)) < 1e-8
    # After a half circle of radius 1 the object is 2 m to the side and heads back along -x.
    end = target.position([T1 + T2])[0]
    assert np.allclose(end, [4.0 + 0.0, 2.0, 0.0], atol=1e-12)
    assert np.allclose(target.velocity([T1 + T2 + 1.0])[0], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(target.position([T1 + T2 + 1.0])[0], [3.0, 2.0, 0.0], atol=1e-12)
    speed = np.linalg.norm(target.velocity(TIMES), axis=1)
    assert np.allclose(speed, 1.0, atol=1e-12)
    # A different angle: quarter turn, then straight along +y.
    q = make_target("mixed", {"turn_angle_rad": math.pi / 2, "straight1_s": 1.0})
    t_end = 1.0 + math.pi / 2
    assert np.allclose(q.position([t_end + 2.0])[0], [1.0 + 1.0, 1.0 + 2.0, 0.0], atol=1e-12)


def test_jink_speed_is_exact():
    target = make_target("jink", seed=11)
    t = np.concatenate([TIMES, np.random.default_rng(0).uniform(0, 12, 500)])
    speed = np.linalg.norm(target.velocity(t), axis=1)
    assert np.max(np.abs(speed - 1.0)) < 1e-9
    fast = make_target("jink", {"speed_mps": 2.5}, seed=11)
    assert np.max(np.abs(np.linalg.norm(fast.velocity(t), axis=1) - 2.5)) < 1e-9
    # Chord speed over one fine step is just under the arc speed.
    chord = np.linalg.norm(np.diff(target.position(np.arange(0.0, 1.0, 0.001)), axis=0), axis=1) / 0.001
    assert np.all(chord <= 1.0 + 1e-9) and np.all(chord > 0.999)


def test_jink_starts_heading_plus_x():
    target = make_target("jink", seed=5)
    assert np.allclose(target.velocity([0.0])[0], [1.0, 0.0, 0.0])
    p = target.position([0.001])[0]
    assert p[0] > 0.0009 and abs(p[1]) < 1e-5 and p[2] == 0.0


def test_jink_matches_ct_step_chain():
    target = make_target("jink", seed=7)
    n = 3000
    w = target.turn_rate(np.arange(n) * 0.001)
    x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    path = [x[:3].copy()]
    for k in range(n):
        x = ct_step(np.concatenate([x[:6], [w[k]]]), 0.001)
        path.append(x[:3].copy())
    expected = np.array(path)
    got = target.position(np.arange(n + 1) * 0.001)
    assert np.max(np.abs(got - expected)) < 1e-9


def test_jink_seeded_and_independent_of_requested_times():
    a = make_target("jink", seed=21)
    b = make_target("jink", seed=21)
    c = make_target("jink", seed=22)
    grid = np.arange(0.0, 12.0 + 1e-9, 0.002)
    short = a.position([1.0, 2.0, 3.0])
    full = a.position(grid)
    assert np.array_equal(short, full[[500, 1000, 1500]])
    assert np.array_equal(b.position(grid), full)
    assert not np.allclose(c.position(grid), full)
    # Asking later for a much longer run must not change the earlier path.
    a.position([30.0])
    assert np.array_equal(a.position(grid), full)
    assert np.array_equal(make_target("jink", seed=21).position([30.0]), a.position([30.0]))
    # No seed means seed 0.
    assert np.array_equal(make_target("jink").position(grid), make_target("jink", seed=0).position(grid))


def test_jink_lateral_acceleration_and_ou_autocorrelation():
    sigma, tau, v = 1.0, 1.0, 1.0
    target = make_target("jink", {"speed_mps": v, "turn_rate_std_radps": sigma, "correlation_time_s": tau}, seed=2026)
    t = np.arange(0.0, 2000.0, 0.01)
    w = target.turn_rate(t)
    lateral = v * w
    assert abs(np.sqrt(np.mean(lateral**2)) - v * sigma) < 0.1 * v * sigma
    assert abs(np.mean(w)) < 0.2
    lag = int(round(tau / 0.01))
    rho = np.corrcoef(w[:-lag], w[lag:])[0, 1]
    assert abs(rho - math.exp(-1.0)) < 0.1
    # Scaling check: different sigma and tau.
    other = make_target("jink", {"turn_rate_std_radps": 0.5, "correlation_time_s": 2.0}, seed=1)
    w2 = other.turn_rate(np.arange(0.0, 2000.0, 0.01))
    assert abs(np.std(w2) - 0.5) < 0.05
    lag2 = 200
    assert abs(np.corrcoef(w2[:-lag2], w2[lag2:])[0, 1] - math.exp(-1.0)) < 0.1


def test_turn_rate_only_for_jink():
    with pytest.raises(ValueError):
        make_target("cv").turn_rate([0.0])


def test_cwna_seeded_starts_at_origin_and_interpolates():
    a = make_target("cwna", seed=4)
    b = make_target("cwna", seed=4)
    c = make_target("cwna", seed=5)
    t = np.arange(0.0, 12.0 + 1e-9, 0.02)
    assert np.array_equal(a.position(t), b.position(t))
    assert not np.allclose(a.position(t), c.position(t))
    assert np.array_equal(a.position([0.0]), np.zeros((1, 3)))
    at_samples = a.position([0.1, 0.2])
    mid = a.position([0.15])[0]
    assert np.allclose(mid, 0.5 * (at_samples[0] + at_samples[1]))
    # A longer request leaves the earlier path unchanged.
    first = a.position(t)
    a.position([500.0])
    assert np.array_equal(a.position(t), first)


def test_cwna_noise_covariance_matches_q_of_T():
    q, T = 0.1, 0.1
    target = make_target("cwna", {"q_acc_psd": q, "sample_period_s": T}, seed=9)
    k = np.arange(0, 20001)
    times = k * T
    pos = target.position(times)
    vel = target.velocity(times)
    dp = pos[1:] - pos[:-1] - T * vel[:-1]
    dv = vel[1:] - vel[:-1]
    Q = q * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
    assert abs(np.var(dp) - Q[0, 0]) < 0.05 * Q[0, 0]
    assert abs(np.var(dv) - Q[1, 1]) < 0.05 * Q[1, 1]
    assert abs(np.mean(dp * dv) - Q[0, 1]) < 0.05 * Q[0, 1]
    # Initial velocity is N(0, 1) per axis: the first few samples are modest.
    assert np.all(np.abs(target.velocity([0.0])[0]) < 5.0)
