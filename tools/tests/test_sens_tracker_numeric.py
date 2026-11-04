"""Tests for tools/sens_tracker_numeric.py.

The control test checks the float64 copy against sim/tracker.py bit for bit.
The hand-check tests compare against numbers worked out on paper, not against
the code's own formulas.
"""
import math

import numpy as np
import pytest

from sim.tracker import ConstantVelocityTracker
from tools import sens_tracker_numeric as stn


def _pair(**kw):
    args = dict(position_variance_m2=1.0, velocity_variance_m2ps2=1.0, acceleration_spectral_density=0.1,
                measurement_std_m=0.05)
    args.update(kw)
    sim = ConstantVelocityTracker((0.1, -0.2, 0.3), (0.0, 0.0, 0.0), 0.0, **args)
    copy = stn.NumericTracker((0.1, -0.2, 0.3), (0.0, 0.0, 0.0), 0.0, dtype=np.float64, **args)
    return sim, copy


@pytest.mark.parametrize("motion", ["straight", "turn", "jink"])
def test_float64_copy_matches_sim_tracker_bit_for_bit(motion):
    times, z = stn.make_measurements(motion, 0.05, seed=12345, duration_s=30.0)
    sim, copy = _pair()
    for t, zk in zip(times, z):
        sim.update(zk, float(t))
        copy.update(zk, float(t))
        assert np.array_equal(sim.x, copy.x)
        assert np.array_equal(sim.P, copy.P)
        q = float(t) + 0.0137
        s = sim.predict_at(q)
        cx, cP = copy.predict_at(q)
        assert np.array_equal(np.asarray(s.position_m), cx[:3])
        assert np.array_equal(np.asarray(s.covariance), cP)
    sim.predict(40.0)
    copy.predict(40.0)
    assert np.array_equal(sim.x, copy.x) and np.array_equal(sim.P, copy.P)


def test_float32_variant_really_stays_float32():
    times, z = stn.make_measurements("turn", 0.05, seed=7, duration_s=5.0)
    trk = stn.NumericTracker((0, 0, 0), (0, 0, 0), dtype=np.float32)
    for t, zk in zip(times, z):
        trk.update(zk, t)
    trk.advance(np.float32(0.01))
    assert trk.x.dtype == np.float32 and trk.P.dtype == np.float32
    assert isinstance(trk.timestamp_s, np.float32)


def test_single_update_matches_hand_value():
    # One axis, prior variance 1, measurement variance 0.05^2 = 0.0025, dt = 0.
    # Posterior variance = 1 * 0.0025 / (1 + 0.0025) = 0.0024937655860349...
    expected = 0.0025 / 1.0025
    for dtype, tol in ((np.float64, 1e-13), (np.float32, 1e-6)):
        trk = stn.NumericTracker((0, 0, 0), (0, 0, 0), dtype=dtype, acceleration_spectral_density=0.0)
        trk.update((0.1, 0.0, 0.0), 0.0)
        assert abs(float(trk.P[0, 0]) - expected) / expected < tol
        # Position moves by the gain K = 1 / 1.0025 times the 0.1 m innovation.
        assert abs(float(trk.x[0]) - 0.1 / 1.0025) < 10 * tol


def test_float32_cancellation_hand_example():
    """Why the short form can go wrong in float32, on numbers small enough to do by hand.

    Prior variance P = 1, measurement variance R = 1e-6. The gain K = P / (P + R)
    is 0.999999000001, and the exact posterior is P R / (P + R) = 9.99999e-7.
    The short form computes (1 - K) * P. In float32, K rounds to the nearest
    representable number near 0.999999 (spacing about 6e-8), so 1 - K keeps only
    about one significant digit: a relative error of several percent from one
    subtraction of two nearly equal numbers. The Joseph form adds two positive
    terms, (1 - K)^2 P + K^2 R, and the K^2 R term carries the answer, so it does
    not lose those digits.
    """
    P, R = np.float32(1.0), np.float32(1e-6)
    K = P / (P + R)
    exact = 1.0 * 1e-6 / (1.0 + 1e-6)
    short = float((np.float32(1.0) - K) * P)
    joseph = float((np.float32(1.0) - K) ** 2 * P + K * K * R)
    assert abs(short - exact) / exact > 0.01
    assert abs(joseph - exact) / exact < 1e-3


def test_snap_tick_hand_values():
    assert stn.snap_tick(0.0043, "nearest") == 0
    assert stn.snap_tick(0.0051, "nearest") == 1
    assert stn.snap_tick(0.1043, "nearest") == 10
    assert stn.snap_tick(0.1043, "next") == 11
    assert stn.snap_tick(0.1000000000001, "next") == 10  # float noise does not push a tick time to the next tick
    assert stn.snap_tick(0.2, "next") == 20


def test_truth_paths_hand_values():
    t = [0.0, 0.75, 1.5, 2.25, 3.0]
    j = stn.truth_position("jink", t)
    assert np.allclose(j[:, 0], t)
    assert np.allclose(j[:, 1], [0.0, 0.75, 1.5, 0.75, 0.0])
    c = stn.truth_position("turn", [math.pi / 2])
    assert np.allclose(c[0], [1.0, 1.0, 0.0])


def test_measurement_times_are_irregular_and_off_grid():
    times, _ = stn.make_measurements("straight", 0.05, seed=3, duration_s=60.0)
    gaps = np.diff(times)
    assert gaps.min() > 0.093 and gaps.max() < 0.107
    assert np.std(gaps) > 1e-3
    assert np.any(np.abs(times / stn.TICK_S - np.round(times / stn.TICK_S)) > 0.1)


def test_run_seed_short_smoke():
    rows = stn.run_seed("straight", seed=11, duration_s=6.0)
    by = {r["variant"]: r for r in rows}
    assert set(by) == set(stn.VARIANTS)
    assert by["f64_exact"]["max_pos_diff_vs_ref_m"] == 0.0
    assert by["f32_exact"]["final_P_dtype"] == "float32"
    assert all(r["cholesky_failures"] == 0 for r in rows)
    summary = stn.summarize(rows)
    assert {r["variant"] for r in summary["rows"]} == set(stn.VARIANTS)


def test_control_comparison_can_fail():
    """The bit-for-bit check is not vacuous: the short-form copy drifts from the sim within a few updates."""
    times, z = stn.make_measurements("turn", 0.05, seed=12345, duration_s=5.0)
    sim, _ = _pair()
    other = stn.NumericTracker((0.1, -0.2, 0.3), (0.0, 0.0, 0.0), 0.0, dtype=np.float64, form="simple",
                               symmetrize=False)
    differs = False
    for t, zk in zip(times, z):
        sim.update(zk, float(t))
        other.update(zk, float(t))
        differs = differs or not np.array_equal(sim.P, other.P)
    assert differs
