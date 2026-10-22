import numpy as np
import pytest

from sim.sensor import SensorConfig, SensorModel
from sim.tracker import ConstantVelocityTracker


def test_zero_noise_sensor_is_truth_and_seeded_dropout_is_explicit():
    model = SensorModel(SensorConfig(noise_std_m=0.0, dropout_probability=0.0), seed=4)
    sample = model.sample((1.0, 2.0, 3.0), 0.1)
    assert sample.available and sample.measurement_m == (1.0, 2.0, 3.0)
    dropped = SensorModel(SensorConfig(dropout_probability=1.0), seed=4).sample((1, 2, 3), 0.1)
    assert not dropped.available and dropped.measurement_m is None
    assert dropped.dropout_reason == "independent_dropout"


def test_random_walk_bias_is_reported_and_applied():
    cfg = SensorConfig(random_walk_std_m=0.1)
    model = SensorModel(cfg, seed=7)
    sample = model.sample((0.0, 0.0, 0.0), 0.1)
    assert np.allclose(np.asarray(sample.measurement_m), np.asarray(sample.bias_m) + sample.noise_m)
    assert sample.bias_m != cfg.bias_m


def test_constant_velocity_prediction_matches_half_second_hand_check():
    tracker = ConstantVelocityTracker((0, 0, 0), (1, 0, 0), timestamp_s=0.0,
                                       acceleration_spectral_density=0.0,
                                       position_variance_m2=0.0, velocity_variance_m2ps2=0.0,
                                       measurement_std_m=0.0)
    state = tracker.predict(0.5)
    assert state.position_m == (0.5, 0.0, 0.0)
    assert state.velocity_mps == (1.0, 0.0, 0.0)


def test_update_reduces_position_uncertainty_and_rejects_old_time():
    tracker = ConstantVelocityTracker(measurement_std_m=0.1)
    before = tracker.snapshot().covariance[0][0]
    tracker.update((0.0, 0.0, 0.0), 0.1)
    after = tracker.snapshot().covariance[0][0]
    assert after < before
    with pytest.raises(ValueError):
        tracker.update((0, 0, 0), 0.05)
    with pytest.raises(ValueError):
        tracker.update((0, 0, 0), 0.1)


def test_tracker_rejects_nonfinite_or_negative_noise_parameters():
    with pytest.raises(ValueError):
        ConstantVelocityTracker(measurement_std_m=-0.1)
    with pytest.raises(ValueError):
        ConstantVelocityTracker(acceleration_spectral_density=float("nan"))


def test_exactly_known_position_measurement_is_not_hidden_as_prediction():
    tracker = ConstantVelocityTracker(position_m=(0, 0, 0), velocity_mps=(1, 0, 0),
                                       acceleration_spectral_density=0.0,
                                       position_variance_m2=1.0, velocity_variance_m2ps2=1.0,
                                       measurement_std_m=0.01)
    tracker.predict(0.5)
    predicted = tracker.snapshot().position_m[0]
    tracker.update((0.5, 0.0, 0.0), 0.5)
    corrected = tracker.snapshot().position_m[0]
    assert abs(predicted - 0.5) < 1e-12
    assert abs(corrected - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Independent reference: from-scratch scalar-per-axis constant-velocity filter
# ---------------------------------------------------------------------------

def _scalar_kf_predict(p, v, P, dt, q):
    """One axis, plain floats. P = [[a, b], [b, c]]."""
    a, b, c = P
    p2 = p + v * dt
    # F P F^T with F = [[1, dt], [0, 1]], written out by hand.
    a2 = a + 2 * dt * b + dt * dt * c + q * dt ** 3 / 3.0
    b2 = b + dt * c + q * dt ** 2 / 2.0
    c2 = c + q * dt
    return p2, v, (a2, b2, c2)


def _scalar_kf_update(p, v, P, z, r):
    """Measure position only. Uses the simple (I - KH)P form, not Joseph."""
    a, b, c = P
    s = a + r
    k0, k1 = a / s, b / s
    innov = z - p
    p2, v2 = p + k0 * innov, v + k1 * innov
    return p2, v2, ((1 - k0) * a, (1 - k0) * b, c - k1 * b)


def test_kalman_filter_matches_independent_scalar_implementation():
    rng = np.random.default_rng(2026)
    q, r, pv, vv = 0.3, 0.05 ** 2, 2.0, 1.5
    tracker = ConstantVelocityTracker(
        position_m=(1.0, -2.0, 0.5), velocity_mps=(0.2, 0.0, -0.1),
        position_variance_m2=pv, velocity_variance_m2ps2=vv,
        acceleration_spectral_density=q, measurement_std_m=0.05)
    axes = []
    for i in range(3):
        axes.append([tracker.x[i], tracker.x[i + 3], (pv, 0.0, vv)])
    t = 0.0
    for step in range(60):
        dt = float(rng.uniform(0.01, 0.5))
        t += dt
        if step % 5 == 4:  # predict-only step (a dropped measurement)
            tracker.predict(t)
            for ax in axes:
                ax[0], ax[1], ax[2] = _scalar_kf_predict(ax[0], ax[1], ax[2], dt, q)
        else:
            z = rng.normal(0.0, 1.0, size=3)
            tracker.update(z, t)
            for i, ax in enumerate(axes):
                ax[0], ax[1], ax[2] = _scalar_kf_predict(ax[0], ax[1], ax[2], dt, q)
                ax[0], ax[1], ax[2] = _scalar_kf_update(ax[0], ax[1], ax[2], z[i], r)
    for i, (p, v, (a, b, c)) in enumerate(axes):
        assert abs(tracker.x[i] - p) < 1e-12
        assert abs(tracker.x[i + 3] - v) < 1e-12
        assert abs(tracker.P[i, i] - a) < 1e-12
        assert abs(tracker.P[i, i + 3] - b) < 1e-12
        assert abs(tracker.P[i + 3, i + 3] - c) < 1e-12
    # Axes are independent: every cross-axis covariance block stays zero.
    mask = np.ones((6, 6), dtype=bool)
    for i in range(3):
        for j in (i, i + 3):
            for k in (i, i + 3):
                mask[j, k] = False
    assert np.max(np.abs(tracker.P[mask])) < 1e-12


# ---------------------------------------------------------------------------
# the filter clock only moves forward; late data needs a copy to look ahead
# ---------------------------------------------------------------------------

def test_predicting_to_receive_time_then_updating_at_capture_time_is_rejected():
    tracker = ConstantVelocityTracker(measurement_std_m=0.1)
    tracker.predict(1.0)  # "predict to the receive time" advances the real clock
    with pytest.raises(ValueError):
        tracker.update((0.0, 0.0, 0.0), 0.5)  # capture time is now in the past


def test_documented_way_to_look_ahead_leaves_the_filter_clock_alone():
    import copy

    tracker = ConstantVelocityTracker(velocity_mps=(1.0, 0.0, 0.0), measurement_std_m=0.1)
    tracker.update((0.0, 0.0, 0.0), 0.2)
    ahead = copy.deepcopy(tracker).predict(1.0)  # display estimate only
    assert ahead.position_m[0] > tracker.snapshot().position_m[0]
    assert tracker.timestamp_s == 0.2
    assert tracker.update((0.3, 0.0, 0.0), 0.4)  # still accepts the next capture


# ---------------------------------------------------------------------------
# predict_at: look ahead without moving the filter clock (S6 recovery arm)
# ---------------------------------------------------------------------------

def test_predict_at_looks_ahead_without_changing_the_filter():
    tracker = ConstantVelocityTracker(velocity_mps=(1.0, 0.0, 0.0), measurement_std_m=0.1)
    tracker.update((0.0, 0.0, 0.0), 0.2)
    before = tracker.snapshot()
    ahead = tracker.predict_at(1.0)
    # Hand check: a constant-velocity estimate moves v x dt. The filter's
    # velocity after one update is still close to 1 m/s, so 0.8 s ahead the
    # x position grows by v_hat x 0.8 from the updated position.
    assert ahead.timestamp_s == 1.0
    assert ahead.position_m[0] == pytest.approx(before.position_m[0] + before.velocity_mps[0] * 0.8)
    assert tracker.snapshot() == before  # nothing in the filter moved
    assert tracker.update((0.3, 0.0, 0.0), 0.4)  # still accepts the next capture


def test_predict_at_equals_the_documented_deepcopy_look_ahead():
    import copy

    tracker = ConstantVelocityTracker(velocity_mps=(0.5, -0.2, 0.1), measurement_std_m=0.05)
    tracker.update((0.01, 0.0, 0.0), 0.1)
    tracker.update((0.06, -0.02, 0.01), 0.2)
    reference = copy.deepcopy(tracker).predict(0.55)
    ahead = tracker.predict_at(0.55)
    assert np.allclose(ahead.position_m, reference.position_m, atol=1e-12)
    assert np.allclose(ahead.velocity_mps, reference.velocity_mps, atol=1e-12)
    assert np.allclose(ahead.covariance, reference.covariance, atol=1e-12)


def test_predict_at_rejects_the_past():
    tracker = ConstantVelocityTracker(measurement_std_m=0.1)
    tracker.update((0.0, 0.0, 0.0), 0.5)
    with pytest.raises(ValueError):
        tracker.predict_at(0.4)
    with pytest.raises(ValueError):
        tracker.predict_at(float("nan"))
