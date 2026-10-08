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
