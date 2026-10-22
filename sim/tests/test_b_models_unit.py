import copy

import numpy as np
import pytest

from sim.estimation.base import EstimatorSettings
from sim.estimation.ca import CAEstimator
from sim.estimation.ct import CTEstimator


def _settings():
    return EstimatorSettings(measurement_std_m=1e-4)


def test_ca_constant_acceleration_converges():
    est = CAEstimator([0, 0, 0], _settings(), r_std_m=1e-4)
    for t in np.arange(0.0, 10.0 + 0.05, 0.1):
        est.update([t * t / 2.0, 0.0, 0.0], float(t))
    assert abs(est.x[0] - 50.0) < 1e-3
    assert abs(est.x[6] - 1.0) < 1e-2


def test_ct_known_turn_rate_circle():
    est = CTEstimator([0, 0, 0], _settings(), r_std_m=1e-4, known_turn_rate=1.0)
    for t in np.arange(0.0, 3.0 + 0.05, 0.1):
        est.update([np.sin(t), 1.0 - np.cos(t), 0.0], float(t))
    p = est.predict_at(3.3).position_m
    truth = np.array([np.sin(3.3), 1.0 - np.cos(3.3), 0.0])
    assert np.linalg.norm(p - truth) < 1e-3


def test_ct_unknown_turn_rate_is_estimated():
    est = CTEstimator([0, 0, 0], _settings(), r_std_m=1e-4)
    for t in np.arange(0.0, 6.0 + 0.05, 0.1):
        est.update([np.sin(t), 1.0 - np.cos(t), 0.0], float(t))
    assert abs(est.x[6] - 1.0) < 0.05


def test_ct_near_zero_turn_matches_cv():
    est = CTEstimator([0, 0, 0], _settings(), known_turn_rate=1e-10)
    est.x[3] = 2.0
    est.x[4] = -0.5
    got = est.predict_at(0.1).position_m
    assert np.allclose(got, [0.2, -0.05, 0.0], atol=1e-9)


def test_timestamp_validation_and_prediction_is_non_mutating():
    est = CAEstimator([0, 0, 0], _settings())
    est.update([0, 0, 0], 0.0)
    with pytest.raises(ValueError):
        est.update([0, 0, 0], 0.0)
    with pytest.raises(ValueError):
        est.update([0, 0, 0], -0.1)
    before = copy.deepcopy(est.__dict__)
    est.predict_at(1.0)
    after = est.__dict__
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, after[key])
        else:
            assert value == after[key]


def test_q_scale_changes_process_covariance():
    a = CAEstimator([0, 0, 0], _settings())
    b = CAEstimator([0, 0, 0], _settings())
    a.q_scale = 1.0
    b.q_scale = 2.0
    pa = a.predict_at(0.1).position_cov
    pb = b.predict_at(0.1).position_cov
    assert np.all(pb >= pa)
    assert pb[0, 0] - pa[0, 0] > 0


@pytest.mark.parametrize("model", ["ca", "ct"])
def test_bad_updates_raise_and_leave_the_filter_unchanged(model):
    """A NaN stamp, a stamp slightly in the past, or a wrongly shaped measurement must raise
    ValueError before the filter moves, so the state after the failed call is unchanged."""
    import copy
    from sim.estimation.base import EstimatorSettings
    from sim.estimation.factory import make_estimator
    est = make_estimator(model, EstimatorSettings(), np.zeros(3))
    est.update(np.array([0.1, 0.0, 0.0]), 0.1)
    before = copy.deepcopy(est)
    for z, t in ((np.zeros(3), float("nan")), (np.zeros(3), 0.1 - 1e-13), (np.zeros(2), 0.2), (np.array([np.nan, 0, 0]), 0.2)):
        with pytest.raises(ValueError):
            est.update(z, t)
        assert est.timestamp_s == before.timestamp_s and est.update_count == before.update_count
        assert np.array_equal(est.x, before.x) and np.array_equal(est.P, before.P)


@pytest.mark.parametrize("model", ["cv", "ca", "ct", "imm"])
def test_predict_at_time_contract_is_the_same_for_every_model(model):
    """base.py: predictions are only for times at or after the filter time. Every model must
    reject a time 5e-13 s in the past, NaN and infinity the same way (CA and CT used to allow
    a 1e-12 s backward slack that CV and IMM did not)."""
    from sim.estimation.base import EstimatorSettings
    from sim.estimation.factory import make_estimator
    est = make_estimator(model, EstimatorSettings(), np.zeros(3))
    est.update(np.array([0.1, 0.0, 0.0]), 0.1)
    for t in (0.1 - 5e-13, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            est.predict_at(t)
    est.predict_at(0.1)
