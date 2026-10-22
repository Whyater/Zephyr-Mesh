"""Focused checks for the two-mode interacting multiple-model estimator."""
from __future__ import annotations

import numpy as np
import pytest

from sim.estimation.base import EstimatorSettings
from sim.estimation.imm import IMMEstimator
from sim.estimation.kinematics import cv7_matrix, ct_process_noise, position_update


def _measurements(kind: str):
    rng = np.random.default_rng(20261014)
    times = np.arange(0.0, 12.0 + 0.05, 0.1)
    if kind == "cv":
        truth = np.column_stack((times, np.zeros((times.size, 2))))
    else:
        truth = np.column_stack((np.sin(times), 1.0 - np.cos(times), np.zeros(times.size)))
    return times, truth + rng.normal(0.0, 0.05, truth.shape)


def test_probabilities_prediction_and_timestamps():
    filt = IMMEstimator(np.zeros(3), EstimatorSettings())
    before = (filt.timestamp_s, filt.update_count, filt.mode_probabilities, filt.mode_states)
    prediction = filt.predict_at(0.7)
    assert prediction.timestamp_s == 0.7
    assert np.allclose(filt.mode_probabilities, before[2])
    assert filt.timestamp_s == before[0]
    assert filt.update_count == before[1]
    for (x_a, P_a), (x_b, P_b) in zip(filt.mode_states, before[3]):
        assert np.array_equal(x_a, x_b)
        assert np.array_equal(P_a, P_b)

    filt.update(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        filt.update(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        filt.update(np.zeros(3), -0.1)
    for t in np.arange(0.1, 1.1, 0.1):
        filt.update(np.array([t, 0.0, 0.0]), float(t))
        mu = filt.mode_probabilities
        assert mu.shape == (2,)
        assert np.all(mu >= 0.0)
        assert np.all(mu <= 1.0)
        assert np.isclose(mu.sum(), 1.0)


def test_straight_motion_prefers_cv():
    times, measurements = _measurements("cv")
    filt = IMMEstimator(np.zeros(3), EstimatorSettings(), r_std_m=0.05)
    values = []
    for t, z in zip(times, measurements):
        filt.update(z, float(t))
        if t >= 4.0:
            values.append(filt.mode_probabilities[1])
    assert float(np.mean(values)) < 0.5


def test_circle_motion_prefers_ct():
    times, measurements = _measurements("ct")
    filt = IMMEstimator(np.zeros(3), EstimatorSettings(), r_std_m=0.05)
    values = []
    for t, z in zip(times, measurements):
        filt.update(z, float(t))
        if t >= 4.0:
            values.append(filt.mode_probabilities[1])
    assert float(np.mean(values)) > 0.5


def test_zero_transition_and_cv_probability_match_cv_filter():
    settings = EstimatorSettings(imm_mean_sojourn_s=1e300, imm_initial_ct_probability=0.0)
    imm = IMMEstimator(np.zeros(3), settings, r_std_m=0.05)
    x = np.zeros(7)
    P = np.eye(7)
    R = np.eye(3) * 0.05**2
    times, measurements = _measurements("cv")
    previous_t = 0.0
    for t, z in zip(times[:30], measurements[:30]):
        dt = float(t - previous_t)
        F = cv7_matrix(dt)
        Q = ct_process_noise(dt, settings.q_acc_psd, settings.q_turn_psd)
        x = F @ x
        P = F @ P @ F.T + Q
        P = (P + P.T) / 2.0
        x, P = position_update(x, P, z, R)
        imm.update(z, float(t))
        assert np.array_equal(imm.mode_probabilities, np.array([1.0, 0.0]))
        pred = imm.predict_at(float(t))
        assert np.allclose(pred.position_m, x[:3], atol=1e-9, rtol=0.0)
        assert np.allclose(pred.velocity_mps, x[3:6], atol=1e-9, rtol=0.0)
        assert np.allclose(pred.position_cov, P[:3, :3], atol=1e-9, rtol=0.0)
        previous_t = float(t)


def test_vectorized_positions_match_predict_at_for_ct_and_imm():
    """predict_positions answers many queries in one call; it must give the same
    positions as predict_at at each time (gaps from 1e-7 s to 7 s, after turns)."""
    from sim.estimation.factory import make_estimator
    rng = np.random.default_rng(7)
    for model in ("ct", "imm"):
        est = make_estimator(model, EstimatorSettings(), np.zeros(3))
        for k in range(1, 40):
            t = k * 0.1
            est.update(np.array([np.sin(t), 1.0 - np.cos(t), 0.0]) + rng.normal(0.0, 0.05, 3), t)
            q = t + np.array([0.0, 1e-7, 0.003, 0.05, 0.4, 2.5, 7.0])
            ref = np.array([est.predict_at(x).position_m for x in q])
            assert np.max(np.abs(est.predict_positions(q) - ref)) <= 1e-12
