"""Unit checks for the NIS band, the consistency monitor and the adaptive wrapper."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from sim.estimation.base import Estimator, Prediction
from sim.estimation.consistency import (AdaptiveEstimator, ConsistencyMonitor,
                                        chi2_gate_threshold, nis_band)
from sim.tracker import ConstantVelocityTracker

DT = 0.1
R_STD = 0.05


class _TrackerAdapter(Estimator):
    """Local Estimator over the S6 recovery tracker, so this file needs no other estimation module."""

    def __init__(self, q_acc: float = 0.1, r_std: float = R_STD):
        self.tracker = ConstantVelocityTracker(
            acceleration_spectral_density=q_acc, measurement_std_m=r_std)
        self.base_q = self.tracker.q_acc
        self.base_R = self.tracker.R.copy()
        self.q_scale = 1.0
        self.r_scale = 1.0

    @property
    def timestamp_s(self) -> float:
        return self.tracker.timestamp_s

    @property
    def update_count(self) -> int:
        return self.tracker.update_count

    def _apply_scales(self) -> None:
        self.tracker.q_acc = self.base_q * self.q_scale
        self.tracker.R = self.base_R * self.r_scale

    def measurement_cov(self) -> np.ndarray:
        return self.base_R * self.r_scale

    def update(self, measurement_m, timestamp_s: float) -> None:
        self._apply_scales()
        self.tracker.update(measurement_m, timestamp_s)

    def predict_at(self, timestamp_s: float) -> Prediction:
        self._apply_scales()
        s = self.tracker.predict_at(timestamp_s)
        return Prediction(s.timestamp_s, np.array(s.position_m), np.array(s.velocity_mps),
                          np.array(s.covariance)[:3, :3])


def _run(filt: Estimator, truth: np.ndarray, noise_std: float, seed: int) -> Estimator:
    rng = np.random.default_rng(seed)
    z = truth + rng.normal(0.0, noise_std, truth.shape)
    for k in range(len(truth)):
        filt.update(z[k], k * DT)
    return filt


def _times(duration_s: float = 12.0) -> np.ndarray:
    return np.arange(0.0, duration_s + DT / 2, DT)


def _circle_truth(times: np.ndarray) -> np.ndarray:
    return np.column_stack((np.sin(times), 1.0 - np.cos(times), np.zeros(times.size)))


def _cwna_truth(n: int, q: float, seed: int) -> np.ndarray:
    """Positions from the exact discrete constant-velocity model with process noise Q(T)."""
    rng = np.random.default_rng(seed)
    Q = q * np.array([[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]])
    chol = np.linalg.cholesky(Q)
    F = np.array([[1.0, DT], [0.0, 1.0]])
    out = np.empty((n, 3))
    for axis in range(3):
        state = np.array([0.0, rng.normal()])
        for k in range(n):
            out[k, axis] = state[0]
            state = F @ state + chol @ rng.normal(size=2)
    return out


def test_nis_band_single_and_window():
    low, high = nis_band(1)
    # (1 - 0.95) / 2 is one rounding step from 0.025, so compare to 1e-12, not bit for bit.
    assert low == pytest.approx(chi2.ppf(0.025, 3), rel=1e-12)
    assert high == pytest.approx(chi2.ppf(0.975, 3), rel=1e-12)
    previous = (0.0, np.inf)
    for window in (1, 2, 5, 10, 50, 200):
        low, high = nis_band(window)
        assert low < 3.0 < high
        assert low > previous[0] and high < previous[1]
        previous = (low, high)
    assert nis_band(10, confidence=0.99)[1] > nis_band(10)[1]


def test_gate_threshold_and_bad_inputs():
    assert chi2_gate_threshold(0.99) == chi2.ppf(0.99, 3)
    assert chi2_gate_threshold(0.95, dof=1) == chi2.ppf(0.95, 1)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            chi2_gate_threshold(bad)
    with pytest.raises(ValueError):
        nis_band(0)
    with pytest.raises(ValueError):
        nis_band(5, confidence=1.0)
    with pytest.raises(ValueError):
        ConsistencyMonitor(window=10).add(float("nan"))


def _flag_counts(scale: float, windows: int = 2000, window: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    monitor = ConsistencyMonitor(window=window)
    labels = []
    for value in scale * rng.chisquare(3, size=windows * window):
        label = monitor.add(value)
        if label is not None:
            labels.append(label)
    assert len(monitor.window_means) == windows
    assert [lab for _, lab in monitor.warnings] == labels
    return labels


def test_monitor_flags_about_five_percent_when_consistent():
    labels = _flag_counts(1.0)
    assert 0.03 <= len(labels) / 2000 <= 0.07


def test_monitor_flags_scaled_nis():
    over = _flag_counts(3.0)
    under = _flag_counts(1.0 / 3.0)
    assert len(over) > 0.98 * 2000 and set(over) == {"overconfident"}
    assert len(under) > 0.98 * 2000 and set(under) == {"underconfident"}


def test_monitor_only_evaluates_full_windows():
    monitor = ConsistencyMonitor(window=4)
    assert [monitor.add(100.0) for _ in range(3)] == [None, None, None]
    assert monitor.window_means == []
    assert monitor.add(100.0) == "overconfident"
    assert monitor.window_means == [100.0] and monitor.warnings == [(0, "overconfident")]


def test_adaptive_q_rises_when_q_is_far_too_small():
    times = _times()
    filt = AdaptiveEstimator(_TrackerAdapter(q_acc=0.1 / 100.0), mode="q")
    _run(filt, _circle_truth(times), R_STD, seed=1)
    assert filt.q_scale > 1.0
    assert filt.r_scale == 1.0
    assert filt.scale_history[-1][1] == filt.q_scale
    assert len(filt.scale_history) == len(times) // 10


def test_adaptive_q_stays_near_one_when_model_is_right():
    times = _times(30.0)
    for seed in range(5):
        filt = AdaptiveEstimator(_TrackerAdapter(), mode="q")
        _run(filt, _cwna_truth(times.size, 0.1, seed), R_STD, seed=100 + seed)
        assert 0.25 <= filt.q_scale <= 4.0, (seed, filt.q_scale)


def test_adaptive_r_rises_when_r_is_too_small():
    times = _times()
    truth = np.column_stack((times, np.zeros(times.size), np.zeros(times.size)))
    filt = AdaptiveEstimator(_TrackerAdapter(r_std=R_STD / np.sqrt(5.0)), mode="r")
    _run(filt, truth, R_STD, seed=3)
    assert filt.r_scale > 1.0
    assert filt.q_scale == 1.0


def test_rule_is_clipped_and_window_aligned():
    truth = np.zeros((25, 3))
    # Measurements 100 sigma away give huge NIS: the factor is clipped at 2 per window.
    filt = AdaptiveEstimator(_TrackerAdapter(), mode="q", window=10)
    for k in range(25):
        filt.update(truth[k] + 5.0 * (k % 2), k * DT)
    assert [s for _, s in filt.scale_history] == [2.0, 4.0]
    assert [t for t, _ in filt.scale_history] == [9 * DT, 19 * DT]
    capped = AdaptiveEstimator(_TrackerAdapter(), mode="q", window=2, scale_clip=(0.01, 3.0))
    for k in range(20):
        capped.update(truth[k] + 5.0 * (k % 2), k * DT)
    assert capped.q_scale == 3.0


def test_delegation_and_bad_arguments():
    base = _TrackerAdapter()
    filt = AdaptiveEstimator(base, mode="r")
    assert filt.timestamp_s == 0.0 and filt.update_count == 0
    filt.update(np.zeros(3), 0.0)
    assert filt.update_count == 1
    assert np.array_equal(filt.measurement_cov(), base.measurement_cov())
    assert np.array_equal(filt.predict_at(0.3).position_m, base.predict_at(0.3).position_m)
    assert filt.predict_positions([0.1, 0.2]).shape == (2, 3)
    with pytest.raises(ValueError):
        filt.update(np.zeros(3), 0.0)  # duplicate stamp: raised by the base, buffer untouched
    with pytest.raises(ValueError):
        AdaptiveEstimator(base, mode="s")
    with pytest.raises(ValueError):
        AdaptiveEstimator(base, mode="q", window=0)
    with pytest.raises(ValueError):
        AdaptiveEstimator(base, mode="q", factor_clip=(2.0, 0.5))


def test_copy_deep_copies_the_base():
    filt = AdaptiveEstimator(_TrackerAdapter(), mode="q", window=3)
    for k in range(5):
        filt.update(np.array([0.01 * k, 0.0, 0.0]), k * DT)
    clone = filt.copy()
    assert clone is not filt and clone.base is not filt.base
    assert clone.base.tracker is not filt.base.tracker
    assert np.array_equal(clone.base.tracker.x, filt.base.tracker.x)
    before_x = filt.base.tracker.x.copy()
    before_hist = list(filt.scale_history)
    for k in range(5, 12):
        clone.update(np.array([3.0, 0.0, 0.0]), k * DT)
    assert np.array_equal(filt.base.tracker.x, before_x)
    assert filt.update_count == 5 and filt.scale_history == before_hist
    assert filt.q_scale == before_hist[-1][1] and clone.q_scale != filt.q_scale


@pytest.mark.parametrize("window", [1.2, 2.9, 0, True])
def test_monitor_rejects_non_integer_windows(window):
    """window=1.2 used to be silently truncated to 1."""
    with pytest.raises(ValueError):
        ConsistencyMonitor(window=window)


def test_nis_band_extreme_confidence_stays_finite():
    """At confidence 1 - 1e-12 the tails are 5e-13 each. Rounding them away used to give (0, inf);
    the band must stay finite and positive, and match scipy's own tail quantiles."""
    from scipy.stats import chi2
    low, high = nis_band(1, confidence=0.999999999999)
    tail = (1.0 - 0.999999999999) / 2.0
    assert 0.0 < low < high < np.inf
    assert low == pytest.approx(chi2.ppf(tail, 3), rel=1e-12)
    assert high == pytest.approx(chi2.isf(tail, 3), rel=1e-12)
