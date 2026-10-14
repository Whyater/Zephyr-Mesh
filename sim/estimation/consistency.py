"""Is the filter honest about its own uncertainty? NIS checks and a self-tuning wrapper.

When a Kalman filter's model is right, the normalized innovation squared
(NIS, nu' S^-1 nu) of every measurement follows a chi-square law with 3
degrees of freedom (the 3 position axes), so it averages 3. Averaging
``window`` independent NIS values gives a number that sits inside a known band
95 % of the time. If the window mean lands above the band, the filter is
overconfident (it is surprised too often). If it lands below, the filter is
underconfident (its error bars are wider than they need to be).

This module holds three things:

- ``nis_band`` and ``chi2_gate_threshold``: the band for a window mean and the
  cut-off used by the chi-square gate.
- ``ConsistencyMonitor``: flags each full, non-overlapping window of NIS values.
- ``AdaptiveEstimator``: wraps any estimator and, after every ``window`` fused
  measurements, nudges either its process noise (mode "q") or its measurement
  noise (mode "r") by the ratio of the mean NIS to its expected value of 3.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .base import Estimator, Prediction

OVERCONFIDENT = "overconfident"
UNDERCONFIDENT = "underconfident"
ADAPTIVE_MODES = ("q", "r")


def _check_dof(dof: int) -> int:
    if int(dof) != dof or dof < 1:
        raise ValueError("dof must be a positive integer")
    return int(dof)


def nis_band(window: int, dof: int = 3, confidence: float = 0.95) -> tuple[float, float]:
    """Interval holding the MEAN of ``window`` independent NIS values with probability ``confidence``.

    The sum of ``window`` chi-square(dof) values is chi-square(dof * window),
    so the band is the two tail quantiles of that law divided by ``window``.
    """
    if int(window) != window or window < 1:
        raise ValueError("window must be a positive integer")
    dof = _check_dof(dof)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    window = int(window)
    tail = (1.0 - confidence) / 2.0
    # isf for the upper end keeps precision when the tail is tiny (1 - tail would round to 1).
    return (float(chi2.ppf(tail, dof * window) / window),
            float(chi2.isf(tail, dof * window) / window))


def chi2_gate_threshold(probability: float, dof: int = 3) -> float:
    """NIS value above which a measurement is rejected by a gate of the given probability."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return float(chi2.ppf(probability, _check_dof(dof)))


class ConsistencyMonitor:
    """Flags full, non-overlapping windows of NIS values whose mean leaves the band.

    ``warnings`` holds (window index, label) pairs, with window indices
    counted from 0. ``window_means`` holds the mean of every full window.
    """

    def __init__(self, window: int = 10, dof: int = 3, confidence: float = 0.95):
        if isinstance(window, bool) or int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = int(window)
        self.dof = _check_dof(dof)
        self.confidence = float(confidence)
        self.low, self.high = nis_band(self.window, self.dof, self.confidence)
        self.warnings: list[tuple[int, str]] = []
        self.window_means: list[float] = []
        self._buffer: list[float] = []

    def add(self, nis: float) -> str | None:
        """Record one NIS value. Returns a label when this value completes a flagged window."""
        value = float(nis)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("nis must be finite and nonnegative")
        self._buffer.append(value)
        if len(self._buffer) < self.window:
            return None
        mean = float(np.mean(self._buffer))
        self._buffer = []
        index = len(self.window_means)
        self.window_means.append(mean)
        label = OVERCONFIDENT if mean > self.high else UNDERCONFIDENT if mean < self.low else None
        if label is not None:
            self.warnings.append((index, label))
        return label


class AdaptiveEstimator(Estimator):
    """Wraps an estimator and retunes its noise from its own NIS.

    Each fused measurement's NIS is computed before it is fused, with the
    scales in force at that moment. After every ``window`` fused measurements
    the mean NIS divided by ``dof`` is clipped to ``factor_clip`` and
    multiplies the scale (clipped to ``scale_clip``), then the buffer is
    emptied. Mode "q" moves ``q_scale`` (a surprised filter is told to allow
    more motion). Mode "r" moves ``r_scale``.
    """

    def __init__(self, base: Estimator, mode: str, window: int = 10,
                 factor_clip: tuple[float, float] = (0.5, 2.0),
                 scale_clip: tuple[float, float] = (0.01, 100.0), dof: int = 3):
        if mode not in ADAPTIVE_MODES:
            raise ValueError(f"mode must be one of {ADAPTIVE_MODES}")
        if int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        for name, (low, high) in (("factor_clip", factor_clip), ("scale_clip", scale_clip)):
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must satisfy 0 < low <= high")
        self.base = base
        self.mode = mode
        self.window = int(window)
        self.dof = _check_dof(dof)
        self.factor_clip = (float(factor_clip[0]), float(factor_clip[1]))
        self.scale_clip = (float(scale_clip[0]), float(scale_clip[1]))
        self.scale_history: list[tuple[float, float]] = []
        self._nis_buffer: list[float] = []

    @property
    def timestamp_s(self) -> float:
        return self.base.timestamp_s

    @property
    def update_count(self) -> int:
        return self.base.update_count

    @property
    def q_scale(self) -> float:
        return self.base.q_scale

    @q_scale.setter
    def q_scale(self, value: float) -> None:
        self.base.q_scale = value

    @property
    def r_scale(self) -> float:
        return self.base.r_scale

    @r_scale.setter
    def r_scale(self, value: float) -> None:
        self.base.r_scale = value

    @property
    def scale(self) -> float:
        """The scale this wrapper moves (q_scale in mode "q", r_scale in mode "r")."""
        return self.base.q_scale if self.mode == "q" else self.base.r_scale

    def measurement_cov(self) -> np.ndarray:
        return self.base.measurement_cov()

    def predict_at(self, timestamp_s: float) -> Prediction:
        return self.base.predict_at(timestamp_s)

    def predict_positions(self, times_s) -> np.ndarray:
        return self.base.predict_positions(times_s)

    def update(self, measurement_m, timestamp_s: float) -> None:
        nis = self.base.nis(measurement_m, timestamp_s)
        self.base.update(measurement_m, timestamp_s)  # raises on a bad time, before any bookkeeping
        self._nis_buffer.append(nis)
        if len(self._nis_buffer) < self.window:
            return
        ratio = float(np.mean(self._nis_buffer)) / self.dof
        self._nis_buffer = []
        factor = float(np.clip(ratio, *self.factor_clip))
        scale = float(np.clip(self.scale * factor, *self.scale_clip))
        if self.mode == "q":
            self.base.q_scale = scale
        else:
            self.base.r_scale = scale
        self.scale_history.append((float(timestamp_s), scale))

    def copy(self) -> "AdaptiveEstimator":
        new = object.__new__(AdaptiveEstimator)
        new.__dict__.update(self.__dict__)
        new.base = self.base.copy()
        new.scale_history = list(self.scale_history)
        new._nis_buffer = list(self._nis_buffer)
        return new
