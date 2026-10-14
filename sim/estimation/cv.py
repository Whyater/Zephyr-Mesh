"""Constant-velocity estimator: the S6 recovery tracker behind the common estimator interface.

This is a thin wrapper around ``sim.tracker.ConstantVelocityTracker``, built
exactly like ``sim.s6_recovery._new_tracker``, so the naive and recovery-v1
arms give the same numbers as the S6 recovery sweep down to the last bit. The wrapper adds
what the experiment engine needs:

- ``q_factor``, ``q_scale`` and ``r_scale`` multiply the process noise and
  the measurement noise. They are pushed into the wrapped tracker
  (``q_acc`` and ``R``) right before each update, so with every factor at 1
  the tracker runs with its original numbers.
- ``predict_at`` returns a ``base.Prediction`` and never touches the tracker.
- ``predict_positions`` answers many query times at once. Between two
  fusions a constant-velocity track is a straight line, p + v (t - t0), so
  one NumPy expression replaces a loop of single predictions.
"""
from __future__ import annotations

import copy

import numpy as np

from sim.tracker import ConstantVelocityTracker

from .base import Estimator, EstimatorSettings, Prediction

_EYE3 = np.eye(3)
_EYE3.setflags(write=False)


class CVEstimator(Estimator):
    """Constant-velocity Kalman filter, numerically identical to the S6 recovery tracker."""

    def __init__(self, start_position, settings: EstimatorSettings, *, q_factor=1.0, r_std_m=None,
                 timestamp_s=0.0, **model_options):
        if model_options:
            raise ValueError(f"unknown model options for cv: {sorted(model_options)}")
        if not np.isfinite(q_factor) or q_factor <= 0:
            raise ValueError("q_factor must be positive")
        r_std = float(settings.measurement_std_m if r_std_m is None else r_std_m)
        if not np.isfinite(r_std) or r_std < 0:
            raise ValueError("r_std_m must be finite and nonnegative")
        start = np.asarray(start_position, dtype=float)
        if start.shape != (3,):
            raise ValueError("start_position must have shape (3,)")
        self.settings = settings
        self.q_factor = float(q_factor)
        self.r_std_m = r_std
        self.q_scale = 1.0
        self.r_scale = 1.0
        self.tracker = ConstantVelocityTracker(
            position_m=start, velocity_mps=(0.0, 0.0, 0.0), timestamp_s=timestamp_s,
            position_variance_m2=settings.initial_position_variance_m2,
            velocity_variance_m2ps2=settings.initial_velocity_variance_m2ps2,
            acceleration_spectral_density=settings.q_acc_psd,
            measurement_std_m=r_std,
        )
        self.base_q = self.tracker.q_acc
        self.base_R = self.tracker.R.copy()

    @property
    def timestamp_s(self) -> float:
        return self.tracker.timestamp_s

    @property
    def update_count(self) -> int:
        return self.tracker.update_count

    def _q(self) -> float:
        return self.base_q * self.q_factor * self.q_scale

    def measurement_cov(self) -> np.ndarray:
        return self.base_R * self.r_scale

    def update(self, measurement_m, timestamp_s: float) -> None:
        self.tracker.q_acc = self._q()
        self.tracker.R = self.measurement_cov()
        self.tracker.update(measurement_m, timestamp_s)

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        # Same arithmetic, in the same order, as ConstantVelocityTracker._transition,
        # but with the scaled q passed in so predict_at never writes to the tracker.
        q = self._q()
        F = np.eye(6)
        F[:3, 3:] = _EYE3 * dt
        Q = np.zeros((6, 6))
        Q[:3, :3] = _EYE3 * (dt**3 / 3.0) * q
        Q[:3, 3:] = _EYE3 * (dt**2 / 2.0) * q
        Q[3:, :3] = Q[:3, 3:].T
        Q[3:, 3:] = _EYE3 * dt * q
        return F, Q

    def predict_at(self, timestamp_s: float) -> Prediction:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.tracker.timestamp_s:
            raise ValueError("prediction timestamp must be finite and not before the filter time")
        F, Q = self._transition(t - self.tracker.timestamp_s)
        x = F @ self.tracker.x
        P = F @ self.tracker.P @ F.T + Q
        P = (P + P.T) / 2.0
        return Prediction(t, x[:3].copy(), x[3:].copy(), P[:3, :3].copy())

    def predict_positions(self, times_s) -> np.ndarray:
        times = np.asarray(times_s, dtype=float).reshape(-1)
        t0 = self.tracker.timestamp_s
        if np.any(~np.isfinite(times)) or np.any(times < t0):
            raise ValueError("prediction times must be finite and not before the filter time")
        x = self.tracker.x
        return x[:3][None, :] + (times - t0)[:, None] * x[3:][None, :]

    def copy(self) -> "CVEstimator":
        # A hand-made copy is several times faster than deepcopy, and
        # retrodiction copies the filter once per fused packet. Only x, P and R
        # are ever replaced or written, so those are the arrays copied.
        new = copy.copy(self)
        tracker = copy.copy(self.tracker)
        tracker.x = self.tracker.x.copy()
        tracker.P = self.tracker.P.copy()
        tracker.R = self.tracker.R.copy()
        new.tracker = tracker
        return new
