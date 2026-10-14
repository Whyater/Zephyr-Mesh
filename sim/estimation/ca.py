"""Constant-acceleration Kalman estimator for three-dimensional positions."""
from __future__ import annotations

import numpy as np

from .base import Estimator, EstimatorSettings, Prediction
from .kinematics import ca_matrices, position_update


class CAEstimator(Estimator):
    """Linear Kalman filter with position, velocity and acceleration states."""

    def __init__(self, start_position, settings: EstimatorSettings, *, q_factor=1.0,
                 r_std_m=None, timestamp_s=0.0, **model_options):
        if not np.isfinite(float(timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if q_factor <= 0 or not np.isfinite(q_factor):
            raise ValueError("q_factor must be positive")
        if model_options:
            raise ValueError(f"unknown model options: {tuple(model_options)}")
        self.settings = settings
        self.q_factor = float(q_factor)
        self.r_std_m = float(settings.measurement_std_m if r_std_m is None else r_std_m)
        if self.r_std_m < 0 or not np.isfinite(self.r_std_m):
            raise ValueError("r_std_m must be nonnegative")
        self.timestamp_s = float(timestamp_s)
        self.update_count = 0
        self.q_scale = 1.0
        self.r_scale = 1.0
        self.x = np.zeros(9, dtype=float)
        self.x[:3] = np.asarray(start_position, dtype=float)
        if self.x[:3].shape != (3,):
            raise ValueError("start_position must have shape (3,)")
        self.P = np.zeros((9, 9), dtype=float)
        self.P[:3, :3] = np.eye(3) * settings.initial_position_variance_m2
        self.P[3:6, 3:6] = np.eye(3) * settings.initial_velocity_variance_m2ps2
        self.P[6:9, 6:9] = np.eye(3) * settings.initial_acceleration_variance

    def measurement_cov(self):
        return np.eye(3) * (self.r_std_m ** 2) * self.r_scale

    def _predict_state(self, x, P, dt):
        F, Q = ca_matrices(dt, self.settings.q_jerk_psd * self.q_factor * self.q_scale)
        xp = F @ x
        Pp = F @ P @ F.T + Q
        return xp, (Pp + Pp.T) / 2.0

    def update(self, measurement_m, timestamp_s: float) -> None:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.timestamp_s or (t == self.timestamp_s and self.update_count > 0):
            raise ValueError("timestamp must advance")
        z = np.asarray(measurement_m, dtype=float)
        if z.shape != (3,) or not np.all(np.isfinite(z)):
            raise ValueError("measurement must have shape (3,)")
        # Validate everything before touching the state, so a bad call leaves the filter as it was.
        if t > self.timestamp_s:
            self.x, self.P = self._predict_state(self.x, self.P, t - self.timestamp_s)
            self.timestamp_s = t
        self.x, self.P = position_update(self.x, self.P, z, self.measurement_cov())
        self.update_count += 1

    def predict_at(self, timestamp_s: float) -> Prediction:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.timestamp_s:
            raise ValueError("cannot predict backwards")
        xp, Pp = self._predict_state(self.x.copy(), self.P.copy(), t - self.timestamp_s)
        return Prediction(t, xp[:3].copy(), xp[3:6].copy(), Pp[:3, :3].copy())

    def predict_positions(self, times_s) -> np.ndarray:
        times = np.asarray(times_s, dtype=float)
        if np.any(times < self.timestamp_s - 1e-12):
            raise ValueError("cannot predict backwards")
        dt = times - self.timestamp_s
        return (self.x[:3][None, :] + dt[:, None] * self.x[3:6][None, :] +
                0.5 * dt[:, None] ** 2 * self.x[6:9][None, :])
