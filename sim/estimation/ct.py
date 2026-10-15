"""Extended Kalman estimator for constant turn motion in the x-y plane."""
from __future__ import annotations

import numpy as np

from .base import Estimator, EstimatorSettings, Prediction
from .kinematics import ct_jacobian, ct_positions, ct_process_noise, ct_step, position_update


class CTEstimator(Estimator):
    """Seven-state EKF with a constant-turn-rate state."""

    def __init__(self, start_position, settings: EstimatorSettings, *, q_factor=1.0,
                 r_std_m=None, timestamp_s=0.0, **model_options):
        if not np.isfinite(float(timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if q_factor <= 0 or not np.isfinite(q_factor):
            raise ValueError("q_factor must be positive")
        unknown = set(model_options) - {"known_turn_rate"}
        if unknown:
            raise ValueError(f"unknown model options: {tuple(sorted(unknown))}")
        self.settings = settings
        self.q_factor = float(q_factor)
        self.r_std_m = float(settings.measurement_std_m if r_std_m is None else r_std_m)
        if self.r_std_m < 0 or not np.isfinite(self.r_std_m):
            raise ValueError("r_std_m must be nonnegative")
        self.timestamp_s = float(timestamp_s)
        self.update_count = 0
        self.q_scale = 1.0
        self.r_scale = 1.0
        self.known_turn_rate = model_options.get("known_turn_rate", None)
        self.q_turn = 0.0 if self.known_turn_rate is not None else settings.q_turn_psd * self.q_factor
        self.x = np.zeros(7, dtype=float)
        self.x[:3] = np.asarray(start_position, dtype=float)
        if self.x[:3].shape != (3,):
            raise ValueError("start_position must have shape (3,)")
        if self.known_turn_rate is not None:
            self.x[6] = float(self.known_turn_rate)
        self.P = np.zeros((7, 7), dtype=float)
        self.P[:3, :3] = np.eye(3) * settings.initial_position_variance_m2
        self.P[3:6, 3:6] = np.eye(3) * settings.initial_velocity_variance_m2ps2
        self.P[6, 6] = 0.0 if self.known_turn_rate is not None else settings.initial_turn_rate_variance

    def measurement_cov(self):
        return np.eye(3) * (self.r_std_m ** 2) * self.r_scale

    def _predict_state(self, x, P, dt):
        if dt == 0:
            return x.copy(), P.copy()
        F = ct_jacobian(x, dt)
        Q = ct_process_noise(dt, self.settings.q_acc_psd * self.q_factor * self.q_scale,
                              self.q_turn * self.q_scale)
        xp = ct_step(x, dt)
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
        xp, Pp = self._predict_state(self.x, self.P, t - self.timestamp_s)
        return Prediction(t, xp[:3].copy(), xp[3:6].copy(), Pp[:3, :3].copy())

    def predict_positions(self, times_s) -> np.ndarray:
        times = np.asarray(times_s, dtype=float)
        if np.any(times < self.timestamp_s - 1e-12):
            raise ValueError("cannot predict backwards")
        return ct_positions(self.x, np.maximum(times - self.timestamp_s, 0.0))
