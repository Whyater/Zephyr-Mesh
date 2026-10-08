"""Small, inspectable constant-velocity Kalman tracker.

The filter is an estimator fixture. It is not a flight controller and does
not infer a camera model. Position and velocity are kept in one world frame.
Measurements must be supplied in chronological capture-time order; callers
can predict to a later receive time to make observation age explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackerState:
    timestamp_s: float
    position_m: tuple[float, ...]
    velocity_mps: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    update_count: int
    prediction_count: int

    def as_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps),
            "covariance": [list(row) for row in self.covariance],
            "update_count": self.update_count,
            "prediction_count": self.prediction_count,
        }


class ConstantVelocityTracker:
    """Independent-axis 3D constant-velocity Kalman filter.

    ``acceleration_spectral_density`` is the continuous white-acceleration
    intensity used to build Q. ``measurement_std_m`` applies to each position
    coordinate. The Joseph covariance update keeps symmetry under rounding.
    """

    def __init__(
        self,
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
        timestamp_s=0.0,
        *,
        position_variance_m2=1.0,
        velocity_variance_m2ps2=1.0,
        acceleration_spectral_density=0.1,
        measurement_std_m=0.05,
    ):
        self.x = np.r_[np.asarray(position_m, dtype=float), np.asarray(velocity_mps, dtype=float)]
        if self.x.shape != (6,) or np.any(~np.isfinite(self.x)):
            raise ValueError("position and velocity must be finite 3-vectors")
        self.timestamp_s = float(timestamp_s)
        self.position_variance = float(position_variance_m2)
        self.velocity_variance = float(velocity_variance_m2ps2)
        self.q_acc = float(acceleration_spectral_density)
        measurement_std_m = float(measurement_std_m)
        self.measurement_variance = measurement_std_m ** 2
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        if (not np.isfinite(self.position_variance) or not np.isfinite(self.velocity_variance)
                or not np.isfinite(self.q_acc) or not np.isfinite(measurement_std_m)):
            raise ValueError("variances, acceleration intensity, and measurement std must be finite")
        if min(self.position_variance, self.velocity_variance, self.q_acc, measurement_std_m) < 0:
            raise ValueError("variances, acceleration intensity, and measurement std must be nonnegative")
        self.P = np.diag([self.position_variance] * 3 + [self.velocity_variance] * 3)
        self.update_count = 0
        self.prediction_count = 0
        self.H = np.c_[np.eye(3), np.zeros((3, 3))]
        self.R = np.eye(3) * self.measurement_variance

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.q_acc
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt**3 / 3.0) * q
        Q[:3, 3:] = np.eye(3) * (dt**2 / 2.0) * q
        Q[3:, :3] = Q[:3, 3:].T
        Q[3:, 3:] = np.eye(3) * dt * q
        return F, Q

    def predict(self, timestamp_s: float) -> TrackerState:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.timestamp_s:
            raise ValueError("prediction timestamp must be finite and nondecreasing")
        dt = t - self.timestamp_s
        F, Q = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) / 2.0
        self.timestamp_s = t
        if dt > 0:
            self.prediction_count += 1
        return self.snapshot()

    def update(self, measurement_m, timestamp_s: float) -> bool:
        z = np.asarray(measurement_m, dtype=float)
        t = float(timestamp_s)
        if z.shape != (3,) or np.any(~np.isfinite(z)):
            raise ValueError("measurement must be a finite 3-vector")
        if not np.isfinite(t) or t < 0:
            raise ValueError("measurement timestamp must be finite and nonnegative")
        if t < self.timestamp_s:
            raise ValueError("out-of-order measurements are rejected; caller must reorder or drop")
        if t == self.timestamp_s and self.update_count > 0:
            raise ValueError("duplicate measurement timestamp; caller must deduplicate")
        self.predict(t)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = np.linalg.solve(S, self.H @ self.P).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular measurement covariance") from exc
        self.x = self.x + K @ innovation
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2.0
        self.update_count += 1
        return True

    def snapshot(self) -> TrackerState:
        return TrackerState(
            float(self.timestamp_s),
            tuple(self.x[:3].tolist()),
            tuple(self.x[3:].tolist()),
            tuple(tuple(row) for row in self.P.tolist()),
            self.update_count,
            self.prediction_count,
        )
