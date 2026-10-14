"""Interacting multiple-model estimator with CV and constant-turn modes.

Both modes use the seven-state layout shared by the constant-turn model.  The
IMM mixes their posterior states before each step, updates each mode with the
same position measurement, and combines their results using the mode
probabilities.
"""
from __future__ import annotations

import numpy as np

from .base import Estimator, EstimatorSettings, Prediction
from .kinematics import (
    ct_jacobian,
    ct_positions,
    ct_process_noise,
    ct_step,
    cv7_matrix,
    gaussian_log_likelihood,
    position_update,
)


def switch_probability(dt, mean_sojourn_s: float):
    """Chance of being in the OTHER mode after a gap dt.

    Symmetric two-state chain that leaves each mode at rate 1 / tau: the chance
    is (1 - e^(-2 dt / tau)) / 2. It tends to 1/2 for long gaps (no memory
    left) and never goes above it. Works on a number or an array of gaps.
    """
    return 0.5 * (1.0 - np.exp(-2.0 * np.asarray(dt, dtype=float) / mean_sojourn_s))


class IMMEstimator(Estimator):
    """Two-mode IMM using CV (mode 0) and CT (mode 1)."""

    def __init__(self, start_position, settings: EstimatorSettings, *, q_factor=1.0,
                 r_std_m=None, timestamp_s=0.0):
        if not np.isfinite(q_factor) or q_factor <= 0:
            raise ValueError("q_factor must be positive")
        if r_std_m is not None and (not np.isfinite(r_std_m) or r_std_m < 0):
            raise ValueError("r_std_m must be nonnegative")
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not np.isfinite(settings.imm_mean_sojourn_s) or settings.imm_mean_sojourn_s <= 0:
            raise ValueError("imm_mean_sojourn_s must be positive")
        if not np.isfinite(settings.imm_initial_ct_probability) or not (
                0.0 <= settings.imm_initial_ct_probability <= 1.0):
            raise ValueError("imm_initial_ct_probability must be in [0, 1]")

        start = np.asarray(start_position, dtype=float)
        if start.shape != (3,) or not np.all(np.isfinite(start)):
            raise ValueError("start_position must have shape (3,)")

        self.settings = settings
        self.q_factor = float(q_factor)
        self.r_std_m = float(settings.measurement_std_m if r_std_m is None else r_std_m)
        self.timestamp_s = float(timestamp_s)
        self.update_count = 0
        self.q_scale = 1.0
        self.r_scale = 1.0

        x0 = np.zeros(7, dtype=float)
        x0[:3] = start
        P0 = np.zeros((7, 7), dtype=float)
        P0[:3, :3] = np.eye(3) * settings.initial_position_variance_m2
        P0[3:6, 3:6] = np.eye(3) * settings.initial_velocity_variance_m2ps2
        P0[6, 6] = settings.initial_turn_rate_variance
        self._mode_states = [(x0.copy(), P0.copy()), (x0.copy(), P0.copy())]
        p_ct = float(settings.imm_initial_ct_probability)
        self._mu = np.array([1.0 - p_ct, p_ct], dtype=float)

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Current probabilities ``[mu_cv, mu_ct]`` as a defensive copy."""
        return self._mu.copy()

    @property
    def mode_states(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Current mode states and covariances as defensive copies."""
        return [(x.copy(), P.copy()) for x, P in self._mode_states]

    def measurement_cov(self) -> np.ndarray:
        return np.eye(3) * (self.r_std_m ** 2) * self.r_scale

    def _transition(self, dt: float) -> np.ndarray:
        p = switch_probability(dt, self.settings.imm_mean_sojourn_s)
        return np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=float)

    def _predict_mode(self, mode: int, x: np.ndarray, P: np.ndarray,
                      dt: float) -> tuple[np.ndarray, np.ndarray]:
        if dt == 0.0:
            return x.copy(), P.copy()
        if mode == 0:
            F = cv7_matrix(dt)
            xp = F @ x
        else:
            F = ct_jacobian(x, dt)
            xp = ct_step(x, dt)
        Q = ct_process_noise(
            dt,
            self.settings.q_acc_psd * self.q_factor * self.q_scale,
            self.settings.q_turn_psd * self.q_factor * self.q_scale,
        )
        Pp = F @ P @ F.T + Q
        return xp, (Pp + Pp.T) / 2.0

    def _mixed_states(self, transition: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        weights = transition * self._mu[:, None]
        c = weights.sum(axis=0)
        mixed = []
        for j in range(2):
            if c[j] > 0.0:
                alpha = weights[:, j] / c[j]
                x0 = sum(alpha[i] * self._mode_states[i][0] for i in range(2))
                P0 = np.zeros((7, 7), dtype=float)
                for i in range(2):
                    dx = self._mode_states[i][0] - x0
                    P0 += alpha[i] * (self._mode_states[i][1] + np.outer(dx, dx))
            else:
                x0, P0 = self._mode_states[j][0].copy(), self._mode_states[j][1].copy()
            mixed.append((x0, (P0 + P0.T) / 2.0))
        return c, mixed

    @staticmethod
    def _combine(states: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        x = sum(weights[j] * states[j][0] for j in range(2))
        P = np.zeros((7, 7), dtype=float)
        for j in range(2):
            dx = states[j][0] - x
            P += weights[j] * (states[j][1] + np.outer(dx, dx))
        return x, (P + P.T) / 2.0

    def update(self, measurement_m, timestamp_s: float) -> None:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.timestamp_s or (t == self.timestamp_s and self.update_count > 0):
            raise ValueError("timestamp must advance")
        z = np.asarray(measurement_m, dtype=float)
        if z.shape != (3,) or not np.all(np.isfinite(z)):
            raise ValueError("measurement must have shape (3,)")

        dt = t - self.timestamp_s
        transition = self._transition(dt)
        c, mixed = self._mixed_states(transition)
        predicted = [self._predict_mode(j, mixed[j][0], mixed[j][1], dt) for j in range(2)]
        R = self.measurement_cov()
        log_weights = np.full(2, -np.inf, dtype=float)
        updated = []
        for j, (x_pred, P_pred) in enumerate(predicted):
            nu = z - x_pred[:3]
            S = P_pred[:3, :3] + R
            if c[j] > 0.0:
                log_weights[j] = np.log(c[j]) + gaussian_log_likelihood(nu, S)
            updated.append(position_update(x_pred, P_pred, z, R))

        finite = np.isfinite(log_weights)
        if np.any(finite):
            shifted = log_weights - np.max(log_weights[finite])
            new_mu = np.zeros(2, dtype=float)
            new_mu[finite] = np.exp(shifted[finite])
            new_mu /= new_mu.sum()
        else:
            new_mu = c / c.sum() if c.sum() > 0.0 else self._mu.copy()
        self._mode_states = [(x.copy(), P.copy()) for x, P in updated]
        self._mu = new_mu
        self.timestamp_s = t
        self.update_count += 1

    def predict_positions(self, times_s) -> np.ndarray:
        """Combined positions at many times in one call; equals predict_at's position.

        Only positions are needed here, so no covariances are propagated: each
        mode's state is moved forward in closed form and the two are blended
        with the same weights c_j(dt) that predict_at uses.
        """
        times = np.asarray(times_s, dtype=float)
        if np.any(times < self.timestamp_s - 1e-12):
            raise ValueError("cannot predict backwards")
        dts = np.maximum(times - self.timestamp_s, 0.0)
        p = switch_probability(dts, self.settings.imm_mean_sojourn_s)
        mu_cv, mu_ct = self._mu
        c_cv = (1.0 - p) * mu_cv + p * mu_ct
        c_ct = p * mu_cv + (1.0 - p) * mu_ct
        total = c_cv + c_ct
        c_cv, c_ct = c_cv / total, c_ct / total
        x_cv = self._mode_states[0][0]
        cv_pos = x_cv[:3] + np.outer(dts, x_cv[3:6])
        ct_pos = ct_positions(self._mode_states[1][0], dts)
        return c_cv[:, None] * cv_pos + c_ct[:, None] * ct_pos

    def predict_at(self, timestamp_s: float) -> Prediction:
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.timestamp_s:
            raise ValueError("cannot predict backwards")
        dt = t - self.timestamp_s
        transition = self._transition(dt)
        c, _ = self._mixed_states(transition)
        predicted = [self._predict_mode(j, self._mode_states[j][0], self._mode_states[j][1], dt)
                     for j in range(2)]
        if c.sum() <= 0.0:
            c = self._mu.copy()
        else:
            c = c / c.sum()
        x, P = self._combine(predicted, c)
        return Prediction(t, x[:3].copy(), x[3:6].copy(), P[:3, :3].copy())

