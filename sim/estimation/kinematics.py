"""Motion-model math shared by the CA, CT and IMM estimators.

State layouts (positions first, like the CV tracker):
- CV: [px, py, pz, vx, vy, vz]
- CA: [px, py, pz, vx, vy, vz, ax, ay, az]
- CT and the IMM: [px, py, pz, vx, vy, vz, w], where w is the turn rate in
  the x-y plane (rad/s, positive = turning left, about +z). z is constant velocity.

Process noise comes from continuous white noise driving the highest
derivative (Bar-Shalom, Li and Kirubarajan 2001, the CWNA and white-jerk
models). For one axis over a step dt:
- white acceleration of intensity q on (p, v): q [[dt^3/3, dt^2/2], [dt^2/2, dt]]
- white jerk of intensity q on (p, v, a):
  q [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]
"""
from __future__ import annotations

import numpy as np

# Below this |w dt| the turn coefficients use a Taylor series, because
# sin(w dt) / w loses digits as w goes to zero.
SMALL_TURN = 1e-4


def cwna_block(dt: float, q: float) -> np.ndarray:
    """2x2 process noise on (position, velocity) for one axis."""
    return q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])


def white_jerk_block(dt: float, q: float) -> np.ndarray:
    """3x3 process noise on (position, velocity, acceleration) for one axis."""
    return q * np.array([
        [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
        [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
        [dt**3 / 6.0, dt**2 / 2.0, dt],
    ])


def ca_matrices(dt: float, q_jerk: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition F (9x9) and process noise Q (9x9) for the constant-acceleration model."""
    I3 = np.eye(3)
    F = np.eye(9)
    F[0:3, 3:6] = I3 * dt
    F[0:3, 6:9] = I3 * (dt**2 / 2.0)
    F[3:6, 6:9] = I3 * dt
    block = white_jerk_block(dt, q_jerk)
    Q = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            Q[3 * i:3 * i + 3, 3 * j:3 * j + 3] = I3 * block[i, j]
    return F, Q


def turn_coefficients(w: float, dt: float) -> tuple[float, float, float, float, float, float]:
    """A = sin(w dt)/w, B = (1 - cos(w dt))/w, their w-derivatives, and s, c.

    Returns (A, B, dA_dw, dB_dw, s, c) with s = sin(w dt), c = cos(w dt).
    As w goes to 0, A goes to dt and B goes to 0 (straight-line motion).
    """
    u = w * dt
    s, c = np.sin(u), np.cos(u)
    if abs(u) < SMALL_TURN:
        A = dt * (1.0 - u**2 / 6.0 + u**4 / 120.0)
        B = dt * (u / 2.0 - u**3 / 24.0 + u**5 / 720.0)
        dA = dt**2 * (-u / 3.0 + u**3 / 30.0)
        dB = dt**2 * (0.5 - u**2 / 8.0 + u**4 / 144.0)
    else:
        A = s / w
        B = (1.0 - c) / w
        dA = (dt * c - A) / w
        dB = (dt * s - B) / w
    return float(A), float(B), float(dA), float(dB), float(s), float(c)


def ct_positions(x: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """Positions (N, 3) of the constant-turn state x after each gap in dts, in one call.

    The same closed form as ct_step, done for many gaps at once so a
    tracker can answer every query between two packets without a Python loop.
    """
    dts = np.asarray(dts, dtype=float)
    w = float(x[6])
    u = w * dts
    small = np.abs(u) < SMALL_TURN
    A = np.empty_like(dts)
    B = np.empty_like(dts)
    us, ds = u[small], dts[small]
    A[small] = ds * (1.0 - us**2 / 6.0 + us**4 / 120.0)
    B[small] = ds * (us / 2.0 - us**3 / 24.0 + us**5 / 720.0)
    big = ~small
    if np.any(big):
        A[big] = np.sin(u[big]) / w
        B[big] = (1.0 - np.cos(u[big])) / w
    out = np.empty((dts.size, 3))
    out[:, 0] = x[0] + A * x[3] - B * x[4]
    out[:, 1] = x[1] + B * x[3] + A * x[4]
    out[:, 2] = x[2] + x[5] * dts
    return out


def ct_step(x: np.ndarray, dt: float) -> np.ndarray:
    """Move a CT state forward by dt along a circular arc at turn rate w."""
    px, py, pz, vx, vy, vz, w = x
    A, B, _, _, s, c = turn_coefficients(w, dt)
    return np.array([
        px + A * vx - B * vy,
        py + B * vx + A * vy,
        pz + vz * dt,
        c * vx - s * vy,
        s * vx + c * vy,
        vz,
        w,
    ])


def ct_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Derivative of ct_step with respect to the state (7x7), used by the EKF."""
    _, _, _, vx, vy, _, w = x
    A, B, dA, dB, s, c = turn_coefficients(w, dt)
    F = np.eye(7)
    F[0, 3], F[0, 4], F[0, 6] = A, -B, dA * vx - dB * vy
    F[1, 3], F[1, 4], F[1, 6] = B, A, dB * vx + dA * vy
    F[2, 5] = dt
    F[3, 3], F[3, 4], F[3, 6] = c, -s, dt * (-s * vx - c * vy)
    F[4, 3], F[4, 4], F[4, 6] = s, c, dt * (c * vx - s * vy)
    return F


def cv7_matrix(dt: float) -> np.ndarray:
    """CV transition in the 7-state CT layout. The turn rate is carried unchanged."""
    F = np.eye(7)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F


def ct_process_noise(dt: float, q_acc: float, q_turn: float) -> np.ndarray:
    """7x7 process noise: white acceleration on each axis plus a turn-rate random walk."""
    block = cwna_block(dt, q_acc)
    Q = np.zeros((7, 7))
    for axis in range(3):
        p, v = axis, axis + 3
        Q[p, p], Q[p, v] = block[0, 0], block[0, 1]
        Q[v, p], Q[v, v] = block[1, 0], block[1, 1]
    Q[6, 6] = q_turn * dt
    return Q


def position_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kalman update for a position-only measurement (H = [I3, 0]), Joseph form.

    K = P H' S^-1 with S = H P H' + R. The Joseph form
    (I - K H) P (I - K H)' + K R K' keeps P symmetric and positive under rounding.
    """
    n = x.size
    H = np.zeros((3, n))
    H[:, :3] = np.eye(3)
    nu = z - x[:3]
    S = P[:3, :3] + R
    K = np.linalg.solve(S, P[:3, :]).T
    x_new = x + K @ nu
    IKH = np.eye(n) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return x_new, (P_new + P_new.T) / 2.0


def gaussian_log_likelihood(nu: np.ndarray, S: np.ndarray) -> float:
    """log N(nu; 0, S) for a 3-vector innovation."""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("innovation covariance must be positive definite")
    return float(-0.5 * (nu @ np.linalg.solve(S, nu)) - 0.5 * logdet - 0.5 * nu.size * np.log(2.0 * np.pi))
