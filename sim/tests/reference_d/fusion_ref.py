"""Independent reference filters for simulated target tracking.

batch_central_cv takes the prior (x0, P0) at time t = 0.
"""
import numpy as np


def batch_central_cv(measurements, x0, P0, q):
    """Run a constant velocity Kalman filter and a batch check when q is zero."""
    x = np.asarray(x0, dtype=float).copy()
    P = np.asarray(P0, dtype=float).copy()
    prev_t = 0.0
    I3 = np.eye(3)
    H = np.hstack((I3, np.zeros((3, 3))))
    ordered = sorted(measurements, key=lambda item: item[0])
    for stamp, obs, cov in ordered:
        dt = float(stamp) - prev_t
        F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        Q = q * np.block([[dt**3 / 3 * I3, dt**2 / 2 * I3], [dt**2 / 2 * I3, dt * I3]])
        x = F @ x
        P = F @ P @ F.T + Q
        obs = np.asarray(obs, dtype=float)
        cov = np.asarray(cov, dtype=float)
        innov = obs - H @ x
        S = H @ P @ H.T + cov
        K = np.linalg.solve(S, (P @ H.T).T).T
        x = x + K @ innov
        P = (np.eye(6) - K @ H) @ P
        P = (P + P.T) / 2
        prev_t = float(stamp)
    return x, P


def ci_grid(a, A, b, B, step=1e-4):
    """Brute force covariance intersection over a deterministic weight grid."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    ia = np.linalg.inv(A); ib = np.linalg.inv(B)
    best = None
    for w in np.arange(0.0, 1.0 + step / 2, step):
        P = np.linalg.inv(w * ia + (1.0 - w) * ib)
        value = float(np.trace(P))
        if best is None or value < best[0]:
            best = (value, float(w), P)
    _, w, P = best
    x = P @ (w * ia @ np.asarray(a, dtype=float) + (1.0 - w) * ib @ np.asarray(b, dtype=float))
    return x, P, w


def mean_variance_of_n(sigma, n):
    return sigma**2 / n
