"""Reference constant velocity Kalman filter."""
import numpy as np


def _matrices(dt, q_acc):
    """Build CV transition and process matrices."""
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    Q = np.zeros((6, 6))
    q = q_acc * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    for k in range(3):
        Q[k, k] = q[0, 0]
        Q[k, k + 3] = q[0, 1]
        Q[k + 3, k] = q[1, 0]
        Q[k + 3, k + 3] = q[1, 1]
    return F, Q


def predict(state, P, t_from, t_to, q_acc=0.1):
    """Predict CV state over a time interval."""
    dt = float(t_to) - float(t_from)
    if dt < 0:
        raise ValueError("t_to must be at least t_from")
    F, Q = _matrices(dt, q_acc)
    x = F @ np.asarray(state, dtype=float)
    C = F @ np.asarray(P, dtype=float) @ F.T + Q
    return x, (C + C.T) / 2.0


def _update(x, P, z, r_std):
    """Apply the Joseph position update."""
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    R = (r_std ** 2) * np.eye(3)
    z = np.asarray(z, dtype=float)
    nu = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.solve(S, np.eye(3))
    x = x + K @ nu
    I = np.eye(6) - K @ H
    P = I @ P @ I.T + K @ R @ K.T
    return x, (P + P.T) / 2.0


def run_cv(measurements, stamps, start_position, q_acc=0.1, r_std=0.05,
           p0_pos=1.0, p0_vel=1.0, query=None):
    """Run CV filter updates.

    With query supplied, it is an iterable of times and predictions from the
    final filter state are returned in query order. Without query, the returned
    list contains (state, covariance) after each measurement update.
    """
    measurements = list(measurements); stamps = list(stamps)
    if len(measurements) != len(stamps):
        raise ValueError("measurements and stamps lengths differ")
    if not measurements:
        if query is not None: return []
        return []
    x = np.zeros(6, dtype=float); x[:3] = np.asarray(start_position, dtype=float)
    P = np.diag([p0_pos] * 3 + [p0_vel] * 3).astype(float)
    t = float(stamps[0])
    out = []
    for z, stamp in zip(measurements, stamps):
        stamp = float(stamp)
        x, P = predict(x, P, t, stamp, q_acc)
        x, P = _update(x, P, z, r_std)
        t = stamp
        out.append((x.copy(), P.copy()))
    if query is not None:
        return [predict(x, P, t, float(qt), q_acc) for qt in query]
    return out
