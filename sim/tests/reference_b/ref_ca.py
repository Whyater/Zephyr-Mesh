"""Reference constant acceleration Kalman filter."""
import numpy as np


def _matrices(dt, q_jerk):
    """Build CA transition and process matrices."""
    F = np.eye(9)
    Q = np.zeros((9, 9))
    fa = np.array([[1.0, dt, dt*dt/2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    qa = q_jerk * np.array([[dt**5/20.0, dt**4/8.0, dt**3/6.0],
                            [dt**4/8.0, dt**3/3.0, dt**2/2.0],
                            [dt**3/6.0, dt**2/2.0, dt]])
    for a in range(3):
        inds = [a, a+3, a+6]
        F[np.ix_(inds, inds)] = fa
        Q[np.ix_(inds, inds)] = qa
    return F, Q


def predict(state, P, t_from, t_to, q_jerk=0.1):
    """Predict CA state over a time interval."""
    dt = float(t_to) - float(t_from)
    if dt < 0: raise ValueError("t_to must be at least t_from")
    F, Q = _matrices(dt, q_jerk)
    x = F @ np.asarray(state, dtype=float)
    C = F @ np.asarray(P, dtype=float) @ F.T + Q
    return x, (C + C.T) / 2.0


def _update(x, P, z, r_std):
    """Apply the Joseph position update."""
    H = np.zeros((3, 9)); H[:, :3] = np.eye(3)
    R = r_std ** 2 * np.eye(3)
    nu = np.asarray(z, dtype=float) - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.solve(S, np.eye(3))
    x = x + K @ nu
    I = np.eye(9) - K @ H
    P = I @ P @ I.T + K @ R @ K.T
    return x, (P + P.T) / 2.0


def run_ca(measurements, stamps, start_position, q_jerk=0.1, r_std=0.05,
           p0_pos=1.0, p0_vel=1.0, p0_acc=1.0, query=None):
    """Run CA filter updates.

    With query supplied, returns predictions from the final posterior at each
    requested time. Otherwise returns posterior (state, covariance) per update.
    """
    measurements = list(measurements); stamps = list(stamps)
    if len(measurements) != len(stamps): raise ValueError("measurements and stamps lengths differ")
    if not measurements: return []
    x = np.zeros(9); x[:3] = np.asarray(start_position, dtype=float)
    P = np.diag([p0_pos]*3 + [p0_vel]*3 + [p0_acc]*3).astype(float)
    t = float(stamps[0]); out = []
    for z, stamp in zip(measurements, stamps):
        stamp = float(stamp)
        x, P = predict(x, P, t, stamp, q_jerk)
        x, P = _update(x, P, z, r_std)
        t = stamp; out.append((x.copy(), P.copy()))
    if query is not None: return [predict(x, P, t, float(qt), q_jerk) for qt in query]
    return out
