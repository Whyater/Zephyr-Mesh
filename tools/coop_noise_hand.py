"""Independent reference numbers for the cooperative-noise experiment.

Works one axis of a constant-velocity Kalman filter by exact covariance
propagation, checks it with a small Monte Carlo, and runs two noise-free
limiting cases (straight line, 1 m circle). Everything here is simulated.
Written from the experiment spec only; it imports nothing from sim/.

Run: python -I tools/coop_noise_hand.py
"""
import json

import numpy as np

Q_SPEC = 0.1          # filter process noise intensity q, m^2/s^3
N_CAP = 121           # k = 0..120 (10 Hz), keeping only captures at or before 12 s
T_SCORE = 2.0         # only score query times at or after this
EPS = 1e-9            # float tolerance when comparing times
P0 = np.diag([1.0, 1.0])
H = np.array([1.0, 0.0])
CONFIGS = {"N1": [0.0], "N3sync": [0.0, 0.0, 0.0], "N3stag": [0.0, 1 / 30, 2 / 30]}
QTIMES = np.arange(601) / 50.0          # 0, 0.02, ..., 12.0 s
SCORED = QTIMES >= T_SCORE - EPS        # 2.0 to 12.0 s, 501 points
PAIRS = [(0.02, 0.05), (0.02, 0.02), (0.05, 0.05), (0.10, 0.05), (0.10, 0.10)]


def F(dt):
    # position moves by velocity*dt, velocity stays the same
    return np.array([[1.0, dt], [0.0, 1.0]])


def Qm(dt):
    # how much the filter thinks the target can wander in dt
    return Q_SPEC * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


def capture_times(phases):
    """All capture times from every observer, sorted (ties stay together)."""
    t = [p + k / 10.0 for p in phases for k in range(N_CAP) if p + k / 10.0 <= 12.0 + EPS]
    return np.sort(np.array(t), kind="stable")


def filter_gains(times, r_told):
    """The filter's own Riccati recursion. Returns (dt, K) for each capture.

    The gains never depend on the measured values, so we can compute them once
    and reuse them for every noise case and every Monte Carlo run.
    """
    P, t_prev, out = P0.copy(), 0.0, []
    for t in times:
        dt = t - t_prev                      # 0 for same-time measurements
        P = F(dt) @ P @ F(dt).T + Qm(dt)     # predict
        K = P[:, 0] / (P[0, 0] + r_told)     # gain = P H^T / (H P H^T + R)
        P = P - np.outer(K, P[0, :])         # P = (I - K H) P
        out.append((dt, K))
        t_prev = t
    return out


def plan(times):
    """For each query time: which captures happen first, and time since last update."""
    steps, i, t_last = [], 0, 0.0
    for tq in QTIMES:
        new = []
        while i < len(times) and times[i] <= tq + EPS:
            new.append(i)
            t_last = times[i]
            i += 1
        steps.append((new, tq - t_last))
    return steps


def cov_run(times, gains, sig_true):
    """Exact true-error statistics (no Monte Carlo).

    e = estimate - truth = [pos error, vel error]. The target truly has zero
    acceleration, so predict is e -> F e with nothing added. Update is
    e -> (I - K H) e + K v, where v is the real noise (variance sig_true^2).
    C tracks the noise covariance of e; m tracks its deterministic mean, which
    starts as [0, -1] because the filter starts at speed 0 and truth is 1 m/s.
    """
    C, m = np.zeros((2, 2)), np.array([0.0, -1.0])
    var, bias = [], []
    for new, tau in plan(times):
        for i in new:
            dt, K = gains[i]
            C, m = F(dt) @ C @ F(dt).T, F(dt) @ m
            A = np.eye(2) - np.outer(K, H)
            C = A @ C @ A.T + sig_true**2 * np.outer(K, K)
            m = A @ m
        h = np.array([1.0, tau])             # straight-line extrapolation to query time
        var.append(h @ C @ h)
        bias.append(h @ m)
    return np.array(var), np.array(bias)


def batch_run(times, gains, z, x0):
    """Run the actual filter on many measurement rows at once.

    z: (runs, n_captures) measured positions. x0: (runs, 2) starting state.
    Returns estimates at every query time, shape (runs, n_query).
    """
    X, est = x0.astype(float).copy(), []
    for new, tau in plan(times):
        for i in new:
            dt, K = gains[i]
            X[:, 0] += dt * X[:, 1]                       # predict
            X += np.outer(z[:, i] - X[:, 0], K)           # update with innovation
        est.append(X[:, 0] + tau * X[:, 1])
    return np.array(est).T


def setup(name, r_told):
    times = capture_times(CONFIGS[name])
    return times, filter_gains(times, r_told)


def steady_gain(T, r, n=3000):
    """Run the Riccati recursion with a fixed step until the gain settles."""
    P = P0.copy()
    for _ in range(n):
        P = F(T) @ P @ F(T).T + Qm(T)
        K = P[:, 0] / (P[0, 0] + r)
        P = P - np.outer(K, P[0, :])
    return K


def hand_lag(T, r, a=1.0):
    """Approximate steady lag of an alpha-beta filter under constant acceleration a.

    Classic result: just after an update the position error is
    -a T^2 (1 - alpha) / beta, and just before the next update it is
    -a T^2 / beta (alpha = K[0], beta = K[1] T). In between the extrapolation
    error is p + v tau - a tau^2 / 2; its average over the gap is reported.
    """
    K = steady_gain(T, r)
    alpha, beta = K[0], K[1] * T
    p = -a * T**2 * (1 - alpha) / beta
    pre = -a * T**2 / beta
    v = (pre - p + a * T**2 / 2) / T
    mean = p + v * T / 2 - a * T**2 / 6
    return {"alpha": alpha, "beta": beta, "err_after_update": abs(p),
            "err_before_update": abs(pre), "mean_abs_err_over_gap": abs(mean)}


def stats(x):
    x = np.abs(x[SCORED])
    return {"rms": np.sqrt(np.mean(x**2)), "p95": np.percentile(x, 95), "max": x.max()}


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6)
    return obj


# Part A: noise only, straight-line target
part_a = {"ideal_ratio_1_over_sqrt3": 1 / np.sqrt(3)}
for st, sr in PAIRS:
    row = {}
    for name in CONFIGS:
        times, g = setup(name, sr**2)
        var, bias = cov_run(times, g, st)
        v, b2 = var[SCORED].mean(), (bias[SCORED] ** 2).mean()
        row[name] = {"axis_var_noise": v, "rms3d_noise": np.sqrt(3 * v),
                     "rms3d_noise_plus_bias": np.sqrt(3 * v + b2),
                     "bias_mean_sq_scored": b2, "bias_max_abs_scored": np.abs(bias[SCORED]).max()}
    for name in ("N3sync", "N3stag"):
        row[f"ratio_{name}_over_N1"] = row[name]["rms3d_noise"] / row["N1"]["rms3d_noise"]
    part_a[f"true{st:.2f}_told{sr:.2f}"] = row

# Monte Carlo check of the analytic numbers, pair (0.05, 0.05)
rng = np.random.default_rng(20261021)
RUNS, SIG = 2000, 0.05
mc_check = {"runs": RUNS, "seed": 20261021, "pair": [SIG, SIG]}
for name in CONFIGS:
    times, g = setup(name, SIG**2)
    z = times[None, :] + SIG * rng.standard_normal((RUNS, len(times)))   # truth x = t
    err = batch_run(times, g, z, np.zeros((RUNS, 2))) - QTIMES[None, :]
    mc_var = err[:, SCORED].var(axis=0, ddof=1).mean()
    var, bias = cov_run(times, g, SIG)
    a_rms, m_rms = np.sqrt(3 * var[SCORED].mean()), np.sqrt(3 * mc_var)
    mc_check[name] = {"rms3d_noise_analytic": a_rms, "rms3d_noise_mc": m_rms,
                      "rel_diff": m_rms / a_rms - 1,
                      "mean_err_max_abs_diff_vs_bias": np.abs(err.mean(axis=0) - bias)[SCORED].max()}

# Part B: zero noise limiting cases, sigma_told = 0.05
part_b = {"straight_line_1mps": {}, "circle_r1_v1": {}, "circle_hand_estimate_approx": {}}
R0 = 0.05**2
for name in CONFIGS:
    times, g = setup(name, R0)
    # 1. straight line: truth x = t, measurements exact
    err = batch_run(times, g, times[None, :], np.zeros((1, 2)))[0] - QTIMES
    _, bias = cov_run(times, g, 0.0)
    s = stats(err)
    part_b["straight_line_1mps"][name] = {"max_abs": s["max"], "p95_abs": s["p95"],
                                          "check_vs_cov_bias": np.abs(err - bias).max()}
    # 2. circle x = cos t, y = sin t (radius 1 m, 1 m/s); one filter copy per axis
    z = np.vstack([np.cos(times), np.sin(times)])
    x0 = np.array([[1.0, 0.0], [0.0, 0.0]])    # true start position, zero velocity
    est = batch_run(times, g, z, x0)
    norm = np.hypot(est[0] - np.cos(QTIMES), est[1] - np.sin(QTIMES))
    part_b["circle_r1_v1"][name] = stats(norm)

# 3. hand estimate: N3sync is exactly one update with R/3 every 0.1 s,
# N3stag is approximated as one sensor every 1/30 s with R.
part_b["circle_hand_estimate_approx"] = {
    "N1": hand_lag(0.1, R0), "N3sync": hand_lag(0.1, R0 / 3), "N3stag": hand_lag(1 / 30, R0)}

notes = [
    "Simulated only. One axis is computed; 3D numbers assume 3 identical independent axes.",
    "A capture at exactly a query time is applied before that query is scored (tolerance 1e-9 s).",
    "Captures are k/10 + phase, k = 0..120, kept only up to 12 s (the 12 s capture counts at phase 0).",
    ("Part A noise covariance starts at zero because the filter starts at the true position; "
     "the only start-up error is the 1 m/s velocity error, carried as a separate mean (bias)."),
    "Bias is on one axis only, so rms3d_noise_plus_bias = sqrt(3*axis_var + mean bias^2).",
    "Time averages are plain means over the 501 query times from 2.0 to 12.0 s.",
    ("Hand lag estimate treats the 1 m/s^2 centripetal acceleration as constant in direction "
     "over the filter memory; it ignores the turning of that direction, so it is approximate."),
    "Monte Carlo variance is across runs at each query time (ddof 1), then time-averaged.",
]

print(json.dumps(rounded({"part_a": part_a, "part_b": part_b, "mc_check": mc_check,
                          "notes": notes}), sort_keys=True, indent=1))
