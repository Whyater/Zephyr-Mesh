"""Independent analytic check for constant-velocity Kalman tuning (one axis, CWNA noise).

This file does not import anything from sim/. It rebuilds the filter's steady
state from the math alone so the simulator's numbers can be checked against it.

Model (per axis): state [pos, vel], period T, F = [[1, T], [0, 1]],
Q = q*[[T^3/3, T^2/2], [T^2/2, T]], H = [1, 0], told variance r = s_told^2.
Steady gain K = [alpha, beta/T].

Derivation of the steady-state relations
----------------------------------------
Let M = [[m11, m12], [m12, m22]] be the steady PREDICTED covariance and P the
steady UPDATED covariance. The gain is K = M H^T / (m11 + r), so
    alpha = m11/(m11 + r),  beta/T = m12/(m11 + r),  and m11 + r = r/(1 - alpha).
Update P = (I - K H) M gives
    p11 = alpha*r,  p12 = r*beta/T,  p22 = m22 - (beta/T)*m12.
Predict M = F P F^T + Q gives
    (1) m11 = p11 + 2T p12 + T^2 p22 + q T^3/3
    (2) m12 = p12 + T p22 + q T^2/2
    (3) m22 = p22 + q T
From (3) and the p22 line: (beta/T)*m12 = q T, so m12 = q T^2/beta.
Also m12 = (beta/T)(m11 + r) = beta r / (T (1 - alpha)). Equate them:
    beta^2 = (q T^3 / r) (1 - alpha) = lambda^2 (1 - alpha).            [relation 1]
From (2): T p22 = m12 - p12 - q T^2/2 = (r/(T(1-alpha))) (alpha beta - beta^2/2),
using q T^2/2 = lambda^2 r/(2T) = beta^2 r / (2T(1-alpha)).
From (1): m11 - p11 = alpha^2 r/(1-alpha) = 2 r beta + T^2 p22 + lambda^2 r/3.
Multiply by (1-alpha)/r and substitute:
    alpha^2 = 2 beta (1-alpha) + alpha beta - beta^2/2 + beta^2/3
    alpha^2 + alpha beta - 2 beta + beta^2/6 = 0.                      [relation 2]
Both expected relations are confirmed. p22 > 0 needs beta < 2 alpha; that picks
the physical root (relation 2 has a second, non-physical root when lambda is big).

Solving: put u = sqrt(1 - alpha), so alpha = 1 - u^2 and beta = lambda*u. The
left side of relation 2 is +1 at u = 0 and negative where beta = 2*alpha, so
bisection (brentq) on that bracket finds the unique physical root.
"""
import argparse
import json
import math

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq
from scipy.stats import chi, chi2, ncx2

Q_LIST = (0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0,
          3.1622776601683795, 10.0)
S_LIST = (0.02, 0.05, 0.10)
DELAYS = (0.0, 0.2, 0.3)
QUERY_STEP = 0.02


def steady_gain(q, r_told, T=0.1):
    """Closed form: root of relation 2 with beta = lambda*sqrt(1 - alpha)."""
    lam = math.sqrt(q * T**3 / r_told)

    def g(u):
        a, b = 1.0 - u * u, lam * u
        return a * a + a * b - 2.0 * b + b * b / 6.0

    # u_max is where beta = 2*alpha (p22 = 0); the physical root is below it.
    u_max = (-lam + math.sqrt(lam * lam + 16.0)) / 4.0
    u = brentq(g, 0.0, u_max, xtol=1e-15, rtol=1e-15, maxiter=500)
    return 1.0 - u * u, lam * u


def riccati_gain(q, r_told, T=0.1, tol=1e-15, max_iter=200000):
    """Brute force: iterate predict/update with plain scalar floats."""
    p11, p12, p22 = 1e3, 0.0, 1e3          # big, vague starting covariance
    a_old = b_old = -1.0
    for _ in range(max_iter):
        # predict: M = F P F^T + Q, written out entry by entry
        m11 = p11 + 2 * T * p12 + T * T * p22 + q * T**3 / 3
        m12 = p12 + T * p22 + q * T**2 / 2
        m22 = p22 + q * T
        s = m11 + r_told                   # innovation variance
        k1, k2 = m11 / s, m12 / s          # gain entries
        # update: P = (I - K H) M
        p11, p12, p22 = (1 - k1) * m11, (1 - k1) * m12, m22 - k2 * m12
        a, b = k1, k2 * T
        if abs(a - a_old) < tol and abs(b - b_old) < tol:
            break
        a_old, b_old = a, b
    return a, b


def _matrices(alpha, beta, T):
    F = np.array([[1.0, T], [0.0, 1.0]])
    K = np.array([alpha, beta / T])
    H = np.array([[1.0, 0.0]])
    A = (np.eye(2) - np.outer(K, H)) @ F   # error / state transition between updates
    return F, K, A


def actual_cov(q, s_told, s_true, T=0.1):
    """True steady error covariance after an update when the truth has no acceleration.
    The error obeys e_k = A e_{k-1} + K*noise_k, so P = A P A^T + s_true^2 K K^T
    (same as iterating the mismatched recursion to steady state)."""
    alpha, beta = steady_gain(q, s_told**2, T)
    _, K, A = _matrices(alpha, beta, T)
    return solve_discrete_lyapunov(A, s_true**2 * np.outer(K, K))


def horizons(delay):
    # queries every 0.02 s land at 5 evenly spaced ages past the fused capture time
    return [delay + j * QUERY_STEP for j in range(5)]


def mixture_p95(sigmas, biases=None):
    """95th percentile of the 3D error norm, equal mix over horizons.
    Noise only: norm/sigma is chi with 3 dof. With an x-y bias b the squared
    norm/sigma^2 is noncentral chi-square, 3 dof, noncentrality (b/sigma)^2."""
    biases = biases or [0.0] * len(sigmas)

    def cdf(x):
        tot = 0.0
        for s, b in zip(sigmas, biases):
            nc = (b / s) ** 2
            tot += chi.cdf(x / s, 3) if nc < 1e-12 else ncx2.cdf((x / s) ** 2, 3, nc)
        return tot / len(sigmas) - 0.95

    hi = 10.0 * max(s + b for s, b in zip(sigmas, biases))
    return brentq(cdf, 0.0, hi, xtol=1e-12)


def _sigma(P, h):
    # predict forward h seconds: err = e_pos + h*e_vel
    return math.sqrt(P[0, 0] + 2 * h * P[0, 1] + h * h * P[1, 1])


def straight_p95(q, s_told, s_true, delay, T=0.1):
    P = actual_cov(q, s_told, s_true, T)
    return mixture_p95([_sigma(P, h) for h in horizons(delay)])


def turn_bias(q, s_told, h, T=0.1, radius=1.0, speed=1.0):
    """Steady bias magnitude at horizon h on a circle, via complex phasors.
    Feed z_k = exp(i w k T); the filter settles to x_k = X exp(i w k T), where
    X = A X exp(-i w T) + K, i.e. (I - A exp(-i w T)) X = K."""
    w = speed / radius
    alpha, beta = steady_gain(q, s_told**2, T)
    _, K, A = _matrices(alpha, beta, T)
    X = np.linalg.solve(np.eye(2) - A * np.exp(-1j * w * T), K.astype(complex))
    pred = X[0] + h * X[1]                 # position + h * velocity
    return radius * abs(pred - np.exp(1j * w * h))


def turn_p95(q, s_told, s_true, delay, T=0.1):
    P = actual_cov(q, s_told, s_true, T)
    hs = horizons(delay)
    return mixture_p95([_sigma(P, h) for h in hs], [turn_bias(q, s_told, h, T) for h in hs])


def _mc_turn_check(q, s_told, s_true, delay, n=200000, seed=1):
    """Monte Carlo cross-check of turn_p95 (bias along x, Gaussian noise on 3 axes)."""
    rng = np.random.default_rng(seed)
    P = actual_cov(q, s_told, s_true)
    norms = []
    for h in horizons(delay):
        e = rng.normal(0.0, _sigma(P, h), size=(n, 3))
        e[:, 0] += turn_bias(q, s_told, h)
        norms.append(np.linalg.norm(e, axis=1))
    return float(np.percentile(np.concatenate(norms), 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="optional path for table 2 as JSON")
    args = ap.parse_args()

    print("TABLE 1: steady gain, closed form vs brute-force Riccati")
    print(f"{'q':>8} {'s_told':>6} {'alpha':>12} {'beta':>12} {'K_vel':>10} {'max|diff|':>10}")
    worst = 0.0
    for s in S_LIST:
        for q in Q_LIST:
            a1, b1 = steady_gain(q, s * s)
            a2, b2 = riccati_gain(q, s * s)
            d = max(abs(a1 - a2), abs(b1 - b2))
            worst = max(worst, d)
            print(f"{q:8.4f} {s:6.2f} {a1:12.9f} {b1:12.9f} {b1 / 0.1:10.6f} {d:10.2e}")
    print(f"max disagreement over grid: {worst:.2e}  ({'OK' if worst < 1e-9 else 'FAIL'})")

    print("\nSanity checks")
    k = math.sqrt(chi2.ppf(0.95, 3))
    print(f"  single chi-3 p95 / sigma = {k:.4f} (expect 2.7955); "
          f"mixture of identical sigmas = {mixture_p95([1.0] * 5):.4f}")
    for qq in (1e-6, 1e-4, 0.01):
        a, _ = steady_gain(qq, 0.05**2)
        print(f"  q={qq:g}: alpha={a:.5f}, straight p95 (d=0, s=0.05) = "
              f"{1000 * straight_p95(qq, 0.05, 0.05, 0.0):.2f} mm")
    mc = _mc_turn_check(1.0, 0.05, 0.05, 0.2)
    print(f"  turn p95 q=1, s=0.05, d=0.2: exact {1000 * turn_p95(1.0, 0.05, 0.05, 0.2):.2f} mm, "
          f"Monte Carlo {1000 * mc:.2f} mm")

    print("\nTABLE 2: predicted p95 3D error in mm (* = q with lowest p95 in that row)")
    head = " ".join(f"{q:>7.3g}" for q in Q_LIST)
    rows = []
    for kind, fn in (("straight", straight_p95), ("turn", turn_p95)):
        for told_mode in ("0.05", "match"):
            print(f"\n{kind}, s_told = {told_mode}")
            print(f"{'s_true':>6} {'delay':>5} | q = {head}")
            for s_true in S_LIST:
                s_told = 0.05 if told_mode == "0.05" else s_true
                for d in DELAYS:
                    vals = [fn(q, s_told, s_true, d) for q in Q_LIST]
                    best = int(np.argmin(vals))
                    cells = " ".join(f"{1000 * v:6.1f}{'*' if i == best else ' '}"
                                     for i, v in enumerate(vals))
                    print(f"{s_true:6.2f} {d:5.1f} |     {cells}")
                    rows.append({"kind": kind, "s_told": s_told, "s_true": s_true,
                                 "delay": d, "q": list(Q_LIST), "p95_m": vals,
                                 "best_q": Q_LIST[best]})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"units": "m", "T": 0.1, "query_step": QUERY_STEP, "rows": rows}, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
