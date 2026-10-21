"""Independent checks for the coordinated-turn (CT) and IMM estimators.

Plain-words derivation (no linear algebra needed)

A coordinated turn: the drone keeps its speed but its horizontal velocity
arrow spins at a steady rate w (radians per second). Write the arrow as one
complex number V = vx + i*vy. Spinning it by the angle w*t is multiplying by
exp(i*w*t), so V(T) = V(0)*exp(i*w*T), and position is the running total
(integral) of velocity. Splitting that into real and imaginary parts gives the
sin/cos formulas in the code. As w goes to 0, sin(w*T)/w goes to T and
(1 - cos(w*T))/w goes to 0, so the motion becomes a straight line. To avoid
trusting those formulas, the reference here never uses them: it steps the
turn equations p' = v, vx' = -w*vy, vy' = w*vx forward with Runge-Kutta 4 (a
careful way of adding up many tiny steps). The Jacobian (how much each output
number changes when one input number is nudged) is checked against nudging the
inputs by 1e-5 and re-running that Runge-Kutta integration.

NEES (normalized estimation error squared) is the filter's squared error
divided by the variance the filter claims, summed over all 7 state numbers. If
the filter is honest about its uncertainty, each term is a squared standard
normal, so the sum averages 7 (chi-square with 7 degrees of freedom; 1 degree
for the turn rate alone). Too big means overconfident, too small means timid.
The truth is simulated here from the continuous-time model in 20 small
substeps per step, not from the filter's own matrices.

The NEES band is the 99 percent chi-square band for 400 runs (+/- 0.482 on 7,
+/- 0.182 on 1) plus an allowance of 0.25 (on 7) and 0.10 (on 1). The extra is
there because the extended Kalman filter is only approximately chi-square: it
replaces the product w*v by a tangent line, which costs a few percent of the
turn coupling at mean speed 2 m/s, and its noise matrix ignores rotation of
the noise within a step, worth under 1 percent. A separate 4000-run
measurement gave 6.967 +/- 0.058 (on 7) and 0.977 +/- 0.022 (on 1), so the
real bias is under about 1 percent and the extra is conservative.

Why the Jacobian check is separate: NEES alone can miss a Jacobian sign error.
A wrong sign on one small entry of the turn-rate column (size about vy*T^2/2)
shifts NEES by less than its band, so only the direct Jacobian comparison
catches it. Larger entries do wreck NEES, but the Jacobian check is the one
that is sensitive to the small ones.

For the IMM (interacting multiple model) the checks use facts that need no
filter code: a symmetric two-state Markov chain left at rate 1/tau has a
closed-form switching chance, a mixture of Gaussians has a known mean and
covariance (average covariance plus spread of the means), and two identical
modes must behave like one textbook Kalman filter. Without switching, the log
odds between two models add up the per-step log-likelihood differences
(Bayes). Every expected value comes from numpy or the standard library, or is
a literal worked out by hand; nothing is imported from sim/ except the objects
under test.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.estimation.base import EstimatorSettings
from sim.estimation.ct import CTEstimator
from sim.estimation.imm import IMMEstimator, switch_probability
from sim.estimation.kinematics import ct_jacobian, ct_step

# ------------------------------------------------------------------ helpers

def _turn_rhs(X):
    """Coordinated turn ODE for a batch of states: p' = v, v_xy spins at rate w."""
    d = np.zeros_like(X)
    d[:, 0:3] = X[:, 3:6]
    d[:, 3] = -X[:, 6] * X[:, 4]
    d[:, 4] = X[:, 6] * X[:, 3]
    return d


def _rk4(X, T, n=400):
    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    h = T / n
    for _ in range(n):
        k1 = _turn_rhs(X)
        k2 = _turn_rhs(X + 0.5 * h * k1)
        k3 = _turn_rhs(X + 0.5 * h * k2)
        k4 = _turn_rhs(X + h * k3)
        X = X + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X


def _nees(error, cov):
    return float(error @ np.linalg.solve(cov, error))


def _chi2_mean_half_width(dof, runs, z=2.576):
    # Sum of `runs` chi-square(dof) is chi-square(runs*dof): variance 2*runs*dof.
    return z * np.sqrt(2.0 * runs * dof) / runs


class _TextbookCV:
    """Constant-velocity Kalman filter from the textbook equations (6 states,
    white acceleration of intensity q, position measurement, standard update)."""

    def __init__(self, q, r_std, p0, v0):
        self.q, self.R = q, np.eye(3) * r_std**2
        self.x = np.zeros(6)
        self.P = np.diag([p0] * 3 + [v0] * 3)
        self.t = 0.0
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def predicted(self, t):
        T = t - self.t
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * T
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * self.q * T**3 / 3.0
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * self.q * T**2 / 2.0
        Q[3:, 3:] = np.eye(3) * self.q * T
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, z, t):
        x, P = self.predicted(t)
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        nu = z - self.H @ x
        self.x = x + K @ nu
        self.P = (np.eye(6) - K @ self.H) @ P
        self.t = t
        return nu, S


def _log_gauss(nu, S):
    # -1.5 ln(2 pi) is dropped: it cancels in every comparison below.
    return float(-0.5 * nu @ np.linalg.solve(S, nu) - 0.5 * np.log(np.linalg.det(S)))


# ------------------------------------------------------------------ check 1

TURN_CASES = [
    ([1, -2, 0.5, 2, -1, 0.3, 0.8], 0.1),
    ([1, -2, 0.5, 2, -1, 0.3, 0.8], 0.7),
    ([0, 0, 0, -1.5, 0.7, 0, -2.5], 0.7),
    ([0.3, 0.1, -0.2, 2, -1, 0.3, 0.0], 0.5),
    ([0.3, 0.1, -0.2, 2, -1, 0.3, 1e-4], 0.5),   # |w T| = 5e-5, Taylor branch
    ([0.3, 0.1, -0.2, 2, -1, 0.3, 3e-4], 0.5),   # |w T| = 1.5e-4, sin/cos branch
    ([5, 5, 5, 0, 3, -1, 1.2], 0.25),
]


def test_ct_step_and_jacobian_match_runge_kutta_of_the_turn_ode():
    """Independent fact: ct_step is the exact solution of p' = v, v_xy' = w J v_xy,
    and ct_jacobian is its derivative. The reference is a 400-step Runge-Kutta 4
    integration (no sin/cos formulas) and a central finite difference of it
    (eps 1e-5). RK4 error here is below 1e-11 and the finite-difference noise
    below 3e-9, so 1e-9 and 1e-7 leave a wide margin while a wrong sign or a
    missing 1/w gives errors of 1e-3 or more. The w = 0 case must also reduce
    by hand to straight-line motion, and its w column to (-vy T^2/2, vx T^2/2,
    -vy T, vx T) = (0.125, 0.25, 0.5, 1.0)."""
    eps = 1e-5
    for x, T in TURN_CASES:
        x = np.asarray(x, dtype=float)
        assert np.max(np.abs(ct_step(x, T) - _rk4(x, T)[0])) < 1e-9, (x, T)
        batch = []
        for j in range(7):
            e = np.zeros(7)
            e[j] = eps
            batch += [x + e, x - e]
        out = _rk4(np.array(batch), T)
        fd = np.column_stack([(out[2 * j] - out[2 * j + 1]) / (2 * eps) for j in range(7)])
        err = np.abs(ct_jacobian(x, T) - fd)
        assert np.max(err) < 1e-7, (x, T, np.unravel_index(np.argmax(err), err.shape), err.max())

    x = np.array([0.3, 0.1, -0.2, 2.0, -1.0, 0.3, 0.0])
    assert np.allclose(ct_step(x, 0.5)[:3], x[:3] + 0.5 * x[3:6], atol=1e-12, rtol=0.0)
    assert np.allclose(ct_jacobian(x, 0.5)[[0, 1, 3, 4], 6], [0.125, 0.25, 0.5, 1.0], atol=1e-12, rtol=0.0)


# ------------------------------------------------------------------ check 2

def test_ct_ekf_nees_on_continuous_time_turn_truth():
    """Independent fact: a consistent filter has NEES ~ chi-square(7).

    Truth is NOT drawn from the filter's F or Q. It integrates the continuous
    model dv_xy = w J v_xy dt + sqrt(qa) dB, dv_z = sqrt(qa) dB, dw = sqrt(qt) dB
    in 20 substeps per 0.1 s: velocity is spun exactly by angle w h, position
    moves by the trapezoid rule, then piecewise-constant white noise is added.
    The EKF is only approximately chi-square. Bound on the bias (derivation in the
    module docstring): the filter's Q omits within-step rotation of the noise and
    turn-noise leakage into velocity, worth under 1 percent here; linearization
    of the w*v product is a few percent of the turn coupling at mean speed
    2 m/s with velocity std 0.2 to 0.5. The band is the 99 percent chi-square
    band for 400 runs (+/- 0.482 on 7, +/- 0.182 on 1) widened by 0.25 and 0.10.
    A separate 4000-run check measured 6.967 +/- 0.058 and 0.977 +/- 0.022, so
    the real bias is under about 1 percent and the widening is conservative.
    Limit: this band only catches large process-noise errors. A turn-noise Q
    off by a factor of 10 is caught; one off by about 1.3 is not.
    """
    rng = np.random.default_rng(20261021)
    runs, steps, dt, substeps = 400, 40, 0.1, 20
    qa, qt, sigma = 0.2, 0.05, 0.05
    v_mean, w_mean, v_var, w_var = 2.0, 0.6, 0.25, 0.25
    h = dt / substeps
    settings = EstimatorSettings(measurement_std_m=sigma, q_acc_psd=qa, q_turn_psd=qt,
                                 initial_position_variance_m2=1.0,
                                 initial_velocity_variance_m2ps2=v_var,
                                 initial_turn_rate_variance=w_var)

    pos = rng.normal(0.0, 1.0, (runs, 3))
    vel = rng.normal(0.0, np.sqrt(v_var), (runs, 3))
    vel[:, 0] += v_mean
    w = rng.normal(w_mean, np.sqrt(w_var), runs)
    truths = [np.column_stack([pos, vel, w])]
    for _ in range(steps):
        for _ in range(substeps):
            V = vel[:, 0] + 1j * vel[:, 1]
            Vn = V * np.exp(1j * w * h)
            pos[:, 0] += 0.5 * h * (V.real + Vn.real)
            pos[:, 1] += 0.5 * h * (V.imag + Vn.imag)
            pos[:, 2] += h * vel[:, 2]
            vel[:, 0], vel[:, 1] = Vn.real, Vn.imag
            acc = rng.normal(0.0, np.sqrt(qa / h), (runs, 3))
            pos += 0.5 * acc * h * h
            vel += acc * h
            w = w + rng.normal(0.0, np.sqrt(qt * h), runs)
        truths.append(np.column_stack([pos, vel, w]))
    truths = np.array(truths)                                   # (steps + 1, runs, 7)
    meas = truths[:, :, :3] + rng.normal(0.0, sigma, (steps + 1, runs, 3))

    nees_final = np.empty(runs)
    nees_avg = np.empty(runs)
    nees_w = np.empty(runs)
    for r in range(runs):
        est = CTEstimator((0.0, 0.0, 0.0), settings)
        est.x[3], est.x[6] = v_mean, w_mean
        values = []
        for k in range(steps + 1):
            est.update(meas[k, r], k * dt)
            if k > steps - 10:
                values.append(_nees(truths[k, r] - est.x, est.P))
        e = truths[steps, r] - est.x
        nees_final[r] = values[-1]
        nees_avg[r] = np.mean(values)
        nees_w[r] = e[6] ** 2 / est.P[6, 6]
    half7 = _chi2_mean_half_width(7, runs) + 0.25
    half1 = _chi2_mean_half_width(1, runs) + 0.10
    assert abs(nees_final.mean() - 7.0) < half7, nees_final.mean()
    assert abs(nees_avg.mean() - 7.0) < half7, nees_avg.mean()
    assert abs(nees_w.mean() - 1.0) < half1, nees_w.mean()


# ------------------------------------------------------------------ check 3

def test_imm_with_identical_modes_follows_the_markov_chain_and_equals_one_cv_filter():
    """Independent facts. (1) A symmetric two-state chain left at rate 1/tau has
    dm/dt = (1 - 2m)/tau, so m(t) = 1/2 + (m0 - 1/2) exp(-2t/tau), and the
    chance of switching in a gap T is (1 - exp(-2T/tau))/2 (also checked here
    against scipy's matrix exponential of the generator). (2) With turn-rate
    prior and noise zero, the CT mode never turns and moves exactly like CV, so
    both modes have the same likelihood and mu just follows the chain, whatever
    the irregular gaps are. (3) A mixture of two identical Gaussians is that
    Gaussian, so the IMM output equals a textbook CV Kalman filter."""
    from scipy.linalg import expm
    tau = 1.5
    gen = np.array([[-1.0, 1.0], [1.0, -1.0]]) / tau
    for T in (0.05, 0.3, 1.0, 3.0):
        assert switch_probability(T, tau) == pytest.approx(expm(gen * T)[0, 1], abs=1e-14)

    settings = EstimatorSettings(q_acc_psd=0.1, q_turn_psd=0.0, initial_turn_rate_variance=0.0,
                                 imm_mean_sojourn_s=tau, imm_initial_ct_probability=0.9)
    imm = IMMEstimator((0.0, 0.0, 0.0), settings, r_std_m=0.05)
    ref = _TextbookCV(q=0.1, r_std=0.05, p0=1.0, v0=1.0)
    rng = np.random.default_rng(5)
    times = np.concatenate([[0.0], np.cumsum(rng.uniform(0.02, 0.3, 25))])
    for t in times:
        z = np.array([1.5 * t, -0.5 * t, 0.2]) + rng.normal(0.0, 0.05, 3)
        imm.update(z, float(t))
        ref.update(z, float(t))
        mu_ct = 0.5 + 0.4 * np.exp(-2.0 * t / tau)
        assert imm.mode_probabilities == pytest.approx([1.0 - mu_ct, mu_ct], abs=1e-12)
        pred = imm.predict_at(float(t))
        assert np.allclose(pred.position_m, ref.x[:3], atol=1e-9, rtol=0.0)
        assert np.allclose(pred.velocity_mps, ref.x[3:], atol=1e-9, rtol=0.0)
        assert np.allclose(pred.position_cov, ref.P[:3, :3], atol=1e-9, rtol=0.0)
    ahead = imm.predict_at(float(times[-1]) + 0.37)
    x_ref, P_ref = ref.predicted(float(times[-1]) + 0.37)
    assert np.allclose(ahead.position_m, x_ref[:3], atol=1e-9, rtol=0.0)
    assert np.allclose(ahead.position_cov, P_ref[:3, :3], atol=1e-9, rtol=0.0)


# ------------------------------------------------------------------ check 4

def test_imm_mixing_is_mixture_moments_and_no_switching_is_bayes():
    """Independent facts. (a) Mixing and combining are the mean and covariance of
    a Gaussian mixture: covariance = average covariance + spread of the means.
    Hand numbers for mu = (0.25, 0.75), Pi = [[0.9, 0.1], [0.1, 0.9]], means 0
    and 2 e_px, covariances I and 3I: c = (0.3, 0.7); mixed px mean 0.5 and
    27/14, px variance 2.25 and 3.066326531, other diagonals 1.5 and 82/28;
    combined px mean 1.4, px variance 3.24, other diagonals 2.4.
    (b) With tau = 1e300 nothing switches, so the log odds of CT over CV add up
    the per-step log-likelihood differences of two stand-alone filters (Bayes)."""
    imm = IMMEstimator((0.0, 0.0, 0.0), EstimatorSettings())
    x1 = np.zeros(7)
    x1[0] = 2.0
    states = [(np.zeros(7), np.eye(7)), (x1, 3.0 * np.eye(7))]
    # Private attributes and methods: no public way to inject a chosen mode state
    # and probability or to call the mixing step alone, and this check needs exact
    # hand-computed inputs.
    imm._mode_states = [(x.copy(), P.copy()) for x, P in states]
    imm._mu = np.array([0.25, 0.75])
    c, mixed = imm._mixed_states(np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert c == pytest.approx([0.3, 0.7], abs=1e-12)
    for (xm, Pm), mean, var_px, var_other in ((mixed[0], 0.5, 2.25, 1.5),
                                               (mixed[1], 27.0 / 14.0, 3.066326530612245, 82.0 / 28.0)):
        assert xm == pytest.approx(np.r_[mean, np.zeros(6)], abs=1e-12)
        assert np.allclose(Pm, np.diag([var_px] + [var_other] * 6), atol=1e-12, rtol=0.0)
    xc, Pc = IMMEstimator._combine(states, np.array([0.3, 0.7]))
    assert xc == pytest.approx(np.r_[1.4, np.zeros(6)], abs=1e-12)
    assert np.allclose(Pc, np.diag([3.24] + [2.4] * 6), atol=1e-12, rtol=0.0)

    settings = EstimatorSettings(imm_mean_sojourn_s=1e300, imm_initial_ct_probability=0.3)
    imm = IMMEstimator((0.0, 0.0, 0.0), settings, r_std_m=0.05)
    cv = _TextbookCV(q=settings.q_acc_psd, r_std=0.05, p0=1.0, v0=1.0)
    ct = CTEstimator((0.0, 0.0, 0.0), settings, r_std_m=0.05)
    rng = np.random.default_rng(11)
    log_odds = np.log(0.3 / 0.7)
    swing = 0.0
    for k in range(21):
        t = 0.1 * k
        z = np.array([np.sin(t), 1.0 - np.cos(t), 0.0]) + rng.normal(0.0, 0.05, 3)
        nu_ct, S_ct = ct.innovation(z, t)
        nu_cv, S_cv = cv.update(z, t)
        ct.update(z, t)
        imm.update(z, t)
        log_odds += _log_gauss(nu_ct, S_ct) - _log_gauss(nu_cv, S_cv)
        swing = max(swing, abs(log_odds - np.log(0.3 / 0.7)))
        mu_cv, mu_ct = imm.mode_probabilities
        assert np.log(mu_ct / mu_cv) == pytest.approx(log_odds, abs=1e-6), k
        (x_cv, P_cv), (x_ct, P_ct) = imm.mode_states
        assert np.allclose(x_cv[:6], cv.x, atol=1e-9, rtol=0.0)
        assert np.allclose(P_cv[:6, :6], cv.P, atol=1e-9, rtol=0.0)
        assert np.allclose(x_ct, ct.x, atol=1e-9, rtol=0.0)
        assert np.allclose(P_ct, ct.P, atol=1e-9, rtol=0.0)
    assert swing > 3.0  # the turn really moved the probabilities, so the check has teeth
