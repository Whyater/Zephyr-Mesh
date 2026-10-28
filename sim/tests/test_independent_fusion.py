"""Independent checks for estimate fusion (sim/coop/fusion.py).

Every expected value here comes from outside the code under test:
hand arithmetic, closed forms worked out on paper, a brute-force search done
with plain NumPy, or statistics of errors that the test generates itself.
Nothing is compared to a second copy of the fusion code. One test checks
that the CI output has the textbook CI form (P^-1 = w Pa^-1 + (1 - w) Pb^-1);
that form is the same one the code uses, so the independent parts there are
recovering w from the output, the matrix bound against information fusion,
and the brute-force search over w.

The main idea is the Julier and Uhlmann (1997) fact. If two estimates share
some error (same wind model, same bad map) and the fuser does not know it,
adding their information over-counts and the reported covariance is too small.
Covariance intersection (CI) never reports a covariance smaller than it should.
The test makes that visible with the NEES (normalized estimation error squared,
e^T P^-1 e). For an honest estimate in k dimensions, NEES averages k.
"""
from __future__ import annotations

import numpy as np

from sim.coop.fusion import (
    covariance_intersection,
    covariance_intersection_fast,
    covariance_intersection_many,
    information_fusion,
    naive_average,
    nees,
)

Z = 4.0  # band half-width in standard errors; all seeds are fixed anyway


def _nees_mean_band(dof, runs, scale=1.0):
    # One NEES of an honest estimate is chi-square with dof degrees of freedom:
    # mean dof, variance 2*dof. The average of `runs` independent ones has
    # mean dof and standard deviation sqrt(2*dof/runs). If the claimed
    # covariance is too small by a factor `scale`, every NEES is multiplied by it.
    center = scale * dof
    half = Z * scale * np.sqrt(2.0 * dof / runs)
    return center - half, center + half


def _fused_mean_maps(fuse, covs):
    # For fixed covariances, CI picks its weights from the covariances alone, so
    # the fused mean is a linear function of the input means: x = sum_i A_i m_i.
    # We read A_i out of the code with unit-vector inputs, then apply it to
    # thousands of error samples with one matrix product. This keeps the optimizer
    # (slow, a few milliseconds per call) out of the Monte Carlo loop while the
    # mean still comes from the code under test. Returns the A_i and the covariance.
    k = covs[0].shape[0]
    maps, cov = [], None
    for i in range(len(covs)):
        cols = []
        for j in range(k):
            means = [np.zeros(k) for _ in covs]
            means[i][j] = 1.0
            x, cov = fuse(means, covs)
            cols.append(x)
        maps.append(np.stack(cols, axis=1))
    # Guard: the shortcut is only valid if the fused mean really is linear in
    # the input means (the weights must not depend on the means). Check it on
    # random means against a direct call; if a future change breaks this, the
    # NEES results built on the shortcut would be wrong, so fail loudly here.
    rng = np.random.default_rng(4242)
    for _ in range(3):
        means = [rng.normal(scale=5.0, size=k) for _ in covs]
        direct, _ = fuse(means, covs)
        linear = sum(a @ m for a, m in zip(maps, means))
        np.testing.assert_allclose(direct, linear, rtol=1e-9, atol=1e-9)
    return maps, cov


def _sample_pair_errors(rng, pa, pb, rho, runs):
    # Whiten: za, zb are unit-variance with correlation rho in each coordinate.
    # ea = La za and eb = Lb zb then have covariances pa and pb exactly,
    # with cross-covariance rho * La Lb^T. rho = 1 means fully shared error.
    k = pa.shape[0]
    la, lb = np.linalg.cholesky(pa), np.linalg.cholesky(pb)
    za = rng.normal(size=(runs, k))
    zb = rho * za + np.sqrt(1.0 - rho**2) * rng.normal(size=(runs, k))
    return za @ la.T, zb @ lb.T


def test_information_fusion_two_scalars_by_hand():
    """Independent fact: two independent measurements of one number.

    Variances 1 and 4, readings 0 and 10. Information (1/variance) adds:
    1/1 + 1/4 = 1.25, so the fused variance is 1/1.25 = 0.8. The mean is the
    inverse-variance weighted average: (0/1 + 10/4) * 0.8 = 2.0, which sits
    close to the tighter reading. Pure arithmetic, so the tolerance is round-off.
    """
    mean, cov = information_fusion([np.array([0.0]), np.array([10.0])], [np.array([[1.0]]), np.array([[4.0]])])
    assert abs(cov[0, 0] - 0.8) < 1e-12
    assert abs(mean[0] - 2.0) < 1e-12
    # Three readings: information 1 + 1/2 + 1/4 = 1.75, variance 4/7,
    # mean = (1*3 + 0.5*6 + 0.25*14) * 4/7 = 9.5 * 4/7 = 38/7.
    mean3, cov3 = information_fusion(
        [np.array([3.0]), np.array([6.0]), np.array([14.0])],
        [np.array([[1.0]]), np.array([[2.0]]), np.array([[4.0]])],
    )
    assert abs(cov3[0, 0] - 4.0 / 7.0) < 1e-12
    assert abs(mean3[0] - 38.0 / 7.0) < 1e-12


def test_independent_errors_give_consistent_information_fusion():
    """Independent fact: with truly independent errors, adding information is right.

    The two errors are drawn independently (rho = 0) with different covariances.
    The fused error then really has the reported covariance, so each NEES is
    chi-square(3) and the average over 4000 runs is 3 with standard error
    sqrt(6/4000) = 0.039. The band is +-4 standard errors.
    """
    rng = np.random.default_rng(np.random.SeedSequence(270101))
    pa = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.6]])
    pb = np.array([[0.7, -0.2, 0.1], [-0.2, 1.5, 0.0], [0.1, 0.0, 3.0]])
    runs = 4000
    ea, eb = _sample_pair_errors(rng, pa, pb, 0.0, runs)
    values = []
    for i in range(runs):
        # True state is the origin, so the error of each estimate is its mean.
        x, p = information_fusion([ea[i], eb[i]], [pa, pb])
        values.append(nees(x, p))
    low, high = _nees_mean_band(3, runs)
    assert low <= np.mean(values) <= high


def test_shared_error_makes_information_fusion_and_naive_average_overconfident():
    """Independent fact: fusing correlated errors as if independent is overconfident.

    Both estimates have covariance P and errors with correlation rho. The fused
    mean (equal weights) has error (ea + eb)/2, whose true covariance is
    P (1 + rho)/2. Information fusion claims P/2, so the true error is
    (1 + rho) times larger than claimed and the mean NEES is k (1 + rho):
    rho = 1 gives 2k, rho = 0.8 gives 1.8k. Three fully shared estimates through
    naive_average claim P/3 against a true P, so the mean NEES is 3k.
    The +-4 standard error band comes from scaling a chi-square(k) mean
    (2000 runs here).
    """
    rng = np.random.default_rng(np.random.SeedSequence(270102))
    p = np.array([[1.5, 0.4], [0.4, 0.8]])
    k, runs = 2, 2000
    for rho in (1.0, 0.8):
        ea, eb = _sample_pair_errors(rng, p, p, rho, runs)
        values = []
        for i in range(runs):
            x, cov = information_fusion([ea[i], eb[i]], [p, p])
            values.append(nees(x, cov))
        # Scaled chi-square: the NEES distribution is (1 + rho) times chi-square(k)
        # for equal-covariance estimates, so the same scaling applies to the band.
        low, high = _nees_mean_band(k, runs, scale=1.0 + rho)
        mean_nees = np.mean(values)
        assert low <= mean_nees <= high
        assert mean_nees > k * 1.5  # clearly above the honest value k
    shared = rng.normal(size=(runs, k)) @ np.linalg.cholesky(p).T
    values = []
    for i in range(runs):
        x, cov = naive_average([shared[i]] * 3, [p] * 3)
        values.append(nees(x, cov))
    low, high = _nees_mean_band(k, runs, scale=3.0)
    assert low <= np.mean(values) <= high


def test_ci_stays_consistent_when_errors_are_fully_shared():
    """Independent fact: CI never claims a covariance below the true one.

    Julier and Uhlmann: for any cross-covariance consistent with the two
    inputs, P_ci >= true fused covariance. So the mean NEES of the fused
    estimate must be at most k. We test a hard case, rho = 1 with unequal
    covariances (errors fully shared in whitened coordinates), and a milder rho.
    With equal covariances CI returns P for any weight, and the true error
    covariance is P (w^2 + (1-w)^2 + 2 w (1-w) rho), which is exactly P at
    rho = 1, so the mean NEES is k (band as in the independent case). At
    rho = 0.8 it is at most k, so only the upper bound is asserted (the exact
    value depends on the tied weight). For unequal covariances we also only
    assert the upper bound k + band, since CI is allowed to be conservative.
    """
    rng = np.random.default_rng(np.random.SeedSequence(270103))
    k, runs = 3, 4000
    p = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    # Equal covariances: exact expectations.
    for rho in (1.0, 0.8):
        ea, eb = _sample_pair_errors(rng, p, p, rho, runs)
        maps, cov = _fused_mean_maps(lambda m, c: covariance_intersection(m[0], c[0], m[1], c[1])[:2], [p, p])
        fused = ea @ maps[0].T + eb @ maps[1].T
        values = [nees(x, cov) for x in fused]
        low, high = _nees_mean_band(k, runs)
        assert np.mean(values) <= high
        if rho == 1.0:
            assert np.mean(values) >= low
    # Unequal covariances, fully shared error: only the upper bound is guaranteed.
    pa = np.array([[3.0, 0.5, 0.0], [0.5, 0.4, 0.0], [0.0, 0.0, 1.0]])
    pb = np.array([[0.5, 0.0, 0.1], [0.0, 2.5, 0.3], [0.1, 0.3, 0.8]])
    ea, eb = _sample_pair_errors(rng, pa, pb, 1.0, runs)
    maps, cov = _fused_mean_maps(lambda m, c: covariance_intersection(m[0], c[0], m[1], c[1])[:2], [pa, pb])
    ci_values = [nees(x, cov) for x in ea @ maps[0].T + eb @ maps[1].T]
    if_values = []
    for i in range(runs):
        x, cov_if = information_fusion([ea[i], eb[i]], [pa, pb])
        if_values.append(nees(x, cov_if))
    assert np.mean(ci_values) <= _nees_mean_band(k, runs)[1]
    # Same errors through information fusion: clearly overconfident.
    assert np.mean(if_values) > _nees_mean_band(k, runs)[1]
    assert np.mean(if_values) > np.mean(ci_values)


def test_ci_closed_form_limits():
    """Independent fact: simple cases where the best CI answer is known on paper.

    Information of the blend is w/Pa + (1-w)/Pb with 0 <= w <= 1.
    (1) Pa = Pb = P: blend information is 1/P for every w, so the covariance is
        P (no information is invented), whatever w is chosen.
    (2) One dimension, variances 2 and 5: blend variance 1/(w/2 + (1-w)/5) is
        smallest at w = 1, giving exactly 2, and the mean is the tighter mean.
        This holds for both the trace and the determinant criterion.
    (3) Pa tighter than Pb in every direction (Pb - Pa positive): the blend
        information only grows with w, so w = 1 and the answer is estimate a.
    Tolerances are round-off level because the optimizer may land on the end
    point exactly (1e-8 allows for an interior search that ends next to it).
    """
    p = np.array([[2.0, 0.5], [0.5, 1.0]])
    _, cov, _ = covariance_intersection(np.array([0.0, 1.0]), p, np.array([3.0, -1.0]), p)
    assert np.allclose(cov, p, atol=1e-10)
    for criterion in ("trace", "det"):
        x, cov, w = covariance_intersection(
            np.array([4.0]), np.array([[2.0]]), np.array([9.0]), np.array([[5.0]]), criterion=criterion
        )
        assert abs(cov[0, 0] - 2.0) < 1e-8
        assert abs(x[0] - 4.0) < 1e-7
        assert w > 1.0 - 1e-6
        # Swap the inputs: now b is the tighter one and w should go to 0.
        x, cov, w = covariance_intersection(
            np.array([9.0]), np.array([[5.0]]), np.array([4.0]), np.array([[2.0]]), criterion=criterion
        )
        assert abs(cov[0, 0] - 2.0) < 1e-8
        assert abs(x[0] - 4.0) < 1e-7
        assert w < 1e-6
    pa = np.array([[1.0, 0.2], [0.2, 2.0]])
    pb = pa + np.array([[9.0, 1.0], [1.0, 18.0]])  # difference is positive definite
    xa, xb = np.array([1.0, 2.0]), np.array([5.0, -3.0])
    x, cov, w = covariance_intersection(xa, pa, xb, pb)
    assert w > 1.0 - 1e-6
    assert np.allclose(x, xa, atol=1e-6)
    assert np.allclose(cov, pa, atol=1e-6)


def test_ci_result_has_the_ci_structure_and_is_not_better_than_information_fusion():
    """Independent fact: the defining equations of covariance intersection.

    CI is defined by P^-1 = w Pa^-1 + (1-w) Pb^-1 with 0 <= w <= 1. We do not
    ask the code for w: we solve for it from the returned covariance (least
    squares of P^-1 - Pb^-1 against Pa^-1 - Pb^-1) and require a near-zero
    residual, w inside [0, 1], and agreement with the w the function reports.
    Because w Pa^-1 + (1-w) Pb^-1 <= Pa^-1 + Pb^-1 in the positive-semidefinite
    order, the CI covariance can never be smaller than the information-fusion
    covariance (Pa^-1 + Pb^-1)^-1, which we build here with np.linalg.inv.
    Finally, the chosen w must beat every weight on a fine grid for trace(P),
    searched here with plain NumPy. Tolerances: 1e-8 for linear algebra on small
    well-conditioned matrices; 1e-6 relative for the optimizer.
    """
    rng = np.random.default_rng(np.random.SeedSequence(270104))
    for _ in range(10):
        qa, qb = rng.normal(size=(2, 3, 3))
        pa = qa @ qa.T + 0.3 * np.eye(3)
        pb = qb @ qb.T + 0.3 * np.eye(3)
        xa, xb = rng.normal(size=(2, 3))
        x, cov, w_reported = covariance_intersection(xa, pa, xb, pb)
        ia, ib = np.linalg.inv(pa), np.linalg.inv(pb)
        info = np.linalg.inv(cov)
        diff, span = info - ib, ia - ib
        w = float(np.sum(diff * span) / np.sum(span * span))
        assert -1e-9 <= w <= 1.0 + 1e-9
        assert np.allclose(info, w * ia + (1.0 - w) * ib, atol=1e-8)
        assert abs(w - w_reported) < 1e-6
        # Mean from the same definition: x = P (w Pa^-1 xa + (1-w) Pb^-1 xb).
        assert np.allclose(x, cov @ (w * ia @ xa + (1.0 - w) * ib @ xb), atol=1e-8)
        # Never tighter than information fusion.
        p_if = np.linalg.inv(ia + ib)
        assert np.min(np.linalg.eigvalsh(cov - p_if)) >= -1e-9
        # Trace-optimal against a brute-force grid done here.
        grid = np.linspace(0.0, 1.0, 2001)
        traces = [np.trace(np.linalg.inv(g * ia + (1.0 - g) * ib)) for g in grid]
        assert np.trace(cov) <= min(traces) * (1.0 + 1e-6)


def test_nees_matches_hand_quadratic_form():
    """Independent fact: NEES is e^T P^-1 e, worked on paper.

    P = [[2, 1], [1, 2]] has determinant 3 and inverse (1/3) [[2, -1], [-1, 2]].
    For e = (1, 2): P^-1 e = (1/3)(2 - 2, -1 + 4) = (0, 1), so e^T P^-1 e = 2.
    A diagonal P cannot catch a transposed or un-inverted off-diagonal term,
    this one can. Using P instead of P^-1 would give 1*(2+2) + 2*(1+4) = 14.
    Scaling the error by 3 scales NEES by 9, and scaling P by 4 divides by 4.
    """
    p = np.array([[2.0, 1.0], [1.0, 2.0]])
    e = np.array([1.0, 2.0])
    assert abs(nees(e, p) - 2.0) < 1e-12
    assert abs(nees(3.0 * e, p) - 18.0) < 1e-11
    assert abs(nees(e, 4.0 * p) - 0.5) < 1e-12
    # e = (2, -2): P^-1 e = (1/3)(4 + 2, -2 - 4) = (2, -2), so NEES = 4 + 4 = 8.
    assert abs(nees(np.array([2.0, -2.0]), p) - 8.0) < 1e-11


def test_fast_ci_and_many_ci_stay_consistent_under_shared_error():
    """Independent fact: any convex CI weights give a consistent estimate.

    The fast variant uses weights 1/trace(P_i), normalized to sum to 1. For any
    weights in the simplex, P^-1 = sum w_i P_i^-1 is a valid CI blend, so even
    when all three estimates share one underlying error (fully correlated in
    whitened coordinates, a hard case) the mean NEES must be at most k.
    The simplex-weight optimizer version must satisfy the same bound. Information
    fusion on the same errors is the control: its claimed covariance is about
    three times too small in the shared direction, so it must exceed the band.
    The band is the +-4 standard error one around k.
    """
    rng = np.random.default_rng(np.random.SeedSequence(270105))
    k, runs = 2, 3000
    covs = [
        np.array([[2.0, 0.3], [0.3, 1.0]]),
        np.array([[0.6, -0.1], [-0.1, 1.8]]),
        np.array([[1.2, 0.0], [0.0, 0.7]]),
    ]
    chols = [np.linalg.cholesky(c) for c in covs]
    z = rng.normal(size=(runs, k))
    errors = [z @ ch.T for ch in chols]
    _, _, weights = covariance_intersection_fast([e[0] for e in errors], covs)
    assert abs(float(np.sum(weights)) - 1.0) < 1e-12 and np.all(weights >= 0.0)
    values = {}
    for name, fuse in (
        ("fast", lambda m, c: covariance_intersection_fast(m, c)[:2]),
        ("many", lambda m, c: covariance_intersection_many(m, c)[:2]),
    ):
        maps, cov = _fused_mean_maps(fuse, covs)
        fused = sum(e @ a.T for e, a in zip(errors, maps))
        values[name] = np.mean([nees(x, cov) for x in fused])
    if_values = []
    for i in range(runs):
        x, cov = information_fusion([e[i] for e in errors], covs)
        if_values.append(nees(x, cov))
    upper = _nees_mean_band(k, runs)[1]
    assert values["fast"] <= upper
    assert values["many"] <= upper
    assert np.mean(if_values) > upper
