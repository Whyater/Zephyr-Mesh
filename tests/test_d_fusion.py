import numpy as np
import pytest

from sim.coop.fusion import (
    covariance_intersection_fast,
    covariance_intersection,
    covariance_intersection_many,
    information_fusion,
    naive_average,
    nees,
)


def test_information_fusion_equal_variance_hand_check():
    sigma2 = 2.5
    means = [np.array([1.0, -2.0, 0.5]) for _ in range(6)]
    for n in range(1, 7):
        _, p = information_fusion(means[:n], [sigma2 * np.eye(3)] * n)
        assert np.allclose(p, (sigma2 / n) * np.eye(3), atol=1e-14)


def test_information_fusion_monte_carlo_covariance():
    rng = np.random.default_rng(np.random.SeedSequence(20261015))
    sigma = 1.7
    for n in (2, 3, 4, 6):
        draws = []
        p = sigma**2 * np.eye(3)
        for _ in range(20000):
            noisy = [rng.normal(size=3) * sigma for _ in range(n)]
            draws.append(information_fusion(noisy, [p] * n)[0])
        empirical = np.cov(np.asarray(draws), rowvar=False, ddof=1)
        expected = (sigma**2 / n) * np.eye(3)
        se = (sigma**2 / n) * np.sqrt(2.0 / 19999.0)
        assert np.all(np.abs(np.diag(empirical) - np.diag(expected)) <= 3 * se)
        assert np.all(np.abs(empirical[np.triu_indices(3, 1)]) <= 3 * se)


def test_ci_identical_estimates_and_symmetric_case():
    mean = np.array([1.0, 2.0])
    _, naive_p = information_fusion([mean, mean], [np.eye(2), np.eye(2)])
    x, p, w = covariance_intersection(mean, np.eye(2), mean, np.eye(2))
    assert np.array_equal(naive_p, 0.5 * np.eye(2))
    assert np.array_equal(p, np.eye(2))
    assert np.array_equal(x, mean)
    assert 0.0 <= w <= 1.0
    a_cov = np.diag([1.0, 4.0])
    b_cov = np.diag([4.0, 1.0])
    _, p2, w2 = covariance_intersection(np.zeros(2), a_cov, np.zeros(2), b_cov)
    assert abs(w2 - 0.5) < 1e-8
    assert np.allclose(p2, np.diag([1.6, 1.6]), atol=1e-10)


def test_ci_consistency_for_correlated_errors():
    rng = np.random.default_rng(np.random.SeedSequence(77))
    for _ in range(8):
        q1 = rng.normal(size=(3, 3))
        q2 = rng.normal(size=(3, 3))
        a = q1 @ q1.T + 0.5 * np.eye(3)
        b = q2 @ q2.T + 0.5 * np.eye(3)
        if _ == 0:
            c = np.linalg.cholesky(a) @ np.linalg.cholesky(b).T
        elif _ == 1:
            c = np.zeros((3, 3))
        else:
            r = rng.normal(size=(3, 3))
            c = 0.2 * r
            # Construct a valid joint covariance by shrinking cross-covariance.
            scale = min(1.0, 0.8 / max(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(a), c @ np.linalg.inv(np.linalg.cholesky(b).T)), 2), 1e-12))
            c *= scale
        _, p, w = covariance_intersection(np.zeros(3), a, np.zeros(3), b)
        ka = w * p @ np.linalg.solve(a, np.eye(3))
        kb = (1.0 - w) * p @ np.linalg.solve(b, np.eye(3))
        true = ka @ a @ ka.T + ka @ c @ kb.T + kb @ c.T @ ka.T + kb @ b @ kb.T
        assert np.min(np.linalg.eigvalsh(p - true)) >= -1e-12


def test_many_ci_average_and_naive_average():
    means = [np.array([1.0, 2.0]), np.array([3.0, 0.0])]
    covs = [np.eye(2), 2.0 * np.eye(2)]
    x1, p1, w1 = covariance_intersection_many(means, covs)
    x2, p2, w2 = covariance_intersection(means[0], covs[0], means[1], covs[1])
    assert np.allclose(p1, p2, atol=1e-6)
    assert np.allclose(x1, x2, atol=1e-6)
    assert np.isclose(np.sum(w1), 1.0)
    assert np.all(w1 >= 0.0)
    _, identical_p, identical_w = covariance_intersection_many(means, [np.eye(2)] * 2)
    assert np.allclose(identical_p, np.eye(2))
    assert np.all(identical_w >= 0.0) and np.isclose(np.sum(identical_w), 1.0)
    x, p = naive_average(means, covs)
    assert np.array_equal(x, np.mean(means, axis=0))
    assert np.array_equal(p, (covs[0] + covs[1]) / 4.0)
    _, p_same = naive_average(means + [np.array([0.0, 0.0])], [3.0 * np.eye(2)] * 3)
    assert np.array_equal(p_same, np.eye(2))


def test_nees_and_validation():
    assert nees(np.array([1.0, 0.0, 0.0]), np.diag([4.0, 1.0, 1.0])) == 0.25
    with pytest.raises(ValueError):
        nees(np.ones(2), np.eye(3))
    with pytest.raises(ValueError):
        nees(np.array([np.nan]), np.eye(1))
    with pytest.raises(ValueError):
        nees(np.ones(2), np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        information_fusion([np.zeros(2)], [np.eye(3)])
    with pytest.raises(ValueError):
        covariance_intersection(np.zeros(1), np.eye(1), np.zeros(1), np.eye(1), criterion="bad")


def test_determinism():
    args = ([np.array([1.0, 2.0]), np.array([-1.0, 3.0])], [np.diag([1.0, 2.0]), np.diag([2.0, 1.0])])
    first = covariance_intersection_many(*args)
    second = covariance_intersection_many(*args)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_fast_ci_hand_values_and_conservative_versus_optimal():
    # Two diagonal tracks: traces 2 and 8 give weights 0.8 and 0.2.
    pa, pb = np.diag([1.0, 1.0]), np.diag([4.0, 4.0])
    xa, xb = np.array([0.0, 0.0]), np.array([1.0, 2.0])
    mean, cov, weights = covariance_intersection_fast([xa, xb], [pa, pb])
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-15)
    info = 0.8 * 1.0 + 0.2 * 0.25  # per axis
    np.testing.assert_allclose(cov, np.eye(2) / info, atol=1e-14)
    np.testing.assert_allclose(mean, (0.2 * 0.25 * xb) / info, atol=1e-14)
    rng = np.random.default_rng(20261015)
    for _ in range(20):
        mats = [m @ m.T + 0.1 * np.eye(3) for m in rng.normal(size=(3, 3, 3))]
        means = list(rng.normal(size=(3, 3)))
        _, fast_cov, _ = covariance_intersection_fast(means, mats)
        _, best_cov, _ = covariance_intersection_many(means, mats)
        assert np.trace(fast_cov) >= np.trace(best_cov) - 1e-9
