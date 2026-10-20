"""Cross-checks: each sim/coop module against a reference written separately from the math.

The references in tests/reference_d/ were written from the math, separately
from sim/coop/; later fixes to them were checked against hand cases. Agreement
here is evidence that both follow the math, not that one copies the other.
"""
import numpy as np
import pytest

from sim.coop import allocation, consensus, fusion
from tests.reference_d.allocation_ref import brute_force_min_cost
from tests.reference_d.consensus_ref import consensus_ref, laplacian_ref, rate_ref
from tests.reference_d.fusion_ref import ci_grid, mean_variance_of_n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_hungarian_matches_brute_force_reference(n):
    rng = np.random.default_rng(np.random.SeedSequence([20261015, n]))
    for trial in range(150):
        if trial % 3 == 0:
            cost = rng.integers(0, 4, size=(n, n)).astype(float)  # many ties
        else:
            cost = rng.uniform(0.0, 10.0, size=(n, n))
        rows, cols = allocation.assign(cost)
        total, _ = brute_force_min_cost(cost)
        assert cost[rows, cols].sum() == pytest.approx(total, abs=1e-12)
        assert sorted(rows.tolist()) == list(range(n))
        assert len(set(cols.tolist())) == n


def test_hungarian_rectangular_matches_reference():
    rng = np.random.default_rng(np.random.SeedSequence([20261015, 99]))
    for shape in [(2, 4), (3, 5), (4, 6)]:
        for _ in range(100):
            cost = rng.uniform(0.0, 5.0, size=shape)
            rows, cols = allocation.assign(cost)
            total, _ = brute_force_min_cost(cost)
            assert cost[rows, cols].sum() == pytest.approx(total, abs=1e-12)


@pytest.mark.parametrize("graph", ["ring6", "line5", "star5", "complete4"])
def test_consensus_matches_closed_form_reference(graph):
    adjacency = {"ring6": consensus.ring(6), "line5": consensus.line(5),
                 "star5": consensus.star(5), "complete4": consensus.complete(4)}[graph]
    n = adjacency.shape[0]
    eps = 1.0 / (consensus.max_degree(adjacency) + 1.0)
    x0 = np.random.default_rng(np.random.SeedSequence([7, n])).uniform(-1, 1, size=(n, 3))
    result = consensus.run_consensus(x0, adjacency, eps, 25)
    L_ref = laplacian_ref(adjacency)
    np.testing.assert_allclose(consensus.laplacian(adjacency), L_ref, atol=0, rtol=0)
    for k in (0, 1, 5, 25):
        np.testing.assert_allclose(result.states[k], consensus_ref(L_ref, eps, x0, k)[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result.states[k], consensus_ref(L_ref, eps, x0, k)[1], atol=1e-10, rtol=0)
    assert consensus.spectral_rate(consensus.laplacian(adjacency), eps) == pytest.approx(rate_ref(L_ref, eps), abs=1e-12)


def test_ci_matches_brute_force_grid_reference():
    rng = np.random.default_rng(np.random.SeedSequence(31))
    for _ in range(20):
        A = rng.normal(size=(3, 3)); A = A @ A.T + 0.2 * np.eye(3)
        B = rng.normal(size=(3, 3)); B = B @ B.T + 0.2 * np.eye(3)
        a = rng.normal(size=3); b = rng.normal(size=3)
        x, P, w = fusion.covariance_intersection(a, A, b, B, criterion="trace")
        x_ref, P_ref, w_ref = ci_grid(a, A, b, B, step=1e-4)
        # the grid reference is only as fine as its step, so compare the objective
        assert np.trace(P) <= np.trace(P_ref) + 1e-9
        assert abs(w - w_ref) < 2e-3 or abs(np.trace(P) - np.trace(P_ref)) < 1e-7


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_information_fusion_variance_matches_sigma2_over_n(n):
    sigma = 0.05
    covs = [sigma**2 * np.eye(3)] * n
    _, P = fusion.information_fusion([np.zeros(3)] * n, covs)
    np.testing.assert_allclose(P, mean_variance_of_n(sigma, n) * np.eye(3), rtol=1e-12, atol=0)


def test_rate_matches_reference_on_disconnected_graph():
    a = np.zeros((4, 4))
    a[0, 1] = a[1, 0] = 1.0
    a[2, 3] = a[3, 2] = 1.0
    L = consensus.laplacian(a)
    assert consensus.spectral_rate(L, 0.4) == pytest.approx(rate_ref(laplacian_ref(a), 0.4), abs=1e-12)
    assert rate_ref(laplacian_ref(a), 0.4) == pytest.approx(1.0, abs=1e-12)


def test_ci_many_two_inputs_no_worse_than_grid_reference():
    rng = np.random.default_rng(np.random.SeedSequence(41))
    for _ in range(15):
        A = rng.normal(size=(3, 3)); A = A @ A.T + 0.2 * np.eye(3)
        B = rng.normal(size=(3, 3)); B = B @ B.T + 0.2 * np.eye(3)
        a = rng.normal(size=3); b = rng.normal(size=3)
        _, P, w = fusion.covariance_intersection_many([a, b], [A, B])
        _, P_ref, _ = ci_grid(a, A, b, B, step=1e-4)
        assert np.trace(P) <= np.trace(P_ref) + 1e-9
        assert w.sum() == pytest.approx(1.0, abs=1e-9) and np.all(w >= -1e-12)


def test_ci_many_three_inputs_no_worse_than_simplex_grid():
    rng = np.random.default_rng(np.random.SeedSequence(43))
    covs = []
    for _ in range(3):
        M = rng.normal(size=(2, 2)); covs.append(M @ M.T + 0.3 * np.eye(2))
    means = [rng.normal(size=2) for _ in range(3)]
    _, P, _ = fusion.covariance_intersection_many(means, covs)
    inverses = [np.linalg.inv(C) for C in covs]
    best = np.inf
    steps = np.linspace(0.0, 1.0, 201)
    for w1 in steps:
        for w2 in steps[steps <= 1.0 - w1 + 1e-12]:
            w3 = max(0.0, 1.0 - w1 - w2)
            info = w1 * inverses[0] + w2 * inverses[1] + w3 * inverses[2]
            best = min(best, np.trace(np.linalg.inv(info)))
    assert np.trace(P) <= best + 1e-9


def test_module_brute_force_matches_reference_brute_force():
    rng = np.random.default_rng(np.random.SeedSequence(47))
    for n in range(1, 7):
        for _ in range(30):
            cost = rng.uniform(0.0, 5.0, size=(n, n))
            total, cols = allocation.brute_force_assign(cost)
            ref_total, _ = brute_force_min_cost(cost)
            assert total == pytest.approx(ref_total, abs=1e-12)
            assert cost[np.arange(n), list(cols)].sum() == pytest.approx(total, abs=1e-12)
