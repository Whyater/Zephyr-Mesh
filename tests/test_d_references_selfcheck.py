import numpy as np

from tests.reference_d.fusion_ref import batch_central_cv, ci_grid
from tests.reference_d.allocation_ref import brute_force_min_cost
from tests.reference_d.consensus_ref import consensus_ref, laplacian_ref
from tests.reference_d.loss_ref import gap_pmf, simulate_gaps


def test_kalman_matches_batch_zero_process_noise():
    x0 = np.array([0.2, -0.1, 0.4, 1.0, 0.5, -0.2])
    P0 = np.diag([2.0, 1.5, 1.2, 0.8, 0.7, 0.9])
    measurements = [(0.0, np.array([0.3, -0.2, 0.5]), np.diag([0.2, 0.3, 0.25])),
                    (0.7, np.array([1.0, 0.2, 0.1]), np.diag([0.4, 0.35, 0.3])),
                    (0.7, np.array([0.8, 0.1, 0.2]), np.diag([0.5, 0.45, 0.4]))]
    x, P = batch_central_cv(measurements, x0, P0, 0.0)
    I3 = np.eye(3)
    normal = np.linalg.inv(P0)
    vector = normal @ x0
    for stamp, obs, cov in measurements:
        G = np.hstack((I3, float(stamp) * I3))
        weight = np.linalg.inv(cov)
        normal += G.T @ weight @ G
        vector += G.T @ weight @ obs
    x_batch0 = np.linalg.solve(normal, vector)
    P_batch0 = np.linalg.inv(normal)
    final_t = max(stamp for stamp, _, _ in measurements)
    F = np.block([[I3, final_t * I3], [np.zeros((3, 3)), I3]])
    x_batch = F @ x_batch0
    P_batch = F @ P_batch0 @ F.T
    assert np.allclose(x, x_batch, atol=1e-9, rtol=0)
    assert np.allclose(P, P_batch, atol=1e-9, rtol=0)
    assert x.shape == (6,) and P.shape == (6, 6)


def test_consensus_forms_agree():
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    L = laplacian_ref(A)
    a, b = consensus_ref(L, 0.1, np.array([1.0, 2.0, -1.0]), 7)
    assert np.allclose(a, b, atol=1e-10)


def test_gap_monte_carlo_matches_pmf():
    for n in range(1, 5):
        gaps = np.asarray(simulate_gaps(0.5, n, 200000, 1234 + n))
        for k in (1, 2, 3):
            observed = np.mean(gaps == k)
            expected = gap_pmf(k, 0.5, n)
            se = np.sqrt(expected * (1 - expected) / len(gaps))
            assert abs(observed - expected) <= 3 * se + 1e-12


def test_ci_and_assignment_hand_cases():
    _, P, w = ci_grid(np.zeros(2), np.diag([1.0, 4.0]), np.zeros(2), np.diag([4.0, 1.0]), step=1e-4)
    assert abs(w - 0.5) < 1e-6
    assert np.allclose(P, np.diag([1.6, 1.6]), atol=1e-6)
    total, assignment = brute_force_min_cost([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert total == 5 and assignment == (1, 0, 2)
