import numpy as np

from sim.coop.consensus import (
    algebraic_connectivity,
    complete,
    eigenvalues,
    laplacian,
    line,
    measured_rate,
    max_degree,
    random_geometric,
    ring,
    run_consensus,
    spectral_rate,
    steps_to_agreement,
    star,
)


def test_hand_graph_spectra():
    assert np.allclose(eigenvalues(laplacian(complete(4))), [0, 4, 4, 4], atol=1e-12)
    assert np.allclose(eigenvalues(laplacian(ring(4))), [0, 2, 2, 4], atol=1e-12)
    assert np.allclose(eigenvalues(laplacian(star(5))), [0, 1, 1, 1, 5], atol=1e-12)
    assert np.allclose(eigenvalues(laplacian(line(3))), [0, 1, 3], atol=1e-12)


def test_matrix_power_and_average():
    a = ring(6)
    initial = np.arange(6.0)[:, None]
    eps = 1.0 / (np.max(np.sum(a, axis=1)) + 1)
    result = run_consensus(initial, a, eps, 8)
    expected = np.linalg.matrix_power(np.eye(6) - eps * laplacian(a), 8) @ initial
    assert np.allclose(result.states[8], expected, atol=1e-10)
    assert np.allclose(result.average, np.mean(initial, axis=0), atol=1e-12)


def test_bipartite_boundary_rate_and_oscillation():
    a = ring(4)
    initial = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    result = run_consensus(initial, a, 0.5, 50)
    assert np.isclose(spectral_rate(laplacian(a), 0.5), 1.0, atol=1e-12)
    assert np.allclose(result.disagreement, 2.0, atol=1e-12)


def test_loss_modes_and_disconnected_graph():
    a = ring(6)
    initial = np.arange(6.0)[:, None]
    symmetric = run_consensus(initial, a, 0.2, 30, loss_probability=0.3, symmetric_loss=True, seed=3)
    directed = run_consensus(initial, a, 0.2, 30, loss_probability=0.3, seed=3)
    assert np.allclose(symmetric.average, symmetric.average[0], atol=1e-12)
    assert abs(directed.average[-1, 0] - directed.average[0, 0]) > 1e-8
    disconnected = np.zeros((4, 4), dtype=float)
    disconnected[0, 1] = disconnected[1, 0] = disconnected[2, 3] = disconnected[3, 2] = 1
    assert abs(algebraic_connectivity(laplacian(disconnected))) <= 1e-12
    starts = np.array([[0.0], [2.0], [10.0], [14.0]])
    result = run_consensus(starts, disconnected, 0.2, 80)
    component_avgs = np.array([1.0, 1.0, 12.0, 12.0])[:, None]
    between = np.linalg.norm(component_avgs - np.mean(starts, axis=0))
    assert np.isclose(result.disagreement[-1], between, atol=1e-10)


def test_measured_rate_helper():
    values = np.array([1.0, 0.25, 0.0625])
    assert measured_rate(values, 0, 2) == 0.25


def test_measured_rate_matches_spectral_rate_on_connected_graphs():
    graphs = [ring(6), line(5), star(5)]
    geometric, _ = random_geometric(6, 2.0, seed=17)
    graphs.append(geometric)
    rng = np.random.default_rng(np.random.SeedSequence(91))
    for adjacency in graphs:
        initial = rng.normal(size=(adjacency.shape[0], 1))
        eps = 1.0 / (max_degree(adjacency) + 1.0)
        result = run_consensus(initial, adjacency, eps, 60)
        measured = measured_rate(result.disagreement, 20, 60)
        expected = spectral_rate(laplacian(adjacency), eps)
        assert np.isclose(measured, expected, rtol=1e-6, atol=0.0)


def test_complete_graph_agrees_in_one_step():
    initial = np.array([[2.0], [-1.0], [8.0], [3.0]])
    result = run_consensus(initial, complete(4), 0.25, 3)
    assert np.allclose(result.states[1], np.mean(initial, axis=0), atol=1e-12)
    assert np.allclose(result.disagreement[1:], 0.0, atol=1e-12)


def test_loss_rate_medians_match_mean_field_prediction():
    adjacency = ring(6)
    expected = spectral_rate(0.7 * laplacian(adjacency), 0.2)
    for symmetric in (False, True):
        rates = []
        for seed in range(200):
            result = run_consensus(np.arange(6.0)[:, None], adjacency, 0.2, 60,
                                   loss_probability=0.3, symmetric_loss=symmetric,
                                   seed=seed)
            rates.append(measured_rate(result.disagreement, 20, 60))
        assert abs(float(np.median(rates)) - expected) <= 0.02


def test_weighted_matrix_power_and_negative_weights():
    adjacency = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 2.0], [0.0, 2.0, 0.0]])
    initial = np.array([[1.0], [4.0], [-2.0]])
    eps = 0.1
    result = run_consensus(initial, adjacency, eps, 7)
    expected = np.linalg.matrix_power(np.eye(3) - eps * laplacian(adjacency), 7) @ initial
    assert np.allclose(result.states[7], expected, atol=1e-12)
    with np.testing.assert_raises(ValueError):
        laplacian(np.array([[0.0, -1.0], [-1.0, 0.0]]))


def test_fresh_delay_semantics_and_steps_to_agreement():
    initial = np.array([[0.0], [1.0], [4.0], [2.0], [7.0], [3.0]])
    immediate = run_consensus(initial, ring(6), 0.2, 100)
    # The ring is regular (every agent has two neighbors), which is why the average survives the delay below.
    delayed = run_consensus(initial, ring(6), 0.2, 100, delay_steps=2)
    assert steps_to_agreement(immediate, 1e-3) < steps_to_agreement(delayed, 1e-3)
    assert np.allclose(immediate.average, np.mean(initial, axis=0), atol=1e-12)
    assert np.allclose(delayed.average, np.mean(initial, axis=0), atol=1e-12)
    assert steps_to_agreement(immediate, 1.0) == 0


def test_delay_breaks_average_on_irregular_graph():
    # star(5) is not regular (the hub has four neighbors, the leaves one), so comparing delayed
    # neighbor values with current own values moves the agreed value away from the start average.
    initial = np.array([[0.0], [1.0], [4.0], [2.0], [7.0]])
    a = star(5)
    result = run_consensus(initial, a, 1.0 / (max_degree(a) + 1.0), 300, delay_steps=2)
    assert abs(float(result.average[-1, 0]) - float(np.mean(initial))) > 1e-3
