import numpy as np

from sim.estimation.morris import (
    elementary_effects,
    morris_screen,
    morris_trajectories,
    scale,
)


def test_trajectories_have_single_positive_step():
    design, orders = morris_trajectories(5, 20, levels=4, seed=3)
    delta = 4 / (2 * 3)
    differences = np.diff(design, axis=1)
    assert design.shape == (20, 6, 5)
    assert orders.shape == (20, 5)
    assert np.all((design >= 0) & (design <= 1))
    assert np.all(np.sum(np.abs(differences) > 1e-12, axis=2) == 1)
    assert np.allclose(np.max(differences, axis=2), delta)
    assert np.allclose(np.min(differences, axis=2), 0.0)


def test_linear_effects_are_exact_and_ranked():
    design, orders = morris_trajectories(5, 25, levels=4, seed=4)
    b = np.array([5.0, 3.0, 1.0, 0.5, 0.0])
    y = np.sum(design * b, axis=2)
    effects = elementary_effects(design, orders, y)
    assert np.allclose(effects["mu_star"], np.abs(b))
    assert np.allclose(effects["mu"], b)
    assert np.allclose(effects["sigma"], 0.0)
    assert np.array_equal(np.argsort(-effects["mu_star"]), [0, 1, 2, 3, 4])


def test_sobol_g_ranking_and_ignored_input():
    a = np.array([0.0, 1.0, 4.5, 9.0, 99.0])

    def g(x):
        return float(np.prod((np.abs(4.0 * x - 2.0) + a) / (1.0 + a)))

    result = morris_screen(g, 5, 50, levels=4, seed=7)
    assert np.array_equal(np.argsort(-result["mu_star"]), [0, 1, 2, 3, 4])

    def ignored(x):
        return float(x[0] + 2.0 * x[1])

    ignored_result = morris_screen(ignored, 3, 20, levels=4, seed=9)
    assert ignored_result["mu_star"][2] == 0.0


def test_scale_linear_and_log():
    unit = np.array([[0.0, 0.5, 1.0]])
    assert np.allclose(scale(unit, [1, 10, 100], [3, 20, 1000]), [[1, 15, 1000]])
    assert np.allclose(scale(unit, [1, 10, 100], [3, 100, 1000], log=[False, True, True]), [[1, np.sqrt(1000), 1000]])
