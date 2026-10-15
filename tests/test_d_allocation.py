import itertools

import numpy as np
import pytest

from sim.coop.allocation import (
    assign,
    brute_force_assign,
    count_role_changes,
    distance_cost,
    observation_slots,
    reallocate,
    slots_for_alive,
)


def _independent_brute(cost):
    cost = np.asarray(cost, dtype=float)
    best = None
    for cols in itertools.permutations(range(cost.shape[1]), cost.shape[0]):
        total = sum(cost[row, col] for row, col in enumerate(cols))
        candidate = (float(total), tuple(cols))
        if best is None or candidate < best:
            best = candidate
    return best


@pytest.mark.parametrize("shape", [(n, n) for n in range(1, 7)] + [(3, 5), (4, 6)])
def test_assignment_matches_independent_brute(shape):
    rng = np.random.default_rng(np.random.SeedSequence(20261015 + sum(shape)))
    for _ in range(200):
        for cost in (rng.uniform(0.0, 10.0, shape), rng.integers(0, 5, shape).astype(float)):
            rows, cols = assign(cost)
            expected_total, _ = _independent_brute(cost)
            assert abs(float(cost[rows, cols].sum()) - expected_total) <= 1e-12
            module_total, _ = brute_force_assign(cost)
            assert abs(module_total - expected_total) <= 1e-12


def test_hand_assignment():
    # The optimum is rows 0, 1, 2 to columns 1, 0, 2: 1 + 2 + 2 = 5.
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    rows, cols = assign(cost)
    assert tuple(cols) == (1, 0, 2)
    assert cost[rows, cols].sum() == 5
    total, mapping = brute_force_assign(cost)
    assert total == 5.0
    assert mapping == (1, 0, 2)


def test_brute_force_lexicographic_tie_break():
    cost = np.ones((3, 5), dtype=float)
    assert brute_force_assign(cost) == (3.0, (0, 1, 2))


def test_observation_slots_ring():
    slots = observation_slots((1, 1, 99), 4, 2, 1.5)
    expected = np.array([[3, 1, 1.5], [1, 3, 1.5], [-1, 1, 1.5], [1, -1, 1.5]])
    assert np.allclose(slots, expected, atol=1e-12)
    assert np.allclose(np.linalg.norm(slots[:, :2] - np.array([1, 1]), axis=1), 2.0, atol=1e-12)


def test_reallocate_switch_penalty_and_swap():
    slots = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    positions = {"a": np.array([1, 0, 0]), "b": np.array([0, 0, 0])}
    assert reallocate(["a"], positions, slots, {"a": 1}, switch_penalty_m=0.1) == {"a": 1}
    assert reallocate(["a"], positions, slots, {"a": 1}, switch_penalty_m=0.0) == {"a": 0}
    assert reallocate(["a", "b"], positions, slots, {"a": 0, "b": 1}, switch_penalty_m=0.0) == {"a": 1, "b": 0}


def test_alive_replan_spacing_and_previous_ring_rule():
    original = slots_for_alive((0, 0), 4, 2, 1.0)
    alive = ["a", "b", "c"]
    positions = {agent: original[i] for i, agent in enumerate(alive)}
    slots = slots_for_alive((0, 0), 3, 2, 1.0)
    angles = np.sort(np.mod(np.arctan2(slots[:, 1], slots[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    assert np.allclose(gaps, 2 * np.pi / 3, atol=1e-12)
    allocation = reallocate(alive, positions, slots, previous=None)
    assert set(allocation.values()) == {0, 1, 2}

    midpoint = {"a": np.array([1.0, 0.0, 0.0])}
    assert reallocate(["a"], midpoint, np.array([[0, 0, 0], [2, 0, 0]], dtype=float), {"a": 8}, switch_penalty_m=0.1) == {"a": 0}


def test_count_role_changes_hand_history():
    history = [
        {"a": 0, "b": 1, "c": 2},
        {"a": 1, "b": 1, "c": 2},
        {"a": 1, "b": 0, "c": 2},
        {"a": 2, "b": 0, "c": 1},
        {"a": 2, "b": 2, "c": 1},
    ]
    assert count_role_changes(history) == 5


def test_forbidden_pairs():
    rows, cols = assign(np.array([[0, np.inf], [np.inf, 0.0]]))
    assert tuple(cols) == (0, 1)
    with pytest.raises(ValueError):
        assign(np.array([[np.inf, np.inf], [0.0, 1.0]]))


def test_distance_cost_squared():
    got = distance_cost(np.array([[0, 0, 0]]), np.array([[3, 4, 0]]), squared=True)
    assert got.shape == (1, 1) and got[0, 0] == 25
