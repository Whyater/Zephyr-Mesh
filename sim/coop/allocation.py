"""Deterministic simulated observer-to-slot allocation helpers."""

from itertools import permutations
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def _cost_array(cost: NDArray) -> NDArray:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError("cost must be a two-dimensional array")
    if np.isnan(arr).any() or np.isneginf(arr).any():
        raise ValueError("cost contains an invalid non-finite value")
    return arr


def assign(cost: NDArray) -> tuple[NDArray, NDArray]:
    """Return a minimum-cost assignment, sorted by observer row."""
    arr = _cost_array(cost)
    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    forbidden = np.isposinf(arr)
    work = np.where(forbidden, 1e9, arr)
    rows, cols = linear_sum_assignment(work)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    chosen_forbidden = forbidden[rows, cols]
    if chosen_forbidden.any():
        cardinality = min(n_rows, n_cols)
        finite_rows, finite_cols = linear_sum_assignment(
            np.where(forbidden, 1.0, 0.0)
        )
        finite_matching = int(np.count_nonzero(~forbidden[finite_rows, finite_cols]))
        if finite_matching >= cardinality:
            raise ValueError("forbidden pair selected despite a finite assignment")
        raise ValueError("no assignment exists without a forbidden pair")
    order = np.argsort(rows, kind="stable")
    return rows[order], cols[order]


def brute_force_assign(cost: NDArray) -> tuple[float, tuple[int, ...]]:
    """Compute the minimum injective row-to-column map for small arrays."""
    arr = _cost_array(cost)
    n_rows, n_cols = arr.shape
    if n_rows > n_cols:
        raise ValueError("brute_force_assign requires rows <= columns")
    if n_rows > 8:
        raise ValueError("brute_force_assign supports at most 8 rows")
    best_total = float("inf")
    best_cols: tuple[int, ...] | None = None
    for candidate in permutations(range(n_cols), n_rows):
        total = float(sum(arr[i, candidate[i]] for i in range(n_rows)))
        if total < best_total or (total == best_total and (best_cols is None or candidate < best_cols)):
            best_total = total
            best_cols = candidate
    assert best_cols is not None
    return best_total, best_cols


def observation_slots(
    center, n_slots: int, radius_m: float, altitude_m: float, phase_rad: float = 0.0
) -> NDArray:
    """Return a simulated non-contact standoff ring at an absolute altitude."""
    c = np.asarray(center, dtype=float).reshape(-1)
    if c.size < 2 or not np.isfinite(c[:2]).all():
        raise ValueError("center must provide finite x and y")
    if n_slots < 0 or int(n_slots) != n_slots or radius_m < 0 or not np.isfinite([radius_m, altitude_m, phase_rad]).all():
        raise ValueError("invalid ring parameters")
    n_slots = int(n_slots)
    if n_slots == 0:
        return np.empty((0, 3), dtype=float)
    angles = phase_rad + 2.0 * np.pi * np.arange(n_slots) / n_slots
    return np.column_stack(
        (c[0] + radius_m * np.cos(angles), c[1] + radius_m * np.sin(angles), np.full(n_slots, altitude_m))
    )


def distance_cost(positions: NDArray, slots: NDArray, *, squared: bool = False) -> NDArray:
    """Return Euclidean distances between simulated observer positions and slots."""
    p = np.asarray(positions, dtype=float)
    s = np.asarray(slots, dtype=float)
    if p.ndim != 2 or s.ndim != 2 or p.shape[1:] != (3,) or s.shape[1:] != (3,):
        raise ValueError("positions and slots must have shape (n, 3) and (m, 3)")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("positions and slots must be finite")
    d2 = ((p[:, None, :] - s[None, :, :]) ** 2).sum(axis=2)
    return d2 if squared else np.sqrt(d2)


def reallocate(
    agent_ids: Sequence[str],
    positions: Mapping[str, NDArray],
    slots: NDArray,
    previous: Mapping[str, int] | None = None,
    *,
    switch_penalty_m: float = 0.0,
) -> dict[str, int]:
    """Assign sorted alive observers to current slots with optional hysteresis.

    Previous indices refer to the current slot ring. An out-of-range previous
    index is ignored and treated as if no previous assignment existed.
    """
    ids = sorted(agent_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("agent_ids must be unique")
    if not np.isfinite(switch_penalty_m) or switch_penalty_m < 0:
        raise ValueError("switch_penalty_m must be finite and nonnegative")
    try:
        pos = np.asarray([positions[agent] for agent in ids], dtype=float)
    except KeyError as exc:
        raise ValueError(f"missing position for {exc.args[0]}") from exc
    base = distance_cost(pos, slots)
    if len(ids) > base.shape[1]:
        raise ValueError("there are fewer slots than alive observers")
    if previous is not None:
        for i, agent in enumerate(ids):
            if agent in previous:
                old = int(previous[agent])
                if old < 0 or old >= base.shape[1]:
                    continue
                if switch_penalty_m:
                    base[i, :] += switch_penalty_m
                    base[i, old] -= switch_penalty_m
    rows, cols = assign(base)
    return {ids[int(row)]: int(col) for row, col in zip(rows, cols)}


def slots_for_alive(center, n_alive: int, radius_m, altitude_m, phase_rad=0.0) -> NDArray:
    """Re-space a simulated standoff ring after an observer drop."""
    return observation_slots(center, n_alive, radius_m, altitude_m, phase_rad)


def count_role_changes(history: Sequence[Mapping[str, int]]) -> int:
    """Count slot changes for observers present in adjacent allocation snapshots."""
    changes = 0
    for before, after in zip(history, history[1:]):
        for agent in sorted(set(before).intersection(after)):
            if before[agent] != after[agent]:
                changes += 1
    return changes
