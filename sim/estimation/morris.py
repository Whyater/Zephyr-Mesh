"""Morris elementary effects for screening scaled model inputs.

The design is built on a regular unit cube grid. Each trajectory changes one
coordinate at a time by the prescribed Morris step, so a model can be sampled
with a small number of evaluations before ranking its inputs.
"""
from __future__ import annotations

import numpy as np


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def morris_trajectories(
    k: int, r: int, levels: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(design, orders)`` for ``r`` Morris trajectories.

    ``design`` has shape ``(r, k + 1, k)`` and ``orders`` has shape ``(r, k)``.
    All moves are positive and have size ``Delta = levels/(2*(levels-1))``.
    """
    k = _positive_int(k, "k")
    r = _positive_int(r, "r")
    levels = _positive_int(levels, "levels")
    if levels < 2:
        raise ValueError("levels must be at least 2")
    delta = levels / (2.0 * (levels - 1))
    grid_step = 1.0 / (levels - 1)
    max_base = 1.0 - delta
    base_values = np.arange(levels, dtype=float) * grid_step
    base_values = base_values[base_values <= max_base + 1e-12]
    if base_values.size == 0:
        raise ValueError("levels does not leave a valid base grid")

    rng = np.random.default_rng(seed)
    design = np.empty((r, k + 1, k), dtype=float)
    orders = np.empty((r, k), dtype=int)
    for trajectory in range(r):
        point = rng.choice(base_values, size=k)
        order = rng.permutation(k)
        design[trajectory, 0] = point
        orders[trajectory] = order
        for step, coordinate in enumerate(order, start=1):
            point = point.copy()
            point[coordinate] += delta
            design[trajectory, step] = point
    return design, orders


def _infer_orders(design: np.ndarray) -> np.ndarray:
    differences = np.diff(design, axis=1)
    orders = np.argmax(np.abs(differences), axis=2)
    changed = np.sum(np.abs(differences) > 1e-12, axis=2)
    if not np.all(changed == 1):
        raise ValueError("each trajectory step must change exactly one input")
    return orders.astype(int)


def elementary_effects(
    design: np.ndarray, orders_or_y: np.ndarray, y: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Compute Morris ``mu_star``, ``mu`` and ``sigma`` arrays.

    The preferred call is ``elementary_effects(design, orders, y)``. For
    compatibility, ``elementary_effects(design, y)`` infers the changed input
    from each design step.
    """
    design = np.asarray(design, dtype=float)
    if design.ndim != 3 or design.shape[1] < 2:
        raise ValueError("design must have shape (r, k+1, k)")
    r, points, k = design.shape
    if y is None:
        y = np.asarray(orders_or_y, dtype=float)
        orders = _infer_orders(design)
    else:
        orders = np.asarray(orders_or_y, dtype=int)
        y = np.asarray(y, dtype=float)
        if orders.shape != (r, k):
            raise ValueError("orders must have shape (r, k)")
    if y.shape != (r, points):
        raise ValueError("y must have shape (r, k+1)")
    if not np.all(np.isfinite(design)) or np.any(design < -1e-12) or np.any(design > 1 + 1e-12):
        raise ValueError("design values must lie in [0, 1]")
    delta = np.abs(design[:, 1:, :] - design[:, :-1, :])
    step = np.take_along_axis(delta, orders[:, :, None], axis=2)[:, :, 0]
    if not np.allclose(step, step[:, :1], rtol=1e-12, atol=1e-12):
        raise ValueError("trajectory steps must have a common size")
    delta_value = float(step[0, 0])
    if delta_value <= 0:
        raise ValueError("trajectory step must be positive")
    effects = np.empty((r, k), dtype=float)
    for trajectory in range(r):
        for step_index, coordinate in enumerate(orders[trajectory]):
            if coordinate < 0 or coordinate >= k:
                raise ValueError("orders contain an invalid input index")
            effects[trajectory, coordinate] = (
                y[trajectory, step_index + 1] - y[trajectory, step_index]
            ) / delta_value
    return {
        "mu_star": np.mean(np.abs(effects), axis=0),
        "mu": np.mean(effects, axis=0),
        "sigma": np.std(effects, axis=0, ddof=1) if r > 1 else np.zeros(k),
    }


def morris_screen(
    fn, k: int, r: int, levels: int = 4, seed: int = 0
) -> dict[str, np.ndarray]:
    """Evaluate ``fn`` on a Morris design and return effects plus the design."""
    design, orders = morris_trajectories(k, r, levels, seed)
    values = np.asarray([fn(row) for row in design.reshape(-1, design.shape[-1])], dtype=float)
    if values.shape != (design.shape[0] * design.shape[1],):
        raise ValueError("fn must return one scalar per input row")
    return elementary_effects(design, orders, values.reshape(design.shape[:2]))


def scale(
    unit: np.ndarray, lows: np.ndarray, highs: np.ndarray, log=False, *, log_scale=None
) -> np.ndarray:
    """Map unit-cube values to linear or logarithmic physical ranges."""
    unit = np.asarray(unit, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    if unit.ndim == 0 or lows.ndim != 1 or highs.shape != lows.shape or unit.shape[-1] != lows.size:
        raise ValueError("unit, lows and highs have incompatible shapes")
    if np.any(~np.isfinite(unit)) or np.any((unit < 0) | (unit > 1)):
        raise ValueError("unit values must lie in [0, 1]")
    if np.any(~np.isfinite(lows)) or np.any(~np.isfinite(highs)) or np.any(highs <= lows):
        raise ValueError("highs must be greater than lows")
    flags = log if log_scale is None else log_scale
    flags = np.broadcast_to(np.asarray(flags, dtype=bool), lows.shape)
    if np.any(flags & (lows <= 0)):
        raise ValueError("log-scaled ranges must be positive")
    linear = lows + unit * (highs - lows)
    logarithmic = np.exp(np.log(lows) + unit * (np.log(highs) - np.log(lows)))
    return np.where(flags, logarithmic, linear)


def one_at_a_time(fn, center, lows, highs, n_points):
    """Evaluate one physical input at a time and return ``(x_values, y_values)`` curves."""
    center = np.asarray(center, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    n_points = _positive_int(n_points, "n_points")
    if center.ndim != 1 or lows.shape != center.shape or highs.shape != center.shape:
        raise ValueError("center, lows and highs must be one-dimensional and aligned")
    if np.any(highs <= lows) or np.any(center < lows) or np.any(center > highs):
        raise ValueError("center must lie within each input range")
    curves = []
    for coordinate in range(center.size):
        values = np.linspace(lows[coordinate], highs[coordinate], n_points)
        outputs = np.empty(n_points, dtype=float)
        for index, value in enumerate(values):
            point = center.copy()
            point[coordinate] = value
            outputs[index] = float(fn(point))
        curves.append((values, outputs))
    return curves
