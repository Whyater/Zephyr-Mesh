"""Simulated cooperative estimate fusion utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

Estimate = tuple[np.ndarray, np.ndarray]


def _validate(means: Sequence[np.ndarray], covs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    try:
        count = len(means)
        cov_count = len(covs)
    except TypeError as exc:
        raise ValueError("means and covs must be sequences") from exc
    if count == 0 or count != cov_count:
        raise ValueError("means and covs must be non-empty and have equal length")
    xs: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    d = None
    for mean, cov in zip(means, covs):
        x = np.asarray(mean, dtype=float)
        p = np.asarray(cov, dtype=float)
        if x.ndim != 1 or (d is not None and x.shape != (d,)):
            raise ValueError("means must be vectors with matching shapes")
        if d is None:
            d = x.size
            if d == 0:
                raise ValueError("means must be non-empty vectors")
        if p.shape != (d, d):
            raise ValueError("covariances must be square and match means")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
            raise ValueError("inputs must be finite")
        if not np.allclose(p, p.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariances must be symmetric positive definite")
        try:
            np.linalg.cholesky(p)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariances must be symmetric positive definite") from exc
        xs.append(x)
        ps.append((p + p.T) * 0.5)
    return xs, ps, int(d)


def information_fusion(means: Sequence[np.ndarray], covs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fuse independent Gaussian estimates in information form."""
    xs, ps, _ = _validate(means, covs)
    info = sum((np.linalg.solve(p, np.eye(p.shape[0])) for p in ps), np.zeros_like(ps[0]))
    weighted = sum((np.linalg.solve(p, x) for p, x in zip(ps, xs)), np.zeros_like(xs[0]))
    cov = np.linalg.solve(info, np.eye(info.shape[0]))
    cov = (cov + cov.T) * 0.5
    return cov @ weighted, cov


def _ci_at(xa: np.ndarray, pa: np.ndarray, xb: np.ndarray, pb: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    ia = np.linalg.solve(pa, np.eye(pa.shape[0]))
    ib = np.linalg.solve(pb, np.eye(pb.shape[0]))
    info = w * ia + (1.0 - w) * ib
    cov = np.linalg.solve(info, np.eye(info.shape[0]))
    cov = (cov + cov.T) * 0.5
    mean = cov @ (w * (ia @ xa) + (1.0 - w) * (ib @ xb))
    return mean, cov


def covariance_intersection(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    criterion: str = "trace",
    grid: int = 101,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Blend two guesses without pretending they are independent."""
    xa, pa, _ = _validate([mean_a], [cov_a])
    xb, pb, _ = _validate([mean_b], [cov_b])
    if xa[0].shape != xb[0].shape:
        raise ValueError("means must have matching shapes")
    if criterion not in {"trace", "det"} or isinstance(grid, (bool, np.bool_)) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("criterion must be 'trace' or 'det', and grid must be at least 2")
    a, b = xa[0], xb[0]
    pa0, pb0 = pa[0], pb[0]

    def objective(w: float) -> float:
        _, p = _ci_at(a, pa0, b, pb0, float(w))
        return float(np.trace(p) if criterion == "trace" else np.linalg.slogdet(p)[1])

    grid_values = np.linspace(0.0, 1.0, int(grid))
    values = np.array([objective(w) for w in grid_values])
    idx = int(np.argmin(values))
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-14})
    candidates = [(float(grid_values[idx]), float(values[idx]))]
    if result.success and np.isfinite(result.fun):
        candidates.append((float(result.x), float(result.fun)))
    candidates.extend([(0.0, objective(0.0)), (1.0, objective(1.0))])
    w = min(candidates, key=lambda item: (item[1], item[0]))[0]
    mean, cov = _ci_at(a, pa0, b, pb0, w)
    return mean, cov, float(w)


def covariance_intersection_many(
    means: Sequence[np.ndarray], covs: Sequence[np.ndarray], *, criterion: str = "trace"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N-way covariance intersection with simplex-constrained weights."""
    xs, ps, d = _validate(means, covs)
    if criterion not in {"trace", "det"}:
        raise ValueError("criterion must be 'trace' or 'det'")
    n = len(xs)
    infos = [np.linalg.solve(p, np.eye(d)) for p in ps]

    def evaluate(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        info = sum((float(w) * i for w, i in zip(weights, infos)), np.zeros((d, d)))
        p = np.linalg.solve(info, np.eye(d))
        p = (p + p.T) * 0.5
        x = p @ sum((float(w) * (i @ m) for w, i, m in zip(weights, infos, xs)), np.zeros(d))
        return x, p

    def objective(weights: np.ndarray) -> float:
        _, p = evaluate(weights)
        return float(np.trace(p) if criterion == "trace" else np.linalg.slogdet(p)[1])

    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    result = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, 1.0)] * n,
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("covariance intersection optimization failed")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    weights /= np.sum(weights)
    mean, cov = evaluate(weights)
    return mean, cov, weights


def covariance_intersection_fast(
    means: Sequence[np.ndarray], covs: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N-way covariance intersection with closed-form weights proportional to 1/trace(P_i).

    Any convex weights give a consistent CI result, so this keeps the
    no-double-counting guarantee while skipping the trace optimization
    (fast covariance intersection, Niehsen 2002). The fused covariance is
    never smaller than the optimal CI result, so it is at least as conservative.
    """
    xs, ps, d = _validate(means, covs)
    inv_traces = np.asarray([1.0 / float(np.trace(p)) for p in ps])
    weights = inv_traces / float(np.sum(inv_traces))
    infos = [np.linalg.solve(p, np.eye(d)) for p in ps]
    info = sum((float(w) * i for w, i in zip(weights, infos)), np.zeros((d, d)))
    cov = np.linalg.solve(info, np.eye(d))
    cov = (cov + cov.T) * 0.5
    mean = cov @ sum((float(w) * (i @ m) for w, i, m in zip(weights, infos, xs)), np.zeros(d))
    return mean, cov, weights


def naive_average(means: Sequence[np.ndarray], covs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average estimates and claim the covariance for independent errors."""
    xs, ps, _ = _validate(means, covs)
    n = len(xs)
    return np.mean(xs, axis=0), sum(ps, np.zeros_like(ps[0])) / (n * n)


def nees(error: np.ndarray, cov: np.ndarray) -> float:
    """Return normalized estimation error squared for a simulated estimate."""
    e = np.asarray(error, dtype=float)
    p = np.asarray(cov, dtype=float)
    if e.ndim != 1 or e.size == 0 or p.shape != (e.size, e.size) or not np.all(np.isfinite(e)) or not np.all(np.isfinite(p)):
        raise ValueError("invalid error or covariance shape/value")
    if not np.allclose(p, p.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric positive definite")
    try:
        np.linalg.cholesky(p)
        solved = np.linalg.solve(p, e)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    return float(e @ solved)
