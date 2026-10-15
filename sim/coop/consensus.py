"""Average consensus for a simulated cooperative observer network."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _graph_array(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(a)) or not np.allclose(a, a.T):
        raise ValueError("adjacency must be finite and symmetric")
    if np.any(a < 0.0):
        raise ValueError("adjacency weights must be nonnegative")
    if not np.allclose(np.diag(a), 0.0):
        raise ValueError("adjacency diagonal must be zero")
    return a


def laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the combinatorial Laplacian of a symmetric adjacency matrix."""
    a = _graph_array(adjacency)
    return np.diag(np.sum(a, axis=1)) - a


def eigenvalues(L: np.ndarray) -> np.ndarray:
    """Return sorted Laplacian eigenvalues."""
    matrix = np.asarray(L, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("L must be square")
    return np.linalg.eigvalsh(matrix)


def algebraic_connectivity(L: np.ndarray) -> float:
    """Return the second smallest Laplacian eigenvalue, or zero for one node."""
    vals = eigenvalues(L)
    return float(vals[1]) if vals.size > 1 else 0.0


def spectral_rate(L: np.ndarray, eps: float) -> float:
    """Return the slowest disagreement-mode factor for one update."""
    if not np.isfinite(eps):
        raise ValueError("eps must be finite")
    vals = eigenvalues(L)
    if vals.size <= 1:
        return 0.0
    return float(max(abs(1.0 - eps * vals[1]), abs(1.0 - eps * vals[-1])))


def max_degree(adjacency: np.ndarray) -> float:
    """Return the largest weighted degree."""
    return float(np.max(np.sum(_graph_array(adjacency), axis=1), initial=0.0))


def ring(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    a = np.zeros((n, n), dtype=float)
    if n == 2:
        a[0, 1] = a[1, 0] = 1.0
    elif n > 2:
        for i in range(n):
            a[i, (i - 1) % n] = a[i, (i + 1) % n] = 1.0
    return a


def line(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    a = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def star(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    a = np.zeros((n, n), dtype=float)
    if n > 1:
        a[0, 1:] = 1.0
        a[1:, 0] = 1.0
    return a


def complete(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    return np.ones((n, n), dtype=float) - np.eye(n, dtype=float)


def random_geometric(n: int, radius_m: float, *, seed: int, box_m: float = 4.0):
    if n < 1 or radius_m < 0 or box_m <= 0:
        raise ValueError("invalid geometric graph parameters")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    positions = np.zeros((n, 3), dtype=float)
    positions[:, :2] = rng.uniform(0.0, box_m, size=(n, 2))
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta[:, :, :2], axis=2)
    adjacency = (distances <= radius_m).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency, positions


@dataclass(frozen=True)
class ConsensusResult:
    states: np.ndarray
    disagreement: np.ndarray
    average: np.ndarray
    messages_sent: int
    messages_lost: int


def run_consensus(
    initial: np.ndarray,
    adjacency: np.ndarray,
    eps: float,
    steps: int,
    *,
    loss_probability: float = 0.0,
    delay_steps: int = 0,
    symmetric_loss: bool = False,
    seed: int = 0,
) -> ConsensusResult:
    """Run delayed, lossy average consensus in a fully simulated network.

    A neighbor contributes to an agent's update only when a message from that
    neighbor arrives at that step. There is no stale-value reuse. With a
    positive delay, an arrival at step ``k`` is the message sent at
    ``k - delay_steps`` that was not lost. Symmetric loss drops both directed
    messages on an edge together; directed loss draws independently.
    """
    a = _graph_array(adjacency)
    x0 = np.asarray(initial, dtype=float)
    if x0.ndim == 1:
        x0 = x0[:, None]
    if x0.ndim != 2 or x0.shape[0] != a.shape[0] or not np.all(np.isfinite(x0)):
        raise ValueError("initial must have shape (n, d)")
    if steps < 0 or delay_steps < 0 or not np.isfinite(eps):
        raise ValueError("invalid steps, delay, or eps")
    if not 0.0 <= loss_probability <= 1.0:
        raise ValueError("loss_probability must be in [0, 1]")
    n, d = x0.shape
    neighbors = [np.flatnonzero(a[i] != 0.0).tolist() for i in range(n)]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    arrivals: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    states = np.empty((steps + 1, n, d), dtype=float)
    states[0] = x0
    sent = lost = 0
    for k in range(steps):
        for sender in range(n):
            for receiver in range(n):
                if a[sender, receiver] == 0.0:
                    continue
                sent += 1
                if symmetric_loss and sender < receiver:
                    drop = bool(rng.random() < loss_probability)
                    if drop:
                        lost += 2
                    if not drop:
                        arrivals.setdefault(k + delay_steps, []).append(
                            (sender, receiver, states[k, sender].copy())
                        )
                        arrivals.setdefault(k + delay_steps, []).append(
                            (receiver, sender, states[k, receiver].copy())
                        )
                elif symmetric_loss and sender > receiver:
                    pass
                else:
                    if rng.random() < loss_probability:
                        lost += 1
                    else:
                        arrivals.setdefault(k + delay_steps, []).append(
                            (sender, receiver, states[k, sender].copy())
                        )
        fresh: dict[tuple[int, int], np.ndarray] = {
            (receiver, sender): value for sender, receiver, value in arrivals.pop(k, [])
        }
        current = states[k]
        updated = current.copy()
        for i in range(n):
            total = np.zeros(d, dtype=float)
            for j in neighbors[i]:
                value = fresh.get((i, j))
                if value is not None:
                    total += a[i, j] * (value - current[i])
            updated[i] = current[i] + eps * total
        states[k + 1] = updated
    averages = np.mean(states, axis=1)
    disagreements = np.linalg.norm(states - averages[:, None, :], axis=(1, 2))
    return ConsensusResult(states, disagreements, averages, sent, lost)


def measured_rate(disagreement: np.ndarray, start: int, stop: int) -> float:
    values = np.asarray(disagreement, dtype=float)
    if stop <= start or start < 0 or stop >= values.size or values[start] <= 0:
        raise ValueError("invalid measurement interval")
    return float((values[stop] / values[start]) ** (1.0 / (stop - start)))


def steps_to_agreement(result: ConsensusResult, tolerance: float) -> Optional[int]:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    threshold = tolerance * float(result.disagreement[0])
    hits = np.flatnonzero(result.disagreement <= threshold)
    return int(hits[0]) if hits.size else None
