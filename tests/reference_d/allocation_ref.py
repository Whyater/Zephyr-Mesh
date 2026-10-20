"""Independent brute force assignment reference for simulated allocation."""
import itertools
import numpy as np


def brute_force_min_cost(cost):
    matrix = np.asarray(cost, dtype=float)
    rows, cols = matrix.shape
    if rows > cols:
        raise ValueError("requires rows <= columns")
    best_total = float("inf")
    best = None
    for assignment in itertools.permutations(range(cols), rows):
        total = float(sum(matrix[i, j] for i, j in enumerate(assignment)))
        if total < best_total:
            best_total, best = total, assignment
    return best_total, tuple(best)
