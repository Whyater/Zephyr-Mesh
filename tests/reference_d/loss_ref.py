"""Independent update-loss distribution references."""
import numpy as np


def all_lost_prob(p, n):
    return p**n


def gap_pmf(k, p, n):
    q = p**n
    return (1.0 - q) * q**(k - 1) if k >= 1 else 0.0


def mean_gap(p, n):
    return 1.0 / (1.0 - p**n)


def survival(k, p, n):
    return (p**n)**(k - 1) if k >= 1 else 1.0


def simulate_gaps(p, n, steps, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    lost = rng.random((n, steps)) < p
    arrivals = ~np.all(lost, axis=0)
    gaps = []
    since = 0
    for arrived in arrivals:
        since += 1
        if arrived:
            gaps.append(since)
            since = 0
    return gaps
