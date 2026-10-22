"""Explain the EXP-028 gap between the hand value and the measured steps to 1 % agreement.

Rebuilds the zero-loss, zero-delay cells of sim/runs/d8-consensus for the ring and the line with the
same graphs, eps, seeds and starting values as tools/d_run_consensus.py, then splits the gap between
ln(0.01)/ln(rho) and the measured median into parts. Read only: it never writes into sim/runs/.

    PYTHONPATH=. python tools/d_exp028_gap.py --output <dir>   (run from the repo root)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from sim.coop.consensus import (
    laplacian,
    max_degree,
    run_consensus,
    spectral_rate,
    steps_to_agreement,
)
from tools.d_common import seeds_for
from tools.d_run_consensus import GRAPH_NAMES, _graph

TOL = 0.01
STEPS = 400
N_SEEDS = 50
SLOW_MATCH = 1e-9


def _registered_steps(graph: str) -> list[int]:
    """Steps to agreement at p = 0, delay 0 from the registered CSV (both loss flags agree at p = 0)."""
    path = Path("sim/runs/d8-consensus/loss_delay.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["graph"] == graph and float(r["p"]) == 0.0 and int(r["delay_steps"]) == 0
                and r["symmetric_loss"] == "true"]
    return [int(r["steps_to_agreement"]) for r in rows]


def _modes(a: np.ndarray, eps: float):
    """Split the disagreement into patterns: shapes across the nodes plus a shrink factor per step.

    No function in sim/coop returns the shapes themselves, so this uses numpy's eigh on the same
    Laplacian. Drops the flat pattern (everyone equal), which is the average and never changes.
    """
    vals, vecs = np.linalg.eigh(laplacian(a))
    factors = 1.0 - eps * vals
    keep = np.arange(1, vals.size)
    return factors[keep], vecs[:, keep]


def _fractional_crossing(energy: np.ndarray, factors: np.ndarray) -> float:
    """Real-valued t where sqrt(sum E_k r_k^(2t)) / sqrt(sum E_k) = TOL, by bisection."""
    total = energy.sum()

    def ratio(t: float) -> float:
        return math.sqrt(float(np.sum(energy * np.abs(factors) ** (2.0 * t))) / total)

    lo, hi = 0.0, 1.0
    while ratio(hi) > TOL:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio(mid) > TOL:
            lo = mid
        else:
            hi = mid
    return hi


def _max_minus_min_steps(states: np.ndarray) -> int | None:
    """Alternative metric: largest per-axis max minus min, relative to its start."""
    spread = np.max(np.ptp(states, axis=1), axis=1)
    hits = np.flatnonzero(spread <= TOL * spread[0])
    return int(hits[0]) if hits.size else None


def analyse(graph: str) -> dict:
    a, _ = _graph(graph)
    eps = 1.0 / (max_degree(a) + 1.0)
    rho = spectral_rate(laplacian(a), eps)
    hand = math.log(TOL) / math.log(rho)
    factors, vecs = _modes(a, eps)
    slow = np.abs(np.abs(factors) - rho) < SLOW_MATCH
    gi = GRAPH_NAMES.index(graph)
    seeds = seeds_for("consensus", N_SEEDS)
    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, gi, 31]))
        initial = rng.uniform(-1, 1, size=(a.shape[0], 3))
        result = run_consensus(initial, a, eps, STEPS, seed=seed)
        measured = steps_to_agreement(result, TOL)
        coeffs = vecs.T @ (initial - initial.mean(axis=0))      # amount of each pattern, per axis
        energy = np.sum(coeffs ** 2, axis=1)                     # summed over the 3 axes
        share = float(energy[slow].sum() / energy.sum())
        frac_all = _fractional_crossing(energy, factors)
        frac_slow_only = (math.log(TOL) - 0.5 * math.log(share)) / math.log(rho)
        int_all = math.ceil(frac_all - 1e-12)
        per_seed.append({"seed": int(seed), "measured": measured, "predicted_integer_all_modes": int_all,
                         "fractional_all_modes": frac_all, "fractional_slowest_only": frac_slow_only,
                         "slowest_share": share, "max_minus_min_steps": _max_minus_min_steps(result.states)})
    registered = _registered_steps(graph)
    measured = [r["measured"] for r in per_seed]

    def med(key):
        return float(np.median([r[key] for r in per_seed]))

    # Start made only of the slowest pattern (first slow shape on every axis).
    pure = np.repeat(vecs[:, np.flatnonzero(slow)[0]][:, None], 3, axis=1)
    pure_steps = steps_to_agreement(run_consensus(pure, a, eps, STEPS), TOL)
    n_patterns = factors.size
    expected_share = int(slow.sum()) / n_patterns
    share_shift = 0.5 * math.log(med("slowest_share")) / abs(math.log(rho))
    return {
        "graph": graph, "n_nodes": int(a.shape[0]), "eps": eps, "rho": rho,
        "pattern_factors": factors.tolist(), "n_slowest_patterns": int(slow.sum()),
        "hand_fractional": hand, "hand_ceiling_registered": math.ceil(hand),
        "hand_with_rounded_rho_0_873": math.log(TOL) / math.log(0.873) if graph == "line" else None,
        "measured_median": float(np.median(measured)), "registered_median": float(np.median(registered)),
        "matches_registered_csv": sorted(measured) == sorted(registered),
        "integer_all_modes_equals_measured_every_seed": all(
            r["predicted_integer_all_modes"] == r["measured"] for r in per_seed),
        "slowest_share_median": med("slowest_share"),
        "slowest_share_min": float(min(r["slowest_share"] for r in per_seed)),
        "slowest_share_max": float(max(r["slowest_share"] for r in per_seed)),
        "slowest_share_expected_random_start": expected_share,
        "fractional_slowest_only_median": med("fractional_slowest_only"),
        "fractional_all_modes_median": med("fractional_all_modes"),
        "max_minus_min_steps_median": med("max_minus_min_steps"),
        "pure_slowest_start_steps": pure_steps,
        "decomposition_steps": {
            "a_share_shift_at_median_share": share_shift,
            "a_share_shift_at_expected_share": 0.5 * math.log(expected_share) / abs(math.log(rho)),
            "a_median_of_slowest_only_minus_hand": med("fractional_slowest_only") - hand,
            "d_fast_patterns_median": float(np.median(
                [r["fractional_all_modes"] - r["fractional_slowest_only"] for r in per_seed])),
            "c_ceiling_median": float(np.median(
                [r["measured"] - r["fractional_all_modes"] for r in per_seed])),
            "b_metric_max_minus_min_minus_measured_median": float(np.median(
                [r["max_minus_min_steps"] - r["measured"] for r in per_seed])),
        },
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    out_dir = parser.parse_args().output
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"tolerance": TOL, "steps": STEPS, "n_seeds": N_SEEDS,
              "graphs": [analyse("line"), analyse("ring")]}
    (out_dir / "gap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for g in report["graphs"]:
        short = {k: v for k, v in g.items() if k not in ("per_seed", "pattern_factors")}
        print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
