"""EXP-040 (proposal): does choosing the motion noise q recover delay and noise tolerance?

The S6 recovery tracker runs with seven values of its process noise q
(0.01 to 10 m^2/s^3 in half-decade steps, including the registered 0.1), told
either 5 cm (the registered setting) or the true camera noise, on the same
delivered packets.

Grid: true noise 0.02, 0.05, 0.10 m x delay 0, 0.2, 0.3 s x (no loss, 30 %
loss in bursts of about 5 packets) x (straight line, 1 m turn), 50 seeds, the
same seeds as the S6 recovery sweep and EXP-037. The pass/fail rule below was
written before the full run and is evaluated by this script, not by hand.
At q = 0.1 the script checks every seed against the stored S6 recovery and
EXP-037 results (gate R0); if that fails, the run is void.

Everything is simulated: synthetic link, synthetic Gaussian noise, no hardware.
Run it with:

    PYTHONPATH=. ./.venv/bin/python tools/run_s6_q_sweep.py --output sim/runs/s6-q-sweep
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.q_sweep import Q_GRID, REGISTERED_Q, simulate_q_condition
from sim.s6_recovery import RecoveryConfig, median_interval, seed_list, wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
HASHED_SOURCES = ("sim/q_sweep.py", "sim/adaptive_r.py", "sim/tracker.py", "sim/s6_recovery.py", "sim/link.py",
                  "sim/sensor.py", "tools/run_s6_q_sweep.py")
REGISTERED_S6 = REPO_ROOT / "sim" / "runs" / "s6-recovery" / "per_seed.csv"
EXP037 = REPO_ROOT / "sim" / "runs" / "s6-adaptive-r" / "per_seed.csv"
MOTIONS = ("constant_velocity", "constant_turn")
NOISES = (0.02, 0.05, 0.10)
DELAYS = (0.0, 0.2, 0.3)
LOSSES = (("none", 0.0, "independent"), ("burst30", 0.30, "burst"))
TOLD = ("told_5cm", "told_truth")
TOLD_STD_M = 0.05
H2_MAX_REGRET_M = 0.02
H3_MIN_GAIN_M = 0.02
H3_MIN_CELLS = 3

RULES = {
    "R0_reproduction": (
        "At q = 0.1, told_5cm equals the registered sim/runs/s6-recovery recovery arm per seed (p95 and max, exact) "
        "wherever that cell exists, told_truth at 0.10 m with no loss equals its oracle slice, and every q = 0.1 "
        "cell equals sim/runs/s6-adaptive-r told_5cm and told_truth. If R0 fails the run is void."),
    "H1_q_is_the_knob": (
        "Straight line, no loss, 0.3 s: (a) at true noise 0.02 m and 0.10 m, the best grid q told 5 cm has a "
        "median p95 at or below the q = 0.1 told_truth median p95; (b) at 0.10 m that best-q cell passes."),
    "H2_one_fixed_q": (
        f"Some single grid q, told 5 cm, (a) never turns a cell that passes at q = 0.1 into a fail (all 36 "
        f"conditions) and (b) in every no-loss condition (18) has a median p95 within {H2_MAX_REGRET_M} m of the "
        "best grid q for that condition."),
    "H3_schedule_worth_a_later_proposal": (
        "Oracle schedule: on straight conditions the grid q with the smallest worst-case regret over straight "
        "no-loss conditions, on turn conditions the same for turns (told 5 cm). Single q: smallest worst-case "
        "regret over all 18 no-loss conditions. Passes if the schedule beats the single q by at least "
        f"{H3_MIN_GAIN_M} m median p95 in at least {H3_MIN_CELLS} no-loss conditions, or turns at least one of "
        "the 36 conditions from fails to passes. Upper bound: the schedule knows the motion with no lag."),
    "regret": "median p95 at that q minus the smallest median p95 over the grid, same condition, told 5 cm",
    "ties": "added after the critic review, matching the code as first written: on a tie the smaller grid q wins",
    "cell_label": "a cell fails when at least half of its 50 seeds fail (p95 > 0.25 m or max > 0.50 m)",
}


def build_conditions() -> list[dict[str, Any]]:
    conditions = []
    for motion in MOTIONS:
        for noise in NOISES:
            for delay in DELAYS:
                for loss_name, loss, loss_model in LOSSES:
                    conditions.append({
                        "condition_id": f"{motion}/n{noise:.2f}/d{delay:.1f}/{loss_name}",
                        "motion": motion, "noise_std_m": noise, "delay_s": delay, "loss": loss_name,
                        "loss_probability": loss, "loss_model": loss_model,
                    })
    return conditions


def _told_stds(noise: float) -> dict[str, float]:
    # At 5 cm the two settings are the same run; it is computed once and reported under both names.
    if noise == TOLD_STD_M:
        return {"told_5cm": TOLD_STD_M}
    return {"told_5cm": TOLD_STD_M, "told_truth": noise}


def _run_condition(condition: dict[str, Any], seeds: list[int], config: RecoveryConfig) -> dict[str, list]:
    """Per-seed rows keyed by f"{repr(q)}|{told}"."""
    told = _told_stds(condition["noise_std_m"])
    rows: dict[str, list] = {f"{q!r}|{name}": [] for q in Q_GRID for name in TOLD}
    for seed in seeds:
        result = simulate_q_condition(condition["motion"], delay_s=condition["delay_s"],
                                      loss_probability=condition["loss_probability"],
                                      noise_std_m=condition["noise_std_m"], loss_model=condition["loss_model"],
                                      seed=seed, config=config, q_values=Q_GRID, told_stds=told)
        for q in Q_GRID:
            for name in TOLD:
                r = result[(q, name if name in told else "told_5cm")]
                rows[f"{q!r}|{name}"].append({
                    "p95_error_m": r.p95_error_m, "max_error_m": r.max_error_m, "rmse_m": r.rmse_m,
                    "mean_nis": r.mean_nis, "fused_count": r.fused_count, "dropped_count": r.dropped_count,
                    "failed": r.failed,
                })
    return rows


def _arm_summary(rows: list[dict]) -> dict[str, Any]:
    p95 = [r["p95_error_m"] for r in rows]
    median, low, high = median_interval(p95)
    fails = sum(r["failed"] for r in rows)
    fail_low, fail_high = wilson_interval(fails, len(rows))
    nis = [r["mean_nis"] for r in rows if r["mean_nis"] is not None]
    return {"p95_median_m": median, "p95_ci_low_m": low, "p95_ci_high_m": high, "fail_count": fails,
            "n_seeds": len(rows), "fail_rate_ci": [fail_low, fail_high],
            "label": "fails" if 2 * fails >= len(rows) else "passes",
            "mean_nis_median": float(np.median(nis)) if nis else None}


def summarize_condition(condition: dict[str, Any], rows: dict[str, list]) -> dict[str, Any]:
    by_q = {repr(q): {name: _arm_summary(rows[f"{q!r}|{name}"]) for name in TOLD} for q in Q_GRID}
    return {**condition, "by_q": by_q}


# ---- rule evaluation --------------------------------------------------------------------------------

def _med(cell: dict, q: float, told: str = "told_5cm") -> float:
    return cell["by_q"][repr(q)][told]["p95_median_m"]


def _label(cell: dict, q: float, told: str = "told_5cm") -> str:
    return cell["by_q"][repr(q)][told]["label"]


def best_q(cell: dict, told: str = "told_5cm") -> float:
    """Grid q with the smallest median p95 (the first in grid order on a tie)."""
    return min(Q_GRID, key=lambda q: _med(cell, q, told))


def _regret(cell: dict, q: float) -> float:
    return _med(cell, q) - _med(cell, best_q(cell))


def _minimax_q(cells: list[dict]) -> float:
    return min(Q_GRID, key=lambda q: max(_regret(c, q) for c in cells))


def _find(cells, motion, noise, delay, loss):
    for c in cells:
        if (c["motion"], c["noise_std_m"], c["delay_s"], c["loss"]) == (motion, noise, delay, loss):
            return c
    raise KeyError((motion, noise, delay, loss))


def schedule_comparison(cells: list[dict]) -> dict[str, Any]:
    no_loss = [c for c in cells if c["loss"] == "none"]
    straight = [c for c in no_loss if c["motion"] == "constant_velocity"]
    turn = [c for c in no_loss if c["motion"] == "constant_turn"]
    q_single = _minimax_q(no_loss)
    q_straight, q_turn = _minimax_q(straight), _minimax_q(turn)

    def scheduled(cell):
        return q_straight if cell["motion"] == "constant_velocity" else q_turn

    gains = [{"condition_id": c["condition_id"], "gain_m": _med(c, q_single) - _med(c, scheduled(c))}
             for c in no_loss]
    winning = [g for g in gains if g["gain_m"] >= H3_MIN_GAIN_M]
    flips = [c["condition_id"] for c in cells
             if _label(c, q_single) == "fails" and _label(c, scheduled(c)) == "passes"]
    return {"pass": len(winning) >= H3_MIN_CELLS or bool(flips), "q_single": q_single, "q_straight": q_straight,
            "q_turn": q_turn, "single_worst_regret_m": max(_regret(c, q_single) for c in no_loss),
            "gains": gains, "cells_gaining_at_least_min": len(winning), "flips_to_pass": flips}


def evaluate_rules(cells: list[dict]) -> dict[str, Any]:
    """Apply the pre-registered rule to the condition summaries."""
    h1_rows = []
    for noise in (0.02, 0.10):
        cell = _find(cells, "constant_velocity", noise, 0.3, "none")
        q = best_q(cell)
        h1_rows.append({"noise_std_m": noise, "best_q": q, "best_q_told_5cm_median_m": _med(cell, q),
                        "q0.1_told_truth_median_m": _med(cell, REGISTERED_Q, "told_truth"),
                        "best_q_label": _label(cell, q),
                        "beats_truth": _med(cell, q) <= _med(cell, REGISTERED_Q, "told_truth")})
    h1 = all(r["beats_truth"] for r in h1_rows) and h1_rows[1]["best_q_label"] == "passes"

    per_q = []
    for q in Q_GRID:
        flips = [c["condition_id"] for c in cells
                 if _label(c, REGISTERED_Q) == "passes" and _label(c, q) == "fails"]
        worst = max(_regret(c, q) for c in cells if c["loss"] == "none")
        per_q.append({"q": q, "flips_vs_q0.1": flips, "worst_no_loss_regret_m": worst,
                      "qualifies": not flips and worst <= H2_MAX_REGRET_M})
    qualifying = [r["q"] for r in per_q if r["qualifies"]]

    schedule = schedule_comparison(cells)
    return {"H1_q_is_the_knob": {"pass": h1, "rows": h1_rows},
            "H2_one_fixed_q": {"pass": bool(qualifying), "qualifying_q": qualifying, "per_q": per_q},
            "H3_schedule_worth_a_later_proposal": schedule}


def tolerance_table(cells: list[dict]) -> list[dict[str, Any]]:
    """Largest grid delay that still passes, per (motion, noise, loss, q, told); None if 0 s already fails."""
    out = []
    for motion in MOTIONS:
        for noise in NOISES:
            for loss_name, _, _ in LOSSES:
                group = sorted((c for c in cells if (c["motion"], c["noise_std_m"], c["loss"]) ==
                                (motion, noise, loss_name)), key=lambda c: c["delay_s"])
                for q in Q_GRID:
                    for told in TOLD:
                        tolerated = None
                        for c in group:
                            if _label(c, q, told) != "passes":
                                break
                            tolerated = c["delay_s"]
                        out.append({"motion": motion, "noise_std_m": noise, "loss": loss_name, "q": q,
                                    "told": told, "tolerated_delay_s": tolerated})
    return out


# ---- reproduction gate R0 ---------------------------------------------------------------------------

def _stored_s6(path: Path) -> dict[tuple, dict[int, tuple[float, float]]]:
    out: dict[tuple, dict[int, tuple[float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != "recovery" or float(row["jitter_s"]) != 0.0:
                continue
            key = (row["slice"], row["motion"], float(row["delay_s"]), float(row["loss_probability"]),
                   float(row["noise_std_m"]), row["loss_model"], float(row["tracker_measurement_std_m"]))
            out.setdefault(key, {})[int(row["seed_index"])] = (float(row["p95_error_m"]), float(row["max_error_m"]))
    return out


def _stored_exp037(path: Path) -> dict[tuple, dict[int, tuple[float, float]]]:
    out: dict[tuple, dict[int, tuple[float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in TOLD:
                out.setdefault((row["condition_id"], row["arm"]), {})[int(row["seed_index"])] = (
                    float(row["p95_error_m"]), float(row["max_error_m"]))
    return out


def reproduction_check(conditions: list[dict], results: list[dict[str, list]], *, s6_path: Path = REGISTERED_S6,
                       exp037_path: Path = EXP037) -> dict[str, Any]:
    """Gate R0. Every expected stored cell must exist and every one of our seeds must match it exactly.

    Registered S6 cells exist for: no loss told 0.05 m (main slice, every noise), 30 % bursts told 0.05 m at
    0.05 m true noise only (burst slice), and no loss told the true 0.10 m (oracle slice). EXP-037 has every
    condition for both told settings. A missing file, cell or seed fails the gate, as does any mismatch.
    """
    s6 = _stored_s6(s6_path) if s6_path.exists() else {}
    exp037 = _stored_exp037(exp037_path) if exp037_path.exists() else {}
    seeds_compared = {"s6_registered": 0, "exp037": 0}
    cells_compared = {"s6_registered": 0, "exp037": 0}
    mismatches, missing = [], []

    def compare(source, label, ours, stored):
        if stored is None:
            missing.append({"source": source, "cell": label, "seed_index": None})
            return
        cells_compared[source] += 1
        for index, row in enumerate(ours):
            if index not in stored:
                missing.append({"source": source, "cell": label, "seed_index": index})
                continue
            seeds_compared[source] += 1
            if (row["p95_error_m"], row["max_error_m"]) != stored[index]:
                mismatches.append({"source": source, "cell": label, "seed_index": index})

    for condition, rows in zip(conditions, results):
        noise, loss = condition["noise_std_m"], condition["loss_probability"]
        for told in TOLD:
            ours = rows[f"{REGISTERED_Q!r}|{told}"]
            told_std = TOLD_STD_M if told == "told_5cm" else noise
            slice_name = None
            if condition["loss"] == "none":
                model = "independent"
                if told_std == TOLD_STD_M:
                    slice_name = "main"
                elif noise == 0.10:
                    slice_name = "oracle"
            else:
                model = "burst"
                if told_std == TOLD_STD_M and noise == TOLD_STD_M:
                    slice_name = "burst"
            if slice_name is not None:
                key = (slice_name, condition["motion"], condition["delay_s"], loss, noise, model, told_std)
                compare("s6_registered", f"{slice_name}:{condition['condition_id']}:{told}", ours, s6.get(key))
            stored = exp037.get((condition["condition_id"], told))
            compare("exp037", f"{condition['condition_id']}:{told}", ours, stored)
    return {"pass": not mismatches and not missing and all(seeds_compared.values()),
            "cells_compared": cells_compared, "seeds_compared": seeds_compared,
            "mismatches": mismatches[:20], "mismatch_count": len(mismatches),
            "missing": missing[:20], "missing_count": len(missing)}


# ---- outputs ----------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def run(output: Path, *, seeds_count: int = 50, processes: int = 1, overwrite: bool = False,
        argv: list[str] | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}; pass --overwrite")
    if seeds_count < 1 or processes < 1:
        raise ValueError("--seeds and --processes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    seeds = seed_list(seeds_count)
    config = RecoveryConfig()
    code_sha256 = {relative: _sha256(REPO_ROOT / relative) for relative in HASHED_SOURCES}
    stored_inputs_sha256 = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in (REGISTERED_S6, EXP037)
                            if path.exists()}
    git_revision = _git_value(["rev-parse", "HEAD"])
    porcelain = _git_value(["status", "--porcelain"])
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=processes) as pool:
        results = list(pool.map(_run_condition, conditions, [seeds] * len(conditions), [config] * len(conditions)))
    runtime = time.perf_counter() - start

    cells = [summarize_condition(c, r) for c, r in zip(conditions, results)]
    summary = {
        "schema": "zephyr-s6-q-sweep-1",
        "status": "EXP-040 proposal: synthetic link and sensor, not flight performance, not registered",
        "generator": "tools/run_s6_q_sweep.py",
        "config": asdict(config),
        "design": {"n_seeds": seeds_count, "seeds": seeds, "q_grid": list(Q_GRID), "registered_q": REGISTERED_Q,
                   "told": list(TOLD), "told_std_m": TOLD_STD_M,
                   "statistic": "per-condition median of per-seed p95 error (recovery arm only), order-statistic "
                                "95% interval, Wilson interval for failure rates",
                   "rules": RULES},
        "code_sha256": code_sha256,
        "stored_inputs_sha256": stored_inputs_sha256,
        "reproduction_R0": reproduction_check(conditions, results),
        "cells": cells,
        "tolerance": tolerance_table(cells),
        "rules_result": evaluate_rules(cells),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["condition_id", "motion", "noise_std_m", "delay_s", "loss", "q", "told", "seed_index", "seed",
              "p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count", "failed"]
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, rows in zip(conditions, results):
            for q in Q_GRID:
                for told in TOLD:
                    for index, (seed, row) in enumerate(zip(seeds, rows[f"{q!r}|{told}"])):
                        record = {key: condition[key] for key in ("condition_id", "motion", "noise_std_m",
                                                                  "delay_s", "loss")}
                        record.update({"q": q, "told": told, "seed_index": index, "seed": seed, **row})
                        writer.writerow({key: _csv_value(record[key]) for key in fields})
    manifest = {
        "schema": "zephyr-s6-q-sweep-manifest-1",
        "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_q_sweep.py " + " ".join(argv or []),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "git_revision": git_revision,
        "git_tree_state": "unknown" if porcelain is None else ("dirty" if porcelain else "clean"),
        "runtime_s": runtime, "n_conditions": len(conditions), "n_seeds": seeds_count, "n_processes": processes,
        "outputs": {name: _sha256(output / name) for name in ("summary.json", "per_seed.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"conditions: {len(conditions)}  seeds: {seeds_count}  runtime_s: {runtime:.1f}  "
          f"R0: {summary['reproduction_R0']['pass']}", file=sys.stderr)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run(args.output, seeds_count=args.seeds, processes=args.processes, overwrite=args.overwrite,
        argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
