"""EXP-037 (proposal): can the recovery tracker learn its own measurement noise?

Three versions of the S6 recovery tracker run on the same delivered packets:

- told_5cm: told 0.05 m, the setting every registered S6 run uses.
- told_truth: told the true noise (an oracle; a real tracker cannot know it).
- adaptive: starts at 0.05 m and re-estimates the noise from its innovations
  (``sim.adaptive_r.AdaptiveRTracker``, window 20, floor 1 mm).

Grid: true noise 0.02, 0.05, 0.10 m x delay 0, 0.2, 0.3 s x (no loss, 30 %
loss in bursts of about 5 packets) x (straight line, 1 m turn), 50 seeds, the
same seeds as the S6 recovery sweep. The pass/fail rule below was written
before the full run and is evaluated by this script, not by hand.

Everything is simulated: synthetic link, synthetic Gaussian noise, no hardware.
Run it with:

    PYTHONPATH=. ./.venv/bin/python tools/run_s6_adaptive_r.py --output sim/runs/s6-adaptive-r
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

from sim.adaptive_r import DEFAULT_FLOOR_STD_M, DEFAULT_WINDOW, simulate_condition
from sim.s6_recovery import RecoveryConfig, median_interval, seed_list, wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
HASHED_SOURCES = ("sim/adaptive_r.py", "sim/tracker.py", "sim/s6_recovery.py", "sim/link.py", "sim/sensor.py",
                  "tools/run_s6_adaptive_r.py")
MOTIONS = ("constant_velocity", "constant_turn")
NOISES = (0.02, 0.05, 0.10)
DELAYS = (0.0, 0.2, 0.3)
LOSSES = (("none", 0.0, "independent"), ("burst30", 0.30, "burst"))
ARMS = ("told_5cm", "told_truth", "adaptive")
TOLD_STD_M = 0.05
GAP_MIN_M = 0.01      # H1 only scores conditions where telling the truth helps by at least 1 cm
H1_MIN_FRACTION = 0.5
H2_MAX_COST_M = 0.01

RULES = {
    "H1_recovers": (
        "Straight line, true noise 0.10 m, no loss, every delay where median p95(told_5cm) - median p95(told_truth) "
        f">= {GAP_MIN_M} m: the adaptive arm closes at least {H1_MIN_FRACTION:.0%} of that gap, "
        "and at 0.3 s its cell label equals the told_truth label."),
    "H2_no_harm": (
        f"True noise 0.05 m (told_5cm is already right), straight line, every delay and loss: the median paired "
        f"per-seed p95 difference adaptive - told_5cm is at most {H2_MAX_COST_M} m, and no cell goes from passes "
        "(told_5cm) to fails (adaptive)."),
    "H3_no_turn_flip": (
        "1 m turn, every noise, delay and loss: no cell goes from passes (told_5cm) to fails (adaptive)."),
    "verdict": (
        "worth proposing if H1, H2 and H3 pass; worth proposing only with a turn guard if H1 and H2 pass and H3 "
        "fails; not worth proposing as is if H1 or H2 fails."),
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


def _arms_for(noise: float) -> dict[str, tuple[str, float]]:
    return {"told_5cm": ("told", TOLD_STD_M), "told_truth": ("told", noise), "adaptive": ("adaptive", TOLD_STD_M)}


def _run_condition(condition: dict[str, Any], seeds: list[int], config: RecoveryConfig) -> dict[str, list]:
    rows: dict[str, list] = {arm: [] for arm in ARMS}
    for seed in seeds:
        result = simulate_condition(condition["motion"], delay_s=condition["delay_s"],
                                    loss_probability=condition["loss_probability"],
                                    noise_std_m=condition["noise_std_m"], loss_model=condition["loss_model"],
                                    seed=seed, config=config, arms=_arms_for(condition["noise_std_m"]))
        for arm in ARMS:
            r = result[arm]
            rows[arm].append({
                "p95_error_m": r.arm.p95_error_m, "max_error_m": r.arm.max_error_m, "rmse_m": r.arm.rmse_m,
                "mean_nis": r.arm.mean_nis, "fused_count": r.arm.fused_count, "dropped_count": r.arm.dropped_count,
                "failed": r.arm.failed, "final_std_m": r.final_std_m, "mean_scored_std_m": r.mean_scored_std_m,
            })
    return rows


def _arm_summary(rows: list[dict]) -> dict[str, Any]:
    p95 = [r["p95_error_m"] for r in rows]
    median, low, high = median_interval(p95)
    fails = sum(r["failed"] for r in rows)
    fail_low, fail_high = wilson_interval(fails, len(rows))
    out = {"p95_median_m": median, "p95_ci_low_m": low, "p95_ci_high_m": high, "fail_count": fails,
           "n_seeds": len(rows), "fail_rate_ci": [fail_low, fail_high],
           "label": "fails" if 2 * fails >= len(rows) else "passes"}
    scored = [r["mean_scored_std_m"] for r in rows if r["mean_scored_std_m"] is not None]
    if scored:
        out["estimated_std_median_m"] = float(np.median(scored))
    return out


def _paired(rows_a: list[dict], rows_b: list[dict]) -> dict[str, float]:
    diff = [a["p95_error_m"] - b["p95_error_m"] for a, b in zip(rows_a, rows_b)]
    median, low, high = median_interval(diff)
    return {"median_m": median, "ci_low_m": low, "ci_high_m": high}


def summarize_condition(condition: dict[str, Any], rows: dict[str, list]) -> dict[str, Any]:
    return {**condition,
            "arms": {arm: _arm_summary(rows[arm]) for arm in ARMS},
            "paired_adaptive_minus_told_5cm": _paired(rows["adaptive"], rows["told_5cm"]),
            "paired_adaptive_minus_told_truth": _paired(rows["adaptive"], rows["told_truth"])}


def evaluate_rules(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule to the condition summaries."""
    def med(cell, arm):
        return cell["arms"][arm]["p95_median_m"]

    h1_scored, h1_label_ok = [], None
    for cell in cells:
        if cell["motion"] != "constant_velocity" or cell["noise_std_m"] != 0.10 or cell["loss"] != "none":
            continue
        gap = med(cell, "told_5cm") - med(cell, "told_truth")
        if gap >= GAP_MIN_M:
            fraction = (med(cell, "told_5cm") - med(cell, "adaptive")) / gap
            h1_scored.append({"condition_id": cell["condition_id"], "gap_m": gap, "closed_fraction": fraction})
        if cell["delay_s"] == 0.3:
            h1_label_ok = cell["arms"]["adaptive"]["label"] == cell["arms"]["told_truth"]["label"]
    h1 = bool(h1_scored) and all(s["closed_fraction"] >= H1_MIN_FRACTION for s in h1_scored) and bool(h1_label_ok)

    h2_rows, h2 = [], True
    for cell in cells:
        if cell["motion"] != "constant_velocity" or cell["noise_std_m"] != 0.05:
            continue
        cost = cell["paired_adaptive_minus_told_5cm"]["median_m"]
        flip = cell["arms"]["told_5cm"]["label"] == "passes" and cell["arms"]["adaptive"]["label"] == "fails"
        h2_rows.append({"condition_id": cell["condition_id"], "median_cost_m": cost, "flip": flip})
        h2 = h2 and cost <= H2_MAX_COST_M and not flip

    flips = [cell["condition_id"] for cell in cells if cell["motion"] == "constant_turn"
             and cell["arms"]["told_5cm"]["label"] == "passes" and cell["arms"]["adaptive"]["label"] == "fails"]
    h3 = not flips
    if h1 and h2 and h3:
        verdict = "worth proposing"
    elif h1 and h2:
        verdict = "worth proposing only with a turn guard"
    else:
        verdict = "not worth proposing as is"
    return {"H1_recovers": {"pass": h1, "scored": h1_scored, "label_at_0.3s_matches_truth": h1_label_ok},
            "H2_no_harm": {"pass": h2, "rows": h2_rows},
            "H3_no_turn_flip": {"pass": h3, "flips": flips},
            "verdict": verdict}


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
    output.mkdir(parents=True, exist_ok=True)
    if seeds_count < 1 or processes < 1:
        raise ValueError("--seeds and --processes must be positive")
    conditions = build_conditions()
    seeds = seed_list(seeds_count)
    config = RecoveryConfig()
    code_sha256 = {relative: _sha256(REPO_ROOT / relative) for relative in HASHED_SOURCES}
    git_revision = _git_value(["rev-parse", "HEAD"])
    porcelain = _git_value(["status", "--porcelain"])
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=processes) as pool:
        results = list(pool.map(_run_condition, conditions, [seeds] * len(conditions), [config] * len(conditions)))
    runtime = time.perf_counter() - start

    cells = [summarize_condition(c, r) for c, r in zip(conditions, results)]
    summary = {
        "schema": "zephyr-s6-adaptive-r-1",
        "status": "EXP-037 proposal: synthetic link and sensor, not flight performance",
        "generator": "tools/run_s6_adaptive_r.py",
        "config": asdict(config),
        "design": {"n_seeds": seeds_count, "seeds": seeds, "arms": list(ARMS), "told_std_m": TOLD_STD_M,
                   "adaptive_window": DEFAULT_WINDOW, "adaptive_floor_std_m": DEFAULT_FLOOR_STD_M,
                   "statistic": "per-condition median of per-seed p95 error (recovery arm only), order-statistic "
                                "95% interval, Wilson interval for failure rates, paired per-seed differences",
                   "rules": RULES},
        "code_sha256": code_sha256,
        "cells": cells,
        "rules_result": evaluate_rules(cells) if seeds_count >= 2 else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["condition_id", "motion", "noise_std_m", "delay_s", "loss", "arm", "seed_index", "seed",
              "p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count", "failed",
              "final_std_m", "mean_scored_std_m"]
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, rows in zip(conditions, results):
            for arm in ARMS:
                for index, (seed, row) in enumerate(zip(seeds, rows[arm])):
                    record = {key: condition[key] for key in ("condition_id", "motion", "noise_std_m", "delay_s",
                                                              "loss")}
                    record.update({"arm": arm, "seed_index": index, "seed": seed, **row})
                    writer.writerow({key: _csv_value(record[key]) for key in fields})
    manifest = {
        "schema": "zephyr-s6-adaptive-r-manifest-1",
        "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_adaptive_r.py " + " ".join(argv or []),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "git_revision": git_revision,
        "git_tree_state": "unknown" if porcelain is None else ("dirty" if porcelain else "clean"),
        "runtime_s": runtime, "n_conditions": len(conditions), "n_seeds": seeds_count, "n_processes": processes,
        "outputs": {name: _sha256(output / name) for name in ("summary.json", "per_seed.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"conditions: {len(conditions)}  seeds: {seeds_count}  runtime_s: {runtime:.1f}", file=sys.stderr)
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
