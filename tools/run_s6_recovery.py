"""Run the S6 recovery sweep over synthetic link and sensor conditions.

The sweep measures simulated tracker recovery under a grid of delays, losses,
noise, burst loss, and jitter. Results are synthetic simulated-link and
simulated-sensor results, not flight performance. Run it with:

    PYTHONPATH=. ./.venv/bin/python tools/run_s6_recovery.py --output runs/s6-recovery
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from sim.s6_recovery import RecoveryConfig, paired_summary, seed_list, simulate_run, summarize_arm

REPO_ROOT = Path(__file__).resolve().parents[1]
HASHED_SOURCES = ("sim/tracker.py", "sim/s6_recovery.py", "sim/link.py", "sim/sensor.py", "tools/run_s6_recovery.py")


MAIN_DELAYS = (0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
MAIN_LOSSES = (0, 0.10, 0.30, 0.50)
MAIN_NOISES = (0, 0.02, 0.05, 0.10)
PILOT_DELAYS = (0, 0.1, 0.3)
PILOT_LOSSES = (0, 0.3)
PILOT_NOISES = (0, 0.05)


def build_cells(pilot: bool = False) -> list[dict[str, Any]]:
    """Build the pre-registered grid in its deterministic order."""
    motions = ("constant_velocity", "constant_turn")
    cells: list[dict[str, Any]] = []
    if pilot:
        slices = (("main", PILOT_DELAYS, PILOT_LOSSES, PILOT_NOISES, "independent", 0.0, 0.05),)
    else:
        slices = (
            ("main", MAIN_DELAYS, MAIN_LOSSES, MAIN_NOISES, "independent", 0.0, 0.05),
            ("burst", MAIN_DELAYS, (0.10, 0.30, 0.50), (0.05,), "burst", 0.0, 0.05),
            ("jitter", (0.025, 0.05, 0.10, 0.15, 0.20, 0.30), (0, 0.30), (0.05,), "independent", None, 0.05),
            ("oracle", MAIN_DELAYS, (0, 0.30), (0.10,), "independent", 0.0, 0.10),
        )
    for slice_name, delays, losses, noises, loss_model, fixed_jitter, tracker_std in slices:
        for motion in motions:
            for delay in delays:
                for loss in losses:
                    for noise in noises:
                        jitter = delay / 2 if fixed_jitter is None else fixed_jitter
                        cell = {
                            "slice": slice_name,
                            "motion": motion,
                            "delay_s": float(delay),
                            "loss_probability": float(loss),
                            "noise_std_m": float(noise),
                            "loss_model": loss_model,
                            "jitter_s": float(jitter),
                            "tracker_measurement_std_m": float(tracker_std),
                        }
                        cell["cell_id"] = (
                            f"{slice_name}/{motion}/d{delay:.3f}/l{loss:.2f}/n{noise:.3f}/"
                            f"{loss_model}/j{jitter:.3f}/r{tracker_std:.3f}"
                        )
                        cells.append(cell)
    return cells


def _run_cell(cell: dict[str, Any], seeds: list[int], config: RecoveryConfig) -> dict[str, Any]:
    naive = []
    recovery = []
    for seed in seeds:
        arms = simulate_run(
            cell["motion"],
            delay_s=cell["delay_s"],
            loss_probability=cell["loss_probability"],
            noise_std_m=cell["noise_std_m"],
            seed=seed,
            config=config,
            loss_model=cell["loss_model"],
            jitter_s=cell["jitter_s"],
            tracker_measurement_std_m=cell["tracker_measurement_std_m"],
        )
        # The per-query error arrays are not needed for the summaries; drop
        # them before results cross the process boundary.
        naive.append(replace(arms["naive"], query_times_s=None, errors_m=None))
        recovery.append(replace(arms["recovery"], query_times_s=None, errors_m=None))
    return {"naive": naive, "recovery": recovery}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    """Return git's output, "" for empty output, or None when git cannot answer."""
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cell_summary(cell: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        **cell,
        "arms": {"naive": summarize_arm(result["naive"]), "recovery": summarize_arm(result["recovery"])},
        "paired": paired_summary(result["naive"], result["recovery"]),
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _write_csv(path: Path, cells: list[dict[str, Any]], results: list[dict[str, Any]], seeds: list[int]) -> None:
    fields = ["cell_id", "slice", "motion", "delay_s", "loss_probability", "noise_std_m", "loss_model",
              "jitter_s", "tracker_measurement_std_m", "arm", "seed_index", "seed", "p95_error_m",
              "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count",
              "fused_out_of_order_count", "failed"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell, result in zip(cells, results):
            for arm_name in ("naive", "recovery"):
                for index, (seed, arm) in enumerate(zip(seeds, result[arm_name])):
                    row = {key: cell[key] for key in ("cell_id", "slice", "motion", "delay_s", "loss_probability", "noise_std_m", "loss_model", "jitter_s", "tracker_measurement_std_m")}
                    row.update({"arm": arm_name, "seed_index": index, "seed": seed})
                    for key in ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count", "fused_out_of_order_count", "failed"):
                        row[key] = getattr(arm, key)
                    writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def run_sweep(output: Path, *, pilot: bool = False, seeds_count: int = 50, processes: int | None = None,
              base_seed: int = 20261013, overwrite: bool = False, argv: list[str] | None = None) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output path is not a directory: {output}")
        if any(output.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory is non-empty: {output}; pass --overwrite")
    else:
        output.mkdir(parents=True)
    if seeds_count < 1:
        raise ValueError("--seeds must be positive")
    cells = build_cells(pilot)
    seeds = seed_list(seeds_count, base_seed=base_seed)
    n_processes = processes if processes is not None else (os.cpu_count() or 1)
    if n_processes < 1:
        raise ValueError("--processes must be positive")
    config = RecoveryConfig()
    # Provenance is captured before anything is written, so the new output
    # files cannot make the tree look dirty, and the hashes describe the code
    # that actually ran.
    code_sha256 = {relative: _sha256(REPO_ROOT / relative) for relative in HASHED_SOURCES}
    git_revision = _git_value(["rev-parse", "HEAD"]) or None
    porcelain = _git_value(["status", "--porcelain"])
    git_tree_state = "unknown" if porcelain is None else ("dirty" if porcelain else "clean")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_processes) as pool:
        futures = [pool.submit(_run_cell, cell, seeds, config) for cell in cells]
        results = [future.result() for future in futures]
    runtime = time.perf_counter() - start

    summary = {
        "schema": "zephyr-s6-recovery-1",
        "status": "synthetic investigation: simulated link and sensor, not flight performance",
        "generator": "tools/run_s6_recovery.py",
        "config": asdict(config),
        "design": {
            "pilot": bool(pilot),
            "n_seeds": seeds_count,
            "base_seed": base_seed,
            "seeds": seeds,
            "statistic": "per-cell medians of per-seed p95 error, with paired per-seed recovery differences",
            "interval": ("distribution-free 95% order-statistic interval for medians (for 50 seeds the 18th and 33rd "
                         "smallest values, coverage 96.7%; with 5 or fewer seeds it falls back to min and max) and "
                         "Wilson 95% interval for failure rates"),
            "criterion": (f"a seed fails when its scored-window p95 error exceeds {config.failure_p95_m} m or its maximum "
                          f"error exceeds {config.failure_max_m} m (a research criterion, not a safety limit)"),
            "cell_label_rule": "a cell is labelled fails when at least half of its seeds fail, otherwise passes",
            "cell_id": "slice/motion/delay/loss/noise/loss-model/jitter/tracker standard deviation, fixed decimals",
            "arms": "naive fuses at receive time while recovery fuses at capture time and drops late out-of-order packets",
        },
        "code_sha256": code_sha256,
        "cells": [_cell_summary(cell, result) for cell, result in zip(cells, results)],
        "limits": [
            "simulated link and sensor (synthetic, not measured)",
            "constant-velocity filter only",
            "recovery drops late out-of-order packets (no retrodiction)",
            "fixed tuning: measurement std 0.05 m (R = 0.0025 m^2) and q = 0.1 m^2/s^3, except the labelled oracle slice",
            "targets at 1.0 m/s, turn radius 1.0 m",
            "10 Hz measurements",
            "criterion is a research choice, not a safety limit",
            f"{seeds_count} seeds per cell",
            "no hardware data",
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = output / "per_seed.csv"
    _write_csv(csv_path, cells, results, seeds)
    manifest = {
        "schema": "zephyr-s6-recovery-manifest-1",
        "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_recovery.py " + " ".join(argv or []),
        "argv": argv or [],
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "platform": platform.platform(),
        "git_revision": git_revision,
        "git_tree_state": git_tree_state,
        "runtime_s": runtime,
        "n_runs": len(cells) * seeds_count,
        "n_cells": len(cells),
        "n_seeds": seeds_count,
        "n_processes": n_processes,
        "outputs": {"summary.json": _sha256(summary_path), "per_seed.csv": _sha256(csv_path)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rate = (len(cells) * seeds_count) / runtime if runtime > 0 else float("inf")
    print(f"runs: {len(cells) * seeds_count}", file=sys.stderr)
    print(f"runtime_s: {runtime!r}", file=sys.stderr)
    print(f"runs_per_second: {rate!r}", file=sys.stderr)
    if pilot:
        full_runs = len(build_cells(False)) * 50
        projection = full_runs / rate if rate > 0 else float("inf")
        print(f"projected_full_grid_runtime_s (318 cells x 50 seeds): {projection!r}", file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--base-seed", type=int, default=20261013)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run_sweep(args.output, pilot=args.pilot, seeds_count=args.seeds, processes=args.processes,
              base_seed=args.base_seed, overwrite=args.overwrite, argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
