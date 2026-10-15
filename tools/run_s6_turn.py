"""Run the turn experiment, asking how turn-aware filters behave over delayed, lossy links.

All results are simulated link and simulated sensor results, with nothing measured.
Run with ``PYTHONPATH=. python tools/run_s6_turn.py --output DIR``.
The full grid has 399 cells (336 main and 63 burst), and each slice runs seven arms;
It includes the ``ca_q10`` and ``ct_q10`` arms to both slices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.estimation.base import ArmSpec, EstimatorSettings, NAIVE, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm_detailed, run_cell
from sim.estimation.sweep import (capture_provenance, run_jobs, sha256_file, sha256_json,
                                  write_csv, write_csv_gz, write_json, write_manifest)
from sim.estimation.targets import make_target
from sim.s6_recovery import RecoveryConfig, seed_list, summarize_arm

DELAYS = (0, .025, .05, .10, .15, .20, .30)
MAIN_LOSSES = (0, .10, .30, .50)
MAIN_NOISES = (0, .02, .05, .10)
BURST_LOSSES = (.10, .30, .50)
TARGETS = ("cv", "ct", "mixed")
ARMS = (NAIVE, RECOVERY_V1, ArmSpec("ca", model="ca"), ArmSpec("ct", model="ct"),
        ArmSpec("imm", model="imm"), ArmSpec("ca_q10", model="ca", q_factor=10.0),
        ArmSpec("ct_q10", model="ct", q_factor=10.0))


def build_cells(pilot: bool = False) -> list[dict[str, Any]]:
    delays = (0, .10, .30) if pilot else DELAYS
    losses = (0, .30) if pilot else MAIN_LOSSES
    noises = (.05,) if pilot else MAIN_NOISES
    cells = []
    for target in TARGETS:
        for delay in delays:
            for loss in losses:
                for noise in noises:
                    cells.append({"slice": "main", "target": target, "delay_s": float(delay),
                                  "loss_probability": float(loss), "loss_model": "independent", "noise_std_m": float(noise),
                                  "jitter_s": 0.0})
    if pilot:
        for target in TARGETS:
            cells.append({"slice": "burst", "target": target, "delay_s": 0.0,
                          "loss_probability": .30, "loss_model": "burst", "noise_std_m": .05,
                          "jitter_s": 0.0, "burst_mean_length_packets": 5.0})
    else:
        for target in TARGETS:
            for delay in DELAYS:
                for loss in BURST_LOSSES:
                    cells.append({"slice": "burst", "target": target, "delay_s": float(delay),
                                  "loss_probability": float(loss), "loss_model": "burst", "noise_std_m": .05,
                                  "jitter_s": 0.0, "burst_mean_length_packets": 5.0})
    expected = 21 if pilot else 399
    assert len(cells) == expected
    return cells


def build_arms() -> list[ArmSpec]:
    return list(ARMS)


def _cell(job: dict[str, Any]) -> CellSpec:
    fields = {k: job[k] for k in ("target", "delay_s", "loss_probability", "loss_model", "noise_std_m", "jitter_s")}
    if "burst_mean_length_packets" in job:
        fields["burst_mean_length_packets"] = job["burst_mean_length_packets"]
    return CellSpec(**fields)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def job_cell(job: dict[str, Any]) -> dict[str, Any]:
    cell = _cell(job)
    config = RecoveryConfig()
    seeds = seed_list(int(job["seeds"]))
    result = run_cell(cell, build_arms(), seeds, config, EstimatorSettings())
    if cell.target == "mixed":
        mixed_params = make_target("mixed", dict(cell.target_params)).params
        turn_start = float(mixed_params["straight1_s"])
        turn_duration = float(mixed_params["turn_angle_rad"]) / (
            float(mixed_params["speed_mps"]) / float(mixed_params["radius_m"])
        )
        turn_end = turn_start + turn_duration
        phase_values = {arm.name: {"straight1": [], "turn": [], "straight2": []} for arm in build_arms()}
        for seed in seeds:
            bundle = deliver_cell(cell, seed, config)
            for arm in build_arms():
                errors = run_arm_detailed(bundle, cell, arm, config).outcome.errors_m
                times = np.arange(0.0, config.duration_s + .02 / 2, .02)
                masks = (times >= 2.0) & (times < turn_start), (times >= turn_start) & (times < turn_end), times >= turn_end
                for name, mask in zip(("straight1", "turn", "straight2"), masks):
                    phase_values[arm.name][name].append(float(np.percentile(errors[mask], 95)))
        result["phase_p95"] = {arm: {phase: float(np.median(vals)) for phase, vals in phases.items()}
                                for arm, phases in phase_values.items()}
    imm_rows = [r.get("x_imm_ct_prob_mean") for r in result["per_seed"] if r["arm"] == "imm"]
    result["imm_ct_prob_mean"] = float(np.median([v for v in imm_rows if v is not None])) if any(v is not None for v in imm_rows) else None
    result["slice"] = job["slice"]
    return _clean(result)


def _fingerprint(seeds: int) -> str:
    paths = sorted(Path("sim/estimation").glob("*.py")) + [Path("sim/link.py"), Path("sim/sensor.py"), Path("sim/tracker.py"), Path("sim/s6_recovery.py"), Path(__file__)]
    return sha256_json({str(p): sha256_file(p) for p in paths} | {"seeds": seeds})


def run(output: Path, *, pilot: bool, seeds: int, processes: int | None, checkpoint_dir: Path | None, no_checkpoint: bool, argv: list[str]) -> None:
    provenance = capture_provenance()  # before any output exists
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [{**cell, "seeds": seeds} for cell in build_cells(pilot)]
    nproc = processes if processes is not None else max(1, (os.cpu_count() or 1) // 3)
    start = time.perf_counter()
    results = run_jobs(jobs, "tools.run_s6_turn:job_cell", processes=nproc,
                       checkpoint_dir=None if no_checkpoint else (checkpoint_dir or Path.home() / "zm-runb-cache" / "turn"),
                       fingerprint=_fingerprint(seeds))
    runtime = time.perf_counter() - start
    cells = []
    seed_rows = []
    for job, result in zip(jobs, results):
        entry = {"slice": job["slice"], "cell": result["cell"], "key": result["key"], "arms": result["arms"],
                 "paired_vs_naive": result["paired_vs_naive"], "paired_vs_recovery": result["paired_vs_recovery"]}
        for field in ("phase_p95", "imm_ct_prob_mean"):
            if field in result:
                entry[field] = result[field]
        cells.append(entry)
        seed_rows.extend(result["per_seed"])
    summary = {"experiment": "s6-turn", "question": "How do turn-aware filters recover over delayed and lossy links?", "simulated": True,
               "failure_rule": {"cell_rule_status": "proposed, not yet adopted", "p95_m": RecoveryConfig().failure_p95_m, "max_m": RecoveryConfig().failure_max_m},
               "config": asdict(RecoveryConfig()), "seeds": seed_list(seeds), "arms": [a.as_dict() for a in build_arms()], "cells": cells}
    write_json(output / "summary.json", summary)
    seed_columns = sorted({k for row in seed_rows for k in row})
    write_csv_gz(output / "per_seed.csv.gz", seed_rows, seed_columns)
    columns = ["slice", "key", "arm", "p95_median_m", "p95_ci_low_m", "p95_ci_high_m", "fail_rate", "fail_rate_ci_low", "fail_rate_ci_high", "max_error_median_m", "mean_nis_median", "fused_total", "dropped_total", "fused_out_of_order_total", "refiltered_total", "gated_total", "paired_vs_naive_median_m", "paired_vs_naive_ci_low_m", "paired_vs_naive_ci_high_m", "paired_vs_recovery_median_m", "paired_vs_recovery_ci_low_m", "paired_vs_recovery_ci_high_m"]
    rows = []
    for c in cells:
        cell_rows = [r for r in seed_rows if r.get("cell_key") == c["key"]]
        for arm, summary in c["arms"].items():
            paired_n = c["paired_vs_naive"].get(arm, {})
            paired_r = c["paired_vs_recovery"].get(arm, {})
            values = {"slice": c["slice"], "key": c["key"], "arm": arm, **summary,
                      "refiltered_total": sum(int(r.get("refiltered_count", 0)) for r in cell_rows if r.get("arm") == arm),
                      "gated_total": sum(int(r.get("gated_count", 0)) for r in cell_rows if r.get("arm") == arm),
                      "paired_vs_naive_median_m": paired_n.get("p95_diff_median_m"),
                      "paired_vs_naive_ci_low_m": paired_n.get("p95_diff_ci_low_m"),
                      "paired_vs_naive_ci_high_m": paired_n.get("p95_diff_ci_high_m"),
                      "paired_vs_recovery_median_m": paired_r.get("p95_diff_median_m"),
                      "paired_vs_recovery_ci_low_m": paired_r.get("p95_diff_ci_low_m"),
                      "paired_vs_recovery_ci_high_m": paired_r.get("p95_diff_ci_high_m")}
            rows.append({column: values.get(column) for column in columns})
    write_csv(output / "cells.csv", rows, columns)
    write_manifest(output, outputs=["summary.json", "cells.csv", "per_seed.csv.gz"], argv=argv, n_processes=nproc, runtime_s=runtime,
                   extra={"replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_turn.py " + shlex.join(argv), "n_cells": len(jobs), "n_runs": len(jobs) * seeds, "n_seeds": seeds, "config_sha256": sha256_json(asdict(RecoveryConfig())), "seeds": seed_list(seeds), "limits": ["simulated link and sensor", "10 Hz", "no jitter"]}, schema="s6-turn-manifest-1", provenance=provenance)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); p.add_argument("--output", type=Path, required=True); p.add_argument("--pilot", action="store_true"); p.add_argument("--seeds", type=int, default=50); p.add_argument("--processes", type=int); p.add_argument("--checkpoint-dir", type=Path); p.add_argument("--no-checkpoint", action="store_true"); a = p.parse_args(argv)
    if a.seeds < 1: raise ValueError("--seeds must be positive")
    run(a.output, pilot=a.pilot, seeds=a.seeds, processes=a.processes, checkpoint_dir=a.checkpoint_dir, no_checkpoint=a.no_checkpoint, argv=list(sys.argv[1:] if argv is None else argv)); return 0


if __name__ == "__main__": raise SystemExit(main())
