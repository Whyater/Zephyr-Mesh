"""The tuning experiment asks how filter mis-tuning appears in a simulated link and sensor.

All results come from a simulated lossy delayed radio link and simulated sensor,
with no measured hardware data. Run with ``PYTHONPATH=. python tools/run_s6_tuning.py
--output PATH``.
The 12-cell grid now has 51 arms, including Q-to-R-ratio twin arms at factors
0.04 and 25 for every model.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from sim.estimation.base import ArmSpec, NAIVE, RECOVERY_V1, EstimatorSettings
from sim.estimation.consistency import nis_band
from sim.estimation.engine import CellSpec, run_cell
from sim.estimation.sweep import (capture_provenance, run_jobs, sha256_file,
                                  sha256_json, write_csv, write_csv_gz, write_json,
                                  write_manifest)
from sim.s6_recovery import RecoveryConfig, seed_list

EXPERIMENT = "s6-tuning"
BASE_SEED = 20261013


def build_cells(pilot: bool = False) -> list[dict]:
    cells = []
    targets = ("cv", "ct")
    delays = (0.0, 0.1, 0.3)
    losses = (0.0, 0.30)
    for target in targets:
        for delay in delays:
            for loss in losses:
                cells.append({"slice": "main", "target": target, "delay_s": delay,
                              "loss_probability": loss, "noise_std_m": 0.05,
                              "rate_hz": 10.0, "jitter_s": 0.0,
                              "loss_model": "independent", "noise_model": "gaussian"})
    if pilot:
        cells = [c for c in cells if (c["target"], c["delay_s"], c["loss_probability"]) in
                 (("cv", 0.1, 0.3), ("ct", 0.3, 0.3))]
    return cells


def _fmt(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def build_arms() -> list[ArmSpec]:
    arms = [NAIVE, RECOVERY_V1]
    for model in ("cv", "ca", "ct", "imm"):
        for f in (0.2, 0.5, 2.0, 5.0):
            arms.append(ArmSpec(f"{model}_r{_fmt(f)}", model=model, r_std_m=f * 0.05))
        for f in (0.04, 0.2, 0.5, 2.0, 5.0, 25.0):
            arms.append(ArmSpec(f"{model}_q{_fmt(f)}", model=model, q_factor=f))
        if model != "cv":
            arms.append(ArmSpec(f"{model}_nominal", model=model))
    arms.extend([
        ArmSpec("cv_adaptq_r0.2", model="cv", adaptive="q", r_std_m=0.01),
        ArmSpec("cv_adaptq_q0.2", model="cv", adaptive="q", q_factor=0.2),
        ArmSpec("cv_adaptr_r0.2", model="cv", adaptive="r", r_std_m=0.01),
        ArmSpec("cv_adaptr_q0.2", model="cv", adaptive="r", q_factor=0.2),
        ArmSpec("imm_adaptq_r0.2", model="imm", adaptive="q", r_std_m=0.01),
        ArmSpec("imm_adaptq_q0.2", model="imm", adaptive="q", q_factor=0.2),
    ])
    return arms


def _cell_from_dict(data: dict) -> CellSpec:
    allowed = {f.name for f in CellSpec.__dataclass_fields__.values()}
    return CellSpec(**{k: v for k, v in data.items() if k in allowed})


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def job_cell(job: dict) -> dict:
    cell_data = dict(job["cell"])
    cell_data.pop("slice", None)
    cell = _cell_from_dict(cell_data)
    arms = [ArmSpec(**a) if not isinstance(a, ArmSpec) else a for a in job["arms"]]
    config = RecoveryConfig(**job.get("config", {}))
    settings = EstimatorSettings(**job.get("settings", {}))
    result = run_cell(cell, arms, seed_list(int(job["n_seeds"])), config, settings)
    result["slice"] = job.get("slice", "main")
    return _json_safe(result)


def _fingerprint(seeds: int) -> str:
    paths = [Path(__file__), Path("sim/s6_recovery.py"), Path("sim/link.py"), Path("sim/sensor.py"),
             Path("sim/tracker.py"), Path("sim/timesync.py")]
    paths.extend(sorted(Path("sim/estimation").glob("*.py")))
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode()); h.update(sha256_file(path).encode())
    h.update(str(int(seeds)).encode())
    return h.hexdigest()


def _write_outputs(out_dir: Path, results: list[dict], cells: list[dict], arms: list[ArmSpec],
                   seeds: list[int], argv: list[str], processes: int, runtime: float,
                   provenance: dict | None = None) -> None:
    rows = []
    per_seed = []
    for result in results:
        seed_rows = result.get("per_seed", [])
        for arm in arms:
            summary = result["arms"][arm.name]
            arm_rows = [r for r in seed_rows if r.get("arm") == arm.name]
            warns = [r.get("x_monitor_warn_fraction") for r in arm_rows
                      if r.get("x_monitor_warn_fraction") is not None]
            qscales = [r.get("x_final_q_scale") for r in arm_rows
                       if r.get("x_final_q_scale") is not None]
            rscales = [r.get("x_final_r_scale") for r in arm_rows
                       if r.get("x_final_r_scale") is not None]
            first_warnings = [r.get("x_monitor_first_warning_s") for r in arm_rows
                              if r.get("x_monitor_first_warning_s") is not None]
            paired_n = result.get("paired_vs_naive", {}).get(arm.name, {})
            paired_r = result.get("paired_vs_recovery", {}).get(arm.name, {})
            row = {"slice": result["slice"], "cell_key": result["key"], "arm": arm.name}
            row.update(summary)
            row.update({"x_monitor_warn_fraction_median": float(np.median(warns)) if warns else None,
                        "x_final_q_scale_median": float(np.median(qscales)) if qscales else None,
                        "x_final_r_scale_median": float(np.median(rscales)) if rscales else None,
                        "x_monitor_first_warning_s_median": float(np.median(first_warnings))
                        if first_warnings else None,
                        "monitor_alarm_rate": float(np.mean(np.asarray(warns) > 0.5)) if warns else None})
            summary.update({"x_monitor_warn_fraction_median": row["x_monitor_warn_fraction_median"],
                            "x_final_q_scale_median": row["x_final_q_scale_median"],
                            "x_final_r_scale_median": row["x_final_r_scale_median"],
                            "x_monitor_first_warning_s_median": row["x_monitor_first_warning_s_median"],
                            "monitor_alarm_rate": row["monitor_alarm_rate"]})
            row.update({"paired_vs_naive_median": paired_n.get("p95_diff_median_m"),
                        "paired_vs_naive_low": paired_n.get("p95_diff_ci_low_m"),
                        "paired_vs_naive_high": paired_n.get("p95_diff_ci_high_m"),
                        "paired_vs_recovery_median": paired_r.get("p95_diff_median_m"),
                        "paired_vs_recovery_low": paired_r.get("p95_diff_ci_low_m"),
                        "paired_vs_recovery_high": paired_r.get("p95_diff_ci_high_m")})
            rows.append(_json_safe(row))
        per_seed.extend(result.get("per_seed", []))
    columns = sorted({k for row in rows for k in row})
    write_csv(out_dir / "cells.csv", rows, columns)
    pcols = sorted({k for row in per_seed for k in row})
    write_csv_gz(out_dir / "per_seed.csv.gz", per_seed, pcols)
    cell_summaries = []
    for result in results:
        cell_summary = dict(result)
        cell_summary.pop("per_seed", None)
        cell_summaries.append(cell_summary)
    recovery_config = {name: getattr(RecoveryConfig(), name) for name in RecoveryConfig.__dataclass_fields__}
    summary = {"experiment": EXPERIMENT,
               "question": "How can mis-tuned filters be noticed from error and NIS?",
               "simulated": True,
               "failure_rule": {"cell_rule_status": "proposed, not yet adopted",
                                "p95_error_m": RecoveryConfig().failure_p95_m,
                                "max_error_m": RecoveryConfig().failure_max_m},
               "config": {"grid": cells, "n_arms": len(arms), "recovery": recovery_config}, "seeds": seeds,
               "arms": [a.as_dict() for a in arms], "cells": cell_summaries}
    write_json(out_dir / "summary.json", _json_safe(summary))
    write_json(out_dir / "consistency_check.json", _consistency_check(seeds))
    outputs = ["summary.json", "cells.csv", "per_seed.csv.gz", "consistency_check.json"]
    write_manifest(out_dir, outputs=outputs, argv=argv, n_processes=processes, runtime_s=runtime,
                   extra={"replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_tuning.py " + shlex.join(argv),
                          "n_cells": len(results), "n_runs": len(results) * len(seeds),
                          "n_seeds": len(seeds), "config_sha256": sha256_json(summary["config"]),
                          "seeds": seeds, "limits": "simulation only"}, schema="s6-tuning-manifest-1",
                   provenance=provenance if provenance is not None else capture_provenance())


def _consistency_check(seeds: list[int]) -> dict:
    cell = CellSpec(target="cwna", noise_std_m=0.05)
    result = run_cell(cell, [RECOVERY_V1], seeds, RecoveryConfig(), EstimatorSettings())
    rows = [row for row in result["per_seed"] if row["arm"] == RECOVERY_V1.name]
    nis_values = [row["mean_nis"] for row in rows if row["mean_nis"] is not None]
    warn = [row["x_monitor_warn_fraction"] for row in rows if row["x_monitor_warn_fraction"] is not None]
    low, high = nis_band(10)
    # A window is "inside the band" when the monitor did not flag it, so the inside fraction is
    # 1 minus the warn fraction (every seed has the same number of windows here: no loss, no delay).
    return {"target": "cwna", "arm": "recovery", "q": 0.1, "r": 0.05,
            "delay_s": 0.0, "loss_probability": 0.0, "n_seeds": len(seeds),
            "mean_of_seed_mean_nis": float(np.mean(nis_values)) if nis_values else None,
            "fraction_windows_inside_band": float(1.0 - np.mean(warn)) if warn else None,
            "band_low": low, "band_high": high}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--test-small", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out = Path(args.output)
    provenance = capture_provenance()  # before any output exists
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty or absent")
    out.mkdir(parents=True, exist_ok=True)
    if args.seeds < 1:
        raise ValueError("--seeds must be at least one")
    cells = build_cells(args.pilot)
    arms = build_arms()
    if args.test_small:
        cells = cells[:2]
        arms = arms[:4]
    jobs = [{"slice": c["slice"], "cell": c, "arms": [a.as_dict() for a in arms],
             "n_seeds": args.seeds} for c in cells]
    checkpoint = None if args.no_checkpoint else (args.checkpoint_dir or os.path.expanduser("~/zm-runb-cache/tuning"))
    processes = args.processes or max(1, (os.cpu_count() or 1) // 3)
    start = time.perf_counter()
    results = run_jobs(jobs, "tools.run_s6_tuning:job_cell", processes=processes,
                       checkpoint_dir=checkpoint, fingerprint=_fingerprint(args.seeds))
    replay_args = list(sys.argv[1:] if argv is None else argv)
    _write_outputs(out, results, cells, arms, seed_list(args.seeds), replay_args,
                   processes, time.perf_counter() - start, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
