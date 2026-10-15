"""Run the late-packet experiment: re-filtering of late packets.

This experiment asks whether fixed-lag re-filtering improves tracking when
packets arrive out of capture-time order. Results use a simulated link and a
simulated sensor only, with nothing measured. Run it with:

    PYTHONPATH=. .venv/bin/python tools/run_s6_retro.py --output DIR

The full grid has 56 cells and three arms (naive, recovery, and retro). The
census uses packets received by the final query and reports
late-within-window drops inferred from detailed-arm replay counts.
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

from sim.estimation.base import ArmSpec, NAIVE, RECOVERY_V1, EstimatorSettings
from sim.estimation.engine import (TIME_TOLERANCE_S, CellSpec, deliver_cell,
                                   retro_window_for, run_arm_detailed, run_cell)
from sim.estimation.sweep import (capture_provenance, run_jobs, sha256_file,
                                  sha256_json, write_csv, write_csv_gz, write_json,
                                  write_manifest)
from sim.s6_recovery import RecoveryConfig, paired_summary, seed_list, summarize_arm

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261013
DELAYS = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
ARMS = (NAIVE, RECOVERY_V1, ArmSpec("retro", model="cv", stamping="capture", late_policy="retrodict"))


def build_cells(pilot: bool = False) -> list[dict[str, Any]]:
    """Return the deterministic late-packet experiment cell grid."""
    cells: list[dict[str, Any]] = []
    if pilot:
        specs = [("J1", (0.10, 0.30), (0.30,), "half"),
                 ("J2", (0.10, 0.30), (0.30,), "full"),
                 ("J0", (0.10,), (0.30,), "zero")]
        targets = {"J1": ("cv",), "J2": ("cv",), "J0": ("cv",)}
    else:
        specs = [("J1", DELAYS, (0.0, 0.30), "half"),
                 ("J2", DELAYS, (0.0, 0.30), "full"),
                 ("J0", (0.10, 0.30), (0.0, 0.30), "zero")]
        targets = {name: ("cv", "ct") for name, *_ in specs}
    for slice_name, delays, losses, jitter_kind in specs:
        for target in targets[slice_name]:
            for delay in delays:
                for loss in losses:
                    jitter = 0.0 if jitter_kind == "zero" else (delay / 2.0 if jitter_kind == "half" else delay)
                    cells.append({"slice": slice_name, "target": target, "target_params": {},
                                  "rate_hz": 10.0, "delay_s": float(delay), "jitter_s": float(jitter),
                                  "loss_probability": float(loss), "loss_model": "independent",
                                  "noise_model": "gaussian", "noise_std_m": 0.05})
    expected = 5 if pilot else 56
    assert len(cells) == expected, len(cells)
    return cells


def build_arms() -> list[ArmSpec]:
    """Return the three late-packet experiment arms."""
    return list(ARMS)


def _cell_from_job(job: dict[str, Any]) -> tuple[str, CellSpec]:
    raw = dict(job.get("cell", job))
    slice_name = str(raw.pop("slice"))
    raw.pop("cell_id", None)
    allowed = {f.name for f in CellSpec.__dataclass_fields__.values()}
    fields = {k: v for k, v in raw.items() if k in allowed}
    return slice_name, CellSpec(**fields)


def _census(cell: CellSpec, seeds: list[int], config: RecoveryConfig) -> dict[str, Any]:
    count = delivered = 0
    per_seed: list[float] = []
    for seed in seeds:
        bundle = deliver_cell(cell, seed, config)
        eligible = [d for d in bundle.deliveries
                    if d.receive_time_s <= config.duration_s + TIME_TOLERANCE_S]
        seed_count = 0
        newest = float("-inf")
        for delivery in eligible:
            delivered += 1
            capture = float(delivery.capture_time_s)
            if capture < newest:
                count += 1
                seed_count += 1
            newest = max(newest, capture)
        per_seed.append(seed_count / len(eligible) if eligible else 0.0)
    pooled = count / delivered if delivered else 0.0
    mean = sum(per_seed) / len(per_seed) if per_seed else 0.0
    return {"fraction": pooled, "count": count, "delivered": delivered,
            "mean": mean, "min": min(per_seed, default=0.0), "max": max(per_seed, default=0.0),
            "per_seed": per_seed}


def _late_drop_diagnostics(cell: CellSpec, seeds: list[int], config: RecoveryConfig,
                           window: float, settings: EstimatorSettings) -> list[dict[str, int]]:
    """Classify late deliveries and infer retro drops from detailed arm counts."""
    diagnostics = []
    recovery_arm = next(arm for arm in build_arms() if arm.name == "recovery")
    retro_arm = next(arm for arm in build_arms() if arm.name == "retro")
    for seed in seeds:
        bundle = deliver_cell(cell, seed, config, settings)
        recovery = run_arm_detailed(bundle, cell, recovery_arm, config, settings)
        retro = run_arm_detailed(bundle, cell, retro_arm, config, settings)
        fused_recovery = {packet.seq for packet in recovery.fused}
        fused_retro = {packet.seq for packet in retro.fused}
        newest_recovery = float("-inf")
        newest_retro = float("-inf")
        recovery_out_of_window = 0
        retro_within_window = 0
        for delivery in bundle.deliveries:
            if delivery.receive_time_s > config.duration_s + TIME_TOLERANCE_S:
                continue
            stamp = float(delivery.stamp_s)
            if delivery.seq not in fused_recovery and stamp < newest_recovery:
                if newest_recovery - stamp > window:
                    recovery_out_of_window += 1
            if (delivery.seq not in fused_retro and newest_retro != float("-inf")
                    and stamp >= newest_retro - window):
                retro_within_window += 1
            if delivery.seq in fused_recovery:
                newest_recovery = max(newest_recovery, stamp)
            if delivery.seq in fused_retro:
                newest_retro = max(newest_retro, stamp)
        diagnostics.append({
            "late_within_window_dropped": retro_within_window,
            "out_of_window_late_dropped": recovery_out_of_window,
            "recovery_dropped_count": int(recovery.outcome.dropped_count),
            "retro_refiltered_count": int(retro.outcome.refiltered_count),
        })
    return diagnostics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return value


def job_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Run one cell from a plain job dictionary."""
    slice_name, cell = _cell_from_job(job)
    base = int(job.get("base_seed", BASE_SEED))
    requested = job.get("seeds", job.get("n_seeds", 50))
    seeds = [int(v) for v in requested] if isinstance(requested, (list, tuple)) else seed_list(int(requested), base_seed=base)
    config = RecoveryConfig()
    result = run_cell(cell, build_arms(), seeds, config, EstimatorSettings())
    result["slice"] = slice_name
    result["retro_window_s"] = retro_window_for(cell)
    result["reorder_census"] = _census(cell, seeds, config)
    diagnostics = _late_drop_diagnostics(cell, seeds, config, result["retro_window_s"], EstimatorSettings())
    result["late_within_window_dropped"] = sum(d["late_within_window_dropped"] for d in diagnostics)
    result["out_of_window_late_dropped"] = sum(d["out_of_window_late_dropped"] for d in diagnostics)
    result["reorder_census"]["late_within_window_dropped"] = result["late_within_window_dropped"]
    result["reorder_census"]["out_of_window_late_dropped"] = result["out_of_window_late_dropped"]
    by_seed = {seed: diagnostic for seed, diagnostic in zip(seeds, diagnostics)}
    census_by_seed = {seed: fraction for seed, fraction in zip(seeds, result["reorder_census"]["per_seed"])}
    for row in result.get("per_seed", []):
        row["slice"] = slice_name
        row["retro_window_s"] = result["retro_window_s"]
        row["reorder_fraction"] = census_by_seed[int(row["seed"])]
        diagnostic = by_seed[int(row["seed"])]
        row["late_within_window_dropped"] = diagnostic["late_within_window_dropped"]
        row["out_of_window_late_dropped"] = diagnostic["out_of_window_late_dropped"]
    return _json_safe(result)


def _fingerprint(seeds_count: int) -> str:
    paths = sorted((REPO_ROOT / "sim/estimation").glob("*.py"))
    paths += [REPO_ROOT / p for p in ("sim/timesync.py", "sim/link.py", "sim/sensor.py", "sim/tracker.py", "sim/s6_recovery.py", "tools/run_s6_retro.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(sha256_file(path).encode())
    digest.update(str(int(seeds_count)).encode())
    return digest.hexdigest()


def _rows(results: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    cells, per_seed, cell_arm_rows = [], [], []
    for result in results:
        cells.append({"slice": result["slice"], "cell": result["cell"], "key": result["key"],
                      "arms": result["arms"], "paired_vs_naive": result.get("paired_vs_naive", {}),
                      "paired_vs_recovery": result.get("paired_vs_recovery", {}),
                      "retro_window_s": result["retro_window_s"], "reorder_census": result["reorder_census"],
                      "late_within_window_dropped": result.get("late_within_window_dropped", 0),
                      "out_of_window_late_dropped": result.get("out_of_window_late_dropped", 0)})
        rows = result.get("per_seed", [])
        per_seed.extend(rows)
        for arm_name, summary in result["arms"].items():
            paired_n = result.get("paired_vs_naive", {}).get(arm_name, {})
            paired_r = result.get("paired_vs_recovery", {}).get(arm_name, {})
            cell_arm_rows.append({"cell_key": result["key"], "slice": result["slice"], "arm": arm_name,
                "p95_median_m": summary.get("p95_median_m"), "p95_ci_low_m": summary.get("p95_ci_low_m"),
                "p95_ci_high_m": summary.get("p95_ci_high_m"), "fail_rate": summary.get("fail_rate"),
                "fail_rate_ci_low": summary.get("fail_rate_ci_low"), "fail_rate_ci_high": summary.get("fail_rate_ci_high"),
                "max_error_median_m": summary.get("max_error_median_m"), "mean_nis_median": summary.get("mean_nis_median"),
                "fused_total": summary.get("fused_total"), "dropped_total": summary.get("dropped_total"),
                "refiltered_total": sum(int(r.get("refiltered_count", 0)) for r in rows if r.get("arm") == arm_name),
                "gated_total": sum(int(r.get("gated_count", 0)) for r in rows if r.get("arm") == arm_name),
                "paired_vs_naive_median_m": paired_n.get("p95_diff_median_m"),
                "paired_vs_naive_ci_low_m": paired_n.get("p95_diff_ci_low_m"),
                "paired_vs_naive_ci_high_m": paired_n.get("p95_diff_ci_high_m"),
                "paired_vs_recovery_median_m": paired_r.get("p95_diff_median_m"),
                "paired_vs_recovery_ci_low_m": paired_r.get("p95_diff_ci_low_m"),
                "paired_vs_recovery_ci_high_m": paired_r.get("p95_diff_ci_high_m"),
                "retro_window_s": result["retro_window_s"], "reorder_fraction": result["reorder_census"]["fraction"],
                "late_within_window_dropped": result.get("late_within_window_dropped", 0),
                "out_of_window_late_dropped": result.get("out_of_window_late_dropped", 0)})
    return cells, per_seed, cell_arm_rows


def run_sweep(output: Path, *, pilot: bool = False, seeds_count: int = 50,
              processes: int | None = None, checkpoint_dir: Path | None = None,
              no_checkpoint: bool = False, argv: list[str] | None = None) -> dict:
    if seeds_count < 1:
        raise ValueError("--seeds must be positive")
    provenance = capture_provenance()  # before any output exists
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output directory is non-empty: {output}")
    else:
        output.mkdir(parents=True)
    nproc = processes if processes is not None else max(1, (os.cpu_count() or 1) // 3)
    if nproc < 1:
        raise ValueError("--processes must be positive")
    cells = build_cells(pilot)
    jobs = [dict(cell, seeds=seeds_count, base_seed=BASE_SEED) for cell in cells]
    ckpt = None if no_checkpoint else (checkpoint_dir or Path.home() / "zm-runb-cache" / "retro")
    start = time.perf_counter()
    results = run_jobs(jobs, "tools.run_s6_retro:job_cell", processes=nproc,
                       checkpoint_dir=ckpt, fingerprint=_fingerprint(seeds_count))
    runtime = time.perf_counter() - start
    cell_rows, seed_rows, cell_arm_rows = _rows(results)
    config = asdict(RecoveryConfig())
    summary = {"experiment": "s6-retro", "experiment_id": "s6-retro",
               "question": "Does re-filtering late packets improve tracking?",
               "simulated": True, "cell_rule_status": "proposed, not yet adopted",
               "failure_rule": {"p95_m": config["failure_p95_m"], "max_m": config["failure_max_m"],
                                "cell_rule": "a cell fails when at least 50 % of seeds fail",
                                "cell_rule_status": "proposed, not yet adopted"}, "config": config,
               "seeds": seed_list(seeds_count), "arms": [a.as_dict() for a in build_arms()], "cells": cell_rows}
    write_json(output / "summary.json", summary)
    cell_columns = list(cell_arm_rows[0]) if cell_arm_rows else []
    seed_base = ["cell_key", "slice", "arm", "seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m",
                 "mean_nis", "fused_count", "dropped_count", "fused_out_of_order_count", "refiltered_count",
                 "gated_count", "failed", "retro_window_s", "reorder_fraction"]
    seed_extra = sorted({key for row in seed_rows for key in row if key not in seed_base})
    seed_columns = seed_base + seed_extra
    write_csv(output / "cells.csv", cell_arm_rows, cell_columns)
    write_csv_gz(output / "per_seed.csv.gz", seed_rows, seed_columns)
    extra = {"n_cells": len(cells), "n_runs": len(cells) * seeds_count, "n_seeds": seeds_count,
             "config_sha256": sha256_json(config), "seeds": seed_list(seeds_count),
             "limits": ["simulated link and sensor only", "10 Hz", "targets cv and ct", "no hardware data"]}
    args = list(sys.argv[1:] if argv is None else argv)
    extra["replay_command"] = "PYTHONPATH=. ./.venv/bin/python tools/run_s6_retro.py " + shlex.join(args)
    manifest = write_manifest(output, outputs=["summary.json", "cells.csv", "per_seed.csv.gz"],
                             argv=args, n_processes=nproc, runtime_s=runtime, extra=extra,
                             schema="s6-retro-manifest-1", provenance=provenance)
    if pilot:
        estimate = runtime * (56 * 50) / (len(cells) * seeds_count)
        print(f"pilot cells: {len(cells)}, seeds: {seeds_count}, wall time: {runtime:.3f} s, estimated full-run wall time at 50 seeds: {estimate:.3f} s", file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    run_sweep(args.output, pilot=args.pilot, seeds_count=args.seeds, processes=args.processes,
              checkpoint_dir=args.checkpoint_dir, no_checkpoint=args.no_checkpoint,
              argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
