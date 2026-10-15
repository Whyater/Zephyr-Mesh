"""Run the sending experiment, testing whether a sender can send fewer packets while keeping tracking quality.

All results use a simulated link and simulated sensor, with nothing measured on hardware.
Run with: PYTHONPATH=. ./.venv/bin/python tools/run_s6_sending.py --output DIR
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
from sim.estimation.engine import CellSpec, deliver_cell, run_cell
from sim.estimation.policies import packet_bytes
from sim.estimation.sweep import (capture_provenance, compare_outputs, run_jobs, sha256_file,
                                  sha256_json, write_csv, write_csv_gz, write_json, write_manifest)
from sim.s6_recovery import RecoveryConfig, paired_summary, seed_list, summarize_arm

BASE_SEED = 20261013
POLICY_PARAMS: dict[str, dict[str, Any]] = {}
for rate in (5, 10, 20, 50):
    POLICY_PARAMS[f"per{rate}_f32"] = {"kind": "periodic", "rate_hz": float(rate), "quantization_m": None}
    POLICY_PARAMS[f"per{rate}_i16_1mm"] = {"kind": "periodic", "rate_hz": float(rate), "quantization_m": 0.001}
POLICY_PARAMS["per10_i16_1cm"] = {"kind": "periodic", "rate_hz": 10.0, "quantization_m": 0.01}
for delta, tag in ((0.02, "2cm"), (0.05, "5cm"), (0.10, "10cm"), (0.20, "20cm"), (0.40, "40cm")):
    POLICY_PARAMS[f"evt{tag}_f32"] = {"kind": "event", "delta_m": delta, "quantization_m": None}
    POLICY_PARAMS[f"evt{tag}_i16_1mm"] = {"kind": "event", "delta_m": delta, "quantization_m": 0.001}
POLICY_PARAMS["evt10cm_hb1s_f32"] = {"kind": "event", "delta_m": 0.10, "heartbeat_s": 1.0, "quantization_m": None}
POLICY_PARAMS["evt10cm_hb1s_i16_1mm"] = {"kind": "event", "delta_m": 0.10, "heartbeat_s": 1.0, "quantization_m": 0.001}
POLICY_LABELS = tuple(POLICY_PARAMS)

HASHED_SOURCES = tuple(sorted([str(p.relative_to(Path(__file__).resolve().parents[1]))
    for p in Path(__file__).resolve().parents[1].glob("sim/estimation/*.py")])) + (
    "sim/timesync.py", "sim/link.py", "sim/sensor.py", "sim/tracker.py", "sim/s6_recovery.py", "tools/run_s6_sending.py")


def build_arms() -> list[ArmSpec]:
    """Return the registered receiver arms."""
    return [NAIVE, RECOVERY_V1, ArmSpec("imm", model="imm")]


def build_cells(pilot: bool = False) -> list[dict[str, Any]]:
    """Build the sending experiment grid in deterministic order."""
    targets = ("cv", "ct", "jink")
    delays = (0.0, 0.1)
    losses = (0.0, 0.3)
    noises = (0.02, 0.05)
    policies = POLICY_LABELS
    if pilot:
        targets, noises, policies = ("cv", "jink"), (0.05,), ("per10_f32", "per50_f32", "evt5cm_f32", "evt10cm_hb1s_f32")
        delays, losses = (0.1,), (0.0, 0.3)
    cells = []
    for target in targets:
        for delay in delays:
            for loss in losses:
                for noise in noises:
                    for policy in policies:
                        cells.append({"slice": "main", "target": target, "rate_hz": 50.0,
                                      "delay_s": delay, "loss_probability": loss,
                                      "loss_model": "independent", "noise_std_m": noise,
                                      "policy": policy, "sender": tuple(sorted(POLICY_PARAMS[policy].items()))})
    return cells


def _cell_from_job(job: dict[str, Any]) -> CellSpec:
    fields = {k: v for k, v in job.items() if k in {"target", "target_params", "rate_hz", "delay_s", "jitter_s", "loss_probability", "loss_model", "burst_mean_length_packets", "noise_model", "noise_std_m", "noise_dof", "outlier_probability", "outlier_m", "clock", "sender"}}
    return CellSpec(**fields)


def _clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def job_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Run one cell with common random numbers across the three arms."""
    cell = _cell_from_job(job)
    if "seeds" in job:
        seeds = [int(v) for v in job["seeds"]]
    else:
        seeds = seed_list(int(job.get("n_seeds", 1)), BASE_SEED)
    result = run_cell(cell, build_arms(), seeds, RecoveryConfig())
    duration = RecoveryConfig().duration_s
    budget_rows = []
    for seed in seeds:
        bundle = deliver_cell(cell, seed, RecoveryConfig(), EstimatorSettings())
        budget_rows.append({
            "seed": seed,
            "sent_count": int(bundle.sent_count),
            "packets_per_second": bundle.sent_count / duration,
            "data_bytes_per_second": bundle.sent_count * bundle.data_bytes_per_packet / duration,
            "on_air_bytes_per_second": bundle.sent_count * bundle.on_air_bytes_per_packet / duration,
        })
    budget_sent = [row["packets_per_second"] for row in budget_rows]
    budget_data = [row["data_bytes_per_second"] for row in budget_rows]
    budget_air = [row["on_air_bytes_per_second"] for row in budget_rows]
    budget_by_seed = {row["seed"]: row for row in budget_rows}
    for row in result["per_seed"]:
        row.update(budget_by_seed[row["seed"]])
    result["policy"] = job["policy"]
    result["budget"] = {"packets_per_second": float(np.mean(budget_sent)),
                         "data_bytes_per_second": float(np.mean(budget_data)),
                         "on_air_bytes_per_second": float(np.mean(budget_air)),
                         "packets_per_second_median": float(np.median(budget_sent)),
                         "data_bytes_per_second_median": float(np.median(budget_data)),
                         "on_air_bytes_per_second_median": float(np.median(budget_air)),
                         "data_bytes_per_packet": int(packet_bytes(POLICY_PARAMS[job["policy"]].get("quantization_m") is not None)["data"]),
                         "on_air_bytes_per_packet": int(packet_bytes(POLICY_PARAMS[job["policy"]].get("quantization_m") is not None)["on_air"]),
                         "overhead_basis": "43 B fixed ESP-NOW overhead, literature-derived, not measured"}
    return _clean(result)


def _fingerprint(seeds_count: int) -> str:
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for name in HASHED_SOURCES:
        path = root / name
        if path.is_file():
            h.update(name.encode()); h.update(path.read_bytes())
    h.update(str(seeds_count).encode())
    return h.hexdigest()


def _csv_rows(results: list[dict[str, Any]], cells: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    summary_rows, seed_rows = [], []
    for cell, result in zip(cells, results):
        for arm, summary in result["arms"].items():
            paired_n = result.get("paired_vs_naive", {}).get(arm, {})
            paired_r = result.get("paired_vs_recovery", {}).get(arm, {})
            summary_rows.append({"key": result["key"], "slice": "main", "policy": result["policy"], "arm": arm,
                                 **summary, "paired_median_vs_naive": paired_n.get("p95_diff_median_m"), "paired_low_vs_naive": paired_n.get("p95_diff_ci_low_m"), "paired_high_vs_naive": paired_n.get("p95_diff_ci_high_m"),
                                 "paired_median_vs_recovery": paired_r.get("p95_diff_median_m"), "paired_low_vs_recovery": paired_r.get("p95_diff_ci_low_m"), "paired_high_vs_recovery": paired_r.get("p95_diff_ci_high_m")})
        seed_rows.extend(result["per_seed"])
    return summary_rows, seed_rows


def run_sweep(output: Path, *, pilot: bool = False, seeds_count: int = 50, processes: int | None = None,
              checkpoint_dir: Path | None = None, no_checkpoint: bool = False, argv: list[str] | None = None) -> dict[str, Any]:
    if seeds_count < 1:
        raise ValueError("seeds_count must be positive")
    provenance = capture_provenance()  # before any output exists
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    cells = build_cells(pilot)
    seeds = seed_list(seeds_count, BASE_SEED)
    jobs = [dict(c, seeds=seeds) for c in cells]
    checkpoint = None if no_checkpoint else (checkpoint_dir or Path.home() / "zm-runb-cache" / "sending")
    nproc = processes if processes is not None else max(1, (os.cpu_count() or 1) // 3)
    start_time = time.perf_counter()
    results = run_jobs(jobs, "tools.run_s6_sending:job_cell", processes=nproc, checkpoint_dir=checkpoint, fingerprint=_fingerprint(seeds_count))
    runtime = time.perf_counter() - start_time
    clean_results = [_clean(r) for r in results]
    cells_out = [{"slice": "main", "cell": _cell_from_job(c).as_dict(), "key": r["key"], "policy": r["policy"], "arms": r["arms"], "paired_vs_naive": r.get("paired_vs_naive", {}), "paired_vs_recovery": r.get("paired_vs_recovery", {}), "budget": r["budget"]} for c, r in zip(cells, clean_results)]
    config = asdict(RecoveryConfig())
    limits = ["simulated link and simulated sensor, nothing measured", "43 B on-air overhead is literature-derived"]
    summary = {"experiment": "s6-sending", "question": "Can sending fewer packets preserve tracking quality?", "simulated": True, "failure_rule": {"cell_rule_status": "proposed, not yet adopted", "p95_m": config["failure_p95_m"], "max_m": config["failure_max_m"]}, "config": config, "seeds": seeds, "arms": [a.as_dict() for a in build_arms()], "policies": POLICY_PARAMS, "cells": cells_out, "limits": limits}
    write_json(output / "summary.json", summary)
    srows, prows = _csv_rows(clean_results, cells)
    write_csv(output / "cells.csv", srows, sorted({k for r in srows for k in r}))
    write_csv_gz(output / "per_seed.csv.gz", prows, sorted({k for r in prows for k in r}))
    replay_args = list(sys.argv[1:] if argv is None else argv)
    return write_manifest(output, outputs=["summary.json", "cells.csv", "per_seed.csv.gz"], argv=replay_args, n_processes=nproc, runtime_s=runtime, extra={"replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_sending.py " + shlex.join(replay_args), "n_cells": len(cells), "n_runs": len(cells) * seeds_count, "n_seeds": seeds_count, "config_sha256": sha256_json(config), "seeds": seeds, "limits": limits}, schema="s6-sending-manifest-1", provenance=provenance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    run_sweep(args.output, pilot=args.pilot, seeds_count=args.seeds, processes=args.processes, checkpoint_dir=args.checkpoint_dir, no_checkpoint=args.no_checkpoint, argv=list(sys.argv[1:] if argv is None else argv))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
