"""Run the clock-sync experiment, asking how accurate sensor clock sync must be.

All results are simulated link and simulated sensor results, with nothing
measured. Run with ``PYTHONPATH=. python tools/run_s6_clocks.py --output DIR``.
The full grid has 132 U cells and 216 N cells, with naive, recovery, and imm
arms.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shlex, sys, time
from pathlib import Path
from typing import Any
import numpy as np

from sim.estimation.base import ArmSpec, NAIVE, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_cell
from sim.estimation.sweep import (capture_provenance, run_jobs, sha256_file,
                                  sha256_json, write_csv, write_csv_gz, write_json,
                                  write_manifest)
from sim.s6_recovery import RecoveryConfig, seed_list

ARMS = [NAIVE, RECOVERY_V1, ArmSpec("imm", model="imm")]
U_OFFSETS = (-0.1, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1)

def _clock(mode, **kw):
    d = {"mode": mode, **kw}
    return tuple(sorted(d.items()))

def build_cells(pilot: bool = False) -> list[dict[str, Any]]:
    cells = []
    targets = ("cv", "ct")
    if pilot:
        for off in (-0.05, 0.0, 0.05):
            for target in targets:
                cells.append({"slice": "U", "target": target, "delay_s": 0.1,
                              "loss_probability": 0.0, "clock": _clock("unsynced", offset_s=off,
                              drift_ppm=0.0, stamp_jitter_s=0.0)})
        for drift in (200.0,):
            for interval in (0.1, 10.0):
                for asym in (-0.04, 0.04):
                    cells.append({"slice": "N", "target": "cv", "delay_s": 0.1,
                                  "loss_probability": 0.0, "clock": _clock("ntp", offset_s=0.05,
                                  drift_ppm=drift, stamp_jitter_s=0.0, sync_interval_s=interval,
                                  sync_delay_s=0.025, sync_jitter_s=0.0,
                                  sync_loss_probability=0.0, sync_asymmetry_s=asym),
                                  "sync_loss_per_exchange": 0})
        return cells
    for off in U_OFFSETS:
        for delay in (0.05, 0.1, 0.3):
            for loss in (0.0, 0.30):
                for target in targets:
                    cells.append({"slice": "U", "target": target, "delay_s": delay,
                                  "loss_probability": loss, "clock": _clock("unsynced", offset_s=off,
                                  drift_ppm=0.0, stamp_jitter_s=0.0)})
    for offset in (0.05, -0.05):
      for drift in (0.0, 50.0, 200.0):
        for interval in (0.1, 1.0, 10.0):
            for asym in (-0.04, 0.0, 0.04):
                for sync_loss, per_exchange in ((0.0, 0.0),
                                                 (0.29289321881345254, 0.5)):
                    for target in targets:
                        cells.append({"slice": "N", "target": target, "delay_s": 0.1,
                                      "loss_probability": 0.0, "clock": _clock("ntp", offset_s=offset,
                                      drift_ppm=drift, stamp_jitter_s=0.0, sync_interval_s=interval,
                                      sync_delay_s=0.025, sync_jitter_s=0.0,
                                      sync_loss_probability=sync_loss, sync_asymmetry_s=asym),
                                      "sync_loss_per_exchange": per_exchange})
    return cells

def build_arms() -> list[ArmSpec]:
    return list(ARMS)

def _cell(job):
    d = dict(job["cell"])
    d.pop("slice", None)
    d.pop("sync_loss_per_exchange", None)
    d["clock"] = tuple(tuple(x) for x in d.get("clock", ()))
    return CellSpec(**d)

def _clean(value):
    if isinstance(value, dict): return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def job_cell(job: dict) -> dict:
    cell = _cell(job); seeds = [int(x) for x in job["seeds"]]
    result = run_cell(cell, build_arms(), seeds, RecoveryConfig())
    result["cell"]["sync_loss_per_exchange"] = job["cell"].get("sync_loss_per_exchange")
    errors = []
    for seed in seeds:
        errors.extend(np.atleast_1d(np.asarray(deliver_cell(cell, seed).stamp_errors_s, dtype=float)).tolist())
    arr = np.asarray(errors, dtype=float)
    result["slice"] = job["cell"]["slice"]
    result["stamp_error_stats"] = {"mean": float(np.mean(arr)) if arr.size else None,
                                    "std": float(np.std(arr)) if arr.size else None,
                                    "max_abs": float(np.max(np.abs(arr))) if arr.size else None}
    if dict(cell.clock).get("mode") == "ntp" and arr.size > len(seeds):
        result["stamp_error_stats"]["post_sync_mean"] = float(np.mean(arr.reshape(len(seeds), -1)[:, 1:]))
    return _clean(result)

def _fingerprint(seeds_count):
    paths = sorted(Path("sim/estimation").glob("*.py")) + [Path("sim/timesync.py"), Path("sim/link.py"),
             Path("sim/sensor.py"), Path("sim/tracker.py"), Path("sim/s6_recovery.py"), Path(__file__)]
    h = hashlib.sha256()
    for p in paths: h.update(str(p).encode()); h.update(bytes.fromhex(sha256_file(p)))
    h.update(str(seeds_count).encode()); return h.hexdigest()

def run(output: Path, *, pilot=False, seeds_count=50, processes=None, checkpoint_dir=None, no_checkpoint=False, argv=None, cells_override=None):
    provenance = capture_provenance()  # before any output exists
    if output.exists() and any(output.iterdir()): raise ValueError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if seeds_count < 1: raise ValueError("--seeds must be positive")
    seeds = seed_list(seeds_count); cells = build_cells(pilot) if cells_override is None else list(cells_override)
    jobs = [{"cell": c, "seeds": seeds} for c in cells]
    cp = None if no_checkpoint else (checkpoint_dir or Path.home() / "zm-runb-cache" / "clocks")
    nproc = processes or max(1, (os.cpu_count() or 1)//3)
    start = time.perf_counter(); results = run_jobs(jobs, "tools.run_s6_clocks:job_cell", processes=nproc, checkpoint_dir=cp, fingerprint=_fingerprint(seeds_count)); runtime = time.perf_counter()-start
    rows = []
    per_seed = []
    for c, r in zip(cells, results):
        for row in r["per_seed"]: per_seed.append({**row, "slice": c["slice"]})
        for arm, summary in r["arms"].items():
            rows.append({"slice": c["slice"], "key": r["key"], "arm": arm, **summary,
                         "paired_median_vs_naive": r.get("paired_vs_naive", {}).get(arm, {}).get("p95_diff_median_m"),
                         "paired_median_vs_recovery": r.get("paired_vs_recovery", {}).get(arm, {}).get("p95_diff_median_m")})
    config = {"experiment": "s6-clocks", "failure_rule": "same RecoveryConfig p95/max rule", "placeholder_arm": "imm is the placeholder for the best maneuver arm from the turn experiment"}
    recovery_config = RecoveryConfig()
    failure_rule = {"p95_m": recovery_config.failure_p95_m,
                    "max_m": recovery_config.failure_max_m,
                    "cell_rule": "a cell fails when at least 50 % of seeds fail",
                    "cell_rule_status": "proposed, not yet adopted"}
    args = list(sys.argv[1:] if argv is None else argv)
    cell_summaries = [{k: v for k, v in r.items() if k != "per_seed"} for r in results]
    summary = {"experiment_id": "s6-clocks", "question": "How good must clock sync be?", "simulated": True,
               "cell_rule_status": "proposed, not yet adopted", "failure_rule": failure_rule,
               "config": config, "seeds": seeds,
               "notes": "When stamp error e is larger than link delay d, the stamp lands in the tracker's future; the tracker then holds its estimate (no backward prediction) until query time passes the stamp.",
               "arms": [a.as_dict() for a in build_arms()], "cells": [{**r, "slice": c["slice"]} for c,r in zip(cells,cell_summaries)]}
    write_json(output/"summary.json", summary)
    cols = sorted({k for row in rows for k in row}); write_csv(output/"cells.csv", rows, cols)
    write_csv_gz(output/"per_seed.csv.gz", per_seed, sorted({k for row in per_seed for k in row}))
    write_manifest(output, outputs=["summary.json","cells.csv","per_seed.csv.gz"], argv=args, n_processes=nproc, runtime_s=runtime,
                   extra={"n_cells":len(cells),"n_runs":len(cells)*seeds_count,"n_seeds":seeds_count,"config_sha256":sha256_json(config),"seeds":seeds,"limits":["simulated link and sensor only"],
                          "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_clocks.py " + shlex.join(args)}, schema="s6-clocks-manifest-1", provenance=provenance)
    return runtime

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--pilot",action="store_true"); p.add_argument("--seeds",type=int,default=50); p.add_argument("--processes",type=int); p.add_argument("--checkpoint-dir",type=Path); p.add_argument("--no-checkpoint",action="store_true"); args=list(sys.argv[1:] if argv is None else argv); a=p.parse_args(args); run(a.output,pilot=a.pilot,seeds_count=a.seeds,processes=a.processes,checkpoint_dir=a.checkpoint_dir,no_checkpoint=a.no_checkpoint,argv=args)
if __name__ == "__main__": main()
