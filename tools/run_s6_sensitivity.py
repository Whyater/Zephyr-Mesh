"""Sensitivity screening with simulated radio and sensor data only.

The experiment asks which link, target, sensor, clock, and tuning inputs move
the p95 tracking error. Results are simulations of a lossy delayed link and a
simulated sensor, with nothing measured. Run it with::

    PYTHONPATH=. ./.venv/bin/python tools/run_s6_sensitivity.py --output runs/s6-sensitivity

Two outputs come from the same nine inputs:

* one-at-a-time (OAT) sweeps around a central case (straight line 1 m/s, or a
  circle of radius 1 m at 1 m/s; 10 Hz, 5 cm noise, 100 ms delay, 10 %
  independent loss);
* Morris screening (r = 20 trajectories, 4 levels), run separately for two
  target classes. In class "straight" the target is always a straight line and
  the turn-rate input is present but cannot change anything, so its mu* must be
  exactly 0 (the real-run null check). In class "turn" the target is a circle
  with radius = speed / turn rate and the turn rate runs 0.25 to 2 rad/s.
"""
from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from sim.estimation.base import ArmSpec
from sim.estimation.engine import CellSpec, run_cell
from sim.estimation.morris import elementary_effects, morris_trajectories
from sim.estimation.sweep import (capture_provenance, default_processes, run_jobs, sha256_file,
                                  write_csv, write_csv_gz, write_json, write_manifest)
from sim.s6_recovery import RecoveryConfig, seed_list

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "s6-sensitivity"
BASE_SEED = 20261013
DEFAULT_SEEDS = 10
MORRIS_R = 20
MORRIS_LEVELS = 4
REAL_JOB_FUNCTION = "tools.run_s6_sensitivity:job_cell"

INPUTS = ("delay", "loss", "burst_mean_length", "noise", "speed", "turn_rate",
          "sensor_rate", "clock_error", "tuning_error")
UNITS = ("s", "probability", "packets", "m", "m/s", "rad/s", "Hz", "s", "factor on std")
INDEX = {name: i for i, name in enumerate(INPUTS)}
CLASSES = ("straight", "turn")
ARM_NAMES = ("recovery", "imm")

# Registered ranges. The turn-rate range depends on the target class: the
# straight class keeps the registered 0 to 2 rad/s (it has no effect there),
# the turn class starts at 0.25 rad/s so that radius = speed / turn rate stays
# finite.
LOWS = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 5.0, 0.0, 0.2])
HIGHS = np.array([0.3, 0.5, 10.0, 0.10, 2.0, 2.0, 50.0, 0.1, 5.0])
TURN_CLASS_LOW_TURN_RATE = 0.25
LOG_FLAGS = np.array([False, False, False, False, False, False, True, False, True])
# Central case. Burst length 1 packet means independent loss. The turn-rate
# entry is the straight case; the turn case centres at 1 rad/s (radius 1 m).
CENTERS = np.array([0.1, 0.10, 1.0, 0.05, 1.0, 0.0, 10.0, 0.0, 1.0])
TURN_CLASS_CENTER_TURN_RATE = 1.0

PER_SEED_COLUMNS = (
    "slice", "cell_key", "arm", "seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m",
    "mean_nis", "fused_count", "dropped_count", "fused_out_of_order_count", "gated_count",
    "refiltered_count", "failed", "x_final_q_scale", "x_final_r_scale",
    "x_gate_checked_clean", "x_gate_checked_outliers", "x_gate_forced_accepts", "x_gated_clean",
    "x_gated_outliers", "x_imm_ct_prob_mean", "x_monitor_first_warning_s",
    "x_monitor_warn_fraction",
)
OAT_COLUMNS = ("slice", "case", "input", "level", "arm", "p95_median_m", "p95_ci_low_m",
               "p95_ci_high_m", "fail_rate")

LIMITS = [
    "Simulated link and sensor only; nothing measured.",
    "Morris moves one input at a time from random starting points, so it ranks inputs and does "
    "not split the variance into exact shares.",
    "The ranking depends on the ranges chosen; a wider range makes an input look more important.",
    "sigma only says an input interacts or is nonlinear, not with what.",
    "Each point is the median p95 over the same common seeds (10 by default), so responses carry "
    "seed noise.",
    "Burst length is in packets while the sensor rate also varies, so rate and gap duration are "
    "mixed in this screening (the atlas sets bursts in seconds instead).",
    "The OAT levels are evenly spaced over each range and skip the central value for delay, loss, "
    "speed and sensor rate.",
    "The straight class has the turn-rate input present but without effect; its mu* of 0 is the "
    "null check. The turn class corner (2 m/s and 2 rad/s) is harder than any atlas target.",
]


def class_ranges(target_class: str) -> tuple[np.ndarray, np.ndarray]:
    """Registered low and high ends of every input for one target class."""
    if target_class not in CLASSES:
        raise ValueError(f"target_class must be one of {CLASSES}")
    lows = LOWS.copy()
    if target_class == "turn":
        lows[INDEX["turn_rate"]] = TURN_CLASS_LOW_TURN_RATE
    return lows, HIGHS.copy()


def class_center(target_class: str) -> np.ndarray:
    center = CENTERS.copy()
    if target_class == "turn":
        center[INDEX["turn_rate"]] = TURN_CLASS_CENTER_TURN_RATE
    return center


def map_inputs(unit, target_class: str = "straight") -> np.ndarray:
    """Map unit cube rows to the registered physical ranges of one class."""
    unit = np.asarray(unit, dtype=float)
    if unit.ndim == 0 or unit.shape[-1] != len(INPUTS):
        raise ValueError("unit must have one coordinate per sensitivity input")
    if np.any(~np.isfinite(unit)) or np.any((unit < 0) | (unit > 1)):
        raise ValueError("unit values must be in [0, 1]")
    lows, highs = class_ranges(target_class)
    mapped = lows + unit * (highs - lows)
    # Logarithmic coordinates are computed only for the two positive ranges, so
    # the linear ranges that start at zero never see log(0).
    for index in np.flatnonzero(LOG_FLAGS):
        mapped[..., index] = np.exp(np.log(lows[index]) + unit[..., index] *
                                    (np.log(highs[index]) - np.log(lows[index])))
    return mapped


def _cell(values, target_class: str = "straight") -> CellSpec:
    """Simulated conditions for one input vector in one target class."""
    values = np.asarray(values, dtype=float)
    if target_class not in CLASSES:
        raise ValueError(f"target_class must be one of {CLASSES}")
    speed = float(values[INDEX["speed"]])
    turn = float(values[INDEX["turn_rate"]])
    params = {"speed_mps": speed}
    if target_class == "turn":
        if turn <= 0:
            raise ValueError("the turn class needs a positive turn rate")
        params["radius_m"] = speed / turn
        target = "ct"
    else:
        target = "cv"  # the turn-rate input is deliberately ignored here
    clock_error = float(values[INDEX["clock_error"]])
    # A sensor clock that runs behind stamps packets early, the harmful side.
    clock = () if clock_error == 0 else (("mode", "unsynced"), ("offset_s", -clock_error))
    burst = float(values[INDEX["burst_mean_length"]])
    loss_model = "independent" if burst <= 1.0 + 1e-12 else "burst"
    return CellSpec(target=target, target_params=tuple(sorted(params.items())),
                    rate_hz=float(values[INDEX["sensor_rate"]]),
                    delay_s=float(values[INDEX["delay"]]),
                    loss_probability=float(values[INDEX["loss"]]), loss_model=loss_model,
                    burst_mean_length_packets=burst,
                    noise_std_m=float(values[INDEX["noise"]]), clock=clock)


def build_arms(values=None) -> list[ArmSpec]:
    """Recovery (cv) and IMM arms; the tuning factor scales the filter's measurement std."""
    factor = float(values[INDEX["tuning_error"]] if values is not None else 1.0)
    std = factor * 0.05
    return [ArmSpec("recovery", model="cv", stamping="capture", late_policy="drop", r_std_m=std),
            ArmSpec("imm", model="imm", stamping="capture", late_policy="drop", r_std_m=std)]


def build_oat_jobs(pilot: bool = False, seeds=None, inputs=None) -> list[dict]:
    """The deterministic OAT grid: two central cases, each input across its range.

    Case "cv" is the straight class; case "ct" is the turn class (circle).
    ``inputs`` restricts the swept inputs (reduced runs in tests).
    """
    levels = 3 if pilot else 5
    names = INPUTS if inputs is None else tuple(inputs)
    unknown = sorted(set(names) - set(INPUTS))
    if unknown:
        raise ValueError(f"unknown inputs: {unknown}")
    rows = []
    for case, target_class in (("cv", "straight"), ("ct", "turn")):
        lows, highs = class_ranges(target_class)
        center = class_center(target_class)
        for name in INPUTS:
            if name not in names:
                continue
            i = INDEX[name]
            if LOG_FLAGS[i]:
                grid = np.geomspace(lows[i], highs[i], levels)
            else:
                grid = np.linspace(lows[i], highs[i], levels)
            for value in grid:
                point = center.copy()
                point[i] = value
                rows.append({"slice": f"oat-{case}-{name}", "values": point.tolist(),
                             "case": case, "input": name, "level": float(value),
                             "target_class": target_class,
                             "seeds": [int(s) for s in seeds] if seeds is not None else []})
    return rows


def build_morris_jobs(r: int = MORRIS_R, seeds=None):
    """Morris jobs for both target classes (same unit design, same seed).

    Returns ``(jobs, design, orders)``; jobs are ordered class by class, then
    trajectory by trajectory, then design point by design point.
    """
    design, orders = morris_trajectories(len(INPUTS), int(r), MORRIS_LEVELS, BASE_SEED)
    points = design.shape[1]
    jobs = []
    for target_class in CLASSES:
        for i, row in enumerate(design.reshape(-1, len(INPUTS))):
            jobs.append({"slice": f"morris-{target_class}-{i // points}-{i % points}",
                         "values": map_inputs(row, target_class).tolist(),
                         "seeds": [int(s) for s in seeds] if seeds is not None else [],
                         "case": "morris", "target_class": target_class,
                         "trajectory": int(i // points), "point": int(i % points)})
    return jobs, design, orders


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def job_cell(job: dict) -> dict:
    """Run both arms on every seed of one OAT or Morris point (the real engine)."""
    values = np.asarray(job["values"], dtype=float)
    cell = _cell(values, job.get("target_class", "straight"))
    out = run_cell(cell, build_arms(values), job["seeds"], RecoveryConfig())
    out["slice"] = job.get("slice", "morris")
    out["values"] = values.tolist()
    out["case"] = job.get("case")
    out["input"] = job.get("input")
    out["target_class"] = job.get("target_class", "straight")
    return _clean(out)


def _fingerprint_files() -> list[Path]:
    """Every source file whose change can alter a result, found from the repo root."""
    files = [Path(__file__).resolve()]
    files += [REPO_ROOT / name for name in ("sim/s6_recovery.py", "sim/link.py", "sim/sensor.py",
                                            "sim/tracker.py", "sim/timesync.py")]
    files += sorted((REPO_ROOT / "sim" / "estimation").glob("*.py"))
    return files


def _fingerprint(seeds, job_function: str = REAL_JOB_FUNCTION) -> str:
    digest = hashlib.sha256()
    for path in _fingerprint_files():
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(sha256_file(path).encode())
    digest.update(",".join(str(int(s)) for s in seeds).encode())
    digest.update(job_function.encode())
    return digest.hexdigest()


def rank_from_mu_star(mu_star) -> list[int]:
    """Each input's rank, 1 for the largest mu*; ties go to the lower index."""
    mu_star = np.asarray(mu_star, dtype=float)
    order = np.argsort(-mu_star, kind="stable")
    ranks = np.empty(mu_star.size, dtype=int)
    ranks[order] = np.arange(1, mu_star.size + 1)
    return ranks.tolist()


def estimate_full_run(pilot_wall_s: float, pilot_jobs: int, full_jobs: int, pilot_seeds: int,
                      full_seeds: int) -> float:
    """Scale a measured pilot wall time to the full run at the same process count.

    The pilot wall time was already measured with its processes working at
    once, so it is scaled by job count and seed count only.
    """
    return pilot_wall_s * (full_jobs / max(1, pilot_jobs)) * (full_seeds / max(1, pilot_seeds))


def _summary_cell(result: dict) -> dict:
    """A point's summary without its per-seed rows (those live in per_seed.csv.gz)."""
    return {key: value for key, value in result.items() if key != "per_seed"}


def _per_seed_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for row in result.get("per_seed", []):
            extra = sorted(set(row) - set(PER_SEED_COLUMNS) - {"slice"})
            if extra:
                raise ValueError(f"per-seed row has columns missing from PER_SEED_COLUMNS: {extra}")
            rows.append({**row, "slice": result["slice"]})
    return rows


def _full_job_count() -> int:
    return len(build_oat_jobs(False)) + len(build_morris_jobs(MORRIS_R)[0])


def run_experiment(output, *, pilot: bool = False, seeds_count: int = DEFAULT_SEEDS,
                   processes: int | None = None, checkpoint_dir=None, morris_r: int | None = None,
                   oat_inputs=None, argv: list[str] | None = None,
                   job_function: str | None = None) -> dict:
    """Run the registered OAT and Morris design into an empty directory.

    ``job_function`` ("module:function") exists for tests only. It replaces the
    engine with a synthetic response function; the outputs are then labelled
    "synthetic_test": true and never "simulated". main() never passes it.
    ``checkpoint_dir`` None means no checkpoints. ``oat_inputs`` restricts the
    OAT sweep to some inputs (a reduced run); Morris always varies all nine.
    """
    out = Path(output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory must be empty")
    if seeds_count < 1:
        raise ValueError("seeds_count must be positive")
    synthetic = job_function is not None
    fn = job_function or REAL_JOB_FUNCTION
    n_processes = default_processes() if processes is None else int(processes)
    if n_processes < 1:
        raise ValueError("processes must be positive")
    r_count = int(morris_r if morris_r is not None else (2 if pilot else MORRIS_R))
    seeds = seed_list(seeds_count)
    oat_jobs = build_oat_jobs(pilot, seeds, oat_inputs)
    morris_jobs, design, orders = build_morris_jobs(r_count, seeds)
    jobs = oat_jobs + morris_jobs
    provenance = capture_provenance()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = run_jobs(jobs, fn, processes=n_processes, checkpoint_dir=checkpoint_dir,
                       progress=False if synthetic else None, fingerprint=_fingerprint(seeds, fn))
    jobs_runtime = time.perf_counter() - t0

    label = {"synthetic_test": True} if synthetic else {"simulated": True}
    n_oat = len(oat_jobs)
    oat = results[:n_oat]
    points = design.shape[1]
    per_class = r_count * points
    morris_results = {cls: results[n_oat + i * per_class:n_oat + (i + 1) * per_class]
                      for i, cls in enumerate(CLASSES)}

    # OAT outputs: fixed arm order, rows in job order.
    oat_rows = []
    for result in oat:
        for arm in ARM_NAMES:
            summary = result["arms"][arm]
            oat_rows.append({"slice": result["slice"], "case": result["case"], "input": result["input"],
                             "level": result["values"][INDEX[result["input"]]], "arm": arm,
                             "p95_median_m": summary["p95_median_m"],
                             "p95_ci_low_m": summary.get("p95_ci_low_m"),
                             "p95_ci_high_m": summary.get("p95_ci_high_m"),
                             "fail_rate": summary.get("fail_rate")})
    write_json(out / "oat.json", {"experiment": EXPERIMENT, **label,
                                  "cells": [_summary_cell(r) for r in oat]})
    write_csv(out / "oat.csv", oat_rows, OAT_COLUMNS)
    write_csv_gz(out / "per_seed.csv.gz", _per_seed_rows(results), PER_SEED_COLUMNS)

    # Morris outputs, per class and per arm.
    classes_doc = {}
    for target_class in CLASSES:
        lows, highs = class_ranges(target_class)
        responses, effects = {}, {}
        for arm in ARM_NAMES:
            y = np.array([r["arms"][arm]["p95_median_m"] for r in morris_results[target_class]],
                         dtype=float).reshape(r_count, points)
            e = elementary_effects(design, orders, y)
            order = np.argsort(-e["mu_star"], kind="stable")
            responses[arm] = y.tolist()
            effects[arm] = {"mu_star": e["mu_star"].tolist(), "mu": e["mu"].tolist(),
                            "sigma": e["sigma"].tolist(), "rank": rank_from_mu_star(e["mu_star"]),
                            "order": order.tolist(), "order_names": [INPUTS[i] for i in order]}
        classes_doc[target_class] = {
            "target": "cv" if target_class == "straight" else "ct (radius = speed / turn rate)",
            "input_ranges": {name: {"low": float(lows[i]), "high": float(highs[i]),
                                    "log_scale": bool(LOG_FLAGS[i]), "unit": UNITS[i]}
                             for i, name in enumerate(INPUTS)},
            "responses": responses, "effects": effects}
    write_json(out / "morris.json", {
        "experiment": EXPERIMENT, **label,
        "inputs": {"names": list(INPUTS), "units": list(UNITS),
                   "clock_error": "early sensor stamps; unsynced offset_s = -e",
                   "tuning_error": "measurement std; r_std_m = factor x 0.05",
                   "burst_mean_length": "packets; 1 means independent loss",
                   "turn_rate": "straight class: present but no effect (null check); "
                                "turn class: radius = speed / turn rate"},
        "design": {"seed": BASE_SEED, "r": r_count, "levels": MORRIS_LEVELS,
                   "points_per_trajectory": points, "seeds": seeds,
                   "unit_design": design.tolist()},
        "classes": classes_doc, "limits": LIMITS})

    ranking = {"by_mu_star": {cls: {arm: classes_doc[cls]["effects"][arm]["order_names"]
                                    for arm in ARM_NAMES} for cls in CLASSES}}
    oat_ranges, oat_rank = {}, {}
    for case in ("cv", "ct"):
        oat_ranges[case], oat_rank[case] = {}, {}
        for arm in ARM_NAMES:
            spans = {}
            for name in INPUTS:
                if oat_inputs is not None and name not in oat_inputs:
                    continue
                values = [r["arms"][arm]["p95_median_m"] for r in oat
                          if r["case"] == case and r["input"] == name]
                spans[name] = float(max(values) - min(values))
            names = sorted(spans, key=lambda n: (-spans[n], INPUTS.index(n)))
            oat_ranges[case][arm] = spans
            oat_rank[case][arm] = {name: names.index(name) + 1 for name in names}
    ranking["oat_ranges"] = oat_ranges
    ranking["oat_rank"] = oat_rank
    write_json(out / "ranking.json", ranking)

    config = RecoveryConfig()
    write_json(out / "summary.json", {
        "experiment": EXPERIMENT, "question": "Sensitivity of p95 error", **label,
        "failure_rule": {"per_seed": {"p95_error_m": config.failure_p95_m,
                                      "max_error_m": config.failure_max_m,
                                      "scored_window_s": [config.burn_in_s, config.duration_s]},
                         "cell_rule_status": "proposed, not yet adopted"},
        "seeds": seeds, "arms": [a.as_dict() for a in build_arms()],
        "design": {"oat_levels": 3 if pilot else 5, "morris_r": r_count,
                   "morris_levels": MORRIS_LEVELS, "morris_seed": BASE_SEED,
                   "classes": list(CLASSES), "n_oat_jobs": n_oat, "n_morris_jobs": len(morris_jobs)},
        "cells": [_summary_cell(r) for r in results], "limits": LIMITS})

    runtime = time.perf_counter() - t0
    used_argv = list(sys.argv[1:] if argv is None else argv)
    extra = {"n_cells": len(jobs), "n_runs": len(jobs) * len(seeds), "n_seeds": len(seeds),
             "seeds": seeds, "limits": LIMITS,
             "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_sensitivity.py "
                               + shlex.join(used_argv), **label}
    write_manifest(out, outputs=["summary.json", "oat.json", "oat.csv", "morris.json", "ranking.json",
                                 "per_seed.csv.gz"],
                   argv=used_argv, n_processes=n_processes, runtime_s=runtime, extra=extra,
                   schema=EXPERIMENT, provenance=provenance)

    if pilot and not synthetic:
        full_jobs = _full_job_count()
        full_seconds = estimate_full_run(jobs_runtime, len(jobs), full_jobs, seeds_count, DEFAULT_SEEDS)
        print(f"pilot jobs: {len(jobs)}, seeds: {seeds_count}, processes: {n_processes}",
              file=sys.stderr)
        print(f"pilot wall time s: {jobs_runtime!r}", file=sys.stderr)
        print(f"estimated full run s ({full_jobs} jobs, {DEFAULT_SEEDS} seeds, {n_processes} processes): "
              f"{full_seconds!r}", file=sys.stderr)
        for name in ("summary.json", "oat.csv", "per_seed.csv.gz"):
            size = (out / name).stat().st_size * full_jobs / len(jobs)
            size *= DEFAULT_SEEDS / seeds_count if name == "per_seed.csv.gz" else 1.0
            print(f"estimated full {name} bytes: {size!r}", file=sys.stderr)
    return {"results": results, "design": design, "label": label}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = None if args.no_checkpoint else (
        args.checkpoint_dir or Path.home() / "zm-runb-cache" / "sensitivity")
    run_experiment(args.output, pilot=args.pilot, seeds_count=args.seeds, processes=args.processes,
                   checkpoint_dir=checkpoint, argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
