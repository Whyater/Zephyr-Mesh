"""Run the simulated stale-neighbor separation-buffer experiment."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Iterable

from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import argv_or_default, default_processes, prepare_output, seeds_for, write_json, write_manifest

MIN_SEPARATION_M = 0.35
BUFFER_STEP_M = 0.05
MAX_RADIUS_M = 1.50
N_SEEDS = 75
CONTROL_DT_S = 0.05
SHARE_RATE_HZ = 10.0


def separation_grid(*, pilot: bool = False) -> list[dict[str, Any]]:
    """Return the deterministic condition grid used by run, config, and pilot."""
    if pilot:
        ring = [{"kind": "ring_swap", "speed_mps": 1.0, "delay_s": 0.0,
                 "loss_probability": 0.0, "noise_std_m": 0.05, "dead_reckoning": False,
                 "lateral_offset_m": 0.0}]
        other = [{"kind": "head_on", "speed_mps": 1.0, "delay_s": 0.0,
                  "loss_probability": 0.0, "noise_std_m": 0.05, "dead_reckoning": False,
                  "lateral_offset_m": 0.0}]
        return ring + other
    ring = [{"kind": "ring_swap", "speed_mps": 1.0, "delay_s": delay,
             "loss_probability": loss, "noise_std_m": 0.05, "dead_reckoning": dr,
             "lateral_offset_m": 0.0}
            for delay in (0.0, 0.1, 0.2, 0.5) for loss in (0.0, 0.3) for dr in (False, True)]
    full = [{"kind": kind, "speed_mps": speed, "delay_s": delay,
             "loss_probability": loss, "noise_std_m": noise, "dead_reckoning": dr,
             "lateral_offset_m": offset if kind == "head_on" else 0.0}
            for kind in ("head_on", "crossing") for offset in ((0.0, 0.1) if kind == "head_on" else (0.0,))
            for speed in (0.5, 1.0, 2.0)
            for delay in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
            for loss in (0.0, 0.1, 0.3) for noise in (0.0, 0.05) for dr in (False, True)]
    return ring + full


def _grid_config(pilot: bool = False) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Any]]] = {}
    for cell in separation_grid(pilot=pilot):
        spec = grouped.setdefault(cell["kind"], {k: [] for k in ("speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning", "lateral_offset_m")})
        for key in spec:
            if cell[key] not in spec[key]:
                spec[key].append(cell[key])
    return grouped


def _spec(job: dict[str, Any]):
    from sim.coop.separation import EncounterSpec
    return EncounterSpec(kind=str(job["kind"]), speed_mps=float(job["speed_mps"]), delay_s=float(job["delay_s"]),
                         loss_probability=float(job["loss_probability"]), noise_std_m=float(job["noise_std_m"]),
                         dead_reckoning=bool(job["dead_reckoning"]), lateral_offset_m=float(job.get("lateral_offset_m", 0.1)),
                         avoidance_radius_m=float(job["avoidance_radius_m"]),
                         share_rate_hz=SHARE_RATE_HZ, control_dt_s=CONTROL_DT_S)


def _result_row(result: dict[str, Any], seed: int, seed_index: int) -> dict[str, Any]:
    return {"seed": int(seed), "seed_index": int(seed_index),
            "min_center_distance_m": float(result["min_center_distance_m"]),
            "violated": bool(result["violated"]), "near_miss": bool(result["near_miss"]),
            "min_ellipsoid_separation": float(result["min_ellipsoid_separation"]),
            "ellipsoid_violated": bool(result["ellipsoid_violated"]),
            "penetration_m": float(result["penetration_m"]), "reached_goals": bool(result["reached_goals"]),
            "mean_message_age_s": float(result["mean_message_age_s"])}


def _aggregate(radius: float, rows: list[dict[str, Any]], first_violation_seed_index: int | None) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("condition produced no rows")
    distances = [float(row["min_center_distance_m"]) for row in rows]
    violations = sum(bool(row["violated"]) for row in rows)
    near_misses = sum(bool(row["near_miss"]) for row in rows)
    ellipsoid = sum(bool(row["ellipsoid_violated"]) for row in rows)
    low, upper = wilson_interval(violations, len(rows))
    near_low, near_upper = wilson_interval(near_misses, len(rows))
    ell_low, ell_upper = wilson_interval(ellipsoid, len(rows))
    return {"avoidance_radius_m": round(float(radius), 2), "seeds_run": len(rows),
            "violations": violations, "violation_rate": violations / len(rows), "wilson_low": low, "wilson_upper": upper,
            "first_violation_seed_index": first_violation_seed_index, "near_miss_count": near_misses,
            "near_miss_rate": near_misses / len(rows), "near_miss_wilson_low": near_low, "near_miss_wilson_upper": near_upper,
            "ellipsoid_violation_count": ellipsoid, "ellipsoid_violation_rate": ellipsoid / len(rows),
            "ellipsoid_wilson_low": ell_low, "ellipsoid_wilson_upper": ell_upper,
            "min_center_distance_m": min(distances), "median_min_center_distance_m": median_interval(distances)[0],
            "max_center_distance_m": max(distances),
            "mean_message_age_s": sum(float(row["mean_message_age_s"]) for row in rows) / len(rows)}


def _condition_job(job: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one condition using common random numbers across radii."""
    from sim.coop.separation import run_encounter
    condition = {key: job[key] for key in ("kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning", "lateral_offset_m")}
    seeds = [int(seed) for seed in job["seeds"]]
    started_cpu = time.process_time()
    baseline_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        spec_job = {**condition, "avoidance_radius_m": MIN_SEPARATION_M}
        baseline_rows.append(_result_row(run_encounter(_spec(spec_job), seed), seed, index))
    tested: list[dict[str, Any]] = []
    passing_radius: float | None = None
    failing_radius: float | None = None
    for step in range(int(round((MAX_RADIUS_M - MIN_SEPARATION_M) / BUFFER_STEP_M)) + 1):
        radius = round(MIN_SEPARATION_M + step * BUFFER_STEP_M, 2)
        if step == 0:
            rows = baseline_rows
        else:
            rows = []
            for index, seed in enumerate(seeds):
                spec_job = {**condition, "avoidance_radius_m": radius}
                row = _result_row(run_encounter(_spec(spec_job), seed), seed, index)
                rows.append(row)
                if row["violated"]:
                    break
        first = next((row["seed_index"] for row in rows if row["violated"]), None)
        aggregate = _aggregate(radius, rows, first)
        aggregate["pass"] = bool(len(rows) == len(seeds) and aggregate["violations"] == 0 and aggregate["wilson_upper"] <= 0.05)
        tested.append(aggregate)
        if aggregate["pass"]:
            passing_radius = radius
            failing_radius = round(radius - BUFFER_STEP_M, 2) if radius > MIN_SEPARATION_M else None
            break
    baseline = _aggregate(MIN_SEPARATION_M, baseline_rows,
                          next((row["seed_index"] for row in baseline_rows if row["violated"]), None))
    return {"condition": condition, "baseline": baseline, "tested": tested,
            "passing_radius_m": passing_radius, "failing_radius_m": failing_radius,
            "cpu_seconds": time.process_time() - started_cpu}


def search_radius_table(table: Iterable[dict[str, Any]], *, n_seeds: int = N_SEEDS) -> dict[str, Any]:
    """Apply the registered first-pass rule to synthetic radius aggregates."""
    ordered = sorted(table, key=lambda item: float(item["avoidance_radius_m"]))
    for row in ordered:
        violations = int(row["violations"])
        seeds_run = int(row.get("seeds_run", n_seeds))
        _, upper = wilson_interval(violations, seeds_run)
        if violations == 0 and seeds_run == n_seeds and upper <= 0.05:
            radius = round(float(row["avoidance_radius_m"]), 2)
            return {"required_buffer_m": round(radius - MIN_SEPARATION_M, 2),
                    "failing_radius_m": round(radius - BUFFER_STEP_M, 2) if radius > MIN_SEPARATION_M else None,
                    "censored": False, "max_radius_searched_m": radius,
                    "first_violation_seed_index": row.get("first_violation_seed_index")}
    first = next((row.get("first_violation_seed_index") for row in ordered if row.get("first_violation_seed_index") is not None), None)
    return {"required_buffer_m": None, "failing_radius_m": None, "censored": True, "max_radius_searched_m": MAX_RADIUS_M,
            "first_violation_seed_index": first}


def _condition_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    condition, baseline = outcome["condition"], outcome["baseline"]
    search = search_radius_table(outcome["tested"])
    from sim.coop.separation import EncounterSpec, hand_buffer_m
    hand = hand_buffer_m(EncounterSpec(**condition, avoidance_radius_m=MIN_SEPARATION_M))
    closing = 2.0 * float(condition["speed_mps"]) if condition["kind"] in ("head_on", "ring_swap") else math.sqrt(2.0) * float(condition["speed_mps"])
    return {**condition, "closing_speed_mps": closing, **search,
            "baseline_violation_count": baseline["violations"], "baseline_violation_rate": baseline["violation_rate"],
            "baseline_violation_wilson_low": baseline["wilson_low"], "baseline_violation_wilson_upper": baseline["wilson_upper"],
            "baseline_near_miss_count": baseline["near_miss_count"], "baseline_near_miss_rate": baseline["near_miss_rate"],
            "baseline_near_miss_wilson_low": baseline["near_miss_wilson_low"], "baseline_near_miss_wilson_upper": baseline["near_miss_wilson_upper"],
            "baseline_ellipsoid_violation_count": baseline["ellipsoid_violation_count"], "baseline_ellipsoid_violation_rate": baseline["ellipsoid_violation_rate"],
            "baseline_ellipsoid_wilson_low": baseline["ellipsoid_wilson_low"], "baseline_ellipsoid_wilson_upper": baseline["ellipsoid_wilson_upper"],
            "baseline_seeds_run": baseline["seeds_run"], "tested_radii": outcome["tested"],
            "hand_buffer_m": round(float(hand), 2), "n_seeds": baseline["seeds_run"]}


def _write_outputs(output: Path, outcomes: list[dict[str, Any]], *, pilot: bool, n: int, proc: int, seeds: list[int], argv: list[str], started: float) -> None:
    entries = [_condition_entry(item) for item in outcomes]
    csv_rows = []
    for item in outcomes:
        condition = item["condition"]
        for tested in item["tested"]:
            csv_rows.append({**condition, **tested, "radius_status": "pass" if tested["pass"] else "reject"})
    columns = ["kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning", "lateral_offset_m", "avoidance_radius_m", "radius_status", "seeds_run", "violations", "violation_rate", "wilson_low", "wilson_upper", "first_violation_seed_index", "near_miss_count", "near_miss_rate", "near_miss_wilson_low", "near_miss_wilson_upper", "ellipsoid_violation_count", "ellipsoid_violation_rate", "ellipsoid_wilson_low", "ellipsoid_wilson_upper", "min_center_distance_m", "median_min_center_distance_m", "max_center_distance_m", "mean_message_age_s", "pass"]
    write_csv(output / "encounters.csv", csv_rows, columns)
    envelopes = []
    for dr in (False, True):
        speeds = sorted({float(e["speed_mps"]) for e in entries if bool(e["dead_reckoning"]) == dr})
        delays = sorted({float(e["delay_s"]) for e in entries if bool(e["dead_reckoning"]) == dr})
        for speed in speeds:
            for delay in delays:
                selected = [e for e in entries if bool(e["dead_reckoning"]) == dr and float(e["speed_mps"]) == speed and float(e["delay_s"]) == delay]
                if not selected:
                    continue
                kinds = sorted({str(e["kind"]) for e in selected})
                censored = any(bool(e["censored"]) for e in selected)
                values = [float(e["required_buffer_m"]) for e in selected if e["required_buffer_m"] is not None]
                envelopes.append({"delay_s": delay, "speed_mps": speed, "dead_reckoning": dr, "required_buffer_m": None if censored else round(max(values), 2), "censored": censored, "max_radius_searched_m": MAX_RADIUS_M if censored else round(MIN_SEPARATION_M + (max(values) if values else 0.0), 2), "kinds_included": kinds})
    limits = ["Kinematic point-mass agents move with a 4 m/s^2 acceleration cap and a 2 m/s speed cap.", "Neighbor information uses an abstract directed link at 10 Hz with fixed delay and independent loss.", "Position and filtered velocity noise are simulated Gaussian draws; nothing is measured.", "The 0.35 m minimum and 5% Wilson rule are simulation settings, not safety standards.", "The ellipsoid check uses combined radii (0.12, 0.12, 0.30) m."]
    buffer = {"schema": "zephyr-d4-buffer-1", "simulated": True, "min_separation_m": MIN_SEPARATION_M, "share_rate_hz": SHARE_RATE_HZ, "control_dt_s": CONTROL_DT_S, "max_accel_mps2": 4.0, "time_horizon_s": 2.0, "criterion": "Wilson 95% upper bound of violations over 75 seeds <= 0.05", "avoidance": {"parameter": "avoidance_radius_m", "safety_margin_m": 0.0}, "entries": entries, "envelope": envelopes, "note": "All values are simulated. Baseline statistics are at avoidance_radius_m = 0.35 m.", "limits": limits}
    write_json(output / "buffer.json", buffer)
    summary_conditions = []
    keys = ("kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning", "lateral_offset_m", "n_seeds", "baseline_violation_count", "baseline_violation_rate", "baseline_violation_wilson_low", "baseline_violation_wilson_upper", "baseline_near_miss_count", "baseline_near_miss_rate", "baseline_near_miss_wilson_low", "baseline_near_miss_wilson_upper", "baseline_ellipsoid_violation_count", "baseline_ellipsoid_violation_rate", "baseline_ellipsoid_wilson_low", "baseline_ellipsoid_wilson_upper")
    for entry in entries:
        summary_conditions.append({key: entry[key] for key in keys})
    write_json(output / "summary.json", {"simulated": True, "near_miss_definition": "true minimum center distance < 0.50 m", "baseline_radius_m": MIN_SEPARATION_M, "conditions": summary_conditions})
    from tools.d_make_separation_figure import make_figures
    make_figures(output)
    write_manifest(output, schema="zephyr-d-separation-manifest-1", argv=argv, command="PYTHONPATH=. {python} tools/d_run_separation.py --output {out}" + (" --pilot" if pilot else ""), config={"pilot": pilot, "n_seeds": n, "grid": separation_grid(pilot=pilot), "avoidance_parameter": "avoidance_radius_m", "min_separation_m": MIN_SEPARATION_M, "radius_step_m": BUFFER_STEP_M, "max_radius_m": MAX_RADIUS_M}, seeds=seeds, outputs=["encounters.csv", "summary.json", "buffer.json", "separation_buffer.png", "separation_baseline.png"], limits=limits, runtime_s=time.perf_counter() - started, n_processes=proc, hashed_sources=["tools/d_run_separation.py", "tools/d_make_separation_figure.py", "sim/coop/separation.py", "sim/coop/avoidance.py", "sim/link.py", "sim/interaction.py", "sim/s6_recovery.py", "sim/coop/buffer_schema.json"])


def run(output: Path, *, seeds_n: int = N_SEEDS, pilot: bool = False, processes: int | None = None, checkpoint_dir: Path | None = None, argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    output = prepare_output(output)
    n = 2 if pilot else int(seeds_n)
    if not pilot and n != N_SEEDS:
        raise ValueError("full separation runs use the registered 75 seeds")
    seeds = seeds_for("separation", n)
    proc = min(4, int(processes or default_processes()))
    if proc < 1:
        raise ValueError("processes must be positive")
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint is not None:
        try:
            checkpoint.relative_to(output)
        except ValueError:
            checkpoint.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("--checkpoint must be outside --output")
    conditions = separation_grid(pilot=pilot)
    jobs = [{**condition, "seeds": seeds} for condition in conditions]
    outcomes = run_jobs(jobs, "tools.d_run_separation:_condition_job", processes=proc, checkpoint_dir=checkpoint, fingerprint="separation-r3c-v1", progress=False)
    if len(outcomes) != len(conditions):
        raise RuntimeError(f"expected {len(conditions)} condition outcomes, got {len(outcomes)}")
    for outcome in outcomes:
        if int(outcome["baseline"]["seeds_run"]) != n:
            raise RuntimeError("baseline did not run every seed")
    _write_outputs(output, outcomes, pilot=pilot, n=n, proc=proc, seeds=seeds, argv=argv_or_default(argv), started=started)
    if pilot:
        total_cpu = sum(float(item["cpu_seconds"]) for item in outcomes)
        full_scale = (len(separation_grid(pilot=False)) * N_SEEDS) / max(1, len(conditions) * n)
        estimated_cpu = total_cpu * full_scale
        print(f"pilot runtime: {time.perf_counter() - started:.2f} s; measured pilot CPU: {total_cpu:.2f} s; full-run estimate: {estimated_cpu:.1f} CPU s, {estimated_cpu / proc:.1f} wall s at {proc} free processes, assuming cores are free", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args_list = argv_or_default(argv)
    args = parser.parse_args(args_list)
    run(args.output, seeds_n=args.seeds, pilot=args.pilot, processes=args.processes, checkpoint_dir=args.checkpoint, argv=args_list)


if __name__ == "__main__":
    main()
