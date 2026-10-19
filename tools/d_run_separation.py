"""Run the simulated stale-neighbor separation-buffer experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
import sim.coop.separation
from tools.d_common import (
    REPO_ROOT,
    argv_or_default,
    default_processes,
    prepare_output,
    seeds_for,
    write_json,
    write_manifest,
)

MIN_SEPARATION_M = 0.35
BUFFER_STEP_M = 0.05
MAX_RADIUS_M = 1.50
N_SEEDS = 75
CONTROL_DT_S = 0.05
SHARE_RATE_HZ = 10.0
RADIUS_COUNT = int(round((MAX_RADIUS_M - MIN_SEPARATION_M) / BUFFER_STEP_M)) + 1


def separation_grid(*, pilot: bool = False) -> list[dict[str, Any]]:
    """Return the deterministic condition grid used by run, config, and pilot."""
    if pilot:
        ring = [{"kind": "ring_swap", "speed_mps": 1.0, "delay_s": 0.0,
                 "loss_probability": 0.0, "noise_std_m": 0.05, "dead_reckoning": False,
                 "lateral_offset_m": 0.0}]
        other = [{"kind": "head_on", "speed_mps": 1.0, "delay_s": 0.0,
                  "loss_probability": 0.0, "noise_std_m": 0.05, "dead_reckoning": False,
                  "lateral_offset_m": 0.0}]
        crossing = [{"kind": "crossing", "speed_mps": 1.0, "delay_s": 0.0,
                     "loss_probability": 0.0, "noise_std_m": 0.05, "dead_reckoning": False,
                     "lateral_offset_m": 0.0}]
        return ring + other + crossing
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
        keys = ("speed_mps", "delay_s", "loss_probability", "noise_std_m",
                "dead_reckoning", "lateral_offset_m")
        spec = grouped.setdefault(cell["kind"], {key: [] for key in keys})
        for key in spec:
            if cell[key] not in spec[key]:
                spec[key].append(cell[key])
    return grouped


def _manifest_config(pilot: bool, n: int) -> dict[str, Any]:
    """Return every simulation setting that can affect a separation result."""
    return {
        "pilot": pilot,
        "n_seeds": n,
        "grid": separation_grid(pilot=pilot),
        "avoidance_parameter": "avoidance_radius_m",
        "min_separation_m": MIN_SEPARATION_M,
        "radius_step_m": BUFFER_STEP_M,
        "max_radius_m": MAX_RADIUS_M,
        "control_dt_s": CONTROL_DT_S,
        "share_rate_hz": SHARE_RATE_HZ,
        "max_accel_mps2": 4.0,
        "max_speed_mps": 2.0,
        "time_horizon_s": 2.0,
        "near_miss_threshold_m": 0.50,
        "ellipsoid_radii_m": [0.12, 0.12, 0.30],
        "ellipsoid_solver": "exact_segment_minimum",
        "criterion": "Wilson 95% upper bound of violations over 75 seeds <= 0.05",
    }


def _repo_module_sources() -> list[str]:
    """List repository Python modules loaded by the completed in-process job."""
    paths: set[str] = {
        "tools/d_run_separation.py",
        "tools/d_make_separation_figure.py",
        "tools/horizon_mpl.py",
        "sim/link.py",
    }
    root = REPO_ROOT.resolve()
    for module in sys.modules.values():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            relative = Path(filename).resolve().relative_to(root)
        except ValueError:
            continue
        if relative.suffix == ".py":
            paths.add(relative.as_posix())
    return sorted(paths)


def _checkpoint_fingerprint(jobs: list[dict[str, Any]]) -> str:
    """Hash runner and cooperative simulation sources together with job config."""
    digest = hashlib.sha256()
    for relative in _repo_module_sources():
        path = REPO_ROOT / relative
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _check_checkpoint_fingerprint(checkpoint: Path | None, fingerprint: str) -> None:
    """Reject a checkpoint directory created for different runner inputs."""
    if checkpoint is None or not checkpoint.is_dir():
        return
    for path in checkpoint.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint file: {path}") from exc
        if record.get("fingerprint") != fingerprint:
            raise ValueError(f"checkpoint fingerprint mismatch: {path}")


def _spec(job: dict[str, Any]):
    from sim.coop.separation import EncounterSpec
    return EncounterSpec(
        kind=str(job["kind"]),
        speed_mps=float(job["speed_mps"]),
        delay_s=float(job["delay_s"]),
        loss_probability=float(job["loss_probability"]),
        noise_std_m=float(job["noise_std_m"]),
        dead_reckoning=bool(job["dead_reckoning"]),
        lateral_offset_m=float(job.get("lateral_offset_m", 0.1)),
        avoidance_radius_m=float(job["avoidance_radius_m"]),
        share_rate_hz=SHARE_RATE_HZ,
        control_dt_s=CONTROL_DT_S,
    )


def _result_row(result: dict[str, Any], seed: int, seed_index: int) -> dict[str, Any]:
    return {"seed": int(seed), "seed_index": int(seed_index),
            "min_center_distance_m": float(result["min_center_distance_m"]),
            "violated": bool(result["violated"]), "near_miss": bool(result["near_miss"]),
            "min_ellipsoid_separation": float(result["min_ellipsoid_separation"]),
            "ellipsoid_violated": bool(result["ellipsoid_violated"]),
            "penetration_m": float(result["penetration_m"]), "reached_goals": bool(result["reached_goals"]),
            "mean_message_age_s": float(result["mean_message_age_s"])}


def _aggregate(
    radius: float,
    rows: list[dict[str, Any]],
    first_violation_seed_index: int | None,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("condition produced no rows")
    distances = [float(row["min_center_distance_m"]) for row in rows]
    violations = sum(bool(row["violated"]) for row in rows)
    near_misses = sum(bool(row["near_miss"]) for row in rows)
    ellipsoid = sum(bool(row["ellipsoid_violated"]) for row in rows)
    low, upper = wilson_interval(violations, len(rows))
    near_low, near_upper = wilson_interval(near_misses, len(rows))
    ell_low, ell_upper = wilson_interval(ellipsoid, len(rows))
    return {
        "avoidance_radius_m": round(float(radius), 2),
        "seeds_run": len(rows),
        "violations": violations,
        "violation_rate": violations / len(rows),
        "wilson_low": low,
        "wilson_upper": upper,
        "first_violation_seed_index": first_violation_seed_index,
        "near_miss_count": near_misses,
        "near_miss_rate": near_misses / len(rows),
        "near_miss_wilson_low": near_low,
        "near_miss_wilson_upper": near_upper,
        "ellipsoid_violation_count": ellipsoid,
        "ellipsoid_violation_rate": ellipsoid / len(rows),
        "ellipsoid_wilson_low": ell_low,
        "ellipsoid_wilson_upper": ell_upper,
        "min_center_distance_m": min(distances),
        "median_min_center_distance_m": median_interval(distances)[0],
        "max_center_distance_m": max(distances),
        "mean_message_age_s": sum(
            float(row["mean_message_age_s"]) for row in rows
        ) / len(rows),
    }


def _condition_job(job: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one condition using common random numbers across radii."""
    from sim.coop.separation import run_encounter
    condition_keys = (
        "kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m",
        "dead_reckoning", "lateral_offset_m",
    )
    condition = {key: job[key] for key in condition_keys}
    seeds = [int(seed) for seed in job["seeds"]]
    started_cpu = time.process_time()
    baseline_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        spec_job = {**condition, "avoidance_radius_m": MIN_SEPARATION_M}
        baseline_rows.append(_result_row(run_encounter(_spec(spec_job), seed), seed, index))
    tested: list[dict[str, Any]] = []
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
        aggregate["pass"] = bool(
            len(rows) == len(seeds)
            and aggregate["violations"] == 0
            and aggregate["wilson_upper"] <= 0.05
        )
        tested.append(aggregate)
        if aggregate["pass"]:
            break
    baseline = _aggregate(MIN_SEPARATION_M, baseline_rows,
                          next((row["seed_index"] for row in baseline_rows if row["violated"]), None))
    return {
        "condition": condition,
        "baseline": baseline,
        "tested": tested,
        "cpu_seconds": time.process_time() - started_cpu,
    }


def search_radius_table(table: Iterable[dict[str, Any]], *, n_seeds: int = N_SEEDS) -> dict[str, Any]:
    """Apply the registered first-pass rule to synthetic radius aggregates."""
    ordered = sorted(table, key=lambda item: float(item["avoidance_radius_m"]))
    first_violation = next(
        (row.get("first_violation_seed_index") for row in ordered
         if row.get("first_violation_seed_index") is not None),
        None,
    )
    for row in ordered:
        violations = int(row["violations"])
        seeds_run = int(row.get("seeds_run", n_seeds))
        _, upper = wilson_interval(violations, seeds_run)
        if violations == 0 and seeds_run == n_seeds and upper <= 0.05:
            radius = round(float(row["avoidance_radius_m"]), 2)
            return {"required_buffer_m": round(radius - MIN_SEPARATION_M, 2),
                    "failing_radius_m": (
                        round(radius - BUFFER_STEP_M, 2)
                        if radius > MIN_SEPARATION_M else None
                    ),
                    "censored": False, "max_radius_searched_m": radius,
                    "first_violation_seed_index": first_violation}
    return {
        "required_buffer_m": None,
        "failing_radius_m": None,
        "censored": True,
        "max_radius_searched_m": MAX_RADIUS_M,
        "first_violation_seed_index": first_violation,
    }


def _pilot_cpu_rates(outcomes: Iterable[dict[str, Any]], *, pilot_seeds: int) -> dict[str, float]:
    rows = list(outcomes)
    if not rows or pilot_seeds < 1:
        raise ValueError("pilot outcomes and seed count are required")
    rates: dict[str, list[float]] = defaultdict(list)
    for outcome in rows:
        radii = len(outcome["tested"])
        if radii < 1:
            raise ValueError("pilot condition has no tested radii")
        condition = outcome["condition"]
        rates[str(condition["kind"])].append(
            float(outcome["cpu_seconds"]) / (pilot_seeds * radii)
        )
    return {kind: sum(values) / len(values) for kind, values in rates.items()}


def estimate_full_run_cpu(outcomes: Iterable[dict[str, Any]], *, pilot_seeds: int,
                          full_grid: Iterable[dict[str, Any]] | None = None) -> tuple[float, float]:
    """Estimate full-grid CPU from measured per-kind condition CPU.

    The worst estimate assumes every condition tests all 24 radii and all 75
    seeds. The best estimate assumes the baseline radius passes immediately,
    so each condition runs one radius over all 75 seeds. The per-kind rate is
    measured at 1 m/s only and applied to all speeds. Pair conditions are
    likely about 20% low because speed-specific costs are omitted.
    """
    rows = list(outcomes)
    full = list(separation_grid(pilot=False) if full_grid is None else full_grid)
    by_kind = _pilot_cpu_rates(rows, pilot_seeds=pilot_seeds)
    if any(kind not in by_kind for kind in {str(cell["kind"]) for cell in full}):
        raise ValueError("pilot must include each encounter kind")
    counts = Counter(str(cell["kind"]) for cell in full)
    best = 0.0
    worst = 0.0
    counts_by_kind = {kind: sum(1 for cell in full if cell["kind"] == kind) for kind in counts}
    for kind in counts:
        per_encounter = by_kind[kind]
        best += per_encounter * counts_by_kind[kind] * N_SEEDS
        worst += per_encounter * counts_by_kind[kind] * N_SEEDS * RADIUS_COUNT
    return best, worst


def _condition_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    condition, baseline = outcome["condition"], outcome["baseline"]
    search = search_radius_table(outcome["tested"])
    from sim.coop.separation import EncounterSpec, hand_buffer_m
    hand = hand_buffer_m(EncounterSpec(**condition, avoidance_radius_m=MIN_SEPARATION_M))
    closing = (
        2.0 * float(condition["speed_mps"])
        if condition["kind"] in ("head_on", "ring_swap")
        else math.sqrt(2.0) * float(condition["speed_mps"])
    )
    return {
        **condition,
        "closing_speed_mps": closing,
        **search,
        "baseline_violation_count": baseline["violations"],
        "baseline_violation_rate": baseline["violation_rate"],
        "baseline_violation_wilson_low": baseline["wilson_low"],
        "baseline_violation_wilson_upper": baseline["wilson_upper"],
        "baseline_near_miss_count": baseline["near_miss_count"],
        "baseline_near_miss_rate": baseline["near_miss_rate"],
        "baseline_near_miss_wilson_low": baseline["near_miss_wilson_low"],
        "baseline_near_miss_wilson_upper": baseline["near_miss_wilson_upper"],
        "baseline_ellipsoid_violation_count": baseline["ellipsoid_violation_count"],
        "baseline_ellipsoid_violation_rate": baseline["ellipsoid_violation_rate"],
        "baseline_ellipsoid_wilson_low": baseline["ellipsoid_wilson_low"],
        "baseline_ellipsoid_wilson_upper": baseline["ellipsoid_wilson_upper"],
        "baseline_seeds_run": baseline["seeds_run"],
        "tested_radii": outcome["tested"],
        "hand_buffer_m": round(float(hand), 2),
        "n_seeds": baseline["seeds_run"],
    }


def _write_outputs(
    output: Path,
    outcomes: list[dict[str, Any]],
    *,
    pilot: bool,
    n: int,
    proc: int,
    seeds: list[int],
    argv: list[str],
    started: float,
) -> None:
    entries = [_condition_entry(item) for item in outcomes]
    csv_rows = []
    for item in outcomes:
        condition = item["condition"]
        for tested in item["tested"]:
            status = "pass" if tested["pass"] else ("not passed (pilot)" if pilot else "reject")
            csv_rows.append({**condition, **tested, "radius_status": status})
    columns = [
        "kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m",
        "dead_reckoning", "lateral_offset_m", "avoidance_radius_m",
        "radius_status", "seeds_run", "violations", "violation_rate",
        "wilson_low", "wilson_upper", "first_violation_seed_index",
        "near_miss_count", "near_miss_rate", "near_miss_wilson_low",
        "near_miss_wilson_upper", "ellipsoid_violation_count",
        "ellipsoid_violation_rate", "ellipsoid_wilson_low",
        "ellipsoid_wilson_upper", "min_center_distance_m",
        "median_min_center_distance_m", "max_center_distance_m",
        "mean_message_age_s", "pass",
    ]
    write_csv(output / "encounters.csv", csv_rows, columns)
    envelopes = []
    for dr in (False, True):
        speeds = sorted({float(e["speed_mps"]) for e in entries if bool(e["dead_reckoning"]) == dr})
        delays = sorted({float(e["delay_s"]) for e in entries if bool(e["dead_reckoning"]) == dr})
        for speed in speeds:
            for delay in delays:
                selected = [
                    e for e in entries
                    if bool(e["dead_reckoning"]) == dr
                    and float(e["speed_mps"]) == speed
                    and float(e["delay_s"]) == delay
                ]
                if not selected:
                    continue
                kinds = sorted({str(e["kind"]) for e in selected})
                censored = any(bool(e["censored"]) for e in selected)
                values = [
                    float(e["required_buffer_m"])
                    for e in selected
                    if e["required_buffer_m"] is not None
                ]
                envelope_value = None if censored else round(max(values), 2)
                max_searched = (
                    MAX_RADIUS_M
                    if censored
                    else round(MIN_SEPARATION_M + (max(values) if values else 0.0), 2)
                )
                envelopes.append({
                    "delay_s": delay,
                    "speed_mps": speed,
                    "dead_reckoning": dr,
                    "required_buffer_m": envelope_value,
                    "censored": censored,
                    "max_radius_searched_m": max_searched,
                    "kinds_included": kinds,
                })
    limits = [
        "Kinematic point-mass agents move with a 4 m/s^2 acceleration cap and a 2 m/s speed cap.",
        "Neighbor information uses an abstract directed link at 10 Hz with fixed delay and independent loss.",
        "Position and filtered velocity noise are simulated Gaussian draws; nothing is measured.",
        "The 0.35 m minimum and 5% Wilson rule are simulation settings, not safety standards.",
        "Ellipsoid limits are per-vehicle radii (0.12, 0.12, 0.30) m and "
        "combined radii (0.24, 0.24, 0.60) m.",
        "Entry first_violation_seed_index is at the lowest radius with a violation, normally 0.35 m.",
    ]
    buffer = {
        "schema": "zephyr-d4-buffer-1",
        "simulated": True,
        "min_separation_m": MIN_SEPARATION_M,
        "share_rate_hz": SHARE_RATE_HZ,
        "control_dt_s": CONTROL_DT_S,
        "max_accel_mps2": 4.0,
        "time_horizon_s": 2.0,
        "criterion": "Wilson 95% upper bound of violations over 75 seeds <= 0.05",
        "avoidance": {"parameter": "avoidance_radius_m", "safety_margin_m": 0.0},
        "entries": entries,
        "envelope": envelopes,
        "note": (
            "All values are simulated. Baseline statistics are at "
            "avoidance_radius_m = 0.35 m. Entry first_violation_seed_index "
            "is the lowest radius with a violation."
        ),
        "limits": limits,
    }
    write_json(output / "buffer.json", buffer)
    summary_conditions = []
    keys = (
        "kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m",
        "dead_reckoning", "lateral_offset_m", "n_seeds",
        "baseline_violation_count", "baseline_violation_rate",
        "baseline_violation_wilson_low", "baseline_violation_wilson_upper",
        "baseline_near_miss_count", "baseline_near_miss_rate",
        "baseline_near_miss_wilson_low", "baseline_near_miss_wilson_upper",
        "baseline_ellipsoid_violation_count", "baseline_ellipsoid_violation_rate",
        "baseline_ellipsoid_wilson_low", "baseline_ellipsoid_wilson_upper",
    )
    for entry in entries:
        summary_conditions.append({key: entry[key] for key in keys})
    write_json(output / "summary.json", {
        "simulated": True,
        "near_miss_definition": "true minimum center distance < 0.50 m",
        "baseline_radius_m": MIN_SEPARATION_M,
        "conditions": summary_conditions,
    })
    from tools.d_make_separation_figure import make_figures
    make_figures(output)
    write_manifest(
        output,
        schema="zephyr-d-separation-manifest-1",
        argv=argv,
        command="PYTHONPATH=. {python} tools/d_run_separation.py --output {out}"
        + (" --pilot" if pilot else ""),
        config=_manifest_config(pilot, n),
        seeds=seeds,
        outputs=[
            "encounters.csv", "summary.json", "buffer.json",
            "separation_buffer.png", "separation_baseline.png",
        ],
        limits=limits,
        runtime_s=time.perf_counter() - started,
        n_processes=proc,
        hashed_sources=_repo_module_sources(),
    )


def run(
    output: Path,
    *,
    seeds_n: int = N_SEEDS,
    pilot: bool = False,
    processes: int | None = None,
    checkpoint_dir: Path | None = None,
    argv: list[str] | None = None,
) -> None:
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
    fingerprint = _checkpoint_fingerprint(jobs)
    _check_checkpoint_fingerprint(checkpoint, fingerprint)
    outcomes = run_jobs(
        jobs,
        "tools.d_run_separation:_condition_job",
        processes=proc,
        checkpoint_dir=checkpoint,
        fingerprint=fingerprint,
        progress=False,
    )
    if len(outcomes) != len(conditions):
        raise RuntimeError(f"expected {len(conditions)} condition outcomes, got {len(outcomes)}")
    for outcome in outcomes:
        if int(outcome["baseline"]["seeds_run"]) != n:
            raise RuntimeError("baseline did not run every seed")
    _write_outputs(
        output,
        outcomes,
        pilot=pilot,
        n=n,
        proc=proc,
        seeds=seeds,
        argv=argv_or_default(argv),
        started=started,
    )
    if pilot:
        total_cpu = sum(float(item["cpu_seconds"]) for item in outcomes)
        best_cpu, worst_cpu = estimate_full_run_cpu(outcomes, pilot_seeds=n)
        expected_cpu = best_cpu * 3.0
        rates = _pilot_cpu_rates(outcomes, pilot_seeds=n)
        rate_text = ", ".join(f"{kind}={rates[kind]:.10f}" for kind in sorted(rates))
        print(
            f"pilot runtime: {time.perf_counter() - started:.2f} s; measured pilot CPU: {total_cpu:.2f} s; "
            f"full-run best-case estimate: {best_cpu:.1f} CPU s, {best_cpu / 4.0:.1f} wall s; "
            f"expected estimate: {expected_cpu:.1f} CPU s, {expected_cpu / 4.0:.1f} wall s (about 3x best; "
            f"sampled full conditions ran about 3x the one-radius cost); "
            f"full-run worst-case estimate: {worst_cpu:.1f} CPU s, {worst_cpu / 4.0:.1f} wall s "
            f"at 4 free processes, assuming cores are free; formula: "
            f"sum_k(mean_cpu_k / (pilot_seeds * tested_radii_k)) * "
            f"full_conditions_k * 75 * [1 or {RADIUS_COUNT} radii]; "
            f"measured CPU s per encounter by kind: {rate_text}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args_list = argv_or_default(argv)
    args = parser.parse_args(args_list)
    run(
        args.output,
        seeds_n=args.seeds,
        pilot=args.pilot,
        processes=args.processes,
        checkpoint_dir=args.checkpoint,
        argv=args_list,
    )


if __name__ == "__main__":
    main()
