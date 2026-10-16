"""Run the simulated stale-neighbor separation-buffer experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import (argv_or_default, default_processes, prepare_output,
                            MOVING_AGENT_LIMITS, seeds_for, write_json, write_manifest)

KINDS = ("head_on", "crossing", "ring_swap")
SPEEDS = (0.5, 1.0, 2.0)
DELAYS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
LOSSES = (0.0, 0.3)
NOISES = (0.0, 0.05)
DRS = (False, True)
MIN_SEPARATION = 0.35


def _encounter_job(job: dict) -> dict:
    """Worker wrapper kept at module scope for process-pool execution."""
    from sim.coop.separation import EncounterSpec, run_encounter
    spec = EncounterSpec(
        kind=str(job["kind"]), speed_mps=float(job["speed_mps"]), delay_s=float(job["delay_s"]),
        loss_probability=float(job["loss_probability"]), noise_std_m=float(job["noise_std_m"]),
        dead_reckoning=bool(job["dead_reckoning"]), avoidance_radius_m=float(job["avoidance_radius_m"]),
    )
    result = dict(run_encounter(spec, int(job["seed"])))
    # Keep checkpoint and CSV worker results small. The detailed trace is for
    # diagnostics in direct simulation calls and is not a runner output.
    result.pop("spec", None)
    result.pop("perceived_trace", None)
    result.pop("duration_s", None)
    flags = result.pop("flags", {})
    for key in ("orca", "infeasible", "emergency", "speed_cap"):
        result[f"flag_{key}"] = int(flags.get(key, result.get(f"flag_{key}", 0)))
    result.update({
        "kind": str(job["kind"]), "speed_mps": float(job["speed_mps"]),
        "delay_s": float(job["delay_s"]), "loss_probability": float(job["loss_probability"]),
        "noise_std_m": float(job["noise_std_m"]), "dead_reckoning": bool(job["dead_reckoning"]),
        "avoidance_radius_m": float(job["avoidance_radius_m"]), "seed": int(job["seed"]),
    })
    return result


def _grid(pilot: bool) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[bool, ...]]:
    if pilot:
        return ("head_on", "ring_swap"), (1.0,), (0.0, 0.2), (0.0,), (0.05,), DRS
    return KINDS, SPEEDS, DELAYS, LOSSES, NOISES, DRS


def _conditions(pilot: bool) -> list[tuple[str, float, float, float, float, bool]]:
    if pilot:
        kinds, speeds, delays, losses, noises, drs = _grid(True)
        return [(kind, speed, delay, loss, noise, dr)
                for kind in kinds for speed in speeds for delay in delays
                for loss in losses for noise in noises for dr in drs]
    cells: list[tuple[str, float, float, float, float, bool]] = []
    for kind in ("head_on", "crossing"):
        for speed in SPEEDS:
            for delay in DELAYS:
                for loss in LOSSES:
                    for noise in NOISES:
                        for dr in DRS:
                            cells.append((kind, speed, delay, loss, noise, dr))
    for speed in (1.0,):
        for delay in (0.0, 0.1, 0.2, 0.5):
            for loss in LOSSES:
                for noise in (0.05,):
                    for dr in DRS:
                        cells.append(("ring_swap", speed, delay, loss, noise, dr))
    return cells


def _grid_config(pilot: bool) -> dict:
    if pilot:
        return {
            "head_on": {"speed_mps": [1.0], "delay_s": [0.0, 0.2], "loss_probability": [0.0],
                        "noise_std_m": [0.05], "dead_reckoning": [False, True]},
            "ring_swap": {"speed_mps": [1.0], "delay_s": [0.0, 0.2], "loss_probability": [0.0],
                          "noise_std_m": [0.05], "dead_reckoning": [False, True]},
        }
    return {
        "head_on": {"speed_mps": list(SPEEDS), "delay_s": list(DELAYS), "loss_probability": list(LOSSES),
                     "noise_std_m": list(NOISES), "dead_reckoning": list(DRS)},
        "crossing": {"speed_mps": list(SPEEDS), "delay_s": list(DELAYS), "loss_probability": list(LOSSES),
                     "noise_std_m": list(NOISES), "dead_reckoning": list(DRS)},
        "ring_swap": {"speed_mps": [1.0], "delay_s": [0.0, 0.1, 0.2, 0.5],
                       "loss_probability": list(LOSSES), "noise_std_m": [0.05], "dead_reckoning": list(DRS)},
    }


def _condition_jobs(kind: str, speed: float, delay: float, loss: float, noise: float, dr: bool,
                    seeds: list[int], radius: float) -> list[dict]:
    return [{"kind": kind, "speed_mps": speed, "delay_s": delay, "loss_probability": loss,
             "noise_std_m": noise, "dead_reckoning": dr, "avoidance_radius_m": radius, "seed": seed}
            for seed in seeds]


def _ceil_centimetre(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(math.ceil(value * 100.0 - 1.0e-12) / 100.0)


def _print_timing_estimate(seeds: list[int], processes: int) -> None:
    """Time representative three-seed cells and print a reproducible estimate."""
    from sim.estimation.sweep import run_jobs as _run_jobs

    samples = [
        ("ring_swap", 1.0, 0.5, 0.3, 0.05, False),
        ("head_on", 1.0, 0.0, 0.0, 0.0, False),
    ]
    sample_seeds = list(seeds[:3])
    jobs = []
    for kind, speed, delay, loss, noise, dr in samples:
        jobs.extend(_condition_jobs(kind, speed, delay, loss, noise, dr, sample_seeds, MIN_SEPARATION))
    cpu_start = time.process_time()
    _run_jobs(jobs, "tools.d_run_separation:_encounter_job", processes=1, progress=False)
    cpu_s = time.process_time() - cpu_start
    wall_start = time.perf_counter()
    _run_jobs(jobs, "tools.d_run_separation:_encounter_job", processes=4, progress=False)
    wall_s = time.perf_counter() - wall_start
    # Every pass with zero failures has a Wilson upper bound above 0.05 at
    # n=50, so the conservative estimate includes all ten buffer attempts and
    # the one missing ring perfect-information cell.
    full_jobs = len(_conditions(False)) * 50 * 11 + 50
    scale = full_jobs / len(jobs)
    print(
        "timing samples: 3 seeds ring_swap(delay=0.5, loss=0.3) and "
        "3 seeds head_on; "
        f"sample CPU (serial) {cpu_s:.2f} s, sample wall (4 processes) {wall_s:.2f} s; "
        f"estimated full run at 4 processes (including up to 10 buffer attempts): CPU {cpu_s * scale:.1f} s, "
        f"wall {wall_s * scale:.1f} s",
        flush=True,
    )


def _row_columns() -> list[str]:
    return ["kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning",
            "pass", "attempt", "seed", "avoidance_radius_m", "min_center_distance_m", "violated",
            "near_miss", "min_ellipsoid_separation", "ellipsoid_violated", "penetration_m",
            "reached_goals", "flag_orca", "flag_infeasible", "flag_emergency", "flag_speed_cap",
            "mean_message_age_s"]


def run(output: Path, *, seeds_n: int = 50, pilot: bool = False, processes: int | None = None,
        checkpoint_dir: Path | None = None, argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    output = prepare_output(output)
    n = 5 if pilot else int(seeds_n)
    if n < 1:
        raise ValueError("seeds must be positive")
    seeds = seeds_for("separation", n)
    if not pilot:
        _print_timing_estimate(seeds, 4)
    kinds, speeds, delays, losses, noises, drs = _grid(pilot)
    proc = int(processes or default_processes())
    checkpoint = None
    if checkpoint_dir is not None:
        requested = Path(checkpoint_dir)
        checkpoint = requested
        try:
            checkpoint.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError("--checkpoint must be outside --output")
        checkpoint.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    condition_keys = _conditions(pilot)
    for kind, speed, delay, loss, noise, dr in condition_keys:
        jobs.extend(_condition_jobs(kind, speed, delay, loss, noise, dr, seeds, MIN_SEPARATION))
    if not jobs:
        raise RuntimeError("separation grid produced no jobs")
    fingerprint_material = json.dumps(_grid_config(pilot), sort_keys=True).encode("utf-8")
    for source in (Path(__file__), Path(__file__).parents[1] / "sim/coop/separation.py",
                   Path(__file__).parents[1] / "sim/coop/avoidance.py"):
        fingerprint_material += source.read_bytes()
    fingerprint = "separation-v3-" + hashlib.sha256(fingerprint_material).hexdigest()[:16]
    baseline_raw = run_jobs(jobs, "tools.d_run_separation:_encounter_job", processes=proc,
                            checkpoint_dir=checkpoint, fingerprint=fingerprint, progress=False)
    expected_baseline = len(condition_keys) * n
    if len(baseline_raw) != expected_baseline:
        raise RuntimeError(f"baseline returned {len(baseline_raw)} rows, expected {expected_baseline}")
    baseline_rows = []
    for result in baseline_raw:
        row = dict(result); row.update({"pass": "baseline", "attempt": 0}); baseline_rows.append(row)

    # Keep the perfect-information check complete even in pilot mode. Pilot
    # cells use nonzero noise by design, so missing zero-noise cells are run as
    # a small, separate diagnostic grid rather than represented by empty rows.
    perfect_jobs = []
    perfect_keys = sorted({(kind, speed) for kind, speed, *_ in condition_keys})
    for kind, speed in perfect_keys:
        existing = [r for r in baseline_rows if r["kind"] == kind and r["speed_mps"] == speed and r["delay_s"] == 0.0 and r["loss_probability"] == 0.0 and r["noise_std_m"] == 0.0 and not r["dead_reckoning"]]
        if len(existing) != n:
            perfect_jobs.extend(_condition_jobs(kind, speed, 0.0, 0.0, 0.0, False, seeds, MIN_SEPARATION))
    perfect_extra = run_jobs(perfect_jobs, "tools.d_run_separation:_encounter_job", processes=proc,
                             checkpoint_dir=checkpoint, fingerprint=fingerprint + "-perfect", progress=False) if perfect_jobs else []
    perfect_rows = list(baseline_rows)
    for result in perfect_extra:
        row = dict(result); row.update({"pass": "perfect_info", "attempt": 0}); perfect_rows.append(row)

    entries = []
    all_rows = list(baseline_rows)
    condition_data: dict[tuple, dict] = {}
    for key in condition_keys:
        kind, speed, delay, loss, noise, dr = key
        selected = [r for r in baseline_rows if (r["kind"], r["speed_mps"], r["delay_s"], r["loss_probability"], r["noise_std_m"], r["dead_reckoning"]) == key]
        if len(selected) != n:
            raise RuntimeError(f"condition {key} returned {len(selected)} rows, expected {n}")
        penetrations = np.asarray([float(r["penetration_m"]) for r in selected], dtype=float)
        buffer_m = _ceil_centimetre(float(np.percentile(penetrations, 95, method="linear"))) if np.any(penetrations > 0.0) else 0.0
        baseline_fail = sum(bool(r["violated"]) for r in selected)
        from sim.coop.separation import EncounterSpec, hand_buffer_m
        hand_spec = EncounterSpec(kind=kind, speed_mps=speed, delay_s=delay, loss_probability=loss,
                                  noise_std_m=noise, dead_reckoning=dr, avoidance_radius_m=MIN_SEPARATION)
        closing = 2.0 * speed if kind in ("head_on", "ring_swap") else math.sqrt(2.0) * speed
        hand = hand_buffer_m(hand_spec)
        condition_data[key] = {"buffer_m": buffer_m, "attempts": [], "final_rows": [],
                               "baseline_fail": baseline_fail, "closing": closing, "hand": hand}

    active = set(condition_keys)
    for attempt in range(1, 11):
        if not active:
            break
        attempt_jobs = []
        for key in condition_keys:
            if key not in active:
                continue
            kind, speed, delay, loss, noise, dr = key
            attempt_jobs.extend(_condition_jobs(kind, speed, delay, loss, noise, dr, seeds,
                                                MIN_SEPARATION + condition_data[key]["buffer_m"]))
        raw = run_jobs(attempt_jobs, "tools.d_run_separation:_encounter_job", processes=proc,
                        checkpoint_dir=checkpoint, fingerprint=fingerprint + f"-attempt-{attempt}", progress=False)
        expected_attempt = len(active) * n
        if len(raw) != expected_attempt:
            raise RuntimeError(f"buffer attempt {attempt} returned {len(raw)} rows, expected {expected_attempt}")
        by_key: dict[tuple, list[dict]] = {key: [] for key in active}
        for result in raw:
            key = (result["kind"], result["speed_mps"], result["delay_s"], result["loss_probability"], result["noise_std_m"], result["dead_reckoning"])
            row = dict(result); row.update({"pass": "buffer", "attempt": attempt})
            by_key[key].append(row)
        for key in list(active):
            rows = by_key[key]
            if len(rows) != n:
                raise RuntimeError(f"condition {key} attempt {attempt} returned {len(rows)} rows, expected {n}")
            all_rows.extend(rows)
            data = condition_data[key]
            failures = sum(bool(r["violated"]) for r in rows)
            rate = failures / n
            low, upper = wilson_interval(failures, n)
            data["attempts"].append({"attempt": attempt, "buffer_m": data["buffer_m"], "violations": failures,
                                     "violation_rate": rate, "wilson_low": low, "wilson_upper": upper})
            data["final_rows"] = rows
            if upper <= 0.05:
                active.remove(key)
            else:
                data["buffer_m"] += 0.05

    for key in condition_keys:
        kind, speed, delay, loss, noise, dr = key
        data = condition_data[key]
        final = data["attempts"][-1]
        entries.append({"kind": kind, "speed_mps": speed, "closing_speed_mps": data["closing"], "delay_s": delay,
                        "loss_probability": loss, "noise_std_m": noise, "dead_reckoning": dr,
                        "required_buffer_m": data["buffer_m"], "baseline_violation_rate": data["baseline_fail"] / n,
                        "verified_violation_rate": final["violation_rate"], "verified_wilson_upper": final["wilson_upper"],
                        "attempts": len(data["attempts"]), "attempt_history": data["attempts"],
                        "hand_buffer_m": data["hand"], "n_seeds": n})

    columns = _row_columns()
    csv_rows = []
    for row in all_rows:
        copy = dict(row)
        for field in ("violated", "near_miss", "ellipsoid_violated", "reached_goals"):
            copy[field] = int(bool(copy.get(field)))
        csv_rows.append(copy)
    write_csv(output / "encounters.csv", csv_rows, columns)
    envelopes = []
    for delay in sorted({key[2] for key in condition_keys}):
        for speed in sorted({key[1] for key in condition_keys}):
            for dr in drs:
                vals = [e["required_buffer_m"] for e in entries if e["delay_s"] == delay and e["speed_mps"] == speed and e["dead_reckoning"] == dr]
                if vals:
                    envelopes.append({"delay_s": delay, "speed_mps": speed, "dead_reckoning": dr, "buffer_m": max(vals)})
    limits = list(MOVING_AGENT_LIMITS) + [
        "Agents change velocity toward the avoidance command with a 4 m/s^2 acceleration cap; ORCA itself assumes an instantaneous velocity change.",
        "Neighbor information travels over abstract directed links at 10 Hz with independent per-message loss.",
        "Position noise is Gaussian per axis. Velocity noise has the same numerical standard deviation in m/s and stands for a filtered velocity estimate.",
        "The minimum center distance of 0.35 m is the S7 default and is not a measured drone size.",
        "Ellipsoid radii (0.12, 0.12, 0.30) m are literature-derived Crazyflie placeholders.",
        "Full runs retain all 50 seeds for loss 0 and noise 0 even though every seed is deterministic.",
    ]
    buffer = {"schema": "zephyr-d4-buffer-1", "simulated": True, "min_separation_m": MIN_SEPARATION,
              "share_rate_hz": 10, "control_dt_s": 0.05, "max_accel_mps2": 4.0, "time_horizon_s": 2.0,
              "criterion": "violation rate Wilson 95 % upper bound <= 0.05 over n seeds",
              "avoidance": {"safety_margin_m": 0.0},
              "entries": entries, "envelope": envelopes, "limits": limits}
    write_json(output / "buffer.json", buffer)

    perfect = []
    for kind, speed in perfect_keys:
        selected = [r for r in perfect_rows if r["kind"] == kind and r["speed_mps"] == speed and r["delay_s"] == 0.0 and r["loss_probability"] == 0.0 and r["noise_std_m"] == 0.0 and not r["dead_reckoning"]]
        if len(selected) != n:
            raise RuntimeError(f"perfect-information condition {(kind, speed)} returned {len(selected)} rows, expected {n}")
        violations = sum(bool(r["violated"]) for r in selected)
        perfect.append({"kind": kind, "speed_mps": speed, "avoidance_radius_m": MIN_SEPARATION,
                        "violations": violations, "n_seeds": len(selected), "zero_violations": violations == 0})
    write_json(output / "perfect_info.json", {"simulated": True, "rows": perfect})

    summary = []
    for key in condition_keys:
        selected = [r for r in baseline_rows if (r["kind"], r["speed_mps"], r["delay_s"], r["loss_probability"], r["noise_std_m"], r["dead_reckoning"]) == key]
        med, low, high = median_interval([float(r["min_center_distance_m"]) for r in selected])
        summary.append({"kind": key[0], "speed_mps": key[1], "delay_s": key[2], "loss_probability": key[3], "noise_std_m": key[4], "dead_reckoning": key[5], "n_seeds": n, "median_min_distance_m": med, "median_interval_low_m": low, "median_interval_high_m": high})
    write_json(output / "summary.json", summary)

    from tools.d_make_separation_figure import make_figures
    make_figures(output)
    outputs = ["encounters.csv", "buffer.json", "perfect_info.json", "summary.json", "separation_buffer.png", "separation_baseline.png"]
    command = "PYTHONPATH=. {python} tools/d_run_separation.py --output {out}"
    if pilot: command += " --pilot"
    elif seeds_n != 50: command += f" --seeds {seeds_n}"
    if processes is not None: command += f" --processes {processes}"
    if checkpoint_dir is not None: command += f" --checkpoint {checkpoint_dir}"
    config = {"pilot": pilot, "n_seeds": n, "grid": _grid_config(pilot),
              "min_separation_m": MIN_SEPARATION, "buffer_increment_m": 0.05,
              "max_attempts": 10, "safety_margin_m": 0.0}
    write_manifest(output, schema="zephyr-d-separation-manifest-1", argv=argv_or_default(argv), command=command, config=config, seeds=seeds, outputs=outputs,
                   limits=limits, runtime_s=time.perf_counter() - started, n_processes=proc,
                   hashed_sources=["tools/d_run_separation.py", "tools/d_make_separation_figure.py", "sim/coop/separation.py", "sim/coop/avoidance.py", "sim/link.py", "sim/interaction.py", "sim/s6_recovery.py", "sim/coop/buffer_schema.json"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args_list = argv_or_default(argv)
    args = parser.parse_args(args_list)
    run(args.output, seeds_n=args.seeds, pilot=args.pilot, processes=args.processes, checkpoint_dir=args.checkpoint, argv=args_list)


if __name__ == "__main__":
    main()
