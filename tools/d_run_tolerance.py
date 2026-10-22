"""Run the simulated cooperative tracking tolerance sweep and pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import sim.estimation.cv  # noqa: F401
from sim.coop.boundary import tolerated
from sim.coop.net import LinkSpec, predicted_channel_use
from sim.coop.observers import CoopCell, run_seed
from sim.s6_recovery import median_interval, wilson_interval
from sim.estimation.sweep import run_jobs, sha256_json, write_csv, write_csv_gz
from tools.d_common import argv_or_default, prepare_output, seeds_for, write_json, write_manifest
from tools.d_make_tolerance_figure import make_figure

NS = (1, 2, 3, 4, 6)
DELAYS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
LOSSES = (("independent", 0.0), ("independent", 0.1), ("independent", 0.3),
          ("independent", 0.5), ("burst", 0.3))
TARGETS = (("cv_1", "cv", ()), ("ct_r1", "ct", ()), ("jink", "jink", ()))
DISPLAY_ARMS = ("central", "star_avg", "peer_ci")
MESSAGE_BYTES = {"single": 32, "central": 32, "star_avg": 108, "peer_ci": 216}


def _positions(n_observers: int) -> dict[str, tuple[float, float, float]]:
    positions = {"gs": (0.0, 0.0, 1.0)}
    for index in range(n_observers):
        angle = 2.0 * np.pi * index / n_observers
        positions[f"o{index + 1}"] = (2.0 + 3.0 * float(np.cos(angle)),
                                         3.0 * float(np.sin(angle)), 1.5)
    return positions


def _cell(job: dict) -> CoopCell:
    """Build the registered simulated configuration for one worker job."""
    n_observers = int(job["n"])
    link_name = str(job["link"])
    rate_hz = 50.0 if link_name == "radio_stress" else 10.0
    positions = _positions(n_observers)
    background = ()
    if link_name == "radio_stress":
        positions.update({"bg_sink": (0.0, -1.0, 1.0), "b0": (-2.0, -2.0, 1.0),
                          "b1": (-2.0, 2.0, 1.0), "b2": (4.0, -3.0, 1.0),
                          "b3": (4.0, 3.0, 1.0)})
        background = tuple((f"b{index}", "bg_sink", 50.0, 200, 0.0, 12.0, 0.0)
                           for index in range(4))
    link_kwargs = {"mode": "radio" if link_name != "abstract" else "abstract",
                   "delay_s": float(job["delay"]),
                   "loss_probability": float(job["loss"]),
                   "loss_model": str(job["loss_model"])}
    if link_name != "abstract":
        link_kwargs.update({"radio_preset": "indoor_gym", "positions": positions,
                            "background": background})
    link = LinkSpec(**link_kwargs)
    target = next(item[1:] for item in TARGETS if item[0] == str(job["target"]))
    return CoopCell(target=target[0], target_params=target[1], n_observers=n_observers,
                    rate_hz=rate_hz, noise_std_m=0.05, r_std_m=None, link=link,
                    phase="random", ci="fast")


def _registered_arms(n_observers: int) -> tuple[str, ...]:
    return ("single",) if int(n_observers) == 1 else DISPLAY_ARMS


def _gap_check(table, n_observers: int, loss_model: str, loss: float,
               rate_hz: float) -> dict:
    """Measure all-observer capture gaps directly from one delivery table."""
    routes = [tuple(table.get((f"o{index + 1}", "gs"), ()))
              for index in range(n_observers)]
    n_steps = max((len(route) for route in routes), default=0)
    valid = []
    all_lost = []
    for step in range(n_steps):
        entries = [route[step] for route in routes if step < len(route)]
        if len(entries) != n_observers or any(item.loss_reason == "late_end" for item in entries):
            continue
        valid.append(step)
        all_lost.append(all(item.receive_time_s is None and item.loss_reason == "link"
                            for item in entries))
    lost_count = sum(all_lost)
    successful_steps = [step for step, lost in zip(valid, all_lost) if not lost]
    period = 1.0 / float(rate_hz)
    gaps = [(right - left) * period for left, right in
            zip(successful_steps[:-1], successful_steps[1:])]
    hand = float(loss) ** int(n_observers) if loss_model == "independent" else None
    hand_gap = (1.0 / ((1.0 - hand) * rate_hz)
                if hand is not None and hand < 1.0 else None)
    return {
        "all_lost_fraction": (float(lost_count) / len(valid) if valid else None),
        "hand_all_lost_fraction": hand,
        "mean_gap_s": (float(np.mean(gaps)) if gaps else None),
        "hand_mean_gap_s": hand_gap,
        "valid_capture_steps": int(len(valid)),
        "all_lost_steps": int(lost_count),
    }


def _job(job: dict) -> dict:
    """Run one cell and seed, returning only registered arm results."""
    cpu_start = time.process_time()
    cell = _cell(job)
    arms = _registered_arms(int(job["n"]))
    outcomes = run_seed(cell, int(job["seed"]), arms=arms)
    rows = []
    for arm in arms:
        outcome = outcomes[arm]
        table = outcome.extras.get("delivery_table")
        summary = table.summary() if table is not None else {}
        gap_check = (_gap_check(table, cell.n_observers, str(job["loss_model"]),
                                float(job["loss"]), cell.rate_hz)
                     if cell.link.mode == "abstract" and table is not None else None)
        rows.append({"link": str(job["link"]), "target": str(job["target"]),
                     "loss_model": str(job["loss_model"]), "loss": float(job["loss"]),
                     "arm": arm, "n": int(job["n"]), "delay": float(job["delay"]),
                     "seed": int(job["seed"]), "failed": bool(outcome.failed),
                     "p95": float(outcome.p95_error_m), "max": float(outcome.max_error_m),
                     "rmse": float(outcome.rmse_m), "mean_nees": outcome.mean_nees,
                     "busy_fraction": summary.get("busy_fraction"),
                     "delivery_ratio": summary.get("delivery_ratio"),
                     "gap_check": gap_check})
    return {"rows": rows, "worker_cpu_s": time.process_time() - cpu_start}


def _condition_key(link, target, loss_model, loss, arm, n):
    return f"{link}|{target}|{loss_model}:{loss:g}|{arm}|{n}"


def _mean(values):
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _write_cells(output: Path, grouped: dict[tuple, list[dict]], seeds: list[int]) -> None:
    columns = ("link", "target", "loss_model", "loss", "arm", "n", "delay", "n_seeds",
               "fail_count", "fail_rate", "fail_rate_wilson_lo", "fail_rate_wilson_hi",
               "p95_median", "p95_median_interval", "max_median", "rmse_median",
               "mean_nees", "busy_fraction", "delivery_ratio")
    cell_rows = []
    for key, selected in sorted(grouped.items()):
        failed = sum(bool(r["failed"]) for r in selected)
        median, low, high = median_interval([r["p95"] for r in selected])
        wilson_low, wilson_high = wilson_interval(failed, len(seeds))
        values = [key[0], key[1], key[2], key[3], key[4], key[5], key[6], len(seeds),
                  failed, failed / len(seeds), wilson_low, wilson_high, median,
                  json.dumps({"median": median, "lo": low, "hi": high,
                              "n": len(selected)}, sort_keys=True),
                  float(np.median([r["max"] for r in selected])),
                  float(np.median([r["rmse"] for r in selected])),
                  _mean([r["mean_nees"] for r in selected]),
                  _mean([r["busy_fraction"] for r in selected]),
                  _mean([r["delivery_ratio"] for r in selected])]
        cell_rows.append(dict(zip(columns, values)))
    write_csv(output / "cells.csv", cell_rows, columns)


def _group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["link"], row["target"], row["loss_model"], row["loss"], row["arm"],
               row["n"], row["delay"])
        grouped.setdefault(key, []).append(row)
    return grouped


def _build_boundaries(grouped: dict[tuple, list[dict]], delays: tuple[float, ...], n_seeds: int) -> dict:
    """Aggregate per-seed failure counts into atlas-compatible boundaries."""
    boundaries = {}
    keys = sorted({key[:6] for key in grouped})
    for link, target, loss_model, loss, arm, n in keys:
        counts = [sum(bool(r["failed"]) for r in grouped.get(
            (link, target, loss_model, loss, arm, n, delay), ()))
                  for delay in delays]
        result = tolerated(delays, counts, n_seeds)
        result["fail_counts"] = counts
        boundaries[_condition_key(link, target, loss_model, loss, arm, n)] = result
    return boundaries


def _build_best(boundaries, links, targets, losses, ns):
    best = {}
    for link in links:
        for target, _, _ in targets:
            for loss_model, loss in losses:
                for arm in DISPLAY_ARMS:
                    candidates = []
                    for n in ns:
                        source_arm = "single" if n == 1 else arm
                        key = _condition_key(link, target, loss_model, loss, source_arm, n)
                        if key not in boundaries:
                            continue
                        boundary = boundaries[key]
                        score = (-1.0 if boundary["fails_at_zero"] else
                                 float("inf") if boundary["censored"] else float(boundary["value"]))
                        candidates.append((score, n))
                    if candidates:
                        best[f"{link}|{target}|{loss_model}:{loss:g}|{arm}"] = min(
                            candidates, key=lambda item: (-item[0], item[1]))[1]
    return best


def _build_gaps(grouped: dict[tuple, list[dict]]) -> dict:
    fields = ("all_lost_fraction", "hand_all_lost_fraction", "mean_gap_s",
              "hand_mean_gap_s", "valid_capture_steps", "all_lost_steps")
    keys = sorted(key for key, values in grouped.items()
                  if any(row["gap_check"] is not None for row in values))
    output = {}
    for key in keys:
        values = [r["gap_check"] for r in grouped[key] if r["gap_check"] is not None]
        label = "|".join((str(key[0]), str(key[1]), str(key[2]), f"{key[3]:g}",
                          str(key[4]), str(key[5]), f"{key[6]:g}"))
        output[label] = {field: _mean([item[field] for item in values]) for field in fields}
    return output


def _build_channel_use(grouped: dict[tuple, list[dict]]) -> dict:
    keys = sorted(grouped)
    entries = {}
    for link, target, loss_model, loss, arm, n, delay in keys:
        values = grouped[(link, target, loss_model, loss, arm, n, delay)]
        traffic_n = 1 if arm == "single" else n
        rate_hz = 50.0 if link == "radio_stress" else 10.0
        predicted = predicted_channel_use(traffic_n, rate_hz, MESSAGE_BYTES[arm],
                                          arm not in {"peer_ci"})
        label = "|".join((str(link), str(target), str(loss_model), f"{loss:g}", str(arm),
                          str(n), f"{delay:g}"))
        entries[label] = {"predicted_channel_use": float(predicted),
                          "busy_fraction": _mean([r["busy_fraction"] for r in values])}
    return {"note": "predicted_channel_use includes idle DIFS and backoff time; "
                     "busy_fraction is on-air occupancy and the two are not comparable.",
            "entries": entries}


def _config(ns, delays, losses, targets, links, seeds):
    return {
        "n_observers": list(ns), "delays_s": list(delays),
        "loss_conditions": [{"model": model, "probability": probability}
                            for model, probability in losses],
        "targets": [{"name": name, "target": target, "params": dict(params)}
                    for name, target, params in targets],
        "links": list(links), "arms_per_n": {"1": ["single"], "2+": list(DISPLAY_ARMS)},
        "estimator": "cv",
        "fusion_per_arm": {"single": "local", "central": "sequential_kalman",
                            "star_avg": "naive_average", "peer_ci": "covariance_intersection"},
        "rates_hz": {"abstract": 10.0, "radio_gym": 10.0, "radio_stress": 50.0},
        "radio_geometry": "ground station at (0,0,1), observers on radius-3 m circle around (2,0,1.5)",
        "background": {"nodes": 4, "rate_hz": 50.0, "message_bytes": 200, "receiver": "bg_sink"},
        "phase": "random (o1 at 0, others seeded per run)", "noise_std_m": 0.05,
        "r_std_m": None, "ci": "fast",
        "duration_s": 12.0, "burn_in_s": 2.0, "query_dt_s": 0.02,
        "control_dt_s": 0.02, "log_every": 1, "burst_mean_packets": 5,
        "delay_added_after_radio": True,
        "failure_rule": {"p95_error_m": 0.25, "max_error_m": 0.50}, "seeds": list(seeds),
    }


def _source_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    paths = set()
    paths.update({Path(__file__).resolve(), root / "tools/d_make_tolerance_figure.py",
                  root / "tools/horizon_mpl.py", root / "sim/coop/__init__.py",
                  root / "sim/coop/observers.py", root / "sim/coop/net.py",
                  root / "sim/coop/fusion.py", root / "sim/coop/boundary.py",
                  root / "sim/estimation/cv.py"})
    for module in sys.modules.values():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        paths.add(path)
    return sorted(paths)


def _checkpoint_fingerprint(config: dict) -> str:
    root = Path(__file__).resolve().parents[1]
    names = set(_source_paths())
    digest = hashlib.sha256()
    for path in sorted(names):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "tolerance-" + digest.hexdigest()


def _check_checkpoint_fingerprint(checkpoint: Path, fingerprint: str) -> None:
    for path in sorted(checkpoint.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"checkpoint directory {checkpoint} contains invalid JSON") from exc
        if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
            raise ValueError(f"checkpoint directory {checkpoint} contains a mismatched fingerprint")


def _count_cached_jobs(checkpoint: Path | None, jobs: list[dict], fingerprint: str) -> int:
    if checkpoint is None:
        return 0
    cached = 0
    for job in jobs:
        path = checkpoint / f"{sha256_json(job)[:16]}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("fingerprint") == fingerprint:
            cached += 1
    return cached


def _validate_rows(rows: list[dict], jobs: list[dict]) -> None:
    """Ensure every registered cell, seed, and arm appears exactly once."""
    expected = {
        (job["link"], job["target"], job["loss_model"], float(job["loss"]),
         int(job["n"]), float(job["delay"]), int(job["seed"]), arm)
        for job in jobs for arm in _registered_arms(int(job["n"]))
    }
    actual = {
        (row["link"], row["target"], row["loss_model"], float(row["loss"]),
         int(row["n"]), float(row["delay"]), int(row["seed"]), row["arm"])
        for row in rows
    }
    if len(rows) != len(expected) or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"runner rows do not match grid; missing={missing[:3]}, extra={extra[:3]}")


def _cpu_per_job_by_n(measured: dict[int, float], wanted: tuple[int, ...]) -> dict[int, float]:
    """Mean worker CPU seconds per job for each wanted N.

    Measured values are used as they are. An N that was not measured is placed on the straight
    line through the two nearest measured points (interpolation or extrapolation). If cost
    grows faster than linearly in N, extrapolating past the largest measured N is a lower bound.
    """
    known = sorted(measured)
    if len(known) < 2:
        raise ValueError("need CPU measured at two or more N values to estimate the rest")
    estimate = {}
    for n in wanted:
        if n in measured:
            estimate[n] = float(measured[n])
            continue
        left, right = sorted(sorted(known, key=lambda item: (abs(item - n), item))[:2])
        slope = (measured[right] - measured[left]) / float(right - left)
        estimate[n] = float(measured[left] + slope * (n - left))
    return estimate


def _stress_background_note(rows: list[dict]) -> str:
    """Text for the figure: median on-air occupancy at N=1 on radio_stress, to the nearest 5 %."""
    values = [float(r["busy_fraction"]) for r in rows
              if r["link"] == "radio_stress" and int(r["n"]) == 1
              and r["busy_fraction"] is not None]
    if not values:
        return ""
    return f"~{5 * round(100.0 * float(np.median(values)) / 5.0):d} % background"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.processes < 1 or args.processes > 4:
        raise ValueError("--processes must be between 1 and 4")
    if args.pilot and args.processes > 2:
        raise ValueError("pilot runs allow at most 2 processes")
    output = prepare_output(Path(args.output))
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and (checkpoint.resolve() == output.resolve()
                                   or output.resolve() in checkpoint.resolve().parents):
        raise ValueError("--checkpoint must be outside --output")
    pilot = bool(args.pilot)
    seeds = seeds_for("tolerance", 4 if pilot else 20)
    ns = (1, 2, 4) if pilot else NS
    delays = (0.0, 0.1, 0.2) if pilot else DELAYS
    losses = (("independent", 0.5), ("burst", 0.3)) if pilot else LOSSES
    targets = (TARGETS[1],) if pilot else TARGETS
    links = ("abstract", "radio_gym", "radio_stress") if pilot else (
        "abstract", "radio_gym", "radio_stress")
    config = _config(ns, delays, losses, targets, links, seeds)
    jobs = [{"link": link, "target": target[0], "loss_model": model, "loss": loss,
             "n": n, "delay": delay, "seed": seed}
            for link in links for target in targets for model, loss in losses for n in ns
            for delay in delays for seed in seeds]
    fingerprint = _checkpoint_fingerprint(config)
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        _check_checkpoint_fingerprint(checkpoint, fingerprint)
    cached_jobs = _count_cached_jobs(checkpoint, jobs, fingerprint)
    start = time.perf_counter()
    raw = run_jobs(jobs, "tools.d_run_tolerance:_job", processes=args.processes,
                   checkpoint_dir=checkpoint, fingerprint=fingerprint,
                   progress=False)
    runtime_s = time.perf_counter() - start
    worker_cpu_s = float(sum(float(result["worker_cpu_s"]) for result in raw))
    pilot_cpu_groups: dict[tuple[str, int], list[float]] = {}
    for job, result in zip(jobs, raw):
        key = (str(job["link"]), int(job["n"]))
        pilot_cpu_groups.setdefault(key, []).append(float(result["worker_cpu_s"]))
    pilot_cpu_by_link_n = {
        f"{link}|{n}": float(np.mean(values))
        for (link, n), values in sorted(pilot_cpu_groups.items())
    }
    rows = [row for result in raw for row in result["rows"]]
    _validate_rows(rows, jobs)
    columns = ("link", "target", "loss_model", "loss", "arm", "n", "delay", "seed",
               "failed", "p95", "max", "rmse", "mean_nees", "busy_fraction",
               "delivery_ratio", "gap_check")
    encoded_rows = []
    for row in rows:
        encoded = dict(row)
        encoded["gap_check"] = (json.dumps(row["gap_check"], sort_keys=True)
                                 if row["gap_check"] is not None else None)
        encoded_rows.append(encoded)
    write_csv_gz(output / "per_seed.csv.gz", encoded_rows, columns)
    grouped = _group_rows(rows)
    boundaries = _build_boundaries(grouped, delays, len(seeds))
    write_json(output / "boundaries.json", boundaries)
    write_json(output / "best_n.json", _build_best(boundaries, links, targets, losses, ns))
    _write_cells(output, grouped, seeds)
    write_json(output / "gaps.json", _build_gaps(grouped))
    write_json(output / "channel_use.json", _build_channel_use(grouped))
    make_figure(output / "boundaries.json", output / "tolerance_vs_n.png", pilot=pilot,
                rates_hz=config["rates_hz"],
                link_notes={"radio_stress": _stress_background_note(rows)})
    full_jobs_per_link_n = len(DELAYS) * len(LOSSES) * len(TARGETS) * 20
    estimated_cpu_s = 0.0
    estimated_cpu_per_job = {}
    for link in links:
        measured = {n: float(np.mean(pilot_cpu_groups[(link, n)])) for n in ns}
        per_job = _cpu_per_job_by_n(measured, NS)
        estimated_cpu_per_job.update({f"{link}|{n}": value for n, value in per_job.items()})
        estimated_cpu_s += sum(per_job.values()) * full_jobs_per_link_n
    estimated_wall_s = estimated_cpu_s / 4.0
    timing = {"pilot_worker_cpu_s": worker_cpu_s, "full_run_estimate_cpu_s": estimated_cpu_s,
              "full_run_estimate_wall_s_at_4_processes": estimated_wall_s, "runtime_s": runtime_s,
              "checkpoint_loaded_jobs": cached_jobs,
              "checkpoint_recomputed_jobs": len(jobs) - cached_jobs,
              "pilot_cpu_by_link_n": pilot_cpu_by_link_n,
              "estimated_cpu_per_job_by_link_n": estimated_cpu_per_job}
    write_json(output / "timing.json", timing)
    print(f"pilot_runtime_s={runtime_s:.3f}")
    print(f"pilot_worker_cpu_s={worker_cpu_s:.3f}")
    print(f"full_run_estimate_cpu_s={estimated_cpu_s:.3f}")
    print(f"full_run_estimate_wall_s_at_4_processes={estimated_wall_s:.3f}")
    if estimated_wall_s > 3.0 * 60.0 * 60.0:
        print("full-run estimate exceeds about 3 hours at 4 processes")
        print("safe options: use --checkpoint and resume, or split the registered grid by cells")
    command = "PYTHONPATH=. {python} tools/d_run_tolerance.py --output {out}"
    if pilot:
        command += " --pilot"
    if args.processes != 2:
        command += f" --processes {args.processes}"
    if args.checkpoint:
        command += " --checkpoint {checkpoint}"
    source_paths = _source_paths()
    repo_root = Path(__file__).resolve().parents[1]
    hashed_sources = [path.relative_to(repo_root).as_posix() for path in source_paths]
    outputs = ["per_seed.csv.gz", "cells.csv", "boundaries.json", "best_n.json", "gaps.json",
               "channel_use.json", "tolerance_vs_n.png"]
    write_manifest(output, schema="zephyr-d-tolerance-manifest-1", argv=argv_or_default(argv),
                   command=command, config=config, seeds=seeds, outputs=outputs,
                   limits=["Agents move as kinematic point masses in the tracking simulation.",
                           "Sensor samples are simulated Gaussian noise on the true target position.",
                           ("Links use abstract simulations or literature-derived radio parameters; "
                            "none are measured."),
                           "Twenty seeds are thin and one-step boundary differences are weak.",
                           ("The burst model is simple, the thresholds are research criteria, and "
                            "background load is chosen."),
                           ("radio_stress runs at 50 Hz, so its N=1 result is not the 10 Hz atlas "
                            "anchor, and a 5-packet burst lasts 0.1 s there instead of 0.5 s; "
                            "compare links only at the same capture rate.")],
                   runtime_s=runtime_s, n_processes=args.processes, hashed_sources=hashed_sources,
                   extra={"volatile_fields": ["runtime_s", "git_revision", "git_tree_state", "platform",
                                               "n_processes", "argv", "pilot_worker_cpu_s",
                                               "full_run_estimate_cpu_s",
                                               "full_run_estimate_wall_s_at_4_processes",
                                               "checkpoint_loaded_jobs", "checkpoint_recomputed_jobs",
                                               "pilot_cpu_by_link_n"],
                          "n_cells": len(jobs) // len(seeds), "n_runs": len(jobs),
                          "source_note": ("At N=1 the o1 route uses the S6 single-tracker loss seed. "
                                          "For N>=2 each route loss stream is the same for every N; "
                                          "common random numbers therefore hold only across N>=2.")})


if __name__ == "__main__":
    main()
