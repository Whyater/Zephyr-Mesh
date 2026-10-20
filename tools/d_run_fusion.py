"""Run the simulated cooperative fusion comparison."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

# Keep estimator modules loaded in the parent. Pool workers do not report their
# module lists back to the parent process.
import sim
import sim.adapters
import sim.authority
import sim.coop
import sim.coop.fusion
import sim.coop.net
import sim.estimation
import sim.estimation.base
import sim.estimation.consistency
import sim.estimation.cv
import sim.estimation.factory
import sim.estimation.kinematics
import sim.estimation.sweep
import sim.estimation.targets
import sim.failsafe
import sim.hardware
import sim.radio
import sim.radio.channel
import sim.radio.mac
import sim.radio.params
import sim.sensor
import sim.s6_recovery
import sim.tracker

from sim.coop.fusion import (
    covariance_intersection_fast,
    covariance_intersection_many,
    information_fusion,
)
from sim.coop.net import LinkSpec
from sim.coop.observers import ARMS, CoopCell, run_seed
from sim.estimation.consistency import nis_band
from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import RecoveryConfig, median_interval, wilson_interval
from tools.d_common import (
    REPO_ROOT,
    argv_or_default,
    default_processes,
    prepare_output,
    seeds_for,
    write_json,
    write_manifest,
)

TARGETS = (("straight", "cv", ()), ("turn", "ct", ()), ("jinking", "jink", ()))
PILOT_TARGETS = (TARGETS[1],)
ROW_COLUMNS = (
    "n_observers", "noise_std_m", "target", "loss_probability", "arm", "ci", "seed",
    "p95_error_m", "max_error_m", "rmse_m", "failed", "mean_nees",
    "frac_windows_above_band", "fused_count", "diverged", "diverged_time_s",
)
METRIC_FIELDS = ("p95_error_m", "max_error_m", "rmse_m", "mean_nees",
                 "frac_windows_above_band")


def _cell(job: dict) -> CoopCell:
    """Build the public observer cell for one simulated job."""
    return CoopCell(
        target=str(job["target"]),
        target_params=tuple(tuple(item) for item in job["target_params"]),
        n_observers=int(job["n_observers"]),
        rate_hz=10.0,
        noise_std_m=float(job["noise_std_m"]),
        link=LinkSpec(mode="abstract", delay_s=0.05,
                      loss_probability=float(job["loss_probability"])),
        phase="random",
        ci=str(job["ci"]),
    )


def _finite_or_none(value) -> float | None:
    """Convert a simulated metric to JSON and CSV safe form."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _job(job: dict) -> dict:
    """Run one simulated cell; a diverged seed is a failure with empty metrics."""
    started = time.process_time()
    cell = _cell(job)
    selected_arms = tuple(job.get("arms", ARMS))
    outcomes = run_seed(cell, int(job["seed"]), arms=selected_arms,
                        config=RecoveryConfig())
    rows = []
    for arm in selected_arms:
        outcome = outcomes[arm]
        extras = outcome.extras or {}
        diverged = bool(extras.get("diverged", False))
        diverged_time = _finite_or_none(extras.get("diverged_time_s"))
        row = {
            "n_observers": cell.n_observers,
            "noise_std_m": cell.noise_std_m,
            "target": job["target_label"],
            "loss_probability": cell.link.loss_probability,
            "arm": arm,
            "ci": cell.ci,
            "seed": int(job["seed"]),
            "failed": bool(outcome.failed) or diverged,
            "fused_count": int(outcome.fused_count),
            "diverged": diverged,
            "diverged_time_s": diverged_time,
        }
        for name in METRIC_FIELDS:
            row[name] = _finite_or_none(getattr(outcome, name))
        row["nonfinite_metric_count"] = sum(row[name] is None for name in METRIC_FIELDS)
        if diverged:
            # A diverged seed is a failure, never a number in the medians.
            for name in METRIC_FIELDS:
                row[name] = None
        rows.append(row)
    return {
        "rows": rows,
        "job_type": {"arms": list(selected_arms), "n_observers": cell.n_observers,
                     "ci": cell.ci},
        "worker_cpu_s": time.process_time() - started,
    }


def _interval(values: list[float | None]) -> dict:
    """Return a labelled median interval and count omitted non-finite values."""
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    missing = len(values) - len(finite)
    if not finite:
        return {"median": None, "lo": None, "hi": None, "n": 0,
                "nonfinite_count": missing}
    median, low, high = median_interval(finite)
    return {"median": float(median), "lo": float(low), "hi": float(high),
            "n": len(finite), "nonfinite_count": missing}


def _summarize(rows: list[dict]) -> list[dict]:
    """Summarize simulated error, failure, divergence, and consistency measures."""
    band_low, band_high = nis_band(10, 3, 0.95)
    keys = sorted({(r["n_observers"], r["noise_std_m"], r["target"],
                    r["loss_probability"], r["arm"], r["ci"]) for r in rows})
    output = []
    for key in keys:
        selected = [row for row in rows if tuple(row[name] for name in (
            "n_observers", "noise_std_m", "target", "loss_probability", "arm", "ci")) == key]
        failed = sum(bool(row["failed"]) for row in selected)
        fail_low, fail_high = wilson_interval(failed, len(selected))
        output.append({
            "n_observers": key[0], "noise_std_m": key[1], "target": key[2],
            "loss_probability": key[3], "arm": key[4], "ci": key[5],
            "n_seeds": len(selected),
            "p95_error_m": _interval([row["p95_error_m"] for row in selected]),
            "fail_rate": {"rate": failed / len(selected), "lo": fail_low, "hi": fail_high,
                          "count": failed, "n": len(selected)},
            "mean_nees": _interval([row["mean_nees"] for row in selected]),
            "frac_windows_above_band": _interval(
                [row["frac_windows_above_band"] for row in selected]),
            "diverged_count": sum(bool(row["diverged"]) for row in selected),
            "nonfinite_metric_count": sum(row["nonfinite_metric_count"] for row in selected),
            "nis_band": {"lo": float(band_low), "hi": float(band_high),
                         "window_queries": 10, "dof": 3, "confidence": 0.95},
        })
    return output


def _exact_subset(fast_rows: list[dict], exact_rows: list[dict]) -> list[dict]:
    """Compare fast and exact simulated CI on the registered turn subset."""
    keys = sorted({(r["target"], r["noise_std_m"], r["n_observers"], r["loss_probability"])
                   for r in exact_rows})
    output = []
    for target, noise, n_observers, loss in keys:
        def same_cell(r: dict) -> bool:
            return (r["arm"] == "peer_ci" and r["target"] == target and
                    r["noise_std_m"] == noise and r["n_observers"] == n_observers and
                    r["loss_probability"] == loss)
        fast = [r for r in fast_rows if same_cell(r)]
        exact = [r for r in exact_rows if same_cell(r)]
        fast_by_seed = {r["seed"]: r for r in fast}
        exact_by_seed = {r["seed"]: r for r in exact}
        common = sorted(set(fast_by_seed) & set(exact_by_seed))
        p95_relative = []
        nees_relative = []
        for seed in common:
            fast_row, exact_row = fast_by_seed[seed], exact_by_seed[seed]
            if fast_row["p95_error_m"] is not None and exact_row["p95_error_m"] is not None:
                p95_relative.append(abs(exact_row["p95_error_m"] - fast_row["p95_error_m"]) /
                                    max(abs(fast_row["p95_error_m"]), 1e-15))
            if fast_row["mean_nees"] is not None and exact_row["mean_nees"] is not None:
                nees_relative.append(abs(exact_row["mean_nees"] - fast_row["mean_nees"]) /
                                     max(abs(fast_row["mean_nees"]), 1e-15))
        output.append({
            "target": target, "noise_std_m": noise,
            "n_observers": n_observers, "loss_probability": loss, "n_seeds": len(common),
            "fast": {"p95_error_m": _interval([fast_by_seed[s]["p95_error_m"] for s in common]),
                     "mean_nees": _interval([fast_by_seed[s]["mean_nees"] for s in common])},
            "exact": {"p95_error_m": _interval([exact_by_seed[s]["p95_error_m"] for s in common]),
                      "mean_nees": _interval([exact_by_seed[s]["mean_nees"] for s in common])},
            "per_seed_relative_difference": {
                "p95_error_m": _interval(p95_relative), "mean_nees": _interval(nees_relative),
            },
        })
    return output


def _random_spd(rng: np.random.Generator) -> np.ndarray:
    """Draw a full SPD covariance for the simulated consistency check."""
    matrix = rng.normal(size=(3, 3))
    return matrix @ matrix.T + np.eye(3) * 0.2


def _math_checks() -> dict:
    """Run fusion module checks on simulated draws and correlated covariances."""
    sigma = 0.7
    rng = np.random.default_rng(np.random.SeedSequence([20261015, 23, 700]))
    sigma_rows = []
    for n in range(1, 7):
        covariance = np.eye(3) * sigma * sigma
        fused_means = []
        for _ in range(20000):
            draws = rng.normal(0.0, sigma, (n, 3))
            fused, _ = information_fusion([draws[i] for i in range(n)],
                                          [covariance] * n)
            fused_means.append(fused)
        measured = float(np.var(np.asarray(fused_means), axis=0, ddof=1).mean())
        hand = sigma * sigma / n
        sigma_rows.append({"n": n, "measured_variance": measured,
                           "hand_variance": hand, "ratio": measured / hand})

    zero = np.zeros(3)
    eye = np.eye(3)
    _, information_covariance = information_fusion([zero, zero], [eye, eye])
    _, fast_covariance, fast_weights = covariance_intersection_fast([zero, zero], [eye, eye])
    _, exact_covariance, exact_weights = covariance_intersection_many([zero, zero], [eye, eye])
    rng = np.random.default_rng(np.random.SeedSequence([20261015, 23, 701]))
    minima = {"fast": float("inf"), "exact": float("inf"), "information": float("inf")}
    too_little = {key: 0 for key in minima}
    trials = 2000
    for _ in range(trials):
        a = _random_spd(rng)
        b = _random_spd(rng)
        u, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        choice = int(rng.integers(0, 3))
        rho = 0.0 if choice == 0 else 1.0 if choice == 1 else float(rng.uniform(0.0, 0.95))
        cross = np.linalg.cholesky(a) @ (u @ np.diag([rho, rho * 0.7, rho * 0.3]) @ v.T)
        cross = cross @ np.linalg.cholesky(b).T
        means = [np.zeros(3), np.zeros(3)]
        _, fast_p, fw = covariance_intersection_fast(means, [a, b])
        _, exact_p, ew = covariance_intersection_many(means, [a, b])
        _, info_p = information_fusion(means, [a, b])
        ia = np.linalg.solve(a, np.eye(3))
        ib = np.linalg.solve(b, np.eye(3))
        for name, claimed, weights in (("fast", fast_p, fw), ("exact", exact_p, ew)):
            gains = [claimed @ (float(weights[i]) * (ia if i == 0 else ib)) for i in range(2)]
            truth = (gains[0] @ a @ gains[0].T + gains[1] @ b @ gains[1].T +
                     gains[0] @ cross @ gains[1].T + gains[1] @ cross.T @ gains[0].T)
            eigenvalue = float(np.linalg.eigvalsh(claimed - truth).min())
            minima[name] = min(minima[name], eigenvalue)
            too_little[name] += eigenvalue < -1e-12
        gains = [info_p @ ia, info_p @ ib]
        truth = (gains[0] @ a @ gains[0].T + gains[1] @ b @ gains[1].T +
                 gains[0] @ cross @ gains[1].T + gains[1] @ cross.T @ gains[0].T)
        eigenvalue = float(np.linalg.eigvalsh(info_p - truth).min())
        minima["information"] = min(minima["information"], eigenvalue)
        too_little["information"] += eigenvalue < -1e-12
    return {
        "sigma2_over_n": sigma_rows,
        "ci_hand": {"information_covariance": information_covariance.tolist(),
                     "fast_covariance": fast_covariance.tolist(), "fast_weights": fast_weights.tolist(),
                     "exact_covariance": exact_covariance.tolist(), "exact_weights": exact_weights.tolist()},
        "ci_consistency": {
            "trials": trials,
            "fast": {"min_eigenvalue": minima["fast"],
                      "claims_too_little_fraction": too_little["fast"] / trials},
            "exact": {"min_eigenvalue": minima["exact"],
                      "claims_too_little_fraction": too_little["exact"] / trials},
            "information": {"min_eigenvalue": minima["information"],
                            "claims_too_little_fraction": too_little["information"] / trials},
        },
    }


def _jobs(ns, noises, targets, losses, seeds, ci):
    return [{"n_observers": n, "noise_std_m": noise, "target_label": label,
             "target": target, "target_params": params, "loss_probability": loss,
             "seed": seed, "ci": ci}
            for n in ns for noise in noises for label, target, params in targets
            for loss in losses for seed in seeds]


def _source_paths() -> list[str]:
    """Return imported repository Python modules plus the Horizon token file."""
    paths = {"design/horizon_tokens.json"}
    for module in sys.modules.values():
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            relative = Path(path).resolve().relative_to(REPO_ROOT)
        except ValueError:
            continue
        if relative.suffix == ".py":
            paths.add(relative.as_posix())
    return sorted(paths)


def run(output: Path, *, pilot: bool = False, processes: int | None = None, argv=None) -> dict:
    """Run the pilot or registered grid and write all simulated outputs."""
    started = time.perf_counter()
    output = prepare_output(output)
    seed_count = 5 if pilot else 50
    seeds = seeds_for("fusion", seed_count)
    ns = (2, 6) if pilot else (2, 3, 4, 6)
    noises = (0.05,) if pilot else (0.05, 0.10)
    targets = PILOT_TARGETS if pilot else TARGETS
    losses = (0.0, 0.3)
    fast_jobs = _jobs(ns, noises, targets, losses, seeds, "fast")
    exact_ns = (2, 6) if pilot else (2, 3, 6)
    exact_jobs = _jobs(exact_ns, (0.05,), (TARGETS[1],), losses, seeds, "exact")
    for job in exact_jobs:
        job["arms"] = ("peer_ci",)
    worker_count = processes or default_processes()
    fast_results = run_jobs(fast_jobs, "tools.d_run_fusion:_job", processes=worker_count, progress=False)
    exact_results = run_jobs(exact_jobs, "tools.d_run_fusion:_job", processes=worker_count, progress=False)
    calibration_jobs = []
    if pilot:
        calibration_jobs.extend(_jobs((3, 4), (0.05,), (TARGETS[1],), (0.0,), [seeds[0]], "fast"))
        exact_calibration = _jobs((3,), (0.05,), (TARGETS[1],), (0.0,), [seeds[0]], "exact")
        for job in exact_calibration:
            job["arms"] = ("peer_ci",)
        calibration_jobs.extend(exact_calibration)
    calibration_results = run_jobs(calibration_jobs, "tools.d_run_fusion:_job",
                                   processes=worker_count, progress=False)
    fast_rows = [row for result in fast_results for row in result["rows"]]
    exact_rows = [row for result in exact_results for row in result["rows"]]
    if len(fast_rows) != len(fast_jobs) * len(ARMS):
        raise RuntimeError("fast row count does not match the requested grid")
    if len(exact_rows) != len(exact_jobs):
        raise RuntimeError("exact row count does not match the requested grid")
    write_csv(output / "per_seed.csv", fast_rows + exact_rows, ROW_COLUMNS)
    write_json(output / "summary.json", {"cells": _summarize(fast_rows),
                                         "exact_ci_subset": _exact_subset(fast_rows, exact_rows)})
    write_json(output / "math_checks.json", _math_checks())
    from tools import d_make_fusion_figure
    d_make_fusion_figure.make_figures(output)
    elapsed = time.perf_counter() - started
    all_results = fast_results + exact_results + calibration_results
    pilot_cpu = sum(float(result["worker_cpu_s"]) for result in all_results)
    timings: dict[tuple, list[float]] = {}
    for result in all_results:
        job_type = (tuple(result["job_type"]["arms"]), int(result["job_type"]["n_observers"]),
                    result["job_type"]["ci"])
        timings.setdefault(job_type, []).append(float(result["worker_cpu_s"]))
    estimates = {}
    for job_type, values in sorted(timings.items(), key=lambda item: repr(item[0])):
        arms, n_observers, ci = job_type
        full_count_for_n = ((2 * len(TARGETS) * 2 * 50) if ci == "fast" else (2 * 50))
        estimate = float(np.mean(values)) * full_count_for_n
        key = f"{ci}:{','.join(arms)}:N={n_observers}"
        estimates[key] = {"pilot_jobs": len(values), "full_jobs": full_count_for_n,
                          "mean_worker_cpu_s": float(np.mean(values)),
                          "full_cpu_s": estimate, "full_wall_s_4_processes": estimate / 4.0}
    full_cpu = float(sum(item["full_cpu_s"] for item in estimates.values()))
    full_wall = full_cpu / 4.0
    config = {
        "n_observers": [2, 3, 4, 6], "noise_std_m": [0.05, 0.10],
        "targets": [{"label": label, "name": target, "params": list(params)}
                    for label, target, params in TARGETS],
        "loss_probability": [0.0, 0.3],
        "link": {"mode": "abstract", "delay_s": 0.05, "loss_model": "independent"},
        "arms": list(ARMS), "rate_hz": 10.0, "phase": "random", "ci_modes": ["fast", "exact"],
        "filter_measurement_std_m": 0.05,
        "seeds": seed_count, "active_n_observers": list(ns),
        "active_noise_std_m": list(noises), "active_targets": [label for label, _, _ in targets],
        "exact_ci_subset": {"n_observers": list(exact_ns), "target": "turn", "noise_std_m": 0.05,
                            "loss_probability": [0.0, 0.3], "seeds": seed_count},
        "recovery_config": dataclasses.asdict(RecoveryConfig()),
    }
    command = "PYTHONPATH=. {python} tools/d_run_fusion.py --output {out}"
    if pilot:
        command += " --pilot"
    if processes is not None:
        command += f" --processes {int(processes)}"
    volatile = ["runtime_s", "git_revision", "git_tree_state", "platform", "n_processes",
                "argv", "pilot_worker_cpu_s", "full_run_estimate_cpu_s",
                "full_run_estimate_wall_s_4_processes", "full_run_estimate_by_job_type"]
    write_manifest(
        output, schema="zephyr-d-fusion-manifest-1", argv=argv_or_default(argv), command=command,
        config=config, seeds=seeds,
        outputs=["per_seed.csv", "summary.json", "math_checks.json", "fusion_error.png",
                 "fusion_consistency.png"],
        limits=[
            "Agents are simulated kinematic point masses; nothing here is measured.",
            "Sensor noise is simulated Gaussian noise on the target position and independent over time.",
            "Links are abstract with a fixed delay and independent loss.",
            "At sensor noise 0.10 m the filter still assumes 0.05 m, as in Run B's atlas, so NEES is inflated for every arm there; consistency is judged at noise 0.05 m.",
            "Failure thresholds are research criteria applied to simulated outcomes.",
        ],
        runtime_s=elapsed, n_processes=worker_count, hashed_sources=_source_paths(),
        extra={"volatile_fields": volatile, "pilot_worker_cpu_s": pilot_cpu,
               "full_run_estimate_cpu_s": full_cpu,
               "full_run_estimate_wall_s_4_processes": full_wall,
               "full_run_estimate_by_job_type": estimates},
    )
    result = {"pilot_runtime_s": elapsed, "pilot_worker_cpu_s": pilot_cpu,
              "full_run_estimate_cpu_s": full_cpu,
              "full_run_estimate_wall_s_4_processes": full_wall,
              "full_run_estimate_by_job_type": estimates}
    print(json.dumps(result, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    args = parser.parse_args(argv)
    run(args.output, pilot=args.pilot, processes=args.processes, argv=argv)


if __name__ == "__main__":
    main()
