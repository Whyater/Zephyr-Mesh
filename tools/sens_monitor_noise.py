"""Side run: the EXP-033 monitor mission with its sensor noise varied.

The monitor mission hardcodes 0.5 m twice: the observers' camera noise
(``sim/missions/monitor.py`` ``SensorConfig(noise_std_m=0.5)``) and the ground
station filter's measurement setting (``EstimatorSettings(measurement_std_m=0.5)``).
This wrapper reruns the registered point-mass scenarios (delay 0.0 s, crossing
and skimming paths, clean and iid_10 presets, the first 30 EXP-033 seeds) with
those two values changed.

How the values are changed: ``monitor.py`` looks both class names up in its
own module namespace when a run starts, so the worker replaces the two names
``sim.missions.monitor.SensorConfig`` and ``sim.missions.monitor.EstimatorSettings``
with small factories for the length of one run, then puts the real classes
back.  Each factory checks that the mission still passes the literal 0.5, builds
the real object with the new value, and records the value it built, so every
row shows the sigma that was in effect inside the worker process.  The patch
is applied inside the worker (not in the parent), because the process pool
starts fresh interpreters on macOS and Windows.

Arms:
  planned            sensor = estimator = sigma, sigma in {0.25, 0.5, 1.0, 2.0} (the pre-registered plan)
  posthoc_matched    sensor = estimator = sigma, sigma in {4.0, 8.0}  (post hoc, not in the plan)
  posthoc_sensor     sensor sigma in {2, 4, 8}, estimator held at 0.5  (post hoc)
  posthoc_estimator  estimator sigma in {2, 4, 8}, sensor held at 0.5  (post hoc)

Control: the planned sigma 0.5 rows must equal the registered
``sim/runs/e-exp033-monitor/exp033.csv`` rows of the same cell and seed on every
registered column, compared as CSV text.

The EXP-033 runner (``tools/e_exp033_monitor.py``) is imported read-only: its
``build_jobs`` gives the registered scenario dicts, and ``make_job`` and
``run_jobs`` build and run the jobs.  Outputs are simulated only and go to a
folder outside ``sim/runs``.

Usage: ``PYTHONPATH=. python tools/sens_monitor_noise.py --output <side folder> --processes 3``
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from sim.missions import monitor
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    _cell_text,
    group_rows,
    median_summary,
    processes,
    rate_summary,
    replay_command,
    run_job,
    write_rows,
)
from tools.e_exp033_monitor import build_jobs, make_job, run_jobs

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CSV = ROOT / "sim" / "runs" / "e-exp033-monitor" / "exp033.csv"
REGISTERED_CSV_SHA256 = "2e13cbbdffca8917c5840193c3c1077926f4daf8f19b41a99d381df61c752e52"
LITERAL_M = 0.5
PATHS = ("crossing", "skimming")
PRESETS = ("clean", "iid_10")
DELAY_S = 0.0
N_SEEDS = 30
PLANNED_SIGMAS = (0.25, 0.5, 1.0, 2.0)
POSTHOC_MATCHED_SIGMAS = (4.0, 8.0)
POSTHOC_SPLIT_SIGMAS = (2.0, 4.0, 8.0)
LABEL_ORDER = ("arm", "sensor_sigma_m", "estimator_sigma_m", "adapter", "path", "preset", "delay_s", "seed")

_real_sensor_config = monitor.SensorConfig
_real_estimator_settings = monitor.EstimatorSettings


def arms(include_posthoc: bool = True) -> list[tuple[str, float, float]]:
    """(arm, sensor sigma, estimator sigma) for every arm, planned first."""
    result = [("planned", s, s) for s in PLANNED_SIGMAS]
    if include_posthoc:
        result += [("posthoc_matched", s, s) for s in POSTHOC_MATCHED_SIGMAS]
        result += [("posthoc_sensor", s, LITERAL_M) for s in POSTHOC_SPLIT_SIGMAS]
        result += [("posthoc_estimator", LITERAL_M, s) for s in POSTHOC_SPLIT_SIGMAS]
    return result


def base_scenarios(n_seeds: int = N_SEEDS) -> list[dict]:
    """The registered EXP-033 point-mass scenarios this sweep reruns, from ``build_jobs``."""
    registered = build_jobs(pilot=False)
    seed_order: list[int] = []
    for job in registered:
        seed = int(job["scenario"]["seed"])
        if seed not in seed_order:
            seed_order.append(seed)
    seeds = set(seed_order[:n_seeds])
    selected = [job for job in registered
                if job["labels"]["adapter"] == "point_mass"
                and job["labels"]["path"] in PATHS
                and job["labels"]["preset"] in PRESETS
                and job["labels"]["delay_s"] == DELAY_S
                and int(job["scenario"]["seed"]) in seeds]
    return selected


def build_noise_jobs(*, include_posthoc: bool = True, n_seeds: int = N_SEEDS) -> list[dict]:
    """One job per arm, sigma pair and registered scenario."""
    jobs = []
    for arm, sensor_sigma, estimator_sigma in arms(include_posthoc):
        for base in base_scenarios(n_seeds):
            labels = {"arm": arm, "sensor_sigma_m": sensor_sigma, "estimator_sigma_m": estimator_sigma,
                      **base["labels"]}
            job = make_job(labels, base["scenario"])
            job["noise"] = {"sensor_sigma_m": sensor_sigma, "estimator_sigma_m": estimator_sigma}
            jobs.append(job)
    return jobs


class _Patch:
    """Replace the two monitor-module class names for one run and record what was built."""

    def __init__(self, sensor_sigma_m: float, estimator_sigma_m: float):
        self.sensor_sigma_m = float(sensor_sigma_m)
        self.estimator_sigma_m = float(estimator_sigma_m)
        self.sensor_built: list[float] = []
        self.estimator_built: list[float] = []

    def sensor_config(self, *, noise_std_m, **kwargs):
        if float(noise_std_m) != LITERAL_M:
            raise RuntimeError(f"monitor.py sensor literal is {noise_std_m!r}, expected {LITERAL_M}")
        config = _real_sensor_config(noise_std_m=self.sensor_sigma_m, **kwargs)
        self.sensor_built.append(float(config.noise_std_m))
        return config

    def estimator_settings(self, *, measurement_std_m, **kwargs):
        if float(measurement_std_m) != LITERAL_M:
            raise RuntimeError(f"monitor.py estimator literal is {measurement_std_m!r}, expected {LITERAL_M}")
        settings = _real_estimator_settings(measurement_std_m=self.estimator_sigma_m, **kwargs)
        self.estimator_built.append(float(settings.measurement_std_m))
        return settings

    def __enter__(self):
        monitor.SensorConfig = self.sensor_config
        monitor.EstimatorSettings = self.estimator_settings
        return self

    def __exit__(self, *exc):
        monitor.SensorConfig = _real_sensor_config
        monitor.EstimatorSettings = _real_estimator_settings
        return False


def run_noise_job(job: dict) -> dict:
    """Worker: run one job with the job's two sigma values in effect, recording what was built."""
    noise = job["noise"]
    with _Patch(noise["sensor_sigma_m"], noise["estimator_sigma_m"]) as patch:
        row = run_job(job)
    row["n_sensor_configs_built"] = len(patch.sensor_built)
    row["n_estimator_settings_built"] = len(patch.estimator_built)
    row["effective_sensor_sigma_m"] = (patch.sensor_built[0] if len(set(patch.sensor_built)) == 1 else None)
    row["effective_estimator_sigma_m"] = (
        patch.estimator_built[0] if len(set(patch.estimator_built)) == 1 else None)
    return row


def effective_sigma_check(rows: list[dict]) -> dict:
    """Every row built two sensor configs and one estimator setting, each at its labelled sigma."""
    bad = [row for row in rows
           if row["n_sensor_configs_built"] != 2 or row["n_estimator_settings_built"] != 1
           or row["effective_sensor_sigma_m"] != row["sensor_sigma_m"]
           or row["effective_estimator_sigma_m"] != row["estimator_sigma_m"]]
    built = group_rows(rows, ("arm", "effective_sensor_sigma_m", "effective_estimator_sigma_m"))
    return {"n_rows": len(rows), "n_rows_wrong": len(bad), "passed": not bad,
            "first_wrong": [{key: row[key] for key in LABEL_ORDER} for row in bad[:5]],
            "by_arm": [[arm, sensor, estimator, len(group)] for (arm, sensor, estimator), group in built.items()]}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_text(ours: str, registered: str) -> bool:
    """How the control compares a value with the registered text: character for character."""
    return ours == registered


def control_check(rows: list[dict], registered_csv: Path = REGISTERED_CSV) -> dict:
    """Planned sigma 0.5 rows against the registered rows, column by column, as CSV text."""
    with open(registered_csv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        registered = {(r["adapter"], r["path"], r["preset"], r["delay_s"], r["seed"]): r for r in reader}
    control = [row for row in rows if row["arm"] == "planned" and row["sensor_sigma_m"] == LITERAL_M]
    mismatches = []
    missing = 0
    for row in control:
        key = tuple(_cell_text(row[name]) for name in ("adapter", "path", "preset", "delay_s", "seed"))
        reference = registered.get(key)
        if reference is None:
            missing += 1
            continue
        for column in columns:
            ours = _cell_text(row.get(column))
            if not _same_text(ours, reference[column]):
                mismatches.append({"key": list(key), "column": column, "ours": ours, "registered": reference[column]})
    return {
        "registered_csv": str(Path(registered_csv).relative_to(ROOT)) if Path(registered_csv).is_relative_to(ROOT)
        else str(registered_csv),
        "registered_csv_sha256": sha256(registered_csv),
        "registered_csv_sha256_expected": REGISTERED_CSV_SHA256,
        "columns_compared": columns,
        "n_control_rows": len(control),
        "n_rows_without_registered_partner": missing,
        "n_mismatched_values": len(mismatches),
        "first_mismatches": mismatches[:10],
        "passed": bool(control) and missing == 0 and not mismatches,
    }


def _cell_summary(cell: list[dict]) -> dict:
    latencies = [row["alert_latency_s"] for row in cell]
    return {
        "n_runs": len(cell),
        "false_alert": rate_summary([bool(row["false_alert"]) for row in cell]),
        "missed_alert": rate_summary([bool(row["missed_alert"]) for row in cell]),
        "alert_latency_s": median_summary(latencies),
        "alert_latency_s_range": ([min(v for v in latencies if v is not None), max(v for v in latencies if v is not None)]
                                  if any(v is not None for v in latencies) else None),
        "alerts_raised": median_summary([float(row["alerts_raised"]) for row in cell]),
        "max_alerts_raised": max(int(row["alerts_raised"]) for row in cell),
        "continuity": median_summary([row["continuity"] for row in cell]),
        "n_runs_with_observer_landing": sum(1 for row in cell if int(row["observer_landings"]) > 0),
    }


def first_failing_sigma(cells: list[dict], arm_names: tuple[str, ...], vary: str) -> dict:
    """Per path: the smallest varied sigma with any false or missed alert, over the given arms (both presets)."""
    result = {}
    for path in PATHS:
        failing = sorted({cell[vary] for cell in cells
                          if cell["arm"] in arm_names and cell["path"] == path
                          and (cell["metrics"]["false_alert"]["count"] > 0
                               or cell["metrics"]["missed_alert"]["count"] > 0)})
        result[path] = failing[0] if failing else None
    return result


def summarize(rows: list[dict]) -> dict:
    keys = ("arm", "sensor_sigma_m", "estimator_sigma_m", "path", "preset")
    cells = []
    for key, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, key))
        item["metrics"] = _cell_summary(grouped)
        cells.append(item)
    matched_arms = ("planned", "posthoc_matched")
    return {
        "experiment": "Side run on EXP-033 (monitor mission), sensor noise varied",
        "simulated": True,
        "fixed": {"adapter": "point_mass", "delay_s": DELAY_S, "n_seeds": N_SEEDS,
                  "seeds": "seeds_for(33, 30), the first 30 of the registered 50"},
        "posthoc_arms": ["posthoc_matched", "posthoc_sensor", "posthoc_estimator"],
        "posthoc_note": ("Only the planned arm is in the pre-registered plan. The post hoc arms were added before any sim run, "
                         "after an independent hand check said the planned grid would not break."),
        "control": control_check(rows),
        "effective_sigma": effective_sigma_check(rows),
        "first_failing_sigma": {
            "planned_only": first_failing_sigma(cells, ("planned",), "sensor_sigma_m"),
            "matched_including_posthoc": first_failing_sigma(cells, matched_arms, "sensor_sigma_m"),
            "sensor_varied_estimator_0.5": first_failing_sigma(cells, ("posthoc_sensor",), "sensor_sigma_m"),
            "estimator_varied_sensor_0.5": first_failing_sigma(cells, ("posthoc_estimator",), "estimator_sigma_m"),
        },
        "cells": cells,
        "limits": list(MISSION_LIMITS) + [
            ("The unknown drone flies straight at constant speed, so the filter's constant-velocity model is exact "
             "and a large measurement setting costs only lag, not model error."),
            "Sensor noise is white Gaussian per axis; no bias, no dropout beyond the link preset.",
        ],
    }


def _check_output(output: Path) -> None:
    resolved = Path(output).resolve()
    runs = (ROOT / "sim" / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        raise SystemExit(f"{output} is inside sim/runs; side runs never write there")


def run_sweep(output: Path, *, include_posthoc: bool = True, n_seeds: int = N_SEEDS,
              n_processes: int | None = None, argv: list[str] | None = None) -> dict:
    _check_output(output)
    output = prepare_output(Path(output))
    start = time.perf_counter()
    jobs = build_noise_jobs(include_posthoc=include_posthoc, n_seeds=n_seeds)
    rows = run_jobs(jobs, n_processes=processes(n_processes), worker=run_noise_job)
    write_rows(output / "monitor_noise.csv", rows, first_columns=LABEL_ORDER)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    raw_argv = list(argv or [])
    write_manifest(output, schema="zephyr-monitor-noise-1", argv=raw_argv,
                   command=replay_command("tools/sens_monitor_noise.py", raw_argv),
                   config={"include_posthoc": include_posthoc, "n_seeds": n_seeds, "delay_s": DELAY_S,
                           "arms": [list(arm) for arm in arms(include_posthoc)]},
                   seeds=sorted({int(job["scenario"]["seed"]) for job in jobs}),
                   outputs=["monitor_noise.csv", "summary.json"], limits=summary["limits"],
                   runtime_s=time.perf_counter() - start, n_processes=processes(n_processes),
                   hashed_sources=["sim/missions/monitor.py", "tools/e_exp033_monitor.py", "tools/e_common.py",
                                   "tools/sens_monitor_noise.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = argv_or_default(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--planned-only", action="store_true", help="skip the post hoc arms")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="seeds per cell (pilot: fewer)")
    args = parser.parse_args(raw_argv)
    summary = run_sweep(args.output, include_posthoc=not args.planned_only, n_seeds=args.seeds,
                        n_processes=args.processes, argv=raw_argv)
    print("control passed:", summary["control"]["passed"],
          "| effective sigma passed:", summary["effective_sigma"]["passed"])
    print("effective sigma built inside the worker runs (arm: sensor m, estimator m, runs):")
    for arm, sensor_sigma, estimator_sigma, n_runs in summary["effective_sigma"]["by_arm"]:
        print(f"  {arm}: {sensor_sigma}, {estimator_sigma}, {n_runs}")
    print("first failing sigma:", summary["first_failing_sigma"])


if __name__ == "__main__":
    main()
