"""EXP-033 simulated monitoring runner.

Usage: ``PYTHONPATH=. python tools/e_exp033_monitor.py --output runs/e-exp033``

The scenario is simulated only. Observers hold fixed posts and alert humans
when the estimated unknown drone is in the alert zone.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.missions.engine import COMMON_LIMITS
from tests.reference_e.hand_checks import alert_latency_s, expected_detection_s
from tools.c_common import prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    group_rows,
    make_job,
    median_summary,
    processes,
    rate_summary,
    run_jobs,
    seeds_for,
    write_rows,
)
from tools.horizon_mpl import apply_style


PATHS = ("crossing", "skimming", "loiter")
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
DELAYS = (0.0, 0.2, 0.5)
# The unknown drone flies at 3 m/s across the 60 m venue, so 20 s covers the
# whole crossing and skimming pass (zone edge at x = 20 to 40 m, t = 6.7 to
# 13.3 s) and about 6.7 s of the loiter circle (it starts circling at 11.3 s).
DURATION_S = 20.0
FIX_PERIOD_S = 0.1
DEBOUNCE_K = 3
SAFETY_LIMIT_M = 0.5
PREDICTION = (
    "Claude's prediction: clean: latency about 0.25 to 0.4 s, no misses; "
    "skimming at 2 m outside with 0.5 m noise (fused about 0.35 m) gives a few "
    "percent false alerts at most (3 consecutive needed). Loss raises latency by "
    "about the mean gap; bursts raise it by about the burst length (0.5 s) and "
    "cut continuity; misses only on burst_30 at long delay."
)


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build the preregistered point mass jobs and physics subset."""
    seeds = seeds_for(33, 3 if pilot else 50)
    jobs: list[dict] = []
    if pilot:
        paths, presets, delays = ("crossing", "skimming"), ("clean", "burst_10_0.5s"), (0.0, 0.5)
    else:
        paths, presets, delays = PATHS, PRESETS, DELAYS
    for path in paths:
        for preset in presets:
            for delay in delays:
                for seed in seeds:
                    labels = {"path": path, "preset": preset, "delay_s": delay, "adapter": "point_mass"}
                    scenario = {
                        "mission": "monitor", "seed": seed, "duration_s": DURATION_S,
                        "adapter": "point_mass", "preset": preset,
                        "measurement_delay_s": delay, "params": {"path": path},
                    }
                    jobs.append(make_job(labels, scenario))
    physics_seeds = seeds[:2] if pilot else seeds[:20]
    physics_cells = (("crossing", "clean"),) if pilot else tuple((p, preset) for p in PATHS for preset in ("clean", "burst_10_0.5s"))
    for path, preset in physics_cells:
        for seed in physics_seeds:
            labels = {"path": path, "preset": preset, "delay_s": 0.0, "adapter": "physics"}
            scenario = {
                "mission": "monitor", "seed": seed, "duration_s": DURATION_S,
                "adapter": "physics", "preset": preset,
                "measurement_delay_s": 0.0, "params": {"path": path},
            }
            jobs.append(make_job(labels, scenario))
    return jobs


build_grid = build_jobs


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n_runs": len(rows),
        "continuity": median_summary([row["continuity"] for row in rows]),
        "alert_latency_s": median_summary([row["alert_latency_s"] for row in rows]),
        "false_alert": rate_summary([bool(row["false_alert"]) for row in rows]),
        "missed_alert": rate_summary([bool(row["missed_alert"]) for row in rows]),
        "max_estimate_age_s": median_summary([row["max_estimate_age_s"] for row in rows]),
        "alerts_raised": median_summary([float(row["alerts_raised"]) for row in rows]),
        "observer_max_horizontal_displacement_m": {
            "maximum": max(float(row["observer_max_horizontal_displacement_m"]) for row in rows),
            "under_limit": all(float(row["observer_max_horizontal_displacement_m"]) < SAFETY_LIMIT_M for row in rows),
        },
        "observer_max_displacement_m": max(float(row["observer_max_displacement_m"]) for row in rows),
        "runs_with_observer_landing": rate_summary([int(row["observer_landings"]) > 0 for row in rows]),
    }


def _summary(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("adapter", "path", "preset", "delay_s")
    cells = []
    for key, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, key))
        item["metrics"] = _cell_summary(grouped)
        cells.append(item)
    safety_max = max(float(row["observer_max_horizontal_displacement_m"]) for row in rows)
    detection_s = expected_detection_s(FIX_PERIOD_S, DEBOUNCE_K)
    observed = []
    for delay in DELAYS:
        selected = [r["alert_latency_s"] for r in rows
                    if r["adapter"] == "point_mass" and r["path"] == "crossing"
                    and r["preset"] == "clean" and r["delay_s"] == delay and r["alert_latency_s"] is not None]
        if selected:
            observed.append({"delay_s": delay, "observed_median_s": median_summary(selected)["median"],
                             "hand_check_s": alert_latency_s(detection_s, delay, 0.0)})
    return {
        "experiment": "EXP-033",
        "pilot": bool(pilot),
        "simulated": True,
        "cells": cells,
        "hand_check": {"formula": "(k - 0.5) * 0.1 + delay + estimator_lag (lag taken as 0)", "k": DEBOUNCE_K,
                        "expected_by_delay_s": [{"delay_s": d, "value_s": alert_latency_s(detection_s, d, 0.0)} for d in DELAYS],
                        "observed_clean_crossing": observed},
        "prediction": PREDICTION,
        "safety_check": {"limit_m": SAFETY_LIMIT_M,
                         "quantity": "maximum horizontal distance of any observer from its post, over every run",
                         "maximum_observer_horizontal_displacement_m": safety_max,
                         "passed": safety_max < SAFETY_LIMIT_M,
                         "flag": "OBSERVER LEFT ITS POST SIDEWAYS" if safety_max >= SAFETY_LIMIT_M else None,
                         "note": ("A lost-link failsafe landing is straight down; it shows in observer_max_displacement_m "
                                  "and runs_with_observer_landing, and a landing observer stops sending fixes.")},
        "deviations": [],
        "missing_metrics": [],
        "limits": list(MISSION_LIMITS + COMMON_LIMITS),
    }


def _figures(output: Path, rows: list[dict]) -> list[str]:
    colors = apply_style("light")
    names = []
    for path in ("crossing", "loiter"):
        fig, ax = plt.subplots()
        for preset in PRESETS:
            x, y = [], []
            for delay in DELAYS:
                vals = [r["alert_latency_s"] for r in rows if r["adapter"] == "point_mass" and r["path"] == path and r["preset"] == preset and r["delay_s"] == delay and r["alert_latency_s"] is not None]
                if vals:
                    x.append(delay); y.append(median_summary(vals)["median"])
            if x:
                ax.plot(x, y, marker="o", label=preset)
        ax.set_xlabel("measurement delay (s)"); ax.set_ylabel("alert latency (s)")
        ax.set_title(f"Alert latency simulated: {path}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        name = f"alert_latency_{path}.png"; fig.savefig(output / name, dpi=140); plt.close(fig); names.append(name)
    fig, ax = plt.subplots()
    for preset in PRESETS:
        vals = []
        labels = []
        for path in PATHS:
            selected = [r["continuity"] for r in rows if r["adapter"] == "point_mass" and r["path"] == path and r["preset"] == preset]
            if selected:
                labels.append(path); vals.append(median_summary(selected)["median"])
        if vals:
            ax.plot(labels, vals, marker="o", label=preset)
    ax.set_xlabel("path"); ax.set_ylabel("continuity")
    ax.set_title("Continuity simulated")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    name = "continuity_simulated.png"; fig.savefig(output / name, dpi=140); plt.close(fig); names.append(name)
    return names


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   jobs: list[dict] | None = None) -> dict:
    output = prepare_output(Path(output))
    start = time.perf_counter()
    run_list = build_jobs(pilot=pilot) if jobs is None else list(jobs)
    rows = run_jobs(run_list, n_processes=processes(n_processes))
    write_rows(output / "exp033.csv", rows, first_columns=("adapter", "path", "preset", "delay_s", "seed"))
    summary = _summary(rows, pilot=pilot)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows)
    outputs = ["exp033.csv", "summary.json", *figures]
    elapsed = time.perf_counter() - start
    write_manifest(output, schema="zephyr-exp033-1", argv=[], command="python tools/e_exp033_monitor.py", config={"pilot": pilot, "duration_s": DURATION_S}, seeds=seeds_for(33, 3 if pilot else 50), outputs=outputs, limits=list(MISSION_LIMITS), runtime_s=elapsed, n_processes=processes(n_processes), hashed_sources=list(MISSION_SOURCES) + ["sim/missions/monitor.py", "tools/e_exp033_monitor.py"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes)


if __name__ == "__main__":
    main()
