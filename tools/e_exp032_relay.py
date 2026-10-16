"""EXP-032 relay range runner for the simulated relay mission.

Usage: PYTHONPATH=. ./.venv/bin/python tools/e_exp032_relay.py --output runs/e-exp032 --pilot

The mission owns radio physics, retries, forwarding, and telemetry timing. This
runner only constructs the preregistered cells, aggregates returned metrics,
and writes reproducible tables, summaries, and figures.
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
import numpy as np

from sim.missions.api import Scenario
from sim.missions.relay import RelayMission
from tests.reference_e.hand_checks import max_range_dual_slope_m
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (MISSION_LIMITS, MISSION_SOURCES, group_rows, make_job,
                            apply_mission_style, median_summary, processes, replay_command,
                            run_jobs, seeds_for, series_colors, write_rows)

DISTANCES = (300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0)
FRACTIONS = (None, 0.25, 0.4, 0.5, 0.6, 0.75)
FULL_SEEDS = 50
PILOT_SEEDS = 3
PHYSICS_DISTANCES = (500.0, 900.0)
PHYSICS_FRACTIONS = (None, 0.5)

PREDICTION_SOURCE = "pre-registered before the runs (EXP-032 section of the private pre-registration)"
PREDICTION_TEXT = ("d_max with 0 dB margin is about 600 to 700 m (Run C: d90 409 m, "
                   "d10 746 m for the stress corner); direct delivery near 100 % at 300 m, near 0 beyond "
                   "about 900 m; f = 0.5 keeps delivery above 99 % to about 1100 m; asymmetric relays "
                   "(0.25, 0.75) fail first on the long hop. Latency roughly doubles with the relay (two hops, "
                   "plus retries on the weaker hop) but stays under 10 ms median.")
PREDICTION_CORRECTION_3 = ("3. EXP-032 prediction: the median-signal range for the stress corner with 0 dB margin "
                           "and -98 dBm sensitivity is about 518 m (reference computation), not 600 to 700 m. "
                           "Corrected: direct delivery falls through 50 % somewhere around 500 to 750 m "
                           "(shadowing and 5 retries push it outward); a relay at f = 0.5 holds high delivery to "
                           "about 1000 to 1300 m.")


def build_jobs(*, pilot: bool = False, duration_s: float = 60.0,
               seed_count: int | None = None) -> tuple[list[dict], list[int]]:
    """Build point-mass jobs followed by the preregistered physics subset."""
    n = PILOT_SEEDS if seed_count is None and pilot else FULL_SEEDS if seed_count is None else int(seed_count)
    seeds = seeds_for(32, n)
    jobs: list[dict] = []
    distances = (500.0, 900.0) if pilot else DISTANCES
    fractions = (None, 0.5) if pilot else FRACTIONS
    for distance in distances:
        for fraction in fractions:
            for seed in seeds:
                params = {"distance_m": distance, "relay_fraction": fraction}
                labels = {"distance_m": distance, "relay_fraction": "none" if fraction is None else fraction,
                          "adapter": "point_mass"}
                scenario = {"mission": "relay", "seed": seed, "duration_s": float(duration_s),
                            "adapter": "point_mass", "params": params}
                jobs.append(make_job(labels, scenario))
    if pilot:
        physics_seeds = seeds[:2]
        physics_cells = ((900.0, 0.5),)
    else:
        physics_seeds = seeds[:20]
        physics_cells = tuple((d, f) for d in PHYSICS_DISTANCES for f in PHYSICS_FRACTIONS)
    for distance, fraction in physics_cells:
        for seed in physics_seeds:
            params = {"distance_m": distance, "relay_fraction": fraction}
            labels = {"distance_m": distance, "relay_fraction": "none" if fraction is None else fraction,
                      "adapter": "physics"}
            scenario = {"mission": "relay", "seed": seed, "duration_s": float(duration_s),
                        "adapter": "physics", "params": params}
            jobs.append(make_job(labels, scenario))
    return jobs, seeds


def _cell_summary(rows: list[dict]) -> dict:
    values = [float(row["delivery_ratio"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "delivery_ratio_mean": float(np.mean(values)),
        "delivery_ratio_interval": median_summary(values),
        "median_latency_s": median_summary([row["median_latency_s"] for row in rows]),
        "p95_latency_s": median_summary([row["p95_latency_s"] for row in rows]),
        "hop1_delivery": median_summary([row["hop1_delivery"] for row in rows]),
        "hop2_delivery": median_summary([row["hop2_delivery"] for row in rows]),
        "retries_mean": median_summary([row["retries_mean"] for row in rows]),
    }


def _values_equal(left, right) -> bool:
    """Equality for metric values where two missing numbers (NaN) count as equal."""
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True
    return left == right


def physics_matches_point_mass(cell_rows: list[dict], all_rows: list[dict]) -> bool:
    """True when every physics row equals the point-mass row of the same D, f and seed on every metric."""
    label_keys = {"adapter", "distance_m", "relay_fraction", "seed"}
    point_by_key = {(r["distance_m"], r["relay_fraction"], r["seed"]): r
                    for r in all_rows if r["adapter"] == "point_mass"}
    for row in cell_rows:
        partner = point_by_key.get((row["distance_m"], row["relay_fraction"], row["seed"]))
        if partner is None:
            return False
        metric_keys = set(row) | set(partner)
        for key in sorted(metric_keys - label_keys):
            if key not in row or key not in partner or not _values_equal(row[key], partner[key]):
                return False
    return True


def _interpolated_crossing(cell_values: dict[float, float]) -> float | None:
    ordered = sorted(cell_values.items())
    for (d0, v0), (d1, v1) in zip(ordered, ordered[1:]):
        if (v0 - 0.5) * (v1 - 0.5) <= 0.0 and v0 != v1:
            return float(d0 + (0.5 - v0) * (d1 - d0) / (v1 - v0))
    return None


def _make_figures(output: Path, rows: list[dict]) -> None:
    apply_mission_style("light")
    colors = series_colors(6)
    point = [r for r in rows if r["adapter"] == "point_mass"]
    grouped = group_rows(point, ("relay_fraction", "distance_m"))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for index, fraction in enumerate(("none", 0.25, 0.4, 0.5, 0.6, 0.75)):
        means = []
        for distance in DISTANCES:
            cell = grouped.get((fraction, distance), [])
            if cell:
                means.append((distance, float(np.mean([r["delivery_ratio"] for r in cell]))))
        if means:
            ax.plot([x for x, _ in means], [y for _, y in means], marker="o", color=colors[index], label=f"f={fraction}")
    ax.set(xlabel="far-drone distance (m)", ylabel="delivery ratio", title="Relay delivery ratio, simulated")
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "delivery_vs_distance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for index, fraction in enumerate(("none", 0.5)):
        means = []
        for distance in DISTANCES:
            cell = grouped.get((fraction, distance), [])
            if cell:
                vals = [r["median_latency_s"] for r in cell if r["median_latency_s"] is not None]
                if vals:
                    means.append((distance, float(np.median(vals))))
        if means:
            ax.plot([x for x, _ in means], [y for _, y in means], marker="o", color=colors[index], label=f"f={fraction}")
    ax.set(xlabel="far-drone distance (m)", ylabel="median latency (s)", title="Relay median latency, simulated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_vs_distance.png", dpi=160)
    plt.close(fig)


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   duration_s: float = 60.0, seed_count: int | None = None,
                   argv: list[str] | None = None) -> dict:
    output_arg = output
    output = prepare_output(Path(output))
    jobs, seeds = build_jobs(pilot=pilot, duration_s=duration_s, seed_count=seed_count)
    nproc = processes(n_processes)
    started = time.perf_counter()
    rows = run_jobs(jobs, n_processes=nproc)
    runtime = time.perf_counter() - started
    write_rows(output / "relay.csv", rows,
               first_columns=("adapter", "distance_m", "relay_fraction", "seed"))
    cells = {}
    for key, cell_rows in group_rows(rows, ("adapter", "distance_m", "relay_fraction")).items():
        cell = _cell_summary(cell_rows)
        if key[0] == "physics":
            cell["physics_matches_point_mass"] = physics_matches_point_mass(cell_rows, rows)
        cells["|".join(str(v) for v in key)] = cell
    breakpoint_m = 4.0 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)
    reference = max_range_dual_slope_m(19.5 - 6.0, -98.0, 40.2, 2.0, breakpoint_m, 4.0, 0.0)
    direct = {float(d): float(np.mean([r["delivery_ratio"] for r in rows
                                        if r["adapter"] == "point_mass" and r["relay_fraction"] == "none" and r["distance_m"] == d]))
              for d in DISTANCES if any(r["adapter"] == "point_mass" and r["relay_fraction"] == "none" and r["distance_m"] == d for r in rows)}
    midpoint = {float(d): float(np.mean([r["delivery_ratio"] for r in rows
                                          if r["adapter"] == "point_mass" and r["relay_fraction"] == 0.5 and r["distance_m"] == d]))
                for d in DISTANCES if any(r["adapter"] == "point_mass" and r["relay_fraction"] == 0.5 and r["distance_m"] == d for r in rows)}
    point_cell_keys = {(row["distance_m"], row["relay_fraction"]) for row in rows if row["adapter"] == "point_mass"}
    physics_cell_keys = {(row["distance_m"], row["relay_fraction"]) for row in rows if row["adapter"] == "physics"}
    point_seed_count = (sum(1 for row in rows if row["adapter"] == "point_mass") // len(point_cell_keys)
                        if point_cell_keys else 0)
    physics_seed_count = (sum(1 for row in rows if row["adapter"] == "physics") // len(physics_cell_keys)
                          if physics_cell_keys else 0)
    deviations = []
    if pilot:
        deviations.append("Pilot uses the preregistered subset and is not the full grid.")
    if duration_s != 60.0:
        deviations.append("Run duration was shortened for a test invocation.")
    summary = {
        "experiment": "EXP-032",
        "pilot": bool(pilot),
        "duration_s": float(duration_s),
        "grid": {"distances_m": list(DISTANCES), "relay_fractions": ["none", *FRACTIONS[1:]],
                 "point_mass_seeds": FULL_SEEDS, "physics_distances_m": list(PHYSICS_DISTANCES),
                 "physics_fractions": ["none", 0.5], "physics_seeds": 20,
                 "actual_point_mass_seeds": point_seed_count, "actual_physics_seeds": physics_seed_count},
        "mission_defaults": dict(RelayMission.defaults),
        "simulated": True,
        "limits": list(MISSION_LIMITS) + list(RelayMission.limits),
        "cells": cells,
        "hand_check": {"reference_median_signal_range_m": reference,
                        "two_d_max_m": 2.0 * reference,
                        "direct_50_percent_crossing_m_interpolation": _interpolated_crossing(direct),
                        "relay_f_0_5_50_percent_crossing_m_interpolation": _interpolated_crossing(midpoint)},
        "prediction": PREDICTION_TEXT,
        "prediction_source": PREDICTION_SOURCE,
        "prediction_correction_3": PREDICTION_CORRECTION_3,
        "prediction_observed": {"text": PREDICTION_CORRECTION_3,
                       "observed_direct": direct, "observed_f_0_5": midpoint},
        "physics_geometry_note": ("Physics rows equal point-mass rows by construction: path loss uses the fixed "
                                  "commanded positions (sim/missions/relay.py limits), so the physics adapter "
                                  "only changes the vehicles, not the radio geometry."),
        "missing_metrics": ["retries_mean: RadioMissionLink does not expose per-packet retry counts; mission returns null."],
        "deviations": deviations,
    }
    write_json(output / "summary.json", summary)
    _make_figures(output, rows)
    if argv is None:
        # Called from code: there is no command line, so record none and name a replay command.
        actual_argv = []
        replay_argv = ["--output", str(output_arg)] + (["--pilot"] if pilot else [])
    else:
        actual_argv = list(argv)
        replay_argv = actual_argv
    write_manifest(output, schema="zephyr-exp032-runner-1", argv=actual_argv,
                   command=replay_command("tools/e_exp032_relay.py", replay_argv), config=summary["grid"],
                   seeds=seeds, outputs=["relay.csv", "summary.json", "delivery_vs_distance.png", "latency_vs_distance.png"],
                   limits=list(MISSION_LIMITS), n_processes=nproc, runtime_s=runtime,
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/relay.py", "tools/e_exp032_relay.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    actual_argv = argv_or_default(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args(actual_argv)
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes, argv=actual_argv)


if __name__ == "__main__":
    main()
