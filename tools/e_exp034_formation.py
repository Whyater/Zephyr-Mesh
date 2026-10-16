"""EXP-034 gym formation demo for simulated missions.

Usage (repo root):
    PYTHONPATH=. ./.venv/bin/python tools/e_exp034_formation.py --output runs/e-exp034-formation --pilot

All motion, camera fixes, delays, and link loss are simulated.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.reference_e.hand_checks import first_order_loop_jitter_std
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    apply_mission_style,
    group_rows,
    make_job,
    median_summary,
    processes,
    replay_command,
    rate_summary,
    run_jobs,
    seeds_for,
    series_colors,
    write_rows,
)

EXP_NUMBER = 34
SIGMAS = (0.01, 0.02, 0.05, 0.10)
LATENCIES = (0.0, 0.05, 0.1, 0.2)
PRESETS = ("clean", "iid_10")
SEEDS = seeds_for(EXP_NUMBER, 50)
PREDICTION_TEXT = (
    "Claude's prediction: point mass at zero latency matches the hand formula within 10 %; latency up to 0.2 s raises the error by under 30 % "
    "(the first-order loop with delay stays stable below about (pi/2) tau = 0.39 s, but the margin shrinks); physics adapter errors larger than the point mass at the same sigma "
    "(PD loop with lower damping passes more noise) and grows faster with latency (Run A: position feedback edge 0.33 s). Separation never near 0.5 m (slots 2 m apart, errors of centimetres)."
)


def _scenario(seed: int, *, sigma: float, latency: float, preset: str, adapter: str,
              duration_s: float = 30.0) -> dict:
    return {
        "mission": "formation",
        "seed": int(seed),
        "duration_s": float(duration_s),
        "adapter": adapter,
        "preset": preset,
        "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": float(sigma),
                   "fix_latency_s": float(latency)},
    }


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build jobs in stable grid order.

    Full grid arithmetic: point mass 4 sigma x 4 latency x 2 presets x 50
    seeds = 1600, physics 2 sigma x 2 latency x 2 presets x 50 = 400.
    """
    jobs: list[dict] = []
    if pilot:
        for sigma in (0.02, 0.10):
            for latency in (0.0, 0.2):
                for seed_index, seed in enumerate(SEEDS[:3]):
                    labels = {"slice": "pilot", "sigma_m": sigma, "fix_latency_s": latency,
                              "preset": "clean", "adapter": "point_mass_offboard", "seed_index": seed_index}
                    jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency,
                                                           preset="clean", adapter="point_mass_offboard", duration_s=10.0)))
        for seed_index, seed in enumerate(SEEDS[:2]):
            labels = {"slice": "pilot_physics", "sigma_m": 0.05, "fix_latency_s": 0.1,
                      "preset": "clean", "adapter": "physics_offboard", "seed_index": seed_index}
            jobs.append(make_job(labels, _scenario(seed, sigma=0.05, latency=0.1,
                                                   preset="clean", adapter="physics_offboard", duration_s=10.0)))
        return jobs

    for adapter, sigmas, latencies in (
        ("point_mass_offboard", SIGMAS, LATENCIES),
        ("physics_offboard", (0.02, 0.05), (0.0, 0.1)),
    ):
        for sigma in sigmas:
            for latency in latencies:
                for preset in PRESETS:
                    for seed_index, seed in enumerate(SEEDS):
                        labels = {"slice": "main" if adapter == "point_mass_offboard" else "physics",
                                  "sigma_m": sigma, "fix_latency_s": latency, "preset": preset,
                                  "adapter": adapter, "seed_index": seed_index}
                        jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency,
                                                               preset=preset, adapter=adapter)))
    return jobs


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "rms_error_m": median_summary([r["rms_error_m"] for r in rows]),
        "p95_error_m": median_summary([r["p95_error_m"] for r in rows]),
        "per_axis_std_m": median_summary([r["per_axis_std_m"] for r in rows]),
        "minimum_pairwise_separation_m": median_summary([r["min_separation_m"] for r in rows]),
        "min_separation_min_m": min(r["min_separation_m"] for r in rows),
        "guard_activations": {"total": sum(int(r["guard_activations"]) for r in rows),
                              "runs_with_any": rate_summary([bool(r["guard_activations"]) for r in rows])},
        "downwash_wind_max_mps": median_summary([r["downwash_wind_max_mps"] for r in rows]),
    }


def summarize(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("slice", "sigma_m", "fix_latency_s", "preset", "adapter")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        item = {key: value for key, value in zip(keys, coords)}
        item.update(_cell_summary(grouped))
        cells.append(item)

    hand = []
    for adapter, sigmas in (("point_mass_offboard", SIGMAS), ("physics_offboard", (0.02, 0.05))):
        for sigma in sigmas:
            selected = [r for r in rows if r["preset"] == "clean" and r["adapter"] == adapter
                        and r["sigma_m"] == sigma and r["fix_latency_s"] == 0.0]
            if selected:
                observed_axis = median_summary([r["per_axis_std_m"] for r in selected])
                observed_rms = median_summary([r["rms_error_m"] for r in selected])
                reference_axis = first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
                reference_rms = float(np.sqrt(3.0) * reference_axis)
                ratio = None if observed_rms["median"] is None else observed_rms["median"] / reference_rms
                hand.append({"adapter": adapter, "sigma_m": sigma,
                             "observed_per_axis_std_m": observed_axis,
                             "reference_per_axis_std_m": reference_axis,
                             "observed_to_reference": None if observed_axis["median"] is None else observed_axis["median"] / reference_axis,
                             "observed_to_reference_per_axis": None if observed_axis["median"] is None else observed_axis["median"] / reference_axis,
                             "observed_rms_error_m": observed_rms,
                             "reference_rms_error_m": reference_rms,
                             "observed_to_reference_rms": ratio,
                             "within_10_percent": None if ratio is None else abs(ratio - 1.0) <= 0.10})
    return {
        "experiment": "EXP-034", "simulated": True, "pilot": bool(pilot),
        "grid_counts": {"point_mass_offboard": 4 * 4 * 2 * 50,
                         "physics_offboard": 2 * 2 * 2 * 50},
        "cells": cells,
        "hand_check": {"fix_period_s": 1.0 / 30.0, "response_time_s": 0.25,
                        "hand_formula": "per-axis std = sigma * sqrt(a / (2 - a)), a = 1 - exp(-(1/30)/0.25)",
                        "zero_latency_clean": hand},
        "notes": [
            "min_separation_m is taken over the whole run, including the start from randomly offset positions.",
            "The offboard physics adapter differences the last two position fixes for velocity, a contract design choice. At 30 Hz, sigma of fix noise becomes about sqrt(2) x 30 x sigma of velocity noise.",
        ],
        "prediction": PREDICTION_TEXT,
        "prediction_observations": {
            "observed_cells": [{key: value for key, value in cell.items()
                                if key in ("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "rms_error_m", "minimum_pairwise_separation_m")}
                               for cell in cells],
        },
        "deviations": [],
        "missing_metrics": [], "limits": list(MISSION_LIMITS),
    }


def _figures(output: Path, rows: list[dict]) -> list[str]:
    apply_mission_style("light")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, adapter in zip(axes, ("point_mass_offboard", "physics_offboard")):
        sigmas = SIGMAS if adapter == "point_mass_offboard" else (0.02, 0.05)
        for series_index, sigma in enumerate(sigmas):
            values = []
            for latency in LATENCIES:
                selected = [r for r in rows if r["adapter"] == adapter and r["preset"] == "clean"
                            and r["sigma_m"] == sigma and r["fix_latency_s"] == latency]
                values.append(median_summary([r["rms_error_m"] for r in selected])["median"] if selected else float("nan"))
            ax.plot(LATENCIES, values, marker="o", color=series_colors(len(sigmas))[series_index], label=f"sigma {sigma:g} m")
            reference = (3.0 ** 0.5) * first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
            if values and not all(v != v for v in values):
                ax.plot([0.0], [reference], marker="x", color="#555555")
        ax.set_title(("point mass" if adapter == "point_mass_offboard" else "physics") + " simulated")
        ax.set_xlabel("fix latency (s)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("formation RMS error (m)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    name = "formation_rms_latency.png"
    fig.savefig(output / name, dpi=140)
    plt.close(fig)
    return [name]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args_list = argv_or_default(argv)
    args = parser.parse_args(args_list)
    output = prepare_output(args.output)
    started = time.perf_counter()
    jobs = build_jobs(pilot=args.pilot)
    rows = run_jobs(jobs, n_processes=processes(args.processes))
    csv_name = "formation.csv"
    write_rows(output / csv_name, rows, first_columns=("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "seed_index", "seed"))
    write_json(output / "summary.json", summarize(rows, pilot=args.pilot))
    figures = _figures(output, rows)
    outputs = [csv_name, "summary.json", *figures]
    write_manifest(output, schema="zephyr-exp034-1", argv=args_list,
                   command=replay_command("tools/e_exp034_formation.py", args_list),
                   config={"experiment": "EXP-034", "pilot": args.pilot},
                   seeds=SEEDS[:3] if args.pilot else SEEDS, outputs=outputs,
                   limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(args.processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/formation.py", "tools/e_exp034_formation.py"])


if __name__ == "__main__":
    main()
