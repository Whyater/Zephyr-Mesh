"""EXP-034 gym formation demo for simulated missions.

Usage (repo root):
    PYTHONPATH=. ./.venv/bin/python tools/e_exp034_formation.py --output runs/e-exp034-formation --pilot

All motion, camera fixes, delays, and link loss are simulated.
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.missions.formation import FormationMission
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
from tools.horizon_mpl import palette

EXP_NUMBER = 34
SIGMAS = (0.01, 0.02, 0.05, 0.10)
LATENCIES = (0.0, 0.05, 0.1, 0.2)
PRESETS = ("clean", "iid_10")
SEEDS = seeds_for(EXP_NUMBER, 50)
PREDICTION_SOURCE = "pre-registered before the runs (EXP-034 section of the private pre-registration)"
PREDICTION_TEXT = (
    "point mass at zero latency matches the hand formula within 10 %; latency up to 0.2 s raises the error by under 30 % "
    "(the first-order loop with delay stays stable below about (pi/2) tau = 0.39 s, but the margin shrinks); physics adapter errors larger than the point mass at the same sigma "
    "(PD loop with lower damping passes more noise) and grows faster with latency (Run A: position feedback edge 0.33 s). Separation never near 0.5 m (slots 2 m apart, errors of centimetres)."
)
# Pre-registered slices per adapter (the pilot slices stand in for them in a
# pilot run).  Both run the registered 0.8 m guard ("local", decision D-126).
PREREGISTERED_SLICES = {
    "point_mass_offboard": ("main", "pilot"),
    "physics_offboard": ("physics", "pilot_physics"),
}
MAIN_RULE = "local"
COMPARISON_RULE = "coop"
COMPARISON_SLICE = "coop_comparison"
RULE_DEVIATIONS = (
    "The pre-registered grid (main and physics slices) runs the registered pairwise 0.8 m guard "
    "(separation_rule local), as pre-registered (decision D-126).",
    "The coop_comparison slice (point mass, clean preset, Run D's cooperative reciprocal velocity rule, "
    "separation_rule coop, decision D-123) was added after the pre-registration, for comparison only; it is "
    "not part of the pre-registered grid.",
    "guard_activations counts pair-steps inside 0.8 m in the local slices; in the coop_comparison slice it "
    "counts drone control steps where the coop rule changed the velocity.",
)


def _scenario(seed: int, *, sigma: float, latency: float, preset: str, adapter: str,
              duration_s: float = 30.0, separation_rule: str = MAIN_RULE) -> dict:
    return {
        "mission": "formation",
        "seed": int(seed),
        "duration_s": float(duration_s),
        "adapter": adapter,
        "preset": preset,
        "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": float(sigma),
                   "fix_latency_s": float(latency), "separation_rule": separation_rule},
    }


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build jobs in stable grid order.

    Full grid arithmetic: point mass 4 sigma x 4 latency x 2 presets x 50
    seeds = 1600, physics 2 sigma x 2 latency x 2 presets x 50 = 400, both on
    the registered 0.8 m guard ("local", decision D-126), plus a comparison
    slice on Run D's separation rule ("coop", decision D-123): point mass,
    4 sigma x 4 latency, clean, 50 seeds = 800.
    """
    jobs: list[dict] = []
    if pilot:
        for sigma in (0.02, 0.10):
            for latency in (0.0, 0.2):
                for seed_index, seed in enumerate(SEEDS[:3]):
                    labels = {"slice": "pilot", "sigma_m": sigma, "fix_latency_s": latency,
                              "preset": "clean", "adapter": "point_mass_offboard", "separation_rule": MAIN_RULE,
                              "seed_index": seed_index}
                    jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency,
                                                           preset="clean", adapter="point_mass_offboard", duration_s=10.0)))
        for seed_index, seed in enumerate(SEEDS[:2]):
            labels = {"slice": "pilot_physics", "sigma_m": 0.05, "fix_latency_s": 0.1,
                      "preset": "clean", "adapter": "physics_offboard", "separation_rule": MAIN_RULE,
                      "seed_index": seed_index}
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
                                  "adapter": adapter, "separation_rule": MAIN_RULE, "seed_index": seed_index}
                        jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency,
                                                               preset=preset, adapter=adapter)))
    for sigma in SIGMAS:
        for latency in LATENCIES:
            for seed_index, seed in enumerate(SEEDS):
                labels = {"slice": COMPARISON_SLICE, "sigma_m": sigma, "fix_latency_s": latency,
                          "preset": "clean", "adapter": "point_mass_offboard", "separation_rule": COMPARISON_RULE,
                          "seed_index": seed_index}
                jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency, preset="clean",
                                                       adapter="point_mass_offboard", separation_rule=COMPARISON_RULE)))
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


def _grid_counts() -> dict:
    return dict(Counter(job["labels"]["slice"] for job in build_jobs()))


def _hand_entry(rows: list[dict], *, adapter: str, sigma: float, slices: tuple, rule: str) -> dict | None:
    """Zero-latency clean rows of one adapter, sigma, slice set and rule against the hand formula."""
    selected = [r for r in rows if r["slice"] in slices and r["separation_rule"] == rule
                and r["preset"] == "clean" and r["adapter"] == adapter
                and r["sigma_m"] == sigma and r["fix_latency_s"] == 0.0]
    if not selected:
        return None
    observed_axis = median_summary([r["per_axis_std_m"] for r in selected])
    observed_rms = median_summary([r["rms_error_m"] for r in selected])
    reference_axis = first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
    reference_rms = float(np.sqrt(3.0) * reference_axis)
    ratio = None if observed_rms["median"] is None else observed_rms["median"] / reference_rms
    return {"adapter": adapter, "sigma_m": sigma, "slice": selected[0]["slice"], "separation_rule": rule,
            "observed_per_axis_std_m": observed_axis,
            "reference_per_axis_std_m": reference_axis,
            "observed_to_reference": None if observed_axis["median"] is None else observed_axis["median"] / reference_axis,
            "observed_rms_error_m": observed_rms,
            "reference_rms_error_m": reference_rms,
            "observed_to_reference_rms": ratio,
            "within_10_percent": None if ratio is None else abs(ratio - 1.0) <= 0.10}


def summarize(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "separation_rule")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        item = {key: value for key, value in zip(keys, coords)}
        item.update(_cell_summary(grouped))
        cells.append(item)

    hand = []
    for adapter, sigmas in (("point_mass_offboard", SIGMAS), ("physics_offboard", (0.02, 0.05))):
        for sigma in sigmas:
            entry = _hand_entry(rows, adapter=adapter, sigma=sigma,
                                slices=PREREGISTERED_SLICES[adapter], rule=MAIN_RULE)
            if entry is not None:
                hand.append(entry)
    comparison_hand = []
    for sigma in SIGMAS:
        entry = _hand_entry(rows, adapter="point_mass_offboard", sigma=sigma,
                            slices=(COMPARISON_SLICE,), rule=COMPARISON_RULE)
        if entry is not None:
            comparison_hand.append(entry)
    deviations = list(RULE_DEVIATIONS)
    if pilot:
        deviations.append("Pilot runs 10 s per run (the full grid runs 30 s) and covers a small subset of the grid with 3 seeds.")
    return {
        "experiment": "EXP-034", "simulated": True, "pilot": bool(pilot),
        "grid_counts": _grid_counts(),
        "cells": cells,
        "hand_check": {"fix_period_s": 1.0 / 30.0, "response_time_s": 0.25,
                        "hand_formula": "per-axis std = sigma * sqrt(a / (2 - a)), a = 1 - exp(-(1/30)/0.25)",
                        "zero_latency_clean": hand,
                        "zero_latency_clean_coop_rule_comparison": comparison_hand,
                        "coop_rule_comparison_note": "Same hand formula applied to the coop_comparison "
                                                     "slice (separation_rule coop), added after the "
                                                     "pre-registration for comparison only."},
        "notes": [
            "min_separation_m is taken over the whole run, including the start from randomly offset positions.",
            "The offboard physics adapter differences the last two position fixes for velocity, a contract design choice. Velocity noise per axis is about sqrt(2) x 30 x sigma at 30 Hz (difference of two fixes 1/30 s apart).",
        ],
        "prediction": PREDICTION_TEXT,
        "prediction_source": PREDICTION_SOURCE,
        "prediction_observations": {
            "observed_cells": [{key: value for key, value in cell.items()
                                if key in ("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "separation_rule",
                                           "rms_error_m", "minimum_pairwise_separation_m")}
                               for cell in cells],
        },
        "deviations": deviations,
        "missing_metrics": [], "limits": list(MISSION_LIMITS) + list(FormationMission.limits),
    }


def _median_points(rows: list[dict], *, adapter: str, sigma: float, slices: tuple, rule: str) -> list[tuple]:
    """(latency, median RMS error) for clean rows of one adapter, sigma, slice set and rule."""
    points = []
    for latency in LATENCIES:
        selected = [r for r in rows if r["slice"] in slices and r["separation_rule"] == rule
                    and r["adapter"] == adapter and r["preset"] == "clean"
                    and r["sigma_m"] == sigma and r["fix_latency_s"] == latency]
        if selected:
            value = median_summary([r["rms_error_m"] for r in selected])["median"]
            if value is not None and math.isfinite(value):
                points.append((latency, value))
    return points


def _figures(output: Path, rows: list[dict]) -> list[str]:
    apply_mission_style("light")
    colors = dict(zip(SIGMAS, series_colors(len(SIGMAS))))
    reference_color = palette("light")["textSecondary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, adapter in zip(axes, ("point_mass_offboard", "physics_offboard")):
        sigmas = SIGMAS if adapter == "point_mass_offboard" else (0.02, 0.05)
        reference_labelled = False
        for sigma in sigmas:
            points = _median_points(rows, adapter=adapter, sigma=sigma,
                                    slices=PREREGISTERED_SLICES[adapter], rule=MAIN_RULE)
            if not points:
                continue
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                    color=colors[sigma], label=f"sigma {sigma:g} m, registered 0.8 m guard")
            if adapter == "point_mass_offboard":
                comparison_points = _median_points(rows, adapter=adapter, sigma=sigma,
                                                   slices=(COMPARISON_SLICE,), rule=COMPARISON_RULE)
                if comparison_points:
                    ax.plot([x for x, _ in comparison_points], [y for _, y in comparison_points], marker="o",
                            linestyle="--", color=colors[sigma],
                            label=f"sigma {sigma:g} m, Run D rule (comparison)")
            reference = math.sqrt(3.0) * first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
            ax.plot([0.0], [reference], marker="x", linestyle="none", color=reference_color,
                    label=None if reference_labelled else "hand formula")
            reference_labelled = True
        ax.set_title(("point mass" if adapter == "point_mass_offboard" else "physics") + " simulated")
        ax.set_xlabel("fix latency (s)")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "not in this run", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_ylabel("formation RMS error (m)")
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
    write_rows(output / csv_name, rows, first_columns=("slice", "sigma_m", "fix_latency_s", "preset", "adapter",
                                                   "separation_rule", "seed_index", "seed"))
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
