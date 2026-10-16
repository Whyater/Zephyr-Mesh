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

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.interaction import downwash_wind_at
from tests.reference_e.hand_checks import first_order_loop_jitter_std
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

EXP_NUMBER = 34
SIGMAS = (0.01, 0.02, 0.05, 0.10)
LATENCIES = (0.0, 0.05, 0.1, 0.2)
PRESETS = ("clean", "iid_10")
SEEDS = seeds_for(EXP_NUMBER, 50)
PHYSICS_SEED_COUNT = 20


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
    seeds = 1600, physics 2 sigma x 2 latency x 2 presets x 20 = 160.
    """
    jobs: list[dict] = []
    if pilot:
        for sigma in (0.02, 0.10):
            for latency in (0.0, 0.2):
                for seed_index, seed in enumerate(SEEDS[:3]):
                    labels = {"slice": "pilot", "sigma_m": sigma, "fix_latency_s": latency,
                              "preset": "clean", "adapter": "point_mass_offboard", "seed_index": seed_index}
                    jobs.append(make_job(labels, _scenario(seed, sigma=sigma, latency=latency,
                                                           preset="clean", adapter="point_mass_offboard")))
        for seed_index, seed in enumerate(SEEDS[:2]):
            labels = {"slice": "pilot_physics", "sigma_m": 0.05, "fix_latency_s": 0.1,
                      "preset": "clean", "adapter": "physics_offboard", "seed_index": seed_index}
            jobs.append(make_job(labels, _scenario(seed, sigma=0.05, latency=0.1,
                                                   preset="clean", adapter="physics_offboard")))
        return jobs

    for adapter, sigmas, latencies, seed_count in (
        ("point_mass_offboard", SIGMAS, LATENCIES, 50),
        ("physics_offboard", (0.02, 0.05), (0.0, 0.1), PHYSICS_SEED_COUNT),
    ):
        for sigma in sigmas:
            for latency in latencies:
                for preset in PRESETS:
                    for seed_index, seed in enumerate(SEEDS[:seed_count]):
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
    for sigma in SIGMAS:
        selected = [r for r in rows if r["preset"] == "clean" and r["adapter"] == "point_mass_offboard"
                    and r["sigma_m"] == sigma and r["fix_latency_s"] == 0.0]
        if selected:
            observed = median_summary([r["per_axis_std_m"] for r in selected])
            reference = first_order_loop_jitter_std(sigma, 1.0 / 30.0, 0.25)
            hand.append({"sigma_m": sigma, "observed_per_axis_std_m": observed,
                         "reference_per_axis_std_m": reference,
                         "observed_to_reference": None if observed["median"] is None else observed["median"] / reference})

    downwash = float(__import__("numpy").linalg.norm(downwash_wind_at(
        (7.0, 4.0, 1.5), 0.030 * 9.81, 4, 0.023, (9.0, 4.0, 1.5))))
    return {
        "experiment": "EXP-034", "simulated": True, "pilot": bool(pilot),
        "grid_counts": {"point_mass_offboard": 4 * 4 * 2 * 50,
                         "physics_offboard": 2 * 2 * 2 * PHYSICS_SEED_COUNT},
        "physics_seed_count": PHYSICS_SEED_COUNT,
        "cells": cells,
        "hand_check": {"fix_period_s": 1.0 / 30.0, "response_time_s": 0.25,
                        "hand_formula": "per-axis std = sigma * sqrt(a / (2 - a)), a = 1 - exp(-(1/30)/0.25)",
                        "zero_latency_clean": hand},
        "downwash_wind_max_mps_observed": downwash,
        "prediction": {
            "text": "Point mass at zero latency matches the hand formula within 10 percent; latency up to 0.2 s raises error by under 30 percent; physics errors are larger and grow faster with latency; minimum separation stays far above 0.5 m.",
            "observed_cells": [{key: value for key, value in cell.items()
                                if key in ("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "rms_error_m", "minimum_pairwise_separation_m")}
                               for cell in cells],
        },
        "deviations": ["D-121: physics cells use the first 20 of the preregistered 50 seeds because the pilot projects about 2.72 hours for the 50-seed full grid."],
        "missing_metrics": [], "limits": list(MISSION_LIMITS),
    }


def _figures(output: Path, rows: list[dict]) -> list[str]:
    apply_style("light")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, adapter in zip(axes, ("point_mass_offboard", "physics_offboard")):
        sigmas = SIGMAS if adapter == "point_mass_offboard" else (0.02, 0.05)
        for sigma in sigmas:
            values = []
            for latency in LATENCIES:
                selected = [r for r in rows if r["adapter"] == adapter and r["preset"] == "clean"
                            and r["sigma_m"] == sigma and r["fix_latency_s"] == latency]
                values.append(median_summary([r["rms_error_m"] for r in selected])["median"] if selected else float("nan"))
            ax.plot(LATENCIES, values, marker="o", label=f"sigma {sigma:g} m")
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
    args = parser.parse_args(argv)
    output = prepare_output(args.output)
    started = time.perf_counter()
    jobs = build_jobs(pilot=args.pilot)
    rows = run_jobs(jobs, n_processes=processes(args.processes))
    csv_name = "formation.csv"
    write_rows(output / csv_name, rows, first_columns=("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "seed_index", "seed"))
    write_json(output / "summary.json", summarize(rows, pilot=args.pilot))
    figures = _figures(output, rows)
    outputs = [csv_name, "summary.json", *figures]
    write_manifest(output, schema="zephyr-exp034-1", argv=[] if argv is None else argv,
                   command="PYTHONPATH=. ./.venv/bin/python tools/e_exp034_formation.py --output " + str(args.output) + (" --pilot" if args.pilot else ""),
                   config={"experiment": "EXP-034", "pilot": args.pilot},
                   seeds=SEEDS[:3] if args.pilot else SEEDS, outputs=outputs,
                   limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(args.processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/formation.py", "tools/e_exp034_formation.py"])


if __name__ == "__main__":
    main()
