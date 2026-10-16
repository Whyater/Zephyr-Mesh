"""EXP-030: tag under latency and loss in simulated missions.

Usage (repo root):
    PYTHONPATH=. ./.venv/bin/python tools/e_exp030_tag.py --output runs/e-exp030-tag

All motion, delays, and link loss are simulated. The runner writes one row per
mission run, aggregate cell summaries, two simple figures, and a manifest.
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

from sim.missions.api import Scenario
from tests.reference_e.hand_checks import tail_follow_time_s
from tools.c_common import prepare_output, write_json, write_manifest
from tools.e_common import MISSION_LIMITS, MISSION_SOURCES, group_rows, make_job, median_summary, processes, rate_summary, run_jobs, seeds_for, write_rows
from tools.horizon_mpl import apply_style

EXP_NUMBER = 30
CLASSES = ("straight", "turn", "figure8", "jink")
DELAYS = (0.0, 0.1, 0.2, 0.3, 0.5)
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
ARMS = ("naive", "recovery", "ct", "imm")
SEEDS = seeds_for(EXP_NUMBER, 50)


def _scenario(seed: int, *, target_class: str, delay: float, preset: str, arm: str,
              n_followers: int = 1, adapter: str = "point_mass", policy: dict | None = None) -> dict:
    return {
        "mission": "tag", "seed": int(seed), "duration_s": 30.0,
        "adapter": adapter, "preset": preset, "policy": dict(policy or {}),
        "measurement_delay_s": float(delay),
        "params": {"target_class": target_class, "arm": arm, "n_followers": int(n_followers), "mode": "tag"},
    }


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build the preregistered jobs in stable order."""
    jobs: list[dict] = []
    if pilot:
        for target_class in ("straight", "jink"):
            for delay in (0.0, 0.3):
                for preset in ("clean", "burst_10_0.5s"):
                    for seed_index, seed in enumerate(SEEDS[:3]):
                        labels = {"slice": "pilot", "target_class": target_class, "measurement_delay_s": delay,
                                  "preset": preset, "arm": "recovery", "n_followers": 1, "adapter": "point_mass",
                                  "seed_index": seed_index, "policy_name": "default"}
                        jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                               preset=preset, arm="recovery")))
        for seed_index, seed in enumerate(SEEDS[:2]):
            labels = {"slice": "pilot_physics", "target_class": "straight", "measurement_delay_s": 0.0,
                      "preset": "clean", "arm": "recovery", "n_followers": 1, "adapter": "physics",
                      "seed_index": seed_index, "policy_name": "default"}
            jobs.append(make_job(labels, _scenario(seed, target_class="straight", delay=0.0, preset="clean",
                                                   arm="recovery", adapter="physics")))
        return jobs

    for target_class in CLASSES:
        for delay in DELAYS:
            for preset in PRESETS:
                for arm in ARMS:
                    for seed_index, seed in enumerate(SEEDS):
                        labels = {"slice": "main", "target_class": target_class, "measurement_delay_s": delay,
                                  "preset": preset, "arm": arm, "n_followers": 1, "adapter": "point_mass",
                                  "seed_index": seed_index, "policy_name": "default"}
                        jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                               preset=preset, arm=arm)))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in PRESETS:
                for seed_index, seed in enumerate(SEEDS):
                    labels = {"slice": "calm", "target_class": target_class, "measurement_delay_s": delay,
                              "preset": preset, "arm": "recovery", "n_followers": 1, "adapter": "point_mass",
                              "seed_index": seed_index, "policy_name": "calm"}
                    jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                           preset=preset, arm="recovery",
                                                           policy={"degraded_after_s": 0.25, "land_after_s": 1.0})))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in ("clean", "iid_30", "burst_10_0.5s"):
                for n_followers in (2, 3):
                    for seed_index, seed in enumerate(SEEDS):
                        labels = {"slice": "followers", "target_class": target_class, "measurement_delay_s": delay,
                                  "preset": preset, "arm": "recovery", "n_followers": n_followers,
                                  "adapter": "point_mass", "seed_index": seed_index, "policy_name": "default",
                                  "separation_rule": "local stand-in separation rule, until Run D"}
                        jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                               preset=preset, arm="recovery", n_followers=n_followers)))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in ("clean", "burst_10_0.5s"):
                for seed_index, seed in enumerate(SEEDS[:20]):
                    labels = {"slice": "physics", "target_class": target_class, "measurement_delay_s": delay,
                              "preset": preset, "arm": "recovery", "n_followers": 1, "adapter": "physics",
                              "seed_index": seed_index, "policy_name": "default"}
                    jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                           preset=preset, arm="recovery", adapter="physics")))
    return jobs


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "tag_success": rate_summary([bool(r["tagged"]) for r in rows]),
        "time_to_tag_s": median_summary([r["time_to_tag_s"] for r in rows if r["tagged"]]),
        "minimum_true_separation_m": {"median": median_summary([r["min_separation_m"] for r in rows]),
                                      "minimum": min(float(r["min_separation_m"]) for r in rows if r["min_separation_m"] is not None)},
        "separation_violations": sum(int(r["separation_violations"]) for r in rows),
        "failsafe_triggers": {"holds": sum(int(r["holds"]) for r in rows),
                              "landings": sum(int(r["landings"]) for r in rows)},
        "outcome_reason_counts": {reason: sum(1 for r in rows if r["end_reason"] == reason)
                                   for reason in sorted({r["end_reason"] for r in rows})},
    }


def summarize(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers", "adapter", "policy_name")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        entry = {key: value for key, value in zip(keys, coords)}
        entry.update(_cell_summary(grouped))
        if any("separation_rule" in row for row in grouped):
            entry["separation_rule"] = "local stand-in separation rule, until Run D"
        cells.append(entry)
    hand_rows = [r for r in rows if r["slice"] in {"main", "pilot"} and r["target_class"] == "straight" and
                 r["measurement_delay_s"] == 0.0 and r["preset"] == "clean" and r["arm"] == "recovery"]
    observed = median_summary([r["time_to_tag_s"] for r in hand_rows if r["tagged"]])
    reference_zone = tail_follow_time_s(4.0, 1.0, 1.5, 0.5)
    return {
        "experiment": "EXP-030", "simulated": True, "pilot": bool(pilot),
        "grid_counts": {"main": 4 * 5 * 5 * 4 * 50, "calm": 4 * 5 * 2 * 50,
                         "followers": 4 * 2 * 3 * 2 * 50, "physics": 4 * 2 * 2 * 20},
        "cells": cells,
        "hand_check": {"reference_zone_entry_s": reference_zone, "predicted_time_to_tag_s": "5.0 to 5.6 s",
                        "observed_time_to_tag_s": observed},
        "prediction": {
            "text": "Corrected prediction: default policy tag success about 45 to 60 % on burst_10_0.5s for straight and turn, lower for figure eight and jinking (longer rounds); burst_30_0.5s about 15 to 30 %; calm (T = 1.0 s) raises both by 20 to 40 points.",
            "observed_rates": [{key: value for key, value in cell.items() if key in ("slice", "target_class", "measurement_delay_s", "preset", "arm", "tag_success")} for cell in cells
                               if cell["arm"] == "recovery"],
        },
        "deviations": [], "missing_metrics": [],
        "limits": list(MISSION_LIMITS),
    }


def _figures(output: Path, rows: list[dict]) -> list[str]:
    apply_style("light")
    colors = {arm: color for arm, color in zip(ARMS, ("#1769AA", "#00897B", "#7B61A8", "#D17A00"))}
    fig, axes = plt.subplots(len(CLASSES), len(PRESETS), figsize=(15, 10), sharey=True)
    for row_index, target_class in enumerate(CLASSES):
        for col_index, preset in enumerate(PRESETS):
            ax = axes[row_index, col_index]
            for arm in ARMS:
                values = []
                for delay in DELAYS:
                    selected = [r for r in rows if r["slice"] == "main" and r["target_class"] == target_class and r["preset"] == preset and r["arm"] == arm and r["measurement_delay_s"] == delay]
                    values.append(sum(bool(r["tagged"]) for r in selected) / len(selected) if selected else float("nan"))
                ax.plot(DELAYS, values, marker="o", label=arm, color=colors[arm])
            if row_index == 0:
                ax.set_title(f"{preset} simulated")
            if col_index == 0:
                ax.set_ylabel(f"{target_class}\ntag success")
            if row_index == len(CLASSES) - 1:
                ax.set_xlabel("measurement delay (s)")
            ax.set_ylim(0, 1.05)
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    first = "tag_success_delay.png"
    fig.savefig(output / first, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for preset in PRESETS:
        vals = []
        for policy in ("default", "calm"):
            selected = [r for r in rows if r["target_class"] in CLASSES and r["measurement_delay_s"] == 0.3 and r["preset"] == preset and r["arm"] == "recovery" and r["policy_name"] == policy]
            vals.append(sum(bool(r["tagged"]) for r in selected) / len(selected) if selected else float("nan"))
        ax.plot(("default", "calm"), vals, marker="o", label=preset)
    ax.set_title("Default versus calm simulated tag success")
    ax.set_ylabel("tag success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    second = "tag_success_default_calm.png"
    fig.savefig(output / second, dpi=140)
    plt.close(fig)
    return [first, second]


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
    csv_name = "tag.csv"
    write_rows(output / csv_name, rows, first_columns=("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers", "adapter", "seed_index", "policy_name", "seed"))
    summary = summarize(rows, pilot=args.pilot)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows)
    outputs = [csv_name, "summary.json", *figures]
    write_manifest(output, schema="zephyr-exp030-1", argv=[] if argv is None else argv,
                   command="PYTHONPATH=. ./.venv/bin/python tools/e_exp030_tag.py --output " + str(args.output) + (" --pilot" if args.pilot else ""),
                   config={"experiment": "EXP-030", "pilot": args.pilot}, seeds=SEEDS[:3] if args.pilot else SEEDS,
                   outputs=outputs, limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(args.processes), hashed_sources=list(MISSION_SOURCES) + ["sim/missions/tag.py", "tools/e_exp030_tag.py"])


if __name__ == "__main__":
    main()
