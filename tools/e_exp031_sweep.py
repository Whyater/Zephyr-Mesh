"""EXP-031 area sweep with lossy coverage sharing.

Usage (repo root):
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/e_exp031_sweep.py --output runs/e-exp031-sweep --pilot --processes 1

All scenarios and figures are simulated.  The mission models three volunteer
search drones sweeping a field and sharing coverage over synthetic links.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.missions.engine import COMMON_LIMITS
from tests.reference_e.hand_checks import coverage_time_s
from tools.c_common import prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    apply_mission_style,
    group_rows,
    make_job,
    median_summary,
    processes,
    rate_summary,
    run_jobs,
    seeds_for,
    write_rows,
)


EXP_NUMBER = 31
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
PHYSICS_PRESETS = ("clean", "iid_30")
MESSAGE_MODES = ("delta", "cumulative")
DROPOUTS = ("none", "one")
DURATION_S = 120.0
PILOT_DURATION_S = 60.0  # long enough for clean runs to finish (hand check 35 to 42 s)
CALM_POLICY = {"degraded_after_s": 0.25, "land_after_s": 1.0}
EXPLORATORY_LABEL = "exploratory, added after the pilot, not pre-registered"
PREDICTION = (
    "Claude's prediction: without dropout, loss changes nothing (each drone sweeps its own strip). "
    "With a dropout and delta messages, loss makes survivors believe covered cells are uncovered, so "
    "duplication rises roughly with the loss rate (iid_30: about 30 % of the dead drone's reported cells "
    "unknown, duplicated); cumulative messages remove almost all of it (any single message carries the "
    "whole map). Bursts hurt delta most. Completion time with dropout: the dead drone's remaining two "
    "thirds of a strip (about 177 m2) shared by two survivors adds about 177/(2 x 4 x 2) = 11 s plus "
    "transit, so about 45 to 55 s."
)


def _scenario(seed: int, *, preset: str, dropout: str, message_mode: str,
              adapter: str = "point_mass", policy: dict | None = None,
              duration_s: float = DURATION_S) -> dict:
    return {
        "mission": "sweep", "seed": int(seed), "duration_s": float(duration_s),
        "adapter": adapter, "preset": preset, "policy": dict(policy or {}),
        "params": {"message_mode": message_mode, "dropout": dropout},
    }


def build_jobs(*, pilot: bool = False) -> list[dict]:
    seeds = seeds_for(EXP_NUMBER, 3 if pilot else 50)
    jobs: list[dict] = []
    main_presets = ("clean", "iid_30") if pilot else PRESETS
    for preset in main_presets:
        for dropout in DROPOUTS:
            for mode in MESSAGE_MODES:
                for seed_index, seed in enumerate(seeds):
                    labels = {"slice": "main", "preset": preset, "dropout": dropout,
                              "message_mode": mode, "adapter": "point_mass", "seed_index": seed_index}
                    jobs.append(make_job(labels, _scenario(seed, preset=preset, dropout=dropout,
                                                           message_mode=mode,
                                                           duration_s=PILOT_DURATION_S if pilot else DURATION_S)))
    physics_seeds = seeds[:2] if pilot else seeds[:20]
    physics_cells = (("clean", "one", "delta"),) if pilot else tuple(
        (preset, "one", mode) for preset in PHYSICS_PRESETS for mode in MESSAGE_MODES
    )
    for preset, dropout, mode in physics_cells:
        for seed_index, seed in enumerate(physics_seeds):
            labels = {"slice": "physics", "preset": preset, "dropout": dropout,
                      "message_mode": mode, "adapter": "physics", "seed_index": seed_index}
            jobs.append(make_job(labels, _scenario(seed, preset=preset, dropout=dropout,
                                                   message_mode=mode, adapter="physics",
                                                   duration_s=PILOT_DURATION_S if pilot else DURATION_S)))
    if not pilot:
        for preset in ("iid_30", "burst_10_0.5s"):
            for dropout in DROPOUTS:
                for mode in MESSAGE_MODES:
                    for seed_index, seed in enumerate(seeds):
                        labels = {"slice": EXPLORATORY_LABEL, "preset": preset, "dropout": dropout,
                                  "message_mode": mode, "adapter": "point_mass", "policy_name": "calm",
                                  "seed_index": seed_index}
                        jobs.append(make_job(labels, _scenario(seed, preset=preset, dropout=dropout,
                                                               message_mode=mode, policy=CALM_POLICY)))
    return jobs


build_grid = build_jobs


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n_runs": len(rows),
        "p_t100_reached": rate_summary([row["t100_s"] is not None for row in rows]),
        "t95_s": median_summary([row["t95_s"] for row in rows]),
        "t100_s": median_summary([row["t100_s"] for row in rows]),
        "duplicated_fraction": median_summary([row["duplicated_fraction"] for row in rows]),
        "missed_cells": {"median": median_summary([row["missed_cells"] for row in rows]),
                         "max": max(int(row["missed_cells"]) for row in rows)},
        "replans": {"mean": sum(float(row["replans"]) for row in rows) / len(rows)},
        "belief_error_cells": median_summary([row["belief_error_cells"] for row in rows]),
        "false_loss_declarations": {
            "total": sum(int(row["false_loss_declarations"]) for row in rows),
            "runs_with_any": rate_summary([int(row["false_loss_declarations"]) > 0 for row in rows]),
        },
        "failsafe_landings": {
            "total": sum(int(row["failsafe_landings"]) for row in rows),
            "runs_with_any": rate_summary([int(row["failsafe_landings"]) > 0 for row in rows]),
        },
    }


def _summarize(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("slice", "preset", "dropout", "message_mode", "adapter")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, coords))
        item["metrics"] = _cell_summary(grouped)
        if any("policy_name" in row for row in grouped):
            item["policy_name"] = "calm"
        cells.append(item)
    hand_reference = coverage_time_s(800, 3, 4.0, 2.0)
    hand = []
    for mode in MESSAGE_MODES:
        selected = [r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"
                    and r["preset"] == "clean" and r["dropout"] == "none" and r["message_mode"] == mode]
        if selected:
            hand.append({"message_mode": mode, "reference_sweeping_s": hand_reference,
                         "predicted_t100_range_s": [35.0, 42.0],
                         "observed_t100": median_summary([r["t100_s"] for r in selected])})
    atlas = {}
    for coords, grouped in group_rows([r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"],
                                      ("preset", "dropout", "message_mode")).items():
        preset, dropout, mode = coords
        atlas[f"{preset}|{dropout}|{mode}"] = {
            "wilson_lower_p_t100_within_120_s": rate_summary([r["t100_s"] is not None for r in grouped])["wilson_low"],
            "median_duplicated_fraction": median_summary([r["duplicated_fraction"] for r in grouped])["median"],
        }
    exploratory = any(r["slice"] == EXPLORATORY_LABEL for r in rows)
    return {
        "experiment": "EXP-031", "simulated": True, "pilot": bool(pilot),
        "duration_s": PILOT_DURATION_S if pilot else DURATION_S, "cells": cells, "atlas": atlas,
        "hand_check": {"reference_sweeping_s": hand_reference, "predicted_t100_range_s": [35.0, 42.0],
                        "observed_clean_no_dropout": hand},
        "prediction": {"text": PREDICTION, "observed_clean_no_dropout": hand},
        "exploratory_slice": EXPLORATORY_LABEL if exploratory else None,
        "deviations": (["Pilot uses a 60 s horizon; the preregistered grid horizon is 120 s."] if pilot else []),
        "missing_metrics": [],
        "limits": list(MISSION_LIMITS + COMMON_LIMITS),
    }


def _figures(output: Path, rows: list[dict]) -> list[str]:
    apply_mission_style("light")
    names: list[str] = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for preset in PRESETS:
        labels, values = [], []
        for dropout in DROPOUTS:
            for mode in MESSAGE_MODES:
                selected = [r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"
                            and r["preset"] == preset and r["dropout"] == dropout and r["message_mode"] == mode]
                if selected:
                    labels.append(f"{dropout}/{mode}")
                    values.append(rate_summary([r["t100_s"] is not None for r in selected])["rate"])
        if values:
            ax.plot(labels, values, marker="o", label=preset)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(complete within 120 s)")
    ax.set_xlabel("dropout / message mode")
    ax.set_title("Completion within 120 s simulated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    name = "sweep_completion_simulated.png"
    fig.savefig(output / name, dpi=140)
    plt.close(fig)
    names.append(name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for preset in PRESETS:
        vals = []
        for mode in MESSAGE_MODES:
            selected = [r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"
                        and r["preset"] == preset and r["dropout"] == "one" and r["message_mode"] == mode]
            vals.append(median_summary([r["belief_error_cells"] for r in selected])["median"] if selected else float("nan"))
        if any(v == v for v in vals):
            ax.plot(MESSAGE_MODES, vals, marker="o", label=preset)
    ax.set_ylabel("belief error cells (median)")
    ax.set_xlabel("message mode")
    ax.set_title("Belief error cells simulated")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    fig.tight_layout()
    name = "sweep_belief_error_simulated.png"
    fig.savefig(output / name, dpi=140)
    plt.close(fig)
    names.append(name)
    return names


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   argv: list[str] | None = None, jobs: list[dict] | None = None) -> dict:
    output = prepare_output(Path(output))
    started = time.perf_counter()
    run_list = build_jobs(pilot=pilot) if jobs is None else list(jobs)
    rows = run_jobs(run_list, n_processes=processes(n_processes))
    csv_name = "sweep.csv"
    write_rows(output / csv_name, rows, first_columns=("slice", "preset", "dropout", "message_mode", "adapter", "policy_name", "seed_index", "seed"))
    summary = _summarize(rows, pilot=pilot)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows)
    outputs = [csv_name, "summary.json", *figures]
    actual_argv = list(argv or [])
    command = "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/e_exp031_sweep.py" + "".join(" " + str(a) for a in actual_argv)
    write_manifest(output, schema="zephyr-exp031-1", argv=actual_argv, command=command,
                   config={"experiment": "EXP-031", "pilot": pilot,
                           "duration_s": PILOT_DURATION_S if pilot else DURATION_S},
                   seeds=seeds_for(EXP_NUMBER, 3 if pilot else 50), outputs=outputs,
                   limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(n_processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/sweep.py", "tools/e_exp031_sweep.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args(argv)
    recorded_argv = list(sys.argv[1:] if argv is None else argv)
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes,
                   argv=recorded_argv)


if __name__ == "__main__":
    main()
