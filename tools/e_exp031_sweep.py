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
from sim.missions.sweep import SweepMission
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
    replay_command,
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
    "without dropout, loss changes nothing (each drone sweeps its own strip). With a dropout and delta "
    "messages, loss makes survivors believe covered cells are uncovered, so duplication rises roughly with "
    "the loss rate (iid_30: about 30 % of the dead drone's reported cells unknown, duplicated); cumulative "
    "messages remove almost all of it (any single message carries the whole map). Bursts hurt delta most. "
    "Completion time with dropout: the dead drone's remaining two thirds of a strip (about 177 m^2) shared "
    "by two survivors adds about 177/(2 x 4 x 2) = 11 s plus transit, so about 45 to 55 s."
)
PREDICTION_SOURCE = "pre-registered before the runs (EXP-031 section of the private pre-registration)"
PREDICTION_NOTE = (
    'Prediction-before-run check: EXP-031 prediction "without dropout, loss changes nothing" was already '
    "contradicted by the pilot on iid_30 under the default policy (failsafe landings); this is reported as "
    "a miss of the prediction, not reworded."
)
HAND_CHECK_BAND_S = (35.0, 42.0)
EXPLORATORY_REASON = (
    "the default 0.30 s lost-link rule landed drones on iid_30 and burst presets and the field was never "
    "finished, because a landed drone that keeps sharing is never declared lost"
)
MISSION_NOTE = (
    "Mission changes before any full run: a drone that is landing or landed covers no cells; a drone that "
    "lands on its own failsafe keeps sharing and is never declared lost, so on lossy links P(complete) "
    "measures the failsafe (see the mission limits); duplicated_fraction counts cells seen by two or more "
    "different drones (contract definition)."
)


def _scenario(seed: int, *, preset: str, dropout: str, message_mode: str,
              adapter: str = "point_mass", policy: dict | None = None,
              duration_s: float = DURATION_S) -> dict:
    return {
        "mission": "sweep", "seed": int(seed), "duration_s": float(duration_s),
        "adapter": adapter, "preset": preset, "policy": dict(policy or {}),
        "params": {"message_mode": message_mode, "dropout": dropout},
    }


def _calm_job(seed: int, seed_index: int, *, preset: str, dropout: str, mode: str,
              duration_s: float) -> dict:
    labels = {"slice": EXPLORATORY_LABEL, "preset": preset, "dropout": dropout,
              "message_mode": mode, "adapter": "point_mass", "policy_name": "calm",
              "seed_index": seed_index}
    return make_job(labels, _scenario(seed, preset=preset, dropout=dropout, message_mode=mode,
                                      policy=CALM_POLICY, duration_s=duration_s))


def build_jobs(*, pilot: bool = False) -> list[dict]:
    seeds = seeds_for(EXP_NUMBER, 3 if pilot else 50)
    duration_s = PILOT_DURATION_S if pilot else DURATION_S
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
                                                           duration_s=duration_s)))
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
                                                   duration_s=duration_s)))
    if pilot:
        for seed_index, seed in enumerate(seeds[:2]):
            jobs.append(_calm_job(seed, seed_index, preset="iid_30", dropout="one", mode="delta",
                                  duration_s=duration_s))
    else:
        for preset in ("iid_30", "burst_10_0.5s"):
            for dropout in DROPOUTS:
                for mode in MESSAGE_MODES:
                    for seed_index, seed in enumerate(seeds):
                        jobs.append(_calm_job(seed, seed_index, preset=preset, dropout=dropout,
                                              mode=mode, duration_s=duration_s))
    return jobs



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


def _hand_check_verdicts(rows: list[dict]) -> list[dict]:
    """Per message mode: how many clean, no-dropout runs reached t100 and whether the median is in the band."""
    low, high = HAND_CHECK_BAND_S
    verdicts = []
    for mode in MESSAGE_MODES:
        selected = [r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"
                    and r["preset"] == "clean" and r["dropout"] == "none" and r["message_mode"] == mode]
        item = {"message_mode": mode, "reference_sweeping_s": coverage_time_s(800, 3, 4.0, 2.0),
                "predicted_t100_range_s": [low, high], "n_runs": len(selected)}
        if not selected:
            item.update({"n_reached_t100": 0, "observed_t100": None, "verdict": "not run"})
        else:
            observed = median_summary([r["t100_s"] for r in selected])
            median = observed["median"]
            # A cell where no run reached t100 has no median, so it cannot be inside the band.
            inside = median is not None and low <= median <= high
            item.update({"n_reached_t100": observed["n"], "observed_t100": observed,
                         "verdict": "inside" if inside else "outside"})
        verdicts.append(item)
    return verdicts


def _summarize(rows: list[dict], *, pilot: bool, horizon_s: float) -> dict:
    keys = ("slice", "preset", "dropout", "message_mode", "adapter")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, coords))
        item["metrics"] = _cell_summary(grouped)
        if any("policy_name" in row for row in grouped):
            item["policy_name"] = "calm"
        cells.append(item)
    hand_reference = coverage_time_s(800, 3, 4.0, 2.0)
    hand = _hand_check_verdicts(rows)
    atlas = {}
    for coords, grouped in group_rows([r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"],
                                      ("preset", "dropout", "message_mode")).items():
        preset, dropout, mode = coords
        atlas[f"{preset}|{dropout}|{mode}"] = {
            f"wilson_lower_p_t100_within_{horizon_s:g}_s":
                rate_summary([r["t100_s"] is not None for r in grouped])["wilson_low"],
            "median_duplicated_fraction": median_summary([r["duplicated_fraction"] for r in grouped])["median"],
        }
    exploratory = any(r["slice"] == EXPLORATORY_LABEL for r in rows)
    deviations = []
    if pilot:
        deviations.append(f"Pilot uses a {horizon_s:g} s horizon; the preregistered grid horizon is {DURATION_S:g} s.")
    if exploratory:
        calm_seeds = "2 seeds, one cell" if pilot else "both dropouts and message modes, 50 seeds, 400 runs"
        deviations.append(
            "Exploratory slice, not pre-registered (Addendum B, written after the pilots): the calm policy "
            f"(degraded after {CALM_POLICY['degraded_after_s']} s, land after {CALM_POLICY['land_after_s']} s) "
            f"on iid_30 and burst_10_0.5s ({calm_seeds}). Reason: {EXPLORATORY_REASON}. "
            "It is reported separately after the pre-registered grid."
        )
    return {
        "experiment": "EXP-031", "simulated": True, "pilot": bool(pilot),
        "duration_s": horizon_s, "cells": cells, "atlas": atlas,
        "hand_check": {"reference_sweeping_s": hand_reference, "predicted_t100_range_s": list(HAND_CHECK_BAND_S),
                       "observed_clean_no_dropout": hand},
        "prediction": {"text": PREDICTION, "prediction_source": PREDICTION_SOURCE,
                       "prediction_note": PREDICTION_NOTE, "observed_clean_no_dropout": hand},
        "exploratory_slice": EXPLORATORY_LABEL if exploratory else None,
        "deviations": deviations,
        "missing_metrics": [],
        "notes": [MISSION_NOTE],
        "limits": list(MISSION_LIMITS) + list(COMMON_LIMITS) + list(SweepMission.limits),
    }


def _figures(output: Path, rows: list[dict], horizon_s: float) -> list[str]:
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
    ax.set_ylabel(f"P(complete within {horizon_s:g} s)")
    ax.set_xlabel("dropout / message mode")
    ax.set_title(f"Completion within {horizon_s:g} s simulated")
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
    horizon_s = max(float(job["scenario"]["duration_s"]) for job in run_list)
    summary = _summarize(rows, pilot=pilot, horizon_s=horizon_s)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows, horizon_s)
    outputs = [csv_name, "summary.json", *figures]
    actual_argv = list(argv or [])
    write_manifest(output, schema="zephyr-exp031-1", argv=actual_argv,
                   command=replay_command("tools/e_exp031_sweep.py", actual_argv),
                   config={"experiment": "EXP-031", "pilot": pilot,
                           "duration_s": horizon_s},
                   seeds=seeds_for(EXP_NUMBER, 3 if pilot else 50), outputs=outputs,
                   limits=list(MISSION_LIMITS) + list(SweepMission.limits), runtime_s=time.perf_counter() - started,
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
