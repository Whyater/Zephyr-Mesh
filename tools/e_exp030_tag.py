"""EXP-030: tag under latency and loss in simulated missions.

Usage (repo root):
    PYTHONPATH=. ./.venv/bin/python tools/e_exp030_tag.py --output sim/runs/e-exp030-tag

All motion, delays, and link loss are simulated. The runner writes one row per
mission run, aggregate cell summaries, two simple figures, and a manifest.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.missions.api import Scenario
from sim.missions.tag import TagMission
from sim.tests.reference_e.hand_checks import tail_follow_time_s
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (MISSION_LIMITS, MISSION_SOURCES, apply_mission_style, group_rows,
                             make_job, median_summary, processes, rate_summary, replay_command,
                             run_jobs, seeds_for, series_colors, write_rows)

EXP_NUMBER = 30
CLASSES = ("straight", "turn", "figure8", "jink")
DELAYS = (0.0, 0.1, 0.2, 0.3, 0.5)
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
ARMS = ("naive", "recovery", "ct", "imm")
SEEDS = seeds_for(EXP_NUMBER, 50)
# Hand check inputs, all from the pre-registration for EXP-030: initial gap 4 m,
# zone radius 1.0 m, follower limit 1.5 m/s, target 0.5 m/s, and a continuous
# 2.0 s hold inside the zone.  The pre-registration states the zone-entry band
# as 3.0 to 3.6 s (3.0 s plus the speed ramp), so the ramp allowance is 0.6 s.
HAND_GAP_M = 4.0
HAND_ZONE_RADIUS_M = 1.0
HAND_FOLLOWER_SPEED_MPS = 1.5
HAND_TARGET_SPEED_MPS = 0.5
HAND_SPEED_RAMP_S = 0.6
HAND_HOLD_S = 2.0
PREDICTION_SOURCE = "pre-registered before the runs (EXP-030 section of the private pre-registration)"
PREDICTION_TEXT = ("clean: straight and turn about 100 %, figure eight about 100 %, "
                   "jinking 85 to 100 %. Delay up to 0.3 s barely matters at 0.5 m/s (lag v*d = 0.15 m "
                   "is inside the 0.5 m wide zone ring) for recovery; naive loses a little on jinking at "
                   "0.5 s. iid_10: still high (default T = 0.3 s at 20 Hz fires about 0.07 times per hour "
                   "at 10 %). iid_30: about 37 unnecessary landings per hour, so about 26 % of 30 s rounds "
                   "end in a landing. Bursty presets are where tag fails: burst_10_0.5s triggers about 400 "
                   "landings per hour, so most 30 s rounds end in a landing before the tag completes (predict "
                   "tag success under 20 % with \"default\", recovered to 50 to 80 % by \"calm\"). Separation "
                   "violations: 0 on clean; a few on jinking under bursts. Two and three followers: higher "
                   "success on jinking (someone is in position) but more follower-follower guard activity; no "
                   "gain when the limit is the landing failsafe (every follower sees the same bad radio). Physics "
                   "adapter: same ordering, slightly lower success from the controller lag.")
PREDICTION_CORRECTION_2 = ('2. EXP-030 prediction for bursty presets: rounds end at the tag (about 5 s on easy '
                           'classes), not at 30 s. With about 400 unnecessary landings per hour on burst_10_0.5s, '
                           'the expected number in 5.3 s is 0.59, so P(no landing) is about e^-0.59 = 0.55. '
                           'Corrected prediction: default policy tag success about 45 to 60 % on burst_10_0.5s for '
                           'straight and turn, lower for figure eight and jinking (longer rounds); burst_30_0.5s '
                           'about 15 to 30 %; "calm" (T = 1.0 s) raises both by 20 to 40 points.')


def _scenario(seed: int, *, target_class: str, delay: float, preset: str, arm: str,
              n_followers: int = 1, adapter: str = "point_mass", policy: dict | None = None,
              separation_rule: str = "coop") -> dict:
    return {
        "mission": "tag", "seed": int(seed), "duration_s": 30.0,
        "adapter": adapter, "preset": preset, "policy": dict(policy or {}),
        "measurement_delay_s": float(delay),
        "params": {"target_class": target_class, "arm": arm, "n_followers": int(n_followers), "mode": "tag",
                   "separation_rule": separation_rule},
    }


# Follower-to-follower separation (decision D-123, within D-120): with Run D
# merged, the 2 and 3 follower slice runs Run D's rule ("coop") and, for
# comparison, the earlier local stand-in ("local").  With one follower the rule
# has no effect, so those rows are labelled "one follower".
ONE_FOLLOWER_RULE = "one follower"
RULE_DEVIATIONS = (
    "Follower-to-follower separation in the 2 and 3 follower slice uses Run D's cooperative reciprocal "
    "velocity rule (separation_rule coop, decision D-123, made after the pre-registration), not the "
    "pre-registered pairwise 0.6 m stand-in guard.",
    "The separation_rule local rows of the followers slice (the pre-registered 0.6 m stand-in guard) were "
    "added afterwards, for comparison only.",
    "guard_activations now counts follower control steps where the coop rule changed the velocity; in the "
    "separation_rule local rows it counts pair-steps inside 0.6 m.",
)


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build the preregistered jobs in stable order."""
    jobs: list[dict] = []
    if pilot:
        for target_class in ("straight", "jink"):
            for delay in (0.0, 0.3):
                for preset in ("clean", "burst_10_0.5s"):
                    for seed_index, seed in enumerate(SEEDS[:3]):
                        labels = {"slice": "pilot", "target_class": target_class, "measurement_delay_s": delay,
                                  "preset": preset, "arm": "recovery", "n_followers": 1,
                                  "separation_rule": ONE_FOLLOWER_RULE, "adapter": "point_mass",
                                  "seed_index": seed_index, "policy_name": "default"}
                        jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                               preset=preset, arm="recovery")))
        for seed_index, seed in enumerate(SEEDS[:2]):
            labels = {"slice": "pilot_physics", "target_class": "straight", "measurement_delay_s": 0.0,
                      "preset": "clean", "arm": "recovery", "n_followers": 1,
                      "separation_rule": ONE_FOLLOWER_RULE, "adapter": "physics",
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
                                  "preset": preset, "arm": arm, "n_followers": 1,
                                  "separation_rule": ONE_FOLLOWER_RULE, "adapter": "point_mass",
                                  "seed_index": seed_index, "policy_name": "default"}
                        jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                               preset=preset, arm=arm)))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in PRESETS:
                for seed_index, seed in enumerate(SEEDS):
                    labels = {"slice": "calm", "target_class": target_class, "measurement_delay_s": delay,
                              "preset": preset, "arm": "recovery", "n_followers": 1,
                              "separation_rule": ONE_FOLLOWER_RULE, "adapter": "point_mass",
                              "seed_index": seed_index, "policy_name": "calm"}
                    jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                           preset=preset, arm="recovery",
                                                           policy={"degraded_after_s": 0.25, "land_after_s": 1.0})))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in ("clean", "iid_30", "burst_10_0.5s"):
                for n_followers in (2, 3):
                    for rule in ("coop", "local"):
                        for seed_index, seed in enumerate(SEEDS):
                            labels = {"slice": "followers", "target_class": target_class,
                                      "measurement_delay_s": delay, "preset": preset, "arm": "recovery",
                                      "n_followers": n_followers, "separation_rule": rule,
                                      "adapter": "point_mass", "seed_index": seed_index, "policy_name": "default"}
                            jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                                   preset=preset, arm="recovery",
                                                                   n_followers=n_followers, separation_rule=rule)))
    for target_class in CLASSES:
        for delay in (0.0, 0.3):
            for preset in ("clean", "burst_10_0.5s"):
                for seed_index, seed in enumerate(SEEDS[:20]):
                    labels = {"slice": "physics", "target_class": target_class, "measurement_delay_s": delay,
                              "preset": preset, "arm": "recovery", "n_followers": 1,
                              "separation_rule": ONE_FOLLOWER_RULE, "adapter": "physics",
                              "seed_index": seed_index, "policy_name": "default"}
                    jobs.append(make_job(labels, _scenario(seed, target_class=target_class, delay=delay,
                                                           preset=preset, arm="recovery", adapter="physics")))
    return jobs


def hand_check_band() -> dict:
    """Predicted time to tag for a straight target, no delay, clean radio.

    The zone entry reference comes from the reference module; the band adds
    the 2.0 s hold, and the upper end also adds the speed ramp.
    """
    zone_entry_s = tail_follow_time_s(HAND_GAP_M, HAND_ZONE_RADIUS_M,
                                      HAND_FOLLOWER_SPEED_MPS, HAND_TARGET_SPEED_MPS)
    return {"zone_entry_reference_s": zone_entry_s,
            "band_s": [zone_entry_s + 0.0 + HAND_HOLD_S,
                       zone_entry_s + HAND_SPEED_RAMP_S + HAND_HOLD_S]}


def _calm_slice_present(rows: list[dict]) -> bool:
    return any(row["slice"] == "calm" for row in rows)


def _grid_counts() -> dict:
    counts = Counter(job["labels"]["slice"] for job in build_jobs())
    return {name: counts[name] for name in ("main", "calm", "followers", "physics")}


def _cell_summary(rows: list[dict]) -> dict:
    result = {
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
    if any(int(row["n_followers"]) > 1 for row in rows):
        guard_values = [int(row["guard_activations"]) for row in rows]
        follower_values = [row["follower_follower_min_m"] for row in rows]
        result["guard_activations"] = {
            "total": sum(guard_values),
            "runs_with_any": sum(value > 0 for value in guard_values),
        }
        result["follower_follower_min_m"] = {
            "median_interval": median_summary(follower_values),
            "minimum": min(float(value) for value in follower_values if value is not None),
        }
    return result


# Label keys kept beside each recovery-arm tag success rate; n_followers and
# separation_rule keep the followers cells apart.
RATE_KEYS = ("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers",
             "separation_rule", "tag_success")


def summarize(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers", "separation_rule",
            "adapter", "policy_name")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        entry = {key: value for key, value in zip(keys, coords)}
        entry.update(_cell_summary(grouped))
        cells.append(entry)
    hand_rows = [r for r in rows if r["slice"] in {"main", "pilot"} and r["target_class"] == "straight" and
                 r["measurement_delay_s"] == 0.0 and r["preset"] == "clean" and r["arm"] == "recovery"]
    observed = median_summary([r["time_to_tag_s"] for r in hand_rows if r["tagged"]])
    observed_median = observed["median"]
    band = hand_check_band()
    low, high = band["band_s"]
    calm_drawn = _calm_slice_present(rows)
    return {
        "experiment": "EXP-030", "simulated": True, "pilot": bool(pilot),
        "grid_counts": _grid_counts(),
        "cells": cells,
        "hand_check": {"zone_entry_reference_s": band["zone_entry_reference_s"],
                        "predicted_time_to_tag_band_s": band["band_s"],
                        "observed_time_to_tag_s": observed,
                        "inside_predicted_band": observed_median is not None and low <= observed_median <= high},
        "prediction": PREDICTION_TEXT,
        "prediction_source": PREDICTION_SOURCE,
        "prediction_correction_2": PREDICTION_CORRECTION_2,
        "prediction_observed_rates": [{key: value for key, value in cell.items() if key in RATE_KEYS}
                                       for cell in cells if cell["arm"] == "recovery"],
        "figures_not_drawn": [] if calm_drawn else [
            {"file": "tag_success_default_calm.png", "reason": "this run has no calm policy slice"}],
        "deviations": list(RULE_DEVIATIONS),
        "missing_metrics": ["zone entry time (the mission logs tag_zone_enter events but exposes no zone-entry metric)"],
        "limits": list(MISSION_LIMITS) + list(TagMission.limits),
    }


def _success_rate(selected: list[dict]) -> float | None:
    if not selected:
        return None
    return sum(bool(r["tagged"]) for r in selected) / len(selected)


def _figures(output: Path, rows: list[dict], *, pilot: bool) -> list[str]:
    apply_mission_style("light")
    colors = dict(zip(ARMS, series_colors(len(ARMS))))
    main_slices = ("pilot",) if pilot else ("main",)
    pilot_note = " (pilot)" if pilot else ""
    fig, axes = plt.subplots(len(CLASSES), len(PRESETS), figsize=(15, 10), sharey=True)
    for row_index, target_class in enumerate(CLASSES):
        for col_index, preset in enumerate(PRESETS):
            ax = axes[row_index, col_index]
            drew_any = False
            for arm in ARMS:
                points = []
                for delay in DELAYS:
                    selected = [r for r in rows if r["slice"] in main_slices and r["target_class"] == target_class
                                and r["preset"] == preset and r["arm"] == arm
                                and r["measurement_delay_s"] == delay]
                    rate = _success_rate(selected)
                    if rate is not None:
                        points.append((delay, rate))
                if points:
                    drew_any = True
                    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=arm,
                            color=colors[arm])
            if not drew_any:
                ax.text(0.5, 0.5, "not in this run", ha="center", va="center", transform=ax.transAxes)
            if row_index == 0:
                ax.set_title(f"{preset} simulated")
            if col_index == 0:
                ax.set_ylabel(f"{target_class}\ntag success")
            if row_index == len(CLASSES) - 1:
                ax.set_xlabel("measurement delay (s)")
            ax.set_ylim(0, 1.05)
    for ax in axes.flat:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
            break
    fig.suptitle(f"Tag success by measurement delay, simulated{pilot_note}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    first = "tag_success_delay.png"
    fig.savefig(output / first, dpi=140)
    plt.close(fig)
    names = [first]

    if not _calm_slice_present(rows):
        return names
    fig, ax = plt.subplots(figsize=(7, 4))
    for preset in PRESETS:
        vals = []
        for policy in ("default", "calm"):
            selected = [r for r in rows if r["slice"] == ("main" if policy == "default" else "calm")
                        and r["measurement_delay_s"] == 0.3
                        and r["preset"] == preset and r["arm"] == "recovery"
                        and r["policy_name"] == policy]
            rate = _success_rate(selected)
            vals.append(float("nan") if rate is None else rate)
        ax.plot(("default", "calm"), vals, marker="o", label=preset)
    ax.set_title(f"Default versus calm simulated tag success{pilot_note}")
    ax.set_ylabel("tag success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    second = "tag_success_default_calm.png"
    fig.savefig(output / second, dpi=140)
    plt.close(fig)
    names.append(second)
    return names


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
    write_rows(output / csv_name, rows, first_columns=("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers",
                                                   "separation_rule", "adapter", "seed_index", "policy_name", "seed"))
    summary = summarize(rows, pilot=args.pilot)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows, pilot=args.pilot)
    outputs = [csv_name, "summary.json", *figures]
    actual_argv = argv_or_default(argv)
    write_manifest(output, schema="zephyr-exp030-1", argv=actual_argv,
                   command=replay_command("tools/e_exp030_tag.py", actual_argv),
                   config={"experiment": "EXP-030", "pilot": args.pilot}, seeds=SEEDS[:3] if args.pilot else SEEDS,
                   outputs=outputs, limits=list(MISSION_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(args.processes), hashed_sources=list(MISSION_SOURCES) + ["sim/missions/tag.py", "tools/e_exp030_tag.py"])


if __name__ == "__main__":
    main()
