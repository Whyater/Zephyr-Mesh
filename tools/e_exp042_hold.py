"""EXP-042: the registered lost-link landing rule vs the opt-in hold-in-place rule.

Usage (repo root):
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/e_exp042_hold.py \\
        --output sim/runs/e-exp042-hold --processes 1 [--pilot]

All scenarios and figures are simulated.  The run repeats the burst-loss cells
of EXP-029 Part B (tag, 2.0 s scripted outage) and EXP-031 (area sweep) on the
same seeds under two policies:

* ``default``: the registered policy (0.15 s silence holds, 0.30 s latches a landing).
* ``hold``: the same thresholds plus ``lost_link_action="hold"``: past 0.30 s the
  drone keeps hovering and resumes when packets return; it lands only after
  5.0 s of silence or on low battery.

The verdict is reported only when every expected cell is present with both
arms and the default arm reproduces the registered EXP-029 Part B and EXP-031
rows column for column; otherwise the run is marked void.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.c_common import REPO_ROOT, prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    apply_mission_style,
    make_job,
    processes,
    rate_summary,
    replay_command,
    run_jobs,
    seeds_for,
    series_colors,
    write_rows,
)
from tools.e_exp029_part_b import _labels as part_b_labels
from tools.e_exp029_part_b import draw_outage_start
from tools.e_exp029_part_b import scenario_for as tag_scenario
from tools.e_exp031_sweep import _scenario as sweep_scenario

SCRIPT = "tools/e_exp042_hold.py"
ARMS = ("default", "hold")
HOLD_FIELDS = {"lost_link_action": "hold", "hold_land_after_s": 5.0}
BURSTS = ("burst_10_0.5s", "burst_30_0.5s")
TAG_CLASSES = ("turn", "jink", "figure8")
TAG_OUTAGE_S = 2.0
TAG_POLICY = {"weak_rule": "age", "T_w_s": 0.15, "T_s": 0.3, "degraded_action": "hold"}
SWEEP_DROPOUTS = ("none", "one")
SWEEP_MODE = "delta"
SWEEP_DURATION_S = 120.0
N_SEEDS, PILOT_SEEDS = 20, 2
TAG_FOLLOWERS, SWEEP_DRONES = 1, 3
SEPARATION_TOLERANCE_M = 0.05
LANDING_RATIO_LIMIT = 0.10
TAG_CSV = Path("sim/runs/e-exp029-part-b/part_b.csv")
SWEEP_CSV = Path("sim/runs/e-exp031-sweep/sweep.csv")
OUTPUTS = ("exp042.csv", "summary.json", "exp042_hold_simulated.png")
EXPECTED_CELLS = tuple([("tag", c, p) for c in TAG_CLASSES for p in BURSTS]
                       + [("sweep", d, p) for p in BURSTS for d in SWEEP_DROPOUTS])

# Label columns of this run's rows, and of the registered rows, by part.  Everything
# else in a row is a mission metric and must match the registered row exactly.
RUN_LABELS = {
    "tag": {"part", "arm", "seed_index", "seed", "adapter", "target_class", "outage_length_s", "preset",
            "weak_rule", "T_w_s", "T_s", "degraded_action"},
    "sweep": {"part", "arm", "seed_index", "seed", "adapter", "preset", "dropout", "message_mode"},
}
REGISTERED_LABELS = {
    "tag": {"seed", "adapter", "target_class", "outage_length_s", "preset", "weak_rule", "T_w_s", "T_s",
            "degraded_action", "blackout_start_s", "late_reaction"},
    "sweep": {"slice", "seed_index", "seed", "adapter", "preset", "dropout", "message_mode", "policy_name"},
}

PREDICTION_SOURCE = ("written in the project's experiment notes (kept outside this repository) before the "
                     "full run, after a 2-seed pilot of this grid; the repository alone cannot show the order")
PREDICTION = (
    "Landings: default lands about once per run in every burst cell; hold has 0 landings everywhere. "
    "Rule 1 passes. "
    "Tag success: hold at least default; about 18 to 20 of 20 on burst_10 and 12 to 18 of 20 on burst_30. "
    "Sweeps: default 0 of 20 in all four cells; hold at least 15 of 20 per cell, t100 about 40 to 60 s "
    "without dropout and 60 to 90 s with one. Rule 2 passes. "
    "Safety: rule 3 fails (more violation steps with hold, mostly on jink, and at least one tag cell's "
    "minimum separation more than 0.05 m below default). "
    "Overall predicted verdict: FAIL on safety, with large gains on landings and completion."
)
PASS_RULE = (
    "Rule 1, landings: in every cell hold landings are no more than default, and pooled hold landings "
    "are at most 10 % of pooled default landings. "
    "Rule 2, success: pooled tags with hold are at least the default count, and pooled completed sweeps "
    "with hold are at least the default count. "
    "Rule 3, safety: pooled separation-violation steps with hold are no more than default, and in every "
    "tag cell the hold arm's minimum separation is no more than 0.05 m below the default arm's. "
    "The run passes only if all three rules pass; the sweep has no separation metric, so safety is judged "
    "on the tag cells only."
)
EXP_LIMITS = (
    "Point-mass adapter only; the physics adapter and the monitor mission were not run.",
    ("Each tag run has one follower and one scripted 2.0 s total outage; the 5.0 s hold ceiling is a "
     "test value, not a measured burst length."),
    ("Comparison of two policies on synthetic burst presets; it does not show the hold rule is safe on "
     "real links."),
)


def seeds(part: str, *, pilot: bool = False) -> list[int]:
    """The registered seeds: the first 20 of EXP-029 Part B and the first 20 of the 50 EXP-031 seeds."""
    n = PILOT_SEEDS if pilot else N_SEEDS
    return seeds_for(29, N_SEEDS)[:n] if part == "tag" else seeds_for(31, 50)[:n]


def build_jobs(*, pilot: bool = False) -> list[dict]:
    jobs = []
    for arm in ARMS:
        for target_class in TAG_CLASSES:
            for preset in BURSTS:
                for seed_index, seed in enumerate(seeds("tag", pilot=pilot)):
                    scenario = tag_scenario(target_class=target_class, length_s=TAG_OUTAGE_S, preset=preset,
                                            policy=TAG_POLICY, seed=seed)
                    if arm == "hold":
                        scenario["policy"].update(HOLD_FIELDS)
                    labels = part_b_labels(target_class=target_class, length_s=TAG_OUTAGE_S, preset=preset,
                                           policy=TAG_POLICY, seed=seed, adapter="point_mass")
                    del labels["seed"], labels["adapter"]  # run_job writes both from the scenario
                    jobs.append(make_job({"part": "tag", "arm": arm, **labels, "seed_index": seed_index}, scenario))
        for preset in BURSTS:
            for dropout in SWEEP_DROPOUTS:
                for seed_index, seed in enumerate(seeds("sweep", pilot=pilot)):
                    policy = dict(HOLD_FIELDS) if arm == "hold" else None
                    scenario = sweep_scenario(seed, preset=preset, dropout=dropout, message_mode=SWEEP_MODE,
                                              policy=policy, duration_s=SWEEP_DURATION_S)
                    labels = {"part": "sweep", "arm": arm, "preset": preset, "dropout": dropout,
                              "message_mode": SWEEP_MODE, "seed_index": seed_index}
                    jobs.append(make_job(labels, scenario))
    return jobs


def cell_of(row: dict) -> tuple[str, str, str]:
    if row["part"] == "tag":
        return ("tag", row["target_class"], row["preset"])
    return ("sweep", row["dropout"], row["preset"])


def _text(value) -> str:
    """The CSV text of a value, as tools.e_common.write_rows writes it."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def registered_rows(repo: Path = REPO_ROOT) -> dict[tuple, dict]:
    """Registered rows keyed by (part, cell key, preset, seed), one per key."""
    out: dict[tuple, dict] = {}

    def add(key: tuple, row: dict) -> None:
        if key in out:
            raise ValueError(f"two registered rows for {key}")
        out[key] = row

    with open(repo / TAG_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["outage_length_s"] == "2.0" and row["T_w_s"] == "0.15" and row["T_s"] == "0.3"
                    and row["weak_rule"] == "age" and row["degraded_action"] == "hold"
                    and row["adapter"] == "point_mass" and row["preset"] in BURSTS):
                add(("tag", row["target_class"], row["preset"], row["seed"]), row)
    with open(repo / SWEEP_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["slice"] == "main" and row["message_mode"] == SWEEP_MODE and row["adapter"] == "point_mass"
                    and row["preset"] in BURSTS):
                add(("sweep", row["dropout"], row["preset"], row["seed"]), row)
    return out


def default_reproduction(rows: list[dict], reference: dict[tuple, dict]) -> dict:
    """Check every default-arm row against the registered row: same metric columns, same values.

    Tag rows also check the blackout start: the registered runner stores it as a label column,
    this run draws it from the same seed (``draw_outage_start``).
    """
    checked = mismatched = missing = 0
    examples = []
    for row in rows:
        if row["arm"] != "default":
            continue
        part = row["part"]
        ref = reference.get((*cell_of(row), str(row["seed"])))
        if ref is None:
            missing += 1
            continue
        checked += 1
        run_metrics = set(row) - RUN_LABELS[part]
        ref_metrics = set(ref) - REGISTERED_LABELS[part]
        problems = sorted(f"missing:{k}" for k in ref_metrics - run_metrics)
        problems += sorted(f"extra:{k}" for k in run_metrics - ref_metrics)
        problems += sorted(k for k in run_metrics & ref_metrics if _text(row[k]) != ref[k])
        if _text(row["adapter"]) != ref["adapter"]:
            problems.append("adapter")
        if part == "tag" and _text(draw_outage_start(int(row["seed"]))) != ref["blackout_start_s"]:
            problems.append("blackout_start_s")
        if problems:
            mismatched += 1
            if len(examples) < 5:
                examples.append({"part": part, "cell": list(cell_of(row)), "seed": row["seed"],
                                 "fields": problems[:6]})
    return {"rows_checked": checked, "rows_mismatched": mismatched, "rows_without_reference": missing,
            "examples": examples, "reproduced": checked > 0 and mismatched == 0 and missing == 0}


def cell_stats(rows: list[dict], part: str) -> dict:
    n = len(rows)
    if part == "tag":
        hours = sum(float(r["duration_s"]) for r in rows) * TAG_FOLLOWERS / 3600.0
        landings = sum(int(r["landings"]) for r in rows)
        seps = [float(r["min_separation_m"]) for r in rows if r["min_separation_m"] is not None]
        return {
            "runs": n,
            "landings": landings,
            "landings_before_outage": sum(int(r["landings_before_outage"]) for r in rows),
            "landings_per_run": landings / n,
            "landings_per_drone_hour": landings / hours,
            "tagged": rate_summary([bool(r["tagged"]) for r in rows]),
            "min_separation_m": min(seps) if seps else None,
            "separation_violation_steps": sum(int(r["separation_violations"]) for r in rows),
            "runs_with_violation": sum(int(r["separation_violations"]) > 0 for r in rows),
            "time_degraded_s_mean": sum(float(r["time_degraded_s"]) for r in rows) / n,
        }
    hours = sum(float(r["duration_s"]) for r in rows) * SWEEP_DRONES / 3600.0
    landings = sum(int(r["failsafe_landings"]) for r in rows)
    done = sorted(float(r["t100_s"]) for r in rows if r["end_reason"] == "complete")
    return {
        "runs": n,
        "landings": landings,
        "landings_per_run": landings / n,
        "landings_per_drone_hour": landings / hours,
        "completed": rate_summary([r["end_reason"] == "complete" for r in rows]),
        "t100_s_median_of_completed": done[len(done) // 2] if done else None,
        "false_loss_declarations": sum(int(r["false_loss_declarations"]) for r in rows),
        "duplicated_fraction_mean": sum(float(r["duplicated_fraction"]) for r in rows) / n,
    }


def summarize_cells(rows: list[dict]) -> list[dict]:
    cells = []
    for cell in sorted({cell_of(r) for r in rows}):
        part, key, preset = cell
        entry = {"part": part, ("target_class" if part == "tag" else "dropout"): key, "preset": preset}
        for arm in ARMS:
            arm_rows = [r for r in rows if r["arm"] == arm and cell_of(r) == cell]
            entry[arm] = cell_stats(arm_rows, part) if arm_rows else None
        cells.append(entry)
    return cells


def _cell_key(entry: dict) -> tuple[str, str, str]:
    return (entry["part"], entry.get("target_class", entry.get("dropout")), entry["preset"])


def validity(cells: list[dict], reproduction: dict, runs_per_arm: int) -> dict:
    """The run counts only if all expected cells exist, each arm has every seed, and the default arm reproduces."""
    present = {_cell_key(c): c for c in cells}
    missing = [list(k) for k in EXPECTED_CELLS if k not in present]
    unexpected = [list(k) for k in present if k not in EXPECTED_CELLS]
    short = [list(k) for k, c in present.items()
             if any(c[arm] is None or c[arm]["runs"] != runs_per_arm for arm in ARMS)]
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} expected cells missing")
    if unexpected:
        reasons.append(f"{len(unexpected)} unexpected cells")
    if short:
        reasons.append(f"{len(short)} cells without {runs_per_arm} runs in both arms")
    if not reproduction["reproduced"]:
        reasons.append("default arm does not reproduce the registered rows")
    return {"expected_cells": len(EXPECTED_CELLS), "cells_present": len(present), "runs_per_arm": runs_per_arm,
            "missing_cells": missing, "unexpected_cells": unexpected, "short_cells": short,
            "valid": not reasons, "reasons": reasons}


def verdict(cells: list[dict], valid: dict) -> dict:
    """The pre-registered pass rule.  A void run gets no pass or fail."""
    if not valid["valid"]:
        return {"status": "void", "pass": None, "reasons": list(valid["reasons"])}
    tag = [c for c in cells if c["part"] == "tag"]
    sweep = [c for c in cells if c["part"] == "sweep"]
    land_d = sum(c["default"]["landings"] for c in cells)
    land_h = sum(c["hold"]["landings"] for c in cells)
    per_cell = all(c["hold"]["landings"] <= c["default"]["landings"] for c in cells)
    rule1 = per_cell and land_h <= LANDING_RATIO_LIMIT * land_d
    tag_d = sum(c["default"]["tagged"]["count"] for c in tag)
    tag_h = sum(c["hold"]["tagged"]["count"] for c in tag)
    sw_d = sum(c["default"]["completed"]["count"] for c in sweep)
    sw_h = sum(c["hold"]["completed"]["count"] for c in sweep)
    rule2 = tag_h >= tag_d and sw_h >= sw_d
    viol_d = sum(c["default"]["separation_violation_steps"] for c in tag)
    viol_h = sum(c["hold"]["separation_violation_steps"] for c in tag)
    drops = []
    for c in tag:
        d, h = c["default"]["min_separation_m"], c["hold"]["min_separation_m"]
        ok = d is not None and h is not None and h >= d - SEPARATION_TOLERANCE_M
        if not ok:
            drops.append({"target_class": c["target_class"], "preset": c["preset"],
                          "default_min_separation_m": d, "hold_min_separation_m": h})
    rule3 = viol_h <= viol_d and not drops
    passed = rule1 and rule2 and rule3
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "rule_1_landings": {"pass": rule1, "every_cell_hold_at_most_default": per_cell,
                            "pooled_default": land_d, "pooled_hold": land_h,
                            "pooled_hold_limit": LANDING_RATIO_LIMIT * land_d},
        "rule_2_success": {"pass": rule2, "tagged_default": tag_d, "tagged_hold": tag_h,
                           "sweeps_completed_default": sw_d, "sweeps_completed_hold": sw_h},
        "rule_3_safety": {"pass": rule3, "violation_steps_default": viol_d, "violation_steps_hold": viol_h,
                          "tag_cells_min_separation_more_than_0.05_m_below_default": drops},
    }


def _in(value: float | None, low: float, high: float) -> bool:
    return value is not None and low <= value <= high


def prediction_check(cells: list[dict], result: dict) -> list[dict]:
    """Each sentence of the stored prediction against what this run observed (computed, not typed in)."""
    if result["status"] == "void":
        return []
    tag = [c for c in cells if c["part"] == "tag"]
    sweep = [c for c in cells if c["part"] == "sweep"]
    name = {"tag": lambda c: f"tag {c['target_class']} {c['preset']}",
            "sweep": lambda c: f"sweep dropout {c['dropout']} {c['preset']}"}
    # Landings per drone per run: a tag run has one follower, a sweep run three drones.
    per_drone = {name[c["part"]](c): c["default"]["landings_per_run"] / (TAG_FOLLOWERS if c["part"] == "tag"
                                                                         else SWEEP_DRONES) for c in cells}
    tags = {name["tag"](c): c["hold"]["tagged"]["count"] for c in tag}
    tag_bands = {"burst_10_0.5s": (18, 20), "burst_30_0.5s": (12, 18)}
    t100 = {name["sweep"](c): c["hold"]["t100_s_median_of_completed"] for c in sweep}
    t100_bands = {"none": (40.0, 60.0), "one": (60.0, 90.0)}
    jink_steps = sum(c["hold"]["separation_violation_steps"] for c in tag if c["target_class"] == "jink")
    safety = result["rule_3_safety"]
    items = [
        ("Default lands about once per run in every burst cell (read as once per drone per run).",
         {"default_landings_per_drone_per_run": per_drone},
         all(_in(v, 0.5, 1.5) for v in per_drone.values())),
        ("Hold has 0 landings everywhere.", {"hold_landings": result["rule_1_landings"]["pooled_hold"]},
         result["rule_1_landings"]["pooled_hold"] == 0),
        ("Rule 1 passes.", {"rule_1": result["rule_1_landings"]["pass"]}, result["rule_1_landings"]["pass"]),
        ("Tag success with hold: about 18 to 20 of 20 on burst_10 and 12 to 18 of 20 on burst_30.",
         {"hold_tagged_of_20": tags},
         all(_in(tags[name["tag"](c)], *tag_bands[c["preset"]]) for c in tag)),
        ("Sweeps: default completes 0 of 20 in all four cells.",
         {"default_completed": {name["sweep"](c): c["default"]["completed"]["count"] for c in sweep}},
         all(c["default"]["completed"]["count"] == 0 for c in sweep)),
        ("Sweeps: hold completes at least 15 of 20 per cell.",
         {"hold_completed": {name["sweep"](c): c["hold"]["completed"]["count"] for c in sweep}},
         all(c["hold"]["completed"]["count"] >= 15 for c in sweep)),
        ("Hold t100 about 40 to 60 s without dropout and 60 to 90 s with one.",
         {"hold_t100_s_median_of_completed": t100},
         all(_in(t100[name["sweep"](c)], *t100_bands[c["dropout"]]) for c in sweep)),
        ("Rule 2 passes.", {"rule_2": result["rule_2_success"]["pass"]}, result["rule_2_success"]["pass"]),
        ("Rule 3 fails: more violation steps with hold, mostly on jink, and at least one tag cell's minimum "
         "separation more than 0.05 m below default.",
         {"violation_steps_default": safety["violation_steps_default"],
          "violation_steps_hold": safety["violation_steps_hold"], "hold_steps_on_jink": jink_steps,
          "cells_more_than_0.05_m_below": len(safety["tag_cells_min_separation_more_than_0.05_m_below_default"])},
         safety["violation_steps_hold"] > safety["violation_steps_default"]
         and jink_steps > safety["violation_steps_hold"] / 2
         and bool(safety["tag_cells_min_separation_more_than_0.05_m_below_default"])),
        ("Overall verdict FAIL.", {"status": result["status"]}, result["status"] == "FAIL"),
    ]
    return [{"prediction": text, "observed": observed, "held": bool(held)} for text, observed, held in items]


def _figure(cells: list[dict]):
    pal = apply_mission_style()
    colors = dict(zip(ARMS, series_colors(len(ARMS))))
    tag = [c for c in cells if c["part"] == "tag"]
    names = [f"{c['part']}\n{c.get('target_class', c.get('dropout'))}\n{c['preset'].split('_0')[0]}"
             for c in cells]
    tag_names = [f"{c['target_class']}\n{c['preset'].split('_0')[0]}" for c in tag]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.6))
    width = 0.38
    panels = (
        (axes[0], names, cells, lambda s: s["landings_per_run"], "failsafe landings per run"),
        (axes[1], names, cells,
         lambda s: (s["tagged"] if "tagged" in s else s["completed"])["rate"], "tagged or sweep completed (rate)"),
        (axes[2], tag_names, tag, lambda s: s["min_separation_m"], "tag: minimum separation (m)"),
    )
    for ax, labels, group, value, ylabel in panels:
        xs = range(len(group))
        for k, arm in enumerate(ARMS):
            heights = [value(c[arm]) if c[arm] and value(c[arm]) is not None else 0.0 for c in group]
            ax.bar([x + (k - 0.5) * width for x in xs], heights, width, color=colors[arm], label=arm)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(0.0, 1.05)
    axes[2].axhline(0.5, color=pal["textSecondary"], linestyle="--", linewidth=1.0,
                    label="0.5 m separation floor")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="upper center", ncol=3, fontsize=8)
    axes[0].set_title("EXP-042 (simulated): registered landing rule vs hold-in-place, point mass, burst presets")
    fig.tight_layout()
    return fig


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   argv: list[str] | None = None, jobs: list[dict] | None = None,
                   repo: Path = REPO_ROOT) -> dict:
    output = prepare_output(Path(output))
    started = time.perf_counter()
    run_list = build_jobs(pilot=pilot) if jobs is None else list(jobs)
    rows = run_jobs(run_list, n_processes=processes(n_processes))
    write_rows(output / OUTPUTS[0], rows,
               first_columns=("part", "arm", "target_class", "dropout", "preset", "seed_index", "seed"))
    cells = summarize_cells(rows)
    reproduction = default_reproduction(rows, registered_rows(repo))
    valid = validity(cells, reproduction, PILOT_SEEDS if pilot else N_SEEDS)
    result = verdict(cells, valid)
    summary = {
        "experiment": "EXP-042",
        "label": "EXP-042, simulated comparison of two lost-link policies",
        "simulated": True,
        "pilot": bool(pilot),
        "jobs": len(run_list),
        "prediction": PREDICTION,
        "prediction_source": PREDICTION_SOURCE,
        "pass_rule": PASS_RULE,
        "hold_policy_fields": dict(HOLD_FIELDS),
        "cells": cells,
        "default_arm_reproduces_registered_rows": reproduction,
        "validity": valid,
        "verdict": result,
        "prediction_check": prediction_check(cells, result),
        "limits": list(MISSION_LIMITS) + list(EXP_LIMITS),
    }
    write_json(output / OUTPUTS[1], summary)
    fig = _figure(cells)
    fig.savefig(output / OUTPUTS[2], dpi=140)
    plt.close(fig)
    actual_argv = list(argv or [])
    write_manifest(output, schema="zephyr-exp042-1", argv=actual_argv,
                   command=replay_command(SCRIPT, actual_argv),
                   config={"experiment": "EXP-042", "pilot": pilot, "hold_policy_fields": dict(HOLD_FIELDS),
                           "tag_policy": dict(TAG_POLICY), "tag_outage_s": TAG_OUTAGE_S,
                           "sweep_duration_s": SWEEP_DURATION_S, "sweep_message_mode": SWEEP_MODE,
                           "tag_seeds": seeds("tag", pilot=pilot), "sweep_seeds": seeds("sweep", pilot=pilot)},
                   seeds=seeds("tag", pilot=pilot) + seeds("sweep", pilot=pilot), outputs=list(OUTPUTS),
                   limits=list(MISSION_LIMITS) + list(EXP_LIMITS), runtime_s=time.perf_counter() - started,
                   n_processes=processes(n_processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/tag.py", "sim/missions/sweep.py",
                                                           "tools/e_exp029_part_b.py", "tools/e_exp031_sweep.py",
                                                           SCRIPT])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    recorded_argv = list(sys.argv[1:] if argv is None else argv)
    summary = run_experiment(args.output, pilot=args.pilot, n_processes=args.processes, argv=recorded_argv)
    result = summary["verdict"]
    print(f"EXP-042 verdict: {result['status']}" + (f" ({'; '.join(result['reasons'])})"
                                                     if result["status"] == "void" else ""))
    return 2 if result["status"] == "void" else 0


if __name__ == "__main__":
    sys.exit(main())
