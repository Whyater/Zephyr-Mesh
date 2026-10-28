"""EXP-042: today's lost-link landing rule vs an opt-in hold-in-place rule.

Usage (repo root):
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/hold_exp042_compare.py \\
        --output sim/runs/exp042-hold --processes 3 [--pilot]

Simulated only.  This is a simulated comparison run and is not a registered
experiment.  It reruns the burst-loss cells of EXP-029 Part B (tag) and
EXP-031 (area sweep) with the same seeds under two policies:

* ``default``: the registered policy (a 0.30 s silence latches a landing).
* ``hold``: the same thresholds, but a silence of 0.30 s or more makes the
  drone hover in place and resume when packets return; it lands only after
  5.0 s of silence or on low battery (``lost_link_action="hold"``).

The default arm must reproduce the registered rows exactly; the script
checks that and records the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

from tools.e_common import make_job, run_jobs, seeds_for, write_rows
from tools.e_exp029_part_b import _labels, scenario_for
from tools.e_exp031_sweep import _scenario as sweep_scenario

HOLD_FIELDS = {"lost_link_action": "hold", "hold_land_after_s": 5.0}
BURSTS = ("burst_10_0.5s", "burst_30_0.5s")
TAG_CLASSES = ("turn", "jink", "figure8")
TAG_OUTAGE_S = 2.0
TAG_POLICY = {"weak_rule": "age", "T_w_s": 0.15, "T_s": 0.3, "degraded_action": "hold"}
SWEEP_DROPOUTS = ("none", "one")
SWEEP_MODE = "delta"
SWEEP_DURATION_S = 120.0
TAG_SEEDS, SWEEP_SEEDS = 20, 20
SWEEP_DRONES, TAG_FOLLOWERS = 3, 1
SEPARATION_TOLERANCE_M = 0.05


def build_jobs(*, pilot: bool = False) -> list[dict]:
    tag_seeds = seeds_for(29, TAG_SEEDS)[: 2 if pilot else TAG_SEEDS]
    sweep_seeds = seeds_for(31, 50)[: 2 if pilot else SWEEP_SEEDS]
    jobs = []
    for arm in ("default", "hold"):
        for target_class in TAG_CLASSES:
            for preset in BURSTS:
                for seed in tag_seeds:
                    scenario = scenario_for(target_class=target_class, length_s=TAG_OUTAGE_S, preset=preset,
                                            policy=TAG_POLICY, seed=seed)
                    if arm == "hold":
                        scenario["policy"].update(HOLD_FIELDS)
                    labels = {"part": "tag", "arm": arm, **_labels(
                        target_class=target_class, length_s=TAG_OUTAGE_S, preset=preset,
                        policy=TAG_POLICY, seed=seed, adapter="point_mass")}
                    labels.pop("seed")
                    labels.pop("adapter")
                    jobs.append(make_job(labels, scenario))
        for preset in BURSTS:
            for dropout in SWEEP_DROPOUTS:
                for seed_index, seed in enumerate(sweep_seeds):
                    policy = dict(HOLD_FIELDS) if arm == "hold" else None
                    scenario = sweep_scenario(seed, preset=preset, dropout=dropout, message_mode=SWEEP_MODE,
                                              policy=policy, duration_s=SWEEP_DURATION_S)
                    labels = {"part": "sweep", "arm": arm, "preset": preset, "dropout": dropout,
                              "message_mode": SWEEP_MODE, "seed_index": seed_index}
                    jobs.append(make_job(labels, scenario))
    return jobs


def _f(value) -> float | None:
    return None if value in (None, "") else float(value)


def _registered(path: Path, keys: tuple[str, ...], where: dict) -> dict[tuple, dict]:
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if all(row[k] == v for k, v in where.items()):
                out[tuple(row[k] for k in keys)] = row
    return out


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def default_reproduction(rows: list[dict], repo: Path) -> dict:
    """Compare every default-arm row with the registered row of the same cell and seed."""
    tag_ref = _registered(repo / "sim/runs/e-exp029-part-b/part_b.csv",
                          ("target_class", "preset", "seed"),
                          {"outage_length_s": "2.0", "T_w_s": "0.15", "T_s": "0.3",
                           "degraded_action": "hold", "adapter": "point_mass"})
    sweep_ref = _registered(repo / "sim/runs/e-exp031-sweep/sweep.csv", ("preset", "dropout", "seed"),
                            {"slice": "main", "message_mode": SWEEP_MODE, "adapter": "point_mass"})
    checked = mismatched = missing = 0
    examples = []
    for row in rows:
        if row["arm"] != "default":
            continue
        if row["part"] == "tag":
            ref = tag_ref.get((row["target_class"], row["preset"], str(row["seed"])))
        else:
            ref = sweep_ref.get((row["preset"], row["dropout"], str(row["seed"])))
        if ref is None:
            missing += 1
            continue
        shared = [k for k in row if k in ref and k not in {"part", "arm", "seed_index", "outage_length_s",
                                                          "T_w_s", "T_s"}]
        bad = [k for k in shared if _text(row[k]) != ref[k]]
        checked += 1
        if bad:
            mismatched += 1
            if len(examples) < 5:
                examples.append({"part": row["part"], "seed": row["seed"], "fields": bad[:6]})
    return {"rows_checked": checked, "rows_mismatched": mismatched, "rows_without_reference": missing,
            "examples": examples, "reproduced": checked > 0 and mismatched == 0 and missing == 0}


def _cell_stats(rows: list[dict], part: str) -> dict:
    n = len(rows)
    if part == "tag":
        hours = sum(float(r["duration_s"]) for r in rows) * TAG_FOLLOWERS / 3600.0
        landings = sum(int(r["landings"]) for r in rows)
        seps = [_f(r["min_separation_m"]) for r in rows if _f(r["min_separation_m"]) is not None]
        return {
            "runs": n,
            "landings": landings,
            "landings_before_outage": sum(int(r["landings_before_outage"]) for r in rows),
            "landings_per_run": landings / n,
            "landings_per_drone_hour": landings / hours if hours else None,
            "tagged": sum(bool(r["tagged"]) for r in rows),
            "tag_success": sum(bool(r["tagged"]) for r in rows) / n,
            "min_separation_m": min(seps) if seps else None,
            "separation_violation_steps": sum(int(r["separation_violations"]) for r in rows),
            "runs_with_violation": sum(int(r["separation_violations"]) > 0 for r in rows),
            "time_degraded_s_mean": sum(float(r["time_degraded_s"]) for r in rows) / n,
        }
    hours = sum(float(r["duration_s"]) for r in rows) * SWEEP_DRONES / 3600.0
    landings = sum(int(r["failsafe_landings"]) for r in rows)
    done = [float(r["t100_s"]) for r in rows if r["end_reason"] == "complete"]
    return {
        "runs": n,
        "landings": landings,
        "landings_per_run": landings / n,
        "landings_per_drone_hour": landings / hours if hours else None,
        "completed": len(done),
        "completion": len(done) / n,
        "t100_s_median_of_completed": sorted(done)[len(done) // 2] if done else None,
        "false_loss_declarations": sum(int(r["false_loss_declarations"]) for r in rows),
        "duplicated_fraction_mean": sum(float(r["duplicated_fraction"]) for r in rows) / n,
    }


def summarize(rows: list[dict]) -> dict:
    cells = []
    keys = {"tag": ("target_class", "preset"), "sweep": ("preset", "dropout")}
    for part, cell_keys in keys.items():
        part_rows = [r for r in rows if r["part"] == part]
        for cell in sorted({tuple(r[k] for k in cell_keys) for r in part_rows}):
            entry = {"part": part, **dict(zip(cell_keys, cell))}
            for arm in ("default", "hold"):
                arm_rows = [r for r in part_rows if r["arm"] == arm and tuple(r[k] for k in cell_keys) == cell]
                entry[arm] = _cell_stats(arm_rows, part) if arm_rows else None
            cells.append(entry)
    return {"cells": cells, "verdict": verdict(cells)}


def verdict(cells: list[dict]) -> dict:
    """The pre-registered pass rule (written before any full run).

    Raises ``ValueError`` when the tag or sweep part has no cell with both
    arms, because every rule would then compare zero with zero and pass.
    """
    complete = [c for c in cells if c["default"] and c["hold"]]
    for part in ("tag", "sweep"):
        if not any(c["part"] == part for c in complete):
            raise ValueError(f"no complete {part} cells (both arms run): the verdict would pass on nothing")
    land_d = sum(c["default"]["landings"] for c in complete)
    land_h = sum(c["hold"]["landings"] for c in complete)
    rule1 = all(c["hold"]["landings"] <= c["default"]["landings"] for c in complete) and (
        land_h <= 0.10 * land_d)
    tag = [c for c in complete if c["part"] == "tag"]
    sweep = [c for c in complete if c["part"] == "sweep"]
    tag_d = sum(c["default"]["tagged"] for c in tag)
    tag_h = sum(c["hold"]["tagged"] for c in tag)
    sw_d = sum(c["default"]["completed"] for c in sweep)
    sw_h = sum(c["hold"]["completed"] for c in sweep)
    rule2 = tag_h >= tag_d and sw_h >= sw_d
    viol_d = sum(c["default"]["separation_violation_steps"] for c in tag)
    viol_h = sum(c["hold"]["separation_violation_steps"] for c in tag)
    sep_ok = all(
        c["hold"]["min_separation_m"] is not None and c["default"]["min_separation_m"] is not None
        and c["hold"]["min_separation_m"] >= c["default"]["min_separation_m"] - SEPARATION_TOLERANCE_M
        for c in tag)
    rule3 = viol_h <= viol_d and sep_ok
    return {
        "rule_1_landings": {"pass": rule1, "default": land_d, "hold": land_h},
        "rule_2_success": {"pass": rule2, "tagged_default": tag_d, "tagged_hold": tag_h,
                           "sweeps_completed_default": sw_d, "sweeps_completed_hold": sw_h},
        "rule_3_safety": {"pass": rule3, "violation_steps_default": viol_d, "violation_steps_hold": viol_h,
                          "every_cell_min_separation_within_0.05_m": sep_ok},
        "pass": rule1 and rule2 and rule3,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(pilot=args.pilot)
    started, cpu0 = time.time(), time.process_time()
    rows = run_jobs(jobs, n_processes=args.processes)
    for row in rows:
        row.pop("policy_name", None)
    write_rows(args.output / "rows.csv", rows, ("part", "arm", "target_class", "preset", "dropout", "seed"))
    try:
        summary = summarize(rows)
    except ValueError as error:
        raise SystemExit(f"EXP-042 verdict not computed: {error}") from error
    summary["default_arm_reproduces_registered_rows"] = default_reproduction(rows, repo)
    summary["simulated"] = True
    summary["label"] = "EXP-042, simulated comparison; not registered"
    summary["pilot"] = bool(args.pilot)
    summary["jobs"] = len(jobs)
    summary["wall_s"] = round(time.time() - started, 1)
    summary["parent_cpu_s"] = round(time.process_time() - cpu0, 1)
    with open(args.output / "summary.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"verdict": summary["verdict"],
                      "default_reproduction": summary["default_arm_reproduces_registered_rows"],
                      "wall_s": summary["wall_s"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
