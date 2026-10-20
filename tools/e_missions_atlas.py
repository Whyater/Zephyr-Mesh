"""Missions atlas: where each simulated Run E mission stops meeting its goal.

Usage (repo root):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/e_missions_atlas.py --output runs/missions-atlas

Everything summarised here is simulated.  Link loss patterns are synthetic
presets (sim/missions/presets.py) and vehicle parameters are literature-derived.

The atlas reads the per-run CSV of every experiment folder (EXP-029 Part A and
Part B, EXP-030 to EXP-034) and applies the goal definitions that were fixed
before any full-grid run (pre-registration Addendum A, copied word for word
into GOAL_TEXT below).  Every condition is recomputed from the CSV rows with
tools/e_common.py rate_summary (Wilson 95 %) and median_summary (order-statistic
95 % interval); the runner summaries are read only to detect pilots and, for
EXP-029 Part A and EXP-031, as a cross-check.

Rules applied to every axis (delay, distance or fix latency): a cell meets
the goal only if every condition holds, and a cell with no runs counts as not
met.  The tolerated value is the largest grid value where the goal is met at
that value and at every smaller one; the bracket is [last met, first not met],
with first not met = null when every value is met and tolerated = null when
the smallest value already fails.

An input folder that is missing, unfinished, or does not match its own
manifest gives "not available"; a pilot folder gives "pilot only, not a
result" and its numbers are left out.  With --pilot-preview the pilot numbers
are written under "pilot_entries" (never "entries"), only to check the
atlas on pilot outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import jsonschema

from sim.missions.presets import PRESET_NAMES, get_preset
from tools.c_common import REPO_ROOT, argv_or_default, prepare_output, sha256_file, write_json, write_manifest
from tools.e_common import group_rows, median_summary, rate_summary, replay_command

SCRIPT = "tools/e_missions_atlas.py"
SCHEMA_PATH = REPO_ROOT / "sim" / "missions" / "missions_atlas_schema.json"
ATLAS_SCHEMA_ID = "zephyr-missions-atlas-1"
OUTPUT_NAME = "missions_atlas.json"
DEFAULT_INPUTS = REPO_ROOT / "runs"

RESULT = "result"
NOT_AVAILABLE = "not available"
PILOT_ONLY = "pilot only, not a result"

# key, default folder name under --inputs, CSV name, experiment label
INPUT_SPECS = (
    ("exp029_part_a", "e-exp029-part-a", "part_a.csv", "EXP-029 Part A"),
    ("exp029_part_b", "e-exp029-part-b", "part_b.csv", "EXP-029 Part B"),
    ("exp030", "e-exp030-tag", "tag.csv", "EXP-030"),
    ("exp031", "e-exp031-sweep", "sweep.csv", "EXP-031"),
    ("exp032", "e-exp032-relay", "relay.csv", "EXP-032"),
    ("exp033", "e-exp033-monitor", "exp033.csv", "EXP-033"),
    ("exp034", "e-exp034-formation", "formation.csv", "EXP-034"),
)
REQUIRED_COLUMNS = {
    "exp029_part_a": ("preset", "h_hz", "weak_rule", "T_w_s", "T_s", "hours", "unnecessary_landings",
                      "landings_per_hour", "landings_per_hour_low_95", "landings_per_hour_high_95",
                      "closed_form_per_hour"),
    "exp029_part_b": ("preset", "T_w_s", "T_s", "degraded_action", "adapter", "airborne_at_blackout",
                      "violation_during_or_after_outage", "nominal_at_age_0_33"),
    "exp030": ("slice", "target_class", "measurement_delay_s", "preset", "arm", "n_followers", "adapter",
               "policy_name", "tagged", "separation_violations"),
    "exp031": ("slice", "preset", "dropout", "message_mode", "adapter", "t100_s", "duplicated_fraction"),
    "exp032": ("adapter", "distance_m", "relay_fraction", "delivery_ratio"),
    "exp033": ("adapter", "path", "preset", "delay_s", "missed_alert", "false_alert", "alert_latency_s"),
    "exp034": ("slice", "sigma_m", "fix_latency_s", "preset", "adapter", "rms_error_m", "min_separation_m"),
}
FULL_PART_A_HOURS = 10.0

# Pre-registration Addendum A, copied word for word (bullet markers removed).
GOAL_SOURCE = ("Run E pre-registration, Addendum A: E7 missions atlas goal definitions (written 2026-10-15 "
               "18:40, before any full-grid run; only pilots of a few cells had run)")
GOAL_RULE_TEXT = (
    'The atlas answers "at what latency and loss does each mission stop meeting its goal". A goal is met in a '
    "cell only if every condition holds; conditions use the 95 % interval, so a cell with too few seeds to tell "
    "counts as not met. These thresholds are Run E design choices for reporting, not product rules (changing a "
    "product threshold is a Proposed decision)."
)
GOAL_TEXT = {
    "tag": ("Tag (EXP-030, 1 follower, default policy): Wilson lower bound of tag success >= 0.80 AND zero "
            "separation violations in the cell. Per class, preset and arm: the largest grid delay where the goal "
            'is met at that delay and every smaller delay ("tolerated delay"), reported as a bracket [last met, '
            "first not met] (first not met = null if all met; tolerated = null if delay 0 already fails)."),
    "sweep": ("Sweep (EXP-031): goal met if Wilson lower bound of P(t100 reached within 120 s) >= 0.80 AND median "
              "duplicated_fraction <= 0.25. Per preset, dropout and message mode: met or not (no delay axis in "
              "this experiment)."),
    "relay": ("Relay (EXP-032): goal met at distance D if mean delivery ratio over seeds >= 0.95 AND the lower end "
              "of its median interval >= 0.90. Per f: the largest D met at that D and every smaller D, as a "
              "bracket."),
    "monitor": ("Monitor (EXP-033, crossing and loiter paths): Wilson upper bound of missed-alert rate <= 0.10 AND "
                "median alert latency <= 1.0 s; skimming path: Wilson upper bound of false-alert rate <= 0.10. Per "
                "path and preset: tolerated delay bracket as for tag."),
    "formation": ("Formation (EXP-034): goal met if the upper end of the median interval of rms_error_m <= 0.10 m "
                  "AND minimum pairwise separation over every seed >= 0.5 m. Per adapter, preset and sigma: "
                  "tolerated fix latency bracket."),
    "failsafe": ("Failsafe (EXP-029): the recommended policy per preset is the pre-registered rule (smallest T "
                 "with under 1 unnecessary landing per hour at 20 Hz, T >= T_w + 0.05, T_w 0.15), with the Part B "
                 'late-reaction rate at that T and at 0.30 s; "none up to 2 s" is a valid entry.'),
}
LOSS_AXIS_TEXT = (
    "Loss axis: presets are not one number, so the atlas lists each preset by its long-run loss fraction and mean "
    "burst length (from sim/missions/presets.py) and gives the delay bracket per preset; it does not interpolate "
    "between presets."
)

# Goal thresholds (Addendum A).
TAG_SUCCESS_LOW = 0.80
SWEEP_COMPLETE_LOW = 0.80
SWEEP_HORIZON_S = 120.0
SWEEP_DUPLICATED_MEDIAN = 0.25
RELAY_MEAN = 0.95
RELAY_MEDIAN_LOW = 0.90
MONITOR_MISSED_HIGH = 0.10
MONITOR_LATENCY_MEDIAN_S = 1.0
MONITOR_FALSE_HIGH = 0.10
FORMATION_RMS_HIGH_M = 0.10
FORMATION_SEPARATION_M = 0.5
# Failsafe rule (Addendum A and the EXP-029 pre-registration).
RULE_H_HZ = 20.0
RULE_T_W_S = 0.15
MIN_T_ABOVE_T_W_S = 0.05
RULE_LANDINGS_PER_HOUR = 1.0
PLANNED_T_S = 0.30
DEGRADED_ACTIONS = ("hold", "slow")

SCOPE_GOAL = "pre-registered grid, Addendum A goal"
SCOPE_EXTRA = "pre-registered extra slice, Addendum A goal applied"
SCOPE_PHYSICS = "pre-registered physics subset, Addendum A goal applied"
SCOPE_EXPLORATORY = "exploratory slice (Addendum B, added after the pilots), not pre-registered"
SCOPE_PILOT = "pilot run, not a result"
SCOPE_COMPARISON = "comparison slice added after the pre-registration, not pre-registered"
SWEEP_EXPLORATORY_SLICE = "exploratory, added after the pilot, not pre-registered"
SLICE_SCOPES = {
    "tag": {"main": SCOPE_GOAL, "calm": SCOPE_EXTRA, "followers": SCOPE_EXTRA, "physics": SCOPE_PHYSICS,
            "pilot": SCOPE_PILOT, "pilot_physics": SCOPE_PILOT},
    "sweep": {"main": SCOPE_GOAL, "physics": SCOPE_PHYSICS, SWEEP_EXPLORATORY_SLICE: SCOPE_EXPLORATORY},
    "relay": {"main": SCOPE_GOAL, "physics": SCOPE_PHYSICS},
    "monitor": {"main": SCOPE_GOAL, "physics": SCOPE_PHYSICS},
    "formation": {"main": SCOPE_GOAL, "physics": SCOPE_PHYSICS,
                  "coop_comparison": SCOPE_COMPARISON, "pilot": SCOPE_PILOT,
                  "pilot_physics": SCOPE_PILOT},
}
ADAPTER_LABELS = {
    "point_mass": "point mass (SimAdapter)",
    "physics": "physics (crazyflie-class PhysicsAdapter, literature-derived, drag off, modelled floor)",
    "point_mass_offboard": "point mass, position loop closed on camera-like fixes",
    "physics_offboard": "physics (crazyflie-class, literature-derived), position loop closed on camera-like fixes",
}
FOLLOWERS_NOTE = ("Two and three followers keep apart with Run D's reciprocal velocity rule (separation_rule coop). "
                  "Rows with separation_rule local are labelled as the comparison rule.")
EXPLORATORY_NOTE = ("Calm policy (degraded after 0.25 s, land after 1.0 s) added after the pilots (Addendum B); "
                    "exploratory, not pre-registered.")
RELAY_PHYSICS_NOTE = ("Physics rows equal point-mass rows by construction: path loss uses the fixed commanded "
                      "positions, so the physics adapter changes the vehicles, not the radio geometry.")

LIMITS = (
    "Simulated only: no drone, camera or radio hardware was used.",
    "Link loss patterns are synthetic presets (sim/missions/presets.py), never measured; presets are listed by "
    "their long-run loss fraction and mean burst length and are never interpolated.",
    "Vehicle parameters are literature-derived (point mass, or the crazyflie-class physics profile with drag off "
    "and a modelled floor).",
    "Every entry names its adapter: the full grids run on the point-mass adapter; physics entries are the "
    "pre-registered physics subsets (fewer seeds and cells) and are labelled as such.",
    "Multi-follower tag entries use Run D's reciprocal velocity rule (decision D-123); rows with separation_rule "
    "local are labelled as the comparison rule and exist only for comparison.",
    "Formation entries use the registered pairwise 0.8 m guard (separation_rule local, decision D-126); the "
    "coop_comparison slice runs Run D's rule, was added after the pre-registration and exists only for comparison.",
    "The EXP-031 calm-policy slice is exploratory (Addendum B, added after the pilots, not pre-registered) and is "
    "labelled in every entry; pre-registered grids are listed first.",
    "Goal thresholds are Run E reporting choices fixed before any full-grid run (Addendum A), not product rules.",
    "EXP-029 Part B ran a reduced grid (Addendum C): five presets, seven (T_w, T) pairs, 20 seeds; Part A rates "
    "cover every preset.",
    "EXP-033 counts an estimate older than 1.0 s as outside the alert zone (Addendum D).",
    "EXP-034 minimum separation is taken over the whole run, including the start from offset positions.",
)


# ---- reading inputs -----------------------------------------------------------------

def parse_cell(text: str):
    """One CSV cell as written by tools/e_common.write_rows: empty is None, true/false are booleans."""
    if text == "":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        return text
    return value


def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [{key: parse_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def display_path(path: Path) -> str:
    """Folder path relative to the repo when inside it, else only its name (no absolute paths)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pilot_reasons(key: str, summary: dict, manifest: dict, rows: list[dict]) -> list[str]:
    reasons = []
    if summary.get("pilot") is True:
        reasons.append('summary.json has "pilot": true')
    if (manifest.get("config") or {}).get("pilot") is True:
        reasons.append('manifest.json config has "pilot": true')
    if key == "exp029_part_a":
        hours = sorted({float(row["hours"]) for row in rows if row.get("hours") is not None})
        if any(h < FULL_PART_A_HOURS for h in hours):
            reasons.append(f"Part A rows cover {hours} simulated hours, not the full run's {FULL_PART_A_HOURS:g} h")
    return reasons


def load_input(key: str, folder: Path, csv_name: str, experiment: str) -> dict:
    """Read one experiment folder and decide whether it holds a result."""
    item = {"key": key, "experiment": experiment, "folder": display_path(folder), "csv": csv_name,
            "status": NOT_AVAILABLE, "reason": None, "pilot_reasons": [], "git_revision": None,
            "git_tree_state": None, "manifest_schema": None, "files": {}, "rows": None, "summary": None}
    folder = Path(folder)
    if not folder.is_dir():
        item["reason"] = "folder not found"
        return item
    names = (csv_name, "summary.json", "manifest.json")
    for name in names:
        if (folder / name).is_file():
            item["files"][name] = sha256_file(folder / name)
    missing = [name for name in names if name not in item["files"]]
    if missing:
        item["reason"] = "missing " + ", ".join(missing) + " (the run may not have finished)"
        return item
    manifest = _read_json(folder / "manifest.json")
    summary = _read_json(folder / "summary.json")
    item["git_revision"] = manifest.get("git_revision")
    item["git_tree_state"] = manifest.get("git_tree_state")
    item["manifest_schema"] = manifest.get("schema")
    listed = manifest.get("outputs") or {}
    for name in (csv_name, "summary.json"):
        if listed.get(name) != item["files"][name]:
            item["reason"] = f"{name} does not match the sha256 in manifest.json"
            return item
    rows = read_rows(folder / csv_name)
    if not rows:
        item["reason"] = f"{csv_name} has no rows"
        return item
    absent = [c for c in REQUIRED_COLUMNS[key] if c not in rows[0]]
    if absent:
        item["reason"] = f"{csv_name} lacks columns {absent}"
        return item
    item["rows"] = rows
    item["summary"] = summary
    item["pilot_reasons"] = pilot_reasons(key, summary, manifest, rows)
    item["status"] = PILOT_ONLY if item["pilot_reasons"] else RESULT
    return item


def input_record(item: dict) -> dict:
    """The JSON part of a loaded input (no rows)."""
    return {name: item[name] for name in ("experiment", "folder", "csv", "status", "reason", "pilot_reasons",
                                          "git_revision", "git_tree_state", "manifest_schema", "files")}


# ---- conditions and brackets ---------------------------------------------------------

def _empty_rate() -> dict:
    return {"n": 0, "count": 0, "rate": None, "wilson_low": None, "wilson_high": None}


def _compare(value, comparison: str, threshold: float) -> bool:
    if value is None:
        return False
    if comparison == ">=":
        return value >= threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == "==":
        return value == threshold
    raise ValueError(f"unknown comparison {comparison!r}")


def _condition(cid: str, text: str, statistic: str, summary: dict, compared, comparison: str,
               threshold: float) -> dict:
    return {"id": cid, "text": text, "statistic": statistic, "summary": summary, "compared": compared,
            "comparison": comparison, "threshold": threshold, "met": _compare(compared, comparison, threshold)}


def rate_condition(cid: str, text: str, flags: list[bool], bound: str, comparison: str, threshold: float) -> dict:
    """A Wilson bound ("wilson_low" or "wilson_high") compared with a threshold; no runs is not met."""
    summary = rate_summary([bool(f) for f in flags]) if flags else _empty_rate()
    return _condition(cid, text, bound, summary, summary[bound], comparison, threshold)


def median_condition(cid: str, text: str, values: list, which: str, comparison: str, threshold: float) -> dict:
    """The median ("median") or an end of its 95 % interval ("low", "high") compared with a threshold."""
    summary = median_summary(values)
    statistic = {"median": "median", "low": "median_low", "high": "median_high"}[which]
    return _condition(cid, text, statistic, summary, summary[which], comparison, threshold)


def mean_condition(cid: str, text: str, values: list, comparison: str, threshold: float) -> dict:
    present = [float(v) for v in values if v is not None]
    mean = math.fsum(present) / len(present) if present else None
    return _condition(cid, text, "mean", {"n": len(present), "mean": mean}, mean, comparison, threshold)


def total_condition(cid: str, text: str, values: list, comparison: str, threshold: float) -> dict:
    present = [int(v) for v in values if v is not None]
    total = sum(present) if present else None
    return _condition(cid, text, "total", {"n": len(present), "total": total}, total, comparison, threshold)


def minimum_condition(cid: str, text: str, values: list, comparison: str, threshold: float) -> dict:
    present = [float(v) for v in values if v is not None]
    minimum = min(present) if present else None
    return _condition(cid, text, "minimum", {"n": len(present), "minimum": minimum}, minimum, comparison, threshold)


def make_cell(value, rows: list[dict], conditions_for) -> dict:
    """One grid cell: the goal is met only if there are runs and every condition holds."""
    conditions = conditions_for(rows)
    return {"value": value, "n_runs": len(rows), "missing": not rows, "conditions": conditions,
            "met": bool(rows) and all(c["met"] for c in conditions)}


def bracket(cells: list[dict], axis: str, unit: str) -> dict:
    """Tolerated value and [last met, first not met] over cells sorted by their axis value."""
    ordered = sorted(cells, key=lambda c: c["value"])
    last_met = None
    first_not_met = None
    met_again = False
    for cell in ordered:
        if first_not_met is None:
            if cell["met"]:
                last_met = cell["value"]
            else:
                first_not_met = cell["value"]
        elif cell["met"]:
            met_again = True
    return {"axis": axis, "unit": unit, "values": [c["value"] for c in ordered], "tolerated": last_met,
            "bracket": [last_met, first_not_met], "met_again_after_first_not_met": met_again}


def axis_entries(rows: list[dict], *, mission: str, group_keys: tuple, axis: str, unit: str,
                 conditions_for, describe) -> list[dict]:
    """One entry per group of rows; the cells are the axis values present in the group's slice.

    A group that lacks one of its slice's axis values gets an empty cell there
    (not met), so a missing cell can never make the tolerated value larger.
    """
    slice_values: dict[tuple, set] = {}
    for row in rows:
        slice_values.setdefault((row["slice"], row["adapter"]), set()).add(row[axis])
    entries = []
    for coords, group in group_rows(rows, group_keys).items():
        labels = dict(zip(group_keys, coords))
        by_value = group_rows(group, (axis,))
        values = sorted(slice_values[(labels["slice"], labels["adapter"])])
        cells = [make_cell(v, by_value.get((v,), []), conditions_for) for v in values]
        entry = common_labels(mission, labels)
        entry.update(describe(labels))
        entry["cells"] = cells
        entry["bracket"] = bracket(cells, axis, unit)
        entries.append(entry)
    return entries


def common_labels(mission: str, labels: dict) -> dict:
    scopes = SLICE_SCOPES[mission]
    if labels["slice"] not in scopes:
        raise ValueError(f"{mission}: unrecognised slice {labels['slice']!r}")
    adapter = labels["adapter"]
    return {"slice": labels["slice"], "scope": scopes[labels["slice"]], "adapter": adapter,
            "adapter_label": ADAPTER_LABELS.get(adapter, adapter)}


def _with_slice(rows: list[dict]) -> list[dict]:
    """EXP-032 and EXP-033 rows have no slice column: point mass is the main grid, physics the subset."""
    return [dict(row, slice="main" if row["adapter"] == "point_mass" else "physics") for row in rows]


# ---- missions -------------------------------------------------------------------------

def tag_conditions(rows: list[dict]) -> list[dict]:
    return [
        rate_condition("tag_success_wilson_low", "Wilson lower bound of tag success >= 0.80",
                       [r["tagged"] for r in rows], "wilson_low", ">=", TAG_SUCCESS_LOW),
        total_condition("separation_violations_total", "zero separation violations in the cell",
                        [r["separation_violations"] for r in rows], "==", 0),
    ]


def tag_entries(rows: list[dict]) -> list[dict]:
    rows = [dict(row, separation_rule=row.get("separation_rule", "coop")) for row in rows]
    def describe(labels: dict) -> dict:
        notes = [FOLLOWERS_NOTE] if int(labels["n_followers"]) > 1 else []
        if labels.get("separation_rule") == "local":
            notes.append("local separation rule, comparison rule")
        return {"target_class": labels["target_class"], "preset": labels["preset"], "arm": labels["arm"],
                "n_followers": int(labels["n_followers"]), "policy": labels["policy_name"] or "default",
                "notes": notes}

    keys = ("slice", "target_class", "preset", "arm", "n_followers", "adapter", "policy_name", "separation_rule")
    return axis_entries(rows, mission="tag", group_keys=keys, axis="measurement_delay_s", unit="s",
                        conditions_for=tag_conditions, describe=describe)


def sweep_conditions(rows: list[dict]) -> list[dict]:
    complete = [r["t100_s"] is not None and float(r["t100_s"]) <= SWEEP_HORIZON_S for r in rows]
    return [
        rate_condition("t100_within_120_s_wilson_low", "Wilson lower bound of P(t100 reached within 120 s) >= 0.80",
                       complete, "wilson_low", ">=", SWEEP_COMPLETE_LOW),
        median_condition("duplicated_fraction_median", "median duplicated_fraction <= 0.25",
                         [r["duplicated_fraction"] for r in rows], "median", "<=", SWEEP_DUPLICATED_MEDIAN),
    ]


def sweep_entries(rows: list[dict]) -> list[dict]:
    keys = ("slice", "preset", "dropout", "message_mode", "adapter", "policy_name")
    entries = []
    for coords, group in group_rows(rows, keys).items():
        labels = dict(zip(keys, coords))
        entry = common_labels("sweep", labels)
        cell = make_cell(None, group, sweep_conditions)
        entry.update({"preset": labels["preset"], "dropout": labels["dropout"],
                      "message_mode": labels["message_mode"], "policy": labels["policy_name"] or "default",
                      "cell": cell, "met": cell["met"],
                      "notes": [EXPLORATORY_NOTE] if entry["scope"] == SCOPE_EXPLORATORY else []})
        entries.append(entry)
    # Pre-registered grid first, then the physics subset, then the exploratory slice.
    order = {SCOPE_GOAL: 0, SCOPE_PHYSICS: 1, SCOPE_EXPLORATORY: 2, SCOPE_PILOT: 3}
    return sorted(entries, key=lambda e: order[e["scope"]])


def relay_conditions(rows: list[dict]) -> list[dict]:
    values = [r["delivery_ratio"] for r in rows]
    return [
        mean_condition("delivery_ratio_mean", "mean delivery ratio over seeds >= 0.95", values, ">=", RELAY_MEAN),
        median_condition("delivery_ratio_median_low", "lower end of the median interval of delivery ratio >= 0.90",
                         values, "low", ">=", RELAY_MEDIAN_LOW),
    ]


def relay_entries(rows: list[dict]) -> list[dict]:
    def describe(labels: dict) -> dict:
        notes = [RELAY_PHYSICS_NOTE] if labels["adapter"] == "physics" else []
        return {"relay_fraction": labels["relay_fraction"], "notes": notes}

    return axis_entries(_with_slice(rows), mission="relay", group_keys=("slice", "relay_fraction", "adapter"),
                        axis="distance_m", unit="m", conditions_for=relay_conditions, describe=describe)


def monitor_conditions_for(path: str):
    if path == "skimming":
        def conditions(rows: list[dict]) -> list[dict]:
            return [rate_condition("false_alert_wilson_high", "Wilson upper bound of false-alert rate <= 0.10",
                                   [r["false_alert"] for r in rows], "wilson_high", "<=", MONITOR_FALSE_HIGH)]
        return conditions
    if path in ("crossing", "loiter"):
        def conditions(rows: list[dict]) -> list[dict]:
            return [
                rate_condition("missed_alert_wilson_high", "Wilson upper bound of missed-alert rate <= 0.10",
                               [r["missed_alert"] for r in rows], "wilson_high", "<=", MONITOR_MISSED_HIGH),
                median_condition("alert_latency_median", "median alert latency <= 1.0 s (runs with an alert)",
                                 [r["alert_latency_s"] for r in rows], "median", "<=", MONITOR_LATENCY_MEDIAN_S),
            ]
        return conditions
    raise ValueError(f"unknown monitor path {path!r}")


def monitor_entries(rows: list[dict]) -> list[dict]:
    rows = _with_slice(rows)
    entries = []
    for path in sorted({r["path"] for r in rows}, key=lambda p: ("crossing", "loiter", "skimming").index(p)):
        selected = [r for r in rows if r["path"] == path]
        entries += axis_entries(selected, mission="monitor", group_keys=("slice", "path", "preset", "adapter"),
                                axis="delay_s", unit="s", conditions_for=monitor_conditions_for(path),
                                describe=lambda labels: {"path": labels["path"], "preset": labels["preset"],
                                                         "notes": []})
    for entry in entries:
        note = seed_count_note(entry)
        if note:
            entry["notes"].append(note)
    order = {SCOPE_GOAL: 0, SCOPE_PHYSICS: 1}
    return sorted(entries, key=lambda e: order[e["scope"]])


def seed_count_note(entry: dict) -> str | None:
    """A note for cells that fail a Wilson upper-bound goal with zero events.

    With n runs and no events the Wilson upper bound is still above zero, so a
    small seed count can miss the goal even when nothing went wrong.
    """
    counts = sorted({c["summary"]["n"] for cell in entry["cells"] for c in cell["conditions"]
                     if c["statistic"] == "wilson_high" and not c["met"]
                     and c["summary"]["n"] and c["summary"]["count"] == 0})
    if not counts:
        return None
    seeds = ", ".join(str(n) for n in counts)
    return (f"Some cells are not met only because {seeds} runs with zero events still give a Wilson upper "
            "bound above the goal; no event was observed in those cells.")


def formation_conditions(rows: list[dict]) -> list[dict]:
    return [
        median_condition("rms_error_median_high", "upper end of the median interval of rms_error_m <= 0.10 m",
                         [r["rms_error_m"] for r in rows], "high", "<=", FORMATION_RMS_HIGH_M),
        minimum_condition("min_separation_minimum", "minimum pairwise separation over every seed >= 0.5 m",
                          [r["min_separation_m"] for r in rows], ">=", FORMATION_SEPARATION_M),
    ]


def formation_entries(rows: list[dict]) -> list[dict]:
    rows = [dict(row, separation_rule=row.get("separation_rule", "local")) for row in rows]
    def describe(labels: dict) -> dict:
        if labels.get("separation_rule") == "coop":
            notes = ["Run D separation rule, comparison slice added after the pre-registration"]
        else:
            notes = ["registered 0.8 m separation guard"]
        return {"sigma_m": labels["sigma_m"], "preset": labels["preset"], "notes": notes}

    return axis_entries(rows, mission="formation", group_keys=("slice", "sigma_m", "preset", "adapter", "separation_rule"),
                        axis="fix_latency_s", unit="s", conditions_for=formation_conditions, describe=describe)


# ---- failsafe (EXP-029) ----------------------------------------------------------------

def _landing_row(row: dict) -> dict:
    return {"T_s": float(row["T_s"]), "count": int(row["unnecessary_landings"]), "hours": float(row["hours"]),
            "landings_per_hour": float(row["landings_per_hour"]),
            "low_95": float(row["landings_per_hour_low_95"]), "high_95": float(row["landings_per_hour_high_95"]),
            "closed_form_per_hour": float(row["closed_form_per_hour"]),
            "under_1_per_hour": float(row["landings_per_hour"]) < RULE_LANDINGS_PER_HOUR}


def recommended_timeout(landing_rows: list[dict]) -> float | None:
    """Smallest T under 1 unnecessary landing per hour with T >= T_w + 0.05 s (T_w 0.15 s)."""
    allowed = [r["T_s"] for r in landing_rows
               if r["T_s"] >= RULE_T_W_S + MIN_T_ABOVE_T_W_S - 1e-9 and r["under_1_per_hour"]]
    return min(allowed) if allowed else None


def late_items(part_b_rows: list[dict], preset: str, timeout: float) -> list[dict]:
    """Part B late-reaction rates at one T (T_w 0.15 s), pooled over classes and blackout lengths."""
    items = []
    selected = [r for r in part_b_rows if r["preset"] == preset and float(r["T_w_s"]) == RULE_T_W_S
                and abs(float(r["T_s"]) - timeout) < 1e-9]
    for (adapter, action), group in group_rows(selected, ("adapter", "degraded_action")).items():
        late = [bool(r["violation_during_or_after_outage"]) or bool(r["nominal_at_age_0_33"]) for r in group]
        airborne = [flag for flag, r in zip(late, group) if r["airborne_at_blackout"]]
        items.append({"adapter": adapter, "adapter_label": ADAPTER_LABELS.get(adapter, adapter),
                      "degraded_action": action, "T_s": timeout, "n_runs": len(group),
                      "late": rate_summary(late),
                      "late_airborne_at_blackout": rate_summary(airborne) if airborne else _empty_rate(),
                      "grounded_before_blackout": len(group) - len(airborne)})
    return sorted(items, key=lambda i: (i["adapter"] != "point_mass", i["adapter"],
                                        DEGRADED_ACTIONS.index(i["degraded_action"])))


def failsafe_entries(part_a_rows: list[dict], part_b_rows: list[dict] | None, part_b_status: str) -> list[dict]:
    rule_rows = [r for r in part_a_rows if r["weak_rule"] == "age" and float(r["h_hz"]) == RULE_H_HZ
                 and float(r["T_w_s"]) == RULE_T_W_S]
    by_preset = group_rows(rule_rows, ("preset",))
    part_b_presets = set() if part_b_rows is None else {r["preset"] for r in part_b_rows}
    entries = []
    for preset in [p for p in PRESET_NAMES if (p,) in by_preset]:
        landing_rows = sorted((_landing_row(r) for r in by_preset[(preset,)]), key=lambda r: r["T_s"])
        chosen = recommended_timeout(landing_rows)
        at_chosen = next((r for r in landing_rows if r["T_s"] == chosen), None)
        entry = {"preset": preset, "h_hz": RULE_H_HZ, "weak_rule": "age", "T_w_s": RULE_T_W_S,
                 "recommended_T_s": chosen,
                 "rule_result": "none up to 2 s" if chosen is None else f"T = {chosen:g} s",
                 "landings_at_recommended_T": at_chosen, "landings_by_T": landing_rows, "notes": []}
        if at_chosen is not None and at_chosen["high_95"] >= RULE_LANDINGS_PER_HOUR:
            entry["notes"].append(f"The rule uses the observed rate; the 95 % upper bound at T = {chosen:g} s is "
                                  f"{at_chosen['high_95']:.2f} per hour, so under 1 per hour is not shown with "
                                  "95 % confidence.")
        for name, timeout in (("late_reaction_at_recommended_T", chosen), ("late_reaction_at_0_30_s", PLANNED_T_S)):
            if part_b_rows is None:
                entry[name] = part_b_status
            elif timeout is None:
                entry[name] = "no T up to 2 s meets the rule"
            elif preset not in part_b_presets:
                entry[name] = "preset not run in Part B (Addendum C grid)"
            else:
                items = late_items(part_b_rows, preset, timeout)
                entry[name] = items if items else f"T = {timeout:g} s not run in Part B for this preset"
        entries.append(entry)
    return entries


def part_a_summary_check(entries: list[dict], summary: dict) -> dict:
    """Compare the recommended T with the Part A runner's own summary at 20 Hz."""
    theirs = {item["preset"]: item.get("smallest_T_s_under_1_per_hour")
              for item in summary.get("recommendation", []) if float(item.get("h_hz", 0.0)) == RULE_H_HZ}
    differing = [e["preset"] for e in entries
                 if e["preset"] not in theirs or theirs[e["preset"]] != e["recommended_T_s"]]
    if differing:
        return {"status": "disagrees", "detail": f"recommended T differs from summary.json for {differing}"}
    return {"status": "agrees", "detail": f"recommended T equals summary.json for all {len(entries)} presets"}


def sweep_summary_check(entries: list[dict], summary: dict) -> dict:
    """Compare with the EXP-031 runner's atlas field (pre-registered point-mass grid only)."""
    atlas = summary.get("atlas") or {}
    key_name = f"wilson_lower_p_t100_within_{SWEEP_HORIZON_S:g}_s"
    compared = 0
    differing = []
    for entry in entries:
        if entry["scope"] != SCOPE_GOAL:
            continue
        theirs = atlas.get(f"{entry['preset']}|{entry['dropout']}|{entry['message_mode']}")
        if theirs is None or key_name not in theirs:
            differing.append(entry["preset"])
            continue
        compared += 1
        ours = {c["id"]: c["compared"] for c in entry["cell"]["conditions"]}
        same_rate = abs(theirs[key_name] - ours["t100_within_120_s_wilson_low"]) <= 1e-12
        their_dup = theirs["median_duplicated_fraction"]
        our_dup = ours["duplicated_fraction_median"]
        same_dup = (their_dup is None and our_dup is None) or (
            their_dup is not None and our_dup is not None and abs(their_dup - our_dup) <= 1e-12)
        if not (same_rate and same_dup):
            differing.append(f"{entry['preset']}|{entry['dropout']}|{entry['message_mode']}")
    if differing:
        return {"status": "disagrees", "detail": f"differs from summary.json atlas for {differing}"}
    return {"status": "agrees", "detail": f"both conditions equal summary.json atlas for {compared} cells"}


# ---- assembling the atlas ---------------------------------------------------------------

MISSION_INFO = {
    "failsafe": {"experiment": "EXP-029", "title": "Lost-link and weak-link rules from data",
                 "inputs": ["exp029_part_a", "exp029_part_b"], "axis": None,
                 "computed_from": ("part_a.csv rows at 20 Hz, age rule, T_w 0.15 s (observed landings per hour "
                                   "and their exact Poisson 95 % interval, as written by the runner); part_b.csv "
                                   "late reaction recomputed per run as violation_during_or_after_outage or "
                                   "nominal_at_age_0_33, pooled over target classes and blackout lengths, "
                                   "Wilson 95 % with rate_summary")},
    "tag": {"experiment": "EXP-030", "title": "Tag under latency and loss", "inputs": ["exp030"],
            "axis": {"name": "measurement_delay_s", "unit": "s", "meaning": "measurement delay"},
            "computed_from": ("tag.csv: tagged (rate_summary, Wilson 95 %) and separation_violations (sum) per "
                              "cell")},
    "sweep": {"experiment": "EXP-031", "title": "Area sweep with lossy coverage sharing", "inputs": ["exp031"],
              "axis": None,
              "computed_from": ("sweep.csv: t100_s present and <= 120 s (rate_summary, Wilson 95 %) and "
                                "duplicated_fraction (median_summary) per cell")},
    "relay": {"experiment": "EXP-032", "title": "Relay drone for range", "inputs": ["exp032"],
              "axis": {"name": "distance_m", "unit": "m", "meaning": "ground station to far drone distance"},
              "computed_from": ("relay.csv: delivery_ratio mean over seeds (math.fsum) and its median interval "
                                "(median_summary) per cell")},
    "monitor": {"experiment": "EXP-033", "title": "Monitoring an unknown drone, alerting humans",
                "inputs": ["exp033"],
                "axis": {"name": "delay_s", "unit": "s", "meaning": "measurement delay"},
                "computed_from": ("exp033.csv: missed_alert and false_alert (rate_summary, Wilson 95 %) and "
                                  "alert_latency_s over runs with an alert (median_summary) per cell")},
    "formation": {"experiment": "EXP-034", "title": "Gym formation demo", "inputs": ["exp034"],
                  "axis": {"name": "fix_latency_s", "unit": "s", "meaning": "camera fix latency"},
                  "computed_from": ("formation.csv: rms_error_m (median_summary interval) and min_separation_m "
                                    "(minimum over seeds) per cell")},
}
ENTRY_BUILDERS = {
    "tag": tag_entries,
    "sweep": sweep_entries,
    "relay": relay_entries,
    "monitor": monitor_entries,
    "formation": formation_entries,
}


def _mission_shell(name: str, status: str, reason: str | None) -> dict:
    info = MISSION_INFO[name]
    return {"experiment": info["experiment"], "title": info["title"], "status": status, "status_reason": reason,
            "inputs": list(info["inputs"]), "goal_definition": GOAL_TEXT[name],
            "computed_from": info["computed_from"], "axis": info["axis"], "entries": [], "notes": []}


def _status_reason(item: dict) -> str | None:
    if item["status"] == PILOT_ONLY:
        return f"{item['experiment']} input is a pilot: " + "; ".join(item["pilot_reasons"])
    if item["status"] == NOT_AVAILABLE:
        return f"{item['experiment']} input {item['folder']}: {item['reason']}"
    return None


def build_mission(name: str, item: dict, pilot_preview: bool) -> dict:
    mission = _mission_shell(name, item["status"], _status_reason(item))
    if item["status"] == RESULT:
        mission["entries"] = ENTRY_BUILDERS[name](item["rows"])
        if name == "sweep":
            mission["runner_summary_check"] = sweep_summary_check(mission["entries"], item["summary"])
    elif item["status"] == PILOT_ONLY and pilot_preview:
        mission["pilot_entries"] = ENTRY_BUILDERS[name](item["rows"])
    return mission


def build_failsafe(part_a: dict, part_b: dict, pilot_preview: bool) -> dict:
    mission = _mission_shell("failsafe", part_a["status"], _status_reason(part_a))
    b_reason = _status_reason(part_b)
    if part_a["status"] == RESULT:
        if part_b["status"] == RESULT:
            mission["entries"] = failsafe_entries(part_a["rows"], part_b["rows"], RESULT)
        else:
            mission["status_reason"] = f"recommendation from Part A; late-reaction rates not available ({b_reason})"
            mission["entries"] = failsafe_entries(part_a["rows"], None, f"{part_b['status']}: {b_reason}")
            if pilot_preview and part_b["status"] == PILOT_ONLY:
                mission["pilot_entries"] = failsafe_entries(part_a["rows"], part_b["rows"], PILOT_ONLY)
        mission["runner_summary_check"] = part_a_summary_check(mission["entries"], part_a["summary"])
    elif part_a["status"] == PILOT_ONLY and pilot_preview:
        rows_b = part_b["rows"] if part_b["status"] != NOT_AVAILABLE else None
        mission["pilot_entries"] = failsafe_entries(part_a["rows"], rows_b, f"{NOT_AVAILABLE}: {b_reason}")
    mission["notes"] = ["Late reaction: separation violation during or within 3 s after the blackout, or the "
                        "follower still NOMINAL when its command age reaches 0.33 s (EXP-029 pre-registration).",
                        "Part B rows exist only for its reduced grid (Addendum C); other presets show the Part A "
                        "recommendation alone."]
    return mission


def preset_descriptors() -> list[dict]:
    out = []
    for name in PRESET_NAMES:
        preset = get_preset(name)
        out.append({"name": name, "loss_model": preset.loss_model,
                    "long_run_loss_fraction": float(preset.loss_probability),
                    "mean_burst_s": None if preset.burst_mean_s is None else float(preset.burst_mean_s),
                    "description": preset.description})
    return out


def build_atlas(items: dict[str, dict], *, pilot_preview: bool = False) -> dict:
    missions = {"failsafe": build_failsafe(items["exp029_part_a"], items["exp029_part_b"], pilot_preview)}
    for name, key in (("tag", "exp030"), ("sweep", "exp031"), ("relay", "exp032"), ("monitor", "exp033"),
                      ("formation", "exp034")):
        missions[name] = build_mission(name, items[key], pilot_preview)
    missions["tag"]["notes"] = [FOLLOWERS_NOTE]
    missions["sweep"]["notes"] = [EXPLORATORY_NOTE]
    missions["relay"]["notes"] = [RELAY_PHYSICS_NOTE, "The relay channel is Run C's outdoor stress corner, not a "
                                  "loss preset, so relay entries have no preset."]
    missions["formation"]["notes"] = ["Minimum separation is over the whole run, including the start from offset "
                                      "positions."]
    return {
        "schema": ATLAS_SCHEMA_ID,
        "simulated": True,
        "title": "Run E missions atlas (simulated)",
        "contains_pilot_numbers": any("pilot_entries" in m for m in missions.values()),
        "goal_source": GOAL_SOURCE,
        "goal_rule": GOAL_RULE_TEXT,
        "loss_axis": LOSS_AXIS_TEXT,
        "presets": preset_descriptors(),
        "limits": list(LIMITS),
        "inputs": {key: input_record(items[key]) for key, *_ in INPUT_SPECS},
        "missions": missions,
    }


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_atlas(atlas: dict) -> None:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(atlas)


def input_folders(inputs: Path, overrides: dict[str, Path | None]) -> dict[str, Path]:
    return {key: Path(overrides.get(key) or Path(inputs) / folder) for key, folder, *_ in INPUT_SPECS}


def run(output: Path, argv: list[str], *, inputs: Path = DEFAULT_INPUTS,
        overrides: dict[str, Path | None] | None = None, pilot_preview: bool = False) -> dict:
    start = time.perf_counter()
    folders = input_folders(inputs, overrides or {})
    items = {key: load_input(key, folders[key], csv_name, experiment)
             for key, _, csv_name, experiment in INPUT_SPECS}
    atlas = build_atlas(items, pilot_preview=pilot_preview)
    validate_atlas(atlas)
    output = prepare_output(output)
    write_json(output / OUTPUT_NAME, atlas)
    for item in items.values():
        reason = _status_reason(item)
        print(f"{item['experiment']}: {item['status']}" + ("" if reason is None else f" ({reason})"))
    config = {"pilot_preview": bool(pilot_preview),
              "inputs": {key: {"folder": items[key]["folder"], "status": items[key]["status"],
                               "files_sha256": items[key]["files"]} for key in items}}
    return write_manifest(
        output, schema="zephyr-missions-atlas-manifest-1", argv=argv, command=replay_command(SCRIPT, argv),
        config=config, seeds=[], outputs=[OUTPUT_NAME], limits=list(LIMITS), runtime_s=time.perf_counter() - start,
        n_processes=1, hashed_sources=[SCRIPT, "sim/missions/missions_atlas_schema.json", "tools/e_common.py",
                                       "sim/missions/presets.py", "sim/s6_recovery.py"])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="new directory for missions_atlas.json")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                        help="directory holding the experiment output folders (default runs/)")
    for key, folder, _, experiment in INPUT_SPECS:
        option = "--" + key.replace("_", "-")
        parser.add_argument(option, type=Path, default=None, dest=key,
                            help=f"{experiment} folder (default <inputs>/{folder})")
    parser.add_argument("--pilot-preview", action="store_true",
                        help="also write pilot numbers under pilot_entries (never as results)")
    args = parser.parse_args(argv)
    overrides = {key: getattr(args, key) for key, *_ in INPUT_SPECS}
    run(args.output, argv, inputs=args.inputs, overrides=overrides, pilot_preview=args.pilot_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
