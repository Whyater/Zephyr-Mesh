"""EXP-029 Part B: set the lost-link and weak-link rules from mission data.

Usage (repo root)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python tools/e_exp029_part_b.py --output runs/e-exp029-part-b --pilot

Everything in this runner is simulated.  The tag mission uses synthetic link
loss, a scripted total blackout, and literature-derived vehicle parameters.

Part B needs the finished Part A table (runs/e-exp029-part-a/part_a.csv, or
--part-a PATH).  Before any mission runs, the table is checked: it must
exist, must not be a pilot output, and must hold one 20 Hz age-rule row for
every preset and every (T_w, T) pair.  Otherwise the runner stops.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from sim.missions.api import Scenario  # noqa: E402
from sim.missions.engine import run_mission  # noqa: E402
from sim.missions.presets import PRESET_NAMES  # noqa: E402
from tools import e_exp029_part_a as part_a  # noqa: E402
from tools.c_common import (  # noqa: E402
    REPO_ROOT,
    argv_or_default,
    prepare_output,
    write_json,
    write_manifest,
)
from tools.e_common import (  # noqa: E402
    MISSION_LIMITS,
    MISSION_SOURCES,
    make_job,
    median_summary,
    processes as process_count,
    rate_summary,
    replay_command,
    run_jobs,
    seeds_for,
    series_colors,
    write_rows,
)
from tools.horizon_mpl import apply_style  # noqa: E402


SCRIPT = "tools/e_exp029_part_b.py"
DT_S = 0.01
COMMAND_RATE_HZ = 20.0
T_W_VALUES_S = (0.10, 0.15, 0.25)
T_VALUES_S = (0.15, 0.20, 0.30, 0.50, 1.0, 2.0)
MIN_T_ABOVE_T_W_S = 0.05
RULE_T_W_S = 0.15
PLANNED_T_S = 0.30
DEGRADED_ACTIONS = ("hold", "slow")
TARGET_CLASSES = ("turn", "figure8", "jink")
OUTAGE_LENGTHS_S = (0.5, 2.0, None)
PILOT_CLASSES = ("turn", "jink")
PILOT_LENGTHS_S = (2.0, None)
PILOT_PRESETS = ("clean", "burst_10_0.5s")
PILOT_T_VALUES_S = (0.30, 1.0)
N_SEEDS = 20
# Grid reduced before the full run (orchestrator, 2026-10-15; prereg Addendum C):
# the pre-registered Part B grid is 121,500 point-mass runs of 30 s, more than a
# day on a machine shared with two other runs.  Kept: the five presets used by
# every other Run E experiment, the full T axis at the rule's T_w = 0.15 s plus
# T_w 0.10 and 0.25 s at the planned T = 0.30 s (7 pairs), both degraded
# actions (7 pairs x 2 actions = 14 policy cells per preset), every class and
# blackout length, and the first 20 seeds of the same list.
RUN_PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
RUN_POLICY_PAIRS = ((0.15, 0.20), (0.15, 0.30), (0.15, 0.50), (0.15, 1.0), (0.15, 2.0),
                    (0.10, 0.30), (0.25, 0.30))
GRID_DEVIATION = ("Grid reduced before the full run for compute (prereg Addendum C): presets "
                  "clean, iid_10, iid_30, burst_10_0.5s, burst_30_0.5s instead of all nine; "
                  "policies T_w 0.15 s with T 0.2, 0.3, 0.5, 1.0, 2.0 s plus T_w 0.10 and 0.25 s at "
                  "T 0.30 s (7 of 15 pairs, each with hold and slow); 20 seeds instead of 50.  Part A "
                  "unnecessary-landing rates still cover every preset and pair.")
NOT_RUN_ADDENDUM_C = "not run in Part B (Addendum C)"
NOT_IN_JOB_LIST = "not in this run's job list (pilot or test subset)"
MISSION_DURATION_S = 30.0
OUTAGE_START_RANGE_S = (8.0, 18.0)
OUTAGE_DRAW_COMPONENT = 29029
PHYSICS_T_VALUES_S = (0.30, 1.0)
PHYSICS_OUTAGE_LENGTHS_S = (2.0, None)
PHYSICS_SEEDS = 20
HAND_CHECK_PRESETS = ("burst_10_0.5s", "iid_30")
Z_LIMIT = part_a.Z_FLAG
MIN_EXPECTED_EVENTS = part_a.MIN_EXPECTED_FOR_Z
DEFAULT_PART_A_CSV = REPO_ROOT / "runs" / "e-exp029-part-a" / "part_a.csv"
OUTPUTS = ("part_b.csv", "summary.json", "tradeoff.png")
REQUIRED_METRICS = (
    "landings_before_outage", "first_landing_s", "airborne_at_blackout",
    "violation_during_or_after_outage", "nominal_at_age_0_33",
    "outage_reaction_hold_s", "outage_reaction_land_s",
    "outage_hold_link_age_s", "outage_land_link_age_s", "blind_in_plane_s",
)

# Pre-registered text, copied word for word from the EXP-029 section of the
# private pre-registration with only the leading label removed.
PREDICTION_SOURCE = "pre-registered before the runs (EXP-029 section of the private pre-registration)"
PREDICTION_TEXT = (
    "at 20 Hz the 300 ms rule (m = 6) is calm on clean, iid_1, iid_5 (iid_5: 3600*20*0.95*0.05^6 = "
    "0.001 per hour) and iid_10 (0.068 per hour), noisy on iid_30 (about 36 per hour) and useless on "
    "every 0.5 s burst preset (burst_10_0.5s about 400 per hour; mean bad sojourn 0.5 s is longer than "
    "0.3 s so most bursts trip it). No T up to 2 s gets burst_10_0.5s under 1 per hour (bursts start "
    "about 720 times per hour: mean cycle 0.5 s bad + 4.5 s good; the bad sojourn is about exponential "
    "with mean 0.5 s, so P(sojourn > 2 s) = e^-4 = 0.018, about 13 per hour at T = 2 s; at T = 0.3 s, "
    "e^-0.6 = 0.55, about 400 per hour). Late reactions: separation violations during a blind hold will "
    "happen on the jinking and figure-eight targets regardless of T, because a held follower cannot get "
    "out of the way; their rate rises with T (longer blind time in the target's plane). Hold vs slow: "
    "slow keeps closing on a stale point, so it will raise violations near a moving target. Expected "
    "recommendation: T = 0.3 s on clean and iid presets up to 10 %, T = 0.5 s at iid_30, and \"no single "
    "timeout works\" for 0.5 s bursts, where the fix must be elsewhere (better link, or a reversible hold "
    "instead of a landing)."
)
CORRECTION_1_TEXT = (
    "EXP-029 prediction: by the pre-registered rule (smallest T with under 1 unnecessary landing per "
    "hour, at 20 Hz), the formula gives T = 0.15 s on iid_1 and 0.2 s on iid_5, not 0.3 s; iid_10 gives "
    "0.3 s (0.065 per hour at 0.3 s). Corrected prediction: 0.15 s (iid_1), 0.2 s (iid_5), 0.3 s "
    "(iid_10), 0.5 s (iid_30), none up to 2 s on the 0.5 s burst presets and on burst_10_0.2s probably "
    "1.0 to 2.0 s. The rule is also constrained to T >= T_w + 0.05 s. The late-reaction rate is reported "
    "beside it."
)
# Predicted T per preset from correction 1.  "kind": single (one T), range
# (inclusive bounds), none (no T up to 2 s), formula (correction 1 gives no
# value for clean, so the closed form decides).
PREDICTED_T = {
    "clean": {"kind": "formula", "text": "as the closed form gives (no value written in correction 1)"},
    "iid_1": {"kind": "single", "values": (0.15,), "text": "0.15 s"},
    "iid_5": {"kind": "single", "values": (0.2,), "text": "0.2 s"},
    "iid_10": {"kind": "single", "values": (0.3,), "text": "0.3 s"},
    "iid_30": {"kind": "single", "values": (0.5,), "text": "0.5 s"},
    "burst_1_0.5s": {"kind": "none", "text": "none up to 2 s"},
    "burst_10_0.5s": {"kind": "none", "text": "none up to 2 s"},
    "burst_30_0.5s": {"kind": "none", "text": "none up to 2 s"},
    "burst_10_0.2s": {"kind": "range", "values": (1.0, 2.0), "text": "probably 1.0 to 2.0 s"},
}


# ---- grid ----------------------------------------------------------------------

def policies(*, actions: tuple[str, ...] = DEGRADED_ACTIONS,
             t_values: tuple[float, ...] = T_VALUES_S) -> list[dict]:
    """Return the Part A age-rule (T_w, T) pairs, each with every degraded action.

    The Part A constraint T >= T_w + 0.05 s leaves 6 + 5 + 4 = 15 pairs, so the
    pre-registered Part B grid has 15 x 2 = 30 policy/action cells.  The run
    (Addendum C) keeps 7 of the pairs (RUN_POLICY_PAIRS) with both actions,
    7 x 2 = 14 cells per preset; build_point_jobs does that selection.
    """
    age_policies = [p for p in part_a.policies() if p["weak_rule"] == "age"]
    allowed_t = {float(v) for v in t_values}
    out = []
    for policy in age_policies:
        if float(policy["T_s"]) not in allowed_t:
            continue
        for action in actions:
            item = {
                "weak_rule": "age",
                "T_w_s": float(policy["T_w_s"]),
                "T_s": float(policy["T_s"]),
                "degraded_action": str(action),
            }
            out.append(item)
    return out


def draw_outage_start(seed: int) -> float:
    """Draw one blackout start from [8, 18] using a SeedSequence per run."""
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), OUTAGE_DRAW_COMPONENT]))
    return float(generator.uniform(*OUTAGE_START_RANGE_S))


def outage_item(seed: int, length_s: float | None) -> dict:
    start = draw_outage_start(seed)
    end = None if length_s is None else start + float(length_s)
    return {"start_s": start, "end_s": end, "streams": None}


def _labels(*, target_class: str, length_s: float | None, preset: str,
            policy: dict, seed: int, adapter: str) -> dict:
    return {
        "target_class": target_class,
        "outage_length_s": "permanent" if length_s is None else float(length_s),
        "preset": preset,
        "weak_rule": policy["weak_rule"],
        "T_w_s": float(policy["T_w_s"]),
        "T_s": float(policy["T_s"]),
        "degraded_action": policy["degraded_action"],
        "seed": int(seed),
        "adapter": adapter,
    }


def scenario_for(*, target_class: str, length_s: float | None, preset: str,
                 policy: dict, seed: int, adapter: str = "point_mass",
                 duration_s: float = MISSION_DURATION_S) -> dict:
    """Build one validated Scenario dictionary for a Part B run."""
    return {
        "mission": "tag",
        "seed": int(seed),
        "duration_s": float(duration_s),
        "adapter": adapter,
        "preset": preset,
        "policy": {
            "degraded_after_s": float(policy["T_w_s"]),
            "land_after_s": float(policy["T_s"]),
            "weak_rule": "age",
            "degraded_action": policy["degraded_action"],
            "loss_window_packets": 20,
            "weak_loss_fraction": 0.3,
        },
        "measurement_delay_s": 0.0,
        "measurement_jitter_s": 0.0,
        "command_delay_s": 0.002,
        "outages": (outage_item(seed, length_s),),
        "params": {
            "target_class": target_class,
            "mode": "hold",
            "target_speed_mps": 0.5,
        },
    }


def _seed_values(values: list[int] | tuple[int, ...] | None, count: int) -> list[int]:
    if values is None:
        return seeds_for(29, count)
    return [int(value) for value in values]


def build_point_jobs(*, pilot: bool = False, seeds: list[int] | None = None,
                     duration_s: float = MISSION_DURATION_S) -> list[dict]:
    """Build the point-mass jobs in deterministic grid order.

    Order: class, blackout length, preset, policy pair and action, seed.  The
    full run is 3 x 3 x 5 x (7 x 2) x 20 = 12,600 jobs.
    """
    seed_list = _seed_values(seeds, 3 if pilot else N_SEEDS)
    classes = PILOT_CLASSES if pilot else TARGET_CLASSES
    lengths = PILOT_LENGTHS_S if pilot else OUTAGE_LENGTHS_S
    presets = PILOT_PRESETS if pilot else RUN_PRESETS
    t_values = PILOT_T_VALUES_S if pilot else T_VALUES_S
    policy_list = policies(t_values=t_values)
    if pilot:
        policy_list = [policy for policy in policy_list if policy["T_w_s"] == RULE_T_W_S]
    else:
        policy_list = [policy for policy in policy_list
                       if (policy["T_w_s"], policy["T_s"]) in RUN_POLICY_PAIRS]
    jobs = []
    for target_class in classes:
        for length_s in lengths:
            for preset in presets:
                for policy in policy_list:
                    for seed in seed_list:
                        scenario = scenario_for(
                            target_class=target_class, length_s=length_s,
                            preset=preset, policy=policy, seed=seed,
                            adapter="point_mass", duration_s=duration_s,
                        )
                        jobs.append(make_job(
                            _labels(target_class=target_class, length_s=length_s,
                                    preset=preset, policy=policy, seed=seed,
                                    adapter="point_mass"), scenario,
                        ))
    return jobs


def build_physics_jobs(*, pilot: bool = False, seeds: list[int] | None = None,
                       duration_s: float = MISSION_DURATION_S) -> list[dict]:
    """Build the pre-registered physics subset, and its two-run pilot cell."""
    if pilot:
        seed_list = _seed_values(seeds, 2)
        cells = (("turn", 2.0, "clean", 0.30, "hold"),)
    else:
        seed_list = _seed_values(seeds, PHYSICS_SEEDS)
        cells = tuple(
            (target_class, length_s, "clean", timeout, "hold")
            for target_class in TARGET_CLASSES
            for length_s in PHYSICS_OUTAGE_LENGTHS_S
            for timeout in PHYSICS_T_VALUES_S
        )
    jobs = []
    for target_class, length_s, preset, timeout, action in cells:
        policy = {"weak_rule": "age", "T_w_s": RULE_T_W_S, "T_s": timeout,
                  "degraded_action": action}
        for seed in seed_list:
            scenario = scenario_for(
                target_class=target_class, length_s=length_s, preset=preset,
                policy=policy, seed=seed, adapter="physics", duration_s=duration_s,
            )
            jobs.append(make_job(
                _labels(target_class=target_class, length_s=length_s,
                        preset=preset, policy=policy, seed=seed,
                        adapter="physics"), scenario,
            ))
    return jobs


def build_jobs(*, pilot: bool = False, seeds: list[int] | None = None,
               duration_s: float = MISSION_DURATION_S) -> list[dict]:
    return build_point_jobs(pilot=pilot, seeds=seeds, duration_s=duration_s) + build_physics_jobs(
        pilot=pilot, seeds=seeds, duration_s=duration_s,
    )


# ---- one run ---------------------------------------------------------------------

def _run_job(job: dict) -> dict:
    scenario = Scenario.from_dict(job["scenario"])
    result = run_mission(scenario, log=False)
    row = dict(job["labels"])
    row["blackout_start_s"] = float(scenario.outages[0]["start_s"])
    for key, value in sorted(result.metrics.items()):
        if key in row:
            raise ValueError(f"metric {key!r} collides with a cell label")
        row[key] = value
    row["late_reaction"] = bool(
        row.get("violation_during_or_after_outage", False)
        or row.get("nominal_at_age_0_33", False)
    )
    return row


def _timed_job(job: dict) -> tuple[dict, float]:
    """Run one job and return its row and the CPU seconds it took (printed only)."""
    start = time.process_time()
    row = _run_job(job)
    return row, time.process_time() - start


# ---- Part A dependency ------------------------------------------------------------

def _display_path(path: Path) -> str:
    """Path relative to the repo when inside it, else only the file name (no absolute paths)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _pilot_flags(folder: Path) -> list[str]:
    """Reasons the Part A folder looks like a pilot, from its summary.json and manifest.json."""
    reasons = []
    summary_path = folder / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("pilot") is True:
            reasons.append("summary.json has \"pilot\": true")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("config") or {}).get("pilot") is True:
            reasons.append("manifest.json config has \"pilot\": true")
    return reasons


def load_part_a(path: Path | None = None) -> dict:
    """Read and check the Part A table before any Part B job runs.

    Refuses (SystemExit with the reason) when the file is missing, when its
    folder says it is a pilot, when a row's simulated hours differ from the
    full Part A run, or when a 20 Hz age-rule row is missing or repeated for
    any preset and (T_w, T) pair.  Returns the rates and the file's sha256.
    """
    source = DEFAULT_PART_A_CSV if path is None else Path(path)
    if not source.exists():
        raise SystemExit(f"Part B needs the Part A table, but {_display_path(source)} does not exist; "
                         "run tools/e_exp029_part_a.py first")
    pilot_reasons = _pilot_flags(source.parent)
    if pilot_reasons:
        raise SystemExit(f"Part A table {_display_path(source)} is a pilot output ("
                         + "; ".join(pilot_reasons) + "); Part B needs the full Part A run")
    rates: dict[tuple[str, float, float], float] = {}
    repeated = []
    short_hours = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["weak_rule"] != "age" or float(row["h_hz"]) != COMMAND_RATE_HZ:
                continue
            if float(row["hours"]) != part_a.HOURS:
                short_hours.add(float(row["hours"]))
            key = (row["preset"], float(row["T_w_s"]), float(row["T_s"]))
            if key in rates:
                repeated.append(key)
            rates[key] = float(row["landings_per_hour"])
    if short_hours:
        raise SystemExit(f"Part A table {_display_path(source)} has rows of {sorted(short_hours)} simulated "
                         f"hours, not the full run's {part_a.HOURS:g} h (a pilot output?); Part B refuses it")
    if repeated:
        raise SystemExit(f"Part A table {_display_path(source)} repeats rows for {sorted(set(repeated))}")
    needed = [(preset, float(p["T_w_s"]), float(p["T_s"]))
              for preset in PRESET_NAMES for p in part_a.policies() if p["weak_rule"] == "age"]
    missing = [key for key in needed if key not in rates]
    if missing:
        raise SystemExit(f"Part A table {_display_path(source)} lacks {len(missing)} of {len(needed)} "
                         f"20 Hz age-rule rows (first missing: {missing[:3]}); Part B refuses to run")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"rates": rates, "sha256": digest, "path": _display_path(source), "rows_used": len(needed)}


# ---- mission-observed unnecessary landings ------------------------------------------

def pre_blackout_exposure_s(row: dict) -> float:
    """Time at risk of an unnecessary landing before the blackout, for one run.

    A landing latches, so a follower that landed before the blackout is at
    risk only until its first landing: min(first landing, blackout start).
    Otherwise the whole time before the blackout counts.
    """
    start = float(row["blackout_start_s"])
    first = row.get("first_landing_s")
    if not row["airborne_at_blackout"] and first is not None:
        return min(float(first), start)
    return start


def _pre_blackout_signature(row: dict) -> tuple:
    first = row.get("first_landing_s")
    landed_before = (not row["airborne_at_blackout"]) and first is not None
    return (int(row["landings_before_outage"]), float(first) if landed_before else None)


def representative_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """One run per seed and cell, and how many seeds disagree with their representative.

    For one seed the link history before the blackout is the same for every
    target class and blackout length (same seed, same loss process, same
    command schedule), so the 3 x 3 runs of a seed are one observation.
    Rule: within each (preset, T_w, T, degraded action, adapter, seed) the
    first row in grid order is kept (class turn, then figure8, then jink;
    blackout 0.5 s, then 2.0 s, then permanent).  The second value counts
    seeds whose other runs show a different pre-blackout landing history.
    """
    kept: dict[tuple, dict] = {}
    inconsistent = set()
    for row in rows:
        key = (row["preset"], row["T_w_s"], row["T_s"], row["degraded_action"], row["adapter"], row["seed"])
        if key not in kept:
            kept[key] = row
        elif _pre_blackout_signature(row) != _pre_blackout_signature(kept[key]):
            inconsistent.add(key)
    return list(kept.values()), len(inconsistent)


def mission_landing_check(rows: list[dict], *, preset: str, t_w: float, timeout: float,
                          action: str, adapter: str = "point_mass") -> dict | None:
    """Mission-observed unnecessary landings before the blackout vs the closed form.

    Expected count = closed-form rate (Part A's iid formula or 0.01 s burst
    chain viewed every 1/h s, at 20 Hz) x exposure hours.  z = (observed -
    expected) / sqrt(expected).  Verdict "agrees" when |z| <= 4 on at least
    10 expected events, else "disagrees"; below 10 expected events "too few
    expected events".  Returns None when the cell has no rows.
    """
    cell = [row for row in rows if row["preset"] == preset and row["adapter"] == adapter
            and float(row["T_w_s"]) == t_w and float(row["T_s"]) == timeout
            and row["degraded_action"] == action]
    if not cell:
        return None
    reps, inconsistent = representative_rows(cell)
    count = int(sum(int(row["landings_before_outage"]) for row in reps))
    exposure_h = sum(pre_blackout_exposure_s(row) for row in reps) / 3600.0
    expected_rate = part_a.closed_form_per_hour(preset, COMMAND_RATE_HZ, timeout)
    expected = expected_rate * exposure_h
    z = (count - expected) / math.sqrt(expected) if expected > 0.0 else None
    if expected >= MIN_EXPECTED_EVENTS:
        verdict = "agrees" if abs(z) <= Z_LIMIT else "disagrees"
    elif expected == 0.0 and count > 0:
        verdict = "landings observed where the closed form expects none"
    else:
        verdict = "too few expected events"
    return {
        "preset": preset,
        "T_w_s": t_w,
        "T_s": timeout,
        "degraded_action": action,
        "adapter": adapter,
        "runs_in_cell": len(cell),
        "seeds_counted": len(reps),
        "seeds_with_inconsistent_history": inconsistent,
        "grounded_before_blackout": int(sum(1 for row in reps if not row["airborne_at_blackout"])),
        "count": count,
        "exposure_h": exposure_h,
        "observed_rate_per_hour": count / exposure_h if exposure_h > 0.0 else None,
        "expected_rate_per_hour": expected_rate,
        "expected_count": expected,
        "z": z,
        "verdict": verdict,
    }


# ---- summaries ----------------------------------------------------------------------

def _group_key(row: dict) -> tuple:
    return (row["preset"], row["T_w_s"], row["T_s"], row["degraded_action"], row["adapter"])


def _median(rows: list[dict], key: str) -> dict:
    return median_summary([row.get(key) for row in rows])


def _rate_or_empty(flags: list[bool]) -> dict:
    if not flags:
        return {"n": 0, "count": 0, "rate": None, "wilson_low": None, "wilson_high": None}
    return rate_summary(flags)


def _metric_summary(rows: list[dict], part_a_rates: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    summaries = []
    for key, group in groups.items():
        preset, t_w, timeout, action, adapter = key
        airborne = [row for row in group if row["airborne_at_blackout"]]
        observed = mission_landing_check(group, preset=preset, t_w=t_w, timeout=timeout,
                                         action=action, adapter=adapter)
        summaries.append({
            "preset": preset,
            "T_w_s": t_w,
            "T_s": timeout,
            "degraded_action": action,
            "adapter": adapter,
            "n": len(group),
            "late": rate_summary([bool(row["late_reaction"]) for row in group]),
            "late_airborne_at_blackout": _rate_or_empty([bool(row["late_reaction"]) for row in airborne]),
            "grounded_before_blackout": len(group) - len(airborne),
            "separation_violation": rate_summary([bool(row["violation_during_or_after_outage"]) for row in group]),
            "nominal_at_age_0_33": rate_summary([bool(row["nominal_at_age_0_33"]) for row in group]),
            "outage_reaction_hold_s": _median(group, "outage_reaction_hold_s"),
            "outage_reaction_land_s": _median(group, "outage_reaction_land_s"),
            "outage_hold_link_age_s": _median(group, "outage_hold_link_age_s"),
            "outage_land_link_age_s": _median(group, "outage_land_link_age_s"),
            "blind_in_plane_s": _median(group, "blind_in_plane_s"),
            "landings": int(sum(int(row["landings"]) for row in group)),
            "holds": int(sum(int(row["holds"]) for row in group)),
            "separation_violations": int(sum(int(row["separation_violations"]) for row in group)),
            "part_a_unnecessary_landings_per_hour": part_a_rates.get((preset, float(t_w), float(timeout))),
            "mission_observed_unnecessary_landings": {
                "count": observed["count"],
                "seeds_counted": observed["seeds_counted"],
                "exposure_h": observed["exposure_h"],
                "rate_per_hour": observed["observed_rate_per_hour"],
                "seeds_with_inconsistent_history": observed["seeds_with_inconsistent_history"],
            },
        })
    return sorted(summaries, key=lambda item: (
        item["adapter"], PRESET_NAMES.index(item["preset"]), item["T_w_s"],
        item["T_s"], item["degraded_action"],
    ))


def smallest_allowed_t(rates_by_t: dict[float, float], t_w: float = RULE_T_W_S) -> float | None:
    """The pre-registered rule: smallest T under 1 per hour with T >= T_w + 0.05 s."""
    valid = [t for t in T_VALUES_S if t >= t_w + MIN_T_ABOVE_T_W_S - 1e-9
             and t in rates_by_t and rates_by_t[t] < 1.0]
    return min(valid) if valid else None


def _late_at(summaries: list[dict], preset: str, timeout: float) -> dict:
    out = {}
    for action in DEGRADED_ACTIONS:
        cell = next((s for s in summaries if s["adapter"] == "point_mass" and s["preset"] == preset
                     and s["T_w_s"] == RULE_T_W_S and s["T_s"] == timeout
                     and s["degraded_action"] == action), None)
        if cell is None:
            out[action] = NOT_IN_JOB_LIST
        else:
            out[action] = {
                "late_all_runs": cell["late"],
                "late_airborne_at_blackout": cell["late_airborne_at_blackout"],
                "grounded_before_blackout": cell["grounded_before_blackout"],
            }
    return out


def _recommendations(summaries: list[dict], part_a_rates: dict) -> list[dict]:
    run_presets = {s["preset"] for s in summaries if s["adapter"] == "point_mass"}
    out = []
    for preset in PRESET_NAMES:
        rates_by_t = {t: part_a_rates[(preset, RULE_T_W_S, t)] for t in T_VALUES_S
                      if (preset, RULE_T_W_S, t) in part_a_rates}
        chosen = smallest_allowed_t(rates_by_t)
        item = {
            "preset": preset,
            "T_w_s": RULE_T_W_S,
            "smallest_T_s_under_1_per_hour": chosen,
            "rule_result": "none up to 2 s" if chosen is None else f"T = {chosen:g} s",
            "part_a_landings_per_hour_at_T": None if chosen is None else rates_by_t[chosen],
            "part_a_landings_per_hour_at_planned_0.30": rates_by_t.get(PLANNED_T_S),
        }
        if preset not in RUN_PRESETS:
            item["at_recommended_T"] = NOT_RUN_ADDENDUM_C
            item["at_planned_0.30"] = NOT_RUN_ADDENDUM_C
        elif preset not in run_presets:
            item["at_recommended_T"] = NOT_IN_JOB_LIST
            item["at_planned_0.30"] = NOT_IN_JOB_LIST
        else:
            item["at_recommended_T"] = ("no T up to 2 s meets the rule" if chosen is None
                                        else _late_at(summaries, preset, chosen))
            item["at_planned_0.30"] = _late_at(summaries, preset, PLANNED_T_S)
        out.append(item)
    return out


def _prediction_comparison(recommendations: list[dict]) -> list[dict]:
    table = []
    for item in recommendations:
        preset = item["preset"]
        predicted = PREDICTED_T[preset]
        computed = item["smallest_T_s_under_1_per_hour"]
        closed_form = smallest_allowed_t({t: part_a.closed_form_per_hour(preset, COMMAND_RATE_HZ, t)
                                          for t in T_VALUES_S})
        if predicted["kind"] == "formula":
            predicted_values = [] if closed_form is None else [closed_form]
            agrees = computed == closed_form
        elif predicted["kind"] == "none":
            predicted_values = []
            agrees = computed is None
        elif predicted["kind"] == "single":
            predicted_values = list(predicted["values"])
            agrees = computed is not None and computed in predicted["values"]
        else:
            low, high = predicted["values"]
            predicted_values = [low, high]
            agrees = computed is not None and low <= computed <= high
        notes = []
        floor_t = RULE_T_W_S + MIN_T_ABOVE_T_W_S
        if predicted["kind"] == "single" and min(predicted["values"]) < floor_t - 1e-9:
            notes.append(f"predicted T is below T_w + 0.05 s = {floor_t:g} s, which the constrained rule cannot return")
        table.append({
            "preset": preset,
            "predicted_T": predicted["text"],
            "predicted_T_s": predicted_values,
            "computed_T_s": computed,
            "computed_rule_result": item["rule_result"],
            "closed_form_T_s": closed_form,
            "agrees": bool(agrees),
            "notes": notes,
        })
    return table


def _hand_checks(rows: list[dict]) -> tuple[dict, list[str]]:
    """Mission-observed unnecessary landings vs the closed form (prereg hand check).

    Uses the planned 20 Hz age policy (T_w 0.15 s, T 0.30 s, hold) on the two
    presets where these landings are common.  A preset this run did not
    simulate is listed under missing_metrics.
    """
    checks = {}
    missing = []
    for preset in HAND_CHECK_PRESETS:
        check = mission_landing_check(rows, preset=preset, t_w=RULE_T_W_S, timeout=PLANNED_T_S,
                                      action="hold")
        if check is None:
            checks[preset] = {"status": "missing_metrics", "reason": NOT_IN_JOB_LIST,
                              "T_w_s": RULE_T_W_S, "T_s": PLANNED_T_S, "degraded_action": "hold"}
            missing.append(f"hand_check mission-observed unnecessary landings on {preset} "
                           f"(T_w 0.15 s, T 0.30 s, hold): {NOT_IN_JOB_LIST}")
        else:
            checks[preset] = {"status": "computed", **check}
    return checks, missing


def _missing_metrics(rows: list[dict]) -> list[str]:
    if not rows:
        return ["no rows"]
    return [f"mission metric {key} absent from some rows" for key in REQUIRED_METRICS
            if any(key not in row for row in rows)]


# ---- figure -------------------------------------------------------------------------

def _make_figure(summaries: list[dict], output: Path) -> None:
    pal = apply_style("light")
    point = [s for s in summaries if s["adapter"] == "point_mass"]
    presets = [p for p in PRESET_NAMES if any(s["preset"] == p for s in point)]
    colors = dict(zip(presets, series_colors(len(presets))))
    rates = [float(s["part_a_unnecessary_landings_per_hour"]) for s in point
             if s["part_a_unnecessary_landings_per_hour"] is not None]
    nonzero = [rate for rate in rates if rate > 0.0]
    smallest = min(nonzero) if nonzero else 1.0
    floor = 10.0 ** (math.floor(math.log10(min(smallest, 1.0))) - 1)

    def x_of(cell: dict) -> float:
        rate = cell["part_a_unnecessary_landings_per_hour"]
        return floor if rate is None or float(rate) <= 0.0 else float(rate)

    markers = {"hold": "o", "slow": "^"}
    fig, axes = plt.subplots(1, len(DEGRADED_ACTIONS), figsize=(11.0, 4.8), sharey=True, squeeze=False)
    for ax, action in zip(axes[0], DEGRADED_ACTIONS):
        for preset in presets:
            cells = [s for s in point if s["preset"] == preset and s["degraded_action"] == action]
            curve = sorted((s for s in cells if s["T_w_s"] == RULE_T_W_S), key=lambda s: s["T_s"])
            if curve:
                ax.plot([x_of(s) for s in curve], [s["late"]["rate"] for s in curve],
                        marker=markers[action], linestyle="-", linewidth=1.2, markersize=5,
                        color=colors[preset])
                for cell in curve:
                    ax.annotate(f"{cell['T_s']:g}", (x_of(cell), cell["late"]["rate"]), fontsize=6,
                                xytext=(3, 3), textcoords="offset points", color=pal["textSecondary"])
            for cell in cells:
                if cell["T_w_s"] == RULE_T_W_S:
                    continue
                ax.plot([x_of(cell)], [cell["late"]["rate"]], marker=markers[action], linestyle="none",
                        markersize=7, markerfacecolor="none", markeredgecolor=colors[preset])
                offset = (-30, -9) if cell["T_w_s"] < RULE_T_W_S else (5, -9)
                ax.annotate(f"T_w {cell['T_w_s']:g}", (x_of(cell), cell["late"]["rate"]), fontsize=6,
                            xytext=offset, textcoords="offset points", color=pal["textSecondary"])
        ax.axvline(1.0, color=pal["lineStrong"], linewidth=0.9, linestyle="--")
        ax.text(1.05, 1.0, "1 per hour (rule)", fontsize=7, color=pal["textSecondary"], va="top")
        ax.axvline(floor, color=pal["lineStrong"], linewidth=0.8, linestyle=":")
        ax.text(floor * 1.15, 0.5, "0 per hour, drawn at the floor", fontsize=7, rotation=90,
                color=pal["textSecondary"], va="center")
        ax.set_xscale("log")
        ax.set_xlim(floor / 2.0, max(max(rates, default=1.0), 1.0) * 3.0)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("unnecessary landings per hour, Part A at 20 Hz (log scale)")
        ax.set_title(f"degraded action: {action}", fontsize=9)
        ax.grid(True, which="major", alpha=0.2)
    axes[0][0].set_ylabel("late-reaction rate (all runs)")
    handles = [Line2D([0], [0], color=colors[p], linewidth=1.2, label=p) for p in presets]
    handles += [
        Line2D([0], [0], color=pal["textSecondary"], marker="o", linestyle="none", label="hold"),
        Line2D([0], [0], color=pal["textSecondary"], marker="^", linestyle="none", label="slow"),
        Line2D([0], [0], color=pal["textSecondary"], marker="o", markerfacecolor="none", linestyle="none",
               label="T_w 0.10 or 0.25 s at T 0.3 s"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Simulated: lost-link timeout tradeoff, EXP-029 Part B (point mass)\n"
                 "curves join T_w 0.15 s cells, labels give T in s", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "tradeoff.png", dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---- run ------------------------------------------------------------------------------

def run(output: Path, argv: list[str], *, pilot: bool = False,
        processes: int | None = None, jobs: list[dict] | None = None,
        duration_s: float = MISSION_DURATION_S, part_a_csv: Path | None = None) -> dict:
    start = time.perf_counter()
    part_a_table = load_part_a(part_a_csv)
    output = prepare_output(output)
    if jobs is None:
        jobs = build_jobs(pilot=pilot, duration_s=duration_s)
    proc = process_count(processes)
    timed = run_jobs(jobs, n_processes=proc, worker=_timed_job, chunksize=1)
    rows = [row for row, _ in timed]
    part_a_rates = part_a_table["rates"]
    summaries = _metric_summary(rows, part_a_rates)
    hand_check, hand_check_missing = _hand_checks(rows)
    recommendation = _recommendations(summaries, part_a_rates)
    full_point_jobs = len(build_point_jobs())
    full_physics_jobs = len(build_physics_jobs())
    summary = {
        "experiment": "EXP-029 Part B",
        "status": "simulated",
        "pilot": bool(pilot),
        "mission_duration_s": float(duration_s),
        "outage_draw": "np.random.default_rng(np.random.SeedSequence([seed, 29029])).uniform(8, 18)",
        "outage_streams": None,
        "presets": sorted({row["preset"] for row in rows}, key=PRESET_NAMES.index),
        "preset_scope_reading": "Part B does not restrict presets; the run uses five of nine for compute (see deviations).",
        "grid": {
            "point_mass_jobs": full_point_jobs,
            "physics_jobs": full_physics_jobs,
            "pre_registered_point_mass_jobs": 121500,
            "pilot_point_mass_jobs": len(build_point_jobs(pilot=True)),
            "pilot_physics_jobs": len(build_physics_jobs(pilot=True)),
            "jobs_in_this_run": len(jobs),
        },
        "part_a": {"csv": part_a_table["path"], "sha256": part_a_table["sha256"],
                   "rows_used": part_a_table["rows_used"], "h_hz": COMMAND_RATE_HZ, "weak_rule": "age"},
        "rules": {
            "late_reaction": "separation violation (true distance < 0.5 m) during the blackout or within 3 s "
                             "after it, or the follower still NOMINAL when its command age reaches 0.33 s "
                             "during the blackout",
            "late_airborne_at_blackout": "the same rate over runs whose follower had not landed before the "
                                         "blackout (a follower grounded by link loss cannot react late)",
            "outage_reaction": "outage_reaction_*: time from the newest command received before the blackout "
                               "to the transition; outage_*_link_age_s: the drone's own command age at it",
            "mission_observed_landings": "count landings before the blackout; exposure per run = min(first "
                                         "landing, blackout start) when the follower landed before the "
                                         "blackout, else the blackout start; one run per seed and cell, the "
                                         "first in grid order (class turn, figure8, jink; blackout 0.5 s, "
                                         "2.0 s, permanent), because the pre-blackout link history does not "
                                         "depend on the class or blackout length",
            "hand_check_verdict": f"z = (observed - expected) / sqrt(expected); agrees if |z| <= {Z_LIMIT:g} "
                                  f"when at least {MIN_EXPECTED_EVENTS:g} events are expected",
            "recommendation": "smallest T with under 1 unnecessary landing per hour in Part A at 20 Hz, age "
                              "rule, T_w 0.15 s, T >= T_w + 0.05 s",
        },
        "summaries": summaries,
        "recommendation": recommendation,
        "prediction": {
            "prediction_source": PREDICTION_SOURCE,
            "prediction_text": PREDICTION_TEXT,
            "correction_1_text": CORRECTION_1_TEXT,
            "comparison": _prediction_comparison(recommendation),
        },
        "hand_check": hand_check,
        "missing_metrics": _missing_metrics(rows) + hand_check_missing,
        "deviations": [GRID_DEVIATION],
        "projection": {
            "full_grid_jobs": full_point_jobs + full_physics_jobs,
            "method": "per-run CPU seconds of this run (printed, kept out of the data files) times the full "
                      "grid's point-mass and physics job counts, divided by 2 processes",
        },
    }
    runtime = time.perf_counter() - start
    write_rows(output / "part_b.csv", rows, first_columns=(
        "target_class", "outage_length_s", "preset", "weak_rule", "T_w_s", "T_s",
        "degraded_action", "seed", "adapter",
    ))
    _make_figure(summaries, output)
    write_json(output / "summary.json", summary)
    print(f"runtime {runtime:.3f} s for {len(jobs)} runs on {proc} processes")
    cpu_by_adapter: dict[str, list[float]] = {}
    for row, cpu_s in timed:
        cpu_by_adapter.setdefault(row["adapter"], []).append(cpu_s)
    full_counts = {"point_mass": full_point_jobs, "physics": full_physics_jobs}
    if all(cpu_by_adapter.get(kind) for kind in full_counts):
        projected = 0.0
        for kind, count in full_counts.items():
            per_run = sum(cpu_by_adapter[kind]) / len(cpu_by_adapter[kind])
            projected += per_run * count / 2.0
            print(f"cpu per {kind} run {per_run:.3f} s over {len(cpu_by_adapter[kind])} runs")
        print(f"projection {projected:.0f} s ({projected / 3600.0:.2f} h) for "
              f"{full_point_jobs + full_physics_jobs} full-grid runs at 2 processes")
    config = {
        "pilot": bool(pilot),
        "duration_s": float(duration_s),
        "target_classes": list(PILOT_CLASSES if pilot else TARGET_CLASSES),
        "outage_lengths_s": ["permanent" if v is None else v for v in (PILOT_LENGTHS_S if pilot else OUTAGE_LENGTHS_S)],
        "presets": list(PRESET_NAMES),
        "run_presets": list(PILOT_PRESETS if pilot else RUN_PRESETS),
        "run_policy_pairs": [list(pair) for pair in RUN_POLICY_PAIRS],
        "T_w_s": list(T_W_VALUES_S),
        "T_s": list(T_VALUES_S),
        "degraded_actions": list(DEGRADED_ACTIONS),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "total_blackout_streams": None,
        "outage_draw_component": OUTAGE_DRAW_COMPONENT,
        "part_a_csv": part_a_table["path"],
        "part_a_csv_sha256": part_a_table["sha256"],
    }
    return write_manifest(
        output, schema="zephyr-run-e-manifest-1", argv=argv,
        command=replay_command(SCRIPT, argv),
        config=config, seeds=sorted({int(row["seed"]) for row in rows}), outputs=list(OUTPUTS),
        limits=list(MISSION_LIMITS) + [
            "Part B uses the tag mission in hold mode with a 0.75 m follower standoff.",
            "A total blackout is expressed with Scenario.outages streams=None, which applies to every mission stream.",
            "Mission duration is 30 s, covering the latest possible blackout, three seconds, and landing time.",
            "Part A unnecessary-landing rates are read from the Part A table (sha256 in config) and never regenerated here.",
        ], runtime_s=runtime, n_processes=proc,
        hashed_sources=list(MISSION_SOURCES) + ["sim/missions/tag.py", "tools/e_exp029_part_a.py",
                                                SCRIPT, "tests/reference_e/hand_checks.py"],
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=process_count())
    parser.add_argument("--part-a", type=Path, default=None,
                        help="Part A table (default runs/e-exp029-part-a/part_a.csv)")
    args = parser.parse_args(argv)
    run(args.output, argv, pilot=args.pilot, processes=args.processes, part_a_csv=args.part_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
