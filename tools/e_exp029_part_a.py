"""EXP-029 Part A: unnecessary landings from synthetic command loss, link only.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/e_exp029_part_a.py --output runs/e-exp029-part-a

Simulated only.  The drone-side failsafe sees nothing but the arrival times of
the ground station's command packets (each command is also the heartbeat), so
no vehicle is simulated.  For every link preset and command rate h, one
synthetic loss process (sim.missions.comms.LossProcess) runs for 10 simulated
hours.  Command k is sent at k / h on stream "cmd:d1", is lost if
LossProcess.is_lost says so, and otherwise arrives 0.002 s later (the mission
command delay).  Loss patterns are synthetic, never measured.

Counting follows sim/missions/stack.py exactly.  At each control step
t_i = i * 0.01 s, every arrival with arrival <= t_i + 1e-12 is processed
first, then the drone-side link age is t_i - newest arrival (t = 0 counts as a
received arming command, so the age starts at 0).  For a policy
(T_w, T, weak rule):
- an unnecessary landing happens in a gap between consecutive arrivals (a, b)
  if some control step t_i in [a, b) has t_i - a >= T;
- after a landing the drone is assumed to be reset at the next arrival
  (renewal), so every gap can land at most once;
- the DEGRADED state (hold) is entered when the age reaches T_w, or, with the
  "age_or_loss" rule, when the loss fraction over the last 20 expected command
  sequence numbers reaches 0.3.

Every count is checked three ways: per-step arrays, the gap formula, and on a
1 h slice of four cells the real FailsafeController driven step by step.
Rates are compared with closed forms from tests/reference_e/hand_checks.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from sim import s6_recovery  # noqa: E402
from sim.failsafe import FailsafeConfig, FailsafeController, FailsafeState, SequenceLossWindow  # noqa: E402
from sim.missions.comms import LossProcess  # noqa: E402
from sim.missions.presets import PRESET_NAMES, get_preset  # noqa: E402
from tests.reference_e.hand_checks import (coarse_grid_chain, false_alarms_per_hour_markov,  # noqa: E402
                                           heartbeat_false_alarms_per_hour_iid, misses_needed)
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest  # noqa: E402
from tools.horizon_mpl import apply_style  # noqa: E402

DT_S = 0.01
TOL_S = 1e-12  # the stack's delivery tolerance (deliveries_until(now) uses now + 1e-12)
DUE_EPS = 1e-9  # the stack's send-schedule tolerance (floor(now * h + 1e-9))
COMMAND_DELAY_S = 0.002
STREAM_ID = "cmd:d1"
BASE_SEED = 20261029  # EXP-029 (contract section 5)
RATES_HZ = (20.0, 50.0)
T_W_VALUES_S = (0.10, 0.15, 0.25)
T_VALUES_S = (0.15, 0.2, 0.3, 0.5, 1.0, 2.0)
MIN_T_ABOVE_T_W_S = 0.05
WEAK_RULES = ("age", "age_or_loss")
LOSS_WINDOW_PACKETS = 20
WEAK_LOSS_FRACTION = 0.3
HOURS = 10.0
SLICE_HOURS = 1.0
SLICE_CELLS = (("iid_30", 20.0), ("burst_10_0.5s", 20.0))
SLICE_T_S = (0.3, 1.0)
SLICE_T_W_S = 0.15
RULE_T_W_S = 0.15  # the pre-registered recommendation uses the default degraded threshold
PLANNED_T_S = 0.30
PILOT_PRESETS = ("iid_30", "burst_10_0.5s")
PILOT_RATES_HZ = (20.0,)
PILOT_HOURS = 1.0
Z_FLAG = 4.0
MIN_EXPECTED_FOR_Z = 10.0
OUTPUTS = ("part_a.csv", "slice_check.json", "summary.json", "part_a_landings.png")


# ---- grid --------------------------------------------------------------------

def build_cells(pilot: bool = False) -> list[dict]:
    """Every (preset, rate) cell.  The cell index is the full-grid index, so a pilot
    cell uses exactly the same loss draws as the matching full-run cell."""
    seed = s6_recovery.seed_list(1, base_seed=BASE_SEED)[0]
    cells = []
    for p_index, preset in enumerate(PRESET_NAMES):
        for h_index, h in enumerate(RATES_HZ):
            index = p_index * len(RATES_HZ) + h_index
            if pilot and (preset not in PILOT_PRESETS or h not in PILOT_RATES_HZ):
                continue
            cell_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
            cells.append({"cell": index, "preset": preset, "h_hz": h, "seed": cell_seed})
    return cells


def policies() -> list[dict]:
    out = []
    for rule in WEAK_RULES:
        for t_w in T_W_VALUES_S:
            for t in T_VALUES_S:
                if t >= t_w + MIN_T_ABOVE_T_W_S - 1e-9:
                    out.append({"weak_rule": rule, "T_w_s": t_w, "T_s": t})
    return out


def failsafe_config(policy: dict) -> FailsafeConfig:
    return FailsafeConfig(degraded_after_s=policy["T_w_s"], land_after_s=policy["T_s"],
                          weak_rule=policy["weak_rule"], loss_window_packets=LOSS_WINDOW_PACKETS,
                          weak_loss_fraction=WEAK_LOSS_FRACTION)


# ---- command arrivals ----------------------------------------------------------

def n_steps_for(hours: float) -> int:
    return int(round(hours * 3600.0 / DT_S))


def n_commands_for(n_steps: int, h: float) -> int:
    """Commands sent at or before the last control step: k / h <= (n_steps - 1) * DT."""
    return int(math.floor((n_steps - 1) * DT_S * h + DUE_EPS)) + 1


def command_losses(loss: LossProcess, h: float, n_commands: int) -> np.ndarray:
    """lost[k] for command k sent at k / h, equal to loss.is_lost(STREAM_ID, k, k / h).

    Burst presets read the shared 0.01 s bad-slot grid directly with the same
    slot formula as LossProcess (floor(send / dt + 1e-12), clipped), which
    avoids one Python call per command.  Independent presets call is_lost in
    sequence order, which is how their per-stream generator is defined.
    """
    model = loss.preset.loss_model
    if model == "none":
        return np.zeros(n_commands, dtype=bool)
    k = np.arange(n_commands)
    send = k / h
    if model == "burst":
        slot = np.floor(send / loss.dt_s + 1e-12).astype(np.int64)
        slot = np.clip(slot, 0, len(loss.bad) - 1)
        return loss.bad[slot].copy()
    return np.fromiter((loss.is_lost(STREAM_ID, int(i), float(i) / h) for i in range(n_commands)),
                       dtype=bool, count=n_commands)


def processing_steps(arrival_s: np.ndarray) -> np.ndarray:
    """First control step i with arrival <= i * DT + 1e-12 (the stack's delivery rule)."""
    i = np.ceil((arrival_s - TOL_S) / DT_S).astype(np.int64)
    i = np.maximum(i, 0)
    while True:
        too_late = (i > 0) & ((i - 1) * DT_S + TOL_S >= arrival_s)
        too_early = i * DT_S + TOL_S < arrival_s
        if not (too_late.any() or too_early.any()):
            return i
        i = i - too_late.astype(np.int64) + too_early.astype(np.int64)


def make_arrivals(lost: np.ndarray, h: float, n_steps: int) -> dict:
    """Received commands that are processed inside the horizon, in sequence order.

    With a fixed delay the arrivals keep the send order, so the newest arrival
    is always the one with the highest sequence number.
    """
    seq = np.flatnonzero(~lost)
    arrival = seq / h + COMMAND_DELAY_S
    proc = processing_steps(arrival)
    keep = proc < n_steps
    return {"seq": seq[keep], "arrival_s": arrival[keep], "proc_step": proc[keep], "n_commands": int(lost.size),
            "received": ~lost}


# ---- vectorised counting -------------------------------------------------------

def step_arrays(arr: dict, n_steps: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-step drone-side link age and the age_or_loss rule's loss flag.

    Loss fraction (the convention of LinkedDrone._expected_next_seq in
    sim/missions/stack.py): the drone knows the command rate but not the
    ground station's counter, so every full command period since the newest
    arrival counts as one missing packet:

        expected_next_seq = newest seq + 1 + max(0, floor((t - newest arrival) h + 1e-9)).

    The window covers the last 20 expected numbers ending at expected_next_seq
    - 1.  A packet still in flight is never counted as lost, and before the
    first arrival the fraction is unknown (None), which counts as healthy.
    """
    t = np.arange(n_steps) * DT_S
    proc = arr["proc_step"]
    n_proc = np.searchsorted(proc, np.arange(n_steps), side="right")  # real arrivals processed by step i
    times = np.concatenate(([0.0], arr["arrival_s"]))  # index 0 is the arming command at t = 0
    newest = times[n_proc]
    age = np.maximum(0.0, t - newest)

    seqs = np.concatenate(([0], arr["seq"]))  # index 0 is a placeholder, masked out below
    missed = np.floor((t - newest) * h + DUE_EPS).astype(np.int64)
    end = seqs[n_proc] + np.maximum(0, missed)  # expected_next_seq - 1
    received = arr["received"]
    if end.max() >= received.size:
        raise ValueError("loss array shorter than the expected sequence range")
    cum = np.concatenate(([0], np.cumsum(received, dtype=np.int64)))  # cum[k] = received with seq < k
    in_flight = cum[end + 1] - n_proc  # received with seq <= end but not yet processed
    if np.any(in_flight[n_proc > 0] != 0):
        # Under this convention a missed period means the next packet's arrival time has passed,
        # so no received packet inside the window can still be in flight.
        raise AssertionError("a packet still in flight fell inside the loss window")
    start = np.maximum(0, end - LOSS_WINDOW_PACKETS + 1)
    total = (end - start + 1).astype(float)
    got = (cum[end + 1] - cum[start] - in_flight).astype(float)
    loss_fraction = (total - got) / total
    lossy = (n_proc > 0) & (loss_fraction >= WEAK_LOSS_FRACTION)  # None (no packet yet) counts as healthy
    return age, lossy


def _rising(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))


def policy_counts(age: np.ndarray, lossy: np.ndarray, policy: dict) -> dict:
    """State per step from the age (monotone inside a gap) and the loss flag.

    Under renewal the state is a pure function of the current step: LANDING or
    LANDED while age >= T (the age only grows until the next arrival, where
    the reset happens), else DEGRADED if the weak rule fires, else NOMINAL.
    """
    landing = age >= policy["T_s"]
    weak = age >= policy["T_w_s"]
    if policy["weak_rule"] == "age_or_loss":
        weak = weak | lossy
    degraded = ~landing & weak
    return {"landings": _rising(landing), "degraded_entries": _rising(degraded),
            "degraded_steps": int(np.count_nonzero(degraded))}


def gap_landings(arrival_s: np.ndarray, proc_step: np.ndarray, n_steps: int, timeout_s: float) -> int:
    """Gap formula: a landing in gap (a, b) iff the last control step before b is at least T after a.

    arrival_s and proc_step include the arming command (0.0, step 0).  The last
    gap ends at the final control step of the horizon.
    """
    next_proc = np.append(proc_step[1:], n_steps)
    last_step = next_proc - 1
    open_gap = last_step >= proc_step  # two arrivals processed in the same step leave an empty gap
    max_age = last_step * DT_S - arrival_s
    return int(np.count_nonzero(open_gap & (max_age >= timeout_s)))


# ---- ground truth: the real controller ---------------------------------------

def drive_controller(arr: dict, n_steps: int, h: float, policy: dict) -> dict:
    """Run FailsafeController (and SequenceLossWindow for age_or_loss) step by step.

    Same arrivals, same grid.  A landing is acknowledged at once (landed=True),
    then controller.reset is called at the first step that processes a new
    arrival (renewal), before that step's observation.
    """
    config = failsafe_config(policy)
    ctrl = FailsafeController(config)
    window = SequenceLossWindow(LOSS_WINDOW_PACKETS) if policy["weak_rule"] == "age_or_loss" else None
    times = arr["arrival_s"].tolist()
    seqs = arr["seq"].tolist()
    n = len(times)
    j = 0
    newest = 0.0
    newest_seq = None
    landings = entries = degraded_steps = 0
    for i in range(n_steps):
        t = i * DT_S
        fresh = False
        while j < n and times[j] <= t + TOL_S:
            newest = max(newest, times[j])
            newest_seq = seqs[j] if newest_seq is None else max(newest_seq, seqs[j])
            if window is not None:
                window.record(seqs[j])
            j += 1
            fresh = True
        age = max(0.0, t - newest)
        if fresh and ctrl.state is FailsafeState.LANDED:
            ctrl.reset(t, link_age_s=age, battery_pct=100.0)
        loss = None
        if window is not None and newest_seq is not None:
            expected = newest_seq + 1 + max(0, int(math.floor((t - newest) * h + DUE_EPS)))
            loss = window.loss_fraction(expected)
        d = ctrl.observe(t, link_age_s=age, battery_pct=100.0, recent_loss_fraction=loss)
        if d.state is FailsafeState.LANDING and d.previous_state is not FailsafeState.LANDING:
            landings += 1
            ctrl.observe(t, link_age_s=age, battery_pct=100.0, landed=True, recent_loss_fraction=loss)
        if d.state is FailsafeState.DEGRADED:
            degraded_steps += 1
            if d.previous_state is not FailsafeState.DEGRADED:
                entries += 1
    return {"landings": landings, "degraded_entries": entries, "degraded_steps": degraded_steps}


# ---- closed forms ------------------------------------------------------------------

def effective_misses(h: float, timeout_s: float) -> int:
    """Misses needed with the 0.002 s delay and the 0.01 s grid.

    Command k arrives at k/h + 0.002 and is processed at the next grid step,
    k/h + 0.01 (send times are on the grid at 20 and 50 Hz).  If the next
    arrival is command k + g, the last step before it is processed is
    (k + g)/h, where the age is g/h - 0.002.  A landing needs
    g/h - 0.002 >= T, so g >= h (T + 0.002) and the misses are
    m = g - 1 = ceil(h (T + 0.002)) - 1.  For h T a whole number n this is n;
    for h T = n + 0.5 (50 Hz, T = 0.15 s) it is also n, so m = floor(T h) on
    every grid value (checked in the summary).
    """
    return int(math.ceil(h * (timeout_s + COMMAND_DELAY_S) - 1e-9)) - 1


def closed_form_per_hour(preset_name: str, h: float, timeout_s: float) -> float:
    preset = get_preset(preset_name)
    if preset.loss_model == "none":
        return 0.0
    p = float(preset.loss_probability)
    if preset.loss_model == "independent":
        return heartbeat_false_alarms_per_hour_iid(p, h, timeout_s)
    r0 = DT_S / float(preset.burst_mean_s)
    q0 = p * r0 / (1.0 - p)
    k = int(round(1.0 / (h * DT_S)))
    q, r = coarse_grid_chain(q0, r0, k)
    return false_alarms_per_hour_markov(q, r, h, timeout_s)


def poisson_interval(count: int, hours: float) -> tuple[float, float]:
    """Exact (chi-square) 95 % interval for a Poisson rate."""
    low = 0.0 if count == 0 else float(chi2.ppf(0.025, 2 * count) / 2.0)
    high = float(chi2.ppf(0.975, 2 * (count + 1)) / 2.0)
    return low / hours, high / hours


# ---- one cell --------------------------------------------------------------------

def run_cell(job: dict) -> dict:
    cell, hours = job["cell"], job["hours"]
    h = float(cell["h_hz"])
    n_steps = n_steps_for(hours)
    n_commands = n_commands_for(n_steps, h)
    loss = LossProcess(cell["preset"], seed=cell["seed"], dt_s=DT_S, duration_s=hours * 3600.0)
    lost = command_losses(loss, h, n_commands)
    arr = make_arrivals(lost, h, n_steps)
    age, lossy = step_arrays(arr, n_steps, h)
    all_times = np.concatenate(([0.0], arr["arrival_s"]))
    all_proc = np.concatenate(([0], arr["proc_step"]))
    rows = []
    for policy in policies():
        counts = policy_counts(age, lossy, policy)
        by_gap = gap_landings(all_times, all_proc, n_steps, policy["T_s"])
        if by_gap != counts["landings"]:
            raise AssertionError(f"gap formula {by_gap} != per-step {counts['landings']} for {cell} {policy}")
        rows.append({**policy, **counts})
    checks = []
    if (cell["preset"], h) in SLICE_CELLS:
        n_slice = n_steps_for(min(SLICE_HOURS, hours))
        for rule in WEAK_RULES:
            for t in SLICE_T_S:
                policy = {"weak_rule": rule, "T_w_s": SLICE_T_W_S, "T_s": t}
                vec = policy_counts(age[:n_slice], lossy[:n_slice], policy)
                ctrl = drive_controller(arr, n_slice, h, policy)
                checks.append({"preset": cell["preset"], "h_hz": h, **policy, "hours": n_slice * DT_S / 3600.0,
                               "vectorised": vec, "controller": ctrl, "equal": vec == ctrl})
    return {"cell": cell, "n_commands": n_commands, "n_lost": int(np.count_nonzero(lost)), "rows": rows,
            "checks": checks}


# ---- tables -------------------------------------------------------------------------

def table_rows(results: list[dict], hours: float) -> list[dict]:
    out = []
    for res in results:
        cell = res["cell"]
        h = float(cell["h_hz"])
        n_steps = n_steps_for(hours)
        for r in res["rows"]:
            count = r["landings"]
            rate = count / hours
            low, high = poisson_interval(count, hours)
            cf = closed_form_per_hour(cell["preset"], h, r["T_s"])
            expected = cf * hours
            out.append({
                "preset": cell["preset"], "h_hz": h, "weak_rule": r["weak_rule"], "T_w_s": r["T_w_s"],
                "T_s": r["T_s"], "hours": hours, "seed": cell["seed"], "commands_sent": res["n_commands"],
                "commands_lost": res["n_lost"], "unnecessary_landings": count, "landings_per_hour": rate,
                "landings_per_hour_low_95": low, "landings_per_hour_high_95": high,
                "degraded_entries": r["degraded_entries"], "degraded_entries_per_hour": r["degraded_entries"] / hours,
                "fraction_time_degraded": r["degraded_steps"] / n_steps,
                "closed_form_per_hour": cf, "expected_landings": expected,
                "observed_over_expected": (count / expected) if expected > 0 else None,
                "poisson_se_per_hour": (math.sqrt(expected) / hours) if expected > 0 else None,
                "z": ((count - expected) / math.sqrt(expected)) if expected > 0 else None,
                "z_checked": bool(expected >= MIN_EXPECTED_FOR_Z),
                "z_flag": bool(expected >= MIN_EXPECTED_FOR_Z and abs((count - expected) / math.sqrt(expected)) > Z_FLAG),
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return repr(v)
        return v
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: fmt(v) for k, v in r.items()})


def summarize(rows: list[dict], cells: list[dict], checks: list[dict], hours: float) -> dict:
    per_cell = []
    for cell in cells:
        base = [r for r in rows if r["preset"] == cell["preset"] and r["h_hz"] == cell["h_hz"]
                and r["T_w_s"] == RULE_T_W_S]
        age_rows = sorted((r for r in base if r["weak_rule"] == "age"), key=lambda r: r["T_s"])
        loss_rows = {r["T_s"]: r for r in base if r["weak_rule"] == "age_or_loss"}
        smallest = next((r["T_s"] for r in age_rows if r["landings_per_hour"] < 1.0), None)
        smallest_cf = next((r["T_s"] for r in age_rows if r["closed_form_per_hour"] < 1.0), None)
        planned = next(r for r in age_rows if r["T_s"] == PLANNED_T_S)
        planned_loss = loss_rows[PLANNED_T_S]
        at_smallest = next((r for r in age_rows if r["T_s"] == smallest), None)
        per_cell.append({
            "preset": cell["preset"], "h_hz": cell["h_hz"], "T_w_s": RULE_T_W_S,
            "smallest_T_s_under_1_per_hour": smallest,
            "landings_per_hour_at_smallest_T": None if at_smallest is None else at_smallest["landings_per_hour"],
            "smallest_T_s_under_1_per_hour_closed_form": smallest_cf,
            "planned_T_s": PLANNED_T_S,
            "planned_landings_per_hour": planned["landings_per_hour"],
            "planned_landings_per_hour_low_95": planned["landings_per_hour_low_95"],
            "planned_landings_per_hour_high_95": planned["landings_per_hour_high_95"],
            "planned_closed_form_per_hour": planned["closed_form_per_hour"],
            "planned_z": planned["z"],
            "planned_degraded_entries_per_hour_age": planned["degraded_entries_per_hour"],
            "planned_degraded_entries_per_hour_age_or_loss": planned_loss["degraded_entries_per_hour"],
            "planned_fraction_time_degraded_age": planned["fraction_time_degraded"],
            "planned_fraction_time_degraded_age_or_loss": planned_loss["fraction_time_degraded"],
        })
    # Landings depend only on T (the weak rule and T_w change DEGRADED, never the age), so the
    # closed-form comparison uses one row per (preset, h, T).
    unique = {}
    for r in rows:
        unique.setdefault((r["preset"], r["h_hz"], r["T_s"]), r)
    for key, r in unique.items():
        same = {x["unnecessary_landings"] for x in rows if (x["preset"], x["h_hz"], x["T_s"]) == key}
        if len(same) != 1:
            raise AssertionError(f"landings changed with T_w or weak rule at {key}")
    checked = [r for r in unique.values() if r["z_checked"]]
    worst = max(checked, key=lambda r: (abs(r["z"]), r["preset"], r["h_hz"], r["T_s"]), default=None)
    agreement = {
        "rule": f"|z| <= {Z_FLAG:g} on cells expecting at least {MIN_EXPECTED_FOR_Z:g} landings",
        "cells_checked": len(checked),
        "cells_flagged": sum(1 for r in checked if r["z_flag"]),
        "flagged": [{"preset": r["preset"], "h_hz": r["h_hz"], "T_s": r["T_s"], "z": r["z"]}
                    for r in sorted(checked, key=lambda r: (r["preset"], r["h_hz"], r["T_s"])) if r["z_flag"]],
        "worst": None if worst is None else {"preset": worst["preset"], "h_hz": worst["h_hz"], "T_s": worst["T_s"],
                                             "z": worst["z"], "observed": worst["unnecessary_landings"],
                                             "expected": worst["expected_landings"]},
        "max_abs_z": None if worst is None else abs(worst["z"]),
    }
    misses = [{"h_hz": h, "T_s": t, "effective_misses": effective_misses(h, t),
               "floor_T_h": misses_needed(h, t), "equal": effective_misses(h, t) == misses_needed(h, t)}
              for h in RATES_HZ for t in T_VALUES_S]
    return {
        "experiment": "EXP-029 Part A",
        "status": "simulated; synthetic loss patterns; drone-side rule only",
        "hours_per_cell": hours,
        "recommendation": per_cell,
        "closed_form_agreement": agreement,
        "effective_misses_check": misses,
        "slice_check_all_equal": all(c["equal"] for c in checks),
        "slice_checks": len(checks),
        "rules": {
            "unnecessary_landing": "a gap between consecutive command arrivals (a, b) lands if a control step "
                                   "t_i in [a, b) has t_i - a >= T",
            "renewal": "after a landing the drone is reset at the next arrival; at most one landing per gap",
            "recommendation": "smallest T with under 1 unnecessary landing per hour (observed), T >= T_w + 0.05 s, "
                              "T_w = 0.15 s; landings do not depend on T_w or the weak rule",
            "degraded": "DEGRADED when age >= T_w, or (age_or_loss) loss over the last 20 expected sequence "
                        "numbers >= 0.3; expected_next_seq = newest seq + 1 + max(0, floor((t - newest arrival) h + 1e-9)), "
                        "so a packet in flight never counts as lost",
            "fraction_time_degraded": "control steps in DEGRADED (not LANDING or LANDED) over all steps",
            "closed_form": "iid: 3600 h (1 - p) p^m, m = floor(T h); burst: the 0.01 s chain "
                           "(q0 = p r0 / (1 - p), r0 = 0.01 / burst_mean_s) seen every 100 / h slots",
            "interval": "exact Poisson 95 % (chi-square) on the landing count, divided by hours",
        },
    }


# ---- figure -----------------------------------------------------------------------

def make_figure(rows: list[dict], output: Path) -> None:
    colors = apply_style("light")
    by_p = {0.01: colors["accent"], 0.05: colors["ok"], 0.10: colors["keepOut"], 0.30: colors["target"]}
    rates = sorted({r["h_hz"] for r in rows})
    presets = [p for p in PRESET_NAMES if any(r["preset"] == p for r in rows)]
    fig, axes = plt.subplots(1, len(rates), figsize=(5.2 * len(rates), 4.4), sharey=True, squeeze=False)
    nonzero = [r["landings_per_hour"] for r in rows if r["landings_per_hour"] > 0]
    y_low = min(min(nonzero, default=1.0), 1.0) / 3.0  # keep the 1 per hour line in view
    y_high = max(max(nonzero, default=1.0), 1.0) * 3.0
    for ax, h in zip(axes[0], rates):
        for preset in presets:
            pr = get_preset(preset)
            sub = sorted((r for r in rows if r["preset"] == preset and r["h_hz"] == h and r["weak_rule"] == "age"
                          and r["T_w_s"] == T_W_VALUES_S[0]), key=lambda r: r["T_s"])
            pts = [(r["T_s"], r["landings_per_hour"]) for r in sub if r["landings_per_hour"] > 0]
            if pr.loss_model == "none":
                style = dict(color=colors["text"], ls="-", marker="o")
            elif pr.loss_model == "independent":
                style = dict(color=by_p[pr.loss_probability], ls="-", marker="o")
            elif pr.burst_mean_s == 0.5:
                style = dict(color=by_p[pr.loss_probability], ls="--", marker="s")
            else:
                style = dict(color=colors["textSecondary"], ls=":", marker="^")
            never = not any(r["landings_per_hour"] > 0 for r in rows if r["preset"] == preset)
            label = f"{preset} (no landings at any rate)" if never else preset
            ax.plot([x for x, _ in pts], [y for _, y in pts], ms=4, lw=1.3, label=label, **style)
        ax.axhline(1.0, color=colors["alert"], lw=1.0, ls="--")
        ax.text(T_VALUES_S[0], 0.85, "1 per hour", color=colors["alert"], fontsize=8, ha="left", va="top")
        ax.set_xscale("log")
        ax.set_xticks(T_VALUES_S, [f"{t:g}" for t in T_VALUES_S])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yscale("log")
        ax.set_ylim(y_low, y_high)
        ax.set_xlabel("lost-link timeout T (s, log scale)")
        ax.set_title(f"commands at {h:g} Hz", fontsize=9)
    axes[0][0].set_ylabel("unnecessary landings per hour (log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    hours = rows[0]["hours"]
    fig.suptitle(f"Simulated: unnecessary landings per hour vs T (synthetic loss, {hours:g} h per preset)\n"
                 "drone-side age rule, reset at the next arrival; zero counts are not drawn", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "part_a_landings.png", dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---- run ------------------------------------------------------------------------------

def default_processes() -> int:
    return max(1, min(2, os.cpu_count() or 1))


def run(output: Path, argv: list[str], *, pilot: bool = False, processes: int | None = None) -> dict:
    start = time.perf_counter()
    output = prepare_output(output)
    hours = PILOT_HOURS if pilot else HOURS
    cells = build_cells(pilot)
    jobs = [{"cell": c, "hours": hours} for c in cells]
    proc = default_processes() if processes is None else max(1, int(processes))
    if proc > 1:
        with ProcessPoolExecutor(max_workers=proc) as pool:
            results = list(pool.map(run_cell, jobs, chunksize=1))
    else:
        results = [run_cell(job) for job in jobs]
    rows = table_rows(results, hours)
    checks = [c for res in results for c in res["checks"]]
    write_csv(output / "part_a.csv", rows)
    write_json(output / "slice_check.json", {
        "experiment": "EXP-029 Part A", "status": "simulated",
        "what": "first hour of each listed cell, counted by the vectorised arrays and by the real "
                "FailsafeController (SequenceLossWindow for age_or_loss), reset at the next arrival after a landing",
        "all_equal": all(c["equal"] for c in checks), "checks": checks})
    if not all(c["equal"] for c in checks):
        raise AssertionError("vectorised counts differ from the FailsafeController on the slice")
    write_json(output / "summary.json", summarize(rows, cells, checks, hours))
    make_figure(rows, output)
    runtime = time.perf_counter() - start
    print(f"runtime {runtime:.1f} s for {len(cells)} cells of {hours:g} h on {proc} processes")
    config = {"pilot": pilot, "hours_per_cell": hours, "presets": [c["preset"] for c in cells],
              "rates_hz": sorted({c["h_hz"] for c in cells}), "T_w_s": list(T_W_VALUES_S), "T_s": list(T_VALUES_S),
              "min_T_above_T_w_s": MIN_T_ABOVE_T_W_S, "weak_rules": list(WEAK_RULES),
              "loss_window_packets": LOSS_WINDOW_PACKETS, "weak_loss_fraction": WEAK_LOSS_FRACTION,
              "command_delay_s": COMMAND_DELAY_S, "dt_s": DT_S, "stream_id": STREAM_ID, "base_seed": BASE_SEED,
              "cell_seeds": {f"{c['preset']}@{c['h_hz']:g}": c["seed"] for c in cells},
              "slice": {"hours": SLICE_HOURS, "cells": [[p, h] for p, h in SLICE_CELLS], "T_s": list(SLICE_T_S),
                        "T_w_s": SLICE_T_W_S}}
    return write_manifest(
        output, schema="zephyr-run-e-manifest-1", argv=argv,
        command="PYTHONPATH=. ./.venv/bin/python tools/e_exp029_part_a.py --output {out}", config=config,
        seeds=[c["seed"] for c in cells], outputs=list(OUTPUTS),
        limits=["Loss patterns are synthetic (sim/missions/presets.py), not measured on any link.",
                "Drone-side rule only: the failsafe sees command arrivals; no vehicle, telemetry or ground-station "
                "logic is simulated in Part A.",
                "Renewal assumption: after an unnecessary landing the drone is reset at the next command arrival, "
                "so each gap between arrivals can land at most once.",
                "One loss stream per cell (10 simulated hours), so cells share no draws and the rate intervals "
                "are Poisson intervals on a single long run.",
                "The loss fraction follows the mission stack: each full command period since the newest arrival "
                "counts as one missing packet, so a packet still in flight never counts as lost."],
        runtime_s=runtime, n_processes=proc,
        hashed_sources=["tools/e_exp029_part_a.py", "sim/missions/comms.py", "sim/missions/presets.py",
                        "sim/failsafe.py", "sim/missions/stack.py", "tests/reference_e/hand_checks.py",
                        "sim/s6_recovery.py"])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=default_processes())
    args = parser.parse_args(argv)
    run(args.output, argv, pilot=args.pilot, processes=args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
