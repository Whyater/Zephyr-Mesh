from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from sim.coop.allocation import assign, brute_force_assign, slots_for_alive
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator
from sim.estimation.sweep import run_jobs
# The simulator's estimator factory imports this lazily inside each job. Import it here
# so the manifest's hashed sources list it even when only worker processes run jobs.
import sim.estimation.cv  # noqa: F401
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import (
    MOVING_AGENT_LIMITS,
    argv_or_default,
    default_processes,
    prepare_output,
    seeds_for,
    write_json,
    write_manifest,
)


THRESHOLD_PATH = Path("sim/runs/c5-heartbeat/thresholds.json")
DROP_TIME_S = 10.0
DURATION_S = 25.0
HOLD_S = 1.0
RING_RADIUS_M = 1.5
SLOT_ALTITUDE_M = 1.5
SWITCH_PENALTY_M = 0.3
GS_WINDOWS = ((5.0, 10.0), (10.0, 15.0))


def _thresholds() -> dict[str, float]:
    """Read the registered heartbeat timeout for each link-loss level."""
    data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    values: dict[float, float] = {}
    for row in data["records"]:
        condition = row.get("loss_condition", "")
        if (
            row.get("h_hz") == 10
            and condition in {
                "independent_p_0.01",
                "independent_p_0.1",
                "independent_p_0.3",
            }
            and row.get("false_alarm_below_1_per_hour")
        ):
            values[float(condition.rsplit("_", 1)[1])] = float(
                row["false_alarm_below_1_per_hour"]["T_s"]
            )
    return {"0.0": values[0.1], "0.1": values[0.1], "0.3": values[0.3]}


def _config(
    n: int,
    k: int,
    mode: str,
    loss: float,
    seed: int,
    timeout: float,
) -> CoopSwarmConfig:
    """Build the registered simulated dropout configuration."""
    dropped = tuple(
        (f"drone-{index:03d}", DROP_TIME_S, mode)
        for index in range(n - k + 1, n + 1)
    )
    return CoopSwarmConfig(
        n_agents=n,
        seed=seed,
        duration_s=DURATION_S,
        control_dt_s=0.02,
        sensor_rate_hz=10.0,
        noise_std_m=0.05,
        estimator="cv",
        fusion="ci",
        decision="onboard",
        target="ct",
        target_params=(("radius", 3.0), ("speed", 0.8)),
        standoff_radius_m=RING_RADIUS_M,
        slot_altitude_m=SLOT_ALTITUDE_M,
        link=CoopLinkConfig(
            delay_s=0.05,
            loss_probability=loss,
            loss_model="independent",
            share_rate_hz=10.0,
        ),
        heartbeat_timeout_s=timeout,
        dropouts=dropped,
        switch_penalty_m=SWITCH_PENALTY_M,
        log_every=1,
        metric_windows=GS_WINDOWS,
        ci_mode="fast",
    )


def _survivors(n: int, k: int) -> tuple[str, ...]:
    """Return the agents that remain after the registered dropout."""
    return tuple(f"drone-{index:03d}" for index in range(1, n - k + 1))


def _dropped(n: int, k: int) -> set[str]:
    """Return the agents scheduled to leave at the fixed drop time."""
    return {f"drone-{index:03d}" for index in range(n - k + 1, n + 1)}


def _measure_detection(
    steps: list[dict], survivors: tuple[str, ...], dropped: set[str]
) -> float | None:
    """Measure the first post-drop time when every survivor knows all dropped IDs are gone."""
    for step in steps:
        if float(step["time_s"]) < DROP_TIME_S:
            continue
        agents = {row["agent_id"]: row for row in step["agents"]}
        if all(dropped.isdisjoint(agents[agent]["believed_alive"]) for agent in survivors):
            return float(step["time_s"]) - DROP_TIME_S
    return None


def _new_plan_error(step: dict, agent_id: str, survivors: tuple[str, ...]) -> float | None:
    """Measure distance from one survivor to its assigned slot in the new survivor ring."""
    agents = {row["agent_id"]: row for row in step["agents"]}
    agent = agents.get(agent_id)
    if agent is None or not agent.get("active"):
        return None
    believed_alive = tuple(agent.get("believed_alive", ()))
    if len(believed_alive) != len(survivors):
        return None
    assignment = agent.get("believed_assignment", {})
    slot_index = assignment.get(agent_id, agent.get("slot_index"))
    if slot_index is None:
        return None
    slots = slots_for_alive(
        np.asarray(step["target_truth_position_m"], dtype=float),
        len(survivors),
        RING_RADIUS_M,
        SLOT_ALTITUDE_M,
        0.0,
    )
    if int(slot_index) < 0 or int(slot_index) >= len(slots):
        return None
    position = np.asarray(agent["position_m"], dtype=float)
    return float(np.linalg.norm(position - slots[int(slot_index)]))


RECOVERAGE_DEFINITION = (
    "First logged t >= drop from which, for a full 1 s inside the logged data, every survivor"
    " (the dropped drones excluded) has len(believed_alive) == N-k and is within 0.3 m of the"
    " slot believed_assignment gives it, on the ring of radius 1.5 m at altitude 1.5 m around"
    " the true target position (world phase 0)."
)


def _survivor_gap_deg(step: dict, survivors: tuple[str, ...]) -> float:
    """Measure the largest angular gap around the target using survivors only."""
    target = np.asarray(step["target_truth_position_m"], dtype=float)
    agents = {row["agent_id"]: row for row in step["agents"]}
    angles = []
    for agent in survivors:
        if agent not in agents or not agents[agent].get("active"):
            continue
        delta = np.asarray(agents[agent]["position_m"], dtype=float) - target
        angles.append(float(np.arctan2(delta[1], delta[0])) % (2.0 * np.pi))
    angles.sort()
    if not angles:
        return 360.0
    gaps = [second - first for first, second in zip(angles, angles[1:])]
    gaps.append(2.0 * np.pi + angles[0] - angles[-1])
    return float(np.degrees(max(gaps)))


def _measure_recoverage(
    steps: list[dict], survivors: tuple[str, ...]
) -> tuple[float | None, int | None, list[tuple[float, float]]]:
    """Return re-coverage time, censoring flag and the survivor gap timeline.

    Re-coverage is defined in RECOVERAGE_DEFINITION.
    """
    post = [step for step in steps if float(step["time_s"]) >= DROP_TIME_S]
    timeline = [
        (float(step["time_s"]), _survivor_gap_deg(step, survivors))
        for step in steps
        if float(step["time_s"]) >= 5.0
    ]
    good: list[bool] = []
    for step in post:
        errors = [_new_plan_error(step, agent, survivors) for agent in survivors]
        good.append(bool(errors) and all(error is not None and error <= 0.3 for error in errors))
    for index, step in enumerate(post):
        start = float(step["time_s"])
        end = start + HOLD_S
        if not post or end > float(post[-1]["time_s"]) + 1e-12:
            continue
        end_index = max(
            candidate
            for candidate, candidate_step in enumerate(post)
            if float(candidate_step["time_s"]) <= end + 1e-12
        )
        if float(post[end_index]["time_s"]) >= end - 1e-12 and all(good[index : end_index + 1]):
            return start - DROP_TIME_S, index, timeline
    return None, None, timeline


def _measure_group_errors(
    steps: list[dict], start: float, end: float
) -> tuple[float | None, float | None]:
    """Measure ground-station group error p95 and maximum over a half-open window."""
    values = [
        float(step["gs_group_error_m"])
        for step in steps
        if start <= float(step["time_s"]) < end and step["gs_group_error_m"] is not None
    ]
    if not values:
        return None, None
    return float(np.percentile(values, 95)), float(np.max(values))


def _measure_separation(steps: list[dict]) -> tuple[float | None, int]:
    """Measure minimum pair separation and the count below the configured safety radius."""
    minimum: float | None = None
    violations = 0
    for step in steps:
        positions = [
            np.asarray(agent["position_m"], dtype=float)
            for agent in step["agents"]
            if agent.get("active")
        ]
        for index, first in enumerate(positions):
            for second in positions[index + 1 :]:
                distance = float(np.linalg.norm(first - second))
                minimum = distance if minimum is None else min(minimum, distance)
                violations += int(distance < 0.35 - 1e-12)
    return minimum, violations


def _measure_role_changes(
    steps: list[dict], survivors: tuple[str, ...]
) -> tuple[int, int]:
    """Measure survivor role changes as changes of (believed count, slot index)."""
    previous: dict[str, tuple[int, int]] = {}
    changes = 0
    plan_steps = 0
    for step in steps:
        time_s = float(step["time_s"])
        if not 10.0 <= time_s < 15.0:
            continue
        rows = {row["agent_id"]: row for row in step["agents"]}
        plan_steps += int(bool(step.get("plan_disagreement", False)))
        for agent_id in survivors:
            row = rows[agent_id]
            current = (len(row.get("believed_alive", ())), int(row.get("slot_index", 0)))
            changes += int(agent_id in previous and current != previous[agent_id])
            previous[agent_id] = current
    return changes, plan_steps


def _allocation_check() -> dict[str, int]:
    """Check the Hungarian assignment against brute force on seeded small matrices."""
    rng = np.random.default_rng(np.random.SeedSequence([20261018, 406]))
    passed = 0
    total = 0
    for n in range(1, 7):
        for _ in range(4):
            cost = rng.uniform(0, 10, (n, n))
            rows, columns = assign(cost)
            brute, _ = brute_force_assign(cost)
            total += 1
            passed += int(
                np.isclose(float(cost[rows, columns].sum()), brute, rtol=0, atol=1e-12)
            )
    return {"pass_count": passed, "case_count": total}


def _job(job: dict) -> dict:
    """Run one simulated seed and compute all registered dropout measures."""
    cpu_started = time.process_time()
    n = int(job["n"])
    k = int(job["k"])
    mode = str(job["mode"])
    loss = float(job["link_loss"])
    seed = int(job["seed"])
    timeout = float(job["timeout_s"])
    raw = CoopSwarmSimulator(_config(n, k, mode, loss, seed, timeout)).run().as_dict()
    steps = raw["steps"]
    survivors = _survivors(n, k)
    dropped = _dropped(n, k)
    detection = _measure_detection(steps, survivors, dropped)
    recovery, recovery_index, timeline = _measure_recoverage(steps, survivors)
    before_p95, before_max = _measure_group_errors(steps, *GS_WINDOWS[0])
    gap_p95, gap_max = _measure_group_errors(steps, *GS_WINDOWS[1])
    scored = [
        float(step["gs_group_error_m"])
        for step in steps
        if float(step["time_s"]) >= 2.0 and step["gs_group_error_m"] is not None
    ]
    overall_p95 = float(np.percentile(scored, 95)) if scored else None
    overall_max = float(np.max(scored)) if scored else None
    minimum_separation, separation_violations = _measure_separation(steps)
    role_changes, plan_steps = _measure_role_changes(steps, survivors)
    covered_fraction = None
    if recovery_index is not None:
        recovery_start = DROP_TIME_S + float(recovery)
        covered = []
        for step in steps:
            if float(step["time_s"]) < recovery_start:
                continue
            errors = [_new_plan_error(step, agent, survivors) for agent in survivors]
            covered.append(
                bool(errors) and all(error is not None and error <= 0.3 for error in errors)
            )
        covered_fraction = float(sum(covered) / max(1, len(covered)))
    hand_registered = None if mode == "landed" else timeout + 0.05
    hand_brief = None if mode == "landed" else timeout + 0.1 * loss / (1.0 - loss)
    return {
        "n": n,
        "k": k,
        "mode": mode,
        "link_loss": loss,
        "seed": seed,
        "detection_time_s": detection,
        "never_detected": detection is None,
        "recoverage_time_s": recovery,
        "recoverage_censored": recovery is None,
        "gs_before_p95_m": before_p95,
        "gs_before_max_m": before_max,
        "gs_gap_p95_m": gap_p95,
        "gs_gap_max_m": gap_max,
        "failed_in_gap": bool(gap_p95 is not None and (gap_p95 > 0.25 or gap_max > 0.50)),
        "seed_failed": bool(overall_p95 is not None and (overall_p95 > 0.25 or overall_max > 0.50)),
        "gs_overall_p95_m": overall_p95,
        "gs_overall_max_m": overall_max,
        "role_changes_after_drop": role_changes,
        "plan_disagreement_steps": plan_steps,
        "min_pairwise_true_separation_m": minimum_separation,
        "separation_violations": separation_violations,
        "violation": bool(separation_violations),
        "hand_detection_registered_s": hand_registered,
        "hand_detection_brief_s": hand_brief,
        "covered_fraction_after_recovery": covered_fraction,
        "survivor_gap_deg": max((gap for _, gap in timeline), default=None),
        "worker_cpu_s": time.process_time() - cpu_started,
        "timeline": timeline,
    }


CSV_COLUMNS = [
    "n",
    "k",
    "mode",
    "link_loss",
    "seed",
    "detection_time_s",
    "never_detected",
    "recoverage_time_s",
    "recoverage_censored",
    "gs_before_p95_m",
    "gs_before_max_m",
    "gs_gap_p95_m",
    "gs_gap_max_m",
    "gs_overall_p95_m",
    "gs_overall_max_m",
    "seed_failed",
    "failed_in_gap",
    "role_changes_after_drop",
    "plan_disagreement_steps",
    "min_pairwise_true_separation_m",
    "separation_violations",
    "violation",
    "hand_detection_registered_s",
    "hand_detection_brief_s",
    "covered_fraction_after_recovery",
    "survivor_gap_deg",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write per-seed results with stable columns and newlines."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_COLUMNS} for row in rows)


def _interval(values: Iterable[float | None]) -> dict | None:
    """Summarize non-censored values with a median and a labelled interval."""
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    median, low, high = median_interval(usable)
    return {
        "median": float(median),
        "lo": float(low),
        "hi": float(high),
        "n": len(usable),
    }


def _rate(values: Iterable[bool]) -> dict:
    """Summarize a Boolean seed rate with a Wilson interval and labelled counts."""
    values_list = [bool(value) for value in values]
    count = sum(values_list)
    low, high = wilson_interval(count, len(values_list))
    return {
        "rate": count / len(values_list),
        "lo": float(low),
        "hi": float(high),
        "count": count,
        "n": len(values_list),
    }


def _loaded_repo_sources() -> list[str]:
    """List loaded Python modules under sim/ and tools/ for manifest provenance."""
    root = Path(__file__).resolve().parents[1]
    paths: set[str] = set()
    for module in sys.modules.values():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            relative = Path(filename).resolve().relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in {"sim", "tools"}:
            paths.add(relative.as_posix())
    return sorted(paths)


def _full_cells() -> list[tuple[int, int, str, float]]:
    """Return the registered grid of cells."""
    return [
        (n, k, mode, loss)
        for n in (3, 4, 6)
        for k in (1, 2)
        if k <= n - 2
        for mode in ("silent", "landed")
        for loss in (0.0, 0.1, 0.3)
    ]


def _pilot_cells() -> list[tuple[int, int, str, float]]:
    """Return a small pilot containing every N,k group and both requested modes."""
    groups = [(3, 1), (4, 1), (4, 2), (6, 1), (6, 2)]
    cells = [(n, k, "silent", 0.1) for n, k in groups]
    cells.extend([(4, 1, "silent", loss) for loss in (0.0, 0.3)])
    cells.append((4, 1, "landed", 0.1))
    return cells


def _cpu_estimate(rows: list[dict], full_cells: list[tuple[int, int, str, float]]) -> dict:
    """Scale measured per-job CPU by each (N,k) group to the full registered grid."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row["n"]), int(row["k"]))
        grouped.setdefault(key, []).append(float(row["worker_cpu_s"]))
    full_counts: dict[tuple[int, int], int] = {}
    for n, k, _mode, _loss in full_cells:
        full_counts[(n, k)] = full_counts.get((n, k), 0) + 50
    estimates = {
        f"{n},{k}": float(np.mean(grouped[(n, k)]) * count)
        for (n, k), count in sorted(full_counts.items())
        if (n, k) in grouped
    }
    total = float(sum(estimates.values()))
    return {
        "per_group_cpu_s": {
            key: value / full_counts[tuple(int(part) for part in key.split(","))]
            for key, value in estimates.items()
        },
        "full_cpu_s": total,
        "full_wall_s_at_4_processes": total / 4.0,
    }


def run(output: Path, argv: list[str], pilot: bool = False, processes: int | None = None):
    """Run the simulated dropout grid, write results, figures, and a manifest."""
    started = time.perf_counter()
    output = prepare_output(output)
    thresholds = _thresholds()
    cells = _pilot_cells() if pilot else _full_cells()
    seeds = seeds_for("dropout", 3 if pilot else 50)
    jobs = [
        {
            "n": n,
            "k": k,
            "mode": mode,
            "link_loss": loss,
            "seed": seed,
            "timeout_s": thresholds[f"{loss:.1f}"],
        }
        for n, k, mode, loss in cells
        for seed in seeds
    ]
    process_limit = 2 if pilot else 4
    nproc = min(processes if processes is not None else default_processes(), process_limit)
    if nproc < 1:
        raise ValueError("processes must be positive")
    rows = run_jobs(jobs, "tools.d_run_dropout:_job", processes=nproc, progress=False)
    if len(rows) != len(jobs) or any(row is None for row in rows):
        raise RuntimeError("unexpected result row count")
    cpu_estimate = _cpu_estimate(rows, _full_cells())
    measured_cpu_s = sum(float(row.pop("worker_cpu_s", 0.0)) for row in rows)
    if pilot:
        print(
            "pilot worker CPU {:.3f} s; full estimate {:.3f} CPU s, {:.3f} wall s at 4 processes".format(
                measured_cpu_s,
                cpu_estimate["full_cpu_s"],
                cpu_estimate["full_wall_s_at_4_processes"],
            )
        )
    _write_csv(output / "per_seed.csv", rows)
    allocation = _allocation_check()
    summary: list[dict] = []
    for key in sorted({(row["n"], row["k"], row["mode"], row["link_loss"]) for row in rows}):
        group = [
            row
            for row in rows
            if (row["n"], row["k"], row["mode"], row["link_loss"]) == key
        ]
        seed_failure = _rate(row["seed_failed"] for row in group)
        recovery_censored = [row["recoverage_censored"] for row in group]
        summary.append(
            {
                "n": key[0],
                "k": key[1],
                "mode": key[2],
                "link_loss": key[3],
                "n_seeds": len(group),
                "detection_time_s": _interval(row["detection_time_s"] for row in group),
                "recoverage_time_s": _interval(row["recoverage_time_s"] for row in group),
                "gs_before_p95_m": _interval(row["gs_before_p95_m"] for row in group),
                "gs_before_max_m": _interval(row["gs_before_max_m"] for row in group),
                "gs_gap_p95_m": _interval(row["gs_gap_p95_m"] for row in group),
                "gs_gap_max_m": _interval(row["gs_gap_max_m"] for row in group),
                "gs_overall_p95_m": _interval(row["gs_overall_p95_m"] for row in group),
                "gs_overall_max_m": _interval(row["gs_overall_max_m"] for row in group),
                "seed_failure_rate": seed_failure,
                "cell_failed": bool(seed_failure["rate"] >= 0.5),
                "covered_fraction_after_recovery": _interval(
                    row["covered_fraction_after_recovery"] for row in group
                ),
                "survivor_gap_deg": _interval(row["survivor_gap_deg"] for row in group),
                "never_recovered_count": sum(recovery_censored),
                "never_recovered_rate": _rate(recovery_censored),
                "never_detected_count": sum(row["never_detected"] for row in group),
                "never_detected_rate": _rate(row["never_detected"] for row in group),
                "failed_in_gap_rate": _rate(row["failed_in_gap"] for row in group),
                "violation_rate": _rate(row["violation"] for row in group),
                "allocation_check_pass_count": allocation["pass_count"],
                "allocation_check_case_count": allocation["case_count"],
            }
        )
    write_json(output / "summary.json", summary)
    from tools.d_make_dropout_figure import make_figures

    make_figures(output, rows, summary)
    config = {
        "grid": cells,
        "duration_s": DURATION_S,
        "drop_time_s": DROP_TIME_S,
        "heartbeat_rate_hz": 10.0,
        "heartbeat_timeouts_s_by_loss": thresholds,
        "thresholds_sha256": hashlib.sha256(THRESHOLD_PATH.read_bytes()).hexdigest(),
        "control_dt_s": 0.02,
        "log_every": 1,
        "sensor_rate_hz": 10.0,
        "noise_std_m": 0.05,
        "estimator": "cv",
        "fusion": "ci",
        "ci_mode": "fast",
        "switch_penalty_m": SWITCH_PENALTY_M,
        "decision": "onboard",
        "target": "ct",
        "target_params": {"radius_m": 3.0, "speed_mps": 0.8},
        "standoff_radius_m": RING_RADIUS_M,
        "slot_altitude_m": SLOT_ALTITUDE_M,
        "separation_violation_radius_m": 0.35,
        "error_windows_s": [[start, end] for start, end in GS_WINDOWS],
        "link": {
            "mode": "abstract",
            "delay_s": 0.05,
            "loss_model": "independent",
            "share_rate_hz": 10.0,
            "loss_probabilities": [0.0, 0.1, 0.3],
        },
        "hand_detection_registered_formula": "T + 0.05",
        "hand_detection_brief_formula": "T + period*p/(1-p) for silent rows; None for landed rows",
        "recoverage_definition": RECOVERAGE_DEFINITION,
        "recovery_hold_s": HOLD_S,
        "role_change_definition": (
            "Per survivor, changes of (len(believed_alive), slot_index) during [10, 15) s."
        ),
        "survivor_gap_deg_definition": (
            "Largest angular gap between survivors around the target, maximum over t >= 5 s; "
            "it includes the geometry before the drop and is not a recovery measure."
        ),
        "plan_disagreement_note": (
            "plan_disagreement_steps cannot see plans built for different slot counts."
        ),
        "failure_rule": (
            "A seed fails when p95 > 0.25 m or max > 0.50 m over t >= 2 s; "
            "a cell fails when at least 50% of seeds fail."
        ),
        "seed_count": len(seeds),
    }
    limits = list(MOVING_AGENT_LIMITS) + [
        "Neighbor links are abstract simulated links.",
        "The drop time is fixed at 10 s.",
        "A landed leaving message is sent once and never retried.",
        "The heartbeat schedule has the same phase relative to the 10 s drop on every seed, "
        "so detection time has almost no seed-to-seed spread. This is a simulator limit, not "
        "measured behaviour.",
    ]
    return write_manifest(
        output,
        schema="zephyr-d-dropout-manifest-1",
        argv=argv,
        command="PYTHONPATH=. python tools/d_run_dropout.py",
        config=config,
        seeds=seeds,
        outputs=[
            "per_seed.csv",
            "summary.json",
            "dropout_times.png",
            "dropout_timeline.png",
        ],
        limits=limits,
        runtime_s=time.perf_counter() - started,
        n_processes=nproc,
        hashed_sources=_loaded_repo_sources(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    args = parser.parse_args()
    run(Path(args.output), argv_or_default(None), args.pilot, args.processes)
