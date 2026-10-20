"""Run the simulated ground versus onboard decision experiment."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, run as sim_run
from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import (
    argv_or_default,
    default_processes,
    prepare_output,
    seeds_for,
    write_json,
    write_manifest,
)

MODES = ("ground", "onboard")
LOSSES = (0.0, 0.1, 0.3)
OUTAGES = ("none", "10_15")
PILOT_LOSSES = (0.0, 0.1, 0.3)
KINDS = ("state", "raw", "track", "command", "leaving")

def _config(job):
    """Build one simulated decision experiment configuration."""
    outage = () if job["outage"] == "none" else ((10.0, 15.0),)
    fusion = "central" if job["mode"] == "ground" else "ci"
    return CoopSwarmConfig(
        n_agents=4,
        seed=int(job["seed"]),
        duration_s=20.0,
        control_dt_s=0.02,
        sensor_rate_hz=10.0,
        estimator="cv",
        fusion=fusion,
        decision=job["mode"],
        target="ct",
        target_params=(("radius", 3.0), ("speed", 0.8)),
        noise_std_m=0.05,
        link=CoopLinkConfig(
            delay_s=0.05,
            loss_probability=float(job["loss"]),
            loss_model="independent",
            share_rate_hz=10.0,
        ),
        standoff_radius_m=1.5,
        heartbeat_timeout_s=0.5,
        log_every=1,
        gs_outage=outage,
        dropouts=(),
        ci_mode="fast",
    )

def _worker(job):
    """Run one simulated seed and return its per-seed measures.

    decision_estimate_error_p95_m / _max_m: error of the estimate each mode decides from
    (the ground station's fused estimate in ground mode, the median over drones of each
    drone's own estimate per step in onboard mode), over 2 s <= t <= the run end.
    agent_track_error_p95_m / _max_m: the median over drones of each drone's own track error
    p95 and max from the simulator's per-agent summary, in both modes. In ground mode these
    tracks are not what decisions use; they are reported for comparison only.
    """
    cpu_start = time.process_time()
    raw = sim_run(_config(job)).as_dict()
    steps = raw["steps"]
    scored = [step for step in steps if float(step["time_s"]) >= 2.0 - 1e-12]
    if job["mode"] == "ground":
        decision = [
            float(step["gs_fuser_error_m"])
            for step in scored
            if step.get("gs_fuser_error_m") is not None
        ]
    else:
        decision = [
            float(np.median([
                agent["truth_error_m"]
                for agent in step["agents"]
                if agent.get("active") and agent.get("truth_error_m") is not None
            ]))
            for step in scored
            if any(agent.get("active") for agent in step["agents"])
        ]
    if not decision:
        raise RuntimeError("no scored decision estimates")
    per_agent = raw["summary"]["per_agent"]
    agent_p95 = [float(per_agent[aid]["p95_error_m"]) for aid in sorted(per_agent)]
    agent_max = [float(per_agent[aid]["max_error_m"]) for aid in sorted(per_agent)]
    def values_in_window(lo, hi, key="standoff_error_m", include_end=False):
        """Collect active agent values in a time window."""
        def in_window(value):
            end_ok = value <= hi if include_end else value < hi
            return lo <= value and end_ok

        return [
            float(agent[key])
            for step in steps
            if in_window(float(step["time_s"]))
            for agent in step["agents"]
            if agent.get("active") and agent.get(key) is not None
        ]

    allv = values_in_window(0.0, 20.0, include_end=True)
    v10 = values_in_window(10.0, 15.0)
    v15 = values_in_window(15.0, 20.0)
    cov = values_in_window(10.0, 15.0, "coverage_largest_gap_deg")
    active_steps = sum(
        1 for step in steps for agent in step["agents"] if agent.get("active")
    )
    timeout = sum(
        "command_timeout" in agent.get("flags", [])
        for step in steps
        for agent in step["agents"]
        if agent.get("active")
    )
    logged_steps = len(steps)
    disagreement_steps = sum(bool(step.get("plan_disagreement")) for step in steps)
    traffic = raw["summary"]["traffic"]
    timeline = []
    for step in steps:
        values = [
            float(agent["standoff_error_m"])
            for agent in step["agents"]
            if agent.get("active") and agent.get("standoff_error_m") is not None
        ]
        if values:
            timeline.append({
                "mode": job["mode"],
                "loss": float(job["loss"]),
                "outage": job["outage"],
                "seed": int(job["seed"]),
                "time_s": float(step["time_s"]),
                "standoff_error_m": float(np.median(values)),
            })
    p95 = float(np.percentile(decision, 95))
    max_error = float(np.max(decision))
    result = {
        "mode": job["mode"],
        "loss": float(job["loss"]),
        "outage": job["outage"],
        "seed": int(job["seed"]),
        "decision_estimate_error_p95_m": p95,
        "decision_estimate_error_max_m": max_error,
        "agent_track_error_p95_m": float(np.median(agent_p95)),
        "agent_track_error_max_m": float(np.median(agent_max)),
        "decision_failed": bool(p95 > 0.25 or max_error > 0.50),
        "gs_group_estimate_p95_m": raw["summary"]["gs_group_error_p95_m"],
        "standoff_error_m": float(np.median(allv)),
        "standoff_error_10_15_m": float(np.median(v10)),
        "standoff_error_15_20_m": float(np.median(v15)),
        "standoff_error_p95_10_15_m": float(np.percentile(v10, 95)),
        "coverage_largest_gap_median_deg": float(np.median(cov)),
        "min_true_separation_m": raw["summary"]["min_pairwise_separation_m"],
        "violation_count": int(raw["summary"]["spherical_violations_count"]),
        "plan_disagreement_steps": disagreement_steps,
        "logged_steps": logged_steps,
        "plan_disagreement_fraction": disagreement_steps / max(1, logged_steps),
        "command_timeout_agent_steps": timeout,
        "active_agent_steps": active_steps,
        "command_timeout_fraction": timeout / max(1, active_steps),
        "messages_per_second": sum(traffic[k]["messages"] for k in KINDS) / 20.0,
        "bytes_per_second": sum(traffic[k]["bytes"] for k in KINDS) / 20.0,
        "timeline": timeline,
        "traffic_message_bytes": {
            kind: int(traffic[kind]["message_bytes"]) for kind in KINDS
        },
        "worker_cpu_s": float(time.process_time() - cpu_start),
    }
    result.update({
        f"{kind}_bytes_per_second": traffic[kind]["bytes"] / 20.0
        for kind in KINDS
    })
    return result

COLS = [
    "mode", "loss", "outage", "seed",
    "decision_estimate_error_p95_m", "decision_estimate_error_max_m",
    "agent_track_error_p95_m", "agent_track_error_max_m",
    "decision_failed", "gs_group_estimate_p95_m", "standoff_error_m",
    "standoff_error_10_15_m", "standoff_error_15_20_m",
    "standoff_error_p95_10_15_m", "coverage_largest_gap_median_deg",
    "min_true_separation_m", "violation_count", "plan_disagreement_steps",
    "logged_steps", "plan_disagreement_fraction", "command_timeout_agent_steps",
    "active_agent_steps", "command_timeout_fraction", "messages_per_second",
    "bytes_per_second", *[f"{kind}_bytes_per_second" for kind in KINDS],
]

def _interval(values):
    """Return the median interval for one per-seed measure."""
    median, low, high = median_interval(values)
    return {"median": median, "lo": low, "hi": high, "n": len(values)}

def _rate(successes, n):
    """Return a labelled Wilson interval for a pooled step rate."""
    low, high = wilson_interval(int(successes), int(n))
    return {
        "rate": float(successes / n),
        "lo": low,
        "hi": high,
        "count": int(successes),
        "n": int(n),
    }

def _paired_differences(rows):
    """Summarize paired ground minus onboard standoff differences."""
    paired = []
    losses = sorted({float(row["loss"]) for row in rows})
    for loss in losses:
        for outage in OUTAGES:
            ground = {
                row["seed"]: row["standoff_error_10_15_m"]
                for row in rows
                if row["mode"] == "ground"
                and float(row["loss"]) == loss
                and row["outage"] == outage
            }
            onboard = {
                row["seed"]: row["standoff_error_10_15_m"]
                for row in rows
                if row["mode"] == "onboard"
                and float(row["loss"]) == loss
                and row["outage"] == outage
            }
            differences = [
                ground[seed] - onboard[seed]
                for seed in sorted(set(ground) & set(onboard))
            ]
            if differences:
                paired.append({
                    "loss": loss,
                    "outage": outage,
                    "median_ground_minus_onboard_standoff_10_15_m": _interval(
                        differences
                    ),
                })
    return paired


def _registered_comparison(rows):
    """Summarize registered traffic fields for each loss and outage cell."""
    comparison = []
    losses = sorted({float(row["loss"]) for row in rows})
    for loss in losses:
        for outage in OUTAGES:
            onboard = [
                row["state_bytes_per_second"]
                for row in rows
                if row["mode"] == "onboard"
                and float(row["loss"]) == loss
                and row["outage"] == outage
            ]
            ground = [
                row["raw_bytes_per_second"] + row["command_bytes_per_second"]
                for row in rows
                if row["mode"] == "ground"
                and float(row["loss"]) == loss
                and row["outage"] == outage
            ]
            if onboard and ground:
                ratios = [
                    ground_value / onboard_value
                    for ground_value, onboard_value in zip(ground, onboard)
                    if onboard_value
                ]
                comparison.append({
                    "loss": loss,
                    "outage": outage,
                    "onboard_state_bytes_per_second": _interval(onboard),
                    "ground_raw_plus_command_bytes_per_second": _interval(ground),
                    "ground_over_onboard_ratio": _interval(ratios),
                })
    return comparison


def _timeline_medians(rows):
    """Pool timeline values by cell and time, then take one median per group."""
    grouped = {}
    for row in rows:
        key = (row["mode"], row["loss"], row["outage"])
        for item in row["timeline"]:
            grouped.setdefault((*key, item["time_s"]), []).append(
                item["standoff_error_m"]
            )
    return [
        {
            "mode": key[0],
            "loss": key[1],
            "outage": key[2],
            "time_s": key[3],
            "standoff_error_m": float(np.median(values)),
        }
        for key, values in sorted(grouped.items())
    ]


def _summary(rows):
    """Build summary cells, pooled rates, paired values, and registration fields."""
    cells = []
    paired = _paired_differences(rows)
    comparison = _registered_comparison(rows)
    numeric = [
        "decision_estimate_error_p95_m", "decision_estimate_error_max_m",
        "agent_track_error_p95_m", "agent_track_error_max_m",
        "gs_group_estimate_p95_m", "standoff_error_m",
        "standoff_error_10_15_m", "standoff_error_15_20_m",
        "standoff_error_p95_10_15_m", "coverage_largest_gap_median_deg",
        "min_true_separation_m", "violation_count", "plan_disagreement_fraction",
        "command_timeout_fraction", "messages_per_second", "bytes_per_second",
        *[f"{kind}_bytes_per_second" for kind in KINDS],
    ]
    for mode in MODES:
        for loss in sorted({float(row["loss"]) for row in rows}):
            for outage in OUTAGES:
                rr = [
                    row for row in rows
                    if row["mode"] == mode
                    and float(row["loss"]) == loss
                    and row["outage"] == outage
                ]
                if not rr:
                    continue
                cell = {
                    "mode": mode,
                    "loss": loss,
                    "outage": outage,
                    "n_seeds": len(rr),
                }
                cell.update({
                    key: _interval([float(row[key]) for row in rr])
                    for key in numeric
                })
                failed = sum(bool(row["decision_failed"]) for row in rr)
                cell["decision_failed_fraction"] = _rate(failed, len(rr))
                if outage == "none":
                    cell["failure_rule_registered"] = True
                    cell["cell_failed"] = (
                        cell["decision_failed_fraction"]["rate"] >= 0.5
                    )
                else:
                    cell["failure_rule_registered"] = False
                    cell["cell_failed"] = None
                    if mode == "ground":
                        cell["note"] = (
                            "During the outage the ground estimate coasts with no "
                            "measurements, so its error exceeds the thresholds. It "
                            "drives no commands then."
                        )
                    else:
                        cell["note"] = (
                            "Onboard decisions use each drone's own shared estimate, "
                            "so the ground link outage does not change them."
                        )
                cell["seeds_with_violation_wilson"] = _rate(
                    sum(row["violation_count"] > 0 for row in rr), len(rr)
                )
                cell["plan_disagreement_steps_pooled_wilson"] = _rate(
                    sum(row["plan_disagreement_steps"] for row in rr),
                    sum(row["logged_steps"] for row in rr),
                )
                cell["command_timeout_agent_steps_pooled_wilson"] = _rate(
                    sum(row["command_timeout_agent_steps"] for row in rr),
                    sum(row["active_agent_steps"] for row in rr),
                )
                cells.append(cell)
    traffic_sizes = rows[0].get("traffic_message_bytes", {}) if rows else {}
    traffic = {
        kind: {"message_bytes": int(traffic_sizes.get(kind, 0))}
        for kind in KINDS
    }
    raw_bytes = traffic["raw"]["message_bytes"]
    command_bytes = traffic["command"]["message_bytes"]
    state_bytes = traffic["state"]["message_bytes"]
    return {
        "cells": cells,
        "paired_differences": paired,
        "traffic": traffic,
        "registered_comparison": {
            "cells": comparison,
            "note": (
                "Onboard sends no separate track messages, so onboard track bytes/s "
                "is 0 by construction because the shared track rides inside the "
                f"{state_bytes} B state message. The simulator raw message is "
                f"{raw_bytes} B and the command is {command_bytes} B, read from "
                "traffic[kind].message_bytes, rather "
                "than the 32 B assumed at registration. Prediction 3 is judged on "
                "totals over all kinds, per the orchestrator decision."
            ),
            "interval_note": (
                "Pooled step Wilson intervals assume independent steps and are too "
                "narrow because steps are autocorrelated."
            ),
        },
    }

def _jobs(losses, seeds):
    """Build the complete mode, loss, outage, and seed grid."""
    return [
        {"mode": mode, "loss": loss, "outage": outage, "seed": seed}
        for mode in MODES
        for loss in losses
        for outage in OUTAGES
        for seed in seeds
    ]


def _source_paths():
    """Return imported repository modules under sim/ and tools/ plus both scripts."""
    root = Path(__file__).resolve().parents[1]
    paths = {
        Path("tools/d_run_decisions.py"),
        Path("tools/d_make_decisions_figure.py"),
    }
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            relative = Path(filename).resolve().relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in {"sim", "tools"}:
            paths.add(relative)
    return sorted(str(path) for path in paths)


def run(output: Path, *, pilot=False, processes=None, argv=None):
    """Run the simulated grid, figures, summary, and manifest."""
    start = time.perf_counter()
    output = prepare_output(output)
    n = 3 if pilot else 50
    seeds = seeds_for("decisions", n)
    losses = PILOT_LOSSES if pilot else LOSSES
    requested = int(processes or default_processes())
    nproc = min(4, requested)
    if nproc < 1:
        raise ValueError("processes must be positive")
    jobs = _jobs(losses, seeds)
    _worker(jobs[0])
    rows = run_jobs(
        jobs,
        "tools.d_run_decisions:_worker",
        processes=nproc,
        progress=False,
    )
    if len(rows) != len(jobs) or any(not row for row in rows):
        raise RuntimeError("decision grid returned incomplete data")
    write_csv(output / "per_seed.csv", rows, COLS)
    timeline = _timeline_medians(rows)
    write_csv(
        output / "timeline.csv",
        timeline,
        ["mode", "loss", "outage", "time_s", "standoff_error_m"],
    )
    write_json(output / "summary.json", _summary(rows))
    from tools.d_make_decisions_figure import make_figures

    make_figures(output)
    runtime = time.perf_counter() - start
    cpu_runtime = float(sum(row["worker_cpu_s"] for row in rows))
    full_cpu = cpu_runtime * (600.0 / len(jobs))
    outputs = [
        "per_seed.csv",
        "timeline.csv",
        "summary.json",
        "decisions_outage.png",
        "decisions_traffic.png",
    ]
    write_manifest(
        output,
        schema="zephyr-d-decisions-manifest-1",
        argv=argv_or_default(argv),
        command="PYTHONPATH=. ./.venv/bin/python tools/d_run_decisions.py --output {out}",
        config={
            "pilot": pilot,
            "n_seeds": n,
            "n_agents": 4,
            "modes": list(MODES),
            "losses": list(losses),
            "outages": list(OUTAGES),
            "delay_s": 0.05,
            "rate_hz": 10.0,
            "duration_s": 20.0,
            "control_dt_s": 0.02,
            "log_every": 1,
            "heartbeat_timeout_s": 0.5,
            "ci_mode": "fast",
            "fusion": {"ground": "central", "onboard": "ci"},
            "decision": list(MODES),
            "target": "ct",
            "target_params": {"radius_m": 3.0, "speed_mps": 0.8},
            "noise_std_m": 0.05,
            "link": {
                "delay_s": 0.05,
                "loss_model": "independent",
                "share_rate_hz": 10.0,
            },
            "dropouts": "none",
            "process_cap": 4,
            "failure_rule": {
                "scored_from_s": 2.0,
                "scored_through_s": 20.0,
                "p95_m": 0.25,
                "max_m": 0.50,
                "cell_failure_fraction": 0.5,
            },
            "failure_queries_dt_s": 0.02,
            "amendment": (
                "20 s run with one ground link outage over [10, 15), "
                "with failure scoring through t = 20.0 s"
            ),
            "failure_thresholds_note": "Failure thresholds are research criteria, not safety limits.",
        },
        seeds=seeds,
        outputs=outputs,
        limits=[
            "Failure thresholds are research criteria, not safety limits.",
            "Agents are simulated kinematic point masses. Target motion and Gaussian "
            "sensor noise are simulated.",
            "Abstract links and assumed simulator constants for message sizes are used. "
            "nothing is measured.",
            "One circling target, idealized hold behavior, and the ground station sees "
            "the same link loss as agents.",
            "The run uses 20 s and a registered 5 s outage amendment. Processes are "
            "capped at 4.",
        ],
        runtime_s=runtime,
        n_processes=nproc,
        hashed_sources=_source_paths(),
    )
    print(
        f"pilot={pilot} seeds={n} runtime_s={runtime:.3f} "
        f"worker_cpu_s={cpu_runtime:.3f} "
        f"full_run_estimate_cpu_s={full_cpu:.3f} "
        f"full_run_estimate_wall_s={full_cpu / 4.0:.3f}"
    )

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run simulated decisions. Processes are capped at 4."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int)
    args = parser.parse_args(argv)
    run(
        args.output,
        pilot=args.pilot,
        processes=args.processes,
        argv=argv_or_default(argv),
    )
if __name__=='__main__': main()
