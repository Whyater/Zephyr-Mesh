"""EXP-029 Part B: set the lost-link and weak-link rules from mission data.

Usage (repo root)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "/Users/akashi0281/Desktop/Zephyr Mesh/Zephyr-Mesh/.venv/bin/python" tools/e_exp029_part_b.py --output runs/e-exp029-part-b --pilot

Everything in this runner is simulated.  The tag mission uses synthetic link
loss, a scripted total blackout, and literature-derived vehicle parameters.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim.missions.api import Scenario  # noqa: E402
from sim.missions.engine import run_mission  # noqa: E402
from sim.missions.presets import PRESET_NAMES, get_preset  # noqa: E402
from tests.reference_e.hand_checks import (  # noqa: E402
    coarse_grid_chain,
    false_alarms_per_hour_markov,
    heartbeat_false_alarms_per_hour_iid,
    per_slot_chain,
)
from tools import e_exp029_part_a as part_a  # noqa: E402
from tools.c_common import (  # noqa: E402
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
    run_jobs,
    seeds_for,
    write_rows,
)
from tools.horizon_mpl import apply_style, palette  # noqa: E402


DT_S = 0.01
COMMAND_RATE_HZ = 20.0
T_W_VALUES_S = (0.10, 0.15, 0.25)
T_VALUES_S = (0.15, 0.20, 0.30, 0.50, 1.0, 2.0)
MIN_T_ABOVE_T_W_S = 0.05
DEGRADED_ACTIONS = ("hold", "slow")
TARGET_CLASSES = ("turn", "figure8", "jink")
OUTAGE_LENGTHS_S = (0.5, 2.0, None)
PILOT_CLASSES = ("turn", "jink")
PILOT_PRESETS = ("clean", "burst_10_0.5s")
PILOT_T_VALUES_S = (0.30, 1.0)
N_SEEDS = 20
# Grid reduced before the full run (orchestrator, 2026-10-15; prereg Addendum C):
# the pre-registered Part B grid is 121,500 point-mass runs of 30 s, more than a
# day on a machine shared with two other runs.  Kept: the five presets used by
# every other Run E experiment, the full T axis at the rule's T_w = 0.15 s plus
# T_w 0.10 and 0.25 s at the planned T = 0.30 s, both degraded actions, every
# class and blackout length, and the first 20 seeds of the same list.
RUN_PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
RUN_POLICY_PAIRS = ((0.15, 0.20), (0.15, 0.30), (0.15, 0.50), (0.15, 1.0), (0.15, 2.0),
                    (0.10, 0.30), (0.25, 0.30))
GRID_DEVIATION = ("Grid reduced before the full run for compute (prereg Addendum C): presets "
                  "clean, iid_10, iid_30, burst_10_0.5s, burst_30_0.5s instead of all nine; "
                  "policies T_w 0.15 s with T 0.2, 0.3, 0.5, 1.0, 2.0 s plus T_w 0.10 and 0.25 s at "
                  "T 0.30 s (7 of 15 pairs); 20 seeds instead of 50.  Part A unnecessary-landing "
                  "rates still cover every preset and pair.")
MISSION_DURATION_S = 30.0
OUTAGE_START_RANGE_S = (8.0, 18.0)
OUTAGE_DRAW_COMPONENT = 29029
PHYSICS_T_VALUES_S = (0.30, 1.0)
PHYSICS_OUTAGE_LENGTHS_S = (2.0, None)
PHYSICS_SEEDS = 20
OUTPUTS = ("part_b.csv", "summary.json", "tradeoff.png")


def policies(*, actions: tuple[str, ...] = DEGRADED_ACTIONS,
             t_values: tuple[float, ...] = T_VALUES_S) -> list[dict]:
    """Return the Part A age-rule grid with each allowed degraded action.

    The Part A constraint leaves 6 + 5 + 4 = 15 threshold pairs.  Part B
    doubles them for hold and slow, giving 30 policy/action cells.
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
    """Build the point-mass jobs in deterministic grid order."""
    seed_list = _seed_values(seeds, 3 if pilot else N_SEEDS)
    classes = PILOT_CLASSES if pilot else TARGET_CLASSES
    lengths = (2.0, None) if pilot else OUTAGE_LENGTHS_S
    presets = PILOT_PRESETS if pilot else RUN_PRESETS
    t_values = PILOT_T_VALUES_S if pilot else T_VALUES_S
    policy_list = policies(t_values=t_values)
    if pilot:
        policy_list = [policy for policy in policy_list if policy["T_w_s"] == 0.15]
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
        policy = {"weak_rule": "age", "T_w_s": 0.15, "T_s": timeout,
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


def _read_part_a_rates(path: Path | None = None) -> dict[tuple[str, float, float], float]:
    source = Path("runs/e-exp029-part-a/part_a.csv") if path is None else Path(path)
    rates = {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["weak_rule"] != "age" or float(row["h_hz"]) != COMMAND_RATE_HZ:
                continue
            rates[(row["preset"], float(row["T_w_s"]), float(row["T_s"]))] = float(row["landings_per_hour"])
    return rates


def _group_key(row: dict) -> tuple:
    return (row["preset"], row["T_w_s"], row["T_s"], row["degraded_action"], row["adapter"])


def _metric_summary(rows: list[dict], part_a_rates: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    summaries = []
    for key, group in groups.items():
        preset, t_w, timeout, action, adapter = key
        a = [bool(row.get("violation_during_or_after_outage", False)) for row in group]
        b = [bool(row.get("nominal_at_age_0_33", False)) for row in group]
        late = [x or y for x, y in zip(a, b)]
        pre_metric = [row.get("landings_before_outage") for row in group]
        present = [float(value) for value in pre_metric if value is not None]
        exposure_h = sum(float(row["blackout_start_s"]) for row in group) / 3600.0
        mission_rate = None if len(present) != len(group) or exposure_h <= 0.0 else sum(present) / exposure_h
        summaries.append({
            "preset": preset,
            "T_w_s": t_w,
            "T_s": timeout,
            "degraded_action": action,
            "adapter": adapter,
            "n": len(group),
            "late": rate_summary(late),
            "separation_violation": rate_summary(a),
            "nominal_at_age_0_33": rate_summary(b),
            "outage_reaction_hold_s": _median(group, "outage_reaction_hold_s"),
            "outage_reaction_land_s": _median(group, "outage_reaction_land_s"),
            "blind_in_plane_s": _median(group, "blind_in_plane_s"),
            "landings": int(sum(int(row.get("landings", 0)) for row in group)),
            "holds": int(sum(int(row.get("holds", 0)) for row in group)),
            "separation_violations": int(sum(int(row.get("separation_violations", 0)) for row in group)),
            "part_a_unnecessary_landings_per_hour": part_a_rates.get((preset, float(t_w), float(timeout))),
            "mission_observed_unnecessary_landings_per_hour": mission_rate,
            "pre_outage_exposure_h": exposure_h,
        })
    return sorted(summaries, key=lambda item: (
        item["adapter"], PRESET_NAMES.index(item["preset"]), item["T_w_s"],
        item["T_s"], item["degraded_action"],
    ))


def _median(rows: list[dict], key: str) -> dict:
    return median_summary([row.get(key) for row in rows])


def _recommendations(summaries: list[dict], part_a_rates: dict) -> list[dict]:
    out = []
    for preset in PRESET_NAMES:
        valid = [t for t in T_VALUES_S if t >= 0.15 + MIN_T_ABOVE_T_W_S - 1e-9
                 and part_a_rates.get((preset, 0.15, t), math.inf) < 1.0]
        chosen = min(valid) if valid else None
        item = {"preset": preset, "T_w_s": 0.15, "smallest_T_s_under_1_per_hour": chosen}
        for timeout_name, timeout in (("recommended", chosen), ("planned_0.30", 0.30)):
            cells = [s for s in summaries if s["adapter"] == "point_mass" and s["preset"] == preset
                     and s["T_w_s"] == 0.15 and s["T_s"] == timeout]
            item[timeout_name] = {
                action: next((cell["late"] for cell in cells if cell["degraded_action"] == action), None)
                for action in DEGRADED_ACTIONS
            } if timeout is not None else None
        out.append(item)
    return out


def _hand_checks(rows: list[dict]) -> dict:
    # The mission-observed check uses the planned 20 Hz age policy at T_w=0.15,
    # T=0.30 and hold, matching the Part A row used for the planned comparison.
    checks = {}
    for preset in ("burst_10_0.5s", "iid_30"):
        candidates = [row for row in rows if row["preset"] == preset and row["adapter"] == "point_mass"
                      and float(row.get("T_w_s", 0.0)) == 0.15
                      and float(row.get("T_s", 0.0)) == 0.30
                      and row.get("degraded_action") == "hold"]
        if not candidates or any(row.get("landings_before_outage") is None for row in candidates):
            checks[preset] = {"status": "missing_metrics", "count": None,
                              "T_w_s": 0.15, "T_s": 0.30, "degraded_action": "hold"}
            continue
        selected = candidates
        count = int(sum(int(row["landings_before_outage"]) for row in selected))
        exposure_h = sum(float(row["blackout_start_s"]) for row in selected) / 3600.0
        preset_info = get_preset(preset)
        if preset_info.loss_model == "independent":
            expected_rate = heartbeat_false_alarms_per_hour_iid(
                preset_info.loss_probability, COMMAND_RATE_HZ, 0.30,
            )
        else:
            q0, r0 = per_slot_chain(preset_info.loss_probability, preset_info.burst_mean_s, 100.0)
            q, r = coarse_grid_chain(q0, r0, 5)
            expected_rate = false_alarms_per_hour_markov(q, r, COMMAND_RATE_HZ, 0.30)
        expected = expected_rate * exposure_h
        z = None if expected <= 0.0 else (count - expected) / math.sqrt(expected)
        checks[preset] = {
            "status": "computed",
            "T_w_s": 0.15,
            "T_s": 0.30,
            "degraded_action": "hold",
            "count": count,
            "exposure_h": exposure_h,
            "observed_rate_per_hour": count / exposure_h if exposure_h else None,
            "expected_rate_per_hour": expected_rate,
            "expected_count": expected,
            "z": z,
        }
    return checks


def _make_figure(summaries: list[dict], output: Path) -> None:
    apply_style("light")
    colors = palette("light")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for action, marker in (("hold", "o"), ("slow", "^")):
        for preset in PRESET_NAMES:
            cells = [s for s in summaries if s["adapter"] == "point_mass" and s["preset"] == preset
                     and s["degraded_action"] == action]
            cells = sorted(cells, key=lambda cell: cell["T_s"])
            if not cells:
                continue
            xs = [max(float(cell["part_a_unnecessary_landings_per_hour"] or 0.0), 1e-6) for cell in cells]
            ys = [cell["late"]["rate"] for cell in cells]
            ax.plot(xs, ys, marker=marker, linestyle="-", linewidth=0.8, markersize=4,
                    color=colors.get("text", "#333333"), alpha=0.55,
                    label=f"{preset}, {action}")
            for x, y, cell in zip(xs, ys, cells):
                ax.annotate(f"{cell['T_s']:g}", (x, y), fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("unnecessary landings per hour from Part A (log scale)")
    ax.set_ylabel("late-reaction rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Simulated: lost-link policy tradeoff by preset")
    ax.grid(True, which="major", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "tradeoff.png", dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def run(output: Path, argv: list[str], *, pilot: bool = False,
        processes: int | None = None, jobs: list[dict] | None = None,
        duration_s: float = MISSION_DURATION_S) -> dict:
    start = time.perf_counter()
    output = prepare_output(output)
    if jobs is None:
        jobs = build_jobs(pilot=pilot, duration_s=duration_s)
    proc = process_count(processes)
    rows = run_jobs(jobs, n_processes=proc, worker=_run_job, chunksize=1)
    part_a_rates = _read_part_a_rates()
    summaries = _metric_summary(rows, part_a_rates)
    missing = []
    if not rows or any(row.get("landings_before_outage") is None for row in rows):
        missing.append("landings_before_outage")
    summary = {
        "experiment": "EXP-029 Part B",
        "status": "simulated",
        "pilot": bool(pilot),
        "mission_duration_s": float(duration_s),
        "outage_draw": "np.random.default_rng(np.random.SeedSequence([seed, 29029])).uniform(8, 18)",
        "outage_streams": None,
        "presets": list(PILOT_PRESETS if pilot else RUN_PRESETS),
        "preset_scope_reading": "Part B does not restrict presets; the run uses five of nine for compute (see deviations).",
        "grid": {
            "point_mass_jobs": len(build_point_jobs()),
            "physics_jobs": len(build_physics_jobs()),
            "pre_registered_point_mass_jobs": 121500,
            "pilot_point_mass_jobs": len(build_point_jobs(pilot=True)),
            "pilot_physics_jobs": len(build_physics_jobs(pilot=True)),
        },
        "summaries": summaries,
        "recommendation": _recommendations(summaries, part_a_rates),
        "hand_check": _hand_checks(rows),
        "missing_metrics": missing,
        "deviations": [GRID_DEVIATION],
        "projection": None,
    }
    runtime = time.perf_counter() - start
    if pilot and jobs:
        summary["projection"] = {
            "pilot_jobs": len(jobs),
            "full_grid_jobs": len(build_point_jobs()) + len(build_physics_jobs()),
            "method": "multiply the printed pilot wall time per run by full_grid_jobs / 2",
        }
    write_rows(output / "part_b.csv", rows, first_columns=(
        "target_class", "outage_length_s", "preset", "weak_rule", "T_w_s", "T_s",
        "degraded_action", "seed", "adapter",
    ))
    _make_figure(summaries, output)
    write_json(output / "summary.json", summary)
    print(f"runtime {runtime:.3f} s for {len(jobs)} runs on {proc} processes")
    if pilot and jobs:
        per_run = runtime / len(jobs)
        full_jobs = len(build_point_jobs()) + len(build_physics_jobs())
        projected = per_run * full_jobs / 2.0
        print(f"projection {projected:.1f} s for {full_jobs} full-grid runs at 2 processes ({per_run:.4f} s/run)")
    config = {
        "pilot": bool(pilot),
        "duration_s": float(duration_s),
        "target_classes": list(PILOT_CLASSES if pilot else TARGET_CLASSES),
        "outage_lengths_s": ["permanent" if v is None else v for v in (OUTAGE_LENGTHS_S if not pilot else (2.0, None))],
        "presets": list(PRESET_NAMES),
        "run_presets": list(PILOT_PRESETS if pilot else RUN_PRESETS),
        "T_w_s": list(T_W_VALUES_S),
        "T_s": list(T_VALUES_S),
        "degraded_actions": list(DEGRADED_ACTIONS),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "total_blackout_streams": None,
        "outage_draw_component": OUTAGE_DRAW_COMPONENT,
    }
    return write_manifest(
        output, schema="zephyr-run-e-manifest-1", argv=argv,
        command="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/akashi0281/Desktop/Zephyr\\ Mesh/Zephyr-Mesh/.venv/bin/python tools/e_exp029_part_b.py --output {out}",
        config=config, seeds=sorted({int(row["seed"]) for row in rows}), outputs=list(OUTPUTS),
        limits=list(MISSION_LIMITS) + [
            "Part B uses the tag mission in hold mode with a 0.75 m follower standoff.",
            "A total blackout is expressed with Scenario.outages streams=None, which applies to every mission stream.",
            "Mission duration is 30 s, covering the latest possible blackout, three seconds, and landing time.",
            "Part A unnecessary-landing rates are read from runs/e-exp029-part-a/part_a.csv and never regenerated here.",
        ], runtime_s=runtime, n_processes=proc,
        hashed_sources=list(MISSION_SOURCES) + ["sim/missions/tag.py", "tools/e_exp029_part_a.py", "tools/e_exp029_part_b.py", "tests/reference_e/hand_checks.py"],
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=process_count())
    args = parser.parse_args(argv)
    run(args.output, argv, pilot=args.pilot, processes=args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
