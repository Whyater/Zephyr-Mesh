"""Pilot: the S6 tracker inside a closed loop with a simulated drone chasing its estimate.

The S6 reruns score the tracker alone (estimate against truth). This pilot puts the same tracker
in a loop: camera packets -> ground tracker -> goal -> setpoint link -> PhysicsAdapter drone, and
scores how far the drone is from where it should be. Everything is simulated with synthetic
inputs; nothing here is a measurement or flight performance, and nothing is registered.

Pieces reused unchanged:

- camera packets: sim.s6_recovery.deliver (sim/sensor.py noise, sim/link.py delay and loss), so
  with the same seed they are the S6 packets;
- ground tracker: the S6 recovery arm rule (fuse at capture time, drop a packet older than one
  already fused, predict forward), on sim/tracker.py with the registered tuning;
- setpoint link: sim/link.SimulatedLink at 50 Hz;
- drone: sim.missions.physics_adapter.PhysicsAdapter (Crazyflie-class v2 profile, drag off,
  onboard feedback, so the drone knows its own true state), stepped at 100 Hz, with the default
  FailsafeConfig and the link age that sim/missions/stack.py feeds it (time since the newest
  setpoint arrived).

Pilot choices (not existing modules):

- goal = tracker estimate predicted lead_s ahead + a fixed offset of (0, -1, +1) m, so the drone
  follows 1 m to the side and 1 m above (non-contact). lead_s defaults to 0.75 s, the PD
  position loop's steady lag per m/s (Kd / Kp = 1.5 / 2 on the horizontal axes);
- the drone applies the newest setpoint by sequence number and ignores an older one that
  arrives late;
- a 1 s pre-roll before t = 0: the ground sends hover setpoints at the start point over the
  same link, so the failsafe sees a running link when the target starts moving.

Wirings:

- w1 (S6 placement): the cell's whole delay, jitter and loss sit on the camera path, as in S6;
  the setpoint link is perfect;
- w2 (architecture placement: camera to laptop, setpoints to the drone over ESP-NOW): the
  camera path carries only the vision delay and jitter (no loss); the radio delay, jitter and
  loss sit on the setpoint link. Variant w2_comp adds the mean radio delay to lead_s. Variant
  w2_truth builds the goal from the true target state instead of the tracker, which leaves only
  the controller and the radio.

Goal rules (goal_rule):

- "velocity" (the default, unchanged): the tracker's own constant-velocity prediction lead_s
  ahead, which runs along the tangent; from the true state, position + lead_s x velocity;
- "lead" (option): the target's predicted position lead_s ahead along its curve. From the
  tracker: a constant-turn-rate prediction from the last fused position and velocity, with the
  turn rate estimated from how fast the tracker's velocity heading turns (estimate_turn_rate).
  From the true state: the true position lead_s ahead. The --lead runs use
  lead_s = setpoint-path radio delay + CONTROLLER_LAG_S. Variants w2_truth_lead, w1_lead and
  w2_lead run beside the old four.

Scoring: the S6 fail rule (p95 > 0.25 m or max > 0.50 m after the 2 s burn-in) on the follow
error, the distance from the drone to target + offset. A secondary window (t >= 6 s, chosen after
the pre-registration) drops the start transient: the target jumps from rest to 1 m/s at t = 0.

    PYTHONPATH=. .venv/Scripts/python.exe tools/pilot_full_chain.py --output DIR [--seeds 5] [--lead]
    PYTHONPATH=. .venv/Scripts/python.exe tools/pilot_full_chain.py --output DIR --delay-sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.adapters import Capabilities
from sim.link import LinkConfig, SimulatedLink
from sim.missions.physics_adapter import PhysicsAdapter
from sim.s6_recovery import (
    BURN_IN_TOLERANCE_S,
    MOTIONS,
    TIME_TOLERANCE_S,
    RecoveryConfig,
    _burst_start_probability,
    _new_tracker,
    deliver,
    run_arms,
    seed_list,
    truth_position,
)

SCHEMA = "zephyr-pilot-full-chain-1"
DT_S = 0.01
GROUND_EVERY = 2  # ground tick every 2 drone steps: 50 Hz, the S6 query rate
PRE_ROLL_S = 1.0
LEAD_S = 0.75
OFFSET_M = (0.0, -1.0, 1.0)
REGISTERED_TRACKER_STD_M = 0.05
MAX_SPEED_MPS = 1.5  # PhysicsAdapter default capabilities

# Lead goal rule. The horizontal position loop is x'' = Kp (g - x) - Kd x' with Kp = 2, Kd = 1.5.
# Chasing a straight line g = v t, the steady drone is x = v (t - T): x'' = 0 and x' = v give
# 0 = Kp v T - Kd v, so T = Kd / Kp = 0.75 s. Aiming that far ahead cancels the lag on a line.
# On a curve the lag depends on the turn rate (0.98 s at 1 rad/s), so one fixed constant is a
# compromise; this one is exact for straight lines.
CONTROLLER_LAG_S = 0.75
TURN_RATE_WINDOW_S = 0.5  # heading change measured over about this long
TURN_RATE_MIN_SPEED_MPS = 0.2  # below this the heading means nothing: no turn
TURN_RATE_MAX_RADPS = 2.0
GOAL_RULES = ("velocity", "lead")
LEAD_VARIANTS = ("w2_truth_lead", "w1_lead", "w2_lead")
SWEEP_DELAYS_S = (0.0, 0.1, 0.2, 0.4)
SWEEP_VARIANTS = ("w1", "w2_comp", "w2_truth_lead", "w1_lead", "w2_lead")


@dataclass(frozen=True)
class Cell:
    name: str
    vision_delay_s: float = 0.0
    vision_jitter_s: float = 0.0
    radio_delay_s: float = 0.0
    radio_jitter_s: float = 0.0
    loss_probability: float = 0.0
    loss_model: str = "independent"
    burst_mean_length_packets: float = 5.0
    noise_std_m: float = 0.0

    @property
    def total_delay_s(self) -> float:
        return self.vision_delay_s + self.radio_delay_s

    @property
    def total_jitter_s(self) -> float:
        return math.hypot(self.vision_jitter_s, self.radio_jitter_s)


# rehearsal: the synthetic values tools/rehearse_measured_inputs.py feeds its generators
# (camera 120 ms + 15 ms processing, 8 ms jitter; EchoParams 2.5 ms, 0.8 ms, burst start 0.02 and
# end 0.3, so a 0.02 / 0.32 = 6.25 % lost fraction in runs of 1 / 0.3 = 3.3 packets; 0.03 m noise).
CELLS = {
    "zero": Cell("zero"),
    "rehearsal": Cell("rehearsal", vision_delay_s=0.135, vision_jitter_s=0.008, radio_delay_s=0.0025,
                      radio_jitter_s=0.0008, loss_probability=0.02 / 0.32, loss_model="burst",
                      burst_mean_length_packets=1 / 0.3, noise_std_m=0.03),
    "s6_pass": Cell("s6_pass", radio_delay_s=0.10, loss_probability=0.10, noise_std_m=0.05),
    "s6_fail": Cell("s6_fail", radio_delay_s=0.20, loss_probability=0.30, noise_std_m=0.05),
}
VARIANTS = ("w2_truth", "w1", "w2", "w2_comp")
CHECKPOINT = "checkpoint.jsonl"
STEADY_FROM_S = 6.0  # secondary window, chosen after the pre-registration (post hoc)


def _link_config(delay_s: float, jitter_s: float, loss_probability: float, loss_model: str,
                 burst_mean_length_packets: float) -> LinkConfig:
    """The same LinkConfig sim.s6_recovery.deliver builds for these settings."""
    if loss_model == "burst":
        end = 1.0 / float(burst_mean_length_packets)
        return LinkConfig(delay_s=delay_s, delay_jitter_s=jitter_s, loss_model="burst",
                          burst_start_probability=_burst_start_probability(loss_probability, end),
                          burst_end_probability=end)
    return LinkConfig(delay_s=delay_s, delay_jitter_s=jitter_s, loss_model="independent",
                      loss_probability=loss_probability)


def _truth_velocity(motion: str, t: float, config: RecoveryConfig, h: float = 1e-5) -> np.ndarray:
    """True target velocity by a central difference of truth_position."""
    ahead, behind = truth_position(motion, [t + h, max(t - h, 0.0)], config)
    return (ahead - behind) / (t + h - max(t - h, 0.0))


def _setpoint_seed(seed: int) -> int:
    """A stream separate from deliver's link and sensor seeds (SeedSequence(seed) state 0 and 1)."""
    return int(np.random.SeedSequence([int(seed), 1]).generate_state(1)[0])


def turn_predict(position, velocity, turn_rate_radps: float, horizon_s: float) -> np.ndarray:
    """Position after horizon_s at constant speed and constant turn rate (left positive).

    The horizontal velocity turns at w, v(s) = cos(w s) v + sin(w s) Jv, where Jv is v turned
    90 degrees left. Integrating over 0..h gives (sin(w h) v + (1 - cos(w h)) Jv) / w. The
    vertical part keeps its velocity. With w = 0 this is the straight line p + v h.
    """
    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    w, h = float(turn_rate_radps), float(horizon_s)
    if abs(w * h) < 1e-9:
        return p + v * h
    flat = np.array([v[0], v[1], 0.0])
    left = np.array([-v[1], v[0], 0.0])
    return p + (math.sin(w * h) * flat + (1.0 - math.cos(w * h)) * left) / w + np.array([0.0, 0.0, v[2] * h])


def estimate_turn_rate(history: list[tuple[float, float, float]], window_s: float = TURN_RATE_WINDOW_S) -> float:
    """Turn rate (rad/s, left positive) from (fuse time, vx, vy) entries, oldest first.

    Heading now minus the heading at the entry closest to window_s earlier, wrapped to
    [-pi, pi), over the time between them. 0 when there is no older entry or either horizontal
    speed is under TURN_RATE_MIN_SPEED_MPS; clipped to +-TURN_RATE_MAX_RADPS.
    """
    if len(history) < 2:
        return 0.0
    t1, vx1, vy1 = history[-1]
    t0, vx0, vy0 = min(history[:-1], key=lambda entry: abs(entry[0] - (t1 - window_s)))
    if t1 <= t0 or min(math.hypot(vx0, vy0), math.hypot(vx1, vy1)) < TURN_RATE_MIN_SPEED_MPS:
        return 0.0
    turn = (math.atan2(vy1, vx1) - math.atan2(vy0, vx0) + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.clip(turn / (t1 - t0), -TURN_RATE_MAX_RADPS, TURN_RATE_MAX_RADPS))


@dataclass
class LoopResult:
    follow_p95_m: float
    follow_max_m: float
    follow_median_m: float
    follow_p95_steady_m: float
    failed_steady: bool
    estimate_p95_m: float
    estimate_max_m: float
    min_separation_m: float
    non_nominal_fraction: float
    landed: bool
    failed: bool


def run_loop(motion: str, cell: Cell, *, wiring: str = "w1", lead_s: float = LEAD_S, seed: int,
             config: RecoveryConfig | None = None, keep_trace: bool = False,
             max_speed_mps: float = MAX_SPEED_MPS, goal_source: str = "tracker",
             goal_rule: str = "velocity"):
    """Run one closed-loop episode. Returns a LoopResult (and the trace dict when keep_trace).

    max_speed_mps is the adapter's carrot speed cap (PhysicsAdapter's default 1.5 m/s); the
    tests lift it to check the linear PD hand value without the cap. goal_source "truth" builds
    the goal from the true position and velocity instead of the tracker (the controller's own
    floor); the camera packets and the tracker still run, so the estimate error is unchanged.
    goal_rule "velocity" (default) predicts lead_s ahead along the tangent; "lead" predicts
    lead_s ahead along the curve (see the module docstring).
    """
    if goal_source not in ("tracker", "truth"):
        raise ValueError("goal_source must be tracker or truth")
    if goal_rule not in GOAL_RULES:
        raise ValueError(f"goal_rule must be one of {GOAL_RULES}")
    config = config or RecoveryConfig()
    if wiring == "w1":
        cam = deliver(motion, delay_s=cell.total_delay_s, loss_probability=cell.loss_probability,
                      noise_std_m=cell.noise_std_m, seed=seed, config=config, loss_model=cell.loss_model,
                      jitter_s=cell.total_jitter_s, burst_mean_length_packets=cell.burst_mean_length_packets)
        radio = LinkConfig()
    elif wiring == "w2":
        cam = deliver(motion, delay_s=cell.vision_delay_s, loss_probability=0.0, noise_std_m=cell.noise_std_m,
                      seed=seed, config=config, jitter_s=cell.vision_jitter_s)
        radio = _link_config(cell.radio_delay_s, cell.radio_jitter_s, cell.loss_probability, cell.loss_model,
                             cell.burst_mean_length_packets)
    else:
        raise ValueError("wiring must be w1 or w2")
    offset = np.asarray(OFFSET_M, dtype=float)
    start = truth_position(motion, [0.0], config)[0]
    tracker = _new_tracker(motion, config, REGISTERED_TRACKER_STD_M)
    link = SimulatedLink(radio, seed=_setpoint_seed(seed))
    drone = PhysicsAdapter("chaser", initial_position_m=start + offset, dt_s=DT_S,
                           capabilities=Capabilities(max_speed_mps=max_speed_mps))
    drone.connect()

    n_pre = int(round(PRE_ROLL_S / DT_S))
    n_run = int(round(config.duration_s / DT_S))
    goals: dict[int, np.ndarray] = {}
    sent = 0
    newest = -1
    last_receive = 0.0  # link clock (t + PRE_ROLL_S); the adapter also starts with age 0
    cursor = 0
    last_capture = -np.inf
    est_times, est_err = [], []
    times, follow, separation, nominal, positions = [], [], [], [], []
    headings: list[tuple[float, float, float]] = []  # (fuse time, vx, vy) for the lead rule
    landed = False
    for i in range(-n_pre, n_run):
        t = i * DT_S
        clock = t + PRE_ROLL_S
        if i % GROUND_EVERY == 0:
            if i < 0:
                goal = start + offset
            else:
                while cursor < len(cam) and cam[cursor].receive_time_s <= t + TIME_TOLERANCE_S:
                    packet = cam[cursor]
                    cursor += 1
                    if packet.capture_time_s <= last_capture:
                        continue  # late: the recovery arm drops it
                    tracker.update(np.asarray(packet.measurement_m, dtype=float), packet.capture_time_s)
                    last_capture = packet.capture_time_s
                    if goal_rule == "lead":
                        fused = tracker.predict_at(tracker.timestamp_s).velocity_mps
                        headings.append((tracker.timestamp_s, fused[0], fused[1]))
                        while headings[0][0] < tracker.timestamp_s - 2.0 * TURN_RATE_WINDOW_S:
                            headings.pop(0)
                now = max(t, tracker.timestamp_s)
                estimate = np.asarray(tracker.predict_at(now).position_m)
                est_times.append(t)
                est_err.append(float(np.linalg.norm(estimate - truth_position(motion, [t], config)[0])))
                if goal_rule == "lead" and goal_source == "truth":
                    goal = truth_position(motion, [t + lead_s], config)[0] + offset
                elif goal_rule == "lead":
                    fused = tracker.predict_at(tracker.timestamp_s)
                    goal = turn_predict(fused.position_m, fused.velocity_mps, estimate_turn_rate(headings),
                                        t + lead_s - tracker.timestamp_s) + offset
                elif goal_source == "truth":
                    goal = truth_position(motion, [t], config)[0] + lead_s * _truth_velocity(motion, t, config) + offset
                else:
                    goal = np.asarray(tracker.predict_at(max(t + lead_s, tracker.timestamp_s)).position_m) + offset
            goals[sent] = goal
            link.send("ground", "drone", sent, clock)
            sent += 1
        for event in link.advance(clock + TIME_TOLERANCE_S):
            last_receive = max(last_receive, float(event.receive_time))
            if event.seq > newest:
                newest = event.seq
                drone.send_setpoint(goals[event.seq])
        state = drone.step(DT_S, link_age_s=max(0.0, clock - last_receive))
        landed = landed or bool(state.landed)
        if i >= 0:
            t_end = (i + 1) * DT_S
            target = truth_position(motion, [t_end], config)[0]
            position = np.asarray(state.position_m)
            times.append(t_end)
            positions.append(position)
            follow.append(float(np.linalg.norm(position - (target + offset))))
            separation.append(float(np.linalg.norm(position - target)))
            nominal.append(state.failsafe_state == "nominal")
    times_a = np.asarray(times)
    follow_a = np.asarray(follow)
    scored = times_a >= config.burn_in_s - BURN_IN_TOLERANCE_S
    est_t = np.asarray(est_times)
    est_a = np.asarray(est_err)
    est_scored = est_t >= config.burn_in_s - BURN_IN_TOLERANCE_S
    window = follow_a[scored]
    p95 = float(np.percentile(window, 95))
    maximum = float(np.max(window))
    steady = follow_a[times_a >= STEADY_FROM_S - BURN_IN_TOLERANCE_S]
    steady_p95 = float(np.percentile(steady, 95))
    result = LoopResult(
        follow_p95_m=p95, follow_max_m=maximum, follow_median_m=float(np.median(window)),
        follow_p95_steady_m=steady_p95,
        failed_steady=bool(steady_p95 > config.failure_p95_m or float(np.max(steady)) > config.failure_max_m),
        estimate_p95_m=float(np.percentile(est_a[est_scored], 95)),
        estimate_max_m=float(np.max(est_a[est_scored])),
        min_separation_m=float(np.min(np.asarray(separation)[scored])),
        non_nominal_fraction=float(1.0 - np.mean(np.asarray(nominal)[scored])),
        landed=landed,
        failed=bool(p95 > config.failure_p95_m or maximum > config.failure_max_m),
    )
    if keep_trace:
        return result, {"times_s": times_a, "follow_m": follow_a, "estimate_times_s": est_t, "estimate_m": est_a,
                        "position_m": np.asarray(positions)}
    return result


def tracker_only(motion: str, cell: Cell, seed: int, config: RecoveryConfig | None = None):
    """The S6 score for this cell: run_arms recovery on the combined-rerun placement."""
    config = config or RecoveryConfig()
    packets = deliver(motion, delay_s=cell.total_delay_s, loss_probability=cell.loss_probability,
                      noise_std_m=cell.noise_std_m, seed=seed, config=config, loss_model=cell.loss_model,
                      jitter_s=cell.total_jitter_s, burst_mean_length_packets=cell.burst_mean_length_packets)
    return run_arms(packets, motion, config, tracker_measurement_std_m=REGISTERED_TRACKER_STD_M)["recovery"]


def _label(failures: int, n: int) -> str:
    return "fails" if 2 * failures >= n else "passes"


def summarize(rows: list[dict], variants: tuple[str, ...] = VARIANTS, cells=None) -> list[dict]:
    """One entry per (cell, motion, variant): medians and ranges over seeds, fail counts, labels."""
    summary_cells = []
    for cell in (cells or CELLS):
        for motion in MOTIONS:
            for variant in ("tracker_only",) + tuple(variants):
                sub = [r for r in rows if r["cell"] == cell and r["motion"] == motion and r["variant"] == variant]
                if not sub:
                    continue  # e.g. the delay sweep has no tracker_only rows
                fails = sum(r["failed"] for r in sub)
                steady_fails = int(sum(r["failed_steady"] for r in sub))
                nn = [r["non_nominal_fraction"] for r in sub if r["non_nominal_fraction"] is not None]
                sep = [r["min_separation_m"] for r in sub if r["min_separation_m"] is not None]
                summary_cells.append({
                    "cell": cell, "motion": motion, "variant": variant, "n_seeds": len(sub),
                    "p95_median_m": float(np.median([r["p95_m"] for r in sub])),
                    "p95_min_m": float(np.min([r["p95_m"] for r in sub])),
                    "p95_max_m": float(np.max([r["p95_m"] for r in sub])),
                    "max_median_m": float(np.median([r["max_m"] for r in sub])),
                    "estimate_p95_median_m": float(np.median([r["estimate_p95_m"] for r in sub])),
                    "fail_count": int(fails), "label": _label(fails, len(sub)),
                    "steady_p95_median_m": float(np.median([r["p95_steady_m"] for r in sub])),
                    "steady_p95_min_m": float(np.min([r["p95_steady_m"] for r in sub])),
                    "steady_p95_max_m": float(np.max([r["p95_steady_m"] for r in sub])),
                    "steady_fail_count": steady_fails, "steady_label": _label(steady_fails, len(sub)),
                    "non_nominal_fraction_max": float(max(nn)) if nn else None,
                    "min_separation_min_m": float(min(sep)) if sep else None,
                    "landed_count": int(sum(bool(r["landed"]) for r in sub if r["landed"] is not None)),
                })
    return summary_cells


def read_rows(path: Path) -> list[dict]:
    """Read per_run.csv back into typed rows (empty cells are None)."""
    text = {"cell", "motion", "variant"}
    flags = {"failed", "failed_steady", "landed"}
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in text:
                    row[key] = value
                elif value == "":
                    row[key] = None
                elif key in flags:
                    row[key] = value == "True"
                elif key == "seed":
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def replot(folder: Path) -> dict:
    """Rebuild cells_summary and the figure of a finished pilot folder from its per_run.csv."""
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    variants = tuple(summary.get("variant_order", VARIANTS))
    summary["cells_summary"] = summarize(read_rows(folder / "per_run.csv"), variants)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(summary["cells_summary"], folder / "pilot_full_chain.png", variants)
    return summary


def variant_settings(variant: str, cell: Cell) -> dict:
    """run_loop keyword arguments for one variant in one cell."""
    radio = cell.radio_delay_s
    table = {
        "w2_truth": dict(wiring="w2", lead_s=LEAD_S, goal_source="truth", goal_rule="velocity"),
        "w1": dict(wiring="w1", lead_s=LEAD_S, goal_source="tracker", goal_rule="velocity"),
        "w2": dict(wiring="w2", lead_s=LEAD_S, goal_source="tracker", goal_rule="velocity"),
        "w2_comp": dict(wiring="w2", lead_s=LEAD_S + radio, goal_source="tracker", goal_rule="velocity"),
        # lead rule: aim the setpoint-path radio delay + the controller lag ahead along the curve
        # (in w1 the delay is on the camera path, which the tracker already predicts across)
        "w2_truth_lead": dict(wiring="w2", lead_s=CONTROLLER_LAG_S + radio, goal_source="truth", goal_rule="lead"),
        "w1_lead": dict(wiring="w1", lead_s=CONTROLLER_LAG_S, goal_source="tracker", goal_rule="lead"),
        "w2_lead": dict(wiring="w2", lead_s=CONTROLLER_LAG_S + radio, goal_source="tracker", goal_rule="lead"),
    }
    return table[variant]


def _loop_row(cell: Cell, motion: str, variant: str, seed: int) -> dict:
    r = run_loop(motion, cell, seed=seed, **variant_settings(variant, cell))
    return {"cell": cell.name, "motion": motion, "variant": variant, "seed": seed,
            "p95_m": r.follow_p95_m, "max_m": r.follow_max_m, "median_m": r.follow_median_m,
            "p95_steady_m": r.follow_p95_steady_m, "failed_steady": r.failed_steady,
            "estimate_p95_m": r.estimate_p95_m, "min_separation_m": r.min_separation_m,
            "non_nominal_fraction": r.non_nominal_fraction, "landed": r.landed,
            "failed": r.failed}


def run_pilot(output: Path, n_seeds: int = 5, with_lead: bool = False) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} exists and is not empty; pick a new folder")
    if "sim" in output.resolve().parts and "runs" in output.resolve().parts:
        raise SystemExit("this pilot never writes under sim/runs")
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    seeds = seed_list(n_seeds)
    variants = VARIANTS + (LEAD_VARIANTS if with_lead else ())
    # Checkpoint: one JSON line per finished (cell, motion) group. A rerun with the same output
    # resumes from it; the finished folder only appears when every group is done.
    checkpoint = partial / CHECKPOINT
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            record = json.loads(line)
            if record["seeds"] != seeds:
                raise SystemExit(f"{checkpoint} was made with other seeds; use a new output folder")
            if record.get("variants", list(VARIANTS)) != list(variants):
                raise SystemExit(f"{checkpoint} was made with other variants; use a new output folder")
            rows.extend(record["rows"])
            done.add((record["cell"], record["motion"]))
        print(f"resuming: {len(done)} groups already in {checkpoint}", flush=True)
    started = time.perf_counter()
    for cell in CELLS.values():
        for motion in MOTIONS:
            if (cell.name, motion) in done:
                continue
            group: list[dict] = []
            for seed in seeds:
                s6 = tracker_only(motion, cell, seed)
                late = s6.errors_m[s6.query_times_s >= STEADY_FROM_S - BURN_IN_TOLERANCE_S]
                late_p95 = float(np.percentile(late, 95))
                group.append({"cell": cell.name, "motion": motion, "variant": "tracker_only", "seed": seed,
                              "p95_m": s6.p95_error_m, "max_m": s6.max_error_m, "median_m": None,
                              "p95_steady_m": late_p95,
                              "failed_steady": bool(late_p95 > 0.25 or float(np.max(late)) > 0.5),
                              "estimate_p95_m": s6.p95_error_m, "min_separation_m": None,
                              "non_nominal_fraction": None, "landed": None, "failed": s6.failed})
                for variant in variants:
                    group.append(_loop_row(cell, motion, variant, seed))
            with open(checkpoint, "a") as handle:
                record = {"cell": cell.name, "motion": motion, "seeds": seeds, "rows": group}
                if with_lead:
                    record["variants"] = list(variants)
                handle.write(json.dumps(record) + "\n")
            rows.extend(group)
            print(f"done {cell.name} {motion} ({time.perf_counter() - started:.0f} s)", flush=True)
    summary_cells = summarize(rows, variants)
    summary = {
        "schema": SCHEMA, "status": "pilot; synthetic inputs; simulated only; not registered",
        "seeds": seeds, "cells": {name: asdict(c) for name, c in CELLS.items()},
        "settings": {"dt_s": DT_S, "ground_rate_hz": 1.0 / (DT_S * GROUND_EVERY), "pre_roll_s": PRE_ROLL_S,
                     "lead_s": LEAD_S, "offset_m": list(OFFSET_M),
                     "tracker_measurement_std_m": REGISTERED_TRACKER_STD_M,
                     "fail_rule": "p95 > 0.25 m or max > 0.50 m after the 2 s burn-in",
                     "label_rule": "fails when at least half the seeds fail",
                     "steady_window": f"secondary, post hoc: same fail rule on t >= {STEADY_FROM_S} s only"},
        "variants": {"tracker_only": "S6 recovery estimate error, whole delay and loss on the camera path",
                     "w2_truth": "w2 wiring but the goal comes from the true target state (controller and radio only)",
                     "w1": "closed loop, whole delay and loss on the camera path, perfect setpoint link",
                     "w2": "closed loop, vision delay on the camera path, radio delay and loss on the setpoints",
                     "w2_comp": "w2 with lead_s increased by the mean radio delay"},
        "limits": ["simulated drone (literature-derived profile, drag off), onboard feedback (true own state)",
                   "synthetic link and sensor; 5 seeds is a pilot, not a tolerance estimate",
                   "the goal law (lead plus fixed offset) is a pilot choice, not a registered controller",
                   "follow error is sampled every 0.01 s; the S6 estimate error every 0.02 s"],
        "cells_summary": summary_cells,
        "elapsed_s_last_invocation": time.perf_counter() - started,
    }
    if with_lead:
        summary["variant_order"] = list(variants)
        summary["settings"]["lead_rule"] = _LEAD_RULE_SETTINGS
        summary["variants"].update(_LEAD_VARIANT_TEXT)
    with open(partial / "per_run.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(summary_cells, partial / "pilot_full_chain.png", variants)
    checkpoint.unlink()  # per_run.csv now holds the same rows
    if output.exists():
        _move_into(partial, output)  # an empty folder was given
    else:
        partial.rename(output)
    return summary


def _move_into(partial: Path, output: Path) -> None:
    for item in partial.iterdir():
        item.rename(output / item.name)
    partial.rmdir()


_LEAD_RULE_SETTINGS = {
    "controller_lag_s": CONTROLLER_LAG_S, "turn_rate_window_s": TURN_RATE_WINDOW_S,
    "turn_rate_min_speed_mps": TURN_RATE_MIN_SPEED_MPS, "turn_rate_max_radps": TURN_RATE_MAX_RADPS,
    "lead_s": "setpoint-path radio delay + controller_lag_s (radio delay is 0 in w1)"}
_LEAD_VARIANT_TEXT = {
    "w2_truth_lead": "w2 wiring, lead rule from the true state: the true position lead_s ahead on the path",
    "w1_lead": "w1 wiring, lead rule: constant-turn prediction from the tracker, controller_lag_s ahead",
    "w2_lead": "w2 wiring, lead rule: constant-turn prediction, radio delay + controller_lag_s ahead"}


def _plot(cells: list[dict], path: Path, plot_variants: tuple[str, ...] = VARIANTS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style

    colors = apply_style("light")
    # Red and amber stay reserved for alerts, so no variant uses them (or the orange target colour).
    shades = {"tracker_only": colors["textSecondary"], "w2_truth": colors["lineStrong"], "w1": colors["accent"],
              "w2": colors["keepOut"], "w2_comp": colors["ok"],
              "w2_truth_lead": colors["lineStrong"], "w1_lead": colors["accent"], "w2_lead": colors["ok"]}
    names = {"tracker_only": "tracker only (the S6 score)", "w2_truth": "loop, goal from the true state",
             "w1": "loop, delay and loss on the camera path",
             "w2": "loop, radio on the setpoints", "w2_comp": "loop, radio on the setpoints, lead + radio delay",
             "w2_truth_lead": "lead rule (hatched), goal from the true state",
             "w1_lead": "lead rule, delay and loss on the camera path",
             "w2_lead": "lead rule, radio on the setpoints"}
    windows = (("", "pre-registered window, t >= 2 s"), ("steady_", "steady window, t >= 6 s (post hoc)"))
    top = 3.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
    variants = ("tracker_only",) + tuple(plot_variants)
    width = 0.8 / len(variants)
    for row, (prefix, window) in enumerate(windows):
        for col, motion in enumerate(MOTIONS):
            ax = axes[row, col]
            for k, variant in enumerate(variants):
                sub = [c for c in cells if c["motion"] == motion and c["variant"] == variant]
                x = np.arange(len(sub)) + (k - (len(variants) - 1) / 2) * width
                mid = np.array([c[f"{prefix}p95_median_m"] for c in sub])
                low = np.array([c[f"{prefix}p95_min_m"] for c in sub])
                high = np.array([c[f"{prefix}p95_max_m"] for c in sub])
                shown = np.minimum(mid, top)
                if variant.endswith("_lead"):
                    ax.bar(x, shown, width, color=shades[variant], label=names[variant], hatch="////",
                           edgecolor=colors["surface"], linewidth=0)
                else:
                    ax.bar(x, shown, width, color=shades[variant], label=names[variant])
                ax.errorbar(x, shown, yerr=[shown - np.minimum(low, top), np.minimum(high, top) - shown],
                            fmt="none", ecolor=colors["text"], capsize=2, lw=0.8)
                for xi, hi_value in zip(x, high):
                    if hi_value > top:  # a seed off the scale: print its value instead of stretching the axis
                        ax.annotate(f"{hi_value:.1f} m", (xi, top), xytext=(2, -3), textcoords="offset points",
                                    ha="left", va="top", rotation=90, fontsize=6, color=colors["text"])
            ax.axhline(0.25, color=colors["textSecondary"], ls="--", lw=1)
            ax.set_ylim(0, top * 1.08)
            ax.set_xticks(np.arange(len(CELLS)))
            ax.set_xticklabels(list(CELLS))
            ax.set_title(f"{motion.replace('_', ' ')}, {window}")
            if col == 0:
                ax.set_ylabel("p95 error (m): bar = median of 5 seeds,\nwhisker = seed range")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.008, "Simulated, synthetic inputs, 5 seeds, pilot (not registered). Dashed line: 0.25 m p95 "
             "fail threshold. A value printed under the top edge is a seed beyond the scale.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replot", action="store_true",
                        help="rebuild the summary table and figure of a finished --output folder from per_run.csv")
    args = parser.parse_args(argv)
    summary = replot(args.output) if args.replot else run_pilot(args.output, args.seeds)
    for c in summary["cells_summary"]:
        print(f"{c['cell']:10s} {c['motion']:18s} {c['variant']:13s} p95 {c['p95_median_m']:.3f} "
              f"[{c['p95_min_m']:.3f}, {c['p95_max_m']:.3f}] fails {c['fail_count']}/{c['n_seeds']} {c['label']}"
              f" | steady p95 {c['steady_p95_median_m']:.3f} fails {c['steady_fail_count']}/{c['n_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
