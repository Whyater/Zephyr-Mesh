"""Simulated tag-zone mission with delayed synthetic observations.

The follower remains separated from the moving target and scores a tag by
holding inside a distance zone.  Link loss, delays, and vehicle motion are all
simulated.  The slot and separation rules below are stand-ins until Run D.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.adapters import Geofence
from sim.missions.engine import (
    Mission,
    agent_frame,
    register_mission,
    stopping_buffer_m,
    target_path,
)
from sim.missions.estimate_feed import EstimateFeed


class _SoftMirroredTarget:
    """Wrap a placed jink target and softly reflect it inside the gym box.

    Folding the two horizontal coordinates and reflecting the corresponding
    velocity component keeps the target in [1, 15] x [1, 9].  This wrapper is
    a Run E change made to keep the generated jink in the gym.
    """

    def __init__(self, placed):
        self.base = placed
        self.altitude_m = placed.altitude_m

    @staticmethod
    def _fold(values, lo, hi):
        length = hi - lo
        period = 2.0 * length
        u = np.mod(values - lo, period)
        reflected = np.where(u <= length, lo + u, hi - (u - length))
        sign = np.where(u <= length, 1.0, -1.0)
        return reflected, sign

    def position(self, times_s):
        values = np.asarray(self.base.position(times_s), dtype=float).copy()
        x, _ = self._fold(values[:, 0], 1.0, 15.0)
        y, _ = self._fold(values[:, 1], 1.0, 9.0)
        values[:, 0] = x
        values[:, 1] = y
        return values

    def velocity(self, times_s):
        values = np.asarray(self.base.position(times_s), dtype=float)
        velocity = np.asarray(self.base.velocity(times_s), dtype=float).copy()
        _, sx = self._fold(values[:, 0], 1.0, 15.0)
        _, sy = self._fold(values[:, 1], 1.0, 9.0)
        velocity[:, 0] *= sx
        velocity[:, 1] *= sy
        return velocity

    @property
    def start_position(self):
        return self.position([0.0])[0]


class _StationaryTarget:
    """Constant target used for the registered zero-speed separation check."""

    def __init__(self, position):
        self._position = np.asarray(position, dtype=float).copy()

    def position(self, times_s):
        times = np.asarray(times_s, dtype=float)
        if times.ndim == 0:
            times = times.reshape(1)
        return np.repeat(self._position.reshape(1, 3), times.size, axis=0)

    def velocity(self, times_s):
        times = np.asarray(times_s, dtype=float)
        if times.ndim == 0:
            times = times.reshape(1)
        return np.zeros((times.size, 3), dtype=float)

    @property
    def start_position(self):
        return self._position.copy()


def _unit_xy(vector, fallback=(1.0, 0.0)):
    value = np.asarray(vector, dtype=float).copy()
    value[2] = 0.0
    norm = float(np.linalg.norm(value[:2]))
    if norm <= 1e-12:
        value = np.asarray((*fallback, 0.0), dtype=float)
        norm = float(np.linalg.norm(value[:2]))
    return value / norm


@register_mission
class TagMission(Mission):
    """Simulated tag-zone mission for one to three separated followers."""

    name = "tag"
    rng_names = ("sensor", "target")
    limits = (
        "Simulated only: target paths, radio delays, and link losses are synthetic.",
        "Stopping-buffer separation guard: stand-in until Run D.",
        "Pairwise follower guard and slot assignment: stand-in until Run D.",
        "The jink path is softly mirrored inside [1,15] x [1,9], a Run E change.",
    )
    defaults = {
        "target_class": "straight",
        "arm": "recovery",
        "n_followers": 1,
        "initial_gap_m": 4.0,
        "zone_radius_m": 1.0,
        "min_separation_m": 0.5,
        "hold_s": 2.0,
        "follower_speed_mps": 1.5,
        "target_speed_mps": 0.5,
        "standoff_m": 0.75,
        "lead_s": 0.75,
        "follower_guard_m": 0.6,
        "dwell_s": 1.0,
        "mode": "tag",
        "separation_guard": True,
    }

    def validate_params(self, params: dict) -> dict:
        result = super().validate_params(params)
        if result["target_class"] not in {"straight", "turn", "figure8", "jink"}:
            raise ValueError("target_class must be straight, turn, figure8, or jink")
        if result["arm"] not in {"naive", "recovery", "ct", "imm"}:
            raise ValueError("arm must be naive, recovery, ct, or imm")
        n = result["n_followers"]
        if isinstance(n, bool) or int(n) != n or not 1 <= int(n) <= 3:
            raise ValueError("n_followers must be an integer from 1 to 3")
        result["n_followers"] = int(n)
        if result["mode"] not in {"tag", "hold"}:
            raise ValueError("mode must be tag or hold")
        if not isinstance(result["separation_guard"], bool):
            raise ValueError("separation_guard must be bool")
        for name in (
            "initial_gap_m", "zone_radius_m", "min_separation_m", "hold_s",
            "follower_speed_mps", "target_speed_mps", "standoff_m", "lead_s",
            "follower_guard_m", "dwell_s",
        ):
            value = float(result[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
        if result["follower_speed_mps"] <= 0.0:
            raise ValueError("follower_speed_mps must be positive")
        if result["hold_s"] <= 0.0:
            raise ValueError("hold_s must be positive")
        return result

    def setup(self, ctx):
        p = ctx.params
        origin = {
            "straight": (4.0, 5.0),
            "turn": (8.0, 3.5),
            "figure8": (8.0, 5.0),
            "jink": (6.0, 5.0),
        }[p["target_class"]]
        path = target_path(
            p["target_class"], ctx.seed_for("target"), origin=origin,
            heading_rad=0.0, altitude_m=1.5, speed_mps=p["target_speed_mps"],
        ) if p["target_speed_mps"] > 0.0 else _StationaryTarget((*origin, 1.5))
        if p["target_class"] == "jink":
            path = _SoftMirroredTarget(path)
        ctx.state["target"] = path
        ctx.state["feed"] = EstimateFeed(
            p["arm"], target=path, sensor_noise_std_m=0.05, rate_hz=10.0,
            link=ctx.link(
                "meas:obs", delay_s=ctx.scenario.measurement_delay_s,
                jitter_s=ctx.scenario.measurement_jitter_s,
            ), seed=ctx.seed_for("sensor"), duration_s=ctx.scenario.duration_s,
        )
        initial_target = path.start_position
        initial_heading = _unit_xy(path.velocity([0.0])[0])
        initial_gap = p["standoff_m"] if p["mode"] == "hold" else p["initial_gap_m"]
        side = np.array([[-0.5], [0.5]]) if p["n_followers"] == 2 else np.array([[-1.0], [0.0], [1.0]])
        if p["n_followers"] == 1:
            side_offsets = [0.0]
        else:
            side_offsets = [float(v) for v in side[:, 0]]
        geofence = Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)
        ids = []
        for index in range(p["n_followers"]):
            follower_id = f"f{index + 1}"
            lateral = np.array([-initial_heading[1], initial_heading[0], 0.0]) * side_offsets[index]
            start = initial_target - initial_gap * initial_heading + lateral
            ctx.add_drone(
                follower_id, initial_position_m=start, geofence=geofence,
                max_speed_mps=p["follower_speed_mps"],
            )
            ids.append(follower_id)
        ctx.start_log(ids)
        ctx.state.update({
            "ids": ids,
            "zone": {agent: False for agent in ids},
            "hold_started": {agent: None for agent in ids},
            "separation_active": {agent: False for agent in ids},
            "tag_complete": False,
            "tag_time": None,
            "min_separation": math.inf,
            "separation_violations": 0,
            "separation_episodes": 0,
            "follower_min": math.inf,
            "guard_activations": 0,
            "max_estimate_age": 0.0,
            "time_degraded": 0.0,
            "nominal_at_age_0_33": False,
            "violation_outage": False,
            "blind_in_plane": 0.0,
            "last_command_before_outage": None,
            "slot_assignment": {},
            "last_assignment": -math.inf,
            "last_heading": initial_heading,
        })
        self._heading = initial_heading

    def _slots(self, estimate, velocity, n, standoff):
        """Return slot geometry using a stand-in until Run D."""
        heading = _unit_xy(velocity, fallback=tuple(self._heading[:2]))
        self._heading = heading
        angles = [180.0] if n == 1 else ([180.0, 130.0] if n == 2 else [180.0, 130.0, 230.0])
        slots = []
        for angle in angles:
            radians = math.radians(angle)
            direction = np.array([
                math.cos(radians) * heading[0] - math.sin(radians) * heading[1],
                math.sin(radians) * heading[0] + math.cos(radians) * heading[1],
                0.0,
            ])
            slots.append(np.asarray(estimate, dtype=float) + standoff * direction)
        return slots

    def _command_goals(self, ctx, t, estimate, velocity, estimate_age):
        """Build telemetry-only goals with guards that stand in until Run D."""
        p = ctx.params
        ids = ctx.state["ids"]
        telemetry = {}
        telemetry_age = {}
        for agent in ids:
            state, age = ctx.drones[agent].telemetry_view(t)
            telemetry[agent] = None if state is None else np.asarray(state.position_m, dtype=float)
            telemetry_age[agent] = float(age)
        if len(ids) > 1 and (t - ctx.state["last_assignment"] >= p["dwell_s"] or not ctx.state["slot_assignment"]):
            slots = self._slots(estimate, velocity, len(ids), p["standoff_m"])
            costs = np.zeros((len(ids), len(slots)), dtype=float)
            for row, agent in enumerate(ids):
                position = telemetry[agent]
                for col, slot in enumerate(slots):
                    costs[row, col] = 1e6 + col if position is None else float(np.linalg.norm(position - slot))
            rows, cols = linear_sum_assignment(costs)
            ctx.state["slot_assignment"] = {ids[int(row)]: int(col) for row, col in zip(rows, cols)}
            ctx.state["last_assignment"] = t
        slots = self._slots(estimate, velocity, len(ids), p["standoff_m"])
        heading = _unit_xy(velocity, fallback=tuple(ctx.state["last_heading"][:2]))
        goals = {}
        for agent in ids:
            follower_position = telemetry[agent]
            if len(ids) > 1:
                goal = slots[ctx.state["slot_assignment"].get(agent, ids.index(agent))].copy()
            else:
                if follower_position is None:
                    direction = heading
                else:
                    direction = _unit_xy(np.asarray(estimate) - follower_position, fallback=tuple(heading[:2]))
                goal = np.asarray(estimate, dtype=float) + p["lead_s"] * np.asarray(velocity, dtype=float) - p["standoff_m"] * direction
            if p["separation_guard"] and follower_position is not None:
                direction = _unit_xy(np.asarray(estimate) - follower_position, fallback=tuple(heading[:2]))
                distance = float(np.linalg.norm(np.asarray(estimate) - follower_position))
                age = estimate_age + telemetry_age[agent] if math.isfinite(estimate_age) and math.isfinite(telemetry_age[agent]) else 0.0
                buffer = stopping_buffer_m(p["follower_speed_mps"], age)
                if distance - buffer < p["min_separation_m"]:
                    goal = np.asarray(estimate, dtype=float) - p["standoff_m"] * direction
            goals[agent] = goal
        if p["follower_guard_m"] > 0.0:
            for left_index, left in enumerate(ids):
                if telemetry[left] is None:
                    continue
                for right in ids[left_index + 1:]:
                    if telemetry[right] is None:
                        continue
                    separation = telemetry[right] - telemetry[left]
                    distance = float(np.linalg.norm(separation))
                    if distance < p["follower_guard_m"]:
                        direction = separation / distance if distance > 1e-12 else np.array([1.0, 0.0, 0.0])
                        goals[right] = goals[right] + direction * (p["follower_guard_m"] - distance)
                        ctx.state["guard_activations"] += 1
        return goals

    def tick(self, ctx, i, t):
        p = ctx.params
        target = ctx.state["target"]
        feed = ctx.state["feed"]
        estimate, velocity, estimate_age = feed.estimate_at(t)
        if math.isfinite(estimate_age):
            ctx.state["max_estimate_age"] = max(ctx.state["max_estimate_age"], estimate_age)
        goals = self._command_goals(ctx, t, estimate, velocity, estimate_age)
        for agent in ctx.state["ids"]:
            ctx.drones[agent].set_goal(goals[agent])
        states = {agent: ctx.drones[agent].tick(i) for agent in ctx.state["ids"]}
        target_position = np.asarray(target.position([t])[0], dtype=float)
        outage_window = next((outage for outage in ctx.outages if outage.start_s <= t < outage.end_s), None)
        if outage_window is None:
            for agent in ctx.state["ids"]:
                age = states[agent].link_age_s
                if age is not None and math.isfinite(age):
                    received_at = t + ctx.dt - age
                    previous = ctx.state["last_command_before_outage"]
                    ctx.state["last_command_before_outage"] = received_at if previous is None else max(previous, received_at)
        for agent in ctx.state["ids"]:
            state = states[agent]
            distance = float(np.linalg.norm(np.asarray(state.position_m) - target_position))
            ctx.state["min_separation"] = min(ctx.state["min_separation"], distance)
            in_zone = distance <= p["zone_radius_m"]
            if in_zone and not ctx.state["zone"][agent]:
                ctx.event(t, "tag_zone_enter", agent, {"distance_m": distance})
            if not in_zone and ctx.state["zone"][agent]:
                ctx.event(t, "tag_zone_exit", agent, {"distance_m": distance})
            ctx.state["zone"][agent] = in_zone
            violates = distance < p["min_separation_m"]
            if violates:
                ctx.state["separation_violations"] += 1
                if not ctx.state["separation_active"][agent]:
                    ctx.state["separation_episodes"] += 1
                    ctx.event(t, "separation_violation", agent, {"distance_m": distance})
                ctx.state["separation_active"][agent] = True
            else:
                ctx.state["separation_active"][agent] = False
            if in_zone and not violates:
                if ctx.state["hold_started"][agent] is None:
                    ctx.state["hold_started"][agent] = t
                if not ctx.state["tag_complete"] and t - ctx.state["hold_started"][agent] + ctx.dt >= p["hold_s"]:
                    ctx.state["tag_complete"] = True
                    ctx.state["tag_time"] = t
                    ctx.event(t, "tag_complete", agent, {"hold_s": p["hold_s"]})
            else:
                ctx.state["hold_started"][agent] = None
            if state.failsafe_state == "degraded":
                ctx.state["time_degraded"] += ctx.dt
                if outage_window is not None and abs(float(state.position_m[2]) - target_position[2]) <= 0.3:
                    ctx.state["blind_in_plane"] += ctx.dt
            if state.link_age_s is not None and state.link_age_s >= 0.33 and state.failsafe_state == "nominal":
                ctx.state["nominal_at_age_0_33"] = True
            if outage_window is not None and violates:
                ctx.state["violation_outage"] = True
        if len(ctx.state["ids"]) > 1:
            positions = [states[agent].position_m for agent in ctx.state["ids"]]
            for left in range(len(positions)):
                for right in range(left + 1, len(positions)):
                    ctx.state["follower_min"] = min(
                        ctx.state["follower_min"],
                        float(np.linalg.norm(np.asarray(positions[left]) - positions[right])),
                    )
        ctx.frame(
            i, t, target_position,
            {
                agent: agent_frame(states[agent], estimate_position_m=estimate, estimate_age_s=estimate_age)
                for agent in ctx.state["ids"]
            },
        )

    def done(self, ctx):
        if ctx.state["tag_complete"] and ctx.params["mode"] == "tag":
            return "tagged"
        states = [ctx.adapters[agent].get_state() for agent in ctx.state["ids"]]
        if states and all(state.failsafe_state in {"landing", "landed"} for state in states):
            return "landed"
        if any(state.failsafe_state == "killed" for state in states):
            return "killed"
        return None

    def metrics(self, ctx):
        holds = sum(
            1 for event in ctx.events
            if event["kind"] == "failsafe_transition" and event["detail"].get("state") == "degraded"
        )
        landings = sum(
            1 for event in ctx.events
            if event["kind"] == "failsafe_transition" and event["detail"].get("state") == "landing"
        )
        result = {
            "tagged": bool(ctx.state["tag_complete"]),
            "time_to_tag_s": None if ctx.state["tag_time"] is None else float(ctx.state["tag_time"]),
            "min_separation_m": None if not math.isfinite(ctx.state["min_separation"]) else float(ctx.state["min_separation"]),
            "separation_violations": int(ctx.state["separation_violations"]),
            "separation_episodes": int(ctx.state["separation_episodes"]),
            "holds": int(holds),
            "landings": int(landings),
            "time_degraded_s": float(ctx.state["time_degraded"]),
            "max_estimate_age_s": float(ctx.state["max_estimate_age"]),
            "follower_follower_min_m": (
                None if len(ctx.state["ids"]) == 1 or not math.isfinite(ctx.state["follower_min"])
                else float(ctx.state["follower_min"])
            ),
            "guard_activations": int(ctx.state["guard_activations"]),
        }
        if ctx.params["mode"] == "hold":
            start = ctx.scenario.outages[0]["start_s"] if ctx.scenario.outages else None
            reaction_origin = ctx.state["last_command_before_outage"]
            transitions = [
                event for event in ctx.events
                if event["kind"] == "failsafe_transition"
                and event["detail"].get("state") in {"degraded", "landing"}
                and (start is None or event["time_s"] >= start)
            ]
            degraded = next((event for event in transitions if event["detail"].get("state") == "degraded"), None)
            landing = next((event for event in transitions if event["detail"].get("state") == "landing"), None)
            result.update({
                "outage_reaction_hold_s": None if degraded is None or reaction_origin is None else float(degraded["time_s"] - reaction_origin),
                "outage_reaction_land_s": None if landing is None or reaction_origin is None else float(landing["time_s"] - reaction_origin),
                "nominal_at_age_0_33": bool(ctx.state["nominal_at_age_0_33"]),
                "violation_during_or_after_outage": bool(ctx.state["violation_outage"]),
                "blind_in_plane_s": float(ctx.state["blind_in_plane"]),
            })
        return result
