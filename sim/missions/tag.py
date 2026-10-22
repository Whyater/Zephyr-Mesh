"""Simulated tag-zone mission with delayed synthetic observations.

The follower remains separated from the moving target and scores a tag by
holding inside a distance zone.  Link loss, delays, and vehicle motion are all
simulated.  With two or three followers, the followers keep apart from each
other with Run D's reciprocal velocity rule (``separation_rule`` "coop", the
default) or with the older local goal guard ("local", kept for comparison).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.adapters import Geofence
from sim.coop.avoidance import AvoidanceConfig, extrapolate, safe_velocity
from sim.missions.engine import (
    Mission,
    agent_frame,
    register_mission,
    stopping_buffer_m,
    target_path,
)
from sim.missions.estimate_feed import EstimateFeed


# Run D's separation rule, as used by the tag and formation missions ------------
#
# Floor: Run D's EXP-025 minimum centre distance, sim/runs/d4-separation/buffer.json
# line 70725 ("min_separation_m": 0.35), which is also AvoidanceConfig's default
# radius_m (sim/coop/avoidance.py line 33).
COOP_FLOOR_M = 0.35
# Run D's required buffer with dead reckoning on, keyed by (speed m/s, delay s):
# the "envelope" rows of sim/runs/d4-separation/buffer.json, lines 70497 to 70713
# (worst case over head-on and crossing encounters, independent loss up to 30 %
# and 0.05 m noise, 10 Hz sharing).  sim/tests/test_e_coop_separation.py checks this
# copy against the file.
COOP_BUFFER_M = {
    (0.5, 0.0): 0.05, (0.5, 0.05): 0.1, (0.5, 0.1): 0.1, (0.5, 0.2): 0.15, (0.5, 0.3): 0.15, (0.5, 0.5): 0.2,
    (1.0, 0.0): 0.2, (1.0, 0.05): 0.1, (1.0, 0.1): 0.15, (1.0, 0.2): 0.3, (1.0, 0.3): 0.15, (1.0, 0.5): 0.75,
    (2.0, 0.0): 0.1, (2.0, 0.05): 0.15, (2.0, 0.1): 0.15, (2.0, 0.2): 0.15, (2.0, 0.3): 0.3, (2.0, 0.5): 0.75,
}
# Run D's encounter settings (sim/coop/separation.py, the AvoidanceConfig built
# in run_encounter): 2 s time horizon, no extra safety margin, and the
# defaults for responsibility (0.5) and the right-hand tie break.
COOP_TIME_HORIZON_S = 2.0
# Run D's simulator drops a neighbour whose newest message is older than its
# heartbeat timeout (sim/coop/simulator.py, CoopSwarmConfig.heartbeat_timeout_s).
COOP_NEIGHBOR_TIMEOUT_S = 0.5
# The safe velocity is sent as a position goal one response time ahead.  The
# point-mass adapters steer at (goal - position) / 0.25 s (engine.make_adapter),
# so a goal 0.25 s ahead asks for the safe velocity itself.  A lost command
# then leaves the drone heading for a point at most 1.5 m/s x 0.25 s = 0.375 m
# away, where it stops, instead of flying on.
GOAL_LOOKAHEAD_S = 0.25


def coop_buffer_m(link_delay_s: float, speed_limit_mps: float) -> float:
    """Run D's buffer for neighbour data of this delay and drones this fast.

    Run D's buffer is not smooth in delay or speed (a threshold effect at its
    0.05 m grid), so this takes the worst row at or below the first grid speed
    and grid delay that cover the inputs.  Inputs past Run D's grid (2 m/s,
    0.5 s) use every row, which is an extrapolation.
    """
    speeds = sorted({speed for speed, _ in COOP_BUFFER_M})
    delays = sorted({delay for _, delay in COOP_BUFFER_M})
    speed_cover = next((value for value in speeds if value >= speed_limit_mps - 1e-12), speeds[-1])
    delay_cover = next((value for value in delays if value >= link_delay_s - 1e-12), delays[-1])
    return max(
        buffer for (speed, delay), buffer in COOP_BUFFER_M.items()
        if speed <= speed_cover and delay <= delay_cover
    )


def coop_avoidance_config(link_delay_s: float, speed_limit_mps: float) -> AvoidanceConfig:
    """Run D's avoidance settings with the radius widened by its buffer.

    The speed ball is the drone's own speed limit instead of Run D's 2 m/s, so
    the adapter never has to shorten a safe velocity (which could leave the
    permitted half-spaces).
    """
    return AvoidanceConfig(
        radius_m=COOP_FLOOR_M + coop_buffer_m(link_delay_s, speed_limit_mps),
        time_horizon_s=COOP_TIME_HORIZON_S,
        max_speed_mps=float(speed_limit_mps),
        safety_margin_m=0.0,
    )


def telemetry_estimate(position, velocity, age_s):
    """Dead-reckon delivered telemetry to now, as Run D does with neighbour data.

    Returns ``(position, velocity)`` or None when there is no data or it is
    older than Run D's heartbeat timeout.
    """
    if position is None or not math.isfinite(age_s) or age_s > COOP_NEIGHBOR_TIMEOUT_S:
        return None
    velocity = np.asarray(velocity, dtype=float)
    return extrapolate(position, velocity, max(0.0, float(age_s))), velocity


def coop_goal(goal, reference, own, neighbors, config: AvoidanceConfig):
    """Pass one drone's goal through Run D's safe_velocity.

    ``reference`` is the ground station's view of the point the drone's own
    position loop steers from, ``own`` its ``(position, velocity)`` estimate
    and ``neighbors`` the other drones' estimates.  The preferred velocity is
    what the adapter would do with ``goal``, capped at the speed limit.  When
    the rule leaves it unchanged the original goal is sent; otherwise the goal
    becomes ``reference + safe velocity x GOAL_LOOKAHEAD_S``.  Returns
    ``(goal, changed)``.
    """
    goal = np.asarray(goal, dtype=float)
    reference = np.asarray(reference, dtype=float)
    preferred = (goal - reference) / GOAL_LOOKAHEAD_S
    speed = float(np.linalg.norm(preferred))
    if speed > config.max_speed_mps:
        preferred = preferred * (config.max_speed_mps / speed)
    if not neighbors:
        return goal, False
    safe, _ = safe_velocity(own[0], own[1], preferred, neighbors, config)
    if float(np.linalg.norm(safe - preferred)) <= 1e-9:
        return goal, False
    return reference + safe * GOAL_LOOKAHEAD_S, True


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
        "Separation from the target uses the pre-registered stopping-buffer guard on the estimate "
        "(assumes 4 m/s^2 braking); the target takes no part in avoidance, so Run D's rule does not apply to it.",
        "Slot assignment (linear_sum_assignment with a 1 s dwell) is a Run E rule; Run D's allocation is not used.",
        "separation_rule coop (default, 2 or 3 followers): followers avoid each other with Run D's reciprocal "
        "velocity rule (sim.coop.avoidance.safe_velocity) on delivered, dead-reckoned telemetry, radius 0.35 m plus "
        "Run D's buffer from runs/d4-separation/buffer.json.  Run D tested it only on kinematic point masses with "
        "independent loss, so bursty presets and the physics adapter are outside that evidence; telemetry velocity is "
        "the simulated vehicle's own, without estimation noise.  guard_activations counts follower control steps "
        "where the rule changed the preferred velocity.",
        "separation_rule local: pairwise 0.6 m follower-follower goal guard, a stand-in until Run D, kept for "
        "comparison; guard_activations counts pair-steps inside 0.6 m.",
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
        "separation_rule": "coop",
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
        if result["separation_rule"] not in {"coop", "local"}:
            raise ValueError("separation_rule must be coop or local")
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
            # Telemetry travels on the tel links, which carry the scenario's
            # command delay, so that sets which of Run D's buffers applies.
            "coop_config": coop_avoidance_config(ctx.scenario.command_delay_s, p["follower_speed_mps"]),
        })
        self._heading = initial_heading

    def _slots(self, estimate, velocity, n, standoff):
        """Return the pre-registered slot geometry around the estimate."""
        heading = _unit_xy(velocity, fallback=tuple(self._heading[:2]))
        self._heading = heading
        # Prereg EXP-030: "slots at 0.75 m behind, behind-left, behind-right
        # (120 and 240 degrees for 3)", angles measured counterclockwise from
        # the estimated target heading.
        angles = [180.0] if n == 1 else ([180.0, 120.0] if n == 2 else [180.0, 120.0, 240.0])
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
        """Build goals from delivered telemetry and estimates only (never truth)."""
        p = ctx.params
        ids = ctx.state["ids"]
        telemetry = {}
        views = {}
        for agent in ids:
            state, age = ctx.drones[agent].telemetry_view(t)
            telemetry[agent] = None if state is None else np.asarray(state.position_m, dtype=float)
            views[agent] = (telemetry[agent], None if state is None else state.velocity_mps, age)
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
                # Prereg EXP-030 buffer: v * (estimate age + 0.05) + v^2 / (2 * 4),
                # with v the follower speed limit.  Only the estimate age enters.
                age = estimate_age if math.isfinite(estimate_age) else 0.0
                buffer = stopping_buffer_m(p["follower_speed_mps"], age)
                if distance - buffer < p["min_separation_m"]:
                    goal = np.asarray(estimate, dtype=float) - p["standoff_m"] * direction
            goals[agent] = goal
        if p["separation_rule"] == "coop":
            if len(ids) > 1:
                goals = self._coop_goals(ctx, goals, views)
        elif p["follower_guard_m"] > 0.0:
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

    def _coop_goals(self, ctx, goals, views):
        """Run D's follower-to-follower rule on the goals built above.

        Every follower is solved from the same telemetry snapshot, so each pair
        shares the avoidance as Run D's reciprocal rule assumes.  A follower
        with no fresh telemetry of its own keeps its goal (there is nothing to
        steer from).  guard_activations counts follower steps where the rule
        changed the preferred velocity.
        """
        ids = ctx.state["ids"]
        config = ctx.state["coop_config"]
        estimates = {agent: telemetry_estimate(*views[agent]) for agent in ids}
        result = {}
        for agent in ids:
            own = estimates[agent]
            if own is None:
                result[agent] = goals[agent]
                continue
            neighbors = [estimates[other] for other in ids if other != agent and estimates[other] is not None]
            # The point-mass follower steers from its own position, so the
            # dead-reckoned telemetry is also the steering reference.
            goal, changed = coop_goal(goals[agent], own[0], own, neighbors, config)
            if changed:
                ctx.state["guard_activations"] += 1
            result[agent] = goal
        return result

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
        ctx.state["last_states"] = states
        target_position = np.asarray(target.position([t])[0], dtype=float)
        outage_window = next((outage for outage in ctx.outages if outage.start_s <= t < outage.end_s), None)
        # Prereg EXP-029 Part B: a separation violation counts as a late
        # reaction during the blackout or within 3 s after it ends.
        late_window = any(outage.start_s <= t < outage.end_s + 3.0 for outage in ctx.outages)
        # Reaction times are measured from the newest command a follower got
        # before the first blackout began (commands after it ends do not count).
        first_outage_start = min((outage.start_s for outage in ctx.outages), default=math.inf)
        if t < first_outage_start:
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
            # Blind in the target's plane: the blackout is on, the follower is
            # still airborne and within 0.3 m of the target's altitude, whatever
            # its failsafe is doing (holding, slowing or starting to land).
            if outage_window is not None and not state.landed \
                    and abs(float(state.position_m[2]) - target_position[2]) <= 0.3:
                ctx.state["blind_in_plane"] += ctx.dt
            # Contract section 7: still executing mission motion when the command
            # age reaches 0.33 s DURING the blackout.
            if outage_window is not None and state.link_age_s is not None \
                    and state.link_age_s >= 0.33 and state.failsafe_state == "nominal":
                ctx.state["nominal_at_age_0_33"] = True
            if late_window and violates:
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
        # Decide from the states the drones reported on this step: their
        # failsafe transition events were recorded in that same step, so a round
        # that ends on a landing always has the landing event.
        last = ctx.state.get("last_states") or {}
        states = [last[agent] for agent in ctx.state["ids"] if agent in last]
        if any(state.failsafe_state == "killed" for state in states):
            return "killed"
        if ctx.params["mode"] == "hold":
            # Hold mode measures the reaction to a scripted blackout, so it runs
            # to the scenario duration: the landing, the end of a finite
            # blackout and the 3 s after it all stay inside the run.
            return None
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
        # Landings that no scripted outage caused: failsafe transitions to the
        # landing state strictly before the earliest outage start.  With no
        # outage this is every landing.
        if ctx.scenario.outages:
            first_outage_s = min(float(item["start_s"]) for item in ctx.scenario.outages)
            landings_before_outage = sum(
                1 for event in ctx.events
                if event["kind"] == "failsafe_transition"
                and event["detail"].get("state") == "landing"
                and event["time_s"] < first_outage_s
            )
        else:
            landings_before_outage = landings
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
            "landings_before_outage": int(landings_before_outage),
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
            landings_all = [event for event in ctx.events if event["kind"] == "failsafe_transition"
                            and event["detail"].get("state") == "landing"]
            first_landing = min((event["time_s"] for event in landings_all), default=None)
            result.update({
                # A landing latches, so a follower that landed from link loss
                # before the blackout spends it on the ground: report that.
                "first_landing_s": None if first_landing is None else float(first_landing),
                "airborne_at_blackout": bool(start is None or first_landing is None or first_landing >= start),
                "outage_reaction_hold_s": None if degraded is None or reaction_origin is None else float(degraded["time_s"] - reaction_origin),
                "outage_reaction_land_s": None if landing is None or reaction_origin is None else float(landing["time_s"] - reaction_origin),
                # The drone's own command age at each transition (its failsafe
                # reaction, which can be shorter than the time since the
                # blackout began when a command was still in flight).
                "outage_hold_link_age_s": None if degraded is None else degraded["detail"].get("link_age_s"),
                "outage_land_link_age_s": None if landing is None else landing["detail"].get("link_age_s"),
                "nominal_at_age_0_33": bool(ctx.state["nominal_at_age_0_33"]),
                "violation_during_or_after_outage": bool(ctx.state["violation_outage"]),
                "blind_in_plane_s": float(ctx.state["blind_in_plane"]),
            })
        return result
