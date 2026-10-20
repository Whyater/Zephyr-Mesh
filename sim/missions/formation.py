"""Simulated scenario of a planned use case: a four-drone gym formation demo."""
from __future__ import annotations

import math

import numpy as np

from sim.adapters import Geofence
from sim.interaction import downwash_wind_at
from sim.missions.engine import Mission, agent_frame, register_mission
from sim.missions.tag import COOP_NEIGHBOR_TIMEOUT_S, coop_avoidance_config, coop_goal, telemetry_estimate
from sim.sensor import SensorConfig, SensorModel


SLOTS = {
    "f1": np.array((7.0, 4.0, 1.5), dtype=float),
    "f2": np.array((9.0, 4.0, 1.5), dtype=float),
    "f3": np.array((9.0, 6.0, 1.5), dtype=float),
    "f4": np.array((7.0, 6.0, 1.5), dtype=float),
}
DRONE_IDS = tuple(SLOTS)
SPEED_LIMIT_MPS = 1.5


@register_mission
class FormationMission(Mission):
    """Simulated scenario of a planned use case for a gym formation demo."""

    name = "formation"
    rng_names = ("start", "fix_f1", "fix_f2", "fix_f3", "fix_f4")
    limits = (
        "Camera fixes, link loss, and vehicle dynamics are synthetic.",
        "separation_rule coop (default): each drone's goal passes through Run D's reciprocal velocity rule "
        "(sim.coop.avoidance.safe_velocity) using the delivered camera fixes dead-reckoned with telemetry velocity, "
        "radius 0.35 m plus Run D's buffer for the fix latency (runs/d4-separation/buffer.json).  Run D tested it "
        "only on kinematic point masses with independent loss; telemetry velocity is the simulated vehicle's own, "
        "without estimation noise.  guard_activations counts drone control steps where the rule changed the "
        "preferred velocity.",
        "separation_rule local: pairwise 0.8 m goal guard moving a goal 0.4 m away, a stand-in until Run D, kept for "
        "comparison; guard_activations counts pair-steps inside 0.8 m.",
        "The downwash value uses the narrow equivalent-disk interaction model and default hover inputs.",
    )
    defaults = {
        "fix_rate_hz": 30.0,
        "fix_noise_std_m": 0.02,
        "fix_latency_s": 0.0,
        "separation_rule": "coop",
    }

    def validate_params(self, params: dict) -> dict:
        result = super().validate_params(params)
        rate = float(result["fix_rate_hz"])
        noise = float(result["fix_noise_std_m"])
        latency = float(result["fix_latency_s"])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("fix_rate_hz must be finite and positive")
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError("fix_noise_std_m must be finite and nonnegative")
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("fix_latency_s must be finite and nonnegative")
        result["fix_rate_hz"] = rate
        result["fix_noise_std_m"] = noise
        result["fix_latency_s"] = latency
        if result["separation_rule"] not in {"coop", "local"}:
            raise ValueError("separation_rule must be coop or local")
        return result

    @staticmethod
    def _start_offset(rng: np.random.Generator) -> np.ndarray:
        direction = np.asarray(rng.normal(size=3), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-15:
            direction = np.array((1.0, 0.0, 0.0), dtype=float)
            norm = 1.0
        radius = float(rng.uniform(0.0, 0.3))
        return direction / norm * radius

    def _sample_fixes(self, ctx, t: float) -> None:
        period = self._fix_period
        while self._next_capture_time <= t + 1e-12:
            capture_time = self._next_capture_time
            for agent in DRONE_IDS:
                truth = np.asarray(ctx.adapters[agent].get_state().position_m, dtype=float)
                sample = self._sensors[agent].sample(truth, capture_time)
                if sample.measurement_m is not None:
                    message = tuple(float(value) for value in sample.measurement_m)
                    self._fix_links[agent].send(self._fix_sequences[agent], capture_time, message)
                    self._fix_sequences[agent] += 1
            self._next_capture_time += period

    def _consume_fixes(self, ctx, t: float) -> None:
        for agent in DRONE_IDS:
            for packet in self._fix_links[agent].deliveries_until(t):
                measurement = np.asarray(packet.message, dtype=float)
                capture_time = float(packet.send_time_s)
                ctx.adapters[agent].set_position_fix(measurement, capture_time)
                self._latest_fixes[agent] = measurement.copy()
                self._latest_fix_times[agent] = capture_time

    def _goals(self, ctx, t: float) -> dict[str, np.ndarray]:
        if ctx.params["separation_rule"] == "coop":
            return self._coop_goals(ctx, t)
        return self._local_goals()

    def _coop_goals(self, ctx, t: float) -> dict[str, np.ndarray]:
        """Pass every slot goal through Run D's rule, from one snapshot of fixes.

        The ground station knows each drone's newest delivered fix (with its
        capture time) and its delivered telemetry velocity.  Positions are the
        fixes dead-reckoned to now.  The offboard drone steers from its newest
        fix, so that fix is the reference the safe velocity is measured from.
        """
        estimates = {}
        for agent in DRONE_IDS:
            fix = self._latest_fixes[agent]
            if fix is None:
                estimates[agent] = None
                continue
            state, telemetry_age = ctx.drones[agent].telemetry_view(t)
            fresh = state is not None and math.isfinite(telemetry_age) and telemetry_age <= COOP_NEIGHBOR_TIMEOUT_S
            velocity = state.velocity_mps if fresh else np.zeros(3)
            estimates[agent] = telemetry_estimate(fix, velocity, max(0.0, t - self._latest_fix_times[agent]))
        goals = {}
        for agent in DRONE_IDS:
            own = estimates[agent]
            if own is None:
                goals[agent] = SLOTS[agent].copy()
                continue
            neighbors = [estimates[other] for other in DRONE_IDS if other != agent and estimates[other] is not None]
            goal, changed = coop_goal(SLOTS[agent], self._latest_fixes[agent], own, neighbors, self._coop_config)
            if changed:
                self._guard_activations += 1
            goals[agent] = goal
        return goals

    def _local_goals(self) -> dict[str, np.ndarray]:
        """The 0.8 m stand-in guard from before Run D, kept for comparison."""
        goals = {agent: SLOTS[agent].copy() for agent in DRONE_IDS}
        for left_index, left in enumerate(DRONE_IDS):
            if self._latest_fixes[left] is None:
                continue
            for right in DRONE_IDS[left_index + 1:]:
                if self._latest_fixes[right] is None:
                    continue
                delta = self._latest_fixes[right] - self._latest_fixes[left]
                distance = float(np.linalg.norm(delta))
                if distance >= 0.8:
                    continue
                if distance <= 1e-12:
                    direction = np.array((1.0, 0.0, 0.0), dtype=float)
                else:
                    direction = delta / distance
                higher_id = right
                goals[higher_id] = goals[higher_id] + 0.4 * direction
                self._guard_activations += 1
        return goals

    def setup(self, ctx) -> None:
        if ctx.scenario.adapter not in {"point_mass_offboard", "physics_offboard"}:
            raise ValueError("formation requires an offboard adapter")

        self._ctx = ctx
        self._fix_period = 1.0 / ctx.params["fix_rate_hz"]
        self._next_capture_time = 0.0
        self._sensors = {
            agent: SensorModel(
                SensorConfig(noise_std_m=ctx.params["fix_noise_std_m"]),
                seed=ctx.seed_for(f"fix_{agent}"),
            )
            for agent in DRONE_IDS
        }
        self._fix_links = {
            agent: ctx.link(f"fix:{agent}", delay_s=ctx.params["fix_latency_s"])
            for agent in DRONE_IDS
        }
        self._fix_sequences = {agent: 0 for agent in DRONE_IDS}
        self._latest_fixes = {agent: None for agent in DRONE_IDS}
        self._latest_fix_times = {agent: None for agent in DRONE_IDS}
        self._guard_activations = 0
        # Neighbour positions come from fixes that are fix_latency_s old (plus
        # up to one 30 Hz period), so that sets which of Run D's buffers applies.
        self._coop_config = coop_avoidance_config(ctx.params["fix_latency_s"], SPEED_LIMIT_MPS)

        geofence = Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)
        start_rng = ctx.rng("start")
        for agent in DRONE_IDS:
            initial = SLOTS[agent] + self._start_offset(start_rng)
            ctx.add_drone(
                agent,
                initial_position_m=initial,
                geofence=geofence,
                max_speed_mps=SPEED_LIMIT_MPS,
            )

        downwash_values = []
        hover_thrust_n = 0.030 * 9.81
        for upper in DRONE_IDS:
            for lower in DRONE_IDS:
                if upper == lower:
                    continue
                wind = downwash_wind_at(
                    SLOTS[upper],
                    hover_thrust_n,
                    4,
                    0.023,
                    SLOTS[lower],
                )
                downwash_values.append(float(np.linalg.norm(wind)))

        self._samples = []
        self._min_separation = math.inf
        self._downwash_max = max(downwash_values) if downwash_values else None
        ctx.state["slots"] = {agent: SLOTS[agent].copy() for agent in DRONE_IDS}
        ctx.state["samples"] = self._samples
        ctx.state["guard_activations"] = self._guard_activations
        ctx.state["downwash_wind_max_mps"] = self._downwash_max
        ctx.start_log(DRONE_IDS)

    def tick(self, ctx, i: int, t: float) -> None:
        self._consume_fixes(ctx, t)
        self._sample_fixes(ctx, t)
        self._consume_fixes(ctx, t)

        goals = self._goals(ctx, t)
        states = {}
        for agent in DRONE_IDS:
            ctx.drones[agent].set_goal(goals[agent])
            states[agent] = ctx.drones[agent].tick(i)

        positions = np.vstack([np.asarray(states[agent].position_m, dtype=float) for agent in DRONE_IDS])
        errors = positions - np.vstack([SLOTS[agent] for agent in DRONE_IDS])
        self._samples.append((float(t), errors.copy()))
        for left_index in range(len(DRONE_IDS)):
            for right_index in range(left_index + 1, len(DRONE_IDS)):
                separation = float(np.linalg.norm(positions[right_index] - positions[left_index]))
                self._min_separation = min(self._min_separation, separation)

        ctx.state["guard_activations"] = self._guard_activations
        per_agent = {}
        for agent in DRONE_IDS:
            latest = self._latest_fixes[agent]
            capture_time = self._latest_fix_times[agent]
            age = None if capture_time is None else max(0.0, float(t) - float(capture_time))
            per_agent[agent] = agent_frame(states[agent], estimate_position_m=latest, estimate_age_s=age)
        ctx.frame(i, t, None, per_agent)

    def metrics(self, ctx) -> dict:
        rows = [errors for time_s, errors in self._samples if time_s >= 5.0]
        if rows:
            errors = np.stack(rows, axis=0)
            norms = np.linalg.norm(errors, axis=2).reshape(-1)
            rms = float(np.sqrt(np.mean(norms * norms)))
            p95 = float(np.percentile(norms, 95.0))
            axis_stds = np.std(errors, axis=0, ddof=0)
            per_axis = float(np.mean(axis_stds))
        else:
            rms = None
            p95 = None
            per_axis = None
        min_separation = None if not math.isfinite(self._min_separation) else float(self._min_separation)
        return {
            "rms_error_m": rms,
            "p95_error_m": p95,
            "per_axis_std_m": per_axis,
            "min_separation_m": min_separation,
            "guard_activations": int(self._guard_activations),
            "downwash_wind_max_mps": None if self._downwash_max is None else float(self._downwash_max),
        }
