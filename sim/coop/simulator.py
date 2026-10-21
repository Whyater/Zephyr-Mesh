"""Deterministic cooperative point-mass simulation for the simulated S7 run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from sim.coop.allocation import reallocate, slots_for_alive
from sim.coop.avoidance import AvoidanceConfig, extrapolate, safe_velocity
from sim.coop.fusion import (
    covariance_intersection_fast,
    covariance_intersection_many,
    information_fusion,
    nees,
)
from sim.estimation.base import EstimatorSettings, MODELS
from sim.estimation.factory import make_estimator
from sim.estimation.targets import make_target
from sim.interaction import ellipsoid_separation
from sim.link import LinkConfig, PacketEvent, SimulatedLink
from sim.scenarios import default_hardware_profile
from sim.sensor import SensorConfig, SensorModel


def _vec(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return result.copy()


def _finite(value: float, name: str, minimum: float = 0.0) -> float:
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


@dataclass(frozen=True)
class CoopLinkConfig:
    delay_s: float = 0.05
    loss_probability: float = 0.1
    loss_model: str = "independent"
    share_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        _finite(self.delay_s, "delay_s")
        probability = _finite(self.loss_probability, "loss_probability")
        if probability > 1.0:
            raise ValueError("loss_probability must be at most one")
        if self.loss_model not in {"none", "independent", "burst"}:
            raise ValueError("unsupported loss model")
        if not np.isfinite(self.share_rate_hz) or self.share_rate_hz <= 0:
            raise ValueError("share_rate_hz must be positive")


@dataclass(frozen=True)
class CoopSwarmConfig:
    n_agents: int = 4
    seed: int = 23
    duration_s: float = 20.0
    control_dt_s: float = 0.05
    sensor_rate_hz: float = 10.0
    noise_std_m: float = 0.05
    estimator: str = "cv"
    fusion: str = "ci"
    decision: str = "onboard"
    target: str = "ct"
    target_params: tuple = (("radius", 3.0), ("speed", 0.8))
    target_altitude_m: float = 1.5
    standoff_radius_m: float = 1.5
    slot_altitude_m: float = 1.5
    keep_out_radius_m: float = 0.75
    link: CoopLinkConfig = field(default_factory=CoopLinkConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    use_avoidance: bool = True
    dead_reckon_neighbors: bool = True
    max_speed_mps: float = 2.0
    max_accel_mps2: float = 4.0
    position_gain: float = 1.8
    velocity_gain: float = 1.1
    heartbeat_timeout_s: float = 0.5
    dropouts: tuple = (("drone-001", 10.0, "silent"),)
    switch_penalty_m: float = 0.3
    log_every: int = 2
    initial_radius_m: float = 4.0
    gs_outage: tuple = ()
    metric_windows: tuple = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0))
    ci_mode: str = "fast"
    # Proposal (EXP-026 close-pass review), opt-in: each drone broadcasts the
    # slot it is flying to and the alive list it planned with, and every
    # planner uses those shared claims for the switch penalty. The default
    # False keeps the original private hysteresis (and the registered Run D
    # outputs), which can lock two survivors onto the same slot after a dropout.
    share_slot_claims: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.n_agents, bool) or int(self.n_agents) != self.n_agents or not 1 <= int(self.n_agents) <= 12:
            raise ValueError("n_agents must be an integer from 1 through 12")
        scalar_names = (
            "duration_s", "control_dt_s", "sensor_rate_hz",
            "target_altitude_m", "standoff_radius_m", "slot_altitude_m",
            "keep_out_radius_m", "max_speed_mps", "max_accel_mps2",
            "position_gain", "velocity_gain", "heartbeat_timeout_s",
            "initial_radius_m",
        )
        for name in scalar_names:
            _finite(getattr(self, name), name)
        if self.control_dt_s <= 0 or self.sensor_rate_hz <= 0 or self.duration_s <= 0:
            raise ValueError("time and rate values must be positive")
        if self.estimator not in MODELS or self.fusion not in {"none", "ci", "naive", "central"}:
            raise ValueError("unsupported estimator or fusion mode")
        if self.decision not in {"onboard", "ground"}:
            raise ValueError("decision must be onboard or ground")
        if self.ci_mode not in {"fast", "exact"}:
            raise ValueError("ci_mode must be fast or exact")
        if not isinstance(self.share_slot_claims, (bool, np.bool_)):
            raise ValueError("share_slot_claims must be boolean")
        if int(self.log_every) != self.log_every or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
        ids = set(self.agent_ids)
        for row in self.dropouts:
            if len(row) != 3 or row[0] not in ids or row[2] not in {"silent", "landed"}:
                raise ValueError("dropouts must contain valid agent id, time, and mode")
        for row in self.gs_outage:
            if len(row) != 2 or not np.isfinite(row[0]) or not np.isfinite(row[1]) or row[0] < 0 or row[1] < row[0]:
                raise ValueError("gs_outage windows must be finite (start, end) pairs")

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(f"drone-{i + 1:03d}" for i in range(self.n_agents))


@dataclass(frozen=True)
class CoopRun:
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.data

    @property
    def steps(self):
        return self.data["steps"]

    @property
    def summary(self):
        return self.data["summary"]


@dataclass
class _Agent:
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    sensor: SensorModel
    estimator: Any
    phase_s: float
    next_capture_s: float
    active: bool = True
    landed: bool = False
    silent: bool = False
    latest: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_gs_track: dict[str, Any] | None = None
    latest_command: dict[str, Any] | None = None
    previous_slots: dict[str, int] = field(default_factory=dict)
    previous_alive: tuple[str, ...] = ()
    slot_index: int = 0
    slot_members: tuple[str, ...] = ()
    believed_assignment: dict[str, int] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)
    leaving_sent: bool = False
    left_peers: set[str] = field(default_factory=set)


def _count_disagreements(
    alive_views: Mapping[str, tuple[str, ...]],
    assignments: Mapping[str, Mapping[str, int]],
) -> int:
    """Count planner pairs that disagree on assignments they both know."""
    count = 0
    planners = sorted(alive_views)
    for left_index, left in enumerate(planners):
        for right in planners[left_index + 1 :]:
            if left not in alive_views[right] or right not in alive_views[left]:
                continue
            shared = set(alive_views[left]).intersection(alive_views[right])
            if any(
                assignments[left].get(peer) != assignments[right].get(peer)
                for peer in shared
            ):
                count += 1
    return count


class CoopSwarmSimulator:
    """Run a deterministic simulated cooperative point-mass swarm."""

    def __init__(self, config: CoopSwarmConfig | None = None):
        self.config = config or CoopSwarmConfig()
        c = self.config
        self.settings = EstimatorSettings()
        self.target_seed = int(np.random.SeedSequence([c.seed, 1]).generate_state(1)[0])
        target_params = {("radius_m" if k == "radius" else "speed_mps" if k == "speed" else k): v for k, v in c.target_params}
        self.target = make_target(c.target, target_params, seed=self.target_seed)
        self._target_start = self.target.start_position + np.array((0.0, 0.0, c.target_altitude_m))
        self.agents: dict[str, _Agent] = {}
        period = 1.0 / c.sensor_rate_hz
        for i, aid in enumerate(c.agent_ids):
            angle = 2.0 * np.pi * i / c.n_agents
            position = self._target_start + c.initial_radius_m * np.array((np.cos(angle), np.sin(angle), 0.0))
            sensor_seed = int(np.random.SeedSequence([c.seed, 100 + i]).generate_state(1)[0])
            sensor = SensorModel(SensorConfig(noise_std_m=c.noise_std_m), seed=sensor_seed)
            estimator = make_estimator(c.estimator, self.settings, self._target_start)
            if i == 0:
                phase = 0.0
            else:
                phase = float(np.random.default_rng(np.random.SeedSequence([c.seed, 3000 + i])).uniform(0.0, period))
            self.agents[aid] = _Agent(
                aid, position, np.zeros(3), sensor, estimator, phase, phase
            )
        self.gs_fuser = make_estimator(c.estimator, self.settings, self._target_start) if (c.fusion == "central" or c.decision == "ground") else None
        self._gs_last_fused_stamp = -float("inf")
        self._gs_last_update_stamp = -float("inf")
        self._gs_reject_count = 0
        self._gs_state_messages: dict[str, dict[str, Any]] = {}
        self._ground_command_sent = False
        self._links: dict[tuple[str, str], SimulatedLink] = {}
        self._messages: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._events: list[PacketEvent] = []
        self._event_kinds: dict[int, str] = {}
        self._seq = 0
        nodes = list(c.agent_ids) + ["gs"]
        route_index = 0
        for sender in nodes:
            for receiver in nodes:
                if sender == receiver:
                    continue
                link_loss = c.link.loss_probability if c.link.loss_model != "none" else 0.0
                link_model = c.link.loss_model
                self._links[(sender, receiver)] = SimulatedLink(
                    LinkConfig(
                        delay_s=c.link.delay_s,
                        loss_probability=link_loss,
                        loss_model=link_model,
                    ),
                    seed=int(
                        np.random.SeedSequence([c.seed, 1000 + route_index])
                        .generate_state(1)[0]
                    ),
                )
                route_index += 1
        self._traffic_by_kind = {
            "state": 240,
            "raw": 56,
            "track": 216,
            "command": 24,
            "leaving": 1,
        }
        self._traffic_kind = {
            kind: {"messages": 0, "bytes": 0}
            for kind in self._traffic_by_kind
        }

    def _ci(self, means, covariances):
        if self.config.ci_mode == "fast":
            return covariance_intersection_fast(means, covariances)
        return covariance_intersection_many(means, covariances)

    def model_provenance(self) -> dict[str, Any]:
        # runs/c5-heartbeat/thresholds.json gives h = 10 Hz, independent p = 0.1,
        # and T = 0.5 s for fewer than one false alarm per hour.
        return {
            "model": "kinematic_point_mass",
            "simulated": True,
            "vehicle_fleet": (
                "sim/vehicle_fleet.py is not used because it is not wired into "
                "the swarm and v1 hardware profiles forbid the interaction hook"
            ),
            "separation": (
                "Run A ellipsoid separation from sim/interaction.py is used "
                "for the separation check"
            ),
            "links": "abstract directed links with literature-derived delay and loss thresholds",
            "sensor_noise": "isotropic and independent of observer position",
            "neighbor_prediction": (
                f"constant-velocity model with q_acc_psd = {self.settings.q_acc_psd}"
            ),
            "neighbor_prediction_q_acc": float(self.settings.q_acc_psd),
        }

    def _truth(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self.target.position([t])[0] + np.array((0.0, 0.0, self.config.target_altitude_m)), self.target.velocity([t])[0]

    def _dropout_state(self, t: float) -> None:
        for aid, when, mode in self.config.dropouts:
            if t + 1e-12 < float(when):
                continue
            agent = self.agents[aid]
            if mode == "landed":
                if not agent.leaving_sent:
                    for receiver in self.config.agent_ids:
                        if receiver != aid:
                            self._send(aid, receiver, float(when), "leaving", {})
                    self._send(aid, "gs", float(when), "leaving", {})
                    agent.leaving_sent = True
                agent.active = False
                agent.landed = True
                agent.velocity[:] = 0.0
            else:
                agent.silent = True

    def _outage(self, t: float) -> bool:
        return any(float(start) <= t <= float(end) for start, end in self.config.gs_outage)

    def _full_track(self, estimator: Any, t: float) -> tuple[np.ndarray, np.ndarray]:
        prediction = estimator.predict_at(max(float(t), float(estimator.timestamp_s)))
        mean = np.r_[prediction.position_m, prediction.velocity_mps]
        if hasattr(estimator, "tracker"):
            tracker = estimator.tracker
            dt = max(0.0, float(t) - float(tracker.timestamp_s))
            transition, process = estimator._transition(dt)
            covariance = transition @ tracker.P @ transition.T + process
        else:
            covariance = np.eye(6)
            covariance[:3, :3] = prediction.position_cov
        return mean, (covariance + covariance.T) * 0.5

    def _predict_track(self, mean: np.ndarray, covariance: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, float(dt))
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = float(self.settings.q_acc_psd)
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * dt**3 / 3.0 * q
        Q[:3, 3:] = np.eye(3) * dt**2 / 2.0 * q
        Q[3:, :3] = Q[:3, 3:].T
        Q[3:, 3:] = np.eye(3) * dt * q
        result = F @ np.asarray(covariance, dtype=float) @ F.T + Q
        return F @ np.asarray(mean, dtype=float), (result + result.T) * 0.5

    def _send(self, sender: str, receiver: str, t: float, kind: str, message: dict[str, Any]) -> None:
        link = self._links[(sender, receiver)]
        seq = self._seq
        self._seq += 1
        event = link.send(sender, receiver, seq, float(t), message)
        if (sender == "gs" or receiver == "gs") and self._outage(float(t)):
            event.receive_time = None
            event.packet_age_s = None
            event.loss_reason = "gs_outage"
            event.outcome = "lost"
            # SimulatedLink has no public cancellation API, so remove the event
            # from its pending queue after marking the outage loss.
            link._pending = [item for item in link._pending if item is not event]
        self._messages[(sender, receiver, seq)] = {"kind": kind, **message}
        self._event_kinds[id(event)] = kind
        self._events.append(event)
        size = self._traffic_by_kind[kind]
        self._traffic_kind[kind]["messages"] += 1
        self._traffic_kind[kind]["bytes"] += size

    def _deliver(self, t: float) -> None:
        ready: list[PacketEvent] = []
        for link in self._links.values():
            ready.extend(link.advance(t))
        ready.sort(key=lambda event: (float(event.receive_time), event.sender, event.seq))
        raw_rows = []
        for event in ready:
            if (event.sender == "gs" or event.receiver == "gs") and self._outage(float(event.receive_time)):
                event.receive_time = None
                event.packet_age_s = None
                event.loss_reason = "gs_outage"
                event.outcome = "lost"
                continue
            message = self._messages.get((event.sender, event.receiver, event.seq))
            if message is None:
                continue
            if event.receiver == "gs":
                if message["kind"] == "raw" and self.gs_fuser is not None:
                    raw_rows.append((float(message["capture_time"]), event.sender, event.seq, np.asarray(message["measurement"], dtype=float)))
                elif message["kind"] == "state":
                    self._gs_state_messages[event.sender] = dict(message, receive_time=float(event.receive_time), sent_at=float(event.send_time))
                elif message["kind"] == "leaving":
                    self._gs_state_messages.pop(event.sender, None)
            elif message["kind"] == "state":
                receiver = self.agents[event.receiver]
                if self.config.share_slot_claims:
                    # Opt-in claims only: a late packet must not bring back a drone
                    # that already said it left, or replace a newer claim with an
                    # older one.
                    newest = receiver.latest.get(event.sender)
                    if event.sender in receiver.left_peers or (
                        newest is not None and float(event.send_time) < float(newest["sent_at"])
                    ):
                        continue
                receiver.latest[event.sender] = dict(message, receive_time=float(event.receive_time), sent_at=float(event.send_time))
            elif message["kind"] == "leaving":
                self.agents[event.receiver].latest.pop(event.sender, None)
                self.agents[event.receiver].left_peers.add(event.sender)
            elif message["kind"] == "track":
                self.agents[event.receiver].latest_gs_track = dict(message, receive_time=float(event.receive_time), sent_at=float(event.send_time))
            elif message["kind"] == "command":
                self.agents[event.receiver].latest_command = dict(message, receive_time=float(event.receive_time), sent_at=float(event.send_time))
        for stamp, sender, seq, measurement in sorted(raw_rows, key=lambda item: (item[0], item[1], item[2])):
            del sender, seq
            if stamp < self._gs_last_fused_stamp - 1e-12:
                self._gs_reject_count += 1
                continue
            try:
                update_stamp = stamp if stamp > self._gs_last_update_stamp else np.nextafter(self._gs_last_update_stamp, float("inf"))
                self.gs_fuser.update(measurement, update_stamp)
            except ValueError:
                self._gs_reject_count += 1
                continue
            self._gs_last_fused_stamp = stamp
            self._gs_last_update_stamp = update_stamp

    def _track_for(self, agent: _Agent, t: float) -> tuple[np.ndarray, np.ndarray, str, float | None]:
        local_mean, local_cov = self._full_track(agent.estimator, t)
        if self.config.fusion == "central" and agent.latest_gs_track is not None:
            message = agent.latest_gs_track
            receive_age = max(0.0, t - float(message["receive_time"]))
            if receive_age <= self.config.heartbeat_timeout_s + 1e-12:
                age = max(0.0, t - float(message["sent_at"]))
                mean, covariance = self._predict_track(message["mean"], message["cov"], age)
                return mean, covariance, "central", float(message.get("source_time", t))
        if self.config.fusion == "none":
            return local_mean, local_cov, "local", float(agent.estimator.timestamp_s)
        tracks = [(local_mean, local_cov)]
        for sender, message in sorted(agent.latest.items()):
            raw_age = max(0.0, t - float(message["receive_time"]))
            if raw_age <= self.config.heartbeat_timeout_s + 1e-12:
                age = max(0.0, t - float(message["sent_at"]))
                tracks.append(self._predict_track(message["mean"], message["cov"], age))
        means, covariances = zip(*tracks)
        if self.config.fusion == "naive":
            mean, covariance = information_fusion(means, covariances)
        else:
            mean, covariance, _ = self._ci(means, covariances)
        return np.asarray(mean), np.asarray(covariance), self.config.fusion, float(agent.estimator.timestamp_s)

    def _gs_group(self, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
        rows = []
        for _, message in sorted(self._gs_state_messages.items()):
            raw_age = max(0.0, t - float(message["receive_time"]))
            if raw_age <= self.config.heartbeat_timeout_s + 1e-12:
                age = max(0.0, t - float(message["sent_at"]))
                rows.append(self._predict_track(message["mean"], message["cov"], age))
        if not rows:
            return None, None
        means, covariances = zip(*rows)
        mean, covariance, _ = self._ci(means, covariances)
        return np.asarray(mean), np.asarray(covariance)

    def _project_keep_out(self, position: np.ndarray, velocity: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, bool]:
        candidate = position + velocity * self.config.control_dt_s
        delta = candidate - center
        distance = float(np.linalg.norm(delta))
        if distance >= self.config.keep_out_radius_m:
            return velocity, False
        direction = delta / distance if distance > 1e-12 else np.array((1.0, 0.0, 0.0))
        projected = center + direction * self.config.keep_out_radius_m
        return (projected - position) / self.config.control_dt_s, True

    def _alive_view(self, agent: _Agent, t: float) -> tuple[str, ...]:
        alive = [agent.agent_id]
        for peer, message in sorted(agent.latest.items()):
            raw_age = max(0.0, t - float(message["receive_time"]))
            if raw_age <= self.config.heartbeat_timeout_s + 1e-12:
                alive.append(peer)
        return tuple(sorted(set(alive)))

    def _positions_view(self, agent: _Agent, alive: tuple[str, ...], t: float) -> dict[str, np.ndarray]:
        positions = {agent.agent_id: agent.position.copy()}
        for peer in alive:
            if peer == agent.agent_id:
                continue
            message = agent.latest[peer]
            age = max(0.0, t - float(message["sent_at"]))
            position = np.asarray(message["position"], dtype=float)
            if self.config.dead_reckon_neighbors:
                position = extrapolate(position, message["velocity"], age)
            positions[peer] = position
        return positions

    def _previous_slots(self, agent: _Agent, alive: tuple[str, ...]) -> dict[str, int]:
        """Return the slot indices that the switch penalty should favour."""
        if not self.config.share_slot_claims:
            return {key: value for key, value in agent.previous_slots.items() if key in alive}
        # Shared claims: own slot plus each peer's last broadcast slot. A claim
        # counts only when it was planned with exactly this planner's alive list
        # (its membership identity), so a claim made while a departed drone was
        # still counted, or on a ring of another size, never pulls a drone.
        claims: dict[str, int] = {}
        if agent.slot_members == alive:
            claims[agent.agent_id] = int(agent.slot_index)
        for peer in alive:
            message = agent.latest.get(peer)
            if peer == agent.agent_id or message is None:
                continue
            if tuple(message.get("slot_members", ())) == alive:
                claims[peer] = int(message["slot_index"])
        # Two drones claiming one slot: the lowest ID keeps the bonus and the
        # others lose theirs, so every planner holding the same claims breaks
        # the tie the same way.
        previous: dict[str, int] = {}
        for peer in sorted(claims):
            if claims[peer] not in previous.values():
                previous[peer] = claims[peer]
        return previous

    def _ground_command(self, t: float, gs_mean: np.ndarray | None) -> None:
        if self.config.decision != "ground" or gs_mean is None:
            return
        positions: dict[str, np.ndarray] = {}
        alive = []
        for aid, message in sorted(self._gs_state_messages.items()):
            raw_age = max(0.0, t - float(message["receive_time"]))
            if raw_age <= self.config.heartbeat_timeout_s + 1e-12:
                age = max(0.0, t - float(message["sent_at"]))
                alive.append(aid)
                positions[aid] = extrapolate(message["position"], message["velocity"], age) if self.config.dead_reckon_neighbors else np.asarray(message["position"], dtype=float)
        if not alive:
            return
        alive = sorted(alive)
        slots = slots_for_alive(gs_mean[:3], len(alive), self.config.standoff_radius_m, self.config.slot_altitude_m, 0.0)
        assignment = reallocate(alive, positions, slots, None, switch_penalty_m=self.config.switch_penalty_m)
        assignment_json = {key: int(value) for key, value in assignment.items()}
        for aid, index in sorted(assignment.items()):
            self._send(
                "gs",
                aid,
                t,
                "command",
                {
                    "setpoint": slots[index].copy(),
                    "source_time": t,
                    "slot_index": int(index),
                    "alive_count": len(alive),
                    "assignment": assignment_json,
                },
            )
        self._ground_command_sent = True

    def run(self) -> CoopRun:
        c = self.config
        n_steps = int(round(c.duration_s / c.control_dt_s))
        steps: list[dict[str, Any]] = []
        errors = {aid: [] for aid in c.agent_ids}
        standoff = {aid: [] for aid in c.agent_ids}
        min_pair = float("inf")
        separation_violations = 0
        spherical_violations = 0
        keep_true_entries = 0
        keep_fused_entries = 0
        role_changes = 0
        plan_disagreements = 0
        assignments: dict[str, dict[str, int]] = {}
        gs_errors = []
        gs_fuser_errors = []
        gs_window = {f"{float(a):g}-{float(b):g}": [] for a, b in c.metric_windows}
        for k in range(n_steps + 1):
            t = round(k * c.control_dt_s, 12)
            self._dropout_state(t)
            truth, _ = self._truth(t)
            for agent in self.agents.values():
                while agent.active and agent.next_capture_s <= t + 1e-12:
                    capture = round(agent.next_capture_s, 12)
                    sample = agent.sensor.sample(self._truth(capture)[0], capture)
                    if sample.available:
                        agent.estimator.update(sample.measurement_m, capture)
                        if self.gs_fuser is not None and not agent.silent:
                            self._send(agent.agent_id, "gs", capture, "raw", {"measurement": np.asarray(sample.measurement_m), "capture_time": capture})
                    agent.next_capture_s += 1.0 / c.sensor_rate_hz
            self._deliver(t)
            share_due = k == 0 or abs((t * c.link.share_rate_hz) - round(t * c.link.share_rate_hz)) < 1e-9
            if share_due:
                for aid, agent in sorted(self.agents.items()):
                    if not agent.active or agent.silent:
                        continue
                    mean, covariance, _, source_time = self._track_for(agent, t)
                    message = {
                        "position": agent.position.copy(),
                        "velocity": agent.velocity.copy(),
                        "mean": mean.copy(),
                        "cov": covariance.copy(),
                        "source_time": source_time if source_time is not None else t,
                    }
                    if c.share_slot_claims:
                        message["slot_index"] = int(agent.slot_index)
                        message["slot_members"] = agent.slot_members
                    for receiver in c.agent_ids:
                        if receiver != aid and receiver not in agent.left_peers:
                            self._send(aid, receiver, t, "state", message)
                    self._send(aid, "gs", t, "state", message)
                gs_mean = None
                if self.gs_fuser is not None and (c.fusion == "central" or c.decision == "ground"):
                    gs_mean, gs_cov = self._full_track(self.gs_fuser, t)
                    if c.fusion == "central":
                        for receiver in c.agent_ids:
                            self._send(
                                "gs", receiver, t, "track",
                                {
                                    "mean": gs_mean.copy(),
                                    "cov": gs_cov.copy(),
                                    "source_time": (
                                        self._gs_last_fused_stamp
                                        if np.isfinite(self._gs_last_fused_stamp)
                                        else t
                                    ),
                                },
                            )
                else:
                    gs_mean, _ = self._gs_group(t)
                self._deliver(t)
                if c.decision == "ground":
                    self._ground_command(t, gs_mean)
                    self._deliver(t)
            track_cache: dict[str, tuple[np.ndarray, np.ndarray, str, float | None]] = {}
            slot_cache: dict[str, np.ndarray] = {}
            alive_cache: dict[str, tuple[str, ...]] = {}
            command_alive_count: dict[str, int] = {}
            for aid, agent in sorted(self.agents.items()):
                if not agent.active:
                    continue
                track = self._track_for(agent, t)
                track_cache[aid] = track
                fused = track[0]
                alive = self._alive_view(agent, t)
                alive_cache[aid] = alive
                if c.decision == "ground" and agent.latest_command is not None:
                    command_age = max(0.0, t - float(agent.latest_command["receive_time"]))
                else:
                    command_age = float("inf")
                if c.decision == "ground" and command_age <= c.heartbeat_timeout_s + 1e-12:
                    command = agent.latest_command
                    assignment = {key: int(value) for key, value in command.get("assignment", {}).items()}
                    alive_count = int(command.get("alive_count", len(alive)))
                    command_alive_count[aid] = alive_count
                    if assignment:
                        alive = tuple(sorted(assignment))
                        alive_cache[aid] = alive
                    slot_index = int(command["slot_index"])
                    slot_cache[aid] = np.asarray(command["setpoint"], dtype=float)
                else:
                    command_alive_count[aid] = len(alive)
                    slots = slots_for_alive(fused[:3], len(alive), c.standoff_radius_m, c.slot_altitude_m, 0.0)
                    assignment = reallocate(
                        alive,
                        self._positions_view(agent, alive, t),
                        slots,
                        self._previous_slots(agent, alive),
                        switch_penalty_m=c.switch_penalty_m,
                    )
                    slot_index = int(assignment[aid])
                    slot_cache[aid] = slots[slot_index]
                if aid in assignments and assignments[aid].get(aid) != slot_index:
                    role_changes += 1
                assignments[aid] = dict(assignment)
                agent.previous_slots = dict(assignment)
                agent.previous_alive = alive
                agent.slot_index = slot_index
                agent.slot_members = alive
                agent.believed_assignment = dict(assignment)
            step_disagreement_count = _count_disagreements(alive_cache, assignments)
            if k < n_steps:
                for aid, agent in sorted(self.agents.items()):
                    if not agent.active or agent.landed:
                        continue
                    fused = track_cache[aid][0]
                    flags: set[str] = set()
                    if agent.silent:
                        preferred = np.zeros(3)
                    elif c.decision == "ground":
                        raw_age = float("inf") if agent.latest_command is None else max(0.0, t - float(agent.latest_command["receive_time"]))
                        if agent.latest_command is None:
                            preferred = np.zeros(3)
                        elif raw_age > c.heartbeat_timeout_s + 1e-12 and self._ground_command_sent:
                            preferred = np.zeros(3)
                            flags.add("command_timeout")
                        else:
                            pd_accel = c.position_gain * (np.asarray(agent.latest_command["setpoint"]) - agent.position) - c.velocity_gain * (agent.velocity - fused[3:])
                            preferred = agent.velocity + pd_accel * c.control_dt_s
                    else:
                        pd_accel = c.position_gain * (slot_cache[aid] - agent.position) - c.velocity_gain * (agent.velocity - fused[3:])
                        preferred = agent.velocity + pd_accel * c.control_dt_s
                    speed = float(np.linalg.norm(preferred))
                    if speed > c.max_speed_mps:
                        preferred *= c.max_speed_mps / speed
                        flags.add("speed_cap")
                    neighbors = []
                    for peer, message in sorted(agent.latest.items()):
                        raw_age = max(0.0, t - float(message["receive_time"]))
                        if raw_age <= c.heartbeat_timeout_s + 1e-12:
                            age = max(0.0, t - float(message["sent_at"]))
                            position = extrapolate(message["position"], message["velocity"], age) if c.dead_reckon_neighbors else np.asarray(message["position"], dtype=float)
                            neighbors.append((position, np.asarray(message["velocity"], dtype=float)))
                    velocity = preferred
                    if c.use_avoidance:
                        velocity, avoid_flags = safe_velocity(agent.position, agent.velocity, preferred, neighbors, c.avoidance)
                        flags.update(avoid_flags)
                    velocity, keep_flag = self._project_keep_out(agent.position, velocity, fused[:3])
                    if keep_flag:
                        flags.add("keep_out")
                    # The acceleration clamp deliberately runs after keep-out projection.
                    acceleration = (velocity - agent.velocity) / c.control_dt_s
                    acceleration_norm = float(np.linalg.norm(acceleration))
                    if acceleration_norm > c.max_accel_mps2:
                        acceleration *= c.max_accel_mps2 / acceleration_norm
                        flags.add("speed_cap")
                    agent.velocity += acceleration * c.control_dt_s
                    speed = float(np.linalg.norm(agent.velocity))
                    if speed > c.max_speed_mps:
                        agent.velocity *= c.max_speed_mps / speed
                        flags.add("speed_cap")
                    agent.position += agent.velocity * c.control_dt_s
                    if agent.position[2] < 0:
                        agent.position[2] = 0.0
                        agent.velocity[2] = 0.0
                        flags.add("ground")
                    agent.flags = flags
            active_ids = [aid for aid, agent in sorted(self.agents.items()) if agent.active]
            violating: set[str] = set()
            radii = c.avoidance.ellipsoid_radii or (0.12, 0.12, 0.30)
            for aid in active_ids:
                if float(np.linalg.norm(self.agents[aid].position - truth)) < c.keep_out_radius_m:
                    keep_true_entries += 1
                if aid in track_cache and float(np.linalg.norm(self.agents[aid].position - track_cache[aid][0][:3])) < c.keep_out_radius_m:
                    keep_fused_entries += 1
            for index, aid in enumerate(active_ids):
                for peer in active_ids[index + 1:]:
                    distance = float(np.linalg.norm(self.agents[aid].position - self.agents[peer].position))
                    min_pair = min(min_pair, distance)
                    ellipsoid_value = ellipsoid_separation(self.agents[aid].position, self.agents[peer].position, radii)
                    if ellipsoid_value < 1.0 - 1e-12:
                        separation_violations += 1
                        violating.update((aid, peer))
                    if distance < c.avoidance.radius_m - 1e-12:
                        spherical_violations += 1
            if k % c.log_every == 0 or k == n_steps:
                plan_disagreements += int(step_disagreement_count > 0)
                group_mean, group_cov = self._gs_group(t)
                group_error = None
                group_nees = None
                if group_mean is not None:
                    group_error = float(np.linalg.norm(group_mean[:3] - truth))
                    group_nees = float(nees(group_mean[:3] - truth, group_cov[:3, :3]))
                    gs_errors.append((t, group_error))
                    for start, end in c.metric_windows:
                        if start <= t <= end:
                            gs_window[f"{float(start):g}-{float(end):g}"].append(group_error)
                gs_fuser_estimate = None
                gs_fuser_error = None
                if self.gs_fuser is not None:
                    gs_fuser_estimate = self._full_track(self.gs_fuser, t)[0][:3]
                    gs_fuser_error = float(np.linalg.norm(gs_fuser_estimate - truth))
                    gs_fuser_errors.append((t, gs_fuser_error))
                step_row = {
                    "step_index": k,
                    "time_s": t,
                    "target_truth_position_m": truth.tolist(),
                    "gs_group_estimate_m": None if group_mean is None else group_mean[:3].tolist(),
                    "gs_group_error_m": group_error,
                    "gs_group_nees": group_nees,
                    "gs_fuser_estimate_m": None if gs_fuser_estimate is None else gs_fuser_estimate.tolist(),
                    "gs_fuser_error_m": gs_fuser_error,
                    "plan_disagreements_count": int(step_disagreement_count),
                    "plan_disagreement": bool(step_disagreement_count),
                    "agents": [],
                }
                for aid in c.agent_ids:
                    agent = self.agents[aid]
                    if not agent.active:
                        continue
                    fused, covariance, _, source_time = track_cache[aid]
                    local, _ = self._full_track(agent.estimator, t)
                    error = float(np.linalg.norm(fused[:3] - truth))
                    errors[aid].append((t, error))
                    nval = float(nees(fused[:3] - truth, covariance[:3, :3]))
                    ideal_slots = slots_for_alive(truth, command_alive_count.get(aid, len(alive_cache[aid])), c.standoff_radius_m, c.slot_altitude_m, 0.0)
                    ideal = ideal_slots[agent.slot_index] if agent.slot_index < len(ideal_slots) else truth
                    standoff[aid].append((t, float(np.linalg.norm(agent.position - ideal))))
                    angles = sorted(float(np.arctan2(self.agents[x].position[1] - truth[1], self.agents[x].position[0] - truth[0])) % (2.0 * np.pi) for x in active_ids)
                    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)] + ([2.0 * np.pi + angles[0] - angles[-1]] if angles else [])
                    coverage = float(np.degrees(max(gaps))) if gaps else 360.0
                    ages = [min(c.heartbeat_timeout_s, max(0.0, t - float(msg["sent_at"]))) for msg in agent.latest.values()]
                    min_distance = min((float(np.linalg.norm(agent.position - self.agents[x].position)) for x in active_ids if x != aid), default=None)
                    min_ellipsoid = min((ellipsoid_separation(agent.position, self.agents[x].position, radii) for x in active_ids if x != aid), default=0.0)
                    flags = set(agent.flags)
                    if aid in violating:
                        flags.add("separation_violation")
                    step_row["agents"].append(
                        {
                            "agent_id": aid,
                            "active": True,
                            "position_m": agent.position.tolist(),
                            "velocity_mps": agent.velocity.tolist(),
                            "target_estimate_m": local[:3].tolist(),
                            "fused_target_m": fused[:3].tolist(),
                            "fused_cov_trace_m2": float(np.trace(covariance[:3, :3])),
                            "truth_error_m": error,
                            "nees": nval,
                            "slot_index": int(agent.slot_index),
                            "believed_alive": list(alive_cache[aid]),
                            "believed_assignment": {
                                key: int(value)
                                for key, value in sorted(agent.believed_assignment.items())
                            },
                            "oldest_neighbor_age_s": max(ages) if ages else None,
                            "min_neighbor_distance_m": min_distance,
                            "min_ellipsoid_separation": float(min_ellipsoid),
                            "standoff_error_m": float(standoff[aid][-1][1]),
                            "coverage_largest_gap_deg": coverage,
                            "flags": sorted(flags),
                            "target_source_time_s": source_time,
                            "target_age_s": (
                                None if source_time is None else min(
                                    c.heartbeat_timeout_s,
                                    max(0.0, t - source_time),
                                )
                            ),
                        }
                    )
                for aid, agent in sorted(self.agents.items()):
                    if agent.active:
                        continue
                    step_row["agents"].append(
                        {
                            "agent_id": aid,
                            "active": False,
                            "position_m": agent.position.tolist(),
                            "velocity_mps": agent.velocity.tolist(),
                            "target_estimate_m": None,
                            "fused_target_m": None,
                            "fused_cov_trace_m2": None,
                            "truth_error_m": None,
                            "nees": None,
                            "slot_index": 0,
                            "believed_alive": [],
                            "believed_assignment": {},
                            "oldest_neighbor_age_s": None,
                            "min_neighbor_distance_m": None,
                            "min_ellipsoid_separation": 0.0,
                            "standoff_error_m": None,
                            "coverage_largest_gap_deg": None,
                            "flags": sorted(agent.flags),
                            "target_source_time_s": None,
                            "target_age_s": None,
                        }
                    )
                steps.append(step_row)
        event_dicts = []
        for event in sorted(self._events, key=lambda item: (item.send_time, item.sender, item.receiver, item.seq)):
            pending = event.receive_time is not None and float(event.receive_time) > c.duration_s + 1e-12
            receive = None if pending or event.receive_time is None else float(event.receive_time)
            loss_reason = None if pending else event.loss_reason
            row = {
                "sender": event.sender,
                "receiver": event.receiver,
                "seq": int(event.seq),
                "kind": self._event_kinds.get(id(event), "state"),
                "send_time": float(event.send_time),
                "receive_time": receive,
                "loss_reason": loss_reason,
            }
            if pending:
                row["outcome"] = "pending"
            elif receive is not None:
                row["outcome"] = "received"
            elif loss_reason is not None:
                row["outcome"] = "lost"
            else:
                row["outcome"] = "pending"
            event_dicts.append(row)
        per_agent = {}
        for aid, values in errors.items():
            scored = np.asarray([value for time, value in values if time >= 2.0 - 1e-12], dtype=float)
            per_agent[aid] = {
                "p95_error_m": float(np.percentile(scored, 95)) if scored.size else None,
                "max_error_m": float(np.max(scored)) if scored.size else None,
                "failed": bool(
                    scored.size
                    and (np.percentile(scored, 95) > 0.25 or np.max(scored) > 0.5)
                ),
            }
        gs_scored = np.asarray([value for time, value in gs_errors if time >= 2.0 - 1e-12], dtype=float)
        window_metrics = {key: {"p95_error_m": float(np.percentile(values, 95)) if values else None, "max_error_m": float(np.max(values)) if values else None} for key, values in gs_window.items()}
        traffic = {
            key: {
                "messages": int(value["messages"]),
                "bytes": int(value["bytes"]),
                "messages_per_second": float(value["messages"] / c.duration_s),
                "bytes_per_second": float(value["bytes"] / c.duration_s),
                "message_bytes": int(self._traffic_by_kind[key]),
            }
            for key, value in self._traffic_kind.items()
        }
        gs_fuser_scored = np.asarray([value for time, value in gs_fuser_errors if time >= 2.0 - 1e-12], dtype=float)
        summary = {
            "per_agent": per_agent,
            "min_pairwise_separation_m": min_pair if np.isfinite(min_pair) else None,
            "separation_violations_count": int(separation_violations),
            "spherical_violations_count": int(spherical_violations),
            "keep_out_entries": int(keep_true_entries),
            "keep_out_fused_estimate_entries": int(keep_fused_entries),
            "role_changes_count": int(role_changes),
            "plan_disagreements_count": int(plan_disagreements),
            "plan_disagreement_fraction": float(plan_disagreements / max(1, len(steps))),
            "gs_reject_count": int(self._gs_reject_count),
            "gs_group_error_p95_m": float(np.percentile(gs_scored, 95)) if gs_scored.size else None,
            "gs_group_error_max_m": float(np.max(gs_scored)) if gs_scored.size else None,
            "gs_group_error_windows": window_metrics,
            "gs_fuser_error_p95_m": float(np.percentile(gs_fuser_scored, 95)) if gs_fuser_scored.size else None,
            "gs_fuser_error_max_m": float(np.max(gs_fuser_scored)) if gs_fuser_scored.size else None,
            "traffic": traffic,
            "traffic_documentation": (
                "Counts are per directed route, in messages and bytes. Message "
                "sizes are recorded in each kind bucket. In ground mode, "
                "believed_alive records the ground station's current plan. After "
                "a leaving message is received, an agent stops sending state to "
                "that peer."
            ),
        }
        return CoopRun(
            {
                "schema": "zephyr-s7-coop-run-2",
                "seed": int(c.seed),
                "simulated": True,
                "config": _config_dict(c),
                "model_provenance": self.model_provenance(),
                "steps": steps,
                "link_events": event_dicts,
                "summary": summary,
                "evidence_boundary": (
                    "Simulated only. Kinematic point-mass agents, abstract links, "
                    "literature-derived thresholds, and no hardware."
                ),
            }
        )


def _config_dict(config: CoopSwarmConfig) -> dict[str, Any]:
    result = asdict(config)
    result["avoidance"] = asdict(config.avoidance)
    if not config.share_slot_claims:
        # Original behaviour: write the same config block as before the option existed.
        result.pop("share_slot_claims")
    result["dropouts"] = [list(row) for row in config.dropouts]
    result["target_params"] = [list(row) for row in config.target_params]
    result["gs_outage"] = [list(row) for row in config.gs_outage]
    result["metric_windows"] = [list(row) for row in config.metric_windows]
    period = 1.0 / config.sensor_rate_hz
    result["observation_phases_s"] = [0.0] + [
        float(
            np.random.default_rng(
                np.random.SeedSequence([config.seed, 3000 + i])
            ).uniform(0.0, period)
        )
        for i in range(1, config.n_agents)
    ]
    return result


def run(config: CoopSwarmConfig | None = None, **kwargs) -> CoopRun:
    if config is None:
        config = CoopSwarmConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either config or keyword configuration")
    return CoopSwarmSimulator(config).run()


def to_replay_v1(run_value: CoopRun | Mapping[str, Any]) -> dict[str, Any]:
    raw = run_value.as_dict() if isinstance(run_value, CoopRun) else dict(run_value)
    config = raw.get("config", {})
    steps = []
    for step in raw["steps"]:
        rows = []
        for agent in step["agents"]:
            rows.append(
                {
                    "agent_id": agent["agent_id"],
                    "profile_id": f"{agent['agent_id']}-synthetic-profile",
                    "active": bool(agent["active"]),
                    "position_m": agent["position_m"],
                    "velocity_mps": agent["velocity_mps"],
                    "target_estimate_m": agent["target_estimate_m"],
                    "fused_target_m": agent["fused_target_m"],
                    "target_source_time_s": agent.get("target_source_time_s"),
                    "target_age_s": agent.get("target_age_s"),
                    "neighbor_count": max(
                        0, len(agent.get("believed_alive", [])) - 1
                    ),
                    "min_neighbor_distance_m": agent.get("min_neighbor_distance_m"),
                    "constraint_flags": agent.get("flags", []),
                }
            )
        steps.append(
            {
                "step_index": int(step["step_index"]),
                "time_s": float(step["time_s"]),
                "target_position_m": step["target_truth_position_m"],
                "active_count": sum(1 for row in rows if row["active"]),
                "agents": rows,
            }
        )
    links = []
    for event in raw.get("link_events", []):
        receive = event.get("receive_time")
        row = {
            "sender": event["sender"],
            "receiver": event["receiver"],
            "seq": int(event["seq"]),
            "send_time": event["send_time"],
            "receive_time": receive,
            "duplicate": False,
            "out_of_order": False,
            "loss_reason": event.get("loss_reason"),
            "packet_age": None if receive is None else receive - event["send_time"],
        }
        if receive is not None:
            row["outcome"] = "received"
        elif event.get("loss_reason") is not None:
            row["outcome"] = "lost"
        links.append(row)
    n_agents = int(config.get("n_agents", len(steps[0]["agents"])))
    profiles = [default_hardware_profile(i).as_dict() for i in range(n_agents)]
    return {
        "schema": "zephyr-s7-swarm-run-1",
        "seed": int(raw["seed"]),
        "dt_s": float(config.get("control_dt_s", 0.05))
        * int(config.get("log_every", 1)),
        "steps": steps,
        "link_events": links,
        "agent_profiles": profiles,
        "scenario_id": "s7-coop-replay-v1",
        "evidence_boundary": (
            "Synthetic deterministic replay over an abstract point-mass link; "
            "hardware values are inputs."
        ),
        "generation_command": "sim.coop.simulator",
        "status": "synthetic deterministic replay; not flight performance",
        "command_authority": "replay only",
        "failsafe": "synthetic constraints only; live failsafe unavailable",
    }
