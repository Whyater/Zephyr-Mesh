"""Deterministic multi-agent coordination fixtures for simulator stage S7.

This module adds the swarm coordination layer on top of the existing S2 rigid
body and S3 link fixtures.  Agent motion here is a bounded point-mass
coordination model: it is useful for testing message age, dropouts, formation
logic, and safety constraints, but it is not a replacement for the 6-DOF
vehicle dynamics or a measured radio model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from sim.link import LinkConfig, PacketEvent, SimulatedLink
from sim.hardware import HardwareProfile, default_hardware_profile

_TARGET_ID = "__target__"
MAX_KEEP_OUT_SPEED_MPS = 50.0


def _vector(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(v) for v in array)


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


@dataclass(frozen=True)
class AgentSpec:
    """Initial state and formation slot for one simulated vehicle."""

    agent_id: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slot_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hardware_profile: HardwareProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id or self.agent_id == _TARGET_ID:
            raise ValueError("agent_id must be a non-empty string other than the reserved target id")
        object.__setattr__(self, "position_m", _vector(self.position_m, name="position_m"))
        object.__setattr__(self, "velocity_mps", _vector(self.velocity_mps, name="velocity_mps"))
        object.__setattr__(self, "slot_offset_m", _vector(self.slot_offset_m, name="slot_offset_m"))
        if self.hardware_profile is None:
            object.__setattr__(self, "hardware_profile", default_hardware_profile(self.agent_id))
        elif not isinstance(self.hardware_profile, HardwareProfile):
            raise ValueError("hardware_profile must be a HardwareProfile")


@dataclass(frozen=True)
class KeepOutSphere:
    """A spherical region with an optional constant-velocity trajectory."""

    center_m: tuple[float, float, float]
    radius_m: float
    label: str = "obstacle"
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_m", _vector(self.center_m, name="center_m"))
        velocity = _vector(self.velocity_mps, name="velocity_mps")
        if any(abs(component) > MAX_KEEP_OUT_SPEED_MPS for component in velocity):
            raise ValueError(f"velocity_mps components must be within ±{MAX_KEEP_OUT_SPEED_MPS:g}")
        object.__setattr__(self, "velocity_mps", velocity)
        radius = float(self.radius_m)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius_m must be finite and nonnegative")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        object.__setattr__(self, "radius_m", radius)

    def center_at(self, time_s: float) -> tuple[float, float, float]:
        """Return the deterministic center at simulation time ``time_s``."""
        time = float(time_s)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        return tuple(float(center + velocity * time) for center, velocity in zip(self.center_m, self.velocity_mps))


@dataclass(frozen=True)
class SwarmConfig:
    """Configuration for a reproducible, bounded swarm coordination run.

    ``target_link`` and ``neighbor_link`` deliberately use the existing S3
    abstract packet model.  They can be replaced by replayed measured traces
    later, but no field here claims ESP-NOW PHY fidelity.
    """

    agents: tuple[AgentSpec, ...]
    dt_s: float = 0.05
    target_position_m: tuple[float, float, float] = (0.0, 0.0, 1.5)
    target_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_link: LinkConfig = LinkConfig(loss_model="none")
    neighbor_link: LinkConfig = LinkConfig(loss_model="none")
    # Optional per-agent target-link overrides model asymmetric reception.
    target_link_overrides: tuple[tuple[str, LinkConfig], ...] = ()
    seed: int = 0
    position_gain: float = 1.8
    velocity_gain: float = 1.1
    max_speed_mps: float = 2.0
    max_acceleration_mps2: float = 4.0
    min_separation_m: float = 0.35
    target_keepout_radius_m: float = 0.45
    keep_out_spheres: tuple[KeepOutSphere, ...] = ()
    max_estimate_age_s: float = 1.0
    ground_plane_z_m: float = 0.0
    constraint_iterations: int = 4

    def __post_init__(self) -> None:
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("at least one agent is required")
        if len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("agent ids must be unique")
        object.__setattr__(self, "agents", agents)
        if not isinstance(self.target_link, LinkConfig) or not isinstance(self.neighbor_link, LinkConfig):
            raise ValueError("target_link and neighbor_link must be LinkConfig values")
        object.__setattr__(self, "target_position_m", _vector(self.target_position_m, name="target_position_m"))
        object.__setattr__(self, "target_velocity_mps", _vector(self.target_velocity_mps, name="target_velocity_mps"))
        dt = float(self.dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", dt)
        for name in ("position_gain", "velocity_gain", "max_speed_mps", "max_acceleration_mps2",
                     "min_separation_m", "target_keepout_radius_m", "max_estimate_age_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        ground = float(self.ground_plane_z_m)
        if not np.isfinite(ground):
            raise ValueError("ground_plane_z_m must be finite")
        object.__setattr__(self, "ground_plane_z_m", ground)
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed or int(self.seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        if isinstance(self.constraint_iterations, (bool, np.bool_)) or int(self.constraint_iterations) < 1:
            raise ValueError("constraint_iterations must be a positive integer")
        object.__setattr__(self, "constraint_iterations", int(self.constraint_iterations))
        overrides = tuple(self.target_link_overrides)
        override_ids = [agent_id for agent_id, _ in overrides]
        if len(set(override_ids)) != len(override_ids):
            raise ValueError("target link override ids must be unique")
        agent_ids = {agent.agent_id for agent in agents}
        if any(agent_id not in agent_ids for agent_id in override_ids):
            raise ValueError("target link override id must name an agent")
        if any(not isinstance(link, LinkConfig) for _, link in overrides):
            raise ValueError("target link overrides must contain LinkConfig values")
        object.__setattr__(self, "target_link_overrides", overrides)
        zones = tuple(self.keep_out_spheres)
        if not all(isinstance(zone, KeepOutSphere) for zone in zones):
            raise ValueError("keep_out_spheres must contain KeepOutSphere values")
        object.__setattr__(self, "keep_out_spheres", zones)


@dataclass(frozen=True)
class NeighborMessage:
    """Latest received state advertisement from one neighbor."""

    sender_id: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    target_estimate_m: tuple[float, float, float] | None
    target_source_time_s: float | None
    receive_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentState:
    """Serializable state and data-quality indicators at one simulation step."""

    agent_id: str
    profile_id: str
    active: bool
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    target_estimate_m: tuple[float, float, float] | None
    fused_target_m: tuple[float, float, float] | None
    target_source_time_s: float | None
    target_age_s: float | None
    neighbor_count: int
    min_neighbor_distance_m: float | None
    constraint_flags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SwarmStep:
    """One immutable observation of a swarm simulation step."""

    step_index: int
    time_s: float
    target_position_m: tuple[float, float, float]
    active_count: int
    agents: tuple[AgentState, ...]
    keep_out_spheres: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["agents"] = [agent.as_dict() for agent in self.agents]
        return result


@dataclass(frozen=True)
class SwarmRun:
    """Machine-readable result of a seeded run."""

    schema: str
    seed: int
    dt_s: float
    steps: tuple[SwarmStep, ...]
    link_events: tuple[dict[str, Any], ...]
    agent_profiles: tuple[dict[str, Any], ...] = ()
    scenario_id: str = "s7-swarm-run"
    evidence_boundary: str = (
        "Deterministic point-mass coordination fixture over an abstract packet link; "
        "hardware values are scenario inputs, not measured flight performance."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "dt_s": self.dt_s,
            "steps": [step.as_dict() for step in self.steps],
            "link_events": list(self.link_events),
            "agent_profiles": list(self.agent_profiles),
            "scenario_id": self.scenario_id,
            "evidence_boundary": self.evidence_boundary,
        }


@dataclass
class _AgentRuntime:
    spec: AgentSpec
    position: np.ndarray
    velocity: np.ndarray
    active: bool = True
    target_estimate: np.ndarray | None = None
    target_source_time_s: float | None = None
    target_receive_time_s: float | None = None
    neighbors: dict[str, NeighborMessage] | None = None

    def __post_init__(self) -> None:
        if self.neighbors is None:
            self.neighbors = {}


class SwarmSimulator:
    """Run deterministic multi-agent formation and tracking scenarios.

    The simulator keeps message transport, estimator state, and constrained
    coordination separate.  ``run`` accepts a dropout schedule so failure and
    recovery scenarios are reproducible without changing source code.
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.time_s = 0.0
        self.step_index = 0
        self.target_position = np.asarray(config.target_position_m, dtype=float).copy()
        self.target_velocity = np.asarray(config.target_velocity_mps, dtype=float).copy()
        self._agents: dict[str, _AgentRuntime] = {
            agent.agent_id: _AgentRuntime(
                spec=agent,
                position=np.asarray(agent.position_m, dtype=float).copy(),
                velocity=np.asarray(agent.velocity_mps, dtype=float).copy(),
            )
            for agent in config.agents
        }
        routes = [(_TARGET_ID, agent.agent_id) for agent in config.agents]
        routes.extend(
            (sender.agent_id, receiver.agent_id)
            for sender in config.agents
            for receiver in config.agents
            if sender.agent_id != receiver.agent_id
        )
        self._links: dict[tuple[str, str], SimulatedLink] = {}
        self._payloads: dict[tuple[str, str], dict[int, Any]] = {}
        for route_index, route in enumerate(routes):
            sender, receiver = route
            if sender == _TARGET_ID:
                override = dict(config.target_link_overrides).get(receiver)
                link_config = config.target_link if override is None else override
            else:
                link_config = config.neighbor_link
            route_seed = config.seed + 104729 * route_index
            self._links[route] = SimulatedLink(link_config, seed=route_seed)
            self._payloads[route] = {}
        self._all_events: list[PacketEvent] = []

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(self._agents)

    @property
    def link_events(self) -> tuple[PacketEvent, ...]:
        """All attempted packet events, sorted independently of dict order."""
        return tuple(sorted(self._all_events, key=lambda event: (
            event.send_time, event.sender, event.receiver, event.seq
        )))

    def set_agent_active(self, agent_id: str, active: bool) -> None:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        self._agents[agent_id].active = bool(active)

    def _send(self, sender: str, receiver: str, seq: int, payload: Any) -> PacketEvent:
        route = (sender, receiver)
        event = self._links[route].send(sender, receiver, seq, self.time_s, payload=payload)
        self._payloads[route][seq] = payload
        self._all_events.append(event)
        return event

    def _apply_delivery(self, route: tuple[str, str], event: PacketEvent) -> None:
        payload = self._payloads[route].pop(event.seq, None)
        if payload is None:
            return
        sender, receiver = route
        if receiver not in self._agents or not self._agents[receiver].active:
            return
        agent = self._agents[receiver]
        if sender == _TARGET_ID:
            source_time = float(payload["source_time_s"])
            if agent.target_source_time_s is None or source_time > agent.target_source_time_s:
                agent.target_estimate = np.asarray(payload["position_m"], dtype=float)
                agent.target_source_time_s = source_time
                agent.target_receive_time_s = float(event.receive_time)
            return
        if sender not in self._agents:
            return
        source_time_value = payload.get("target_source_time_s")
        source_time = None if source_time_value is None else float(source_time_value)
        previous = agent.neighbors.get(sender)
        if previous is not None and source_time is not None and previous.target_source_time_s is not None:
            if source_time < previous.target_source_time_s:
                return
        agent.neighbors[sender] = NeighborMessage(
            sender_id=sender,
            position_m=_vector(payload["position_m"], name="neighbor position"),
            velocity_mps=_vector(payload["velocity_mps"], name="neighbor velocity"),
            target_estimate_m=None if payload.get("target_estimate_m") is None else _vector(
                payload["target_estimate_m"], name="neighbor target estimate"
            ),
            target_source_time_s=source_time,
            receive_time_s=float(event.receive_time),
        )

    def _advance_links(self) -> None:
        for route in sorted(self._links):
            delivered = self._links[route].advance(self.time_s)
            for event in delivered:
                self._apply_delivery(route, event)

    def _send_messages(self) -> None:
        seq = self.step_index
        for agent_id in self.agent_ids:
            agent = self._agents[agent_id]
            if not agent.active:
                continue
            self._send(
                _TARGET_ID,
                agent_id,
                seq,
                {
                    "source_time_s": self.time_s,
                    "position_m": tuple(self.target_position.tolist()),
                    "velocity_mps": tuple(self.target_velocity.tolist()),
                },
            )
        for sender_id in self.agent_ids:
            sender = self._agents[sender_id]
            if not sender.active:
                continue
            for receiver_id in self.agent_ids:
                if sender_id == receiver_id or not self._agents[receiver_id].active:
                    continue
                self._send(
                    sender_id,
                    receiver_id,
                    seq,
                    {
                        "source_time_s": self.time_s,
                        "position_m": tuple(sender.position.tolist()),
                        "velocity_mps": tuple(sender.velocity.tolist()),
                        "target_estimate_m": None if sender.target_estimate is None else tuple(
                            sender.target_estimate.tolist()
                        ),
                        "target_source_time_s": sender.target_source_time_s,
                    },
                )

    def _fused_target(self, agent: _AgentRuntime) -> np.ndarray | None:
        estimates: list[np.ndarray] = []
        if agent.target_estimate is not None and agent.target_source_time_s is not None:
            if self.time_s - agent.target_source_time_s <= self.config.max_estimate_age_s:
                estimates.append(agent.target_estimate.copy())
        for message in agent.neighbors.values():
            source_time = message.target_source_time_s
            if message.target_estimate_m is None or source_time is None:
                continue
            if self.time_s - source_time <= self.config.max_estimate_age_s:
                estimates.append(np.asarray(message.target_estimate_m, dtype=float))
        if not estimates:
            return None
        return np.mean(np.stack(estimates), axis=0)

    @staticmethod
    def _limited(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm == 0.0:
            return vector
        return vector * (limit / norm)

    @staticmethod
    def _separation_direction(delta: np.ndarray, *, index: int) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm > 1e-12:
            return delta / norm
        axis = index % 3
        direction = np.zeros(3, dtype=float)
        direction[axis] = 1.0
        return direction

    def _constrain(
        self,
        candidates: dict[str, np.ndarray],
        current: dict[str, np.ndarray],
        keep_out_spheres: tuple[KeepOutSphere, ...] | None = None,
    ) -> dict[str, set[str]]:
        flags = {agent_id: set() for agent_id in self.agent_ids}
        active_ids = [agent_id for agent_id in self.agent_ids if self._agents[agent_id].active]
        for _ in range(self.config.constraint_iterations):
            for pair_index, left_id in enumerate(active_ids):
                for right_id in active_ids[pair_index + 1:]:
                    delta = candidates[left_id] - candidates[right_id]
                    distance = float(np.linalg.norm(delta))
                    minimum = self.config.min_separation_m
                    if distance >= minimum:
                        continue
                    direction = self._separation_direction(delta, index=pair_index)
                    correction = 0.5 * (minimum - distance) * direction
                    candidates[left_id] += correction
                    candidates[right_id] -= correction
                    flags[left_id].add("separation")
                    flags[right_id].add("separation")
            zones = list(self.config.keep_out_spheres if keep_out_spheres is None else keep_out_spheres)
            zones.append(KeepOutSphere(
                tuple(self.target_position.tolist()),
                self.config.target_keepout_radius_m,
                "target_keepout",
            ))
            for agent_id in active_ids:
                for zone in zones:
                    delta = candidates[agent_id] - np.asarray(zone.center_m, dtype=float)
                    distance = float(np.linalg.norm(delta))
                    if distance >= zone.radius_m:
                        continue
                    direction = self._separation_direction(delta, index=self.agent_ids.index(agent_id))
                    candidates[agent_id] = np.asarray(zone.center_m, dtype=float) + direction * zone.radius_m
                    flags[agent_id].add(zone.label)
                if candidates[agent_id][2] < self.config.ground_plane_z_m:
                    candidates[agent_id][2] = self.config.ground_plane_z_m
                    flags[agent_id].add("ground")
        return flags

    def _snapshot(
        self,
        flags: Mapping[str, set[str]],
        keep_out_spheres: tuple[KeepOutSphere, ...] = (),
    ) -> SwarmStep:
        active_ids = [agent_id for agent_id in self.agent_ids if self._agents[agent_id].active]
        states: list[AgentState] = []
        for agent_id in self.agent_ids:
            agent = self._agents[agent_id]
            distances = [
                float(np.linalg.norm(agent.position - self._agents[other_id].position))
                for other_id in active_ids if other_id != agent_id
            ]
            target_age = None
            if agent.target_source_time_s is not None:
                target_age = max(0.0, self.time_s - agent.target_source_time_s)
            states.append(AgentState(
                agent_id=agent_id,
                profile_id=agent.spec.hardware_profile.profile_id,
                active=agent.active,
                position_m=tuple(float(v) for v in agent.position),
                velocity_mps=tuple(float(v) for v in agent.velocity),
                target_estimate_m=None if agent.target_estimate is None else tuple(float(v) for v in agent.target_estimate),
                fused_target_m=None if self._fused_target(agent) is None else tuple(float(v) for v in self._fused_target(agent)),
                target_source_time_s=agent.target_source_time_s,
                target_age_s=target_age,
                neighbor_count=0 if agent.neighbors is None else len(agent.neighbors),
                min_neighbor_distance_m=None if not distances else min(distances),
                constraint_flags=tuple(sorted(flags.get(agent_id, set()))),
            ))
        return SwarmStep(
            step_index=self.step_index,
            time_s=self.time_s,
            target_position_m=tuple(float(v) for v in self.target_position),
            active_count=len(active_ids),
            agents=tuple(states),
            keep_out_spheres=tuple({
                "center_m": zone.center_m,
                "radius_m": zone.radius_m,
                "label": zone.label,
            } for zone in keep_out_spheres),
        )

    def step(self) -> SwarmStep:
        """Advance one fixed-duration step and return its immutable snapshot."""
        # Deliver old packets first, then include zero-delay packets from this step.
        self._advance_links()
        self._send_messages()
        self._advance_links()

        current = {agent_id: self._agents[agent_id].position.copy() for agent_id in self.agent_ids}
        candidates: dict[str, np.ndarray] = {}
        for agent_id in self.agent_ids:
            agent = self._agents[agent_id]
            if not agent.active:
                candidates[agent_id] = agent.position.copy()
                continue
            estimate = self._fused_target(agent)
            if estimate is None:
                acceleration = -self.config.velocity_gain * agent.velocity
            else:
                desired = estimate + np.asarray(agent.spec.slot_offset_m, dtype=float)
                acceleration = (
                    self.config.position_gain * (desired - agent.position)
                    - self.config.velocity_gain * agent.velocity
                )
            acceleration = self._limited(acceleration, self.config.max_acceleration_mps2)
            velocity = self._limited(agent.velocity + acceleration * self.config.dt_s, self.config.max_speed_mps)
            candidates[agent_id] = agent.position + velocity * self.config.dt_s

        moving_keep_out_spheres = tuple(
            KeepOutSphere(zone.center_at(self.time_s), zone.radius_m, zone.label, zone.velocity_mps)
            for zone in self.config.keep_out_spheres
        )
        flags = self._constrain(candidates, current, moving_keep_out_spheres)
        for agent_id in self.agent_ids:
            agent = self._agents[agent_id]
            if not agent.active:
                continue
            agent.velocity = self._limited(
                (candidates[agent_id] - agent.position) / self.config.dt_s,
                self.config.max_speed_mps,
            )
            agent.position = candidates[agent_id]

        snapshot = self._snapshot(flags, moving_keep_out_spheres)
        self.target_position = self.target_position + self.target_velocity * self.config.dt_s
        self.time_s += self.config.dt_s
        self.step_index += 1
        return snapshot

    def run(
        self,
        steps: int,
        *,
        dropout_schedule: Mapping[int, Mapping[str, bool] | Iterable[str]] | None = None,
    ) -> SwarmRun:
        """Run ``steps`` fixed steps with optional deterministic agent dropouts.

        A schedule value can be ``{"drone-2": False}`` for explicit state, or
        an iterable of ids to mark inactive at that step.  No schedule entry
        is inferred from link loss: transport failure and vehicle dropout are
        kept as separate events.
        """
        if isinstance(steps, (bool, np.bool_)) or int(steps) != steps or int(steps) < 0:
            raise ValueError("steps must be a nonnegative integer")
        schedule = dropout_schedule or {}
        records: list[SwarmStep] = []
        for _ in range(int(steps)):
            if self.step_index in schedule:
                change = schedule[self.step_index]
                if isinstance(change, Mapping):
                    for agent_id, active in change.items():
                        self.set_agent_active(agent_id, bool(active))
                else:
                    for agent_id in change:
                        self.set_agent_active(agent_id, False)
            records.append(self.step())
        return SwarmRun(
            schema="zephyr-s7-swarm-run-1",
            seed=self.config.seed,
            dt_s=self.config.dt_s,
            steps=tuple(records),
            link_events=tuple(event.as_dict() for event in self.link_events),
            agent_profiles=tuple(
                self._agents[agent_id].spec.hardware_profile.as_dict()
                for agent_id in self.agent_ids
            ),
            scenario_id="s7-swarm-run",
        )
