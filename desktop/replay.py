"""Pure-Python replay model shared by the desktop preview and tests.

The model reads the canonical S7 event-log projection. It deliberately exposes
only deterministic replay state. It never opens a radio, camera, controller,
or vehicle adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ReplayFormatError(ValueError):
    """Raised when a replay file does not satisfy the S7 event-log contract."""


@dataclass(frozen=True)
class LinkStats:
    """Observed link metrics for one receiver in a replay manifest.

    These values are calculated from the event log, so the desktop surfaces can
    show a useful link diagnostic without pretending that a live radio is
    attached.  ``None`` means the manifest contained no packet-age samples.
    """

    receiver: str
    event_count: int
    loss_count: int
    mean_delay_ms: float | None
    max_delay_ms: float | None

    @property
    def loss_pct(self) -> float:
        if self.event_count <= 0:
            return 0.0
        return self.loss_count / self.event_count * 100.0


@dataclass(frozen=True)
class HardwareProfile:
    """A compact, typed projection of an agent's synthetic parts manifest."""

    profile_id: str
    motor_part_id: str
    motor_max_torque_nm: float
    motor_max_power_w: float
    motor_max_rpm: float
    motor_nominal_voltage_v: float
    propeller_part_id: str
    propeller_diameter_m: float
    propeller_pitch_m: float
    propeller_max_rpm: float
    motor_count: int
    arm_length_m: float
    frame_mass_kg: float
    battery_capacity_wh: float
    total_mass_kg: float
    estimated_max_thrust_n: float


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    active: bool
    confidence: float
    neighbor_count: int
    min_neighbor_distance_m: float
    constraint_flags: tuple[str, ...]
    profile_id: str
    battery_pct: float | None = None
    link_delay_ms: float | None = None


@dataclass(frozen=True)
class ReplayFrame:
    step_index: int
    time_s: float
    target_position_m: tuple[float, float, float]
    active_count: int
    agents: tuple[AgentSnapshot, ...]


class ReplayModel:
    """A bounded, deterministic cursor over ``runs/s7-swarm/run.json``."""

    def __init__(self, document: Mapping[str, Any], *, source_path: Path | None = None) -> None:
        self.schema = _required_str(document, "schema")
        self.scenario_id = _required_str(document, "scenario_id")
        self.seed = _required_int(document, "seed")
        self.dt_s = _required_float(document, "dt_s")
        self.status = _required_str(document, "status")
        self.command_authority = _required_str(document, "command_authority")
        self.failsafe = _required_str(document, "failsafe")
        self.evidence_boundary = _required_str(document, "evidence_boundary")
        self.agent_profiles = _decode_profiles(document.get("agent_profiles", []))
        self._link_stats = _decode_link_stats(document.get("link_events", []))
        self.source_path = source_path
        raw_steps = document.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ReplayFormatError("run.json must contain a non-empty steps list")
        self.frames = tuple(_decode_frame(step) for step in raw_steps)
        if any(len(frame.agents) == 0 for frame in self.frames):
            raise ReplayFormatError("every replay frame must contain at least one agent")
        self._index = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayModel":
        source = Path(path)
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayFormatError(f"could not read replay {source}: {exc}") from exc
        if not isinstance(document, dict):
            raise ReplayFormatError("replay root must be a JSON object")
        return cls(document, source_path=source)

    @classmethod
    def from_scenario_document(cls, document: Mapping[str, Any]) -> "ReplayModel":
        """Load a validated interactive scenario through the replay contract.

        The conversion is lazy-imported to keep the replay model usable by
        existing frozen fixtures without changing the checked-in replay file.
        """
        try:
            from sim.scenario_runner import to_replay_document
            replay = to_replay_document(document)
        except (TypeError, ValueError) as exc:
            raise ReplayFormatError(f"invalid scenario document: {exc}") from exc
        return cls(replay)

    @property
    def index(self) -> int:
        return self._index

    @property
    def frame(self) -> ReplayFrame:
        return self.frames[self._index]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def progress(self) -> float:
        if len(self.frames) <= 1:
            return 1.0
        return self._index / (len(self.frames) - 1)

    def step(self, delta: int = 1) -> ReplayFrame:
        if not isinstance(delta, int):
            raise TypeError("delta must be an integer")
        self._index = max(0, min(self._index + delta, len(self.frames) - 1))
        return self.frame

    def reset(self) -> ReplayFrame:
        self._index = 0
        return self.frame

    def summary(self) -> dict[str, Any]:
        """Return labels appropriate for a replay-only operator surface."""
        frame = self.frame
        losses = [agent.confidence for agent in frame.agents]
        return {
            "frame": f"{self._index + 1}/{len(self.frames)}",
            "time_s": frame.time_s,
            "active_count": frame.active_count,
            "agent_count": len(frame.agents),
            "min_confidence": min(losses),
            "command_authority": self.command_authority,
            "failsafe": self.failsafe,
            "status": self.status,
            "link_event_count": sum(item.event_count for item in self._link_stats.values()),
            "link_loss_pct": _aggregate_loss_pct(self._link_stats.values()),
            "profile_count": len(self.agent_profiles),
        }

    def link_stats_for(self, agent_id: str) -> LinkStats | None:
        """Return replay-derived link metrics for ``agent_id`` when available."""

        return self._link_stats.get(agent_id)

    def profile_for(self, agent_id: str) -> HardwareProfile | None:
        """Return the immutable synthetic parts profile for ``agent_id``."""

        agent = next((item for item in self.frame.agents if item.agent_id == agent_id), None)
        if agent is None:
            return None
        return self.agent_profiles.get(agent.profile_id)


def filter_agents(agents: Sequence[AgentSnapshot], query: str) -> tuple[AgentSnapshot, ...]:
    """Filter a fleet by stable ID or profile ID using case-insensitive matching."""

    needle = query.strip().casefold()
    if not needle:
        return tuple(agents)
    return tuple(
        agent
        for agent in agents
        if needle in agent.agent_id.casefold() or needle in agent.profile_id.casefold()
    )


def _decode_frame(raw: Any) -> ReplayFrame:
    if not isinstance(raw, Mapping):
        raise ReplayFormatError("each step must be a JSON object")
    agents_raw = raw.get("agents")
    if not isinstance(agents_raw, list):
        raise ReplayFormatError("each step must contain an agents list")
    agents = tuple(_decode_agent(agent) for agent in agents_raw)
    target = _vector(raw.get("target_position_m"), "target_position_m")
    return ReplayFrame(
        step_index=_required_int(raw, "step_index"),
        time_s=_required_float(raw, "time_s"),
        target_position_m=target,
        active_count=_required_int(raw, "active_count"),
        agents=agents,
    )


def _decode_profiles(raw: Any) -> dict[str, HardwareProfile]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ReplayFormatError("agent_profiles must be a list when present")
    profiles: dict[str, HardwareProfile] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReplayFormatError("each agent profile must be an object")
        profile_id = _required_str(item, "profile_id")
        motor = item.get("motor")
        propeller = item.get("propeller")
        if not isinstance(motor, Mapping) or not isinstance(propeller, Mapping):
            raise ReplayFormatError(f"profile {profile_id} must contain motor and propeller objects")
        if profile_id in profiles:
            raise ReplayFormatError(f"duplicate agent profile {profile_id}")
        profiles[profile_id] = HardwareProfile(
            profile_id=profile_id,
            motor_part_id=_required_str(motor, "part_id"),
            motor_max_torque_nm=_required_float(motor, "max_torque_nm"),
            motor_max_power_w=_required_float(motor, "max_power_w"),
            motor_max_rpm=_required_float(motor, "max_rpm"),
            motor_nominal_voltage_v=_required_float(motor, "nominal_voltage_v"),
            propeller_part_id=_required_str(propeller, "part_id"),
            propeller_diameter_m=_required_float(propeller, "diameter_m"),
            propeller_pitch_m=_required_float(propeller, "pitch_m"),
            propeller_max_rpm=_required_float(propeller, "max_rpm"),
            motor_count=_required_int(item, "motor_count"),
            arm_length_m=_required_float(item, "arm_length_m"),
            frame_mass_kg=_required_float(item, "frame_mass_kg"),
            battery_capacity_wh=_required_float(item, "battery_capacity_wh"),
            total_mass_kg=_required_float(item, "total_mass_kg"),
            estimated_max_thrust_n=_required_float(item, "estimated_max_thrust_n"),
        )
    return profiles


def _decode_link_stats(raw: Any) -> dict[str, LinkStats]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ReplayFormatError("link_events must be a list when present")
    counts: dict[str, int] = {}
    losses: dict[str, int] = {}
    delays: dict[str, list[float]] = {}
    for event in raw:
        if not isinstance(event, Mapping):
            raise ReplayFormatError("each link event must be an object")
        receiver = event.get("receiver")
        if not isinstance(receiver, str) or not receiver:
            raise ReplayFormatError("each link event must have a receiver")
        counts[receiver] = counts.get(receiver, 0) + 1
        if event.get("loss_reason") is not None:
            losses[receiver] = losses.get(receiver, 0) + 1
        packet_age = event.get("packet_age")
        if packet_age is not None:
            if isinstance(packet_age, bool) or not isinstance(packet_age, (int, float)) or packet_age < 0:
                raise ReplayFormatError("packet_age must be a non-negative number when present")
            delays.setdefault(receiver, []).append(float(packet_age) * 1000.0)
    return {
        receiver: LinkStats(
            receiver=receiver,
            event_count=count,
            loss_count=losses.get(receiver, 0),
            mean_delay_ms=(sum(delays[receiver]) / len(delays[receiver])) if delays.get(receiver) else None,
            max_delay_ms=max(delays[receiver]) if delays.get(receiver) else None,
        )
        for receiver, count in counts.items()
    }


def _aggregate_loss_pct(stats: Iterable[LinkStats]) -> float:
    items = tuple(stats)
    events = sum(item.event_count for item in items)
    return sum(item.loss_count for item in items) / events * 100.0 if events else 0.0


def _decode_agent(raw: Any) -> AgentSnapshot:
    if not isinstance(raw, Mapping):
        raise ReplayFormatError("each agent must be a JSON object")
    flags = raw.get("constraint_flags", [])
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise ReplayFormatError("constraint_flags must be a list of strings")
    return AgentSnapshot(
        agent_id=_required_str(raw, "agent_id"),
        position_m=_vector(raw.get("position_m"), "position_m"),
        velocity_mps=_vector(raw.get("velocity_mps"), "velocity_mps"),
        active=_required_bool(raw, "active"),
        confidence=_required_float(raw, "confidence", default=1.0),
        neighbor_count=_required_int(raw, "neighbor_count", default=0),
        min_neighbor_distance_m=_required_float(raw, "min_neighbor_distance_m", default=0.0),
        constraint_flags=tuple(flags),
        profile_id=_required_str(raw, "profile_id", default="unidentified"),
        battery_pct=_optional_float(raw, "battery_pct"),
        link_delay_ms=_optional_float(raw, "link_delay_ms"),
    )


def _vector(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ReplayFormatError(f"{field} must be a three-element numeric list")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ReplayFormatError(f"{field} must be a three-element numeric list") from exc
    return vector  # type: ignore[return-value]


def _required_str(raw: Mapping[str, Any], field: str, default: str | None = None) -> str:
    value = raw.get(field, default)
    if not isinstance(value, str) or not value:
        raise ReplayFormatError(f"{field} must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], field: str, default: int | None = None) -> int:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayFormatError(f"{field} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], field: str, default: float | None = None) -> float:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayFormatError(f"{field} must be numeric")
    return float(value)


def _optional_float(raw: Mapping[str, Any], field: str) -> float | None:
    value = raw.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayFormatError(f"{field} must be numeric when present")
    return float(value)


def _required_bool(raw: Mapping[str, Any], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise ReplayFormatError(f"{field} must be a boolean")
    return value
