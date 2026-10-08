"""Pure-Python replay model shared by the desktop preview and tests.

The model reads the canonical S7 event-log projection. It deliberately exposes
only deterministic replay state. It never opens a radio, camera, controller,
or vehicle adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReplayFormatError(ValueError):
    """Raised when a replay file does not satisfy the S7 event-log contract."""


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
        }


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
