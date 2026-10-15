"""Scenario description for simulated missions.

A scenario is plain JSON so a student can write one by hand, run it with
``tools/run_mission.py`` and share it.  Everything a scenario describes is
simulated: link loss patterns are synthetic and vehicle parameters are
literature-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any

from sim.failsafe import FailsafeConfig
from sim.missions.presets import PRESET_NAMES

ADAPTER_KINDS = ("point_mass", "physics", "point_mass_offboard", "physics_offboard")


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class Scenario:
    """One simulated mission run: which mission, which link, which seed.

    ``outages`` are scripted total blackouts, each
    ``{"start_s": float, "end_s": float | None, "streams": list | None}``.
    ``end_s`` None means the blackout lasts until the mission ends;
    ``streams`` None means every stream in the mission goes silent.
    """

    mission: str
    seed: int
    duration_s: float
    adapter: str = "point_mass"
    preset: str = "clean"
    policy: dict = field(default_factory=dict)
    measurement_delay_s: float = 0.0
    measurement_jitter_s: float = 0.0
    command_delay_s: float = 0.002
    outages: tuple = ()
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mission, str) or not self.mission:
            raise ValueError("mission must be a non-empty string")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or int(self.seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        duration = _finite_nonnegative("duration_s", self.duration_s)
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        object.__setattr__(self, "duration_s", duration)
        if self.adapter not in ADAPTER_KINDS:
            raise ValueError(f"adapter must be one of {ADAPTER_KINDS}")
        if self.preset not in PRESET_NAMES:
            raise ValueError(f"preset must be one of {PRESET_NAMES}")
        if not isinstance(self.policy, dict):
            raise ValueError("policy must be a mapping")
        FailsafeConfig.from_dict(self.policy)  # validates field names and values
        for name in ("measurement_delay_s", "measurement_jitter_s", "command_delay_s"):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))
        outages = []
        for item in self.outages:
            if not isinstance(item, dict) or set(item) - {"start_s", "end_s", "streams"} or "start_s" not in item:
                raise ValueError("each outage needs start_s and may have end_s and streams only")
            start = _finite_nonnegative("outage start_s", item["start_s"])
            end = item.get("end_s")
            if end is not None:
                end = _finite_nonnegative("outage end_s", end)
                if end < start:
                    raise ValueError("outage end_s must not be before start_s")
            streams = item.get("streams")
            if streams is not None:
                streams = sorted(str(s) for s in streams)
            outages.append({"start_s": start, "end_s": end, "streams": streams})
        object.__setattr__(self, "outages", tuple(outages))
        if not isinstance(self.params, dict):
            raise ValueError("params must be a mapping")

    @property
    def failsafe_config(self) -> FailsafeConfig:
        return FailsafeConfig.from_dict(self.policy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "command_delay_s": self.command_delay_s,
            "duration_s": self.duration_s,
            "measurement_delay_s": self.measurement_delay_s,
            "measurement_jitter_s": self.measurement_jitter_s,
            "mission": self.mission,
            "outages": [dict(item) for item in self.outages],
            "params": dict(self.params),
            "policy": dict(self.policy),
            "preset": self.preset,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scenario":
        if not isinstance(value, dict):
            raise ValueError("a scenario must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        data = dict(value)
        data["outages"] = tuple(data.get("outages", ()))
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace(self, **changes: Any) -> "Scenario":
        data = self.as_dict()
        data.update(changes)
        return Scenario.from_dict(data)
