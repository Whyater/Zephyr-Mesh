"""Utilities for deterministic simulated mission logs."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from typing import Any
import jsonschema

_SCHEMA_PATH = Path(__file__).with_name("mission_log_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def scenario_sha256(scenario: Any) -> str:
    return hashlib.sha256(canonical_json(scenario).encode("utf-8")).hexdigest()

def round_floats(obj: Any) -> Any:
    if isinstance(obj, bool) or isinstance(obj, int) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        value = round(obj, 6)
        return 0.0 if value == 0.0 else value
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [round_floats(v) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    return obj

def build_log(**fields: Any) -> dict:
    result = dict(fields)
    result.setdefault("schema", "zephyr-mission-log-1")
    result.setdefault("simulated", True)
    if "scenario" in result:
        result["scenario_sha256"] = scenario_sha256(result["scenario"])
    return round_floats(result)

def _path(error) -> str:
    path = "".join(f"[{p!r}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path)
    return path.lstrip(".") or "$"

def validate_log(obj: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        raise ValueError(f"{_path(e)}: {e.message}")
    if scenario_sha256(obj["scenario"]) != obj["scenario_sha256"]:
        raise ValueError("scenario_sha256: does not match scenario")
    if obj["agents"] != sorted(obj["agents"]):
        raise ValueError("agents: must be sorted")
    frames = obj["frames"]
    if set(frames["agents"]) != set(obj["agents"]):
        raise ValueError("frames.agents: keys must match agents")
    n = len(frames["time_s"])
    for i in range(1, n):
        if frames["time_s"][i] <= frames["time_s"][i-1]:
            raise ValueError(f"frames.time_s[{i}]: times must be strictly increasing")
    for agent, series in frames["agents"].items():
        for key, values in series.items():
            if isinstance(values, list) and len(values) != n:
                raise ValueError(f"frames.agents.{agent}.{key}: expected length {n}, got {len(values)}")
    previous = None
    for i, event in enumerate(obj["events"]):
        if previous is not None and event["time_s"] < previous:
            raise ValueError(f"events[{i}].time_s: events must be sorted by time")
        previous = event["time_s"]

def write_log(path: str | Path, log: dict) -> None:
    validate_log(log)
    text = json.dumps(log, indent=None, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")

def load_log(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_log(obj)
    return obj

class MissionLogRecorder:
    def __init__(self, agents, dt_s, log_every):
        self.agents = sorted(agents)
        self.dt_s = dt_s
        self.log_every = log_every
        self._times=[]; self._targets=[]; self._series={a:{k:[] for k in ("position_m","velocity_mps","failsafe_state","authority","link_age_s","estimate_position_m","estimate_age_s")} for a in self.agents}
        self._events=[]
    def frame(self, i, t, target_position, per_agent):
        if i % self.log_every: return
        self._times.append(t); self._targets.append(target_position)
        for a in self.agents:
            d=per_agent[a]
            for k in self._series[a]: self._series[a][k].append(d.get(k))
    def event(self, t, kind, agent, detail): self._events.append({"time_s":t,"kind":kind,"agent":agent,"detail":detail})
    def to_log(self, **header):
        fields = dict(header)
        fields.update(dt_s=self.dt_s, log_every=self.log_every, agents=self.agents,
                      frames={"time_s": self._times, "target_position_m": self._targets, "agents": self._series},
                      events=self._events)
        return build_log(**fields)
