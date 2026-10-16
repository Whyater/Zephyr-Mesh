"""Mission logs for simulated drone missions.

Every log written here describes a SIMULATED run: no drone was flown and no
radio was measured.  A log is one JSON document (schema id
``zephyr-mission-log-1``, see ``mission_log_schema.json``) with the scenario,
a hash of the scenario, per-drone time series stored column by column, a list
of events, link counters and a flat metrics dictionary.

Floats are rounded to six decimals when a log is built or written, negative
zero becomes zero and non-finite numbers become null, so the same run always
produces the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from jsonschema.exceptions import best_match

from sim.failsafe import FailsafeConfig

_SCHEMA_PATH = Path(__file__).with_name("mission_log_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)

_SERIES_KEYS = (
    "position_m",
    "velocity_mps",
    "failsafe_state",
    "authority",
    "link_age_s",
    "estimate_position_m",
    "estimate_age_s",
)
# Columns that may be null as a whole when no value was ever recorded.
_OPTIONAL_COLUMNS = ("estimate_position_m", "estimate_age_s")


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted, ASCII-only JSON text used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def scenario_sha256(scenario: Any) -> str:
    """Hex SHA-256 of the canonical JSON text of a scenario."""
    return hashlib.sha256(canonical_json(scenario).encode("utf-8")).hexdigest()


def round_floats(obj: Any) -> Any:
    """Return a JSON-ready copy of ``obj``.

    Floats are rounded to six decimals (negative zero becomes zero, NaN and
    infinity become None).  NumPy arrays become lists, NumPy scalars become
    plain Python numbers or booleans, and tuples become lists.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round_floats(float(obj))
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        value = round(obj, 6)
        return 0.0 if value == 0.0 else value
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [round_floats(item) for item in obj]
    if isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    return obj


def build_log(**fields: Any) -> dict:
    """Assemble a log dictionary from header fields and series.

    The fields are rounded first.  When a scenario is given, its hash is
    computed from the rounded scenario, so the stored scenario and its hash
    always agree.  ``schema`` and ``simulated`` are filled in when missing.
    """
    result = round_floats(dict(fields))
    result.setdefault("schema", "zephyr-mission-log-1")
    result.setdefault("simulated", True)
    if "scenario" in result:
        result["scenario_sha256"] = scenario_sha256(result["scenario"])
    return result


def _path(path_parts) -> str:
    text = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path_parts)
    return text.lstrip(".") or "$"


def _check_json_ready(value: Any, where: str) -> None:
    """Raise ValueError unless ``value`` is plain, finite JSON data.

    ``where`` is the path of ``value``; the top-level object uses "".
    """
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where or '$'}: non-finite number is not JSON-ready")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_ready(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where or '$'}: object keys must be strings, got {type(key).__name__}")
            _check_json_ready(item, f"{where}.{key}" if where else key)
        return
    raise ValueError(f"{where or '$'}: {type(value).__name__} is not JSON-ready")


def validate_log(obj: dict) -> None:
    """Check a log against the schema and the rules a schema cannot state.

    Raises ValueError that starts with the path of the first problem.  Every
    number anywhere in the log must be finite.
    """
    _check_json_ready(obj, "")
    error = best_match(_VALIDATOR.iter_errors(obj))
    if error is not None:
        raise ValueError(f"{_path(error.absolute_path)}: {error.message}")

    try:
        FailsafeConfig.from_dict(obj["policy"])
    except Exception as exc:
        raise ValueError(f"policy: {exc}") from exc

    if scenario_sha256(obj["scenario"]) != obj["scenario_sha256"]:
        raise ValueError("scenario_sha256: does not match scenario")

    agents = obj["agents"]
    if agents != sorted(agents):
        raise ValueError("agents: must be sorted")

    frames = obj["frames"]
    if set(frames["agents"]) != set(agents):
        raise ValueError("frames.agents: keys must match agents")

    times = frames["time_s"]
    count = len(times)
    for index in range(1, count):
        if times[index] <= times[index - 1]:
            raise ValueError(f"frames.time_s[{index}]: times must be strictly increasing")

    target = frames["target_position_m"]
    if target is not None and len(target) != count:
        raise ValueError(f"frames.target_position_m: expected length {count}, got {len(target)}")

    for agent in agents:
        for key, values in frames["agents"][agent].items():
            if values is not None and len(values) != count:
                raise ValueError(
                    f"frames.agents.{agent}.{key}: expected length {count}, got {len(values)}")

    known = set(agents)
    previous_time = None
    for index, event in enumerate(obj["events"]):
        if event["agent"] is not None and event["agent"] not in known:
            raise ValueError(f"events[{index}].agent: {event['agent']!r} is not one of the log's agents")
        if previous_time is not None and event["time_s"] < previous_time:
            raise ValueError(f"events[{index}].time_s: events must be sorted by time")
        previous_time = event["time_s"]


def write_log(path: str | Path, log: dict) -> None:
    """Round, validate and write a log as one line of compact JSON (UTF-8, LF)."""
    log = round_floats(log)
    validate_log(log)
    text = json.dumps(log, indent=None, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def load_log(path: str | Path) -> dict:
    """Read a log from disk and validate it."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_log(obj)
    return obj


class MissionLogRecorder:
    """Collect frames and events during a simulated run and build the log.

    ``frame`` keeps every ``log_every``-th control step.  A column that holds
    no value at all (for example estimates when no estimator ran) is written
    as null for the whole series.
    """

    def __init__(self, agents, dt_s: float, log_every: int) -> None:
        self.agents = sorted(agents)
        self.dt_s = dt_s
        self.log_every = log_every
        self._times: list = []
        self._targets: list = []
        self._series = {agent: {key: [] for key in _SERIES_KEYS} for agent in self.agents}
        self._events: list = []

    def frame(self, i: int, t: float, target_position, per_agent: dict) -> None:
        """Record one control step if it falls on the logging grid."""
        if i % self.log_every:
            return
        self._times.append(t)
        self._targets.append(target_position)
        for agent in self.agents:
            values = per_agent[agent]
            for key in _SERIES_KEYS:
                self._series[agent][key].append(values.get(key))

    def event(self, t: float, kind: str, agent: str | None, detail: dict | None = None) -> None:
        """Record one event."""
        self._events.append({"time_s": t, "kind": kind, "agent": agent, "detail": dict(detail or {})})

    def to_log(self, **header: Any) -> dict:
        """Build the log from the recorded frames, the events and the header fields.

        The finished log is validated, so a bad header raises ValueError.
        """
        targets = None if all(item is None for item in self._targets) else list(self._targets)
        series = {}
        for agent in self.agents:
            columns = {key: list(values) for key, values in self._series[agent].items()}
            for key in _OPTIONAL_COLUMNS:
                if all(item is None for item in columns[key]):
                    columns[key] = None
            series[agent] = columns
        fields = dict(header)
        fields.update(
            dt_s=self.dt_s,
            log_every=self.log_every,
            agents=self.agents,
            frames={"time_s": list(self._times), "target_position_m": targets, "agents": series},
            events=list(self._events),
        )
        log = build_log(**fields)
        validate_log(log)
        return log
