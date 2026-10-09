"""Validated, deterministic JSON scenarios for the S7 coordination fixture.

This module is a portable boundary around :class:`sim.swarm.SwarmSimulator`.
It accepts a small JSON-friendly scenario description and returns a canonical
document suitable for a local, read-only replay.  The link model is synthetic:
the result is not a radio, tracker, flight, or safety measurement.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from sim.link import LinkConfig
from sim.scenarios import make_ring_swarm_config
from sim.swarm import MAX_KEEP_OUT_SPEED_MPS, KeepOutSphere, SwarmConfig, SwarmSimulator


SCENARIO_SCHEMA = "zephyr-s7-scenario-run-1"
SCENARIO_VERSION = 1
# Cross-language hash contract: sorted object keys, compact JSON separators,
# ASCII escaping, and one trailing LF, encoded as UTF-8 bytes.
CANONICAL_PARAMETER_ENCODING = "utf-8 JSON; sort_keys; compact separators; ensure_ascii; trailing LF"
MAX_CONFIG_BYTES = 1 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_AGENT_COUNT = 256
MAX_STEPS = 10_000
MAX_KEEP_OUT_SPHERES = 64
# Keep every integer representable by all contract readers, including
# JavaScript-style JSON implementations and the Swift decoder.
JSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
# Each step sends one target packet and one packet per ordered neighbor pair.
# Bound this product before constructing the event log so a hostile config
# cannot request an unbounded in-memory replay.
MAX_SIMULATION_EVENTS = 250_000

EVIDENCE_BOUNDARY = (
    "Synthetic deterministic point-mass coordination replay with linear moving keep-out spheres over abstract links; "
    "read-only output does not establish measured radio tolerance, tracking "
    "tolerance, flight performance, safety, or live control."
)


class ScenarioConfigError(ValueError):
    """Raised when a scenario document is malformed or outside its bounds."""


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioConfigError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ScenarioConfigError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ScenarioConfigError(f"{field} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ScenarioConfigError(f"{field} must be at most {maximum:g}")
    # Canonical parameters should not distinguish the two spellings of zero.
    return 0.0 if number == 0.0 else number


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioConfigError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ScenarioConfigError(f"{field} must be between {minimum} and {maximum}")
    return value


def _vector(value: Any, field: str, *, component_limit: float = 100.0) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ScenarioConfigError(f"{field} must be a three-element list")
    return [
        _number(component, f"{field}[{index}]", minimum=-component_limit, maximum=component_limit)
        for index, component in enumerate(value)
    ]


def _alias(raw: Mapping[str, Any], names: Sequence[str], field: str, *, required: bool = True) -> Any:
    present = [name for name in names if name in raw]
    if len(present) > 1:
        raise ScenarioConfigError(f"{field} has duplicate aliases: {', '.join(present)}")
    if not present:
        if required:
            raise ScenarioConfigError(f"missing {field}")
        return None
    return raw[present[0]]


def _link(raw: Any, field: str) -> tuple[LinkConfig, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ScenarioConfigError(f"{field} must be an object")
    allowed = {
        "delay_s", "delay", "delay_jitter_s", "jitter_s", "jitter",
        "loss_probability", "loss", "loss_model", "burst_start_probability",
        "burst_end_probability", "contention_mode", "packet_duration_s",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ScenarioConfigError(f"{field} contains unknown field(s): {', '.join(unknown)}")
    delay_raw = _alias(raw, ("delay_s", "delay"), f"{field}.delay_s", required=False)
    delay = _number(0.0 if delay_raw is None else delay_raw,
                    f"{field}.delay_s", minimum=0.0, maximum=10.0)
    jitter_raw = _alias(raw, ("delay_jitter_s", "jitter_s", "jitter"),
                        f"{field}.delay_jitter_s", required=False)
    jitter = _number(0.0 if jitter_raw is None else jitter_raw,
                     f"{field}.delay_jitter_s", minimum=0.0, maximum=10.0)
    if delay == 0.0 and jitter > 0.0:
        raise ScenarioConfigError(f"{field}.delay_jitter_s must be zero when delay_s is zero")
    if jitter > delay:
        raise ScenarioConfigError(f"{field}.delay_jitter_s cannot exceed delay_s")
    loss_raw = _alias(raw, ("loss_probability", "loss"), f"{field}.loss_probability", required=False)
    loss = _number(0.0 if loss_raw is None else loss_raw,
                   f"{field}.loss_probability", minimum=0.0, maximum=1.0)
    loss_model = raw.get("loss_model", "none" if loss == 0.0 else "independent")
    if not isinstance(loss_model, str) or loss_model not in {"none", "independent", "burst"}:
        raise ScenarioConfigError(f"{field}.loss_model must be none, independent, or burst")
    if loss_model == "none" and loss != 0.0:
        raise ScenarioConfigError(f"{field}.loss_probability must be zero when loss_model is none")
    burst_start = _number(raw.get("burst_start_probability", 0.0),
                          f"{field}.burst_start_probability", minimum=0.0, maximum=1.0)
    burst_end = _number(raw.get("burst_end_probability", 1.0),
                        f"{field}.burst_end_probability", minimum=0.0, maximum=1.0)
    if loss_model == "burst" and loss != 0.0:
        raise ScenarioConfigError(f"{field}.loss_probability is ignored by burst loss_model; omit it or set zero")
    if loss_model != "burst" and (burst_start != 0.0 or burst_end != 1.0):
        raise ScenarioConfigError(f"{field} burst probabilities require loss_model burst")
    contention = raw.get("contention_mode", "none")
    if not isinstance(contention, str) or contention not in {"none", "serialized"}:
        raise ScenarioConfigError(f"{field}.contention_mode must be none or serialized")
    packet_duration = _number(raw.get("packet_duration_s", 0.0),
                              f"{field}.packet_duration_s", minimum=0.0, maximum=1.0)
    if contention == "serialized" and packet_duration <= 0.0:
        raise ScenarioConfigError(f"{field}.packet_duration_s must be positive for serialized contention")
    if contention != "serialized" and packet_duration != 0.0:
        raise ScenarioConfigError(f"{field}.packet_duration_s requires serialized contention")
    config = LinkConfig(
        delay_s=delay,
        delay_jitter_s=jitter,
        loss_model=loss_model,
        loss_probability=loss,
        burst_start_probability=burst_start,
        burst_end_probability=burst_end,
        contention_mode=contention,
        packet_duration_s=packet_duration,
    )
    normalized = {
        "delay_s": delay,
        "delay_jitter_s": jitter,
        "loss_model": loss_model,
        "loss_probability": loss,
        "burst_start_probability": burst_start,
        "burst_end_probability": burst_end,
        "contention_mode": contention,
        "packet_duration_s": packet_duration,
    }
    return config, normalized


def _config_from_mapping(raw: Mapping[str, Any]) -> tuple[SwarmConfig, dict[str, Any]]:
    allowed = {
        "scenario_id", "agent_count", "radius", "radius_m", "altitude", "altitude_m",
        "steps", "seed", "dt_s", "target_velocity", "target_velocity_mps",
        "target_link", "neighbor_link", "keep_out_spheres",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ScenarioConfigError(f"scenario contains unknown field(s): {', '.join(unknown)}")
    scenario_id = raw.get("scenario_id", "s7-interactive")
    if not isinstance(scenario_id, str) or not scenario_id.strip() or len(scenario_id) > 120:
        raise ScenarioConfigError("scenario_id must be a non-empty string of at most 120 characters")
    if any(ord(char) < 32 for char in scenario_id):
        raise ScenarioConfigError("scenario_id contains a control character")
    agent_count = _integer(raw.get("agent_count"), "agent_count", minimum=1, maximum=MAX_AGENT_COUNT)
    radius = _number(_alias(raw, ("radius_m", "radius"), "radius_m"), "radius_m", minimum=1e-9, maximum=1_000.0)
    altitude = _number(_alias(raw, ("altitude_m", "altitude"), "altitude_m"), "altitude_m", minimum=0.0, maximum=100.0)
    steps = _integer(raw.get("steps"), "steps", minimum=1, maximum=MAX_STEPS)
    if steps * agent_count * agent_count > MAX_SIMULATION_EVENTS:
        raise ScenarioConfigError(
            "agent_count and steps request too many synthetic link events "
            f"(maximum {MAX_SIMULATION_EVENTS})"
        )
    seed = _integer(raw.get("seed"), "seed", minimum=0, maximum=JSON_SAFE_INTEGER_MAX)
    dt = _number(raw.get("dt_s", 0.05), "dt_s", minimum=1e-6, maximum=1.0)
    velocity_raw = _alias(raw, ("target_velocity_mps", "target_velocity"), "target_velocity_mps", required=False)
    target_velocity = [0.0, 0.0, 0.0] if velocity_raw is None else _vector(velocity_raw, "target_velocity_mps", component_limit=50.0)
    target_link, target_link_document = _link(raw.get("target_link", {}), "target_link")
    neighbor_link, neighbor_link_document = _link(raw.get("neighbor_link", {}), "neighbor_link")
    zones_raw = raw.get("keep_out_spheres", [])
    if not isinstance(zones_raw, list) or len(zones_raw) > MAX_KEEP_OUT_SPHERES:
        raise ScenarioConfigError(f"keep_out_spheres must be a list of at most {MAX_KEEP_OUT_SPHERES} objects")
    zones: list[KeepOutSphere] = []
    zone_document: list[dict[str, Any]] = []
    for index, raw_zone in enumerate(zones_raw):
        field = f"keep_out_spheres[{index}]"
        if not isinstance(raw_zone, Mapping):
            raise ScenarioConfigError(f"{field} must be an object")
        unknown_zone = sorted(set(raw_zone) - {"center", "center_m", "radius", "radius_m", "label", "velocity_mps"})
        if unknown_zone:
            raise ScenarioConfigError(f"{field} contains unknown field(s): {', '.join(unknown_zone)}")
        center = _vector(_alias(raw_zone, ("center_m", "center"), f"{field}.center_m"),
                         f"{field}.center_m", component_limit=100.0)
        radius_zone = _number(_alias(raw_zone, ("radius_m", "radius"), f"{field}.radius_m"),
                              f"{field}.radius_m", minimum=0.0, maximum=100.0)
        velocity_zone = _vector(raw_zone.get("velocity_mps", (0.0, 0.0, 0.0)),
                                f"{field}.velocity_mps", component_limit=MAX_KEEP_OUT_SPEED_MPS)
        label = raw_zone.get("label", f"keep_out_{index + 1}")
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ScenarioConfigError(f"{field}.label must be a non-empty string of at most 80 characters")
        zones.append(KeepOutSphere(tuple(center), radius_zone, label.strip(), tuple(velocity_zone)))
        zone_document.append({"center_m": center, "radius_m": radius_zone, "label": label.strip(), "velocity_mps": velocity_zone})
    config = make_ring_swarm_config(
        agent_count=agent_count,
        radius_m=radius,
        altitude_m=altitude,
        seed=seed,
        target_link=target_link,
        neighbor_link=neighbor_link,
    )
    config = replace(config, dt_s=dt, target_velocity_mps=tuple(target_velocity), keep_out_spheres=tuple(zones))
    parameters = {
        "scenario_id": scenario_id.strip(),
        "agent_count": agent_count,
        "radius_m": radius,
        "altitude_m": altitude,
        "steps": steps,
        "seed": seed,
        "dt_s": dt,
        "target_velocity_mps": target_velocity,
        "target_link": target_link_document,
        "neighbor_link": neighbor_link_document,
        "keep_out_spheres": zone_document,
    }
    return config, parameters


def validate_scenario_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return normalized parameters without running the simulator."""
    if not isinstance(raw, Mapping):
        raise ScenarioConfigError("scenario root must be a JSON object")
    _, parameters = _config_from_mapping(raw)
    return parameters


def _canonical_bytes(value: Any) -> bytes:
    """Encode values according to :data:`CANONICAL_PARAMETER_ENCODING`."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _jsonable(value: Any) -> Any:
    """Convert simulator tuples and NumPy scalars to JSON-native values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_hash(parameters: Mapping[str, Any]) -> str:
    """Return the SHA-256 of normalized scenario parameters."""
    return hashlib.sha256(_canonical_bytes(dict(parameters))).hexdigest()


def _code_revision() -> str | None:
    configured = os.environ.get("ZEPHYR_CODE_REVISION")
    if configured and all(char.isalnum() or char in ".-_" for char in configured):
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def run_scenario(raw: Mapping[str, Any], *, code_revision: str | None = None) -> dict[str, Any]:
    """Validate, run, and return a canonical scenario document."""
    config, parameters = _config_from_mapping(raw)
    result = SwarmSimulator(config).run(parameters["steps"])
    revision = _code_revision() if code_revision is None else code_revision
    if revision is not None and (not isinstance(revision, str) or not revision.strip() or len(revision) > 200):
        raise ScenarioConfigError("code_revision must be a non-empty string of at most 200 characters")
    frames = [_jsonable(frame.as_dict()) for frame in result.steps]
    link_events = [_jsonable(event) for event in result.link_events]
    profiles = [_jsonable(profile) for profile in result.agent_profiles]
    payload = {"frames": frames, "link_events": link_events, "profiles": profiles}
    payload_canonical = _canonical_bytes(payload).decode("utf-8")
    document = {
        "schema": SCENARIO_SCHEMA,
        "version": SCENARIO_VERSION,
        "scenario_hash": scenario_hash(parameters),
        "parameters_canonical": _canonical_bytes(parameters).decode("utf-8"),
        "payload_canonical": payload_canonical,
        "payload_sha256": hashlib.sha256(payload_canonical.encode("utf-8")).hexdigest(),
        "parameters": parameters,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "frames": frames,
        "link_events": link_events,
        "profiles": profiles,
        "provenance": {
            "seed": parameters["seed"],
            "dt_s": parameters["dt_s"],
            "code_revision": revision,
            "generator": "sim.scenario_runner",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "status": "synthetic deterministic replay; read-only",
        },
    }
    _validate_run_document(document)
    return document


def _validate_run_document(document: Mapping[str, Any]) -> None:
    """Check the output envelope before it can be serialized or written."""
    required = {"schema", "version", "scenario_hash", "parameters_canonical", "payload_canonical", "payload_sha256", "parameters", "evidence_boundary", "frames", "link_events", "profiles", "provenance"}
    if set(document) != required:
        raise ScenarioConfigError("run document has an invalid field set")
    if document["schema"] != SCENARIO_SCHEMA or document["version"] != SCENARIO_VERSION:
        raise ScenarioConfigError("run document schema/version is invalid")
    parameters = document["parameters"]
    if not isinstance(parameters, Mapping):
        raise ScenarioConfigError("parameters must be an object")
    if "steps" not in parameters or "seed" not in parameters or "dt_s" not in parameters:
        raise ScenarioConfigError("parameters is missing required provenance fields")
    if isinstance(parameters["steps"], bool) or not isinstance(parameters["steps"], int):
        raise ScenarioConfigError("parameters.steps must be an integer")
    canonical_parameters = document["parameters_canonical"]
    if not isinstance(canonical_parameters, str) or not canonical_parameters.endswith("\n"):
        raise ScenarioConfigError("parameters_canonical must be canonical UTF-8 JSON ending in LF")
    if (
        not isinstance(document["scenario_hash"], str)
        or len(document["scenario_hash"]) != 64
        or any(char not in "0123456789abcdef" for char in document["scenario_hash"])
    ):
        raise ScenarioConfigError("scenario_hash must be a SHA-256 hex string")
    if scenario_hash(parameters) != document["scenario_hash"]:
        raise ScenarioConfigError("scenario_hash does not match parameters")
    if canonical_parameters != _canonical_bytes(parameters).decode("utf-8"):
        raise ScenarioConfigError("parameters_canonical does not match parameters")
    payload = {"frames": document["frames"], "link_events": document["link_events"], "profiles": document["profiles"]}
    canonical_payload = document["payload_canonical"]
    if not isinstance(canonical_payload, str):
        raise ScenarioConfigError("payload_canonical must be canonical UTF-8 JSON")
    try:
        expected_payload = _canonical_bytes(payload).decode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioConfigError(f"payload contains a non-JSON value: {exc}") from exc
    if canonical_payload != expected_payload:
        raise ScenarioConfigError("payload_canonical does not match payload")
    if not isinstance(document["payload_sha256"], str) or len(document["payload_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in document["payload_sha256"]):
        raise ScenarioConfigError("payload_sha256 must be a SHA-256 hex string")
    if hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest() != document["payload_sha256"]:
        raise ScenarioConfigError("payload_sha256 does not match payload")
    if not isinstance(document["frames"], list) or len(document["frames"]) != parameters["steps"]:
        raise ScenarioConfigError("frames must contain one entry per requested step")
    if not isinstance(document["link_events"], list) or not isinstance(document["profiles"], list):
        raise ScenarioConfigError("link_events and profiles must be lists")
    if not isinstance(document["provenance"], Mapping):
        raise ScenarioConfigError("provenance must be an object")
    if document["provenance"].get("seed") != parameters["seed"]:
        raise ScenarioConfigError("provenance seed does not match parameters")
    if document["provenance"].get("dt_s") != parameters["dt_s"]:
        raise ScenarioConfigError("provenance dt_s does not match parameters")
    try:
        serialized = _canonical_bytes(document)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioConfigError(f"run document contains a non-JSON value: {exc}") from exc
    schema_path = Path(__file__).with_name("scenario_run_schema.json")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda error: list(error.path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioConfigError(f"could not load scenario run schema: {exc}") from exc
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "run"
        raise ScenarioConfigError(f"{location}: {errors[0].message}")
    if len(serialized) > MAX_OUTPUT_BYTES:
        raise ScenarioConfigError(f"run exceeds {MAX_OUTPUT_BYTES // (1024 * 1024)} MiB")


def to_replay_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a scenario-run envelope to the existing desktop replay contract.

    The converter intentionally returns a new object and strips optional
    hardware-profile fields that the frozen ``s7_run_schema`` does not carry.
    It does not write or mutate ``runs/s7-swarm/run.json``.
    """
    _validate_run_document(document)
    allowed_profile_keys = (
        "profile_id", "motor", "propeller", "motor_count", "arm_length_m",
        "frame_mass_kg", "battery_capacity_wh", "total_mass_kg", "estimated_max_thrust_n",
    )
    profiles = [
        {key: profile[key] for key in allowed_profile_keys if key in profile}
        for profile in document["profiles"]
    ]
    replay_event_keys = (
        "sender", "receiver", "seq", "send_time", "receive_time", "duplicate",
        "out_of_order", "loss_reason", "packet_age",
    )
    link_events = [
        {key: event[key] for key in replay_event_keys if key in event}
        for event in document["link_events"]
    ]
    replay_frame_keys = ("step_index", "time_s", "target_position_m", "active_count", "agents")
    steps = [
        {key: frame[key] for key in replay_frame_keys if key in frame}
        for frame in document["frames"]
    ]
    replay = {
        "schema": "zephyr-s7-swarm-run-1",
        "seed": document["provenance"]["seed"],
        "dt_s": document["provenance"]["dt_s"],
        "steps": steps,
        "link_events": link_events,
        "agent_profiles": profiles,
        "scenario_id": document["parameters"]["scenario_id"],
        "evidence_boundary": document["evidence_boundary"],
        "generation_command": "sim.scenario_runner.run_scenario",
        "status": "synthetic deterministic replay; not flight performance",
        "command_authority": "replay only",
        "failsafe": "synthetic constraints only; live failsafe unavailable",
    }
    return replay


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load one bounded JSON config with duplicate-key and constant checks."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ScenarioConfigError("config must be a regular file")
    data = source.read_bytes()
    if len(data) > MAX_CONFIG_BYTES:
        raise ScenarioConfigError(f"config exceeds {MAX_CONFIG_BYTES // 1024} KiB")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScenarioConfigError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ScenarioConfigError(f"invalid JSON constant: {value}")

    try:
        document = json.loads(data, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioConfigError(f"invalid JSON config: {exc}") from exc
    if not isinstance(document, dict):
        raise ScenarioConfigError("config root must be a JSON object")
    return document


def serialize_run(document: Mapping[str, Any]) -> bytes:
    """Validate and serialize a run with stable key order and no NaN values."""
    _validate_run_document(document)
    return _canonical_bytes(document)


def summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, synthetic-only summary for a CLI or caller."""
    _validate_run_document(document)
    return {
        "schema": document["schema"],
        "version": document["version"],
        "scenario_hash": document["scenario_hash"],
        "agent_count": document["parameters"]["agent_count"],
        "frame_count": len(document["frames"]),
        "link_event_count": len(document["link_events"]),
        "profile_count": len(document["profiles"]),
        "status": document["provenance"]["status"],
        "evidence_boundary": document["evidence_boundary"],
    }
