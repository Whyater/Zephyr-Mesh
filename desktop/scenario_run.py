"""Portable decoder for the canonical S7 scenario-run contract.

The decoder is deliberately read-only. It validates the generated JSON
envelope and exposes only summary and frame data for desktop inspection. It
does not run the simulator, open an adapter, or claim measured performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCENARIO_SCHEMA = "zephyr-s7-scenario-run-1"
SCENARIO_VERSION = 1
MAX_SCENARIO_BYTES = 64 * 1024 * 1024
JSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
FRAME_TIME_TOLERANCE_S = 1e-9


class ScenarioRunFormatError(ValueError):
    """Raised when a scenario-run JSON file is malformed or unsupported."""


@dataclass(frozen=True)
class ScenarioFrame:
    step_index: int
    time_s: float
    target_position_m: tuple[float, float, float]
    active_count: int
    agent_count: int
    agents: tuple[Mapping[str, Any], ...]
    keep_out_spheres: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ScenarioPlotBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass(frozen=True)
class ScenarioAltitudeBounds:
    """Finite altitude range used by the selected-frame synthetic profile."""

    min_z: float
    max_z: float


@dataclass(frozen=True)
class ScenarioPlotGeometry:
    bounds: ScenarioPlotBounds
    altitude_bounds: ScenarioAltitudeBounds
    target: tuple[float, float]
    target_altitude: float
    agents: tuple[tuple[str, float, float, bool], ...]
    agent_altitudes: tuple[float, ...]
    keep_out_spheres: tuple[tuple[str, float, float, float], ...]
    keep_out_altitudes: tuple[float, ...]


def scenario_frame_geometry(frame: ScenarioFrame) -> ScenarioPlotGeometry:
    """Return bounded top-down geometry for one synthetic frame."""
    target = (float(frame.target_position_m[0]), float(frame.target_position_m[1]))
    agents = tuple(
        (
            str(agent["agent_id"]),
            float(agent["position_m"][0]),
            float(agent["position_m"][1]),
            bool(agent["active"]),
        )
        for agent in frame.agents
    )
    agent_altitudes = tuple(float(agent["position_m"][2]) for agent in frame.agents)
    keep_out_spheres = tuple(
        (
            str(sphere["label"]),
            float(sphere["center_m"][0]),
            float(sphere["center_m"][1]),
            float(sphere["radius_m"]),
        )
        for sphere in frame.keep_out_spheres
    )
    keep_out_altitudes = tuple(float(sphere["center_m"][2]) for sphere in frame.keep_out_spheres)
    x_values = [target[0], *(agent[1] for agent in agents)]
    y_values = [target[1], *(agent[2] for agent in agents)]
    x_values.extend(center_x + radius for _, center_x, _, radius in keep_out_spheres)
    x_values.extend(center_x - radius for _, center_x, _, radius in keep_out_spheres)
    y_values.extend(center_y + radius for _, _, center_y, radius in keep_out_spheres)
    y_values.extend(center_y - radius for _, _, center_y, radius in keep_out_spheres)
    span = max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1.0)
    margin = max(span * 0.05, 0.25)
    z_values = [float(frame.target_position_m[2]), *agent_altitudes]
    z_values.extend(center_z + radius for center_z, (_, _, _, radius) in zip(keep_out_altitudes, keep_out_spheres))
    z_values.extend(center_z - radius for center_z, (_, _, _, radius) in zip(keep_out_altitudes, keep_out_spheres))
    z_span = max(max(z_values) - min(z_values), 1.0)
    z_margin = max(z_span * 0.05, 0.25)
    return ScenarioPlotGeometry(
        bounds=ScenarioPlotBounds(min(x_values) - margin, max(x_values) + margin, min(y_values) - margin, max(y_values) + margin),
        altitude_bounds=ScenarioAltitudeBounds(min(z_values) - z_margin, max(z_values) + z_margin),
        target=target,
        target_altitude=float(frame.target_position_m[2]),
        agents=agents,
        agent_altitudes=agent_altitudes,
        keep_out_spheres=keep_out_spheres,
        keep_out_altitudes=keep_out_altitudes,
    )


def project_scenario_point(point: tuple[float, float], bounds: ScenarioPlotBounds, width: float, height: float, inset: float = 18.0) -> tuple[float, float]:
    """Project one top-down metre point into a bounded canvas rectangle."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    inset = max(float(inset), 0.0)
    span_x = max(bounds.max_x - bounds.min_x, 1.0)
    span_y = max(bounds.max_y - bounds.min_y, 1.0)
    scale = min(max(width - 2.0 * inset, 1.0) / span_x, max(height - 2.0 * inset, 1.0) / span_y)
    offset_x = (width - span_x * scale) / 2.0
    offset_y = (height - span_y * scale) / 2.0
    return (offset_x + (point[0] - bounds.min_x) * scale, height - offset_y - (point[1] - bounds.min_y) * scale)


def project_scenario_radius(radius_m: float, bounds: ScenarioPlotBounds, width: float, height: float, inset: float = 18.0) -> float:
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    inset = max(float(inset), 0.0)
    span_x = max(bounds.max_x - bounds.min_x, 1.0)
    span_y = max(bounds.max_y - bounds.min_y, 1.0)
    scale = min(max(width - 2.0 * inset, 1.0) / span_x, max(height - 2.0 * inset, 1.0) / span_y)
    return max(float(radius_m), 0.0) * scale


def project_scenario_altitude(altitude_m: float, bounds: ScenarioAltitudeBounds, height: float, inset: float = 18.0) -> float:
    """Project altitude into a vertical profile rail in canvas coordinates."""

    height = max(float(height), 1.0)
    inset = min(max(float(inset), 0.0), height / 2.0)
    span = max(bounds.max_z - bounds.min_z, 1.0)
    usable = max(height - 2.0 * inset, 0.0)
    projected = height - inset - (float(altitude_m) - bounds.min_z) * usable / span
    return min(max(projected, 0.0), height)


def project_scenario_altitude_radius(radius_m: float, bounds: ScenarioAltitudeBounds, height: float, inset: float = 18.0) -> float:
    """Project a sphere radius onto the altitude profile rail."""

    height = max(float(height), 1.0)
    inset = min(max(float(inset), 0.0), height / 2.0)
    span = max(bounds.max_z - bounds.min_z, 1.0)
    usable = max(height - 2.0 * inset, 0.0)
    return min(max(float(radius_m), 0.0) * usable / span, height / 2.0)


def clamp_scenario_rail_x(width: float, preferred_x: float) -> float:
    """Keep the synthetic altitude rail coordinate inside a canvas."""

    width = max(float(width), 0.0)
    return min(max(float(preferred_x), 0.0), width)


def scenario_rail_endpoints(height: float, top_inset: float = 30.0, bottom_inset: float = 18.0) -> tuple[float, float]:
    """Return clamped altitude rail endpoints, with a midpoint fallback."""

    height = max(float(height), 0.0)
    top = min(max(float(top_inset), 0.0), height)
    bottom = min(max(height - float(bottom_inset), 0.0), height)
    if bottom < top:
        midpoint = height / 2.0
        return midpoint, midpoint
    return top, bottom


@dataclass(frozen=True)
class ScenarioRun:
    document: Mapping[str, Any]
    source_path: Path
    frames: tuple[ScenarioFrame, ...]

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.document["parameters"]

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.document["provenance"]

    def summary(self) -> dict[str, Any]:
        parameters = self.parameters
        return {
            "schema": self.document["schema"],
            "version": self.document["version"],
            "scenario_hash": self.document["scenario_hash"],
            "payload_sha256": self.document["payload_sha256"],
            "scenario_id": parameters["scenario_id"],
            "agent_count": parameters["agent_count"],
            "dt_s": parameters["dt_s"],
            "frame_count": len(self.frames),
            "link_event_count": len(self.document["link_events"]),
            "profile_count": len(self.document["profiles"]),
            "generator": self.provenance["generator"],
            "python": self.provenance["python"],
            "numpy": self.provenance["numpy"],
            "code_revision": self.provenance["code_revision"],
            "code_revision_semantics": "base revision; does not fingerprint uncommitted files",
            "status": self.provenance["status"],
            "evidence_boundary": self.document["evidence_boundary"],
        }


@dataclass
class ScenarioRunCursor:
    """Deterministic, local cursor for the read-only scenario inspector."""

    frame_count: int
    index: int = 0

    def __post_init__(self) -> None:
        self.frame_count = max(1, int(self.frame_count))
        self.index = 0

    def seek(self, requested: int) -> int:
        self.index = min(max(int(requested), 0), self.frame_count - 1)
        return self.index

    def step(self, delta: int = 1) -> int:
        return self.seek(self.index + int(delta))

    def reset(self) -> int:
        self.index = 0
        return self.index

    def replace(self, frame_count: int) -> int:
        self.frame_count = max(1, int(frame_count))
        self.index = 0
        return self.index


def selected_frame_summary(run: ScenarioRun, cursor: ScenarioRunCursor) -> str:
    """Format one selected frame for the read-only scenario inspector."""

    frame = run.frames[cursor.index]
    target = ", ".join(f"{value:.3f}" for value in frame.target_position_m)
    return (
        f"Frame {cursor.index + 1} of {len(run.frames)} · t={frame.time_s:.3f} s · "
        f"target=[{target}] m · {frame.active_count}/{frame.agent_count} active · "
        f"keep-out {len(frame.keep_out_spheres)} · synthetic altitude {frame.target_position_m[2]:.3f} m · altitude profile"
    )


def load_scenario_run(path: str | Path) -> ScenarioRun:
    """Load and validate one generated scenario-run JSON file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ScenarioRunFormatError("scenario run must be a regular file")
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ScenarioRunFormatError(f"could not inspect scenario run: {exc}") from exc
    if size > MAX_SCENARIO_BYTES:
        raise ScenarioRunFormatError("scenario run exceeds the 64 MiB size limit")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScenarioRunFormatError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates, parse_constant=_reject_constant)
    except ScenarioRunFormatError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioRunFormatError(f"could not read scenario run: {exc}") from exc
    if not isinstance(document, dict):
        raise ScenarioRunFormatError("scenario run root must be a JSON object")
    _validate(document)
    frames = tuple(_decode_frame(frame) for frame in document["frames"])
    _validate_frame_sequence(frames, float(document["parameters"]["dt_s"]))
    return ScenarioRun(document=document, source_path=source, frames=frames)


def _reject_constant(value: str) -> None:
    raise ScenarioRunFormatError(f"invalid JSON constant: {value}")


def _validate(document: Mapping[str, Any]) -> None:
    required = {"schema", "version", "scenario_hash", "parameters_canonical", "payload_canonical", "payload_sha256", "parameters", "evidence_boundary", "frames", "link_events", "profiles", "provenance"}
    if set(document) != required:
        raise ScenarioRunFormatError("scenario run has an invalid field set")
    if document["schema"] != SCENARIO_SCHEMA or document["version"] != SCENARIO_VERSION:
        raise ScenarioRunFormatError("scenario run schema/version is invalid")
    digest = document["scenario_hash"]
    if not isinstance(digest, str) or digest != digest.lower() or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ScenarioRunFormatError("scenario_hash must be a lowercase SHA-256 hex string")
    parameters = document["parameters"]
    if not isinstance(parameters, dict):
        raise ScenarioRunFormatError("parameters must be an object")
    _validate_parameters(parameters)
    canonical = (json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n")
    if document["parameters_canonical"] != canonical:
        raise ScenarioRunFormatError("parameters_canonical does not match parameters")
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
        raise ScenarioRunFormatError("scenario_hash does not match parameters")
    payload = {"frames": document["frames"], "link_events": document["link_events"], "profiles": document["profiles"]}
    payload_canonical = document["payload_canonical"]
    if not isinstance(payload_canonical, str) or payload_canonical != (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"):
        raise ScenarioRunFormatError("payload_canonical does not match payload")
    payload_digest = document["payload_sha256"]
    if not isinstance(payload_digest, str) or len(payload_digest) != 64 or any(c not in "0123456789abcdef" for c in payload_digest) or hashlib.sha256(payload_canonical.encode("utf-8")).hexdigest() != payload_digest:
        raise ScenarioRunFormatError("payload_sha256 does not match payload")
    boundary = document["evidence_boundary"]
    if not isinstance(boundary, str) or not boundary.strip():
        raise ScenarioRunFormatError("evidence_boundary must be a non-empty string")
    frames = document["frames"]
    if not isinstance(frames, list) or not frames:
        raise ScenarioRunFormatError("frames must be a non-empty list")
    if len(frames) != parameters["steps"]:
        raise ScenarioRunFormatError("frames must contain one entry per requested step")
    if not isinstance(document["link_events"], list) or not isinstance(document["profiles"], list):
        raise ScenarioRunFormatError("link_events and profiles must be lists")
    provenance = document["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"seed", "dt_s", "code_revision", "generator", "python", "numpy", "status"} or provenance.get("status") != "synthetic deterministic replay; read-only":
        raise ScenarioRunFormatError("provenance status must identify a synthetic read-only replay")
    if provenance.get("seed") != parameters.get("seed") or provenance.get("dt_s") != parameters.get("dt_s"):
        raise ScenarioRunFormatError("provenance does not match parameters")
    if provenance.get("generator") != "sim.scenario_runner" or not isinstance(provenance.get("python"), str) or not isinstance(provenance.get("numpy"), str):
        raise ScenarioRunFormatError("provenance generator and versions are invalid")
    if provenance.get("code_revision") is not None and (not isinstance(provenance.get("code_revision"), str) or not provenance["code_revision"].strip() or len(provenance["code_revision"]) > 200):
        raise ScenarioRunFormatError("provenance code_revision is invalid")
    for event in document["link_events"]:
        _validate_link_event(event)
    for profile in document["profiles"]:
        _validate_profile(profile)


def _validate_frame_sequence(frames: tuple[ScenarioFrame, ...], dt_s: float) -> None:
    for index, frame in enumerate(frames):
        if frame.step_index != index:
            raise ScenarioRunFormatError("frame step_index sequence is invalid")
        if abs(frame.time_s - index * dt_s) > FRAME_TIME_TOLERANCE_S:
            raise ScenarioRunFormatError("frame time_s sequence is invalid")


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    required = {"scenario_id", "agent_count", "radius_m", "altitude_m", "steps", "seed", "dt_s", "target_velocity_mps", "target_link", "neighbor_link", "keep_out_spheres"}
    if set(parameters) != required or not isinstance(parameters.get("scenario_id"), str) or not parameters["scenario_id"] or len(parameters["scenario_id"]) > 120 or any(ord(char) < 32 for char in parameters["scenario_id"]):
        raise ScenarioRunFormatError("parameters have an invalid field set")
    _integer(parameters["agent_count"], "agent_count", 1, 256)
    _integer(parameters["steps"], "steps", 1, 10000)
    _integer(parameters["seed"], "seed", 0, JSON_SAFE_INTEGER_MAX)
    _number(parameters["radius_m"], "radius_m", 0, 1000, strict_min=True)
    _number(parameters["altitude_m"], "altitude_m", 0, 100)
    _number(parameters["dt_s"], "dt_s", 0, 1, strict_min=True)
    _vector(parameters["target_velocity_mps"], "target_velocity_mps")
    _validate_link(parameters["target_link"], "target_link")
    _validate_link(parameters["neighbor_link"], "neighbor_link")
    spheres = parameters["keep_out_spheres"]
    if not isinstance(spheres, list) or len(spheres) > 64:
        raise ScenarioRunFormatError("keep_out_spheres must contain at most 64 entries")
    for sphere in spheres:
        if not isinstance(sphere, dict) or not {"center_m", "radius_m", "label"}.issubset(sphere) or not set(sphere).issubset({"center_m", "radius_m", "label", "velocity_mps"}) or not isinstance(sphere["label"], str) or not sphere["label"]:
            raise ScenarioRunFormatError("keep_out_spheres entries are invalid")
        _vector(sphere["center_m"], "keep_out_spheres.center_m")
        _number(sphere["radius_m"], "keep_out_spheres.radius_m", 0, 100)
        if "velocity_mps" in sphere:
            velocity = sphere["velocity_mps"]
            if not isinstance(velocity, list) or len(velocity) != 3:
                raise ScenarioRunFormatError("keep_out_spheres.velocity_mps is invalid")
            for component in velocity:
                _number(component, "keep_out_spheres.velocity_mps", -50, 50)


def _validate_link(value: Any, field: str) -> None:
    required = {"delay_s", "delay_jitter_s", "loss_model", "loss_probability", "burst_start_probability", "burst_end_probability", "contention_mode", "packet_duration_s"}
    if not isinstance(value, dict) or set(value) != required or value.get("loss_model") not in {"none", "independent", "burst"} or value.get("contention_mode") not in {"none", "serialized"}:
        raise ScenarioRunFormatError(f"{field} is invalid")
    _number(value["delay_s"], f"{field}.delay_s", 0, 10)
    _number(value["delay_jitter_s"], f"{field}.delay_jitter_s", 0, 10)
    for key in ("loss_probability", "burst_start_probability", "burst_end_probability"):
        _number(value[key], f"{field}.{key}", 0, 1)
    _number(value["packet_duration_s"], f"{field}.packet_duration_s", 0, 1)
    if value["delay_s"] == 0.0 and value["delay_jitter_s"] != 0.0:
        raise ScenarioRunFormatError(f"{field}.delay_jitter_s must be zero when delay_s is zero")
    if value["delay_jitter_s"] > value["delay_s"]:
        raise ScenarioRunFormatError(f"{field}.delay_jitter_s cannot exceed delay_s")
    if value["loss_model"] in {"none", "burst"} and value["loss_probability"] != 0.0:
        raise ScenarioRunFormatError(f"{field}.loss_probability is ignored by {value['loss_model']} loss_model")
    if value["loss_model"] in {"none", "independent"} and (value["burst_start_probability"] != 0.0 or value["burst_end_probability"] != 1.0):
        raise ScenarioRunFormatError(f"{field} burst probabilities require loss_model burst")
    if value["contention_mode"] == "none" and value["packet_duration_s"] != 0.0:
        raise ScenarioRunFormatError(f"{field}.packet_duration_s requires serialized contention")
    if value["contention_mode"] == "serialized" and value["packet_duration_s"] <= 0.0:
        raise ScenarioRunFormatError(f"{field}.packet_duration_s must be positive for serialized contention")


def _validate_link_event(value: Any) -> None:
    required = {"sender", "receiver", "seq", "send_time", "receive_time", "duplicate", "out_of_order", "loss_reason", "packet_age", "outcome"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["sender"], str) or not isinstance(value["receiver"], str) or value["outcome"] not in {"received", "lost"}:
        raise ScenarioRunFormatError("link event is invalid")
    _integer(value["seq"], "link event seq", 0, JSON_SAFE_INTEGER_MAX)
    _number(value["send_time"], "link event send_time", 0, None)
    if not isinstance(value["duplicate"], bool) or not isinstance(value["out_of_order"], bool):
        raise ScenarioRunFormatError("link event flags are invalid")
    if value["outcome"] == "received":
        if value["loss_reason"] is not None:
            raise ScenarioRunFormatError("received link event is invalid")
        _number(value["receive_time"], "link event receive_time", 0, None)
        _number(value["packet_age"], "link event packet_age", 0, None)
        if value["receive_time"] < value["send_time"] or abs((value["receive_time"] - value["send_time"]) - value["packet_age"]) > 1e-9:
            raise ScenarioRunFormatError("received link event timing is inconsistent")
    else:
        if value["receive_time"] is not None or not isinstance(value["loss_reason"], str) or not value["loss_reason"] or value["packet_age"] is not None:
            raise ScenarioRunFormatError("lost link event is invalid")


def _validate_profile(value: Any) -> None:
    required = {"profile_id", "motor", "propeller", "motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh", "total_mass_kg", "estimated_max_thrust_n"}
    allowed = required | {"status", "source", "calibration_id", "code_revision", "notes", "estimated_thrust_at_rpm_limit_n"}
    if not isinstance(value, dict) or not required.issubset(value) or not set(value).issubset(allowed) or not isinstance(value["profile_id"], str) or not value["profile_id"]:
        raise ScenarioRunFormatError("profile is invalid")
    motor = value["motor"]
    motor_keys = {"part_id", "max_torque_nm", "max_power_w", "max_rpm", "mass_kg", "nominal_voltage_v"}
    prop = value["propeller"]
    prop_keys = {"part_id", "diameter_m", "pitch_m", "max_rpm", "thrust_coefficient", "power_coefficient", "mass_kg"}
    if not isinstance(motor, dict) or set(motor) != motor_keys or not isinstance(prop, dict) or set(prop) != prop_keys:
        raise ScenarioRunFormatError("profile motor or propeller is invalid")
    for key in motor_keys - {"part_id"}:
        _number(motor[key], f"profile.motor.{key}", 0, None, strict_min=True)
    for key in prop_keys - {"part_id"}:
        _number(prop[key], f"profile.propeller.{key}", 0, None, strict_min=True)
    _integer(value["motor_count"], "profile.motor_count", 1, JSON_SAFE_INTEGER_MAX)
    for key in ("arm_length_m", "frame_mass_kg", "battery_capacity_wh", "total_mass_kg", "estimated_max_thrust_n"):
        _number(value[key], f"profile.{key}", 0, None, strict_min=True)
    if "status" in value and not isinstance(value["status"], str):
        raise ScenarioRunFormatError("profile.status is invalid")
    for key in ("source", "calibration_id", "code_revision", "notes"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise ScenarioRunFormatError(f"profile.{key} is invalid")
    if "estimated_thrust_at_rpm_limit_n" in value:
        _number(value["estimated_thrust_at_rpm_limit_n"], "profile.estimated_thrust_at_rpm_limit_n", 0, None, strict_min=True)


def _integer(value: Any, field: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ScenarioRunFormatError(f"{field} must be an integer in range")


def _number(value: Any, field: str, minimum: float | None, maximum: float | None, *, strict_min: bool = False) -> None:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or (minimum is not None and (value <= minimum if strict_min else value < minimum)) or (maximum is not None and value > maximum):
        raise ScenarioRunFormatError(f"{field} must be a finite number in range")


def _decode_frame(raw: Any) -> ScenarioFrame:
    if not isinstance(raw, dict):
        raise ScenarioRunFormatError("each frame must be an object")
    required = {"step_index", "time_s", "target_position_m", "active_count", "agents"}
    if not required.issubset(raw) or not set(raw).issubset(required | {"keep_out_spheres"}):
        raise ScenarioRunFormatError("frame has an invalid field set")
    step = raw["step_index"]
    time_s = raw["time_s"]
    active = raw["active_count"]
    agents = raw["agents"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > JSON_SAFE_INTEGER_MAX:
        raise ScenarioRunFormatError("frame step_index must be a non-negative integer")
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or time_s < 0:
        raise ScenarioRunFormatError("frame time_s must be a non-negative number")
    if isinstance(active, bool) or not isinstance(active, int) or active < 0 or active > JSON_SAFE_INTEGER_MAX:
        raise ScenarioRunFormatError("frame active_count must be a non-negative integer")
    target = _vector(raw["target_position_m"], "target_position_m")
    if not isinstance(agents, list) or not agents:
        raise ScenarioRunFormatError("frame agents must be a non-empty list")
    for agent in agents:
        _validate_agent(agent)
    keep_out_spheres = raw.get("keep_out_spheres", [])
    if not isinstance(keep_out_spheres, list) or len(keep_out_spheres) > 64:
        raise ScenarioRunFormatError("frame keep_out_spheres must be a bounded list")
    for sphere in keep_out_spheres:
        if not isinstance(sphere, dict) or set(sphere) != {"center_m", "radius_m", "label"}:
            raise ScenarioRunFormatError("frame keep-out sphere is invalid")
        _vector(sphere["center_m"], "frame keep_out_spheres.center_m")
        _number(sphere["radius_m"], "frame keep_out_spheres.radius_m", 0, 100)
        if not isinstance(sphere["label"], str) or not sphere["label"]:
            raise ScenarioRunFormatError("frame keep_out_spheres.label is invalid")
    return ScenarioFrame(int(step), float(time_s), target, active, len(agents), tuple(agents), tuple(keep_out_spheres))


def _vector(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ScenarioRunFormatError(f"{field} must be a three-element numeric list")
    for item in value:
        _number(item, field, None, None)
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _validate_agent(value: Any) -> None:
    required = {"agent_id", "profile_id", "active", "position_m", "velocity_mps", "target_estimate_m", "fused_target_m", "target_source_time_s", "target_age_s", "neighbor_count", "min_neighbor_distance_m", "constraint_flags"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["agent_id"], str) or not value["agent_id"] or not isinstance(value["profile_id"], str) or not value["profile_id"] or not isinstance(value["active"], bool):
        raise ScenarioRunFormatError("agent is invalid")
    _vector(value["position_m"], "agent.position_m")
    _vector(value["velocity_mps"], "agent.velocity_mps")
    for key in ("target_estimate_m", "fused_target_m"):
        if value[key] is not None:
            _vector(value[key], f"agent.{key}")
    for key in ("target_source_time_s", "target_age_s", "min_neighbor_distance_m"):
        if value[key] is not None:
            _number(value[key], f"agent.{key}", 0, None)
    _integer(value["neighbor_count"], "agent.neighbor_count", 0, JSON_SAFE_INTEGER_MAX)
    if not isinstance(value["constraint_flags"], list) or not all(isinstance(flag, str) and flag for flag in value["constraint_flags"]):
        raise ScenarioRunFormatError("agent constraint_flags are invalid")
