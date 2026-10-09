"""Validated camera/tracker trace observations.

The schema is an evidence boundary for future measured vision inputs. It keeps
capture provenance, timing domains, detections, misses, and unresolved records
explicit. It does not infer target identity, camera calibration, or tracker
performance from missing fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

VISION_TRACE_SCHEMA = "zephyr-vision-trace-1"
TRACE_STATUSES = {"measured", "synthetic", "fixture"}
OUTCOMES = {"detected", "missed", "unknown"}
CLOCK_DOMAINS = {"shared_monotonic", "device_and_host", "unknown"}


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown field(s): {', '.join(unknown)}")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _optional_nonnegative(value: Any, field: str) -> float | None:
    return None if value is None else _nonnegative(value, field)


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _vector3(value: Any, field: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a three-element numeric array or null")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _bbox(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("bbox_px must be an object or null")
    required = {"x", "y", "width", "height"}
    if set(value) != required:
        raise ValueError("bbox_px must contain exactly x, y, width, and height")
    x = _nonnegative(value["x"], "bbox_px.x")
    y = _nonnegative(value["y"], "bbox_px.y")
    width = _finite(value["width"], "bbox_px.width")
    height = _finite(value["height"], "bbox_px.height")
    if width <= 0 or height <= 0:
        raise ValueError("bbox_px width and height must be positive")
    return {"x": x, "y": y, "width": width, "height": height}


@dataclass(frozen=True)
class VisionObservation:
    """One subject observation associated with one captured frame."""

    observation_id: str
    frame_id: str
    subject_id: str
    capture_time_s: float
    outcome: str
    processing_time_s: float | None = None
    confidence: float | None = None
    bbox_px: dict[str, float] | None = None
    position_m: tuple[float, float, float] | None = None
    position_frame: str | None = None
    miss_reason: str | None = None
    unknown_reason: str | None = None
    duplicate: bool | None = None
    out_of_order: bool | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VisionObservation":
        if not isinstance(raw, Mapping):
            raise ValueError("observation must be an object")
        _reject_unknown(raw, {
            "observation_id", "frame_id", "subject_id", "capture_time_s", "outcome",
            "processing_time_s", "confidence", "bbox_px", "position_m", "position_frame",
            "miss_reason", "unknown_reason", "duplicate", "out_of_order",
        }, "observation")
        outcome = _required_text(raw.get("outcome"), "outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        confidence = raw.get("confidence")
        if confidence is not None:
            confidence = _finite(confidence, "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
        bbox_px = _bbox(raw.get("bbox_px"))
        position_m = _vector3(raw.get("position_m"), "position_m")
        position_frame = raw.get("position_frame")
        if position_frame is not None:
            position_frame = _required_text(position_frame, "position_frame")
        if position_m is not None and position_frame is None:
            raise ValueError("position_m requires position_frame")
        if position_m is None and position_frame is not None:
            raise ValueError("position_frame requires position_m")
        miss_reason = raw.get("miss_reason")
        if miss_reason is not None:
            miss_reason = _required_text(miss_reason, "miss_reason")
        unknown_reason = raw.get("unknown_reason")
        if unknown_reason is not None:
            unknown_reason = _required_text(unknown_reason, "unknown_reason")
        if outcome == "detected" and bbox_px is None and position_m is None:
            raise ValueError("detected observations require bbox_px or position_m")
        if outcome == "missed" and miss_reason is None:
            raise ValueError("missed observations require miss_reason")
        if outcome == "unknown" and unknown_reason is None:
            raise ValueError("unknown observations require unknown_reason")
        if outcome != "detected" and (bbox_px is not None or position_m is not None or confidence is not None):
            raise ValueError(f"{outcome} observations cannot include detection fields")
        if outcome != "missed" and miss_reason is not None:
            raise ValueError(f"{outcome} observations cannot include miss_reason")
        if outcome != "unknown" and unknown_reason is not None:
            raise ValueError(f"{outcome} observations cannot include unknown_reason")
        return cls(
            observation_id=_required_text(raw.get("observation_id"), "observation_id"),
            frame_id=_required_text(raw.get("frame_id"), "frame_id"),
            subject_id=_required_text(raw.get("subject_id"), "subject_id"),
            capture_time_s=_nonnegative(raw.get("capture_time_s"), "capture_time_s"),
            outcome=outcome,
            processing_time_s=_optional_nonnegative(raw.get("processing_time_s"), "processing_time_s"),
            confidence=confidence,
            bbox_px=bbox_px,
            position_m=position_m,
            position_frame=position_frame,
            miss_reason=miss_reason,
            unknown_reason=unknown_reason,
            duplicate=_optional_bool(raw.get("duplicate"), "duplicate"),
            out_of_order=_optional_bool(raw.get("out_of_order"), "out_of_order"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisionTrace:
    """Validated trace manifest plus ordered camera/tracker observations."""

    trace_id: str
    status: str
    clock_domain: str
    observations: tuple[VisionObservation, ...]
    capture: dict[str, Any]
    source: str | None = None
    code_revision: str | None = None
    clock_uncertainty_s: float | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VisionTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("trace must be an object")
        _reject_unknown(raw, {
            "schema", "trace_id", "status", "clock_domain", "capture", "observations",
            "source", "code_revision", "clock_uncertainty_s", "notes",
        }, "trace")
        if raw.get("schema") != VISION_TRACE_SCHEMA:
            raise ValueError(f"schema must be {VISION_TRACE_SCHEMA}")
        status = _required_text(raw.get("status"), "status")
        if status not in TRACE_STATUSES:
            raise ValueError(f"status must be one of {sorted(TRACE_STATUSES)}")
        clock_domain = _required_text(raw.get("clock_domain"), "clock_domain")
        if clock_domain not in CLOCK_DOMAINS:
            raise ValueError(f"clock_domain must be one of {sorted(CLOCK_DOMAINS)}")
        capture_raw = raw.get("capture")
        if not isinstance(capture_raw, Mapping):
            raise ValueError("capture must be an object")
        _reject_unknown(capture_raw, {
            "source_type", "source_id", "device_ids", "model", "calibration_id",
            "resolution_px", "frame_rate_hz",
        }, "capture")
        source_type = _required_text(capture_raw.get("source_type"), "capture.source_type")
        if source_type not in {"camera", "tracker", "recording"}:
            raise ValueError("capture.source_type must be camera, tracker, or recording")
        device_ids = capture_raw.get("device_ids")
        if not isinstance(device_ids, list) or not device_ids or any(not isinstance(value, str) or not value.strip() for value in device_ids):
            raise ValueError("capture.device_ids must be a non-empty string array")
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("capture.device_ids must be unique")
        resolution = capture_raw.get("resolution_px")
        if resolution is not None:
            if not isinstance(resolution, Mapping) or set(resolution) != {"width", "height"}:
                raise ValueError("capture.resolution_px must contain width and height")
            width = resolution["width"]
            height = resolution["height"]
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or isinstance(height, bool) or not isinstance(height, int) or height <= 0:
                raise ValueError("capture.resolution_px width and height must be positive integers")
            resolution = {"width": width, "height": height}
        frame_rate = _optional_nonnegative(capture_raw.get("frame_rate_hz"), "capture.frame_rate_hz")
        capture = {
            "source_type": source_type,
            "source_id": _required_text(capture_raw.get("source_id"), "capture.source_id"),
            "device_ids": list(device_ids),
            "model": _required_text(capture_raw.get("model"), "capture.model"),
            "resolution_px": resolution,
            "frame_rate_hz": frame_rate,
        }
        calibration_id = capture_raw.get("calibration_id")
        if "calibration_id" in capture_raw and calibration_id is None:
            raise ValueError("capture.calibration_id must be a non-empty string when supplied")
        if calibration_id is not None:
            capture["calibration_id"] = _required_text(calibration_id, "capture.calibration_id")
        observations_raw = raw.get("observations")
        if not isinstance(observations_raw, list):
            raise ValueError("observations must be an array")
        observations = tuple(VisionObservation.from_dict(item) for item in observations_raw)
        observation_ids = [item.observation_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id values must be unique")
        if clock_domain == "shared_monotonic":
            for item in observations:
                if item.processing_time_s is not None and item.processing_time_s < item.capture_time_s:
                    raise ValueError("shared_monotonic processing_time_s cannot precede capture_time_s")
        if status == "measured" and raw.get("source") is None:
            raise ValueError("measured traces require source")
        if "source" in raw and raw["source"] is None:
            raise ValueError("source must be a non-empty string when supplied")
        for field in ("code_revision", "notes"):
            if field in raw and raw[field] is None:
                raise ValueError(f"{field} must be a non-empty string when supplied")
        return cls(
            trace_id=_required_text(raw.get("trace_id"), "trace_id"),
            status=status,
            clock_domain=clock_domain,
            observations=observations,
            capture=capture,
            source=None if raw.get("source") is None else _required_text(raw["source"], "source"),
            code_revision=None if raw.get("code_revision") is None else _required_text(raw["code_revision"], "code_revision"),
            clock_uncertainty_s=_optional_nonnegative(raw.get("clock_uncertainty_s"), "clock_uncertainty_s"),
            notes=None if raw.get("notes") is None else _required_text(raw["notes"], "notes"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "VisionTrace":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": VISION_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "status": self.status,
            "clock_domain": self.clock_domain,
            "capture": dict(self.capture),
            "observations": [item.as_dict() for item in self.observations],
        }
        for key in ("source", "code_revision", "clock_uncertainty_s", "notes"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def validate_vision_trace(raw: Mapping[str, Any]) -> VisionTrace:
    """Validate a decoded JSON object without filling missing observations."""

    return VisionTrace.from_dict(raw)


def iter_detected(trace: VisionTrace) -> Iterable[VisionObservation]:
    """Yield detections in processing-time order, falling back to capture time."""

    def order_key(item: VisionObservation) -> tuple[float, str]:
        return (item.processing_time_s if item.processing_time_s is not None else item.capture_time_s, item.observation_id)

    return iter(sorted((item for item in trace.observations if item.outcome == "detected"), key=order_key))
