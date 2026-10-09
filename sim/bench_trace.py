"""Validated motor/propeller bench observations.

The trace preserves raw test conditions and incomplete readings so future
calibration can be reproducible. Summaries report measured electrical input
power and ranges only; they do not fit a motor map or claim flight behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

BENCH_TRACE_SCHEMA = "zephyr-bench-trace-1"
TRACE_STATUSES = {"measured", "synthetic", "fixture"}
SAMPLE_QUALITIES = {"valid", "invalid", "unknown"}
CLOCK_DOMAINS = {"shared_monotonic", "device_and_host", "unknown"}
MEASUREMENT_SCOPES = {"single_rotor", "airframe"}


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


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown field(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class BenchSample:
    """One steady-state bench observation or an explicitly incomplete record."""

    sample_id: str
    timestamp_s: float
    quality: str
    rpm: float | None = None
    thrust_n: float | None = None
    voltage_v: float | None = None
    current_a: float | None = None
    duration_s: float | None = None
    temperature_c: float | None = None
    quality_reason: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchSample":
        if not isinstance(raw, Mapping):
            raise ValueError("sample must be an object")
        _reject_unknown(raw, {"sample_id", "timestamp_s", "quality", "rpm", "thrust_n", "voltage_v", "current_a", "duration_s", "temperature_c", "quality_reason"}, "sample")
        quality = _required_text(raw.get("quality"), "quality")
        if quality not in SAMPLE_QUALITIES:
            raise ValueError(f"quality must be one of {sorted(SAMPLE_QUALITIES)}")
        numeric = {
            "rpm": _optional_nonnegative(raw.get("rpm"), "rpm"),
            "thrust_n": _optional_nonnegative(raw.get("thrust_n"), "thrust_n"),
            "voltage_v": _optional_nonnegative(raw.get("voltage_v"), "voltage_v"),
            "current_a": _optional_nonnegative(raw.get("current_a"), "current_a"),
            "duration_s": _optional_nonnegative(raw.get("duration_s"), "duration_s"),
            "temperature_c": None if raw.get("temperature_c") is None else _finite(raw["temperature_c"], "temperature_c"),
        }
        reason = raw.get("quality_reason")
        if reason is not None:
            reason = _required_text(reason, "quality_reason")
        if quality == "valid":
            required = ["rpm", "thrust_n", "voltage_v", "current_a", "duration_s"]
            missing = [name for name in required if numeric[name] is None]
            if missing:
                raise ValueError(f"valid samples require {', '.join(missing)}")
            if reason is not None:
                raise ValueError("valid samples cannot have quality_reason")
        elif reason is None:
            raise ValueError(f"{quality} samples require quality_reason")
        return cls(
            sample_id=_required_text(raw.get("sample_id"), "sample_id"),
            timestamp_s=_nonnegative(raw.get("timestamp_s"), "timestamp_s"),
            quality=quality,
            quality_reason=reason,
            **numeric,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchTrace:
    """Validated bench manifest plus ordered observations."""

    trace_id: str
    status: str
    profile_id: str
    motor_part_id: str
    propeller_part_id: str
    measurement_scope: str
    clock_domain: str
    capture: dict[str, Any]
    samples: tuple[BenchSample, ...]
    source: str | None = None
    calibration_id: str | None = None
    code_revision: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("trace must be an object")
        _reject_unknown(raw, {"schema", "trace_id", "status", "profile_id", "motor_part_id", "propeller_part_id", "measurement_scope", "clock_domain", "capture", "samples", "source", "calibration_id", "code_revision", "notes"}, "trace")
        if raw.get("schema") != BENCH_TRACE_SCHEMA:
            raise ValueError(f"schema must be {BENCH_TRACE_SCHEMA}")
        status = _required_text(raw.get("status"), "status")
        if status not in TRACE_STATUSES:
            raise ValueError(f"status must be one of {sorted(TRACE_STATUSES)}")
        scope = _required_text(raw.get("measurement_scope"), "measurement_scope")
        if scope not in MEASUREMENT_SCOPES:
            raise ValueError(f"measurement_scope must be one of {sorted(MEASUREMENT_SCOPES)}")
        clock_domain = _required_text(raw.get("clock_domain"), "clock_domain")
        if clock_domain not in CLOCK_DOMAINS:
            raise ValueError(f"clock_domain must be one of {sorted(CLOCK_DOMAINS)}")
        capture_raw = raw.get("capture")
        if not isinstance(capture_raw, Mapping):
            raise ValueError("capture must be an object")
        _reject_unknown(capture_raw, {"device_ids", "instrument_ids", "air_density_kg_m3", "ambient_temperature_c", "test_notes"}, "capture")
        device_ids = capture_raw.get("device_ids")
        instrument_ids = capture_raw.get("instrument_ids")
        for values, field in ((device_ids, "capture.device_ids"), (instrument_ids, "capture.instrument_ids")):
            if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field} must be a non-empty string array")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must be unique")
        capture = {
            "device_ids": list(device_ids),
            "instrument_ids": list(instrument_ids),
            "air_density_kg_m3": _nonnegative(capture_raw.get("air_density_kg_m3"), "capture.air_density_kg_m3"),
            "ambient_temperature_c": _finite(capture_raw.get("ambient_temperature_c"), "capture.ambient_temperature_c"),
        }
        if capture["air_density_kg_m3"] <= 0:
            raise ValueError("capture.air_density_kg_m3 must be positive")
        if "test_notes" in capture_raw and capture_raw["test_notes"] is None:
            raise ValueError("capture.test_notes must be a non-empty string when supplied")
        if capture_raw.get("test_notes") is not None:
            capture["test_notes"] = _required_text(capture_raw["test_notes"], "capture.test_notes")
        samples_raw = raw.get("samples")
        if not isinstance(samples_raw, list):
            raise ValueError("samples must be an array")
        samples = tuple(BenchSample.from_dict(item) for item in samples_raw)
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique")
        if clock_domain == "shared_monotonic":
            timestamps = [sample.timestamp_s for sample in samples]
            if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
                raise ValueError("shared_monotonic sample timestamps must be strictly increasing")
        if status == "measured" and raw.get("source") is None:
            raise ValueError("measured traces require source")
        if status == "measured" and raw.get("calibration_id") is None:
            raise ValueError("measured traces require calibration_id")
        for field in ("source", "calibration_id", "code_revision", "notes"):
            if field in raw and raw[field] is None:
                raise ValueError(f"{field} must be a non-empty string when supplied")
        return cls(
            trace_id=_required_text(raw.get("trace_id"), "trace_id"),
            status=status,
            profile_id=_required_text(raw.get("profile_id"), "profile_id"),
            motor_part_id=_required_text(raw.get("motor_part_id"), "motor_part_id"),
            propeller_part_id=_required_text(raw.get("propeller_part_id"), "propeller_part_id"),
            measurement_scope=scope,
            clock_domain=clock_domain,
            capture=capture,
            samples=samples,
            source=None if raw.get("source") is None else _required_text(raw["source"], "source"),
            calibration_id=None if raw.get("calibration_id") is None else _required_text(raw["calibration_id"], "calibration_id"),
            code_revision=None if raw.get("code_revision") is None else _required_text(raw["code_revision"], "code_revision"),
            notes=None if raw.get("notes") is None else _required_text(raw["notes"], "notes"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "BenchTrace":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": BENCH_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "status": self.status,
            "profile_id": self.profile_id,
            "motor_part_id": self.motor_part_id,
            "propeller_part_id": self.propeller_part_id,
            "measurement_scope": self.measurement_scope,
            "clock_domain": self.clock_domain,
            "capture": dict(self.capture),
            "samples": [sample.as_dict() for sample in self.samples],
        }
        for key in ("source", "calibration_id", "code_revision", "notes"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def validate_bench_trace(raw: Mapping[str, Any]) -> BenchTrace:
    return BenchTrace.from_dict(raw)
