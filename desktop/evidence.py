"""Portable, read-only evidence reports for the desktop applications.

The report is an intentionally small envelope around the existing trace and
profile summaries. It gives both desktop surfaces one decoder and one honest
boundary: a report can be inspected, but it cannot command a vehicle or turn
synthetic observations into measured performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sim.bench_metrics import summarize_bench_trace
from sim.bench_trace import BENCH_TRACE_SCHEMA, BenchTrace
from sim.hardware import HARDWARE_PROFILE_SCHEMA, HardwareProfile
from sim.trace import TRACE_SCHEMA, EspNowTrace
from sim.trace_metrics import summarize_trace
from sim.vision_metrics import summarize_vision_trace
from sim.vision_trace import VISION_TRACE_SCHEMA, VisionTrace

EVIDENCE_REPORT_SCHEMA = "zephyr-evidence-report-1"
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64
SAFE_INTEGER_MAX = 9_007_199_254_740_991
SUPPORTED_KINDS = frozenset({"espnow", "vision", "hardware", "bench"})
SUPPORTED_STATUSES = frozenset({"measured", "synthetic", "fixture"})
RAW_SCHEMAS = {
    "espnow": TRACE_SCHEMA,
    "vision": VISION_TRACE_SCHEMA,
    "hardware": HARDWARE_PROFILE_SCHEMA,
    "bench": BENCH_TRACE_SCHEMA,
}


class EvidenceError(ValueError):
    """A report is missing a portable-envelope requirement."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} must be a non-empty string")
    return value


def _finite_json(value: Any, field: str = "report", depth: int = 0) -> Any:
    """Reject non-JSON finite values while retaining nulls and nesting."""
    if depth > MAX_JSON_DEPTH:
        raise EvidenceError(f"{field} exceeds the {MAX_JSON_DEPTH}-level nesting limit")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > SAFE_INTEGER_MAX:
            raise EvidenceError(f"{field} contains an integer outside the cross-platform safe range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_finite_json(item, f"{field}[]", depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: _finite_json(item, f"{field}.{key}", depth + 1) for key, item in value.items()}
    raise EvidenceError(f"{field} contains an unsupported JSON value")


def _parse_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise EvidenceError(f"report exceeds {MAX_REPORT_BYTES // (1024 * 1024)} MiB")
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(str(exc)) from exc
    if len(raw) > MAX_REPORT_BYTES:
        raise EvidenceError(f"report exceeds {MAX_REPORT_BYTES // (1024 * 1024)} MiB")
    try:
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = item
            return result

        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"invalid constant {constant}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError("report must be a JSON object")
    return value, raw


def _validate_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"schema", "kind", "source", "status", "summary", "limitations"}
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if missing:
        raise EvidenceError(f"report missing required field(s): {', '.join(missing)}")
    if unknown:
        raise EvidenceError(f"report contains unknown field(s): {', '.join(unknown)}")
    if value["schema"] != EVIDENCE_REPORT_SCHEMA:
        raise EvidenceError(f"schema must be {EVIDENCE_REPORT_SCHEMA}")
    kind = _text(value["kind"], "kind")
    if kind not in SUPPORTED_KINDS:
        raise EvidenceError(f"kind must be one of {sorted(SUPPORTED_KINDS)}")
    status = _text(value["status"], "status")
    if status not in SUPPORTED_STATUSES:
        raise EvidenceError(f"status must be one of {sorted(SUPPORTED_STATUSES)}")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"name", "sha256", "schema"}:
        raise EvidenceError("source must contain exactly name, sha256, and schema")
    name = _text(source["name"], "source.name")
    digest = _text(source["sha256"], "source.sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise EvidenceError("source.sha256 must be 64 lowercase hexadecimal characters")
    source_schema = _text(source["schema"], "source.schema")
    if source_schema != RAW_SCHEMAS[kind]:
        raise EvidenceError(f"source.schema must be {RAW_SCHEMAS[kind]} for kind={kind}")
    if not isinstance(value["summary"], dict):
        raise EvidenceError("summary must be an object")
    limitations = value["limitations"]
    if not isinstance(limitations, list) or any(not isinstance(item, str) or not item.strip() for item in limitations):
        raise EvidenceError("limitations must be an array of non-empty strings")
    validated = {
        "schema": EVIDENCE_REPORT_SCHEMA,
        "kind": kind,
        "source": {"name": name, "sha256": digest, "schema": source_schema},
        "status": status,
        "summary": _finite_json(value["summary"], "summary"),
        "limitations": limitations,
    }
    if not validated["limitations"]:
        raise EvidenceError("limitations must contain at least one entry")
    summary = validated["summary"]
    summary_status = summary.get("status")
    if kind in {"espnow", "vision", "bench"}:
        provenance = summary.get("provenance")
        summary_status = provenance.get("status") if isinstance(provenance, dict) else None
    if summary_status != status:
        raise EvidenceError("status must match the status retained in summary provenance")
    return validated


def load_evidence_report(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a portable report envelope."""
    report, _ = _parse_json(Path(path))
    return _validate_envelope(report)


def _load_raw(kind: str, value: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if kind == "espnow":
        trace = EspNowTrace.from_dict(value)
        return trace.status, TRACE_SCHEMA, summarize_trace(trace)
    if kind == "vision":
        trace = VisionTrace.from_dict(value)
        return trace.status, VISION_TRACE_SCHEMA, summarize_vision_trace(trace)
    if kind == "hardware":
        profile = HardwareProfile.from_dict(value)
        return profile.status, HARDWARE_PROFILE_SCHEMA, profile.as_document()
    if kind == "bench":
        trace = BenchTrace.from_dict(value)
        return trace.status, BENCH_TRACE_SCHEMA, summarize_bench_trace(trace)
    raise EvidenceError(f"unsupported kind: {kind}")


def build_evidence_report(path: str | Path, kind: str) -> dict[str, Any]:
    """Summarize one supported raw artifact into the shared desktop envelope."""
    kind = _text(kind, "kind")
    if kind not in SUPPORTED_KINDS:
        raise EvidenceError(f"kind must be one of {sorted(SUPPORTED_KINDS)}")
    source_path = Path(path)
    value, raw = _parse_json(source_path)
    try:
        raw_schema_path = {
            "espnow": Path(__file__).parents[1] / "sim" / "trace_schema.json",
            "vision": Path(__file__).parents[1] / "sim" / "vision_trace_schema.json",
            "hardware": Path(__file__).parents[1] / "sim" / "hardware_profile_schema.json",
            "bench": Path(__file__).parents[1] / "sim" / "bench_trace_schema.json",
        }[kind]
        Draft202012Validator(json.loads(raw_schema_path.read_text(encoding="utf-8"))).validate(value)
        status, schema, summary = _load_raw(kind, value)
        summary = _finite_json(summary, "summary")
    except (ValueError, TypeError, OverflowError, RecursionError, ValidationError) as exc:
        raise EvidenceError(str(exc)) from exc
    limitations = {
        "espnow": ["Observed records only. This report does not establish radio tolerance, coverage, or flight performance."],
        "vision": ["Observed records only. This report does not establish camera calibration, recall, or tracker performance."],
        "hardware": ["Profile inputs are not bench measurements unless status and provenance say so. Derived estimates are not flight performance."],
        "bench": ["Descriptive bench observations only. No motor map is fitted and electrical input power is not shaft power."],
    }[kind]
    report = {
        "schema": EVIDENCE_REPORT_SCHEMA,
        "kind": kind,
        "source": {"name": source_path.name, "sha256": hashlib.sha256(raw).hexdigest(), "schema": schema},
        "status": status,
        "summary": summary,
        "limitations": limitations,
    }
    return _validate_envelope(report)


def write_evidence_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Validate and write a report with stable formatting and no NaN values."""
    validated = _validate_envelope(report)
    Path(path).write_text(json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
