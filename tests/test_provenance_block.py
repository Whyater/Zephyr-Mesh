import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from sim.bench_trace import BENCH_TRACE_SCHEMA, validate_bench_trace
from sim.vision_trace import VISION_TRACE_SCHEMA, validate_vision_trace


ROOT = Path(__file__).parents[1]
SHA = "a" * 64


def provenance(*, calibration_ref=SHA, ground_truth=None):
    value = {
        "instrument": {"kind": "camera", "model": "model-1", "settings": {}},
        "calibration_ref": calibration_ref,
        "raw_sha256": SHA,
        "code_revision": "abc123-dirty",
        "captured_at": "2026-10-14T09:30:00-06:00",
        "operator": "operator one",
        "ground_truth": ground_truth if ground_truth is not None else {
            "method": "survey",
            "values": {"position_m": [1.0, 2.0, 3.0]},
            "uncertainty": {"position_m": 0.01},
        },
    }
    if calibration_ref is None:
        value["calibration_ref_null_reason"] = "calibration unavailable"
    return value


def vision_payload(status="measured"):
    return {
        "schema": VISION_TRACE_SCHEMA,
        "trace_id": "vision-1",
        "status": status,
        "clock_domain": "shared_monotonic",
        "capture": {
            "source_type": "camera",
            "source_id": "camera-1",
            "device_ids": ["host-1"],
            "model": "detector-1",
            "resolution_px": {"width": 640, "height": 480},
            "frame_rate_hz": 30,
        },
        "source": "capture.jsonl",
        "provenance": provenance(),
        "observations": [{
            "observation_id": "obs-1", "frame_id": "frame-1", "subject_id": "subject-1",
            "capture_time_s": 1.0, "outcome": "detected", "position_m": [1.0, 2.0, 3.0],
            "position_frame": "world",
            "ground_truth": {"position_m": [1.0, 2.0, 3.0], "uncertainty_m": 0.1, "method": "survey"},
        }],
    }


def bench_payload(status="measured"):
    return {
        "schema": BENCH_TRACE_SCHEMA,
        "trace_id": "bench-1",
        "status": status,
        "profile_id": "profile-1",
        "motor_part_id": "motor-1",
        "propeller_part_id": "prop-1",
        "measurement_scope": "single_rotor",
        "clock_domain": "shared_monotonic",
        "capture": {
            "device_ids": ["host-1"], "instrument_ids": ["load-1"],
            "air_density_kg_m3": 1.2, "ambient_temperature_c": 22.0,
        },
        "source": "bench.jsonl",
        "calibration_id": "legacy-calibration-id",
        "provenance": provenance(),
        "samples": [{
            "sample_id": "sample-1", "timestamp_s": 1.0, "quality": "valid",
            "rpm": 1000, "thrust_n": 0.2, "voltage_v": 3.7, "current_a": 1.0, "duration_s": 2.0,
        }],
    }


def schema_for(kind):
    filename = "vision_trace_schema.json" if kind == "vision" else "bench_trace_schema.json"
    return Draft202012Validator(
        json.loads((ROOT / "sim" / filename).read_text()), format_checker=FormatChecker()
    )


@pytest.mark.parametrize("kind", ["vision", "bench"])
def test_measured_source_only_is_rejected(kind):
    payload = vision_payload() if kind == "vision" else bench_payload()
    payload.pop("provenance")
    if kind == "vision":
        payload["observations"][0].pop("ground_truth")
    if kind == "bench":
        payload["calibration_id"] = "legacy-calibration-id"
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
def test_complete_measured_trace_passes_python_and_jsonschema(kind):
    payload = vision_payload() if kind == "vision" else bench_payload()
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    assert schema_for(kind).is_valid(payload)
    trace = validator(payload)
    assert trace.as_dict()["provenance"] == payload["provenance"]


@pytest.mark.parametrize("kind", ["vision", "bench"])
@pytest.mark.parametrize("key", ["instrument", "calibration_ref", "raw_sha256", "code_revision", "captured_at", "operator", "ground_truth"])
def test_missing_required_provenance_key_is_rejected(kind, key):
    payload = vision_payload() if kind == "vision" else bench_payload()
    del payload["provenance"][key]
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
@pytest.mark.parametrize("location", ["provenance", "instrument", "ground_truth"])
def test_provenance_unknown_key_is_rejected(kind, location):
    payload = vision_payload() if kind == "vision" else bench_payload()
    target = payload["provenance"] if location == "provenance" else payload["provenance"][location]
    target["unexpected"] = 1
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
@pytest.mark.parametrize("field", ["captured_at", "raw_sha256"])
def test_bad_timestamp_or_sha_is_rejected(kind, field):
    payload = vision_payload() if kind == "vision" else bench_payload()
    payload["provenance"][field] = "2026-10-14T09:30:00" if field == "captured_at" else "bad"
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
def test_null_calibration_requires_reason_and_ground_truth_null_requires_reason(kind):
    payload = vision_payload() if kind == "vision" else bench_payload()
    payload["provenance"]["calibration_ref"] = None
    payload["provenance"].pop("calibration_ref_null_reason", None)
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)
    payload["provenance"]["calibration_ref_null_reason"] = "not available"
    assert schema_for(kind).is_valid(payload)
    validator(payload)
    payload["provenance"]["ground_truth"] = None
    payload["provenance"].pop("ground_truth_null_reason", None)
    with pytest.raises(ValueError):
        validator(payload)
    assert not schema_for(kind).is_valid(payload)
    payload["provenance"]["ground_truth_null_reason"] = "not available"
    assert schema_for(kind).is_valid(payload)
    validator(payload)


def test_vision_observation_ground_truth_rules_and_round_trip():
    payload = vision_payload()
    trace = validate_vision_trace(payload)
    assert trace.as_dict()["observations"][0]["ground_truth"] == payload["observations"][0]["ground_truth"]
    assert validate_vision_trace(trace.as_dict()).as_dict() == trace.as_dict()
    payload["observations"][0]["ground_truth"]["unexpected"] = 1
    with pytest.raises(ValueError):
        validate_vision_trace(payload)
    assert not schema_for("vision").is_valid(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
def test_synthetic_examples_still_validate(kind):
    filename = "vision_trace_example.json" if kind == "vision" else "bench_trace_example.json"
    payload = json.loads((ROOT / "sim" / filename).read_text())
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    assert schema_for(kind).is_valid(payload)
    validator(payload)


@pytest.mark.parametrize("kind", ["vision", "bench"])
def test_python_and_schema_agree_on_cases(kind):
    base = vision_payload() if kind == "vision" else bench_payload()
    cases = [base]
    missing = copy.deepcopy(base)
    del missing["provenance"]["operator"]
    cases.append(missing)
    unknown = copy.deepcopy(base)
    unknown["provenance"]["extra"] = True
    cases.append(unknown)
    validator = validate_vision_trace if kind == "vision" else validate_bench_trace
    schema = schema_for(kind)
    for payload in cases:
        python_valid = True
        try:
            validator(payload)
        except ValueError:
            python_valid = False
        assert python_valid == schema.is_valid(payload)
