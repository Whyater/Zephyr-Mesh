import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.vision_metrics import summarize_vision_trace
from sim.vision_trace import VISION_TRACE_SCHEMA, VisionTrace, iter_detected, validate_vision_trace
from tools.analyze_vision_trace import main as analyze_vision_trace


def vision_payload():
    return {
        "schema": VISION_TRACE_SCHEMA,
        "trace_id": "camera-lab-2026-10-09-001",
        "status": "measured",
        "clock_domain": "shared_monotonic",
        "capture": {
            "source_type": "camera",
            "source_id": "front-camera",
            "device_ids": ["host-a"],
            "model": "detector-0.1",
            "calibration_id": "cal-1",
            "resolution_px": {"width": 1280, "height": 720},
            "frame_rate_hz": 30,
        },
        "source": "camera-log.jsonl",
        "clock_uncertainty_s": 0.0005,
        "observations": [
            {
                "observation_id": "obs-2",
                "frame_id": "frame-2",
                "subject_id": "tag-a",
                "capture_time_s": 1.1,
                "processing_time_s": 1.13,
                "outcome": "detected",
                "confidence": 0.8,
                "bbox_px": {"x": 10, "y": 20, "width": 30, "height": 40},
            },
            {
                "observation_id": "obs-1",
                "frame_id": "frame-1",
                "subject_id": "tag-a",
                "capture_time_s": 1.0,
                "processing_time_s": 1.02,
                "outcome": "detected",
                "confidence": 0.9,
                "position_m": [1.0, 2.0, 3.0],
                "position_frame": "world",
            },
            {
                "observation_id": "obs-3",
                "frame_id": "frame-3",
                "subject_id": "tag-a",
                "capture_time_s": 1.2,
                "processing_time_s": 1.25,
                "outcome": "missed",
                "miss_reason": "occluded",
            },
            {
                "observation_id": "obs-4",
                "frame_id": "frame-4",
                "subject_id": "tag-a",
                "capture_time_s": 1.3,
                "outcome": "unknown",
                "unknown_reason": "recording_ended",
            },
        ],
    }


def test_schema_and_python_validator_accept_example_and_measured_fixture():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "vision_trace_schema.json").read_text())
    Draft202012Validator(schema).validate(vision_payload())
    trace = validate_vision_trace(vision_payload())
    assert trace.capture["resolution_px"]["width"] == 1280
    example = json.loads((Path(__file__).parents[1] / "sim" / "vision_trace_example.json").read_text())
    Draft202012Validator(schema).validate(example)
    assert validate_vision_trace(example).status == "fixture"


def test_json_schema_rejects_outcome_field_conflicts_and_position_without_frame():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "vision_trace_schema.json").read_text())
    payload = vision_payload()
    payload["observations"][0]["outcome"] = "missed"
    payload["observations"][0]["miss_reason"] = "occluded"
    assert not Draft202012Validator(schema).is_valid(payload)
    payload = vision_payload()
    payload["observations"][1].pop("position_frame")
    assert not Draft202012Validator(schema).is_valid(payload)
    payload = vision_payload()
    payload["observations"][0]["unexpected"] = 1
    assert not Draft202012Validator(schema).is_valid(payload)
    payload = vision_payload()
    payload["source"] = None
    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValueError, match="measured traces require source"):
        validate_vision_trace(payload)


def test_round_trip_and_detection_iteration_are_stable(tmp_path):
    trace = validate_vision_trace(vision_payload())
    path = tmp_path / "vision.json"
    trace.write(path)
    decoded = VisionTrace.from_path(path)
    assert decoded.as_dict() == trace.as_dict()
    assert [item.observation_id for item in iter_detected(decoded)] == ["obs-1", "obs-2"]


def test_observation_fields_are_outcome_specific():
    payload = vision_payload()
    payload["observations"][0]["outcome"] = "missed"
    payload["observations"][0]["miss_reason"] = "occluded"
    with pytest.raises(ValueError, match="detection fields"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][0].pop("bbox_px")
    payload["observations"][0].pop("confidence")
    with pytest.raises(ValueError, match="bbox_px or position_m"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][2].pop("miss_reason")
    with pytest.raises(ValueError, match="miss_reason"):
        validate_vision_trace(payload)


def test_trace_rejects_clock_and_numeric_ambiguity():
    payload = vision_payload()
    payload["observations"][0]["processing_time_s"] = 1.0
    with pytest.raises(ValueError, match="precede"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][0]["confidence"] = "0.8"
    with pytest.raises(ValueError, match="finite"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][1]["position_m"] = [1, 2, 3]
    payload["observations"][1].pop("position_frame")
    with pytest.raises(ValueError, match="position_frame"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][0]["observation_id"] = payload["observations"][1]["observation_id"]
    with pytest.raises(ValueError, match="observation_id"):
        validate_vision_trace(payload)
    payload = vision_payload()
    payload["observations"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        validate_vision_trace(payload)


def test_unknown_observations_are_kept_as_unresolved_and_bounds_are_explicit():
    summary = summarize_vision_trace(validate_vision_trace(vision_payload()))
    observations = summary["observations"]
    assert observations["observed_records"] == 4
    assert observations["unknown_unresolved"] == 1
    assert observations["observed_detection_rate"] is None
    assert observations["detection_rate_bounds"] == {"interpretation": "identification bounds over observed records; not confidence intervals", "lower": 0.5, "upper": 0.75}
    assert summary["processing_latency"]["p50"] == pytest.approx(0.03)
    assert summary["confidence"]["p95"] == pytest.approx(0.895)
    assert observations["duplicates"] is None
    assert observations["duplicate_flag_unknown"] == 4
    assert observations["duplicate_metadata_available"] is False
    assert observations["out_of_order"] is None


def test_separate_clock_does_not_invent_processing_latency():
    payload = vision_payload()
    payload["clock_domain"] = "device_and_host"
    summary = summarize_vision_trace(validate_vision_trace(payload))
    assert summary["processing_latency"]["available"] is False


def test_empty_trace_is_json_safe():
    payload = vision_payload()
    payload["observations"] = []
    summary = summarize_vision_trace(validate_vision_trace(payload))
    assert summary["observations"]["observed_records"] == 0
    assert summary["confidence"]["available"] is False


def test_mixed_duplicate_and_order_flags_keep_unknown_count():
    payload = vision_payload()
    payload["observations"][0]["duplicate"] = True
    payload["observations"][1]["duplicate"] = False
    payload["observations"][0]["out_of_order"] = True
    summary = summarize_vision_trace(validate_vision_trace(payload))["observations"]
    assert summary["duplicates"] == 1
    assert summary["duplicate_flag_unknown"] == 2
    assert summary["duplicate_metadata_available"] is True
    assert summary["out_of_order"] == 1
    assert summary["out_of_order_flag_unknown"] == 3


def test_vision_cli_writes_nested_output_and_reports_input_errors(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(vision_payload()), encoding="utf-8")
    output_path = tmp_path / "nested" / "summary.json"
    assert analyze_vision_trace([str(trace_path), "--output", str(output_path)]) == 0
    summary = json.loads(output_path.read_text())
    assert summary["schema"] == "zephyr-vision-trace-summary-1"
    assert len(summary["provenance"]["input_sha256"]) == 64
    with pytest.raises(SystemExit) as same_path:
        analyze_vision_trace([str(trace_path), "--output", str(trace_path)])
    assert same_path.value.code == 2
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as bad_input:
        analyze_vision_trace([str(malformed)])
    assert bad_input.value.code == 2
