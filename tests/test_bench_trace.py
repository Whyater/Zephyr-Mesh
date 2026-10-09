import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.bench_metrics import summarize_bench_trace
from sim.bench_trace import BENCH_TRACE_SCHEMA, BenchTrace, validate_bench_trace
from tools.analyze_bench_trace import main as analyze_bench_trace


def bench_payload():
    return {
        "schema": BENCH_TRACE_SCHEMA,
        "trace_id": "bench-lab-2026-10-09-001",
        "status": "measured",
        "profile_id": "quad-profile-1",
        "motor_part_id": "motor-a",
        "propeller_part_id": "prop-a",
        "measurement_scope": "single_rotor",
        "clock_domain": "shared_monotonic",
        "capture": {
            "device_ids": ["bench-host"],
            "instrument_ids": ["tach-a", "load-a", "power-a"],
            "air_density_kg_m3": 1.21,
            "ambient_temperature_c": 23.0,
        },
        "source": "bench-run.jsonl",
        "calibration_id": "cal-001",
        "samples": [
            {"sample_id": "s1", "timestamp_s": 1.0, "quality": "valid", "rpm": 10000, "thrust_n": 0.3, "voltage_v": 3.7, "current_a": 1.0, "duration_s": 2.0, "temperature_c": 22.0},
            {"sample_id": "s2", "timestamp_s": 4.0, "quality": "valid", "rpm": 20000, "thrust_n": 1.2, "voltage_v": 3.6, "current_a": 2.0, "duration_s": 2.0, "temperature_c": 24.0},
            {"sample_id": "s3", "timestamp_s": 7.0, "quality": "invalid", "quality_reason": "load_cell_saturated", "rpm": 25000},
            {"sample_id": "s4", "timestamp_s": 8.0, "quality": "unknown", "quality_reason": "recording_ended"},
        ],
    }


def test_schema_and_python_validator_accept_measured_fixture():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "bench_trace_schema.json").read_text())
    Draft202012Validator(schema).validate(bench_payload())
    trace = validate_bench_trace(bench_payload())
    assert trace.samples[0].rpm == 10000
    example = json.loads((Path(__file__).parents[1] / "sim" / "bench_trace_example.json").read_text())
    Draft202012Validator(schema).validate(example)
    assert validate_bench_trace(example).status == "fixture"


def test_round_trip_and_summary_keep_unknowns_and_power_transparent(tmp_path):
    trace = validate_bench_trace(bench_payload())
    path = tmp_path / "bench.json"
    trace.write(path)
    decoded = BenchTrace.from_path(path)
    assert decoded.as_dict() == trace.as_dict()
    summary = summarize_bench_trace(decoded)
    assert summary["samples"]["valid"] == 2
    assert summary["samples"]["invalid"] == 1
    assert summary["samples"]["unknown_unresolved"] == 1
    assert summary["measurements"]["electrical_input_power"]["mean"] == pytest.approx(5.45)
    assert summary["method"]["fit_status"].startswith("not fitted")


def test_quality_and_clock_rules_reject_ambiguous_records():
    payload = bench_payload()
    payload["samples"][0].pop("current_a")
    with pytest.raises(ValueError, match="valid samples require"):
        validate_bench_trace(payload)
    payload = bench_payload()
    payload["samples"][2].pop("quality_reason")
    with pytest.raises(ValueError, match="invalid samples require"):
        validate_bench_trace(payload)
    payload = bench_payload()
    payload["samples"][1]["timestamp_s"] = 1.0
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_bench_trace(payload)
    payload = bench_payload()
    payload["samples"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        validate_bench_trace(payload)
    payload = bench_payload()
    payload["capture"]["test_notes"] = None
    assert not Draft202012Validator(json.loads((Path(__file__).parents[1] / "sim" / "bench_trace_schema.json").read_text())).is_valid(payload)
    with pytest.raises(ValueError, match="capture.test_notes"):
        validate_bench_trace(payload)


def test_measured_trace_requires_provenance_and_schema_matches():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "bench_trace_schema.json").read_text())
    payload = bench_payload()
    del payload["calibration_id"]
    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValueError, match="calibration_id"):
        validate_bench_trace(payload)
    payload = bench_payload()
    payload["samples"][0]["quality"] = "invalid"
    payload["samples"][0]["quality_reason"] = "bad"
    assert Draft202012Validator(schema).is_valid(payload)
    payload["samples"][0]["quality_reason"] = None
    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValueError, match="invalid samples require"):
        validate_bench_trace({**payload, "samples": [{"sample_id": "s", "timestamp_s": 0, "quality": "invalid"}]})


def test_bench_cli_writes_nested_output_and_reports_errors(tmp_path):
    source = tmp_path / "bench.json"
    source.write_text(json.dumps(bench_payload()), encoding="utf-8")
    output = tmp_path / "nested" / "summary.json"
    assert analyze_bench_trace([str(source), "--output", str(output)]) == 0
    summary = json.loads(output.read_text())
    assert summary["schema"] == "zephyr-bench-trace-summary-1"
    assert len(summary["provenance"]["input_sha256"]) == 64
    with pytest.raises(SystemExit) as same_path:
        analyze_bench_trace([str(source), "--output", str(source)])
    assert same_path.value.code == 2
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as bad:
        analyze_bench_trace([str(malformed)])
    assert bad.value.code == 2
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema": BENCH_TRACE_SCHEMA}), encoding="utf-8")
    with pytest.raises(SystemExit) as invalid_exit:
        analyze_bench_trace([str(invalid)])
    assert invalid_exit.value.code == 2


def test_summary_keeps_declared_scope_and_excludes_overflowed_power():
    payload = bench_payload()
    payload["measurement_scope"] = "airframe"
    payload["samples"][0]["voltage_v"] = 1e308
    payload["samples"][0]["current_a"] = 1e308
    summary = summarize_bench_trace(validate_bench_trace(payload))
    assert "measurement_scope=airframe" in summary["method"]["weighting"]
    assert summary["samples"]["electrical_power_nonfinite_excluded"] == 1
    assert summary["measurements"]["electrical_input_power"]["count"] == 1
