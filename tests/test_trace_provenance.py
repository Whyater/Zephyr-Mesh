import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from sim.trace import EspNowTrace, TRACE_SCHEMA, validate_trace

ROOT = Path(__file__).parents[1]
SHA = "a" * 64


def provenance(*, calibration_ref=SHA, ground_truth=None):
    value = {
        "instrument": {"kind": "esp32-sniffer", "model": "model-1", "settings": {"channel": 6}},
        "calibration_ref": calibration_ref,
        "raw_sha256": SHA,
        "code_revision": "abc123-dirty",
        "captured_at": "2026-10-14T09:30:00-06:00",
        "operator": "operator one",
        "ground_truth": ground_truth if ground_truth is not None else {
            "method": "tape measure",
            "values": {"distance_m": 3.0},
            "uncertainty": {"distance_m": 0.05},
        },
    }
    if calibration_ref is None:
        value["calibration_ref_null_reason"] = "calibration unavailable"
    return value


def payload(status="measured", with_provenance=True):
    data = {
        "schema": TRACE_SCHEMA,
        "trace_id": "lab-1",
        "status": status,
        "clock_domain": "shared_monotonic",
        "capture": {"transport": "ESP-NOW", "device_ids": ["a", "b"], "firmware": "fw-1", "channel": 6, "phy": "802.11n"},
        "source": "capture.json",
        "packets": [
            {"packet_id": "p0", "sender_id": "a", "receiver_id": "b", "sender_session_id": "s", "seq": 0,
             "send_time_s": 1.0, "receive_time_s": 1.1, "outcome": "received"},
        ],
    }
    if with_provenance:
        data["provenance"] = provenance()
    return data


@pytest.fixture(scope="module")
def schema():
    return Draft202012Validator(json.loads((ROOT / "sim" / "trace_schema.json").read_text()), format_checker=FormatChecker())


def assert_rejected(schema, data):
    with pytest.raises(ValueError):
        EspNowTrace.from_dict(data)
    assert not schema.is_valid(data)


def test_measured_without_provenance_is_rejected(schema):
    assert_rejected(schema, payload(with_provenance=False))


def test_measured_with_full_block_is_accepted(schema):
    data = payload()
    assert schema.is_valid(data)
    trace = validate_trace(data)
    assert trace.as_dict()["provenance"] == data["provenance"]


@pytest.mark.parametrize("status", ["synthetic", "fixture"])
def test_non_measured_traces_do_not_require_provenance(schema, status):
    data = payload(status=status, with_provenance=False)
    assert schema.is_valid(data)
    trace = validate_trace(data)
    assert trace.provenance is None
    assert "provenance" not in trace.as_dict()


@pytest.mark.parametrize("status", ["synthetic", "fixture"])
def test_non_measured_provenance_is_validated_when_supplied(schema, status):
    data = payload(status=status)
    assert schema.is_valid(data)
    validate_trace(data)
    data["provenance"]["raw_sha256"] = "bad"
    assert_rejected(schema, data)


def test_example_still_validates(schema):
    example = json.loads((ROOT / "sim" / "trace_example.json").read_text())
    assert schema.is_valid(example)
    assert validate_trace(example).provenance is None


@pytest.mark.parametrize("key", ["instrument", "calibration_ref", "raw_sha256", "code_revision", "captured_at", "operator", "ground_truth"])
def test_missing_required_key_is_rejected(schema, key):
    data = payload()
    del data["provenance"][key]
    assert_rejected(schema, data)


def test_missing_key_with_reason_present_is_still_rejected(schema):
    data = payload()
    del data["provenance"]["calibration_ref"]
    data["provenance"]["calibration_ref_null_reason"] = "unavailable"
    assert_rejected(schema, data)
    data = payload()
    del data["provenance"]["ground_truth"]
    data["provenance"]["ground_truth_null_reason"] = "unavailable"
    assert_rejected(schema, data)


@pytest.mark.parametrize("field", ["raw_sha256", "calibration_ref"])
def test_bad_hex_is_rejected(schema, field):
    for bad in ("bad", "g" * 64, "a" * 63):
        data = payload()
        data["provenance"][field] = bad
        assert_rejected(schema, data)


@pytest.mark.parametrize("stamp", ["2026-10-14T09:30:00", "2026-10-14 09:30:00Z", "not a time", "2026-10-14"])
def test_naive_or_malformed_timestamp_is_rejected(schema, stamp):
    data = payload()
    data["provenance"]["captured_at"] = stamp
    assert_rejected(schema, data)


@pytest.mark.parametrize("stamp", ["2026-10-14T09:30:00Z", "2026-10-14T09:30:00+05:30"])
def test_offset_timestamps_are_accepted(schema, stamp):
    data = payload()
    data["provenance"]["captured_at"] = stamp
    assert schema.is_valid(data)
    validate_trace(data)


def test_null_calibration_requires_reason(schema):
    data = payload()
    data["provenance"]["calibration_ref"] = None
    assert_rejected(schema, data)
    data["provenance"]["calibration_ref_null_reason"] = ""
    assert_rejected(schema, data)
    data["provenance"]["calibration_ref_null_reason"] = "not available"
    assert schema.is_valid(data)
    assert validate_trace(data).provenance["calibration_ref"] is None


def test_reason_with_non_null_calibration_is_rejected(schema):
    data = payload()
    data["provenance"]["calibration_ref_null_reason"] = "should not be here"
    assert_rejected(schema, data)


def test_null_ground_truth_requires_reason_and_reason_requires_null(schema):
    data = payload()
    data["provenance"]["ground_truth"] = None
    assert_rejected(schema, data)
    data["provenance"]["ground_truth_null_reason"] = "no reference available"
    assert schema.is_valid(data)
    validate_trace(data)
    data = payload()
    data["provenance"]["ground_truth_null_reason"] = "should not be here"
    assert_rejected(schema, data)


@pytest.mark.parametrize("location", ["provenance", "instrument", "ground_truth"])
def test_unknown_key_is_rejected(schema, location):
    data = payload()
    target = data["provenance"] if location == "provenance" else data["provenance"][location]
    target["unexpected"] = 1
    assert_rejected(schema, data)


@pytest.mark.parametrize("mutate", [
    lambda p: p["instrument"].pop("settings"),
    lambda p: p["instrument"].__setitem__("settings", [1]),
    lambda p: p["instrument"].__setitem__("kind", ""),
    lambda p: p["ground_truth"].pop("uncertainty"),
    lambda p: p["ground_truth"].__setitem__("values", "x"),
    lambda p: p.__setitem__("code_revision", ""),
    lambda p: p.__setitem__("operator", ""),
])
def test_other_malformed_fields_are_rejected(schema, mutate):
    data = payload()
    mutate(data["provenance"])
    assert_rejected(schema, data)


def test_provenance_must_be_an_object(schema):
    for bad in (None, [], "x"):
        data = payload()
        data["provenance"] = bad
        assert_rejected(schema, data)


def test_as_dict_round_trips_provenance(schema, tmp_path):
    data = payload()
    data["provenance"] = provenance(calibration_ref=None, ground_truth=None)
    data["provenance"]["ground_truth"] = None
    data["provenance"]["ground_truth_null_reason"] = "none available"
    trace = validate_trace(data)
    out = trace.as_dict()
    assert out["provenance"] == data["provenance"]
    assert validate_trace(out).as_dict() == out
    path = tmp_path / "t.json"
    trace.write(path)
    assert EspNowTrace.from_path(path).provenance == data["provenance"]
    assert schema.is_valid(json.loads(path.read_text()))


def test_python_and_schema_agree(schema):
    base = payload()
    cases = [base]
    for edit in (
        lambda d: d["provenance"].pop("operator"),
        lambda d: d["provenance"].update(extra=True),
        lambda d: d["provenance"].update(raw_sha256="bad"),
        lambda d: d["provenance"].update(captured_at="2026-10-14T09:30:00"),
        lambda d: d.pop("provenance"),
    ):
        case = copy.deepcopy(base)
        edit(case)
        cases.append(case)
    for case in cases:
        try:
            EspNowTrace.from_dict(case)
            python_valid = True
        except ValueError:
            python_valid = False
        assert python_valid == schema.is_valid(case)
