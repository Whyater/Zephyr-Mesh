import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sim.link import replay_trace
from sim.trace import EspNowTrace, TRACE_SCHEMA, iter_received, validate_trace


def trace_payload():
    return {
        "schema": TRACE_SCHEMA,
        "trace_id": "lab-2026-10-09-001",
        "status": "measured",
        "clock_domain": "shared_monotonic",
        "capture": {
            "transport": "ESP-NOW",
            "device_ids": ["a", "b"],
            "firmware": "esp-idf-5.3.2",
            "channel": 6,
            "phy": "802.11n",
        },
        "source": "esp-now-capture.json",
        "clock_uncertainty_s": 0.0001,
        "packets": [
            {"packet_id": "boot-a-0", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 0, "send_time_s": 1.0, "receive_time_s": 1.1, "outcome": "received", "rssi_dbm": -54.0, "callback_time_s": 1.101, "payload_bytes": 32, "retry_count": 0},
            {"packet_id": "boot-a-2", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 2, "send_time_s": 1.2, "receive_time_s": 1.3, "outcome": "received"},
            {"packet_id": "boot-a-1", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 1, "send_time_s": 1.1, "receive_time_s": 1.4, "outcome": "received"},
            {"packet_id": "boot-a-2b", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 2, "send_time_s": 1.25, "receive_time_s": 1.5, "outcome": "received", "duplicate": True},
            {"packet_id": "boot-a-3", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 3, "send_time_s": 1.3, "outcome": "lost", "loss_reason": "no_ack"},
            {"packet_id": "boot-a-4", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a", "seq": 4, "send_time_s": 1.4, "outcome": "unknown", "unknown_reason": "capture_ended"},
        ],
    }


def test_trace_round_trip_keeps_measurement_metadata(tmp_path):
    trace = validate_trace(trace_payload())
    path = tmp_path / "trace.json"
    trace.write(path)
    decoded = EspNowTrace.from_path(path)
    assert decoded.trace_id == "lab-2026-10-09-001"
    assert decoded.capture["channel"] == 6
    assert decoded.packets[0].rssi_dbm == pytest.approx(-54.0)
    assert json.loads(path.read_text())["schema"] == TRACE_SCHEMA


def test_json_schema_and_python_validator_accept_the_same_valid_fixture():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "trace_schema.json").read_text())
    Draft202012Validator(schema).validate(trace_payload())
    validate_trace(trace_payload())
    example = json.loads((Path(__file__).parents[1] / "sim" / "trace_example.json").read_text())
    Draft202012Validator(schema).validate(example)
    assert validate_trace(example).status == "fixture"

    invalid = trace_payload()
    invalid["packets"][4].pop("loss_reason")
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid)
    with pytest.raises(ValueError, match="loss_reason"):
        validate_trace(invalid)


def test_trace_rejects_ambiguous_or_unidentified_observations():
    payload = trace_payload()
    payload["packets"][0]["outcome"] = "lost"
    with pytest.raises(ValueError, match="cannot have receive_time"):
        validate_trace(payload)

    payload = trace_payload()
    payload["packets"][1]["packet_id"] = payload["packets"][0]["packet_id"]
    with pytest.raises(ValueError, match="packet_id"):
        validate_trace(payload)

    payload = trace_payload()
    payload["packets"][0]["rssi_dbm"] = True
    with pytest.raises(ValueError, match="rssi_dbm"):
        validate_trace(payload)


def test_replay_trace_preserves_observed_outcomes_and_derives_ordering():
    events = replay_trace(validate_trace(trace_payload()))
    assert len(events) == 6
    assert events[0].packet_age_s == pytest.approx(0.1)
    assert events[1].out_of_order is False
    assert events[2].out_of_order is True
    assert events[3].duplicate is True
    assert events[4].receive_time is None
    assert events[4].loss_reason == "no_ack"
    assert events[5].outcome == "unknown"
    assert events[5].unknown_reason == "capture_ended"
    assert events[0].callback_time == pytest.approx(1.101)
    assert events[0].rssi_dbm == pytest.approx(-54.0)
    assert events[0].payload_bytes == 32


def test_separate_clocks_do_not_invent_packet_age():
    payload = trace_payload()
    payload["clock_domain"] = "device_and_host"
    trace = validate_trace(payload)
    assert replay_trace(trace)[0].packet_age_s is None


def test_trace_requires_session_namespace_and_validates_shared_clock_order():
    payload = trace_payload()
    del payload["packets"][0]["sender_session_id"]
    with pytest.raises(ValueError, match="sender_session_id"):
        validate_trace(payload)

    payload = trace_payload()
    payload["packets"][0]["receive_time_s"] = 0.5
    with pytest.raises(ValueError, match="precede"):
        validate_trace(payload)

    payload = trace_payload()
    payload["packets"][0]["rssi_dbm"] = "-54"
    with pytest.raises(ValueError, match="rssi_dbm"):
        validate_trace(payload)

    payload = trace_payload()
    payload["packets"][0]["callback_time_s"] = 0.9
    with pytest.raises(ValueError, match="callback_time_s"):
        validate_trace(payload)


def test_received_iteration_prefers_callback_time():
    payload = trace_payload()
    payload["packets"][0]["callback_time_s"] = 1.45
    payload["packets"][1]["callback_time_s"] = 1.35
    payload["packets"][1]["receive_time_s"] = 1.3
    payload["packets"][2]["callback_time_s"] = 1.4
    ordered = list(iter_received(validate_trace(payload)))
    assert [packet.packet_id for packet in ordered[:3]] == ["boot-a-2", "boot-a-1", "boot-a-0"]


# ---------------------------------------------------------------------------
# unknown fields are rejected (same strictness as the vision trace)
# ---------------------------------------------------------------------------

def test_trace_rejects_unknown_packet_root_and_capture_fields():
    bad_packet = trace_payload()
    bad_packet["packets"][0]["bogus"] = 1
    with pytest.raises(ValueError, match="bogus"):
        validate_trace(bad_packet)
    bad_root = trace_payload()
    bad_root["bogus_top"] = 1
    with pytest.raises(ValueError, match="bogus_top"):
        validate_trace(bad_root)
    bad_capture = trace_payload()
    bad_capture["capture"]["bogus_cap"] = 1
    with pytest.raises(ValueError, match="bogus_cap"):
        validate_trace(bad_capture)


def test_python_allowed_field_sets_equal_the_json_schema_property_lists():
    import sim.trace as trace_module

    schema = json.loads((Path(__file__).parents[1] / "sim" / "trace_schema.json").read_text())
    assert trace_module._TRACE_FIELDS == set(schema["properties"])
    assert trace_module._CAPTURE_FIELDS == set(schema["properties"]["capture"]["properties"])
    assert trace_module._PACKET_FIELDS == set(schema["properties"]["packets"]["items"]["properties"])
    # Both validators also reject the same unknown packet field.
    bad = trace_payload()
    bad["packets"][1]["bogus"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)
    with pytest.raises(ValueError):
        validate_trace(bad)


# ---------------------------------------------------------------------------
# wrapped sequence numbers in a captured trace
# ---------------------------------------------------------------------------

def _wrapped_trace(extra=None):
    payload = trace_payload()
    payload["packets"] = [
        {"packet_id": f"p{i}", "sender_id": "a", "receiver_id": "b", "sender_session_id": "boot-a",
         "seq": i % 256, "send_time_s": i * 0.01, "receive_time_s": i * 0.01 + 0.005,
         "outcome": "received", **(extra or {})}
        for i in range(600)
    ]
    return validate_trace(payload)


def test_wrapped_sequence_is_misread_by_replay_unless_capture_supplies_flags():
    # Honest current behavior: replay derives duplicate flags from (route, seq).
    # A uint8 counter repeats identities, so 600 - 256 = 344 packets are
    # flagged.  A capture tool that unwraps the counter itself can pass the
    # explicit flags, which replay treats as authoritative.
    derived = replay_trace(_wrapped_trace())
    assert sum(e.duplicate for e in derived) == 600 - 256
    explicit = replay_trace(_wrapped_trace({"duplicate": False, "out_of_order": False}))
    assert sum(e.duplicate or e.out_of_order for e in explicit) == 0
