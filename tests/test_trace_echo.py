import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from sim.link import replay_trace
from sim.trace import TRACE_SCHEMA, EspNowPacket, EspNowTrace


def echo_payload():
    return {
        "schema": TRACE_SCHEMA,
        "trace_id": "echo-1",
        "status": "synthetic",
        "clock_domain": "sender_round_trip",
        "capture": {"transport": "ESP-NOW", "device_ids": ["A", "B"], "firmware": "fw", "channel": 1, "phy": "unrecorded"},
        "packets": [{
            "packet_id": "s:0", "sender_id": "A", "receiver_id": "B", "sender_session_id": "s",
            "seq": 0, "send_time_s": 1.0, "outcome": "received", "echo_time_s": 1.004,
            "echo_turnaround_s": 0.001,
        }],
    }


def validators():
    schema = json.loads((Path(__file__).parents[1] / "sim" / "trace_schema.json").read_text())
    return Draft202012Validator(schema)


def test_echo_only_received_packet_passes_both_validators():
    payload = echo_payload()
    validators().validate(payload)
    trace = EspNowTrace.from_dict(payload)
    assert trace.packets[0].round_trip_s() == pytest.approx(0.003)


def test_echo_fields_in_shared_monotonic_trace_rejected_by_both():
    payload = echo_payload()
    payload["clock_domain"] = "shared_monotonic"
    with pytest.raises(ValueError, match="only allowed"):
        EspNowTrace.from_dict(payload)
    with pytest.raises(ValidationError):
        validators().validate(payload)


def test_echo_on_lost_packet_rejected_by_both():
    payload = echo_payload()
    payload["packets"][0]["outcome"] = "lost"
    payload["packets"][0]["loss_reason"] = "not_in_echoer_log"
    with pytest.raises(ValueError, match="must be received"):
        EspNowTrace.from_dict(payload)
    with pytest.raises(ValidationError):
        validators().validate(payload)


def test_turnaround_without_echo_rejected_by_both():
    payload = echo_payload()
    payload["packets"][0].pop("echo_time_s")
    with pytest.raises(ValueError, match="requires echo_time_s"):
        EspNowTrace.from_dict(payload)
    with pytest.raises(ValidationError):
        validators().validate(payload)


def test_received_without_receive_or_echo_rejected_by_both():
    payload = echo_payload()
    payload["packets"][0].pop("echo_time_s")
    payload["packets"][0].pop("echo_turnaround_s")
    with pytest.raises(ValueError, match="require receive_time_s"):
        EspNowTrace.from_dict(payload)
    with pytest.raises(ValidationError):
        validators().validate(payload)


def test_echo_ordering_checks_are_python_only():
    payload = echo_payload()
    payload["packets"][0]["echo_time_s"] = 0.9
    with pytest.raises(ValueError, match="cannot precede"):
        EspNowTrace.from_dict(payload)
    payload = echo_payload()
    payload["packets"][0]["echo_turnaround_s"] = 0.01
    with pytest.raises(ValueError, match="cannot exceed"):
        EspNowTrace.from_dict(payload)


def test_round_trip_is_sender_delta_minus_receiver_hold():
    packet = EspNowPacket.from_dict({
        "packet_id": "p", "sender_id": "A", "receiver_id": "B", "sender_session_id": "s",
        "seq": 4, "send_time_s": 10.0, "outcome": "received", "echo_time_s": 10.0075,
        "echo_turnaround_s": 0.0025,
    }, "sender_round_trip")
    assert packet.round_trip_s() == pytest.approx(0.005)
    assert EspNowPacket.from_dict({
        "packet_id": "p2", "sender_id": "A", "receiver_id": "B", "sender_session_id": "s",
        "seq": 5, "send_time_s": 10.0, "outcome": "received", "echo_time_s": 10.0075,
    }, "sender_round_trip").round_trip_s() == pytest.approx(0.0075)


def test_non_echo_as_dict_omits_echo_keys_and_round_trips():
    from tests.test_trace import trace_payload
    payload = trace_payload()
    trace = EspNowTrace.from_dict(payload)
    encoded = trace.as_dict()
    assert all("echo_time_s" not in packet and "echo_turnaround_s" not in packet for packet in encoded["packets"])
    assert EspNowTrace.from_dict(encoded).as_dict() == encoded


def test_replay_trace_refuses_echo_trace():
    with pytest.raises(ValueError, match="sender_round_trip"):
        replay_trace(EspNowTrace.from_dict(echo_payload()))
