import json
import math
from pathlib import Path

import pytest

from sim.trace import EspNowTrace
from sim.trace_metrics import summarize_trace
from tools.analyze_trace import main


def example_trace():
    return EspNowTrace.from_path(Path(__file__).parents[1] / "sim" / "trace_example.json")


def test_trace_summary_reports_observed_records_and_delay():
    summary = summarize_trace(example_trace())
    assert summary["schema"] == "zephyr-espnow-trace-summary-1"
    assert summary["packets"]["observed_transmission_records"] == 2
    assert summary["packets"]["received"] == 1
    assert summary["packets"]["lost"] == 1
    assert summary["packets"]["unknown_unresolved"] == 0
    assert summary["packets"]["loss_rate"] == 0.5
    assert summary["delay"]["mean"] == 0.012
    assert summary["rssi"]["mean"] == -55.0


def test_trace_summary_with_unknown_outcome_reports_bounds_and_no_rate():
    payload = example_trace().as_dict()
    payload["packets"].append({
        "packet_id": "example-a-2",
        "sender_id": "drone-a",
        "receiver_id": "ground-b",
        "sender_session_id": "example-session-a",
        "seq": 2,
        "send_time_s": 0.04,
        "outcome": "unknown",
        "unknown_reason": "capture_ended",
    })
    summary = summarize_trace(EspNowTrace.from_dict(payload))
    assert summary["packets"]["loss_rate"] is None
    assert summary["packets"]["loss_rate_bounds"]["lower"] == 1 / 3
    assert summary["packets"]["loss_rate_bounds"]["upper"] == 2 / 3
    assert summary["packets"]["unknown_reason_counts"] == {"capture_ended": 1}


def test_trace_summary_does_not_invent_age_for_separate_clocks(tmp_path):
    payload = example_trace().as_dict()
    payload["clock_domain"] = "device_and_host"
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_trace(EspNowTrace.from_path(path))
    assert summary["delay"]["available"] is False
    assert "clock_domain" in summary["delay"]["reason"]


def test_trace_cli_writes_summary(tmp_path):
    output = tmp_path / "nested" / "summary.json"
    assert main([str(Path(__file__).parents[1] / "sim" / "trace_example.json"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["schema"] == "zephyr-espnow-trace-summary-1"


def test_duplicate_delivery_does_not_invent_a_new_logical_packet():
    payload = example_trace().as_dict()
    duplicate = dict(payload["packets"][0], packet_id="duplicate-reception", receive_time_s=0.015, callback_time_s=0.016)
    reboot = dict(duplicate, packet_id="new-boot", sender_session_id="boot-2", send_time_s=0.03, receive_time_s=0.04, callback_time_s=0.041)
    payload["packets"].extend([duplicate, reboot])
    packets = summarize_trace(EspNowTrace.from_dict(payload))["packets"]
    assert packets["observed_transmission_records"] == 4
    assert packets["logical_identity_count"] == 3
    assert packets["duplicates"] == 1
    assert packets["loss_rate"] == 0.25


def test_quantiles_match_independent_hand_calculation():
    payload = example_trace().as_dict()
    first = payload["packets"][0]
    payload["packets"] = [dict(first, packet_id=f"sample-{i}", seq=i, send_time_s=0.0, receive_time_s=delay, callback_time_s=delay) for i, delay in enumerate([0.01, 0.03, 0.07])]
    delay = summarize_trace(EspNowTrace.from_dict(payload))["delay"]
    assert delay["mean"] == pytest.approx(0.11 / 3)
    assert delay["p50"] == 0.03
    assert delay["p95"] == pytest.approx(0.066)


def test_empty_summary_has_unknown_rates_and_no_samples():
    payload = example_trace().as_dict()
    payload["packets"] = []
    summary = summarize_trace(EspNowTrace.from_dict(payload))
    assert summary["packets"]["loss_rate"] is None
    assert summary["packets"]["loss_rate_bounds"]["lower"] is None
    assert summary["delay"]["available"] is False
    assert summary["rssi"]["count"] == 0


def test_extreme_finite_samples_produce_standard_finite_json():
    payload = example_trace().as_dict()
    payload["packets"][0]["rssi_dbm"] = -1e308
    payload["packets"][1]["rssi_dbm"] = 1e308
    summary = summarize_trace(EspNowTrace.from_dict(payload))
    assert summary["rssi"]["mean"] == 0.0
    assert summary["rssi"]["p50"] == 0.0
    assert math.isfinite(summary["rssi"]["p95"])
    json.dumps(summary, allow_nan=False)


def test_summary_preserves_code_revision_and_unknown_reason():
    payload = example_trace().as_dict()
    payload["code_revision"] = "abc123"
    payload["notes"] = "props removed; USB power"
    payload["packets"][1].update(outcome="unknown", loss_reason=None, unknown_reason="logger_failure")
    summary = summarize_trace(EspNowTrace.from_dict(payload))
    assert summary["provenance"]["code_revision"] == "abc123"
    assert summary["provenance"]["notes"] == "props removed; USB power"
    assert summary["packets"]["unknown_reason_counts"] == {"logger_failure": 1}
    assert "not inferred" in summary["packets"]["censoring"]


def test_cli_input_and_output_errors_are_concise(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    with pytest.raises(SystemExit) as error:
        main([str(invalid)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main([str(Path(__file__).parents[1] / "sim" / "trace_example.json"), "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
