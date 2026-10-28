"""Tests for tools/latency_budget.py.

The truth every lookup test recovers is the registered S6 summary
(sim/runs/s6-recovery/summary.json, read-only): for the main slice at 5 cm
noise, no loss and no jitter, naive fails the straight line from 200 ms and
the 1 m turn from 200 ms (43 and 50 of 50 seeds), recovery fails only the turn
at 300 ms (50 of 50), naive loses 4 of 50 on the turn at 150 ms, and recovery
loses 2 of 50 on the turn at 200 ms.
"""
import hashlib
import json

import pytest

from tools import latency_budget as lb
from tools.fit_espnow_log import fit_echo_trace
from tools.import_espnow_log import read_echo_logs
from tools.make_espnow_echo_fixture import EchoParams, write_fixture
from tools.vision_latency import main as vision_latency_main

GRID = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3)


def _latency(high_s, status="synthetic"):
    """A minimal latency.json with the camera upper bound set to ``high_s``."""
    return {
        "schema": "zephyr-vision-latency-1",
        "status": status,
        "apparent_latency_s": {"mean": high_s - 0.001, "stated_error": 0.001,
                               "jitter_std_from_straddles": 0.004},
        "camera_latency_s": {"low": high_s - 0.061, "high": high_s},
        "settings": {"display_lag_s": [0.0, 0.06]},
    }


def _processing(median_s, status=None):
    data = {"schema": "zephyr-processing-time-1", "median_s": median_s, "p95_s": median_s * 2, "n": 200}
    if status is not None:
        data["status"] = status
    return data


def _fit(delay_s, ci_high_s, status):
    """The parts of a zephyr-espnow-fit-1 the tool reads."""
    return {
        "schema": "zephyr-espnow-fit-1",
        "source_trace": {"trace_id": "t1", "status": status, "n_packets": 100},
        "delay": {"one_way_estimate": {"delay_s": {"value": delay_s, "ci": [delay_s / 2, ci_high_s]}}},
    }


def _cells(result):
    return {row["motion"]: (row["naive"]["label"], row["naive"]["fail_count"],
                            row["recovery"]["label"], row["recovery"]["fail_count"])
            for row in result["rows"]}


def test_grid_is_read_from_the_registered_summary():
    summary = lb.load_summary(lb.DEFAULT_SUMMARY)
    assert lb.delay_grid(summary, 0.05) == GRID


def test_hand_case_exactly_on_a_grid_point():
    # 50 + 10 + 40 = 100 ms, the 100 ms grid point itself (not rounded up to 150)
    result = lb.budget(_latency(0.050), _processing(0.010))
    assert result["total_delay_s"] == pytest.approx(0.100)
    assert result["grid_point_s"] == 0.1
    assert result["on_grid"] is True
    assert _cells(result) == {"constant_velocity": ("passes", 0, "passes", 0),
                              "constant_turn": ("passes", 0, "passes", 0)}


def test_hand_case_just_above_a_grid_point_rounds_up():
    # 50.5 + 10 + 40 = 100.5 ms, so the next grid point up, 150 ms
    result = lb.budget(_latency(0.0505), _processing(0.010))
    assert result["total_delay_s"] == pytest.approx(0.1005)
    assert result["grid_point_s"] == 0.15
    assert _cells(result) == {"constant_velocity": ("passes", 0, "passes", 0),
                              "constant_turn": ("passes", 4, "passes", 0)}


def test_hand_case_200_ms_row():
    # 140 + 15 + 40 = 195 ms, rounds up to 200 ms
    result = lb.budget(_latency(0.140), _processing(0.015))
    assert result["grid_point_s"] == 0.2
    assert _cells(result) == {"constant_velocity": ("fails", 43, "passes", 0),
                              "constant_turn": ("fails", 50, "passes", 2)}


def test_last_grid_point_is_still_on_the_grid():
    # 250 + 10 + 40 = 300 ms, the largest grid point
    result = lb.budget(_latency(0.250), _processing(0.010))
    assert result["grid_point_s"] == 0.3
    assert _cells(result) == {"constant_velocity": ("fails", 50, "passes", 0),
                              "constant_turn": ("fails", 50, "fails", 50)}


def test_above_the_largest_grid_point_is_off_the_grid_not_the_last_row():
    # 300 + 10 + 40 = 350 ms, above 300 ms
    result = lb.budget(_latency(0.300), _processing(0.010))
    assert result["grid_point_s"] is None
    assert result["on_grid"] is False
    assert result["rows"] == []
    assert "off the grid" in result["verdict"]
    assert "Row read" not in lb.format_report(result)


def test_float_rounding_does_not_push_a_grid_point_up():
    assert lb.round_up_to_grid(0.1 + 1e-12, GRID) == 0.1
    assert lb.round_up_to_grid(0.1 + 1e-6, GRID) == 0.15
    assert lb.round_up_to_grid(0.0, GRID) == 0.0
    assert lb.round_up_to_grid(0.3000001, GRID) is None
    total = 0.1 + 0.2  # a float sum that lands just above 0.3
    assert total > 0.3
    assert lb.round_up_to_grid(total, GRID) == 0.3


def test_default_radio_term_is_labelled_assumed_simulated():
    result = lb.budget(_latency(0.050), _processing(0.010, status="measured"))
    radio = result["components"]["radio"]
    assert radio["value_s"] == 0.040
    assert radio["label"] == "assumed, simulated"
    assert radio["status"] == "assumed (simulated)"


def test_radio_label_and_value_change_when_a_fit_is_given():
    # radio term is the upper end of the fit's one-way mean delay interval
    result = lb.budget(_latency(0.050), _processing(0.010), fit=_fit(0.003, 0.004, "synthetic"))
    radio = result["components"]["radio"]
    assert radio["value_s"] == pytest.approx(0.004)
    assert radio["label"] == "from a synthetic ESP-NOW fit"
    assert lb.RADIO_FIT_NOTE in result["limits"]
    assert result["total_delay_s"] == pytest.approx(0.064)
    assert result["grid_point_s"] == 0.1
    measured = lb.budget(_latency(0.050), _processing(0.010), fit=_fit(0.003, 0.004, "measured"))
    assert measured["components"]["radio"]["label"] == "from a measured ESP-NOW fit"


def test_overall_status_follows_the_weakest_input():
    measured_parts = {"latency": _latency(0.05, "measured"), "processing": _processing(0.01, "measured")}
    assert lb.budget(**measured_parts)["status"] == "partly measured"  # radio assumed
    assert lb.budget(**measured_parts, fit=_fit(0.003, 0.004, "measured"))["status"] == "measured"
    assert lb.budget(**measured_parts, fit=_fit(0.003, 0.004, "fixture"))["status"] == "synthetic"
    assert lb.budget(_latency(0.05, "synthetic"), _processing(0.01, "measured"),
                     fit=_fit(0.003, 0.004, "measured"))["status"] == "synthetic"
    # a processing file without a status field is unknown, so never fully measured
    assert lb.budget(_latency(0.05, "measured"), _processing(0.01),
                     fit=_fit(0.003, 0.004, "measured"))["status"] == "partly measured"


def test_ten_cm_noise_row_uses_the_10_cm_cells():
    # registered 10 cm row at 150 ms: recovery fails the straight line, naive fails both
    result = lb.budget(_latency(0.100), _processing(0.010), noise_std_m=0.10)
    assert result["grid_point_s"] == 0.15
    assert result["noise_std_m"] == 0.10
    # truth from the registered summary, 10 cm row at 150 ms
    assert _cells(result) == {"constant_velocity": ("fails", 50, "fails", 28),
                              "constant_turn": ("fails", 50, "fails", 50)}


@pytest.mark.parametrize("bad", [
    {"schema": "zephyr-vision-latency-run-1"},
    {**_latency(0.05), "camera_latency_s": {"low": 0.0}},
    {**_latency(0.05), "camera_latency_s": {"low": 0.0, "high": float("nan")}},
    {**_latency(0.05), "camera_latency_s": {"low": 0.0, "high": -0.01}},
    [1, 2],
])
def test_reject_malformed_latency(bad):
    with pytest.raises(lb.LatencyBudgetError):
        lb.budget(bad, _processing(0.01))


@pytest.mark.parametrize("bad", [
    {"schema": "zephyr-vision-latency-1", "median_s": 0.01, "p95_s": 0.02, "n": 10},
    {"schema": "zephyr-processing-time-1", "median_s": True, "p95_s": 0.02, "n": 10},
    {"schema": "zephyr-processing-time-1", "median_s": 0.01, "p95_s": 0.02, "n": 0},
    {"schema": "zephyr-processing-time-1", "p95_s": 0.02, "n": 10},
])
def test_reject_malformed_processing(bad):
    with pytest.raises(lb.LatencyBudgetError):
        lb.budget(_latency(0.05), bad)


@pytest.mark.parametrize("bad", [
    {"schema": "zephyr-espnow-trace-1"},
    {"schema": "zephyr-espnow-fit-1", "delay": None, "source_trace": {"status": "synthetic"}},
    {"schema": "zephyr-espnow-fit-1", "source_trace": {"status": "synthetic"},
     "delay": {"one_way_estimate": {"delay_s": {"value": -0.001, "ci": None}}}},
])
def test_reject_malformed_fit(bad):
    with pytest.raises(lb.LatencyBudgetError):
        lb.budget(_latency(0.05), _processing(0.01), fit=bad)


def test_reject_a_summary_with_another_schema():
    summary = lb.load_summary(lb.DEFAULT_SUMMARY)
    with pytest.raises(lb.LatencyBudgetError):
        lb.budget(_latency(0.05), _processing(0.01), summary={**summary, "schema": "other"})


def test_real_fixture_fit_recovers_the_true_leg_delay(tmp_path):
    # truth: both legs take 2.5 ms on average, so the one-way mean is 2.5 ms
    fixture = tmp_path / "fixture"
    write_fixture(fixture, EchoParams(n_packets=400, burst_start=0.0, burst_end=1.0, return_loss=0.0), 3)
    trace = read_echo_logs(fixture / "sender.csv", fixture / "echoer.csv", trace_id="fx", status="fixture")
    fit = fit_echo_trace(trace, n_boot=200, seed=1)
    result = lb.budget(_latency(0.050), _processing(0.010), fit=fit)
    radio = result["components"]["radio"]
    low, high = fit["delay"]["one_way_estimate"]["delay_s"]["ci"]
    assert low <= 0.0025 <= high
    assert radio["value_s"] == high
    assert radio["value_s"] == pytest.approx(0.0025, abs=0.0005)
    assert radio["label"] == "from a fixture ESP-NOW fit"
    assert result["status"] == "synthetic"


def test_synthetic_latency_run_end_to_end(tmp_path, capsys):
    run = tmp_path / "run"
    assert vision_latency_main(["synth", str(run), "--latency-ms", "120", "--jitter-ms", "8"]) == 0
    assert vision_latency_main(["analyze", str(run)]) == 0
    latency = json.loads((run / "latency.json").read_text())
    proc = tmp_path / "proc.json"
    proc.write_text(json.dumps(_processing(0.010, status="synthetic")))
    out = tmp_path / "budget.json"
    summary_bytes = lb.DEFAULT_SUMMARY.read_bytes()
    assert lb.main(["--latency", str(run / "latency.json"), "--processing", str(proc), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    high = latency["camera_latency_s"]["high"]
    # truth: the synth run's injected mean latency (120 ms plus exposure effects) sits inside the camera range
    true_mean_s = latency["run_meta"]["injected"]["true_mean_latency_ms"] / 1000.0
    assert latency["camera_latency_s"]["low"] <= true_mean_s <= high
    assert high == pytest.approx(true_mean_s, abs=0.005)
    assert result["components"]["camera"]["value_s"] == high
    assert result["total_delay_s"] == pytest.approx(high + 0.010 + 0.040)
    assert result["grid_point_s"] == 0.2  # about 127 + 10 + 40 ms rounds up to 200 ms
    assert result["status"] == "synthetic"
    assert result["summary"]["sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert lb.DEFAULT_SUMMARY.read_bytes() == summary_bytes  # registered summary untouched
    assert "synthetic" in capsys.readouterr().out


def test_typed_processing_median_is_labelled(tmp_path):
    lat = tmp_path / "latency.json"
    lat.write_text(json.dumps(_latency(0.050, "measured")))
    out = tmp_path / "b.json"
    assert lb.main(["--latency", str(lat), "--processing-median-ms", "10", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["components"]["processing"]["value_s"] == pytest.approx(0.010)
    assert result["components"]["processing"]["status"] == "typed in by hand"
    assert result["status"] == "partly measured"


def test_output_inside_sim_runs_is_refused(tmp_path):
    lat = tmp_path / "latency.json"
    lat.write_text(json.dumps(_latency(0.050)))
    target = lb.REPO_ROOT / "sim" / "runs" / "zz-latency-budget-test.json"
    assert lb.main(["--latency", str(lat), "--processing-median-ms", "10", "--out", str(target)]) == 2
    assert not target.exists()
    with pytest.raises(lb.LatencyBudgetError):
        lb._check_output(target)


@pytest.mark.parametrize("text", ['{"schema": "zephyr-s6-recovery-1", "cells": [', 
                                  '{"schema": "zephyr-s6-recovery-1", "cells": [{"slice": "main", "motion": '
                                  '"constant_velocity", "loss_model": "independent", "loss_probability": 0.0, '
                                  '"jitter_s": 0.0, "noise_std_m": 0.05, "delay_s": 0.1}]}'])
def test_broken_summary_is_an_error_not_a_traceback(tmp_path, text):
    lat = tmp_path / "latency.json"
    lat.write_text(json.dumps(_latency(0.050)))
    bad = tmp_path / "summary.json"
    bad.write_text(text)
    assert lb.main(["--latency", str(lat), "--processing-median-ms", "10", "--summary", str(bad)]) == 2
