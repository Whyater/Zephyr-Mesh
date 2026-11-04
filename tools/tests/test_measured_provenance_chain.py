"""The word "measured" on an output must be backed by provenance from the bench files.

Truth in these tests is what each input file actually contains, not what any reader
computes: a file that only says ``"status": "measured"``, with no operator, capture
time, code revision or raw-file hash anywhere in it, cannot be traced to a capture,
so no reader may call its result measured. A file made from a trace that carries
its provenance block must still be called measured (the downgrade must not fire on
honest inputs).
"""
import json

import pytest

from sim.detection_log import read_detection_log
from sim.vision_noise_fit import fit_vision_noise
from sim.vision_trace import VisionTrace
from tools import latency_budget as lb
from tools import make_detection_fixture as fixture_cli
from tools import s6_combined_rerun as combined
from tools import vision_latency_to_link as conv
from tools.fit_espnow_log import fit_echo_trace
from tools.import_espnow_log import read_echo_logs
from tools.make_espnow_echo_fixture import EchoParams, write_fixture
from tools.s6_link_rerun import describe_input, run_rerun, status_label
from tools.s6_measured_rerun import fit_status

RADIO_PROVENANCE = {
    "instrument": {"kind": "esp32-echo-pair", "model": "test-board", "settings": {"channel": 1}},
    "calibration_ref": None, "calibration_ref_null_reason": "test trace",
    "raw_sha256": "c" * 64, "code_revision": "rev-radio", "captured_at": "2026-10-27T12:00:00-06:00",
    "operator": "radio operator", "ground_truth": None, "ground_truth_null_reason": "test trace",
}
VISION_PROVENANCE = {
    "instrument": {"kind": "camera", "model": "bench-cam", "settings": {}},
    "calibration_ref": None, "calibration_ref_null_reason": "test",
    "raw_sha256": "d" * 64, "code_revision": "rev-vision", "captured_at": "2026-10-30T09:30:00-06:00",
    "operator": "vision operator", "ground_truth": None, "ground_truth_null_reason": "fixture truth",
}
RECORDER_META = {"schema": "zephyr-vision-latency-run-1", "status": "measured", "n_bits": 8,
                 "opencv_version": "4.x", "platform": "bench-test-platform", "clock": "time.monotonic"}
PROCESSING_PROVENANCE = {"operator": "p", "captured_at": "2026-10-30T10:00:00-06:00", "code_revision": "r",
                         "raw_sha256": "e" * 64}


def _radio_fit(tmp_path, status, provenance=None):
    fixture = tmp_path / f"echo-{status}"
    write_fixture(fixture, EchoParams(n_packets=80, burst_start=0.0, burst_end=1.0, return_loss=0.0), 4)
    extra = {"source": "bench capture", "provenance": provenance} if provenance else {}
    trace = read_echo_logs(fixture / "sender.csv", fixture / "echoer.csv", trace_id="bench-1", status=status,
                           **extra)
    return fit_echo_trace(trace, n_boot=10, seed=2)


def _vision_trace(tmp_path, status, provenance=None):
    log = tmp_path / "det.jsonl"
    fixture_cli.main([str(log), "--frames", "200", "--rate", "30", "--noise-std", "0.03", "--bias", "0", "0", "0",
                      "--dropout", "0.1", "--latency-mean", "0.04", "--latency-std", "0.01", "--seed", "3"])
    doc = read_detection_log(log).as_dict()
    doc["status"] = status
    if provenance:
        doc["source"] = "bench-camera"
        doc["provenance"] = provenance
    return VisionTrace.from_dict(doc)


def _bare_radio_fit():
    """A hand-typed fit: right schema and numbers, says measured, nothing traceable."""
    return {"schema": "zephyr-espnow-fit-1", "source_trace": {"trace_id": "t1", "status": "measured"},
            "s6_deliver": {"delay_s": 0.003, "jitter_s": 0.001, "loss_probability": 0.0},
            "delay": {"one_way_estimate": {"delay_s": {"value": 0.003, "ci": [0.002, 0.004]}}}}


def _bare_vision_fit():
    return {"schema": "zephyr-vision-noise-fit-1", "input": {"trace_id": "hand", "status": "measured"},
            "sensor_config": {"noise_std_m": 0.02, "bias_m": [0.0, 0.0, 0.0], "dropout_probability": 0.0},
            "per_axis": {"axes": ["x", "y", "z"], "std_m": [0.02] * 3, "position_frame": "camera"},
            "scenario_delay": None}


def _bare_latency(high_s=0.05):
    return {"schema": "zephyr-vision-latency-1", "status": "measured",
            "apparent_latency_s": {"mean": high_s - 0.001, "stated_error": 0.001, "jitter_std_from_straddles": 0.004},
            "camera_latency_s": {"low": high_s - 0.061, "high": high_s},
            "settings": {"display_lag_s": [0.0, 0.06]}, "warnings": []}


def _recorded_latency(high_s=0.05):
    """What analyze writes for a recorded run: hashes of the run files and the recorder metadata."""
    return {**_bare_latency(high_s), "run_meta": dict(RECORDER_META),
            "inputs": {"frames.npz": "1" * 64, "display_log.csv": "2" * 64, "run_meta.json": "3" * 64}}


# ------------------------------------------------------------ the fit writers keep provenance

def test_radio_fit_keeps_the_trace_provenance(tmp_path):
    fit = _radio_fit(tmp_path, "measured", RADIO_PROVENANCE)
    kept = fit["source_trace"]["provenance"]
    assert kept["raw_sha256"] == "c" * 64
    assert kept["operator"] == "radio operator"
    assert kept["captured_at"] == "2026-10-27T12:00:00-06:00"
    assert kept["code_revision"] == "rev-radio"


def test_radio_fit_of_a_fixture_trace_has_the_same_keys_as_before(tmp_path):
    fit = _radio_fit(tmp_path, "fixture")
    assert set(fit["source_trace"]) == {"trace_id", "status", "n_packets"}


def test_vision_fit_keeps_the_trace_provenance(tmp_path):
    fit = fit_vision_noise(_vision_trace(tmp_path, "measured", VISION_PROVENANCE), n_bootstrap=0)
    kept = fit["input"]["provenance"]
    assert kept["raw_sha256"] == "d" * 64
    assert kept["operator"] == "vision operator"
    assert kept["code_revision"] == "rev-vision"


def test_vision_fit_of_a_synthetic_trace_has_the_same_keys_as_before(tmp_path):
    fit = fit_vision_noise(_vision_trace(tmp_path, "synthetic"), n_bootstrap=0)
    assert "provenance" not in fit["input"]


# ------------------------------------------------------------ the readers

def test_link_rerun_does_not_call_a_hand_typed_measured_fit_measured():
    data = _bare_radio_fit()
    raw = json.dumps(data).encode()
    provenance = describe_input(data, raw, "fit.json")
    assert provenance["input_status"] == "unverified"
    assert "provenance" in provenance["provenance_note"]
    label, limits = status_label(provenance)
    assert "measured fit" not in label and "unverified" in label


def test_link_rerun_end_to_end_keeps_a_real_measured_fit_measured(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(_radio_fit(tmp_path, "measured", RADIO_PROVENANCE)))
    run_rerun(path, tmp_path / "out", pilot=True, seeds_count=1)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["status"] == "investigation: link from a measured fit, sensor simulated"


def test_link_rerun_end_to_end_downgrades_a_hand_typed_measured_fit(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(_bare_radio_fit()))
    run_rerun(path, tmp_path / "out", pilot=True, seeds_count=1)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert "measured fit" not in summary["status"]
    assert summary["provenance"]["input_status"] == "unverified"


def test_combined_rerun_with_three_hand_typed_measured_inputs_is_not_measured(tmp_path):
    radio = tmp_path / "radio.json"
    radio.write_text(json.dumps({"delay_s": 0.003, "jitter_s": 0.001, "loss_probability": 0.0,
                                 "status": "measured"}))
    vlink = tmp_path / "vlink.json"
    vlink.write_text(json.dumps({"schema": "zephyr-vision-latency-link-1", "delay_s": 0.05, "jitter_s": 0.004,
                                 "loss_probability": 0.0, "status": "measured"}))
    vfit = tmp_path / "vfit.json"
    vfit.write_text(json.dumps(_bare_vision_fit()))
    out = tmp_path / "out"
    manifest = combined.run_rerun(out, radio_path=radio, vision_link_path=vlink, vision_fit_path=vfit,
                                  seeds_count=1)
    summary = json.loads((out / "summary.json").read_text())
    assert manifest["input_status"] == summary["input_status"] == "unverified"
    assert "measured" not in summary["status"]


def test_combined_rerun_keeps_real_measured_inputs_measured(tmp_path):
    radio = tmp_path / "radio.json"
    radio.write_text(json.dumps(_radio_fit(tmp_path, "measured", RADIO_PROVENANCE)))
    vfit = tmp_path / "vfit.json"
    vfit.write_text(json.dumps(fit_vision_noise(_vision_trace(tmp_path, "measured", VISION_PROVENANCE),
                                                n_bootstrap=0)))
    out = tmp_path / "out"
    manifest = combined.run_rerun(out, radio_path=radio, vision_fit_path=vfit, seeds_count=1)
    assert manifest["input_status"] == "measured"


def test_measured_rerun_does_not_call_a_hand_typed_measured_fit_measured():
    assert fit_status(_bare_vision_fit()) == "unverified"


def test_measured_rerun_keeps_a_real_measured_fit_measured(tmp_path):
    fit = fit_vision_noise(_vision_trace(tmp_path, "measured", VISION_PROVENANCE), n_bootstrap=0)
    assert fit_status(fit) == "measured"


def test_link_converter_does_not_call_a_bare_measured_latency_file_measured():
    link = conv.build_link(_bare_latency())
    assert link["status"] == "unverified"
    assert link["components"]["camera"]["status"] == "unverified"


def test_link_converter_keeps_a_recorded_latency_file_measured():
    link = conv.build_link(_recorded_latency())
    assert link["status"] == "measured"


def test_link_converter_does_not_call_a_bare_measured_vision_fit_measured():
    fit = {**_bare_vision_fit(), "scenario_delay": {"measurement_delay_s": 0.01, "measurement_jitter_s": 0.002}}
    link = conv.build_link(_recorded_latency(), vision_fit=fit)
    assert link["status"] == "unverified"


def test_budget_with_hand_typed_measured_inputs_is_not_measured():
    measured_processing = {"schema": "zephyr-processing-time-1", "median_s": 0.01, "p95_s": 0.02, "n": 100,
                           "status": "measured", "provenance": dict(PROCESSING_PROVENANCE)}
    result = lb.budget(_bare_latency(), measured_processing, fit=_bare_radio_fit())
    assert result["status"] == "partly measured"  # only the processing file is traceable
    assert result["components"]["camera"]["status"] == "unverified"
    assert result["components"]["radio"]["status"] == "unverified"
    assert "measured" not in result["components"]["radio"]["label"].replace("says measured", "")


def test_budget_with_traceable_inputs_is_measured(tmp_path):
    measured_processing = {"schema": "zephyr-processing-time-1", "median_s": 0.01, "p95_s": 0.02, "n": 100,
                           "status": "measured", "provenance": dict(PROCESSING_PROVENANCE)}
    fit = _radio_fit(tmp_path, "measured", RADIO_PROVENANCE)
    assert lb.budget(_recorded_latency(), measured_processing, fit=fit)["status"] == "measured"


# ------------------------------------------------------------ "partly measured" and the typed processing median

def test_budget_with_nothing_measured_is_not_partly_measured():
    # camera status unknown, processing typed by hand, radio assumed: not one input is measured
    latency = {**_bare_latency(), "status": "unknown"}
    result = lb.budget(latency, processing_term_override=lb.typed_processing_term(0.01))
    assert result["status"] == "not measured"


@pytest.mark.parametrize("typed", ["0.012", "0.3", "600", "2000"])
def test_a_processing_median_outside_half_a_ms_to_500_ms_is_refused(tmp_path, capsys, typed):
    latency = tmp_path / "latency.json"
    latency.write_text(json.dumps(_bare_latency()))
    code = lb.main(["--latency", str(latency), "--processing-median-ms", typed])
    assert code == 2
    err = capsys.readouterr().err
    assert "milliseconds" in err and "seconds" in err


@pytest.mark.parametrize("typed", ["0.5", "12", "500"])
def test_a_plausible_processing_median_is_accepted(tmp_path, capsys, typed):
    latency = tmp_path / "latency.json"
    latency.write_text(json.dumps(_bare_latency()))
    assert lb.main(["--latency", str(latency), "--processing-median-ms", typed]) == 0
