"""Tests for tools/vision_latency_to_link.py (no OpenCV, no camera).

The end-to-end test makes a synthetic recording with a known latency, runs the
existing analyzer on it, converts the result and checks the link against the
recording's answer key. The synthetic jitter is abs(normal(0, sigma)), a
half-normal, so the truth is the realized mean and std of the injected
per-frame latencies (about 126.6 ms and 4.8 ms for --latency-ms 120
--jitter-ms 8), not the two command-line numbers.
"""
import json
import math

import numpy as np
import pytest

from sim.vision_latency import SynthConfig, synth_recording
from tools import vision_latency as latency_cli
from tools import vision_latency_to_link as tool
from tools.s6_link_rerun import parse_link

PROCESSING = {"schema": "zephyr-processing-time-1", "median_s": 0.015, "p95_s": 0.021, "n": 300,
              "status": "synthetic"}
EXPOSURE_S = SynthConfig().exposure_s  # the synth default the end-to-end run uses


def _latency(low=0.05, high=0.13, jitter=0.008, status="measured"):
    """A hand-written latency.json with round numbers, for exact hand checks."""
    return {"schema": "zephyr-vision-latency-1", "status": status,
            "settings": {"display_lag_s": [0.0, 0.06]},
            "apparent_latency_s": {"mean": 0.125, "stated_error": 0.005, "jitter_std_from_straddles": jitter},
            "camera_latency_s": {"low": low, "high": high, "note": "x"}, "warnings": []}


def _fit(delay=0.03, jitter=0.006, status="measured"):
    return {"schema": "zephyr-vision-noise-fit-1", "input": {"status": status},
            "scenario_delay": {"measurement_delay_s": delay, "measurement_jitter_s": jitter}}


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("vl") / "run"
    assert latency_cli.main(["synth", str(run_dir), "--latency-ms", "120", "--jitter-ms", "8"]) == 0
    assert latency_cli.main(["analyze", str(run_dir)]) == 0
    # The answer key: the same config as cmd_synth builds from those flags.
    truth = synth_recording(SynthConfig(latency_s=0.120, latency_jitter_s=0.008, seed=1)).true_latency_s
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert 1000 * float(np.mean(truth)) == pytest.approx(meta["injected"]["true_mean_latency_ms"], abs=1e-9)
    return run_dir, truth


def test_synthetic_run_recovers_the_injected_delay_and_jitter(synth_run, tmp_path):
    run_dir, truth = synth_run
    analysis = json.loads((run_dir / "latency.json").read_text())
    stated = analysis["apparent_latency_s"]["stated_error"]
    true_mean, true_std = float(np.mean(truth)), float(np.std(truth, ddof=1))
    link = tool.convert_files(run_dir / "latency.json", tmp_path / "link.json")
    # Truth: realized mean latency (about 0.1266 s). The analyzer states mean +/- stated covers it,
    # and display lag was 0, so the upper bound (mean + stated) lies in [truth, truth + 2 * stated].
    assert true_mean <= link["delay_s"] <= true_mean + 2 * stated
    # Truth: realized latency std (about 4.8 ms). The straddle estimate reads high by roughly
    # exposure / 7 (sim/vision_latency.py) and has sampling error std / sqrt(2 (n - 1)).
    n = analysis["apparent_latency_s"]["straddle_frames"]
    assert abs(link["jitter_s"] - true_std) <= EXPOSURE_S / 7 + 3 * true_std / math.sqrt(2 * (n - 1))
    assert link["status"] == "synthetic"
    assert link["loss_probability"] == 0.0
    cam = link["components"]["camera"]["latency_s"]
    assert cam["lower"] <= true_mean <= cam["upper"] == link["delay_s"]


def test_processing_median_adds_exactly_and_keeps_jitter(synth_run, tmp_path):
    run_dir, truth = synth_run
    plain = tool.convert_files(run_dir / "latency.json", tmp_path / "a.json")
    (tmp_path / "proc.json").write_text(json.dumps(PROCESSING))
    with_proc = tool.convert_files(run_dir / "latency.json", tmp_path / "b.json",
                                   processing_path=tmp_path / "proc.json")
    # Truth: realized mean latency plus the 0.015 s processing median, within the analyzer's interval.
    stated = json.loads((run_dir / "latency.json").read_text())["apparent_latency_s"]["stated_error"]
    assert float(np.mean(truth)) + 0.015 <= with_proc["delay_s"] <= float(np.mean(truth)) + 0.015 + 2 * stated
    assert with_proc["delay_s"] - plain["delay_s"] == pytest.approx(0.015, abs=1e-15)
    assert with_proc["jitter_s"] == plain["jitter_s"]
    assert any("no standard deviation" in w for w in with_proc["warnings"])
    assert set(with_proc["inputs"]) == {"latency", "processing"}


def test_output_parses_in_the_link_rerun():
    link = tool.build_link(_latency(), processing=dict(PROCESSING, status="measured"))
    assert parse_link(link) == {"delay_s": pytest.approx(0.145), "jitter_s": 0.008,
                                "loss_probability": 0.0, "loss_model": "independent"}
    assert parse_link(json.loads(json.dumps(link)))["delay_s"] == pytest.approx(0.145)


def test_fit_delay_adds_and_jitters_combine_in_quadrature():
    # Hand check: 0.13 + 0.03 = 0.16 s; sqrt(0.008^2 + 0.006^2) = 0.010 s (a 6-8-10 triangle).
    link = tool.build_link(_latency(), vision_fit=_fit())
    assert link["delay_s"] == pytest.approx(0.16, abs=1e-15)
    assert link["jitter_s"] == pytest.approx(0.010, abs=1e-15)
    assert link["components"]["processing"]["statistic"] == "mean"
    assert link["components"]["camera"]["latency_s"] == {"lower": 0.05, "mid": pytest.approx(0.09), "upper": 0.13}
    assert link["status"] == "measured"


def test_display_lag_is_passed_through_as_an_assumption():
    camera = tool.build_link(_latency())["components"]["camera"]
    assert camera["display_lag_assumed_s"] == [0.0, 0.06]
    assert "assumption" in camera["display_lag_note"]
    assert camera["used"] == "upper"


@pytest.mark.parametrize("camera, extra, expected", [
    ("synthetic", None, "synthetic"),
    ("measured", None, "measured"),
    ("measured", {"processing": dict(PROCESSING, status="measured")}, "measured"),
    ("measured", {"processing": PROCESSING}, "synthetic"),
    ("synthetic", {"vision_fit": _fit(status="measured")}, "synthetic"),
    ("measured", {"processing": {k: v for k, v in PROCESSING.items() if k != "status"}}, "unknown"),
    ("unknown", None, "unknown"),
    ("fixture", None, "unknown"),
])
def test_status_synthetic_stays_synthetic(camera, extra, expected):
    assert tool.build_link(_latency(status=camera), **(extra or {}))["status"] == expected


def test_missing_camera_jitter_is_flagged_not_hidden():
    link = tool.build_link(_latency(jitter=None), vision_fit=_fit())
    assert link["jitter_s"] == pytest.approx(0.006)
    assert any("camera jitter was not measured" in w for w in link["warnings"])


def _without(data, *path):
    data = json.loads(json.dumps(data))
    target = data
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return data


@pytest.mark.parametrize("latency, extra, message", [
    ([1, 2], {}, "JSON object"),
    (dict(_latency(), schema="zephyr-espnow-fit-1"), {}, "schema zephyr-vision-latency-1"),
    (dict(_latency(), error="only 3 usable frames; need at least 10"), {}, "analysis failed"),
    (_without(_latency(), "camera_latency_s"), {}, "camera_latency_s"),
    (_without(_latency(), "camera_latency_s", "high"), {}, "high"),
    (_latency(high=float("nan")), {}, "finite"),
    (_latency(high=True), {}, "finite"),
    (_latency(low=0.2, high=0.1), {}, "above"),
    (_latency(low=-0.1, high=0.0), {}, "not positive"),
    (_latency(jitter=-0.001), {}, "zero or more"),
    (_latency(), {"processing": PROCESSING, "vision_fit": _fit()}, "not both"),
    (_latency(), {"processing": dict(PROCESSING, schema="zephyr-detection-log-1")}, "zephyr-processing-time-1"),
    (_latency(), {"processing": dict(PROCESSING, p95_s=0.01)}, "below median_s"),
    (_latency(), {"processing": dict(PROCESSING, n=0)}, "positive integer"),
    (_latency(), {"processing": _without(PROCESSING, "median_s")}, "median_s"),
    (_latency(), {"vision_fit": dict(_fit(), schema="zephyr-espnow-fit-1")}, "zephyr-vision-noise-fit-1"),
    (_latency(), {"vision_fit": {**_fit(), "scenario_delay": None, "scenario_delay_null_reason": "clock"}},
     "no scenario_delay: clock"),
])
def test_malformed_or_other_schema_input_is_rejected(latency, extra, message):
    with pytest.raises(tool.LinkConversionError, match=message):
        tool.build_link(latency, **extra)


def test_cli_rejects_bad_files_and_sim_runs(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert tool.main([str(bad), "--output", str(tmp_path / "out.json")]) == 2
    good = tmp_path / "latency.json"
    good.write_text(json.dumps(_latency()))
    assert tool.main([str(good), "--output", str(tool._REPO_ROOT / "sim" / "runs" / "x.json")]) == 2
    assert "sim/runs" in capsys.readouterr().err
    assert not (tool._REPO_ROOT / "sim" / "runs" / "x.json").exists()
    with pytest.raises(SystemExit):
        tool.main([str(good), "--output", str(tmp_path / "o.json"), "--processing", "a", "--vision-fit", "b"])


def test_same_input_is_byte_identical(tmp_path, capsys):
    good = tmp_path / "latency.json"
    good.write_text(json.dumps(_latency()))
    assert tool.main([str(good), "--output", str(tmp_path / "one.json")]) == 0
    assert tool.main([str(good), "--output", str(tmp_path / "two.json")]) == 0
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    assert "Link delay 130.0 ms" in capsys.readouterr().out


@pytest.mark.parametrize("latency, extra, message", [
    (dict(_latency(), status=["measured"]), {}, "status must be a string"),
    (dict(_latency(), warnings=None), {}, "warnings must be a list of strings"),
    (dict(_latency(), warnings="frame gaps"), {}, "warnings must be a list of strings"),
    ({**_latency(), "settings": {"display_lag_s": [0.06, 0.0]}}, {}, "min is above max"),
    ({**_latency(), "settings": {"display_lag_s": [0.0]}}, {}, "display_lag_s"),
    (_latency(), {"processing": dict(PROCESSING, median_s=-0.001)}, "zero or more"),
    (_latency(), {"processing": dict(PROCESSING, median_s=float("nan"))}, "finite"),
    (_latency(), {"processing": dict(PROCESSING, status=1)}, "status must be a string"),
    (_latency(), {"vision_fit": {**_fit(), "scenario_delay": {"measurement_jitter_s": 0.006}}},
     "measurement_delay_s"),
])
def test_bad_fields_are_rejected_not_crashed(latency, extra, message):
    with pytest.raises(tool.LinkConversionError, match=message):
        tool.build_link(latency, **extra)


def test_assumptions_and_skew_are_stated():
    link = tool.build_link(_latency(), processing=PROCESSING)
    assert "no overlap" in link["components"]["processing"]["assumption"]
    assert any("median" in w and "understate" in w for w in link["warnings"])
    assert link["components"]["processing"]["p95_s"] == 0.021


def test_negative_camera_lower_bound_is_flagged():
    link = tool.build_link(_latency(low=-0.01, high=0.05))
    assert any("lower bound is negative" in w for w in link["warnings"])


def test_output_may_not_overwrite_the_input(tmp_path, capsys):
    good = tmp_path / "latency.json"
    good.write_text(json.dumps(_latency()))
    before = good.read_bytes()
    assert tool.main([str(good), "--output", str(good)]) == 2
    assert good.read_bytes() == before
    assert "overwrite" in capsys.readouterr().err
