"""Realistic camera-log quirks: the readers and the fit must not mislead quietly.

Each test builds a synthetic detection log with a known setup (a static marker
seen by a camera at 30 frames per second, Normal position noise), adds one
problem a real webcam bench can produce, and checks what the reader and the
fit do with it. A quirk is either refused with a plain message, flagged with a
"WARNING" line in the fit's assumptions (which tools/intake_vision.py prints in
INTAKE.md), or accepted on purpose, in which case the test says why.

Run ``python tools/tests/test_vision_log_quirks.py`` from the repo root (with
PYTHONPATH=.) to print the outlier-skew numbers used in the session notes.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.detection_log import DetectionLogError, detection_log_lines, parse_detection_log_lines, read_detection_log
from sim.sensor import SensorConfig
from sim.vision_noise_fit import fit_vision_noise
from sim.vision_trace import VISION_TRACE_SCHEMA, VisionTrace
from tools.intake_vision import _fit_section
from tools.make_detection_fixture import make_fixture

TRUTH = (0.10, -0.05, 1.50)
RATE_HZ = 30.0
SIGMA = 0.01
LATENCY_S = 0.030


def make_records(n: int = 600, *, seed: int = 3, sigma: float = SIGMA, subject: str = "aruco-4x4_50-id0") -> list[dict]:
    """One detected record per frame, in capture order, like tools/capture_aruco.py writes."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(n, 3))
    lat = LATENCY_S + rng.normal(0.0, 0.004, size=n).clip(-0.02, 0.02)
    records = []
    for i in range(n):
        capture = 1.0 + i / RATE_HZ
        records.append({
            "observation_id": f"obs-{i:06d}", "frame_id": f"frame-{i:06d}", "subject_id": subject,
            "capture_time_s": capture, "processing_time_s": capture + float(lat[i]), "outcome": "detected",
            "confidence": 1.0, "position_m": [TRUTH[k] + float(noise[i, k]) for k in range(3)],
            "position_frame": "camera",
            "ground_truth": {"position_m": list(TRUTH), "uncertainty_m": 0.0, "method": "tape"},
        })
    return records


def trace_of(records: list[dict], clock_domain: str = "shared_monotonic") -> VisionTrace:
    return VisionTrace.from_dict({
        "schema": VISION_TRACE_SCHEMA, "trace_id": "quirk", "status": "synthetic", "clock_domain": clock_domain,
        "capture": {"source_type": "camera", "source_id": "cam", "device_ids": ["laptop"], "model": "aruco",
                    "frame_rate_hz": RATE_HZ},
        "observations": records,
    })


def fit(records: list[dict], **kwargs) -> dict:
    kwargs.setdefault("n_bootstrap", 0)
    return fit_vision_noise(trace_of(records), **kwargs)


def warnings_of(result: dict) -> list[str]:
    return [line for line in result["assumptions"] if line.startswith("WARNING")]


# --------------------------------------------------------------------------- baseline

def test_clean_log_gives_no_warning():
    result = fit(make_records())
    assert warnings_of(result) == []
    assert result["sensor_config"]["noise_std_m"] == pytest.approx(SIGMA, rel=0.1)


@pytest.mark.parametrize("config, trajectory", [
    (SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05), dropout_probability=0.1), "circle"),
    (SensorConfig(noise_std_m=0.02, mode="burst", burst_start_probability=0.05, burst_end_probability=0.3), "circle"),
    (SensorConfig(noise_std_m=0.01), "static"),
])
def test_clean_fixture_logs_give_no_warning(config, trajectory):
    # Logs from tools/make_detection_fixture.py list every frame, misses included.
    trace = make_fixture(n_frames=3000, frame_rate_hz=30, sensor_config=config, seed=7,
                         latency_mean_s=0.03, latency_std_s=0.008, trajectory=trajectory)
    assert warnings_of(fit_vision_noise(trace, n_bootstrap=0)) == []


# --------------------------------------------------------------------------- dropped frames

def test_dropped_frames_are_flagged():
    # A webcam that skips frames writes no line for them. The fit cannot count
    # them as misses, so dropout_probability reads 0. The fit must say so.
    records = [r for i, r in enumerate(make_records()) if not (100 <= i < 130 or 400 <= i < 420)]
    result = fit(records)
    assert result["sensor_config"]["dropout_probability"] == 0.0  # what the log shows
    warn = " ".join(warnings_of(result))
    assert "gap" in warn and "dropout_probability" in warn
    assert "about 50 frames" in warn


def test_normal_frame_jitter_is_not_a_gap():
    records = make_records()
    rng = np.random.default_rng(1)
    for r in records:  # +/- 8 ms of timestamp jitter at 33 ms per frame
        shift = float(rng.uniform(-0.008, 0.008))
        r["capture_time_s"] += shift
        r["processing_time_s"] += shift
    assert not any("gap" in w for w in warnings_of(fit(records)))


def test_camera_pause_then_resume_is_flagged_as_a_gap():
    records = make_records()
    for r in records[300:]:  # the stream stalls for 2 s, then frames resume
        r["capture_time_s"] += 2.0
        r["processing_time_s"] += 2.0
    assert any("gap" in w for w in warnings_of(fit(records)))


def test_camera_that_stops_for_good_is_acceptable():
    # tools/capture_aruco.py stops writing when read() fails, so a camera that
    # dies just gives a shorter log. The fit of the frames that exist is still
    # a fair fit; nothing to warn about.
    result = fit(make_records()[:360])
    assert warnings_of(result) == []
    assert result["input"]["counts"]["observed"] == 360


# --------------------------------------------------------------------------- capture stamps

def test_duplicate_capture_stamps_are_flagged():
    records = make_records()
    for i in range(50, 110):  # the same frame logged twice under new ids
        copy = dict(records[i], observation_id=f"dup-{i}", frame_id=f"dup-frame-{i}")
        records.append(copy)
    warn = " ".join(warnings_of(fit(records)))
    assert "60 records share a capture_time_s" in warn


def test_backward_capture_stamps_are_flagged_but_fit_sorts_them():
    records = make_records()
    shuffled = records[:200] + records[205:210] + records[200:205] + records[210:]
    result = fit(shuffled)
    baseline = fit(records)
    # The fit sorts by capture time, so the numbers do not change ...
    assert result["sensor_config"] == baseline["sensor_config"]
    # ... but a monotonic camera clock never runs backward in file order, so say so.
    assert any("goes backward" in w for w in warnings_of(result))


def test_backward_stamps_marked_out_of_order_are_not_flagged():
    records = make_records()
    for r in records[200:205]:
        r["out_of_order"] = True
    shuffled = records[:200] + records[205:210] + records[200:205] + records[210:]
    assert not any("goes backward" in w for w in warnings_of(fit(shuffled)))


# --------------------------------------------------------------------------- marker id flips

def flipped_ids(records: list[dict], every: int = 50) -> list[dict]:
    for i in range(0, len(records), every):  # detector misreads id 0 as id 3 on 2% of frames
        records[i]["subject_id"] = "aruco-4x4_50-id3"
    return records


def test_marker_id_flip_error_names_the_counts():
    with pytest.raises(ValueError) as info:
        fit(flipped_ids(make_records()))
    message = str(info.value)
    assert "aruco-4x4_50-id0 (588 records)" in message and "aruco-4x4_50-id3 (12 records)" in message
    assert "misread" in message


def test_marker_id_flip_with_subject_chosen_flags_hidden_frames():
    result = fit(flipped_ids(make_records()), subject_id="aruco-4x4_50-id0")
    warn = " ".join(warnings_of(result))
    assert "12 frames" in warn and "other subject_id" in warn


def test_two_real_targets_sharing_frames_still_need_a_subject():
    a = make_records(subject="marker-a")
    b = [dict(r, observation_id=r["observation_id"] + "-b", subject_id="marker-b") for r in make_records(seed=4)]
    with pytest.raises(ValueError, match="choose one with subject_id"):
        fit(a + b)


# --------------------------------------------------------------------------- pose flips

def with_z_flips(records: list[dict], every: int = 50) -> list[dict]:
    for i in range(0, len(records), every):
        p = records[i]["position_m"]
        records[i]["position_m"] = [p[0], p[1], -p[2]]
    return records


def test_z_sign_flip_is_flagged():
    result = fit(with_z_flips(make_records()))
    warn = " ".join(warnings_of(result))
    assert "behind the camera" in warn
    assert "outlier" in warn


def test_rotation_180_about_optical_axis_is_flagged():
    records = make_records()
    for i in range(0, len(records), 50):  # x and y both change sign
        p = records[i]["position_m"]
        records[i]["position_m"] = [-p[0], -p[1], p[2]]
    assert any("outlier" in w for w in warnings_of(fit(records)))


def test_heavy_flip_contamination_is_still_flagged():
    # At 10% flips the plain std is so big that flips sit only about 3 std
    # out, so a "beyond 4 std" count alone would miss them.
    result = fit(with_z_flips(make_records(), every=10))
    assert any("outlier" in w for w in warnings_of(result))


def flip_skew(n: int = 3000, p: float = 0.02, seed: int = 11) -> dict:
    """Fit a log with known sigma plus exactly p z-sign flips; return fitted and hand values."""
    every = round(1 / p)
    records = with_z_flips(make_records(n, seed=seed), every=every)
    result = fit(records)
    flips = len(range(0, n, every))
    p_exact = flips / n
    d = -2.0 * TRUTH[2]
    second_moment = math.sqrt((1 - p_exact) * SIGMA**2 + p_exact * d**2)  # sqrt((1-p)s^2 + p d^2)
    mixture_std = math.sqrt(SIGMA**2 + p_exact * (1 - p_exact) * d**2)   # subtracts the mean shift p d
    pooled_hand = math.sqrt((2 * SIGMA**2 + mixture_std**2) / 3)
    return {"p": p_exact, "d_m": d, "fitted_std_z_m": result["per_axis"]["std_m"][2],
            "hand_second_moment_m": second_moment, "hand_mixture_std_m": mixture_std,
            "fitted_noise_std_m": result["sensor_config"]["noise_std_m"], "hand_pooled_m": pooled_hand,
            "fitted_bias_z_m": result["sensor_config"]["bias_m"][2], "hand_bias_z_m": p_exact * d,
            "robust_std_z_m": result["per_axis"]["robust_std_m"][2],
            "fraction_beyond_4_std_z": result["per_axis"]["fraction_beyond_4_std"][2]}


def test_flip_skew_matches_the_hand_mixture_formula():
    # Independent check: the variance of a two-part mixture by hand, not the fit's code.
    s = flip_skew()
    assert s["fitted_std_z_m"] == pytest.approx(s["hand_mixture_std_m"], rel=0.02)
    assert s["fitted_std_z_m"] == pytest.approx(s["hand_second_moment_m"], rel=0.02)
    assert s["fitted_noise_std_m"] == pytest.approx(s["hand_pooled_m"], rel=0.02)
    assert s["fitted_bias_z_m"] == pytest.approx(s["hand_bias_z_m"], abs=0.002)
    # The outlier-resistant spread barely moves, which is why the warning compares the two.
    assert s["robust_std_z_m"] == pytest.approx(SIGMA, rel=0.1)


# --------------------------------------------------------------------------- units

def test_millimeter_positions_with_meter_truth_are_refused():
    records = make_records()
    for r in records:  # marker length typed in mm, so solvePnP gives tvec in mm
        r["position_m"] = [1000.0 * v for v in r["position_m"]]
    with pytest.raises(ValueError, match="not in meters"):
        fit(records)


def test_meter_positions_with_millimeter_truth_are_refused():
    records = make_records()
    for r in records:
        r["ground_truth"]["position_m"] = [1000.0 * v for v in TRUTH]
    with pytest.raises(ValueError, match="not in meters"):
        fit(records)


def test_pixel_positions_with_meter_truth_are_refused():
    records = make_records()
    rng = np.random.default_rng(5)
    for r in records:  # marker center in pixels put in position_m
        r["position_m"] = [640.0 + float(rng.normal(0, 0.5)), 360.0 + float(rng.normal(0, 0.5)), 1.0]
    with pytest.raises(ValueError, match="not in meters"):
        fit(records)


def test_pixels_on_both_sides_are_flagged():
    # Truth and measurement both in pixels agree with each other, so only the
    # size of the numbers gives it away. That is a warning, not a refusal,
    # because a far target in meters is possible.
    records = make_records()
    for r in records:
        r["position_m"] = [640.0 + 100 * (v - t) for v, t in zip(r["position_m"], TRUTH)]
        r["ground_truth"]["position_m"] = [640.0, 640.0, 640.0]
    warn = " ".join(warnings_of(fit(records)))
    assert "pixels" in warn


# --------------------------------------------------------------------------- latency

def test_latency_spike_over_5_s_blocks_the_latency_fit():
    records = make_records()
    records[250]["processing_time_s"] = records[250]["capture_time_s"] + 7.0  # laptop stalled
    result = fit(records)
    assert result["scenario_delay"] is None
    assert "5 s" in result["scenario_delay_null_reason"]
    assert any("5 s" in w for w in warnings_of(result))
    # The noise fit itself is unaffected.
    assert result["sensor_config"]["noise_std_m"] == pytest.approx(fit(make_records())["sensor_config"]["noise_std_m"])


def test_wall_clock_processing_time_blocks_the_latency_fit():
    records = make_records()
    for r in records:  # processing stamped with time.time() while capture uses a monotonic clock
        r["processing_time_s"] = 1.7e9 + r["capture_time_s"]
    result = fit(records)
    assert result["scenario_delay"] is None
    assert "clock" in result["scenario_delay_null_reason"]


def test_latency_under_5_s_is_kept():
    records = make_records()
    records[250]["processing_time_s"] = records[250]["capture_time_s"] + 0.9
    assert fit(records)["scenario_delay"] is not None


# --------------------------------------------------------------------------- frozen stream

def test_frozen_stream_is_flagged():
    # Some webcam drivers keep returning the last buffer when the stream stalls.
    records = make_records()
    for r in records[400:]:
        r["position_m"] = list(records[399]["position_m"])
    result = fit(records)
    assert any("repeat the previous position_m exactly" in w for w in warnings_of(result))


# --------------------------------------------------------------------------- empty and truncated logs

def lines_of(records: list[dict]) -> list[str]:
    return detection_log_lines(trace_of(records))


def test_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    with pytest.raises(DetectionLogError, match="empty file"):
        read_detection_log(path)


def test_header_only_log_reads_but_fit_says_no_frames(tmp_path):
    path = tmp_path / "header.jsonl"
    path.write_bytes((lines_of(make_records(3))[0] + "\n").encode("utf-8"))
    trace = read_detection_log(path)
    assert trace.observations == ()
    with pytest.raises(ValueError, match="no observation lines"):
        fit_vision_noise(trace, n_bootstrap=0)


def test_last_line_cut_off_mid_write_says_so(tmp_path):
    lines = lines_of(make_records(20))
    path = tmp_path / "cut.jsonl"
    path.write_bytes(("\n".join(lines[:-1]) + "\n" + lines[-1][:40]).encode("utf-8"))  # program killed mid-line
    with pytest.raises(DetectionLogError) as info:
        read_detection_log(path)
    message = str(info.value)
    assert ":21:" in message and "cut off" in message


def test_bad_middle_line_does_not_claim_a_cut_off(tmp_path):
    lines = lines_of(make_records(20))
    lines[5] = lines[5][:40]
    path = tmp_path / "bad.jsonl"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    with pytest.raises(DetectionLogError) as info:
        read_detection_log(path)
    assert "cut off" not in str(info.value)


def test_parse_lines_round_trip_still_valid():
    trace = parse_detection_log_lines(lines_of(make_records(20)))
    assert len(trace.observations) == 20


# --------------------------------------------------------------------------- intake shows the warnings

def test_intake_section_prints_fit_warnings(tmp_path):
    result = fit(with_z_flips(make_records()))
    text = "\n".join(_fit_section("cal", tmp_path / "cal.jsonl", result))
    assert "WARNING" in text


if __name__ == "__main__":
    print(json.dumps(flip_skew(), indent=2))
