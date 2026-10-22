"""Tests for tools/capture_aruco.py.

Nothing here opens a real camera. The live loop is driven by a fake frame
source that hands out pictures drawn by tools.aruco_geometry.render_marker_frame.
Tests that need OpenCV skip themselves when it is not installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.detection_log import read_detection_log
from sim.vision_noise_fit import fit_vision_noise
from tools import aruco_geometry as geo
from tools import capture_aruco as cap

# --- the scene used by the OpenCV tests ------------------------------------
WIDTH, HEIGHT, HFOV_DEG = 960, 540, 70.0
MARKER_L = 0.15
NOMINAL = (0.05, -0.03, 1.2)  # x right, y down, z out, meters: what the "tape measure" says
JITTER_STD_M = 0.005
FRAMES = 120
BLANK_FRAMES = 24  # marker absent: a known dropout rate of 0.2
WARMUP = 3
FPS = 30.0
LISTED = ["FaceTime HD Camera", "OBS Virtual Camera"]
REAL_LIST_CAMERAS = cap.list_cameras  # kept before the autouse fixture swaps it for a fake


@pytest.fixture(autouse=True)
def no_real_camera(monkeypatch):
    """Make sure no test can open a camera or a window by accident."""
    try:
        import cv2
    except ImportError:
        return
    real_capture = cv2.VideoCapture

    def guarded(*args, **kwargs):
        if args and isinstance(args[0], int):
            raise AssertionError("a test tried to open a camera by index")
        return real_capture(*args, **kwargs)

    def no_window(*args, **kwargs):
        raise AssertionError("a test tried to open a preview window")

    monkeypatch.setattr(cv2, "VideoCapture", guarded)
    monkeypatch.setattr(cv2, "namedWindow", no_window)
    monkeypatch.setattr(cv2, "imshow", no_window)


@pytest.fixture(autouse=True)
def no_system_profiler_and_no_repo_guard(monkeypatch, tmp_path_factory):
    """Never run system_profiler, and move the repo guard off the test temp folder.

    pytest's temp folder sits inside the repo when --basetemp points there, so
    the guard is aimed at a separate empty folder; test_live_output_inside_the_repo_exits_2
    checks the real guard.
    """
    monkeypatch.setattr(cap, "list_cameras", lambda *args, **kwargs: list(LISTED))
    monkeypatch.setattr(cap, "PRIVACY_ROOT", tmp_path_factory.mktemp("pretend-repo"))


class FakeClock:
    """A clock that starts at a big number (so t0 matters) and moves a little on every read."""

    def __init__(self) -> None:
        self.t = 5000.0

    def __call__(self) -> float:
        self.t += 0.0002
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSource:
    """Stands in for cv2.VideoCapture: read(), get(prop), release()."""

    def __init__(self, cv2, frames, size, fps=FPS, clock=None, fail_at=None, error=None, gap_at=None) -> None:
        self.cv2, self.frames, self.size, self.fps, self.clock = cv2, frames, size, fps, clock
        self.fail_at, self.error, self.gap_at = fail_at, error, gap_at
        self.index = 0
        self.released = False

    def read(self):
        if self.clock is not None:
            self.clock.advance((3.0 if self.index == self.gap_at else 1.0) / self.fps)
        if self.index == self.fail_at:  # read number fail_at (counting from 0) blows up
            raise self.error or RuntimeError("camera unplugged")
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, prop):
        values = {self.cv2.CAP_PROP_FRAME_WIDTH: self.size[0], self.cv2.CAP_PROP_FRAME_HEIGHT: self.size[1],
                  self.cv2.CAP_PROP_FPS: self.fps}
        return float(values.get(prop, 0.0))

    def release(self) -> None:
        self.released = True


@pytest.fixture(scope="module")
def cv2():
    return pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def scene(cv2):
    """A static marker with per-frame position jitter, plus blank frames, drawn once for all tests."""
    camera = geo.camera_from_hfov(WIDTH, HEIGHT, HFOV_DEG)
    image = cv2.aruco.generateImageMarker(cap.get_dictionary(cv2, "DICT_4X4_50"), 0, 400)
    rvec = geo.rvec_from_euler_deg(0.0, 12.0, 8.0)
    rng = np.random.default_rng(20261020)
    injected = rng.normal(0.0, JITTER_STD_M, size=(FRAMES, 3))
    blank = np.zeros(FRAMES, dtype=bool)
    blank[rng.choice(FRAMES, size=BLANK_FRAMES, replace=False)] = True
    frames = []
    for k in range(FRAMES):
        frame_rng = np.random.default_rng([7, k])
        if blank[k]:
            frame = _blank_frame(frame_rng)
        else:
            tvec = tuple(np.asarray(NOMINAL) + injected[k])
            frame = geo.render_marker_frame(cv2, image, camera, rvec, tvec, MARKER_L,
                                            pixel_noise_std=1.0, rng=frame_rng)
        frames.append(frame)
    warmup = [_blank_frame(np.random.default_rng([9, k])) for k in range(WARMUP)]
    return SimpleNamespace(cv2=cv2, camera=camera, image=image, rvec=rvec, injected=injected, blank=blank,
                           frames=frames, warmup=warmup)


def _blank_frame(rng) -> np.ndarray:
    gray = np.clip(128.0 + rng.normal(0.0, 1.0, size=(HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def write_calibration(path: Path, camera: geo.CameraModel) -> None:
    path.write_text(json.dumps({
        "schema": geo.CALIBRATION_SCHEMA,
        "camera_matrix": [list(row) for row in camera.camera_matrix],
        "dist_coeffs": list(camera.dist_coeffs),
        "image_size_px": {"width": camera.width_px, "height": camera.height_px},
    }), encoding="utf-8")


def live_argv(tmp_path: Path, name: str, *extra: str) -> list[str]:
    return ["--condition", "test-spot", "--distance-m", str(NOMINAL[2]), "--lateral-m", str(NOMINAL[0]),
            "--vertical-m", str(NOMINAL[1]), "--marker-length-m", str(MARKER_L),
            "--operator", "Test Operator", "--output", str(tmp_path / f"{name}.jsonl"),
            "--no-preview", "--warmup-frames", str(WARMUP), "--queue-frames", "1000", "--yes",
            "--countdown-s", "0", *extra]


def run_live(scene, tmp_path: Path, name: str, *extra: str, frames=None, count=None, with_calibration=True,
             ask=None, **source_options):
    """Run the whole tool in live mode on the fake camera. Returns (exit code, log lines, source)."""
    pieces = list(extra)
    if with_calibration:
        calibration = tmp_path / "camera.json"
        write_calibration(calibration, scene.camera)
        pieces += ["--calibration", str(calibration)]
    frames = scene.frames if frames is None else frames
    count = len(frames) if count is None else count
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.warmup + list(frames), (WIDTH, HEIGHT), clock=clock, **source_options)
    messages: list[str] = []
    code = cap.main([*live_argv(tmp_path, name, *pieces), "--frames", str(count)],
                    source_factory=lambda args, cv2: source, clock=clock, log=messages.append, ask=ask)
    return code, messages, source


# --- header and observation building (no OpenCV needed) ----------------------

TRUTH = cap.Truth((0.05, -0.03, 1.2), 0.01, "tape measure")
POSE = geo.MarkerPose(rvec=(3.0, 0.1, 0.0), tvec=(0.051, -0.029, 1.203), reprojection_error_px=0.07,
                      ambiguity_ratio=0.4)
CORNERS = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]


def hand_observations() -> list[dict]:
    results = [
        cap.FrameResult("detected", None, (0,), CORNERS, (150.0, 150.0), POSE,
                        {"x": 100.5, "y": 100.5, "width": 100.0, "height": 100.0}),
        cap.FrameResult("missed", "marker_not_found", ()),
        cap.FrameResult("missed", "other_markers_only", (3,)),
        cap.FrameResult("unknown", "multiple_target_markers", (0, 0)),
        cap.FrameResult("unknown", "pose_failed", (0,), CORNERS, (150.0, 150.0)),
        cap.FrameResult("detected", None, (0,), CORNERS, (150.0, 150.0), POSE, None),
    ]
    return [cap.build_observation(k, subject_id=cap.subject_id_for("DICT_4X4_50", 0), capture_time_s=k / 30,
                                  processing_time_s=k / 30 + 0.01, truth=TRUTH, result=r)
            for k, r in enumerate(results)]


def hand_capture(live: bool) -> dict:
    return {"source_type": "camera" if live else "recording", "source_id": "mac-built-in-camera",
            "device_ids": ["this-mac"], "model": "opencv-aruco 5.0.0", "resolution_px": {"width": 1280, "height": 720},
            "frame_rate_hz": 30.0}


def hand_provenance(calibration_sha: str | None) -> dict:
    return cap.build_provenance(
        camera_model="built-in camera", settings={"requested_fps": 30.0, "video_codec": "ffv1"},
        calibration_ref=calibration_sha,
        calibration_null_reason=None if calibration_sha else "no calibration file; focal length from a 70 degree horizontal field of view guess",
        raw_sha256="ab" * 32, code_revision="abc123-dirty", captured_at="2026-10-20T20:00:00-05:00",
        operator="Test", condition="test-spot", truth=TRUTH, marker_side_m=0.15, range_m=1.202)


@pytest.mark.parametrize("calibration_sha", [None, "cd" * 32])
def test_live_header_and_observations_pass_the_real_reader(tmp_path, calibration_sha):
    observations = hand_observations()
    header = cap.build_header(live=True, trace_id="aruco-test-spot-20261020T200000", source="run.jsonl",
                              capture=hand_capture(True), clock_domain="shared_monotonic", notes="notes",
                              code_revision="abc123-dirty", provenance=hand_provenance(calibration_sha))
    paths = cap.output_paths(tmp_path / "run.jsonl")
    cap.write_log_atomically(cap.make_trace(header, observations), paths)
    assert not paths["tmp"].exists()  # the temp file was moved into place
    trace = read_detection_log(paths["log"])
    assert trace.status == "measured"
    assert [o.outcome for o in trace.observations] == ["detected", "missed", "missed", "unknown", "unknown", "detected"]
    assert trace.observations[0].position_m == (0.051, -0.029, 1.203)
    assert trace.observations[0].position_frame == "camera"
    assert trace.observations[0].bbox_px["width"] == 100.0
    assert trace.observations[1].miss_reason == "marker_not_found"
    assert trace.observations[2].miss_reason == "other_markers_only"
    assert trace.observations[3].unknown_reason == "multiple_target_markers"
    assert trace.observations[4].unknown_reason == "pose_failed"
    assert all(o.ground_truth["position_m"] == [0.05, -0.03, 1.2] for o in trace.observations)
    assert all(o.confidence is None for o in trace.observations)  # ArUco gives no confidence
    assert trace.observations[0].subject_id == "aruco-dict_4x4_50-id0"
    assert [o.observation_id for o in trace.observations][:2] == ["obs-000001", "obs-000002"]
    assert trace.observations[0].frame_id == "frame-000001"
    provenance = trace.provenance
    assert provenance["ground_truth"]["values"] == {"x_m": 0.05, "y_m": -0.03, "z_m": 1.2, "condition": "test-spot",
                                                    "marker_side_m": 0.15, "range_m": 1.202}
    assert provenance["ground_truth"]["uncertainty"] == {"x_m": 0.01, "y_m": 0.01, "z_m": 0.01}
    assert provenance["calibration_ref"] == calibration_sha
    assert ("calibration_ref_null_reason" in provenance) == (calibration_sha is None)


def test_dry_run_header_is_a_fixture_without_provenance(tmp_path):
    header = cap.build_header(live=False, trace_id="aruco-test-replay", source="video.mkv",
                              capture=hand_capture(False), clock_domain="shared_monotonic", notes="replay",
                              code_revision="unknown", provenance=None)
    paths = cap.output_paths(tmp_path / "replay.jsonl")
    cap.write_log_atomically(cap.make_trace(header, hand_observations()), paths)
    trace = read_detection_log(paths["log"])
    assert trace.status == "fixture"
    assert trace.provenance is None
    assert trace.capture["source_type"] == "recording"


def test_sidecar_row_has_the_documented_fields_and_strict_json():
    result = cap.FrameResult("detected", None, (0,), CORNERS, (150.0, 150.0), POSE, None)
    row = cap.build_sidecar_row(4, capture_time_s=0.5, processing_time_s=0.51, result=result)
    assert set(row) == {"frame_index", "frame_id", "capture_time_s", "processing_time_s", "ids_seen", "corners_px",
                        "center_px", "rvec", "tvec", "reprojection_error_px", "ambiguity_ratio", "outcome",
                        "reason", "video_frame_index", "pose_candidates"}
    assert row["frame_id"] == "frame-000005"
    line = cap.dump_line(row)
    assert json.loads(line) == row
    assert line == json.dumps(row, sort_keys=True, allow_nan=False)
    missed = cap.build_sidecar_row(0, capture_time_s=0.0, processing_time_s=0.0,
                                   result=cap.FrameResult("missed", "marker_not_found", ()))
    assert missed["corners_px"] is None and missed["tvec"] is None and missed["ids_seen"] == []


def test_output_paths_and_replay_time_agree_with_the_geometry_helpers(tmp_path):
    paths = cap.output_paths(tmp_path / "z1.5m.jsonl")
    assert paths["frames"].name == "z1.5m.frames.jsonl"
    assert paths["raw"].name == "z1.5m.raw.mkv"
    assert paths["tmp"].name == "z1.5m.jsonl.tmp"
    reference = geo.replay_capture_times(200, 29.97)
    assert [cap.replay_time(k, 29.97) for k in range(200)] == reference


def test_position_stats_by_hand():
    observations = [{"outcome": "detected", "position_m": [0.0, 0.0, 1.0]},
                    {"outcome": "detected", "position_m": [0.002, 0.0, 1.004]},
                    {"outcome": "missed"}]
    stats = cap.position_stats(observations)
    assert stats["n"] == 2
    assert stats["mean_mm"] == pytest.approx([1.0, 0.0, 1002.0])
    assert stats["std_mm"] == pytest.approx([np.sqrt(2.0), 0.0, np.sqrt(8.0)])
    assert stats["pooled_std_mm"] == pytest.approx(np.sqrt((2.0 + 0.0 + 8.0) / 3.0))  # sqrt of the mean variance
    assert stats["range_mean_mm"] == pytest.approx((1000.0 + np.hypot(2.0, 1004.0)) / 2.0)
    one = cap.position_stats(observations[:1])
    assert one["std_mm"] == [None, None, None] and one["pooled_std_mm"] is None
    assert cap.position_stats([{"outcome": "missed"}]) is None
    assert cap.count_outcomes(observations) == {"detected": 2, "missed": 1, "unknown": 0}


def test_load_frame_times(tmp_path):
    good = tmp_path / "a.frames.jsonl"
    good.write_text("\n".join([
        json.dumps({"frame_index": 0, "capture_time_s": 0.0, "video_frame_index": 0}),
        json.dumps({"frame_index": 1, "capture_time_s": 0.033, "video_frame_index": None}),
        json.dumps({"frame_index": 2, "capture_time_s": 0.066, "video_frame_index": 1}),
    ]) + "\n", encoding="utf-8")
    rows, skipped = cap.load_frame_times(good)
    assert skipped == 1
    assert rows == {0: {"frame_index": 0, "capture_time_s": 0.0}, 1: {"frame_index": 2, "capture_time_s": 0.066}}
    old = tmp_path / "old.jsonl"  # a frames file with no video_frame_index: position equals frame_index
    old.write_text(json.dumps({"frame_index": 3, "capture_time_s": 1.0}) + "\n", encoding="utf-8")
    assert cap.load_frame_times(old)[0] == {3: {"frame_index": 3, "capture_time_s": 1.0}}
    for bad_text in ("not json\n", json.dumps({"frame_index": 0}) + "\n",
                     json.dumps({"frame_index": 0, "capture_time_s": -1.0}) + "\n",
                     json.dumps({"frame_index": 0, "capture_time_s": 0.0}) + "\n"
                     + json.dumps({"frame_index": 0, "capture_time_s": 0.1}) + "\n", ""):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(bad_text, encoding="utf-8")
        with pytest.raises(ValueError):
            cap.load_frame_times(bad)


# --- the background video writer (no OpenCV needed) --------------------------

class BlockingWriter:
    """A video writer that waits until told to go, so the queue can fill up."""

    def __init__(self) -> None:
        self.go = threading.Event()
        self.written = 0
        self.released = False

    def write(self, frame) -> None:
        self.go.wait(timeout=10)
        self.written += 1

    def release(self) -> None:
        self.released = True


def test_writer_drops_frames_instead_of_blocking_when_the_queue_is_full():
    inner = BlockingWriter()
    writer = cap.RawVideoWriter(inner, max_frames=3)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    positions = [writer.submit(frame) for _ in range(20)]
    accepted = [p for p in positions if p is not None]
    assert writer.dropped == 20 - len(accepted) > 0
    assert accepted == list(range(len(accepted)))  # video positions are 0, 1, 2, ... with no gaps
    inner.go.set()
    writer.close()
    assert inner.written == len(accepted) and inner.released


def test_writer_gets_its_own_copy_of_each_frame():
    seen = []

    class Recorder:
        def write(self, frame) -> None:
            seen.append(frame.copy())

        def release(self) -> None:
            pass

    writer = cap.RawVideoWriter(Recorder(), max_frames=10)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    writer.submit(frame)
    frame[:] = 255  # the caller reuses its buffer right away
    writer.close()
    assert seen[0].max() == 0


def test_writer_failure_is_reported_on_close():
    class Failing:
        def write(self, frame) -> None:
            raise OSError("disk full")

        def release(self) -> None:
            pass

    writer = cap.RawVideoWriter(Failing(), max_frames=5)
    writer.submit(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(cap.CaptureError, match="disk full"):
        writer.close()


# --- argument errors exit with code 2 (checked before OpenCV is loaded) -----

def refuse_to_ask(prompt: str) -> str:
    raise EOFError


def base_args(tmp_path: Path) -> list[str]:
    return ["--condition", "spot", "--distance-m", "1.5", "--marker-length-m", "0.1", "--hfov-deg", "70",
            "--operator", "T", "--output", str(tmp_path / "out.jsonl"), "--no-preview"]


def replace_option(argv: list[str], option: str, value: str | None) -> list[str]:
    """Copy of argv with an option's value changed, or the option removed when value is None."""
    out = list(argv)
    index = out.index(option)
    if value is None:
        del out[index:index + 2]
    else:
        out[index + 1] = value
    return out


def expect_exit_2(argv: list[str], capsys, fragment: str | None = None) -> None:
    with pytest.raises(SystemExit) as info:
        cap.main(argv, ask=refuse_to_ask)
    assert info.value.code == 2
    if fragment is not None:
        assert fragment in capsys.readouterr().err


def test_bad_condition_labels_are_rejected(tmp_path, capsys):
    for label in ("has space", "semi;colon", "slash/inside", "", "café"):
        expect_exit_2(replace_option(base_args(tmp_path), "--condition", label), capsys, "--condition")
    for label in ("ok", "z1.5m-bright", "A_b.c-9"):
        assert cap.CONDITION_RE.fullmatch(label)


def test_camera_model_must_be_given_exactly_once(tmp_path, capsys):
    argv = base_args(tmp_path)
    expect_exit_2(replace_option(argv, "--hfov-deg", None), capsys, "exactly one")
    calibration = tmp_path / "c.json"
    calibration.write_text("{}", encoding="utf-8")
    expect_exit_2([*argv, "--calibration", str(calibration)], capsys, "exactly one")
    expect_exit_2([*replace_option(argv, "--hfov-deg", None), "--calibration", str(tmp_path / "missing.json")],
                  capsys, "not found")


@pytest.mark.parametrize("option,value", [
    ("--hfov-deg", "0.5"), ("--hfov-deg", "179.5"), ("--hfov-deg", "abc"), ("--distance-m", "-1"),
    ("--distance-m", "nan"), ("--marker-length-m", "0"), ("--marker-length-m", "-0.1"),
    ("--truth-uncertainty-m", "-0.01"), ("--output", "out.txt"), ("--output", "no_such_folder/out.jsonl"),
])
def test_bad_numbers_and_paths_exit_2(tmp_path, capsys, option, value):
    if option == "--output":
        value = str(tmp_path / value)
    argv = base_args(tmp_path)
    if option == "--truth-uncertainty-m":
        argv += [option, value]
    else:
        argv = replace_option(argv, option, value)
    expect_exit_2(argv, capsys)


def test_missing_required_options_exit_2(tmp_path, capsys):
    expect_exit_2(replace_option(base_args(tmp_path), "--marker-length-m", None), capsys, "--marker-length-m")
    expect_exit_2(replace_option(base_args(tmp_path), "--output", None), capsys, "--output")
    # live mode asks for a missing condition; with no keyboard that is a usage error
    expect_exit_2(replace_option(base_args(tmp_path), "--condition", None), capsys, "--condition")
    expect_exit_2(replace_option(base_args(tmp_path), "--operator", None), capsys, "--operator")


def test_refuses_to_overwrite_existing_outputs(tmp_path, capsys):
    argv = base_args(tmp_path)
    for name in ("out.jsonl", "out.frames.jsonl", "out.raw.mkv"):
        target = tmp_path / name
        target.write_text("keep me", encoding="utf-8")
        expect_exit_2(argv, capsys, "already exists")
        target.unlink()
    (tmp_path / "out.jsonl").write_text("keep me", encoding="utf-8")
    expect_exit_2(argv, capsys, "--overwrite")
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == "keep me"


def test_dry_run_option_errors(tmp_path, capsys):
    argv = base_args(tmp_path)
    expect_exit_2([*argv, "--dry-run", str(tmp_path / "no_video.mkv")], capsys, "not found")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    expect_exit_2([*argv, "--dry-run", str(video), "--frame-times", str(tmp_path / "nothing.jsonl")], capsys, "not found")
    expect_exit_2([*argv, "--frame-times", str(video)], capsys, "--dry-run")
    same = tmp_path / "out.frames.jsonl"
    same.write_text("{}\n", encoding="utf-8")
    expect_exit_2([*replace_option(argv, "--output", str(tmp_path / "out.jsonl")), "--dry-run", str(video),
                   "--frame-times", str(same), "--overwrite"], capsys, "same file")
    # a dry run does not prompt: it needs the truth on the command line
    expect_exit_2(replace_option([*argv, "--dry-run", str(video)], "--condition", None), capsys, "--condition")


# --- OpenCV tests -------------------------------------------------------------

def test_fake_camera_run_recovers_injected_jitter_bias_and_dropout(scene, tmp_path):
    code, messages, source = run_live(scene, tmp_path, "fit")
    assert code == 0 and source.released
    paths = cap.output_paths(tmp_path / "fit.jsonl")
    trace = read_detection_log(paths["log"])
    assert len(trace.observations) == FRAMES  # the 3 warmup frames are not in the log
    detected = [o for o in trace.observations if o.outcome == "detected"]
    missed = [o for o in trace.observations if o.outcome == "missed"]
    assert len(missed) == BLANK_FRAMES and len(detected) == FRAMES - BLANK_FRAMES
    assert all(o.miss_reason == "marker_not_found" for o in missed)
    # blank frames are the ones we planted, frame by frame
    assert [o.outcome == "missed" for o in trace.observations] == scene.blank.tolist()

    fit = fit_vision_noise(trace, n_bootstrap=200, seed=0)
    config = fit["sensor_config"]

    # Dropout: the planted rate is exactly 24 / 120 = 0.2 and every other frame was found.
    assert config["dropout_probability"] == pytest.approx(BLANK_FRAMES / FRAMES, abs=1e-12)
    assert fit["uncertainty"]["dropout_probability"]["ci_low"] <= 0.2 <= fit["uncertainty"]["dropout_probability"]["ci_high"]

    # Noise: the true position of each frame was nominal + injected jitter, so the fit should see about the
    # spread we injected. The pooled spread of 96 frames has a standard error of about 1/sqrt(2 * 96 * 3),
    # which is 4 percent, so 15 percent (almost 4 standard errors) around the planted 5 mm is a fair test of
    # the pipeline. The exact draws that were used are known, so we also compare to their own spread. The
    # detector adds only a little in quadrature (measured: under 1 percent here), so 5 percent is generous.
    used = scene.injected[~scene.blank]
    drawn_std = float(np.sqrt(np.mean(np.var(used, axis=0, ddof=1))))
    assert drawn_std == pytest.approx(JITTER_STD_M, rel=0.15)
    assert config["noise_std_m"] == pytest.approx(JITTER_STD_M, rel=0.15)
    assert config["noise_std_m"] == pytest.approx(drawn_std, rel=0.05)

    # Bias: the fit compares to the NOMINAL position, so it should find the mean of the injected jitter
    # (about 0.3 mm per axis from sampling) plus a small detector bias. x and y: within 0.5 mm of that
    # (measured: 0.1 mm). z: OpenCV's corner refinement pulls the corners inward a little, so z reads long
    # by about 0.2 percent of the distance at this marker size (measured: 2.4 mm at 1.2 m). Allow 0.5 percent
    # of Z, which still catches a wrong sign, a wrong scale or a mixed up axis.
    bias = np.asarray(config["bias_m"])
    expected_mean = used.mean(axis=0)
    assert abs(bias[0] - expected_mean[0]) < 0.0005
    assert abs(bias[1] - expected_mean[1]) < 0.0005
    assert 0.0 < bias[2] - expected_mean[2] < 0.005 * NOMINAL[2]

    # The header: a measured log from a camera, with every provenance piece present and checkable.
    assert trace.status == "measured" and trace.source == "fit.jsonl"
    assert trace.capture["source_type"] == "camera"
    assert trace.capture["resolution_px"] == {"width": WIDTH, "height": HEIGHT}
    assert trace.capture["frame_rate_hz"] == pytest.approx(FPS, abs=0.5)
    provenance = trace.provenance
    assert provenance["raw_sha256"] == hashlib.sha256(paths["raw"].read_bytes()).hexdigest()
    assert provenance["calibration_ref"] == hashlib.sha256((tmp_path / "camera.json").read_bytes()).hexdigest()
    assert provenance["operator"] == "Test Operator"
    assert provenance["instrument"]["settings"]["video_codec"] == "ffv1"
    assert provenance["instrument"]["settings"]["dictionary"] == "DICT_4X4_50"
    assert provenance["ground_truth"]["values"] == {"x_m": NOMINAL[0], "y_m": NOMINAL[1], "z_m": NOMINAL[2],
                                                    "condition": "test-spot", "marker_side_m": MARKER_L}
    assert provenance["ground_truth"]["uncertainty"] == {"x_m": 0.01, "y_m": 0.01, "z_m": 0.01}
    settings = provenance["instrument"]["settings"]
    assert settings["exposure"] == "auto, cannot be locked by OpenCV on macOS"
    assert settings["focus"] == "auto, cannot be locked by OpenCV on macOS"
    assert settings["frame_timestamps"].startswith("host monotonic clock when read() returned")
    assert settings["pose_solver"] == "IPPE_SQUARE with SQPNP fallback, lowest reprojection error"
    assert settings["listed_cameras"] == LISTED and settings["camera_confirmed"] == "not asked (--yes)"
    assert settings["warmup_frames"] == WARMUP and settings["countdown_frames"] == 0
    assert (settings["actual_width_px"], settings["actual_height_px"]) == (WIDTH, HEIGHT)
    assert trace.capture["model"] == f"opencv-aruco {scene.cv2.__version__} DICT_4X4_50 IPPE_SQUARE+SQPNP refine=SUBPIX"
    assert trace.capture["calibration_id"] == "camera.json"
    assert provenance["captured_at"][-6] in "+-" or provenance["captured_at"].endswith("Z")
    assert trace.code_revision and trace.code_revision == provenance["code_revision"]

    # Times: the clock starts at 5000 s, but capture times start near zero and processing is never earlier.
    first = trace.observations[0]
    assert 0.0 <= first.capture_time_s < 1.0
    assert all(o.processing_time_s >= o.capture_time_s for o in trace.observations)

    # The frames file: one strict JSON line per recorded frame, in order.
    lines = paths["frames"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == FRAMES
    rows = [json.loads(line) for line in lines]
    assert all(line == cap.dump_line(row) for line, row in zip(lines, rows))
    assert [r["frame_index"] for r in rows] == list(range(FRAMES))
    assert [r["video_frame_index"] for r in rows] == list(range(FRAMES))
    detected_rows = [r for r in rows if r["outcome"] == "detected"]
    assert all(len(r["corners_px"]) == 4 and len(r["rvec"]) == 3 and r["ambiguity_ratio"] is not None
               for r in detected_rows)
    # Both IPPE tilts and the SQPNP pose are in the frames file, and the kept one is among them.
    for r in detected_rows:
        methods = [c["method"] for c in r["pose_candidates"]]
        assert methods.count("IPPE_SQUARE") == 2 and "SQPNP" in methods
        assert r["tvec"] in [c["tvec"] for c in r["pose_candidates"]]
    assert all(r["pose_candidates"] is None for r in rows if r["outcome"] == "missed")
    assert f"SHA-256 {hashlib.sha256(paths['frames'].read_bytes()).hexdigest()}" in trace.notes
    assert max(r["reprojection_error_px"] for r in detected_rows) < 1.0
    # the position in the log is the tvec in the frames file
    assert [o.position_m for o in detected] == [tuple(r["tvec"]) for r in detected_rows]

    # The summary: per-axis numbers, the pooled number the fit uses, and the exact fit command.
    stats = cap.position_stats([{"outcome": o.outcome, "position_m": o.position_m} for o in trace.observations])
    assert stats["pooled_std_mm"] / 1000.0 == pytest.approx(config["noise_std_m"], rel=1e-9)
    assert any(line.startswith("Pooled std, the one noise number the fit uses:") for line in messages)
    assert any(line.startswith("  z ") for line in messages) and any(line.startswith("  x ") for line in messages)
    assert f"Pooled std (the one number the fit uses): {stats['pooled_std_mm']:.2f} mm" in trace.notes
    log_abs, fit_abs = paths["log"].resolve(), paths["fit"].resolve()
    assert f'  PYTHONPATH=. .venv/bin/python tools/fit_vision_noise.py "{log_abs}" --output "{fit_abs}"' in messages
    assert any("repo's .venv" in line for line in messages)
    assert any(line.startswith("Cameras macOS lists: FaceTime HD Camera, OBS Virtual Camera") for line in messages)
    assert not any(line.startswith("WARNING") for line in messages)


def test_dry_run_of_the_saved_video_reproduces_the_live_positions_exactly(scene, tmp_path):
    frames = scene.frames[:40]
    code, _, _ = run_live(scene, tmp_path, "live", frames=frames)
    assert code == 0
    live_paths = cap.output_paths(tmp_path / "live.jsonl")
    live = read_detection_log(live_paths["log"])

    replay_out = tmp_path / "replay.jsonl"
    argv = ["--dry-run", str(live_paths["raw"]), "--frame-times", str(live_paths["frames"]),
            "--condition", "test-spot", "--distance-m", str(NOMINAL[2]), "--lateral-m", str(NOMINAL[0]),
            "--vertical-m", str(NOMINAL[1]), "--marker-length-m", str(MARKER_L),
            "--calibration", str(tmp_path / "camera.json"), "--output", str(replay_out)]
    assert cap.main(argv, ask=refuse_to_ask, log=lambda text: None) == 0
    replay = read_detection_log(replay_out)

    assert replay.status == "fixture" and replay.provenance is None
    assert replay.capture["source_type"] == "recording"
    assert len(replay.observations) == len(live.observations) == 40
    for a, b in zip(live.observations, replay.observations):
        assert a.outcome == b.outcome
        assert a.position_m == b.position_m  # exactly equal: lossless video, same code path
        assert a.bbox_px == b.bbox_px
        assert a.capture_time_s == b.capture_time_s
        assert a.observation_id == b.observation_id and a.frame_id == b.frame_id
        assert b.processing_time_s >= b.capture_time_s
    live_rows = [json.loads(line) for line in live_paths["frames"].read_text(encoding="utf-8").splitlines()]
    replay_rows = [json.loads(line) for line in replay_out.with_name("replay.frames.jsonl").read_text(encoding="utf-8").splitlines()]
    for a, b in zip(live_rows, replay_rows):
        for key in ("corners_px", "center_px", "rvec", "tvec", "reprojection_error_px", "ambiguity_ratio",
                    "ids_seen", "outcome"):
            assert a[key] == b[key]


def test_dry_run_of_an_image_pattern_matches_the_same_frames_processed_live(scene, tmp_path):
    cv2 = scene.cv2
    frames = scene.frames[:15]
    folder = tmp_path / "pictures"
    folder.mkdir()
    for k, frame in enumerate(frames):
        assert cv2.imwrite(str(folder / f"{k:04d}.png"), frame)
    opened = []
    real_capture = cv2.VideoCapture

    def recording_capture(*args, **kwargs):
        opened.append(args[0])
        return real_capture(*args, **kwargs)

    cv2.VideoCapture = recording_capture  # the autouse fixture restores the original afterwards
    try:
        out = tmp_path / "sequence.jsonl"
        argv = ["--dry-run", str(folder / "%04d.png"), "--fps", "30", "--condition", "pictures",
                "--distance-m", str(NOMINAL[2]), "--lateral-m", str(NOMINAL[0]), "--vertical-m", str(NOMINAL[1]),
                "--marker-length-m", str(MARKER_L), "--hfov-deg", str(HFOV_DEG), "--output", str(out)]
        assert cap.main(argv, ask=refuse_to_ask, log=lambda text: None) == 0
    finally:
        cv2.VideoCapture = real_capture
    assert opened and all(isinstance(item, str) for item in opened)  # never a camera index
    trace = read_detection_log(out)
    assert len(trace.observations) == 15
    assert trace.capture["source_id"] == "%04d.png"
    # capture times were made up from the 30 Hz rate: k / 30, the same numbers the geometry helper gives
    assert [o.capture_time_s for o in trace.observations] == geo.replay_capture_times(15, 30.0)
    # same positions as running the identical frames through process_frame directly
    camera = geo.camera_from_hfov(WIDTH, HEIGHT, HFOV_DEG)
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=camera, truth=cap.Truth(NOMINAL, 0.01, "tape measure"))
    for k, observation in enumerate(trace.observations):
        direct, _ = cap.process_frame(frames[k], k, k / 30, ctx)
        assert observation.position_m == (tuple(direct["position_m"]) if direct["outcome"] == "detected" else None)
    assert "made up from a frame rate of 30" in trace.notes


def test_dry_run_without_a_stored_frame_rate_asks_for_one(scene, tmp_path, capsys):
    cv2 = scene.cv2
    folder = tmp_path / "pictures"
    folder.mkdir()
    for k, frame in enumerate(scene.frames[:3]):
        cv2.imwrite(str(folder / f"{k:04d}.png"), frame)
    argv = ["--dry-run", str(folder / "%04d.png"), "--condition", "p", "--distance-m", "1.2",
            "--marker-length-m", "0.15", "--hfov-deg", "70", "--output", str(tmp_path / "x.jsonl")]
    try:
        code = cap.main(argv, ask=refuse_to_ask, log=lambda text: None)
    except SystemExit as exit_info:  # this OpenCV reports no rate for pictures
        assert exit_info.code == 2
        assert "--fps" in capsys.readouterr().err
    else:
        assert code == 0  # this OpenCV reports a default rate, which is also acceptable


def test_second_run_refuses_to_overwrite_and_overwrite_flag_replaces(scene, tmp_path, capsys):
    frames = scene.frames[:6]
    assert run_live(scene, tmp_path, "twice", frames=frames)[0] == 0
    before = (tmp_path / "twice.jsonl").read_bytes()
    with pytest.raises(SystemExit) as info:
        run_live(scene, tmp_path, "twice", frames=frames)
    assert info.value.code == 2 and "already exists" in capsys.readouterr().err
    assert (tmp_path / "twice.jsonl").read_bytes() == before
    assert run_live(scene, tmp_path, "twice", "--overwrite", frames=frames[:4])[0] == 0
    assert len(read_detection_log(tmp_path / "twice.jsonl").observations) == 4


def test_missing_values_are_asked_for_in_live_mode(scene, tmp_path):
    answers = iter(["typed-spot", "1.2", "Typed Name"])
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    frames = scene.frames[:5]
    calibration = tmp_path / "camera.json"
    write_calibration(calibration, scene.camera)
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.warmup + list(frames), (WIDTH, HEIGHT), clock=clock)
    argv = ["--marker-length-m", str(MARKER_L), "--calibration", str(calibration), "--output",
            str(tmp_path / "asked.jsonl"), "--no-preview", "--warmup-frames", str(WARMUP), "--frames", "5",
            "--yes", "--countdown-s", "0"]
    code = cap.main(argv, source_factory=lambda args, cv2: source, clock=clock, ask=ask, log=lambda text: None)
    assert code == 0 and len(prompts) == 3
    trace = read_detection_log(tmp_path / "asked.jsonl")
    assert trace.provenance["operator"] == "Typed Name"
    assert trace.provenance["ground_truth"]["values"]["condition"] == "typed-spot"
    assert trace.observations[0].ground_truth["position_m"] == [0.0, 0.0, 1.2]
    assert trace.trace_id.startswith("aruco-typed-spot-")


def test_hfov_guess_is_recorded_as_a_null_calibration_with_a_reason(scene, tmp_path):
    code, _, _ = run_live(scene, tmp_path, "guess", "--hfov-deg", str(HFOV_DEG), frames=scene.frames[:5],
                          with_calibration=False)
    assert code == 0
    provenance = read_detection_log(tmp_path / "guess.jsonl").provenance
    assert provenance["calibration_ref"] is None
    assert provenance["calibration_ref_null_reason"] == (
        f"no calibration file; focal length from a {HFOV_DEG:g} degree horizontal field of view guess")


def test_calibration_for_the_wrong_picture_size_exits_2(scene, tmp_path, capsys):
    other = geo.camera_from_hfov(1280, 720, 70.0)
    calibration = tmp_path / "camera.json"
    write_calibration(calibration, other)
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.warmup + scene.frames[:3], (WIDTH, HEIGHT), clock=clock)
    argv = live_argv(tmp_path, "mismatch", "--calibration", str(calibration), "--frames", "3")
    with pytest.raises(SystemExit) as info:
        cap.main(argv, source_factory=lambda args, cv2: source, clock=clock, log=lambda text: None)
    assert info.value.code == 2
    assert "1280" in capsys.readouterr().err
    assert source.released


def test_camera_that_stops_early_keeps_the_frames_it_gave(scene, tmp_path):
    code, _, _ = run_live(scene, tmp_path, "short", frames=scene.frames[:7], count=20)
    assert code == 0
    trace = read_detection_log(tmp_path / "short.jsonl")
    assert len(trace.observations) == 7
    assert "end of stream" in trace.notes


def test_no_frames_at_all_is_a_runtime_failure(scene, tmp_path, capsys):
    code, _, _ = run_live(scene, tmp_path, "nothing", frames=[], count=5)
    assert code == 1
    assert "no frames" in capsys.readouterr().err
    assert not (tmp_path / "nothing.jsonl").exists()  # no half-written log
    assert (tmp_path / "nothing.frames.jsonl").exists()


def test_pressing_q_stops_early_and_keeps_what_was_recorded(scene, tmp_path):
    cv2 = scene.cv2

    class QuitAfterFive:
        def __init__(self) -> None:
            self.calls = 0
            self.seen_frames = []

        def show(self, frame, corners_px, text) -> bool:
            self.calls += 1
            self.seen_frames.append(frame)
            return self.calls >= 5

    camera = scene.camera
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=camera, truth=cap.Truth(NOMINAL, 0.01, "tape"))
    clock = FakeClock()
    source = FakeSource(cv2, scene.frames[:30], (WIDTH, HEIGHT), clock=clock)
    preview = QuitAfterFive()
    recording = cap.record_frames(source, ctx, frames=30, clock=clock, preview=preview, log=lambda text: None)
    assert recording.stopped_early
    assert len(recording.observations) == 5 == len(recording.rows)


def test_preview_overlay_is_drawn_on_a_copy_not_on_the_detected_frame(scene):
    cv2 = scene.cv2
    frame = scene.frames[0].copy()
    reference = frame.copy()
    corners = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    shown = cap.draw_overlay(cv2, frame, corners, "frame 1/3")
    assert np.array_equal(frame, reference)  # untouched
    assert shown is not frame and not np.array_equal(shown, frame)


def test_preview_turns_itself_off_when_the_opencv_build_has_no_window_support(scene, monkeypatch):
    cv2 = scene.cv2

    def no_gui(*args, **kwargs):
        raise cv2.error("The function is not implemented. Rebuild the library with GUI support")

    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    messages: list[str] = []
    preview = cap.Preview(cv2, messages.append)
    assert not preview.enabled
    assert len(messages) == 1 and "no GUI" in messages[0]
    assert preview.show(scene.frames[0], None, "x") is False  # does nothing, does not raise
    preview.close()


def test_full_run_without_no_preview_flag_still_records_when_there_is_no_window(scene, tmp_path, monkeypatch):
    cv2 = scene.cv2

    def no_gui(*args, **kwargs):
        raise cv2.error("no GUI")

    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    pieces = [piece for piece in live_argv(tmp_path, "nogui") if piece != "--no-preview"]
    calibration = tmp_path / "camera.json"
    write_calibration(calibration, scene.camera)
    clock = FakeClock()
    source = FakeSource(cv2, scene.warmup + scene.frames[:5], (WIDTH, HEIGHT), clock=clock)
    messages: list[str] = []
    code = cap.main([*pieces, "--calibration", str(calibration), "--frames", "5"],
                    source_factory=lambda args, module: source, clock=clock, log=messages.append)
    assert code == 0
    assert sum("Preview window is not available" in text for text in messages) == 1


def test_detector_does_not_change_the_frame_it_is_given(scene):
    cv2 = scene.cv2
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=scene.camera, truth=cap.Truth(NOMINAL, 0.01, "tape"))
    frame = next(f for f, blank in zip(scene.frames, scene.blank) if not blank)
    before = frame.copy()
    observation, row = cap.process_frame(frame, 0, 0.0, ctx)
    assert observation["outcome"] == "detected" and row["outcome"] == "detected"
    assert np.array_equal(frame, before)


def test_other_markers_and_duplicate_target_markers(scene):
    cv2 = scene.cv2
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=scene.camera, truth=cap.Truth(NOMINAL, 0.01, "tape"))
    tvec = NOMINAL
    other_image = cv2.aruco.generateImageMarker(cap.get_dictionary(cv2, "DICT_4X4_50"), 1, 400)
    other = geo.render_marker_frame(cv2, other_image, scene.camera, scene.rvec, tvec, MARKER_L)
    obs, row = cap.process_frame(other, 0, 0.0, ctx)
    assert (obs["outcome"], obs["miss_reason"], row["ids_seen"]) == ("missed", "other_markers_only", [1])

    single = geo.render_marker_frame(cv2, scene.image, scene.camera, scene.rvec, tvec, MARKER_L)
    twice = np.minimum(single, np.roll(single, 330, axis=1))  # a second copy of marker 0, far from the first
    obs, row = cap.process_frame(twice, 0, 0.0, ctx)
    assert (obs["outcome"], obs["unknown_reason"], row["ids_seen"]) == ("unknown", "multiple_target_markers", [0, 0])
    assert "position_m" not in obs and row["tvec"] is None

    blank = _blank_frame(np.random.default_rng(1))
    obs, _ = cap.process_frame(blank, 0, 0.0, ctx)
    assert (obs["outcome"], obs["miss_reason"]) == ("missed", "marker_not_found")


def test_a_failed_pose_is_logged_as_unknown_pose_failed(scene, monkeypatch):
    cv2 = scene.cv2
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=scene.camera, truth=cap.Truth(NOMINAL, 0.01, "tape"))
    monkeypatch.setattr(geo, "estimate_marker_pose", lambda *args, **kwargs: None)
    frame = next(f for f, blank in zip(scene.frames, scene.blank) if not blank)
    obs, row = cap.process_frame(frame, 0, 0.0, ctx)
    assert (obs["outcome"], obs["unknown_reason"]) == ("unknown", "pose_failed")
    assert row["corners_px"] is not None and row["tvec"] is None


def png_chunks(data: bytes) -> list[tuple[bytes, bytes, int]]:
    """Split a PNG into (type, data, stored CRC) chunks, written here separately from the tool."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, position = [], 8
    while position < len(data):
        (length,) = struct.unpack(">I", data[position:position + 4])
        kind = data[position + 4:position + 8]
        body = data[position + 8:position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length:position + 12 + length])
        chunks.append((kind, body, crc))
        position += 12 + length
    return chunks


def test_print_marker_writes_a_picture_that_prints_at_the_right_size(scene, tmp_path, capsys):
    cv2 = scene.cv2
    target = tmp_path / "marker.png"
    messages: list[str] = []
    assert cap.main(["--print-marker", str(target), "--marker-id", "7"], log=messages.append) == 0
    image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    # 600 px black square (6 cells of 100 px) plus 20 mm = 80 px of white on each side at 4000 px per meter.
    assert image.shape == (760, 760)
    assert image[0, 0] == 255 and image[380, 79] == 255 and image[380, 80] == 0  # the black square starts at 80
    assert image[380, 679] == 0 and image[380, 680] == 255
    chunks = png_chunks(target.read_bytes())
    kinds = [kind for kind, _, _ in chunks]
    assert kinds[0] == b"IHDR" and kinds.count(b"pHYs") == 1 and kinds.index(b"pHYs") < kinds.index(b"IDAT")
    for kind, body, crc in chunks:
        assert zlib.crc32(kind + body) & 0xFFFFFFFF == crc
    phys = next(body for kind, body, _ in chunks if kind == b"pHYs")
    x_ppm, y_ppm, unit = struct.unpack(">IIB", phys)
    assert (x_ppm, y_ppm, unit) == (4000, 4000, 1)  # unit 1 = meter
    assert 600 / x_ppm * 1000.0 == pytest.approx(150.0)  # the black square in mm at 100 percent
    assert 80 / x_ppm * 1000.0 == pytest.approx(20.0)
    assert any("150.0 mm" in text and "20.0 mm" in text for text in messages)
    detector = cv2.aruco.ArucoDetector(cap.get_dictionary(cv2, "DICT_4X4_50"), cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(image)
    assert ids.reshape(-1).tolist() == [7]
    # Another size: the pixels stay a whole number of cells, the density changes.
    other = tmp_path / "small.png"
    assert cap.main(["--print-marker", str(other), "--marker-print-mm", "100", "--marker-margin-mm", "10"],
                    log=lambda text: None) == 0
    phys = next(body for kind, body, _ in png_chunks(other.read_bytes()) if kind == b"pHYs")
    assert struct.unpack(">IIB", phys) == (6000, 6000, 1)
    assert cv2.imread(str(other), cv2.IMREAD_GRAYSCALE).shape == (720, 720)
    with pytest.raises(SystemExit) as info:
        cap.main(["--print-marker", str(target)])
    assert info.value.code == 2 and "--overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        cap.main(["--print-marker", str(tmp_path / "big.png"), "--marker-id", "99"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        cap.main(["--print-marker", str(tmp_path / "marker.jpg")])
    assert info.value.code == 2


def test_marker_print_layout_by_hand():
    layout = cap.marker_print_layout(6, 150.0, 20.0)
    assert (layout["side_px"], layout["margin_px"], layout["pixels_per_meter"]) == (600, 80, 4000)
    assert cap.marker_print_layout(7, 150.0, 20.0)["side_px"] == 602  # a 5x5 marker has 7 cells: 7 x 86
    phys_png = cap.png_with_physical_size(cap.PNG_SIGNATURE + cap.png_chunk(b"IHDR", b"x" * 13)
                                          + cap.png_chunk(b"pHYs", b"old") + cap.png_chunk(b"IEND", b""), 1234)
    assert [kind for kind, _, _ in png_chunks(phys_png)] == [b"IHDR", b"pHYs", b"IEND"]
    with pytest.raises(ValueError):
        cap.png_with_physical_size(b"not a png", 1)


def test_marker_id_beyond_the_dictionary_exits_2(scene, tmp_path, capsys):
    argv = [*base_args(tmp_path), "--marker-id", "50"]
    with pytest.raises(SystemExit) as info:
        cap.main(argv, source_factory=lambda args, cv2: pytest.fail("must not open a source"))
    assert info.value.code == 2 and "--marker-id" in capsys.readouterr().err


def test_mjpg_codec_runs_and_is_recorded(scene, tmp_path):
    code, _, _ = run_live(scene, tmp_path, "mjpg", "--video-codec", "mjpg", frames=scene.frames[:4])
    assert code == 0
    trace = read_detection_log(tmp_path / "mjpg.jsonl")
    assert trace.provenance["instrument"]["settings"]["video_codec"] == "mjpg"
    assert (tmp_path / "mjpg.raw.mkv").stat().st_size > 0


def test_full_writer_queue_is_noted_and_replay_skips_those_frames(scene, tmp_path):
    frames = scene.frames[:12]
    code, _, _ = run_live(scene, tmp_path, "tight", "--queue-frames", "1", frames=frames)
    paths = cap.output_paths(tmp_path / "tight.jsonl")
    rows = [json.loads(line) for line in paths["frames"].read_text(encoding="utf-8").splitlines()]
    assert code == 0 and len(rows) == 12
    in_video = [r["video_frame_index"] for r in rows if r["video_frame_index"] is not None]
    assert in_video == list(range(len(in_video)))  # no gaps in the video positions
    trace = read_detection_log(paths["log"])
    dropped = 12 - len(in_video)
    # --queue-frames 1 is added after the default 1000, and argparse keeps the last value
    assert ("frames were not saved" in trace.notes) == (dropped > 0)
    replay_out = tmp_path / "tight-replay.jsonl"
    argv = ["--dry-run", str(paths["raw"]), "--frame-times", str(paths["frames"]), "--condition", "test-spot",
            "--distance-m", "1.2", "--marker-length-m", str(MARKER_L), "--calibration", str(tmp_path / "camera.json"),
            "--output", str(replay_out)]
    assert cap.main(argv, ask=refuse_to_ask, log=lambda text: None) == 0
    replay = read_detection_log(replay_out)
    assert len(replay.observations) == len(in_video)
    live_by_id = {o.observation_id: o for o in trace.observations}
    for observation in replay.observations:
        assert live_by_id[observation.observation_id].position_m == observation.position_m


def test_dry_run_with_missing_frame_time_rows_is_a_runtime_error(scene, tmp_path, capsys):
    frames = scene.frames[:6]
    assert run_live(scene, tmp_path, "src", frames=frames)[0] == 0
    paths = cap.output_paths(tmp_path / "src.jsonl")
    short = tmp_path / "short.frames.jsonl"
    short.write_text("\n".join(paths["frames"].read_text(encoding="utf-8").splitlines()[:3]) + "\n", encoding="utf-8")
    argv = ["--dry-run", str(paths["raw"]), "--frame-times", str(short), "--condition", "c", "--distance-m", "1.2",
            "--marker-length-m", str(MARKER_L), "--hfov-deg", "70",
            "--output", str(tmp_path / "bad.jsonl")]
    # the video is 960x540 but an hfov guess builds its model from the video size, so this only fails on the rows
    code = cap.main(argv, ask=refuse_to_ask, log=lambda text: None)
    assert code == 1 and "frame-times" in capsys.readouterr().err


# --- repair round: the writer thread ------------------------------------------

def close_promptly(writer) -> BaseException | None:
    """Call writer.close() in a thread; fail if it hangs. Returns what it raised."""
    outcome: list[BaseException | None] = []

    def target() -> None:
        try:
            writer.close()
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001  handed back to the test
            outcome.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "close() hung"
    return outcome[0]


class ReleaseOnly:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize("failure", ["returns False", "raises KeyError"])
def test_writer_failure_of_any_kind_makes_close_return_promptly_and_raise(failure):
    class Bad(ReleaseOnly):
        def write(self, frame):
            if failure == "returns False":
                return False
            raise KeyError("not an OSError")

    inner = Bad()
    writer = cap.RawVideoWriter(inner, max_frames=2)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in range(50):  # many more than the queue holds: the thread keeps draining after the failure
        writer.submit(frame)
        time.sleep(0.001)
    error = close_promptly(writer)
    assert isinstance(error, cap.CaptureError)
    assert ("returned False" if failure == "returns False" else "not an OSError") in str(error)
    assert inner.released
    assert writer.close() is None  # a second close does nothing


def test_writer_checks_the_frame_count_after_release():
    class Fine(ReleaseOnly):
        def write(self, frame):
            return True

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    good = cap.RawVideoWriter(Fine(), max_frames=10, count_frames=lambda: 3)
    for _ in range(3):
        good.submit(frame)
    assert close_promptly(good) is None
    short = cap.RawVideoWriter(Fine(), max_frames=10, count_frames=lambda: 2)
    for _ in range(3):
        short.submit(frame)
    error = close_promptly(short)
    assert isinstance(error, cap.CaptureError) and "holds 2 frames but 3" in str(error)


def test_failed_atomic_write_leaves_no_tmp_file(tmp_path, monkeypatch):
    header = cap.build_header(live=False, trace_id="t", source="v.mkv", capture=hand_capture(False),
                              clock_domain="shared_monotonic", notes="n", code_revision="unknown", provenance=None)
    paths = cap.output_paths(tmp_path / "out.jsonl")

    def broken_replace(*args, **kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(cap.os, "replace", broken_replace)
    with pytest.raises(OSError):
        cap.write_log_atomically(cap.make_trace(header, hand_observations()), paths)
    assert not paths["tmp"].exists() and not paths["log"].exists()


def test_usage_text_is_a_raw_string_with_plain_line_continuations():
    doc = cap.__doc__
    assert " \\\n" in doc and "\\\\" not in doc  # one backslash, so commands copy straight from the file
    assert "\u2014" not in doc and "\u2013" not in doc
    assert doc.index('mkdir -p "$HOME/zephyr-data/aruco"') < doc.index("--condition 2026-10-24")
    assert doc.index('cd Zephyr-Mesh') < doc.index("tools/capture_aruco.py")
    assert "PYTHONPATH=. .venv/bin/python tools/fit_vision_noise.py" in doc
    assert '--print-marker "$HOME/Desktop/' in doc and "tools/calibrate_charuco.py" in doc
    assert "2026-10-24_d1.0_center_yaw0_r1" in doc and "--check-camera" in doc
    result = subprocess.run([sys.executable, str(cap.ROOT / "tools" / "capture_aruco.py"), "--help"],
                            capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": str(cap.ROOT)})
    assert result.stdout.startswith("usage: capture_aruco.py")
    assert "--check-camera" in result.stdout and "--marker-print-mm" in result.stdout


def test_list_cameras_reads_system_profiler_json_without_running_it():
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"SPCameraDataType": [
            {"_name": "FaceTime HD Camera", "spcamera_model-id": "x"},
            {"_name": "OBS Virtual Camera"}, {"no name": 1}]}))

    assert REAL_LIST_CAMERAS(run=fake_run, platform="darwin") == ["FaceTime HD Camera", "OBS Virtual Camera"]
    assert calls == [["system_profiler", "SPCameraDataType", "-json"]]
    assert REAL_LIST_CAMERAS(run=fake_run, platform="linux") is None and len(calls) == 1

    def broken(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    assert REAL_LIST_CAMERAS(run=broken, platform="darwin") is None
    assert REAL_LIST_CAMERAS(run=lambda c, **k: SimpleNamespace(stdout="not json"), platform="darwin") is None


def test_live_output_inside_the_repo_exits_2_but_a_dry_run_may_write_there(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cap, "PRIVACY_ROOT", cap.ROOT)  # the real guard
    inside = cap.ROOT / "tools" / "zz-privacy-guard-test.jsonl"
    argv = replace_option(base_args(tmp_path), "--output", str(inside))
    expect_exit_2(argv, capsys, "inside the repo")
    assert not inside.exists()
    expect_exit_2([*base_args(tmp_path), "--check-camera"], capsys, "leave out --output")
    # A dry run writes no video, so the same path is accepted (validate_args writes nothing itself).
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    parser = cap.build_parser()
    args = parser.parse_args([*argv, "--dry-run", str(video), "--fps", "30"])
    cap.validate_args(parser, args, "replay")
    assert not inside.exists()



def _case_insensitive_disk(folder: Path) -> bool:
    swapped = folder.parent / folder.name.swapcase()
    return swapped != folder and swapped.exists()


def test_privacy_guard_catches_a_differently_cased_repo_path(tmp_path):
    repo = tmp_path / "Repo-Folder"
    (repo / "tools").mkdir(parents=True)
    if not _case_insensitive_disk(repo):
        pytest.skip("this disk is case-sensitive, so a differently cased path is a different folder")
    assert cap.inside_folder(tmp_path / "REPO-FOLDER" / "tools" / "x.jsonl", repo)
    assert cap.inside_folder(tmp_path / "repo-folder" / "new-dir" / "x.jsonl", repo)
    assert not cap.inside_folder(tmp_path / "other" / "x.jsonl", repo)
    assert not cap.inside_folder(tmp_path / "Repo-Folder-2" / "x.jsonl", repo)

# --- repair round: OpenCV tests -------------------------------------------------

def test_read_error_at_frame_12_keeps_the_frames_before_it_and_exits_1(scene, tmp_path, capsys):
    # Reads 0..2 are warmup, so read 3 + 11 = 14 is the 12th recorded read.
    code, messages, source = run_live(scene, tmp_path, "crash", frames=scene.frames[:30], count=30,
                                      fail_at=WARMUP + 11)
    assert code == 1 and source.released
    err = capsys.readouterr().err
    assert "RuntimeError: camera unplugged" in err
    paths = cap.output_paths(tmp_path / "crash.jsonl")
    trace = read_detection_log(paths["log"])
    assert len(trace.observations) == 11
    assert "ended on an error after 11 frames: RuntimeError: camera unplugged" in trace.notes
    assert len(paths["frames"].read_text(encoding="utf-8").splitlines()) == 11
    assert cap.count_video_frames(scene.cv2, paths["raw"]) == 11  # the video was finished and closed
    assert trace.provenance["raw_sha256"] == hashlib.sha256(paths["raw"].read_bytes()).hexdigest()
    assert any("fit_vision_noise.py" in line for line in messages)


def test_opencv_error_during_detection_also_keeps_the_frames_before_it(scene, tmp_path, capsys, monkeypatch):
    real = cap.analyze_frame
    calls = {"n": 0}

    def flaky(frame, ctx):
        calls["n"] += 1
        if calls["n"] == 5:
            raise scene.cv2.error("detector blew up")
        return real(frame, ctx)

    monkeypatch.setattr(cap, "analyze_frame", flaky)
    code, _, _ = run_live(scene, tmp_path, "cverr", frames=scene.frames[:10])
    assert code == 1 and "detector blew up" in capsys.readouterr().err
    trace = read_detection_log(tmp_path / "cverr.jsonl")
    assert len(trace.observations) == 4 and "ended on an error" in trace.notes


def test_frame_is_saved_after_the_result_is_stamped(scene, tmp_path, monkeypatch):
    order: list[str] = []
    real_process, real_submit = cap.process_frame, cap.RawVideoWriter.submit

    def process(*args, **kwargs):
        order.append("process")
        return real_process(*args, **kwargs)

    def submit(self, frame):
        order.append("submit")
        return real_submit(self, frame)

    monkeypatch.setattr(cap, "process_frame", process)
    monkeypatch.setattr(cap.RawVideoWriter, "submit", submit)
    assert run_live(scene, tmp_path, "order", frames=scene.frames[:3])[0] == 0
    assert order == ["process", "submit"] * 3
    rows = [json.loads(line) for line in (tmp_path / "order.frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["video_frame_index"] for r in rows] == [0, 1, 2]  # the row is written after the frame is queued


def test_camera_size_comes_from_the_frames_not_from_what_the_backend_reports(scene, tmp_path):
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.warmup + scene.frames[:4], (1280, 720), clock=clock)  # get() is wrong
    argv = [*live_argv(tmp_path, "size", "--hfov-deg", str(HFOV_DEG)), "--frames", "4"]
    assert cap.main(argv, source_factory=lambda args, cv2: source, clock=clock, log=lambda text: None) == 0
    trace = read_detection_log(tmp_path / "size.jsonl")
    assert trace.capture["resolution_px"] == {"width": WIDTH, "height": HEIGHT}
    assert cap.count_video_frames(scene.cv2, tmp_path / "size.raw.mkv") == 4
    settings = trace.provenance["instrument"]["settings"]
    assert (settings["actual_width_px"], settings["requested_width_px"]) == (WIDTH, 1280)


def test_frame_size_change_mid_run_names_the_hfov_guess(scene, tmp_path, capsys):
    bigger = np.zeros((HEIGHT + 2, WIDTH, 3), dtype=np.uint8)
    frames = [*scene.frames[:3], bigger, *scene.frames[3:6]]
    code, _, _ = run_live(scene, tmp_path, "resize", "--hfov-deg", str(HFOV_DEG), frames=frames,
                          with_calibration=False)
    assert code == 1
    assert "--hfov-deg focal length was worked out for 960x540" in capsys.readouterr().err
    assert len(read_detection_log(tmp_path / "resize.jsonl").observations) == 3
    camera = geo.camera_from_hfov(WIDTH, HEIGHT, HFOV_DEG)
    calibrated = geo.CameraModel(camera.camera_matrix, camera.dist_coeffs, WIDTH, HEIGHT, "calibration_file", "c")
    assert "--calibration file is for 960x540" in cap.resolution_problem(calibrated, 1280, 720)
    assert cap.resolution_problem(calibrated, WIDTH, HEIGHT) is None


def test_few_detections_print_a_warning(scene, tmp_path):
    frames = [_blank_frame(np.random.default_rng(k)) for k in range(6)] + [scene.frames[0]]
    frames[-1] = next(f for f, blank in zip(scene.frames, scene.blank) if not blank)
    code, messages, _ = run_live(scene, tmp_path, "few", frames=frames)
    assert code == 0
    warning = [line for line in messages if line.startswith("WARNING: the marker was found in only 1 of 7")]
    assert warning and "white margin" in warning[0] and "--overwrite" in warning[0]


def test_gaps_in_the_capture_times_are_counted_and_warned_about(scene, tmp_path):
    code, messages, _ = run_live(scene, tmp_path, "gap", frames=scene.frames[:12], gap_at=WARMUP + 6)
    assert code == 0
    trace = read_detection_log(tmp_path / "gap.jsonl")
    assert "1 gaps longer than 1.5 x the median interval" in trace.notes
    assert any(line.startswith("WARNING: 1 gaps longer") for line in messages)


def test_dry_run_notes_frame_time_rows_past_the_end_of_the_video(scene, tmp_path):
    assert run_live(scene, tmp_path, "src", frames=scene.frames[:5])[0] == 0
    paths = cap.output_paths(tmp_path / "src.jsonl")
    rows = paths["frames"].read_text(encoding="utf-8").splitlines()
    extra = [json.dumps({"frame_index": 5 + k, "capture_time_s": 1.0 + k, "video_frame_index": 5 + k}) for k in range(3)]
    longer = tmp_path / "longer.frames.jsonl"
    longer.write_text("\n".join(rows + extra) + "\n", encoding="utf-8")
    out = tmp_path / "replay.jsonl"
    argv = ["--dry-run", str(paths["raw"]), "--frame-times", str(longer), "--condition", "c", "--distance-m", "1.2",
            "--marker-length-m", str(MARKER_L), "--calibration", str(tmp_path / "camera.json"), "--output", str(out)]
    assert cap.main(argv, ask=refuse_to_ask, log=lambda text: None) == 0
    assert "3 rows of that file are past the end of the video" in read_detection_log(out).notes


def test_camera_question_and_enter_prompt_and_countdown(scene, tmp_path):
    calibration = tmp_path / "camera.json"
    write_calibration(calibration, scene.camera)
    stream = scene.frames[:40]  # every frame differs (its own jitter), so we can tell which ones were logged
    argv = [piece for piece in live_argv(tmp_path, "asked", "--calibration", str(calibration))
            if piece != "--yes"]
    argv = replace_option(argv, "--countdown-s", "0.2") + ["--frames", "5"]

    def run(answers):
        prompts: list[str] = []
        replies = iter(answers)

        def ask(prompt):
            prompts.append(prompt)
            return next(replies)

        clock = FakeClock()
        source = FakeSource(scene.cv2, scene.warmup + list(stream), (WIDTH, HEIGHT), clock=clock)
        messages: list[str] = []
        code = cap.main(argv, source_factory=lambda args, cv2: source, clock=clock, ask=ask, log=messages.append)
        return code, prompts, source, messages

    # "n": stop before anything is written, exit 2.
    code, prompts, source, messages = run(["n"])
    assert code == 2 and len(prompts) == 1 and "Mac's own camera" in prompts[0] and "960x540" in prompts[0]
    assert not (tmp_path / "asked.frames.jsonl").exists() and not (tmp_path / "asked.raw.mkv").exists()
    assert source.released

    # "y", then Enter: count down 0.2 s while throwing frames away, then record 5.
    code, prompts, source, messages = run(["y", ""])
    assert code == 0 and prompts[1] == "Press Enter, then step away. "
    trace = read_detection_log(tmp_path / "asked.jsonl")
    settings = trace.provenance["instrument"]["settings"]
    counted = settings["countdown_frames"]
    assert 5 <= counted <= 7  # 0.2 s at 30 fps
    assert source.index == WARMUP + counted + 5
    assert settings["camera_confirmed"] == "yes, answered at the prompt"
    assert f"{WARMUP} warmup frames and {counted} countdown frames" in trace.notes
    # The logged frames are exactly the 5 that came after the countdown frames.
    recorded = stream[counted:counted + 5]
    direct = cap.build_context(scene.cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                               corner_refinement="subpix", camera=scene.camera, truth=cap.Truth(NOMINAL, 0.01, "t"))
    for k, observation in enumerate(trace.observations):
        expected, _ = cap.process_frame(recorded[k], k, 0.0, direct)
        assert observation.outcome == expected["outcome"]
        assert observation.position_m == (tuple(expected["position_m"]) if "position_m" in expected else None)
    assert any(line.strip() == "1" for line in messages)  # the countdown is shown


def test_per_axis_truth_uncertainty_range_and_notes(scene, tmp_path):
    code, messages, _ = run_live(scene, tmp_path, "xyz", "--truth-uncertainty-xyz-m", "0.002", "0.003", "0.0034",
                                 "--range-m", "1.2015", "--notes", "2 shims, lid 90 deg", frames=scene.frames[:8])
    assert code == 0
    trace = read_detection_log(tmp_path / "xyz.jsonl")
    ground_truth = trace.provenance["ground_truth"]
    assert ground_truth["uncertainty"] == {"x_m": 0.002, "y_m": 0.003, "z_m": 0.0034}
    assert ground_truth["values"]["range_m"] == 1.2015
    assert all(o.ground_truth["uncertainty_m"] == 0.0034 for o in trace.observations)  # the largest of the three
    assert "Operator notes: 2 shims, lid 90 deg" in trace.notes
    stats = cap.position_stats([{"outcome": o.outcome, "position_m": o.position_m} for o in trace.observations])
    difference = stats["range_mean_mm"] - 1201.5
    assert (f"Range: measured mean {stats['range_mean_mm']:.1f} mm, taped 1201.5 mm, difference {difference:+.1f} mm "
            f"({difference / 1201.5 * 100:+.2f}%).") in messages


def test_both_truth_uncertainty_options_exit_2(tmp_path, capsys):
    expect_exit_2([*base_args(tmp_path), "--truth-uncertainty-m", "0.01", "--truth-uncertainty-xyz-m", "0.1", "0.1",
                   "0.1"], capsys, "not both")


def test_video_codec_none_hashes_the_frames_file(scene, tmp_path):
    code, _, _ = run_live(scene, tmp_path, "novideo", "--video-codec", "none", frames=scene.frames[:5])
    assert code == 0
    paths = cap.output_paths(tmp_path / "novideo.jsonl")
    assert not paths["raw"].exists()
    trace = read_detection_log(paths["log"])
    frames_sha = hashlib.sha256(paths["frames"].read_bytes()).hexdigest()
    assert trace.provenance["raw_sha256"] == frames_sha
    assert "raw_sha256 is the SHA-256 of the frames file" in trace.notes
    rows = [json.loads(line) for line in paths["frames"].read_text(encoding="utf-8").splitlines()]
    assert all(r["video_frame_index"] is None for r in rows)


def test_check_readout_by_hand(scene):
    pose = geo.MarkerPose(rvec=(3.1, 0.0, 0.0), tvec=(0.003, -0.004, 1.2), reprojection_error_px=0.1,
                          ambiguity_ratio=None)
    cx, cy = scene.camera.matrix()[0, 2], scene.camera.matrix()[1, 2]
    result = cap.FrameResult("detected", None, (0,), CORNERS, (cx + 2.5, cy - 1.25), pose, None)
    assert cap.check_readout(result, scene.camera) == [
        "x +3.0  y -4.0  z 1200.0 mm   range 1200.0 mm",  # sqrt(3^2 + 4^2 + 1200^2) = 1200.0104
        "marker center from crosshair: u-cx +2.5 px, v-cy -1.2 px"]
    assert cap.check_readout(cap.FrameResult("missed", "marker_not_found", ()), scene.camera) == ["marker not found"]


def test_check_camera_loop_draws_crosshair_on_a_copy_and_reports_timing(scene, tmp_path):
    cv2 = scene.cv2
    ctx = cap.build_context(cv2, dictionary_name="DICT_4X4_50", marker_id=0, marker_length_m=MARKER_L,
                            corner_refinement="subpix", camera=scene.camera, truth=cap.Truth(NOMINAL, 0.01, "t"))
    shown = []

    class DrawingPreview:
        def show(self, frame, corners_px, text, crosshair=None):
            before = frame.copy()
            shown.append((cap.draw_overlay(cv2, frame, corners_px, text, crosshair), crosshair, list(text)))
            assert np.array_equal(frame, before)
            return False

    clock = FakeClock()
    source = FakeSource(cv2, scene.frames[:40], (WIDTH, HEIGHT), clock=clock, gap_at=10)
    result = cap.check_camera_loop(source, ctx, clock=clock, preview=DrawingPreview(), log=lambda text: None,
                                   seconds=0.5)
    assert 12 <= len(result.capture_times_s) <= 15  # 0.5 s at 30 fps is 15 intervals, one of them 3 x long
    timing = geo.summarize_frame_timing(result.capture_times_s, None)
    assert timing["late_frames"] == 1 and timing["measured_frame_rate_hz"] == pytest.approx(FPS, rel=0.02)
    image, crosshair, text = shown[-1]
    assert crosshair == (WIDTH / 2, HEIGHT / 2)
    assert tuple(image[HEIGHT - 5, WIDTH // 2]) == (255, 255, 0) and tuple(image[HEIGHT // 2, WIDTH - 5]) == (255, 255, 0)
    assert any(line.startswith("x ") or line == "marker not found" for line in text)
    assert result.frame_shape == (HEIGHT, WIDTH, 3)


def test_check_camera_mode_prints_size_rate_and_gaps_and_writes_nothing(scene, tmp_path):
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.frames[:60], (1280, 720), clock=clock, gap_at=20)
    messages: list[str] = []
    before = set(tmp_path.iterdir())
    argv = ["--check-camera", "--check-seconds", "1", "--marker-length-m", str(MARKER_L), "--hfov-deg",
            str(HFOV_DEG), "--no-preview", "--yes"]
    assert cap.main(argv, source_factory=lambda args, cv2: source, clock=clock, log=messages.append) == 0
    assert set(tmp_path.iterdir()) == before and source.released
    assert "Frame size read from the frames: 960x540. Requested: 1280x720." in messages
    assert any("different size than requested" in line for line in messages)
    rate_line = next(line for line in messages if line.startswith("Frames: "))
    rate = float(re.search(r"measured frame rate: ([0-9.]+) Hz", rate_line).group(1))
    assert rate == pytest.approx(FPS, rel=0.02)  # the fake clock also ticks 0.2 ms per reading
    assert rate_line.endswith("gaps (longer than 1.5 x the median interval): 1.")
    assert any(line.startswith("Cameras macOS lists: FaceTime HD Camera") for line in messages)
    # Without --yes it asks afterwards; "n" exits 2.
    clock = FakeClock()
    source = FakeSource(scene.cv2, scene.frames[:60], (WIDTH, HEIGHT), clock=clock)
    asked = []
    code = cap.main([a for a in argv if a != "--yes"], source_factory=lambda args, cv2: source, clock=clock,
                    log=lambda text: None, ask=lambda prompt: asked.append(prompt) or "n")
    assert code == 2 and "Mac's own camera" in asked[0]
