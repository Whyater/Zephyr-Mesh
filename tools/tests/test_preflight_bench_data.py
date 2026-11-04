"""Tests for tools/preflight_bench_data.py (no OpenCV, no camera, no radio).

Every bench folder here is built by the helpers below from known settings, so each test can break exactly one
thing and check that the preflight names it. The good folder copies the runbook layout (aruco/, latency/,
radio/) with measured headers. All values in it are made up; none is a measurement.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sim.detection_log import write_detection_log
from sim.vision_trace import VISION_TRACE_SCHEMA, VisionTrace
from tools import aruco_processing_time as processing_tool
from tools import preflight_bench_data as tool
from tools.make_espnow_echo_fixture import EchoParams, write_fixture

CAPTURED_AT = "2026-10-30T14:00:00-06:00"
REVISION = "77be94f0000000000000000000000000000000ab"
LOG_1M = "2026-10-30_d1.0_center_yaw0_r1"
LOG_2M = "2026-10-30_d2.0_side_yaw0_r1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_calibration(path: Path, *, width: int = 1280, height: int = 720, fx: float = 900.0,
                      rms: float = 0.3) -> str:
    doc = {"schema": "zephyr-camera-calibration-1",
           "camera_matrix": [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]],
           "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0], "image_size_px": {"width": width, "height": height},
           "rms_reprojection_error_px": rms}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return _sha(path)


def write_log(folder: Path, name: str, truth: tuple[float, float, float], *, frames: int = 300,
              detected_per_10: int = 9, status: str = "measured", calibration: Path | None = None,
              code_revision: str = REVISION, resolution: tuple[int, int] = (1280, 720), scale: float = 1.0,
              mixed_frames: bool = False, raw_video: bool = True, captured_at: str = CAPTURED_AT,
              subjects: int = 1) -> Path:
    """A detection log plus its frames file and raw video, the way tools/capture_aruco.py lays them out."""
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{name}.jsonl"
    frames_path = folder / f"{name}.frames.jsonl"
    frames_path.write_text("".join(json.dumps({"frame_index": i, "capture_time_s": i / 30}) + "\n"
                                   for i in range(frames)), encoding="utf-8", newline="\n")
    frames_sha = _sha(frames_path)
    raw_sha = frames_sha
    if raw_video:
        raw = folder / f"{name}.raw.mkv"
        raw.write_bytes(f"stand-in video bytes for {name}".encode())
        raw_sha = _sha(raw)
    rng = np.random.default_rng(len(name))
    observations = []
    for i in range(frames):
        gt = {"position_m": list(truth), "uncertainty_m": 0.01, "method": "tape"}
        obs = {"observation_id": f"obs-{i + 1:06d}", "frame_id": f"frame-{i + 1:06d}",
               "subject_id": f"marker-{i % subjects}", "capture_time_s": i / 30,
               "processing_time_s": i / 30 + 0.012, "ground_truth": gt}
        if i % 10 < detected_per_10:
            noise = rng.normal(0.0, 0.003, 3)
            obs.update(outcome="detected", position_m=[scale * t + n for t, n in zip(truth, noise)],
                       position_frame="world" if mixed_frames and i % 2 else "camera")
        else:
            obs.update(outcome="missed", miss_reason="not_found")
        observations.append(obs)
    header = {"trace_id": name, "status": status, "clock_domain": "shared_monotonic", "source": log.name,
              "capture": {"source_type": "camera", "source_id": "test-camera", "device_ids": ["laptop"],
                          "model": "test-detector",
                          "resolution_px": {"width": resolution[0], "height": resolution[1]},
                          "frame_rate_hz": 30.0},
              "code_revision": code_revision,
              "notes": f"Test log. The frames file {frames_path.name} has SHA-256 {frames_sha}."}
    if status == "measured":
        cal_ref = None if calibration is None else _sha(calibration)
        if calibration is not None:
            header["capture"]["calibration_id"] = calibration.name
        header["provenance"] = {
            "instrument": {"kind": "camera", "model": "test-camera", "settings": {}},
            "calibration_ref": cal_ref, "raw_sha256": raw_sha, "code_revision": code_revision,
            "captured_at": captured_at, "operator": "Test Operator",
            "ground_truth": {"method": "tape", "values": {"z_m": truth[2]}, "uncertainty": {"z_m": 0.01}}}
        if cal_ref is None:
            header["provenance"]["calibration_ref_null_reason"] = "field of view guess"
    trace = VisionTrace.from_dict({"schema": VISION_TRACE_SCHEMA, **header, "observations": observations})
    write_detection_log(trace, log)
    return log


def write_processing(folder: Path, *, status: str = "measured", name: str = "processing_time.json") -> Path:
    timings = [0.010 + 0.0001 * i for i in range(50)]
    report = processing_tool.build_report(
        timings, status=status, cv2_version="test", dictionary="DICT_4X4_50", machine={}, source={},
        warmup_frames=10, operator="Test Operator", captured_at=CAPTURED_AT, code_revision=REVISION)
    out = folder / name
    processing_tool.write_report(out, report, timings)
    return out


def write_latency_run(run: Path, *, status: str = "measured", frames: int = 20, camera_error: str | None = None,
                      t_read: int | None = None) -> Path:
    run.mkdir(parents=True)
    np.savez_compressed(run / "frames.npz", frames=np.zeros((frames, 4, 4), dtype=np.uint8),
                        t_read_s=np.arange(frames if t_read is None else t_read) / 30.0)
    (run / "display_log.csv").write_text(("seq,code,t_shown_s,t_drawn_s\n0,0,0.100000,0.100000\n"
                                          "1,1,0.200000,0.200000\n"),
                                         encoding="utf-8", newline="\n")
    meta = {"schema": "zephyr-vision-latency-run-1", "status": status, "n_bits": 8, "frames": frames,
            "opencv_version": "test", "platform": "test-platform", "clock": "time.monotonic",
            "camera_error": camera_error}
    (run / "run_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8", newline="\n")
    return run


def build_good(root: Path) -> Path:
    """A measured bench folder in the runbook layout that should pass every check."""
    aruco = root / "aruco"
    aruco.mkdir(parents=True)
    calibration = aruco / "calibration-1280x720.json"
    write_calibration(calibration)
    write_log(aruco, LOG_1M, (0.0, 0.0, 1.0), calibration=calibration)
    write_log(aruco, LOG_2M, (0.3, 0.0, 2.0), calibration=calibration)
    write_processing(aruco)
    for name in ("run1", "run2"):
        write_latency_run(root / "latency" / name)
    write_fixture(root / "radio", EchoParams(), 20261020)
    return root


def run(folder: Path, *args: str) -> tool.Report:
    return tool.preflight(folder, **dict(_options(args)))


def _options(args):
    it = iter(args)
    for key in it:
        yield key, next(it)


def codes(report: tool.Report, level: str = "FAIL") -> set[str]:
    return {f.code for f in report.findings if f.level == level}


def assert_fails(report: tool.Report, code: str) -> None:
    found = [f for f in report.findings if f.level == "FAIL" and f.code == code]
    assert found, f"expected FAIL {code}, got {[(f.level, f.code, f.reason) for f in report.findings]}"
    for finding in found:
        assert finding.reason and "\n" not in finding.reason, "the reason must be one plain line"
        assert finding.fix and "\n" not in finding.fix, "every failure needs a one-line fix hint"


@pytest.fixture
def good(tmp_path: Path) -> Path:
    return build_good(tmp_path / "bench")


# ---------------------------------------------------------------- the good folder

def test_good_folder_passes(good: Path, capsys):
    report = tool.preflight(good)
    assert not codes(report), [(f.code, f.reason) for f in report.findings]
    assert report.counts["detection logs"] == 2
    assert report.counts["latency runs"] == 2
    assert report.counts["radio pairs"] == 1
    assert tool.main([str(good)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL" not in out


def test_good_folder_with_new_output_and_frame_size_passes(good: Path, tmp_path: Path):
    report = tool.preflight(good, output=tmp_path / "intake-new", frame_size=(1280, 720))
    assert not codes(report)


def test_cli_exit_codes(good: Path, tmp_path: Path, capsys):
    assert tool.main([str(tmp_path / "missing")]) == 2
    assert tool.main([str(good), "--frame-size", "1280by720"]) == 2
    (good / "aruco" / f"{LOG_1M}.raw.mkv").write_bytes(b"changed")
    assert tool.main([str(good)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "fix:" in out and "Do not run the intake" in out


def test_intake_output_folders_are_skipped(good: Path):
    intake = good / "out" / "intake-pilot"
    write_log(intake / "fits", "copy-of-a-log", (0.0, 0.0, 1.0), status="synthetic")
    (intake / "INTAKE.md").write_text("# Vision intake\n", encoding="utf-8")
    report = tool.preflight(good)
    assert not codes(report)
    assert report.counts["detection logs"] == 2


# ---------------------------------------------------------------- detection logs

def test_empty_folder_fails(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert_fails(tool.preflight(tmp_path / "empty"), "no-bench-files")


def test_crlf_log_is_rejected_by_the_reader(good: Path):
    log = good / "aruco" / f"{LOG_1M}.jsonl"
    log.write_bytes(log.read_bytes().replace(b"\n", b"\r\n"))
    assert_fails(tool.preflight(good), "log-unreadable")


def test_measured_log_without_provenance_fails(good: Path):
    log = good / "aruco" / f"{LOG_1M}.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    header = json.loads(lines[0])
    del header["provenance"]
    lines[0] = json.dumps(header) + "\n"
    log.write_text("".join(lines), encoding="utf-8", newline="\n")
    assert_fails(tool.preflight(good), "missing-provenance")


def test_low_detection_rate_fails(good: Path):
    write_log(good / "aruco", LOG_1M, (0.0, 0.0, 1.0), detected_per_10=4,
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "low-detection-rate")


def test_too_few_detections_with_truth_fails(good: Path):
    write_log(good / "aruco", LOG_1M, (0.0, 0.0, 1.0), frames=1,
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "too-few-detections")


def test_mixed_position_frames_fail(good: Path):
    write_log(good / "aruco", LOG_1M, (0.0, 0.0, 1.0), mixed_frames=True,
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "mixed-position-frames")


def test_name_distance_disagreeing_with_truth_fails(good: Path):
    # A stale DIST variable: the file says 2.0 m but every line's truth says 1.0 m.
    write_log(good / "aruco", LOG_2M, (0.3, 0.0, 1.0), calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "name-truth-mismatch")


def test_center_name_with_sideways_truth_fails(good: Path):
    write_log(good / "aruco", LOG_1M, (0.2, 0.0, 1.0), calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "name-truth-mismatch")


def test_side_name_with_center_truth_fails(good: Path):
    write_log(good / "aruco", LOG_2M, (0.0, 0.0, 2.0), calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "name-truth-mismatch")


def test_detected_distance_far_from_truth_fails(good: Path):
    write_log(good / "aruco", LOG_1M, (0.0, 0.0, 1.0), scale=1.2,
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "distance-off")


def test_raw_video_hash_mismatch_fails(good: Path):
    (good / "aruco" / f"{LOG_1M}.raw.mkv").write_bytes(b"a different video")
    assert_fails(tool.preflight(good), "raw-sha256-mismatch")


def test_missing_raw_video_fails(good: Path):
    (good / "aruco" / f"{LOG_1M}.raw.mkv").unlink()
    assert_fails(tool.preflight(good), "raw-sha256-mismatch")


def test_no_video_run_uses_the_frames_file_hash(good: Path):
    write_log(good / "aruco", LOG_1M, (0.0, 0.0, 1.0), raw_video=False,
              calibration=good / "aruco" / "calibration-1280x720.json")
    (good / "aruco" / f"{LOG_1M}.raw.mkv").unlink()
    assert not codes(tool.preflight(good))


def test_frames_file_changed_after_capture_fails(good: Path):
    frames = good / "aruco" / f"{LOG_1M}.frames.jsonl"
    frames.write_text(frames.read_text(encoding="utf-8") + "{}\n", encoding="utf-8", newline="\n")
    assert_fails(tool.preflight(good), "frames-sha256-mismatch")


def test_dirty_code_revision_warns_only(good: Path):
    for name, truth in ((LOG_1M, (0.0, 0.0, 1.0)), (LOG_2M, (0.3, 0.0, 2.0))):
        write_log(good / "aruco", name, truth, code_revision=REVISION + "-dirty",
                  calibration=good / "aruco" / "calibration-1280x720.json")
    report = tool.preflight(good)
    assert not codes(report)
    assert "dirty-code-revision" in codes(report, "WARN")


def test_old_fit_next_to_a_log_warns(good: Path):
    (good / "aruco" / f"{LOG_1M}.fit.json").write_text("{}\n", encoding="utf-8")
    report = tool.preflight(good)
    assert not codes(report)
    assert "fit-would-be-replaced" in codes(report, "WARN")


# ---------------------------------------------------------------- calibration and frame size

def test_unreadable_calibration_fails(good: Path):
    (good / "aruco" / "calibration-extra.json").write_text(
        json.dumps({"schema": "zephyr-camera-calibration-1", "camera_matrix": [[1]]}), encoding="utf-8")
    assert_fails(tool.preflight(good), "calibration-unreadable")


def test_calibration_changed_after_capture_fails(good: Path):
    write_calibration(good / "aruco" / "calibration-1280x720.json", fx=950.0)
    assert_fails(tool.preflight(good), "calibration-sha256-mismatch")


def test_log_frame_size_differs_from_its_calibration(good: Path):
    aruco = good / "aruco"
    calibration = aruco / "calibration-1280x720.json"
    write_calibration(calibration, width=1920, height=1080)
    for name, truth in ((LOG_1M, (0.0, 0.0, 1.0)), (LOG_2M, (0.3, 0.0, 2.0))):
        write_log(aruco, name, truth, calibration=calibration)
    assert_fails(tool.preflight(good), "frame-size-mismatch")


def test_expected_frame_size_flag(good: Path):
    assert_fails(tool.preflight(good, frame_size=(1920, 1080)), "frame-size-mismatch")


def test_high_calibration_rms_warns(good: Path):
    aruco = good / "aruco"
    calibration = aruco / "calibration-1280x720.json"
    write_calibration(calibration, rms=0.9)
    for name, truth in ((LOG_1M, (0.0, 0.0, 1.0)), (LOG_2M, (0.3, 0.0, 2.0))):
        write_log(aruco, name, truth, calibration=calibration)
    report = tool.preflight(good)
    assert not codes(report)
    assert "calibration-rms-high" in codes(report, "WARN")


# ---------------------------------------------------------------- processing time

def test_measured_processing_without_provenance_fails(good: Path):
    path = good / "aruco" / "processing_time.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["provenance"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert_fails(tool.preflight(good), "processing-refused")


def test_processing_timings_hash_mismatch_fails(good: Path):
    csv = good / "aruco" / "processing_time.timings.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace("0.010000000", "0.010000001"), encoding="utf-8",
                   newline="\n")
    assert_fails(tool.preflight(good), "processing-check-failed")


def test_two_processing_files_fail(good: Path):
    write_processing(good / "aruco", name="processing_time_b.json")
    assert_fails(tool.preflight(good), "several-processing-files")


# ---------------------------------------------------------------- latency runs

def test_latency_run_without_display_log_fails(good: Path):
    (good / "latency" / "run1" / "display_log.csv").unlink()
    assert_fails(tool.preflight(good), "latency-run-refused")


def test_measured_latency_run_without_recorder_provenance_fails(good: Path):
    meta_path = good / "latency" / "run1" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["platform"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert_fails(tool.preflight(good), "latency-run-refused")


def test_latency_run_with_camera_error_fails(good: Path):
    shutil.rmtree(good / "latency" / "run2")
    write_latency_run(good / "latency" / "run2", camera_error="camera stopped sending frames")
    assert_fails(tool.preflight(good), "latency-run-bad")


def test_latency_run_with_no_frames_fails(good: Path):
    shutil.rmtree(good / "latency" / "run2")
    write_latency_run(good / "latency" / "run2", frames=0)
    assert_fails(tool.preflight(good), "latency-run-bad")


def test_latency_run_with_unequal_lengths_fails(good: Path):
    shutil.rmtree(good / "latency" / "run2")
    write_latency_run(good / "latency" / "run2", t_read=5)
    assert_fails(tool.preflight(good), "latency-run-bad")


# ---------------------------------------------------------------- radio

def test_bad_radio_log_fails(good: Path):
    sender = good / "radio" / "sender.csv"
    lines = sender.read_text(encoding="utf-8").splitlines(keepends=True)
    del lines[20]  # a lost serial line: seq jumps
    sender.write_text("".join(lines), encoding="utf-8", newline="\n")
    assert_fails(tool.preflight(good), "radio-log-refused")


def test_two_senders_in_one_folder_fail(good: Path, tmp_path: Path):
    write_fixture(tmp_path / "other-test", EchoParams(), 7)
    shutil.copy(tmp_path / "other-test" / "sender.csv", good / "radio" / "sender-b.csv")
    assert_fails(tool.preflight(good), "radio-pairing")


def test_raw_capture_next_to_its_clean_log_counts_once(good: Path):
    clean = (good / "radio" / "sender.csv").read_bytes()
    (good / "radio" / "sender.raw").write_bytes(b"ets Jun  8 2016 00:22:57\nboot text\n" + clean)
    report = tool.preflight(good)
    assert not codes(report), [(f.code, f.reason) for f in report.findings]
    assert report.counts["radio pairs"] == 1


def test_sender_without_echoer_fails(good: Path):
    (good / "radio" / "echoer.csv").unlink()
    assert_fails(tool.preflight(good), "radio-pairing")


# ---------------------------------------------------------------- mixed sessions and collisions

def test_mixed_statuses_fail(good: Path):
    write_log(good / "aruco", "2026-10-30_d1.0_center_yaw0_r1-replay", (0.0, 0.0, 1.0), status="fixture")
    assert_fails(tool.preflight(good), "mixed-status")


def test_logs_from_two_code_revisions_fail(good: Path):
    write_log(good / "aruco", LOG_2M, (0.3, 0.0, 2.0), code_revision="0" * 40,
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "mixed-sessions")


def test_logs_from_two_calibrations_fail(good: Path):
    other = good / "aruco" / "calibration-older.json"
    write_calibration(other, fx=880.0)
    write_log(good / "aruco", LOG_2M, (0.3, 0.0, 2.0), calibration=other)
    assert_fails(tool.preflight(good), "mixed-sessions")


def test_capture_times_far_apart_warn(good: Path):
    write_log(good / "aruco", LOG_2M, (0.3, 0.0, 2.0), captured_at="2026-11-02T14:00:00-06:00",
              calibration=good / "aruco" / "calibration-1280x720.json")
    report = tool.preflight(good)
    assert not codes(report)
    assert "capture-times-far-apart" in codes(report, "WARN")


def test_two_logs_with_the_same_name_fail(good: Path):
    write_log(good / "aruco" / "day2", LOG_1M, (0.0, 0.0, 1.0),
              calibration=good / "aruco" / "calibration-1280x720.json")
    assert_fails(tool.preflight(good), "name-collision")


def test_two_latency_runs_with_the_same_name_fail(good: Path):
    write_latency_run(good / "latency" / "day2" / "run1")
    assert_fails(tool.preflight(good), "name-collision")


def test_same_log_twice_fails(good: Path):
    aruco = good / "aruco"
    for suffix in (".jsonl", ".frames.jsonl", ".raw.mkv"):
        shutil.copy(aruco / f"{LOG_1M}{suffix}", aruco / f"{LOG_1M}-copy{suffix}")
    assert_fails(tool.preflight(good), "duplicate-file")


def test_output_folder_not_empty_fails(good: Path, tmp_path: Path):
    out = tmp_path / "intake-old"
    out.mkdir()
    (out / "INTAKE.md").write_text("old\n", encoding="utf-8")
    assert_fails(tool.preflight(good, output=out), "output-not-usable")


def test_output_folder_under_sim_runs_fails(good: Path):
    assert_fails(tool.preflight(good, output=tool.REPO_ROOT / "sim" / "runs" / "preflight-test"),
                 "output-not-usable")


def test_output_folder_holding_the_inputs_fails(good: Path):
    assert_fails(tool.preflight(good, output=good), "output-not-usable")


# ---------------------------------------------------------------- the synthetic rehearsal inputs

def test_rehearsal_inputs_pass(tmp_path: Path):
    from tools import rehearse_measured_inputs as rehearsal

    work = tmp_path / "rehearsal"
    work.mkdir()
    inputs = rehearsal.build_inputs(work)
    report = tool.preflight(inputs["log"].parent)
    assert not codes(report), [(f.code, f.reason) for f in report.findings]
    assert report.counts["detection logs"] == 1
    assert report.counts["latency runs"] == 1
    assert report.counts["radio pairs"] == 1
