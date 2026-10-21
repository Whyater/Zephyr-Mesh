#!/usr/bin/env python3
r"""Record a printed ArUco marker with the Mac's camera and write a detection log.

The marker stays still. For every frame the script finds the marker, works out
where it is in meters, and writes a log that tools/fit_vision_noise.py turns
into noise settings for the sim.

STATUS: a proposal. The camera code has only run on synthetic frames, never on
a real camera. It says nothing about any real camera until you record data.

Every command below runs from the repo folder:

    cd "$HOME/Desktop/Zephyr Mesh/Zephyr-Mesh"

Set up a camera environment (once)
----------------------------------
The repo's .venv has no OpenCV and should stay that way. Make a second
environment just for the camera. With uv:

    uv venv -p 3.13 ~/zephyr-camera-venv
    uv pip install -p ~/zephyr-camera-venv/bin/python "opencv-python==5.0.0.93" "numpy==2.5.4"

Without uv you need Python 3.12 or newer (check with python3 --version):

    python3 -m venv ~/zephyr-camera-venv
    ~/zephyr-camera-venv/bin/pip install "opencv-python==5.0.0.93" "numpy==2.5.4"

Use "opencv-python", not "opencv-python-headless". Only the first one can open
the preview window. The camera venv is for capture. The repo's .venv is for
the fit (Step 5).

The first time the script opens the camera, macOS asks if the terminal app
may use it. Click OK. If you clicked "Don't Allow", turn it back on in System
Settings > Privacy & Security > Camera.

Make the data folder once, before the first capture:

    mkdir -p "$HOME/Desktop/Zephyr Mesh/Data/aruco"

Step 1: print the marker
------------------------
    ~/zephyr-camera-venv/bin/python tools/capture_aruco.py \
        --print-marker "$HOME/Desktop/aruco-marker-id0.png"

This writes marker id 0 (DICT_4X4_50) sized so the black square prints
150 mm wide, with a 20 mm white margin. Print it at 100% (actual size) and
turn off "fit to page". The black square means the outer edge of the black
frame, not the white margin. Measure it on both edges with a ruler. If it is
not 150 mm, use what you measured, in meters (for example 0.149), for
--marker-length-m. Keep the white margin; the detector needs it. Glue the
paper flat to stiff cardboard. Matte paper is best.

Step 2: tell the script the focal length
----------------------------------------
Give exactly one of these:

  --calibration FILE.json   the good option. Make the file with
        tools/calibrate_charuco.py (run it with --help). Its frame size must
        match the frames. Its hash goes in the log.
  --hfov-deg DEGREES        a guess of the horizontal field of view. Every
        distance scales with the guess, and so does the noise.

The 70 in the examples is a placeholder, not a measured value. To correct a
guess, put the marker at a taped distance and compare. Example: with 70
degrees, z reads 1.65 m but the tape says 1.50 m. tan(35 deg) = 0.700, times
1.65 / 1.50 = 1.10 gives 0.770, and 2 x atan(0.770) is about 75.2 degrees.

Step 3: check the camera first (writes no files)
------------------------------------------------
    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/capture_aruco.py \
        --check-camera --marker-length-m 0.150 --hfov-deg 70

For lining up the taped axis line, use --calibration FILE instead of
--hfov-deg, so the crosshair sits at the calibrated image center.

The preview shows a crosshair at the image center the camera model uses
(cx, cy), the marker outline, and live numbers: x, y, z and range in mm, and
how far the marker center is from the crosshair in pixels. Use it to line up
the marker. Press q to stop, or add --check-seconds 10. Then it prints the
real frame size, the frame rate and the number of gaps.

Which camera: the script first prints the cameras macOS lists. OpenCV does
not promise an order, so index 0 can be an iPhone (Continuity Camera) or a
virtual camera such as OBS. It asks if the preview shows the Mac's own
camera. If not, try --camera-index 1.

Step 4: record one placement (one command, one log)
---------------------------------------------------
    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/capture_aruco.py \
        --condition 2026-10-24_d1.0_center_yaw0_r1 \
        --distance-m 0.998 --lateral-m 0 --vertical-m 0 --range-m 0.998 \
        --marker-length-m 0.150 --hfov-deg 70 --operator "Your Name" \
        --notes "2 paper shims, lid at 90 deg" \
        --output "$HOME/Desktop/Zephyr Mesh/Data/aruco/2026-10-24_d1.0_center_yaw0_r1.jsonl"

The name reads: date, distance 1.0 m, center spot, yaw 0, placement 1. Make
one log per placement. The fit takes one mean and one spread per file, so
mixing placements in one file would count your tape error as camera noise.

What happens: 30 warmup frames are thrown away while the exposure settles.
The script asks if the preview shows the Mac's camera, then says "Press
Enter, then step away", counts down 3 seconds and records 300 frames (about
10 s at 30 fps). --yes skips both questions. Press q in the preview or Ctrl-C
to stop early: the frames so far still go in the log and the script exits 0.
If --condition, --distance-m or --operator is missing, it asks for them.
The bench method waits about 10 s after you step away: add --countdown-s 13
to get that. For the long still run, add --frames 1800.

A good run says almost every frame was detected. How big the spreads should
be is not known yet for this camera. The bench method's guess is a few mm at
1 m and more at 2 m, mostly in z, so write down what you get and do not
discard a run for being bigger than you expected. If the marker is not detected, or less than half the frames are, check:
the whole white margin is in view, the paper is flat and matte, the room is
bright with no glare on the marker, the marker is not too far away, and the
preview draws an outline around it. Then run the same command with
--overwrite.

Axes and the true position
--------------------------
Positions are meters in the camera frame, to the middle of the marker:
    x  is to the right of the camera (positive right)
    y  is DOWN (a marker above the lens has a negative y)
    z  is straight out from the lens
--distance-m is z, --lateral-m is x, --vertical-m is y (both default to 0).
--range-m is the taped straight line from the camera glass to the marker
center; it is optional and only used as a check. What you type is stored as
the truth, and the fit compares the camera to it. --truth-uncertainty-m
(default 0.01) is how far off the tape could be. Or give one number per axis
with --truth-uncertainty-xyz-m UX UY UZ. The fit uses one number for all
axes, so every frame gets the largest of the three.

Files (for --output NAME.jsonl)
-------------------------------
    NAME.jsonl         the detection log, written in one go at the end.
    NAME.frames.jsonl  one line per frame: corners in pixels, both pose
                       solutions, times and outcome. Written as it goes.
    NAME.raw.mkv       saves the recorded frames without losing picture
                       quality (a few can be skipped if the disk is slow;
                       the log says how many). It is big: a synthetic
                       1280x720 test took about 0.3 MB per frame.
--video-codec mjpg makes a smaller video, but it loses detail, so a replay
will not match exactly. --video-codec none saves no video. The video can show
you and your room, so a live run refuses to write inside the repo. Keep it
all in Data/. Nothing is overwritten unless you pass --overwrite.

Step 5: fit the noise (with the repo's .venv)
---------------------------------------------
The end of every run prints the exact command. It looks like this:

    PYTHONPATH=. .venv/bin/python tools/fit_vision_noise.py \
        "$HOME/Desktop/Zephyr Mesh/Data/aruco/NAME.jsonl" \
        --output "$HOME/Desktop/Zephyr Mesh/Data/aruco/NAME.fit.json"

Replay a saved video (a dry run, no camera)
-------------------------------------------
    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/capture_aruco.py \
        --dry-run "$HOME/Desktop/Zephyr Mesh/Data/aruco/NAME.raw.mkv" \
        --frame-times "$HOME/Desktop/Zephyr Mesh/Data/aruco/NAME.frames.jsonl" \
        --condition NAME --distance-m 0.998 --marker-length-m 0.150 --hfov-deg 70 \
        --output "$HOME/Desktop/Zephyr Mesh/Data/aruco/NAME-replay.jsonl"

With --frame-times the replay uses the live capture times and gives exactly
the same positions. Without it, give --fps. A folder of pictures works too,
as a pattern like "frames/%04d.png". The log says "fixture", not "measured".

Limits on macOS (read these before you trust a number)
------------------------------------------------------
  * OpenCV cannot lock exposure or focus. Both stay automatic.
  * There are no camera timestamps. A capture time is the laptop clock when
    the frame reached the program, later than the exposure by an unknown
    amount. The fit's latency only covers the time after that.
  * Late frames are dropped without a message. The script counts gaps in
    the capture times instead and warns about them.
  * The camera can run at another size than you asked for, without an
    error. The script reads the real size from the frames.
  * Index 0 may be another camera (see Step 3).
  * The marker stays still, so this measures one spot, not tracking. An
    error that is the same on every frame (focal length, tape) shows up as
    bias, not noise.

Exit codes: 0 ok (also after q or Ctrl-C), 2 a wrong option or you said
the preview is not the Mac's camera, 1 a failure while running.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.detection_log import write_detection_log
from sim.vision_trace import VISION_TRACE_SCHEMA, VisionTrace
from tools import aruco_geometry as geo

# Live and check runs may not write inside this folder (the raw video can show the room).
PRIVACY_ROOT = ROOT

DICTIONARY_NAMES = (
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16H5", "DICT_APRILTAG_25H9", "DICT_APRILTAG_36H10", "DICT_APRILTAG_36H11",
)
REFINEMENT_NAMES = {
    "subpix": "CORNER_REFINE_SUBPIX",
    "contour": "CORNER_REFINE_CONTOUR",
    "none": "CORNER_REFINE_NONE",
    "apriltag": "CORNER_REFINE_APRILTAG",
}
CONDITION_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 1280, 720, 30.0
DEFAULT_FRAMES = 300
DEFAULT_QUEUE_FRAMES = 60
DEFAULT_TRUTH_UNCERTAINTY_M = 0.01
DEFAULT_COUNTDOWN_S = 3.0
LOW_DETECTION_FRACTION = 0.5
# The printed marker: the black square is about 600 pixels, a whole number of
# cells (6 for a 4x4 marker with its border), so no cell is stretched.
MARKER_PRINT_TARGET_PX = 600
DEFAULT_MARKER_PRINT_MM = 150.0
DEFAULT_MARKER_MARGIN_MM = 20.0
PRINTABLE_WIDTH_MM = 200.0  # letter is 216 mm and A4 is 210 mm wide, minus printer margins
MIN_OPENCV = (4, 7)
VENV_HELP = ("Make the camera environment described at the top of tools/capture_aruco.py "
             "(run it with --help): a separate venv with \"opencv-python==5.0.0.93\".")
MACOS_SETTINGS = {
    "exposure": "auto, cannot be locked by OpenCV on macOS",
    "focus": "auto, cannot be locked by OpenCV on macOS",
    "frame_timestamps": "host monotonic clock when read() returned; the macOS backend gives no camera timestamps",
    "pose_solver": "IPPE_SQUARE with SQPNP fallback, lowest reprojection error",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CaptureError(RuntimeError):
    """Something failed while recording or replaying (exit code 1)."""


# --- small pieces that need no OpenCV --------------------------------------

@dataclass(frozen=True)
class Truth:
    """Where the marker really is, in the camera frame (meters, y down).

    uncertainty_xyz_m is one number per axis. A single number means the same
    for all three. The log lines carry the largest of the three, because the
    fit uses one uncertainty for every axis.
    """

    position_m: tuple[float, float, float]
    uncertainty_xyz_m: tuple[float, float, float] | float
    method: str

    def __post_init__(self) -> None:
        value = self.uncertainty_xyz_m
        triple = (float(value),) * 3 if isinstance(value, (int, float)) else tuple(float(v) for v in value)
        if len(triple) != 3:
            raise ValueError("uncertainty_xyz_m needs one number or three")
        object.__setattr__(self, "uncertainty_xyz_m", triple)
        object.__setattr__(self, "position_m", tuple(float(v) for v in self.position_m))

    @property
    def uncertainty_m(self) -> float:
        return max(self.uncertainty_xyz_m)

    def as_dict(self) -> dict[str, Any]:
        return {"position_m": [float(v) for v in self.position_m],
                "uncertainty_m": float(self.uncertainty_m), "method": self.method}


@dataclass(frozen=True)
class FrameResult:
    """What the detector found in one frame, before it is turned into log lines."""

    outcome: str
    reason: str | None
    ids_seen: tuple[int, ...]
    corners_px: list[list[float]] | None = None
    center_px: tuple[float, float] | None = None
    pose: geo.MarkerPose | None = None
    bbox_px: dict[str, float] | None = None


def subject_id_for(dictionary_name: str, marker_id: int) -> str:
    return f"aruco-{dictionary_name.lower()}-id{marker_id}"


def observation_id(frame_index: int) -> str:
    return f"obs-{frame_index + 1:06d}"


def frame_id(frame_index: int) -> str:
    return f"frame-{frame_index + 1:06d}"


def build_observation(frame_index: int, *, subject_id: str, capture_time_s: float,
                      processing_time_s: float, truth: Truth, result: FrameResult) -> dict[str, Any]:
    """One log line for one frame (a plain dict for VisionTrace.from_dict)."""
    obs: dict[str, Any] = {
        "observation_id": observation_id(frame_index),
        "frame_id": frame_id(frame_index),
        "subject_id": subject_id,
        "capture_time_s": float(capture_time_s),
        "processing_time_s": float(processing_time_s),
        "outcome": result.outcome,
        "ground_truth": truth.as_dict(),
    }
    if result.outcome == "detected":
        assert result.pose is not None
        obs["position_m"] = [float(v) for v in result.pose.tvec]
        obs["position_frame"] = "camera"
        if result.bbox_px is not None:
            obs["bbox_px"] = dict(result.bbox_px)
    elif result.outcome == "missed":
        obs["miss_reason"] = result.reason
    else:
        obs["unknown_reason"] = result.reason
    return obs


def _finite_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def build_sidecar_row(frame_index: int, *, capture_time_s: float, processing_time_s: float,
                      result: FrameResult) -> dict[str, Any]:
    """One line of the frames file: the numbers the log has no room for.

    pose_candidates holds every pose the solvers returned (both IPPE tilts and
    SQPNP), so a flip between the two tilts can be found later.
    """
    pose = result.pose
    return {
        "frame_index": int(frame_index),
        "frame_id": frame_id(frame_index),
        "capture_time_s": float(capture_time_s),
        "processing_time_s": float(processing_time_s),
        "ids_seen": [int(v) for v in result.ids_seen],
        "corners_px": result.corners_px,
        "center_px": None if result.center_px is None else [float(v) for v in result.center_px],
        "rvec": None if pose is None else [float(v) for v in pose.rvec],
        "tvec": None if pose is None else [float(v) for v in pose.tvec],
        "reprojection_error_px": None if pose is None else float(pose.reprojection_error_px),
        "ambiguity_ratio": None if pose is None else _finite_or_none(pose.ambiguity_ratio),
        "pose_candidates": None if pose is None else [dict(c) for c in pose.candidates],
        "outcome": result.outcome,
        "reason": result.reason,
        "video_frame_index": None,
    }


def dump_line(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path = ROOT) -> str:
    """The commit the script is running from, plus -dirty if files changed."""
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return head + ("-dirty" if status else "")


def list_cameras(run: Callable[..., Any] = subprocess.run, platform: str = sys.platform) -> list[str] | None:
    """Names of the cameras macOS knows about, or None if they could not be listed.

    system_profiler only reads the device list; it never opens a camera, so it
    cannot trigger the camera permission prompt.
    """
    if platform != "darwin":
        return None
    try:
        output = run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                     timeout=30, check=True).stdout
        items = json.loads(output).get("SPCameraDataType")
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None
    if not isinstance(items, list):
        return None
    return [item["_name"] for item in items if isinstance(item, dict) and isinstance(item.get("_name"), str)]


def inside_folder(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def output_paths(output: Path) -> dict[str, Path]:
    """NAME.jsonl -> the log, the frames file and the raw video next to it."""
    base = output.name[: -len(".jsonl")]
    return {"log": output, "frames": output.with_name(base + ".frames.jsonl"),
            "raw": output.with_name(base + ".raw.mkv"),
            "fit": output.with_name(base + ".fit.json"),
            "tmp": output.with_name(output.name + ".tmp")}


def position_stats(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Mean and spread of detected positions per axis, in mm.

    pooled_std_mm is the square root of the mean of the three per-axis
    variances, the one noise number tools/fit_vision_noise.py reports.
    range_mean_mm is the mean straight-line distance |position|.
    """
    rows = [o["position_m"] for o in observations if o["outcome"] == "detected"]
    if not rows:
        return None
    array = np.asarray(rows, dtype=float) * 1000.0
    if len(rows) > 1:
        variance = array.var(axis=0, ddof=1)
        std = np.sqrt(variance).tolist()
        pooled: float | None = float(np.sqrt(variance.mean()))
    else:
        std, pooled = [None, None, None], None
    return {"n": len(rows), "mean_mm": array.mean(axis=0).tolist(), "std_mm": std,
            "pooled_std_mm": pooled, "range_mean_mm": float(np.linalg.norm(array, axis=1).mean())}


def count_outcomes(observations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"detected": 0, "missed": 0, "unknown": 0}
    for obs in observations:
        counts[obs["outcome"]] += 1
    return counts


def _mm(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def stats_text(stats: dict[str, Any] | None) -> str:
    """Per-axis mean and spread in one sentence, for the log notes."""
    if stats is None:
        return "No frame was detected, so there are no position numbers."
    axes = "; ".join(f"{axis} mean {stats['mean_mm'][i]:.2f} std {_mm(stats['std_mm'][i])}"
                     for i, axis in enumerate("xyz"))
    return (f"Detected positions in mm (camera frame, y down), {stats['n']} frames: {axes}. "
            f"Pooled std (the one number the fit uses): {_mm(stats['pooled_std_mm'])} mm. "
            f"Mean range |position|: {stats['range_mean_mm']:.2f} mm.")


def build_notes(*, live: bool, condition: str, marker_text: str, camera: geo.CameraModel,
                timing: dict[str, Any], counts: dict[str, int], stats: dict[str, Any] | None,
                extra: list[str]) -> str:
    """Plain-text notes for the log header."""
    if live:
        time_text = ("capture_time_s is the laptop monotonic clock (seconds since recording started) read "
                     "right after read() returned. The macOS backend gives no camera timestamps, so this is "
                     "later than the exposure by an amount that is not measured. processing_time_s is the "
                     "same clock after detection and pose.")
    else:
        time_text = ("This is a replay of a saved video, not a live measurement. capture_time_s comes from "
                     "the frame-times file or from the video frame rate. processing_time_s is capture_time_s "
                     "plus the detection time measured on this machine during the replay.")
    rate = timing.get("measured_frame_rate_hz")
    median = timing.get("median_interval_s")
    parts = [
        f"Condition {condition}. Static marker.",
        marker_text + ".",
        f"Camera model: {camera.description}.",
        f"Timing: {timing['frames']} frames, measured frame rate "
        + ("unknown" if rate is None else f"{rate:.2f} Hz")
        + ("" if median is None else f", median interval {median * 1000:.2f} ms")
        + f", {timing['late_frames']} gaps longer than 1.5 x the median interval.",
        f"Counts: {counts['detected']} detected, {counts['missed']} missed, {counts['unknown']} unknown.",
        stats_text(stats),
        time_text,
    ]
    parts.extend(extra)
    return " ".join(parts)


def build_header(*, live: bool, trace_id: str, source: str, capture: dict[str, Any],
                 clock_domain: str, notes: str, code_revision: str,
                 provenance: dict[str, Any] | None) -> dict[str, Any]:
    header: dict[str, Any] = {
        "trace_id": trace_id,
        "status": "measured" if live else "fixture",
        "clock_domain": clock_domain,
        "source": source,
        "capture": capture,
        "code_revision": code_revision,
        "notes": notes,
    }
    if provenance is not None:
        header["provenance"] = provenance
    return header


def truth_values(truth: Truth, *, condition: str, marker_side_m: float, range_m: float | None) -> dict[str, Any]:
    """The ground-truth numbers as the bench method writes them (x, y, z, range)."""
    values: dict[str, Any] = {"x_m": truth.position_m[0], "y_m": truth.position_m[1],
                              "z_m": truth.position_m[2], "condition": condition,
                              "marker_side_m": float(marker_side_m)}
    if range_m is not None:
        values["range_m"] = float(range_m)
    return values


def build_provenance(*, camera_model: str, settings: dict[str, Any], calibration_ref: str | None,
                     calibration_null_reason: str | None, raw_sha256: str, code_revision: str,
                     captured_at: str, operator: str, condition: str, truth: Truth,
                     marker_side_m: float, range_m: float | None = None) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "instrument": {"kind": "camera", "model": camera_model, "settings": settings},
        "calibration_ref": calibration_ref,
        "raw_sha256": raw_sha256,
        "code_revision": code_revision,
        "captured_at": captured_at,
        "operator": operator,
        "ground_truth": {
            "method": truth.method,
            "values": truth_values(truth, condition=condition, marker_side_m=marker_side_m, range_m=range_m),
            "uncertainty": dict(zip(("x_m", "y_m", "z_m"), truth.uncertainty_xyz_m)),
        },
    }
    if calibration_ref is None:
        provenance["calibration_ref_null_reason"] = calibration_null_reason
    return provenance


def make_trace(header: dict[str, Any], observations: list[dict[str, Any]]) -> VisionTrace:
    """Build the trace the real validator checks."""
    return VisionTrace.from_dict({"schema": VISION_TRACE_SCHEMA, **header, "observations": observations})


def write_log_atomically(trace: VisionTrace, paths: dict[str, Path]) -> None:
    """Write NAME.jsonl.tmp, then move it over NAME.jsonl. Never leaves a .tmp behind."""
    try:
        write_detection_log(trace, paths["tmp"])
        os.replace(paths["tmp"], paths["log"])
    except BaseException:
        paths["tmp"].unlink(missing_ok=True)
        raise


def replay_time(index: int, frame_rate_hz: float) -> float:
    """Same formula as aruco_geometry.replay_capture_times, for one frame."""
    return 0.0 + index / frame_rate_hz


def load_frame_times(path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    """Read a frames file. Returns {position in video: row} and how many rows were not in the video."""
    rows: dict[int, dict[str, Any]] = {}
    not_in_video = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                frame_index = row["frame_index"]
                capture = row["capture_time_s"]
                if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                    raise ValueError("frame_index must be a whole number, 0 or more")
                if isinstance(capture, bool) or not isinstance(capture, (int, float)) \
                        or not math.isfinite(capture) or capture < 0:
                    raise ValueError("capture_time_s must be a finite number, 0 or more")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: not a frames file line ({exc})") from exc
            position = row.get("video_frame_index", frame_index)
            if position is None:
                not_in_video += 1
                continue
            if position in rows:
                raise ValueError(f"{path}:{number}: video_frame_index {position} appears twice")
            rows[position] = {"frame_index": frame_index, "capture_time_s": float(capture)}
    if not rows:
        raise ValueError(f"{path}: no frame times found")
    return rows, not_in_video


def resolution_problem(camera: geo.CameraModel, width: int, height: int) -> str | None:
    """A plain message if a frame does not fit the camera model, else None."""
    if (int(width), int(height)) == (camera.width_px, camera.height_px):
        return None
    model = f"{camera.width_px}x{camera.height_px}"
    if camera.source == "calibration_file":
        return (f"the frames are {width}x{height} pixels but the --calibration file is for {model}. "
                f"Ask the camera for {model} with --width and --height, or make a calibration at "
                f"{width}x{height} with tools/calibrate_charuco.py.")
    return (f"the frame size changed from {model} to {width}x{height} during the run. The --hfov-deg "
            f"focal length was worked out for {model}, so the run stopped there.")


# --- the printed marker (PNG with its physical size) -------------------------

def png_chunk(kind: bytes, data: bytes) -> bytes:
    """One PNG chunk: length, type, data, CRC-32 of type plus data."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_with_physical_size(png: bytes, pixels_per_meter: int) -> bytes:
    """Return the PNG with a pHYs chunk, so 100% printing gives the intended size.

    pHYs stores pixels per meter for x and y and a unit byte (1 = meter). It
    must come before the image data, so it goes right after IHDR. Any older
    pHYs chunk is dropped.
    """
    if png[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    phys = png_chunk(b"pHYs", struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1))
    out = [PNG_SIGNATURE]
    position = 8
    while position < len(png):
        (length,) = struct.unpack(">I", png[position:position + 4])
        kind = png[position + 4:position + 8]
        end = position + 12 + length
        if kind != b"pHYs":
            out.append(png[position:end])
        if kind == b"IHDR":
            out.append(phys)
        position = end
    return b"".join(out)


def marker_print_layout(cells: int, print_mm: float, margin_mm: float) -> dict[str, float | int]:
    """Pixel sizes for the printed marker.

    The black square is a whole number of cells close to 600 pixels. The
    density (pixels per meter) is chosen so that square prints print_mm wide.
    """
    side_px = cells * max(1, round(MARKER_PRINT_TARGET_PX / cells))
    pixels_per_meter = round(side_px / (print_mm / 1000.0))
    margin_px = round(margin_mm / 1000.0 * pixels_per_meter)
    return {"side_px": side_px, "margin_px": margin_px, "pixels_per_meter": pixels_per_meter,
            "side_mm": side_px / pixels_per_meter * 1000.0,
            "margin_mm": margin_px / pixels_per_meter * 1000.0}


# --- OpenCV side ------------------------------------------------------------

def load_cv2() -> Any:
    """Import OpenCV and check the new ArUco API is there. Raises CaptureError with help if not."""
    try:
        import cv2
    except ImportError as exc:
        raise CaptureError("OpenCV (cv2) is not installed in this Python. " + VENV_HELP) from exc
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "ArucoDetector"):
        raise CaptureError(f"This OpenCV ({cv2.__version__}) has no cv2.aruco.ArucoDetector. "
                           f"OpenCV {MIN_OPENCV[0]}.{MIN_OPENCV[1]} or newer is needed. " + VENV_HELP)
    return cv2


def get_dictionary(cv2: Any, name: str) -> Any:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def dictionary_size(dictionary: Any) -> int:
    return int(dictionary.bytesList.shape[0])


def print_marker(cv2: Any, path: Path, dictionary_name: str, marker_id: int, print_mm: float,
                 margin_mm: float) -> dict[str, float | int]:
    """Write the marker PNG with its print size stored in it. Returns the layout used."""
    dictionary = get_dictionary(cv2, dictionary_name)
    layout = marker_print_layout(int(dictionary.markerSize) + 2, print_mm, margin_mm)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, int(layout["side_px"]), borderBits=1)
    margin = int(layout["margin_px"])
    padded = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    ok, encoded = cv2.imencode(".png", padded)
    if not ok:
        raise CaptureError(f"Could not encode {path}")
    try:
        path.write_bytes(png_with_physical_size(encoded.tobytes(), int(layout["pixels_per_meter"])))
    except OSError as exc:
        raise CaptureError(f"Could not write {path}: {exc}") from exc
    return layout


@dataclass
class Context:
    """Everything the per-frame code needs. Built once per run."""

    cv2: Any
    detector: Any
    camera: geo.CameraModel
    marker_length_m: float
    marker_id: int
    subject_id: str
    truth: Truth


def build_context(cv2: Any, *, dictionary_name: str, marker_id: int, marker_length_m: float,
                  corner_refinement: str, camera: geo.CameraModel, truth: Truth) -> Context:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, REFINEMENT_NAMES[corner_refinement])
    detector = cv2.aruco.ArucoDetector(get_dictionary(cv2, dictionary_name), params)
    return Context(cv2=cv2, detector=detector, camera=camera, marker_length_m=marker_length_m,
                   marker_id=marker_id, subject_id=subject_id_for(dictionary_name, marker_id), truth=truth)


def analyze_frame(frame: np.ndarray, ctx: Context) -> FrameResult:
    """Detect the target marker in one BGR frame and work out its pose. Never changes the frame."""
    cv2 = ctx.cv2
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = ctx.detector.detectMarkers(gray)
    seen = () if ids is None else tuple(sorted(int(v) for v in np.asarray(ids).reshape(-1)))
    hits = [] if ids is None else [i for i, v in enumerate(np.asarray(ids).reshape(-1)) if int(v) == ctx.marker_id]
    if not hits:
        return FrameResult("missed", "other_markers_only" if seen else "marker_not_found", seen)
    if len(hits) > 1:
        return FrameResult("unknown", "multiple_target_markers", seen)
    corner_array = np.asarray(corners[hits[0]], dtype=float).reshape(4, 2)
    corner_list = corner_array.tolist()
    center = geo.marker_center_px(corner_array)
    pose = geo.estimate_marker_pose(cv2, corner_array, ctx.camera, ctx.marker_length_m)
    if pose is None:
        return FrameResult("unknown", "pose_failed", seen, corner_list, center)
    bbox = geo.bbox_from_corners(corner_array, ctx.camera.width_px, ctx.camera.height_px)
    return FrameResult("detected", None, seen, corner_list, center, pose, bbox)


def process_frame(frame: np.ndarray, frame_index: int, capture_time_s: float, ctx: Context, *,
                  now: Callable[[], float] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The one per-frame code path, used for live capture and for replay.

    Live: pass now, a function giving the current time on the same clock as
    capture_time_s. processing_time_s is read after detection and pose.
    Replay: leave now out. processing_time_s is capture_time_s plus how long
    detection and pose took on this machine.
    """
    started = time.perf_counter()
    result = analyze_frame(frame, ctx)
    if now is None:
        processing = capture_time_s + (time.perf_counter() - started)
    else:
        processing = max(float(now()), capture_time_s)
    observation = build_observation(frame_index, subject_id=ctx.subject_id, capture_time_s=capture_time_s,
                                    processing_time_s=processing, truth=ctx.truth, result=result)
    row = build_sidecar_row(frame_index, capture_time_s=capture_time_s, processing_time_s=processing,
                            result=result)
    return observation, row


def check_readout(result: FrameResult, camera: geo.CameraModel) -> list[str]:
    """The live numbers --check-camera shows: position and range in mm, and the pixel offset from (cx, cy)."""
    if result.pose is None:
        if result.outcome == "missed":
            return ["marker not found"]
        return [f"marker found but no position ({result.reason})"]
    x, y, z = (v * 1000.0 for v in result.pose.tvec)
    matrix = camera.matrix()
    assert result.center_px is not None
    du, dv = result.center_px[0] - matrix[0, 2], result.center_px[1] - matrix[1, 2]
    return [f"x {x:+.1f}  y {y:+.1f}  z {z:.1f} mm   range {math.sqrt(x * x + y * y + z * z):.1f} mm",
            f"marker center from crosshair: u-cx {du:+.1f} px, v-cy {dv:+.1f} px"]


def draw_overlay(cv2: Any, frame: np.ndarray, corners_px: list[list[float]] | None, text: str | list[str],
                 crosshair: tuple[float, float] | None = None) -> np.ndarray:
    """A copy of the frame with the marker outline, text lines and an optional crosshair.

    The original frame is never touched, so what is detected and saved is the
    camera's own picture.
    """
    canvas = frame.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if crosshair is not None:
        u, v = round(crosshair[0]), round(crosshair[1])
        height, width = canvas.shape[:2]
        cv2.line(canvas, (u, 0), (u, height - 1), (255, 255, 0), 1)
        cv2.line(canvas, (0, v), (width - 1, v), (255, 255, 0), 1)
    if corners_px is not None:
        points = np.round(np.asarray(corners_px, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 255, 0), 2)
    lines = [text] if isinstance(text, str) else list(text)
    for number, line in enumerate(lines):
        origin = (12, 28 + 26 * number)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class Preview:
    """The live preview window. Quietly turns itself off if OpenCV has no window support."""

    WINDOW = "Zephyr ArUco capture"

    def __init__(self, cv2: Any, log: Callable[[str], None] = print) -> None:
        self.cv2 = cv2
        self.enabled = False
        try:
            cv2.namedWindow(self.WINDOW)
            self.enabled = True
        except (cv2.error, AttributeError):
            log("Preview window is not available in this OpenCV build (no GUI support). Going on without it.")

    def show(self, frame: np.ndarray, corners_px: list[list[float]] | None, text: str | list[str],
             crosshair: tuple[float, float] | None = None) -> bool:
        """Show one frame. Returns True if the user pressed q."""
        if not self.enabled:
            return False
        cv2 = self.cv2
        lines = ([text] if isinstance(text, str) else list(text)) + ["press q to stop"]
        cv2.imshow(self.WINDOW, draw_overlay(cv2, frame, corners_px, lines, crosshair))
        return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"))

    def close(self) -> None:
        if self.enabled:
            self.cv2.destroyAllWindows()
            self.cv2.waitKey(1)
            self.enabled = False


class RawVideoWriter:
    """Writes frames to a video from a background thread so capture never waits on the disk.

    The queue is bounded. If the writer cannot keep up, submit() drops the
    frame (it returns None) and counts it, instead of stalling the camera.
    Any failure inside the thread is kept and reported by close(); the thread
    keeps emptying the queue, so close() never hangs on a dead writer. After
    release, count_frames (if given) reads the file back and close() checks
    that it holds exactly the frames that were accepted.
    """

    def __init__(self, video_writer: Any, max_frames: int = DEFAULT_QUEUE_FRAMES,
                 count_frames: Callable[[], int] | None = None) -> None:
        self._writer = video_writer
        self._count_frames = count_frames
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, int(max_frames)))
        self.accepted = 0
        self.dropped = 0
        self.error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="raw-video-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self.error is not None:
                continue  # keep draining so submit() and close() never block on a dead writer
            try:
                if self._writer.write(item) is False:
                    raise OSError("video write returned False")
            except Exception as exc:  # noqa: BLE001  any failure is kept and reported by close()
                self.error = exc

    def submit(self, frame: np.ndarray) -> int | None:
        """Queue a copy of the frame. Returns its position in the video, or None if it was dropped."""
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1
            return None
        position = self.accepted
        self.accepted += 1
        return position

    def close(self) -> None:
        """Finish writing, release the file and check it. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # waits for room, which the thread always makes
        self._thread.join()
        try:
            self._writer.release()
        except Exception as exc:  # noqa: BLE001  reported just below
            self.error = self.error or exc
        if self.error is not None:
            raise CaptureError(f"writing the raw video failed: {self.error}")
        if self._count_frames is not None:
            try:
                written = self._count_frames()
            except Exception as exc:
                raise CaptureError(f"could not read the raw video back to check it: {exc}") from exc
            if written != self.accepted:
                raise CaptureError(f"the raw video holds {written} frames but {self.accepted} were sent to it")


@dataclass
class Recording:
    observations: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    warmup_done: int = 0
    countdown_done: int = 0
    stream_ended: bool = False
    stopped_early: bool = False
    error: Exception | None = None
    frame_times_past_end: int = 0


class PeekedSource:
    """Wraps a frame source so main() can look at the first frame before any loop reads it."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._pending: np.ndarray | None = None

    def peek(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._source.read()
        if ok:
            self._pending = frame
        return ok, frame

    def drop_pending(self) -> None:
        self._pending = None

    def read(self) -> tuple[bool, Any]:
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return True, frame
        return self._source.read()

    def get(self, prop: Any) -> Any:
        return self._source.get(prop)

    def release(self) -> None:
        self._source.release()


def discard_frames(source: Any, *, count: int | None = None, seconds: float | None = None,
                   clock: Callable[[], float] = time.monotonic, preview: Any = None,
                   log: Callable[[str], None] = print, label: str = "warming up") -> tuple[int, bool]:
    """Read and throw away frames: a fixed count (warmup) or for some seconds (countdown).

    Reading keeps the camera's buffer fresh, so the first recorded frame is a
    new one. Returns (frames thrown away, True if q was pressed).
    """
    done = 0
    start = clock()
    shown_second: int | None = None
    while True:
        if count is not None and done >= count:
            break
        if seconds is not None:
            left = seconds - (clock() - start)
            if left <= 0:
                break
            second = math.ceil(left)
            if second != shown_second:
                log(f"  {second}")
                shown_second = second
        ok, frame = source.read()
        if not ok:
            raise CaptureError("The camera stopped giving frames before recording started.")
        done += 1
        text = (f"{label} {done}/{count}" if count is not None
                else f"{label} {math.ceil(max(seconds - (clock() - start), 0.0))}")  # type: ignore[operator]
        if preview is not None and preview.show(frame, None, text):
            return done, True
    return done, False


def record_frames(source: Any, ctx: Context, *, frames: int, clock: Callable[[], float] = time.monotonic,
                  writer: RawVideoWriter | None = None, sidecar: Any = None, preview: Any = None,
                  log: Callable[[str], None] = print, progress_every: int = 30,
                  recording: Recording | None = None) -> Recording:
    """The live loop: detect, save and log each frame.

    source needs read() -> (ok, frame). Press q in the preview (or Ctrl-C) to
    stop early; what was recorded is kept. Any other error ends the loop and is
    stored in recording.error, so the frames before it can still be logged.
    """
    recording = recording if recording is not None else Recording()
    try:
        t0 = clock()
        for frame_index in range(frames):
            ok, frame = source.read()
            capture_time_s = clock() - t0  # read right after the frame arrived
            if not ok:
                recording.stream_ended = True
                break
            problem = resolution_problem(ctx.camera, frame.shape[1], frame.shape[0])
            if problem is not None:
                raise CaptureError(problem)
            observation, row = process_frame(frame, frame_index, capture_time_s, ctx,
                                             now=lambda: clock() - t0)
            # Saving comes after the result is stamped, so the copy is not counted as latency.
            row["video_frame_index"] = None if writer is None else writer.submit(frame)
            recording.observations.append(observation)
            recording.rows.append(row)
            if sidecar is not None:
                sidecar.write(row)
            if progress_every and (frame_index + 1) % progress_every == 0:
                log(f"  recorded {frame_index + 1}/{frames} frames")
            if preview is not None and preview.show(
                    frame, row["corners_px"], f"frame {frame_index + 1}/{frames}  {row['outcome']}"):
                recording.stopped_early = True
                break
    except KeyboardInterrupt:
        recording.stopped_early = True
    except Exception as exc:  # noqa: BLE001  main() logs the frames so far, then reports it
        recording.error = exc
    return recording


def replay_frames(source: Any, ctx: Context, *, frame_times: dict[int, dict[str, Any]] | None,
                  fps: float | None, max_frames: int | None, sidecar: Any = None,
                  recording: Recording | None = None) -> Recording:
    """Dry run: send saved frames through the same process_frame as the live loop."""
    recording = recording if recording is not None else Recording()
    position = 0
    try:
        while max_frames is None or position < max_frames:
            ok, frame = source.read()
            if not ok:
                recording.stream_ended = True
                break
            problem = resolution_problem(ctx.camera, frame.shape[1], frame.shape[0])
            if problem is not None:
                raise CaptureError(problem)
            if frame_times is not None:
                entry = frame_times.get(position)
                if entry is None:
                    raise CaptureError(f"The video has a frame at position {position} but the frame-times "
                                       "file has no row for it. Is it the frames file of this video?")
                frame_index, capture_time_s = entry["frame_index"], entry["capture_time_s"]
            else:
                assert fps is not None
                frame_index, capture_time_s = position, replay_time(position, fps)
            observation, row = process_frame(frame, frame_index, capture_time_s, ctx)
            row["video_frame_index"] = position
            recording.observations.append(observation)
            recording.rows.append(row)
            if sidecar is not None:
                sidecar.write(row)
            position += 1
    except Exception as exc:  # noqa: BLE001  main() logs the frames so far, then reports it
        recording.error = exc
    if frame_times is not None and recording.stream_ended:
        recording.frame_times_past_end = sum(1 for key in frame_times if key >= position)
    return recording


@dataclass
class CheckResult:
    capture_times_s: list[float] = field(default_factory=list)
    frame_shape: tuple[int, ...] | None = None
    detected: int = 0
    last_readout: list[str] = field(default_factory=list)
    stream_ended: bool = False


def check_camera_loop(source: Any, ctx: Context, *, clock: Callable[[], float] = time.monotonic,
                      preview: Any = None, log: Callable[[str], None] = print, seconds: float | None = None,
                      max_frames: int | None = None, report_every: int = 30) -> CheckResult:
    """--check-camera: show live numbers until q, Ctrl-C, or the time is up. Nothing is saved."""
    result = CheckResult()
    camera = ctx.camera
    crosshair = (float(camera.matrix()[0, 2]), float(camera.matrix()[1, 2]))
    t0 = clock()
    try:
        while max_frames is None or len(result.capture_times_s) < max_frames:
            ok, frame = source.read()
            now = clock() - t0
            if not ok:
                result.stream_ended = True
                break
            result.capture_times_s.append(now)
            result.frame_shape = tuple(frame.shape)
            problem = resolution_problem(camera, frame.shape[1], frame.shape[0])
            if problem is not None:
                raise CaptureError(problem)
            found = analyze_frame(frame, ctx)
            result.detected += found.pose is not None
            result.last_readout = check_readout(found, camera)
            if preview is not None and preview.show(frame, found.corners_px, result.last_readout, crosshair):
                break
            if report_every and len(result.capture_times_s) % report_every == 0:
                log("  " + "   ".join(result.last_readout))
            if seconds is not None and now >= seconds:
                break
    except KeyboardInterrupt:
        pass
    return result


class SidecarWriter:
    """Appends one JSON line per frame and flushes it, so a crash keeps the data."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(dump_line(row) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def open_camera(cv2: Any, index: int, width: int, height: int, fps: float) -> Any:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise CaptureError(f"Could not open camera {index}. On a Mac, check System Settings > Privacy & "
                           "Security > Camera and allow your terminal app. Try --camera-index 1 for "
                           "another camera.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def open_video(cv2: Any, path: str) -> Any:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {path} as a video or an image pattern")
    return capture


def count_video_frames(cv2: Any, path: Path) -> int:
    """Read a video file back and count its frames (the stored count can be an estimate)."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"could not open {path}")
        count = 0
        while capture.grab():
            count += 1
        return count
    finally:
        capture.release()


def open_raw_video(cv2: Any, path: Path, codec: str, fps: float, size: tuple[int, int],
                   queue_frames: int) -> RawVideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*("FFV1" if codec == "ffv1" else "MJPG"))
    video_writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not video_writer.isOpened():
        raise CaptureError(f"Could not open {path} for writing a {codec} video. "
                           "Check the folder and the disk space.")
    return RawVideoWriter(video_writer, queue_frames, lambda: count_video_frames(cv2, path))


# --- command line -----------------------------------------------------------

def _number(text: str, *, minimum: float | None = None, strictly_above: float | None = None) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    if minimum is not None and value < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum:g}")
    if strictly_above is not None and value <= strictly_above:
        raise argparse.ArgumentTypeError(f"must be greater than {strictly_above:g}")
    return value


def _any_number(text: str) -> float:
    return _number(text)


def _positive(text: str) -> float:
    return _number(text, strictly_above=0.0)


def _nonnegative(text: str) -> float:
    return _number(text, minimum=0.0)


def _whole(text: str, minimum: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}")
    return value


def _positive_int(text: str) -> int:
    return _whole(text, 1)


def _nonnegative_int(text: str) -> int:
    return _whole(text, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_argument_group("what you are recording")
    what.add_argument("--condition", help="name of this placement, letters, digits, . _ - only (asked for in live mode if missing)")
    what.add_argument("--distance-m", type=_positive, help="z: tape distance from the lens straight out to the marker center, meters")
    what.add_argument("--lateral-m", type=_any_number, default=0.0, help="x: marker center to the camera's right of the lens axis, meters (default 0)")
    what.add_argument("--vertical-m", type=_any_number, default=0.0, help="y: marker center DOWN from the lens axis, meters (default 0)")
    what.add_argument("--range-m", type=_positive, help="taped straight-line distance from the camera glass to the marker center, meters (optional check)")
    what.add_argument("--truth-uncertainty-m", type=_nonnegative, help="how far off the tape could be, meters, same for x, y and z (default 0.01)")
    what.add_argument("--truth-uncertainty-xyz-m", type=_nonnegative, nargs=3, metavar=("UX", "UY", "UZ"), help="the same, one number per axis")
    what.add_argument("--truth-method", default="tape measure", help="how you measured the true position (default 'tape measure')")
    what.add_argument("--notes", help="free text added to the log notes, such as the shim stack or the lid angle")
    marker = parser.add_argument_group("the marker")
    marker.add_argument("--marker-length-m", type=_positive, help="measured edge of the BLACK square, meters")
    marker.add_argument("--dictionary", choices=DICTIONARY_NAMES, metavar="NAME", default="DICT_4X4_50", help="marker dictionary, such as DICT_4X4_50 (default), DICT_5X5_100 or DICT_APRILTAG_36H11")
    marker.add_argument("--marker-id", type=_nonnegative_int, default=0, help="marker id to track (default 0)")
    marker.add_argument("--corner-refinement", choices=sorted(REFINEMENT_NAMES), metavar="METHOD", default="subpix", help="corner refinement: subpix (default), contour, none or apriltag")
    marker.add_argument("--print-marker", type=Path, metavar="PNG", help="write a printable marker image and exit (no camera)")
    marker.add_argument("--marker-print-mm", type=_positive, default=DEFAULT_MARKER_PRINT_MM, help="printed size of the black square, mm (default 150)")
    marker.add_argument("--marker-margin-mm", type=_nonnegative, default=DEFAULT_MARKER_MARGIN_MM, help="white margin on each side of the printed marker, mm (default 20)")
    cam = parser.add_argument_group("camera model (give exactly one)")
    cam.add_argument("--calibration", type=Path, help="camera calibration JSON file from tools/calibrate_charuco.py")
    cam.add_argument("--hfov-deg", type=_positive, help="horizontal field of view in degrees, a guess used to estimate the focal length")
    live = parser.add_argument_group("live camera")
    live.add_argument("--check-camera", action="store_true", help="show the preview with live numbers and print the frame size and rate; writes no files")
    live.add_argument("--check-seconds", type=_positive, help="with --check-camera, stop after this many seconds instead of waiting for q")
    live.add_argument("--camera-index", type=_nonnegative_int, default=0, help="which camera, 0 is the first OpenCV finds (default 0)")
    live.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH, help="requested frame width (default 1280)")
    live.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT, help="requested frame height (default 720)")
    live.add_argument("--fps", type=_positive, default=None, help="requested frame rate (live default 30); in a dry run, the frame rate to assume when there is no --frame-times")
    live.add_argument("--warmup-frames", type=_nonnegative_int, default=30, help="frames thrown away while the exposure settles (default 30)")
    live.add_argument("--countdown-s", type=_nonnegative, default=DEFAULT_COUNTDOWN_S, help="seconds of countdown before recording, frames not logged (default 3)")
    live.add_argument("--frames", type=_positive_int, default=None, help="frames to record (live default 300; dry run default: the whole video)")
    live.add_argument("--yes", action="store_true", help="skip the camera question and the Enter prompt")
    live.add_argument("--no-preview", action="store_true", help="do not show the preview window")
    live.add_argument("--video-codec", choices=("ffv1", "mjpg", "none"), default="ffv1", help="raw video: ffv1 keeps every detail (default), mjpg is smaller but loses detail, none saves no video")
    live.add_argument("--queue-frames", type=_positive_int, default=DEFAULT_QUEUE_FRAMES, help="frames the video writer may hold before it skips one (default 60)")
    live.add_argument("--operator", help="who ran the recording (asked for in live mode if missing)")
    live.add_argument("--camera-model", default="built-in camera", help="camera name for the log (default 'built-in camera')")
    dry = parser.add_argument_group("dry run (no camera)")
    dry.add_argument("--dry-run", type=str, metavar="VIDEO", help="replay a saved video file, or an image pattern such as frames/%%04d.png, through the same code")
    dry.add_argument("--frame-times", type=Path, metavar="FRAMES_JSONL", help="the NAME.frames.jsonl of the live run, so the replay uses its capture times")
    out = parser.add_argument_group("output")
    out.add_argument("--output", type=Path, help="log file to write, must end in .jsonl and (live) be outside the repo")
    out.add_argument("--overwrite", action="store_true", help="replace existing output files")
    out.add_argument("--trace-id", help="name of the run (default aruco-CONDITION-TIMESTAMP)")
    out.add_argument("--source-id", help="source_id for the log (default mac-built-in-camera, or the video file name in a dry run)")
    out.add_argument("--device-id", default="this-mac", help="device id for the log (default this-mac)")
    return parser


def _fill_missing_live_values(parser: argparse.ArgumentParser, args: argparse.Namespace,
                              ask: Callable[[str], str]) -> None:
    def prompt(text: str, name: str) -> str:
        try:
            return ask(text).strip()
        except EOFError:
            parser.error(f"{name} is missing and there is no keyboard to ask for it; pass it on the command line")

    if args.condition is None:
        args.condition = prompt("Placement name (letters, digits, . _ -), for example "
                                "2026-10-24_d1.0_center_yaw0_r1: ", "--condition")
    if args.distance_m is None:
        text = prompt("Tape distance from the lens to the marker center, in meters: ", "--distance-m")
        try:
            args.distance_m = _positive(text)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"--distance-m: {exc}")
    if args.operator is None:
        args.operator = prompt("Your name (operator): ", "--operator")


def _check_camera_model_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.marker_length_m is None:
        parser.error("--marker-length-m is required (edge of the black square, in meters)")
    if (args.calibration is None) == (args.hfov_deg is None):
        parser.error("give exactly one of --calibration FILE or --hfov-deg DEGREES")
    if args.hfov_deg is not None and not 1.0 < args.hfov_deg < 179.0:
        parser.error("--hfov-deg must be between 1 and 179")
    if args.calibration is not None and not args.calibration.is_file():
        parser.error(f"--calibration file not found: {args.calibration}")


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, mode: str) -> None:
    """Checks that need no camera or OpenCV. Calls parser.error on a problem."""
    if args.check_seconds is not None and mode != "check":
        parser.error("--check-seconds only works together with --check-camera")
    if mode == "check":
        if args.dry_run is not None:
            parser.error("--check-camera and --dry-run do not go together")
        if args.output is not None:
            parser.error("--check-camera writes no files, so leave out --output")
        _check_camera_model_args(parser, args)
        return
    if args.condition is None or not CONDITION_RE.fullmatch(args.condition):
        parser.error("--condition may only use letters, digits, '.', '_' and '-' (no spaces)")
    if args.distance_m is None:
        parser.error("--distance-m is required")
    _check_camera_model_args(parser, args)
    if args.truth_uncertainty_m is not None and args.truth_uncertainty_xyz_m is not None:
        parser.error("give --truth-uncertainty-m or --truth-uncertainty-xyz-m, not both")
    if args.output is None:
        parser.error("--output is required")
    if args.output.suffix != ".jsonl" or args.output.name == ".jsonl":
        parser.error("--output must be a file name ending in .jsonl")
    if not args.output.resolve().parent.is_dir():
        parser.error(f"the folder for --output does not exist: {args.output.resolve().parent}")
    if mode == "live" and inside_folder(args.output, PRIVACY_ROOT):
        parser.error(f"--output {args.output} is inside the repo ({PRIVACY_ROOT}). A live run saves raw "
                     "video that can show you and your room, so write it outside the repo, for example "
                     "in \"$HOME/Desktop/Zephyr Mesh/Data/aruco/\".")
    paths = output_paths(args.output)
    wanted = ["log", "frames"] + (["raw"] if mode == "live" and args.video_codec != "none" else [])
    if not args.overwrite:
        for key in wanted:
            if paths[key].exists():
                parser.error(f"{paths[key]} already exists; use --overwrite to replace it")
    if not args.operator and mode == "live":
        parser.error("--operator is required")
    if args.frame_times is not None and mode != "replay":
        parser.error("--frame-times only works together with --dry-run")
    if mode == "replay":
        if "%" not in args.dry_run and not Path(args.dry_run).is_file():
            parser.error(f"--dry-run file not found: {args.dry_run}")
        if args.frame_times is not None:
            if not args.frame_times.is_file():
                parser.error(f"--frame-times file not found: {args.frame_times}")
            if args.frame_times.resolve() == paths["frames"].resolve():
                parser.error("--frame-times is the same file this run would write; choose another --output")


def _camera_for(parser: argparse.ArgumentParser, args: argparse.Namespace, size: tuple[int, int]
                ) -> tuple[geo.CameraModel, str | None]:
    """Build the camera model from the real frame size. Returns (model, calibration file sha256 or None)."""
    try:
        if args.calibration is not None:
            camera, digest = geo.load_camera_calibration(args.calibration)
        else:
            camera, digest = geo.camera_from_hfov(size[0], size[1], args.hfov_deg), None
    except ValueError as exc:
        parser.error(str(exc))
    problem = resolution_problem(camera, size[0], size[1])
    if problem is not None:
        parser.error(problem)
    return camera, digest


def _quote(path: Path) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def fit_command(paths: dict[str, Path]) -> list[str]:
    """The lines that tell the user how to fit this log (repo .venv, absolute paths)."""
    log_path, fit_path = _quote(paths["log"].resolve()), _quote(paths["fit"].resolve())
    return ["Next, fit the noise with the repo's .venv (the camera venv is only for capture):",
            f"  cd {_quote(ROOT)}",
            f"  PYTHONPATH=. .venv/bin/python tools/fit_vision_noise.py {log_path} --output {fit_path}"]


def _summary_lines(args: argparse.Namespace, recording: Recording, timing: dict[str, Any],
                   observations: list[dict[str, Any]], paths: dict[str, Path], truth: Truth,
                   live: bool, size: tuple[int, int]) -> list[str]:
    counts = count_outcomes(observations)
    total = len(observations)
    lines = [(f"Recorded {total} frames ({counts['detected']} detected, "
              f"{counts['missed']} missed, {counts['unknown']} unknown). Frame size {size[0]}x{size[1]} "
              f"(read from the frames).")]
    if counts["detected"] < LOW_DETECTION_FRACTION * total:
        lines.append(f"WARNING: the marker was found in only {counts['detected']} of {total} frames. Check that "
                     "the whole white margin is in view, the paper is flat and matte, there is no glare, the "
                     "marker is not too far away, and the preview draws an outline around it. Then run the "
                     "same command again with --overwrite.")
    rate, median = timing["measured_frame_rate_hz"], timing["median_interval_s"]
    lines.append("Measured frame rate: " + ("unknown" if rate is None else f"{rate:.2f} Hz")
                 + ("" if median is None else f", median interval {median * 1000:.1f} ms")
                 + f", gaps: {timing['late_frames']}.")
    if timing["late_frames"]:
        lines.append(f"WARNING: {timing['late_frames']} gaps longer than 1.5 x the median interval. macOS drops "
                     "late frames without saying so. Close other apps, or try --no-preview.")
    stats = position_stats(observations)
    if stats is not None:
        lines.append("Position in the camera frame, mm (x right, y down, z out):")
        lines.append("           mean       std      tape   mean-tape")
        for i, axis in enumerate("xyz"):
            std = stats["std_mm"][i]
            std_text = "      n/a" if std is None else f"{std:9.2f}"
            tape = truth.position_m[i] * 1000.0
            lines.append(f"  {axis}  {stats['mean_mm'][i]:9.2f} {std_text} {tape:9.1f} {stats['mean_mm'][i] - tape:+11.2f}")
        lines.append(f"Pooled std, the one noise number the fit uses: {_mm(stats['pooled_std_mm'])} mm.")
        if args.range_m is not None:
            taped = args.range_m * 1000.0
            difference = stats["range_mean_mm"] - taped
            lines.append(f"Range: measured mean {stats['range_mean_mm']:.1f} mm, taped {taped:.1f} mm, "
                         f"difference {difference:+.1f} mm ({difference / taped * 100.0:+.2f}%).")
    proc = [o["processing_time_s"] - o["capture_time_s"] for o in observations]
    if proc:
        lines.append(f"Time from frame arrival to result: median {np.median(proc) * 1000:.1f} ms, "
                     f"max {max(proc) * 1000:.1f} ms.")
    if recording.stopped_early:
        lines.append("Stopped early. The frames recorded so far are in the log.")
    lines.append(f"Log:    {paths['log']}")
    lines.append(f"Frames: {paths['frames']}")
    if live and paths["raw"].exists() and args.video_codec != "none":
        lines.append(f"Video:  {paths['raw']}  (large, keep it out of the repo)")
    lines.extend(fit_command(paths))
    return lines


def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


def confirm_camera(args: argparse.Namespace, ask: Callable[[str], str], size: tuple[int, int],
                   preview_on: bool, log: Callable[[str], None]) -> bool | None:
    """Ask if the frames come from the Mac's own camera. None means not asked (--yes)."""
    if args.yes:
        return None
    if preview_on:
        question = "Does the preview show the Mac's own camera? [y/N] "
    else:
        question = (f"Camera index {args.camera_index} gives {size[0]}x{size[1]} frames. "
                    "Is it the Mac's own camera? [y/N] ")
    try:
        answer = _yes(ask(question))
    except EOFError:
        answer = False
    if not answer:
        log("Stopped: not confirmed as the Mac's own camera. Nothing was recorded. Try --camera-index 1, "
            "or add --yes if you are sure.")
    return answer


def _report_cameras(names: list[str] | None, log: Callable[[str], None]) -> None:
    if names is None:
        log("Could not list the cameras (system_profiler is only on macOS).")
    else:
        log("Cameras macOS lists: " + (", ".join(names) if names else "none") + ".")
    log("OpenCV does not promise an order, so camera index 0 may be an iPhone (Continuity Camera) or a "
        "virtual camera, not the Mac's own camera.")


def run_print_marker(parser: argparse.ArgumentParser, args: argparse.Namespace,
                     log: Callable[[str], None]) -> int:
    target = args.print_marker
    if target.suffix.lower() != ".png":
        parser.error("--print-marker must be a file name ending in .png")
    if not target.resolve().parent.is_dir():
        parser.error(f"the folder for --print-marker does not exist: {target.resolve().parent}")
    if target.exists() and not args.overwrite:
        parser.error(f"{target} already exists; use --overwrite to replace it")
    try:
        cv2 = load_cv2()
        if args.marker_id >= dictionary_size(get_dictionary(cv2, args.dictionary)):
            parser.error(f"--marker-id {args.marker_id} is too big for {args.dictionary}")
        layout = print_marker(cv2, target, args.dictionary, args.marker_id, args.marker_print_mm,
                              args.marker_margin_mm)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total_mm = layout["side_mm"] + 2 * layout["margin_mm"]
    log(f"Wrote {target}: {args.dictionary} id {args.marker_id}. At 100% the black square prints "
        f"{layout['side_mm']:.1f} mm wide with a {layout['margin_mm']:.1f} mm white margin "
        f"({total_mm:.0f} mm in all).")
    log("Print at 100% (actual size) with fit to page turned off. Then measure the black square on both "
        "edges. If it is not the size above, use the measured value for --marker-length-m.")
    if total_mm > PRINTABLE_WIDTH_MM:
        log(f"WARNING: {total_mm:.0f} mm may not fit on letter or A4 paper. Use a smaller --marker-print-mm "
            "or --marker-margin-mm.")
    return 0


def main(argv: list[str] | None = None, *, source_factory: Callable[..., Any] | None = None,
         clock: Callable[[], float] = time.monotonic, ask: Callable[[str], str] | None = None,
         log: Callable[[str], None] = print, camera_lister: Callable[[], list[str] | None] | None = None) -> int:
    """Run the tool. source_factory, clock, ask, log and camera_lister let tests run without a camera."""
    parser = build_parser()
    args = parser.parse_args(argv)
    ask = ask or input
    if args.print_marker is not None:
        return run_print_marker(parser, args, log)
    mode = "check" if args.check_camera else ("replay" if args.dry_run is not None else "live")
    if mode == "live":
        _fill_missing_live_values(parser, args, ask)
    validate_args(parser, args, mode)

    try:
        cv2 = load_cv2()
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.marker_id >= dictionary_size(get_dictionary(cv2, args.dictionary)):
        parser.error(f"--marker-id {args.marker_id} is too big for {args.dictionary}")

    listed: list[str] | None = None
    if mode in ("live", "check"):
        listed = (camera_lister or list_cameras)()
        _report_cameras(listed, log)

    try:
        if source_factory is not None:
            raw_source = source_factory(args, cv2)
        elif mode == "replay":
            raw_source = open_video(cv2, args.dry_run)
        else:
            raw_source = open_camera(cv2, args.camera_index, args.width, args.height, args.fps or DEFAULT_FPS)
    except ValueError as exc:
        parser.error(str(exc))
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    source = PeekedSource(raw_source)
    try:
        try:
            ok, first = source.peek()
        except Exception as exc:  # noqa: BLE001  any camera failure becomes a plain message
            print(f"error: reading the first frame failed: {exc}", file=sys.stderr)
            return 1
        if not ok or first is None:
            print("error: the frame source gave no frames at all.", file=sys.stderr)
            return 1
        size = (int(first.shape[1]), int(first.shape[0]))  # the real size, whatever was requested
        camera, calibration_sha = _camera_for(parser, args, size)
        uncertainty = (tuple(args.truth_uncertainty_xyz_m) if args.truth_uncertainty_xyz_m is not None
                       else (args.truth_uncertainty_m if args.truth_uncertainty_m is not None
                             else DEFAULT_TRUTH_UNCERTAINTY_M))
        truth = Truth((float(args.lateral_m), float(args.vertical_m), float(args.distance_m or 0.0)),
                      uncertainty, args.truth_method)
        ctx = build_context(cv2, dictionary_name=args.dictionary, marker_id=args.marker_id,
                            marker_length_m=args.marker_length_m, corner_refinement=args.corner_refinement,
                            camera=camera, truth=truth)
        if mode == "check":
            return run_check(args, cv2, source, ctx, size=size, clock=clock, ask=ask, log=log)
        return run_capture(args, cv2, source, ctx, mode=mode, size=size, calibration_sha=calibration_sha,
                           listed=listed, clock=clock, ask=ask, log=log, parser=parser)
    finally:
        source.release()


def run_check(args: argparse.Namespace, cv2: Any, source: PeekedSource, ctx: Context, *, size: tuple[int, int],
              clock: Callable[[], float], ask: Callable[[str], str], log: Callable[[str], None]) -> int:
    """--check-camera: live numbers, then the frame size, rate and gaps. No files are written."""
    source.drop_pending()  # the first frame was only read for its size; its read time is unknown
    preview = None if args.no_preview else Preview(cv2, log)
    preview_on = bool(preview is not None and preview.enabled)
    log("Checking the camera. " + ("Press q in the preview to stop." if args.check_seconds is None
                                    else f"Stopping after {args.check_seconds:g} s.") + " Nothing is saved.")
    try:
        result = check_camera_loop(source, ctx, clock=clock, preview=preview, log=log, seconds=args.check_seconds)
    except Exception as exc:  # noqa: BLE001  any camera failure becomes a plain message
        print(f"error: the camera check stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()
    timing = geo.summarize_frame_timing(result.capture_times_s, None)
    shape = result.frame_shape or tuple(int(v) for v in (size[1], size[0]))
    rate, median = timing["measured_frame_rate_hz"], timing["median_interval_s"]
    matrix = ctx.camera.matrix()
    log(f"Frame size read from the frames: {shape[1]}x{shape[0]}. Requested: {args.width}x{args.height}.")
    if (shape[1], shape[0]) != (args.width, args.height):
        log("The camera runs at a different size than requested. Use this real size everywhere, "
            "including the calibration.")
    log(f"Frames: {timing['frames']}, measured frame rate: " + ("unknown" if rate is None else f"{rate:.2f} Hz")
        + ", median interval: " + ("unknown" if median is None else f"{median * 1000:.1f} ms")
        + f", gaps (longer than 1.5 x the median interval): {timing['late_frames']}.")
    log(f"Marker found in {result.detected} of {timing['frames']} frames. Crosshair at "
        f"cx={matrix[0, 2]:.1f}, cy={matrix[1, 2]:.1f} px ({ctx.camera.description}).")
    if result.last_readout:
        log("Last reading: " + "   ".join(result.last_readout))
    answer = confirm_camera(args, ask, (shape[1], shape[0]), preview_on, log)
    return 2 if answer is False else 0


def run_capture(args: argparse.Namespace, cv2: Any, source: PeekedSource, ctx: Context, *, mode: str,
                size: tuple[int, int], calibration_sha: str | None, listed: list[str] | None,
                clock: Callable[[], float], ask: Callable[[str], str], log: Callable[[str], None],
                parser: argparse.ArgumentParser) -> int:
    """Live capture or replay, then the log. Every file and window is closed on every way out."""
    live = mode == "live"
    paths = output_paths(args.output)
    code_revision = git_revision()
    captured_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    trace_id = args.trace_id or f"aruco-{args.condition}-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    frame_times: dict[int, dict[str, Any]] | None = None
    not_in_video = 0
    if not live and args.frame_times is not None:
        try:
            frame_times, not_in_video = load_frame_times(args.frame_times)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    video_fps = _float_or_none(source.get(cv2.CAP_PROP_FPS))
    replay_fps = args.fps or video_fps
    if not live and frame_times is None and not replay_fps:
        parser.error("the video does not store a frame rate; give --fps or --frame-times")

    recording = Recording()
    writer: RawVideoWriter | None = None
    sidecar: SidecarWriter | None = None
    preview: Preview | None = None
    error: Exception | None = None
    writer_error: CaptureError | None = None
    camera_confirmed: bool | None = None
    frames_wanted = args.frames or (DEFAULT_FRAMES if live else None)
    try:
        if live:
            source.drop_pending()  # the frame read for its size counts as the first warmup frame
            recording.warmup_done = 1
            if not args.no_preview:
                preview = Preview(cv2, log)
            log(f"Camera gives {size[0]}x{size[1]} frames (asked for {args.width}x{args.height}). "
                f"Warming up for {args.warmup_frames} frames.")
            done, quit_pressed = discard_frames(source, count=max(args.warmup_frames - 1, 0), clock=clock,
                                                preview=preview, log=log, label="warming up")
            recording.warmup_done += done
            if quit_pressed:
                log("Stopped before recording started. No files were written.")
                return 0
            camera_confirmed = confirm_camera(args, ask, size, bool(preview and preview.enabled), log)
            if camera_confirmed is False:
                return 2
            if not args.yes:
                try:
                    ask("Press Enter, then step away. ")
                except EOFError:
                    log("No keyboard to press Enter on. Add --yes to skip the prompt. Nothing was recorded.")
                    return 2
            sidecar = SidecarWriter(paths["frames"])
            if args.video_codec != "none":
                writer = open_raw_video(cv2, paths["raw"], args.video_codec, args.fps or video_fps or DEFAULT_FPS,
                                        size, args.queue_frames)
            if args.countdown_s > 0:
                log(f"Recording starts in {args.countdown_s:g} s.")
            done, quit_pressed = discard_frames(source, seconds=args.countdown_s, clock=clock, preview=preview,
                                                log=log, label="recording starts in")
            recording.countdown_done = done
            if quit_pressed:
                recording.stopped_early = True
            else:
                log(f"Recording {frames_wanted} frames of {ctx.subject_id}. Keep still.")
                record_frames(source, ctx, frames=frames_wanted, clock=clock, writer=writer, sidecar=sidecar,
                              preview=preview, log=log, recording=recording)
        else:
            sidecar = SidecarWriter(paths["frames"])
            log(f"Replaying {args.dry_run} through the detector. No camera is used.")
            replay_frames(source, ctx, frame_times=frame_times, fps=replay_fps, max_frames=frames_wanted,
                          sidecar=sidecar, recording=recording)
    except KeyboardInterrupt:
        recording.stopped_early = True
    except Exception as exc:  # noqa: BLE001  CaptureError, cv2.error or anything else from the camera
        error = exc
    finally:
        if preview is not None:
            preview.close()
        if writer is not None:
            try:
                writer.close()
            except CaptureError as exc:
                writer_error = exc
        if sidecar is not None:
            sidecar.close()
    error = error or recording.error

    observations = recording.observations
    if not observations:
        if error is not None:
            print(f"error: {_describe(error)}", file=sys.stderr)
        if error is not None or not recording.stopped_early:
            where = f" The frames file {paths['frames']} was kept." if paths["frames"].exists() else ""
            print("error: no frames were recorded, so there is no log to write." + where, file=sys.stderr)
            return 1
        log("Stopped before any frame was recorded. No log was written.")
        return 0

    sidecar_sha = sha256_file(paths["frames"])
    times = [o["capture_time_s"] for o in observations]
    nominal = replay_fps if replay_fps else (args.fps or DEFAULT_FPS)
    timing = geo.summarize_frame_timing(times, nominal)
    counts = count_outcomes(observations)
    stats = position_stats(observations)
    truth = ctx.truth
    marker_text = (f"Marker {args.dictionary} id {args.marker_id}, black square {args.marker_length_m:g} m, "
                   f"corner refinement {args.corner_refinement}. Truth: marker center at x={truth.position_m[0]:g}, "
                   f"y={truth.position_m[1]:g} (down), z={truth.position_m[2]:g} m in the camera frame"
                   + ("" if args.range_m is None else f", taped range {args.range_m:g} m")
                   + f", {truth.method}, uncertainty x={truth.uncertainty_xyz_m[0]:g}, "
                   f"y={truth.uncertainty_xyz_m[1]:g}, z={truth.uncertainty_xyz_m[2]:g} m (each log line "
                   f"carries the largest, {truth.uncertainty_m:g} m, because the fit uses one number for all axes)")
    extra: list[str] = []
    if live:
        extra.append(f"{recording.warmup_done} warmup frames and {recording.countdown_done} countdown frames "
                     "were read and thrown away before recording.")
    if live and recording.stopped_early:
        extra.append("Recording was stopped early by the user.")
    if live and recording.stream_ended:
        extra.append("The camera stopped giving frames before the requested count (end of stream).")
    if error is not None:
        extra.append(f"The run ended on an error after {len(observations)} frames: {_describe(error)}. "
                     "This log holds the frames recorded before it.")
    if writer is not None and writer.dropped:
        extra.append(f"WARNING: {writer.dropped} frames were not saved to the raw video because the writer "
                     "queue was full; they are in this log but a replay of the video will skip them.")
    if writer_error is not None:
        extra.append(f"WARNING: the raw video is not trustworthy: {writer_error}.")
    extra.append(f"The frames file {paths['frames'].name} has SHA-256 {sidecar_sha}.")
    if live and writer is None:
        extra.append("No raw video was saved (--video-codec none), so raw_sha256 is the SHA-256 of the "
                     "frames file.")
    if not live and frame_times is not None:
        extra.append(f"Capture times came from {args.frame_times.name}.")
        if not_in_video:
            extra.append(f"{not_in_video} rows of that file are for frames that were not in the video.")
        if recording.frame_times_past_end:
            extra.append(f"{recording.frame_times_past_end} rows of that file are past the end of the video.")
    elif not live:
        extra.append(f"Capture times were made up from a frame rate of {replay_fps:g} Hz.")
    if args.notes:
        extra.append(f"Operator notes: {args.notes}")
    notes = build_notes(live=live, condition=args.condition, marker_text=marker_text, camera=ctx.camera,
                        timing=timing, counts=counts, stats=stats, extra=extra)
    frame_rate = timing["measured_frame_rate_hz"] or nominal
    refine = REFINEMENT_NAMES[args.corner_refinement].removeprefix("CORNER_REFINE_")
    capture: dict[str, Any] = {
        "source_type": "camera" if live else "recording",
        "source_id": args.source_id or ("mac-built-in-camera" if live else Path(args.dry_run).name),
        "device_ids": [args.device_id],
        "model": f"opencv-aruco {cv2.__version__} {args.dictionary} IPPE_SQUARE+SQPNP refine={refine}",
        "resolution_px": {"width": size[0], "height": size[1]},
        "frame_rate_hz": float(frame_rate),
    }
    if args.calibration is not None:
        capture["calibration_id"] = args.calibration.name
    provenance = None
    if live:
        settings = {
            "requested_width_px": args.width, "requested_height_px": args.height,
            "requested_fps": args.fps or DEFAULT_FPS,
            "actual_width_px": size[0], "actual_height_px": size[1], "backend_reported_fps": video_fps,
            "backend": "avfoundation" if sys.platform == "darwin" else "default",
            "camera_index": args.camera_index,
            "listed_cameras": listed if listed is not None else [],
            "camera_confirmed": "not asked (--yes)" if camera_confirmed is None else "yes, answered at the prompt",
            "warmup_frames": recording.warmup_done, "countdown_s": args.countdown_s,
            "countdown_frames": recording.countdown_done,
            "dictionary": args.dictionary, "marker_id": args.marker_id,
            "marker_side_m": args.marker_length_m, "corner_refinement": args.corner_refinement,
            "opencv_version": cv2.__version__, "video_codec": args.video_codec,
            "video_frames_dropped": 0 if writer is None else writer.dropped,
            **MACOS_SETTINGS,
        }
        null_reason = None
        if calibration_sha is None:
            null_reason = ("no calibration file; focal length from a "
                           f"{args.hfov_deg:g} degree horizontal field of view guess")
        raw_sha = sha256_file(paths["raw"]) if writer is not None and paths["raw"].exists() else sidecar_sha
        provenance = build_provenance(
            camera_model=args.camera_model, settings=settings, calibration_ref=calibration_sha,
            calibration_null_reason=null_reason, raw_sha256=raw_sha, code_revision=code_revision,
            captured_at=captured_at, operator=args.operator, condition=args.condition, truth=truth,
            marker_side_m=args.marker_length_m, range_m=args.range_m)
    source_name = paths["log"].name if live else Path(args.dry_run).name
    header = build_header(live=live, trace_id=trace_id, source=source_name, capture=capture,
                          clock_domain="shared_monotonic", notes=notes, code_revision=code_revision,
                          provenance=provenance)
    try:
        write_log_atomically(make_trace(header, observations), paths)
    except (ValueError, OSError) as exc:
        print(f"error: could not write the log: {exc}. The frames file {paths['frames']} was kept.",
              file=sys.stderr)
        return 1
    for line in _summary_lines(args, recording, timing, observations, paths, truth, live, size):
        log(line)
    if error is not None or writer_error is not None:
        problem = error if error is not None else writer_error
        print(f"error: the run ended on an error: {_describe(problem)}. The log keeps the "
              f"{len(observations)} frames recorded before it.", file=sys.stderr)
        return 1
    return 0


def _describe(error: BaseException | None) -> str:
    if error is None:
        return "unknown error"
    if isinstance(error, CaptureError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
