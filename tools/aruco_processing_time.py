#!/usr/bin/env python3
r"""Time the ArUco detector on saved frames or a replay video and write a processing-time file.

The file is schema zephyr-processing-time-1. tools/vision_latency_to_link.py
adds its median to the camera latency to get the vision delay D.

STATUS: a proposal. The statistics, the writer and the check are tested with a
fake timer. The OpenCV path (reading frames and running the real detector) has
NOT been run on a real camera or in the camera environment yet. Until it has,
treat the numbers from it as unchecked.

What is timed, per frame: grayscale conversion, marker detection and the pose
solve, which is the same work tools/capture_aruco.py does between a frame and
its log line. Decoding the video, reading the camera and drawing the preview
are not timed. The timing is on this machine, in this OpenCV build. A median
from a replay is a good stand-in for the live cost only if the replay frames
look like the live ones (same size, same marker distance).

Every command runs from the repo folder, with the camera environment's Python
(the repo's .venv has no OpenCV):

    CAM="$HOME/zephyr-camera-venv/bin/python"
    "$CAM" tools/aruco_processing_time.py measure FRAMES_FOLDER_OR_VIDEO \
        --out "$HOME/zephyr-data/aruco/processing_time.json" --operator "Your Name"
    "$CAM" tools/aruco_processing_time.py check "$HOME/zephyr-data/aruco/processing_time.json"

measure writes two files: the JSON and a .timings.csv beside it with every raw
timing. The JSON's provenance.raw_sha256 is the hash of that CSV's rows, and
`check` recomputes the median, p95 and n from the CSV and compares all three
(and the hash) with the JSON. That catches an accidental edit to the timings or
to those numbers. It is not a signature: it does not cover the other JSON
fields (operator, machine, cv2_version), and someone who edits the CSV on
purpose can recompute the hash too.

A file that says "measured" must carry its provenance (operator, captured_at,
code_revision, raw_sha256). tools/vision_latency_to_link.py, and so
tools/intake_vision.py, refuse it otherwise. tools/latency_budget.py has its own
processing reader and does not check provenance yet.

Percentiles use the nearest-rank rule: p95 of n timings is the value at rank
ceil(0.95 * n) in sorted order. It is always one of the real timings.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vision_latency_to_link import (  # noqa: E402
    PROCESSING_SCHEMA,
    LinkConversionError,
    _processing_from_json,
)

SCHEMA = PROCESSING_SCHEMA
STAGE = "grayscale + detectMarkers + pose (the per-frame work in tools/capture_aruco.py analyze_frame)"
SCOPE = ("timed on this machine, one frame at a time, with the detector only; camera read, video decode and "
         "preview are not included")
PERCENTILE_METHOD = "nearest-rank"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_WARMUP = 10
DEFAULT_MAX_FRAMES = 600


class ProcessingTimeError(ValueError):
    pass


# ------------------------------------------------------------------ pure statistics

def _check_timings(timings: Iterable[Any]) -> list[float]:
    values: list[float] = []
    for value in timings:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProcessingTimeError(f"timings must be numbers, got {value!r}")
        if not math.isfinite(value) or value < 0:
            raise ProcessingTimeError(f"timings must be finite and zero or more, got {value!r}")
        values.append(float(value))
    if not values:
        raise ProcessingTimeError("no timings to summarize")
    return values


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def nearest_rank(values: list[float], fraction: float) -> float:
    """The value at rank ceil(fraction * n) of the sorted values (1-based)."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(fraction * len(ordered) - 1e-12)) - 1]


def summarize(timings: Iterable[Any]) -> dict[str, Any]:
    values = _check_timings(timings)
    return {"median_s": median(values), "p95_s": nearest_rank(values, 0.95), "n": len(values),
            "min_s": min(values), "max_s": max(values), "mean_s": sum(values) / len(values)}


def time_calls(frames: Iterable[Any], detect: Callable[[Any], Any], *, timer: Callable[[], float] = time.perf_counter,
               warmup: int = 0, max_frames: int | None = None,
               after: Callable[[int, Any], None] | None = None) -> list[float]:
    """Time detect(frame) for each frame. The first `warmup` frames run but are not counted.

    max_frames limits the counted frames. Only the call itself sits between the two timer reads.
    after(index, result), if given, runs once the second timer read is done (index counts from 1, warm-up
    included), so bookkeeping on the result never lands inside the timing.
    """
    if warmup < 0:
        raise ProcessingTimeError("warmup must be zero or more")
    timings: list[float] = []
    seen = 0
    for frame in frames:
        seen += 1
        start = timer()
        result = detect(frame)
        stop = timer()
        if after is not None:
            after(seen, result)
        if stop < start:
            raise ProcessingTimeError("the timer went backwards; use a monotonic timer")
        if seen > warmup:
            timings.append(stop - start)
            if max_frames is not None and len(timings) >= max_frames:
                break
    if seen == 0:
        raise ProcessingTimeError("no frames to time")
    if not timings:
        raise ProcessingTimeError(f"only {seen} frame(s) were read, all used as warm-up; need more than {warmup}")
    return timings


def _timing_lines(timings: list[float]) -> list[str]:
    return [f"{t:.9f}" for t in timings]


def timings_sha256(timings: list[float]) -> str:
    """Hash of the raw timings: one line per timing, nine decimals, in capture order."""
    text = "".join(line + "\n" for line in _timing_lines(timings))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ report and files

def build_report(timings: list[float], *, status: str, cv2_version: str, dictionary: str, machine: dict[str, Any],
                 source: dict[str, Any], warmup_frames: int, operator: str, captured_at: str, code_revision: str,
                 frames_with_marker: int | None = None) -> dict[str, Any]:
    if status not in ("measured", "synthetic"):
        raise ProcessingTimeError("status must be measured or synthetic")
    if status == "measured":
        for name, value in (("operator", operator), ("captured_at", captured_at), ("code_revision", code_revision)):
            if not isinstance(value, str) or not value.strip():
                raise ProcessingTimeError(f"a measured file needs {name}")
    stats = summarize(timings)
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": status, **stats, "percentile_method": PERCENTILE_METHOD,
        "stage": STAGE, "scope": SCOPE, "cv2_version": cv2_version, "dictionary": dictionary,
        "machine": machine, "source": source, "warmup_frames": warmup_frames,
        "provenance": {"operator": operator, "captured_at": captured_at, "code_revision": code_revision,
                       "raw_sha256": timings_sha256(timings)},
    }
    if frames_with_marker is not None:
        report["frames_with_marker"] = frames_with_marker
    return report


def sidecar_path(out: Path) -> Path:
    return out.with_name(out.stem + ".timings.csv")


def write_timings_csv(path: Path, timings: list[float], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise ProcessingTimeError(f"{path.name} exists; pick a new name")
    path.write_text("seconds\n" + "".join(line + "\n" for line in _timing_lines(timings)), encoding="utf-8", newline="\n")


def write_report(out: Path, report: dict[str, Any], timings: list[float]) -> None:
    csv_path = sidecar_path(out)
    for path in (out, csv_path):
        if path.exists():
            raise ProcessingTimeError(f"{path.name} exists; pick a new name (nothing is overwritten)")
    write_timings_csv(csv_path, timings)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def check_report(path: Path) -> list[str]:
    """Return a list of problems; empty means the JSON, its provenance and its raw timings agree."""
    if not path.is_file():
        raise ProcessingTimeError(f"{path.name}: file not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProcessingTimeError(f"{path.name}: not readable JSON ({exc})") from exc
    if not isinstance(data, dict):
        return [f"{path.name}: the JSON is not an object"]
    problems: list[str] = []
    try:
        _processing_from_json(data)
    except LinkConversionError as exc:
        problems.append(str(exc))
    csv_path = sidecar_path(path)
    if not csv_path.is_file():
        problems.append(f"timings file {csv_path.name} not found, so the numbers cannot be rechecked")
        return problems
    try:
        rows = csv_path.read_text(encoding="utf-8").split()
        timings = [float(v) for v in rows[1:]]
        stats = summarize(timings)
    except (OSError, ValueError) as exc:
        problems.append(f"timings file {csv_path.name} is not usable ({exc})")
        return problems
    if data.get("n") != stats["n"]:
        problems.append(f"n is {data.get('n')} but the timings file has {stats['n']} rows")
    for key in ("median_s", "p95_s"):
        value = data.get(key)
        if not isinstance(value, (int, float)) or abs(value - stats[key]) > 1e-9:
            problems.append(f"{key} is {value} but the timings file gives {stats[key]}")
    prov = data.get("provenance")
    if isinstance(prov, dict) and prov.get("raw_sha256") != timings_sha256(timings):
        problems.append("provenance.raw_sha256 does not match the timings file (sha256 mismatch)")
    return problems


# ------------------------------------------------------------------ OpenCV side (needs the camera environment)

def _load_cv2() -> Any:
    from tools import capture_aruco as cap
    try:
        return cap.load_cv2()
    except cap.CaptureError as exc:
        raise ProcessingTimeError(str(exc)) from exc


def iter_frames(cv2: Any, source: Path) -> Iterator[Any]:
    """Yield frames from a folder of images (sorted by name) or a video file, one at a time."""
    if source.is_dir():
        # names starting with "." are macOS resource-fork leftovers, not images
        names = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))
        if not names:
            raise ProcessingTimeError(f"{source.name}: no image files ({', '.join(sorted(IMAGE_SUFFIXES))})")
        for path in names:
            # color, like the BGR frames the live capture feeds the detector (a grayscale PNG would skip the conversion)
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ProcessingTimeError(f"could not read image {path.name}")
            yield frame
    elif source.is_file():
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ProcessingTimeError(f"could not open video {source.name}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()
    else:
        raise ProcessingTimeError(f"{source}: not found")


def machine_info(cv2: Any) -> dict[str, Any]:
    import os
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        info["opencv_threads"] = int(cv2.getNumThreads())
    except Exception:  # an OpenCV build without the call; the rest of the record is still useful
        pass
    return info


def measure(args: argparse.Namespace) -> int:
    from tools import aruco_geometry as geo
    from tools import capture_aruco as cap
    cv2 = _load_cv2()
    out = Path(args.out)
    for path in (out, sidecar_path(out)):
        if path.exists():
            raise ProcessingTimeError(f"{path.name} exists; pick a new name (nothing is overwritten)")
    frames = iter_frames(cv2, Path(args.source))
    try:
        first = next(frames)
    except StopIteration:
        raise ProcessingTimeError("the source has no frames") from None
    height, width = first.shape[:2]
    camera = geo.camera_from_hfov(width, height, args.hfov_deg)
    truth = cap.Truth((0.0, 0.0, 1.0), cap.DEFAULT_TRUTH_UNCERTAINTY_M, "none")
    ctx = cap.build_context(cv2, dictionary_name=args.dictionary, marker_id=args.marker_id,
                            marker_length_m=args.marker_length_m, corner_refinement=args.corner_refinement,
                            camera=camera, truth=truth)
    counts = {"read": 0, "marker": 0}

    def stream() -> Iterator[Any]:
        yield first
        yield from frames

    def count(index: int, result: Any) -> None:
        counts["read"] = index
        if index > args.warmup and result.outcome == "detected":
            counts["marker"] += 1

    with contextlib.closing(stream()) as feed:  # closes the video file if max_frames stops the loop early
        timings = time_calls(feed, lambda frame: cap.analyze_frame(frame, ctx), warmup=args.warmup,
                             max_frames=args.max_frames, after=count)
    kind = "frames" if Path(args.source).is_dir() else "video"
    report = build_report(
        timings, status=args.status, cv2_version=cv2.__version__, dictionary=args.dictionary,
        machine=machine_info(cv2),
        source={"kind": kind, "name": Path(args.source).name, "frames_read": counts["read"],
                "frame_size_px": [int(width), int(height)]},
        warmup_frames=args.warmup, operator=args.operator,
        captured_at=args.captured_at or datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        code_revision=args.code_revision or cap.git_revision(), frames_with_marker=counts["marker"])
    write_report(out, report, timings)
    print(f"{report['n']} frames timed ({args.warmup} warm-up skipped): median {1000 * report['median_s']:.2f} ms, "
          f"p95 {1000 * report['p95_s']:.2f} ms, marker found in {counts['marker']} of them")
    print(f"wrote {out.name} and {sidecar_path(out).name}")
    if counts["marker"] < report["n"]:
        print("warning: the marker was not found in every frame; frames without a marker are usually cheaper, "
              "so the median may be low for a scene where the marker is always in view", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ command line

def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _hfov(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0 < value < 180:
        raise argparse.ArgumentTypeError("must be between 0 and 180 degrees")
    return value


def build_parser() -> argparse.ArgumentParser:
    from tools import capture_aruco as cap  # for the dictionary and refinement choices
    parser = argparse.ArgumentParser(description="Time the ArUco detector per frame (zephyr-processing-time-1).")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="time the detector on a folder of images or a video, write the JSON")
    m.add_argument("source", help="folder of image files, or a video file")
    m.add_argument("--out", required=True, help="processing-time JSON to write (must not exist)")
    m.add_argument("--operator", required=True, help="your name, written into the provenance")
    m.add_argument("--status", choices=("measured", "synthetic"), required=True,
                   help="measured only for frames the real camera recorded; synthetic for generated test frames")
    m.add_argument("--dictionary", default="DICT_4X4_50", choices=cap.DICTIONARY_NAMES, metavar="NAME",
                   help="marker dictionary (default DICT_4X4_50)")
    m.add_argument("--marker-id", type=_nonnegative_int, default=0)
    m.add_argument("--marker-length-m", type=_positive, default=0.150,
                   help="black square edge in meters; only the pose solve uses it (default 0.150)")
    m.add_argument("--hfov-deg", type=_hfov, default=70.0,
                   help="guessed horizontal field of view for the pose solve; its cost barely depends on it (default 70)")
    m.add_argument("--corner-refinement", default="subpix", choices=sorted(cap.REFINEMENT_NAMES), metavar="METHOD",
                   help="corner refinement, same as capture_aruco (default subpix)")
    m.add_argument("--warmup", type=_nonnegative_int, default=DEFAULT_WARMUP,
                   help=f"frames to run but not count (default {DEFAULT_WARMUP})")
    m.add_argument("--max-frames", type=_positive_int, default=DEFAULT_MAX_FRAMES,
                   help=f"stop after this many counted frames (default {DEFAULT_MAX_FRAMES})")
    m.add_argument("--captured-at", help="ISO time of the capture (default: now)")
    m.add_argument("--code-revision", help="override the git revision written into the provenance")
    c = sub.add_parser("check", help="recheck a processing-time JSON against its timings file")
    c.add_argument("file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            problems = check_report(Path(args.file))
            if problems:
                for problem in problems:
                    print(f"problem: {problem}", file=sys.stderr)
                return 1
            print(f"{Path(args.file).name}: ok (the timings file matches n, median_s, p95_s and the hash"
                  "; for a measured file the provenance block is complete)")
            return 0
        return measure(args)
    except ProcessingTimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
