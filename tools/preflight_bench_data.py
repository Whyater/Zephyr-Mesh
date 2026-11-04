"""Check a bench data folder before intake, and stop loudly if it would give misleading numbers.

Runs in the repo's .venv (NumPy only; no camera, no radio, no device). It reads files and writes nothing.

    PYTHONPATH=. .venv/bin/python tools/preflight_bench_data.py "$D" [--output "$D/out/intake-pilot"]
        [--frame-size 1280x720] [--min-detection-rate 0.5]

Give it the bench data folder (the one holding aruco/, latency/ and radio/ in the runbook). It looks through
every subfolder, works out what each file is from its contents, and checks the things intake depends on:

- detection logs: the same reader and provenance check the intake uses, the detection rate, the ground truth
  against the file name (d2.0 means a tape value that rounds to 2.0 m, center means x = 0), the detected
  distance and sideways position against the recorded truth, and
  the raw video and frames file against the sha256 values written into the log;
- calibration files: readable, unchanged since the capture, and the same frame size as the logs;
- the processing-time file: the same check the intake uses, plus its timings file and sha256;
- latency run folders: the intake's own check, plus the recorded frames;
- radio logs: the importer's reader, and one sender and one echoer per folder;
- across the folder: one status for every input, one session (calibration, frame size, code revision), no
  two inputs whose intake outputs would collide, and, with --output, an intake output folder that is new.

Each problem is printed as FAIL (intake would refuse, or would run and give numbers that mislead) or WARN
(worth knowing, the numbers stay honest), with one line saying what to do. Folders that hold an INTAKE.md
are intake outputs and are skipped.

Exit code: 0 when nothing failed (warnings allowed), 1 when anything failed, 2 for a usage error.
A pass means the files are consistent. It does not mean the numbers in them are good.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.detection_log import DETECTION_LOG_SCHEMA, read_detection_log
from tools import aruco_processing_time, extract_echo_log, import_espnow_log, vision_latency
from tools.aruco_geometry import CALIBRATION_SCHEMA, load_camera_calibration
from tools.intake_vision import IntakeError, _check_latency_run, _check_log, _check_output
from tools.vision_latency_to_link import (
    PROCESSING_SCHEMA,
    LinkConversionError,
    _processing_from_json,
    combine_status,
)

DEFAULT_MIN_DETECTION_RATE = 0.5   # runbook step 4: under half the frames detected means fix the setup
# Truth against what the camera sees, on the median over the log's frames. Frame-to-frame ArUco noise is a few
# mm at 1 m and more at 2 m, mostly in z (tools/capture_aruco.py), and the median of hundreds of frames shrinks
# it far below that, so these limits are set by the honest steady errors instead:
# - z (relative): marker length and focal length errors scale z by a percent or two. 10% allows several times
#   that. A stale DIST is 50 to 100% off (0.5, 1.0 and 2.0 m crosses).
# - x (absolute plus a part of z): 2 cm for tape and centering the marker on the cross, plus 5% of z for an
#   axis line a few degrees off the lens axis (tan 3 deg = 0.05) or a principal point tens of px off. Side
#   crosses sit at x = 427 z / fx, so x doubles from one cross to the next and a stale LAT is off by half or
#   all of the true x: 0.107 m at the 0.5 and 1 m crosses, where the limit is 0.045 and 0.07 m, and 0.22 m at
#   2 m, where it is 0.12 m. Center crosses (x = 0) get the 2 cm term plus 5% of z.
MAX_DISTANCE_ERROR = 0.10          # runbook step 3: live z should be within a few percent of the tape
LATERAL_ERROR_M = 0.02
LATERAL_ERROR_PER_M_OF_Z = 0.05
MAX_RMS_PX = 0.5                   # runbook step 3: calibration RMS under about 0.5 px
MAX_SESSION_SPAN_H = 12.0
CENTER_TOLERANCE_M = 0.0005        # the runbook types LAT=0 for a center cross
RADIO_SUFFIXES = {".csv", ".txt", ".raw", ".log"}
RADIO_FORMAT = import_espnow_log.FORMAT_LINE
FRAMES_NOTE = re.compile(r"The frames file (\S+) has SHA-256 ([0-9a-f]{64})")
NEVER_EDIT = "never edit a bench file by hand"


@dataclass
class Finding:
    level: str   # FAIL or WARN
    code: str
    file: str
    reason: str
    fix: str


@dataclass
class Report:
    folder: Path
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    ignored: int = 0

    def fail(self, code: str, file: str, reason: str, fix: str) -> None:
        self.findings.append(Finding("FAIL", code, file, _one_line(reason), _one_line(fix)))

    def warn(self, code: str, file: str, reason: str, fix: str) -> None:
        self.findings.append(Finding("WARN", code, file, _one_line(reason), _one_line(fix)))

    @property
    def failed(self) -> bool:
        return any(f.level == "FAIL" for f in self.findings)


@dataclass
class LogInfo:
    path: Path
    status: str
    sha256: str
    resolution: tuple[int, int] | None
    calibration_ref: str | None
    calibration_id: str | None
    code_revision: str | None
    captured_at: datetime | None


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(value: Any) -> str:
    return value if value in ("measured", "synthetic", "fixture") else "unknown"


# ---------------------------------------------------------------- finding the files

@dataclass
class Found:
    logs: list[Path] = field(default_factory=list)
    calibrations: list[Path] = field(default_factory=list)
    processing: list[Path] = field(default_factory=list)
    latency_runs: list[Path] = field(default_factory=list)
    radio: list[Path] = field(default_factory=list)


def _first_line(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.readline(1 << 16)


def _json_schema(path: Path) -> str | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("schema") if isinstance(data, dict) else None


def find_inputs(folder: Path, report: Report) -> Found:
    found = Found()
    stack = [folder]
    while stack:
        current = stack.pop()
        if (current / "INTAKE.md").is_file():
            report.skipped.append(_rel(current, folder) or ".")
            continue
        if (current / "frames.npz").is_file() or (current / "display_log.csv").is_file():
            found.latency_runs.append(current)
            continue
        for child in sorted(current.iterdir(), reverse=True):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                _classify(child, found, report)
    for items in (found.logs, found.calibrations, found.processing, found.latency_runs, found.radio):
        items.sort()
    return found


def _classify(path: Path, found: Found, report: Report) -> None:
    name = path.name
    if name.endswith(".frames.jsonl"):
        return
    if path.suffix == ".jsonl":
        if DETECTION_LOG_SCHEMA.encode() in _first_line(path):
            found.logs.append(path)
        else:
            report.ignored += 1
        return
    if path.suffix == ".json":
        try:
            schema = _json_schema(path)
        except (OSError, UnicodeDecodeError, ValueError):
            report.warn("unreadable-json", _rel(path, report.folder), "this .json file is not readable JSON, so it "
                        "was not checked", "if it is a calibration or processing-time file, make it again with "
                        "the tool that wrote it")
            return
        if schema == CALIBRATION_SCHEMA:
            found.calibrations.append(path)
        elif schema == PROCESSING_SCHEMA:
            found.processing.append(path)
        else:
            report.ignored += 1
        return
    if path.suffix.lower() in RADIO_SUFFIXES and path.stat().st_size < (64 << 20):
        if RADIO_FORMAT.encode() in path.read_bytes():
            found.radio.append(path)
            return
    report.ignored += 1


def _rel(path: Path, folder: Path) -> str:
    try:
        return path.relative_to(folder).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------- detection logs

def _name_problems(stem: str, truth: tuple[float, float, float]) -> list[str]:
    tokens = stem.split("_")
    problems = []
    for token in tokens:
        match = re.fullmatch(r"d(\d+(?:\.\d+)?)", token)
        # The name is the taped distance rounded to the digits it shows ("d1.0" reads "distance 1.0 m" in
        # tools/capture_aruco.py, whose own example names a 0.998 m tape d1.0), so d1.0 covers 0.95 to 1.05 m.
        # That still catches a mix-up between crosses (0.5, 1.0, 2.0 m) or 0.75 m data under d1.0.
        decimals = len(match.group(1).partition(".")[2]) if match else 0
        if match and abs(float(match.group(1)) - truth[2]) > 0.5 * 10 ** -decimals + 1e-9:
            problems.append(f"the name says {match.group(1)} m but the ground truth z in the log is "
                            f"{truth[2]:g} m")
    if "center" in tokens and abs(truth[0]) > CENTER_TOLERANCE_M:
        problems.append(f"the name says center but the ground truth x (sideways) is {truth[0]:g} m, not 0")
    if "side" in tokens and abs(truth[0]) <= CENTER_TOLERANCE_M:
        problems.append("the name says side but the ground truth x (sideways) is 0")
    return problems


def check_log(path: Path, report: Report, min_rate: float) -> LogInfo | None:
    rel = _rel(path, report.folder)
    try:
        _check_log(path.stem, path)
    except IntakeError as exc:
        report.fail("missing-provenance", rel, str(exc), "re-record it with tools/capture_aruco.py and --operator "
                    "(it fills in the rest); " + NEVER_EDIT)
        return None
    try:
        trace = read_detection_log(path)
    except (ValueError, OSError) as exc:
        report.fail("log-unreadable", rel, f"the log reader refused it: {exc}",
                    "copy the file again from where the capture wrote it (an editor or a copy tool may have "
                    "changed its line endings), or re-record it; " + NEVER_EDIT)
        return None

    detected = [o for o in trace.observations if o.outcome == "detected"]
    missed = sum(1 for o in trace.observations if o.outcome == "missed")
    usable = [o for o in detected if o.position_m is not None and isinstance(o.ground_truth, dict)
              and o.ground_truth.get("position_m") is not None]
    if len(usable) < 2:
        report.fail("too-few-detections", rel, f"only {len(usable)} detected frame(s) have both a position and "
                    "ground truth; the noise fit needs at least 2", "record again with the marker in view; "
                    "check the camera preview first (runbook step 2)")
    seen = len(detected) + missed
    rate = len(detected) / seen if seen else 0.0
    if rate < min_rate:
        report.fail("low-detection-rate", rel, f"the marker was found in only {100 * rate:.0f}% of frames "
                    f"({len(detected)} of {seen}); intake needs at least {100 * min_rate:.0f}%",
                    "fix the light or the marker (flat, matte, fully in view), then record again under a new name")
    frames = {o.position_frame for o in detected if o.position_m is not None}
    if len(frames) > 1:
        report.fail("mixed-position-frames", rel, f"positions are in more than one frame ({', '.join(sorted(frames))});"
                    " the fit needs one", "record again; one capture writes one frame")
    subjects = {o.subject_id for o in trace.observations}
    if len(subjects) > 1:
        report.warn("multiple-subjects", rel, f"the log has {len(subjects)} subjects ({', '.join(sorted(subjects))})",
                    "give intake --subject ID to choose one")

    if usable:
        truth = tuple(float(v) for v in usable[0].ground_truth["position_m"])
        for problem in _name_problems(path.stem, truth):
            report.fail("name-truth-mismatch", rel, problem, "a DIST or LAT shell variable was probably left from "
                        "the last placement; set DIST and LAT for this cross and record again under a new name")
        if frames == {"camera"}:
            pairs = [(o.position_m[2], o.ground_truth["position_m"][2]) for o in usable
                     if o.ground_truth["position_m"][2] > 0.1]
            if pairs:
                error = statistics.median(abs(z - t) / t for z, t in pairs)
                if error > MAX_DISTANCE_ERROR:
                    seen_z = statistics.median(z for z, _ in pairs)
                    tape_z = statistics.median(t for _, t in pairs)
                    report.fail("distance-off", rel, f"the camera puts the marker at {seen_z:.3f} m but the tape says "
                                f"{tape_z:.3f} m ({100 * error:.0f}% off; the runbook allows a few percent)",
                                "check --marker-length-m against the ruler and that --calibration is the file made "
                                "at this frame size, then record again")
            seen_x = statistics.median(o.position_m[0] for o in usable)
            truth_x = statistics.median(o.ground_truth["position_m"][0] for o in usable)
            limit = LATERAL_ERROR_M + LATERAL_ERROR_PER_M_OF_Z * abs(truth[2])
            if abs(seen_x - truth_x) > limit:
                report.fail("lateral-off", rel, f"the camera puts the marker {seen_x:.3f} m sideways but the log's "
                            f"truth says {truth_x:.3f} m ({abs(seen_x - truth_x):.3f} m apart; honest placement "
                            f"error is under {limit:.3f} m here)",
                            "LAT was probably left from another cross; set LAT to the x on this cross's card and "
                            "record again under a new name")

    _check_sidecars(path, trace, report, rel)

    prov = trace.provenance or {}
    revision = prov.get("code_revision") or trace.code_revision
    if trace.status == "measured":
        if revision is None or revision == "unknown" or str(revision).endswith("-dirty"):
            report.warn("dirty-code-revision", rel, f"the code revision is {revision!r}, so the code that made "
                        "this log cannot be named exactly", "next time capture from a clean checkout "
                        "(runbook step 0)")
        if prov.get("calibration_ref") is None:
            report.warn("no-calibration", rel, "the focal length came from a field-of-view guess, not a "
                        "calibration file, so every distance and the noise scale with that guess",
                        "calibrate (runbook step 3) and record with --calibration")
    fit = path.with_name(path.stem + ".fit.json")
    if fit.exists():
        report.warn("fit-would-be-replaced", _rel(fit, report.folder), "an older fit sits next to the log; the "
                    "fit command printed after capture replaces it", "rename it first if you want to keep it "
                    "(intake writes its own fits into the output folder)")

    captured = None
    if isinstance(prov.get("captured_at"), str):
        try:
            captured = datetime.fromisoformat(prov["captured_at"])
        except ValueError:
            captured = None
    capture = trace.capture or {}
    size = capture.get("resolution_px")
    resolution = (int(size["width"]), int(size["height"])) if isinstance(size, dict) else None
    return LogInfo(path=path, status=_status(trace.status), sha256=_sha256(path), resolution=resolution,
                   calibration_ref=prov.get("calibration_ref"), calibration_id=capture.get("calibration_id"),
                   code_revision=revision, captured_at=captured)


def _check_sidecars(path: Path, trace: Any, report: Report, rel: str) -> None:
    stem = path.stem
    frames = path.with_name(stem + ".frames.jsonl")
    raw = path.with_name(stem + ".raw.mkv")
    frames_sha = _sha256(frames) if frames.is_file() else None
    match = FRAMES_NOTE.search(trace.notes or "")
    if match:
        if frames_sha is None:
            report.fail("frames-sha256-mismatch", rel, f"the log names the frames file {match.group(1)} but it is "
                        "not next to the log", "put the .frames.jsonl the capture wrote back next to the log")
        elif frames_sha != match.group(2):
            report.fail("frames-sha256-mismatch", rel, f"{frames.name} changed after the capture (its sha256 is "
                        "not the one in the log)", "put back the original file from the capture; " + NEVER_EDIT)
    if trace.status != "measured":
        return
    expected = (trace.provenance or {}).get("raw_sha256")
    if raw.is_file():
        if _sha256(raw) != expected:
            report.fail("raw-sha256-mismatch", rel, f"{raw.name} is not the video this log was made from (its "
                        "sha256 is not the log's raw_sha256)", "put back the original video from the capture, or "
                        "record again; " + NEVER_EDIT)
    elif frames_sha is None or frames_sha != expected:
        report.fail("raw-sha256-mismatch", rel, f"the raw video {raw.name} is missing, so raw_sha256 cannot be "
                    "checked", "put the .raw.mkv the capture wrote back next to the log, with the same name")


# ---------------------------------------------------------------- calibration

def check_calibrations(found: Found, report: Report) -> dict[Path, tuple[str, tuple[int, int]]]:
    good: dict[Path, tuple[str, tuple[int, int]]] = {}
    for path in found.calibrations:
        rel = _rel(path, report.folder)
        try:
            camera, digest = load_camera_calibration(path)
        except (ValueError, OSError) as exc:
            report.fail("calibration-unreadable", rel, f"the calibration reader refused it: {exc}",
                        "make it again with tools/calibrate_charuco.py --overwrite; " + NEVER_EDIT)
            continue
        good[path] = (digest, (camera.width_px, camera.height_px))
        rms = json.loads(path.read_text(encoding="utf-8")).get("rms_reprojection_error_px")
        if isinstance(rms, (int, float)) and rms > MAX_RMS_PX:
            report.warn("calibration-rms-high", rel, f"the calibration RMS is {rms:.2f} px (the runbook wants under "
                        f"about {MAX_RMS_PX} px)", "redo the calibration with a flat board, sharp views and more "
                        "angles (runbook step 3)")
    return good


def check_calibration_links(logs: list[LogInfo], calibrations: dict[Path, tuple[str, tuple[int, int]]],
                            report: Report) -> None:
    by_sha = {digest: (path, size) for path, (digest, size) in calibrations.items()}
    for log in logs:
        if log.status != "measured" or log.calibration_ref is None:
            continue
        rel = _rel(log.path, report.folder)
        named = [p for p in calibrations if p.name == log.calibration_id]
        if named and log.calibration_ref not in by_sha:
            report.fail("calibration-sha256-mismatch", rel, f"{log.calibration_id} changed after this log was "
                        "recorded (its sha256 is not the log's calibration_ref)", "put back the calibration file "
                        "used for the capture, or record again with the new one")
            continue
        if log.calibration_ref not in by_sha:
            report.warn("calibration-not-found", rel, "the calibration file this log used is not in the folder, so "
                        "its frame size cannot be checked", "copy the calibration JSON into the folder")
            continue
        cal_path, size = by_sha[log.calibration_ref]
        if log.resolution is not None and log.resolution != size:
            report.fail("frame-size-mismatch", rel, f"the frames are {log.resolution[0]}x{log.resolution[1]} but "
                        f"{cal_path.name} is for {size[0]}x{size[1]}", "record at the calibration's size, or "
                        "calibrate at this size (a calibration fits one frame size only)")


def check_frame_size(logs: list[LogInfo], calibrations: dict[Path, tuple[str, tuple[int, int]]],
                     expected: tuple[int, int] | None, report: Report) -> None:
    sizes = {log.resolution for log in logs if log.resolution is not None}
    if len(sizes) > 1:
        listed = ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        report.fail("frame-size-mismatch", ".", f"the detection logs use more than one frame size ({listed})",
                    "keep one frame size per bench session (the runbook uses 1280x720)")
    if expected is None:
        return
    for log in logs:
        if log.resolution is not None and log.resolution != expected:
            report.fail("frame-size-mismatch", _rel(log.path, report.folder), f"the frames are "
                        f"{log.resolution[0]}x{log.resolution[1]}, not the expected {expected[0]}x{expected[1]}",
                        "ask the camera for the expected size (capture --width and --height) and record again")
    for path, (_, size) in calibrations.items():
        if size != expected:
            report.fail("frame-size-mismatch", _rel(path, report.folder), f"the calibration is for "
                        f"{size[0]}x{size[1]}, not the expected {expected[0]}x{expected[1]}",
                        "calibrate again at the expected size")


# ---------------------------------------------------------------- processing time and latency runs

def check_processing(found: Found, report: Report) -> list[tuple[Path, str]]:
    statuses = []
    if len(found.processing) > 1:
        names = ", ".join(_rel(p, report.folder) for p in found.processing)
        report.fail("several-processing-files", ".", f"intake takes one processing-time file and the folder has "
                    f"{len(found.processing)} ({names})", "keep the one from this session and move the others out")
    for path in found.processing:
        rel = _rel(path, report.folder)
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            info = _processing_from_json(data)
        except LinkConversionError as exc:
            report.fail("processing-refused", rel, f"intake would refuse it: {exc}", "make it again with "
                        "tools/aruco_processing_time.py measure (runbook step 5); " + NEVER_EDIT)
            continue
        statuses.append((path, info["status"]))
        problems = aruco_processing_time.check_report(path)
        if not problems:
            continue
        if info["status"] == "measured":
            report.fail("processing-check-failed", rel, "; ".join(problems), "make it again with "
                        "tools/aruco_processing_time.py measure into a new --out name; " + NEVER_EDIT)
        else:
            report.warn("processing-unchecked", rel, "; ".join(problems), "fine for a synthetic file; a measured "
                        "one must pass tools/aruco_processing_time.py check")
    return statuses


def check_latency_run(run: Path, report: Report) -> str | None:
    rel = _rel(run, report.folder)
    try:
        _check_latency_run(run.name, run)
    except IntakeError as exc:
        report.fail("latency-run-refused", rel, f"intake would refuse it: {exc}", "record it again with "
                    "tools/vision_latency_run.py into a new folder (runbook step 6)")
        return None
    try:
        frames, t_read, meta = vision_latency.load_run(run)
        n_bits = int(meta.get("n_bits", 8)) if isinstance(meta, dict) else 8
        vision_latency.read_display_log(run / "display_log.csv", n_bits)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        report.fail("latency-run-bad", rel, f"the recording cannot be read: {exc}", "record it again into a new "
                    "folder")
        return None
    meta = meta if isinstance(meta, dict) else {}
    if len(frames) == 0:
        report.fail("latency-run-bad", rel, "the recording has no frames", "record it again; aim the camera so "
                    "the striped bar is in view")
    elif len(frames) != len(t_read):
        report.fail("latency-run-bad", rel, f"frames.npz has {len(frames)} frames but {len(t_read)} read times",
                    "record it again into a new folder")
    if meta.get("camera_error"):
        report.fail("latency-run-bad", rel, f"the recorder logged a camera problem: {meta['camera_error']}",
                    "check --camera-index and record again into a new folder")
    if meta.get("stopped_early"):
        report.warn("latency-run-short", rel, "the recording was stopped early", "fine if enough frames were "
                    "kept; analyze it and check the rejected share")
    return _status(meta.get("status"))


# ---------------------------------------------------------------- radio

def _radio_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(RADIO_FORMAT.encode() + b"\n") or data.startswith(RADIO_FORMAT.encode() + b"\r\n"):
        return data.decode("utf-8")
    texts, _ = extract_echo_log.extract(data, path.name)
    return texts[0]


def _role(text: str) -> str | None:
    match = re.search(r"^# role=(\S+)\s*$", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def check_radio(found: Found, report: Report) -> int:
    folders: dict[Path, dict[str, dict[str, Any]]] = {}
    for path in found.radio:
        rel = _rel(path, report.folder)
        try:
            text = _radio_text(path)
            role = _role(text)
            if role == "sender":
                log = import_espnow_log.parse_sender_log(text, path.name)
            elif role == "echoer":
                log = import_espnow_log.parse_echoer_log(text, path.name)
            else:
                raise ValueError(f"{path.name}: the header has no role=sender or role=echoer line")
        except (ValueError, UnicodeDecodeError) as exc:
            report.fail("radio-log-refused", rel, f"the radio log reader refused it: {exc}", "capture it again "
                        "(tools/capture_serial_log.py) or cut it again with tools/extract_echo_log.py; "
                        + NEVER_EDIT)
            continue
        # A raw capture and the clean log cut from it hold the same text, so they count once.
        folders.setdefault(path.parent, {"sender": {}, "echoer": {}})[role].setdefault(text, (path, log))
    pairs = 0
    for folder, roles in sorted(folders.items()):
        rel = _rel(folder, report.folder) or "."
        senders, echoers = list(roles["sender"].values()), list(roles["echoer"].values())
        if len(senders) > 1 or len(echoers) > 1:
            report.fail("radio-pairing", rel, f"intake takes one sender and one echoer log, and this folder has "
                        f"{len(senders)} different sender and {len(echoers)} different echoer logs",
                        "put each sender and echoer pair from one test in its own folder")
            continue
        if not senders or not echoers:
            have, need = ("sender", "echoer") if senders else ("echoer", "sender")
            report.fail("radio-pairing", rel, f"there is a {have} log but no {need} log; without both no outbound "
                        "loss can be fitted", f"capture the {need} board too (runbook step 10) and put its log "
                        "in the same folder")
            continue
        try:
            raw = import_espnow_log.logs_to_trace_dict(senders[0][1], echoers[0][1], trace_id="preflight",
                                                       status="synthetic")
            import_espnow_log.EspNowTrace.from_dict(raw)
        except ValueError as exc:
            report.fail("radio-log-refused", rel, f"the two logs do not fit together: {exc}", "use the sender and "
                        "echoer logs from the same test; capture both again if one was cut short")
            continue
        pairs += 1
    return pairs


# ---------------------------------------------------------------- across the folder

def check_sessions(logs: list[LogInfo], report: Report) -> None:
    measured = [log for log in logs if log.status == "measured"]
    for label, values in (("calibration file", {log.calibration_ref for log in measured}),
                          ("code revision", {log.code_revision for log in measured})):
        if len(values) > 1:
            report.fail("mixed-sessions", ".", f"the measured logs come from {len(values)} different {label}s, so "
                        "they are not one bench session", "run the preflight and intake on one session's files "
                        "at a time (move the others to their own folder)")
    times = [log.captured_at for log in measured if log.captured_at is not None]
    try:
        span_h = (max(times) - min(times)).total_seconds() / 3600 if len(times) > 1 else 0.0
    except TypeError:  # a time without a UTC offset cannot be compared with one that has it
        span_h = 0.0
    if span_h > MAX_SESSION_SPAN_H:
        report.warn("capture-times-far-apart", ".", f"the measured logs were captured {span_h:.0f} hours apart",
                    "if the laptop lid, the marker or the tape moved in between, keep the sessions apart")


def check_collisions(logs: list[LogInfo], runs: list[Path], report: Report) -> None:
    for kind, names in (("detection logs", [(log.path.stem, log.path) for log in logs]),
                        ("latency runs", [(run.name, run) for run in runs])):
        seen: dict[str, Path] = {}
        for name, path in names:
            if name in seen:
                report.fail("name-collision", _rel(path, report.folder), f"two {kind} are both called {name!r} "
                            f"({_rel(seen[name], report.folder)} too); intake names its outputs after them",
                            "rename one (add b, as the runbook does) or run intake on them separately")
            else:
                seen[name] = path
    by_sha: dict[str, Path] = {}
    for log in logs:
        if log.sha256 in by_sha:
            report.fail("duplicate-file", _rel(log.path, report.folder), f"this log is a byte-for-byte copy of "
                        f"{_rel(by_sha[log.sha256], report.folder)}, so intake would count it twice",
                        "delete the copy or move it out of the folder")
        else:
            by_sha[log.sha256] = log.path


def check_statuses(statuses: list[tuple[Path, str]], report: Report) -> None:
    groups: dict[str, list[Path]] = {}
    for path, status in statuses:
        groups.setdefault(status, []).append(path)
    if len(groups) < 2:
        return
    overall = combine_status(["synthetic" if s == "fixture" else s for s in groups])
    listed = "; ".join(f"{status}: {', '.join(_rel(p, report.folder) for p in paths)}"
                       for status, paths in sorted(groups.items()))
    report.fail("mixed-status", ".", f"the inputs do not share one status ({listed}); intake would label the "
                f"whole result {overall}", "move replays, fixtures and rehearsal files out of the bench folder")


# ---------------------------------------------------------------- driver

def preflight(folder: Path, *, output: Path | None = None, frame_size: tuple[int, int] | None = None,
              min_detection_rate: float = DEFAULT_MIN_DETECTION_RATE) -> Report:
    folder = Path(folder)
    report = Report(folder=folder)
    found = find_inputs(folder, report)
    report.counts = {"detection logs": len(found.logs), "calibration files": len(found.calibrations),
                     "processing-time files": len(found.processing), "latency runs": len(found.latency_runs),
                     "radio logs": len(found.radio), "radio pairs": 0}
    if not any((found.logs, found.calibrations, found.processing, found.latency_runs, found.radio)):
        report.fail("no-bench-files", ".", "found no detection log, calibration, processing-time file, latency "
                    "run or radio log here", "give the bench data folder (the one holding aruco/, latency/ and "
                    "radio/), not an intake output folder")
        return report

    logs = [info for info in (check_log(p, report, min_detection_rate) for p in found.logs) if info is not None]
    calibrations = check_calibrations(found, report)
    check_calibration_links(logs, calibrations, report)
    check_frame_size(logs, calibrations, frame_size, report)
    statuses: list[tuple[Path, str]] = [(log.path, log.status) for log in logs]
    statuses += check_processing(found, report)
    for run in found.latency_runs:
        status = check_latency_run(run, report)
        if status is not None:
            statuses.append((run, status))
    report.counts["radio pairs"] = check_radio(found, report)
    check_statuses(statuses, report)
    check_sessions(logs, report)
    check_collisions(logs, found.latency_runs, report)
    if output is not None:
        try:
            _check_output(Path(output), [*found.logs, *found.latency_runs, *found.processing])
        except IntakeError as exc:
            report.fail("output-not-usable", _rel(Path(output), folder), f"intake would refuse this --output: {exc}",
                        "pick a new folder name under out/ (for example out/intake-pilot-2); never under sim/runs")
    return report


def report_lines(report: Report) -> list[str]:
    c = report.counts
    lines = [f"Preflight for {report.folder}",
             (f"Found {c['detection logs']} detection log(s), {c['calibration files']} calibration file(s), "
              f"{c['processing-time files']} processing-time file(s), {c['latency runs']} latency run(s), "
              f"{c['radio logs']} radio log(s) in {c['radio pairs']} pair(s).")]
    if report.skipped:
        lines.append(f"Skipped {len(report.skipped)} intake output folder(s): {', '.join(report.skipped)}.")
    if report.ignored:
        lines.append(f"Ignored {report.ignored} other file(s) (not bench inputs).")
    if c["radio logs"]:
        lines.append("Radio logs carry no status of their own; intake takes it from --status.")
    lines.append("")
    for finding in sorted(report.findings, key=lambda f: (f.level != "FAIL", f.file, f.code)):
        lines.append(f"{finding.level}  {finding.file}: {finding.reason}")
        lines.append(f"      fix: {finding.fix}")
    fails = sum(1 for f in report.findings if f.level == "FAIL")
    warns = len(report.findings) - fails
    if fails:
        lines.append(f"RESULT: FAIL ({fails} problem(s), {warns} warning(s)). Do not run the intake until every "
                     "FAIL above is fixed.")
    else:
        lines.append(f"RESULT: PASS ({warns} warning(s)). The files are consistent, so the intake can run. A pass "
                     "does not mean the numbers in them are good.")
    return lines


def _frame_size(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not match or int(match.group(1)) < 1 or int(match.group(2)) < 1:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 1280x720")
    return int(match.group(1)), int(match.group(2))


def _rate(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a number from 0 to 1, for example 0.5") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("use a number from 0 to 1, for example 0.5")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="the bench data folder (holds aruco/, latency/, radio/)")
    parser.add_argument("--output", type=Path, help="the --output folder you plan to give the intake")
    parser.add_argument("--frame-size", type=_frame_size, help="expected frame size, such as 1280x720")
    parser.add_argument("--min-detection-rate", type=_rate, default=DEFAULT_MIN_DETECTION_RATE,
                        help=f"lowest share of frames with the marker found (default {DEFAULT_MIN_DETECTION_RATE})")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a folder; give the bench data folder", file=sys.stderr)
        return 2
    report = preflight(args.folder, output=args.output, frame_size=args.frame_size,
                       min_detection_rate=args.min_detection_rate)
    print("\n".join(report_lines(report)))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
