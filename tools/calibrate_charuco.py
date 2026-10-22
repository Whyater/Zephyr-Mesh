#!/usr/bin/env python3
r"""Calibrate a camera with a printed ChArUco board.

This finds the focal length, the image center and the lens distortion of one
camera at one resolution. It writes a zephyr-camera-calibration-1 JSON file that
tools/capture_aruco.py reads with --calibration.

Status: tested on synthetic images only. It has not been run on a real camera yet.

Setup (once). Use the separate camera venv, never the repo .venv:

    uv venv -p 3.13 ~/zephyr-camera-venv
    uv pip install -p ~/zephyr-camera-venv/bin/python "opencv-python==5.0.0.93" "numpy==2.5.4"

All commands below start from the repo folder:

    cd Zephyr-Mesh   # the folder you cloned the repository into

1. Print the board.

    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/calibrate_charuco.py --print-board /tmp/charuco-board.png

   The board is 7 by 5 squares of 35 mm with 26.25 mm DICT_4X4_50 markers, so
   245 by 175 mm, inside a 10 mm white margin. The PNG says 10 pixels per mm,
   so it prints at the right size at 100% scale. Print it on US letter in
   landscape, at 100% (turn off "fit to page"). Glue or tape it flat to
   something rigid, like a clipboard or a box lid.

2. Measure it. Use a ruler to measure several squares and several black
   markers, and take the average. Pass the measured sizes with --square-mm
   and --marker-mm. The printer may not hit 35 mm exactly, and the measured
   size is what sets the distance scale.

3. Calibrate live (this opens the camera and a preview window):

    mkdir -p "$HOME/zephyr-data/aruco"
    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/calibrate_charuco.py --square-mm 35.0 --marker-mm 26.25 --output "$HOME/zephyr-data/aruco/calibration-1280x720.json"

   Before it opens the camera it lists the cameras macOS knows about. Index 0
   may be an iPhone (Continuity Camera) or a virtual camera instead of the
   built-in one. If the preview shows the wrong camera, press q and try
   --camera-index 1. The first run makes macOS ask for camera permission for
   your terminal app; that is expected.

   In the preview, press space to keep a view (it needs at least
   --min-corners board corners, 12 by default) and q to finish. It stops on
   its own after --views views (20 by default).

   Or calibrate from pictures you already saved, which never opens a camera:

    PYTHONPATH=. ~/zephyr-camera-venv/bin/python tools/calibrate_charuco.py --images "$HOME/zephyr-data/aruco/calib-views/*.png" --square-mm 35.0 --marker-mm 26.25 --output "$HOME/zephyr-data/aruco/calibration-1280x720.json"

How to take good views:
- About 20 views.
- Cover the whole picture, including the four corners, across the set.
- Tilt the board up to about 45 degrees, in different directions.
- Keep the board flat and rigid, and hold still so nothing is blurred.
- Do not take many views that look the same.
- Calibrate at the same resolution you will capture at (1280x720 by default).
  A calibration only fits that one resolution.

What it prints: the RMS reprojection error, fx, fy, cx, cy, the horizontal
field of view, how many views and corners it used, and the SHA-256 of the file.
Under about 0.5 px RMS is a rule of thumb, not a standard. A higher number is a
red flag: look for a bent board, blur, or too few angles.

It refuses to overwrite an existing output file unless you add --overwrite.
--save-views DIR keeps the pictures it used; DIR must be outside the repo.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aruco_geometry import CALIBRATION_SCHEMA

BOARD_SQUARES = (7, 5)
DICTIONARY_NAME = "DICT_4X4_50"
NOMINAL_SQUARE_MM = 35.0
NOMINAL_MARKER_MM = 26.25
PRINT_PX_PER_MM = 10
PRINT_MARGIN_MM = 10
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RMS_WARNING_PX = 0.5
MIN_VIEWS_WARNING = 10
MIN_VIEWS_TO_SOLVE = 3
CELL_NAMES = (
    ("top-left", "top-middle", "top-right"),
    ("middle-left", "center", "middle-right"),
    ("bottom-left", "bottom-middle", "bottom-right"),
)
VENV_HELP = (
    "OpenCV with the ArUco module is needed. Use the camera venv, not the repo .venv:\n"
    "  uv venv -p 3.13 ~/zephyr-camera-venv\n"
    '  uv pip install -p ~/zephyr-camera-venv/bin/python "opencv-python==5.0.0.93" "numpy==2.5.4"\n'
    "then run this script with ~/zephyr-camera-venv/bin/python."
)


class CalibrationError(RuntimeError):
    """A failure while running (camera, files, too few views). Exit code 1."""


@dataclass(frozen=True)
class View:
    """Board corners found in one picture."""

    name: str
    object_points: np.ndarray  # (N, 3) float32, board frame, meters
    image_points: np.ndarray  # (N, 2) float32, pixels


# ---------------------------------------------------------------- OpenCV and the board


def import_cv2() -> Any:
    """Import OpenCV only when needed, with a clear message if it is missing."""
    try:
        import cv2
    except ImportError as exc:
        raise CalibrationError(f"could not import cv2 ({exc}).\n{VENV_HELP}") from exc
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise CalibrationError(f"this OpenCV ({cv2.__version__}) has no cv2.aruco.CharucoDetector (needs 4.7 or newer).\n{VENV_HELP}")
    return cv2


def make_board(cv2: Any, square_mm: float, marker_mm: float) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, square_mm / 1000.0, marker_mm / 1000.0, dictionary)


def make_detector(cv2: Any, board: Any) -> Any:
    return cv2.aruco.CharucoDetector(board)


def board_image(cv2: Any) -> np.ndarray:
    """The nominal board at PRINT_PX_PER_MM, with a white margin of PRINT_MARGIN_MM."""
    board = make_board(cv2, NOMINAL_SQUARE_MM, NOMINAL_MARKER_MM)
    width = round(BOARD_SQUARES[0] * NOMINAL_SQUARE_MM * PRINT_PX_PER_MM)
    height = round(BOARD_SQUARES[1] * NOMINAL_SQUARE_MM * PRINT_PX_PER_MM)
    image = board.generateImage((width, height), marginSize=0)
    margin = PRINT_MARGIN_MM * PRINT_PX_PER_MM
    return cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def add_phys_chunk(png: bytes, px_per_m: int) -> bytes:
    """Insert a pHYs chunk (pixels per meter) right after IHDR so it prints at true size."""
    if not png.startswith(PNG_SIGNATURE) or png[12:16] != b"IHDR":
        raise ValueError("not a PNG with IHDR first")
    ihdr_end = 8 + 8 + int.from_bytes(png[8:12], "big") + 4
    data = px_per_m.to_bytes(4, "big") + px_per_m.to_bytes(4, "big") + b"\x01"  # unit 1 = meter
    chunk = len(data).to_bytes(4, "big") + b"pHYs" + data + zlib.crc32(b"pHYs" + data).to_bytes(4, "big")
    return png[:ihdr_end] + chunk + png[ihdr_end:]


def read_phys(png: bytes) -> tuple[int, int, int] | None:
    """(x pixels per unit, y pixels per unit, unit) from a PNG pHYs chunk, or None."""
    position = 8
    while position + 8 <= len(png):
        length = int.from_bytes(png[position:position + 4], "big")
        kind = png[position + 4:position + 8]
        data = png[position + 8:position + 8 + length]
        if kind == b"pHYs":
            return int.from_bytes(data[0:4], "big"), int.from_bytes(data[4:8], "big"), data[8]
        if kind == b"IEND":
            break
        position += 12 + length
    return None


def write_board_png(cv2: Any, path: Path, overwrite: bool) -> None:
    ok, encoded = cv2.imencode(".png", board_image(cv2))
    if not ok:
        raise CalibrationError("OpenCV could not encode the board PNG")
    png = add_phys_chunk(encoded.tobytes(), PRINT_PX_PER_MM * 1000)
    _write_new_file(path, png, overwrite)


# ---------------------------------------------------------------- one view


def find_view(cv2: Any, detector: Any, board: Any, image: np.ndarray, name: str) -> View | None:
    """Detect the board in one picture. None if fewer than 4 corners were found.

    Live mode and --images mode both use this, so they treat pictures the same way.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or corners is None or len(ids) < 4:
        return None
    object_points, image_points = board.matchImagePoints(corners, ids)
    if object_points is None or len(object_points) < 4:
        return None
    return View(
        name=name,
        object_points=np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
        image_points=np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
    )


# ---------------------------------------------------------------- calibration


@dataclass(frozen=True)
class Calibration:
    rms_px: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray  # k1, k2, p1, p2, k3
    width_px: int
    height_px: int
    views: int
    corners: int


def calibrate(cv2: Any, views: list[View], width_px: int, height_px: int) -> Calibration:
    if len(views) < MIN_VIEWS_TO_SOLVE:
        raise CalibrationError(f"only {len(views)} usable view(s); at least {MIN_VIEWS_TO_SOLVE} are needed (about 20 is the plan)")
    rms, matrix, dist, _, _ = cv2.calibrateCamera(
        [view.object_points for view in views],
        [view.image_points for view in views],
        (width_px, height_px),
        None,
        None,
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)[:5]
    if not (math.isfinite(rms) and np.all(np.isfinite(matrix)) and np.all(np.isfinite(dist))):
        raise CalibrationError("calibration gave non-finite numbers; take more varied views")
    return Calibration(
        rms_px=float(rms),
        camera_matrix=matrix,
        dist_coeffs=dist,
        width_px=width_px,
        height_px=height_px,
        views=len(views),
        corners=sum(len(view.image_points) for view in views),
    )


def coverage_counts(views: list[View], width_px: int, height_px: int) -> np.ndarray:
    """For a 3x3 grid of image cells, how many views put at least one corner in each cell."""
    counts = np.zeros((3, 3), dtype=int)
    for view in views:
        points = view.image_points
        columns = np.clip((points[:, 0] + 0.5) * 3 // width_px, 0, 2).astype(int)
        rows = np.clip((points[:, 1] + 0.5) * 3 // height_px, 0, 2).astype(int)
        hit = np.zeros((3, 3), dtype=bool)
        hit[rows, columns] = True
        counts += hit
    return counts


def horizontal_fov_deg(fx: float, width_px: int) -> float:
    return math.degrees(2.0 * math.atan(width_px / (2.0 * fx)))


def warnings_for(result: Calibration, coverage: np.ndarray) -> list[str]:
    notes = []
    if result.rms_px > RMS_WARNING_PX:
        notes.append(
            f"RMS {result.rms_px:.3f} px is above {RMS_WARNING_PX} px (a rule of thumb, not a standard). "
            "Look for a bent board, blur, or too few angles."
        )
    if result.views < MIN_VIEWS_WARNING:
        notes.append(f"only {result.views} views; aim for about 20 (at least {MIN_VIEWS_WARNING}).")
    thin = [CELL_NAMES[r][c] for r in range(3) for c in range(3) if coverage[r, c] < 2]
    if thin:
        notes.append(
            "poor coverage: fewer than 2 views put corners in the " + ", ".join(thin)
            + " part(s) of the picture. Add views with the board there."
        )
    return notes


def calibration_document(cv2: Any, result: Calibration, square_mm: float, marker_mm: float, source: str) -> dict[str, Any]:
    m = result.camera_matrix
    return {
        "schema": CALIBRATION_SCHEMA,
        "camera_matrix": [[float(m[0, 0]), 0.0, float(m[0, 2])], [0.0, float(m[1, 1]), float(m[1, 2])], [0.0, 0.0, 1.0]],
        "dist_coeffs": [float(v) for v in result.dist_coeffs],
        "image_size_px": {"width": result.width_px, "height": result.height_px},
        "rms_reprojection_error_px": result.rms_px,
        "method": (
            f"OpenCV {cv2.__version__} calibrateCamera (k1, k2, p1, p2, k3) on ChArUco corners; board "
            f"{BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} squares, {DICTIONARY_NAME}, measured square {square_mm:g} mm, "
            f"measured marker {marker_mm:g} mm"
        ),
        "notes": f"{result.views} views, {result.corners} corners, from {source}",
    }


# ---------------------------------------------------------------- files


def _inside(path: Path, folder: Path) -> bool:
    """True if path is folder or anywhere under it, also for a differently cased path on a Mac disk."""
    if not folder.exists():
        return False
    for candidate in (path.resolve(), *path.resolve().parents):
        if candidate.exists() and os.path.samefile(candidate, folder):
            return True
    return False


def _write_new_file(path: Path, data: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CalibrationError(f"{path} already exists; add --overwrite to replace it")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_view_image(cv2: Any, folder: Path, index: int, image: np.ndarray) -> None:
    if not cv2.imwrite(str(folder / f"view-{index:02d}.png"), image):
        raise CalibrationError(f"could not write view {index} into {folder}")


# ---------------------------------------------------------------- picture sources


def views_from_images(cv2: Any, detector: Any, board: Any, pattern: str, min_corners: int,
                      save_dir: Path | None) -> tuple[list[View], int, int]:
    paths = sorted(glob.glob(os.path.expanduser(pattern)))
    if not paths:
        raise CalibrationError(f"no files match {pattern!r}")
    views: list[View] = []
    size: tuple[int, int] | None = None
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipped {path}: not a readable image")
            continue
        this_size = (int(image.shape[1]), int(image.shape[0]))
        if size is None:
            size = this_size
        elif this_size != size:
            raise CalibrationError(
                f"{path} is {this_size[0]}x{this_size[1]} but earlier pictures are {size[0]}x{size[1]}; "
                "a calibration fits one resolution only"
            )
        view = find_view(cv2, detector, board, image, Path(path).name)
        found = 0 if view is None else len(view.image_points)
        if view is None or found < min_corners:
            print(f"skipped {path}: {found} corners (needs {min_corners})")
            continue
        views.append(view)
        if save_dir is not None:
            save_view_image(cv2, save_dir, len(views), image)
    if size is None:
        raise CalibrationError(f"none of the files matching {pattern!r} could be read as images")
    return views, size[0], size[1]


def list_macos_cameras() -> list[str]:
    """Camera names from system_profiler. This only reads a list; it never opens a camera."""
    try:
        result = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # Camera names are the 4-space indented headings, like "    FaceTime HD Camera:".
        if line.startswith("    ") and not line.startswith("      ") and stripped.endswith(":"):
            names.append(stripped[:-1])
    return names


def views_from_camera(cv2: Any, detector: Any, board: Any, args: argparse.Namespace) -> tuple[list[View], int, int]:
    if platform.system() == "Darwin":
        names = list_macos_cameras()
        print("Cameras macOS reports: " + (", ".join(names) if names else "(none listed)"))
        print("Index 0 may be an iPhone (Continuity Camera) or a virtual camera, not the built-in one. Check the preview.")
        capture = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise CalibrationError(f"could not open camera index {args.camera_index}")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        window = "ChArUco calibration (space keep, q finish)"
        try:
            cv2.namedWindow(window)
        except cv2.error as exc:
            raise CalibrationError(
                "this OpenCV build has no preview window (headless). Live mode needs opencv-python, "
                "or save pictures and use --images."
            ) from exc
        views: list[View] = []
        size: tuple[int, int] | None = None
        while len(views) < args.views:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CalibrationError("the camera stopped giving frames")
            this_size = (int(frame.shape[1]), int(frame.shape[0]))
            if size is None:
                size = this_size
                print(f"Frames are {size[0]}x{size[1]} (asked for {args.width}x{args.height}).")
            elif this_size != size:
                raise CalibrationError("the frame size changed during calibration")
            view = find_view(cv2, detector, board, frame, f"live-{len(views) + 1:02d}")
            found = 0 if view is None else len(view.image_points)
            preview = frame.copy()
            if view is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, view.image_points.reshape(-1, 1, 2))
            color = (0, 200, 0) if found >= args.min_corners else (0, 0, 255)
            text = f"views {len(views)}/{args.views}  corners {found}  space keep  q finish"
            cv2.putText(preview, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if view is None or found < args.min_corners:
                    print(f"not kept: {found} corners (needs {args.min_corners})")
                    continue
                views.append(view)
                print(f"kept view {len(views)}: {found} corners")
                if args.save_views is not None:
                    save_view_image(cv2, args.save_views, len(views), frame)
        cv2.destroyWindow(window)
    finally:
        capture.release()
    if size is None:
        raise CalibrationError("no frames were read from the camera")
    return views, size[0], size[1]


# ---------------------------------------------------------------- command line


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.print_board is not None:
        if args.images is not None:
            parser.error("--print-board and --images do not go together")
        return
    if args.output is None:
        parser.error("--output FILE.json is required")
    if not args.output.parent.is_dir():
        parser.error(f"the folder {args.output.parent} does not exist; create it first (mkdir -p)")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; add --overwrite to replace it")
    for name, value in (("--square-mm", args.square_mm), ("--marker-mm", args.marker_mm)):
        if not math.isfinite(value) or not 5.0 <= value <= 300.0:
            parser.error(f"{name} must be a measured size between 5 and 300 mm")
    if args.marker_mm >= args.square_mm:
        parser.error("--marker-mm must be smaller than --square-mm (the marker sits inside a square)")
    if args.min_corners < 4:
        parser.error("--min-corners must be at least 4")
    if args.views < MIN_VIEWS_TO_SOLVE:
        parser.error(f"--views must be at least {MIN_VIEWS_TO_SOLVE}")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.save_views is not None:
        if _inside(args.save_views, ROOT):
            parser.error(f"--save-views must be outside the repo ({ROOT}); pictures are big and stay in Data/")
        if args.save_views.exists() and any(args.save_views.glob("view-*.png")) and not args.overwrite:
            parser.error(f"{args.save_views} already holds view-*.png files; pick an empty folder or add --overwrite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print-board", type=Path, metavar="PNG", help="write the printable board PNG and exit (no camera)")
    parser.add_argument("--images", metavar="PATTERN", help="calibrate from saved pictures matching this glob (no camera)")
    parser.add_argument("--output", type=Path, metavar="FILE.json", help="calibration file to write")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--square-mm", type=float, default=NOMINAL_SQUARE_MM,
                        help="MEASURED square size in mm; measure several squares with a ruler (default %(default)s, the design size)")
    parser.add_argument("--marker-mm", type=float, default=NOMINAL_MARKER_MM,
                        help="MEASURED black marker size in mm; measure several with a ruler (default %(default)s, the design size)")
    parser.add_argument("--min-corners", type=int, default=12, help="corners a view needs to be kept (default %(default)s, of 24)")
    parser.add_argument("--views", type=int, default=20, help="live mode stops after this many views (default %(default)s)")
    parser.add_argument("--camera-index", type=int, default=0, help="live mode camera index (default %(default)s)")
    parser.add_argument("--width", type=int, default=1280, help="live mode requested frame width (default %(default)s)")
    parser.add_argument("--height", type=int, default=720, help="live mode requested frame height (default %(default)s)")
    parser.add_argument("--save-views", type=Path, metavar="DIR", help="also save the kept pictures here (outside the repo)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    try:
        cv2 = import_cv2()
        if args.print_board is not None:
            write_board_png(cv2, args.print_board, args.overwrite)
            total_w = BOARD_SQUARES[0] * NOMINAL_SQUARE_MM + 2 * PRINT_MARGIN_MM
            total_h = BOARD_SQUARES[1] * NOMINAL_SQUARE_MM + 2 * PRINT_MARGIN_MM
            print(f"Wrote {args.print_board}: board {BOARD_SQUARES[0] * NOMINAL_SQUARE_MM:g} x "
                  f"{BOARD_SQUARES[1] * NOMINAL_SQUARE_MM:g} mm, page image {total_w:g} x {total_h:g} mm at "
                  f"{PRINT_PX_PER_MM} px per mm. Print at 100% on letter, landscape.")
            return 0
        if args.save_views is not None:
            args.save_views.mkdir(parents=True, exist_ok=True)
        board = make_board(cv2, args.square_mm, args.marker_mm)
        detector = make_detector(cv2, board)
        if args.images is not None:
            views, width, height = views_from_images(cv2, detector, board, args.images, args.min_corners, args.save_views)
            source = f"saved pictures matching {args.images}"
        else:
            views, width, height = views_from_camera(cv2, detector, board, args)
            source = f"live camera index {args.camera_index}"
        result = calibrate(cv2, views, width, height)
        document = calibration_document(cv2, result, args.square_mm, args.marker_mm, source)
        rendered = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        _write_new_file(args.output, rendered, args.overwrite)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    m = result.camera_matrix
    print(f"Image size: {result.width_px}x{result.height_px}")
    print(f"RMS reprojection error: {result.rms_px:.3f} px")
    print(f"fx = {m[0, 0]:.2f} px, fy = {m[1, 1]:.2f} px, cx = {m[0, 2]:.2f} px, cy = {m[1, 2]:.2f} px")
    print(f"Equivalent horizontal field of view: {horizontal_fov_deg(float(m[0, 0]), result.width_px):.1f} degrees")
    print("Distortion k1, k2, p1, p2, k3: " + ", ".join(f"{v:.5f}" for v in result.dist_coeffs))
    print(f"Used {result.views} views and {result.corners} corners")
    print(f"Wrote {args.output}")
    print(f"SHA-256: {hashlib.sha256(rendered).hexdigest()}")
    coverage = coverage_counts(views, result.width_px, result.height_px)
    for note in warnings_for(result, coverage):
        print(f"warning: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
