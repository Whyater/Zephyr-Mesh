"""Tests for tools/calibrate_charuco.py on synthetic pictures only.

The board is drawn into 1280x720 pictures with a known pinhole camera (zero
distortion) at known poses, so the exact homography from the flat board to the
picture is known. The calibration has to recover the camera it was drawn with.
No test opens a camera: a guard fails any test that passes an integer to
cv2.VideoCapture or runs system_profiler.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools import calibrate_charuco as tool
from tools.aruco_geometry import load_camera_calibration

WIDTH, HEIGHT = 1280, 720
TRUE_FX, TRUE_FY, TRUE_CX, TRUE_CY = 930.0, 936.0, 646.3, 355.8
TRUE_K = np.array([[TRUE_FX, 0.0, TRUE_CX], [0.0, TRUE_FY, TRUE_CY], [0.0, 0.0, 1.0]])
SUPERSAMPLE = 4
TEXTURE_PX_PER_M = tool.PRINT_PX_PER_MM * 1000
MARGIN_PX = tool.PRINT_MARGIN_MM * tool.PRINT_PX_PER_MM
BOARD_W_M = tool.BOARD_SQUARES[0] * tool.NOMINAL_SQUARE_MM / 1000.0
BOARD_H_M = tool.BOARD_SQUARES[1] * tool.NOMINAL_SQUARE_MM / 1000.0

# (target pixel of the board center u, v, depth m, tilt about x deg, tilt about y deg, roll deg)
POSES = [
    (640, 360, 0.55, 0, 0, 0),
    (330, 200, 0.70, 25, -20, 5),
    (640, 190, 0.65, -30, 0, -8),
    (950, 200, 0.70, 20, 30, 10),
    (330, 360, 0.60, 0, -40, 0),
    (950, 360, 0.60, 0, 40, 3),
    (330, 530, 0.70, -25, -25, -5),
    (640, 540, 0.65, 35, 5, 0),
    (950, 530, 0.70, -20, 25, 12),
    (640, 360, 0.80, 40, 0, 0),
    (640, 360, 0.80, 0, 40, -15),
    (500, 300, 0.50, -15, 15, 20),
    (800, 420, 0.50, 15, -15, -20),
    (420, 450, 0.75, 30, 30, 0),
    (860, 270, 0.75, -30, -30, 0),
    (640, 360, 0.95, -40, 10, 30),
]


class GuardTripped(AssertionError):
    pass


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    """Fail if anything tries to open a real camera or list cameras."""
    real_capture = cv2.VideoCapture

    def guarded_capture(*args, **kwargs):
        if args and isinstance(args[0], (int, np.integer)):
            raise GuardTripped("a test tried to open a real camera")
        return real_capture(*args, **kwargs)

    real_run = subprocess.run

    def guarded_run(command, *args, **kwargs):
        if "system_profiler" in " ".join(map(str, command if isinstance(command, (list, tuple)) else [command])):
            raise GuardTripped("a test tried to run system_profiler")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", guarded_capture)
    monkeypatch.setattr(subprocess, "run", guarded_run)


def _rotation(ax_deg: float, ay_deg: float, az_deg: float) -> np.ndarray:
    rx, _ = cv2.Rodrigues(np.array([math.radians(ax_deg), 0.0, 0.0]))
    ry, _ = cv2.Rodrigues(np.array([0.0, math.radians(ay_deg), 0.0]))
    rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(az_deg)]))
    return rx @ ry @ rz


def render_view(texture: np.ndarray, pose: tuple, rng: np.random.Generator) -> np.ndarray:
    """Draw the printed board (with its white margin) seen by TRUE_K, no distortion.

    Board frame: origin at the board's top-left inner edge, x right, y down,
    z into the board, which is how OpenCV's CharucoBoard places its corners.
    Identity rotation means the board faces the camera upright. With zero
    distortion the board plane maps to the picture by one exact homography,
    K [r1 r2 t]. It is drawn 4x finer and averaged down (INTER_AREA).
    """
    u, v, depth, ax, ay, az = pose
    rotation = _rotation(ax, ay, az)
    center = np.array([BOARD_W_M / 2.0, BOARD_H_M / 2.0, 0.0])
    target = depth * np.linalg.solve(TRUE_K, np.array([u, v, 1.0]))
    translation = target - rotation @ center
    board_to_image = TRUE_K @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    # Texture pixel k has its center at board coordinate (k - margin + 0.5) / px_per_m.
    texture_to_board = np.array([
        [1.0 / TEXTURE_PX_PER_M, 0.0, (0.5 - MARGIN_PX) / TEXTURE_PX_PER_M],
        [0.0, 1.0 / TEXTURE_PX_PER_M, (0.5 - MARGIN_PX) / TEXTURE_PX_PER_M],
        [0.0, 0.0, 1.0],
    ])
    s = SUPERSAMPLE
    image_to_fine = np.array([[s, 0.0, 0.5 * s - 0.5], [0.0, s, 0.5 * s - 0.5], [0.0, 0.0, 1.0]])
    homography = image_to_fine @ board_to_image @ texture_to_board
    fine = cv2.warpPerspective(texture, homography, (WIDTH * s, HEIGHT * s), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=110)
    gray = cv2.resize(fine, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float64)
    gray += rng.normal(0.0, 2.0, size=gray.shape)
    return cv2.cvtColor(np.clip(np.rint(gray), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def view_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("views")
    texture = tool.board_image(cv2)
    rng = np.random.default_rng(20261021)
    for index, pose in enumerate(POSES, start=1):
        assert cv2.imwrite(str(folder / f"view-{index:02d}.png"), render_view(texture, pose, rng))
    return folder


def test_calibration_recovers_known_camera(view_folder, tmp_path, capsys):
    output = tmp_path / "calibration-1280x720.json"
    code = tool.main(["--images", str(view_folder / "view-*.png"), "--output", str(output)])
    printed = capsys.readouterr().out
    assert code == 0, printed

    model, digest = load_camera_calibration(output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert f"SHA-256: {digest}" in printed
    assert (model.width_px, model.height_px) == (WIDTH, HEIGHT)
    k = model.matrix()
    # Measured on this picture set with OpenCV 5.0.0 and two noise seeds:
    # fx -1.4 / -1.6 px, fy -1.1 / -1.3 px, cx -2.3 / -2.3 px, cy -0.15 / +0.28 px,
    # RMS 0.077 px. The small negative focal bias stays (about -1.2 px) even
    # with distortion fixed at zero, so it comes from the corner detector or
    # this renderer, not from the fit; its cause is not pinned down. The cx
    # error mostly trades against the free tangential and k3 terms (with them
    # fixed, cx is within 0.04 px). Tolerances are about 1.7 times the worst
    # measured error. 3 px is 0.3% of fx (0.3% of a distance), and 4 px in cx
    # moves the axis line about 9 mm at 2 m.
    assert abs(k[0, 0] - TRUE_FX) < 3.0
    assert abs(k[1, 1] - TRUE_FY) < 3.0
    assert abs(k[0, 2] - TRUE_CX) < 4.0
    assert abs(k[1, 2] - TRUE_CY) < 3.0
    # The pictures have no distortion, so the fitted terms should be small
    # (measured: k3 about 0.02, the rest under 0.012).
    assert np.all(np.abs(model.distortion()) < 0.05)
    document_rms = float(printed.split("RMS reprojection error: ")[1].split(" px")[0])
    assert document_rms < 0.2
    assert "Used 16 views" in printed
    hfov = math.degrees(2 * math.atan(WIDTH / (2 * TRUE_FX)))
    assert f"{hfov:.1f} degrees" in printed


def test_board_png_prints_at_true_size(tmp_path):
    path = tmp_path / "board.png"
    assert tool.main(["--print-board", str(path)]) == 0
    data = path.read_bytes()
    x_ppu, y_ppu, unit = tool.read_phys(data)
    assert unit == 1  # pixels per meter
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    height_px, width_px = image.shape
    board_w_mm = (width_px - 2 * MARGIN_PX) / x_ppu * 1000
    board_h_mm = (height_px - 2 * MARGIN_PX) / y_ppu * 1000
    assert (board_w_mm, board_h_mm) == pytest.approx((245.0, 175.0), abs=1e-9)
    page_w_mm, page_h_mm = width_px / x_ppu * 1000, height_px / y_ppu * 1000
    assert page_w_mm <= 279.4 and page_h_mm <= 215.9  # US letter, landscape
    # The margin is plain white, and the board is found in the printed image.
    assert image[:MARGIN_PX].min() == 255 and image[:, :MARGIN_PX].min() == 255
    board = tool.make_board(cv2, tool.NOMINAL_SQUARE_MM, tool.NOMINAL_MARKER_MM)
    small = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    view = tool.find_view(cv2, tool.make_detector(cv2, board), board, small, "printed")
    assert view is not None and len(view.image_points) == 24
    # The pHYs chunk sits right after IHDR and its CRC (over type and data) is correct.
    assert data[33 + 4:33 + 8] == b"pHYs"
    chunk_data = data[33 + 8:33 + 17]
    assert int.from_bytes(data[33 + 17:33 + 21], "big") == zlib.crc32(b"pHYs" + chunk_data)


def test_refuses_to_overwrite(view_folder, tmp_path, capsys):
    output = tmp_path / "calibration.json"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        tool.main(["--images", str(view_folder / "view-*.png"), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep me"
    assert "--overwrite" in capsys.readouterr().err


def test_save_views_inside_repo_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        tool.main(["--images", "nothing-*.png", "--output", str(tmp_path / "c.json"),
                   "--save-views", str(tool.ROOT / "calibration-views")])
    assert error.value.code == 2
    assert "outside the repo" in capsys.readouterr().err
    assert not (tool.ROOT / "calibration-views").exists()


def test_save_views_outside_repo_keeps_pictures(view_folder, tmp_path):
    # pytest's --basetemp may sit inside the repo, which the tool rightly refuses,
    # so use a system temporary folder here.
    with tempfile.TemporaryDirectory() as outside:
        saved = Path(outside) / "kept"
        assert not tool._inside(saved, tool.ROOT)
        code = tool.main(["--images", str(view_folder / "view-*.png"), "--output", str(tmp_path / "c.json"),
                          "--save-views", str(saved)])
        assert code == 0
        assert len(list(saved.glob("view-*.png"))) == len(POSES)


def test_user_errors_exit_2(tmp_path):
    for argv in (
        [],  # no --output
        ["--output", str(tmp_path / "missing-folder" / "c.json")],
        ["--output", str(tmp_path / "c.json"), "--square-mm", "20", "--marker-mm", "25"],
        ["--output", str(tmp_path / "c.json"), "--min-corners", "2"],
    ):
        with pytest.raises(SystemExit) as error:
            tool.main(argv)
        assert error.value.code == 2, argv


def test_no_matching_images_is_runtime_failure(tmp_path):
    assert tool.main(["--images", str(tmp_path / "none-*.png"), "--output", str(tmp_path / "c.json")]) == 1
    assert not (tmp_path / "c.json").exists()


def test_too_few_views_is_runtime_failure(view_folder, tmp_path):
    assert tool.main(["--images", str(view_folder / "view-0[12].png"), "--output", str(tmp_path / "c.json")]) == 1


def test_coverage_warning_for_center_only_views(tmp_path):
    view = tool.View("center", np.zeros((4, 3), np.float32), np.array([[600, 340], [680, 340], [680, 380], [600, 380]], np.float32))
    counts = tool.coverage_counts([view] * 12, WIDTH, HEIGHT)
    assert counts[1, 1] == 12 and counts.sum() == 12
    result = tool.Calibration(0.9, TRUE_K, np.zeros(5), WIDTH, HEIGHT, 5, 20)
    notes = " ".join(tool.warnings_for(result, counts))
    assert "RMS 0.900" in notes and "only 5 views" in notes and "top-left" in notes


def test_guard_blocks_live_mode(tmp_path):
    """Live mode is never run in tests: the guard trips before a camera could open."""
    with pytest.raises(GuardTripped):
        tool.main(["--output", str(tmp_path / "c.json")])
    with pytest.raises(GuardTripped):
        cv2.VideoCapture(0)
