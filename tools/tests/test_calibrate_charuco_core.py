"""Tests for the parts of tools/calibrate_charuco.py that do not need OpenCV.

tools/tests/test_calibrate_charuco.py skips when cv2 is missing, which is the
case in the repo .venv. These tests cover the pure parts so they always run:
the board size math, the RMS rule, the calibration JSON writer (checked with
the reader capture_aruco.py uses), the pHYs print-size chunk, the field of
view, and the size checks on the command line. None of them imports cv2 or
opens a camera.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

import numpy as np
import pytest

from tools import calibrate_charuco as tool
from tools.aruco_geometry import CALIBRATION_SCHEMA, load_camera_calibration

US_LETTER_LANDSCAPE_MM = (279.4, 215.9)


# ---------------------------------------------------------------- board geometry


def test_nominal_board_size_matches_hand_math():
    # Truth, by hand: 7 x 35.0 = 245.0 mm, 5 x 35.0 = 175.0 mm, (7 - 1) x (5 - 1) = 24
    # inner corners, 26.25 / 35.0 = 0.75. At 10 px per mm the board is 2450 x 1750 px,
    # and a 10 mm (100 px) margin on each side makes the page 265 x 195 mm, 2650 x 1950 px.
    board = tool.board_geometry(35.0, 26.25)
    assert board.squares == (7, 5)
    assert board.width_mm == 245.0 and board.height_mm == 175.0
    assert board.inner_corners == 24
    assert board.marker_mm / board.square_mm == 0.75
    assert board.square_m == 0.035 and board.marker_m == 0.02625
    assert (board.page_width_mm, board.page_height_mm) == (265.0, 195.0)
    assert board.print_size_px == (2450, 1750)
    margin_px = tool.PRINT_MARGIN_MM * tool.PRINT_PX_PER_MM
    assert (board.print_size_px[0] + 2 * margin_px, board.print_size_px[1] + 2 * margin_px) == (2650, 1950)
    assert board.page_width_mm <= US_LETTER_LANDSCAPE_MM[0]
    assert board.page_height_mm <= US_LETTER_LANDSCAPE_MM[1]


def test_measured_board_size_scales_with_the_ruler():
    # A printer that came out 1% large: 35.35 mm squares, 26.51 mm markers.
    # Truth: 7 x 35.35 = 247.45 mm, 5 x 35.35 = 176.75 mm, and the meter values
    # OpenCV gets are the millimeter values over 1000.
    board = tool.board_geometry(35.35, 26.51)
    assert board.width_mm == pytest.approx(247.45, abs=1e-12)
    assert board.height_mm == pytest.approx(176.75, abs=1e-12)
    assert board.square_m == pytest.approx(0.03535, abs=1e-15)
    assert board.marker_m == pytest.approx(0.02651, abs=1e-15)
    assert board.inner_corners == 24  # the corner count does not depend on size


def test_nominal_print_size_math():
    # The PNG is drawn at the design size; the measured sizes only change the math.
    # This checks the pure size math only; drawing the PNG needs cv2 and is covered
    # by test_board_png_prints_at_true_size in test_calibrate_charuco.py.
    nominal = tool.board_geometry(tool.NOMINAL_SQUARE_MM, tool.NOMINAL_MARKER_MM)
    assert nominal.print_size_px == (round(245.0 * tool.PRINT_PX_PER_MM), round(175.0 * tool.PRINT_PX_PER_MM))


def test_command_line_rejects_bad_board_sizes(tmp_path):
    # These stop in the argument check, before OpenCV is imported or a camera opens.
    output = str(tmp_path / "c.json")
    for square, marker in (("20", "25"), ("35", "35"), ("4", "3"), ("35", "301"), ("nan", "26.25")):
        with pytest.raises(SystemExit) as error:
            tool.main(["--output", output, "--square-mm", square, "--marker-mm", marker])
        assert error.value.code == 2, (square, marker)
    assert not (tmp_path / "c.json").exists()


def test_command_line_size_check_messages(tmp_path, capsys):
    parser = tool.build_parser()
    output = str(tmp_path / "c.json")
    # Positive control: measured sizes inside the limits pass the check (this calls
    # only the argument check, so nothing imports cv2 or opens a camera).
    tool._check_args(parser, parser.parse_args(["--output", output, "--square-mm", "35.35", "--marker-mm", "26.51"]))
    # A marker larger than its square fails for that reason, not another one.
    with pytest.raises(SystemExit):
        tool._check_args(parser, parser.parse_args(["--output", output, "--square-mm", "20", "--marker-mm", "25"]))
    assert "--marker-mm must be smaller than --square-mm" in capsys.readouterr().err


# ---------------------------------------------------------------- RMS rule


@pytest.mark.parametrize(
    ("rms_px", "flagged"),
    [
        (0.0, False),
        (0.077, False),  # the synthetic calibration in test_calibrate_charuco.py
        (0.5, False),  # the rule is "above 0.5 px", so exactly 0.5 is accepted
        (0.5001, True),
        (0.9, True),
        (float("nan"), True),  # a broken number never counts as good
        (float("inf"), True),
    ],
)
def test_rms_rule(rms_px, flagged):
    assert tool.rms_needs_recalibration(rms_px) is flagged


def _full_coverage() -> np.ndarray:
    return np.full((3, 3), 5, dtype=int)


def test_rms_warning_follows_the_rule():
    matrix = np.array([[930.0, 0.0, 640.0], [0.0, 930.0, 360.0], [0.0, 0.0, 1.0]])
    good = tool.Calibration(0.5, matrix, np.zeros(5), 1280, 720, 20, 400)
    bad = tool.Calibration(0.5001, matrix, np.zeros(5), 1280, 720, 20, 400)
    assert tool.warnings_for(good, _full_coverage()) == []
    notes = tool.warnings_for(bad, _full_coverage())
    assert len(notes) == 1 and notes[0].startswith("RMS 0.500 px is above 0.5 px")


# ---------------------------------------------------------------- calibration JSON


def _calibration(rms_px: float = 0.0771234567891234) -> tool.Calibration:
    # Awkward floats on purpose, so a lossy float format would show up.
    matrix = np.array([[930.1234567890123, 0.0, 646.3000000000001], [0.0, 936.0 / 3.0 * 3.0 + 1e-9, 355.8], [0.0, 0.0, 1.0]])
    dist = np.array([0.1 + 0.2, -1.0 / 3.0, 1e-17, -2.5e-5, 0.0199999])
    return tool.Calibration(rms_px, matrix, dist, 1280, 720, 16, 384)


def _document() -> dict:
    return tool.calibration_document("5.0.0", _calibration(), 35.2, 26.4, "saved pictures matching views/*.png")


def test_document_fields():
    document = _document()
    assert document["schema"] == CALIBRATION_SCHEMA
    assert document["image_size_px"] == {"width": 1280, "height": 720}
    assert document["camera_matrix"][0] == [930.1234567890123, 0.0, 646.3000000000001]
    assert document["camera_matrix"][2] == [0.0, 0.0, 1.0]
    assert len(document["dist_coeffs"]) == 5
    assert "OpenCV 5.0.0" in document["method"]
    assert "measured square 35.2 mm" in document["method"] and "measured marker 26.4 mm" in document["method"]
    assert document["notes"] == "16 views, 384 corners, from saved pictures matching views/*.png"


def test_json_round_trip_is_byte_stable():
    rendered = tool.render_calibration_json(_document())
    again = tool.render_calibration_json(json.loads(rendered.decode("utf-8")))
    assert again == rendered
    text = rendered.decode("utf-8")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    keys = list(json.loads(text))
    assert keys == sorted(keys)
    assert text.splitlines()[1].startswith('  "')  # two-space indent


def test_written_file_reads_back_exactly(tmp_path):
    # The truth is the Calibration that went in: the reader must hand back the
    # same numbers bit for bit, and the digest of the bytes on disk.
    result = _calibration()
    rendered = tool.render_calibration_json(_document())
    path = tmp_path / "calibration-1280x720.json"
    tool._write_new_file(path, rendered, overwrite=False)
    assert path.read_bytes() == rendered
    model, digest = load_camera_calibration(path)
    assert digest == hashlib.sha256(rendered).hexdigest()
    assert model.source == "calibration_file"
    assert (model.width_px, model.height_px) == (1280, 720)
    k = model.matrix()
    for row, column in ((0, 0), (0, 2), (1, 1), (1, 2)):
        assert k[row, column] == result.camera_matrix[row, column]
    assert np.array_equal(model.distortion(), result.dist_coeffs)
    assert "calibration RMS 0.077 px" in model.description


def test_writer_refuses_to_overwrite(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_bytes(b"keep me")
    with pytest.raises(tool.CalibrationError, match="--overwrite"):
        tool._write_new_file(path, b"new", overwrite=False)
    assert path.read_bytes() == b"keep me"
    tool._write_new_file(path, b"new", overwrite=True)
    assert path.read_bytes() == b"new"
    assert not (tmp_path / "calibration.json.tmp").exists()


def test_non_finite_numbers_are_never_written():
    document = tool.calibration_document("5.0.0", _calibration(rms_px=float("nan")), 35.0, 26.25, "test")
    with pytest.raises(ValueError):
        tool.render_calibration_json(document)


# ---------------------------------------------------------------- print size and field of view


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _tiny_png() -> bytes:
    """A 1 by 1 gray PNG built by hand, so no image library is needed."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xff")  # filter byte 0, one white pixel
    return tool.PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", pixels) + _chunk(b"IEND", b"")


def test_phys_chunk_sets_true_print_size():
    # Truth: 10 px per mm is 10000 px per meter, unit 1 (meter). The board
    # page image is 2450 + 2 x 100 = 2650 px wide, so it prints 2650 / 10000 m = 265 mm.
    original = _tiny_png()
    assert tool.read_phys(original) is None
    stamped = tool.add_phys_chunk(original, tool.PRINT_PX_PER_MM * 1000)
    assert tool.read_phys(stamped) == (10000, 10000, 1)
    assert stamped[33 + 4:33 + 8] == b"pHYs"  # right after the 33-byte signature and IHDR
    assert stamped[:33] == original[:33] and stamped[33 + 21:] == original[33:]
    data = stamped[33 + 8:33 + 17]
    assert int.from_bytes(stamped[33 + 17:33 + 21], "big") == zlib.crc32(b"pHYs" + data)
    page_px = tool.board_geometry(35.0, 26.25).print_size_px[0] + 2 * tool.PRINT_MARGIN_MM * tool.PRINT_PX_PER_MM
    assert page_px == 2650 and page_px * 1000 // 10000 == 265  # whole millimeters, no float rounding


def test_phys_chunk_rejects_non_png():
    with pytest.raises(ValueError):
        tool.add_phys_chunk(b"not a png at all, just some bytes", 10000)


@pytest.mark.parametrize(
    ("fx", "width_px", "degrees"),
    [
        (640.0, 1280, 90.0),  # half width over fx is 1, atan(1) is 45 degrees
        (640.0 / math.tan(math.radians(30.0)), 1280, 60.0),
        (930.0, 1280, 69.069323143),  # 2 x atan(640 / 930), worked out once and written down
    ],
)
def test_horizontal_fov(fx, width_px, degrees):
    assert tool.horizontal_fov_deg(fx, width_px) == pytest.approx(degrees, abs=1e-9)
