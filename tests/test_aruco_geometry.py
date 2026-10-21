"""Checks for tools/aruco_geometry.py.

Most checks compare against something worked out separately: hand pinhole
numbers, OpenCV's own Rodrigues and projectPoints, the exact area and centroid
of a projected square, or a limiting case. Tests that need OpenCV skip when it
is not installed; the rest run on plain NumPy.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import aruco_geometry as geo

ROOT = Path(__file__).resolve().parents[1]
L = 0.15
CAMERA = geo.camera_from_hfov(1280, 720, 70.0)
FX = 640.0 / math.tan(math.radians(35.0))  # (width / 2) / tan(hfov / 2), worked out here


def facing(z: float, x: float = 0.0, y: float = 0.0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return geo.rvec_from_euler_deg(0.0, 0.0, 0.0), (x, y, z)


def projected_corners(rvec, tvec, camera=CAMERA, length=L) -> np.ndarray:
    return geo.project_points(geo.pose_points_camera(rvec, tvec, geo.marker_object_points(length)), camera)


def polygon_area_centroid(corners: np.ndarray) -> tuple[float, float, float]:
    """Shoelace area and area centroid of a simple polygon."""
    x, y = corners[:, 0], corners[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    return abs(float(area)), float(cx), float(cy)


# ---------------------------------------------------------------- import


def test_module_imports_without_loading_opencv():
    code = "import sys; import tools.aruco_geometry; print('cv2' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# ---------------------------------------------------------------- marker and camera


def test_marker_object_points_order_and_values():
    points = geo.marker_object_points(0.2)
    assert points.shape == (4, 3)
    assert points.dtype == np.float64
    expected = [[-0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, -0.1, 0.0]]
    assert np.array_equal(points, np.array(expected))


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf, True, "0.1"])
def test_marker_object_points_rejects_bad_length(bad):
    with pytest.raises(ValueError):
        geo.marker_object_points(bad)


def test_camera_from_hfov_hand_numbers():
    # tan(35 deg) = 0.70020754, so fx = 640 / 0.70020754 = 914.0147 px.
    assert CAMERA.camera_matrix[0][0] == pytest.approx(914.0147, abs=1e-4)
    assert CAMERA.camera_matrix[0][0] == pytest.approx(FX, rel=1e-12)
    assert CAMERA.camera_matrix[1][1] == CAMERA.camera_matrix[0][0]
    assert CAMERA.camera_matrix[0][2] == 640.0
    assert CAMERA.camera_matrix[1][2] == 360.0
    assert CAMERA.dist_coeffs == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert CAMERA.source == "hfov_estimate"
    assert "914.0" in CAMERA.description and "70.0 degree" in CAMERA.description
    assert CAMERA.matrix().shape == (3, 3) and CAMERA.matrix().dtype == np.float64
    assert CAMERA.distortion().shape == (5,) and CAMERA.distortion().dtype == np.float64
    # Limiting case: a 90 degree field of view gives fx = width / 2.
    assert geo.camera_from_hfov(1000, 500, 90.0).camera_matrix[0][0] == pytest.approx(500.0)


@pytest.mark.parametrize(
    ("width", "height", "hfov"),
    [(1280, 720, 1.0), (1280, 720, 179.0), (1280, 720, math.nan), (0, 720, 70.0), (1280, -1, 70.0), (1280.0, 720, 70.0), (True, 720, 70.0)],
)
def test_camera_from_hfov_rejects_bad_input(width, height, hfov):
    with pytest.raises(ValueError):
        geo.camera_from_hfov(width, height, hfov)


def test_camera_model_rejects_bad_fields():
    good = {"camera_matrix": ((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0)), "dist_coeffs": (0.0,) * 5, "width_px": 1280, "height_px": 720, "source": "hfov_estimate", "description": "test"}
    geo.CameraModel(**good)
    for change in (
        {"camera_matrix": ((900.0, 0.0, 640.0), (0.0, 900.0, 360.0))},
        {"camera_matrix": ((-1.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0))},
        {"camera_matrix": ((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 2.0))},
        {"dist_coeffs": (0.0,) * 6},
        {"dist_coeffs": (0.0, math.nan, 0.0, 0.0)},
        {"width_px": 0},
        {"source": "guess"},
        {"description": " "},
    ):
        with pytest.raises(ValueError):
            geo.CameraModel(**{**good, **change})


# ---------------------------------------------------------------- calibration file


def calibration_dict(**changes):
    value = {
        "schema": geo.CALIBRATION_SCHEMA,
        "camera_matrix": [[910.5, 0, 641.2], [0, 912.0, 359.4], [0, 0, 1]],
        "dist_coeffs": [0.1, -0.2, 0.001, -0.002, 0.05],
        "image_size_px": {"width": 1280, "height": 720},
        "rms_reprojection_error_px": 0.31,
        "method": "chessboard, 25 views",
        "notes": "example",
    }
    value.update(changes)
    return value


def test_load_camera_calibration_reads_file_and_hashes_raw_bytes(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(calibration_dict(), indent=2), encoding="utf-8")
    model, digest = geo.load_camera_calibration(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert model.source == "calibration_file"
    assert model.camera_matrix == ((910.5, 0.0, 641.2), (0.0, 912.0, 359.4), (0.0, 0.0, 1.0))
    assert model.dist_coeffs == (0.1, -0.2, 0.001, -0.002, 0.05)
    assert (model.width_px, model.height_px) == (1280, 720)
    assert "cal.json" in model.description
    # The optional fields really are optional.
    minimal = calibration_dict()
    for key in ("rms_reprojection_error_px", "method", "notes"):
        minimal.pop(key)
    path.write_text(json.dumps(minimal), encoding="utf-8")
    assert geo.load_camera_calibration(path)[0].width_px == 1280


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (json.dumps(calibration_dict(extra=1)), "extra"),
        (json.dumps(calibration_dict(schema="other")), "schema"),
        (json.dumps(calibration_dict(camera_matrix=[[910.5, 0, 641.2], [0, 912.0, 359.4]])), "camera_matrix"),
        (json.dumps(calibration_dict(camera_matrix=[[0, 0, 641.2], [0, 912.0, 359.4], [0, 0, 1]])), "camera_matrix"),
        (json.dumps(calibration_dict(camera_matrix=[[910.5, 0, 641.2], [0, 912.0, 359.4], [0, 0, 2]])), "camera_matrix"),
        (json.dumps(calibration_dict(camera_matrix=[[910.5, 0, 641.2], [0, 912.0, "359"], [0, 0, 1]])), "camera_matrix"),
        (json.dumps(calibration_dict(dist_coeffs=[0.1, 0.2, 0.3])), "dist_coeffs"),
        (json.dumps(calibration_dict(image_size_px={"width": 1280})), "image_size_px"),
        (json.dumps(calibration_dict(image_size_px={"width": 1280, "height": 0})), "image_size_px"),
        (json.dumps(calibration_dict(rms_reprojection_error_px=-1)), "rms_reprojection_error_px"),
        (json.dumps(calibration_dict(method=3)), "method"),
        (json.dumps(calibration_dict()).replace("0.05", "NaN"), "NaN"),
        ('{"schema": "zephyr-camera-calibration-1", "schema": "zephyr-camera-calibration-1"}', "duplicate"),
        (json.dumps({"schema": geo.CALIBRATION_SCHEMA}), "camera_matrix"),
    ],
)
def test_load_camera_calibration_is_strict(tmp_path, text, field):
    path = tmp_path / "bad-cal.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as info:
        geo.load_camera_calibration(path)
    assert str(path) in str(info.value)
    assert field in str(info.value)


def test_check_resolution():
    geo.check_resolution(CAMERA, 1280, 720)
    with pytest.raises(ValueError, match="640x480"):
        geo.check_resolution(CAMERA, 640, 480)


# ---------------------------------------------------------------- rotations and the pinhole by hand


def test_identity_pose_faces_the_camera_upright():
    rvec = geo.rvec_from_euler_deg(0.0, 0.0, 0.0)
    assert np.allclose(np.abs(rvec), [math.pi, 0.0, 0.0], atol=1e-12)
    axes = geo.pose_points_camera(rvec, (0.0, 0.0, 0.0), np.eye(3))
    assert np.allclose(axes, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], atol=1e-12)


def test_pinhole_hand_numbers_for_a_facing_marker():
    # A square of side L at distance Z spans fx * L / Z pixels: 914.0147 * 0.15 / 1.0 = 137.102 px.
    rvec, tvec = facing(1.0)
    corners = projected_corners(rvec, tvec)
    half = FX * L / 2.0
    expected = [[640 - half, 360 - half], [640 + half, 360 - half], [640 + half, 360 + half], [640 - half, 360 + half]]
    assert np.allclose(corners, expected, atol=1e-9)
    assert corners[1, 0] - corners[0, 0] == pytest.approx(137.102, abs=1e-3)
    # Twice as far gives half the size.
    far = projected_corners(*facing(2.0))
    assert far[1, 0] - far[0, 0] == pytest.approx(137.102 / 2.0, abs=1e-3)
    # The printed top-left corner shows up at the upper left of the image.
    assert corners[0, 0] < 640 and corners[0, 1] < 360
    # Moving the marker 0.1 m to the right at 1 m moves it fx * 0.1 = 91.40 px right.
    shifted = projected_corners(*facing(1.0, x=0.1))
    assert np.allclose(shifted - corners, [[91.40147, 0.0]] * 4, atol=1e-4)


def test_euler_angle_signs_match_the_docstring():
    up = np.array([[0.0, 1.0, 0.0]])
    # Roll +90: the marker's up arrow turns counterclockwise on screen, to image left (camera -x).
    assert np.allclose(geo.pose_points_camera(geo.rvec_from_euler_deg(90, 0, 0), (0, 0, 0), up), [[-1, 0, 0]], atol=1e-12)
    # Pitch +30: the top edge comes toward the camera (smaller z than the center).
    top = geo.pose_points_camera(geo.rvec_from_euler_deg(0, 30, 0), (0, 0, 1), [[0.0, L / 2, 0.0]])
    assert top[0, 2] == pytest.approx(1.0 - (L / 2) * math.sin(math.radians(30)), abs=1e-12)
    # Yaw +30: the right edge moves away from the camera.
    right = geo.pose_points_camera(geo.rvec_from_euler_deg(0, 0, 30), (0, 0, 1), [[L / 2, 0.0, 0.0]])
    assert right[0, 2] == pytest.approx(1.0 + (L / 2) * math.sin(math.radians(30)), abs=1e-12)


def test_rodrigues_limiting_cases():
    points = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]])
    assert np.allclose(geo.pose_points_camera((0, 0, 0), (0, 0, 0), points), points, atol=0)
    assert np.allclose(geo.pose_points_camera((1e-14, 0, 0), (0, 0, 0), points), points, atol=1e-12)
    # 90 degrees about z sends x to y.
    assert np.allclose(geo.pose_points_camera((0, 0, math.pi / 2), (0, 0, 0), [[1, 0, 0]]), [[0, 1, 0]], atol=1e-12)
    # A full turn changes nothing.
    assert np.allclose(geo.pose_points_camera((2 * math.pi, 0, 0), (0, 0, 0), points), points, atol=1e-12)
    with pytest.raises(ValueError):
        geo.pose_points_camera((0, 0), (0, 0, 0), points)
    with pytest.raises(ValueError):
        geo.pose_points_camera((0, 0, 0), (0, math.nan, 0), points)


@pytest.mark.parametrize("angles", [(0, 0, 0), (0, 0, 180), (179.9999, 0, 0), (0, 0.0001, 0), (30, -25, 15), (-170, 60, 80)])
def test_euler_rvec_rebuilds_the_same_rotation(angles):
    # Independent of the log map: rotate the axes with the returned rvec and compare
    # with the product of elementary rotations applied one after another.
    rvec = geo.rvec_from_euler_deg(*angles)
    roll, pitch, yaw = (math.radians(a) for a in angles)
    expected = []
    for axis in np.eye(3):
        p = axis
        p = np.array([p[0] * math.cos(roll) - p[1] * math.sin(roll), p[0] * math.sin(roll) + p[1] * math.cos(roll), p[2]])
        p = np.array([p[0], p[1] * math.cos(pitch) - p[2] * math.sin(pitch), p[1] * math.sin(pitch) + p[2] * math.cos(pitch)])
        p = np.array([p[0] * math.cos(yaw) + p[2] * math.sin(yaw), p[1], -p[0] * math.sin(yaw) + p[2] * math.cos(yaw)])
        expected.append([p[0], -p[1], -p[2]])  # the 180 degree turn about x
    assert np.allclose(geo.pose_points_camera(rvec, (0, 0, 0), np.eye(3)), expected, atol=1e-9)
    assert np.linalg.norm(rvec) <= math.pi + 1e-12


# ---------------------------------------------------------------- projection


def test_project_points_distortion_limiting_cases():
    distorted = geo.CameraModel(CAMERA.camera_matrix, (-0.3, 0.1, 0.002, -0.001, 0.02), 1280, 720, "hfov_estimate", "test")
    # A point on the optical axis is not moved by distortion.
    assert np.allclose(geo.project_points([[0.0, 0.0, 2.0]], distorted), [[640.0, 360.0]], atol=1e-12)
    # Radial only, by hand: x' = 0.1, y' = 0, r^2 = 0.01, factor 1 + k1 r^2 = 1 - 0.002.
    radial = geo.CameraModel(CAMERA.camera_matrix, (-0.2, 0.0, 0.0, 0.0), 1280, 720, "hfov_estimate", "test")
    assert np.allclose(geo.project_points([[0.1, 0.0, 1.0]], radial), [[640.0 + FX * 0.1 * 0.998, 360.0]], atol=1e-9)
    # Higher-order terms that are zero are fine, nonzero ones are refused.
    eight = geo.CameraModel(CAMERA.camera_matrix, (0.0,) * 8, 1280, 720, "hfov_estimate", "test")
    assert np.allclose(geo.project_points([[0.1, 0.0, 1.0]], eight), [[640.0 + FX * 0.1, 360.0]])
    k4 = geo.CameraModel(CAMERA.camera_matrix, (0.0,) * 5 + (0.1, 0.0, 0.0), 1280, 720, "hfov_estimate", "test")
    with pytest.raises(ValueError, match="k3"):
        geo.project_points([[0.1, 0.0, 1.0]], k4)
    with pytest.raises(ValueError, match="front"):
        geo.project_points([[0.1, 0.0, 0.0]], CAMERA)


def test_center_is_the_diagonal_intersection_not_the_corner_mean():
    # Pitch 50 degrees at 0.4 m: the near (top) half looks bigger, so the corner
    # mean slides toward the top of the image by about 16 px.
    rvec, tvec = geo.rvec_from_euler_deg(0, 50, 0), (0.02, -0.01, 0.4)
    corners = projected_corners(rvec, tvec)
    true_center = geo.project_points(geo.pose_points_camera(rvec, tvec, [[0.0, 0.0, 0.0]]), CAMERA)[0]
    center = np.array(geo.marker_center_px(corners))
    assert np.allclose(center, true_center, atol=1e-9)
    mean_gap = corners.mean(axis=0) - true_center
    assert mean_gap[1] < -10.0
    # For a marker facing the camera the two agree.
    flat = projected_corners(*facing(1.0))
    assert np.allclose(geo.marker_center_px(flat), flat.mean(axis=0), atol=1e-9)
    assert np.allclose(geo.marker_center_px(flat[None]), flat.mean(axis=0), atol=1e-9)
    with pytest.raises(ValueError):
        geo.marker_center_px([[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        geo.marker_center_px([[0, 0], [1, 1], [2, 2]])


def test_bbox_from_corners():
    corners = [[10.0, 20.0], [30.0, 21.0], [29.0, 40.0], [11.0, 39.0]]
    # Boxes use edge coordinates: the corners (center convention) plus half a pixel.
    assert geo.bbox_from_corners(corners, 1280, 720) == {"x": 10.5, "y": 20.5, "width": 20.0, "height": 20.0}
    assert geo.bbox_from_corners(np.array(corners)[None], 1280, 720) == {"x": 10.5, "y": 20.5, "width": 20.0, "height": 20.0}
    clipped = geo.bbox_from_corners([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]], 1280, 720)
    assert clipped == {"x": 0.0, "y": 0.0, "width": 5.5, "height": 5.5}
    right = geo.bbox_from_corners([[1270.0, 700.0], [1290.0, 700.0], [1290.0, 730.0], [1270.0, 730.0]], 1280, 720)
    assert right == {"x": 1270.5, "y": 700.5, "width": 9.5, "height": 19.5}
    assert geo.bbox_from_corners([[-50.0, 10.0], [-20.0, 10.0], [-20.0, 30.0], [-50.0, 30.0]], 1280, 720) is None


# ---------------------------------------------------------------- timing


def test_summarize_frame_timing_on_hand_made_intervals():
    times = [0.0, 0.1, 0.2, 0.4, 0.5, 0.5]  # intervals 0.1, 0.1, 0.2, 0.1, 0.0
    summary = geo.summarize_frame_timing(times, 30)
    assert summary["frames"] == 6
    assert summary["duration_s"] == pytest.approx(0.5)
    assert summary["median_interval_s"] == pytest.approx(0.1)
    assert summary["measured_frame_rate_hz"] == pytest.approx(10.0)
    assert summary["max_interval_s"] == pytest.approx(0.2)
    assert summary["late_frames"] == 1
    assert summary["non_increasing"] == 1
    assert summary["nominal_frame_rate_hz"] == 30.0
    json.dumps(summary, allow_nan=False)


def test_summarize_frame_timing_small_and_bad_input():
    empty = geo.summarize_frame_timing([], None)
    assert empty["frames"] == 0 and empty["duration_s"] is None and empty["measured_frame_rate_hz"] is None
    one = geo.summarize_frame_timing([3.0], 30.0)
    assert one["frames"] == 1 and one["duration_s"] == 0.0 and one["median_interval_s"] is None
    json.dumps(empty, allow_nan=False)
    json.dumps(one, allow_nan=False)
    with pytest.raises(ValueError):
        geo.summarize_frame_timing([0.0, math.nan], None)
    with pytest.raises(ValueError):
        geo.summarize_frame_timing([0.0, 1.0], 0.0)


def test_replay_capture_times_do_not_drift():
    times = geo.replay_capture_times(300001, 30.0, start_s=5.0)
    assert len(times) == 300001
    assert times[0] == 5.0
    assert times[-1] == 5.0 + 300000 / 30.0
    assert times[30] == pytest.approx(6.0, abs=1e-12)
    assert geo.replay_capture_times(0, 30.0) == []
    for bad in ((-1, 30.0), (10, 0.0), (10, math.inf), (1.5, 30.0)):
        with pytest.raises(ValueError):
            geo.replay_capture_times(*bad)


# ---------------------------------------------------------------- OpenCV cross-checks


def test_numpy_rodrigues_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    rvecs = [np.zeros(3), np.array([1e-9, 0, 0]), np.array([math.pi, 0, 0]), np.array([0, 0, math.pi - 1e-7])]
    rvecs += [rng.normal(size=3) * rng.uniform(0.1, 3.0) for _ in range(20)]
    for rvec in rvecs:
        expected = cv2.Rodrigues(rvec.reshape(3, 1))[0]
        ours = geo.pose_points_camera(rvec, (0, 0, 0), np.eye(3)).T
        assert np.allclose(ours, expected, atol=1e-12)


def test_euler_rvec_matches_cv2_composition():
    cv2 = pytest.importorskip("cv2")

    def rod(v):
        return cv2.Rodrigues(np.array(v, dtype=np.float64).reshape(3, 1))[0]

    for roll, pitch, yaw in [(0, 0, 0), (25, -40, 70), (-120, 10, 5)]:
        expected = rod([math.pi, 0, 0]) @ rod([0, math.radians(yaw), 0]) @ rod([math.radians(pitch), 0, 0]) @ rod([0, 0, math.radians(roll)])
        assert np.allclose(rod(geo.rvec_from_euler_deg(roll, pitch, yaw)), expected, atol=1e-12)


def test_project_points_matches_cv2_with_distortion():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    points = np.column_stack([rng.uniform(-0.6, 0.6, 50), rng.uniform(-0.4, 0.4, 50), rng.uniform(0.5, 3.0, 50)])
    for dist in [(0.0,) * 5, (-0.28, 0.09, 0.0015, -0.0012, -0.01), (0.12, -0.3, -0.002, 0.003), (0.05, 0.01, 0.001, 0.001, 0.002, 0.0, 0.0, 0.0)]:
        camera = geo.CameraModel(((905.0, 0.0, 637.5), (0.0, 910.0, 362.25), (0.0, 0.0, 1.0)), dist, 1280, 720, "calibration_file", "test")
        expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.matrix(), camera.distortion())
        assert np.allclose(geo.project_points(points, camera), expected.reshape(-1, 2), atol=1e-9)


def test_pose_from_exact_corners_recovers_the_pose():
    cv2 = pytest.importorskip("cv2")
    # The first two face the camera exactly, where IPPE_SQUARE alone goes wrong in OpenCV 5.0.
    cases = [((0, 0, 0), (0.0, 0.0, 1.0)), ((5, 0, 0), (0.3, 0.2, 2.0)), ((10, 25, -15), (0.2, -0.1, 2.0)), ((-40, -30, 35), (-0.3, 0.15, 0.6))]
    for angles, tvec in cases:
        rvec = geo.rvec_from_euler_deg(*angles)
        corners = projected_corners(rvec, tvec)
        pose = geo.estimate_marker_pose(cv2, corners.astype(np.float32)[None], CAMERA, L)
        assert pose is not None
        assert np.allclose(pose.tvec, tvec, atol=1e-5)
        assert np.allclose(cv2.Rodrigues(np.array(pose.rvec))[0], cv2.Rodrigues(np.array(rvec))[0], atol=1e-4)
        assert pose.reprojection_error_px < 1e-3
        assert pose.ambiguity_ratio is None or 0.0 <= pose.ambiguity_ratio <= 1.0


def test_pose_keeps_every_candidate_and_the_kept_one_is_the_best_in_front():
    cv2 = pytest.importorskip("cv2")
    rvec, tvec = geo.rvec_from_euler_deg(10, 25, -15), (0.2, -0.1, 2.0)
    corners = projected_corners(rvec, tvec)
    # A little corner noise so the two IPPE tilts get different errors.
    corners = corners + np.random.default_rng(3).normal(0.0, 0.05, size=corners.shape)
    pose = geo.estimate_marker_pose(cv2, corners, CAMERA, L)
    assert pose is not None
    methods = [c["method"] for c in pose.candidates]
    assert methods.count("IPPE_SQUARE") == 2 and methods.count("SQPNP") >= 1
    assert methods == sorted(methods, key=lambda m: m != "IPPE_SQUARE")  # IPPE first, then SQPNP
    assert all(set(c) == {"method", "rvec", "tvec", "reprojection_error_px"} for c in pose.candidates)
    json.dumps(list(pose.candidates), allow_nan=False)  # plain JSON numbers, no NumPy types
    # Each candidate's error, worked out again here with cv2.projectPoints.
    object_points = geo.marker_object_points(L)
    for candidate in pose.candidates:
        projected, _ = cv2.projectPoints(object_points, np.array(candidate["rvec"]), np.array(candidate["tvec"]),
                                         CAMERA.matrix(), CAMERA.distortion())
        rms = math.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1)))
        assert candidate["reprojection_error_px"] == pytest.approx(rms, rel=1e-9, abs=1e-12)
    # The kept pose is the lowest-error candidate with z > 0, unchanged by listing the others.
    in_front = [c for c in pose.candidates if c["tvec"][2] > 0]
    best = min(in_front, key=lambda c: c["reprojection_error_px"])
    assert list(pose.tvec) == best["tvec"] and list(pose.rvec) == best["rvec"]
    assert pose.reprojection_error_px == best["reprojection_error_px"]
    # The other IPPE tilt is listed and is a real alternative, about 2 x 25 degrees away.
    other_tilts = [c for c in pose.candidates if c["method"] == "IPPE_SQUARE" and c["rvec"] != best["rvec"]]
    assert other_tilts and other_tilts[0]["reprojection_error_px"] >= pose.reprojection_error_px


def test_pose_returns_none_for_degenerate_corners():
    cv2 = pytest.importorskip("cv2")
    assert geo.estimate_marker_pose(cv2, np.full((4, 2), 100.0), CAMERA, L) is None
    with pytest.raises(ValueError):
        geo.estimate_marker_pose(cv2, np.zeros((3, 2)), CAMERA, L)


# ---------------------------------------------------------------- renderer


def aruco(cv2):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return dictionary, cv2.aruco.ArucoDetector(dictionary, params)


def detect_one(cv2, detector, frame):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert ids is not None and ids.ravel().tolist() == [0]
    return corners[0].reshape(4, 2).astype(np.float64)


@pytest.mark.parametrize("angles", [(0, 0, 0), (30, -25, 15), (17, 40, -33)])
@pytest.mark.parametrize("z", [0.5, 3.0])
def test_render_area_and_centroid_are_exact(angles, z):
    # A plain black square on a white background: the total darkness must equal
    # the area of the projected square, and its centroid the polygon centroid.
    # A half-pixel convention slip would move the centroid by 0.5 px.
    cv2 = pytest.importorskip("cv2")
    rvec, tvec = geo.rvec_from_euler_deg(*angles), (0.031, -0.017, z)
    frame = geo.render_marker_frame(cv2, np.zeros((8, 8), np.uint8), CAMERA, rvec, tvec, L, quiet_zone=0.0, background=255)
    coverage = (255.0 - frame[:, :, 0]) / 255.0
    area, cx, cy = polygon_area_centroid(projected_corners(rvec, tvec))
    rows, cols = np.mgrid[0:720, 0:1280]
    total = coverage.sum()
    assert total == pytest.approx(area, rel=2e-4)
    assert (coverage * cols).sum() / total == pytest.approx(cx, abs=0.01)
    assert (coverage * rows).sum() / total == pytest.approx(cy, abs=0.01)


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("angles", [(0, 0, 0), (10, 20, 0), (30, -25, 15)])
def test_render_detect_pose_recovers_known_tvec(z, angles):
    cv2 = pytest.importorskip("cv2")
    dictionary, detector = aruco(cv2)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200, borderBits=1)
    rvec, tvec = geo.rvec_from_euler_deg(*angles), (0.03, -0.02, z)
    frame = geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L)
    detected = detect_one(cv2, detector, frame)
    truth = projected_corners(rvec, tvec)
    errors = detected - truth
    # No shift of the whole marker (that is where a convention slip would show).
    assert np.all(np.abs(errors.mean(axis=0)) < 0.1)
    # OpenCV's subpixel refinement pulls these outer corners about 0.1 to 0.3 px
    # inward (measured); the renderer itself is exact (see the area test).
    assert np.max(np.linalg.norm(errors, axis=1)) < 0.4
    pose = geo.estimate_marker_pose(cv2, detected, CAMERA, L)
    assert pose is not None
    assert abs(pose.tvec[0] - tvec[0]) < 1e-3 and abs(pose.tvec[1] - tvec[1]) < 1e-3
    assert abs(pose.tvec[2] - z) < 0.01 * z
    assert pose.reprojection_error_px < 0.3


def test_render_orientation_through_the_detector():
    cv2 = pytest.importorskip("cv2")
    dictionary, detector = aruco(cv2)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 120, borderBits=1)
    upright = detect_one(cv2, detector, geo.render_marker_frame(cv2, marker, CAMERA, *facing(1.0), L))
    # Detected corner 0 is the printed top-left and sits at the upper left.
    assert np.argmin(upright.sum(axis=1)) == 0
    # Roll +90 (counterclockwise on screen) carries the printed top-left to the lower left.
    rolled = detect_one(cv2, detector, geo.render_marker_frame(cv2, marker, CAMERA, geo.rvec_from_euler_deg(90, 0, 0), (0, 0, 1.0), L))
    assert rolled[0, 0] < 640 and rolled[0, 1] > 360


def test_render_with_distortion_puts_corners_at_distorted_projection():
    cv2 = pytest.importorskip("cv2")
    dictionary, detector = aruco(cv2)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200, borderBits=1)
    camera = geo.CameraModel(CAMERA.camera_matrix, (-0.25, 0.08, 0.001, -0.0008, 0.0), 1280, 720, "calibration_file", "test")
    rvec, tvec = geo.rvec_from_euler_deg(10, 15, -20), (0.35, 0.2, 1.2)
    frame = geo.render_marker_frame(cv2, marker, camera, rvec, tvec, L)
    detected = detect_one(cv2, detector, frame)
    distorted = projected_corners(rvec, tvec, camera)
    undistorted = projected_corners(rvec, tvec)
    assert np.max(np.linalg.norm(detected - distorted, axis=1)) < 0.4
    assert np.min(np.linalg.norm(detected - undistorted, axis=1)) > 2.0  # the distortion is visible
    pose = geo.estimate_marker_pose(cv2, detected, camera, L)
    assert np.allclose(pose.tvec, tvec, atol=0.01 * tvec[2])


def test_render_options_noise_blur_and_errors():
    cv2 = pytest.importorskip("cv2")
    dictionary, _ = aruco(cv2)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 60, borderBits=1)
    rvec, tvec = facing(2.0)
    clean = geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L)
    assert clean.shape == (720, 1280, 3) and clean.dtype == np.uint8
    assert np.array_equal(clean[:, :, 0], clean[:, :, 1]) and np.array_equal(clean[:, :, 0], clean[:, :, 2])
    assert clean[5, 5, 0] == 128  # background far from the paper
    corners = projected_corners(rvec, tvec)
    # The white quiet zone sits just outside the black border.
    gap = round(float(corners[0, 1]) - 0.1 * FX * L / 2.0)
    assert clean[gap, 640, 0] == 255
    noisy_a = geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L, pixel_noise_std=4.0, rng=np.random.default_rng(5))
    noisy_b = geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L, pixel_noise_std=4.0, rng=np.random.default_rng(5))
    assert np.array_equal(noisy_a, noisy_b)
    background_noise = noisy_a[:200, :200, 0].astype(float) - 128.0
    assert background_noise.std() == pytest.approx(4.0, rel=0.05)
    assert abs(background_noise.mean()) < 0.1
    blurred = geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L, blur_sigma_px=2.0)
    row = round(float(corners[0, 1]) + 2)
    assert np.abs(np.diff(blurred[row, 600:680, 0].astype(int))).max() < np.abs(np.diff(clean[row, 600:680, 0].astype(int))).max()
    for kwargs in ({"pixel_noise_std": -1.0}, {"blur_sigma_px": math.nan}, {"quiet_zone": -0.1}, {"background": 300}):
        with pytest.raises(ValueError):
            geo.render_marker_frame(cv2, marker, CAMERA, rvec, tvec, L, **kwargs)
    with pytest.raises(ValueError):
        geo.render_marker_frame(cv2, marker[:, :50], CAMERA, rvec, tvec, L)
    with pytest.raises(ValueError):
        geo.render_marker_frame(cv2, marker.astype(np.float32), CAMERA, rvec, tvec, L)
    with pytest.raises(ValueError):
        geo.render_marker_frame(cv2, marker, CAMERA, rvec, (0.0, 0.0, -1.0), L)
