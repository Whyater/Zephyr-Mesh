"""Pose, projection and timing math for the ArUco capture tool.

Everything here is plain NumPy. The few functions that need OpenCV take the
``cv2`` module as their first argument, so ``import tools.aruco_geometry`` works
on a machine without OpenCV and the pure math can be tested on its own.

Conventions used throughout:

* Camera frame (OpenCV): x to the right of the image, y down the image, z
  straight out of the lens. Units are meters.
* Marker frame: origin at the center of the black square, x to the right of
  the printed marker, y toward its top, z out of the printed face (toward the
  camera when the marker faces it).
* Pixel coordinates follow OpenCV: the integer coordinate (u, v) is the center
  of pixel column u, row v. The image covers -0.5 .. width - 0.5 horizontally.
  ``project_points``, ``render_marker_frame`` and the OpenCV detector all use
  this convention, so their numbers can be compared directly.
* Corner order is the ArUco order: top-left, top-right, bottom-right,
  bottom-left of the printed marker (clockwise in the image when the marker
  faces the camera upright).

None of this says anything about how well a real camera performs. The renderer
makes clean synthetic frames for tests and sanity checks only.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_SCHEMA = "zephyr-camera-calibration-1"
CAMERA_SOURCES = {"calibration_file", "hfov_estimate"}
DIST_COEFF_LENGTHS = {4, 5, 8, 12, 14}
_CALIBRATION_KEYS = {
    "schema",
    "camera_matrix",
    "dist_coeffs",
    "image_size_px",
    "rms_reprojection_error_px",
    "method",
    "notes",
}
_CALIBRATION_REQUIRED = {"schema", "camera_matrix", "dist_coeffs", "image_size_px"}

# The renderer draws at this many sub-pixels per output pixel (in each
# direction) and then averages them down with INTER_AREA. See render_marker_frame.
RENDER_SUPERSAMPLE = 4


# ---------------------------------------------------------------- small checks


def _finite(value: Any, field: str) -> float:
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    result = float(value) if is_number else math.nan
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _corners(corners_px: Any) -> np.ndarray:
    """Return corners as a (4, 2) float64 array. Accepts (4, 2) or (1, 4, 2)."""
    array = np.asarray(corners_px, dtype=np.float64)
    if array.shape == (1, 4, 2):
        array = array[0]
    if array.shape != (4, 2):
        raise ValueError(f"corners_px must have shape (4, 2) or (1, 4, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("corners_px must be finite")
    return array


# ---------------------------------------------------------------- marker and camera


def marker_object_points(marker_length_m: float) -> np.ndarray:
    """Corners of the black square in the marker frame, shape (4, 3), meters.

    Order is top-left, top-right, bottom-right, bottom-left, which is both the
    ArUco detector order and the order SOLVEPNP_IPPE_SQUARE expects.
    """
    length = _finite(marker_length_m, "marker_length_m")
    if length <= 0:
        raise ValueError("marker_length_m must be greater than 0")
    half = length / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera with OpenCV distortion coefficients.

    ``camera_matrix`` is [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] in pixels.
    ``dist_coeffs`` uses the OpenCV order (k1, k2, p1, p2, k3, ...).
    """

    camera_matrix: tuple[tuple[float, float, float], ...]
    dist_coeffs: tuple[float, ...]
    width_px: int
    height_px: int
    source: str
    description: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a 3x3 array of finite numbers")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera_matrix fx and fy must be greater than 0")
        if matrix[0, 1] != 0 or matrix[1, 0] != 0 or tuple(matrix[2]) != (0.0, 0.0, 1.0):
            raise ValueError("camera_matrix must look like [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]")
        dist = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        if dist.size not in DIST_COEFF_LENGTHS or not np.all(np.isfinite(dist)):
            raise ValueError("dist_coeffs must hold 4, 5, 8, 12 or 14 finite numbers")
        _positive_int(self.width_px, "width_px")
        _positive_int(self.height_px, "height_px")
        if self.source not in CAMERA_SOURCES:
            raise ValueError(f"source must be one of {sorted(CAMERA_SOURCES)}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        # Store plain Python floats so the model hashes and prints cleanly.
        object.__setattr__(self, "camera_matrix", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "dist_coeffs", tuple(float(v) for v in dist))
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "height_px", int(self.height_px))

    def matrix(self) -> np.ndarray:
        return np.array(self.camera_matrix, dtype=np.float64)

    def distortion(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float64)


def camera_from_hfov(width_px: int, height_px: int, hfov_deg: float) -> CameraModel:
    """A guessed camera from the horizontal field of view, with no distortion.

    fx = fy = (width / 2) / tan(hfov / 2), cx = width / 2, cy = height / 2.
    Under the pixel-center convention the exact middle of the image is
    ((width - 1) / 2, (height - 1) / 2); the half pixel is far smaller than the
    error of a field of view guess, and it does not matter for synthetic frames
    because the renderer and the pose solver use the same model.
    """
    width = _positive_int(width_px, "width_px")
    height = _positive_int(height_px, "height_px")
    hfov = _finite(hfov_deg, "hfov_deg")
    if not 1.0 < hfov < 179.0:
        raise ValueError("hfov_deg must be between 1 and 179 degrees")
    focal = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return CameraModel(
        camera_matrix=((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        width_px=width,
        height_px=height,
        source="hfov_estimate",
        description=f"fx=fy={focal:.1f} px from a {hfov:.1f} degree horizontal field of view guess",
    )


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not allowed (non-finite number)")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def load_camera_calibration(path: str | Path) -> tuple[CameraModel, str]:
    """Read a ``zephyr-camera-calibration-1`` JSON file.

    Returns the camera model (source "calibration_file") and the SHA-256 hex
    digest of the raw file bytes, which goes in the log as calibration_ref.
    Any problem raises ValueError naming the file and the field.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    def fail(field: str, message: str) -> ValueError:
        return ValueError(f"{path}: {field}: {message}")

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: not a valid UTF-8 JSON file: {exc}") from exc
    if not isinstance(data, dict):
        raise fail("file", "must hold one JSON object")
    unknown = sorted(set(data) - _CALIBRATION_KEYS)
    if unknown:
        raise fail(unknown[0], f"unknown field(s): {', '.join(unknown)}")
    missing = sorted(_CALIBRATION_REQUIRED - set(data))
    if missing:
        raise fail(missing[0], f"missing required field(s): {', '.join(missing)}")
    if data["schema"] != CALIBRATION_SCHEMA:
        raise fail("schema", f"must be {CALIBRATION_SCHEMA!r}")

    matrix = data["camera_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        raise fail("camera_matrix", "must be a 3x3 list of numbers")
    try:
        values = [[_finite(v, "value") for v in row] for row in matrix]
    except ValueError as exc:
        raise fail("camera_matrix", str(exc)) from exc
    if values[0][0] <= 0 or values[1][1] <= 0:
        raise fail("camera_matrix", "fx and fy must be greater than 0")
    if values[2] != [0.0, 0.0, 1.0]:
        raise fail("camera_matrix", "last row must be [0, 0, 1]")
    if values[0][1] != 0.0 or values[1][0] != 0.0:
        raise fail("camera_matrix", "must look like [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew)")

    dist = data["dist_coeffs"]
    if not isinstance(dist, list) or len(dist) not in DIST_COEFF_LENGTHS:
        raise fail("dist_coeffs", "must be a list of 4, 5, 8, 12 or 14 numbers")
    try:
        dist_values = [_finite(v, "value") for v in dist]
    except ValueError as exc:
        raise fail("dist_coeffs", str(exc)) from exc

    size = data["image_size_px"]
    if not isinstance(size, dict) or set(size) != {"width", "height"}:
        raise fail("image_size_px", "must be an object with exactly width and height")
    try:
        width = _positive_int(size["width"], "width")
        height = _positive_int(size["height"], "height")
    except ValueError as exc:
        raise fail("image_size_px", str(exc)) from exc

    description = f"calibration file {path.name}: fx={values[0][0]:.1f} px, fy={values[1][1]:.1f} px"
    if "rms_reprojection_error_px" in data:
        try:
            rms = _finite(data["rms_reprojection_error_px"], "value")
        except ValueError as exc:
            raise fail("rms_reprojection_error_px", str(exc)) from exc
        if rms < 0:
            raise fail("rms_reprojection_error_px", "must not be negative")
        description += f", calibration RMS {rms:.3f} px"
    for key in ("method", "notes"):
        if key in data and not isinstance(data[key], str):
            raise fail(key, "must be text")
    if isinstance(data.get("method"), str) and data["method"].strip():
        description += f", method: {data['method'].strip()}"

    model = CameraModel(
        camera_matrix=tuple(tuple(row) for row in values),
        dist_coeffs=tuple(dist_values),
        width_px=width,
        height_px=height,
        source="calibration_file",
        description=description,
    )
    return model, digest


def check_resolution(camera: CameraModel, width_px: int, height_px: int) -> None:
    """Refuse frames whose size differs from the camera model (no silent rescaling)."""
    if (int(width_px), int(height_px)) != (camera.width_px, camera.height_px):
        raise ValueError(
            f"the frames are {int(width_px)}x{int(height_px)} pixels but the camera model is for "
            f"{camera.width_px}x{camera.height_px}; use a calibration made at this resolution "
            "or set the camera to the calibrated size"
        )


# ---------------------------------------------------------------- rotations


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_from_rvec(rvec: Iterable[float]) -> np.ndarray:
    """Rodrigues formula: R = I + sin(t) K + (1 - cos(t)) K^2, t = |rvec|."""
    r = np.asarray(rvec, dtype=np.float64).reshape(-1)
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise ValueError("rvec must be 3 finite numbers")
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        # First order is exact to rounding at this size.
        return np.eye(3) + _skew(r)
    k = _skew(r / theta)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Inverse of the Rodrigues formula, including angles at and near 180 degrees."""
    r = np.asarray(rotation, dtype=np.float64)
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])  # 2 sin(t) * axis
    cos_t = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_t)
    if theta < 1e-6:
        return vee / 2.0
    if math.pi - theta > 1e-3:
        return vee * (theta / (2.0 * math.sin(theta)))
    # Near 180 degrees sin(t) is tiny, so read the axis from the symmetric part:
    # (R + R^T) / 2 = cos(t) I + (1 - cos(t)) axis axis^T.
    outer = ((r + r.T) / 2.0 - cos_t * np.eye(3)) / (1.0 - cos_t)
    column = int(np.argmax(np.diag(outer)))
    axis = outer[:, column] / math.sqrt(max(outer[column, column], 1e-300))
    axis /= np.linalg.norm(axis)
    if float(axis @ vee) < 0:
        axis = -axis
    return axis * theta


def pose_points_camera(rvec: Iterable[float], tvec: Iterable[float], object_points: Any) -> np.ndarray:
    """Move marker-frame points into the camera frame: camera = R @ obj + t.

    R comes from the Rodrigues formula written out in NumPy, so this is an
    independent check on cv2.Rodrigues.
    """
    t = np.asarray(tvec, dtype=np.float64).reshape(-1)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        raise ValueError("tvec must be 3 finite numbers")
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    return points @ _rotation_from_rvec(rvec).T + t


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rvec_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float]:
    """Rotation (marker frame to camera frame) for a marker tilted from facing the camera.

    (0, 0, 0) is the marker facing the camera upright: marker x goes to camera
    x (right), marker y (up on the paper) goes to camera -y (up in the image),
    and marker z (out of the face) goes to camera -z (back toward the lens).
    That is a 180 degree rotation about the camera x axis, rvec (pi, 0, 0).

    The three angles turn the marker about its own axes, applied in this order
    to a point on the marker: roll first, then pitch, then yaw.

    * roll: about the marker z axis (its face normal). Positive turns the
      marker counterclockwise as the camera sees it.
    * pitch: about the marker x axis. Positive tips the top edge toward the
      camera.
    * yaw: about the marker y axis. Positive swings the face normal toward the
      marker's right, so the right edge moves away from the camera.

    In matrix form R = Rx(180) @ Ry(yaw) @ Rx(pitch) @ Rz(roll).
    """
    roll = _finite(roll_deg, "roll_deg")
    pitch = _finite(pitch_deg, "pitch_deg")
    yaw = _finite(yaw_deg, "yaw_deg")
    rotation = _rot_x(180.0) @ _rot_y(yaw) @ _rot_x(pitch) @ _rot_z(roll)
    rvec = _rvec_from_rotation(rotation)
    return (float(rvec[0]), float(rvec[1]), float(rvec[2]))


# ---------------------------------------------------------------- projection


def project_points(points_camera_m: Any, camera: CameraModel) -> np.ndarray:
    """Pinhole projection with the OpenCV k1, k2, p1, p2, k3 distortion model.

    Input is (N, 3) points in the camera frame, output is (N, 2) pixel
    coordinates in the pixel-center convention. Raises ValueError for a point
    at or behind the camera, or if any distortion term past k3 is nonzero
    (those are not modeled here).
    """
    points = np.asarray(points_camera_m, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if np.any(points[:, 2] <= 0):
        raise ValueError("every point must be in front of the camera (z > 0)")
    dist = camera.distortion()
    if np.any(dist[5:] != 0):
        raise ValueError("project_points models only k1, k2, p1, p2, k3; the higher distortion terms must be 0")
    k1, k2, p1, p2 = dist[:4]
    k3 = dist[4] if dist.size > 4 else 0.0
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    m = camera.matrix()
    return np.column_stack([m[0, 0] * xd + m[0, 2], m[1, 1] * yd + m[1, 2]])


def marker_center_px(corners_px: Any) -> tuple[float, float]:
    """Image of the marker center: where the two diagonals cross.

    A perspective view keeps straight lines straight, so the center of the
    square lands exactly on both diagonals. The plain mean of the four corners
    is pulled toward the nearer (bigger) side of a tilted marker.
    """
    c = _corners(corners_px)
    ones = np.ones((4, 1))
    h = np.hstack([c, ones])
    diagonal_a = np.cross(h[0], h[2])  # top-left to bottom-right
    diagonal_b = np.cross(h[1], h[3])  # top-right to bottom-left
    point = np.cross(diagonal_a, diagonal_b)
    scale = float(np.max(np.abs(c))) + 1.0
    if abs(point[2]) < 1e-12 * scale * scale:
        raise ValueError("the marker diagonals are parallel; the corners are degenerate")
    return (float(point[0] / point[2]), float(point[1] / point[2]))


def bbox_from_corners(corners_px: Any, width_px: int, height_px: int) -> dict[str, float] | None:
    """Axis-aligned box around the corners, clipped to the image.

    The box uses the usual edge convention for boxes: the image runs from
    (0, 0) at the outer top-left corner of the first pixel to (width, height).
    That is the pixel-center convention of the corners shifted by half a pixel,
    so x and y are never negative. Returns None if nothing is left after clipping.
    """
    c = _corners(corners_px) + 0.5
    width = _positive_int(width_px, "width_px")
    height = _positive_int(height_px, "height_px")
    x0 = min(max(float(c[:, 0].min()), 0.0), float(width))
    x1 = min(max(float(c[:, 0].max()), 0.0), float(width))
    y0 = min(max(float(c[:, 1].min()), 0.0), float(height))
    y1 = min(max(float(c[:, 1].max()), 0.0), float(height))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


# ---------------------------------------------------------------- pose


@dataclass(frozen=True)
class MarkerPose:
    """One marker pose from the four detected corners.

    rvec turns marker-frame vectors into the camera frame (Rodrigues form) and
    tvec is the marker center in the camera frame, in meters.
    """

    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]
    reprojection_error_px: float
    ambiguity_ratio: float | None
    # Every pose the solvers returned, kept or not, so a flip between the two
    # IPPE tilts can be spotted later. Each is a plain dict: {"method":
    # "IPPE_SQUARE" or "SQPNP", "rvec": [3], "tvec": [3], "reprojection_error_px"}.
    candidates: tuple[dict[str, Any], ...] = ()


def _rms_corner_error(cv2: Any, object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera: CameraModel) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera.matrix(), camera.distortion())
    distances_sq = np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)
    return float(math.sqrt(float(np.mean(distances_sq))))


def _solve_candidates(cv2: Any, flag: int, object_points: np.ndarray, image_points: np.ndarray, camera: CameraModel) -> list[tuple[np.ndarray, np.ndarray]]:
    """All finite solutions from one solver, including any behind the camera."""
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points.reshape(4, 1, 2),
            camera.matrix(),
            camera.distortion(),
            flags=flag,
        )
    except cv2.error:
        return []
    result = []
    for index in range(int(count)):
        rvec = np.asarray(rvecs[index], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[index], dtype=np.float64).reshape(3)
        if np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
            result.append((rvec, tvec))
    return result


def _rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    relative = _rotation_from_rvec(rvec_a).T @ _rotation_from_rvec(rvec_b)
    return math.degrees(math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def estimate_marker_pose(cv2: Any, corners_px: Any, camera: CameraModel, marker_length_m: float) -> MarkerPose | None:
    """Solve the marker pose from its 4 corners with cv2.solvePnPGeneric.

    SOLVEPNP_IPPE_SQUARE returns up to two poses that explain a square almost
    equally well (the two ways a flat square can tilt). In OpenCV 5.0 it also
    sometimes returns a clearly wrong pair when the marker faces the camera
    almost exactly (errors from 0.4 px up to the size of the marker, measured
    on exact synthetic corners), so the SOLVEPNP_SQPNP solution is added as a
    third candidate. The pose kept is the candidate with the lowest
    reprojection error whose center is in front of the camera (z > 0).

    reprojection_error_px is the RMS distance over the 4 corners, computed
    here with cv2.projectPoints. ambiguity_ratio is the kept error divided by
    the error of the best candidate whose rotation differs from it by more
    than 1 degree (usually the other IPPE tilt). Near 1 means the two tilts
    are hard to tell apart; None means no such second candidate was found.
    candidates lists every finite solution in the order the solvers gave them
    (IPPE_SQUARE first, then SQPNP), with its own reprojection error, whether
    or not it was kept. Returns None for corners that enclose almost no area
    or when no solver gives a valid pose.
    """
    image_points = _corners(corners_px)
    object_points = marker_object_points(marker_length_m)
    x, y = image_points[:, 0], image_points[:, 1]
    if abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.0 < 1.0:
        return None
    solved = [("IPPE_SQUARE", pose) for pose in _solve_candidates(cv2, cv2.SOLVEPNP_IPPE_SQUARE, object_points, image_points, camera)]
    solved += [("SQPNP", pose) for pose in _solve_candidates(cv2, cv2.SOLVEPNP_SQPNP, object_points, image_points, camera)]
    candidates = []
    poses = []
    for method, (rvec, tvec) in solved:
        error = _rms_corner_error(cv2, object_points, image_points, rvec, tvec, camera)
        if not math.isfinite(error):
            continue
        candidates.append({
            "method": method,
            "rvec": [float(v) for v in rvec],
            "tvec": [float(v) for v in tvec],
            "reprojection_error_px": error,
        })
        if tvec[2] > 0:
            poses.append((error, rvec, tvec))
    if not poses:
        return None
    # Same choice as before the candidate list existed: lowest error, in front of the camera.
    scored = sorted(poses, key=lambda item: item[0])
    best_error, best_rvec, best_tvec = scored[0]
    others = [error for error, rvec, _ in scored[1:] if _rotation_angle_deg(best_rvec, rvec) > 1.0]
    ratio: float | None = None
    if others:
        ratio = 1.0 if others[0] <= 0 else float(best_error / others[0])
    return MarkerPose(
        rvec=(float(best_rvec[0]), float(best_rvec[1]), float(best_rvec[2])),
        tvec=(float(best_tvec[0]), float(best_tvec[1]), float(best_tvec[2])),
        reprojection_error_px=float(best_error),
        ambiguity_ratio=ratio,
        candidates=tuple(candidates),
    )


# ---------------------------------------------------------------- rendering


def render_marker_frame(
    cv2: Any,
    marker_image: np.ndarray,
    camera: CameraModel,
    rvec: Iterable[float],
    tvec: Iterable[float],
    marker_length_m: float,
    *,
    quiet_zone: float = 0.25,
    background: int = 128,
    pixel_noise_std: float = 0.0,
    blur_sigma_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a synthetic camera frame of a printed marker. Returns BGR uint8.

    ``marker_image`` is the square gray image from cv2.aruco.generateImageMarker;
    its full extent is the black-bordered square of side marker_length_m. A
    white paper margin of quiet_zone * marker_length_m is drawn around it on a
    flat gray background.

    How it works: the four marker corners are projected with project_points
    (so lens distortion moves the corners), and one homography from the marker
    image to those corners is applied with cv2.warpPerspective (INTER_LINEAR)
    into an image RENDER_SUPERSAMPLE times finer than the output, which is then
    averaged down with cv2.resize(INTER_AREA). Only the area around the paper
    is drawn finely, to keep it fast. A single homography keeps edges straight,
    so with nonzero distortion the corners are right but the slight bending of
    the edges between them is not drawn.

    Pixel convention: texture pixel k covers -0.5 + k .. 0.5 + k, and a fine
    pixel j covers output coordinate (j + 0.5) / s - 0.5 at its center, which
    matches the OpenCV center convention, so corners found by the detector line
    up with project_points.

    After that: optional Gaussian blur (blur_sigma_px, output pixels), then
    additive Gaussian noise (pixel_noise_std, gray levels), clipped to 0..255.
    If noise is asked for and rng is None, a fresh unseeded generator is used.
    """
    marker = np.asarray(marker_image)
    if marker.ndim != 2 or marker.shape[0] != marker.shape[1] or marker.shape[0] < 2:
        raise ValueError("marker_image must be a square 2D gray image")
    if marker.dtype != np.uint8:
        raise ValueError("marker_image must be uint8")
    length = _finite(marker_length_m, "marker_length_m")
    if length <= 0:
        raise ValueError("marker_length_m must be greater than 0")
    quiet = _finite(quiet_zone, "quiet_zone")
    if quiet < 0:
        raise ValueError("quiet_zone must not be negative")
    if isinstance(background, bool) or not 0 <= int(background) <= 255:
        raise ValueError("background must be a gray level from 0 to 255")
    noise_std = _finite(pixel_noise_std, "pixel_noise_std")
    blur = _finite(blur_sigma_px, "blur_sigma_px")
    if noise_std < 0 or blur < 0:
        raise ValueError("pixel_noise_std and blur_sigma_px must not be negative")

    width, height = camera.width_px, camera.height_px
    s = RENDER_SUPERSAMPLE

    # Paper texture: the marker with a white margin, scaled up so that one
    # texture pixel is no bigger than about one fine output pixel.
    object_points = marker_object_points(length)
    corners_cam = pose_points_camera(rvec, tvec, object_points)
    paper_cam = pose_points_camera(rvec, tvec, object_points * (1.0 + 2.0 * quiet))
    corners_px = project_points(corners_cam, camera)
    paper_px = project_points(paper_cam, camera)
    side_px = float(np.max(np.linalg.norm(np.roll(corners_px, -1, axis=0) - corners_px, axis=1)))
    cell = marker.shape[0]
    scale = max(1, math.ceil(side_px * s / cell))
    scale = min(scale, max(1, 4096 // cell))
    texture_marker = cv2.resize(marker, (cell * scale, cell * scale), interpolation=cv2.INTER_NEAREST)
    margin = round(quiet * texture_marker.shape[0])
    texture = cv2.copyMakeBorder(texture_marker, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    side = texture_marker.shape[0]
    texture_corners = np.array(
        [[margin, margin], [margin + side, margin], [margin + side, margin + side], [margin, margin + side]],
        dtype=np.float64,
    ) - 0.5

    frame = np.full((height, width), int(background), dtype=np.uint8)
    # Output region that the paper can touch, in whole output pixels.
    lo = np.floor(paper_px.min(axis=0) - 2.0).astype(int)
    hi = np.ceil(paper_px.max(axis=0) + 2.0).astype(int)
    x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
    x1, y1 = min(int(hi[0]) + 1, width), min(int(hi[1]) + 1, height)
    if x1 > x0 and y1 > y0:
        # Fine pixel j (relative to the region) has its center at output
        # coordinate x0 + (j + 0.5) / s - 0.5, so output u maps to fine
        # j = (u - x0 + 0.5) * s - 0.5.
        fine_corners = (corners_px - np.array([x0, y0], dtype=np.float64) + 0.5) * s - 0.5
        homography = cv2.getPerspectiveTransform(texture_corners.astype(np.float32), fine_corners.astype(np.float32))
        fine = cv2.warpPerspective(
            texture,
            homography,
            ((x1 - x0) * s, (y1 - y0) * s),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(background),
        )
        frame[y0:y1, x0:x1] = cv2.resize(fine, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)

    image = frame.astype(np.float64)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur, sigmaY=blur)
    if noise_std > 0:
        generator = rng if rng is not None else np.random.default_rng()
        image = image + generator.normal(0.0, noise_std, size=image.shape)
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- timing


def summarize_frame_timing(capture_times_s: Iterable[float], nominal_frame_rate_hz: float | None) -> dict[str, Any]:
    """Frame timing summary, safe to write as JSON.

    measured_frame_rate_hz is 1 / median interval. late_frames counts intervals
    longer than 1.5 times the median (a dropped or slow frame), non_increasing
    counts intervals of zero or less (a clock or ordering problem). With fewer
    than 2 frames the interval numbers are None and the counts are 0.
    """
    times = np.asarray(list(capture_times_s), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("capture times must be finite")
    nominal: float | None = None
    if nominal_frame_rate_hz is not None:
        nominal = _finite(nominal_frame_rate_hz, "nominal_frame_rate_hz")
        if nominal <= 0:
            raise ValueError("nominal_frame_rate_hz must be greater than 0")
    n = int(times.size)
    summary: dict[str, Any] = {
        "frames": n,
        "duration_s": float(times[-1] - times[0]) if n >= 1 else None,
        "median_interval_s": None,
        "measured_frame_rate_hz": None,
        "max_interval_s": None,
        "late_frames": 0,
        "non_increasing": 0,
        "nominal_frame_rate_hz": nominal,
    }
    if n < 2:
        return summary
    intervals = np.diff(times)
    median = float(np.median(intervals))
    summary["median_interval_s"] = median
    summary["max_interval_s"] = float(intervals.max())
    summary["non_increasing"] = int(np.count_nonzero(intervals <= 0))
    if median > 0:
        summary["measured_frame_rate_hz"] = 1.0 / median
        summary["late_frames"] = int(np.count_nonzero(intervals > 1.5 * median))
    return summary


def replay_capture_times(n_frames: int, frame_rate_hz: float, start_s: float = 0.0) -> list[float]:
    """Evenly spaced capture times start_s + k / frame_rate_hz for k = 0 .. n - 1.

    Each time is computed directly from k, not by adding up intervals, so
    rounding does not build up over a long video.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)) or int(n_frames) < 0:
        raise ValueError("n_frames must be a whole number, 0 or more")
    rate = _finite(frame_rate_hz, "frame_rate_hz")
    if rate <= 0:
        raise ValueError("frame_rate_hz must be greater than 0")
    start = _finite(start_s, "start_s")
    return [start + k / rate for k in range(int(n_frames))]

