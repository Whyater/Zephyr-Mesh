"""Pure orbit-camera, projection, and drone glyph geometry.

World axes: x east, y north, z up, in meters. The camera orbits a target
point; ``yaw`` is the compass angle from the target to the camera (0 means
the camera sits east of the target looking west) and ``pitch`` is the angle
above the floor. The screen keeps a right-handed view, so east appears to the
right of north when looking down.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, NamedTuple, Sequence, Any


Vec3 = tuple[float, float, float]
PITCH_MIN = 10.0
PITCH_MAX = 85.0
DEFAULT_NEAR_LIMIT = 0.5
DEFAULT_FOV = 50.0
NEAR_PLANE = 0.1
TOP_YAW = -90.0
TOP_PITCH = 90.0
DEFAULT_YAW = -62.0
DEFAULT_PITCH = 32.0


def _vec(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("a world point must have three coordinates")
    return float(value[0]), float(value[1]), float(value[2])


def clamp_pitch(pitch: float) -> float:
    return min(PITCH_MAX, max(PITCH_MIN, float(pitch)))


def clamp_distance(distance: float, swarm_radius: float, near_limit: float = DEFAULT_NEAR_LIMIT) -> float:
    near = max(float(near_limit), 1e-6)
    radius = max(float(swarm_radius), 0.0)
    return min(max(float(distance), near), max(near, radius * 3.0))


@dataclass(frozen=True)
class OrbitCamera:
    yaw: float = 0.0
    pitch: float = 35.0
    distance: float = 10.0
    target: Vec3 = (0.0, 0.0, 0.0)
    near_limit: float = DEFAULT_NEAR_LIMIT
    swarm_radius: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pitch", clamp_pitch(self.pitch))
        object.__setattr__(self, "distance", clamp_distance(self.distance, self.swarm_radius, self.near_limit))
        object.__setattr__(self, "target", _vec(self.target))

    def drag_delta(self, dx: float, dy: float, sensitivity: float = 0.35) -> "OrbitCamera":
        return replace(self, yaw=self.yaw - float(dx) * sensitivity, pitch=clamp_pitch(self.pitch + float(dy) * sensitivity))

    def zoom(self, factor: float) -> "OrbitCamera":
        numeric = float(factor)
        if numeric <= 0:
            raise ValueError("zoom factor must be positive")
        return replace(self, distance=clamp_distance(self.distance * numeric, self.swarm_radius, self.near_limit))

    def focus(self, point: Sequence[float]) -> "OrbitCamera":
        return replace(self, target=_vec(point))

    def top_view(self) -> "OrbitCamera":
        return replace(self, yaw=0.0, pitch=PITCH_MAX)


def drag_delta(camera: OrbitCamera, dx: float, dy: float, sensitivity: float = 0.35) -> OrbitCamera:
    return camera.drag_delta(dx, dy, sensitivity)


def zoom(camera: OrbitCamera, factor: float) -> OrbitCamera:
    return camera.zoom(factor)


def focus_point(camera: OrbitCamera, point: Sequence[float]) -> OrbitCamera:
    return camera.focus(point)


def top_view(camera: OrbitCamera) -> OrbitCamera:
    return camera.top_view()


def swarm_radius(points: Iterable[Sequence[float]], target: Sequence[float] | None = None) -> float:
    items = [_vec(point) for point in points]
    if not items:
        return 0.0
    center = _vec(target) if target is not None else tuple(sum(point[i] for point in items) / len(items) for i in range(3))
    return max(math.dist(point, center) for point in items)


def fit_all(points: Iterable[Sequence[float]], near_limit: float = DEFAULT_NEAR_LIMIT) -> OrbitCamera:
    items = [_vec(point) for point in points]
    if not items:
        return OrbitCamera(distance=near_limit, near_limit=near_limit, swarm_radius=0.0)
    center = tuple((min(point[i] for point in items) + max(point[i] for point in items)) / 2.0 for i in range(3))
    radius = max(swarm_radius(items, center), near_limit / 3.0)
    distance = clamp_distance(max(float(near_limit), radius * 2.2), radius, near_limit)
    return OrbitCamera(yaw=0.0, pitch=35.0, distance=distance, target=center, near_limit=near_limit, swarm_radius=radius)


def framing_camera(points: Iterable[Sequence[float]], near_limit: float = DEFAULT_NEAR_LIMIT) -> OrbitCamera:
    """Reset framing: fit every drone, seen from the south-east at a calm angle."""
    fitted = fit_all(points, near_limit)
    return replace(fitted, yaw=DEFAULT_YAW, pitch=DEFAULT_PITCH)


def fit_distance(
    camera: OrbitCamera,
    points: Iterable[Sequence[float]],
    width: float,
    height: float,
    margin: float = 24.0,
    fov_degrees: float = DEFAULT_FOV,
) -> float:
    """Smallest allowed distance at which every point projects inside the margins."""
    items = [_vec(point) for point in points]
    low = max(camera.near_limit, 1e-3)
    high = max(low, camera.swarm_radius * 3.0)
    if not items or width <= 2 * margin or height <= 2 * margin:
        return high

    def fits(distance: float) -> bool:
        projector = Projector(ViewPose(camera.yaw, camera.pitch, distance, camera.target), width, height, fov_degrees)
        for point in items:
            projected = projector.project(point)
            if projected.depth <= NEAR_PLANE:
                return False
            if not (margin <= projected.x <= width - margin and margin <= projected.y <= height - margin):
                return False
        return True

    if not fits(high):
        return high
    for _ in range(30):
        middle = (low + high) / 2.0
        if fits(middle):
            high = middle
        else:
            low = middle
    return high


def ease_out(start: float | Sequence[float], end: float | Sequence[float], progress: float, reduced_motion: bool = False) -> Any:
    """Interpolate with a quadratic ease-out, or jump when motion is reduced."""
    if reduced_motion:
        return end
    t = min(1.0, max(0.0, float(progress)))
    amount = 1.0 - (1.0 - t) ** 2
    if isinstance(start, Sequence) and not isinstance(start, (str, bytes)):
        return tuple(ease_out(a, b, t, False) for a, b in zip(start, end))
    return float(start) + (float(end) - float(start)) * amount


class ProjectedPoint(NamedTuple):
    x: float
    y: float
    depth: float


@dataclass(frozen=True)
class ViewPose:
    """What the renderer draws. Unlike ``OrbitCamera`` it allows the 90 degree top view."""

    yaw: float
    pitch: float
    distance: float
    target: Vec3

    @classmethod
    def of(cls, camera: Any) -> "ViewPose":
        return cls(float(camera.yaw), float(camera.pitch), float(camera.distance), _vec(camera.target))

    def toward(self, other: "ViewPose", progress: float, reduced_motion: bool = False) -> "ViewPose":
        """Eased step toward ``other``; yaw takes the short way around."""
        if reduced_motion:
            return other
        yaw_delta = (other.yaw - self.yaw + 180.0) % 360.0 - 180.0
        return ViewPose(
            ease_out(self.yaw, self.yaw + yaw_delta, progress),
            ease_out(self.pitch, other.pitch, progress),
            ease_out(self.distance, other.distance, progress),
            ease_out(self.target, other.target, progress),
        )

    def close_to(self, other: "ViewPose", tolerance: float = 1e-3) -> bool:
        yaw_delta = abs((other.yaw - self.yaw + 180.0) % 360.0 - 180.0)
        return (
            yaw_delta < tolerance
            and abs(other.pitch - self.pitch) < tolerance
            and abs(other.distance - self.distance) < tolerance
            and math.dist(other.target, self.target) < tolerance
        )


def top_pose(camera: Any) -> ViewPose:
    """Straight-down view with north up and east to the right."""
    return ViewPose(TOP_YAW, TOP_PITCH, float(camera.distance), _vec(camera.target))


def _basis(camera: Any) -> tuple[Vec3, Vec3, Vec3, Vec3]:
    pitch = math.radians(camera.pitch)
    yaw = math.radians(camera.yaw)
    target = camera.target
    position = (
        target[0] + camera.distance * math.cos(pitch) * math.cos(yaw),
        target[1] + camera.distance * math.cos(pitch) * math.sin(yaw),
        target[2] + camera.distance * math.sin(pitch),
    )
    forward = tuple((target[i] - position[i]) / camera.distance for i in range(3))
    right = (-math.sin(yaw), math.cos(yaw), 0.0)
    up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return position, forward, right, up


class Projector:
    """Perspective projection for one pose and canvas size, with the basis computed once."""

    def __init__(self, camera: Any, width: float, height: float, fov_degrees: float = DEFAULT_FOV) -> None:
        self.width = float(width)
        self.height = float(height)
        self.position, self.forward, self.right, self.up = _basis(camera)
        self.focal = (self.width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
        self.pitch = float(camera.pitch)
        self.ellipse_ratio = max(0.04, abs(math.sin(math.radians(self.pitch))))

    def camera_space(self, point: Sequence[float]) -> Vec3:
        px, py, pz = point[0] - self.position[0], point[1] - self.position[1], point[2] - self.position[2]
        f, r, u = self.forward, self.right, self.up
        return (
            px * r[0] + py * r[1] + pz * r[2],
            px * u[0] + py * u[1] + pz * u[2],
            px * f[0] + py * f[1] + pz * f[2],
        )

    def _screen(self, horizontal: float, vertical: float, depth: float) -> ProjectedPoint:
        safe = max(depth, 1e-6)
        return ProjectedPoint(
            self.width / 2.0 + horizontal * self.focal / safe,
            self.height / 2.0 - vertical * self.focal / safe,
            depth,
        )

    def project(self, point: Sequence[float]) -> ProjectedPoint:
        return self._screen(*self.camera_space(point))

    def scale_at(self, depth: float) -> float:
        """Screen pixels per world meter at a depth."""
        return self.focal / max(depth, 1e-6)

    def segment(self, a: Sequence[float], b: Sequence[float], near: float = NEAR_PLANE) -> tuple[ProjectedPoint, ProjectedPoint] | None:
        """Project a segment, clipped against the near plane; ``None`` when fully behind."""
        ha, va, da = self.camera_space(a)
        hb, vb, db = self.camera_space(b)
        if da < near and db < near:
            return None
        if da < near or db < near:
            t = (near - da) / (db - da)
            clipped = (ha + (hb - ha) * t, va + (vb - va) * t, near)
            if da < near:
                ha, va, da = clipped
            else:
                hb, vb, db = clipped
        return self._screen(ha, va, da), self._screen(hb, vb, db)


def project_point(camera: Any, point: Sequence[float], width: float, height: float, fov_degrees: float = 60.0) -> ProjectedPoint:
    """Perspective-project a world point into canvas coordinates."""
    return Projector(camera, width, height, fov_degrees).project(_vec(point))


def project_points(
    camera: Any,
    points: Iterable[Sequence[float]],
    width: float,
    height: float,
    fov_degrees: float = 60.0,
) -> tuple[ProjectedPoint, ...]:
    projector = Projector(camera, width, height, fov_degrees)
    return tuple(projector.project(_vec(point)) for point in points)


def painter_order(items: Iterable[Any]) -> list[Any]:
    """Return items from farthest to nearest using their depth value."""
    def depth(item: Any) -> float:
        if hasattr(item, "depth"):
            return float(item.depth)
        if isinstance(item, dict):
            return float(item["depth"])
        return float(item[1])
    return sorted(items, key=depth)


def far_to_near(depths: Sequence[float]) -> list[int]:
    """Indices ordered for painter's drawing: largest depth first."""
    return sorted(range(len(depths)), key=lambda index: -float(depths[index]))


@dataclass(frozen=True)
class Primitive:
    kind: str
    points: tuple[ProjectedPoint, ...]
    depth: float
    radius: float = 0.0


def quadcopter_glyph(
    camera: OrbitCamera,
    position: Sequence[float],
    heading: float,
    size_m: float,
    width: float,
    height: float,
) -> tuple[Primitive, ...]:
    """Return body, four arms, four rotor discs, and a heading marker."""
    x, y, z = _vec(position)
    size = max(float(size_m), 0.0)
    angle = float(heading)
    projector = Projector(camera, width, height, 60.0)

    def world(dx: float, dy: float) -> Vec3:
        return (x + dx * math.cos(angle) - dy * math.sin(angle), y + dx * math.sin(angle) + dy * math.cos(angle), z)
    center = projector.project((x, y, z))
    body_world = (world(size * 0.35, 0), world(0, size * 0.2), world(-size * 0.35, 0), world(0, -size * 0.2))
    result = [Primitive("body", tuple(projector.project(point) for point in body_world), center.depth)]
    arm_points = ((size, 0), (0, size), (-size, 0), (0, -size))
    for dx, dy in arm_points:
        result.append(Primitive("arm", (center, projector.project(world(dx, dy))), center.depth))
    for dx, dy in arm_points:
        rotor = projector.project(world(dx, dy))
        rotor_edge = projector.project(world(dx + size * 0.14, dy))
        result.append(Primitive("rotor", (rotor,), rotor.depth, abs(rotor_edge.x - rotor.x)))
    marker = projector.project(world(size * 0.7, 0))
    result.append(Primitive("heading", (center, marker), center.depth))
    return tuple(result)


@dataclass(frozen=True)
class DroneGlyph:
    """Screen geometry for one X-frame quadcopter.

    ``rotors`` are axis-aligned ellipses ``(cx, cy, rx, ry)``: with no camera
    roll, a horizontal circle projects to an ellipse whose axes follow the
    screen axes, so Tk ovals draw them exactly.
    """

    center: ProjectedPoint
    arm_tips: tuple[tuple[float, float], ...]
    rotors: tuple[tuple[float, float, float, float], ...]
    body: tuple[float, float, float, float]
    motor_radius: float
    heading: tuple[tuple[float, float], ...] | None


def xframe_glyph(
    projector: Projector,
    position: Sequence[float],
    heading: float | None,
    arm_m: float = 0.17,
    rotor_m: float = 0.085,
) -> DroneGlyph | None:
    """X-frame quadcopter at ``position``; ``heading`` in radians or ``None`` at rest.

    Returns ``None`` when the drone is behind the near plane. With no heading
    the frame keeps a fixed orientation and draws no heading marker, because
    the data does not say which way a resting drone faces.
    """
    x, y, z = position[0], position[1], position[2]
    center = projector.project((x, y, z))
    if center.depth < NEAR_PLANE * 2:
        return None
    yaw = float(heading) if heading is not None else 0.0
    tips = []
    rotors = []
    scale = projector.scale_at(center.depth)
    rotor_rx = rotor_m * scale
    rotor_ry = rotor_rx * projector.ellipse_ratio
    for quarter in (45.0, 135.0, 225.0, 315.0):
        angle = yaw + math.radians(quarter)
        tip = projector.project((x + arm_m * math.cos(angle), y + arm_m * math.sin(angle), z))
        tips.append((tip.x, tip.y))
        rotors.append((tip.x, tip.y, rotor_rx, rotor_ry))
    body_rx = arm_m * 0.34 * scale
    body_ry = max(body_rx * projector.ellipse_ratio, body_rx * 0.45)
    marker = None
    if heading is not None:
        nose = projector.project((x + arm_m * 1.05 * math.cos(yaw), y + arm_m * 1.05 * math.sin(yaw), z))
        left = projector.project((x + arm_m * 0.55 * math.cos(yaw + 0.42), y + arm_m * 0.55 * math.sin(yaw + 0.42), z))
        right = projector.project((x + arm_m * 0.55 * math.cos(yaw - 0.42), y + arm_m * 0.55 * math.sin(yaw - 0.42), z))
        marker = ((nose.x, nose.y), (left.x, left.y), (right.x, right.y))
    return DroneGlyph(
        center=center,
        arm_tips=tuple(tips),
        rotors=tuple(rotors),
        body=(center.x - body_rx, center.y - body_ry, center.x + body_rx, center.y + body_ry),
        motor_radius=max(1.5, rotor_m * 0.3 * scale),
        heading=marker,
    )


def grid_lines(center: Sequence[float], half_extent: float, spacing: float = 1.0, major_every: int = 5) -> list[tuple[Vec3, Vec3, bool]]:
    """Floor grid at z = 0 aligned to whole meters; ``True`` marks every 5 m line."""
    spacing = max(float(spacing), 1e-6)
    cx, cy = float(center[0]), float(center[1])
    low_x = math.floor((cx - half_extent) / spacing)
    high_x = math.ceil((cx + half_extent) / spacing)
    low_y = math.floor((cy - half_extent) / spacing)
    high_y = math.ceil((cy + half_extent) / spacing)
    lines = []
    for index in range(low_x, high_x + 1):
        x = index * spacing
        lines.append(((x, low_y * spacing, 0.0), (x, high_y * spacing, 0.0), index % major_every == 0))
    for index in range(low_y, high_y + 1):
        y = index * spacing
        lines.append(((low_x * spacing, y, 0.0), (high_x * spacing, y, 0.0), index % major_every == 0))
    return lines


def circle_points(center: Sequence[float], radius: float, segments: int = 32) -> list[Vec3]:
    """Horizontal circle in world space."""
    cx, cy, cz = _vec(center)
    return [
        (cx + radius * math.cos(2 * math.pi * index / segments), cy + radius * math.sin(2 * math.pi * index / segments), cz)
        for index in range(segments)
    ]


def hit_test(points: Sequence[tuple[float, float] | None], x: float, y: float, radius: float) -> int | None:
    """Index of the nearest screen point within ``radius`` pixels, else ``None``."""
    best = None
    best_distance = float(radius)
    for index, point in enumerate(points):
        if point is None:
            continue
        distance = math.hypot(point[0] - x, point[1] - y)
        if distance <= best_distance:
            best = index
            best_distance = distance
    return best


class CameraAnimator:
    """Time-based ease-out between view poses; the Tk layer only calls ``advance``."""

    def __init__(self, pose: ViewPose) -> None:
        self.pose = pose
        self._start = pose
        self._goal = pose
        self._began_ms = 0.0
        self._duration_ms = 0.0

    @property
    def goal(self) -> ViewPose:
        return self._goal

    @property
    def active(self) -> bool:
        return not self.pose.close_to(self._goal, 1e-4)

    def go(self, goal: ViewPose, now_ms: float, duration_ms: float, reduced_motion: bool = False) -> None:
        self._start = self.pose
        self._goal = goal
        self._began_ms = float(now_ms)
        self._duration_ms = 0.0 if reduced_motion else max(0.0, float(duration_ms))
        if self._duration_ms == 0.0:
            self.pose = goal

    def advance(self, now_ms: float) -> ViewPose:
        if self._duration_ms <= 0.0:
            self.pose = self._goal
            return self.pose
        progress = (float(now_ms) - self._began_ms) / self._duration_ms
        self.pose = self._goal if progress >= 1.0 else self._start.toward(self._goal, progress)
        return self.pose


__all__ = [
    "CameraAnimator", "DEFAULT_FOV", "DEFAULT_NEAR_LIMIT", "DroneGlyph", "NEAR_PLANE", "OrbitCamera",
    "PITCH_MAX", "PITCH_MIN", "Primitive", "ProjectedPoint", "Projector", "ViewPose", "circle_points",
    "clamp_distance", "clamp_pitch", "drag_delta", "ease_out", "far_to_near", "fit_all", "focus_point",
    "fit_distance", "framing_camera", "grid_lines", "hit_test", "painter_order", "project_point", "project_points",
    "quadcopter_glyph", "swarm_radius", "top_pose", "top_view", "xframe_glyph", "zoom",
]
