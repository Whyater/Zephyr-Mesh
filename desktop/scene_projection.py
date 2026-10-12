"""Pure orbit-camera, projection, and drone glyph geometry."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, NamedTuple, Sequence, Any


Vec3 = tuple[float, float, float]
PITCH_MIN = 10.0
PITCH_MAX = 85.0
DEFAULT_NEAR_LIMIT = 0.5


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
        return replace(self, yaw=self.yaw + float(dx) * sensitivity, pitch=clamp_pitch(self.pitch + float(dy) * sensitivity))

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


def _basis(camera: OrbitCamera) -> tuple[Vec3, Vec3, Vec3, Vec3]:
    pitch = math.radians(camera.pitch)
    yaw = math.radians(camera.yaw)
    position = (
        camera.target[0] + camera.distance * math.cos(pitch) * math.cos(yaw),
        camera.target[1] + camera.distance * math.cos(pitch) * math.sin(yaw),
        camera.target[2] + camera.distance * math.sin(pitch),
    )
    forward = tuple((camera.target[i] - position[i]) / camera.distance for i in range(3))
    right = (math.sin(yaw), -math.cos(yaw), 0.0)
    up = (
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    )
    return position, forward, right, up


def project_point(camera: OrbitCamera, point: Sequence[float], width: float, height: float, fov_degrees: float = 60.0) -> ProjectedPoint:
    """Perspective-project a world point into canvas coordinates."""
    world = _vec(point)
    position, forward, right, up = _basis(camera)
    relative = tuple(world[i] - position[i] for i in range(3))
    depth = sum(relative[i] * forward[i] for i in range(3))
    horizontal = sum(relative[i] * right[i] for i in range(3))
    vertical = sum(relative[i] * up[i] for i in range(3))
    safe_depth = max(depth, 1e-6)
    scale = (float(width) / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return ProjectedPoint(float(width) / 2.0 + horizontal * scale / safe_depth, float(height) / 2.0 - vertical * scale / safe_depth, depth)


def project_points(
    camera: OrbitCamera,
    points: Iterable[Sequence[float]],
    width: float,
    height: float,
    fov_degrees: float = 60.0,
) -> tuple[ProjectedPoint, ...]:
    return tuple(project_point(camera, point, width, height, fov_degrees) for point in points)


def painter_order(items: Iterable[Any]) -> list[Any]:
    """Return items from farthest to nearest using their depth value."""
    def depth(item: Any) -> float:
        if hasattr(item, "depth"):
            return float(item.depth)
        if isinstance(item, dict):
            return float(item["depth"])
        return float(item[1])
    return sorted(items, key=depth)


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
    def world(dx: float, dy: float) -> Vec3:
        return (x + dx * math.cos(angle) - dy * math.sin(angle), y + dx * math.sin(angle) + dy * math.cos(angle), z)
    center = project_point(camera, (x, y, z), width, height)
    body_world = (world(size * 0.35, 0), world(0, size * 0.2), world(-size * 0.35, 0), world(0, -size * 0.2))
    result = [Primitive("body", tuple(project_point(camera, point, width, height) for point in body_world), center.depth)]
    arm_points = ((size, 0), (0, size), (-size, 0), (0, -size))
    for dx, dy in arm_points:
        result.append(Primitive("arm", (center, project_point(camera, world(dx, dy), width, height)), center.depth))
    for dx, dy in arm_points:
        rotor = project_point(camera, world(dx, dy), width, height)
        rotor_edge = project_point(camera, world(dx + size * 0.14, dy), width, height)
        result.append(Primitive("rotor", (rotor,), rotor.depth, abs(rotor_edge.x - rotor.x)))
    marker = project_point(camera, world(size * 0.7, 0), width, height)
    result.append(Primitive("heading", (center, marker), center.depth))
    return tuple(result)


__all__ = [
    "DEFAULT_NEAR_LIMIT", "OrbitCamera", "PITCH_MAX", "PITCH_MIN", "Primitive", "ProjectedPoint",
    "clamp_distance", "clamp_pitch", "drag_delta", "ease_out", "fit_all", "focus_point",
    "painter_order", "project_point", "project_points", "quadcopter_glyph", "swarm_radius",
    "top_view", "zoom",
]
