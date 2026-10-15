"""Simulated reciprocal velocity avoidance for cooperative observers."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


_EPS = 1.0e-12


def _vec(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return out.copy()


@dataclass(frozen=True)
class AvoidanceConfig:
    """Parameters for the simulated ORCA-lite controller."""

    radius_m: float = 0.35
    time_horizon_s: float = 2.0
    max_speed_mps: float = 2.0
    responsibility: float = 0.5
    emergency_distance_m: float | None = None
    sidestep_right: bool = True
    ellipsoid_radii: tuple[float, float, float] | None = None
    safety_margin_m: float = 0.0

    def __post_init__(self) -> None:
        for name in ("radius_m", "time_horizon_s", "max_speed_mps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        margin = float(self.safety_margin_m)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("safety_margin_m must be finite and non-negative")
        responsibility = float(self.responsibility)
        if not np.isfinite(responsibility) or not 0.0 <= responsibility <= 1.0:
            raise ValueError("responsibility must be between zero and one")
        if self.emergency_distance_m is not None:
            value = float(self.emergency_distance_m)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("emergency_distance_m must be finite and non-negative")
        if not isinstance(self.sidestep_right, (bool, np.bool_)):
            raise ValueError("sidestep_right must be boolean")
        if self.ellipsoid_radii is not None:
            radii = np.asarray(self.ellipsoid_radii, dtype=float)
            if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError("ellipsoid_radii must contain three positive finite values")


def extrapolate(position, velocity, age_s) -> np.ndarray:
    """Dead reckon a simulated position by a supplied age."""
    p = _vec(position, "position")
    v = _vec(velocity, "velocity")
    age = float(age_s)
    if not np.isfinite(age) or age < 0.0:
        raise ValueError("age_s must be finite and non-negative")
    return p + v * age


def closest_approach(rel_pos, rel_vel) -> tuple[float, float]:
    """Return the non-negative closest-approach time and distance."""
    p = _vec(rel_pos, "rel_pos")
    v = _vec(rel_vel, "rel_vel")
    vv = float(np.dot(v, v))
    if vv <= _EPS:
        return 0.0, float(np.linalg.norm(p))
    t = max(0.0, -float(np.dot(p, v)) / vv)
    return t, float(np.linalg.norm(p + t * v))


def _right_direction(axis: np.ndarray) -> np.ndarray:
    """Return the deterministic right direction around a line of sight."""
    z_up = np.array((0.0, 0.0, 1.0))
    right = np.cross(axis, z_up)
    if np.linalg.norm(right) < _EPS:
        right = np.array((1.0, 0.0, 0.0))
    return right / np.linalg.norm(right)


def _vo_correction(
    rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shortest outward correction and normal for a velocity obstacle."""
    distance = float(np.linalg.norm(rel_pos))
    if distance <= radius:
        axis = rel_pos / distance if distance > _EPS else np.array((1.0, 0.0, 0.0))
        w = rel_vel - rel_pos / horizon
        w_length = float(np.linalg.norm(w))
        if w_length > _EPS:
            normal = w / w_length
        else:
            normal = _right_direction(axis)
        return (radius / horizon - w_length) * normal, normal

    axis = rel_pos / distance
    w = rel_vel - rel_pos / horizon
    w_length = float(np.linalg.norm(w))
    dot_w_p = float(np.dot(w, rel_pos))
    if (
        w_length > _EPS
        and dot_w_p < 0.0
        and dot_w_p * dot_w_p > radius * radius * w_length * w_length
    ):
        normal = w / w_length
        sideways = rel_vel - float(np.dot(rel_vel, axis)) * axis
        if float(np.linalg.norm(sideways)) < 1.0e-12:
            coefficient = radius / horizon - w_length
            right = _right_direction(axis)
            # The exact tie is deliberately resolved as a right-hand
            # sidestep, including when the sphere formula has a negative
            # coefficient. The plane normal follows that chosen direction.
            return abs(coefficient) * right, right
        return (radius / horizon - w_length) * normal, normal

    radial_vector = rel_vel - float(np.dot(rel_vel, axis)) * axis
    radial = float(np.linalg.norm(radial_vector))
    side = radial_vector / radial if radial >= 1.0e-12 else _right_direction(axis)
    axial = float(np.dot(rel_vel, axis))
    k = radius / np.sqrt(distance * distance - radius * radius)
    tangent_axial = (distance * distance - radius * radius) / (distance * horizon)
    projected_axial = max(tangent_axial, (axial + k * radial) / (1.0 + k * k))
    boundary = projected_axial * axis + (k * projected_axial) * side
    normal = side - k * axis
    normal /= np.linalg.norm(normal)
    return boundary - rel_vel, normal


def _scaled_pair(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    neighbor_pos: np.ndarray,
    neighbor_vel: np.ndarray,
    config: AvoidanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if config.ellipsoid_radii is None:
        scale = np.ones(3)
        radius = config.radius_m + config.safety_margin_m
    else:
        scale = np.asarray(config.ellipsoid_radii, dtype=float)
        radius = 2.0 + config.safety_margin_m / float(np.min(scale))
    rel_pos = (neighbor_pos - own_pos) / scale
    own_scaled = own_vel / scale
    neighbor_scaled = neighbor_vel / scale
    return rel_pos, own_scaled, neighbor_scaled, scale, radius


def _orca_halfspaces_impl(
    own_pos,
    own_vel,
    neighbors: Sequence[tuple[np.ndarray, np.ndarray]],
    config: AvoidanceConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build permitted-velocity half-spaces from nearby simulated neighbors."""
    own_position = _vec(own_pos, "own_pos")
    own_velocity = _vec(own_vel, "own_vel")
    if not isinstance(config, AvoidanceConfig):
        raise TypeError("config must be AvoidanceConfig")
    result: list[tuple[np.ndarray, np.ndarray]] = []
    scale_for_reach = 1.0 if config.ellipsoid_radii is None else float(np.min(config.ellipsoid_radii))
    for neighbor_position, neighbor_velocity in neighbors:
        npos = _vec(neighbor_position, "neighbor position")
        nvel = _vec(neighbor_velocity, "neighbor velocity")
        rel_pos, own_scaled, neighbor_scaled, scale, radius = _scaled_pair(
            own_position, own_velocity, npos, nvel, config
        )
        distance = float(np.linalg.norm(rel_pos))
        reach = 2.0 * config.max_speed_mps * config.time_horizon_s / scale_for_reach + radius
        if distance > reach:
            continue
        horizon = config.time_horizon_s if distance > radius else 0.05
        rel_velocity = own_scaled - neighbor_scaled
        u_scaled, normal_scaled = _vo_correction(rel_pos, rel_velocity, radius, horizon)
        point_scaled = own_scaled + config.responsibility * u_scaled
        point = point_scaled * scale
        normal = normal_scaled / scale
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= _EPS:
            continue
        result.append((point, normal / normal_norm))
    return result


def orca_halfspaces(
    own_pos, own_vel, neighbors: Sequence[tuple[np.ndarray, np.ndarray]], config: AvoidanceConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build permitted velocity half-spaces using the configured safety margin."""
    return _orca_halfspaces_impl(own_pos, own_vel, neighbors, config)


def _speed_cap(velocity: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(velocity))
    if norm > maximum:
        return maximum * (velocity / norm), True
    return velocity.copy(), False


def _emergency_velocity(
    own_pos: np.ndarray,
    neighbor_pos: np.ndarray,
    config: AvoidanceConfig,
) -> np.ndarray:
    delta = neighbor_pos - own_pos
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance > _EPS else np.array((1.0, 0.0, 0.0))
    away = -direction
    if not config.sidestep_right:
        return config.max_speed_mps * away
    right = _right_direction(direction)
    combined = 0.8 * away + 0.6 * right
    combined_norm = float(np.linalg.norm(combined))
    return config.max_speed_mps * combined / combined_norm


def _project_ball_halfspace(
    preferred: np.ndarray, point: np.ndarray, normal: np.ndarray, maximum: float
) -> np.ndarray:
    """Project onto one half-space and a Euclidean speed ball."""
    normal = normal / np.linalg.norm(normal)
    boundary = float(np.dot(point, normal))
    if float(np.dot(preferred, normal)) >= boundary:
        return preferred.copy()
    projected = preferred + (boundary - float(np.dot(preferred, normal))) * normal
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= maximum:
        return projected
    tangent = preferred - float(np.dot(preferred, normal)) * normal
    available_sq = maximum * maximum - boundary * boundary
    if available_sq < 0.0:
        return _speed_cap(projected, maximum)[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > _EPS:
        return boundary * normal + np.sqrt(max(0.0, available_sq)) * tangent / tangent_norm
    basis = _right_direction(normal)
    return boundary * normal + np.sqrt(max(0.0, available_sq)) * basis


def _affine_projection(
    preferred: np.ndarray,
    active: tuple[int, ...],
    halfspaces: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project onto the active equality set and return a null direction when useful."""
    if not active:
        return preferred.copy(), None
    points = np.asarray([halfspaces[index][0] for index in active], dtype=float)
    normals = np.asarray([halfspaces[index][1] for index in active], dtype=float)
    bounds = np.einsum("ij,ij->i", points, normals)
    if len(active) == 1:
        normal_norm = float(np.linalg.norm(normals[0]))
        if normal_norm <= _EPS:
            return np.full(3, np.nan), None
        normal = normals[0] / normal_norm
        bound = bounds[0] / normal_norm
        projected = preferred + (bound - float(np.dot(preferred, normal))) * normal
        return projected, None
    if len(active) == 2:
        direction = np.cross(normals[0], normals[1])
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            return np.full(3, np.nan), None
        direction /= direction_norm
        gram = normals @ normals.T
        correction = np.linalg.solve(gram, bounds - normals @ preferred)
        projected = preferred + normals.T @ correction
        return projected, direction
    if len(active) == 3:
        if abs(float(np.linalg.det(normals))) <= 1.0e-12:
            return np.full(3, np.nan), None
        return np.linalg.solve(normals, bounds), None
    raise ValueError("at most three active half-spaces are supported")


def _ball_boundary_candidates(
    preferred: np.ndarray,
    active: tuple[int, ...],
    affine_point: np.ndarray,
    line_direction: np.ndarray | None,
    maximum: float,
    halfspaces: list[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """Return candidates on the speed sphere for one affine active set."""
    point_norm = float(np.linalg.norm(affine_point))
    if point_norm <= maximum + 1.0e-12:
        return []
    if not active:
        if float(np.linalg.norm(preferred)) <= _EPS:
            return []
        return [maximum * preferred / float(np.linalg.norm(preferred))]
    if len(active) == 1:
        normal_raw = np.asarray(halfspaces[active[0]][1], dtype=float)
        normal_norm = float(np.linalg.norm(normal_raw))
        if normal_norm <= _EPS:
            return []
        normal = normal_raw / normal_norm
        bound = float(np.dot(halfspaces[active[0]][0], normal))
        if abs(bound) > maximum + 1.0e-12:
            return []
        center = bound * normal
        radius_sq = maximum * maximum - bound * bound
        tangent = affine_point - center
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= _EPS:
            tangent = _right_direction(normal)
            tangent_norm = 1.0
        return [center + np.sqrt(max(0.0, radius_sq)) * tangent / tangent_norm]
    if len(active) == 2 and line_direction is not None:
        normals = np.asarray([halfspaces[index][1] for index in active], dtype=float)
        bounds = np.asarray(
            [float(np.dot(halfspaces[index][0], halfspaces[index][1])) for index in active],
            dtype=float,
        )
        gram = normals @ normals.T
        origin = normals.T @ np.linalg.solve(gram, bounds)
        origin_norm_sq = float(np.dot(origin, origin))
        if origin_norm_sq > maximum * maximum + 1.0e-12:
            return []
        distance = np.sqrt(max(0.0, maximum * maximum - origin_norm_sq))
        return [origin + distance * line_direction, origin - distance * line_direction]
    return []


def _active_set_projection(
    preferred: np.ndarray,
    halfspaces: list[tuple[np.ndarray, np.ndarray]],
    maximum: float,
) -> tuple[np.ndarray, bool]:
    """Project onto a three-dimensional half-space intersection and speed ball."""
    candidates: list[np.ndarray] = []
    candidates.append(preferred.copy())
    candidates.extend(
        _ball_boundary_candidates(
            preferred,
            (),
            preferred,
            None,
            maximum,
            halfspaces,
        )
    )
    limit = min(3, len(halfspaces))
    for count in range(1, limit + 1):
        for active in combinations(range(len(halfspaces)), count):
            affine_point, line_direction = _affine_projection(preferred, active, halfspaces)
            if not np.all(np.isfinite(affine_point)):
                continue
            candidates.append(affine_point)
            candidates.extend(
                _ball_boundary_candidates(
                    preferred,
                    active,
                    affine_point,
                    line_direction,
                    maximum,
                    halfspaces,
                )
            )

    def feasible(value: np.ndarray) -> bool:
        if not np.all(np.isfinite(value)):
            return False
        if any(float(np.dot(value - point, normal)) < -1.0e-9 for point, normal in halfspaces):
            return False
        return float(np.dot(value, value)) <= maximum * maximum + 1.0e-9

    best: np.ndarray | None = None
    best_objective = float("inf")
    for candidate in candidates:
        if not feasible(candidate):
            continue
        objective = float(np.dot(candidate - preferred, candidate - preferred))
        if objective < best_objective:
            best = candidate
            best_objective = objective
    if best is None:
        return preferred.copy(), False
    return best, True


def safe_velocity(
    own_pos,
    own_vel,
    preferred_vel,
    neighbors: Sequence[tuple[np.ndarray, np.ndarray]],
    config: AvoidanceConfig,
    *,
    solver: str = "active_set",
) -> tuple[np.ndarray, frozenset[str]]:
    """Choose a simulated velocity under emergency and reciprocal rules.

    Emergency separation is applied first when a neighbor is inside the
    emergency distance. Otherwise the preferred velocity is projected into
    ORCA half-spaces and the maximum-speed ball. Exact head-on ties use a
    deterministic right-hand sidestep. This creates mirror chirality by
    design, so a reflected exact tie follows the reflected observer's right
    side rather than being a second valid copy of the original path.
    Overlapping pairs use the RVO2 collision rule with a 0.05 second horizon.
    The default active-set projection enumerates the small three-dimensional
    boundary set. ``solver="slsqp"`` keeps the reference optimizer available
    for comparisons. If the selected projection solver cannot find a feasible
    point, a minimum-violation epigraph solve supplies a deterministic fallback
    and sets ``infeasible``. The configured
    safety margin enlarges the separation radius in both half-space creation
    and this solve. In ellipsoid mode positions and velocities are scaled by
    the three radii, so the margin is converted to the conservative scaled
    units before the spherical test; the speed cap remains in world units.
    """
    if solver not in {"active_set", "slsqp"}:
        raise ValueError("solver must be 'active_set' or 'slsqp'")
    own_position = _vec(own_pos, "own_pos")
    own_velocity = _vec(own_vel, "own_vel")
    preferred = _vec(preferred_vel, "preferred_vel")
    if not isinstance(config, AvoidanceConfig):
        raise TypeError("config must be AvoidanceConfig")
    neighbor_data = [(_vec(item[0], "neighbor position"), _vec(item[1], "neighbor velocity")) for item in neighbors]

    effective_radius = config.radius_m + config.safety_margin_m
    if config.ellipsoid_radii is None:
        emergency_default = effective_radius
    else:
        emergency_default = 2.0 + config.safety_margin_m / float(np.min(config.ellipsoid_radii))
    emergency_distance = emergency_default if config.emergency_distance_m is None else config.emergency_distance_m
    nearest: tuple[float, np.ndarray] | None = None
    scale = None if config.ellipsoid_radii is None else np.asarray(config.ellipsoid_radii, dtype=float)
    for npos, _ in neighbor_data:
        delta = npos - own_position
        distance = float(np.linalg.norm(delta if scale is None else delta / scale))
        if distance < emergency_distance and (nearest is None or distance < nearest[0]):
            nearest = (distance, npos)
    if nearest is not None:
        return _emergency_velocity(own_position, nearest[1], config), frozenset({"emergency"})

    capped_preferred, was_capped = _speed_cap(preferred, config.max_speed_mps)
    halfspaces = _orca_halfspaces_impl(own_position, own_velocity, neighbor_data, config)
    flags = {"orca"} if halfspaces else set()
    if not halfspaces:
        if was_capped:
            flags.add("speed_cap")
        return capped_preferred, frozenset(flags)

    if solver == "active_set":
        candidate, feasible = _active_set_projection(
            capped_preferred, halfspaces, config.max_speed_mps
        )
    else:
        constraints = [
            {"type": "ineq", "fun": lambda velocity, point=point, normal=normal: float(np.dot(velocity - point, normal))}
            for point, normal in halfspaces
        ]
        constraints.append({"type": "ineq", "fun": lambda velocity: float(config.max_speed_mps**2 - np.dot(velocity, velocity))})
        result = minimize(
            lambda velocity: float(np.dot(velocity - capped_preferred, velocity - capped_preferred)),
            capped_preferred.copy(),
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1.0e-12, "maxiter": 300, "disp": False},
        )
        candidate = np.asarray(result.x if np.all(np.isfinite(result.x)) else capped_preferred, dtype=float)
        violation = max(
            [max(0.0, -float(np.dot(candidate - point, normal))) for point, normal in halfspaces]
            + [max(0.0, float(np.dot(candidate, candidate) - config.max_speed_mps**2))]
        )
        feasible = bool(result.success and np.all(np.isfinite(candidate)) and violation <= 1.0e-6)
        if feasible and len(halfspaces) == 1:
            candidate = _project_ball_halfspace(
                capped_preferred,
                halfspaces[0][0],
                halfspaces[0][1],
                config.max_speed_mps,
            )
    infeasible = not feasible

    if infeasible:
        start_t = max(
            [max(0.0, -float(np.dot(capped_preferred - point, normal))) for point, normal in halfspaces] + [0.0]
        )
        x0 = np.r_[capped_preferred, start_t]
        epi_constraints = [
            {"type": "ineq", "fun": lambda x, point=point, normal=normal: float(x[3] + np.dot(x[:3] - point, normal))}
            for point, normal in halfspaces
        ]
        epi_constraints.extend(
            [
                {"type": "ineq", "fun": lambda x: float(x[3])},
                {"type": "ineq", "fun": lambda x: float(config.max_speed_mps**2 - np.dot(x[:3], x[:3]))},
            ]
        )
        epi = minimize(
            lambda x: float(x[3]),
            x0,
            method="SLSQP",
            constraints=epi_constraints,
            bounds=[(None, None)] * 3 + [(0.0, None)],
            options={"ftol": 1.0e-12, "maxiter": 300, "disp": False},
        )
        if np.all(np.isfinite(epi.x[:3])):
            candidate = np.asarray(epi.x[:3], dtype=float)
        flags.add("infeasible")

    candidate, candidate_capped = _speed_cap(candidate, config.max_speed_mps)
    if was_capped or candidate_capped:
        flags.add("speed_cap")
    return candidate, frozenset(flags)
