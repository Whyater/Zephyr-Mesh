"""Simple, explicit environmental forces for the reference simulator."""
import numpy as np

GRAVITY = 9.81
AIR_DENSITY = 1.225


def relative_air_velocity(velocity, wind_velocity=(0.0, 0.0, 0.0)):
    """Return inertial velocity relative to the moving air mass (m/s)."""
    velocity = np.asarray(velocity, dtype=float)
    wind_velocity = np.asarray(wind_velocity, dtype=float)
    if velocity.shape != (3,) or wind_velocity.shape != (3,):
        raise ValueError("velocity and wind_velocity must have shape (3,)")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(wind_velocity)):
        raise ValueError("velocity and wind_velocity must be finite")
    return velocity - wind_velocity


def calculate_drag(velocity, drag_coefficient=1.3, cross_sectional_area=0.05):
    """Return quadratic aerodynamic drag opposite a *relative* velocity.

    The input is deliberately named ``velocity`` for backward compatibility,
    but callers must pass velocity relative to the air. ``SixDOFInterceptor``
    computes that explicitly using :func:`relative_air_velocity` semantics.
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("velocity must have shape (3,)")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be finite")
    if not np.isfinite(drag_coefficient) or drag_coefficient < 0:
        raise ValueError("drag_coefficient must be finite and nonnegative")
    if not np.isfinite(cross_sectional_area) or cross_sectional_area < 0:
        raise ValueError("cross_sectional_area must be finite and nonnegative")
    speed = np.linalg.norm(velocity)
    if speed == 0.0:
        return np.zeros(3, dtype=float)
    drag_magnitude = 0.5 * AIR_DENSITY * speed**2 * drag_coefficient * cross_sectional_area
    return -drag_magnitude * velocity / speed
