"""Small interaction models for a civilian multi vehicle simulation.

The functions in this module are opt in.  They return the same disturbance
objects used by the physics chain, so a caller can combine their results with
other environmental models.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import numpy as np

from sim.physics.disturbance_api import DisturbanceSample


_RHO_AIR = 1.225
_DEFAULT_MASS_KG = 0.030
_DEFAULT_ROTOR_COUNT = 4
_DEFAULT_ROTOR_RADIUS_M = 0.023


def _position(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return array


def _radii(value) -> np.ndarray:
    radii = np.asarray(value, dtype=float)
    if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii must be a finite length-3 vector with positive entries")
    return radii


def ellipsoid_separation(
    p, q, radii_each=(0.12, 0.12, 0.30)
) -> float:
    """Return the normalized distance between two vehicle centres.

    Crazyswarm uses an ellipsoidal keep-out volume (Preiss et al., 2017,
    *Crazyswarm: A large nano-quadcopter swarm*, ICRA 2017,
    https://doi.org/10.1109/ICRA.2017.7989239; Honig et al., 2018,
    *Trajectory planning for quadrotor swarms*, IEEE T-RO).  The
    literature-derived per-vehicle radii here are ``(0.12, 0.12, 0.30)`` m.
    This implementation uses the combined ellipsoid convention: both vehicles'
    radii are summed, so separation means
    ``||diag(2*radii_each)^-1 (p-q)|| >= 1``.
    """
    delta = _position(p, "p") - _position(q, "q")
    combined = 2.0 * _radii(radii_each)
    return float(np.linalg.norm(delta / combined))


def is_separated(p, q, radii_each=(0.12, 0.12, 0.30)) -> bool:
    """Whether two vehicle centres meet the ellipsoidal keep-out distance."""
    return ellipsoid_separation(p, q, radii_each) >= 1.0


def min_vertical_clearance(radii) -> float:
    """Return the vertical centre clearance for two equal ellipsoids."""
    return float(2.0 * _radii(radii)[2])


def min_horizontal_clearance(radii) -> float:
    """Return the horizontal centre clearance for two equal ellipsoids."""
    return float(2.0 * min(_radii(radii)[:2]))


def downwash_wind_at(
    upper_position,
    upper_thrust_n: float,
    rotor_count: int,
    rotor_radius_m: float,
    lower_position,
    max_depth_m: float = 1.0,
) -> np.ndarray:
    """Return the upper vehicle's induced wind at a lower vehicle.

    The structure is literature-derived from actuator-disk momentum theory:
    Leishman, 2006, *Principles of Helicopter Aerodynamics*,
    https://doi.org/10.1016/B978-0-12-374518-7.X5000-2.  For ``N`` equivalent
    disks, ``A = N*pi*R^2``, ``v_h = sqrt(T/(2*rho*A))`` with ``rho = 1.225``
    kg/m^3, and far-wake speed is ``2*v_h``.  The contracted wake radius is
    ``R/sqrt(2)`` where ``R = sqrt(A/pi)``.  The model is trusted only inside
    ``max_depth_m`` and has no decay model, so it is conservative inside this
    column.
    """
    upper = _position(upper_position, "upper_position")
    lower = _position(lower_position, "lower_position")
    thrust = float(upper_thrust_n)
    count = int(rotor_count)
    radius = float(rotor_radius_m)
    depth_limit = float(max_depth_m)
    if not np.isfinite(thrust) or thrust < 0.0:
        raise ValueError("upper_thrust_n must be finite and non-negative")
    if count <= 0 or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("rotor_count must be positive and rotor_radius_m must be positive")
    if not np.isfinite(depth_limit) or depth_limit < 0.0:
        raise ValueError("max_depth_m must be finite and non-negative")

    vertical_gap = float(upper[2] - lower[2])
    if vertical_gap <= 0.0 or vertical_gap > depth_limit:
        return np.zeros(3)
    horizontal_gap = float(np.linalg.norm((lower - upper)[:2]))
    area = count * np.pi * radius * radius
    equivalent_radius = np.sqrt(area / np.pi)
    wake_radius = equivalent_radius / np.sqrt(2.0)
    if horizontal_gap > wake_radius or thrust == 0.0:
        return np.zeros(3)
    induced = np.sqrt(thrust / (2.0 * _RHO_AIR * area))
    return np.array((0.0, 0.0, -2.0 * induced))


class DownwashInteraction:
    """Aggregate downwash winds from all configured vehicles.

    ``profiles`` maps vehicle IDs to either ``(thrust_function, rotor_count,
    rotor_radius_m)`` tuples or mappings with those names.  A thrust function
    receives the vehicle ID and returns newtons.  ``thrust_provider`` is an
    optional fallback with that same signature.  If neither is supplied,
    hover thrust is estimated as ``0.030 kg * 9.81 m/s^2`` (a placeholder for a
    Crazyflie-class vehicle; pass real per-vehicle values).  Rotor defaults are
    four rotors of radius 0.023 m, literature-derived Crazyflie geometry (45 to
    46 mm props: Landry 2015 MIT thesis p.26; Luis and Le Ny 2016,
    arXiv:1608.05786).

    Limit: the four rotors are lumped into one equivalent disk on the vehicle's
    vertical axis, so the wake column is narrow (about 3.3 cm radius for the
    defaults) while the real four jets cover a wider footprint.  A vehicle a few
    centimetres off-axis therefore feels nothing here.  Treat this as a simple
    on-axis model, not a wake map.
    """

    def __init__(
        self,
        profiles: Mapping[str, object] | None = None,
        thrust_provider: Callable[[str], float] | None = None,
        max_depth_m: float = 1.0,
    ) -> None:
        if not np.isfinite(max_depth_m) or max_depth_m < 0.0:
            raise ValueError("max_depth_m must be finite and non-negative")
        self.profiles = dict(profiles or {})
        self.thrust_provider = thrust_provider
        self.max_depth_m = float(max_depth_m)

    @staticmethod
    def _profile_values(vehicle_id: str, profile):
        thrust_fn = None
        count = _DEFAULT_ROTOR_COUNT
        radius = _DEFAULT_ROTOR_RADIUS_M
        mass = _DEFAULT_MASS_KG
        if isinstance(profile, Mapping):
            thrust_fn = profile.get(
                "thrust_function",
                profile.get("thrust_fn", profile.get("thrust_provider")),
            )
            count = profile.get("rotor_count", count)
            radius = profile.get("rotor_radius_m", radius)
            mass = profile.get("mass_kg", mass)
        elif isinstance(profile, (tuple, list)):
            if len(profile) != 3:
                raise ValueError("a vehicle profile tuple must have three entries")
            thrust_fn, count, radius = profile
        elif profile is not None:
            raise ValueError("vehicle profile must be a mapping or three-entry tuple")
        if thrust_fn is not None and not callable(thrust_fn):
            raise ValueError("profile thrust function must be callable")
        return thrust_fn, int(count), float(radius), float(mass)

    def __call__(self, positions: dict[str, np.ndarray], velocities: dict[str, np.ndarray]):
        """Return one disturbance sample per vehicle in ``positions``.

        Velocities are accepted to keep the fleet callback small and stable;
        downwash in this model depends on positions and upper thrust only.
        """
        del velocities
        normalized = {vehicle_id: _position(value, f"positions[{vehicle_id!r}]") for vehicle_id, value in positions.items()}
        winds = {vehicle_id: np.zeros(3) for vehicle_id in normalized}
        for upper_id, upper_position in normalized.items():
            profile = self.profiles.get(upper_id)
            thrust_fn, count, radius, mass = self._profile_values(upper_id, profile)
            if thrust_fn is not None:
                try:
                    thrust = float(thrust_fn(upper_id))
                except TypeError:
                    thrust = float(thrust_fn())
            elif self.thrust_provider is not None:
                thrust = float(self.thrust_provider(upper_id))
            else:
                thrust = mass * 9.81
            for lower_id, lower_position in normalized.items():
                if lower_id == upper_id:
                    continue
                winds[lower_id] += downwash_wind_at(
                    upper_position,
                    thrust,
                    count,
                    radius,
                    lower_position,
                    self.max_depth_m,
                )
        return {
            vehicle_id: DisturbanceSample(wind_velocity_mps=wind)
            for vehicle_id, wind in winds.items()
        }
