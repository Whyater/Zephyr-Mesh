"""Small, opt-in wind and near-ground disturbance models.

Each ``sample`` call represents one physics substep.  The vehicle chain calls
it at the beginning of a substep and holds the returned sample until the next
substep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .disturbance_api import DisturbanceSample


def _vector3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return array


@dataclass
class ConstantWind:
    """Return one constant world-frame wind velocity.

    This model has no fitted coefficients.  It is a kinematic test input and
    has no external literature source.  UNCONFIRMED as a physical wind model.
    ``velocity_mps`` is copied on construction, and every sample has zero
    thrust and force disturbance.
    """

    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        self.velocity_mps = _vector3(self.velocity_mps, "velocity_mps")

    def sample(
        self,
        t_s: float,
        dt_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
    ) -> DisturbanceSample:
        del t_s, dt_s, position_m, velocity_mps
        return DisturbanceSample(wind_velocity_mps=self.velocity_mps.copy())


@dataclass
class OUGust:
    """A stationary, per-axis Ornstein-Uhlenbeck gust process.

    The exact update for any substep length ``h`` is
    ``x_next = mean + (x - mean) * a + sigma * sqrt(1 - a**2) * n``, where
    ``a = exp(-h / T)`` and ``n`` is a standard normal draw.  The state starts
    with a stationary draw, so the process is stationary at ``t = 0``.
    The vehicle calls ``sample`` once per substep with ``dt_s`` equal to the
    substep length.  A given seed and call sequence produce deterministic
    samples.

    The OU or first-order Gauss-Markov gust form is the simplest member of the
    Dryden turbulence family.  Dryden spectra are shaped differently, so this
    is a simplification.  Source: Uhlenbeck and Ornstein (1930), "On the
    Theory of the Brownian Motion", *Physical Review*,
    https://doi.org/10.1103/PhysRev.36.823; Dryden context: MIL-HDBK-1797,
    *Flying Qualities of Piloted Aircraft*, 1997.  The OU coefficients here
    are literature-derived in form, with ``mean_mps``, ``sigma_mps``, and
    ``time_constant_s`` supplied by the caller.
    """

    mean_mps: np.ndarray
    sigma_mps: np.ndarray
    time_constant_s: float
    seed: int | None = None
    _state: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean_mps = _vector3(self.mean_mps, "mean_mps")
        self.sigma_mps = _vector3(self.sigma_mps, "sigma_mps")
        if np.any(self.sigma_mps < 0.0):
            raise ValueError("sigma_mps must be nonnegative")
        self.time_constant_s = float(self.time_constant_s)
        if not np.isfinite(self.time_constant_s) or self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be finite and positive")
        self._rng = np.random.default_rng(self.seed)
        self._state = self.mean_mps + self.sigma_mps * self._rng.standard_normal(3)

    def sample(
        self,
        t_s: float,
        dt_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
    ) -> DisturbanceSample:
        del t_s, position_m, velocity_mps
        h = float(dt_s)
        if not np.isfinite(h) or h < 0.0:
            raise ValueError("dt_s must be finite and nonnegative")
        a = np.exp(-h / self.time_constant_s)
        noise_scale = np.sqrt(max(0.0, 1.0 - a * a))
        self._state = self.mean_mps + (self._state - self.mean_mps) * a
        self._state = self._state + self.sigma_mps * noise_scale * self._rng.standard_normal(3)
        return DisturbanceSample(wind_velocity_mps=self._state.copy())


@dataclass
class GroundEffect:
    """Increase thrust near the ground using the hover ground-effect formula.

    The thrust scale is ``1 / (1 - (R / (4 z))**2)``, where ``R`` is rotor
    radius and ``z`` is rotor-plane height above ``ground_z_m``.  Heights below
    ``min_height_ratio * R`` are clamped to that value, including positions at
    or below the ground.  The formula and its hover use are from Cheeseman and
    Bennett (1955), "The effect of the ground on a helicopter rotor in forward
    flight", ARC R&M 3021.  The coefficient and formula are
    literature-derived.  The hover approximation is intended for heights at
    least about one rotor radius, which is why the default clamp ratio is 1.
    """

    rotor_radius_m: float
    ground_z_m: float = 0.0
    min_height_ratio: float = 1.0

    def __post_init__(self) -> None:
        self.rotor_radius_m = float(self.rotor_radius_m)
        self.ground_z_m = float(self.ground_z_m)
        self.min_height_ratio = float(self.min_height_ratio)
        if not np.isfinite(self.rotor_radius_m) or self.rotor_radius_m <= 0.0:
            raise ValueError("rotor_radius_m must be finite and positive")
        if not np.isfinite(self.ground_z_m):
            raise ValueError("ground_z_m must be finite")
        if not np.isfinite(self.min_height_ratio) or self.min_height_ratio <= 0.25:
            raise ValueError("min_height_ratio must be finite and greater than 0.25")

    def sample(
        self,
        t_s: float,
        dt_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
    ) -> DisturbanceSample:
        del t_s, dt_s, velocity_mps
        position = _vector3(position_m, "position_m")
        height = max(position[2] - self.ground_z_m, self.min_height_ratio * self.rotor_radius_m)
        ratio = self.rotor_radius_m / (4.0 * height)
        scale = 1.0 / (1.0 - ratio * ratio)
        return DisturbanceSample(thrust_scale=scale)
