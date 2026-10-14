"""The small interface between the vehicle chain and disturbance models.

A disturbance is anything outside the drone that pushes on it: wind and gusts,
ground effect near the floor, or downwash from a vehicle above.  The vehicle
asks the disturbance for one ``DisturbanceSample`` at the start of every physics
substep and holds it for that substep, the same way the rotor thrust is held.

Every model is off unless a caller passes one in, so older runs reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _vector3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return array


@dataclass(frozen=True)
class DisturbanceSample:
    """What the environment does to the drone for one substep.

    ``wind_velocity_mps``: air velocity in the world frame (m/s). Drag uses
    drone velocity minus this.
    ``thrust_scale``: multiplies the rotor thrust (1.0 means no change). Ground
    effect raises it above 1 close to the floor.
    ``extra_force_n``: any other world-frame force in newtons, for example the
    downward push of another vehicle's downwash.
    """

    wind_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_scale: float = 1.0
    extra_force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "wind_velocity_mps", _vector3(self.wind_velocity_mps, "wind_velocity_mps"))
        object.__setattr__(self, "extra_force_n", _vector3(self.extra_force_n, "extra_force_n"))
        scale = float(self.thrust_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("thrust_scale must be finite and positive")
        object.__setattr__(self, "thrust_scale", scale)


NO_DISTURBANCE = DisturbanceSample()


class Disturbance(Protocol):
    """Anything with this ``sample`` method can be passed to the vehicle chain."""

    def sample(self, t_s: float, dt_s: float, position_m: np.ndarray, velocity_mps: np.ndarray) -> DisturbanceSample:
        ...


def combine(samples) -> DisturbanceSample:
    """Combine several samples: winds add, thrust scales multiply, forces add."""
    wind = np.zeros(3)
    scale = 1.0
    force = np.zeros(3)
    for sample in samples:
        wind = wind + sample.wind_velocity_mps
        scale = scale * sample.thrust_scale
        force = force + sample.extra_force_n
    return DisturbanceSample(wind, scale, force)
