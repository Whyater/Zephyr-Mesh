"""Opt-in flat floor for the rigid-body simulator.

Without a ground, a simulated vehicle can fall through z = 0 forever. Real
drones take off from and land on a floor, and an indoor test ends when one
lands hard. ``GroundPlane`` adds that floor. It is off unless a caller passes
one to ``SixDOFInterceptor(ground=...)`` or ``ActuatedQuadrotor(ground=...)``.

How it works, in plain words:

* After every RK4 step, if the body ended up below the floor, put it back on
  the floor and cancel any downward speed. Upward speed is kept, so the
  vehicle can still take off.
* If the body was above the floor at the start of that step, that step was a
  touchdown. Record how fast it was moving down when it crossed the floor.
* A touchdown faster than ``crash_speed_mps`` is flagged as a crash. The run
  is not stopped; the caller decides what a crash means.

Why a clamp instead of a spring floor: a spring has to be very stiff to act
like a floor, and a stiff spring makes RK4 at dt = 0.01 s unstable or bouncy.
A clamp has no stiffness, so it cannot destabilise the integrator. Its cost is
that the body can sit up to about ``g * dt**2 / 2`` (0.5 mm at 100 Hz) below
the floor inside a step before it is moved back.

Limits: the contact is a single point at the centre of mass. There is no
friction, no bounce, no tipping, and no torque from hitting the floor.

The 2.0 m/s crash speed is a placeholder, not a measured value. It needs a
real landing or drop test on the actual airframe before a result cites it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

PLACEHOLDER_CRASH_SPEED_MPS = 2.0


@dataclass(frozen=True)
class Touchdown:
    """One moment where the body came down onto the floor.

    Speeds are positive numbers in m/s. ``vertical_speed_mps`` is the
    downward speed estimated at the instant of crossing the floor;
    ``end_of_step_vertical_speed_mps`` is the raw speed at the end of the
    step that crossed (slightly too fast, because that step ends after the
    crossing). ``time_s`` counts integration time since the ground was
    created or last reset.
    """

    time_s: float
    vertical_speed_mps: float
    horizontal_speed_mps: float
    end_of_step_vertical_speed_mps: float
    crashed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "time_s": self.time_s,
            "vertical_speed_mps": self.vertical_speed_mps,
            "horizontal_speed_mps": self.horizontal_speed_mps,
            "end_of_step_vertical_speed_mps": self.end_of_step_vertical_speed_mps,
            "crashed": self.crashed,
        }


def _crossing_estimate(z_start: float, vz_start: float, vz_end: float, height: float, dt: float) -> tuple[float, float]:
    """Return ``(vertical velocity, time into the step)`` at the floor crossing.

    Treat the vertical acceleration inside the step as constant,
    ``a = (vz_end - vz_start) / dt``. Then the usual kinematics formula
    ``v**2 = v0**2 + 2 * a * (z - z0)`` gives the speed at the floor. In free
    fall the acceleration really is constant, so this is exact; with drag or
    changing thrust it is a close approximation.
    """
    a = (vz_end - vz_start) / dt
    drop = height - z_start  # negative: the floor is below the start point
    v_sq = vz_start * vz_start + 2.0 * a * drop
    v_cross = -math.sqrt(max(v_sq, 0.0))  # moving down, so negative
    if a != 0.0:
        tau = (v_cross - vz_start) / a
    elif vz_start != 0.0:
        tau = drop / vz_start
    else:
        tau = dt
    return v_cross, min(max(tau, 0.0), dt)


class GroundPlane:
    """A flat, level floor at ``z = height_m`` that a body cannot pass through.

    Use one ``GroundPlane`` per body: it keeps that body's touchdown list and
    its own clock, so sharing one between vehicles would mix their histories.
    """

    def __init__(self, height_m: float = 0.0, crash_speed_mps: float = PLACEHOLDER_CRASH_SPEED_MPS) -> None:
        self.height_m = float(height_m)
        self.crash_speed_mps = float(crash_speed_mps)
        if not math.isfinite(self.height_m):
            raise ValueError("height_m must be finite")
        if not math.isfinite(self.crash_speed_mps) or self.crash_speed_mps <= 0.0:
            raise ValueError("crash_speed_mps must be finite and positive")
        self.reset()

    def reset(self) -> None:
        """Forget all touchdowns and restart the ground's clock at zero."""
        self.touchdowns: list[Touchdown] = []
        self.elapsed_s = 0.0
        self.in_contact = False

    @property
    def crashed(self) -> bool:
        """True once any touchdown has been faster than the crash speed."""
        return any(event.crashed for event in self.touchdowns)

    @property
    def first_crash(self) -> Touchdown | None:
        return next((event for event in self.touchdowns if event.crashed), None)

    def model_provenance(self) -> dict[str, object]:
        return {
            "status": "synthetic",
            "model": "ground_plane_clamp",
            "height_m": self.height_m,
            "crash_speed_mps": self.crash_speed_mps,
            "crash_speed_basis": "placeholder; not measured on any airframe",
            "contact": "point contact at centre of mass; no friction, bounce, tipping, or contact torque",
        }

    def after_step(self, drone, start_state: np.ndarray, dt: float) -> Touchdown | None:
        """Apply the floor to ``drone`` after one integration step.

        ``start_state`` is the 13-number state at the start of the step
        (position, velocity, quaternion, angular velocity). Only the vertical
        position and velocity are ever changed. Returns the new touchdown if
        this step was one, else ``None``.
        """
        step_start_time = self.elapsed_s
        self.elapsed_s += dt
        z_end = float(drone.position[2])
        if z_end > self.height_m:
            self.in_contact = False
            return None
        z_start = float(start_state[2])
        vz_start = float(start_state[5])
        vz_end = float(drone.velocity[2])
        drone.position[2] = self.height_m
        if drone.velocity[2] < 0.0:
            drone.velocity[2] = 0.0
        self.in_contact = True
        if z_start <= self.height_m:
            # Already on the floor at the start of the step: resting, not landing.
            return None
        v_cross, tau = _crossing_estimate(z_start, vz_start, vz_end, self.height_m, dt)
        speed = -v_cross
        event = Touchdown(
            time_s=step_start_time + tau,
            vertical_speed_mps=speed,
            horizontal_speed_mps=float(np.hypot(drone.velocity[0], drone.velocity[1])),
            end_of_step_vertical_speed_mps=max(-vz_end, 0.0),
            crashed=speed > self.crash_speed_mps,
        )
        self.touchdowns.append(event)
        return event
