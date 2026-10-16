"""Simulated rigid-body adapter for the mission seam (Run A vehicle chain).

``PhysicsAdapter`` drives ``sim.vehicle.ActuatedQuadrotor`` with a v2
rotorcraft profile.  The profile values are literature-derived (none were
measured), aerodynamic drag is off because the profile drag is zero, and the
floor is modelled here because the vehicle chain has none.  Everything is
simulated only: no drone was flown.

Authority, failsafe, geofence latch, link telemetry and events come from
``sim.adapters.SimulatedAdapterCore``, the same code ``SimAdapter`` uses.

Carrot setpoint.  The vehicle chain only takes a position setpoint and its PD
position loop has no velocity limit, so the adapter feeds it a carrot:

* POSITION: the carrot moves toward the commanded point at most
  ``capabilities.max_speed_mps * dt`` per step.  This limits the carrot, not
  the body: the PD loop lags a moving carrot and then catches up, so the true
  speed can overshoot the carrot limit (about 21 % for the Crazyflie-class
  profile: 1.82 m/s against a 1.5 m/s carrot, measured in
  tests/test_e_adapters_redo.py).
* SLOW (weak link policy): the carrot moves at most ``slow_speed_mps * dt``
  per step from where it was, toward a point that is the next step toward
  the command held within ``slow_speed_mps * min(Kd / Kp)`` of the sensed
  position.  That lead is the PD loop's steady lag at the slow speed, so the
  loop's steady speed cannot exceed the slow speed and the overshoot of a
  speed-only carrot (0.36 m/s against 0.3 m/s) does not happen.  The carrot
  never jumps, even when SLOW starts with the carrot well ahead of a fast
  vehicle or when an offboard fix jumps.  The rigid body state is never
  edited, so a vehicle that enters SLOW at speed slows down through its own
  dynamics.  Crazyflie-class profile, SLOW entered at 1.76 m/s after 2 s
  toward a far goal (tests/test_e_adapters_rereview.py): the speed still
  rises to 1.80 m/s for 0.19 s while the body tilts back (flying on without
  SLOW peaks at 1.82 m/s), falls under 0.3 m/s 1.34 s after SLOW began, and
  stays under it.
* HOLD (or no command): the carrot is frozen at the sensed position where the
  hold began.
* Whenever the mode changes (position, hold, land, unpowered) the carrot
  restarts at the sensed position.
* LAND: the carrot is frozen at the true position where the landing began,
  then its z drops at ``land_speed_mps`` until the floor.  A body that is
  moving when the landing starts brakes over a finite distance and comes
  back to the frozen point (0.85 m peak from 1.76 m/s for the default
  profile, 2 cm left at touchdown; the linear PD hand value is 0.66 m).
  The landing is the vehicle's own failsafe, so even with
  ``feedback="offboard"`` it is flown on the true (onboard) state: the fixes
  usually stop for the same reason the landing started, and a held fix
  would steer the body away.  Touchdown is
  declared when the TRUE z is within 0.02 m of the floor and the true speed is
  below 0.05 m/s; the body is then set on the floor (z pinned, velocity and
  spin zeroed) and the rotors stop.

Floor.  After every step a true z below ``ground_plane_z_m`` is set to the
floor.  A body that ends the step at or below the floor without climbing
does not slide: its linear velocity is zeroed, and if it was already on the
floor when the step began it keeps its horizontal position.  Powered, the
attitude stays with the controller (so a take-off command still works);
while the kill switch is latched or after touchdown the angular velocity is
zeroed as well, so an unpowered body that hits the floor stays where it hit.

Unpowered (kill switch, or landed): the rotor speeds are set to zero as soon
as the kill switch latches and held there, the controller is not called,
and the body moves under gravity alone (drag is off,
so a fall from height h takes sqrt(2 h / g)).  ``vehicle.time_s`` advances by
the same dt as the adapter clock on every step, powered or not.

Feedback.  ``feedback="onboard"`` gives the controller the true state.
``feedback="offboard"`` gives it the newest position fix from
``set_position_fix`` (held; the initial position before the first fix) and a
velocity from the last two fixes (zero before that); attitude and spin stay
true.  LAND is the exception (see above).  Geofence checks, the floor and
touchdown always use the TRUE state.
"""
from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Iterable

import numpy as np

from sim.adapters import (
    FLOOR_CONTACT_M,
    Capabilities,
    DroneState,
    Geofence,
    PositionFixHold,
    SimulatedAdapterCore,
    _limited,
    _vector3,
)
from sim.authority import CommandKind, ControlCommand
from sim.failsafe import FailsafeAction, FailsafeConfig, FailsafeDecision, FailsafeState
from sim.hardware import RotorcraftProfile, load_hardware_profile
from sim.vehicle import ActuatedQuadrotor

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "physics" / "profiles" / "crazyflie_class.json"
TOUCHDOWN_HEIGHT_M = 0.02
TOUCHDOWN_SPEED_MPS = 0.05


class _SensedDrone:
    """Controller view of the simulated body: sensed position and velocity, true attitude."""

    def __init__(self, drone: Any, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.asarray(velocity, dtype=float).copy()
        self.quaternion = drone.quaternion.copy()
        self.angular_velocity = drone.angular_velocity.copy()
        self.mass = float(drone.mass)
        self.I_x = float(drone.I_x)
        self.I_y = float(drone.I_y)
        self.I_z = float(drone.I_z)


class _OffboardFeedback:
    """Wraps the flight controller so its position loop sees the held fix."""

    def __init__(self, inner: Any, fix: PositionFixHold) -> None:
        self.inner = inner
        self.fix = fix
        self.use_truth = False  # set while a LAND failsafe flies on the onboard state

    def update_control(self, drone: Any, target_position: np.ndarray, dt: float, **kwargs: Any):
        if self.use_truth:
            return self.inner.update_control(drone, target_position, dt, **kwargs)
        sensed = _SensedDrone(drone, self.fix.position, self.fix.velocity)
        return self.inner.update_control(sensed, target_position, dt, **kwargs)


def _step_toward(point: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    return point + _limited(goal - point, max_step)


class PhysicsAdapter(SimulatedAdapterCore):
    """Simulated v2 rotorcraft (literature-derived profile, drag off, modelled floor).

    Simulated only; this is not flight validation.  See the module docstring
    for the carrot, floor, unpowered and feedback rules.
    """

    def __init__(
        self,
        adapter_id: str,
        *,
        profile: str | Path | RotorcraftProfile | None = None,
        capabilities: Capabilities | None = None,
        initial_position_m: Iterable[float] = (0.0, 0.0, 1.0),
        dt_s: float = 0.01,
        ground_plane_z_m: float = 0.0,
        initial_battery_pct: float = 100.0,
        failsafe_config: FailsafeConfig | None = None,
        geofence: Geofence | None = None,
        feedback: str = "onboard",
        land_speed_mps: float = 0.3,
    ) -> None:
        self._init_core(
            adapter_id,
            capabilities=capabilities or Capabilities(max_speed_mps=1.5),
            dt_s=dt_s,
            initial_battery_pct=initial_battery_pct,
            failsafe_config=failsafe_config,
            geofence=geofence,
        )
        if feedback not in {"onboard", "offboard"}:
            raise ValueError("feedback must be 'onboard' or 'offboard'")
        if profile is None:
            profile = DEFAULT_PROFILE
        loaded = load_hardware_profile(profile) if isinstance(profile, (str, Path)) else profile
        if not isinstance(loaded, RotorcraftProfile):
            raise ValueError("profile must be a v2 RotorcraftProfile")
        land_speed = float(land_speed_mps)
        if not math.isfinite(land_speed) or land_speed <= 0.0:
            raise ValueError("land_speed_mps must be finite and positive")
        ground = float(ground_plane_z_m)
        if not math.isfinite(ground):
            raise ValueError("ground_plane_z_m must be finite")
        start = _vector3(initial_position_m, name="initial_position_m")

        self.profile = loaded
        self.feedback = feedback
        self._ground = ground
        self._land_speed = land_speed
        self.vehicle = ActuatedQuadrotor(loaded, initial_position=start, substeps=1)
        self.vehicle.actuator.speed_rad_s[:] = loaded.hover_speed_rad_s()
        controller = self.vehicle.controller
        # Steady lag of the PD position loop per m/s of carrot speed (Kd / Kp,
        # the most restrictive axis).  Used for the SLOW lead limit.
        self._lag_s_per_mps = float(np.min(np.asarray(controller.K_d_pos, dtype=float) / np.asarray(controller.K_p_pos, dtype=float)))
        self._fix = PositionFixHold(start)
        self._offboard: _OffboardFeedback | None = None
        if feedback == "offboard":
            self._offboard = _OffboardFeedback(controller, self._fix)
            self.vehicle.controller = self._offboard
        self._carrot = start.copy()
        self._carrot_mode = "hold"

    # Read-only views ----------------------------------------------------------

    @property
    def carrot_m(self) -> np.ndarray:
        """Copy of the setpoint handed to the vehicle controller on the last step."""
        return self._carrot.copy()

    @property
    def position_fix_m(self) -> np.ndarray:
        """Copy of the held position fix (offboard feedback)."""
        return self._fix.position.copy()

    def set_position_fix(self, position_m: Iterable[float], timestamp_s: float) -> None:
        self._fix.update(position_m, timestamp_s)

    def _truth_position(self) -> np.ndarray:
        return self.vehicle.drone.position

    def _sensed_position(self) -> np.ndarray:
        if self._offboard is not None and not self._offboard.use_truth:
            return self._fix.position.copy()
        return self.vehicle.drone.position.copy()

    # Motion -------------------------------------------------------------------

    def _update_carrot(self, command: ControlCommand | None, decision: FailsafeDecision, dt: float) -> None:
        sensed = self._sensed_position()
        kind = None if command is None else command.kind
        if kind is CommandKind.POSITION:
            mode = "position"
        elif kind is CommandKind.LAND:
            mode = "land"
        else:
            mode = "hold"
        if mode != self._carrot_mode:
            # Every new mode starts the carrot at the sensed position: a hold
            # or landing is anchored where it began, and a position command
            # after a hold or a reset starts from where the vehicle is.
            self._carrot = sensed
        self._carrot_mode = mode

        if mode == "position":
            assert command is not None and command.position_m is not None
            slow = decision.action is FailsafeAction.SLOW
            speed = self.failsafe.config.slow_speed_mps if slow else self.capabilities.max_speed_mps
            goal = np.asarray(command.position_m, dtype=float)
            if slow:
                # Aim at the next step toward the goal held within the lead of
                # the sensed position, but move the carrot only one slow step
                # toward that aim, so it never jumps.
                aim = _step_toward(self._carrot, goal, speed * dt)
                aim = sensed + _limited(aim - sensed, speed * self._lag_s_per_mps)
                goal = aim
            self._carrot = _step_toward(self._carrot, goal, speed * dt)
        elif mode == "land":
            self._carrot = self._carrot.copy()
            self._carrot[2] = max(self._ground, self._carrot[2] - self._land_speed * dt)

    def _on_kill(self) -> None:
        """Kill switch latched: stop the rotors now, not at the next step."""
        self.vehicle.actuator.speed_rad_s[:] = 0.0
        self._carrot_mode = "unpowered"

    def _unpowered_step(self, dt: float) -> None:
        """Kill switch or landed: rotors at zero, no controller, gravity only."""
        self.vehicle.actuator.speed_rad_s[:] = 0.0
        self._carrot_mode = "unpowered"
        self.vehicle.drone.step_physics(np.zeros(3), np.zeros(3), dt)
        self.vehicle.time_s += dt

    def _rest_on_floor(self) -> None:
        drone = self.vehicle.drone
        drone.position[2] = self._ground
        drone.velocity[:] = 0.0
        drone.angular_velocity[:] = 0.0

    def _floor_contact(self, start: np.ndarray) -> None:
        drone = self.vehicle.drone
        if drone.position[2] < self._ground:
            drone.position[2] = self._ground
            drone.velocity[2] = max(0.0, drone.velocity[2])
        if drone.position[2] > self._ground + FLOOR_CONTACT_M or drone.velocity[2] > 0.0:
            return  # airborne, or climbing away from the floor
        # Inelastic, no-slide floor.
        drone.velocity[:] = 0.0
        if start[2] <= self._ground + FLOOR_CONTACT_M:
            drone.position[:2] = start[:2]
        if self.failsafe.state in {FailsafeState.KILLED, FailsafeState.LANDED}:
            self._rest_on_floor()

    def step(
        self,
        dt_s: float | None = None,
        *,
        link_age_s: float | None = None,
        battery_pct: float | None = None,
        kill_requested: bool = False,
        recent_loss_fraction: float | None = None,
    ) -> DroneState:
        """Advance the simulated vehicle one control step and return normalized state."""
        dt, decision, winner = self._begin_step(
            dt_s,
            link_age_s=link_age_s,
            battery_pct=battery_pct,
            kill_requested=kill_requested,
            recent_loss_fraction=recent_loss_fraction,
        )
        start = self.vehicle.drone.position.copy()
        if self.failsafe.state in {FailsafeState.KILLED, FailsafeState.LANDED}:
            self._unpowered_step(dt)
        else:
            if self._offboard is not None:
                landing = winner.command is not None and winner.command.kind is CommandKind.LAND
                self._offboard.use_truth = landing
            self._update_carrot(winner.command, decision, dt)
            self.vehicle.step(self._carrot, dt)
        self._floor_contact(start)

        drone = self.vehicle.drone
        if (
            self.failsafe.state is FailsafeState.LANDING
            and drone.position[2] <= self._ground + TOUCHDOWN_HEIGHT_M
            and float(np.linalg.norm(drone.velocity)) < TOUCHDOWN_SPEED_MPS
        ):
            self._rest_on_floor()
            self.vehicle.actuator.speed_rad_s[:] = 0.0
            self._carrot_mode = "unpowered"
            self._mark_landed(recent_loss_fraction)
        self._end_step()
        return self.get_state()

    def get_state(self) -> DroneState:
        """Return the true simulated position and velocity plus injected link and battery fields."""
        return self._state(self.vehicle.drone.position, self.vehicle.drone.velocity)

    def model_provenance(self) -> dict[str, Any]:
        result = self.vehicle.model_provenance()
        result.update({
            "profile_id": self.profile.profile_id,
            "floor": "modelled",
            "drag": "off",
            "feedback": self.feedback,
            "simulated_only": True,
        })
        return result
