"""Actuator-coupled quadrotor vehicle for the synthetic simulator.

``SixDOFInterceptor`` remains the transparent rigid-body reference model. This
module supplies the missing coupling layer: a position controller requests a
wrench, a four-rotor mixer turns it into motor commands, and the measured
actuator output is what reaches the rigid body. The defaults are synthetic
scenario inputs until a bench calibration replaces them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Any

import numpy as np

from sim.actuator import ActuatorState, QuadrotorActuator
from sim.controller import GeometricFlightController
from sim.drone import SixDOFInterceptor
from sim.hardware import HardwareProfile


@dataclass(frozen=True)
class MixerResult:
    """Rotor allocation for one requested collective force and body torque."""

    requested_force_n: float
    requested_torque_nm: tuple[float, float, float]
    rotor_thrust_n: tuple[float, float, float, float]
    motor_command: tuple[float, float, float, float]
    achieved_force_n: float
    achieved_torque_nm: tuple[float, float, float]
    saturated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_force_n": self.requested_force_n,
            "requested_torque_nm": list(self.requested_torque_nm),
            "rotor_thrust_n": list(self.rotor_thrust_n),
            "motor_command": list(self.motor_command),
            "achieved_force_n": self.achieved_force_n,
            "achieved_torque_nm": list(self.achieved_torque_nm),
            "saturated": self.saturated,
        }


@dataclass(frozen=True)
class VehicleStep:
    """Inspectable output after one actuator-coupled integration step."""

    target_position_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    requested_force_body_n: tuple[float, float, float]
    requested_torque_body_nm: tuple[float, float, float]
    applied_force_body_n: tuple[float, float, float]
    applied_torque_body_nm: tuple[float, float, float]
    mixer: MixerResult
    actuator: ActuatorState

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_position_m": list(self.target_position_m),
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps),
            "requested_force_body_n": list(self.requested_force_body_n),
            "requested_torque_body_nm": list(self.requested_torque_body_nm),
            "applied_force_body_n": list(self.applied_force_body_n),
            "applied_torque_body_nm": list(self.applied_torque_body_nm),
            "mixer": self.mixer.as_dict(),
            "actuator": self.actuator.as_dict(),
        }


class QuadrotorMixer:
    """Allocate a body-z collective and roll/pitch/yaw torque to four rotors."""

    def __init__(self, actuator: QuadrotorActuator) -> None:
        self.actuator = actuator
        arm = actuator.profile.arm_length_m
        yaw = actuator.config.rotor_drag_torque_ratio
        self._allocation = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm, -arm, -arm, arm],
                [-arm, -arm, arm, arm],
                [yaw, -yaw, yaw, -yaw],
            ],
            dtype=float,
        )
        self._max_rotor_thrust_n = actuator.profile.propeller.estimated_static_thrust_n(
            actuator.max_rpm, actuator.config.air_density_kg_m3
        )

    def mix(self, collective_force_n: float, body_torque_nm: Iterable[float]) -> MixerResult:
        collective_force_n = float(collective_force_n)
        requested_torque = np.asarray(tuple(body_torque_nm), dtype=float)
        if not math.isfinite(collective_force_n) or collective_force_n < 0.0:
            raise ValueError("collective_force_n must be finite and nonnegative")
        if requested_torque.shape != (3,) or not np.all(np.isfinite(requested_torque)):
            raise ValueError("body_torque_nm must be a finite length-3 vector")
        wrench = np.concatenate(([collective_force_n], requested_torque))
        raw = np.linalg.lstsq(self._allocation, wrench, rcond=None)[0]
        clipped = np.clip(raw, 0.0, self._max_rotor_thrust_n)
        commands = np.sqrt(np.divide(clipped, self._max_rotor_thrust_n))
        achieved = self._allocation @ clipped
        saturated = bool(np.any(np.abs(clipped - raw) > 1e-10))
        return MixerResult(
            requested_force_n=collective_force_n,
            requested_torque_nm=tuple(float(v) for v in requested_torque),
            rotor_thrust_n=tuple(float(v) for v in clipped),
            motor_command=tuple(float(v) for v in commands),
            achieved_force_n=float(achieved[0]),
            achieved_torque_nm=tuple(float(v) for v in achieved[1:]),
            saturated=saturated,
        )


class ActuatedQuadrotor:
    """Controller, mixer, actuator, and 6-DOF body in one explicit step."""

    def __init__(
        self,
        profile: HardwareProfile,
        *,
        initial_position: Iterable[float] = (0.0, 0.0, 0.0),
        controller: GeometricFlightController | None = None,
        drag_coefficient: float = 1.3,
        cross_sectional_area: float = 0.05,
        wind_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not isinstance(profile, HardwareProfile):
            raise ValueError("profile must be a HardwareProfile")
        self.profile = profile
        self.actuator = QuadrotorActuator(profile)
        self.mixer = QuadrotorMixer(self.actuator)
        self.drone = SixDOFInterceptor(
            init_pos=initial_position,
            mass=profile.total_mass_kg,
            drag_coefficient=drag_coefficient,
            cross_sectional_area=cross_sectional_area,
            wind_velocity=wind_velocity,
        )
        self.controller = controller or GeometricFlightController()

    def step(self, target_position: Iterable[float], dt_s: float) -> VehicleStep:
        target = np.asarray(tuple(target_position), dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be a finite length-3 vector")
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        requested_force, requested_torque = self.controller.update_control(self.drone, target, dt_s)
        mixer = self.mixer.mix(float(requested_force[2]), requested_torque)
        actuator_state = self.actuator.step(mixer.motor_command, dt_s)
        applied_force = np.array([0.0, 0.0, actuator_state.total_thrust_n])
        applied_torque = np.asarray(actuator_state.body_torque_nm, dtype=float)
        self.drone.step_physics(applied_force, applied_torque, dt_s)
        return VehicleStep(
            target_position_m=tuple(float(v) for v in target),
            position_m=tuple(float(v) for v in self.drone.position),
            velocity_mps=tuple(float(v) for v in self.drone.velocity),
            requested_force_body_n=tuple(float(v) for v in requested_force),
            requested_torque_body_nm=tuple(float(v) for v in requested_torque),
            applied_force_body_n=tuple(float(v) for v in applied_force),
            applied_torque_body_nm=tuple(float(v) for v in applied_torque),
            mixer=mixer,
            actuator=actuator_state,
        )

    def reset(self, position: Iterable[float] | None = None) -> None:
        if position is not None:
            value = np.asarray(tuple(position), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("position must be a finite length-3 vector")
            self.drone.position = value.copy()
        self.drone.velocity[:] = 0.0
        self.drone.quaternion[:] = (1.0, 0.0, 0.0, 0.0)
        self.drone.angular_velocity[:] = 0.0
        self.actuator.reset()
