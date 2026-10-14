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

from sim.actuator import ActuatorState, QuadrotorActuator, RotorcraftActuator, RotorcraftActuatorState, _field, _profile_part
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
    actuator: ActuatorState | RotorcraftActuatorState

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


class RotorcraftMixer:
    """Exact X-layout wrench allocation for a v2 rotorcraft profile."""

    def __init__(self, actuator: RotorcraftActuator) -> None:
        self.actuator = actuator
        rotor = _profile_part(actuator.profile, "rotor")
        airframe = _profile_part(actuator.profile, "airframe")
        arm = float(_field(airframe, "arm_length_m"))
        lever = arm / math.sqrt(2.0)
        # Rotor order: front-left, front-right, rear-right, rear-left.
        self._allocation = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [lever, -lever, -lever, lever],
            [-lever, -lever, lever, lever],
            [-actuator.spin_directions[0] * actuator.k_m_ratio,
             -actuator.spin_directions[1] * actuator.k_m_ratio,
             -actuator.spin_directions[2] * actuator.k_m_ratio,
             -actuator.spin_directions[3] * actuator.k_m_ratio],
        ], dtype=float)
        self._inverse = np.linalg.inv(self._allocation)

    def mix(self, collective_force_n: float, body_torque_nm: Iterable[float]) -> MixerResult:
        collective_force_n = float(collective_force_n)
        requested_torque = np.asarray(tuple(body_torque_nm), dtype=float)
        if not math.isfinite(collective_force_n) or collective_force_n < 0.0:
            raise ValueError("collective_force_n must be finite and nonnegative")
        if requested_torque.shape != (3,) or not np.all(np.isfinite(requested_torque)):
            raise ValueError("body_torque_nm must be a finite length-3 vector")
        raw = self._inverse @ np.concatenate(([collective_force_n], requested_torque))
        max_thrust = self.actuator.max_thrust_per_rotor_n(self.actuator.voltage_v)
        clipped = np.clip(raw, 0.0, max_thrust)
        duty = np.sqrt(np.divide(clipped, max_thrust, out=np.zeros(4), where=max_thrust > 0.0))
        achieved = self._allocation @ clipped
        return MixerResult(
            requested_force_n=collective_force_n,
            requested_torque_nm=tuple(float(v) for v in requested_torque),
            rotor_thrust_n=tuple(float(v) for v in clipped),
            motor_command=tuple(float(v) for v in duty),
            achieved_force_n=float(achieved[0]),
            achieved_torque_nm=tuple(float(v) for v in achieved[1:]),
            saturated=bool(np.any(np.abs(clipped - raw) > 1e-12)),
        )

    def torque_from_thrust(self, thrust: np.ndarray) -> np.ndarray:
        return self._allocation[1:, :] @ np.asarray(thrust, dtype=float)


def _is_v2_profile(profile: Any) -> bool:
    return not isinstance(profile, HardwareProfile) and hasattr(profile, "airframe") and hasattr(profile, "rotor")


class ActuatedQuadrotor:
    """Controller, mixer, actuator, and 6-DOF body in one explicit step."""

    def __init__(
        self,
        profile: Any,
        *,
        initial_position: Iterable[float] = (0.0, 0.0, 0.0),
        controller: GeometricFlightController | None = None,
        drag_coefficient: float = 1.3,
        cross_sectional_area: float = 0.05,
        wind_velocity: Iterable[float] = (0.0, 0.0, 0.0),
        substeps: int = 1,
        disturbance: Any = None,
    ) -> None:
        if not isinstance(profile, HardwareProfile) and not _is_v2_profile(profile):
            raise ValueError("profile must be a HardwareProfile")
        if isinstance(substeps, bool) or int(substeps) != substeps or int(substeps) < 1:
            raise ValueError("substeps must be an integer >= 1")
        self.substeps = int(substeps)
        self.disturbance = disturbance
        self.time_s = 0.0
        self.profile = profile
        self._v2 = _is_v2_profile(profile)
        if self._v2:
            self.actuator = RotorcraftActuator(profile)
            self.mixer = RotorcraftMixer(self.actuator)
            airframe = _profile_part(profile, "airframe")
            mass = float(_field(airframe, "mass_kg"))
            inertia = tuple(float(v) for v in _field(airframe, "inertia_kg_m2"))
            if drag_coefficient == 1.3:
                drag_coefficient = float(_field(airframe, "drag_coefficient"))
            if cross_sectional_area == 0.05:
                cross_sectional_area = float(_field(airframe, "cross_sectional_area_m2"))
        else:
            self.actuator = QuadrotorActuator(profile)
            self.mixer = QuadrotorMixer(self.actuator)
            mass = profile.total_mass_kg
            inertia = profile.estimated_inertia_kg_m2()
        self.drone = SixDOFInterceptor(
            init_pos=initial_position,
            mass=mass,
            drag_coefficient=drag_coefficient,
            cross_sectional_area=cross_sectional_area,
            wind_velocity=wind_velocity,
            # Estimated from the profile geometry; see HardwareProfile.estimated_inertia_kg_m2.
            inertia_kg_m2=inertia,
        )
        self.controller = controller or GeometricFlightController()

    def model_provenance(self) -> dict[str, object]:
        """Return the coupled dynamics and actuator provenance envelope."""
        return {
            "status": "synthetic",
            "model": "actuated_quadrotor",
            "dynamics": self.drone.model_provenance(),
            "inertia_basis": "estimate from profile geometry (motor point masses plus a frame plate); not measured",
            "actuator": self.actuator.model_provenance(),
            "evidence_boundary": "synthetic controller, actuator, and rigid-body fixture; no flight validation",
        }

    @staticmethod
    def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        return np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
            [2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)],
            [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)],
        ])

    def _step_v2(self, target: np.ndarray, dt_s: float, extra_disturbance: Any = None) -> VehicleStep:
        requested_force, requested_torque = self.controller.update_control(self.drone, target, dt_s)
        mixer = self.mixer.mix(float(requested_force[2]), requested_torque)
        h = dt_s / self.substeps
        final_state = self.actuator.last_state
        final_force = np.zeros(3)
        final_torque = np.zeros(3)
        for index in range(self.substeps):
            t = self.time_s + index * h
            sample = extra_disturbance
            if self.disturbance is not None:
                own = self.disturbance.sample(t, h, self.drone.position.copy(), self.drone.velocity.copy())
                if sample is None:
                    sample = own
                else:
                    from sim.physics.disturbance_api import combine
                    sample = combine((own, sample))
            if sample is None:
                # Preserve the rigid-body wind supplied at construction when
                # no explicit disturbance source is active.
                from sim.physics.disturbance_api import DisturbanceSample
                sample = DisturbanceSample(wind_velocity_mps=self.drone.wind_velocity.copy())
            final_state = self.actuator.step(mixer.motor_command, h, thrust_scale=sample.thrust_scale, saturated=mixer.saturated)
            # The actuator integrates the exact mean of w(t)^2 over this
            # substep and exposes it separately from end-state telemetry.
            # Apply that mean thrust and its matching reaction torque across
            # the same interval, rather than using either endpoint value.
            mean_thrust = np.asarray(final_state.mean_thrust_n, dtype=float)
            rotor_torque = self.mixer.torque_from_thrust(mean_thrust)
            applied_torque = rotor_torque
            applied_force = np.array([0.0, 0.0, float(np.sum(mean_thrust))])
            applied_force += self._rotation_from_quaternion(self.drone.quaternion).T @ sample.extra_force_n
            self.drone.step_physics(applied_force, applied_torque, h, wind_velocity=sample.wind_velocity_mps)
            final_force, final_torque = applied_force, applied_torque
        self.time_s += dt_s
        return VehicleStep(
            target_position_m=tuple(float(v) for v in target),
            position_m=tuple(float(v) for v in self.drone.position),
            velocity_mps=tuple(float(v) for v in self.drone.velocity),
            requested_force_body_n=tuple(float(v) for v in requested_force),
            requested_torque_body_nm=tuple(float(v) for v in requested_torque),
            applied_force_body_n=tuple(float(v) for v in final_force),
            applied_torque_body_nm=tuple(float(v) for v in final_torque),
            mixer=mixer,
            actuator=final_state,
        )

    def _step_with_extra_disturbance(self, target_position: Iterable[float], dt_s: float, extra_disturbance: Any = None) -> VehicleStep:
        target = np.asarray(tuple(target_position), dtype=float)
        dt_s = float(dt_s)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be a finite length-3 vector")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if self._v2:
            return self._step_v2(target, dt_s, extra_disturbance)
        # Keep this sequence identical to the original v1 path.
        requested_force, requested_torque = self.controller.update_control(self.drone, target, dt_s)
        mixer = self.mixer.mix(float(requested_force[2]), requested_torque)
        actuator_state = self.actuator.step(mixer.motor_command, dt_s)
        applied_force = np.array([0.0, 0.0, actuator_state.total_thrust_n])
        applied_torque = np.asarray(actuator_state.body_torque_nm, dtype=float)
        self.drone.step_physics(applied_force, applied_torque, dt_s)
        self.time_s += dt_s
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

    def step(self, target_position: Iterable[float], dt_s: float) -> VehicleStep:
        target = np.asarray(tuple(target_position), dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be a finite length-3 vector")
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        return self._step_with_extra_disturbance(target, dt_s)

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
        self.time_s = 0.0
