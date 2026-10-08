"""Transparent actuator and battery model for a synthetic quadrotor.

The model is intentionally separated from :mod:`sim.drone`. It gives the
simulator an explicit place to introduce motor lag, thrust saturation, yaw
reaction torque, and battery sag without pretending that catalog coefficients
are measured vehicle data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from sim.hardware import HardwareProfile


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class ActuatorConfig:
    """Dynamic and electrical parameters for one four-rotor airframe."""

    motor_time_constant_s: float = 0.055
    min_battery_voltage_v: float = 3.2
    air_density_kg_m3: float = 1.225
    rotor_drag_torque_ratio: float = 0.015

    def __post_init__(self) -> None:
        object.__setattr__(self, "motor_time_constant_s", _finite_positive("motor_time_constant_s", self.motor_time_constant_s))
        object.__setattr__(self, "min_battery_voltage_v", _finite_positive("min_battery_voltage_v", self.min_battery_voltage_v))
        object.__setattr__(self, "air_density_kg_m3", _finite_positive("air_density_kg_m3", self.air_density_kg_m3))
        ratio = float(self.rotor_drag_torque_ratio)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("rotor_drag_torque_ratio must be finite and nonnegative")
        object.__setattr__(self, "rotor_drag_torque_ratio", ratio)


@dataclass(frozen=True)
class ActuatorState:
    """Inspectable actuator output at one simulation instant."""

    rpm: tuple[float, float, float, float]
    thrust_n: tuple[float, float, float, float]
    total_thrust_n: float
    body_torque_nm: tuple[float, float, float]
    power_w: float
    battery_voltage_v: float
    battery_pct: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuadrotorActuator:
    """Four independent first-order motor channels with explicit saturation."""

    def __init__(self, profile: HardwareProfile, config: ActuatorConfig | None = None) -> None:
        if not isinstance(profile, HardwareProfile):
            raise ValueError("profile must be a HardwareProfile")
        if profile.motor_count != 4:
            raise ValueError("QuadrotorActuator requires a four-motor HardwareProfile")
        self.profile = profile
        self.config = config or ActuatorConfig()
        self.rpm = np.zeros(4, dtype=float)
        self.energy_wh = float(profile.battery_capacity_wh)
        self.last_state = self._state()

    @property
    def max_rpm(self) -> float:
        return min(self.profile.motor.max_rpm, self.profile.propeller.max_rpm)

    @property
    def battery_pct(self) -> float:
        return max(0.0, min(100.0, 100.0 * self.energy_wh / self.profile.battery_capacity_wh))

    def _state(self) -> ActuatorState:
        rho = self.config.air_density_kg_m3
        thrust = np.array([self.profile.propeller.estimated_static_thrust_n(r, rho) for r in self.rpm])
        power = float(sum(self.profile.propeller.estimated_power_w(r, rho) for r in self.rpm))
        arm = self.profile.arm_length_m
        # X-configuration mixer. Rotor order is front-left, front-right,
        # rear-right, rear-left. Opposing rotor signs produce yaw reaction.
        torque = np.array([
            arm * (thrust[0] - thrust[1] - thrust[2] + thrust[3]),
            arm * (-thrust[0] - thrust[1] + thrust[2] + thrust[3]),
            self.config.rotor_drag_torque_ratio * (thrust[0] - thrust[1] + thrust[2] - thrust[3]),
        ])
        voltage = self.config.min_battery_voltage_v + (
            self.profile.motor.nominal_voltage_v - self.config.min_battery_voltage_v
        ) * (self.battery_pct / 100.0)
        return ActuatorState(
            rpm=tuple(float(v) for v in self.rpm),
            thrust_n=tuple(float(v) for v in thrust),
            total_thrust_n=float(np.sum(thrust)),
            body_torque_nm=tuple(float(v) for v in torque),
            power_w=power,
            battery_voltage_v=float(voltage),
            battery_pct=float(self.battery_pct),
        )

    @staticmethod
    def _commands(command: Iterable[float]) -> np.ndarray:
        values = np.asarray(tuple(command), dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("command must be a finite length-4 vector")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("each motor command must be between zero and one")
        return values

    def step(self, command: Iterable[float], dt_s: float) -> ActuatorState:
        """Advance motor lag and battery state by ``dt_s`` seconds."""
        command_values = self._commands(command)
        dt_s = _finite_positive("dt_s", dt_s)
        voltage_ratio = self.last_state.battery_voltage_v / self.profile.motor.nominal_voltage_v
        voltage_ratio = max(0.0, min(1.0, voltage_ratio))
        desired_rpm = command_values * self.max_rpm * voltage_ratio
        alpha = 1.0 - math.exp(-dt_s / self.config.motor_time_constant_s)
        self.rpm += alpha * (desired_rpm - self.rpm)
        self.rpm = np.clip(self.rpm, 0.0, self.max_rpm)
        state = self._state()
        self.energy_wh = max(0.0, self.energy_wh - state.power_w * dt_s / 3600.0)
        self.last_state = self._state()
        return self.last_state

    def reset(self) -> ActuatorState:
        self.rpm[:] = 0.0
        self.energy_wh = self.profile.battery_capacity_wh
        self.last_state = self._state()
        return self.last_state
