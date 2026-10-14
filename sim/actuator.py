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
        rho = self.config.air_density_kg_m3
        propeller = self.profile.propeller
        coefficient = propeller.power_coefficient * rho * propeller.diameter_m**5
        power_limited_rpm = 60.0 * (self.profile.motor.max_power_w / coefficient) ** (1.0 / 3.0)
        torque_limited_rpm = 60.0 * (
            2.0 * math.pi * self.profile.motor.max_torque_nm / coefficient
        ) ** 0.5
        return min(
            self.profile.motor.max_rpm,
            propeller.max_rpm,
            power_limited_rpm,
            torque_limited_rpm,
        )

    @property
    def battery_pct(self) -> float:
        return max(0.0, min(100.0, 100.0 * self.energy_wh / self.profile.battery_capacity_wh))

    def model_provenance(self) -> dict[str, object]:
        """Return JSON-safe provenance for the synthetic actuator envelope.

        A measured hardware profile can identify the eventual calibration
        source, but the coefficient equations and operating limits below
        remain synthetic until those equations are bench-validated.
        """
        profile = self.profile
        return {
            "status": "synthetic",
            "model": "quadrotor_actuator_envelope",
            "units": "SI",
            "equations": {
                "thrust": "Ct * rho * n^2 * D^4",
                "power": "Cp * rho * n^3 * D^5",
                "torque": "P / (2 * pi * n)",
            },
            "air_density_kg_m3": float(self.config.air_density_kg_m3),
            "profile": {
                "profile_id": profile.profile_id,
                "status": profile.status,
                "source": profile.source,
                "calibration_id": profile.calibration_id,
            },
            "limits": {
                "motor_max_rpm": float(profile.motor.max_rpm),
                "propeller_max_rpm": float(profile.propeller.max_rpm),
                "effective_max_rpm": float(self.max_rpm),
                "motor_max_power_w": float(profile.motor.max_power_w),
                "motor_max_torque_nm": float(profile.motor.max_torque_nm),
            },
            "evidence_boundary": "synthetic coefficient envelope; not measured motor map or flight performance",
        }

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
        if self.energy_wh <= 0.0:
            self.rpm[:] = 0.0
            self.last_state = self._state()
            return self.last_state
        voltage_ratio = self.last_state.battery_voltage_v / self.profile.motor.nominal_voltage_v
        voltage_ratio = max(0.0, min(1.0, voltage_ratio))
        desired_rpm = command_values * self.max_rpm * voltage_ratio
        alpha = 1.0 - math.exp(-dt_s / self.config.motor_time_constant_s)
        self.rpm += alpha * (desired_rpm - self.rpm)
        self.rpm = np.clip(self.rpm, 0.0, self.max_rpm)
        state = self._state()
        self.energy_wh = max(0.0, self.energy_wh - state.power_w * dt_s / 3600.0)
        if self.energy_wh <= 0.0:
            self.rpm[:] = 0.0
        self.last_state = self._state()
        return self.last_state

    def reset(self) -> ActuatorState:
        self.rpm[:] = 0.0
        self.energy_wh = self.profile.battery_capacity_wh
        self.last_state = self._state()
        return self.last_state


@dataclass(frozen=True)
class RotorcraftActuatorState:
    """Output of the SI rotor model used by hardware profile v2.

    ``speed_rad_s`` and ``thrust_n`` are the end-of-step telemetry.  When the
    state comes from :meth:`RotorcraftActuator.step`, ``mean_thrust_n`` is the
    exact per-rotor thrust average over that step, which is the value applied
    to the rigid body by the v2 vehicle integrator.
    """

    speed_rad_s: tuple[float, float, float, float]
    rpm: tuple[float, float, float, float]
    thrust_n: tuple[float, float, float, float]
    total_thrust_n: float
    body_torque_nm: tuple[float, float, float]
    power_w: float
    battery_voltage_v: float
    battery_charge_ah: float | None
    battery_pct: float
    duty: tuple[float, float, float, float]
    saturated: bool = False
    mean_thrust_n: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def rotor_speed_rad_s(self) -> tuple[float, float, float, float]:
        return self.speed_rad_s

    @property
    def charge_ah(self) -> float | None:
        return self.battery_charge_ah


def _profile_part(profile: Any, name: str) -> Any:
    """Read a profile section from either a v2 dataclass or a plain mapping."""
    value = profile.get(name) if isinstance(profile, dict) else getattr(profile, name)
    return value


def _field(section: Any, name: str) -> Any:
    return section.get(name) if isinstance(section, dict) else getattr(section, name)


class RotorcraftActuator:
    """Four-rotor lag, thrust, reaction torque, and optional battery model.

    Rotor speed is the state.  Reaction torque is ``-spin_direction * k_m/k_f * T``;
    positive spin direction is counterclockwise viewed from above.
    """

    def __init__(self, profile: Any) -> None:
        self.profile = profile
        rotor = _profile_part(profile, "rotor")
        self.count = int(_field(rotor, "count"))
        if self.count != 4:
            raise ValueError("RotorcraftActuator requires four rotors")
        self.k_f = float(_field(rotor, "thrust_coefficient_n_s2"))
        self.k_m_ratio = float(_field(rotor, "torque_to_thrust_m"))
        self.max_speed_rad_s = float(_field(rotor, "max_speed_rad_s"))
        self.motor_time_constant_s = float(_field(rotor, "motor_time_constant_s"))
        self.spin_directions = np.asarray(_field(rotor, "spin_directions"), dtype=float)
        if self.spin_directions.shape != (4,):
            raise ValueError("spin_directions must have four entries")
        airframe = _profile_part(profile, "airframe")
        self.mass_kg = float(_field(airframe, "mass_kg"))
        battery = _profile_part(profile, "battery")
        self.battery = battery
        if battery is None:
            self.nominal_voltage_v = 1.0
            self.charge_ah: float | None = None
            self.voltage_v = self.nominal_voltage_v
        else:
            self.nominal_voltage_v = float(_field(battery, "nominal_voltage_v"))
            self.cutoff_voltage_v = float(_field(battery, "cutoff_voltage_v"))
            self.capacity_ah = float(_field(battery, "capacity_mah")) / 1000.0
            self.internal_resistance_ohm = float(_field(battery, "internal_resistance_ohm"))
            self.efficiency = float(_field(battery, "efficiency"))
            self.charge_ah = self.capacity_ah
            self.voltage_v = self.nominal_voltage_v
        self.speed_rad_s = np.zeros(4, dtype=float)
        self.duty = np.zeros(4, dtype=float)
        self.last_state = self._state(False, 1.0)

    @property
    def battery_pct(self) -> float:
        if self.charge_ah is None:
            return 100.0
        return max(0.0, min(100.0, 100.0 * self.charge_ah / self.capacity_ah))

    def model_provenance(self) -> dict[str, object]:
        return {
            "status": "synthetic",
            "model": "rotorcraft_profile_v2",
            "equations": {
                "speed_lag": "w_next = w_cmd + (w - w_cmd) exp(-dt/tau)",
                "thrust": "k_f * w^2",
                "reaction_torque": "-spin_direction * (k_m/k_f) * thrust",
            },
            "battery_mode": "linear state of charge with internal resistance" if self.battery is not None else "constant voltage",
        }

    def max_thrust_per_rotor_n(self, voltage_v: float | None = None) -> float:
        voltage = self.nominal_voltage_v if voltage_v is None else float(voltage_v)
        speed = self.max_speed_rad_s
        if self.battery is not None:
            speed *= max(0.0, voltage) / self.nominal_voltage_v
        return self.k_f * speed * speed

    def _state(
        self,
        saturated: bool,
        thrust_scale: float,
        mean_thrust_n: np.ndarray | None = None,
    ) -> RotorcraftActuatorState:
        thrust = self.k_f * self.speed_rad_s**2 * float(thrust_scale)
        torque_z = -self.spin_directions * self.k_m_ratio * thrust
        # Geometry and signs are applied by RotorcraftMixer for roll and pitch.
        torque = np.array((0.0, 0.0, float(np.sum(torque_z))))
        power = float(self.k_m_ratio * self.k_f * np.sum(self.speed_rad_s**3))
        return RotorcraftActuatorState(
            speed_rad_s=tuple(float(v) for v in self.speed_rad_s),
            rpm=tuple(float(v * 60.0 / (2.0 * math.pi)) for v in self.speed_rad_s),
            thrust_n=tuple(float(v) for v in thrust),
            total_thrust_n=float(np.sum(thrust)),
            body_torque_nm=tuple(float(v) for v in torque),
            power_w=power,
            battery_voltage_v=float(self.voltage_v),
            battery_charge_ah=None if self.charge_ah is None else float(self.charge_ah),
            battery_pct=float(self.battery_pct),
            duty=tuple(float(v) for v in self.duty),
            saturated=bool(saturated),
            mean_thrust_n=(
                None
                if mean_thrust_n is None
                else tuple(float(v) for v in np.asarray(mean_thrust_n, dtype=float))
            ),
        )

    def step(self, duty: Iterable[float], dt_s: float, *, thrust_scale: float = 1.0, saturated: bool = False) -> RotorcraftActuatorState:
        values = np.asarray(tuple(duty), dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("duty must be a finite length-4 vector")
        # The profile contract defines duty as a clipped command.  Keep the
        # clipped value in telemetry so callers can safely pass an unbounded
        # controller output and inspect the saturation flag.
        input_saturated = bool(np.any((values < 0.0) | (values > 1.0)))
        values = np.clip(values, 0.0, 1.0)
        dt_s = _finite_positive("dt_s", dt_s)
        thrust_scale = float(thrust_scale)
        if not math.isfinite(thrust_scale) or thrust_scale <= 0.0:
            raise ValueError("thrust_scale must be finite and positive")
        self.duty = values.copy()
        if self.charge_ah is not None and self.charge_ah <= 0.0:
            self.speed_rad_s[:] = 0.0
            self.voltage_v = self.cutoff_voltage_v
            self.last_state = self._state(
                saturated or input_saturated,
                thrust_scale,
                np.zeros(4, dtype=float),
            )
            return self.last_state
        start_voltage = self.voltage_v
        speed_limit = self.max_speed_rad_s
        if self.battery is not None:
            speed_limit *= max(0.0, start_voltage) / self.nominal_voltage_v
        command = values * speed_limit
        start_speed = self.speed_rad_s.copy()
        alpha = math.exp(-dt_s / self.motor_time_constant_s)

        # Integrate (c + (w0-c) exp(-t/tau))^2 over the substep and divide by
        # dt.  The three terms give c^2, 2*c*(w0-c)*(tau/dt)*(1-exp(-dt/tau)),
        # and (w0-c)^2*(tau/(2*dt))*(1-exp(-2*dt/tau)), respectively.  Both
        # c and w0 are clipped to [0, max_speed], so no clipping is needed
        # inside this exact integral.
        tau = self.motor_time_constant_s
        one_minus_alpha = -math.expm1(-dt_s / tau)
        one_minus_alpha2 = -math.expm1(-2.0 * dt_s / tau)
        one_minus_alpha3 = -math.expm1(-3.0 * dt_s / tau)
        delta = start_speed - command
        mean_speed_squared = (
            command**2
            + 2.0 * command * delta * (tau / dt_s) * one_minus_alpha
            + delta**2 * (tau / (2.0 * dt_s)) * one_minus_alpha2
        )
        mean_thrust = self.k_f * mean_speed_squared * thrust_scale
        mean_speed_cubed = (
            command**3
            + 3.0 * command**2 * delta * (tau / dt_s) * one_minus_alpha
            + 3.0 * command * delta**2 * (tau / (2.0 * dt_s)) * one_minus_alpha2
            + delta**3 * (tau / (3.0 * dt_s)) * one_minus_alpha3
        )
        self.speed_rad_s = command + (self.speed_rad_s - command) * alpha
        self.speed_rad_s = np.clip(self.speed_rad_s, 0.0, self.max_speed_rad_s)
        # Battery charge uses the exact substep-average mechanical power.  It
        # comes from integrating (c + (w0-c) exp(-t/tau))^3, whose four terms
        # use exponents 0, 1, 2, and 3 and factors tau/(j*dt) for j > 0.
        mechanical_power = float(self.k_m_ratio * self.k_f * np.sum(mean_speed_cubed))
        if self.charge_ah is not None:
            current = mechanical_power / max(self.efficiency * max(start_voltage, 1e-9), 1e-9)
            self.charge_ah = max(0.0, self.charge_ah - current * dt_s / 3600.0)
            soc = self.charge_ah / self.capacity_ah
            open_circuit = self.cutoff_voltage_v + (self.nominal_voltage_v - self.cutoff_voltage_v) * soc
            self.voltage_v = open_circuit - current * self.internal_resistance_ohm
            if self.charge_ah <= 0.0:
                self.speed_rad_s[:] = 0.0
        else:
            self.voltage_v = self.nominal_voltage_v
        self.last_state = self._state(
            saturated or input_saturated,
            thrust_scale,
            mean_thrust,
        )
        return self.last_state

    def reset(self) -> RotorcraftActuatorState:
        self.speed_rad_s[:] = 0.0
        self.duty[:] = 0.0
        if self.charge_ah is not None:
            self.charge_ah = self.capacity_ah
            self.voltage_v = self.nominal_voltage_v
        self.last_state = self._state(False, 1.0)
        return self.last_state
