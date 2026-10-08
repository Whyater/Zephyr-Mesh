"""Explicit, SI-unit hardware profiles for simulator scenarios.

The profiles describe inputs that can later be replaced by measured motor and
propeller data.  The thrust estimate is a transparent static-propeller
scaling law, not a validated motor map or flight-performance claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class MotorProfile:
    """Motor limits in SI units and manufacturer/profile identity."""

    part_id: str
    max_torque_nm: float
    max_power_w: float
    max_rpm: float
    mass_kg: float = 0.01
    nominal_voltage_v: float = 3.7

    def __post_init__(self) -> None:
        if not isinstance(self.part_id, str) or not self.part_id:
            raise ValueError("part_id must be a non-empty string")
        for name in ("max_torque_nm", "max_power_w", "max_rpm", "mass_kg", "nominal_voltage_v"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PropellerProfile:
    """Propeller geometry and dimensionless coefficients.

    ``thrust_coefficient`` and ``power_coefficient`` are intentionally inputs;
    they must be replaced or calibrated before using the estimate for a real
    vehicle.
    """

    part_id: str
    diameter_m: float
    pitch_m: float
    max_rpm: float
    thrust_coefficient: float = 0.10
    power_coefficient: float = 0.04
    mass_kg: float = 0.002

    def __post_init__(self) -> None:
        if not isinstance(self.part_id, str) or not self.part_id:
            raise ValueError("part_id must be a non-empty string")
        for name in ("diameter_m", "pitch_m", "max_rpm", "mass_kg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("thrust_coefficient", "power_coefficient"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def estimated_static_thrust_n(self, rpm: float, air_density_kg_m3: float = 1.225) -> float:
        """Return ``Ct * rho * n² * D⁴`` for a declared rotor coefficient."""
        rpm = float(rpm)
        rho = float(air_density_kg_m3)
        if not math.isfinite(rpm) or rpm < 0.0:
            raise ValueError("rpm must be finite and nonnegative")
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("air_density_kg_m3 must be finite and positive")
        revolutions_per_second = rpm / 60.0
        return self.thrust_coefficient * rho * revolutions_per_second**2 * self.diameter_m**4

    def estimated_power_w(self, rpm: float, air_density_kg_m3: float = 1.225) -> float:
        """Return ``Cp * rho * n³ * D⁵`` for a declared rotor coefficient."""
        rpm = float(rpm)
        rho = float(air_density_kg_m3)
        if not math.isfinite(rpm) or rpm < 0.0:
            raise ValueError("rpm must be finite and nonnegative")
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("air_density_kg_m3 must be finite and positive")
        revolutions_per_second = rpm / 60.0
        return self.power_coefficient * rho * revolutions_per_second**3 * self.diameter_m**5


@dataclass(frozen=True)
class HardwareProfile:
    """A per-agent airframe and its replaceable motor/propeller parts."""

    profile_id: str
    motor: MotorProfile
    propeller: PropellerProfile
    motor_count: int = 4
    arm_length_m: float = 0.10
    frame_mass_kg: float = 0.03
    battery_capacity_wh: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.motor, MotorProfile) or not isinstance(self.propeller, PropellerProfile):
            raise ValueError("motor and propeller must be typed profiles")
        if isinstance(self.motor_count, bool) or int(self.motor_count) != self.motor_count or int(self.motor_count) < 1:
            raise ValueError("motor_count must be a positive integer")
        object.__setattr__(self, "motor_count", int(self.motor_count))
        for name in ("arm_length_m", "frame_mass_kg", "battery_capacity_wh"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def total_mass_kg(self) -> float:
        return self.frame_mass_kg + self.motor_count * (self.motor.mass_kg + self.propeller.mass_kg)

    def estimated_max_thrust_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Estimate total static thrust at the lower motor/prop RPM limit."""
        rpm = min(self.motor.max_rpm, self.propeller.max_rpm)
        return self.motor_count * self.propeller.estimated_static_thrust_n(rpm, air_density_kg_m3)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["motor"] = self.motor.as_dict()
        result["propeller"] = self.propeller.as_dict()
        result["total_mass_kg"] = self.total_mass_kg
        result["estimated_max_thrust_n"] = self.estimated_max_thrust_n()
        return result


def default_hardware_profile(agent_id: str) -> HardwareProfile:
    """Return a deterministic, visibly unique synthetic profile for one agent.

    The variation is deliberate scenario data, not a claim about a particular
    catalog part. Replace these fields with bench measurements before using a
    profile for flight planning.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    signature = sum((index + 1) * ord(char) for index, char in enumerate(agent_id))
    scale = 1.0 + (signature % 9) * 0.01
    diameter = 0.0762 + (signature % 5) * 0.001
    return HardwareProfile(
        profile_id=f"{agent_id}-synthetic-profile",
        motor=MotorProfile(
            part_id=f"{agent_id}-motor",
            max_torque_nm=0.0018 * scale,
            max_power_w=18.0 * scale,
            max_rpm=38_000.0 * scale,
        ),
        propeller=PropellerProfile(
            part_id=f"{agent_id}-prop",
            diameter_m=diameter,
            pitch_m=0.038 + (signature % 4) * 0.001,
            max_rpm=36_000.0 * scale,
        ),
    )
