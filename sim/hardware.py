"""Explicit, SI-unit hardware profiles for simulator scenarios.

The profiles describe inputs that can later be replaced by measured motor and
propeller data.  The thrust estimate is a transparent static-propeller
scaling law, not a validated motor map or flight-performance claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

HARDWARE_PROFILE_SCHEMA = "zephyr-hardware-profile-1"
PROFILE_STATUSES = {"measured", "synthetic", "fixture"}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _reject_unknown(raw: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown field(s): {', '.join(unknown)}")


def _require_fields(raw: dict[str, Any], required: set[str], field: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"{field} missing required field(s): {', '.join(missing)}")


def _require_numbers(raw: dict[str, Any], fields: set[str], field: str) -> None:
    for name in fields:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field}.{name} must be a numeric value")


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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MotorProfile":
        if not isinstance(raw, dict):
            raise ValueError("motor must be an object")
        fields = {"part_id", "max_torque_nm", "max_power_w", "max_rpm", "mass_kg", "nominal_voltage_v"}
        _reject_unknown(raw, fields, "motor")
        _require_fields(raw, fields, "motor")
        _require_numbers(raw, fields - {"part_id"}, "motor")
        return cls(**raw)


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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PropellerProfile":
        if not isinstance(raw, dict):
            raise ValueError("propeller must be an object")
        fields = {"part_id", "diameter_m", "pitch_m", "max_rpm", "thrust_coefficient", "power_coefficient", "mass_kg"}
        _reject_unknown(raw, fields, "propeller")
        _require_fields(raw, fields, "propeller")
        _require_numbers(raw, fields - {"part_id"}, "propeller")
        return cls(**raw)

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
    status: str = "synthetic"
    source: str | None = None
    calibration_id: str | None = None
    code_revision: str | None = None
    notes: str | None = None

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
        if self.status not in PROFILE_STATUSES:
            raise ValueError(f"status must be one of {sorted(PROFILE_STATUSES)}")
        if self.status == "measured" and self.source is None:
            raise ValueError("measured profiles require source")
        for name in ("source", "calibration_id", "code_revision", "notes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required_text(value, name))
        if self.status == "measured" and self.calibration_id is None:
            raise ValueError("measured profiles require calibration_id")

    @property
    def total_mass_kg(self) -> float:
        return self.frame_mass_kg + self.motor_count * (self.motor.mass_kg + self.propeller.mass_kg)

    def estimated_thrust_at_rpm_limit_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Estimate static thrust at the lower declared motor/prop RPM limit.

        This transparent coefficient estimate does not apply motor torque or
        electrical power limits. It is a scenario envelope, not a maximum
        thrust measurement.
        """
        rpm = min(self.motor.max_rpm, self.propeller.max_rpm)
        return self.motor_count * self.propeller.estimated_static_thrust_n(rpm, air_density_kg_m3)

    def estimated_max_thrust_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Compatibility alias for the legacy, non-physical field name."""
        return self.estimated_thrust_at_rpm_limit_n(air_density_kg_m3)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["motor"] = self.motor.as_dict()
        result["propeller"] = self.propeller.as_dict()
        result["total_mass_kg"] = self.total_mass_kg
        result["estimated_thrust_at_rpm_limit_n"] = self.estimated_thrust_at_rpm_limit_n()
        result["estimated_max_thrust_n"] = result["estimated_thrust_at_rpm_limit_n"]
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HardwareProfile":
        if not isinstance(raw, dict):
            raise ValueError("hardware profile must be an object")
        if raw.get("schema") != HARDWARE_PROFILE_SCHEMA:
            raise ValueError(f"schema must be {HARDWARE_PROFILE_SCHEMA}")
        raw = {key: value for key, value in raw.items() if key != "schema"}
        fields = {"profile_id", "motor", "propeller", "motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh", "status", "source", "calibration_id", "code_revision", "notes"}
        _reject_unknown(raw, fields, "hardware profile")
        _require_fields(raw, {"profile_id", "motor", "propeller", "motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh", "status"}, "hardware profile")
        _require_numbers(raw, {"motor_count", "arm_length_m", "frame_mass_kg", "battery_capacity_wh"}, "hardware profile")
        if isinstance(raw["motor_count"], bool) or not isinstance(raw["motor_count"], int):
            raise ValueError("hardware profile.motor_count must be an integer")
        return cls(
            profile_id=_required_text(raw["profile_id"], "profile_id"),
            motor=MotorProfile.from_dict(raw["motor"]),
            propeller=PropellerProfile.from_dict(raw["propeller"]),
            motor_count=raw["motor_count"],
            arm_length_m=raw["arm_length_m"],
            frame_mass_kg=raw["frame_mass_kg"],
            battery_capacity_wh=raw["battery_capacity_wh"],
            status=raw["status"],
            source=raw.get("source"),
            calibration_id=raw.get("calibration_id"),
            code_revision=raw.get("code_revision"),
            notes=raw.get("notes"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "HardwareProfile":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_document(self) -> dict[str, Any]:
        """Return the strict interchange document, including its schema id."""
        document = {
            "schema": HARDWARE_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "motor": self.motor.as_dict(),
            "propeller": self.propeller.as_dict(),
            "motor_count": self.motor_count,
            "arm_length_m": self.arm_length_m,
            "frame_mass_kg": self.frame_mass_kg,
            "battery_capacity_wh": self.battery_capacity_wh,
            "status": self.status,
        }
        for key in ("source", "calibration_id", "code_revision", "notes"):
            value = getattr(self, key)
            if value is not None:
                document[key] = value
        return document

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_document(), indent=2) + "\n", encoding="utf-8")


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
