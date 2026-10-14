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
from typing import Any, Mapping

HARDWARE_PROFILE_SCHEMA = "zephyr-hardware-profile-1"
PROFILE_STATUSES = {"measured", "synthetic", "fixture"}
HARDWARE_PROFILE_SCHEMA_V2 = "zephyr-hardware-profile-2"
PROFILE_V2_STATUSES = {"literature-derived", "synthetic", "fixture", "measured"}


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

    def estimated_torque_nm(self, rpm: float, air_density_kg_m3: float = 1.225) -> float:
        """Return the declared-coefficient shaft-torque estimate ``P / omega``.

        This is derived from the synthetic propeller power coefficient. It is
        an actuator-envelope input, not measured motor torque data.
        """
        rpm = float(rpm)
        if not math.isfinite(rpm) or rpm < 0.0:
            raise ValueError("rpm must be finite and nonnegative")
        rho = float(air_density_kg_m3)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("air_density_kg_m3 must be finite and positive")
        if rpm == 0.0:
            return 0.0
        revolutions_per_second = rpm / 60.0
        return self.estimated_power_w(rpm, rho) / (2.0 * math.pi * revolutions_per_second)


@dataclass(frozen=True)
class HardwareProfile:
    """A per-agent airframe and its replaceable motor/propeller parts.

    ``arm_length_m`` is the lever arm used for roll and pitch torque: the
    perpendicular distance from each motor axis to the body x (roll) axis and
    to the body y (pitch) axis, so motors sit at (+-arm, +-arm) in the body
    plane.  For an X frame measured centre to motor as ``L``, enter
    ``L / sqrt(2)``.  Entering ``L`` directly overstates roll and pitch torque
    by a factor of sqrt(2), about 41 percent.
    """

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

    def estimated_inertia_kg_m2(self) -> tuple[float, float, float]:
        """Return an ESTIMATE of the principal inertia (I_x, I_y, I_z) in kg m^2.

        Each motor plus propeller is a point mass at (+-arm, +-arm), and the
        frame mass (including battery and electronics) is a uniform square
        plate of side ``2 * arm`` spanning the motors.  This is a geometry
        estimate for a four-motor X layout, not a measured inertia.
        """
        arm = self.arm_length_m
        rotor_mass = self.motor.mass_kg + self.propeller.mass_kg
        # Point masses: distance to the roll (or pitch) axis is arm; to the yaw axis it is sqrt(2) * arm.
        roll = self.motor_count * rotor_mass * arm**2
        # Uniform plate of side s = 2 * arm: m * s**2 / 12 about an in-plane axis.
        plate = self.frame_mass_kg * (2.0 * arm) ** 2 / 12.0
        return (roll + plate, roll + plate, 2.0 * (roll + plate))

    def estimated_thrust_at_rpm_limit_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Estimate static thrust at the lower declared motor/prop RPM limit.

        This transparent coefficient estimate does not apply motor torque or
        electrical power limits, so it overstates the available thrust when
        those limits bind. It is a scenario envelope, not a maximum thrust
        measurement. Use :meth:`estimated_max_thrust_limited_n` when limits
        must be applied.
        """
        rpm = min(self.motor.max_rpm, self.propeller.max_rpm)
        return self.motor_count * self.propeller.estimated_static_thrust_n(rpm, air_density_kg_m3)

    def estimated_max_thrust_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Compatibility alias for the legacy, non-physical field name."""
        return self.estimated_thrust_at_rpm_limit_n(air_density_kg_m3)

    def estimated_max_thrust_limited_n(self, air_density_kg_m3: float = 1.225) -> float:
        """Estimate total static thrust after motor power and torque limits.

        The effective RPM envelope mirrors :class:`QuadrotorActuator`: motor
        and propeller RPM limits are combined with the RPM implied by the
        declared power and torque limits.
        """
        rho = float(air_density_kg_m3)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("air_density_kg_m3 must be finite and positive")
        coefficient = self.propeller.power_coefficient * rho * self.propeller.diameter_m**5
        power_limited_rpm = 60.0 * (self.motor.max_power_w / coefficient) ** (1.0 / 3.0)
        torque_limited_rpm = 60.0 * (
            2.0 * math.pi * self.motor.max_torque_nm / coefficient
        ) ** 0.5
        rpm = min(self.motor.max_rpm, self.propeller.max_rpm, power_limited_rpm, torque_limited_rpm)
        return self.motor_count * self.propeller.estimated_static_thrust_n(rpm, rho)

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


@dataclass(frozen=True)
class AirframeProfile:
    """Rigid-body inputs for a four-rotor literature profile."""

    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    arm_length_m: float
    drag_coefficient: float
    cross_sectional_area_m2: float

    def __post_init__(self) -> None:
        mass = _finite_profile_number("airframe.mass_kg", self.mass_kg, positive=True)
        inertia = _finite_profile_vector("airframe.inertia_kg_m2", self.inertia_kg_m2, 3, positive=True)
        arm = _finite_profile_number("airframe.arm_length_m", self.arm_length_m, positive=True)
        drag = _finite_profile_number("airframe.drag_coefficient", self.drag_coefficient, positive=False)
        area = _finite_profile_number("airframe.cross_sectional_area_m2", self.cross_sectional_area_m2, positive=False)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "inertia_kg_m2", inertia)
        object.__setattr__(self, "arm_length_m", arm)
        object.__setattr__(self, "drag_coefficient", drag)
        object.__setattr__(self, "cross_sectional_area_m2", area)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "arm_length_m": self.arm_length_m,
            "drag_coefficient": self.drag_coefficient,
            "cross_sectional_area_m2": self.cross_sectional_area_m2,
        }


@dataclass(frozen=True)
class RotorProfile:
    """Rotor geometry, allocation, and motor-lag inputs."""

    count: int
    layout: str
    thrust_coefficient_n_s2: float
    torque_to_thrust_m: float
    max_speed_rad_s: float
    motor_time_constant_s: float
    prop_radius_m: float
    spin_directions: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count != 4:
            raise ValueError("rotor.count must be exactly 4")
        if self.layout != "x":
            raise ValueError("rotor.layout must be 'x'")
        k_f = _finite_profile_number("rotor.thrust_coefficient_n_s2", self.thrust_coefficient_n_s2, positive=True)
        ratio = _finite_profile_number("rotor.torque_to_thrust_m", self.torque_to_thrust_m, positive=True)
        max_speed = _finite_profile_number("rotor.max_speed_rad_s", self.max_speed_rad_s, positive=True)
        tau = _finite_profile_number("rotor.motor_time_constant_s", self.motor_time_constant_s, positive=True)
        radius = _finite_profile_number("rotor.prop_radius_m", self.prop_radius_m, positive=True)
        if not isinstance(self.spin_directions, (list, tuple)):
            raise ValueError("rotor.spin_directions must be [1, -1, 1, -1]")
        directions = tuple(self.spin_directions)
        if directions != (1, -1, 1, -1):
            raise ValueError("rotor.spin_directions must be [1, -1, 1, -1]")
        object.__setattr__(self, "count", 4)
        object.__setattr__(self, "thrust_coefficient_n_s2", k_f)
        object.__setattr__(self, "torque_to_thrust_m", ratio)
        object.__setattr__(self, "max_speed_rad_s", max_speed)
        object.__setattr__(self, "motor_time_constant_s", tau)
        object.__setattr__(self, "prop_radius_m", radius)
        object.__setattr__(self, "spin_directions", directions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "layout": self.layout,
            "thrust_coefficient_n_s2": self.thrust_coefficient_n_s2,
            "torque_to_thrust_m": self.torque_to_thrust_m,
            "max_speed_rad_s": self.max_speed_rad_s,
            "motor_time_constant_s": self.motor_time_constant_s,
            "prop_radius_m": self.prop_radius_m,
            "spin_directions": list(self.spin_directions),
        }


@dataclass(frozen=True)
class BatteryProfile:
    """Optional linear state-of-charge battery model for a v2 profile."""

    nominal_voltage_v: float
    cutoff_voltage_v: float
    capacity_mah: float
    internal_resistance_ohm: float
    efficiency: float

    def __post_init__(self) -> None:
        nominal = _finite_profile_number("battery.nominal_voltage_v", self.nominal_voltage_v, positive=True)
        cutoff = _finite_profile_number("battery.cutoff_voltage_v", self.cutoff_voltage_v, positive=True)
        if cutoff >= nominal:
            raise ValueError("battery.cutoff_voltage_v must be below nominal_voltage_v")
        capacity = _finite_profile_number("battery.capacity_mah", self.capacity_mah, positive=True)
        resistance = _finite_profile_number("battery.internal_resistance_ohm", self.internal_resistance_ohm, positive=False)
        efficiency = _finite_profile_number("battery.efficiency", self.efficiency, positive=True)
        if efficiency > 1.0:
            raise ValueError("battery.efficiency must be at most 1")
        object.__setattr__(self, "nominal_voltage_v", nominal)
        object.__setattr__(self, "cutoff_voltage_v", cutoff)
        object.__setattr__(self, "capacity_mah", capacity)
        object.__setattr__(self, "internal_resistance_ohm", resistance)
        object.__setattr__(self, "efficiency", efficiency)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nominal_voltage_v": self.nominal_voltage_v,
            "cutoff_voltage_v": self.cutoff_voltage_v,
            "capacity_mah": self.capacity_mah,
            "internal_resistance_ohm": self.internal_resistance_ohm,
            "efficiency": self.efficiency,
        }


def _finite_profile_number(field: str, value: Any, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric value")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def _finite_profile_vector(field: str, value: Any, length: int, *, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} numeric values")
    return tuple(_finite_profile_number(f"{field}[{index}]", item, positive=positive) for index, item in enumerate(value))


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _range_contains(value: Any, bounds: Any) -> bool:
    if isinstance(value, (list, tuple)):
        if not isinstance(bounds, (list, tuple)):
            return False
        # A scalar [low, high] range applies to every element. A nested
        # range gives each element its own bounds.
        if len(bounds) == 2 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(float(item)) for item in bounds
        ):
            return all(_range_contains(item, bounds) for item in value)
        return len(bounds) == len(value) and all(
            _range_contains(item, item_bounds) for item, item_bounds in zip(value, bounds)
        )
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in bounds) and float(bounds[0]) <= float(value) <= float(bounds[1])


_V2_NUMERIC_PATHS = (
    "airframe.mass_kg", "airframe.inertia_kg_m2", "airframe.arm_length_m",
    "airframe.drag_coefficient", "airframe.cross_sectional_area_m2",
    "rotor.count", "rotor.thrust_coefficient_n_s2", "rotor.torque_to_thrust_m",
    "rotor.max_speed_rad_s", "rotor.motor_time_constant_s", "rotor.prop_radius_m",
    "rotor.spin_directions",
    "battery.nominal_voltage_v", "battery.cutoff_voltage_v", "battery.capacity_mah",
    "battery.internal_resistance_ohm", "battery.efficiency",
)


# Paths that may carry the EXCLUDED status (unsourced, so switched off).
_V2_EXCLUDABLE_PATHS = ("airframe.drag_coefficient", "airframe.cross_sectional_area_m2")


def _path_value(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


@dataclass(frozen=True)
class RotorcraftProfile:
    """Strict v2 literature-profile document for a four-rotor micro-quad."""

    profile_id: str
    status: str
    notes: str
    airframe: AirframeProfile
    rotor: RotorProfile
    battery: BatteryProfile | None
    parameters: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if not isinstance(self.status, str) or self.status not in PROFILE_V2_STATUSES:
            raise ValueError(f"status must be one of {sorted(PROFILE_V2_STATUSES)}")
        if self.status == "measured":
            raise ValueError("measured v2 profiles need the provenance block (not implemented)")
        if not isinstance(self.airframe, AirframeProfile) or not isinstance(self.rotor, RotorProfile):
            raise ValueError("airframe and rotor must be typed profiles")
        if self.battery is not None and not isinstance(self.battery, BatteryProfile):
            raise ValueError("battery must be a BatteryProfile or null")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be an object")
        copied = {str(path): dict(entry) for path, entry in self.parameters.items()}
        object.__setattr__(self, "parameters", copied)

    @property
    def total_mass_kg(self) -> float:
        return self.airframe.mass_kg

    @property
    def mass_kg(self) -> float:
        return self.airframe.mass_kg

    @property
    def inertia_kg_m2(self) -> tuple[float, float, float]:
        return self.airframe.inertia_kg_m2

    @property
    def arm_length_m(self) -> float:
        return self.airframe.arm_length_m

    @property
    def drag_coefficient(self) -> float:
        return self.airframe.drag_coefficient

    @property
    def cross_sectional_area_m2(self) -> float:
        return self.airframe.cross_sectional_area_m2

    @property
    def rotor_count(self) -> int:
        return self.rotor.count

    @property
    def thrust_coefficient_n_s2(self) -> float:
        return self.rotor.thrust_coefficient_n_s2

    @property
    def torque_to_thrust_m(self) -> float:
        return self.rotor.torque_to_thrust_m

    @property
    def max_speed_rad_s(self) -> float:
        return self.rotor.max_speed_rad_s

    @property
    def motor_time_constant_s(self) -> float:
        return self.rotor.motor_time_constant_s

    @property
    def spin_directions(self) -> tuple[int, int, int, int]:
        return self.rotor.spin_directions

    def hover_speed_rad_s(self, gravity_mps2: float = 9.81) -> float:
        gravity = _finite_profile_number("gravity_mps2", gravity_mps2, positive=True)
        return math.sqrt(self.airframe.mass_kg * gravity / (self.rotor.count * self.rotor.thrust_coefficient_n_s2))

    def max_thrust_per_rotor_n(self) -> float:
        return self.rotor.thrust_coefficient_n_s2 * self.rotor.max_speed_rad_s**2

    def thrust_to_weight(self, gravity_mps2: float = 9.81) -> float:
        gravity = _finite_profile_number("gravity_mps2", gravity_mps2, positive=True)
        return self.rotor.count * self.max_thrust_per_rotor_n() / (self.airframe.mass_kg * gravity)

    def as_document(self) -> dict[str, Any]:
        return {
            "schema": HARDWARE_PROFILE_SCHEMA_V2,
            "profile_id": self.profile_id,
            "status": self.status,
            "notes": self.notes,
            "airframe": self.airframe.as_dict(),
            "rotor": self.rotor.as_dict(),
            "battery": None if self.battery is None else self.battery.as_dict(),
            "parameters": {path: dict(entry) for path, entry in self.parameters.items()},
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON document, including its v2 schema identifier."""
        return self.as_document()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RotorcraftProfile":
        if not isinstance(raw, dict):
            raise ValueError("hardware profile must be an object")
        if raw.get("schema") != HARDWARE_PROFILE_SCHEMA_V2:
            raise ValueError(f"schema must be {HARDWARE_PROFILE_SCHEMA_V2}")
        allowed = {"schema", "profile_id", "status", "notes", "airframe", "rotor", "battery", "parameters"}
        _reject_unknown(raw, allowed, "hardware profile v2")
        _require_fields(raw, allowed - {"schema"}, "hardware profile v2")
        airframe_raw = raw["airframe"]
        rotor_raw = raw["rotor"]
        if not isinstance(airframe_raw, dict) or not isinstance(rotor_raw, dict):
            raise ValueError("airframe and rotor must be objects")
        _reject_unknown(airframe_raw, {"mass_kg", "inertia_kg_m2", "arm_length_m", "drag_coefficient", "cross_sectional_area_m2"}, "airframe")
        _require_fields(airframe_raw, {"mass_kg", "inertia_kg_m2", "arm_length_m", "drag_coefficient", "cross_sectional_area_m2"}, "airframe")
        _reject_unknown(rotor_raw, {"count", "layout", "thrust_coefficient_n_s2", "torque_to_thrust_m", "max_speed_rad_s", "motor_time_constant_s", "prop_radius_m", "spin_directions"}, "rotor")
        _require_fields(rotor_raw, {"count", "layout", "thrust_coefficient_n_s2", "torque_to_thrust_m", "max_speed_rad_s", "motor_time_constant_s", "prop_radius_m", "spin_directions"}, "rotor")
        battery_raw = raw["battery"]
        battery = None
        if battery_raw is not None:
            if not isinstance(battery_raw, dict):
                raise ValueError("battery must be an object or null")
            battery_fields = {"nominal_voltage_v", "cutoff_voltage_v", "capacity_mah", "internal_resistance_ohm", "efficiency"}
            _reject_unknown(battery_raw, battery_fields, "battery")
            _require_fields(battery_raw, battery_fields, "battery")
            battery = BatteryProfile(**battery_raw)
        parameters = raw["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        expected = set(_V2_NUMERIC_PATHS)
        if battery is None:
            expected -= {path for path in expected if path.startswith("battery.")}
        document_fields = {key: raw[key] for key in ("airframe", "rotor", "battery")}
        for path, entry in parameters.items():
            if path not in expected:
                raise ValueError(f"parameters contains unknown field path: {path}")
            if not isinstance(entry, dict):
                raise ValueError(f"parameters.{path} must be an object")
            _reject_unknown(entry, {"value", "range", "source", "url", "accessed", "status", "note"}, f"parameters.{path}")
            _require_fields(entry, {"value", "range", "source", "url", "accessed", "status"}, f"parameters.{path}")
            if not _nested_equal(entry["value"], _path_value(document_fields, path)):
                raise ValueError(f"parameters.{path}.value must equal the profile field")
            if not _range_contains(entry["value"], entry["range"]):
                raise ValueError(f"parameters.{path}.range must contain its value")
            for text_field in ("source", "url", "accessed", "status"):
                _required_text(entry[text_field], f"parameters.{path}.{text_field}")
            if len(entry["accessed"]) != 10 or entry["accessed"][4] != "-" or entry["accessed"][7] != "-":
                raise ValueError(f"parameters.{path}.accessed must use YYYY-MM-DD")
            if entry["status"] not in {"CONFIRMED", "CROSS-CHECKED", "SCALED", "UNCONFIRMED", "EXCLUDED"}:
                raise ValueError(f"parameters.{path}.status is invalid")
            if entry["status"] == "SCALED" and not _required_text(entry.get("note"), f"parameters.{path}.note"):
                raise ValueError(f"parameters.{path}.note is required for SCALED values")
            if entry["status"] == "EXCLUDED":
                # EXCLUDED means "no source found, so the effect is switched off".
                # Only body drag can be switched off this way: drag_coefficient
                # must be 0, which makes cross_sectional_area_m2 irrelevant.
                if path not in _V2_EXCLUDABLE_PATHS:
                    raise ValueError(f"parameters.{path} cannot be EXCLUDED; only body drag can be switched off")
                _required_text(entry.get("note"), f"parameters.{path}.note")
                if path == "airframe.drag_coefficient" and float(entry["value"]) != 0.0:
                    raise ValueError("an EXCLUDED airframe.drag_coefficient must be 0 (drag off)")
            if raw.get("status") == "literature-derived":
                if entry["status"] == "UNCONFIRMED":
                    raise ValueError(f"literature-derived profile uses UNCONFIRMED parameter: {path}")
        excluded_paths = {path for path, entry in parameters.items() if entry.get("status") == "EXCLUDED"}
        if "airframe.cross_sectional_area_m2" in excluded_paths and "airframe.drag_coefficient" not in excluded_paths:
            raise ValueError("cross_sectional_area_m2 can only be EXCLUDED when drag_coefficient is EXCLUDED too")
        if raw.get("status") == "literature-derived":
            missing = sorted(expected - set(parameters))
            if missing:
                raise ValueError("literature-derived profile missing parameters: " + ", ".join(missing))
        return cls(
            profile_id=raw["profile_id"], status=raw["status"], notes=raw["notes"],
            airframe=AirframeProfile(**airframe_raw), rotor=RotorProfile(**rotor_raw),
            battery=battery, parameters=parameters,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "RotorcraftProfile":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def load_hardware_profile(path: str | Path) -> HardwareProfile | RotorcraftProfile:
    """Load either a v1 :class:`HardwareProfile` or v2 rotorcraft profile."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("hardware profile must be an object")
    schema = raw.get("schema")
    if schema == HARDWARE_PROFILE_SCHEMA:
        return HardwareProfile.from_dict(raw)
    if schema == HARDWARE_PROFILE_SCHEMA_V2:
        return RotorcraftProfile.from_dict(raw)
    raise ValueError(f"unsupported hardware profile schema: {schema!r}")


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
