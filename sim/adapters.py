"""Typed adapter boundary and deterministic in-memory :class:`SimAdapter`.

The swarm layer talks to a vehicle through ``DroneAdapter`` and never reaches
into a vendor API.  ``SimAdapter`` is the first implementation: it integrates
a bounded point-mass response so authority, stale-link, and failsafe behavior
can be tested without a live radio or motor.  Its values are synthetic
scenario inputs and are not flight-performance claims.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from sim.authority import (
    AuthorityArbiter,
    AuthorityDecision,
    CommandKind,
    CommandSource,
    ControlCommand,
)
from sim.failsafe import FailsafeAction, FailsafeConfig, FailsafeController, FailsafeDecision, FailsafeState
from sim.hardware import HardwareProfile, default_hardware_profile


def _vector3(value: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class Capabilities:
    """Feature declaration used for adapter selection and UI disclosure."""

    has_camera: bool = False
    has_gps: bool = False
    indoor_only: bool = True
    max_speed_mps: float = 2.0
    supports_position_setpoint: bool = True
    supports_land: bool = True
    supports_kill: bool = True

    def __post_init__(self) -> None:
        speed = float(self.max_speed_mps)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("max_speed_mps must be finite and positive")
        object.__setattr__(self, "max_speed_mps", speed)
        for name in (
            "has_camera",
            "has_gps",
            "indoor_only",
            "supports_position_setpoint",
            "supports_land",
            "supports_kill",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_camera": self.has_camera,
            "has_gps": self.has_gps,
            "indoor_only": self.indoor_only,
            "max_speed_mps": self.max_speed_mps,
            "supports_position_setpoint": self.supports_position_setpoint,
            "supports_land": self.supports_land,
            "supports_kill": self.supports_kill,
        }


@dataclass(frozen=True)
class DroneState:
    """Normalized telemetry returned by every adapter.

    ``position`` and ``velocity`` are NumPy vectors in SI units.  Authority and
    failsafe fields are strings so a state serializes directly to JSON without
    importing simulator enums in a ground-station client.
    """

    position: np.ndarray
    velocity: np.ndarray
    battery_pct: float
    link_delay_ms: float
    link_loss_pct: float
    timestamp_s: float = 0.0
    connected: bool = False
    command_authority: str | None = None
    failsafe_state: str = FailsafeState.NOMINAL.value

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, name="position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, name="velocity"))
        for name, lower, upper in (
            ("battery_pct", 0.0, 100.0),
            ("link_loss_pct", 0.0, 100.0),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError(f"{name} must be finite and between {lower:g} and {upper:g}")
            object.__setattr__(self, name, value)
        delay = _finite_nonnegative("link_delay_ms", self.link_delay_ms)
        timestamp = _finite_nonnegative("timestamp_s", self.timestamp_s)
        object.__setattr__(self, "link_delay_ms", delay)
        object.__setattr__(self, "timestamp_s", timestamp)
        if not isinstance(self.connected, bool):
            raise ValueError("connected must be bool")
        if self.command_authority is not None and not isinstance(self.command_authority, str):
            raise ValueError("command_authority must be a string or None")
        if not isinstance(self.failsafe_state, str) or not self.failsafe_state:
            raise ValueError("failsafe_state must be a non-empty string")

    @property
    def position_m(self) -> np.ndarray:
        """Compatibility alias for code that uses explicit SI suffixes."""
        return self.position

    @property
    def velocity_mps(self) -> np.ndarray:
        """Compatibility alias for code that uses explicit SI suffixes."""
        return self.velocity

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "battery_pct": self.battery_pct,
            "link_delay_ms": self.link_delay_ms,
            "link_loss_pct": self.link_loss_pct,
            "timestamp_s": self.timestamp_s,
            "connected": self.connected,
            "command_authority": self.command_authority,
            "failsafe_state": self.failsafe_state,
        }


class DroneAdapter(ABC):
    """Small common contract for simulation and future hardware adapters."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Stable vehicle identity within one mission."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Declared features and limits of this adapter."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the adapter has an active transport/session."""

    @abstractmethod
    def connect(self) -> None:
        """Open the adapter session."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the adapter session."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Read the normalized latest state."""

    @abstractmethod
    def request_command(
        self,
        command: ControlCommand,
        *,
        source: CommandSource,
        ttl_s: float | None = 0.5,
        timestamp_s: float | None = None,
    ) -> AuthorityDecision:
        """Submit a typed command through the authority arbiter."""

    @abstractmethod
    def send_setpoint(self, position: np.ndarray) -> AuthorityDecision:
        """Request a mission position setpoint through authority arbitration."""

    @abstractmethod
    def set_link_metrics(self, *, delay_ms: float, loss_pct: float, age_s: float | None) -> None:
        """Inject normalized link telemetry for a synthetic or replay adapter."""

    @abstractmethod
    def set_battery_pct(self, battery_pct: float) -> None:
        """Inject a normalized battery reading for a synthetic or replay adapter."""

    @abstractmethod
    def step(
        self,
        dt_s: float | None = None,
        *,
        link_age_s: float | None = None,
        battery_pct: float | None = None,
        kill_requested: bool = False,
    ) -> DroneState:
        """Advance a deterministic simulation or poll a replay clock."""

    @abstractmethod
    def land(self) -> FailsafeDecision:
        """Request a latched landing failsafe."""

    @abstractmethod
    def kill(self) -> FailsafeDecision:
        """Request a latched motor-stop failsafe in the synthetic model."""

    @abstractmethod
    def reset_failsafe(self) -> FailsafeDecision:
        """Clear a latched failsafe after an explicit health check."""


class SimAdapter(DroneAdapter):
    """Deterministic point-mass adapter for one synthetic vehicle.

    The adapter is intentionally modest: a bounded acceleration response moves
    the vehicle toward the winning position command.  It provides the seam for
    testing command ownership and recovery before a measured actuator model
    or hardware transport exists.
    """

    def __init__(
        self,
        adapter_id: str,
        *,
        profile: HardwareProfile | None = None,
        capabilities: Capabilities | None = None,
        initial_position_m: Iterable[float] = (0.0, 0.0, 1.0),
        dt_s: float = 0.05,
        response_time_s: float = 0.25,
        max_acceleration_mps2: float = 4.0,
        ground_plane_z_m: float = 0.0,
        initial_battery_pct: float = 100.0,
        failsafe_config: FailsafeConfig | None = None,
    ) -> None:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be a non-empty string")
        self._adapter_id = adapter_id
        self.profile = profile or default_hardware_profile(adapter_id)
        if not isinstance(self.profile, HardwareProfile):
            raise ValueError("profile must be a HardwareProfile")
        self._capabilities = capabilities or Capabilities()
        if not isinstance(self._capabilities, Capabilities):
            raise ValueError("capabilities must be Capabilities")
        self.position = _vector3(initial_position_m, name="initial_position_m")
        self.velocity = np.zeros(3, dtype=float)
        self._dt_s = _finite_nonnegative("dt_s", dt_s)
        if self._dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self._response_time_s = float(response_time_s)
        if not math.isfinite(self._response_time_s) or self._response_time_s <= 0.0:
            raise ValueError("response_time_s must be finite and positive")
        requested_acceleration = float(max_acceleration_mps2)
        if not math.isfinite(requested_acceleration) or requested_acceleration <= 0.0:
            raise ValueError("max_acceleration_mps2 must be finite and positive")
        # The point-mass adapter does not integrate per-rotor RPM.  It still
        # respects the profile's declared static thrust and mass by capping
        # the synthetic acceleration envelope at T_max / m.  The full motor
        # and rotor coupling remains in ``sim.vehicle``.
        profile_acceleration = self.profile.estimated_max_thrust_n() / self.profile.total_mass_kg
        if not math.isfinite(profile_acceleration) or profile_acceleration <= 0.0:
            raise ValueError("profile must provide a finite positive thrust-to-mass limit")
        self._profile_acceleration_mps2 = profile_acceleration
        self._max_acceleration_mps2 = min(requested_acceleration, profile_acceleration)
        self._ground_plane_z_m = float(ground_plane_z_m)
        if not math.isfinite(self._ground_plane_z_m):
            raise ValueError("ground_plane_z_m must be finite")
        self._battery_pct = float(initial_battery_pct)
        if not math.isfinite(self._battery_pct) or self._battery_pct < 0.0 or self._battery_pct > 100.0:
            raise ValueError("initial_battery_pct must be finite and between zero and 100")

        self.authority = AuthorityArbiter()
        self.failsafe = FailsafeController(failsafe_config)
        self._connected = False
        self._time_s = 0.0
        self._link_delay_ms = 0.0
        self._link_loss_pct = 0.0
        self._link_age_s: float | None = 0.0
        self._landed = False
        self._last_failsafe = self.failsafe.last_decision

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def profile_acceleration_limit_mps2(self) -> float:
        """Synthetic ``T_max / mass`` cap derived from the typed profile."""
        return self._profile_acceleration_mps2

    @property
    def failsafe_state(self) -> FailsafeState:
        return self.failsafe.state

    @property
    def last_failsafe(self) -> FailsafeDecision:
        return self._last_failsafe

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        """Close the synthetic session; no hardware operation is implied."""
        self._connected = False

    def _rejected(self, reason: str) -> AuthorityDecision:
        return AuthorityDecision(
            accepted=False,
            source=None,
            command=None,
            reason=reason,
            evaluated_at_s=self._time_s,
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("adapter is disconnected")

    def set_link_metrics(self, *, delay_ms: float, loss_pct: float, age_s: float | None) -> None:
        """Set synthetic link telemetry used by the failsafe policy."""
        delay = _finite_nonnegative("delay_ms", delay_ms)
        loss = float(loss_pct)
        if not math.isfinite(loss) or loss < 0.0 or loss > 100.0:
            raise ValueError("loss_pct must be finite and between zero and 100")
        if age_s is not None:
            age_s = _finite_nonnegative("age_s", age_s)
        self._link_delay_ms = delay
        self._link_loss_pct = loss
        self._link_age_s = age_s

    def set_battery_pct(self, battery_pct: float) -> None:
        value = float(battery_pct)
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            raise ValueError("battery_pct must be finite and between zero and 100")
        self._battery_pct = value

    def request_command(
        self,
        command: ControlCommand,
        *,
        source: CommandSource,
        ttl_s: float | None = 0.5,
        timestamp_s: float | None = None,
    ) -> AuthorityDecision:
        """Submit a typed command and return the authority winner."""
        self._require_connected()
        if not isinstance(command, ControlCommand):
            raise ValueError("command must be a ControlCommand")
        try:
            normalized_source = source if isinstance(source, CommandSource) else CommandSource(source)
        except (TypeError, ValueError) as exc:
            raise ValueError("source must be a valid CommandSource") from exc
        if command.kind in {CommandKind.LAND, CommandKind.KILL} and normalized_source is not CommandSource.FAILSAFE:
            return self._rejected("safety_command_requires_failsafe")
        now = self._time_s if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(now) or now < self._time_s:
            raise ValueError("timestamp_s must be finite and nondecreasing")
        self._time_s = now
        return self.authority.submit(normalized_source, command, timestamp_s=now, ttl_s=ttl_s)

    def send_setpoint(self, position: np.ndarray) -> AuthorityDecision:
        if not self.capabilities.supports_position_setpoint:
            return self._rejected("position_setpoint_unsupported")
        return self.request_command(
            ControlCommand.position(position),
            source=CommandSource.MISSION,
            ttl_s=0.5,
        )

    def _sync_failsafe_authority(self, decision: FailsafeDecision) -> None:
        if decision.state is FailsafeState.NOMINAL:
            self.authority.release(CommandSource.FAILSAFE, timestamp_s=self._time_s)
            return
        # Degraded hold, landing, landed hold, and kill are all safety-owned.
        command = decision.command
        if decision.state is FailsafeState.LANDED:
            command = ControlCommand.hold()
        if command is None:
            raise RuntimeError(f"failsafe state {decision.state.value} has no safety command")
        self.authority.submit(
            CommandSource.FAILSAFE,
            command,
            timestamp_s=self._time_s,
            ttl_s=None,
            request_id=f"failsafe-{decision.state.value}",
        )

    def _observe_failsafe(self, *, kill_requested: bool = False, landed: bool = False) -> FailsafeDecision:
        decision = self.failsafe.observe(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
            kill_requested=kill_requested,
            landed=landed,
        )
        self._last_failsafe = decision
        self._sync_failsafe_authority(decision)
        return decision

    def land(self) -> FailsafeDecision:
        self._require_connected()
        if not self.capabilities.supports_land:
            raise RuntimeError("landing is unsupported by this adapter")
        decision = self.failsafe.force_land(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        self._last_failsafe = decision
        self._sync_failsafe_authority(decision)
        return decision

    def kill(self) -> FailsafeDecision:
        self._require_connected()
        if not self.capabilities.supports_kill:
            raise RuntimeError("kill is unsupported by this adapter")
        decision = self.failsafe.force_kill(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        self._last_failsafe = decision
        self._sync_failsafe_authority(decision)
        return decision

    def reset_failsafe(self) -> FailsafeDecision:
        """Explicitly clear a latched failsafe after healthy synthetic checks."""
        self._require_connected()
        decision = self.failsafe.reset(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        self._landed = False
        self._last_failsafe = decision
        self._sync_failsafe_authority(decision)
        return decision

    @staticmethod
    def _limited(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm == 0.0:
            return vector
        return vector * (limit / norm)

    def step(
        self,
        dt_s: float | None = None,
        *,
        link_age_s: float | None = None,
        battery_pct: float | None = None,
        kill_requested: bool = False,
    ) -> DroneState:
        """Advance the synthetic adapter and return normalized state."""
        self._require_connected()
        dt = self._dt_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if link_age_s is not None:
            self._link_age_s = _finite_nonnegative("link_age_s", link_age_s)
        if battery_pct is not None:
            self.set_battery_pct(battery_pct)
        self._time_s += dt
        decision = self._observe_failsafe(kill_requested=kill_requested, landed=self._landed)
        winner = self.authority.select(self._time_s)

        if winner.command is None or winner.command.kind is CommandKind.KILL:
            self.velocity[:] = 0.0
        else:
            command = winner.command
            if command.kind is CommandKind.POSITION:
                assert command.position_m is not None
                desired = np.asarray(command.position_m, dtype=float)
            elif command.kind is CommandKind.LAND:
                desired = self.position.copy()
                desired[2] = self._ground_plane_z_m
            else:  # HOLD
                desired = self.position.copy()
            desired_velocity = (desired - self.position) / self._response_time_s
            desired_velocity = self._limited(desired_velocity, self.capabilities.max_speed_mps)
            acceleration = self._limited(
                (desired_velocity - self.velocity) / dt,
                self._max_acceleration_mps2,
            )
            self.velocity = self._limited(self.velocity + acceleration * dt, self.capabilities.max_speed_mps)
            self.position = self.position + self.velocity * dt
            if self.position[2] < self._ground_plane_z_m:
                self.position[2] = self._ground_plane_z_m
                self.velocity[2] = max(0.0, self.velocity[2])
            if command.kind is CommandKind.LAND and (
                self.position[2] <= self._ground_plane_z_m + 1e-9
                and np.linalg.norm(self.velocity) <= 1e-6
            ):
                self.position[2] = self._ground_plane_z_m
                self._landed = True
                decision = self._observe_failsafe(landed=True)

        # A killed vehicle is stopped immediately.  A landed one remains held.
        if decision.action is FailsafeAction.KILL:
            self.velocity[:] = 0.0
        return self.get_state()

    def get_state(self) -> DroneState:
        """Return a normalized, serializable state snapshot."""
        authority = self.authority.select(self._time_s)
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            battery_pct=self._battery_pct,
            link_delay_ms=self._link_delay_ms,
            link_loss_pct=self._link_loss_pct,
            timestamp_s=self._time_s,
            connected=self._connected,
            command_authority=None if authority.source is None else authority.source.value,
            failsafe_state=self.failsafe.state.value,
        )
