"""Typed adapter boundary and deterministic in-memory :class:`SimAdapter`.

The swarm layer talks to a vehicle through ``DroneAdapter`` and never reaches
into a vendor API.  ``SimAdapter`` is the first implementation: it integrates
a bounded point-mass response so authority, stale-link, and failsafe behavior
can be tested without a live radio or motor.  Its values are synthetic
scenario inputs and are not flight-performance claims.

``SimulatedAdapterCore`` holds the authority, failsafe, geofence latch, link
and event rules shared by every simulated adapter (this module's point-mass
adapters and ``sim.missions.physics_adapter.PhysicsAdapter``).  Everything here
is simulated.
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
class Geofence:
    """Axis-aligned simulated flight volume."""

    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]
    margin_m: float = 0.5

    def __post_init__(self) -> None:
        lo = _vector3(self.min_m, name="min_m")
        hi = _vector3(self.max_m, name="max_m")
        margin = float(self.margin_m)
        if not math.isfinite(margin) or margin < 0.0 or np.any(lo >= hi - 2.0 * margin):
            raise ValueError("geofence requires min_m < max_m - 2*margin_m on every axis")
        object.__setattr__(self, "min_m", tuple(float(v) for v in lo))
        object.__setattr__(self, "max_m", tuple(float(v) for v in hi))
        object.__setattr__(self, "margin_m", margin)

    def contains(self, p: Iterable[float]) -> bool:
        value = _vector3(p, name="p")
        return bool(np.all(value >= np.asarray(self.min_m)) and np.all(value <= np.asarray(self.max_m)))

    def clip(self, p: Iterable[float]) -> np.ndarray:
        value = _vector3(p, name="p")
        return np.clip(value, np.asarray(self.min_m) + self.margin_m, np.asarray(self.max_m) - self.margin_m)

    def as_dict(self) -> dict[str, Any]:
        return {"min_m": list(self.min_m), "max_m": list(self.max_m), "margin_m": self.margin_m}


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
    link_age_s: float | None = None
    landed: bool = False

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
        if self.link_age_s is not None:
            object.__setattr__(self, "link_age_s", _finite_nonnegative("link_age_s", self.link_age_s))
        if not isinstance(self.landed, bool):
            raise ValueError("landed must be bool")

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

    def as_mission_dict(self) -> dict[str, Any]:
        result = self.as_dict()
        result.update({"link_age_s": self.link_age_s, "landed": self.landed})
        return result


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
        recent_loss_fraction: float | None = None,
    ) -> DroneState:
        """Advance a deterministic simulation or poll a replay clock."""

    @abstractmethod
    def land(self) -> FailsafeDecision:
        """Request a latched landing failsafe."""

    @abstractmethod
    def kill(self) -> FailsafeDecision:
        """Request a latched motor-stop failsafe in the synthetic model.

        In ``SimAdapter`` this only zeroes velocity: the point mass has no
        gravity, so a killed vehicle stays where it was instead of falling.
        """

    @abstractmethod
    def reset_failsafe(self) -> FailsafeDecision:
        """Clear a latched failsafe after an explicit health check."""

    def drain_events(self) -> list[dict[str, Any]]:
        return []

    @property
    def time_s(self) -> float:
        raise NotImplementedError

    @property
    def geofence(self) -> Geofence | None:
        return None


# A body this close to the floor (or below it) counts as touching it.
FLOOR_CONTACT_M = 1e-9


def _limited(vector: np.ndarray, limit: float) -> np.ndarray:
    """Return ``vector`` scaled down so its length is at most ``limit``."""
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector
    return vector * (limit / norm)


class PositionFixHold:
    """Newest externally supplied position fix, held between fixes.

    Used by the simulated offboard adapters, where a laptop closes the
    position loop from camera fixes.  Before the first fix arrives the held
    fix is the initial position with zero velocity.  The velocity estimate is
    the difference of the last two fixes over their time gap (zero until two
    fixes with different timestamps have arrived).  A fix with the same
    timestamp as the previous one replaces the position and keeps the old
    velocity estimate.
    """

    def __init__(self, initial_position_m: Iterable[float]) -> None:
        self.position = _vector3(initial_position_m, name="initial_position_m")
        self.velocity = np.zeros(3, dtype=float)
        self.time_s: float | None = None

    def update(self, position_m: Iterable[float], timestamp_s: float) -> None:
        stamp = _finite_nonnegative("timestamp_s", timestamp_s)
        if self.time_s is not None and stamp < self.time_s:
            raise ValueError("timestamp_s must be nondecreasing")
        point = _vector3(position_m, name="position_m")
        if self.time_s is not None and stamp > self.time_s:
            self.velocity = (point - self.position) / (stamp - self.time_s)
        self.position = point
        self.time_s = stamp


class SimulatedAdapterCore(DroneAdapter):
    """Authority, failsafe, geofence, link and event plumbing shared by the simulated adapters.

    ``SimAdapter`` (point mass) and ``PhysicsAdapter`` (rigid body) both
    inherit this class so the safety rules are written once:

    * LAND and KILL commands are accepted only from the FAILSAFE source.
    * Failsafe states map to a FAILSAFE-source request with no expiry: HOLD
      for a degraded link (unless the policy says SLOW), LAND while landing,
      HOLD once landed, KILL after the kill switch.
    * Geofence breach latch: the first step whose TRUE position is outside
      ``geofence.contains`` (the box itself, faces inclusive; the margin only
      shrinks the clip box) records one "geofence_breach" event and latches a
      FAILSAFE-source HOLD.  The check runs on the position after the step's
      motion, so the step that crosses a face already reports the breach and
      the safety authority; it also runs before the motion, which catches a
      vehicle that starts a step outside.  The latch survives the vehicle coming back inside
      and is released only by :meth:`reset_failsafe`.  A landing or kill
      switch request still outranks the latched hold.  While latched, no
      further breach events are recorded; a breach after a reset is a new
      episode with a new event.
    * Events are recorded with the adapter clock in ``drain_events`` order.

    Subclasses provide the motion model through ``step`` and ``get_state``
    and report their true position through ``_truth_position``.  Everything
    here is simulated; no radio or flight controller is involved.
    """

    def _init_core(
        self,
        adapter_id: str,
        *,
        capabilities: Capabilities,
        dt_s: float,
        initial_battery_pct: float,
        failsafe_config: FailsafeConfig | None,
        geofence: Geofence | None,
    ) -> None:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be a non-empty string")
        if not isinstance(capabilities, Capabilities):
            raise ValueError("capabilities must be Capabilities")
        if geofence is not None and not isinstance(geofence, Geofence):
            raise ValueError("geofence must be Geofence or None")
        if failsafe_config is not None and not isinstance(failsafe_config, FailsafeConfig):
            raise ValueError("failsafe_config must be FailsafeConfig or None")
        dt = _finite_nonnegative("dt_s", dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        battery = float(initial_battery_pct)
        if not math.isfinite(battery) or battery < 0.0 or battery > 100.0:
            raise ValueError("initial_battery_pct must be finite and between zero and 100")
        self._adapter_id = adapter_id
        self._capabilities = capabilities
        self._geofence = geofence
        self._dt_s = dt
        self._battery_pct = battery
        self.authority = AuthorityArbiter()
        self.failsafe = FailsafeController(failsafe_config)
        self._connected = False
        self._time_s = 0.0
        self._link_delay_ms = 0.0
        self._link_loss_pct = 0.0
        self._link_age_s: float | None = 0.0
        self._landed = False
        self._geofence_latched = False
        self._last_failsafe = self.failsafe.last_decision
        self._events: list[dict[str, Any]] = []
        self._last_authority_source: str | None = None

    # Identity and session ---------------------------------------------------

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
    def geofence(self) -> Geofence | None:
        return self._geofence

    @property
    def geofence_latched(self) -> bool:
        """True while a geofence breach hold is latched (until ``reset_failsafe``)."""
        return self._geofence_latched

    @property
    def failsafe_state(self) -> FailsafeState:
        return self.failsafe.state

    @property
    def last_failsafe(self) -> FailsafeDecision:
        return self._last_failsafe

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        """Close the simulated session; no hardware operation is implied."""
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("adapter is disconnected")

    def _rejected(self, reason: str) -> AuthorityDecision:
        return AuthorityDecision(
            accepted=False,
            source=None,
            command=None,
            reason=reason,
            evaluated_at_s=self._time_s,
        )

    # Events -------------------------------------------------------------------

    def drain_events(self) -> list[dict[str, Any]]:
        events = self._events
        self._events = []
        return events

    def _event(self, kind: str, detail: dict[str, Any] | None = None) -> None:
        self._events.append({"time_s": float(self._time_s), "kind": kind, "agent": self._adapter_id, "detail": detail or {}})

    def _record_authority(self, winner: AuthorityDecision) -> None:
        source = None if winner.source is None else winner.source.value
        if source != self._last_authority_source:
            self._last_authority_source = source
            self._event("authority_change", {"source": source})

    # Injected telemetry -------------------------------------------------------

    def set_link_metrics(self, *, delay_ms: float, loss_pct: float, age_s: float | None) -> None:
        """Set simulated link telemetry used by the failsafe policy."""
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

    # Commands -----------------------------------------------------------------

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
        if command.kind is CommandKind.POSITION and self._geofence is not None:
            # Every POSITION command meets the geofence here, whichever path it
            # took in.  A command already inside the clip box is left alone, so
            # it is never clipped (or logged) twice.
            requested = np.asarray(command.position_m, dtype=float)
            clipped = self._geofence.clip(requested)
            if not np.array_equal(clipped, requested):
                self._event("geofence_clip", {"requested_m": requested.tolist(), "clipped_m": clipped.tolist()})
                command = ControlCommand.position(clipped)
        return self.authority.submit(normalized_source, command, timestamp_s=now, ttl_s=ttl_s)

    def send_setpoint(self, position: np.ndarray) -> AuthorityDecision:
        if not self.capabilities.supports_position_setpoint:
            return self._rejected("position_setpoint_unsupported")
        point = _vector3(position, name="position")
        return self.request_command(
            ControlCommand.position(point),
            source=CommandSource.MISSION,
            ttl_s=0.5,
        )

    # Failsafe and authority ---------------------------------------------------

    def _sync_failsafe_authority(self, decision: FailsafeDecision) -> None:
        """Publish (or release) the FAILSAFE-source request for ``decision``."""
        command: ControlCommand | None
        if decision.state is FailsafeState.LANDED:
            command = ControlCommand.hold()
            request_id = "failsafe-landed"
        elif decision.action in {FailsafeAction.NONE, FailsafeAction.SLOW}:
            # Nominal, or SLOW: the mission keeps authority.
            command = None
            request_id = ""
        else:
            # Degraded hold, landing, and kill are safety-owned.
            command = decision.command
            request_id = f"failsafe-{decision.state.value}"
            if command is None:
                raise RuntimeError(f"failsafe state {decision.state.value} has no safety command")
        if command is None and self._geofence_latched:
            command = ControlCommand.hold()
            request_id = "geofence-breach"
        if command is None:
            self.authority.release(CommandSource.FAILSAFE, timestamp_s=self._time_s)
            return
        self.authority.submit(
            CommandSource.FAILSAFE,
            command,
            timestamp_s=self._time_s,
            ttl_s=None,
            request_id=request_id,
        )

    def _apply_failsafe_decision(self, decision: FailsafeDecision) -> FailsafeDecision:
        self._last_failsafe = decision
        if decision.changed:
            self._event("failsafe_transition", decision.as_dict())
            if decision.state is FailsafeState.KILLED:
                self._event("kill", {})
                self._on_kill()
        self._sync_failsafe_authority(decision)
        return decision

    def _on_kill(self) -> None:
        """Called once when the kill switch latches; models with motors stop them here."""

    def _observe_failsafe(self, *, kill_requested: bool = False, recent_loss_fraction: float | None = None) -> FailsafeDecision:
        decision = self.failsafe.observe(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
            kill_requested=kill_requested,
            landed=self._landed,
            recent_loss_fraction=recent_loss_fraction,
        )
        return self._apply_failsafe_decision(decision)

    def land(self) -> FailsafeDecision:
        self._require_connected()
        if not self.capabilities.supports_land:
            raise RuntimeError("landing is unsupported by this adapter")
        decision = self.failsafe.force_land(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        return self._apply_failsafe_decision(decision)

    def kill(self) -> FailsafeDecision:
        self._require_connected()
        if not self.capabilities.supports_kill:
            raise RuntimeError("kill is unsupported by this adapter")
        decision = self.failsafe.force_kill(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        return self._apply_failsafe_decision(decision)

    def reset_failsafe(self) -> FailsafeDecision:
        """Explicitly clear a latched failsafe or geofence hold after healthy checks."""
        self._require_connected()
        decision = self.failsafe.reset(
            self._time_s,
            link_age_s=self._link_age_s,
            battery_pct=self._battery_pct,
        )
        self._landed = False
        self._geofence_latched = False
        return self._apply_failsafe_decision(decision)

    # Step plumbing ------------------------------------------------------------

    def _truth_position(self) -> np.ndarray:
        """True simulated position used for geofence and floor checks."""
        raise NotImplementedError

    def _check_geofence(self) -> bool:
        """Latch the breach hold if the true position is outside; True when it just latched."""
        if self._geofence is None or self._geofence_latched:
            return False
        truth = self._truth_position()
        if not self._geofence.contains(truth):
            self._geofence_latched = True
            self._event("geofence_breach", {"position_m": truth.tolist()})
            return True
        return False

    def _end_step(self) -> None:
        """Check the geofence on the position this step's motion produced.

        A breach found here publishes the latched hold at once, so the state
        returned for the crossing step already shows the safety authority.
        """
        if self._check_geofence():
            self._sync_failsafe_authority(self._last_failsafe)
            self._record_authority(self.authority.select(self._time_s))

    def _begin_step(
        self,
        dt_s: float | None,
        *,
        link_age_s: float | None,
        battery_pct: float | None,
        kill_requested: bool,
        recent_loss_fraction: float | None,
    ) -> tuple[float, FailsafeDecision, AuthorityDecision]:
        """Advance the clock, evaluate safety, and select this step's command.

        The geofence check runs before the authority selection, so a vehicle
        that starts the step outside is held in this same step.  Subclasses
        call ``_end_step`` after their motion for the crossing check.
        """
        self._require_connected()
        dt = self._dt_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if link_age_s is not None:
            self._link_age_s = _finite_nonnegative("link_age_s", link_age_s)
        if battery_pct is not None:
            self.set_battery_pct(battery_pct)
        self._time_s += dt
        self._check_geofence()
        decision = self._observe_failsafe(kill_requested=kill_requested, recent_loss_fraction=recent_loss_fraction)
        winner = self.authority.select(self._time_s)
        self._record_authority(winner)
        return dt, decision, winner

    def _mark_landed(self, recent_loss_fraction: float | None) -> FailsafeDecision:
        """Acknowledge touchdown to the failsafe policy (LANDING to LANDED)."""
        self._landed = True
        self._event("landed", {})
        return self._observe_failsafe(recent_loss_fraction=recent_loss_fraction)

    def _state(self, position: np.ndarray, velocity: np.ndarray) -> DroneState:
        authority = self.authority.select(self._time_s)
        return DroneState(
            position=position.copy(),
            velocity=velocity.copy(),
            battery_pct=self._battery_pct,
            link_delay_ms=self._link_delay_ms,
            link_loss_pct=self._link_loss_pct,
            timestamp_s=self._time_s,
            connected=self._connected,
            command_authority=None if authority.source is None else authority.source.value,
            failsafe_state=self.failsafe.state.value,
            link_age_s=self._link_age_s,
            landed=self._landed,
        )


class SimAdapter(SimulatedAdapterCore):
    """Deterministic point-mass adapter for one simulated vehicle.

    The adapter is intentionally modest: a bounded acceleration response moves
    the vehicle toward the winning position command.  It provides the seam for
    testing command ownership and recovery before a measured actuator model
    or hardware transport exists.  There is no gravity or ground contact
    model beyond a floor clamp, so a killed (motor-stopped) vehicle freezes
    where it is rather than falling.  The floor does not let the vehicle
    slide: a step that ends on the floor without climbing stops the vehicle,
    and one that also started on the floor keeps its horizontal position.
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
        geofence: Geofence | None = None,
    ) -> None:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be a non-empty string")
        self.profile = profile or default_hardware_profile(adapter_id)
        if not isinstance(self.profile, HardwareProfile):
            raise ValueError("profile must be a HardwareProfile")
        self._init_core(
            adapter_id,
            capabilities=capabilities or Capabilities(),
            dt_s=dt_s,
            initial_battery_pct=initial_battery_pct,
            failsafe_config=failsafe_config,
            geofence=geofence,
        )
        self.position = _vector3(initial_position_m, name="initial_position_m")
        self.velocity = np.zeros(3, dtype=float)
        self._response_time_s = float(response_time_s)
        if not math.isfinite(self._response_time_s) or self._response_time_s <= 0.0:
            raise ValueError("response_time_s must be finite and positive")
        requested_acceleration = float(max_acceleration_mps2)
        if not math.isfinite(requested_acceleration) or requested_acceleration <= 0.0:
            raise ValueError("max_acceleration_mps2 must be finite and positive")
        # The point-mass adapter does not integrate per-rotor RPM.  It still
        # respects the profile's declared static thrust and mass by capping
        # the simulated acceleration envelope at T_max / m.  The full motor
        # and rotor coupling remains in ``sim.vehicle``.
        profile_acceleration = self.profile.estimated_max_thrust_n() / self.profile.total_mass_kg
        if not math.isfinite(profile_acceleration) or profile_acceleration <= 0.0:
            raise ValueError("profile must provide a finite positive thrust-to-mass limit")
        self._profile_acceleration_mps2 = profile_acceleration
        self._max_acceleration_mps2 = min(requested_acceleration, profile_acceleration)
        self._ground_plane_z_m = float(ground_plane_z_m)
        if not math.isfinite(self._ground_plane_z_m):
            raise ValueError("ground_plane_z_m must be finite")

    @property
    def profile_acceleration_limit_mps2(self) -> float:
        """Simulated ``T_max / mass`` cap derived from the typed profile."""
        return self._profile_acceleration_mps2

    _limited = staticmethod(_limited)

    def _truth_position(self) -> np.ndarray:
        return self.position

    def _sensed_position(self) -> np.ndarray:
        """Position the motion law steers from.  Onboard, this is the truth."""
        return self.position

    def _move_point_mass(self, command: ControlCommand, *, sensed: np.ndarray, speed_limit: float, dt: float) -> None:
        """Point-mass motion law.

        The velocity request comes from the error between the goal and the
        ``sensed`` position; the TRUE position integrates the resulting
        velocity and is clamped to the floor.  With ``sensed`` equal to the
        truth this is the original ``SimAdapter`` law away from the floor.

        Floor contact: if the step ends at or below the floor and the vehicle
        is not climbing, it stops (full velocity zeroed).  If it was already
        touching the floor when the step began, it also keeps its horizontal
        position, so a vehicle on the floor never slides along it.
        """
        start = self.position.copy()
        if command.kind is CommandKind.POSITION:
            assert command.position_m is not None
            desired = np.asarray(command.position_m, dtype=float)
        elif command.kind is CommandKind.LAND:
            desired = sensed.copy()
            desired[2] = self._ground_plane_z_m
        else:  # HOLD
            desired = sensed.copy()
        desired_velocity = (desired - sensed) / self._response_time_s
        desired_velocity = _limited(desired_velocity, speed_limit)
        acceleration = _limited(
            (desired_velocity - self.velocity) / dt,
            self._max_acceleration_mps2,
        )
        self.velocity = _limited(self.velocity + acceleration * dt, speed_limit)
        self.position = self.position + self.velocity * dt
        if self.position[2] < self._ground_plane_z_m:
            self.position[2] = self._ground_plane_z_m
            self.velocity[2] = max(0.0, self.velocity[2])
        if self.position[2] <= self._ground_plane_z_m + FLOOR_CONTACT_M and self.velocity[2] <= 0.0:
            self.velocity[:] = 0.0
            if start[2] <= self._ground_plane_z_m + FLOOR_CONTACT_M:
                self.position[:2] = start[:2]

    def step(
        self,
        dt_s: float | None = None,
        *,
        link_age_s: float | None = None,
        battery_pct: float | None = None,
        kill_requested: bool = False,
        recent_loss_fraction: float | None = None,
    ) -> DroneState:
        """Advance the simulated adapter and return normalized state."""
        dt, decision, winner = self._begin_step(
            dt_s,
            link_age_s=link_age_s,
            battery_pct=battery_pct,
            kill_requested=kill_requested,
            recent_loss_fraction=recent_loss_fraction,
        )
        command = winner.command
        if command is None or command.kind is CommandKind.KILL:
            self.velocity[:] = 0.0
        else:
            speed_limit = self.capabilities.max_speed_mps
            if decision.action is FailsafeAction.SLOW:
                speed_limit = min(speed_limit, self.failsafe.config.slow_speed_mps)
            # A landing is the vehicle's own failsafe: it steers from the
            # vehicle's own state, never from a held offboard fix.
            sensed = self.position if command.kind is CommandKind.LAND else self._sensed_position()
            self._move_point_mass(command, sensed=sensed, speed_limit=speed_limit, dt=dt)
            # Touchdown is judged on the true position and velocity.
            if command.kind is CommandKind.LAND and (
                self.position[2] <= self._ground_plane_z_m + 1e-9
                and np.linalg.norm(self.velocity) <= 1e-6
            ):
                self.position[2] = self._ground_plane_z_m
                decision = self._mark_landed(recent_loss_fraction)

        # A killed vehicle is stopped immediately and frozen in place (this
        # point-mass model has no gravity, so it does not fall).  A landed one
        # remains held.
        if decision.action is FailsafeAction.KILL:
            self.velocity[:] = 0.0
        self._end_step()
        return self.get_state()

    def get_state(self) -> DroneState:
        """Return a normalized, serializable state snapshot."""
        return self._state(self.position, self.velocity)


class OffboardPointMassAdapter(SimAdapter):
    """Simulated point mass whose position loop uses externally supplied fixes.

    This is the "laptop closes the position loop from camera fixes" case.
    The motion law steers from the newest fix given to
    :meth:`set_position_fix` (held between fixes; the initial position before
    the first fix).  The true position still integrates the commanded
    velocity, sits on the floor clamp, and is what geofence checks, touchdown
    detection and ``get_state()`` use.  A LAND failsafe steers from the true
    state: the landing is the vehicle's own, and the fixes usually stop for
    the same reason the landing started.
    """

    def __init__(self, adapter_id: str, **kwargs: Any) -> None:
        super().__init__(adapter_id, **kwargs)
        self._fix = PositionFixHold(self.position)

    @property
    def position_fix_m(self) -> np.ndarray:
        """Copy of the fix the motion law currently steers from."""
        return self._fix.position.copy()

    def set_position_fix(self, position_m: Iterable[float], timestamp_s: float) -> None:
        self._fix.update(position_m, timestamp_s)

    def _sensed_position(self) -> np.ndarray:
        return self._fix.position.copy()
