"""Actuated vehicle adapter for the simulated mission seam."""
from __future__ import annotations

from pathlib import Path
import math
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from sim.adapters import Capabilities, DroneAdapter, DroneState, Geofence, _finite_nonnegative, _vector3
from sim.authority import AuthorityArbiter, AuthorityDecision, CommandKind, CommandSource, ControlCommand
from sim.failsafe import FailsafeAction, FailsafeConfig, FailsafeController, FailsafeDecision, FailsafeState
from sim.hardware import RotorcraftProfile, load_hardware_profile
from sim.vehicle import ActuatedQuadrotor


class _FeedbackController:
    def __init__(self, inner: Any, adapter: "PhysicsAdapter") -> None:
        self.inner = inner
        self.adapter = adapter

    def update_control(self, drone: Any, target: np.ndarray, dt: float, **kwargs: Any):
        fix = self.adapter._position_fix
        if self.adapter.feedback == "offboard" and fix is not None:
            proxy = SimpleNamespace(
                position=fix.copy(),
                velocity=self.adapter._fix_velocity.copy(),
                quaternion=drone.quaternion,
                angular_velocity=drone.angular_velocity,
                mass=drone.mass,
                I_x=drone.I_x,
                I_y=drone.I_y,
                I_z=drone.I_z,
            )
            return self.inner.update_control(proxy, target, dt, **kwargs)
        return self.inner.update_control(drone, target, dt, **kwargs)


class PhysicsAdapter(DroneAdapter):
    """Simulated v2 vehicle using a literature-derived profile, drag off, and a modelled floor.

    This adapter is simulated only and does not represent flight validation.
    """

    def __init__(self, adapter_id: str, *, profile: str | Path | RotorcraftProfile | None = None,
                 capabilities: Capabilities | None = None, initial_position_m: Iterable[float] = (0.0, 0.0, 1.0),
                 dt_s: float = 0.01, ground_plane_z_m: float = 0.0, initial_battery_pct: float = 100.0,
                 failsafe_config: FailsafeConfig | None = None, geofence: Geofence | None = None,
                 feedback: str = "onboard", land_speed_mps: float = 0.3) -> None:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be a non-empty string")
        if feedback not in {"onboard", "offboard"}:
            raise ValueError("feedback must be 'onboard' or 'offboard'")
        if profile is None:
            profile = Path(__file__).resolve().parents[1] / "physics" / "profiles" / "crazyflie_class.json"
        loaded = load_hardware_profile(profile) if isinstance(profile, (str, Path)) else profile
        if not isinstance(loaded, RotorcraftProfile):
            raise ValueError("profile must be a v2 RotorcraftProfile")
        if not math.isfinite(float(land_speed_mps)) or float(land_speed_mps) <= 0:
            raise ValueError("land_speed_mps must be finite and positive")
        self._adapter_id = adapter_id
        self.profile = loaded
        self._capabilities = capabilities or Capabilities(max_speed_mps=1.5)
        self._dt_s = _finite_nonnegative("dt_s", dt_s)
        if self._dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self._ground = float(ground_plane_z_m)
        self._battery_pct = float(initial_battery_pct)
        if not math.isfinite(self._battery_pct) or not 0 <= self._battery_pct <= 100:
            raise ValueError("initial_battery_pct must be between zero and 100")
        self.feedback = feedback
        self._land_speed = float(land_speed_mps)
        self._geofence = geofence
        self.vehicle = ActuatedQuadrotor(loaded, initial_position=initial_position_m, substeps=1)
        self.vehicle.actuator.speed_rad_s[:] = loaded.hover_speed_rad_s()
        self.vehicle.controller = _FeedbackController(self.vehicle.controller, self)
        self.authority = AuthorityArbiter()
        self.failsafe = FailsafeController(failsafe_config)
        self._connected = False
        self._time_s = 0.0
        self._link_age_s: float | None = 0.0
        self._link_delay_ms = 0.0
        self._link_loss_pct = 0.0
        self._landed = False
        self._carrot = _vector3(initial_position_m, name="initial_position_m")
        self._hold_carrot = self._carrot.copy()
        self._position_fix: np.ndarray | None = None
        self._fix_velocity = np.zeros(3)
        self._fix_time_s: float | None = None
        self._events: list[dict[str, Any]] = []
        self._last_authority: str | None = None
        self._last_command_kind: CommandKind | None = None
        self._last_failsafe = self.failsafe.last_decision

    @property
    def adapter_id(self) -> str: return self._adapter_id
    @property
    def capabilities(self) -> Capabilities: return self._capabilities
    @property
    def connected(self) -> bool: return self._connected
    @property
    def time_s(self) -> float: return self._time_s
    @property
    def geofence(self) -> Geofence | None: return self._geofence

    def drain_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events

    def _event(self, kind: str, detail: dict[str, Any] | None = None) -> None:
        self._events.append({"time_s": self._time_s, "kind": kind, "agent": self._adapter_id, "detail": detail or {}})

    def connect(self) -> None: self._connected = True
    def disconnect(self) -> None: self._connected = False
    def _require(self) -> None:
        if not self._connected: raise RuntimeError("adapter is disconnected")

    def set_position_fix(self, position_m: Iterable[float], timestamp_s: float) -> None:
        stamp = _finite_nonnegative("timestamp_s", timestamp_s)
        if self._fix_time_s is not None and stamp < self._fix_time_s: raise ValueError("timestamp_s must be nondecreasing")
        point = _vector3(position_m, name="position_m")
        if self._position_fix is not None and stamp > self._fix_time_s:
            self._fix_velocity = (point - self._position_fix) / (stamp - self._fix_time_s)
        self._position_fix, self._fix_time_s = point, stamp

    def set_link_metrics(self, *, delay_ms: float, loss_pct: float, age_s: float | None) -> None:
        self._link_delay_ms = _finite_nonnegative("delay_ms", delay_ms)
        self._link_loss_pct = float(loss_pct)
        if not 0 <= self._link_loss_pct <= 100 or not math.isfinite(self._link_loss_pct): raise ValueError("loss_pct out of range")
        self._link_age_s = None if age_s is None else _finite_nonnegative("age_s", age_s)
    def set_battery_pct(self, battery_pct: float) -> None:
        value = float(battery_pct)
        if not math.isfinite(value) or not 0 <= value <= 100: raise ValueError("battery_pct out of range")
        self._battery_pct = value

    def request_command(self, command: ControlCommand, *, source: CommandSource, ttl_s: float | None = 0.5,
                        timestamp_s: float | None = None) -> AuthorityDecision:
        self._require()
        if not isinstance(command, ControlCommand): raise ValueError("command must be a ControlCommand")
        src = source if isinstance(source, CommandSource) else CommandSource(source)
        if command.kind in {CommandKind.LAND, CommandKind.KILL} and src is not CommandSource.FAILSAFE:
            return AuthorityDecision(False, None, None, "safety_command_requires_failsafe", self._time_s)
        now = self._time_s if timestamp_s is None else float(timestamp_s)
        if now < self._time_s or not math.isfinite(now): raise ValueError("timestamp_s must be nondecreasing")
        self._time_s = now
        return self.authority.submit(src, command, timestamp_s=now, ttl_s=ttl_s)

    def send_setpoint(self, position: np.ndarray) -> AuthorityDecision:
        if not self.capabilities.supports_position_setpoint:
            return AuthorityDecision(False, None, None, "position_setpoint_unsupported", self._time_s)
        point = _vector3(position, name="position")
        if self._geofence is not None:
            clipped = self._geofence.clip(point)
            if not np.array_equal(point, clipped): self._event("geofence_clip", {"requested_m": point.tolist(), "clipped_m": clipped.tolist()})
            point = clipped
        return self.request_command(ControlCommand.position(point), source=CommandSource.MISSION, ttl_s=0.5)

    def _sync(self, decision: FailsafeDecision) -> None:
        if decision.state is FailsafeState.NOMINAL or decision.action is FailsafeAction.SLOW:
            self.authority.release(CommandSource.FAILSAFE, timestamp_s=self._time_s); return
        command = ControlCommand.hold() if decision.state is FailsafeState.LANDED else decision.command
        if command is not None:
            self.authority.submit(CommandSource.FAILSAFE, command, timestamp_s=self._time_s, ttl_s=None, request_id=f"failsafe-{decision.state.value}")

    def _observe(self, *, kill_requested=False, recent_loss_fraction=None) -> FailsafeDecision:
        decision = self.failsafe.observe(self._time_s, link_age_s=self._link_age_s, battery_pct=self._battery_pct,
                                         kill_requested=kill_requested, landed=self._landed,
                                         recent_loss_fraction=recent_loss_fraction)
        if decision.changed:
            self._event("failsafe_transition", decision.as_dict())
            if decision.state is FailsafeState.KILLED: self._event("kill", {})
        self._last_failsafe = decision; self._sync(decision); return decision

    def land(self) -> FailsafeDecision:
        self._require()
        if not self.capabilities.supports_land: raise RuntimeError("landing is unsupported by this adapter")
        decision = self.failsafe.force_land(self._time_s, link_age_s=self._link_age_s, battery_pct=self._battery_pct)
        self._last_failsafe = decision
        if decision.changed: self._event("failsafe_transition", decision.as_dict())
        self._sync(decision); return decision
    def kill(self) -> FailsafeDecision:
        self._require()
        if not self.capabilities.supports_kill: raise RuntimeError("kill is unsupported by this adapter")
        decision = self.failsafe.force_kill(self._time_s, link_age_s=self._link_age_s, battery_pct=self._battery_pct)
        self._last_failsafe = decision
        if decision.changed: self._event("failsafe_transition", decision.as_dict())
        self._sync(decision); self._event("kill", {}); return decision
    def reset_failsafe(self) -> FailsafeDecision:
        self._require(); decision = self.failsafe.reset(self._time_s, link_age_s=self._link_age_s, battery_pct=self._battery_pct); self._landed = False; self._last_failsafe = decision
        if decision.changed: self._event("failsafe_transition", decision.as_dict())
        self._sync(decision); return decision

    def step(self, dt_s: float | None = None, *, link_age_s: float | None = None, battery_pct: float | None = None,
             kill_requested: bool = False, recent_loss_fraction: float | None = None) -> DroneState:
        self._require()
        dt = self._dt_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0: raise ValueError("dt_s must be positive")
        if link_age_s is not None: self._link_age_s = _finite_nonnegative("link_age_s", link_age_s)
        if battery_pct is not None: self.set_battery_pct(battery_pct)
        self._time_s += dt
        decision = self._observe(kill_requested=kill_requested, recent_loss_fraction=recent_loss_fraction)
        winner = self.authority.select(self._time_s)
        source = None if winner.source is None else winner.source.value
        if source != self._last_authority:
            self._last_authority = source; self._event("authority_change", {"source": source})
        if self._geofence is not None and not self._geofence.contains(self.vehicle.drone.position):
            self._event("geofence_breach", {"position_m": self.vehicle.drone.position.tolist()})
            self.authority.submit(CommandSource.FAILSAFE, ControlCommand.hold(), timestamp_s=self._time_s, ttl_s=None, request_id="geofence-breach")
            winner = self.authority.select(self._time_s)
        command = winner.command
        if decision.action is FailsafeAction.KILL or (command is not None and command.kind is CommandKind.KILL):
            self.vehicle.actuator.speed_rad_s[:] = 0.0
            self.vehicle.drone.step_physics(np.zeros(3), np.zeros(3), dt)
        else:
            if command is not None and command.kind is CommandKind.POSITION:
                desired = np.asarray(command.position_m, dtype=float)
                speed = self.failsafe.config.slow_speed_mps if decision.action is FailsafeAction.SLOW else self.capabilities.max_speed_mps
                delta = desired - self._carrot
                norm = float(np.linalg.norm(delta))
                self._carrot += delta if norm <= speed * dt else delta * ((speed * dt) / norm)
            elif command is not None and command.kind is CommandKind.HOLD:
                if self._last_command_kind is not CommandKind.HOLD:
                    self._hold_carrot = self.vehicle.drone.position.copy()
                self._carrot = self._hold_carrot.copy()
            elif command is not None and command.kind is CommandKind.LAND:
                self._carrot[:2] = self.vehicle.drone.position[:2]
                self._carrot[2] = max(self._ground, self._carrot[2] - self._land_speed * dt)
            else:
                self._hold_carrot = self.vehicle.drone.position.copy(); self._carrot = self._hold_carrot.copy()
            self._last_command_kind = None if command is None else command.kind
            self.vehicle.step(self._carrot, dt)
            if decision.action is FailsafeAction.SLOW:
                speed = float(np.linalg.norm(self.vehicle.drone.velocity))
                cap = self.failsafe.config.slow_speed_mps
                if speed > cap:
                    self.vehicle.drone.velocity *= cap / speed
        if self.vehicle.drone.position[2] < self._ground:
            self.vehicle.drone.position[2] = self._ground
            self.vehicle.drone.velocity[2] = max(0.0, self.vehicle.drone.velocity[2])
        if decision.state is FailsafeState.LANDING and self.vehicle.drone.position[2] <= self._ground + 0.02 and np.linalg.norm(self.vehicle.drone.velocity) < 0.05:
            self._landed = True; self._event("landed", {}); self._observe(recent_loss_fraction=recent_loss_fraction)
            self.vehicle.actuator.speed_rad_s[:] = 0.0
        return self.get_state()

    def get_state(self) -> DroneState:
        authority = self.authority.select(self._time_s)
        return DroneState(position=self.vehicle.drone.position.copy(), velocity=self.vehicle.drone.velocity.copy(),
                          battery_pct=self._battery_pct, link_delay_ms=self._link_delay_ms, link_loss_pct=self._link_loss_pct,
                          timestamp_s=self._time_s, connected=self._connected,
                          command_authority=None if authority.source is None else authority.source.value,
                          failsafe_state=self.failsafe.state.value, link_age_s=self._link_age_s, landed=self._landed)

    def model_provenance(self) -> dict[str, Any]:
        result = self.vehicle.model_provenance()
        result.update({"profile_id": self.profile.profile_id, "floor": "modelled", "drag": "off", "simulated_only": True})
        return result
