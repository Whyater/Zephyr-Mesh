"""Deterministic link and battery failsafe state machine.

This is a synthetic policy contract for replay and adapter conformance tests.
It does not claim that a particular aircraft, radio, or regulator uses these
thresholds.  Thresholds are explicit inputs so measured fixtures can replace
them later without changing the state-transition logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from sim.authority import ControlCommand


class FailsafeState(str, Enum):
    """Latched safety states exposed to the UI and adapter boundary."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    LANDING = "landing"
    LANDED = "landed"
    KILLED = "killed"


class FailsafeAction(str, Enum):
    """Action the authority layer should publish for the current state."""

    NONE = "none"
    HOLD = "hold"
    LAND = "land"
    KILL = "kill"


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class FailsafeConfig:
    """Policy thresholds for the deterministic fixture.

    ``degraded_after_s`` enters a hold state.  ``land_after_s`` enters the
    latched landing state.  The defaults mirror the product note's rough
    300 ms indoor-link example, but are scenario assumptions rather than a
    field specification.
    """

    degraded_after_s: float = 0.15
    land_after_s: float = 0.30
    low_battery_pct: float = 15.0

    def __post_init__(self) -> None:
        degraded = _finite_nonnegative("degraded_after_s", self.degraded_after_s)
        land = _finite_nonnegative("land_after_s", self.land_after_s)
        battery = float(self.low_battery_pct)
        if not math.isfinite(battery) or battery < 0.0 or battery > 100.0:
            raise ValueError("low_battery_pct must be finite and between zero and 100")
        if land < degraded:
            raise ValueError("land_after_s must be at least degraded_after_s")
        object.__setattr__(self, "degraded_after_s", degraded)
        object.__setattr__(self, "land_after_s", land)
        object.__setattr__(self, "low_battery_pct", battery)


@dataclass(frozen=True)
class FailsafeDecision:
    """One state-machine evaluation, including transition evidence."""

    state: FailsafeState
    action: FailsafeAction
    changed: bool
    previous_state: FailsafeState
    reason: str
    evaluated_at_s: float
    link_age_s: float | None
    battery_pct: float

    @property
    def command(self) -> ControlCommand | None:
        """Map the policy state to the command vocabulary."""
        if self.action is FailsafeAction.HOLD:
            return ControlCommand.hold()
        if self.action is FailsafeAction.LAND:
            return ControlCommand.land()
        if self.action is FailsafeAction.KILL:
            return ControlCommand.kill()
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "action": self.action.value,
            "changed": self.changed,
            "previous_state": self.previous_state.value,
            "reason": self.reason,
            "evaluated_at_s": self.evaluated_at_s,
            "link_age_s": self.link_age_s,
            "battery_pct": self.battery_pct,
        }


class FailsafeController:
    """Evaluate deterministic safety transitions with explicit reset points.

    A degraded link may recover to nominal.  Landing and kill are latched so
    a momentary packet or battery reading cannot silently resume motion.  An
    operator must call :meth:`reset` with a healthy-link and battery check,
    and a landing must be acknowledged with ``landed=True`` before the state
    becomes ``LANDED``.
    """

    def __init__(self, config: FailsafeConfig | None = None) -> None:
        self.config = config or FailsafeConfig()
        self.state = FailsafeState.NOMINAL
        self._last_time_s = 0.0
        self.last_decision = FailsafeDecision(
            state=FailsafeState.NOMINAL,
            action=FailsafeAction.NONE,
            changed=False,
            previous_state=FailsafeState.NOMINAL,
            reason="initial",
            evaluated_at_s=0.0,
            link_age_s=0.0,
            battery_pct=100.0,
        )

    @property
    def last_time_s(self) -> float:
        return self._last_time_s

    def _time(self, timestamp_s: float) -> float:
        timestamp_s = _finite_nonnegative("timestamp_s", timestamp_s)
        if timestamp_s < self._last_time_s:
            raise ValueError("timestamp_s must be nondecreasing")
        self._last_time_s = timestamp_s
        return timestamp_s

    @staticmethod
    def _battery(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            raise ValueError("battery_pct must be finite and between zero and 100")
        return value

    @staticmethod
    def _link_age(value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_nonnegative("link_age_s", value)

    def _decision(
        self,
        *,
        previous: FailsafeState,
        reason: str,
        timestamp_s: float,
        link_age_s: float | None,
        battery_pct: float,
    ) -> FailsafeDecision:
        action = {
            FailsafeState.NOMINAL: FailsafeAction.NONE,
            FailsafeState.DEGRADED: FailsafeAction.HOLD,
            FailsafeState.LANDING: FailsafeAction.LAND,
            FailsafeState.LANDED: FailsafeAction.NONE,
            FailsafeState.KILLED: FailsafeAction.KILL,
        }[self.state]
        decision = FailsafeDecision(
            state=self.state,
            action=action,
            changed=self.state is not previous,
            previous_state=previous,
            reason=reason,
            evaluated_at_s=timestamp_s,
            link_age_s=link_age_s,
            battery_pct=battery_pct,
        )
        self.last_decision = decision
        return decision

    def observe(
        self,
        timestamp_s: float,
        *,
        link_age_s: float | None,
        battery_pct: float,
        kill_requested: bool = False,
        landed: bool = False,
    ) -> FailsafeDecision:
        """Advance the policy using one shared-clock observation."""
        age = self._link_age(link_age_s)
        battery = self._battery(battery_pct)
        now = self._time(timestamp_s)
        previous = self.state

        if self.state is FailsafeState.KILLED:
            return self._decision(
                previous=previous,
                reason="kill_latched",
                timestamp_s=now,
                link_age_s=age,
                battery_pct=battery,
            )
        if kill_requested:
            self.state = FailsafeState.KILLED
            return self._decision(
                previous=previous,
                reason="operator_kill",
                timestamp_s=now,
                link_age_s=age,
                battery_pct=battery,
            )
        if self.state is FailsafeState.LANDED:
            return self._decision(
                previous=previous,
                reason="landed_latched",
                timestamp_s=now,
                link_age_s=age,
                battery_pct=battery,
            )
        if self.state is FailsafeState.LANDING:
            if landed:
                self.state = FailsafeState.LANDED
                reason = "landing_acknowledged"
            else:
                reason = "landing_latched"
            return self._decision(
                previous=previous,
                reason=reason,
                timestamp_s=now,
                link_age_s=age,
                battery_pct=battery,
            )

        stale_age = math.inf if age is None else age
        if battery <= self.config.low_battery_pct:
            self.state = FailsafeState.LANDING
            reason = "low_battery"
        elif stale_age >= self.config.land_after_s:
            self.state = FailsafeState.LANDING
            reason = "link_timeout"
        elif stale_age >= self.config.degraded_after_s:
            self.state = FailsafeState.DEGRADED
            reason = "link_degraded"
        else:
            self.state = FailsafeState.NOMINAL
            reason = "link_healthy"
        return self._decision(
            previous=previous,
            reason=reason,
            timestamp_s=now,
            link_age_s=age,
            battery_pct=battery,
        )

    def force_land(
        self,
        timestamp_s: float,
        *,
        link_age_s: float | None = 0.0,
        battery_pct: float = 100.0,
    ) -> FailsafeDecision:
        """Latch landing for a direct operator or policy request."""
        age = self._link_age(link_age_s)
        battery = self._battery(battery_pct)
        now = self._time(timestamp_s)
        previous = self.state
        if self.state not in {FailsafeState.KILLED, FailsafeState.LANDED}:
            self.state = FailsafeState.LANDING
        return self._decision(
            previous=previous,
            reason="operator_land",
            timestamp_s=now,
            link_age_s=age,
            battery_pct=battery,
        )

    def force_kill(
        self,
        timestamp_s: float,
        *,
        link_age_s: float | None = 0.0,
        battery_pct: float = 100.0,
    ) -> FailsafeDecision:
        """Latch the kill state for a direct emergency stop."""
        age = self._link_age(link_age_s)
        battery = self._battery(battery_pct)
        now = self._time(timestamp_s)
        previous = self.state
        self.state = FailsafeState.KILLED
        return self._decision(
            previous=previous,
            reason="operator_kill",
            timestamp_s=now,
            link_age_s=age,
            battery_pct=battery,
        )

    def reset(
        self,
        timestamp_s: float,
        *,
        link_age_s: float | None,
        battery_pct: float,
    ) -> FailsafeDecision:
        """Explicitly re-arm the policy only after healthy observations."""
        age = self._link_age(link_age_s)
        battery = self._battery(battery_pct)
        if age is None or age >= self.config.degraded_after_s:
            raise ValueError("cannot reset failsafe with a stale link")
        if battery <= self.config.low_battery_pct:
            raise ValueError("cannot reset failsafe with low battery")
        if self.state is FailsafeState.LANDING:
            raise ValueError("cannot reset failsafe before landing is acknowledged")
        now = self._time(timestamp_s)
        previous = self.state
        self.state = FailsafeState.NOMINAL
        return self._decision(
            previous=previous,
            reason="operator_reset",
            timestamp_s=now,
            link_age_s=age,
            battery_pct=battery,
        )
