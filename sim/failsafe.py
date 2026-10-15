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
    SLOW = "slow"
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
    weak_rule: str = "age"
    loss_window_packets: int = 20
    weak_loss_fraction: float = 0.3
    degraded_action: str = "hold"
    slow_speed_mps: float = 0.3

    def __post_init__(self) -> None:
        degraded = _finite_nonnegative("degraded_after_s", self.degraded_after_s)
        land = _finite_nonnegative("land_after_s", self.land_after_s)
        battery = float(self.low_battery_pct)
        if not math.isfinite(battery) or battery < 0.0 or battery > 100.0:
            raise ValueError("low_battery_pct must be finite and between zero and 100")
        if land < degraded:
            raise ValueError("land_after_s must be at least degraded_after_s")
        if not isinstance(self.weak_rule, str) or self.weak_rule not in {"age", "age_or_loss"}:
            raise ValueError("weak_rule must be 'age' or 'age_or_loss'")
        if isinstance(self.loss_window_packets, bool) or int(self.loss_window_packets) != self.loss_window_packets or int(self.loss_window_packets) <= 0:
            raise ValueError("loss_window_packets must be a positive integer")
        weak_loss = float(self.weak_loss_fraction)
        if not math.isfinite(weak_loss) or weak_loss < 0.0 or weak_loss > 1.0:
            raise ValueError("weak_loss_fraction must be finite and between zero and one")
        if not isinstance(self.degraded_action, str) or self.degraded_action not in {"hold", "slow"}:
            raise ValueError("degraded_action must be 'hold' or 'slow'")
        slow_speed = float(self.slow_speed_mps)
        if not math.isfinite(slow_speed) or slow_speed <= 0.0:
            raise ValueError("slow_speed_mps must be finite and positive")
        object.__setattr__(self, "degraded_after_s", degraded)
        object.__setattr__(self, "land_after_s", land)
        object.__setattr__(self, "low_battery_pct", battery)
        object.__setattr__(self, "loss_window_packets", int(self.loss_window_packets))
        object.__setattr__(self, "weak_loss_fraction", weak_loss)
        object.__setattr__(self, "slow_speed_mps", slow_speed)

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in sorted((
            "degraded_action", "degraded_after_s", "land_after_s", "loss_window_packets",
            "low_battery_pct", "slow_speed_mps", "weak_loss_fraction", "weak_rule",
        ))}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FailsafeConfig":
        if not isinstance(value, dict):
            raise ValueError("failsafe config must be a mapping")
        allowed = {"degraded_after_s", "land_after_s", "low_battery_pct", "weak_rule",
                   "loss_window_packets", "weak_loss_fraction", "degraded_action", "slow_speed_mps"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown failsafe config fields: {sorted(unknown)}")
        return cls(**value)


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
        if self.state is FailsafeState.DEGRADED and self.config.degraded_action == "slow":
            action = FailsafeAction.SLOW
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
        recent_loss_fraction: float | None = None,
    ) -> FailsafeDecision:
        """Advance the policy using one shared-clock observation."""
        age = self._link_age(link_age_s)
        battery = self._battery(battery_pct)
        if recent_loss_fraction is not None:
            loss = float(recent_loss_fraction)
            if not math.isfinite(loss) or loss < 0.0 or loss > 1.0:
                raise ValueError("recent_loss_fraction must be finite and between zero and one")
        else:
            loss = None
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
        elif stale_age >= self.config.degraded_after_s or (
            self.config.weak_rule == "age_or_loss" and loss is not None and loss >= self.config.weak_loss_fraction
        ):
            self.state = FailsafeState.DEGRADED
            reason = "link_lossy" if (
                self.config.weak_rule == "age_or_loss" and loss is not None
                and loss >= self.config.weak_loss_fraction and stale_age < self.config.degraded_after_s
            ) else "link_degraded"
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


class SequenceLossWindow:
    """Track received sequence numbers and estimate a trailing loss fraction."""

    def __init__(self, window_packets: int = 20) -> None:
        if isinstance(window_packets, bool) or int(window_packets) != window_packets or int(window_packets) <= 0:
            raise ValueError("window_packets must be a positive integer")
        self.window_packets = int(window_packets)
        self._received: set[int] = set()
        self._newest: int | None = None

    def record(self, seq: int) -> None:
        if isinstance(seq, bool) or int(seq) != seq or int(seq) < 0:
            raise ValueError("seq must be a nonnegative integer")
        seq = int(seq)
        if self._newest is not None and seq <= self._newest:
            return
        self._received.add(seq)
        self._newest = seq
        floor = seq - self.window_packets * 2
        if floor > 0:
            self._received = {item for item in self._received if item >= floor}

    def loss_fraction(self, expected_next_seq: int | None = None) -> float | None:
        if self._newest is None:
            return None
        if expected_next_seq is None:
            end = self._newest
        else:
            if isinstance(expected_next_seq, bool) or int(expected_next_seq) != expected_next_seq or int(expected_next_seq) < 0:
                raise ValueError("expected_next_seq must be a nonnegative integer")
            end = int(expected_next_seq) - 1
        if end < 0:
            return 0.0
        start = max(0, end - self.window_packets + 1)
        total = end - start + 1
        received = sum(seq in self._received for seq in range(start, end + 1))
        return float(total - received) / float(total)
