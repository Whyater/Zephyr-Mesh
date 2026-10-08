"""Deterministic command authority arbitration for simulator adapters.

The simulator receives several *possible* command producers: a mission, a
formation controller, a manual operator, and the safety layer.  This module
keeps the ownership decision explicit and inspectable.  It deliberately
contains no radio, joystick, or flight-controller integration.  A hardware
adapter can later translate the selected command after passing the same
conformance tests.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Iterable

import numpy as np


class CommandSource(str, Enum):
    """Producers that may request control of one vehicle."""

    FAILSAFE = "failsafe"
    MANUAL = "manual"
    MISSION = "mission"
    FORMATION = "formation"
    IDLE = "idle"


class CommandKind(str, Enum):
    """The small command vocabulary shared by adapters."""

    POSITION = "position"
    HOLD = "hold"
    LAND = "land"
    KILL = "kill"


# Safety must be able to pre-empt every normal command.  The remaining order
# is intentional and documented so a tie never depends on dictionary order.
SOURCE_PRIORITY: dict[CommandSource, int] = {
    CommandSource.FAILSAFE: 100,
    CommandSource.MANUAL: 80,
    CommandSource.MISSION: 60,
    CommandSource.FORMATION: 40,
    CommandSource.IDLE: 0,
}


def _finite_time(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _vector3(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class ControlCommand:
    """A serializable command that an adapter may execute.

    Position setpoints use metres in the simulator's world frame.  ``HOLD``,
    ``LAND``, and ``KILL`` intentionally do not carry an arbitrary payload.
    """

    kind: CommandKind
    position_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommandKind):
            try:
                object.__setattr__(self, "kind", CommandKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("kind must be a valid CommandKind") from exc
        if self.kind is CommandKind.POSITION:
            if self.position_m is None:
                raise ValueError("position command requires position_m")
            object.__setattr__(self, "position_m", _vector3(self.position_m, name="position_m"))
        elif self.position_m is not None:
            raise ValueError(f"{self.kind.value} command cannot carry position_m")

    @classmethod
    def position(cls, position_m: Iterable[float]) -> "ControlCommand":
        return cls(CommandKind.POSITION, _vector3(position_m, name="position_m"))

    @classmethod
    def hold(cls) -> "ControlCommand":
        return cls(CommandKind.HOLD)

    @classmethod
    def land(cls) -> "ControlCommand":
        return cls(CommandKind.LAND)

    @classmethod
    def kill(cls) -> "ControlCommand":
        return cls(CommandKind.KILL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "position_m": None if self.position_m is None else list(self.position_m),
        }


@dataclass(frozen=True)
class AuthorityRequest:
    """One source's currently active request."""

    source: CommandSource
    command: ControlCommand
    requested_at_s: float
    expires_at_s: float | None
    request_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "priority": SOURCE_PRIORITY[self.source],
            "command": self.command.as_dict(),
            "requested_at_s": self.requested_at_s,
            "expires_at_s": self.expires_at_s,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class AuthorityDecision:
    """The result of evaluating all active requests at one timestamp.

    ``source`` and ``request_id`` identify the selected winner.  When the
    decision came from :meth:`AuthorityArbiter.submit`, the
    ``submitted_*`` fields identify the request that was just published and
    ``request_selected`` says whether that request owns authority.  This keeps
    a lower-priority request from being mistaken for an accepted command.
    """

    accepted: bool
    source: CommandSource | None
    command: ControlCommand | None
    reason: str
    evaluated_at_s: float
    request_id: str | None = None
    contenders: tuple[str, ...] = ()
    submitted_source: CommandSource | None = None
    submitted_request_id: str | None = None
    request_selected: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "source": None if self.source is None else self.source.value,
            "command": None if self.command is None else self.command.as_dict(),
            "reason": self.reason,
            "evaluated_at_s": self.evaluated_at_s,
            "request_id": self.request_id,
            "contenders": list(self.contenders),
            "submitted_source": None if self.submitted_source is None else self.submitted_source.value,
            "submitted_request_id": self.submitted_request_id,
            "request_selected": self.request_selected,
        }


class AuthorityArbiter:
    """Select one command using priorities, expiry, and deterministic ties.

    At most one request per source is active.  A new request replaces that
    source's old request, which makes manual takeover and command renewal
    explicit rather than an accidental race between loops.  A finite ``ttl_s``
    is recommended for normal commands.  Safety requests may be non-expiring
    until the failsafe is reset.
    """

    def __init__(self) -> None:
        self._requests: dict[CommandSource, AuthorityRequest] = {}
        self._last_time_s = 0.0
        self._sequence = 0

    @property
    def active_requests(self) -> tuple[AuthorityRequest, ...]:
        """Return active requests in stable source order."""
        return tuple(self._requests[source] for source in sorted(self._requests, key=lambda item: item.value))

    @property
    def last_time_s(self) -> float:
        return self._last_time_s

    def _time(self, value: float) -> float:
        value = _finite_time("timestamp_s", value)
        if value < self._last_time_s:
            raise ValueError("timestamp_s must be nondecreasing")
        self._last_time_s = value
        return value

    def submit(
        self,
        source: CommandSource,
        command: ControlCommand,
        *,
        timestamp_s: float,
        ttl_s: float | None = 0.5,
        request_id: str | None = None,
    ) -> AuthorityDecision:
        """Publish a request and return the selected command immediately."""
        if not isinstance(source, CommandSource):
            try:
                source = CommandSource(source)
            except (TypeError, ValueError) as exc:
                raise ValueError("source must be a valid CommandSource") from exc
        if not isinstance(command, ControlCommand):
            raise ValueError("command must be a ControlCommand")
        if ttl_s is not None:
            ttl = float(ttl_s)
            if not math.isfinite(ttl) or ttl <= 0.0:
                raise ValueError("ttl_s must be finite and positive or None")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("request_id must be a non-empty string")
        now = self._time(timestamp_s)
        if ttl_s is not None:
            expires_at_s: float | None = now + ttl
        else:
            expires_at_s = None
        self._sequence += 1
        identifier = request_id or f"{source.value}-{self._sequence:06d}"
        self._requests[source] = AuthorityRequest(source, command, now, expires_at_s, identifier)
        selected = self.select(now)
        request_selected = selected.source is source and selected.request_id == identifier
        return replace(
            selected,
            accepted=request_selected,
            reason="selected_highest_priority" if request_selected else "request_preempted",
            submitted_source=source,
            submitted_request_id=identifier,
            request_selected=request_selected,
        )

    def release(self, source: CommandSource, *, timestamp_s: float | None = None) -> None:
        """Remove a source's request without changing other sources."""
        if not isinstance(source, CommandSource):
            source = CommandSource(source)
        if timestamp_s is not None:
            self._time(timestamp_s)
        self._requests.pop(source, None)

    def clear(self, *, timestamp_s: float | None = None) -> None:
        """Release every request."""
        if timestamp_s is not None:
            self._time(timestamp_s)
        self._requests.clear()

    def select(self, timestamp_s: float) -> AuthorityDecision:
        """Return the winner at ``timestamp_s`` and expire stale requests."""
        now = self._time(timestamp_s)
        expired = [
            source for source, request in self._requests.items()
            if request.expires_at_s is not None and request.expires_at_s <= now
        ]
        for source in expired:
            self._requests.pop(source, None)
        requests = tuple(self._requests.values())
        if not requests:
            return AuthorityDecision(False, None, None, "no_active_command", now)
        ordered = sorted(
            requests,
            key=lambda request: (
                -SOURCE_PRIORITY[request.source],
                -request.requested_at_s,
                request.source.value,
                request.request_id,
            ),
        )
        winner = ordered[0]
        return AuthorityDecision(
            True,
            winner.source,
            winner.command,
            "selected_highest_priority",
            now,
            request_id=winner.request_id,
            contenders=tuple(request.source.value for request in ordered),
        )
