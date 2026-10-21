"""Deterministic, deliberately small model of an imperfect packet link.

This is a research fixture, not an ESP-NOW performance model.  It exposes
delay and loss assumptions explicitly and retains one event per attempted
packet so later metrics can distinguish loss, duplicate delivery, and
reordering.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinkConfig:
    """Parameters for :class:`SimulatedLink`.

    ``contention_mode='serialized'`` is only a declared abstraction: packets
    are put on one FIFO transmission resource. It does not claim to model
    CSMA/CA, airtime, or radio backoff.
    """

    # delay_s is the mean one-way delay and delay_jitter_s is the standard
    # deviation of the delay.  With jitter, delay is lognormal (always above
    # zero, right-skewed), so the realized mean and spread match these two
    # numbers.  See SimulatedLink._sample_delay.
    delay_s: float = 0.0
    delay_jitter_s: float = 0.0
    loss_model: str = "independent"  # independent, burst, or none
    loss_probability: float = 0.0
    burst_start_probability: float = 0.0
    burst_end_probability: float = 1.0
    contention_mode: str = "none"  # none or serialized
    packet_duration_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("delay_s", "delay_jitter_s", "loss_probability",
                     "burst_start_probability", "burst_end_probability",
                     "packet_duration_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.delay_jitter_s > self.delay_s and self.delay_s == 0.0:
            raise ValueError("delay_jitter_s cannot be positive when delay_s is zero")
        if self.loss_model not in {"none", "independent", "burst"}:
            raise ValueError("loss_model must be none, independent, or burst")
        if self.contention_mode not in {"none", "serialized"}:
            raise ValueError("contention_mode must be none or serialized")
        if self.loss_probability > 1 or self.burst_start_probability > 1 or self.burst_end_probability > 1:
            raise ValueError("probabilities must be at most one")


@dataclass
class PacketEvent:
    sender: str
    receiver: str
    seq: int
    send_time: float
    receive_time: float | None
    duplicate: bool = False
    out_of_order: bool = False
    loss_reason: str | None = None
    packet_age_s: float | None = None
    packet_id: str | None = None
    sender_session_id: str | None = None
    outcome: str | None = None
    callback_time: float | None = None
    rssi_dbm: float | None = None
    payload_bytes: int | None = None
    retry_count: int | None = None
    unknown_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["packet_age"] = data.pop("packet_age_s")
        for key in ("packet_id", "sender_session_id", "outcome", "callback_time", "rssi_dbm", "payload_bytes", "retry_count", "unknown_reason"):
            if data[key] is None:
                data.pop(key)
        return data


class SimulatedLink:
    """Seeded packet delay/loss queue with an inspectable event log."""

    def __init__(self, config: LinkConfig | None = None, *, seed: int | None = None):
        self.config = config or LinkConfig()
        # Loss and delay draw from separate streams spawned from one seed, so
        # changing the jitter setting cannot change which packets are lost,
        # and changing the loss setting cannot change any packet's delay.
        loss_seq, delay_seq = np.random.SeedSequence(seed).spawn(2)
        self.loss_rng = np.random.default_rng(loss_seq)
        self.delay_rng = np.random.default_rng(delay_seq)
        self.events: list[PacketEvent] = []
        self._pending: list[PacketEvent] = []
        self._seen: dict[tuple[str, str], set[int]] = {}
        self._highest: dict[tuple[str, str], int] = {}
        self._last_send_time: dict[tuple[str, str], float] = {}
        self._burst_active = False
        self._next_available_time = 0.0

    def _loss_reason(self) -> str | None:
        c = self.config
        if c.loss_model == "none":
            return None
        if c.loss_model == "independent":
            return "independent_loss" if self.loss_rng.random() < c.loss_probability else None
        if not self._burst_active and self.loss_rng.random() < c.burst_start_probability:
            self._burst_active = True
        if self._burst_active:
            reason = "burst_loss"
            if self.loss_rng.random() < c.burst_end_probability:
                self._burst_active = False
            return reason
        return None

    def _sample_delay(self) -> float:
        """Draw one one-way delay in seconds.

        ``delay_s`` is the mean and ``delay_jitter_s`` the standard deviation.
        A Normal(mean, std) clipped at zero would pile packets up at exactly
        0 s and push the realized mean above the configured one (for mean
        0.01 s, std 0.02 s the clipped mean is 0.0140 s, not 0.010 s).  A
        lognormal has no negative values and a right-skewed tail, and its
        two parameters can be solved so the mean and std come out exact:
        sigma^2 = ln(1 + std^2 / mean^2), mu = ln(mean) - sigma^2 / 2.
        This is a modelling choice, not a measured ESP-NOW latency law.
        """
        c = self.config
        if not c.delay_jitter_s:
            return c.delay_s
        sigma2 = float(np.log1p((c.delay_jitter_s / c.delay_s) ** 2))
        mu = float(np.log(c.delay_s)) - sigma2 / 2.0
        return float(self.delay_rng.lognormal(mu, np.sqrt(sigma2)))

    def send(self, sender: str, receiver: str, seq: int, send_time: float,
             payload: Any = None) -> PacketEvent:
        """Attempt one packet and return its eventual event record.

        Payload is accepted for caller convenience but deliberately not copied
        into the event log. The link models timing and identity, not contents.

        ``seq`` must not wrap within a route: duplicate and out-of-order flags
        identify a packet by ``(sender, receiver, seq)``.  A uint8 or uint16
        counter that repeats values will be flagged as duplicates.  Unwrap the
        counter before calling this method (for example, keep a per-session
        count that only increases).
        """
        del payload
        if not isinstance(sender, str) or not isinstance(receiver, str) or not sender or not receiver:
            raise ValueError("sender and receiver must be non-empty strings")
        if isinstance(seq, (bool, np.bool_)) or int(seq) != seq or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        send_time = float(send_time)
        if not np.isfinite(send_time) or send_time < 0:
            raise ValueError("send_time must be finite and nonnegative")
        c = self.config
        route = (sender, receiver)
        actual_send = send_time
        if c.contention_mode == "serialized":
            actual_send = max(send_time, self._next_available_time)
            self._next_available_time = actual_send + c.packet_duration_s
        previous_send = self._last_send_time.get(route)
        if previous_send is not None and actual_send < previous_send:
            raise ValueError("send_time must be nondecreasing per sender/receiver route")
        self._last_send_time[route] = actual_send
        reason = self._loss_reason()
        # The delay is drawn even for a lost packet so the delay stream stays
        # aligned with packet index whatever the loss setting is.
        delay = self._sample_delay()
        receive_time = None if reason else actual_send + delay
        event = PacketEvent(sender, receiver, int(seq), actual_send, receive_time,
                            loss_reason=reason,
                            packet_age_s=None if receive_time is None else receive_time - actual_send,
                            outcome="lost" if reason else "received")
        self.events.append(event)
        if receive_time is not None:
            self._pending.append(event)
        return event

    def advance(self, now: float) -> list[PacketEvent]:
        """Deliver queued packets whose receive time is at or before ``now``."""
        now = float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError("now must be finite and nonnegative")
        ready = [p for p in self._pending if p.receive_time <= now]
        self._pending = [p for p in self._pending if p.receive_time > now]
        for event in sorted(ready, key=lambda p: (p.receive_time, p.seq)):
            key = (event.sender, event.receiver)
            seen = self._seen.setdefault(key, set())
            event.duplicate = event.seq in seen
            event.out_of_order = not event.duplicate and event.seq < self._highest.get(key, event.seq)
            seen.add(event.seq)
            self._highest[key] = max(event.seq, self._highest.get(key, event.seq))
        return sorted(ready, key=lambda p: (p.receive_time, p.seq))

    def deliver_all(self) -> list[PacketEvent]:
        """Deliver every queued packet, useful for deterministic fixtures."""
        if not self._pending:
            return []
        return self.advance(max(p.receive_time for p in self._pending))


def replay_trace(trace: Any) -> list[PacketEvent]:
    """Replay an :class:`sim.trace.EspNowTrace` without fitting a link model.

    The trace's observed outcome and receive timestamp remain authoritative.
    This function only derives duplicate and out-of-order flags when the raw
    capture did not provide them. It intentionally does not estimate missing
    packets, interpolate clocks, or turn RSSI into distance.

    Derived flags identify a packet by ``(sender, receiver, session, seq)``, so
    a sequence counter that wraps (for example uint8, every 256 packets) is
    misread as duplicates.  The schema has no field that says how wide the
    counter is, so replay cannot unwrap it.  A capture tool with a narrow
    counter must supply the explicit ``duplicate`` and ``out_of_order`` flags,
    or write an unwrapped ``seq``; those override the derived values.

    Echo captures (``clock_domain = "sender_round_trip"``) are refused: they
    hold no one-way arrival time on the sender's clock, only a round trip, and
    replay would have to invent one. Fit them with ``tools/fit_espnow_log.py``
    instead.
    """

    from .trace import EspNowTrace

    if not isinstance(trace, EspNowTrace):
        raise TypeError("trace must be a validated EspNowTrace")
    if trace.clock_domain == "sender_round_trip":
        raise ValueError("sender_round_trip (echo) traces have no one-way arrival times to replay; "
                         "fit them with tools/fit_espnow_log.py instead")
    packets = trace.packets
    clock_domain = getattr(trace, "clock_domain", "")
    events: list[PacketEvent] = []
    received = [packet for packet in packets if packet.outcome == "received" and packet.receive_time_s is not None]
    ordered = sorted(received, key=lambda packet: (packet.receive_time_s, packet.packet_id))
    seen: dict[tuple[str, str, str], set[int]] = {}
    highest: dict[tuple[str, str, str], int] = {}
    derived_flags: dict[str, tuple[bool, bool]] = {}
    for packet in ordered:
        route = (packet.sender_id, packet.receiver_id, packet.sender_session_id)
        route_seen = seen.setdefault(route, set())
        duplicate = packet.seq in route_seen
        out_of_order = not duplicate and packet.seq < highest.get(route, packet.seq)
        route_seen.add(packet.seq)
        highest[route] = max(packet.seq, highest.get(route, packet.seq))
        derived_flags[packet.packet_id] = (duplicate, out_of_order)
    for packet in packets:
        if packet.outcome == "received":
            receive_time = packet.receive_time_s
            if receive_time is None:
                raise ValueError("received trace packet is missing receive_time_s")
            if packet.packet_id in derived_flags:
                duplicate, out_of_order = derived_flags[packet.packet_id]
            else:
                duplicate, out_of_order = False, False
            packet_age = None
            if clock_domain == "shared_monotonic":
                packet_age = receive_time - packet.send_time_s
                if packet_age < 0:
                    raise ValueError("shared_monotonic receive_time_s cannot precede send_time_s")
            event = PacketEvent(
                packet.sender_id,
                packet.receiver_id,
                packet.seq,
                packet.send_time_s,
                receive_time,
                duplicate=packet.duplicate if packet.duplicate is not None else duplicate,
                out_of_order=packet.out_of_order if packet.out_of_order is not None else out_of_order,
                packet_age_s=packet_age,
                packet_id=packet.packet_id,
                sender_session_id=packet.sender_session_id,
                outcome=packet.outcome,
                callback_time=packet.callback_time_s,
                rssi_dbm=packet.rssi_dbm,
                payload_bytes=packet.payload_bytes,
                retry_count=packet.retry_count,
            )
        else:
            event = PacketEvent(
                packet.sender_id,
                packet.receiver_id,
                packet.seq,
                packet.send_time_s,
                None,
                loss_reason=packet.loss_reason if packet.outcome == "lost" else None,
                packet_id=packet.packet_id,
                sender_session_id=packet.sender_session_id,
                outcome=packet.outcome,
                callback_time=packet.callback_time_s,
                rssi_dbm=packet.rssi_dbm,
                payload_bytes=packet.payload_bytes,
                retry_count=packet.retry_count,
                unknown_reason=packet.unknown_reason,
            )
        events.append(event)
    return events


def lag_error(distance_rate_mps: float, delay_s: float) -> float:
    """First-order hand check: constant-speed position error ``v × L``."""
    v, delay = float(distance_rate_mps), float(delay_s)
    if not np.isfinite(v) or not np.isfinite(delay) or v < 0 or delay < 0:
        raise ValueError("distance rate and delay must be finite and nonnegative")
    return v * delay
