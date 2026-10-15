"""One link interface with two modes: the old abstract link, or the radio model.

Callers that today build ``sim.link.SimulatedLink`` can build a network model
instead and switch the mode with one config field:

- ``mode="abstract"`` wraps ``SimulatedLink`` unchanged.  Same seed, same
  config, same events, byte for byte.  Runs B and D and EXP-006 depend on it.
- ``mode="radio"`` sends every packet through the channel (distance,
  obstacles, shadowing) and the event-driven MAC (airtime, carrier sense,
  backoff, collisions, ACKs and retries, queues).  Delay and loss are then
  outputs of a physical situation instead of knobs.

Both expose the ``SimulatedLink`` surface: ``send``, ``advance``,
``deliver_all`` and ``events`` (a list of ``sim.link.PacketEvent``).

Radio-mode details worth knowing:

- ``send`` returns the packet's event right away with ``outcome="pending"``;
  it becomes ``"received"`` or ``"lost"`` once the MAC has finished with the
  frame.  ``advance(now)`` returns the packets delivered up to ``now``.
- MAC retries can deliver the same frame twice (the data got through but
  the ACK was lost).  The receiver drops such copies with the 32-bit
  ``SequenceTracker`` (``duplicates_suppressed`` counts them), so callers see
  each packet at most once, like ``SimulatedLink``.
- Background traffic (other drones sending) is part of the model config, so
  the tracker's own stream competes for airtime with it.

Everything is simulated; radio parameters are literature-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

import numpy as np

from ..link import LinkConfig, PacketEvent, SimulatedLink
from .channel import Channel, ChannelConfig
from .mac import BROADCAST, Frame, MacConfig, MacSimulator, PhyConfig
from .params import channel_preset, default_mac, default_phy
from .seq import SEQ_MOD, SequenceTracker


class NetworkModel(Protocol):
    mode: str
    events: list[PacketEvent]

    def send(self, sender: str, receiver: str, seq: int, send_time: float, payload=None) -> PacketEvent: ...

    def advance(self, now: float) -> list[PacketEvent]: ...

    def deliver_all(self) -> list[PacketEvent]: ...


@dataclass(frozen=True)
class BackgroundSource:
    """Another node sending periodic traffic that competes for airtime."""

    node: str
    dst: str
    rate_hz: float
    payload_bytes: int = 32
    start_s: float = 0.0
    stop_s: float = math.inf
    jitter_s: float = 0.0


@dataclass
class RadioLinkConfig:
    """Physical situation for radio mode.

    ``positions`` must include every node that sends or receives.  ``channel``
    overrides the preset when given.  ``unicast`` True sends with ACK and
    retries; False broadcasts (no ACK, no retry) and the named receiver just
    listens.
    """

    positions: dict[str, tuple[float, float, float]]
    preset: str = "indoor_gym"
    channel: ChannelConfig | None = None
    phy: PhyConfig | None = None
    mac: MacConfig | None = None
    payload_bytes: int = 32
    unicast: bool = True
    background: tuple[BackgroundSource, ...] = ()
    walls: dict[tuple[str, str], int] = field(default_factory=dict)
    duplicate_window: int = 64
    drain_step_s: float = 0.5  # deliver_all advances the engine in steps of this size
    drain_limit_s: float = 60.0  # and gives up this long after the last send (a bug guard)


@dataclass
class NetworkModelConfig:
    mode: str = "abstract"  # abstract or radio
    link: LinkConfig = field(default_factory=LinkConfig)
    radio: RadioLinkConfig | None = None


def make_network_model(cfg: NetworkModelConfig, *, seed: int | None) -> "AbstractNetworkModel | RadioNetworkModel":
    if cfg.mode == "abstract":
        return AbstractNetworkModel(cfg.link, seed=seed)
    if cfg.mode == "radio":
        if cfg.radio is None:
            raise ValueError("radio mode needs a RadioLinkConfig")
        return RadioNetworkModel(cfg.radio, seed=seed)
    raise ValueError("mode must be abstract or radio")


class AbstractNetworkModel:
    """``SimulatedLink`` behind the shared interface.  Nothing is changed."""

    mode = "abstract"

    def __init__(self, config: LinkConfig | None = None, *, seed: int | None = None):
        self.link = SimulatedLink(config, seed=seed)

    @property
    def events(self) -> list[PacketEvent]:
        return self.link.events

    def send(self, sender: str, receiver: str, seq: int, send_time: float, payload=None) -> PacketEvent:
        return self.link.send(sender, receiver, seq, send_time, payload=payload)

    def advance(self, now: float) -> list[PacketEvent]:
        return self.link.advance(now)

    def deliver_all(self) -> list[PacketEvent]:
        return self.link.deliver_all()


class RadioNetworkModel:
    """Packets go through the channel and the MAC.  See the module docstring."""

    mode = "radio"

    def __init__(self, cfg: RadioLinkConfig, *, seed: int | None):
        if not isinstance(cfg.payload_bytes, int) or cfg.payload_bytes < 1:
            raise ValueError("payload_bytes must be a positive integer")
        self.cfg = cfg
        channel_cfg = cfg.channel if cfg.channel is not None else channel_preset(cfg.preset)
        channel_seq, mac_seq = np.random.SeedSequence(seed).spawn(2)
        self.channel = Channel(channel_cfg, np.random.default_rng(channel_seq))
        self.mac = MacSimulator(cfg.positions, self.channel, cfg.phy or default_phy(), cfg.mac or default_mac(),
                                seed=int(mac_seq.generate_state(1)[0]), walls=cfg.walls)
        for bg in cfg.background:
            if bg.node not in self.mac.positions:
                raise ValueError(f"background node {bg.node!r} has no position")
            self.mac.add_periodic_source(bg.node, bg.dst, bg.rate_hz, bg.payload_bytes, start_s=bg.start_s,
                                         stop_s=bg.stop_s, jitter_s=bg.jitter_s)
        self.mac.on_receive = self._on_receive
        self.events: list[PacketEvent] = []
        self._by_frame: dict[int, PacketEvent] = {}
        self._unresolved: dict[int, PacketEvent] = {}
        self._ready: list[PacketEvent] = []
        self._tracker = SequenceTracker(window=cfg.duplicate_window)
        self._rx_index: dict[int, list] = {}
        self._indexed = 0
        self._last_send: dict[tuple[str, str], float] = {}
        self.duplicates_suppressed = 0

    # ---- sending ----------------------------------------------------------
    def send(self, sender: str, receiver: str, seq: int, send_time: float, payload=None) -> PacketEvent:
        del payload  # timing and identity only, like SimulatedLink
        if not isinstance(sender, str) or not isinstance(receiver, str) or not sender or not receiver:
            raise ValueError("sender and receiver must be non-empty strings")
        for node in (sender, receiver):
            if node not in self.mac.positions:
                raise ValueError(f"node {node!r} has no position")
        if isinstance(seq, (bool, np.bool_)) or int(seq) != seq or not 0 <= seq < SEQ_MOD:
            raise ValueError("seq must be an integer in [0, 2**32)")
        send_time = float(send_time)
        if not math.isfinite(send_time) or send_time < 0:
            raise ValueError("send_time must be finite and nonnegative")
        route = (sender, receiver)
        if send_time < self._last_send.get(route, 0.0) or send_time < self.mac.now_s:
            raise ValueError("send_time must be nondecreasing and not before the model clock")
        self._last_send[route] = send_time
        self.mac.run_until(send_time)
        dst = receiver if self.cfg.unicast else BROADCAST
        frame = Frame(sender, dst, self.cfg.payload_bytes, send_time, seq=int(seq), meta={"receiver": receiver})
        frame_id = self.mac.enqueue(sender, frame, send_time)
        event = PacketEvent(sender, receiver, int(seq), send_time, None, outcome="pending",
                            payload_bytes=self.cfg.payload_bytes)
        self._by_frame[frame_id] = event
        self._unresolved[frame_id] = event
        self.events.append(event)
        return event

    # ---- MAC callback -----------------------------------------------------
    def _on_receive(self, _mac: MacSimulator, rx: str, frame: Frame, t: float) -> None:
        event = self._by_frame.get(frame.frame_id)
        if event is None or rx != event.receiver:
            return  # background traffic, or an overhearing node
        observation = self._tracker.observe(event.sender, event.seq)
        if not observation.accept:
            self.duplicates_suppressed += 1
            return
        if event.receive_time is not None:
            # Same frame again after a retry whose earlier copy was accepted.
            self.duplicates_suppressed += 1
            return
        event.receive_time = float(t)
        event.packet_age_s = float(t) - event.send_time
        event.outcome = "received"
        event.out_of_order = observation.status == "reordered"
        record = self._delivery_record(frame.frame_id, rx)
        if record is not None:
            event.rssi_dbm = float(record.rssi_dbm)
        self._ready.append(event)

    def _records_for(self, frame_id: int) -> list:
        """RxRecords of one frame, from an index kept up to date incrementally."""
        records = self.mac.rx_records
        while self._indexed < len(records):
            record = records[self._indexed]
            if record.frame_id in self._by_frame:
                self._rx_index.setdefault(record.frame_id, []).append(record)
            self._indexed += 1
        return self._rx_index.get(frame_id, [])

    def _delivery_record(self, frame_id: int, rx: str):
        for record in reversed(self._records_for(frame_id)):
            if record.rx == rx and record.outcome == "delivered":
                return record
        return None

    def _resolve_finished(self) -> None:
        """Mark frames the MAC is done with, and that never reached the receiver, as lost.

        A frame is only declared lost once the MAC has a final status for it
        AND any delivery still inside the receiver's processing delay would
        have landed by now.
        """
        processing = self.mac.mac.processing_delay_s
        for frame_id in list(self._unresolved):
            event = self._unresolved[frame_id]
            record = self.mac.frame_records[frame_id]
            if record.status == "pending":
                continue
            event.retry_count = max(0, int(record.tries) - 1)
            if event.receive_time is None:
                if record.last_tx_end_s is not None and self.mac.now_s < record.last_tx_end_s + processing:
                    continue
                event.outcome = "lost"
                event.loss_reason = self._loss_reason(frame_id, event.receiver, record.status)
            del self._unresolved[frame_id]

    def _loss_reason(self, frame_id: int, rx: str, status: str) -> str:
        if status == "queue_drop":
            return "queue_drop"
        outcomes = [r.outcome for r in self._records_for(frame_id) if r.rx == rx]
        return outcomes[-1] if outcomes else "below_sensitivity"

    # ---- SimulatedLink surface ---------------------------------------------
    def advance(self, now: float) -> list[PacketEvent]:
        now = float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("now must be finite and nonnegative")
        if now > self.mac.now_s:
            self.mac.run_until(now)
        self._resolve_finished()
        ready = [e for e in self._ready if e.receive_time <= now]
        self._ready = [e for e in self._ready if e.receive_time > now]
        return sorted(ready, key=lambda e: (e.receive_time, e.seq))

    def deliver_all(self) -> list[PacketEvent]:
        """Run the engine until every sent packet is resolved, then deliver all."""
        if not self._unresolved:
            return self.advance(self.mac.now_s)
        last_send = max((e.send_time for e in self._unresolved.values()), default=self.mac.now_s)
        t = max(self.mac.now_s, last_send)
        out: list[PacketEvent] = []
        while self._unresolved:
            t += self.cfg.drain_step_s
            if t > last_send + self.cfg.drain_limit_s:
                raise RuntimeError("radio model did not resolve every packet; check the MAC configuration")
            out.extend(self.advance(t))
        out.extend(self.advance(t))
        return sorted(out, key=lambda e: (e.receive_time, e.seq))


def events_to_trace(events: list[PacketEvent], *, trace_id: str, session_id: str = "sim-session-0",
                    channel: int | None = 1, notes: str | None = None) -> dict:
    """Write radio-model events as a synthetic ``zephyr-espnow-trace-1`` document.

    Used to test calibration tools on traces generated with KNOWN parameters.
    Status is always ``synthetic``; it never claims to be a measurement.
    """
    devices = sorted({e.sender for e in events} | {e.receiver for e in events})
    packets = []
    for i, e in enumerate(events):
        if e.outcome not in ("received", "lost"):
            raise ValueError("resolve every event (deliver_all) before exporting")
        packets.append({
            "packet_id": f"p{i:07d}",
            "sender_id": e.sender,
            "receiver_id": e.receiver,
            "sender_session_id": session_id,
            "seq": int(e.seq),
            "send_time_s": float(e.send_time),
            "outcome": e.outcome,
            "receive_time_s": None if e.receive_time is None else float(e.receive_time),
            "callback_time_s": None,
            "rssi_dbm": None if e.rssi_dbm is None else float(e.rssi_dbm),
            "payload_bytes": None if e.payload_bytes is None else int(e.payload_bytes),
            "retry_count": None if e.retry_count is None else int(e.retry_count),
            "loss_reason": e.loss_reason if e.outcome == "lost" else None,
            "unknown_reason": None,
            "duplicate": None,
            "out_of_order": None,
        })
    return {
        "schema": "zephyr-espnow-trace-1",
        "trace_id": trace_id,
        "status": "synthetic",
        "clock_domain": "shared_monotonic",
        "source": None,
        "code_revision": None,
        "clock_uncertainty_s": 0.0,
        "notes": notes,
        "capture": {"transport": "ESP-NOW", "device_ids": devices, "firmware": "simulated",
                    "channel": channel, "phy": "simulated 802.11b 1 Mbps"},
        "packets": packets,
    }
