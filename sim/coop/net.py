"""Precomputed simulated delivery tables for cooperative observer traffic.

The abstract path uses one seeded :class:`SimulatedLink` for every directed
route.  The radio path composes Run C's :class:`MacSimulator` and
:class:`Channel` directly.  A broadcast frame is enqueued once and the MAC
callback records each receiver independently, which is the same one-to-all
handling used by the Run C network layer.

With ``immediate_access=True`` (the Run C default), perfectly synchronized
captures contend at the same instant and collide. In this model synchronized
peer delivery is 0.000, versus 0.909 with ``immediate_access=False`` (seed 1,
four observers 3 m from gs, 10 Hz, 32 B, 12 s; other seeds give 0.88 to 0.94).
Whether
ESP-NOW's MAC behaves like immediate access is UNCONFIRMED. Dephasing sender
captures over one period avoids that synchronized collision pattern.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from sim.link import LinkConfig, SimulatedLink
from sim.radio.channel import Channel
from sim.radio.mac import BROADCAST, Frame, MacSimulator, ack_airtime_s, data_airtime_s
from sim.radio.params import channel_preset, default_mac, default_phy
from sim.s6_recovery import _burst_start_probability


_TIME_TOLERANCE_S = 1e-12


@dataclass(frozen=True)
class TrafficSpec:
    pattern: str
    observers: tuple[str, ...]
    rate_hz: float = 10.0
    message_bytes: int = 32
    duration_s: float = 12.0
    phase_offsets_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.pattern not in {"star", "peer"}:
            raise ValueError("pattern must be star or peer")
        if not self.observers or len(set(self.observers)) != len(self.observers):
            raise ValueError("observers must be a nonempty tuple of unique names")
        if any(not isinstance(name, str) or not name for name in self.observers):
            raise ValueError("observer names must be nonempty strings")
        if not math.isfinite(float(self.rate_hz)) or self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive and finite")
        if isinstance(self.message_bytes, bool) or int(self.message_bytes) != self.message_bytes or self.message_bytes < 1:
            raise ValueError("message_bytes must be a positive integer")
        if not math.isfinite(float(self.duration_s)) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        if self.phase_offsets_s is not None:
            if len(self.phase_offsets_s) != len(self.observers):
                raise ValueError("phase_offsets_s must match observers")
            if any(not math.isfinite(float(x)) or float(x) < 0 for x in self.phase_offsets_s):
                raise ValueError("phase offsets must be finite and nonnegative")


def default_phases(observers: tuple[str, ...], rate_hz: float, seed: int) -> tuple[float, ...]:
    """Return deterministic fixed phases for unsynchronized simulated cameras.

    ``o1`` is anchored at phase zero. Each other observer receives a fixed
    uniform phase in one capture period from its own SeedSequence.
    """
    if not observers or len(set(observers)) != len(observers):
        raise ValueError("observers must be a nonempty tuple of unique names")
    if not math.isfinite(float(rate_hz)) or rate_hz <= 0:
        raise ValueError("rate_hz must be positive and finite")
    period = 1.0 / float(rate_hz)
    phases = []
    for i, name in enumerate(observers):
        if name == "o1":
            phases.append(0.0)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3000 + i]))
            phases.append(float(rng.uniform(0.0, period)))
    return tuple(phases)


@dataclass(frozen=True)
class LinkSpec:
    mode: str
    delay_s: float
    loss_probability: float
    loss_model: str = "independent"
    jitter_s: float = 0.0
    radio_preset: str = "indoor_gym"
    positions: dict[str, tuple[float, float, float]] | None = None
    background: tuple = ()
    immediate_access: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"abstract", "radio"}:
            raise ValueError("mode must be abstract or radio")
        for name in ("delay_s", "loss_probability", "jitter_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.loss_probability > 1:
            raise ValueError("loss_probability must be at most one")
        if self.loss_model not in {"independent", "burst"}:
            raise ValueError("loss_model must be independent or burst")
        if not isinstance(self.immediate_access, (bool, np.bool_)):
            raise ValueError("immediate_access must be boolean")
        if self.mode == "radio" and self.positions is None:
            raise ValueError("radio mode requires positions")
        if self.mode == "radio" and self.jitter_s > 0:
            raise ValueError("radio mode does not support jitter_s")


@dataclass(frozen=True)
class Delivery:
    sender: str
    receiver: str
    seq: int
    send_time_s: float
    receive_time_s: float | None
    loss_reason: str | None


class DeliveryTable(dict[tuple[str, str], tuple[Delivery, ...]]):
    """Immutable-by-convention route table with aggregate metrics."""

    def __init__(self, routes: dict[tuple[str, str], tuple[Delivery, ...]], *, pattern: str,
                 busy_fraction: float | None, duration_s: float):
        super().__init__(routes)
        self.pattern = pattern
        self.busy_fraction = busy_fraction
        self.duration_s = float(duration_s)

    def summary(self) -> dict[str, Any]:
        rows = [item for key in sorted(self) for item in self[key]]
        delivered = [item for item in rows if item.receive_time_s is not None]
        latencies = [item.receive_time_s - item.send_time_s for item in delivered]
        reasons = Counter(item.loss_reason for item in rows if item.loss_reason is not None)
        late_end = int(reasons.get("late_end", 0))
        radio_denominator = len(rows) - late_end
        return {
            "pattern": self.pattern,
            "messages": len(rows),
            "delivered": len(delivered),
            "delivery_ratio": (len(delivered) / len(rows) if rows else 0.0),
            "radio_delivery_ratio": (len(delivered) / radio_denominator if radio_denominator else 0.0),
            "mean_latency_s": (float(np.mean(latencies)) if latencies else None),
            "p95_latency_s": (float(np.percentile(latencies, 95)) if latencies else None),
            "busy_fraction": self.busy_fraction,
            "loss_reasons": {str(k): int(reasons[k]) for k in sorted(reasons)},
        }


def _routes(traffic: TrafficSpec) -> list[tuple[str, str]]:
    if traffic.pattern == "star":
        pairs = [(sender, "gs") for sender in traffic.observers]
    else:
        pairs = [(sender, receiver) for sender in traffic.observers
                 for receiver in (*traffic.observers, "gs") if receiver != sender]
    return sorted(pairs)


def _capture_times(traffic: TrafficSpec, observer_index: int) -> list[float]:
    offsets = traffic.phase_offsets_s or (0.0,) * len(traffic.observers)
    period = 1.0 / float(traffic.rate_hz)
    count = int(round(float(traffic.duration_s) / period))
    # This matches s6_recovery: arange(round(duration / period) + 1) * period.
    raw = np.arange(count + 1, dtype=float) * period + float(offsets[observer_index])
    return [float(t) for t in raw if t <= traffic.duration_s + _TIME_TOLERANCE_S]


def _abstract_table(traffic: TrafficSpec, link: LinkSpec, seed: int) -> DeliveryTable:
    route_list = _routes(traffic)
    out: dict[tuple[str, str], tuple[Delivery, ...]] = {}
    one_agent_star = traffic.pattern == "star" and len(traffic.observers) == 1
    observer_index = {name: i for i, name in enumerate(traffic.observers)}
    end = float(traffic.duration_s)
    for route_index, (sender, receiver) in enumerate(route_list):
        if one_agent_star:
            route_seed = int(np.random.SeedSequence(seed).generate_state(2)[0])
        else:
            route_seed = int(np.random.SeedSequence([int(seed), 1000 + route_index]).generate_state(1)[0])
        if link.loss_model == "burst":
            # Keep burst transitions aligned with sim/link.py's Gilbert-Elliott logic.
            burst_end = 1.0 / 5.0
            config = LinkConfig(delay_s=link.delay_s, delay_jitter_s=link.jitter_s, loss_model="burst",
                                burst_start_probability=_burst_start_probability(link.loss_probability, burst_end),
                                burst_end_probability=burst_end)
        else:
            config = LinkConfig(delay_s=link.delay_s, delay_jitter_s=link.jitter_s, loss_model="independent",
                                loss_probability=link.loss_probability)
        model = SimulatedLink(config, seed=route_seed)
        times = _capture_times(traffic, observer_index[sender])
        events = [model.send(sender, receiver, seq, t) for seq, t in enumerate(times)]
        model.deliver_all()
        deliveries = []
        for event in events:
            if event.receive_time is None:
                deliveries.append(Delivery(sender, receiver, event.seq, float(event.send_time), None, "link"))
            elif event.receive_time > end + _TIME_TOLERANCE_S:
                deliveries.append(Delivery(sender, receiver, event.seq, float(event.send_time), None, "late_end"))
            else:
                deliveries.append(Delivery(sender, receiver, event.seq, float(event.send_time),
                                           float(event.receive_time), None))
        out[(sender, receiver)] = tuple(deliveries)
    return DeliveryTable(out, pattern=traffic.pattern, busy_fraction=None, duration_s=end)


def _background_values(item: Any) -> tuple[str, str, float, int, float, float, float]:
    if isinstance(item, tuple):
        if len(item) != 7:
            raise ValueError("background entries must have seven fields")
        return item
    values = tuple(vars(item).values())
    if len(values) != 7:
        raise ValueError("background entries must have seven fields")
    return values


def _radio_table(traffic: TrafficSpec, link: LinkSpec, seed: int) -> DeliveryTable:
    positions = dict(link.positions or {})
    required = set(traffic.observers) | {"gs"}
    missing = sorted(required - set(positions))
    if missing:
        raise ValueError(f"positions missing nodes: {missing}")
    for item in link.background:
        node, dst, rate, size, start, stop, jitter = _background_values(item)
        if node not in positions:
            raise ValueError(f"background node {node!r} has no position")
        if dst != "*" and dst not in positions:
            raise ValueError(f"background receiver {dst!r} has no position")
        del dst, rate, size, start, stop, jitter
    phy = default_phy()
    channel_seq, mac_seq = np.random.SeedSequence(int(seed)).spawn(2)
    channel = Channel(channel_preset(link.radio_preset), np.random.default_rng(channel_seq))
    mac = MacSimulator(positions, channel, phy, default_mac(unicast_retry_limit=5,
                       immediate_access=link.immediate_access),
                       seed=int(mac_seq.generate_state(1)[0]))
    for item in link.background:
        node, dst, rate, size, start, stop, jitter = _background_values(item)
        mac.add_periodic_source(node, dst, float(rate), int(size), start_s=float(start),
                                stop_s=float(stop), jitter_s=float(jitter))

    route_list = _routes(traffic)
    extra_rng = {route: np.random.default_rng(np.random.SeedSequence([int(seed), 2000 + i]))
                 for i, route in enumerate(route_list)}
    extra_state = {route: [False, _burst_start_probability(link.loss_probability, 1.0 / 5.0), 1.0 / 5.0]
                   for route in route_list}
    own_frames: dict[int, tuple[str, int, float]] = {}
    own_ids: dict[tuple[str, int], int] = {}
    radio_times: dict[tuple[str, int, str], float] = {}
    radio_records: dict[tuple[str, int, str], list[str]] = {}

    def on_receive(_sim: MacSimulator, receiver: str, frame: Frame, when: float) -> None:
        info = own_frames.get(frame.frame_id)
        if info is None:
            return
        sender, seq, _ = info
        key = (sender, seq, receiver)
        radio_times.setdefault(key, float(when))

    mac.on_receive = on_receive
    sends: list[tuple[float, str, str, int]] = []
    indices = {name: i for i, name in enumerate(traffic.observers)}
    for sender in traffic.observers:
        for seq, when in enumerate(_capture_times(traffic, indices[sender])):
            if traffic.pattern == "star":
                sends.append((when, sender, "gs", seq))
            else:
                sends.append((when, sender, BROADCAST, seq))
    sends.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    for when, sender, receiver, seq in sends:
        mac.run_until(float(when))
        frame = Frame(sender, receiver, int(traffic.message_bytes), float(when), seq=int(seq),
                      meta={"route": receiver})
        frame_id = mac.enqueue(sender, frame, float(when))
        own_frames[frame_id] = (sender, seq, float(when))
        own_ids[(sender, seq)] = frame_id
    mac.run_until(float(traffic.duration_s) + 60.0)
    frame_status = {frame_id: mac.frame_records[frame_id].status for frame_id in own_frames}

    for record in mac.rx_records:
        info = own_frames.get(record.frame_id)
        if info is None:
            continue
        sender, seq, _ = info
        key = (sender, seq, record.rx)
        radio_records.setdefault(key, []).append(str(record.outcome))

    out: dict[tuple[str, str], tuple[Delivery, ...]] = {}
    for sender, receiver in route_list:
        deliveries = []
        for seq, send_time in enumerate(_capture_times(traffic, indices[sender])):
            key = (sender, seq, receiver)
            receive = radio_times.get(key)
            rng = extra_rng[(sender, receiver)]
            if link.loss_model == "independent":
                extra_hit = float(rng.random()) < link.loss_probability
            else:
                state = extra_state[(sender, receiver)]
                if not state[0] and float(rng.random()) < state[1]:
                    state[0] = True
                extra_hit = bool(state[0])
                if state[0] and float(rng.random()) < state[2]:
                    state[0] = False
            reason: str | None = None
            if receive is None:
                status = frame_status.get(own_ids.get((sender, seq), -1), "")
                outcomes = radio_records.get(key, [])
                if status == "queue_drop":
                    reason = "queue_drop"
                elif outcomes:
                    reason = outcomes[-1] if outcomes[-1] in {"collision", "channel_error", "below_sensitivity", "queue_drop"} else "collision"
                else:
                    reason = "below_sensitivity"
            else:
                receive = float(receive) + link.delay_s
                if receive > traffic.duration_s + _TIME_TOLERANCE_S:
                    receive = None
                    reason = "late_end"
                elif extra_hit:
                    receive = None
                    reason = "extra"
            deliveries.append(Delivery(sender, receiver, seq, float(send_time), receive, reason))
        out[(sender, receiver)] = tuple(deliveries)
    busy = mac.channel_busy_fraction(0.0, max(float(traffic.duration_s), 1e-12))
    return DeliveryTable(out, pattern=traffic.pattern, busy_fraction=float(busy), duration_s=traffic.duration_s)


def make_deliveries(traffic: TrafficSpec, link: LinkSpec, seed: int) -> DeliveryTable:
    """Build a deterministic simulated table before any tracking code runs."""
    if not isinstance(traffic, TrafficSpec) or not isinstance(link, LinkSpec):
        raise TypeError("traffic and link must be their specification types")
    return _abstract_table(traffic, link, int(seed)) if link.mode == "abstract" else _radio_table(traffic, link, int(seed))


def predicted_channel_use(n: int, rate_hz: float, message_bytes: int, unicast: bool, phy=None) -> float:
    """Estimate offered medium fraction using the Run C airtime convention.

    This is the same DIFS plus 15.5-slot mean backoff calculation as
    ``tools/c_exp014_mac.py:predicted_channel_use``. It includes idle DIFS and
    backoff time, so it is not directly comparable to ``busy_fraction``, which
    measures on-air time only.
    """
    if isinstance(n, bool) or int(n) != n or n < 1 or not math.isfinite(float(rate_hz)) or rate_hz < 0:
        raise ValueError("n must be positive and rate_hz nonnegative")
    if isinstance(message_bytes, bool) or int(message_bytes) != message_bytes or message_bytes < 0:
        raise ValueError("message_bytes must be a nonnegative integer")
    phy = default_phy() if phy is None else phy
    per_frame = phy.difs_s + data_airtime_s(message_bytes, phy) + 15.5 * phy.slot_s
    if unicast:
        per_frame += phy.sifs_s + ack_airtime_s(phy)
    return float(int(n) * float(rate_hz) * per_frame)

