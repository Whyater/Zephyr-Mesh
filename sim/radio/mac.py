"""Deterministic event driven 802.11-like MAC for the radio simulator.

Event queue correctness: events are processed in nondecreasing time. Ties are
broken by a fixed priority and then by insertion counter. Transmission ends
run before starts at the same instant, so back-to-back frames do not overlap;
an overlap exists only when two intervals intersect with positive length. Each
frame reaches exactly one terminal FrameRecord status. Carrier sense uses only
transmissions whose received power at the sensing node is at least the CCA
threshold. Backoff countdown freezes while busy and resumes after DIFS idle.

Access rules (802.11 DCF, simplified):
- A frame that reaches the head of the queue always waits DIFS of continuous
  idle medium first. If the medium was idle at that moment and no backoff is
  owed, it transmits right after that DIFS (immediate access). If the medium
  is busy at that moment, or turns busy during the DIFS, a backoff is owed.
- After every transmission, a backoff is owed if another frame is already
  queued (back-to-back frames always back off). If the queue is empty, the
  802.11 post-transmission backoff is assumed to finish before the next
  arrival, which holds when arrivals are far apart compared with a few slots.
- A node whose countdown (or DIFS) ends at the same instant another node
  starts cannot sense that start in time, so both transmit and collide. This is
  where most 802.11 collisions come from.
- Backoff progress is kept across unrelated events: a transmission the node
  cannot sense, or a new frame joining its queue, does not restart it.
- ACKs occupy the medium: nodes that sense an ACK defer until it ends plus
  DIFS, like any other transmission.
The engine never schedules an event in the past. Consequently, calling
run_until(t) followed by run_until(t2) gives the same result as one call to
run_until(t2), for identical enqueues.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
import math
from typing import Callable, Any

import numpy as np


BROADCAST = "*"


@dataclass(frozen=True)
class PhyConfig:
    rate_mbps: float = 1.0
    modulation: str = "dsss"
    preamble: str = "long"
    slot_s: float = 20e-6
    sifs_s: float = 10e-6
    cw_min: int = 31
    cw_max: int = 1023
    mac_overhead_bytes: int = 43
    ack_bytes: int = 14
    ack_rate_mbps: float = 1.0
    max_payload_bytes: int = 250

    @property
    def difs_s(self) -> float:
        return self.sifs_s + 2 * self.slot_s

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_mbps) or self.rate_mbps <= 0:
            raise ValueError("rate_mbps must be positive")
        if self.modulation not in ("dsss", "ofdm"):
            raise ValueError("unknown modulation")
        if self.preamble not in ("long", "short"):
            raise ValueError("unknown preamble")
        if self.slot_s <= 0 or self.sifs_s < 0 or self.ack_rate_mbps <= 0:
            raise ValueError("invalid PHY timing")
        if self.cw_min < 0 or self.cw_max < self.cw_min:
            raise ValueError("invalid contention window")
        if self.mac_overhead_bytes < 0 or self.ack_bytes < 1 or self.max_payload_bytes < 1:
            raise ValueError("invalid PHY byte count")


def frame_bytes(payload_bytes: int, phy: PhyConfig) -> int:
    if int(payload_bytes) != payload_bytes or payload_bytes < 0:
        raise ValueError("payload_bytes must be a nonnegative integer")
    if payload_bytes > phy.max_payload_bytes:
        raise ValueError("payload exceeds PHY maximum")
    return int(payload_bytes) + phy.mac_overhead_bytes


def airtime_s(n_bytes_mpdu: int, rate_mbps: float, phy: PhyConfig) -> float:
    if int(n_bytes_mpdu) != n_bytes_mpdu or n_bytes_mpdu < 1:
        raise ValueError("n_bytes_mpdu must be positive")
    if not math.isfinite(rate_mbps) or rate_mbps <= 0:
        raise ValueError("rate_mbps must be positive")
    if phy.modulation == "dsss":
        pre = 192e-6 if phy.preamble == "long" else 96e-6
        return pre + 8 * int(n_bytes_mpdu) / (rate_mbps * 1e6)
    symbols = math.ceil((16 + 8 * int(n_bytes_mpdu) + 6) / (4 * rate_mbps))
    return 20e-6 + 4e-6 * symbols


def data_airtime_s(payload_bytes: int, phy: PhyConfig) -> float:
    return airtime_s(frame_bytes(payload_bytes, phy), phy.rate_mbps, phy)


def ack_airtime_s(phy: PhyConfig) -> float:
    return airtime_s(phy.ack_bytes, phy.ack_rate_mbps, phy)


@dataclass(frozen=True)
class MacConfig:
    carrier_sense: bool = True
    backoff: bool = True
    slotted: bool = False
    slot_len_s: float | None = None
    unicast_retry_limit: int = 0
    ack_timeout_s: float | None = None
    queue_capacity: int = 16
    processing_delay_s: float = 0.0
    cca_threshold_dbm: float = -82.0
    capture_threshold_db: float | None = None
    interference_floor_dbm: float = -100.0
    immediate_access: bool = True

    def __post_init__(self) -> None:
        if self.slotted and self.carrier_sense:
            raise ValueError("slotted mode requires carrier_sense=False")
        if self.slot_len_s is not None and self.slot_len_s <= 0:
            raise ValueError("slot_len_s must be positive")
        if self.unicast_retry_limit < 0 or self.queue_capacity < 0:
            raise ValueError("retry limit and queue capacity must be nonnegative")
        if self.processing_delay_s < 0:
            raise ValueError("processing delay must be nonnegative")


@dataclass
class Frame:
    src: str
    dst: str
    payload_bytes: int
    created_s: float
    seq: int = 0
    origin: str | None = None
    origin_seq: int | None = None
    ttl: int = 0
    hops: int = 0
    meta: dict = field(default_factory=dict)
    frame_id: int = -1


@dataclass
class RxRecord:
    frame_id: int
    src: str
    dst: str
    rx: str
    seq: int
    origin: str | None
    origin_seq: int | None
    created_s: float
    tx_start_s: float
    tx_end_s: float
    deliver_s: float | None
    outcome: str
    attempt: int
    rssi_dbm: float
    snr_db: float


@dataclass
class FrameRecord:
    frame_id: int
    src: str
    dst: str
    seq: int
    created_s: float
    status: str
    tries: int
    first_tx_s: float | None
    last_tx_end_s: float | None
    queue_wait_s: float | None


@dataclass
class _Node:
    queue: list[Frame] = field(default_factory=list)
    current: Frame | None = None
    attempt: int = 0
    cw: int = 0
    needs_backoff: bool = False
    phase: str = "idle"
    remaining_slots: int = 0
    backoff_pending: bool = False
    count_start: float = 0.0
    fire_at: float = 0.0
    token: int = 0
    busy_until: float = 0.0


@dataclass
class _Tx:
    src: str
    frame: Frame | None
    start: float
    end: float
    kind: str
    attempt: int
    powers: dict[str, float]
    samples: dict[str, Any]
    ack_parent: int | None = None
    ack_sender: str | None = None
    interference: dict[str, list[float]] = field(default_factory=dict)
    self_collision: set[str] = field(default_factory=set)


class MacSimulator:
    """Run a deterministic, event-driven MAC over fixed node positions."""

    # Lower value runs first. Ends precede starts at equal time.
    _P_TX_END = 0
    _P_DELIVER = 1
    _P_TX_START = 2
    _P_ENQUEUE = 3
    _P_SOURCE = 4
    _P_TIMEOUT = 5

    def __init__(self, positions: dict[str, tuple[float, float, float]], channel, phy: PhyConfig, mac: MacConfig,
                 *, seed: int, walls: dict[tuple[str, str], int] | None = None, noise_floor_fn=None):
        self.positions = {str(k): tuple(float(x) for x in v) for k, v in positions.items()}
        if not self.positions:
            raise ValueError("positions cannot be empty")
        for p in self.positions.values():
            if len(p) != 3 or any(not math.isfinite(x) for x in p):
                raise ValueError("positions must be finite 3-tuples")
        self.channel = channel
        self.phy = phy
        self.mac = mac
        self.walls = walls or {}
        self.noise_floor_fn = noise_floor_fn
        ss = np.random.SeedSequence(seed)
        b, e, s = ss.spawn(3)
        self._backoff_rng = np.random.default_rng(b)
        self._error_rng = np.random.default_rng(e)
        self._source_rng = np.random.default_rng(s)
        self._nodes = {n: _Node(cw=phy.cw_min) for n in sorted(self.positions)}
        self._heap: list[tuple[float, int, int, str, Any]] = []
        self._counter = 0
        self._next_frame_id = 0
        self._active: list[_Tx] = []
        self._ended_same_time: list[_Tx] = []
        self._last_event_time = 0.0
        self._intervals: list[tuple[float, float]] = []
        self._now = 0.0
        self.rx_records: list[RxRecord] = []
        self.frame_records: dict[int, FrameRecord] = {}
        self.on_receive: Callable[[MacSimulator, str, Frame, float], None] | None = None

    @property
    def now_s(self) -> float:
        return self._now

    def _push(self, when: float, priority: int, kind: str, data: Any) -> None:
        if not math.isfinite(when):
            raise ValueError("event time must be finite")
        if when < self._now - 1e-12:
            raise RuntimeError("event scheduled in the past")
        self._counter += 1
        heapq.heappush(self._heap, (max(when, self._now), priority, self._counter, kind, data))

    def enqueue(self, node: str, frame: Frame, t_s: float) -> int:
        if node not in self._nodes:
            raise ValueError("unknown node")
        if not math.isfinite(t_s) or t_s < self._now - 1e-12:
            raise ValueError("enqueue time must be at or after now")
        if frame.src != node:
            raise ValueError("frame source does not match node")
        frame_bytes(frame.payload_bytes, self.phy)
        fid = self._next_frame_id
        self._next_frame_id += 1
        if not math.isfinite(frame.created_s):
            raise ValueError("frame created_s must be finite")
        f = replace(frame, frame_id=fid)
        self.frame_records[fid] = FrameRecord(fid, f.src, f.dst, f.seq, f.created_s, "pending", 0, None, None, None)
        self._push(float(t_s), self._P_ENQUEUE, "enqueue", (node, f))
        return fid

    def add_periodic_source(self, node, dst, rate_hz, payload_bytes, *, start_s=0.0, stop_s=math.inf,
                            phase_s=None, jitter_s=0.0) -> None:
        if node not in self._nodes or rate_hz <= 0 or not math.isfinite(rate_hz):
            raise ValueError("invalid periodic source")
        if start_s < self._now or stop_s < start_s or jitter_s < 0:
            raise ValueError("invalid source interval")
        frame_bytes(payload_bytes, self.phy)
        period = 1.0 / rate_hz
        if phase_s is None:
            phase_s = float(self._source_rng.uniform(0.0, period))
        if phase_s < 0 or phase_s >= period:
            raise ValueError("phase_s must be in one period")
        first = float(start_s + phase_s)
        if jitter_s:
            first += float(self._source_rng.uniform(0.0, jitter_s))
        self._push(first, self._P_SOURCE, "periodic", (node, dst, rate_hz, payload_bytes, start_s, stop_s, phase_s, jitter_s, 0))

    def add_poisson_source(self, node, dst, rate_hz, payload_bytes, *, start_s=0.0, stop_s=math.inf) -> None:
        if node not in self._nodes or rate_hz <= 0 or not math.isfinite(rate_hz):
            raise ValueError("invalid Poisson source")
        if start_s < self._now or stop_s < start_s:
            raise ValueError("invalid source interval")
        frame_bytes(payload_bytes, self.phy)
        first = float(start_s + self._source_rng.exponential(1.0 / rate_hz))
        self._push(first, self._P_SOURCE, "poisson", (node, dst, rate_hz, payload_bytes, start_s, stop_s))

    def _distance(self, a: str, b: str) -> float:
        p, q = self.positions[a], self.positions[b]
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))

    def _wall_count(self, a: str, b: str) -> int:
        return self.walls.get((a, b), self.walls.get((b, a), 0))

    def _sensitivity(self) -> float | None:
        x = getattr(self.channel, "sensitivity_dbm", None)
        if x is None:
            cfg = getattr(self.channel, "cfg", None)
            x = getattr(cfg, "sensitivity_dbm", None) if cfg is not None else None
        return float(x) if x is not None else None

    def _power(self, tx: str, rx: str, t: float) -> float:
        return float(self.channel.received_power_dbm(tx, rx, self._distance(tx, rx), t,
                                                     walls=self._wall_count(tx, rx)))

    def _sensed(self, tx: _Tx, node: str) -> bool:
        return node in tx.powers and tx.powers[node] >= self.mac.cca_threshold_dbm

    def _schedule_node(self, node: str, when: float | None = None) -> None:
        st = self._nodes[node]
        if st.current is not None or not st.queue:
            return
        now = self._now if when is None else max(when, self._now)
        if self.mac.carrier_sense:
            if st.phase in ("ready", "countdown"):
                # Already scheduled. A sensed transmission would have moved it to "busy".
                return
            if now < st.busy_until - 1e-12:
                if self.mac.backoff:
                    st.needs_backoff = True
                st.phase = "busy"
                st.token += 1
                self._push(st.busy_until, self._P_TX_START, "node_ready", (node, st.token))
                return
            base = max(now, st.busy_until) + self.phy.difs_s
            if self.mac.backoff and (st.needs_backoff or not self.mac.immediate_access):
                if not st.backoff_pending:
                    st.remaining_slots = int(self._backoff_rng.integers(0, st.cw + 1))
                    st.backoff_pending = True
                st.count_start = base
                st.phase = "countdown"
                st.fire_at = base + st.remaining_slots * self.phy.slot_s
            else:
                st.phase = "ready"
                st.fire_at = base
            st.token += 1
            self._push(st.fire_at, self._P_TX_START, "node_ready", (node, st.token))
            return
        # ALOHA path
        base = now
        if self.mac.slotted:
            slot = self.mac.slot_len_s or data_airtime_s(st.queue[0].payload_bytes, self.phy)
            base = math.ceil(base / slot - 1e-9) * slot
        st.phase = "ready"
        st.token += 1
        self._push(base, self._P_TX_START, "node_ready", (node, st.token))

    def _freeze_countdowns(self, now: float, nodes: list[str] | None = None) -> None:
        """Nodes that sense a new transmission stop counting (or stop their DIFS wait).

        A node scheduled to fire at this very instant is not frozen: it cannot
        sense a start in the same instant, so it transmits too and they collide.
        """
        for node in sorted(self._nodes if nodes is None else nodes):
            st = self._nodes[node]
            if st.phase not in ("countdown", "ready") or st.fire_at <= now + 1e-12:
                continue
            if st.phase == "countdown":
                elapsed = max(0, int(math.floor((now - st.count_start + 1e-12) / self.phy.slot_s)))
                st.remaining_slots = max(0, st.remaining_slots - elapsed)
            elif self.mac.backoff:
                st.needs_backoff = True
            st.phase = "busy"
            st.token += 1

    def _start_tx(self, node: str) -> None:
        st = self._nodes[node]
        if st.current is not None or not st.queue:
            return
        if self.mac.carrier_sense and any(self._sensed(tx, node) and tx.end > self._now + 1e-12
                                          and tx.start < self._now - 1e-12 for tx in self._active):
            st.phase = "busy"
            if self.mac.backoff:
                st.needs_backoff = True
            self._schedule_node(node, self._now)
            return
        frame = st.queue.pop(0)
        st.backoff_pending = False
        st.remaining_slots = 0
        st.current = frame
        st.attempt += 1
        if st.attempt == 1:
            rec = self.frame_records[frame.frame_id]
            self.frame_records[frame.frame_id] = replace(rec, first_tx_s=self._now)
        rec = self.frame_records[frame.frame_id]
        self.frame_records[frame.frame_id] = replace(rec, tries=st.attempt)
        duration = data_airtime_s(frame.payload_bytes, self.phy)
        powers: dict[str, float] = {}
        samples: dict[str, Any] = {}
        fb = frame_bytes(frame.payload_bytes, self.phy)
        for rx in sorted(self.positions):
            if rx == node:
                continue
            try:
                sample = self.channel.sample(node, rx, self._distance(node, rx), self._now,
                                             frame_bytes=fb, walls=self._wall_count(node, rx),
                                             noise_floor_dbm=self.noise_floor_fn(self._now, rx) if self.noise_floor_fn else None)
            except TypeError:
                sample = self.channel.sample(node, rx, self._distance(node, rx), self._now, frame_bytes=fb)
            samples[rx] = sample
            powers[rx] = float(sample.rssi_dbm)
        tx = _Tx(node, frame, self._now, self._now + duration, "data", st.attempt, powers, samples)
        self._register_overlap(tx)
        self._active.append(tx)
        self._intervals.append((tx.start, tx.end))
        if self.mac.carrier_sense:
            st.busy_until = max(st.busy_until, tx.end)
        sensed_nodes = []
        for n in sorted(self._nodes):
            if n != node and self._sensed(tx, n):
                self._nodes[n].busy_until = max(self._nodes[n].busy_until, tx.end)
                sensed_nodes.append(n)
        self._freeze_countdowns(self._now, sensed_nodes)
        self._push(tx.end, self._P_TX_END, "tx_end", tx)

    def _overlap(self, a: _Tx, b: _Tx) -> bool:
        # 1 ns tolerance: at t = 16 s one float step is 3.6e-15 s, so back-to-back
        # frames computed by different sums must not count as overlapping.
        return max(a.start, b.start) < min(a.end, b.end) - 1e-9

    def _collision(self, tx: _Tx, rx: str) -> bool:
        competitors: list[Any] = list(tx.interference.get(rx, []))
        for other in self._active + self._ended_same_time:
            if other is tx or not self._overlap(tx, other):
                continue
            p = other.powers.get(rx, -math.inf)
            if p >= self.mac.interference_floor_dbm:
                competitors.append((p, other))
        if not competitors:
            return False
        if self.mac.capture_threshold_db is None:
            return True
        main = tx.powers.get(rx, -math.inf)
        strongest = max(p if isinstance(p, (int, float)) else p[0] for p in competitors)
        return main - strongest < self.mac.capture_threshold_db

    def _write_rx(self, tx: _Tx, rx: str, outcome: str, sample: Any) -> None:
        f = tx.frame
        assert f is not None
        deliver = tx.end + self.mac.processing_delay_s if outcome == "delivered" else None
        rec = RxRecord(f.frame_id, f.src, f.dst, rx, f.seq, f.origin, f.origin_seq, f.created_s,
                       tx.start, tx.end, deliver, outcome, tx.attempt - 1,
                       float(sample.rssi_dbm), float(sample.snr_db))
        self.rx_records.append(rec)
        if deliver is not None:
            self._push(deliver, self._P_DELIVER, "deliver", (rx, f, deliver))

    def _data_end(self, tx: _Tx) -> None:
        f = tx.frame
        assert f is not None
        st = self._nodes[tx.src]
        intended = [f.dst] if f.dst != BROADCAST else [n for n in sorted(self.positions) if n != f.src]
        delivered_intended = False
        for rx in intended:
            sample = tx.samples.get(rx)
            if sample is None:
                continue
            sens = self._sensitivity()
            if sens is not None and float(sample.rssi_dbm) < sens - 10.0:
                self._write_rx(tx, rx, "below_sensitivity", sample)
                continue
            if rx in tx.self_collision or any(other.src == rx and self._overlap(tx, other)
                   for other in self._active + self._ended_same_time if other is not tx):
                outcome = "collision"
            elif self._collision(tx, rx):
                outcome = "collision"
            elif float(self._error_rng.random()) < float(sample.per):
                outcome = "channel_error"
            else:
                outcome = "delivered"
            self._write_rx(tx, rx, outcome, sample)
            if rx == f.dst and outcome == "delivered":
                delivered_intended = True
        rec = self.frame_records[f.frame_id]
        self.frame_records[f.frame_id] = replace(
            rec,
            last_tx_end_s=tx.end,
            queue_wait_s=rec.queue_wait_s if rec.queue_wait_s is not None else max(
                0.0, tx.start - f.created_s - (self.phy.difs_s if self.mac.carrier_sense else 0.0)
            ),
        )
        if f.dst == BROADCAST:
            self.frame_records[f.frame_id] = replace(self.frame_records[f.frame_id], status="sent")
            st.current = None; st.attempt = 0; st.cw = self.phy.cw_min; st.phase = "idle"; st.remaining_slots = 0
            st.needs_backoff = self.mac.backoff and bool(st.queue)
            self._schedule_node(tx.src, tx.end)
        elif delivered_intended:
            # ACK starts after SIFS. It is a real on-air interval and can collide.
            self._push(tx.end + self.phy.sifs_s, self._P_TX_START, "ack_start", tx)
            timeout = self.mac.ack_timeout_s
            if timeout is None:
                timeout = self.phy.sifs_s + ack_airtime_s(self.phy) + self.phy.slot_s
            st.phase = "ack_wait"
            st.token += 1
            self._push(tx.end + timeout, self._P_TIMEOUT, "ack_timeout", (tx, st.token))
        else:
            self._attempt_done(tx, False)

    def _ack_start(self, parent: _Tx) -> None:
        f = parent.frame
        assert f is not None
        duration = ack_airtime_s(self.phy)
        sender = parent.src
        powers, samples = {}, {}
        try:
            sample = self.channel.sample(f.dst, sender, self._distance(f.dst, sender), self._now,
                                         frame_bytes=self.phy.ack_bytes, walls=self._wall_count(f.dst, sender),
                                         noise_floor_dbm=self.noise_floor_fn(self._now, sender) if self.noise_floor_fn else None)
        except TypeError:
            sample = self.channel.sample(f.dst, sender, self._distance(f.dst, sender), self._now, frame_bytes=self.phy.ack_bytes)
        samples[sender] = sample; powers[sender] = float(sample.rssi_dbm)
        for n in sorted(self.positions):
            if n not in (f.dst, sender):
                powers[n] = self._power(f.dst, n, self._now)
        tx = _Tx(f.dst, None, self._now, self._now + duration, "ack", parent.attempt, powers, samples, f.frame_id, sender)
        self._register_overlap(tx)
        self._active.append(tx); self._intervals.append((tx.start, tx.end))
        if self.mac.carrier_sense:
            ack_node = self._nodes[f.dst]
            ack_node.busy_until = max(ack_node.busy_until, tx.end)
            sensed_nodes = [f.dst]
            for n in sorted(self._nodes):
                if n != f.dst and self._sensed(tx, n):
                    self._nodes[n].busy_until = max(self._nodes[n].busy_until, tx.end)
                    sensed_nodes.append(n)
            self._freeze_countdowns(self._now, sensed_nodes)
        self._push(tx.end, self._P_TX_END, "tx_end", tx)

    def _ack_end(self, tx: _Tx) -> None:
        sample = tx.samples[tx.ack_sender or ""]
        success = not self._collision(tx, tx.ack_sender or "") and float(self._error_rng.random()) >= float(sample.per)
        parent_frame_id = tx.ack_parent
        if parent_frame_id is None:
            return
        frame = self.frame_records[parent_frame_id]
        st = self._nodes[frame.src]
        if success:
            self._attempt_done_by_id(parent_frame_id, True)
        else:
            self._attempt_done_by_id(parent_frame_id, False)

    def _attempt_done(self, tx: _Tx, success: bool) -> None:
        self._attempt_done_by_id(tx.frame.frame_id, success)  # type: ignore[union-attr]

    def _attempt_done_by_id(self, fid: int, success: bool) -> None:
        rec = self.frame_records[fid]
        st = self._nodes[rec.src]
        f = st.current
        if f is None or f.frame_id != fid:
            return
        if success:
            self.frame_records[fid] = replace(rec, status="acked")
            st.current = None; st.attempt = 0; st.cw = self.phy.cw_min; st.phase = "idle"; st.remaining_slots = 0
            st.needs_backoff = self.mac.backoff and bool(st.queue)
            self._schedule_node(f.src, self._now)
            return
        if st.attempt <= self.mac.unicast_retry_limit:
            st.cw = min(self.phy.cw_max, max(self.phy.cw_min, st.cw * 2 + 1))
            st.queue.insert(0, f)
            st.current = None; st.needs_backoff = self.mac.backoff; st.phase = "idle"; st.remaining_slots = 0
            self._schedule_node(f.src, self._now)
        else:
            self.frame_records[fid] = replace(rec, status="retry_exhausted")
            st.current = None; st.attempt = 0; st.cw = self.phy.cw_min; st.phase = "idle"; st.remaining_slots = 0
            st.needs_backoff = self.mac.backoff and bool(st.queue)
            self._schedule_node(f.src, self._now)

    def _tx_end(self, tx: _Tx) -> None:
        if tx in self._active:
            if tx.kind == "data":
                self._data_end(tx)
            else:
                self._ack_end(tx)
            self._active.remove(tx)
            self._ended_same_time.append(tx)
            # A medium that became idle can release waiting nodes.
            for n in sorted(self._nodes):
                if self._nodes[n].current is None and self._nodes[n].queue:
                    self._schedule_node(n, self._now)

    def _enqueue_event(self, node: str, f: Frame) -> None:
        st = self._nodes[node]
        if len(st.queue) >= self.mac.queue_capacity:
            self.frame_records[f.frame_id] = replace(self.frame_records[f.frame_id], status="queue_drop")
            return
        st.queue.append(f)
        self._schedule_node(node, self._now)

    def _register_overlap(self, tx: _Tx) -> None:
        """Keep overlap powers on each transmission until its end event."""
        for old in self._active:
            if not self._overlap(tx, old):
                continue
            for rx, power in tx.powers.items():
                if rx in old.powers and power >= self.mac.interference_floor_dbm:
                    old.interference.setdefault(rx, []).append(power)
            for rx, power in old.powers.items():
                if rx in tx.powers and power >= self.mac.interference_floor_dbm:
                    tx.interference.setdefault(rx, []).append(power)
            if tx.src in old.powers:
                old.self_collision.add(tx.src)
            if old.src in tx.powers:
                tx.self_collision.add(old.src)

    def _source_event(self, kind: str, data: tuple) -> None:
        if kind == "periodic":
            node, dst, rate, payload, start, stop, phase, jitter, k = data
            t = self._now
            if t <= stop + 1e-12:
                self.enqueue(node, Frame(node, dst, payload, t, seq=k), t)
            period = 1.0 / rate
            nt = start + (k + 1) * period + phase
            if nt <= stop + 1e-12:
                if jitter:
                    nt += float(self._source_rng.uniform(0.0, jitter))
                self._push(nt, self._P_SOURCE, "periodic", (node, dst, rate, payload, start, stop, phase, jitter, k + 1))
        else:
            node, dst, rate, payload, start, stop = data
            t = self._now
            if t <= stop + 1e-12:
                self.enqueue(node, Frame(node, dst, payload, t), t)
                nt = t + float(self._source_rng.exponential(1.0 / rate))
                if nt <= stop + 1e-12:
                    self._push(nt, self._P_SOURCE, "poisson", data)

    def _deliver(self, rx: str, f: Frame, t: float) -> None:
        if self.on_receive is not None:
            self.on_receive(self, rx, f, t)

    def run_until(self, t_s: float) -> None:
        if not math.isfinite(t_s) or t_s < self._now - 1e-12:
            raise ValueError("run_until time must be finite and nondecreasing")
        while self._heap and self._heap[0][0] <= t_s + 1e-12:
            when, _, _, kind, data = heapq.heappop(self._heap)
            if when != self._last_event_time:
                self._ended_same_time.clear()
                self._last_event_time = when
            self._now = max(self._now, when)
            if kind == "enqueue": self._enqueue_event(*data)
            elif kind == "node_ready":
                node, token = data
                if token == self._nodes[node].token:
                    if self._nodes[node].phase == "busy":
                        self._schedule_node(node)
                    else:
                        self._start_tx(node)
            elif kind == "tx_end": self._tx_end(data)
            elif kind == "ack_start": self._ack_start(data)
            elif kind == "ack_timeout":
                tx, token = data
                st = self._nodes[tx.src]
                if st.phase == "ack_wait" and st.token == token: self._attempt_done(tx, False)
            elif kind == "deliver": self._deliver(*data)
            elif kind in ("periodic", "poisson"): self._source_event(kind, data)
        self._now = float(t_s)

    def channel_busy_fraction(self, t0: float, t1: float) -> float:
        if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
            raise ValueError("invalid interval")
        segs = sorted((max(t0, a), min(t1, b)) for a, b in self._intervals if b > t0 and a < t1)
        total = 0.0; cur = t0
        for a, b in segs:
            if a > cur: cur = a
            if b > cur: total += b - cur; cur = b
        return min(1.0, max(0.0, total / (t1 - t0)))
