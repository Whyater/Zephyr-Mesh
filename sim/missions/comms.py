"""Deterministic simulated mission links and synthetic loss processes."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable

import numpy as np

from sim.link import LinkConfig
from sim.radio.model import NetworkModelConfig, RadioLinkConfig, make_network_model
from .presets import LinkPreset, get_preset


def _stable_index(stream_id: str) -> int:
    """Fallback stream index for links constructed independently."""
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream_id must be a non-empty string")
    # Keep the mapping independent of construction order and Python's hash
    # randomization. Mission stream names are grouped by their fixed prefix,
    # then the suffix contributes a stable lexical surrogate.
    prefixes = {"cmd:": 0, "tel:": 1, "meas:": 2, "share:": 3}
    prefix = next((value for key, value in prefixes.items() if stream_id.startswith(key)), 4)
    suffix = stream_id.split(":", 1)[1] if ":" in stream_id else stream_id
    rank = 0
    for i, char in enumerate(suffix):
        rank = (rank * 257 + (i + 1) * ord(char)) % 1000003
    return prefix * 1000003 + rank


class LossProcess:
    """Synthetic independent or shared-burst loss process on a fixed time grid."""

    def __init__(self, preset: LinkPreset | str, *, seed: int, dt_s: float = 0.01,
                 duration_s: float):
        self.preset = get_preset(preset) if isinstance(preset, str) else preset
        if self.preset.loss_model not in {"none", "independent", "burst"}:
            raise ValueError("invalid loss model")
        if seed is None or isinstance(seed, bool):
            raise ValueError("seed is required")
        self.seed = int(seed)
        self.dt_s = float(dt_s)
        self.duration_s = float(duration_s)
        if not math.isfinite(self.dt_s) or self.dt_s <= 0 or not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("dt_s must be positive and duration_s nonnegative")
        self._streams: dict[str, np.random.Generator] = {}
        self._drawn: dict[str, list[bool]] = {}
        self._stream_indices: dict[str, int] = {}
        self.bad = np.zeros(int(math.ceil(self.duration_s / self.dt_s)) + 1, dtype=bool)
        if self.preset.loss_model == "burst" and self.bad.size:
            p = float(self.preset.loss_probability)
            r = self.dt_s / float(self.preset.burst_mean_s)
            q = 0.0 if p == 0 else p * r / (1.0 - p)
            if r > 1.0 or q > 1.0:
                raise ValueError("burst parameters produce invalid transition probabilities")
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7001]))
            state = bool(rng.random() < p)
            for i in range(self.bad.size):
                self.bad[i] = state
                if state:
                    if rng.random() < r:
                        state = False
                elif rng.random() < q:
                    state = True

    def _rng(self, stream_id: str) -> np.random.Generator:
        if stream_id not in self._streams:
            idx = _stable_index(stream_id)
            self._stream_indices[stream_id] = idx
            self._streams[stream_id] = np.random.default_rng(
                np.random.SeedSequence([self.seed, 7100 + idx]))
            self._drawn[stream_id] = []
        return self._streams[stream_id]

    def lost(self, stream_id: str, send_time_s: float, rng: np.random.Generator) -> bool:
        if self.preset.loss_model == "none":
            return False
        if self.preset.loss_model == "burst":
            return bool(self.bad[min(max(int(math.floor(float(send_time_s) / self.dt_s + 1e-12)), 0), len(self.bad) - 1)])
        return bool(rng.random() < self.preset.loss_probability)

    def is_lost(self, stream_id: str, seq: int, send_time_s: float) -> bool:
        if self.preset.loss_model == "none":
            return False
        if self.preset.loss_model == "burst":
            slot = min(max(int(math.floor(float(send_time_s) / self.dt_s + 1e-12)), 0), len(self.bad) - 1)
            return bool(self.bad[slot])
        if int(seq) < 0:
            raise ValueError("seq must be nonnegative")
        rng = self._rng(stream_id)
        draws = self._drawn[stream_id]
        while len(draws) <= int(seq):
            draws.append(bool(rng.random() < self.preset.loss_probability))
        return draws[int(seq)]


@dataclass(frozen=True)
class OutageWindow:
    start_s: float
    end_s: float
    streams: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)) or self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("outage bounds must be finite, nonnegative, and ordered")
        if self.streams is not None:
            object.__setattr__(self, "streams", tuple(sorted(set(self.streams))))

    def applies(self, stream_id: str, send_time_s: float) -> bool:
        return self.start_s <= float(send_time_s) < self.end_s and (self.streams is None or stream_id in self.streams)


@dataclass(frozen=True)
class Delivered:
    seq: int
    send_time_s: float
    receive_time_s: float
    message: Any


@dataclass
class _Packet:
    seq: int
    send_time_s: float
    receive_time_s: float | None
    message: Any
    reason: str | None
    delivered: bool = False


class MissionLink:
    """A one-way deterministic mission link with synthetic loss and delay."""

    def __init__(self, stream_id: str, *, delay_s: float, jitter_s: float = 0.0,
                 loss: LossProcess, outages: Iterable[OutageWindow] = (), seed: int):
        if seed is None:
            raise ValueError("seed is required")
        self.stream_id = str(stream_id)
        self.loss = loss
        self.outages = tuple(outages)
        self._packets: list[_Packet] = []
        self._pending: list[tuple[float, int, int]] = []  # (receive time, seq, packet index)
        self._sent = 0
        self._delivered = 0
        self._lost = {"loss": 0, "outage": 0}
        idx = _stable_index(self.stream_id)
        timing_seed = int(np.random.SeedSequence([int(seed), 7200 + idx]).generate_state(1)[0])
        cfg = NetworkModelConfig(mode="abstract", link=LinkConfig(delay_s=float(delay_s), delay_jitter_s=float(jitter_s), loss_model="none"))
        self._net = make_network_model(cfg, seed=timing_seed)

    def send(self, seq: int, send_time_s: float, message: Any) -> None:
        seq = int(seq); send_time_s = float(send_time_s)
        if seq < 0 or not math.isfinite(send_time_s) or send_time_s < 0:
            raise ValueError("invalid packet sequence or send time")
        event = self._net.send(self.stream_id, f"peer:{self.stream_id}", seq, send_time_s)
        reason = "outage" if any(w.applies(self.stream_id, send_time_s) for w in self.outages) else ("loss" if self.loss.is_lost(self.stream_id, seq, send_time_s) else None)
        self._packets.append(_Packet(seq, send_time_s, event.receive_time, message, reason))
        if reason is None and event.receive_time is not None:
            heapq.heappush(self._pending, (float(event.receive_time), seq, len(self._packets) - 1))
        self._sent += 1
        if reason:
            self._lost[reason] += 1

    def deliveries_until(self, now_s: float) -> list[Delivered]:
        # The abstract model fixes each receive time at send time, so a heap
        # ordered by (receive time, seq) yields deliveries without rescanning.
        limit = float(now_s) + 1e-12
        out = []
        while self._pending and self._pending[0][0] <= limit:
            _, _, index = heapq.heappop(self._pending)
            packet = self._packets[index]
            packet.delivered = True
            self._delivered += 1
            out.append(Delivered(packet.seq, packet.send_time_s, float(packet.receive_time_s), packet.message))
        return out

    def stats(self) -> dict:
        return {"sent": self._sent, "delivered": self._delivered, "lost_by_reason": dict(self._lost)}

    def events(self) -> list[dict]:
        return [{"seq": p.seq, "send_time_s": p.send_time_s, "receive_time_s": p.receive_time_s if p.reason is None else None,
                 "reason": p.reason} for p in self._packets]


class RadioMissionLink:
    """Mission-link surface backed by the simulated Run C radio model."""

    def __init__(self, stream_id: str, *, radio_config: RadioLinkConfig, seed: int,
                 sender_id: str | None = None, receiver_id: str | None = None):
        if not isinstance(radio_config, RadioLinkConfig):
            raise ValueError("radio_config must be a RadioLinkConfig")
        self.stream_id = str(stream_id)
        self.sender_id = sender_id or self.stream_id
        if self.sender_id not in radio_config.positions:
            raise ValueError("sender_id must be present in radio_config.positions")
        if receiver_id is None:
            choices = sorted(name for name in radio_config.positions if name != self.sender_id)
            if not choices:
                raise ValueError("radio_config.positions must contain a receiver")
            receiver_id = choices[0]
        if receiver_id not in radio_config.positions:
            raise ValueError("receiver_id must be present in radio_config.positions")
        self.receiver_id = receiver_id
        self._net = make_network_model(
            NetworkModelConfig(mode="radio", radio=radio_config), seed=int(seed)
        )
        self._packets: list[_Packet] = []
        self._seen: set[int] = set()

    def send(self, seq: int, send_time_s: float, message: Any) -> None:
        event = self._net.send(self.sender_id, self.receiver_id, int(seq), float(send_time_s))
        self._packets.append(_Packet(int(seq), float(send_time_s), event.receive_time, message, None))

    def deliveries_until(self, now_s: float) -> list[Delivered]:
        ready = self._net.advance(float(now_s) + 1e-12)
        out = []
        by_seq = {p.seq: p for p in self._packets}
        for event in ready:
            if event.outcome != "received" or event.seq in self._seen or event.seq not in by_seq:
                continue
            packet = by_seq[event.seq]
            packet.delivered = True
            packet.receive_time_s = event.receive_time
            self._seen.add(event.seq)
            out.append(Delivered(packet.seq, packet.send_time_s, float(event.receive_time), packet.message))
        return sorted(out, key=lambda item: (item.receive_time_s, item.seq))

    def stats(self) -> dict:
        delivered = sum(p.delivered for p in self._packets)
        lost = sum(1 for event in self._net.events if event.outcome == "lost")
        return {"sent": len(self._packets), "delivered": delivered,
                "lost_by_reason": {"loss": lost, "outage": 0}}

    def events(self) -> list[dict]:
        return [{"seq": p.seq, "send_time_s": p.send_time_s,
                 "receive_time_s": p.receive_time_s if p.delivered else None,
                 "reason": None if p.delivered else "loss"} for p in self._packets]

