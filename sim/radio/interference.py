"""Interference episodes and a degraded-link detector (simulation only, defensive).

Nothing here transmits anything.  It models what a receiver sees when the
radio environment gets noisy for a while (another Wi-Fi network, a
microwave oven, a crowded event): the noise floor rises by some dB for some
seconds, SNR drops, and packets start failing.  ``DegradedLinkDetector``
watches a sliding window of packets and declares the link "degraded" when
loss is high AND the received SNR is low, so the vehicle can slow down and
hold instead of acting on stale data.  Its trigger delay and false triggers
are measured in EXP-016.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .channel import Channel, ChannelConfig, packet_error_rate


@dataclass(frozen=True)
class InterferenceConfig:
    onset_rate_per_hour: float
    mean_duration_s: float
    noise_rise_db: float
    duration_model: str = "exponential"

    def __post_init__(self):
        if not (math.isfinite(self.onset_rate_per_hour) and self.onset_rate_per_hour >= 0):
            raise ValueError("onset_rate_per_hour must be nonnegative")
        if not (math.isfinite(self.mean_duration_s) and self.mean_duration_s > 0):
            raise ValueError("mean_duration_s must be positive")
        if not math.isfinite(self.noise_rise_db) or self.noise_rise_db < 0:
            raise ValueError("noise_rise_db must be nonnegative")
        if self.duration_model != "exponential":
            raise ValueError("duration_model must be exponential")


def interference_timeline(cfg: InterferenceConfig, duration_s: float, rng) -> list[tuple[float, float]]:
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be nonnegative and finite")
    if rng is None:
        raise ValueError("rng is required")
    out = []
    t = 0.0
    lam = cfg.onset_rate_per_hour / 3600.0
    if lam == 0:
        return out
    while True:
        t += float(rng.exponential(1.0 / lam))
        if t >= duration_s:
            break
        out.append((t, t + float(rng.exponential(cfg.mean_duration_s))))
    return out


def noise_floor_at(t: float, base_dbm: float, episodes, cfg: InterferenceConfig) -> float:
    if not math.isfinite(t) or not math.isfinite(base_dbm):
        raise ValueError("t and base_dbm must be finite")
    return float(base_dbm + (cfg.noise_rise_db if any(s <= t < e for s, e in episodes) else 0.0))


class DegradedLinkDetector:
    def __init__(self, window_packets: int, loss_threshold: float, snr_threshold_db: float, min_packets: int):
        if int(window_packets) != window_packets or window_packets < 1 or int(min_packets) != min_packets or min_packets < 1:
            raise ValueError("window_packets and min_packets must be positive integers")
        if not 0 < loss_threshold <= 1 or not math.isfinite(snr_threshold_db):
            raise ValueError("invalid detector thresholds")
        self.window_packets = int(window_packets)
        self.loss_threshold = float(loss_threshold)
        self.snr_threshold_db = float(snr_threshold_db)
        self.min_packets = int(min_packets)
        self.state = "ok"
        self._items: list[tuple[float, bool, float]] = []
        self.transitions: list[tuple[float, str]] = []

    def update(self, time_s: float, received: bool, snr_db: float) -> str:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        self._items.append((float(time_s), bool(received), float(snr_db)))
        self._items = self._items[-self.window_packets:]
        if len(self._items) < self.min_packets:
            return self.state
        rec = np.array([x[1] for x in self._items], dtype=bool)
        loss_rate = 1.0 - float(np.mean(rec))
        vals = np.array([x[2] for x in self._items if x[1] and math.isfinite(x[2])])
        bad_snr = vals.size == 0 or float(np.median(vals)) <= self.snr_threshold_db
        want = loss_rate >= self.loss_threshold and bad_snr
        if self.state == "ok" and want:
            self.state = "degraded"
            self.transitions.append((float(time_s), self.state))
        elif self.state == "degraded" and loss_rate < self.loss_threshold / 2.0:
            self.state = "ok"
            self.transitions.append((float(time_s), self.state))
        return self.state

    def process(self, times, received, snr):
        for t, r, s in zip(times, received, snr):
            self.update(float(t), bool(r), float(s))
        return list(self.transitions)


def simulate_link_with_interference(channel_cfg: ChannelConfig, distance_m: float, rate_hz: float,
                                    duration_s: float, frame_bytes: int,
                                    interference_cfg: InterferenceConfig, seed: int) -> dict:
    if not math.isfinite(rate_hz) or rate_hz <= 0 or not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("rate_hz and duration_s are invalid")
    root = np.random.SeedSequence(seed)
    ep_rng, ch_rng, loss_rng = [np.random.default_rng(s) for s in root.spawn(3)]
    episodes = interference_timeline(interference_cfg, duration_s, ep_rng)
    times = np.arange(0.0, duration_s, 1.0 / rate_hz)
    ch = Channel(channel_cfg, ch_rng)
    received = np.empty(times.size, dtype=bool)
    snrs = np.empty(times.size, dtype=float)
    for i, t in enumerate(times):
        nf = noise_floor_at(float(t), channel_cfg.noise_floor_dbm, episodes, interference_cfg)
        sample = ch.sample("tx", "rx", distance_m, float(t), frame_bytes=frame_bytes, noise_floor_dbm=nf)
        snrs[i] = sample.snr_db
        received[i] = loss_rng.random() >= sample.per
    return {"times": times, "received": received, "snr_db": snrs, "episodes": episodes}


def evaluate_detector(times, received, snr_db, episodes, detector=None, *, grace_s: float = 0.0) -> dict:
    """Score a detector against known interference episodes.

    A trigger at time t belongs to an episode (start, end) when
    start <= t <= end + grace_s: the detector needs a window of packets to
    react, so a trigger just after a short episode is a late detection, not a
    false alarm.  An episode that starts while the detector is already in the
    degraded state counts as detected with delay 0.  Every other trigger is a
    false trigger.
    """
    if grace_s < 0 or not math.isfinite(grace_s):
        raise ValueError("grace_s must be finite and nonnegative")
    if detector is None:
        detector = DegradedLinkDetector(10, 0.5, 0.0, 3)
    detector.process(times, received, snr_db)
    transitions = detector.transitions

    def state_at(t):
        state = "ok"
        for when, new_state in transitions:
            if when > t:
                break
            state = new_state
        return state

    starts = [t for t, state in transitions if state == "degraded"]
    delays, missed = [], 0
    for start, end in episodes:
        if state_at(start) == "degraded":
            delays.append(0.0)
            continue
        inside = [t for t in starts if start <= t <= end + grace_s]
        if inside:
            delays.append(inside[0] - start)
        else:
            missed += 1
    duration = float(np.max(times) - np.min(times)) if len(times) > 1 else 0.0
    outside = [t for t in starts if not any(s <= t <= e + grace_s for s, e in episodes)]
    return {"trigger_delays_s": delays, "missed": missed, "false_triggers": len(outside),
            "false_triggers_per_hour": len(outside) * 3600.0 / duration if duration > 0 else 0.0,
            "transitions": transitions}
