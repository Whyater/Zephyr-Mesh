"""Heartbeats and the lost-link alarm (simulation only, diagnostic).

A drone and the ground station send each other a small "I'm here" message,
the heartbeat, ``rate_hz`` times a second.  The receiver declares the link
lost when no heartbeat has arrived for longer than ``timeout_s``.  Two things
trade off:

- a short timeout notices a real failure quickly, but random losses can
  string together and set the alarm off falsely;
- a long timeout rarely false-alarms, but a real failure takes longer to see.

Hand formulas are in the Run C derivations (D9) and are checked against long
simulated runs in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .channel import GilbertElliott


def _check(rate_hz: float, timeout_s: float) -> None:
    if not (math.isfinite(rate_hz) and rate_hz > 0 and math.isfinite(timeout_s) and timeout_s > 0):
        raise ValueError("rate_hz and timeout_s must be positive and finite")


def max_misses_before_alarm(timeout_s: float, rate_hz: float) -> int:
    _check(rate_hz, timeout_s)
    return int(math.floor(round(float(timeout_s) * float(rate_hz), 9)))


def false_alarm_rate_iid_per_hour(p: float, rate_hz: float, timeout_s: float) -> float:
    _check(rate_hz, timeout_s)
    if not 0 <= p <= 1 or not math.isfinite(p):
        raise ValueError("p must be in [0, 1]")
    m = max_misses_before_alarm(timeout_s, rate_hz)
    return 3600.0 * rate_hz * (1.0 - p) * p ** m


def false_alarm_rate_ge_per_hour(ge: GilbertElliott, rate_hz: float, timeout_s: float) -> float:
    _check(rate_hz, timeout_s)
    m = max_misses_before_alarm(timeout_s, rate_hz)
    p = ge.p_good_to_bad
    r = ge.p_bad_to_good
    hg, hb = ge.loss_good, ge.loss_bad
    pi = np.array([(1.0 - ge.stationary_bad()), ge.stationary_bad()])
    # D_s keeps the probability of receiving in each current state.
    D_s = np.diag([1.0 - hg, 1.0 - hb])
    # D_l keeps the probability of losing in each current state.
    D_l = np.diag([hg, hb])
    # P advances the hidden state after each heartbeat slot.
    P = np.array([[1.0 - p, p], [r, 1.0 - r]])
    # pi D_s (P D_l)^m 1 is one received heartbeat followed by m losses.
    prob = float(pi @ D_s @ np.linalg.matrix_power(P @ D_l, m) @ np.ones(2))
    return 3600.0 * rate_hz * prob


def detection_delay_bounds(rate_hz: float, timeout_s: float) -> tuple[float, float, float]:
    _check(rate_hz, timeout_s)
    return (timeout_s - 1.0 / rate_hz, timeout_s, timeout_s - 1.0 / (2.0 * rate_hz))


@dataclass(frozen=True)
class HeartbeatResult:
    alarm_start_times: np.ndarray
    n_episodes: int
    total_time_in_alarm: float
    false_alarm_rate_per_hour: float


def simulate_heartbeats(lost: np.ndarray, rate_hz: float, timeout_s: float, *, jitter_s=None, rng=None) -> HeartbeatResult:
    _check(rate_hz, timeout_s)
    lost = np.asarray(lost, dtype=bool)
    if lost.ndim != 1:
        raise ValueError("lost must be one-dimensional")
    n = lost.size
    if jitter_s is None:
        jitter = np.zeros(n)
    else:
        jitter = np.asarray(jitter_s, dtype=float)
        if jitter.ndim == 0:
            if rng is None:
                raise ValueError("rng is required for scalar jitter_s")
            jitter = np.asarray(rng.normal(0.0, float(jitter), n))
        if jitter.shape != lost.shape or np.any(~np.isfinite(jitter)):
            raise ValueError("jitter_s must match lost and be finite")
    times = np.arange(n, dtype=float) / rate_hz + jitter
    received = np.sort(times[~lost])
    starts: list[float] = []
    alarm_time = 0.0
    if received.size >= 2:
        if jitter_s is None:
            # Exact rule in whole slots: a gap of g slots alarms when g / h > T,
            # i.e. g > m* (comparing floats like 0.30000000000000004 > 0.3
            # would count gaps of exactly T as alarms).
            slots = np.flatnonzero(~lost)
            idx = np.flatnonzero(np.diff(slots) > max_misses_before_alarm(timeout_s, rate_hz))
        else:
            idx = np.flatnonzero(np.diff(received) > timeout_s + 1e-9)
        starts = (received[idx] + timeout_s).tolist()
        for i in idx:
            alarm_time += max(0.0, received[i + 1] - (received[i] + timeout_s))
    if received.size:
        end_time = times[-1] if n else received[-1]
        tail = end_time - (received[-1] + timeout_s)
        if tail > 0 and (n == 0 or times[-1] >= received[-1]):
            alarm_time += tail
    duration_hours = (n / rate_hz) / 3600.0 if n else 0.0
    rate = len(starts) / duration_hours if duration_hours > 0 else 0.0
    return HeartbeatResult(np.asarray(starts, dtype=float), len(starts), float(alarm_time), float(rate))


def detection_delay_samples(rate_hz: float, timeout_s: float, n: int, rng, *, pre_loss=None,
                            history_slots: int = 200) -> np.ndarray:
    """Simulate ``n`` permanent link failures and return how long each took to detect.

    Each trial runs ``history_slots`` heartbeats before the failure.  Each of
    those is lost with the loss process ``pre_loss``: None (no loss), a float
    p (independent loss), or a ``GilbertElliott`` chain.  The link then dies
    at a uniformly random moment inside the next heartbeat period.  The
    detector fires at (last received heartbeat) + timeout.  If it had already
    fired before the failure (a long run of losses), the delay is 0.

    Hand result (derivations D9): with no prior loss the delay is uniform on
    (T - 1/h, T], mean T - 1/(2h).  Prior losses can only make it shorter.
    """
    _check(rate_hz, timeout_s)
    if int(n) != n or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if int(history_slots) != history_slots or history_slots < 1:
        raise ValueError("history_slots must be a positive integer")
    if rng is None:
        raise ValueError("rng is required")
    period = 1.0 / rate_hz
    delays = np.empty(int(n))
    for k in range(int(n)):
        if pre_loss is None:
            lost = np.zeros(history_slots, dtype=bool)
        elif isinstance(pre_loss, GilbertElliott):
            lost = pre_loss.sample(history_slots, rng)[1]
        else:
            p = float(pre_loss)
            if not 0.0 <= p <= 1.0:
                raise ValueError("pre_loss probability must be in [0, 1]")
            lost = rng.random(history_slots) < p
        # Heartbeats at slots 0 .. history_slots - 1; failure inside the last period.
        t_fail = (history_slots - 1) * period + rng.random() * period
        received = np.flatnonzero(~lost)
        last = received[-1] * period if received.size else -np.inf
        delays[k] = max(0.0, last + timeout_s - t_fail) if np.isfinite(last) else 0.0
    return delays
