"""Map a device's boot clock onto sim time.

The simulator has one perfect clock that every part shares. A real drone
does not: it stamps telemetry with its own counter that started at zero when
it powered on (PX4 counts microseconds, MAVLink messages often carry
milliseconds), and that counter ticks a little fast or slow. To use those
stamps, the ground station needs a rule that turns "device time" into "sim
time" (the ground station's clock).

This module uses the same clock model as ``sim/timesync.py``::

    device = sim * (1 + drift_ppm * 1e-6) + offset_s

* ``offset_s`` is how far ahead the device clock reads at sim time zero.
* ``drift_ppm`` is how fast it gains time, in parts per million. 50 ppm means
  it gains 50 microseconds every second, so 0.05 s after 1000 s.

Given matched pairs (the device stamp and the sim time of the same moment),
:func:`fit_clock_map` finds the offset and drift with a straight-line least
squares fit. :func:`pairs_from_sync_log` builds those pairs from the
four-timestamp exchanges in ``sim.timesync``. Like that module, the pairs
cannot see a difference between the uplink and downlink delay: an asymmetry
of ``a`` seconds moves the fitted offset by ``a / 2``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

__all__ = ["ClockMap", "fit_clock_map", "pairs_from_sync_log", "unwrap_counter", "ticks_to_seconds"]


@dataclass(frozen=True)
class ClockMap:
    """A fitted (or known) rule between device time and sim time, in seconds.

    ``n_pairs``, ``span_s`` and ``residual_rms_s`` describe the fit that made
    it (None when the map was written by hand). ``residual_rms_s`` is the
    typical leftover mismatch of one pair, which is roughly the stamp noise.
    """

    offset_s: float
    drift_ppm: float
    n_pairs: int | None = None
    span_s: float | None = None
    residual_rms_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("offset_s", "drift_ppm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.drift_ppm <= -1e6:
            raise ValueError("drift_ppm must be above -1e6 (the clock must move forward)")

    @property
    def rate(self) -> float:
        """Device seconds per sim second, ``1 + drift_ppm * 1e-6``."""
        return 1.0 + self.drift_ppm * 1e-6

    def to_device(self, sim_s):
        """Device clock reading at sim time ``sim_s`` (scalar or array)."""
        value = np.asarray(sim_s, dtype=float) * self.rate + self.offset_s
        return float(value) if value.ndim == 0 else value

    def to_sim(self, device_s):
        """Sim time of a device stamp: undo the offset, then undo the drift."""
        value = (np.asarray(device_s, dtype=float) - self.offset_s) / self.rate
        return float(value) if value.ndim == 0 else value


def fit_clock_map(device_s: Iterable[float], sim_s: Iterable[float]) -> ClockMap:
    """Fit offset and drift from matched (device stamp, sim time) pairs.

    How it works, without matrices: the gap ``device - sim`` is the clock
    error at each moment. With the model above that gap is a straight line in
    sim time, ``gap = offset + drift * sim``. A least squares line through the
    gaps gives the slope (the drift) and the value at sim time zero (the
    offset)::

        slope  = sum((t - t_mean) * (gap - gap_mean)) / sum((t - t_mean) ** 2)
        offset = gap_mean - slope * t_mean

    Fitting the gap instead of the raw device time keeps the numbers small, so
    a 50 ppm slope does not get lost next to a 1 in front of it.

    Needs at least two pairs at different sim times. More pairs spread over a
    longer time pin the drift down better: the drift error shrinks like
    ``noise / sqrt(sum((t - t_mean) ** 2))``.
    """
    device = np.asarray(list(device_s) if not isinstance(device_s, np.ndarray) else device_s, dtype=float)
    sim = np.asarray(list(sim_s) if not isinstance(sim_s, np.ndarray) else sim_s, dtype=float)
    if device.ndim != 1 or sim.ndim != 1 or device.shape != sim.shape:
        raise ValueError("device_s and sim_s must be 1-D and the same length")
    if device.size < 2:
        raise ValueError("need at least two pairs to fit a drift")
    if not (np.all(np.isfinite(device)) and np.all(np.isfinite(sim))):
        raise ValueError("pairs must be finite")
    gap = device - sim
    t_mean = float(np.mean(sim))
    spread = float(np.sum((sim - t_mean) ** 2))
    if spread <= 0.0:
        raise ValueError("pairs must cover more than one sim time")
    gap_mean = float(np.mean(gap))
    slope = float(np.sum((sim - t_mean) * (gap - gap_mean))) / spread
    offset = gap_mean - slope * t_mean
    residual = gap - (offset + slope * sim)
    return ClockMap(
        offset_s=offset,
        drift_ppm=slope * 1e6,
        n_pairs=int(device.size),
        span_s=float(np.max(sim) - np.min(sim)),
        residual_rms_s=float(np.sqrt(np.mean(residual ** 2))),
    )


def pairs_from_sync_log(log: Iterable[Mapping[str, float] | object]) -> tuple[np.ndarray, np.ndarray]:
    """Turn four-timestamp exchanges into (device, sim) pairs.

    Accepts the dicts from ``sim.timesync.sync_log`` or ``SyncSample`` objects.
    Each kept exchange gives one pair: the middle of the device's two stamps,
    ``(t2 + t3) / 2``, matched with the middle of the ground station's two
    stamps, ``(t1 + t4) / 2``. If the trip up and the trip down take the same
    time, those two middles are the same moment. Lost exchanges are skipped.
    """
    device = []
    sim = []
    for row in log:
        get = row.get if isinstance(row, Mapping) else (lambda key, _row=row: getattr(_row, key))
        if get("lost"):
            continue
        t1, t2, t3, t4 = (float(get(key)) for key in ("t1", "t2", "t3", "t4"))
        if not all(math.isfinite(v) for v in (t1, t2, t3, t4)):
            continue
        device.append((t2 + t3) / 2.0)
        sim.append((t1 + t4) / 2.0)
    return np.asarray(device, dtype=float), np.asarray(sim, dtype=float)


DEFAULT_MAX_GAP_TICKS = 5_000_000
"""Default largest gap between samples: 5 s of a microsecond counter."""


def unwrap_counter(ticks, modulus: int, max_gap_ticks: int | None = None) -> np.ndarray:
    """Undo rollovers in a counter that wraps back to zero at ``modulus``.

    Example: a 32-bit microsecond counter (``modulus = 2**32``) wraps every
    4294.97 s, about 71.6 minutes. Samples must be in time order.

    ``max_gap_ticks`` is the largest plausible gap between two samples, in
    counter ticks. When the counter goes down from one sample to the next,
    the drop is judged with it:

    * Drop of at most ``max_gap_ticks``: a reordered packet. Left alone.
    * A rollover: the step forward across the wrap,
      ``(modulus - previous) + next``, is at most ``max_gap_ticks``. That
      means the previous value was near the top AND the next is near zero.
      ``modulus`` is added to that sample and every later one.
    * Anything else raises ``ValueError`` naming the sample index. The usual
      cause is a device reboot (the counter restarted at zero). Split the log
      at that index and fit each piece on its own.

    A reordered packet can also straddle the wrap: one stamped just before
    the top arrives after one stamped just after zero. That shows up as a
    forward step of at least ``modulus - max_gap_ticks`` (it is at most
    ``max_gap_ticks`` older, counted back across the wrap). It is placed in
    the epoch before the previous sample, and later samples are judged as
    usual. Results are counted from the first sample's epoch, so if the very
    first sample is already past the wrap, a late packet from before it
    comes out slightly below zero.

    A reboot can still pass as a rollover if it happens within
    ``max_gap_ticks`` of the top of the range, because the two look the same
    in the counter alone. A tight gap keeps that window small.

    The default is the smaller of ``DEFAULT_MAX_GAP_TICKS`` (5,000,000 ticks,
    which is 5 s of a microsecond counter, 5000 s of a millisecond one) and
    ``modulus // 4`` (so small counters still work). The gap must be positive
    and less than half the modulus, so a reordered packet and a rollover can
    never both explain the same drop.
    """
    modulus = int(modulus)
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if max_gap_ticks is None:
        max_gap_ticks = min(DEFAULT_MAX_GAP_TICKS, modulus // 4)
    max_gap_ticks = int(max_gap_ticks)
    if max_gap_ticks <= 0 or 2 * max_gap_ticks >= modulus:
        raise ValueError("max_gap_ticks must be positive and less than half the modulus")
    raw = np.asarray(ticks)
    if raw.ndim != 1:
        raise ValueError("ticks must be 1-D")
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or np.any(raw != np.round(raw)):
            raise ValueError("ticks must be whole numbers")
    values = raw.astype(np.int64)
    if np.any(values < 0) or np.any(values >= modulus):
        raise ValueError("ticks must be in [0, modulus)")
    if values.size < 2:
        return values.copy()
    prev = values[:-1]
    nxt = values[1:]
    drop = prev - nxt
    backward = drop > max_gap_ticks
    wraps_here = backward & ((modulus - prev) + nxt <= max_gap_ticks)
    bad = np.flatnonzero(backward & ~wraps_here)
    if bad.size:
        i = int(bad[0]) + 1
        raise ValueError(
            f"counter went backward by {int(drop[i - 1])} ticks at index {i} "
            f"({int(values[i - 1])} -> {int(values[i])}), which is neither a reordered packet "
            f"nor a rollover within max_gap_ticks={max_gap_ticks}: likely a device reboot. "
            "Split the log at this index."
        )
    late_pre_wrap = (nxt - prev) >= modulus - max_gap_ticks
    steps = wraps_here.astype(np.int64) - late_pre_wrap.astype(np.int64)
    wraps = np.concatenate(([0], np.cumsum(steps)))
    return values + wraps.astype(np.int64) * modulus


def ticks_to_seconds(ticks, tick_hz: float):
    """Counter ticks to seconds, for example microseconds with ``tick_hz = 1e6``."""
    tick_hz = float(tick_hz)
    if not math.isfinite(tick_hz) or tick_hz <= 0.0:
        raise ValueError("tick_hz must be finite and positive")
    value = np.asarray(ticks, dtype=float) / tick_hz
    return float(value) if value.ndim == 0 else value
