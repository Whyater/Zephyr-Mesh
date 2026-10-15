"""Sensor clock error and an NTP-style sync over the simulated radio link.

The sensor stamps every capture with its own clock. That clock is wrong in
three ways: an offset (it started at the wrong time, ``offset_s``), a drift
(it ticks a little fast or slow, ``drift_ppm`` parts per million) and stamp
jitter (each reading wobbles a little). The tracker works in its own clock,
which is taken to be true time. A stamp that is wrong by ``e`` seconds on a
target moving at speed ``v`` puts the predict-forward estimate off by about
``v * e`` metres, so the stamp error matters directly.

The fix is the classic four-timestamp exchange. The tracker sends a request
at ``t1`` (tracker clock). The sensor reads ``t2`` when it arrives, and sends a
reply at ``t3`` (sensor clock). The tracker reads ``t4`` when the reply lands.
Then

    theta = ((t2 - t1) + (t3 - t4)) / 2        (sensor minus tracker clock)
    delta = (t4 - t1) - (t3 - t2)              (round trip, processing removed)

``t2 - t1`` is the offset plus the trip up, ``t3 - t4`` is the offset minus the
trip down, so adding them cancels the trip only if both trips take the same
time. If the uplink is ``d_up`` and the downlink ``d_down``, theta is off by
``(d_up - d_down) / 2``: path asymmetry shows up as a clock offset error and no
amount of averaging removes it. Jitter on the delays just adds noise to theta.

Between syncs the drift keeps growing the error at ``r`` seconds per second,
so the clock error just before a new sync is about ``r * interval``.

This module only models timing. It does not model a real ESP-NOW or NTP
implementation, and the sensor clock is read without stamp jitter inside the
exchange (jitter is applied to capture stamps only).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .link import LinkConfig, SimulatedLink

MODES = ("perfect", "unsynced", "ntp")

DEFAULT_PARAMS = {
    "mode": "perfect",
    "offset_s": 0.0,
    "drift_ppm": 0.0,
    "stamp_jitter_s": 0.0,
    "sync_interval_s": 1.0,
    "sync_delay_s": 0.005,
    "sync_jitter_s": 0.0,
    "sync_loss_probability": 0.0,
    "sync_asymmetry_s": 0.0,
    "processing_s": 0.001,
}

TRACKER = "tracker"
SENSOR = "sensor"


@dataclass(frozen=True)
class ClockModel:
    """Sensor clock: reading = t + offset_s + drift_ppm * 1e-6 * t (+ jitter)."""

    offset_s: float = 0.0
    drift_ppm: float = 0.0
    stamp_jitter_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("offset_s", "drift_ppm", "stamp_jitter_s"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.stamp_jitter_s < 0:
            raise ValueError("stamp_jitter_s must be nonnegative")

    def read(self, t, rng: np.random.Generator | None = None):
        """What this clock shows at true time ``t`` (scalar or array).

        Jitter is added only if an ``rng`` is given and ``stamp_jitter_s > 0``,
        one independent draw per reading.
        """
        t_arr = np.asarray(t, dtype=float)
        value = t_arr * (1.0 + self.drift_ppm * 1e-6) + self.offset_s
        if rng is not None and self.stamp_jitter_s > 0:
            value = value + rng.normal(0.0, self.stamp_jitter_s, size=t_arr.shape)
        return float(value) if value.ndim == 0 else value


def offset_from_timestamps(t1: float, t2: float, t3: float, t4: float) -> float:
    """Estimated sensor-minus-tracker clock offset from one exchange."""
    return ((t2 - t1) + (t3 - t4)) / 2.0


def round_trip(t1: float, t2: float, t3: float, t4: float) -> float:
    """Round-trip time with the sensor's processing time taken out."""
    return (t4 - t1) - (t3 - t2)


@dataclass(frozen=True)
class SyncSample:
    """One exchange. A lost leg leaves the later fields as NaN and ``lost`` True.

    ``arrival_s`` is the true time the reply reaches the tracker (NaN if lost).
    ``lost_leg`` is "up", "down" or None.
    """

    seq: int
    t1: float
    t2: float
    t3: float
    t4: float
    theta_s: float
    delta_s: float
    arrival_s: float
    lost: bool
    lost_leg: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _lost_sample(seq: int, t1: float, t2: float, t3: float, leg: str) -> SyncSample:
    nan = float("nan")
    return SyncSample(seq, t1, t2, t3, nan, nan, nan, nan, True, leg)


def _exchange(t_true, sensor_clock, up_link, down_link, seq, processing_s,
              min_reply_send_s=0.0) -> tuple[SyncSample, float | None]:
    """The exchange plus the true time the reply was sent (None if the request was lost)."""
    t_true = float(t_true)
    if processing_s < 0 or not np.isfinite(processing_s):
        raise ValueError("processing_s must be finite and nonnegative")
    t1 = t_true
    up = up_link.send(TRACKER, SENSOR, seq, t_true)
    if up.receive_time is None:
        return _lost_sample(seq, t1, float("nan"), float("nan"), "up"), None
    t2 = float(sensor_clock.read(up.receive_time))
    # A reply cannot leave before an earlier reply did on the same route.
    reply_send = max(up.receive_time + processing_s, float(min_reply_send_s))
    t3 = float(sensor_clock.read(reply_send))
    down = down_link.send(SENSOR, TRACKER, seq, reply_send)
    if down.receive_time is None:
        return _lost_sample(seq, t1, t2, t3, "down"), reply_send
    t4 = float(down.receive_time)
    sample = SyncSample(seq, t1, t2, t3, t4, offset_from_timestamps(t1, t2, t3, t4),
                        round_trip(t1, t2, t3, t4), t4, False, None)
    return sample, reply_send


def ntp_exchange(t_true, sensor_clock: ClockModel, up_link: SimulatedLink,
                 down_link: SimulatedLink, seq: int,
                 processing_s: float = 0.001) -> SyncSample | None:
    """One four-timestamp exchange starting at true time ``t_true``.

    ``up_link`` carries the request (tracker to sensor) and ``down_link`` the
    reply, so a different ``delay_s`` on each makes the path asymmetric.
    Returns None if either packet is lost.
    """
    sample, _ = _exchange(t_true, sensor_clock, up_link, down_link, seq, processing_s)
    return None if sample.lost else sample


def _resolve_params(clock_params) -> dict:
    params = dict(DEFAULT_PARAMS)
    if clock_params:
        for key, value in dict(clock_params).items():
            if key not in DEFAULT_PARAMS:
                raise ValueError(f"unknown clock parameter {key!r}")
            params[key] = value
    if params["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for key in DEFAULT_PARAMS:
        if key == "mode":
            continue
        value = float(params[key])
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
        params[key] = value
    for key in ("stamp_jitter_s", "sync_delay_s", "sync_jitter_s", "processing_s"):
        if params[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if params["sync_interval_s"] <= 0:
        raise ValueError("sync_interval_s must be positive")
    if not 0.0 <= params["sync_loss_probability"] <= 1.0:
        raise ValueError("sync_loss_probability must be in [0, 1]")
    half = params["sync_asymmetry_s"] / 2.0
    if params["sync_delay_s"] + half < 0 or params["sync_delay_s"] - half < 0:
        raise ValueError("sync_asymmetry_s is too large: a one-way delay would be negative")
    return params


def _stream(streams, key: str):
    if not streams or key not in streams:
        raise ValueError(f"streams must provide an integer seed under {key!r}")
    return int(streams[key])


def _make_links(params: dict, streams) -> tuple[SimulatedLink, SimulatedLink]:
    loss = params["sync_loss_probability"]
    model = "independent" if loss > 0 else "none"
    half = params["sync_asymmetry_s"] / 2.0

    def config(delay: float) -> LinkConfig:
        return LinkConfig(delay_s=delay, delay_jitter_s=params["sync_jitter_s"],
                          loss_model=model, loss_probability=loss)

    up = SimulatedLink(config(params["sync_delay_s"] + half), seed=_stream(streams, "sync_up"))
    down = SimulatedLink(config(params["sync_delay_s"] - half), seed=_stream(streams, "sync_down"))
    return up, down


def _run_schedule(horizon_s: float, params: dict, streams) -> list[SyncSample]:
    """Exchanges at t = 0, I, 2I, ... up to ``horizon_s`` (inclusive)."""
    up, down = _make_links(params, streams)
    clock = ClockModel(params["offset_s"], params["drift_ppm"], 0.0)
    interval = params["sync_interval_s"]
    samples: list[SyncSample] = []
    last_reply_send = 0.0
    k = 0
    while k * interval <= horizon_s + 1e-12:
        sample, reply_send = _exchange(k * interval, clock, up, down, k,
                                       params["processing_s"], last_reply_send)
        if reply_send is not None:
            last_reply_send = max(last_reply_send, reply_send)
        samples.append(sample)
        k += 1
    return samples


def sync_log(horizon_s: float, clock_params=None, streams=None) -> list[dict]:
    """Every exchange the schedule would make up to ``horizon_s``, as dicts.

    Keys: seq, t1, t2, t3, t4, theta_s, delta_s, arrival_s, lost, lost_leg.
    Use it to report the sync success rate: ``1 - mean(row["lost"])``.
    """
    horizon = float(horizon_s)
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon_s must be finite and nonnegative")
    params = _resolve_params(clock_params)
    if params["mode"] != "ntp":
        return []
    return [s.as_dict() for s in _run_schedule(horizon, params, streams)]


def _theta_at(captures: np.ndarray, samples: list[SyncSample]) -> np.ndarray:
    """Offset estimate in use at each capture time.

    Among the exchanges whose reply has arrived by the capture time, the newest
    EXCHANGE (highest sequence number) wins, not the reply that happened to
    arrive last: with jitter an older exchange's reply can arrive after a newer
    one's, and the newer exchange measured the clock more recently.
    """
    theta = np.zeros(captures.shape)
    got = [s for s in samples if not s.lost]
    if not got:
        return theta
    got.sort(key=lambda s: (s.arrival_s, s.seq))
    arrivals = np.array([s.arrival_s for s in got])
    seqs = np.array([s.seq for s in got])
    thetas = np.array([s.theta_s for s in got])
    # best[i] = index (in arrival order) of the highest-seq exchange among the first i + 1 arrivals.
    best = np.zeros(len(got), dtype=int)
    for i in range(1, len(got)):
        best[i] = i if seqs[i] > seqs[best[i - 1]] else best[i - 1]
    idx = np.searchsorted(arrivals, captures, side="right") - 1
    seen = idx >= 0
    theta[seen] = thetas[best[idx[seen]]]
    return theta


def stamp_errors_with_log(capture_times, clock_params=None, streams=None):
    """Like :func:`stamp_errors` but also returns the list of ``SyncSample``."""
    params = _resolve_params(clock_params)
    captures = np.asarray(capture_times, dtype=float)
    if captures.ndim != 1 or not np.all(np.isfinite(captures)) or np.any(captures < 0):
        raise ValueError("capture_times must be a 1D array of finite nonnegative times")
    if params["mode"] == "perfect" or captures.size == 0:
        return np.zeros(captures.shape), []
    clock = ClockModel(params["offset_s"], params["drift_ppm"], params["stamp_jitter_s"])
    rng = None
    if params["stamp_jitter_s"] > 0:
        rng = np.random.default_rng(_stream(streams, "clock"))
    # Sensor reading minus true time: offset + drift + jitter.
    error = clock.read(captures, rng) - captures
    samples: list[SyncSample] = []
    if params["mode"] == "ntp" and captures.size:
        samples = _run_schedule(float(captures.max()), params, streams)
        error = error - _theta_at(captures, samples)
    return error, samples


def stamp_errors(capture_times, clock_params=None, streams=None) -> np.ndarray:
    """Error of each capture stamp as the tracker sees it (stamp minus true time).

    ``clock_params`` keys (defaults in brackets): mode ["perfect"], offset_s [0],
    drift_ppm [0], stamp_jitter_s [0], sync_interval_s [1.0], sync_delay_s
    [0.005], sync_jitter_s [0], sync_loss_probability [0], sync_asymmetry_s [0],
    processing_s [0.001]. Unknown keys raise ValueError.

    "perfect": all zeros. "unsynced": e0 + r t (+ jitter). "ntp": exchanges at
    t = 0, I, 2I, ...; a capture at time t uses the latest exchange whose reply
    had arrived by t (theta = 0 before the first), and error = reading - theta - t.

    ``streams`` is a dict of integer seeds: "clock" (stamp jitter, one draw per
    capture in order), "sync_up" and "sync_down" (the two sync links).
    """
    return stamp_errors_with_log(capture_times, clock_params, streams)[0]
