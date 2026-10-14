"""Sender policies: which measurements leave the sensor, and how big each packet is.

The sensor captures a position measurement at a fixed rate. The sender may
not transmit every one of them. This module holds the three knobs it has:

- periodic: send every k-th sample (a lower send rate than the sensor rate).
- event-triggered: keep a copy of the receiver's filter (a "replica") fed
  only with what was sent, and send a sample only when it disagrees with the
  replica's prediction by at least ``delta_m`` metres. A sample the receiver
  could already have guessed is not worth the air time. An optional
  heartbeat forces a send if nothing went out for ``heartbeat_s`` seconds.
- int16 quantization: round each coordinate to a multiple of ``q`` metres and
  send it as a 16-bit integer (6 bytes of position instead of 12).

The replica assumes every sent packet arrives (there are no ACKs), so it is
a guess about the receiver, not a view of it. It is the recovery-v1 filter
(``sim.tracker.ConstantVelocityTracker``, the same tuning as the receiver),
started at the first measurement because the sender does not know the truth.

Packet size model: data = sequence number 2 B + timestamp 4 B + position
(12 B as float32, or 6 B as int16). On air the packet carries a fixed 43 B
of ESP-NOW framing. The 43 B figure is literature-derived,
not measured on our hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.tracker import ConstantVelocityTracker

from .base import EstimatorSettings

INT16_MIN = -32768
INT16_MAX = 32767

SEQ_BYTES = 2
TIMESTAMP_BYTES = 4
POSITION_FLOAT32_BYTES = 12
POSITION_INT16_BYTES = 6
# Fixed ESP-NOW overhead per packet on air. Literature-derived,
# NOT measured on our hardware.
ESPNOW_OVERHEAD_BYTES = 43

KINDS = ("periodic", "event")
_PARAM_KEYS = ("kind", "rate_hz", "delta_m", "quantization_m", "heartbeat_s")
_HEARTBEAT_TOLERANCE_S = 1e-12


@dataclass(frozen=True)
class SenderOutput:
    """What the sender transmits for one run.

    send_indices: indices into the capture arrays, increasing.
    sent_measurements: (n, 3) values as sent (already quantized when quantization is on).
    data_bytes_per_packet: seq + timestamp + position bytes.
    on_air_bytes_per_packet: data plus the fixed ESP-NOW overhead.
    """

    send_indices: np.ndarray
    sent_measurements: np.ndarray
    data_bytes_per_packet: int
    on_air_bytes_per_packet: int


def quantize(z, q: float) -> tuple[np.ndarray, np.ndarray]:
    """int16 quantizer: code = clip(round(z / q), -32768, 32767), z_hat = code * q.

    Rounding is NumPy's (half to even). Returns (codes as int16, z_hat as float).
    """
    q = _check_step(q)
    codes = np.clip(np.round(np.asarray(z, dtype=float) / q), INT16_MIN, INT16_MAX).astype(np.int16)
    return codes, dequantize(codes, q)


def dequantize(codes, q: float) -> np.ndarray:
    """Turn int16 codes back into metres."""
    return np.asarray(codes, dtype=np.int16).astype(float) * _check_step(q)


def packet_bytes(quantized: bool) -> dict:
    """Data and on-air size of one packet (see the module docstring)."""
    position = POSITION_INT16_BYTES if quantized else POSITION_FLOAT32_BYTES
    data = SEQ_BYTES + TIMESTAMP_BYTES + position
    return {"data": data, "on_air": data + ESPNOW_OVERHEAD_BYTES}


def apply_sender(capture_times, measurements, params: dict | None, settings: EstimatorSettings,
                 seed: int | None = None) -> SenderOutput:
    """Decide which captured samples are sent under the policy in ``params``.

    params keys (all optional except delta_m for the event kind):
      kind: "periodic" (default) or "event".
      rate_hz: periodic only; send every round(sensor_rate / rate_hz)-th sample
        starting at index 0. Default is the sensor rate (send everything).
      delta_m: event only; send when the sample is at least this far from the
        replica's prediction. 0 sends every sample.
      quantization_m: None (float32 on the wire) or the int16 step in metres.
      heartbeat_s: event only; also send if nothing went out for this long.
    ``seed`` is accepted for the engine's per-stream seeding; both policies are
    deterministic, so it does not change the result.
    """
    params = dict(params or {})
    unknown = sorted(set(params) - set(_PARAM_KEYS))
    if unknown:
        raise ValueError(f"unknown sender params: {unknown}")
    kind = params.get("kind", "periodic")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    times = np.asarray(capture_times, dtype=float)
    z = np.asarray(measurements, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("capture_times must be a non-empty 1D array")
    if z.shape != (times.size, 3):
        raise ValueError("measurements must have shape (n, 3) matching capture_times")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(z))):
        raise ValueError("capture_times and measurements must be finite")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("capture_times must be nonnegative and strictly increasing")
    q = params.get("quantization_m")
    if q is not None:
        q = _check_step(q)

    if kind == "periodic":
        if "delta_m" in params or "heartbeat_s" in params:
            raise ValueError("delta_m and heartbeat_s belong to the event kind")
        indices = _periodic_indices(times, params.get("rate_hz"))
    else:
        if "rate_hz" in params:
            raise ValueError("rate_hz belongs to the periodic kind")
        if "delta_m" not in params:
            raise ValueError("the event kind needs delta_m")
        indices = _event_indices(times, z, float(params["delta_m"]), params.get("heartbeat_s"), q, settings)

    sent = z[indices]
    if q is not None:
        sent = quantize(sent, q)[1]
    sizes = packet_bytes(q is not None)
    return SenderOutput(indices, sent, sizes["data"], sizes["on_air"])


def _check_step(q) -> float:
    q = float(q)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("quantization step must be positive and finite")
    return q


def _periodic_indices(times: np.ndarray, rate_hz) -> np.ndarray:
    n = times.size
    if rate_hz is not None:
        rate_hz = float(rate_hz)
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be positive and finite")
    if rate_hz is None or n == 1:
        stride = 1  # one sample: the sensor rate cannot be inferred, so it is simply sent
    else:
        sensor_rate = 1.0 / float(np.median(np.diff(times)))
        stride = int(round(sensor_rate / rate_hz))
        if stride < 1:
            raise ValueError("rate_hz cannot exceed the sensor rate")
    return np.arange(0, n, stride, dtype=int)


def _event_indices(times: np.ndarray, z: np.ndarray, delta_m: float, heartbeat_s, q, settings) -> np.ndarray:
    if not np.isfinite(delta_m) or delta_m < 0:
        raise ValueError("delta_m must be finite and nonnegative")
    if heartbeat_s is not None:
        heartbeat_s = float(heartbeat_s)
        if not np.isfinite(heartbeat_s) or heartbeat_s <= 0:
            raise ValueError("heartbeat_s must be positive and finite")

    def as_sent(value: np.ndarray) -> np.ndarray:
        return value if q is None else quantize(value, q)[1]

    # Replica of the receiver: starts at the first value as SENT (quantized if
    # quantizing, since that is all the receiver ever sees) with zero velocity,
    # then fuses that same value (the first sample is always sent).
    replica = ConstantVelocityTracker(
        position_m=as_sent(z[0]), velocity_mps=(0.0, 0.0, 0.0), timestamp_s=float(times[0]),
        position_variance_m2=settings.initial_position_variance_m2,
        velocity_variance_m2ps2=settings.initial_velocity_variance_m2ps2,
        acceleration_spectral_density=settings.q_acc_psd,
        measurement_std_m=settings.measurement_std_m,
    )
    replica.update(as_sent(z[0]), float(times[0]))
    sent = [0]
    last_sent_time = float(times[0])
    for i in range(1, times.size):
        t = float(times[i])
        miss = float(np.linalg.norm(z[i] - np.asarray(replica.predict_at(t).position_m)))
        heartbeat_due = heartbeat_s is not None and t - last_sent_time >= heartbeat_s - _HEARTBEAT_TOLERANCE_S
        if miss >= delta_m or heartbeat_due:
            replica.update(as_sent(z[i]), t)
            sent.append(i)
            last_sent_time = t
    return np.asarray(sent, dtype=int)
