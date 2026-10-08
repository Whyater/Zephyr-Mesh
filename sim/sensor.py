"""Seeded sensor fixtures that keep truth, measurement, and availability separate.

The model is intentionally small and inspectable. It is a measurement fixture,
not a camera or IMU model. Hardware coefficients stay inputs until experiments
replace them with measured values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SensorConfig:
    """Parameters for a position-like sensor."""

    noise_std_m: float = 0.0
    bias_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    random_walk_std_m: float = 0.0
    dropout_probability: float = 0.0
    burst_start_probability: float = 0.0
    burst_end_probability: float = 1.0
    mode: str = "independent"  # independent or burst

    def __post_init__(self) -> None:
        for name in ("noise_std_m", "random_walk_std_m", "dropout_probability",
                     "burst_start_probability", "burst_end_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("dropout_probability", "burst_start_probability", "burst_end_probability"):
            if float(getattr(self, name)) > 1:
                raise ValueError(f"{name} must be at most one")
        if self.mode not in {"independent", "burst"}:
            raise ValueError("mode must be independent or burst")
        bias = np.asarray(self.bias_m, dtype=float)
        if bias.ndim != 1 or bias.size == 0 or np.any(~np.isfinite(bias)):
            raise ValueError("bias_m must be a finite non-empty vector")


@dataclass(frozen=True)
class SensorSample:
    """One timestamped observation. ``measurement_m`` is None when unavailable."""

    truth_m: tuple[float, ...]
    measurement_m: tuple[float, ...] | None
    available: bool
    timestamp_s: float
    noise_m: tuple[float, ...]
    bias_m: tuple[float, ...]
    dropout_reason: str | None = None
    replay_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "truth_m": list(self.truth_m),
            "measurement_m": None if self.measurement_m is None else list(self.measurement_m),
            "available": self.available,
            "timestamp_s": self.timestamp_s,
            "noise_m": list(self.noise_m),
            "bias_m": list(self.bias_m),
            "dropout_reason": self.dropout_reason,
            "replay_id": self.replay_id,
        }


class SensorModel:
    """Generate deterministic sensor observations from a truth vector."""

    def __init__(self, config: SensorConfig | None = None, *, seed: int | None = None):
        self.config = config or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self._bias = np.asarray(self.config.bias_m, dtype=float).copy()
        self._burst_active = False
        self._last_timestamp = -np.inf
        self._index = 0

    def _dropout(self) -> str | None:
        c = self.config
        if c.mode == "independent":
            return "independent_dropout" if self.rng.random() < c.dropout_probability else None
        if not self._burst_active and self.rng.random() < c.burst_start_probability:
            self._burst_active = True
        if self._burst_active:
            if self.rng.random() < c.burst_end_probability:
                self._burst_active = False
            return "burst_dropout"
        return None

    def sample(self, truth_m: Iterable[float], timestamp_s: float, *, replay_id: str | None = None) -> SensorSample:
        truth = np.asarray(tuple(truth_m), dtype=float)
        timestamp = float(timestamp_s)
        if truth.ndim != 1 or truth.size == 0 or np.any(~np.isfinite(truth)):
            raise ValueError("truth_m must be a finite non-empty vector")
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= self._last_timestamp:
            raise ValueError("timestamp_s must be finite, nonnegative, and strictly increasing")
        if self._bias.size != truth.size:
            raise ValueError("bias_m dimension must match truth_m")
        self._last_timestamp = timestamp
        self._index += 1
        if self.config.random_walk_std_m:
            self._bias += self.rng.normal(0.0, self.config.random_walk_std_m, size=truth.size)
        bias = self._bias.copy()
        noise = self.rng.normal(0.0, self.config.noise_std_m, size=truth.size)
        reason = self._dropout()
        measurement = None if reason else truth + bias + noise
        return SensorSample(
            tuple(truth.tolist()),
            None if measurement is None else tuple(measurement.tolist()),
            reason is None,
            timestamp,
            tuple(noise.tolist()),
            tuple(bias.tolist()),
            reason,
            replay_id or f"sensor-{self._index:06d}",
        )

    def sample_series(self, truth_rows: Iterable[Iterable[float]], timestamps_s: Iterable[float]) -> list[SensorSample]:
        rows = list(truth_rows)
        times = list(timestamps_s)
        if len(rows) != len(times):
            raise ValueError("truth_rows and timestamps_s must have equal length")
        return [self.sample(row, timestamp) for row, timestamp in zip(rows, times)]
