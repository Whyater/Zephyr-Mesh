"""Online synthetic target estimation over a mission link."""
from __future__ import annotations

import math

import numpy as np

from sim.estimation.base import EstimatorSettings
from sim.estimation.factory import make_estimator
from sim.sensor import SensorConfig, SensorModel


def _field(packet, *names, default=None):
    for name in names:
        if hasattr(packet, name):
            return getattr(packet, name)
        if isinstance(packet, dict) and name in packet:
            return packet[name]
    return default


class EstimateFeed:
    """Generate timestamped synthetic fixes and fuse them online.

    ``target`` is a ``sim.estimation.targets.Target``.  Packets are sent at
    the sensor rate during construction, then consumed as ``estimate_at`` is
    queried in nondecreasing time order.  The link and estimator are wholly
    simulated.
    """

    def __init__(self, arm: str, *, target, sensor_noise_std_m: float = 0.05,
                 rate_hz: float = 10.0, link, seed: int, duration_s: float,
                 estimator_settings: EstimatorSettings | None = None) -> None:
        if arm not in {"naive", "recovery", "ct", "imm"}:
            raise ValueError("arm must be naive, recovery, ct, or imm")
        if seed is None:
            raise ValueError("seed is required")
        self.arm = arm
        self.target = target
        self.link = link
        self.seed = int(seed)
        self.rate_hz = float(rate_hz)
        self.duration_s = float(duration_s)
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        if not np.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        noise = float(sensor_noise_std_m)
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("sensor_noise_std_m must be finite and nonnegative")
        sensor_seed = int(np.random.SeedSequence(self.seed).generate_state(2)[1])
        sensor = SensorModel(SensorConfig(noise_std_m=noise), seed=sensor_seed)
        period = 1.0 / self.rate_hz
        count = int(round(self.duration_s * self.rate_hz))
        self.capture_times_s = np.arange(count + 1, dtype=float) * period
        truth = np.asarray(target.position(self.capture_times_s), dtype=float)
        self.measurements_m: list[tuple[float, ...]] = []
        for t, row in zip(self.capture_times_s, truth):
            sample = sensor.sample(row, float(t))
            if sample.measurement_m is None:
                continue
            seq = len(self.measurements_m)
            self.measurements_m.append(tuple(float(v) for v in sample.measurement_m))
            link.send(seq, float(t), self.measurements_m[-1])
        self._estimator = make_estimator(
            "cv" if arm in {"naive", "recovery"} else arm,
            estimator_settings or EstimatorSettings(), target.start_position,
        )
        self._newest_capture = -math.inf
        self._last_query = -math.inf
        self._fused = 0

    @classmethod
    def from_engine_link(cls, arm: str, *, cell, seed: int, duration_s: float = 12.0,
                         estimator_settings: EstimatorSettings | None = None) -> "EstimateFeed":
        """Build a feed from the estimator engine's own seeded packet bundle.

        ``MissionLink`` deliberately has mission stream seeds, while the Run B
        engine uses its historical ``SimulatedLink`` seed split. This seam
        keeps the online loop directly comparable to ``run_arm`` without
        changing either seeding convention.
        """
        if arm not in {"naive", "recovery", "ct", "imm"}:
            raise ValueError("arm must be naive, recovery, ct, or imm")
        from sim.estimation.engine import deliver_cell
        from sim.s6_recovery import RecoveryConfig

        config = RecoveryConfig(duration_s=float(duration_s))
        bundle = deliver_cell(cell, int(seed), config=config)

        class _BundleLink:
            def __init__(self, deliveries):
                self._deliveries = list(deliveries)
                self._used = 0

            def deliveries_until(self, now_s):
                ready = [item for item in self._deliveries[self._used:]
                         if item.receive_time_s <= float(now_s) + 1e-12]
                self._used += len(ready)
                return ready

        obj = cls.__new__(cls)
        obj.arm = arm
        obj.target = bundle.target
        obj.link = _BundleLink(bundle.deliveries)
        obj.seed = int(seed)
        obj.rate_hz = float(cell.rate_hz)
        obj.duration_s = float(duration_s)
        obj.capture_times_s = np.asarray(bundle.capture_times_s, dtype=float)
        obj.measurements_m = [tuple(float(v) for v in d.measurement_m) for d in bundle.deliveries]
        obj._estimator = make_estimator(
            "cv" if arm in {"naive", "recovery"} else arm,
            estimator_settings or EstimatorSettings(), bundle.target.start_position,
        )
        obj._newest_capture = -math.inf
        obj._last_query = -math.inf
        obj._fused = 0
        return obj

    def estimate_at(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Return position, velocity, and age of the newest fused capture."""
        now = float(t)
        if not np.isfinite(now) or now < 0:
            raise ValueError("t must be finite and nonnegative")
        if now < self._last_query:
            raise ValueError("estimate_at queries must be nondecreasing")
        self._last_query = now
        packets = self.link.deliveries_until(now)
        for packet in packets:
            measurement = _field(packet, "message", "measurement_m")
            if measurement is None:
                continue
            receive = float(_field(packet, "receive_time_s", "receive_time"))
            capture = float(_field(packet, "send_time_s", "send_time", "capture_time_s"))
            stamp = receive if self.arm == "naive" else capture
            if stamp < self._estimator.timestamp_s - 1e-12:
                continue
            try:
                self._estimator.update(np.asarray(measurement, dtype=float), stamp)
            except ValueError:
                continue
            self._fused += 1
            self._newest_capture = max(self._newest_capture, capture)
        pred = self._estimator.predict_at(max(now, self._estimator.timestamp_s))
        pred_position = np.asarray(pred.position_m, dtype=float)
        pred_velocity = np.asarray(pred.velocity_mps, dtype=float)
        age = math.inf if not np.isfinite(self._newest_capture) else now - self._newest_capture
        return pred_position.copy(), pred_velocity.copy(), float(age)

