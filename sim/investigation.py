"""Deterministic S6 latency, loss, and sensing-noise investigation fixtures.

This module answers a bounded model question: how does a simple constant
velocity estimator behave as observation age, dropout, and position noise vary?
It is deliberately separate from flight performance and from the S2 rigid-body
model. The returned rows retain the tested condition and criterion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .tracker import ConstantVelocityTracker


@dataclass(frozen=True)
class SweepConfig:
    duration_s: float = 8.0
    truth_dt_s: float = 0.02
    observation_period_s: float = 0.10
    target_speed_mps: float = 0.7
    target_acceleration_mps2: float = 0.0
    failure_p95_m: float = 0.25
    failure_max_m: float = 0.50
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("duration_s", "truth_dt_s", "observation_period_s", "target_speed_mps",
                     "target_acceleration_mps2", "failure_p95_m", "failure_max_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.duration_s <= 0 or self.truth_dt_s <= 0 or self.observation_period_s <= 0:
            raise ValueError("duration, timestep, and observation period must be positive")
        if self.target_speed_mps < 0 or self.failure_p95_m < 0 or self.failure_max_m < 0:
            raise ValueError("speed and failure criteria must be nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")


def _run_case(delay_s, loss_probability, noise_std_m, *, config: SweepConfig, seed: int):
    times = np.arange(0.0, config.duration_s + config.truth_dt_s / 2, config.truth_dt_s)
    truth_x = config.target_speed_mps * times + 0.5 * config.target_acceleration_mps2 * times**2
    truth = np.c_[truth_x, np.zeros((len(times), 2))]
    observation_times = np.arange(0.0, config.duration_s + 1e-12, config.observation_period_s)
    rng = np.random.default_rng(seed)
    pending = []
    for capture_time in observation_times:
        if rng.random() >= loss_probability:
            receive_at = float(capture_time + delay_s)
            if receive_at <= times[-1]:
                position = np.array((config.target_speed_mps * capture_time +
                                     0.5 * config.target_acceleration_mps2 * capture_time**2, 0.0, 0.0))
                pending.append((receive_at, float(capture_time), position + rng.normal(0, noise_std_m, 3)))
    pending.sort(key=lambda row: (row[0], row[1]))

    tracker = ConstantVelocityTracker(
        position_m=truth[0], velocity_mps=(0.0, 0.0, 0.0),
        measurement_std_m=max(noise_std_m, 1e-6), acceleration_spectral_density=0.05,
    )
    errors = []
    hold_errors = []
    ages = []
    last_measurement = truth[0].copy()
    cursor = 0
    for t, expected in zip(times, truth):
        ready = []
        while cursor < len(pending) and pending[cursor][0] <= t + 1e-12:
            receive_at, capture_time, measurement = pending[cursor]
            # The observation is consumed at receipt time. Its stale age is
            # retained separately instead of rewinding the estimator clock.
            ready.append(measurement)
            last_measurement = measurement
            ages.append(max(0.0, t - capture_time))
            cursor += 1
        # The display grid can contain multiple arrivals. Apply the newest
        # one once at this grid time so tracker timestamps remain monotonic.
        if ready:
            tracker.update(ready[-1], float(t))
        state = tracker.predict(float(t))
        errors.append(float(np.linalg.norm(np.asarray(state.position_m) - expected)))
        hold_errors.append(float(np.linalg.norm(last_measurement - expected)))
    errors = np.asarray(errors)
    hold_errors = np.asarray(hold_errors)
    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    hold_p95 = float(np.percentile(hold_errors, 95))
    hold_max = float(np.max(hold_errors))
    return {
        "delay_s": float(delay_s), "loss_probability": float(loss_probability),
        "noise_std_m": float(noise_std_m), "target_acceleration_mps2": float(config.target_acceleration_mps2),
        "p95_error_m": p95, "max_error_m": maximum,
        "held_last_p95_error_m": hold_p95, "held_last_max_error_m": hold_max,
        "mean_age_s": float(np.mean(ages)) if ages else None,
        "sample_count": int(len(errors)), "criterion": f"p95>{config.failure_p95_m} m OR max>{config.failure_max_m} m",
        "label": "failure-boundary" if p95 > config.failure_p95_m or maximum > config.failure_max_m else "within-scenario-envelope",
    }


def run_sweep(*, config: SweepConfig | None = None, delays=(0.0, 0.05, 0.10, 0.20),
              losses=(0.0, 0.10, 0.30), noises=(0.0, 0.02, 0.05), accelerations=(0.0, 0.5)) -> dict:
    """Return a reproducible grid plus manifest and hand-check statement."""
    cfg = config or SweepConfig()
    def values(name, values, *, lower=0.0, upper=None):
        out = []
        for value in values:
            value = float(value)
            if not np.isfinite(value) or value < lower or (upper is not None and value > upper):
                raise ValueError(f"{name} values must be finite and within bounds")
            out.append(value)
        if not out:
            raise ValueError(f"{name} must not be empty")
        return out
    delays = values("delay", delays)
    losses = values("loss", losses, upper=1.0)
    noises = values("noise", noises)
    accelerations = values("acceleration", accelerations)
    rows = []
    index = 0
    for delay in delays:
        for loss in losses:
            for noise in noises:
                for acceleration in accelerations:
                    case_config = SweepConfig(**{**asdict(cfg), "target_acceleration_mps2": float(acceleration)})
                    rows.append(_run_case(delay, loss, noise, config=case_config, seed=cfg.seed + index))
                    index += 1
    return {
        "schema": "zephyr-s6-sweep-1", "status": "synthetic investigation fixture; not flight performance",
        "config": asdict(cfg), "rows": rows,
        "independent_check": "For a held-last-position estimate, v x L = 0.7 m/s x 0.10 s = 0.07 m before noise or dropout.",
        "criterion_note": "The failure label is a scenario-specific research criterion, not a safety threshold.",
    }
