"""Deterministic S6 latency, loss, and sensing-noise investigation fixtures.

This module answers a bounded model question: how does a simple constant
velocity estimator behave as observation age, dropout, and position noise vary?
It is deliberately separate from flight performance and from the S2 rigid-body
model. The returned rows retain the tested condition and criterion.

What the estimator arm is, plainly: a NAIVE receive-time estimator. Each
measurement is applied when it arrives and is stamped with its arrival time, not
the time it was captured. For a target moving at speed v, a measurement that is
L seconds old therefore sits about v x L behind the truth, so the error grows
like speed x delay by construction. No recovery arm exists yet (capture-time
fusion plus predict-forward is not built), so this sweep does not test what
recovers a tracker. The tracker is also given the true noise level
(measurement_std is set to it), which is oracle tuning and flatters the filter.

Packets travel through the tested ``SimulatedLink`` and measurements come from
the tested ``SensorModel``. Every cell uses the same seed (common random
numbers), so differences between cells come from the setting that changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .link import LinkConfig, SimulatedLink
from .sensor import SensorConfig, SensorModel
from .tracker import ConstantVelocityTracker

# Speed the rest of the sim treats as the vehicle limit (adapters.Capabilities
# max_speed_mps). A pinned test keeps this equal to that value.
REFERENCE_MAX_SPEED_MPS = 2.0
# Peak speed = 0.7 + a x 8 s. With a = 0.15 m/s^2 that is 0.7 + 1.2 = 1.9 m/s,
# inside the 2.0 m/s limit. The old default a = 0.5 reached 0.7 + 4.0 = 4.7 m/s.
DEFAULT_ACCELERATIONS_MPS2 = (0.0, 0.15)


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


def peak_target_speed(config: SweepConfig) -> float:
    """Largest |velocity| of the target over the run, from v(t) = v0 + a t."""
    v0 = config.target_speed_mps
    v_end = v0 + config.target_acceleration_mps2 * config.duration_s
    return float(max(abs(v0), abs(v_end)))


def _delivered_measurements(delay_s, loss_probability, noise_std_m, *, config: SweepConfig, seed: int):
    """Send every observation through the tested link and sensor.

    Returns (receive_time, capture_time, measurement) rows sorted by arrival.
    The link and sensor streams depend only on ``seed``: loss draws one uniform
    per packet and noise draws one normal per sample whatever the settings are,
    so two cells that differ only in loss or noise share the other's draws.
    """
    link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
    link = SimulatedLink(
        LinkConfig(delay_s=float(delay_s), loss_model="independent", loss_probability=float(loss_probability)),
        seed=link_seed,
    )
    sensor = SensorModel(SensorConfig(noise_std_m=float(noise_std_m)), seed=sensor_seed)
    observation_times = np.arange(0.0, config.duration_s + 1e-12, config.observation_period_s)
    measurements = {}
    for seq, capture_time in enumerate(observation_times):
        truth = (config.target_speed_mps * capture_time +
                 0.5 * config.target_acceleration_mps2 * capture_time**2, 0.0, 0.0)
        sample = sensor.sample(truth, float(capture_time))
        link.send("target", "tracker", seq, float(capture_time))
        measurements[seq] = np.asarray(sample.measurement_m, dtype=float)
    last_time = config.duration_s
    delivered = []
    for event in link.advance(last_time + config.truth_dt_s / 2):
        if event.receive_time <= last_time:
            delivered.append((float(event.receive_time), float(event.send_time), measurements[event.seq]))
    delivered.sort(key=lambda row: (row[0], row[1]))
    return delivered


def _run_case(delay_s, loss_probability, noise_std_m, *, config: SweepConfig, seed: int):
    times = np.arange(0.0, config.duration_s + config.truth_dt_s / 2, config.truth_dt_s)
    truth_x = config.target_speed_mps * times + 0.5 * config.target_acceleration_mps2 * times**2
    truth = np.c_[truth_x, np.zeros((len(times), 2))]
    pending = _delivered_measurements(delay_s, loss_probability, noise_std_m, config=config, seed=seed)

    # measurement_std is set to the TRUE noise (oracle tuning). A real tracker
    # would not know it; mismatched tuning is not swept here.
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
            # NAIVE receive-time estimator: the observation is consumed at
            # receipt time and stamped with that time, not its capture time.
            # Its stale age is retained separately instead of rewinding the
            # estimator clock, so error grows like speed x delay by design.
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
              losses=(0.0, 0.10, 0.30), noises=(0.0, 0.02, 0.05),
              accelerations=DEFAULT_ACCELERATIONS_MPS2) -> dict:
    """Return a reproducible grid plus manifest and hand-check statement.

    Every cell uses ``cfg.seed`` (common random numbers). The estimator is the
    naive receive-time arm described in the module docstring.
    """
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

    def case_config(acceleration):
        return SweepConfig(**{**asdict(cfg), "target_acceleration_mps2": float(acceleration)})

    rows = []
    reference_failures = {}
    for delay in delays:
        for loss in losses:
            for noise in noises:
                for acceleration in accelerations:
                    row = _run_case(delay, loss, noise, config=case_config(acceleration), seed=cfg.seed)
                    if row["label"] == "failure-boundary":
                        # Same delay and target motion, loss and noise switched
                        # off, same seed. If this also fails, loss and noise are
                        # not what drives the label.
                        key = (delay, acceleration)
                        if key not in reference_failures:
                            reference = _run_case(delay, 0.0, 0.0, config=case_config(acceleration), seed=cfg.seed)
                            reference_failures[key] = reference["label"] == "failure-boundary"
                        if reference_failures[key]:
                            row["criterion"] += ("; also fails with zero loss and zero noise, so delay and target "
                                                 "motion drive this label, not loss or noise")
                        else:
                            row["criterion"] += "; passes with zero loss and zero noise, so loss or noise contributes"
                    rows.append(row)
    peak = max(peak_target_speed(case_config(a)) for a in accelerations)
    peak_text = (f"Peak target speed in this grid is {peak:.3g} m/s against the {REFERENCE_MAX_SPEED_MPS:g} m/s "
                 "reference vehicle limit" + ("." if peak <= REFERENCE_MAX_SPEED_MPS + 1e-12 else
                                              " (EXCEEDED: failure labels may reflect an unflyable target)."))
    worst_delay = max(delays)
    return {
        "schema": "zephyr-s6-sweep-1", "status": "synthetic investigation fixture; not flight performance",
        "config": asdict(cfg), "rows": rows,
        "independent_check": (
            "The estimator arm is a receive-time (naive) tracker: it applies each measurement at arrival, "
            "stamped with its arrival time, so its error grows like speed x delay by construction: "
            f"v x delay = {cfg.target_speed_mps:g} m/s x {worst_delay:g} s = {cfg.target_speed_mps * worst_delay:.3g} m "
            "at the largest delay (plus up to one display step). "
            "Separately, the held-last observation-period lag is speed x observation period = "
            f"{cfg.target_speed_mps:g} m/s x {cfg.observation_period_s:g} s = "
            f"{cfg.target_speed_mps * cfg.observation_period_s:g} m before noise or dropout. "
            "No recovery arm exists yet, so this sweep does not show what recovers a tracker. "
            "Packets use SimulatedLink and measurements use SensorModel, one shared seed for every cell."
        ),
        "criterion_note": (
            "The failure label is a scenario-specific research criterion, not a safety threshold. "
            "Each failure row's criterion text says whether it still fails with zero loss and zero noise. "
            + peak_text + " "
            "The tracker is told the true noise (oracle tuning), a known limit that flatters it. "
            "One seed per cell: no intervals are reported, so small differences are not evidence."
        ),
    }
