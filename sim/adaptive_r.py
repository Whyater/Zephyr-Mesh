"""Opt-in adaptive measurement noise for the S6 recovery tracker (EXP-037, a proposal).

The S6 recovery tracker is told its measurement noise in advance (0.05 m).
When the real noise is different, it trusts each measurement too much or too
little. ``AdaptiveRTracker`` instead learns the noise level while it runs,
with the windowed innovation covariance-matching estimate (Mohamed 1999,
PhD thesis, Eq. 4.15 and 4.16; also Almagbile, Wang and Ding 2010, Eq. 15
and 16):

    C = mean of (innovation x innovation) over the last N fused measurements
    R_hat = C - (the filter's own predicted position variance), averaged the same way

An innovation is the gap between where the filter predicted the target would
be and where the measurement says it is. That gap has two sources: the
filter's own uncertainty about the target, which it already knows (the
predicted variance), and sensor noise. Subtracting the first from the average
squared gap leaves an estimate of the second. Here the three axes share one
number (the sensor noise is the same in every direction), the window holds the
current measurement, and the new estimate is used for that same update, as in
the source. The estimate is held at ``floor_std_m`` squared or more because the
subtraction can go negative. Until the window is full, the starting value
(``measurement_std_m``) is used.

Known weak spots, all from the sources above: the subtraction is only right if
the filter's motion model and process noise are right, so a target that turns
while the filter assumes a straight line shows up as fake extra noise, and
process noise set too high makes the estimate come out too low. After a
measurement gap of g seconds the first innovation's predicted variance grows
roughly as q g^3 / 3, which can drive the estimate to the floor; one such sample
also dominates the window even when q is right. Short windows are noisy; long
ones are slow.

Nothing here is used by any registered run. ``sim/tracker.py`` and
``sim/s6_recovery.py`` are untouched (registered runs hash both files into
their outputs), so this lives in its own module. ``run_recovery_arm`` repeats
the recovery branch of ``sim.s6_recovery.run_arms`` so a tracker can be passed
in; a test checks the two stay bit-identical.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .s6_recovery import (
    BURN_IN_TOLERANCE_S,
    TIME_TOLERANCE_S,
    ArmResult,
    Delivery,
    RecoveryConfig,
    _nis,
    deliver,
    truth_position,
)
from .tracker import ConstantVelocityTracker

R_MODES = ("told", "adaptive")
DEFAULT_WINDOW = 20
DEFAULT_FLOOR_STD_M = 0.001


class AdaptiveRTracker(ConstantVelocityTracker):
    """Constant-velocity tracker that re-estimates its measurement noise from its innovations.

    ``window`` is the number of fused measurements averaged (N above).
    ``r_history`` holds (time, estimated variance) for every update once the
    window is full.
    """

    def __init__(self, *args, window: int = DEFAULT_WINDOW, floor_std_m: float = DEFAULT_FLOOR_STD_M, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(window, bool) or int(window) != window or window < 2:
            raise ValueError("window must be an integer of at least 2")
        floor = float(floor_std_m)
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("floor_std_m must be finite and positive")
        self.window = int(window)
        self.floor_variance = floor**2
        self._surprise: deque[float] = deque(maxlen=self.window)  # squared innovation, per axis
        self._expected: deque[float] = deque(maxlen=self.window)  # predicted position variance, per axis
        self.r_history: list[tuple[float, float]] = []

    @property
    def estimated_std_m(self) -> float:
        return float(np.sqrt(self.R[0, 0]))

    def update(self, measurement_m, timestamp_s: float) -> bool:
        z = np.asarray(measurement_m, dtype=float)
        t = float(timestamp_s)
        if z.shape != (3,) or not np.isfinite(t) or t < self.timestamp_s:
            return super().update(measurement_m, timestamp_s)  # raises with the usual message
        predicted = self.predict_at(t)
        innovation = z - np.asarray(predicted.position_m)
        saved = (list(self._surprise), list(self._expected), self.R.copy())
        self._surprise.append(float(innovation @ innovation) / 3.0)
        self._expected.append(float(np.trace(np.asarray(predicted.covariance)[:3, :3])) / 3.0)
        full = len(self._surprise) == self.window
        try:
            if full:
                r_hat = float(np.mean(self._surprise)) - float(np.mean(self._expected))
                if not np.isfinite(r_hat):
                    # A finite but enormous measurement can overflow when squared;
                    # refuse it before the filter state is touched.
                    raise ValueError("adaptive noise estimate is not finite")
                self.R = np.eye(3) * max(r_hat, self.floor_variance)
            super().update(z, t)
        except ValueError:
            # A rejected measurement (for example a duplicate time) leaves the estimate as it was.
            self._surprise = deque(saved[0], maxlen=self.window)
            self._expected = deque(saved[1], maxlen=self.window)
            self.R = saved[2]
            raise
        if full:
            self.r_history.append((t, float(self.R[0, 0])))
        return True


@dataclass(frozen=True)
class AdaptiveArmResult:
    arm: ArmResult
    final_std_m: float
    # Mean of the square root of the estimated variance over updates in the
    # scored window; None for an arm that was told its noise.
    mean_scored_std_m: float | None


def make_tracker(motion: str, config: RecoveryConfig, *, r_mode: str, measurement_std_m: float,
                 window: int = DEFAULT_WINDOW, floor_std_m: float = DEFAULT_FLOOR_STD_M) -> ConstantVelocityTracker:
    """A tracker built like ``sim.s6_recovery._new_tracker``; ``r_mode`` "adaptive" starts from ``measurement_std_m``."""
    if r_mode not in R_MODES:
        raise ValueError(f"r_mode must be one of {R_MODES}")
    kwargs = {
        "position_m": truth_position(motion, [0.0], config)[0], "velocity_mps": (0.0, 0.0, 0.0), "timestamp_s": 0.0,
        "position_variance_m2": config.initial_position_variance_m2,
        "velocity_variance_m2ps2": config.initial_velocity_variance_m2ps2,
        "acceleration_spectral_density": config.tracker_acceleration_psd,
        "measurement_std_m": float(measurement_std_m),
    }
    if r_mode == "told":
        return ConstantVelocityTracker(**kwargs)
    return AdaptiveRTracker(**kwargs, window=window, floor_std_m=floor_std_m)


def run_recovery_arm(deliveries: list[Delivery], motion: str, config: RecoveryConfig,
                     tracker: ConstantVelocityTracker) -> ArmResult:
    """The recovery arm of ``sim.s6_recovery.run_arms`` with the tracker passed in."""
    query_times = np.arange(0.0, config.duration_s + config.query_dt_s / 2, config.query_dt_s)
    truth = truth_position(motion, query_times, config)
    scored = query_times >= config.burn_in_s - BURN_IN_TOLERANCE_S
    fused = dropped = 0
    last_capture = -np.inf
    nis_values = []
    errors = np.empty(query_times.size)
    cursor = 0
    for i, t in enumerate(query_times):
        while cursor < len(deliveries) and deliveries[cursor].receive_time_s <= t + TIME_TOLERANCE_S:
            packet = deliveries[cursor]
            cursor += 1
            z = np.asarray(packet.measurement_m, dtype=float)
            stamp = packet.capture_time_s
            if stamp <= last_capture:
                dropped += 1
                continue
            if stamp >= config.burn_in_s - BURN_IN_TOLERANCE_S:
                nis_values.append(_nis(tracker, z, stamp))
            tracker.update(z, stamp)
            last_capture = max(last_capture, packet.capture_time_s)
            fused += 1
        estimate = tracker.predict_at(max(float(t), tracker.timestamp_s))
        errors[i] = np.linalg.norm(np.asarray(estimate.position_m) - truth[i])
    window = errors[scored]
    p95 = float(np.percentile(window, 95))
    maximum = float(np.max(window))
    return ArmResult(
        p95_error_m=p95,
        max_error_m=maximum,
        rmse_m=float(np.sqrt(np.mean(window**2))),
        fused_count=fused,
        dropped_count=dropped,
        fused_out_of_order_count=0,
        mean_nis=float(np.mean(nis_values)) if nis_values else None,
        failed=bool(p95 > config.failure_p95_m or maximum > config.failure_max_m),
        query_times_s=query_times,
        errors_m=errors,
    )


def simulate_condition(motion: str, *, delay_s: float, loss_probability: float, noise_std_m: float,
                       loss_model: str, seed: int, config: RecoveryConfig,
                       arms: dict[str, tuple[str, float]], window: int = DEFAULT_WINDOW,
                       floor_std_m: float = DEFAULT_FLOOR_STD_M) -> dict[str, AdaptiveArmResult]:
    """Run several recovery-arm trackers on the same delivered packets.

    ``arms`` maps an arm name to (r_mode, measurement_std_m). Every arm sees the
    same deliveries (common random numbers), so differences come only from
    how each one sets its measurement noise.
    """
    deliveries = deliver(motion, delay_s=delay_s, loss_probability=loss_probability, noise_std_m=noise_std_m,
                         seed=seed, config=config, loss_model=loss_model)
    out = {}
    for name, (r_mode, std) in arms.items():
        tracker = make_tracker(motion, config, r_mode=r_mode, measurement_std_m=std, window=window,
                               floor_std_m=floor_std_m)
        result = run_recovery_arm(deliveries, motion, config, tracker)
        mean_scored = None
        if isinstance(tracker, AdaptiveRTracker):
            scored = [v for time, v in tracker.r_history if time >= config.burn_in_s - BURN_IN_TOLERANCE_S]
            mean_scored = float(np.mean(np.sqrt(scored))) if scored else None
        out[name] = AdaptiveArmResult(result, float(np.sqrt(tracker.R[0, 0])), mean_scored)
    return out
