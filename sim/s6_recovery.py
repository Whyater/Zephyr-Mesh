"""S6 recovery sweep: a naive tracker next to capture-time fusion plus predict-forward.

The question: how much one-way delay, packet loss and position noise can a
constant-velocity Kalman tracker take, and how much does stamping each
measurement with the time it was captured (instead of the time it arrived),
then predicting forward to "now", recover?

Both arms run the same ``ConstantVelocityTracker`` with the same fixed tuning
on the same delivered packets (common random numbers), so any difference
between them comes from how they use time:

- naive: fuses a packet when it arrives, stamped with its receive time. A
  measurement that is d seconds old is treated as current, so on a target
  moving at v the estimate trails by about v x d.
- recovery: fuses a packet stamped with its capture time, then answers each
  query with ``predict_at(now)``, which moves the estimate forward without
  moving the filter clock. A packet that arrives after a newer one was already
  fused is dropped and counted. There is no retrodiction (no re-filtering of
  late packets), so a dropped late packet acts like a lost one.

Everything here is synthetic: packets go through ``SimulatedLink`` and
measurements come from ``SensorModel``. Nothing is measured on hardware, and
the numbers are not flight performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import binom, norm

from .link import LinkConfig, SimulatedLink
from .sensor import SensorConfig, SensorModel
from .tracker import ConstantVelocityTracker

MOTIONS = ("constant_velocity", "constant_turn")
ARMS = ("naive", "recovery")
BASE_SEED = 20261013
# Arrival times are compared with a small tolerance so a packet captured at
# k x 0.1 s lands on the matching 0.02 s query even with float rounding.
TIME_TOLERANCE_S = 1e-12
# One tolerance for "is this at or after the end of the warm-up", used both for
# the scored query window and for which fusions count toward the mean NIS.
BURN_IN_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class RecoveryConfig:
    duration_s: float = 12.0
    burn_in_s: float = 2.0
    query_dt_s: float = 0.02
    observation_period_s: float = 0.10
    target_speed_mps: float = 1.0
    turn_radius_m: float = 1.0
    tracker_measurement_std_m: float = 0.05
    tracker_acceleration_psd: float = 0.1
    initial_position_variance_m2: float = 1.0
    initial_velocity_variance_m2ps2: float = 1.0
    failure_p95_m: float = 0.25
    failure_max_m: float = 0.50

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.duration_s, self.query_dt_s, self.observation_period_s, self.turn_radius_m) <= 0:
            raise ValueError("duration, time steps and turn radius must be positive")
        if self.burn_in_s >= self.duration_s:
            raise ValueError("burn_in_s must be shorter than duration_s")


@dataclass(frozen=True)
class Delivery:
    seq: int
    capture_time_s: float
    receive_time_s: float
    measurement_m: tuple[float, float, float]


@dataclass(frozen=True)
class ArmResult:
    p95_error_m: float
    max_error_m: float
    rmse_m: float
    fused_count: int
    dropped_count: int
    fused_out_of_order_count: int
    mean_nis: float | None
    failed: bool
    # Per-query arrays; the sweep runner drops them (None) before summarizing.
    query_times_s: np.ndarray | None
    errors_m: np.ndarray | None


def truth_position(motion: str, times_s, config: RecoveryConfig) -> np.ndarray:
    """True target position, shape (N, 3).

    Straight line: x = v t. Turn: a circle of radius r at speed v starting at
    the origin heading +x and turning left, so the turn rate is w = v / r.
    """
    t = np.asarray(times_s, dtype=float)
    v = config.target_speed_mps
    out = np.zeros((t.size, 3))
    if motion == "constant_velocity":
        out[:, 0] = v * t
    elif motion == "constant_turn":
        r = config.turn_radius_m
        w = v / r
        out[:, 0] = r * np.sin(w * t)
        out[:, 1] = r * (1.0 - np.cos(w * t))
    else:
        raise ValueError(f"motion must be one of {MOTIONS}")
    return out


def seed_list(n_seeds: int, base_seed: int = BASE_SEED) -> list[int]:
    """The per-run seeds. Every cell and both arms use this same list."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least one")
    return [int(v) for v in np.random.SeedSequence(base_seed).generate_state(n_seeds)]


def _burst_start_probability(loss_probability: float, burst_end_probability: float) -> float:
    """Start probability that makes the long-run lost fraction equal ``loss_probability``.

    In ``SimulatedLink``'s burst model a packet is lost with probability s if
    the previous one got through, and with probability 1 if a burst is still
    running; a burst ends after each lost packet with probability e. Solving
    the steady state for the lost fraction p gives s = p e / (1 - p (1 - e)).
    """
    p, e = loss_probability, burst_end_probability
    if p >= 1.0:
        return 1.0
    return p * e / (1.0 - p * (1.0 - e))


def deliver(motion: str, *, delay_s: float, loss_probability: float, noise_std_m: float, seed: int,
            config: RecoveryConfig, loss_model: str = "independent", jitter_s: float = 0.0,
            burst_mean_length_packets: float = 5.0) -> list[Delivery]:
    """Capture every 10 Hz measurement, send it through the link, return what arrives.

    The link and sensor streams depend only on ``seed``, so two cells that
    differ in one setting see the same random draws for everything else.
    Returns deliveries sorted by (receive time, seq); packets that would land
    after the end of the run (beyond TIME_TOLERANCE_S) are not delivered.
    """
    link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
    if loss_model == "burst":
        end = 1.0 / float(burst_mean_length_packets)
        link_config = LinkConfig(delay_s=float(delay_s), delay_jitter_s=float(jitter_s), loss_model="burst",
                                 burst_start_probability=_burst_start_probability(float(loss_probability), end),
                                 burst_end_probability=end)
    elif loss_model == "independent":
        link_config = LinkConfig(delay_s=float(delay_s), delay_jitter_s=float(jitter_s), loss_model="independent",
                                 loss_probability=float(loss_probability))
    else:
        raise ValueError("loss_model must be independent or burst")
    link = SimulatedLink(link_config, seed=link_seed)
    sensor = SensorModel(SensorConfig(noise_std_m=float(noise_std_m)), seed=sensor_seed)
    n_captures = int(round(config.duration_s / config.observation_period_s))
    capture_times = np.arange(n_captures + 1) * config.observation_period_s
    truth = truth_position(motion, capture_times, config)
    measurements = {}
    for seq, (capture_time, position) in enumerate(zip(capture_times, truth)):
        sample = sensor.sample(position, float(capture_time))
        link.send("target", "tracker", seq, float(capture_time))
        measurements[seq] = tuple(float(v) for v in sample.measurement_m)
    deliveries = [
        Delivery(event.seq, float(event.send_time), float(event.receive_time), measurements[event.seq])
        for event in link.deliver_all()
        if event.receive_time <= config.duration_s + TIME_TOLERANCE_S
    ]
    deliveries.sort(key=lambda d: (d.receive_time_s, d.seq))
    return deliveries


def _new_tracker(motion: str, config: RecoveryConfig, measurement_std_m: float) -> ConstantVelocityTracker:
    return ConstantVelocityTracker(
        position_m=truth_position(motion, [0.0], config)[0], velocity_mps=(0.0, 0.0, 0.0), timestamp_s=0.0,
        position_variance_m2=config.initial_position_variance_m2,
        velocity_variance_m2ps2=config.initial_velocity_variance_m2ps2,
        acceleration_spectral_density=config.tracker_acceleration_psd,
        measurement_std_m=measurement_std_m,
    )


def _nis(tracker: ConstantVelocityTracker, measurement: np.ndarray, stamp: float) -> float:
    """Normalized innovation squared for a measurement about to be fused at ``stamp``."""
    predicted = tracker.predict_at(stamp)
    innovation = measurement - np.asarray(predicted.position_m)
    S = np.asarray(predicted.covariance)[:3, :3] + tracker.R
    return float(innovation @ np.linalg.solve(S, innovation))


def run_arms(deliveries: list[Delivery], motion: str, config: RecoveryConfig, *,
             tracker_measurement_std_m: float | None = None) -> dict[str, ArmResult]:
    """Run the naive and recovery arms on the same delivered packets."""
    std = config.tracker_measurement_std_m if tracker_measurement_std_m is None else float(tracker_measurement_std_m)
    query_times = np.arange(0.0, config.duration_s + config.query_dt_s / 2, config.query_dt_s)
    truth = truth_position(motion, query_times, config)
    scored = query_times >= config.burn_in_s - BURN_IN_TOLERANCE_S
    results = {}
    for arm in ARMS:
        tracker = _new_tracker(motion, config, std)
        fused = dropped = fused_out_of_order = 0
        last_capture = -np.inf
        nis_values = []
        errors = np.empty(query_times.size)
        cursor = 0
        for i, t in enumerate(query_times):
            while cursor < len(deliveries) and deliveries[cursor].receive_time_s <= t + TIME_TOLERANCE_S:
                packet = deliveries[cursor]
                cursor += 1
                z = np.asarray(packet.measurement_m, dtype=float)
                if arm == "naive":
                    # The naive arm treats the packet as current when it arrives.
                    stamp = packet.receive_time_s
                    if tracker.update_count > 0 and abs(stamp - tracker.timestamp_s) <= TIME_TOLERANCE_S:
                        dropped += 1
                        continue
                    if packet.capture_time_s < last_capture:
                        fused_out_of_order += 1
                else:
                    # The recovery arm stamps the packet with when it was
                    # captured. A packet older than one already fused is late:
                    # drop it and count it rather than fuse it out of order.
                    stamp = packet.capture_time_s
                    if stamp <= last_capture:
                        dropped += 1
                        continue
                if stamp >= config.burn_in_s - BURN_IN_TOLERANCE_S:
                    nis_values.append(_nis(tracker, z, stamp))
                tracker.update(z, stamp)
                last_capture = max(last_capture, packet.capture_time_s)
                fused += 1
            # Predict forward to the query time without moving the filter clock.
            # max() only absorbs float rounding of at most TIME_TOLERANCE_S.
            estimate = tracker.predict_at(max(float(t), tracker.timestamp_s))
            errors[i] = np.linalg.norm(np.asarray(estimate.position_m) - truth[i])
        window = errors[scored]
        p95 = float(np.percentile(window, 95))
        maximum = float(np.max(window))
        results[arm] = ArmResult(
            p95_error_m=p95,
            max_error_m=maximum,
            rmse_m=float(np.sqrt(np.mean(window**2))),
            fused_count=fused,
            dropped_count=dropped,
            fused_out_of_order_count=fused_out_of_order,
            mean_nis=float(np.mean(nis_values)) if nis_values else None,
            failed=bool(p95 > config.failure_p95_m or maximum > config.failure_max_m),
            query_times_s=query_times,
            errors_m=errors,
        )
    return results


def simulate_run(motion: str, *, delay_s: float, loss_probability: float, noise_std_m: float, seed: int,
                 config: RecoveryConfig, loss_model: str = "independent", jitter_s: float = 0.0,
                 tracker_measurement_std_m: float | None = None) -> dict[str, ArmResult]:
    deliveries = deliver(motion, delay_s=delay_s, loss_probability=loss_probability, noise_std_m=noise_std_m,
                         seed=seed, config=config, loss_model=loss_model, jitter_s=jitter_s)
    return run_arms(deliveries, motion, config, tracker_measurement_std_m=tracker_measurement_std_m)


# ---- statistics --------------------------------------------------------------

def median_interval(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Median and a distribution-free interval for it from order statistics.

    Sort the n values. The chance that the true median lies below the k-th
    smallest is the chance that fewer than k of n coin flips come up heads,
    binom.cdf(k - 1, n, 0.5). Choosing the largest k that keeps that chance
    at or below (1 - confidence) / 2 on each side gives [x(k), x(n - k + 1)].
    For n = 50 that is the 18th and 33rd values (coverage 96.7 %).
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("median_interval needs at least one value")
    median = float(np.median(x))
    tail = (1.0 - confidence) / 2.0
    candidates = [k for k in range(1, n // 2 + 1) if binom.cdf(k - 1, n, 0.5) <= tail]
    if not candidates:
        return median, float(x[0]), float(x[-1])
    k = max(candidates)
    return median, float(x[k - 1]), float(x[n - k])


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion (R-P3-TEST-01)."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_arm(results: list[ArmResult]) -> dict:
    n = len(results)
    p95 = [r.p95_error_m for r in results]
    median, low, high = median_interval(p95)
    fails = sum(1 for r in results if r.failed)
    fail_low, fail_high = wilson_interval(fails, n)
    nis = [r.mean_nis for r in results if r.mean_nis is not None]
    return {
        "n_seeds": n,
        "p95_median_m": median,
        "p95_ci_low_m": low,
        "p95_ci_high_m": high,
        "p95_min_m": float(min(p95)),
        "p95_max_m": float(max(p95)),
        "max_error_median_m": float(np.median([r.max_error_m for r in results])),
        "rmse_median_m": float(np.median([r.rmse_m for r in results])),
        "fail_count": fails,
        "fail_rate": fails / n,
        "fail_rate_ci_low": fail_low,
        "fail_rate_ci_high": fail_high,
        "label": "fails" if 2 * fails >= n else "passes",
        "mean_nis_median": float(np.median(nis)) if nis else None,
        "fused_total": int(sum(r.fused_count for r in results)),
        "dropped_total": int(sum(r.dropped_count for r in results)),
        "fused_out_of_order_total": int(sum(r.fused_out_of_order_count for r in results)),
    }


def paired_summary(naive: list[ArmResult], recovery: list[ArmResult]) -> dict:
    """Seed-by-seed comparison. Both arms saw the same packets, so pair them."""
    if len(naive) != len(recovery) or not naive:
        raise ValueError("paired_summary needs equal, non-empty seed lists")
    diff = np.array([a.p95_error_m - b.p95_error_m for a, b in zip(naive, recovery)])
    median, low, high = median_interval(diff)
    return {
        "n_seeds": int(diff.size),
        "p95_diff_median_m": median,
        "p95_diff_ci_low_m": low,
        "p95_diff_ci_high_m": high,
        "recovery_better_count": int(np.sum(diff > 0)),
        "recovery_worse_count": int(np.sum(diff < 0)),
        "tie_count": int(np.sum(diff == 0)),
    }
