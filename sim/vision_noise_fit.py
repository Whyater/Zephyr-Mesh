"""Fit the sim's measurement noise parameters from one detection log.

This is a statistical fit of ONE log that carries ground truth. It inverts the
small model in ``sim/sensor.py`` (truth + bias + Normal noise, plus independent
or burst dropout) and the latency convention used by ``Scenario``. It is not a
camera model. A fit of a synthetic or fixture log says nothing about a real
camera, and a fit of one measured log says only what that one run showed.
"""
from __future__ import annotations

import math
import operator
from itertools import pairwise
from statistics import NormalDist
from typing import Any

import numpy as np

from sim.vision_trace import VisionObservation, VisionTrace

FIT_SCHEMA = "zephyr-vision-noise-fit-1"
DROPOUT_MODES = ("independent", "burst")
_AXES = ("x", "y", "z")
_EVIDENCE_BOUNDARY = (
    "This is a fit of one detection log to the sim's sensor model. A synthetic or "
    "fixture log says nothing about a real camera, and one measured log describes "
    "only that run, that target, and that setup."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(value: float | None) -> float | None:
    """Plain JSON-safe float, or None for missing or non-finite values."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _as_int(value: Any, name: str) -> int:
    if type(value) is bool:
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_float(value: Any, name: str) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _record(estimate, low, high, method: str, n: int) -> dict[str, Any]:
    return {"estimate": _f(estimate), "ci_low": _f(low), "ci_high": _f(high), "method": method, "n": int(n)}


def _block_length(n: int) -> int:
    # Block length rule: about the cube root of the series length, rounded up,
    # but never more than a quarter of the series so there are always at least
    # four blocks to shuffle. Blocks this long keep short-range correlation
    # (neighbouring frames that err the same way) inside each block. The cube
    # root is the usual rule of thumb for bootstrap means and variances; it is
    # not tuned to this log, so a log with very slow drift needs longer blocks.
    return max(1, min(math.ceil(n ** (1.0 / 3.0)), n // 4))


def _block_bootstrap_moments(x: np.ndarray, n_bootstrap: int, block: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap of the per-column mean and sample variance (ddof=1).

    Each replicate glues together randomly chosen runs of ``block`` consecutive
    values until it is ``n`` long (the last block is cut short). Instead of
    building the resampled arrays, block sums come from prefix sums, so memory
    is (replicates x blocks) rather than (replicates x n).
    """
    n, d = x.shape
    center = x.mean(axis=0)
    y = x - center  # centering first keeps the variance formula numerically safe
    cs1 = np.vstack([np.zeros((1, d)), np.cumsum(y, axis=0)])
    cs2 = np.vstack([np.zeros((1, d)), np.cumsum(y * y, axis=0)])
    full, rest = divmod(n, block)
    n_blocks = full + (1 if rest else 0)
    starts = rng.integers(0, n - block + 1, size=(n_bootstrap, n_blocks))
    ends = starts + block
    if rest:
        ends[:, -1] = starts[:, -1] + rest
    s1 = (cs1[ends] - cs1[starts]).sum(axis=1)
    s2 = (cs2[ends] - cs2[starts]).sum(axis=1)
    mean_c = s1 / n
    var = np.maximum((s2 - n * mean_c * mean_c) / (n - 1), 0.0)
    return mean_c + center, var


def _percentile_ci(samples: np.ndarray, level: float) -> tuple[float | None, float | None]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None, None
    alpha = 1.0 - level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _wilson(k: int, n: int, z: float) -> tuple[float, float]:
    # Wilson score interval for a proportion k/n. Unlike the plain normal
    # interval p +/- z*sqrt(p(1-p)/n), it never leaves [0, 1] and it does not
    # collapse to zero width when k is 0 or n, which matters for rare dropouts.
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _variance_of_sample_variance(column: np.ndarray) -> float:
    # Large-sample variance of the sample variance s^2:
    #   Var(s^2) ~ (m4 - s^4 (n - 3)/(n - 1)) / n,  m4 = mean of (x - mean)^4.
    # For a Gaussian this is about 2 s^4 / (n - 1). Using the measured fourth
    # moment instead keeps the interval honest when the data have heavy tails
    # or skew (latency is right-skewed). The spread interval then uses the
    # delta method: se(s) = se(s^2) / (2 s).
    n = column.size
    y = column - column.mean()
    s2 = float(np.dot(y, y)) / (n - 1)
    m4 = float(np.mean(y ** 4))
    return max((m4 - s2 * s2 * (n - 3) / (n - 1)) / n, 0.0)


def _lag1_autocorrelation(column: np.ndarray) -> float | None:
    if column.size < 3:
        return None
    y = column - column.mean()
    denom = float(np.dot(y, y))
    if denom == 0.0:
        return None
    return float(np.dot(y[:-1], y[1:]) / denom)


def _end_from(p_da: np.ndarray | float, s: np.ndarray | float) -> np.ndarray:
    # P(drop -> available) = e * (1 - s), so e = P(drop -> available) / (1 - s),
    # clipped into [0, 1]. When s is 1 the chain never stays available and e
    # cannot be separated from s; report 1.0 there.
    p_da = np.asarray(p_da, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(s < 1.0, p_da / (1.0 - s), 1.0)
    return np.clip(e, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Selection and validation
# ---------------------------------------------------------------------------

def _select(trace: VisionTrace, subject_id: str | None) -> tuple[str, list[VisionObservation]]:
    subjects = sorted({item.subject_id for item in trace.observations})
    if subject_id is None:
        if len(subjects) > 1:
            raise ValueError(f"trace has more than one subject_id ({', '.join(subjects)}); choose one with subject_id")
        if not subjects:
            raise ValueError("trace has no observations to fit")
        subject_id = subjects[0]
    elif subject_id not in subjects:
        raise ValueError(f"subject_id {subject_id!r} is not in the trace")
    records = [item for item in trace.observations if item.subject_id == subject_id]
    records.sort(key=lambda item: (item.capture_time_s, item.observation_id))
    return subject_id, records


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_vision_noise(trace: VisionTrace, *, subject_id: str | None = None, dropout_mode: str = "independent",
                     n_bootstrap: int = 1000, seed: int = 0, confidence_level: float = 0.95) -> dict:
    """Fit SensorConfig and Scenario delay parameters from one log with ground truth."""

    if dropout_mode not in DROPOUT_MODES:
        raise ValueError(f"dropout_mode must be one of {list(DROPOUT_MODES)}")
    n_bootstrap = _as_int(n_bootstrap, "n_bootstrap")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be zero or positive")
    seed = _as_int(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    level = _as_float(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)

    subject_id, records = _select(trace, subject_id)
    frames = sorted({item.position_frame for item in records if item.position_m is not None})
    if len(frames) > 1:
        raise ValueError(f"position_frame differs across records ({', '.join(frames)}); a fit needs one frame")

    residual_rows: list[list[float]] = []
    truth_uncertainty: list[float] = []
    for item in records:
        if item.outcome == "detected" and item.position_m is not None and item.ground_truth is not None:
            truth = item.ground_truth["position_m"]
            residual_rows.append([item.position_m[i] - truth[i] for i in range(3)])
            truth_uncertainty.append(float(item.ground_truth["uncertainty_m"]))
    n_res = len(residual_rows)
    if n_res < 2:
        raise ValueError(f"need at least 2 detected records with position_m and ground_truth; found {n_res}")

    counts = {
        "observed": len(records),
        "detected": sum(item.outcome == "detected" for item in records),
        "detected_with_truth": n_res,
        "missed": sum(item.outcome == "missed" for item in records),
        "unknown": sum(item.outcome == "unknown" for item in records),
    }

    # One seeded stream per bootstrap so changing one part never shifts another.
    rng_res, rng_lat, rng_burst = (np.random.default_rng(child) for child in np.random.SeedSequence(int(seed)).spawn(3))

    # -- Bias and spread ----------------------------------------------------
    # Bias per axis is the plain average residual (measured minus truth).
    # Spread per axis is the sample standard deviation with ddof=1 (divide by
    # n - 1, because the mean was estimated from the same numbers).
    # The sim has one noise number for all three axes. We pool by averaging the
    # three variances and taking the square root (the RMS spread). Averaging
    # variances, not standard deviations, is what keeps the total squared error
    # the same as the per-axis model would give.
    res = np.asarray(residual_rows, dtype=float)
    mean = res.mean(axis=0)
    var = res.var(axis=0, ddof=1)
    std = np.sqrt(var)
    pooled = float(np.sqrt(var.mean()))
    lag1 = [_lag1_autocorrelation(res[:, i]) for i in range(3)]
    # Outlier check, reported only (it does not change noise_std_m). The robust
    # spread is 1.4826 times the median absolute deviation from the median; for
    # a true Gaussian it matches the standard deviation, while a few wild
    # frames barely move it. The tail fraction counts residuals more than 4
    # sample standard deviations from the mean; a Gaussian puts about 0.006%
    # of points there.
    robust_std = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beyond_4 = np.where(std > 0, np.mean(np.abs(res - mean) > 4.0 * std, axis=0), 0.0)

    # Ground-truth error is treated as one shared offset for the whole run
    # (1 sigma = the largest per-record uncertainty_m, the cautious choice).
    # A shared offset moves every residual the same way, so it cannot be seen
    # in the data and adds straight onto the bias uncertainty. It is added in
    # quadrature (square, add, square root) to each side of the bias interval.
    truth_sigma = max(truth_uncertainty)
    truth_half = z * truth_sigma

    res_block = _block_length(n_res)
    if n_bootstrap > 0:
        # Moving-block bootstrap over the residuals in capture order. Frames a
        # few apart from a real camera tend to err the same way, so resampling
        # single frames would make the intervals too narrow. Resampling whole
        # blocks keeps that short-range correlation. Percentile intervals.
        boot_mean, boot_var = _block_bootstrap_moments(res, n_bootstrap, res_block, rng_res)
        bias_ci = [_percentile_ci(boot_mean[:, i], level) for i in range(3)]
        std_ci = [_percentile_ci(np.sqrt(boot_var[:, i]), level) for i in range(3)]
        pooled_ci = _percentile_ci(np.sqrt(boot_var.mean(axis=1)), level)
        res_method = f"moving-block bootstrap, block length {res_block}, {n_bootstrap} resamples, percentile interval"
    else:
        # Analytic fallback: normal approximation. The bias standard error uses
        # an effective sample size n(1 - r)/(1 + r) from the lag-1
        # autocorrelation r (clipped to [0, 0.99]), which shrinks n when
        # neighbouring residuals move together. The spread intervals use the
        # fourth-moment standard error of the variance (see
        # _variance_of_sample_variance), which stays honest for heavy tails.
        bias_ci = []
        for i in range(3):
            r = min(max(lag1[i] or 0.0, 0.0), 0.99)
            n_eff = max(n_res * (1.0 - r) / (1.0 + r), 1.0)
            half = z * std[i] / math.sqrt(n_eff)
            bias_ci.append((mean[i] - half, mean[i] + half))
        var_of_var = [_variance_of_sample_variance(res[:, i]) for i in range(3)]
        std_ci = []
        for i in range(3):
            se = math.sqrt(var_of_var[i]) / (2.0 * std[i]) if std[i] > 0 else 0.0
            std_ci.append((max(std[i] - z * se, 0.0), std[i] + z * se))
        # pooled^2 is the mean of three variances, so its variance is
        # (sum of the three variances-of-variance) / 9, treating axes as independent.
        se_pooled = math.sqrt(sum(var_of_var) / 9.0) / (2.0 * pooled) if pooled > 0 else 0.0
        pooled_ci = (max(pooled - z * se_pooled, 0.0), pooled + z * se_pooled)
        res_method = "normal approximation (no bootstrap): bias uses a lag-1 effective sample size, spread uses a fourth-moment standard error"

    bias_records = {}
    for i, axis in enumerate(_AXES):
        low, high = bias_ci[i]
        lo_half = math.hypot(mean[i] - low, truth_half) if low is not None else None
        hi_half = math.hypot(high - mean[i], truth_half) if high is not None else None
        method = res_method + (f"; ground-truth 1-sigma {truth_sigma:g} m added in quadrature" if truth_sigma > 0 else "; ground-truth uncertainty is 0")
        bias_records[f"bias_m_{axis}"] = _record(mean[i], None if lo_half is None else mean[i] - lo_half,
                                                 None if hi_half is None else mean[i] + hi_half, method, n_res)
    noise_record = _record(pooled, pooled_ci[0], pooled_ci[1], res_method + "; pooled as sqrt(mean of the 3 axis variances)", n_res)

    finite_std = [float(v) for v in std]
    anisotropy = max(finite_std) / min(finite_std) if min(finite_std) > 0 else None
    per_axis = {
        "axes": list(_AXES),
        "position_frame": frames[0] if frames else None,
        "mean_m": [float(v) for v in mean],
        "std_m": finite_std,
        "std_ci_low_m": [_f(lo) for lo, _ in std_ci],
        "std_ci_high_m": [_f(hi) for _, hi in std_ci],
        "robust_std_m": [float(v) for v in robust_std],
        "fraction_beyond_4_std": [float(v) for v in beyond_4],
        "lag1_autocorrelation": [_f(v) for v in lag1],
        "anisotropy_ratio": _f(anisotropy),
        "pooled_std_m": pooled,
        "bootstrap_block_length": res_block,
    }

    # -- Dropout, independent view -----------------------------------------
    # Miss fraction = missed / (detected + missed). Unknown records are left
    # out: we do not know whether the camera saw the target on those frames.
    # Interval: Wilson score (always, even with a bootstrap requested), because
    # it behaves well when misses are rare. It assumes misses are independent;
    # if misses come in bursts it is too narrow, and burst mode is the better fit.
    n_drop_den = counts["detected"] + counts["missed"]
    drop_hat = counts["missed"] / n_drop_den if n_drop_den else 0.0
    drop_ci = _wilson(counts["missed"], n_drop_den, z)
    dropout_record = _record(drop_hat, drop_ci[0], drop_ci[1], "missed/(detected+missed), Wilson score interval", n_drop_den)

    # -- Dropout, burst view (2-state Markov chain) --------------------------
    # In sim/sensor.py burst mode, after an available frame the next frame is
    # dropped with probability s (a burst starts). After a dropped frame the
    # burst ends with probability e, but the next frame can start a new burst
    # straight away, so P(dropped -> available) = e * (1 - s).
    # Maximum likelihood: count the four kinds of step between neighbouring
    # frames (AA, AD, DA, DD). s_hat = n_AD / (n_AD + n_AA),
    # p_DA_hat = n_DA / (n_DA + n_DD), e_hat = p_DA_hat / (1 - s_hat).
    # An unknown record breaks the chain: no step is counted across it.
    segments: list[list[bool]] = []  # True = dropped
    current: list[bool] = []
    for item in records:
        if item.outcome == "unknown":
            if current:
                segments.append(current)
            current = []
        else:
            current.append(item.outcome == "missed")
    if current:
        segments.append(current)
    n_aa = n_ad = n_da = n_dd = 0
    for seg in segments:
        for prev, nxt in pairwise(seg):
            if not prev:
                n_ad += nxt
                n_aa += not nxt
            else:
                n_dd += nxt
                n_da += not nxt
    n_from_a = n_aa + n_ad
    n_from_d = n_da + n_dd
    s_hat = n_ad / n_from_a if n_from_a else 0.0
    p_hat = n_da / n_from_d if n_from_d else None
    e_hat = float(_end_from(p_hat, s_hat)) if p_hat is not None else 1.0

    # Identifiability. s needs at least one step out of the available state.
    # e needs at least one step out of the dropped state, and s must be known
    # and below 1, because e = P(D->A) / (1 - s). A parameter that is not
    # identifiable is never filled with a made-up value: burst mode refuses to
    # fit, and independent mode falls back to the SensorConfig default and
    # reports the uncertainty record as null.
    counts_text = f"(transition counts AA={n_aa}, AD={n_ad}, DA={n_da}, DD={n_dd})"
    s_reason = None if n_from_a else "no steps out of the available state"
    if not n_from_d:
        e_reason = "no steps out of the dropped state"
    elif s_reason is not None:
        e_reason = "burst_start_probability is not identifiable, and e = P(D->A)/(1-s) needs it"
    elif s_hat >= 1.0:
        e_reason = "every step out of the available state is a drop (s = 1), so e cannot be separated from s"
    else:
        e_reason = None
    if dropout_mode == "burst" and (s_reason is not None or e_reason is not None):
        problems = []
        if s_reason is not None:
            problems.append(f"burst_start_probability is not identifiable: {s_reason}")
        if e_reason is not None:
            problems.append(f"burst_end_probability is not identifiable: {e_reason}")
        raise ValueError(f"cannot fit dropout_mode 'burst': {'; '.join(problems)} {counts_text}")

    s_wilson = _wilson(n_ad, n_from_a, z)
    p_wilson = _wilson(n_da, n_from_d, z)
    # Box mapping for e when the bootstrap cannot be used: e grows with both
    # p_DA and s, so the low end uses both low ends and the high end both high
    # ends. That is a little wider than a joint interval would be.
    e_wilson = (float(_end_from(p_wilson[0], s_wilson[0])), float(_end_from(p_wilson[1], s_wilson[1])))

    if s_reason is not None:
        s_record = _record(None, None, None, f"not identifiable: {s_reason} {counts_text}; sensor_config uses the SensorConfig default 0.0", n_from_a)
    elif n_bootstrap == 0 or n_ad in (0, n_from_a):
        why = "no bootstrap" if n_bootstrap == 0 else "count at the 0 or n edge, parametric bootstrap would collapse"
        s_record = _record(s_hat, s_wilson[0], s_wilson[1], f"transition-count MLE, Wilson score interval ({why})", n_from_a)
    else:
        s_record = None
    if e_reason is not None:
        e_record = _record(None, None, None, f"not identifiable: {e_reason} {counts_text}; sensor_config uses the SensorConfig default 1.0", n_from_d)
    elif n_bootstrap == 0 or n_da in (0, n_from_d) or n_ad in (0, n_from_a):
        why = "no bootstrap" if n_bootstrap == 0 else "count at the 0 or n edge, parametric bootstrap would collapse"
        e_record = _record(e_hat, e_wilson[0], e_wilson[1], f"transition-count MLE, Wilson intervals of s and P(D->A) mapped through e = P(D->A)/(1-s) ({why})", n_from_d)
    else:
        e_record = None

    if s_record is None or e_record is None:
        # Parametric bootstrap: simulate the fitted chain many times with the
        # same segment lengths and the same first state in each segment, refit
        # s and e on each copy, and take percentiles. For a Markov chain this is
        # cleaner than cutting the observed sequence into blocks, because block
        # joins would invent transitions that never happened.
        boot_s, boot_e = _simulate_chain(segments, s_hat, p_hat, n_bootstrap, rng_burst)
        method = f"transition-count MLE, parametric bootstrap of the fitted chain, {n_bootstrap} resamples, percentile interval"
        if s_record is None:
            lo, hi = _percentile_ci(boot_s, level)
            s_record = _record(s_hat, lo, hi, method, n_from_a)
        if e_record is None:
            lo, hi = _percentile_ci(boot_e, level)
            e_record = _record(e_hat, lo, hi, method, n_from_d)

    # -- Latency ---------------------------------------------------------------
    # Latency per record = processing_time_s - capture_time_s, over every record
    # of this subject that has processing_time_s. Mean -> measurement_delay_s,
    # standard deviation (ddof=1) -> measurement_jitter_s, the same mean/std
    # pair that the lognormal delay in sim/link.py is built from. Intervals use
    # the same moving-block bootstrap as the residuals rather than resampling
    # single frames: detector latency often runs in streaks (queueing, thermal
    # throttling), and for truly independent latencies the block version costs
    # only a little extra width.
    latencies = [item.processing_time_s - item.capture_time_s for item in records if item.processing_time_s is not None]
    scenario_delay = None
    null_reason = None
    if trace.clock_domain != "shared_monotonic":
        null_reason = f"clock_domain is {trace.clock_domain!r}, so capture and processing times are on different clocks and their difference is not a latency"
    elif len(latencies) < 2:
        null_reason = f"need at least 2 records with processing_time_s, because a standard deviation needs at least 2 records; found {len(latencies)}"
    if null_reason is not None:
        delay_record = _record(None, None, None, f"not fitted: {null_reason}", len(latencies))
        jitter_record = _record(None, None, None, f"not fitted: {null_reason}", len(latencies))
        lat_block = None
    else:
        lat = np.asarray(latencies, dtype=float).reshape(-1, 1)
        n_lat = lat.shape[0]
        lat_mean = float(lat.mean())
        lat_std = float(lat.std(ddof=1))
        lat_block = _block_length(n_lat)
        if n_bootstrap > 0:
            bm, bv = _block_bootstrap_moments(lat, n_bootstrap, lat_block, rng_lat)
            mean_ci = _percentile_ci(bm[:, 0], level)
            std_ci_lat = _percentile_ci(np.sqrt(bv[:, 0]), level)
            lat_method = f"moving-block bootstrap, block length {lat_block}, {n_bootstrap} resamples, percentile interval"
        else:
            half_mean = z * lat_std / math.sqrt(n_lat)
            half_std = z * math.sqrt(_variance_of_sample_variance(lat[:, 0])) / (2.0 * lat_std) if lat_std > 0 else 0.0
            mean_ci = (lat_mean - half_mean, lat_mean + half_mean)
            std_ci_lat = (max(lat_std - half_std, 0.0), lat_std + half_std)
            lat_method = "normal approximation (no bootstrap), fourth-moment standard error for the std; latency is usually right-skewed so this is rough"
        scenario_delay = {"measurement_delay_s": lat_mean, "measurement_jitter_s": lat_std}
        delay_record = _record(lat_mean, mean_ci[0], mean_ci[1], lat_method, n_lat)
        jitter_record = _record(lat_std, std_ci_lat[0], std_ci_lat[1], lat_method, n_lat)

    sensor_config = {
        "noise_std_m": pooled,
        "bias_m": [float(v) for v in mean],
        "random_walk_std_m": 0.0,
        "dropout_probability": float(drop_hat),
        "burst_start_probability": 0.0 if s_reason is not None else float(s_record["estimate"]),
        "burst_end_probability": 1.0 if e_reason is not None else float(e_record["estimate"]),
        "mode": dropout_mode,
    }

    assumptions = [
        "Residual = position_m - ground_truth.position_m, only for detected records that have both, ordered by capture_time_s.",
        "The sim model is truth + constant bias + independent Normal noise per axis; random_walk_std_m is not fitted and is set to 0.0.",
        f"Ground-truth uncertainty is treated as one shared 1-sigma offset for the whole run ({truth_sigma:g} m, the largest per-record value) and is added in quadrature to the bias intervals only. If the truth error is random frame to frame instead, it inflates the fitted spread and is not removed.",
        "noise_std_m pools the three axes as sqrt(mean of the per-axis variances). A real camera is usually much worse along the depth (range) axis than across the image, so one number hides that; check per_axis.anisotropy_ratio before trusting the single value.",
        "noise_std_m uses the plain sample standard deviation, so outlier frames inflate it. per_axis.robust_std_m (1.4826 x MAD) and per_axis.fraction_beyond_4_std are reported for checking: a big gap between std_m and robust_std_m, or a tail fraction far above 0.0001, means a Gaussian is a poor fit for this log.",
        "One log should be one condition (one distance, one lighting, one target). Error usually grows with distance, so noise versus distance needs one fit per log, not one fit of a mixed log.",
        "Bootstrap intervals resample blocks of neighbouring frames to respect short-range correlation; correlation longer than the block length (slow drift, a static target with a fixed error) still makes the intervals too narrow.",
        "dropout_probability = missed / (detected + missed); unknown records are excluded and the Wilson interval assumes independent misses.",
        "Burst fields come from the 2-state Markov chain of sim/sensor.py: P(available -> dropped) = s and P(dropped -> available) = e(1 - s). Unknown records break the chain. The burst MLE is filled in both modes so the config can be switched to mode 'burst'; in independent mode SensorConfig ignores the burst fields.",
        "Frames that never appear in the log (not even as missed or unknown) are invisible; the transition counts assume the log lists every frame of this subject.",
        "Records flagged duplicate or out_of_order are used as they are, not dropped.",
        "Latency = processing_time_s - capture_time_s: detector processing only. Exposure, readout, and transport before the capture stamp are not seen. It is fitted only under clock_domain shared_monotonic.",
    ]
    unidentified = [name for name, reason in (("burst_start_probability", s_reason), ("burst_end_probability", e_reason)) if reason is not None]
    if unidentified:
        assumptions.append(
            f"{' and '.join(unidentified)} {'is' if len(unidentified) == 1 else 'are'} not identifiable from this log {counts_text}; "
            "sensor_config uses the SensorConfig default (burst_start_probability 0.0, burst_end_probability 1.0) and the uncertainty record is null. "
            "Do not switch this config to mode 'burst'."
        )

    uncertainty = {
        "noise_std_m": noise_record,
        **bias_records,
        "dropout_probability": dropout_record,
        "burst_start_probability": s_record,
        "burst_end_probability": e_record,
        "measurement_delay_s": delay_record,
        "measurement_jitter_s": jitter_record,
    }

    result: dict[str, Any] = {
        "schema": FIT_SCHEMA,
        "evidence_boundary": _EVIDENCE_BOUNDARY,
        "input": {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "clock_domain": trace.clock_domain,
            "subject_id": subject_id,
            "counts": counts,
        },
        "sensor_config": sensor_config,
        "scenario_delay": scenario_delay,
        "uncertainty": uncertainty,
        "per_axis": per_axis,
        "transition_counts": {"available_to_available": n_aa, "available_to_dropped": n_ad,
                              "dropped_to_available": n_da, "dropped_to_dropped": n_dd},
        "bootstrap": {"n_bootstrap": n_bootstrap, "seed": int(seed), "confidence_level": level,
                      "residual_block_length": res_block, "latency_block_length": lat_block},
        "assumptions": assumptions,
    }
    if scenario_delay is None:
        result["scenario_delay_null_reason"] = null_reason
    return result


def _simulate_chain(segments: list[list[bool]], s: float, p_da: float | None, n_bootstrap: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the fitted availability chain and refit s and e on every copy."""

    p = 0.0 if p_da is None else p_da
    usable = [seg for seg in segments if len(seg) > 1]
    n_aa = np.zeros(n_bootstrap, dtype=np.int64)
    n_ad = np.zeros(n_bootstrap, dtype=np.int64)
    n_da = np.zeros(n_bootstrap, dtype=np.int64)
    n_dd = np.zeros(n_bootstrap, dtype=np.int64)
    for seg in usable:
        steps = len(seg) - 1
        states = np.empty((steps + 1, n_bootstrap), dtype=bool)  # True = dropped
        states[0] = seg[0]
        u = rng.random((steps, n_bootstrap))
        for t in range(steps):
            # From available: drop if u < s. From dropped: stay dropped if u >= p.
            states[t + 1] = np.where(states[t], u[t] >= p, u[t] < s)
        # Count with two passes instead of four: every switch is either A->D
        # or D->A, and (A->D) - (D->A) = (last state) - (first state).
        from_d = np.count_nonzero(states[:-1], axis=0)
        switches = np.count_nonzero(states[:-1] ^ states[1:], axis=0)
        net = states[-1].astype(np.int64) - states[0].astype(np.int64)
        ad = (switches + net) // 2
        da = switches - ad
        n_ad += ad
        n_da += da
        n_aa += steps - from_d - ad
        n_dd += from_d - da
    from_a = n_aa + n_ad
    from_d = n_da + n_dd
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_s = np.where(from_a > 0, n_ad / np.maximum(from_a, 1), np.nan)
        boot_p = np.where(from_d > 0, n_da / np.maximum(from_d, 1), np.nan)
    boot_e = np.where(np.isfinite(boot_s) & np.isfinite(boot_p), _end_from(np.nan_to_num(boot_p), np.nan_to_num(boot_s)), np.nan)
    return boot_s, boot_e
