"""Independent checks for the S6 recovery statistics, link delivery and truth paths.

Every expected value here comes from outside the code under test: published
textbook numbers, exact binomial arithmetic done with math.comb, the
steady-state solution of a two-state chain worked out by hand, or plain
geometry (a straight line, a circle). None of them call a copy of the
function being checked. Each test says in a comment which fact it uses, a
short derivation, and why its tolerance has the size it has.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.s6_recovery import (
    ArmResult,
    RecoveryConfig,
    _burst_start_probability,
    deliver,
    median_interval,
    paired_summary,
    summarize_arm,
    truth_position,
    wilson_interval,
)

CFG = RecoveryConfig()
Z95 = 1.959964  # the 97.5th percentile of the standard normal, from any z table


def _bin_cdf(k, n):
    """P(X <= k) for X ~ Binomial(n, 1/2), by counting: sum of C(n, i) / 2^n."""
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n


def _binom_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)


def _result(p95, failed=False):
    return ArmResult(p95_error_m=p95, max_error_m=p95, rmse_m=p95, fused_count=1, dropped_count=0,
                     fused_out_of_order_count=0, mean_nis=None, failed=failed,
                     query_times_s=None, errors_m=None)


# ---- Wilson interval -----------------------------------------------------------

def test_wilson_matches_published_values():
    # Independent fact: Newcombe (1998), Statistics in Medicine 17, Table I, lists
    # Wilson score intervals without continuity correction: 81 of 263 gives
    # (0.2553, 0.3662), 15 of 148 gives (0.0624, 0.1605), and 0 of 20 gives
    # (0, 0.1611). The tolerance is half a unit in the last printed digit
    # (the table rounds to 4 places).
    for k, n, lo, hi in ((81, 263, 0.2553, 0.3662), (15, 148, 0.0624, 0.1605), (0, 20, 0.0, 0.1611)):
        got_lo, got_hi = wilson_interval(k, n)
        assert got_lo == pytest.approx(lo, abs=6e-5)
        assert got_hi == pytest.approx(hi, abs=6e-5)


def test_wilson_all_or_nothing_has_closed_form_edge():
    # Independent fact: when no trial succeeds the Wilson score equation has a
    # simple root. With p-hat = 0 the lower end is 0 and the upper end is
    # z^2 / (n + z^2). Check: n = 10 gives 3.8416 / 13.8416 = 0.2775. By
    # symmetry 10 of 10 gives (1 - 0.2775, 1). The 1e-6 tolerance covers the
    # rounding of z to 6 digits; 1e-12 on the edges is float noise (the code
    # returns 5.6e-17 instead of 0 for the lower end at 0 successes).
    for n in (10, 30, 200):
        upper = Z95**2 / (n + Z95**2)
        lo, hi = wilson_interval(0, n)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert hi == pytest.approx(upper, abs=1e-6)
        lo, hi = wilson_interval(n, n)
        assert hi == pytest.approx(1.0, abs=1e-12)
        assert lo == pytest.approx(1.0 - upper, abs=1e-6)


def test_wilson_is_mirror_symmetric_and_brackets_the_estimate():
    # Independent facts: counting failures instead of successes cannot change
    # the answer, so lower(k) = 1 - upper(n - k). The interval must contain
    # the observed proportion k / n, and a higher confidence level must give
    # a wider interval. Rounding tolerance 1e-12 (only float arithmetic).
    for n in (7, 25, 60):
        for k in range(n + 1):
            lo, hi = wilson_interval(k, n)
            mirror_lo, mirror_hi = wilson_interval(n - k, n)
            assert lo == pytest.approx(1.0 - mirror_hi, abs=1e-12)
            assert hi == pytest.approx(1.0 - mirror_lo, abs=1e-12)
            assert lo - 1e-12 <= k / n <= hi + 1e-12
            wide_lo, wide_hi = wilson_interval(k, n, confidence=0.99)
            assert wide_lo <= lo + 1e-12 and hi <= wide_hi + 1e-12


def test_wilson_average_coverage_is_near_95_percent():
    # Independent fact: the true chance that the interval catches the real p
    # is sum over k of Binomial(n, p) probability(k) when p lies inside the
    # interval built from k. For a 95 percent interval this should average
    # about 0.95 over many p values (Brown, Cai and DasGupta 2001 show the
    # Wilson interval hovers around nominal, a little under at single p).
    # Here n = 100 and p runs 0.05 to 0.95 in steps of 0.005. The band
    # 0.94 to 0.96 is wide enough for the known wobble and tight enough that
    # a 90 percent interval (z = 1.645) averages about 0.90 and fails.
    n = 100
    intervals = [wilson_interval(k, n) for k in range(n + 1)]
    ps = np.arange(0.05, 0.9501, 0.005)
    coverage = []
    for p in ps:
        coverage.append(sum(_binom_pmf(k, n, float(p)) for k, (lo, hi) in enumerate(intervals) if lo <= p <= hi))
    assert 0.94 <= float(np.mean(coverage)) <= 0.96


# ---- median interval -----------------------------------------------------------

def test_median_interval_index_is_largest_with_at_least_nominal_coverage():
    # Independent fact: for n continuous values, the true median sits between
    # the k-th smallest and the k-th largest with probability
    # 1 - 2 * P(Bin(n, 1/2) <= k - 1), because each value lands below the
    # median with chance 1/2. Feeding the values 1..n makes the k-th smallest
    # equal to k, so the returned low end IS the index the code picked.
    # Checks, using my own exact binomial sums: (a) that index covers at
    # least 95 percent, (b) one index higher (a narrower interval) would not,
    # and (c) the interval is mirror symmetric, high = n + 1 - low. All exact
    # integer arithmetic, so the tolerance is zero.
    for n in range(8, 61):
        median, low, high = median_interval(list(range(1, n + 1)))
        k = int(low)
        assert 1 <= k <= n // 2
        assert high == n + 1 - k
        assert 1.0 - 2.0 * _bin_cdf(k - 1, n) >= 0.95
        if k + 1 <= n // 2:
            assert 1.0 - 2.0 * _bin_cdf(k, n) < 0.95
        assert median == pytest.approx((n + 1) / 2.0)


def test_median_interval_falls_back_to_full_range_when_n_is_too_small():
    # Independent fact: for n = 5 even the widest order-statistic interval
    # (smallest to largest) covers the median with chance 1 - 2 / 2^5 =
    # 0.9375, below 95 percent, so no honest 95 percent interval narrower
    # than "everything" exists. n = 6 reaches 1 - 2/64 = 0.969 using the
    # extremes, so the answer there is also (min, max). Exact arithmetic.
    assert 1.0 - 2.0 * _bin_cdf(0, 5) < 0.95
    assert median_interval([3.0, 9.0, 1.0, 7.0, 5.0]) == (5.0, 1.0, 9.0)
    assert 1.0 - 2.0 * _bin_cdf(0, 6) >= 0.95
    assert median_interval([6, 1, 5, 2, 4, 3])[1:] == (1.0, 6.0)


def test_median_interval_covers_true_median_in_simulation():
    # Independent fact: draws from a normal(10, 2) have true median exactly 10.
    # For n = 25 my exact binomial sum gives coverage 1 - 2 * P(Bin <= 7) for
    # the code's index (computed in the test from the returned index, not
    # from the code's own formula). With 3000 repeats the standard error of
    # an observed coverage near 0.957 is sqrt(0.957 * 0.043 / 3000) = 0.0037,
    # so +/- 5 standard errors (0.019) is a safe but real band. The expected
    # coverage is computed from the index the code returns, so this test checks
    # that the rank-based interval really covers the true median at the rate
    # the binomial sum predicts, whatever the data; the choice of index itself
    # is checked exactly by the index test above.
    rng = np.random.default_rng(20261027)
    n, repeats = 25, 3000
    probe = median_interval(list(range(1, n + 1)))
    k = int(probe[1])
    expected = 1.0 - 2.0 * _bin_cdf(k - 1, n)
    hits = 0
    for _ in range(repeats):
        _, lo, hi = median_interval(rng.normal(10.0, 2.0, size=n))
        hits += lo <= 10.0 <= hi
    se = math.sqrt(expected * (1.0 - expected) / repeats)
    assert abs(hits / repeats - expected) < 5.0 * se


# ---- burst loss probability ----------------------------------------------------

def test_burst_start_probability_gives_the_target_stationary_loss():
    # Independent fact: the link's burst rule is a two-step chain. A packet is
    # lost for sure while a burst is running, and each lost packet ends the
    # burst with chance e. Call b the steady lost fraction. A burst is still
    # running at the next packet only if the last packet was lost and the
    # burst did not end: probability b * (1 - e). Otherwise a new burst
    # starts with chance s. So b = b(1 - e) + (1 - b(1 - e)) s. Solving
    # for b: b = s / (e + s (1 - e)). The function must pick s so this b
    # equals the requested loss p. Tolerance 1e-12 is float rounding only.
    for p in (0.01, 0.1, 0.3, 0.5, 0.9):
        for e in (0.05, 0.2, 0.5, 1.0):
            s = _burst_start_probability(p, e)
            assert 0.0 <= s <= 1.0
            steady = s / (e + s * (1.0 - e))
            assert steady == pytest.approx(p, abs=1e-12)
    # Limiting facts: no loss needs no bursts, and certain loss starts one at once.
    assert _burst_start_probability(0.0, 0.2) == 0.0
    assert _burst_start_probability(1.0, 0.2) == 1.0
    # With bursts lasting one packet (e = 1) the model is plain independent loss, so s = p.
    assert _burst_start_probability(0.37, 1.0) == pytest.approx(0.37, abs=1e-12)


def test_deliver_burst_loss_rate_and_burst_length_match_theory():
    # Independent facts: (1) the steady lost fraction equals the requested p
    # (derived in the previous test). (2) A burst of consecutive lost packets
    # ends after each lost packet with chance e, so its length is geometric
    # with mean 1/e (here 4 packets). (3) A run of lost packets we can see is
    # not always one burst: right after a burst ends, the next packet starts a
    # new burst with chance s, which glues the two together. So the number of
    # bursts in a visible run is geometric with mean 1 / (1 - s), and the mean
    # visible run is (1/e) / (1 - s). With b = p the steady loss, s solves
    # p = s / (e + s (1 - e)), so s = p e / (1 - p (1 - e)) = 0.0625 / 0.8125
    # = 0.0769, and the mean visible run is 4 / 0.9231 = 4.333 packets.
    # Many short runs are pooled; the error bar for (1) is taken from the
    # spread of per-run loss fractions (standard error of their mean), so it
    # already includes the correlation between neighbouring packets.
    # Tolerance: 5 standard errors for both. Over 30 seeds there are about
    # 3400 visible runs with std near 3.9 packets, so the run-length standard
    # error is about 0.066 and a burst length wrong by 9 percent (visible mean
    # near 3.95) falls about 6 standard errors outside the band.
    p, mean_len = 0.25, 4.0
    cfg = RecoveryConfig(duration_s=200.0)
    n_sent = round(cfg.duration_s / cfg.observation_period_s) + 1
    fractions, run_lengths = [], []
    for seed in range(1, 31):
        out = deliver("constant_velocity", delay_s=0.0, loss_probability=p, noise_std_m=0.0, seed=seed,
                      config=cfg, loss_model="burst", burst_mean_length_packets=mean_len)
        got = {d.seq for d in out}
        fractions.append(1.0 - len(got) / n_sent)
        run = 0
        for seq in range(n_sent):
            if seq in got:
                if run:
                    run_lengths.append(run)
                run = 0
            else:
                run += 1
        if run:
            run_lengths.append(run)
    fractions = np.asarray(fractions)
    se = fractions.std(ddof=1) / math.sqrt(fractions.size)
    assert abs(fractions.mean() - p) < 5.0 * se
    # Bursts cut short by the end of a seed's run (at most 30 of them) shift
    # the mean by well under one standard error.
    s = p * (1.0 / mean_len) / (1.0 - p * (1.0 - 1.0 / mean_len))
    expected_run = mean_len / (1.0 - s)
    assert expected_run == pytest.approx(4.3333, abs=1e-4)
    run_lengths = np.asarray(run_lengths, dtype=float)
    run_se = run_lengths.std(ddof=1) / math.sqrt(run_lengths.size)
    assert abs(run_lengths.mean() - expected_run) < 5.0 * run_se


# ---- deliver: loss, delay, noise -----------------------------------------------

def test_deliver_independent_loss_rate_and_exact_delay():
    # Independent facts: with loss probability p each packet is dropped
    # independently, so the number kept out of N sent is Binomial(N, 1-p) with
    # standard deviation sqrt(N p (1 - p)). With N = 32001 and p = 0.2 that is
    # 72 packets; the test allows 3.5 standard deviations (the run is seeded,
    # so this is not a flaky bound, and a loss rate wrong by 0.01 moves the
    # count by 320, which is 4.5 standard deviations). Every delivered
    # packet must arrive exactly delay_s after capture (a link with no jitter
    # adds one fixed number). The 1e-9 s tolerance is float rounding on
    # times near 3200 s. Packets that would land after the run ends are not
    # delivered; here that is only the very last capture (t = 3200 s plus
    # 0.037 s), one packet out of 32001, far inside the bound.
    p, delay = 0.2, 0.037
    cfg = RecoveryConfig(duration_s=3200.0)
    n_sent = 32001
    out = deliver("constant_velocity", delay_s=delay, loss_probability=p, noise_std_m=0.01, seed=11, config=cfg)
    sd = math.sqrt(n_sent * p * (1.0 - p))
    assert abs(len(out) - n_sent * (1.0 - p)) < 3.5 * sd
    for d in out:
        assert d.receive_time_s - d.capture_time_s == pytest.approx(delay, abs=1e-9)
    # Each capture time is its sequence number times the 0.1 s period.
    assert all(d.capture_time_s == pytest.approx(0.1 * d.seq, abs=1e-9) for d in out)


def test_deliver_drops_packets_that_would_arrive_after_the_run():
    # Independent fact: captures happen at t = 0, 0.1, ..., 12.0. With no loss
    # and one-way delay d, a packet is delivered only if t + d <= 12. For
    # d = 0.25 that means t <= 11.75, so k = 0..117, which is 118 packets.
    # (Counted by hand: 11.75 / 0.1 = 117.5, floor is 117, plus the t = 0
    # packet.) Exact integers, zero tolerance. Receive times must never pass 12.
    out = deliver("constant_velocity", delay_s=0.25, loss_probability=0.0, noise_std_m=0.0, seed=3, config=CFG)
    assert len(out) == 118
    assert max(d.receive_time_s for d in out) <= CFG.duration_s + 1e-9
    assert sorted(d.seq for d in out) == list(range(118))


def test_deliver_measurement_noise_has_requested_spread_and_no_bias():
    # Independent fact: with sigma set, measurement minus truth is
    # Normal(0, sigma^2) in each of x, y and z. The line truth is x = v t, y = z = 0
    # (typed here from the definition of constant velocity along +x, not
    # taken from the code). For M independent values the sample variance
    # divided by sigma^2 has mean 1 and variance 2 / (M - 1), and the sample
    # mean has standard deviation sigma / sqrt(M). Here M = 3 * 6001 = 18003, so
    # the allowed band is +/- 5 * sqrt(2 / M) = +/- 5.3 percent on the
    # variance ratio, and 5 standard errors on the mean.
    sigma = 0.08
    cfg = RecoveryConfig(duration_s=600.0, target_speed_mps=1.0)
    out = deliver("constant_velocity", delay_s=0.0, loss_probability=0.0, noise_std_m=sigma, seed=5, config=cfg)
    assert len(out) == 6001
    residual = np.array([np.asarray(d.measurement_m) - np.array([1.0 * d.capture_time_s, 0.0, 0.0]) for d in out])
    m = residual.size
    assert residual.var(ddof=1) / sigma**2 == pytest.approx(1.0, abs=5.0 * math.sqrt(2.0 / m))
    assert abs(residual.mean()) < 5.0 * sigma / math.sqrt(m)
    # The three axes are separate, independent draws, so each should show the same spread.
    for axis in range(3):
        assert residual[:, axis].std(ddof=1) / sigma == pytest.approx(1.0, abs=0.06)


def test_deliver_with_no_loss_delay_or_noise_returns_the_truth_exactly():
    # Limiting case: sigma = 0, delay = 0, loss = 0 means the sensor reports
    # exactly where the target is at the moment of capture, and every packet
    # arrives at that moment. The expected position is the line x = v t and
    # the circle (r sin(w t), r (1 - cos(w t))) with w = v / r, typed from the
    # definition of circular motion. Tolerance 1e-12 m is float rounding.
    v, r = CFG.target_speed_mps, CFG.turn_radius_m
    for motion in ("constant_velocity", "constant_turn"):
        out = deliver(motion, delay_s=0.0, loss_probability=0.0, noise_std_m=0.0, seed=9, config=CFG)
        assert len(out) == 121
        for d in out:
            t = d.capture_time_s
            if motion == "constant_velocity":
                want = (v * t, 0.0, 0.0)
            else:
                want = (r * math.sin(v / r * t), r * (1.0 - math.cos(v / r * t)), 0.0)
            assert d.receive_time_s == pytest.approx(t, abs=1e-12)
            assert d.measurement_m == pytest.approx(want, abs=1e-12)


# ---- truth_position ------------------------------------------------------------

def test_truth_turn_is_a_circle_traversed_at_constant_speed():
    # Independent geometry: a circle of radius R through the origin heading +x
    # and turning left has its centre at (0, R, 0). So every point must sit
    # distance R from that centre, speed must be v everywhere (check by the
    # distance between points a tiny step apart: 1e-6 s step, error of a chord
    # versus an arc is about (v dt)^3 / (24 R^2), negligible), and after one
    # full period 2 pi R / v the target is back at the origin. After half a
    # period it is at the far side, (0, 2R). Non-default R = 2.5 m and v = 0.7
    # m/s so that a hard-coded default cannot pass by accident. Tolerances: 1e-9 m
    # for positions (float trig), 1e-6 relative for speed.
    r, v = 2.5, 0.7
    cfg = RecoveryConfig(turn_radius_m=r, target_speed_mps=v)
    period = 2.0 * math.pi * r / v
    t = np.linspace(0.0, 2.0 * period, 997)
    p = truth_position("constant_turn", t, cfg)
    centre = np.array([0.0, r, 0.0])
    assert np.allclose(np.linalg.norm(p - centre, axis=1), r, atol=1e-9)
    assert np.all(p[:, 2] == 0.0)
    dt = 1e-6
    p2 = truth_position("constant_turn", t + dt, cfg)
    speed = np.linalg.norm(p2 - p, axis=1) / dt
    assert np.allclose(speed, v, rtol=1e-6)
    ends = truth_position("constant_turn", [0.0, period / 2.0, period], cfg)
    assert ends[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert ends[1] == pytest.approx((0.0, 2.0 * r, 0.0), abs=1e-9)
    assert ends[2] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # It starts out heading along +x: a short time later x is positive and y is tiny.
    start = truth_position("constant_turn", [1e-3], cfg)[0]
    assert start[0] == pytest.approx(v * 1e-3, rel=1e-4)
    assert 0.0 < start[1] < 1e-5


def test_truth_line_and_turn_path_length_match_geometry():
    # Independent geometry: the line x = v t gives distance v t after t seconds
    # in a straight line, so the point at t is v t from the origin and on the
    # x axis. For the turn, the total length of a very fine polyline over one
    # period must equal the circle's circumference 2 pi R. With 200000 chords
    # the polyline falls short of the circle by a relative (2 pi / N)^2 / 24,
    # about 4e-11, so rel=1e-8 is generous. A sloppy angular rate would change
    # the length of one period's worth of motion and fail.
    cfg = RecoveryConfig(turn_radius_m=1.7, target_speed_mps=0.9)
    t = np.array([0.0, 1.0, 2.5, 10.0])
    line = truth_position("constant_velocity", t, cfg)
    assert np.allclose(line[:, 0], 0.9 * t, atol=1e-12)
    assert np.all(line[:, 1:] == 0.0)
    period = 2.0 * math.pi * 1.7 / 0.9
    path = truth_position("constant_turn", np.linspace(0.0, period, 200001), cfg)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    assert length == pytest.approx(2.0 * math.pi * 1.7, rel=1e-8)
    with pytest.raises(ValueError):
        truth_position("figure_eight", t, cfg)


# ---- summaries -----------------------------------------------------------------

def test_summarize_arm_failure_interval_matches_published_zero_of_twenty():
    # Independent fact: Newcombe (1998) lists the Wilson interval for 0 of 20
    # as (0, 0.1611); also 81 of 263 as (0.2553, 0.3662). Build 20 results with
    # no failures, then 263 results of which 81 fail, and read the failure rate
    # interval out of the summary. The label rule is stated in the docstring:
    # "fails" when at least half the seeds fail, which 81/263 (31 percent) is not.
    # Tolerance 6e-5 matches the 4 printed digits.
    s = summarize_arm([_result(0.1 + 0.001 * i) for i in range(20)])
    assert s["fail_count"] == 0 and s["fail_rate"] == 0.0
    # The lower end is max(0, centre - half), which can come out as a tiny
    # float like 1e-17 instead of 0 depending on rounding, so allow 1e-12.
    assert s["fail_rate_ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert s["fail_rate_ci_high"] == pytest.approx(0.1611, abs=6e-5)
    assert s["label"] == "passes"
    rows = [_result(0.1 + 0.001 * i, failed=(i < 81)) for i in range(263)]
    s = summarize_arm(rows)
    assert s["fail_rate"] == pytest.approx(81 / 263)
    assert s["fail_rate_ci_low"] == pytest.approx(0.2553, abs=6e-5)
    assert s["fail_rate_ci_high"] == pytest.approx(0.3662, abs=6e-5)
    assert s["label"] == "passes"


def test_paired_summary_reports_naive_minus_recovery_with_hand_counts():
    # Independent fact (done by hand): twenty seeds where naive's p95 is
    # recovery's plus 0.5 m times each integer k in -9..10, so the differences
    # naive - recovery are 0.5 k. Counted by hand: 10 are positive (k = 1..10,
    # recovery better), 9 are negative (k = -9..-1, recovery worse), 1 is zero
    # (a tie). The median of k = -9..10 is the average of the 10th and 11th
    # values, (0 + 1) / 2 = 0.5, so the median difference is 0.25 m. The 95
    # percent order-statistic interval for n = 20 uses index 6 (exact coverage
    # 1 - 2 * 0.0207 = 0.959; index 7 gives 1 - 2 * 0.0577 = 0.885), so it spans the 6th
    # smallest (k = -4, i.e. -2.0 m) to the 6th largest (k = 5, i.e. 2.5 m).
    # Multiples of 0.5 are exact in floating point, so the tolerance is zero.
    offsets = list(range(-9, 11))
    recovery = [_result(10.0) for _ in offsets]
    naive = [_result(10.0 + 0.5 * k) for k in offsets]
    out = paired_summary(naive, recovery)
    assert out["n_seeds"] == 20
    assert out["recovery_better_count"] == 10
    assert out["recovery_worse_count"] == 9
    assert out["tie_count"] == 1
    assert out["p95_diff_median_m"] == 0.25
    assert out["p95_diff_ci_low_m"] == -2.0
    assert out["p95_diff_ci_high_m"] == 2.5
    with pytest.raises(ValueError):
        paired_summary(naive, recovery[:-1])
