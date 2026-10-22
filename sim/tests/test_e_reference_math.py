"""Checks of the Run E hand-check formulas against hand-worked numbers,
limiting cases and small independent simulations.

Everything here is SIMULATED.  Radio values come from sim/radio/params.py and
are literature-derived, not measured.  The burst loss chains are synthetic.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest


from sim.tests.reference_e import hand_checks as hc  # noqa: E402

SEED_BASE = 20261029


def _rng(index):
    return np.random.default_rng(np.random.SeedSequence([SEED_BASE, index]))


# ---------------------------------------------------------------------------
# 1. Tail follow
# ---------------------------------------------------------------------------
def test_tail_follow_hand_value():
    # gap 4, radius 1: close 3 m at 1.5 - 0.5 = 1 m/s -> 3 s
    assert hc.tail_follow_time_s(4.0, 1.0, 1.5, 0.5) == pytest.approx(3.0, abs=1e-12)
    # gap 10, radius 1, 1.5 vs 0.0: 9 / 1.5 = 6 s
    assert hc.tail_follow_time_s(10.0, 1.0, 1.5, 0.0) == pytest.approx(6.0, abs=1e-12)


def test_tail_follow_limits():
    assert hc.tail_follow_time_s(4.0, 1.0, 0.5, 0.5) == math.inf
    assert hc.tail_follow_time_s(4.0, 1.0, 0.4, 0.5) == math.inf
    assert hc.tail_follow_time_s(0.8, 1.0, 1.5, 0.5) == 0.0
    assert hc.tail_follow_time_s(1.0, 1.0, 1.5, 0.5) == 0.0


def test_tail_follow_time_stepping():
    # step both positions forward and stop when the gap reaches the radius
    dt = 1e-4
    xf, xt, t = 0.0, 4.0, 0.0
    while xt - xf > 1.0 + 1e-12:
        xf += 1.5 * dt
        xt += 0.5 * dt
        t += dt
    assert t == pytest.approx(hc.tail_follow_time_s(4.0, 1.0, 1.5, 0.5), abs=2 * dt)


# ---------------------------------------------------------------------------
# 2. Heartbeat false alarms
# ---------------------------------------------------------------------------
def test_misses_needed_convention():
    assert hc.misses_needed(10, 0.3) == 3    # gap 0.3 s exactly does not alarm, 0.4 s does
    assert hc.misses_needed(10, 0.25) == 2   # gap 0.3 s > 0.25 s alarms after 2 misses
    assert hc.misses_needed(10, 0.7) == 7    # 0.7 * 10 is 7.000000000000001 in floating point
    assert hc.misses_needed(20, 0.3) == 6
    assert hc.misses_needed(10, 0.05) == 0


def test_iid_matches_run_c_value():
    # 36000 * 0.9 * 0.1**3 = 32.4 (Run C published 32.4)
    assert hc.heartbeat_false_alarms_per_hour_iid(0.1, 10, 0.3) == pytest.approx(32.4, rel=1e-12)


@pytest.mark.parametrize("p,h,expected", [
    (0.01, 10, 0.0356), (0.05, 10, 4.275), (0.20, 10, 230.4), (0.30, 10, 680.4),
    (0.10, 20, 0.0648), (0.20, 20, 3.686), (0.30, 20, 36.74),
], ids=["p01h10", "p05h10", "p20h10", "p30h10", "p10h20", "p20h20", "p30h20"])
def test_iid_run_c_table(p, h, expected):
    # Run C closed-form column at T = 0.3 s, worked by hand to 3 or 4 figures
    assert hc.heartbeat_false_alarms_per_hour_iid(p, h, 0.3) == pytest.approx(expected, rel=2e-3)


def test_iid_limits():
    assert hc.heartbeat_false_alarms_per_hour_iid(0.0, 10, 0.3) == 0.0
    assert hc.heartbeat_false_alarms_per_hour_iid(1.0, 10, 0.3) == 0.0  # never a heartbeat, so no fresh alarm
    # T shorter than one period: every delivered heartbeat is followed by an alarm
    assert hc.heartbeat_false_alarms_per_hour_iid(0.0, 10, 0.05) == pytest.approx(36000.0)


@pytest.mark.parametrize("T", [0.3, 0.25], ids=["T030", "T025"])
def test_iid_brute_force_gaps(T):
    # Independent route: draw losses, take actual gaps between arrivals in
    # seconds and count gaps strictly longer than T.  No use of floor(T h).
    h, p, hours = 10, 0.1, 200
    n = int(3600 * h * hours)
    lost = _rng(1).random(n) < p
    arrivals = np.flatnonzero(~lost)
    gaps_s = np.diff(arrivals) / h
    count = int(np.sum(gaps_s > T + 1e-9))
    expected = hc.heartbeat_false_alarms_per_hour_iid(p, h, T) * hours
    assert abs(count - expected) < 5 * math.sqrt(expected)


def test_markov_run_c_convention():
    q, r = hc.per_slot_chain(0.1, 0.5, 10)
    assert r == pytest.approx(0.2)
    assert q == pytest.approx(0.1 * 0.2 / 0.9)
    # 36000 * 0.1 * 0.2 * 0.8**2 = 460.8
    assert hc.false_alarms_per_hour_markov(q, r, 10, 0.3) == pytest.approx(460.8, rel=1e-9)


@pytest.mark.parametrize("p,D,h,T,expected", [
    (0.01, 0.5, 10, 0.3, 46.08), (0.10, 0.2, 10, 0.3, 450.0), (0.30, 0.5, 10, 0.3, 1382.4),
    (0.01, 0.5, 20, 0.3, 42.51), (0.10, 0.2, 20, 0.3, 427.1), (0.10, 0.5, 20, 0.3, 425.2),
    (0.30, 0.5, 20, 0.3, 1275.5), (0.10, 0.5, 10, 2.0, 10.38), (0.10, 0.5, 20, 2.0, 11.82),
], ids=["p01D5h10", "p10D2h10", "p30D5h10", "p01D5h20", "p10D2h20", "p10D5h20",
        "p30D5h20", "p10D5h10T2", "p10D5h20T2"])
def test_markov_run_c_table(p, D, h, T, expected):
    q, r = hc.per_slot_chain(p, D, h)
    assert hc.false_alarms_per_hour_markov(q, r, h, T) == pytest.approx(expected, rel=2e-3)


def test_markov_limits():
    assert hc.false_alarms_per_hour_markov(0.0, 0.2, 10, 0.3) == 0.0
    # r = 1 - q makes the chain memoryless with loss p = q: it must equal the iid formula
    q = 0.1
    assert hc.false_alarms_per_hour_markov(q, 1.0 - q, 10, 0.3) == pytest.approx(
        hc.heartbeat_false_alarms_per_hour_iid(q, 10, 0.3), rel=1e-12)
    with pytest.raises(ValueError):
        hc.false_alarms_per_hour_markov(0.1, 0.2, 10, 0.05)


def test_coarse_chain_small_cases():
    # k = 1 changes nothing
    assert hc.coarse_grid_chain(0.01, 0.2, 1) == pytest.approx((0.01, 0.2))
    # k = 2 by hand: P(g->b in 2) = (1-q) q + q (1-r)
    q, r = 0.01, 0.2
    q2, r2 = hc.coarse_grid_chain(q, r, 2)
    assert q2 == pytest.approx((1 - q) * q + q * (1 - r))
    assert r2 == pytest.approx(r * (1 - q) + (1 - r) * r)
    # large k forgets the start: both rows go to the stationary loss q / (q + r)
    qk, rk = hc.coarse_grid_chain(q, r, 5000)
    assert qk == pytest.approx(q / (q + r), rel=1e-9)
    assert 1 - rk == pytest.approx(q / (q + r), rel=1e-9)
    # the long-run loss of the coarse chain equals the fine chain's
    qc, rc = hc.coarse_grid_chain(q, r, 10)
    assert qc / (qc + rc) == pytest.approx(q / (q + r), rel=1e-12)


def _fine_chain_sampled_runs(q0, r0, k, hours, rng):
    """Brute-force fine chain (0.01 s steps), sampled every k steps.

    Builds the chain from alternating good and bad run lengths (geometric,
    which is exactly what a two-state chain produces), then works out which
    sampled slots fall inside bad runs.  Returns, for every run of
    consecutive bad samples: its length in samples, and how many separate
    fine bursts it merged.
    """
    n_steps = int(round(hours * 3600 * 100))
    mean_cycle = 1.0 / q0 + 1.0 / r0
    n_cycles = int(n_steps / mean_cycle * 1.1) + 1000
    good = rng.geometric(q0, n_cycles).astype(np.int64)
    bad = rng.geometric(r0, n_cycles).astype(np.int64)
    cycle_start = np.concatenate(([0], np.cumsum(good + bad)[:-1]))
    assert cycle_start[-1] > n_steps, "not enough cycles drawn"
    bad_start = cycle_start + good
    bad_end = bad_start + bad                     # exclusive
    keep = bad_end <= n_steps
    bad_start, bad_end = bad_start[keep], bad_end[keep]
    first = -(-bad_start // k)                    # ceil: first sampled slot inside the burst
    last = (bad_end - 1) // k                     # floor: last sampled slot inside the burst
    seen = first <= last
    first, last = first[seen], last[seen]
    # neighbouring bursts merge when no good sample lies between them
    new_group = np.ones(first.size, dtype=bool)
    new_group[1:] = first[1:] > last[:-1] + 1
    group_id = np.cumsum(new_group) - 1
    n_groups = int(group_id[-1]) + 1
    g_first = first[new_group]
    g_last = np.zeros(n_groups, dtype=np.int64)
    np.maximum.at(g_last, group_id, last)
    bursts_per_group = np.bincount(group_id, minlength=n_groups)
    return g_last - g_first + 1, bursts_per_group


@pytest.mark.parametrize("p,D,rate,idx", [(0.1, 0.5, 10, 2), (0.1, 0.2, 20, 3)],
                         ids=["p10D5h10", "p10D2h20"])
def test_coarse_chain_against_brute_force(p, D, rate, idx):
    T, hours = 0.3, 2000
    k = int(round(100 / rate))
    r0 = 1.0 / (D * 100)                          # fine grid: mean burst D seconds
    q0 = p * r0 / (1 - p)
    m = hc.misses_needed(rate, T)
    lengths, bursts = _fine_chain_sampled_runs(q0, r0, k, hours, _rng(idx))
    alarm_runs = lengths >= m
    count = int(np.sum(alarm_runs))
    qc, rc = hc.coarse_grid_chain(q0, r0, k)
    expected = hc.false_alarms_per_hour_markov(qc, rc, rate, T) * hours
    # the k-step chain is exact for heartbeat loss: only sampling noise remains
    assert abs(count - expected) < 5 * math.sqrt(expected)
    # burst identity: share of alarm runs that merged two or more fine bursts
    merged_share = float(np.mean(bursts[alarm_runs] > 1))
    # chance a good spell is shorter than k steps, so it can hide between samples
    short_gap_chance = 1 - (1 - q0) ** (k - 1)
    assert 0.0 < merged_share < 3 * short_gap_chance
    # counting alarms per fine burst instead (no merging) undercounts by at most that share
    assert merged_share < 0.05


def test_coarse_chain_differs_from_per_slot_convention():
    # Run C put the chain directly on the heartbeat grid with r = 1 / (D h).
    # A 0.01 s chain with the same mean burst, seen at 10 Hz, is a different
    # chain: the exit chance per heartbeat is 1 - (1 - r0)**10 plus re-entry
    # terms, not 0.2.  Hand values for p 0.1, D 0.5 s.
    r0 = 0.02
    q0 = 0.1 * r0 / 0.9
    qc, rc = hc.coarse_grid_chain(q0, r0, 10)
    assert 1 - (1 - r0) ** 10 == pytest.approx(0.1829, abs=1e-4)
    assert 0.18 < rc < 0.2
    run_c = hc.false_alarms_per_hour_markov(*hc.per_slot_chain(0.1, 0.5, 10), 10, 0.3)
    fine = hc.false_alarms_per_hour_markov(qc, rc, 10, 0.3)
    # the two agree to within about 10 percent here
    assert abs(fine - run_c) / run_c < 0.12


# ---------------------------------------------------------------------------
# 3. Detection delay
# ---------------------------------------------------------------------------
def test_detection_delay_hand_values():
    lo, hi, mean = hc.detection_delay_bounds(10, 0.3)
    assert (lo, hi, mean) == pytest.approx((0.2, 0.3, 0.25))
    assert hc.detection_delay_bounds(20, 0.3) == pytest.approx((0.25, 0.3, 0.275))
    with pytest.raises(ValueError):
        hc.detection_delay_bounds(10, 0.05)


def test_detection_delay_sampled():
    # failures at random times; newest heartbeat sent at or before the failure
    h, T = 10, 0.3
    t_fail = _rng(4).uniform(0.0, 1000.0, 200_000)
    t_last = np.floor(t_fail * h + 1e-12) / h
    delay = t_last + T - t_fail
    lo, hi, mean = hc.detection_delay_bounds(h, T)
    assert delay.min() > lo - 1e-9
    assert delay.max() <= hi + 1e-9
    assert delay.mean() == pytest.approx(mean, abs=1e-3)


def test_detection_mean_under_loss_explains_run_c():
    # Run C's tables quote mean delays of 0.439 s (10 Hz, T 0.5), 0.270 s
    # (20 Hz, T 0.3) and 0.188 s (50 Hz, T 0.2) at 10 % independent loss,
    # below T - 1/(2h).  Heartbeats lost just before the failure make the
    # newest arrival older: expected p / (1 - p) extra periods.
    p = 0.1
    for h, T, quoted in [(10, 0.5, 0.439), (20, 0.3, 0.270), (50, 0.2, 0.188)]:
        mean = hc.detection_delay_bounds(h, T)[2] - p / (1 - p) / h
        assert mean == pytest.approx(quoted, abs=6e-4)


# ---------------------------------------------------------------------------
# 4. Stopping buffer
# ---------------------------------------------------------------------------
def test_stopping_buffer_hand_value():
    # 1.5 * (0.3 + 0.1) + 1.5**2 / (2 * 2) = 0.6 + 0.5625
    assert hc.stopping_buffer_m(1.5, 0.3, 0.1, 2.0) == pytest.approx(1.1625)
    assert hc.stopping_buffer_m(0.0, 0.3, 0.1, 2.0) == 0.0
    assert hc.stopping_buffer_m(1.0, 0.0, 0.0, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        hc.stopping_buffer_m(1.0, 0.1, 0.1, 0.0)


def test_stopping_buffer_time_stepping():
    v, age, react, a, dt = 1.5, 0.3, 0.1, 2.0, 1e-5
    x, t = 0.0, 0.0
    while v > 0:
        if t >= age + react:
            v = max(v - a * dt, 0.0)
        x += v * dt
        t += dt
    assert x == pytest.approx(hc.stopping_buffer_m(1.5, age, react, a), abs=1e-3)


# ---------------------------------------------------------------------------
# 5. Coverage
# ---------------------------------------------------------------------------
def test_coverage_hand_values():
    # 16 x 10 m arena, 2 drones, 2 m swath, 1 m/s: 160 / 4 = 40 s
    assert hc.coverage_time_s(160.0, 2, 2.0, 1.0) == pytest.approx(40.0)
    # doubling drones halves the time
    assert hc.coverage_time_s(160.0, 4, 2.0, 1.0) == pytest.approx(20.0)
    assert hc.coverage_time_s(160.0, 0, 2.0, 1.0) == math.inf


# ---------------------------------------------------------------------------
# 6. Path loss and relay
# ---------------------------------------------------------------------------
# literature-derived values from sim/radio/params.py (values only)
TX_DBM, SENS_DBM, PL0_DB = 19.5, -98.0, 40.2
BREAKPOINT_M = 4 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)  # two-ray breakpoint, about 73.2 m


def test_log_distance_hand_values():
    assert hc.log_distance_path_loss_db(1.0, PL0_DB, 2.0) == pytest.approx(40.2)
    assert hc.log_distance_path_loss_db(10.0, PL0_DB, 2.0) == pytest.approx(60.2)
    assert hc.log_distance_path_loss_db(100.0, PL0_DB, 3.0) == pytest.approx(100.2)
    assert hc.log_distance_path_loss_db(20.0, PL0_DB, 2.0, d0=2.0) == pytest.approx(60.2)


def test_dual_slope_hand_values():
    assert BREAKPOINT_M == pytest.approx(73.16, abs=0.01)
    assert hc.dual_slope_path_loss_db(50.0, PL0_DB, 2.0, BREAKPOINT_M, 4.0) == pytest.approx(
        hc.log_distance_path_loss_db(50.0, PL0_DB, 2.0))
    # 40.2 + 20 log10(73.16) + 40 log10(200 / 73.16) = 40.2 + 37.285 + 17.470
    assert hc.dual_slope_path_loss_db(200.0, PL0_DB, 2.0, BREAKPOINT_M, 4.0) == pytest.approx(94.955, abs=0.01)
    # continuous at the breakpoint
    below = hc.dual_slope_path_loss_db(BREAKPOINT_M * (1 - 1e-9), PL0_DB, 2.0, BREAKPOINT_M, 4.0)
    above = hc.dual_slope_path_loss_db(BREAKPOINT_M * (1 + 1e-9), PL0_DB, 2.0, BREAKPOINT_M, 4.0)
    assert above == pytest.approx(below, abs=1e-6)


def test_max_range_hand_values():
    # allowed loss 19.5 + 98 = 117.5 dB; minus PL0 40.2 leaves 77.3 dB at 20 dB per decade
    assert hc.max_range_m(TX_DBM, SENS_DBM, PL0_DB, 2.0, 0.0) == pytest.approx(7328.2, rel=1e-4)
    assert hc.max_range_m(TX_DBM, SENS_DBM, PL0_DB, 2.0, 20.0) == pytest.approx(732.82, rel=1e-4)
    # inverting back gives the allowed loss
    d = hc.max_range_m(TX_DBM, SENS_DBM, PL0_DB, 2.6, 10.0)
    assert TX_DBM - hc.log_distance_path_loss_db(d, PL0_DB, 2.6) == pytest.approx(SENS_DBM + 10.0)


def test_max_range_dual_slope_stress_corner():
    # stress corner median signal with 6 dB orientation loss as extra margin:
    # 111.5 - 77.485 = 34.015 dB at 40 dB per decade -> 73.16 * 10**0.8504 = 518 m
    d = hc.max_range_dual_slope_m(TX_DBM, SENS_DBM, PL0_DB, 2.0, BREAKPOINT_M, 4.0, 6.0)
    assert d == pytest.approx(518.4, abs=1.0)
    # sanity against Run C's simulated stress corner: d90 409 m < median range < d10 746 m
    assert 409.0 < d < 746.0
    # margin small enough to stop before the breakpoint: falls back to one slope
    assert hc.max_range_dual_slope_m(TX_DBM, SENS_DBM, PL0_DB, 2.0, BREAKPOINT_M, 4.0, 50.0) == pytest.approx(
        hc.max_range_m(TX_DBM, SENS_DBM, PL0_DB, 2.0, 50.0))


def test_relay_hand_cases():
    assert hc.relay_feasible(180.0, 0.5, 100.0)
    assert not hc.relay_feasible(180.0, 0.3, 100.0)   # second hop 126 m
    assert hc.relay_feasible(200.0, 0.5, 100.0)       # both hops exactly at range
    assert hc.relay_feasible(80.0, 0.0, 100.0)        # direct link already closes
    assert not hc.relay_feasible(250.0, 0.5, 100.0)
    with pytest.raises(ValueError):
        hc.relay_feasible(100.0, 1.5, 100.0)


def test_best_relay_fraction_by_search():
    fractions = np.linspace(0.0, 1.0, 1001)
    longest_hop = np.maximum(fractions, 1.0 - fractions)
    assert fractions[np.argmin(longest_hop)] == pytest.approx(hc.best_relay_fraction())
    # only the midpoint works just under twice the range
    ok = [f for f in fractions if hc.relay_feasible(199.0, f, 100.0)]
    assert min(ok) < 0.5 < max(ok) and max(ok) - min(ok) < 0.02


# ---------------------------------------------------------------------------
# 7. Alert latency
# ---------------------------------------------------------------------------
def test_alert_latency_hand_values():
    assert hc.alert_latency_s(0.25, 0.02, 0.01) == pytest.approx(0.28)
    assert hc.expected_detection_s(0.1, 1) == pytest.approx(0.05)
    assert hc.expected_detection_s(0.1, 3) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        hc.expected_detection_s(0.1, 0)


def test_expected_detection_sampled():
    # fixes every Ts with a random phase; entry at t = 0; alert at the k-th fix after entry
    Ts, k = 0.1, 3
    phase = _rng(5).uniform(0.0, Ts, 200_000)       # time of the first fix after entry, minus a period
    first_fix = Ts - phase                          # in (0, Ts]
    alert = first_fix + (k - 1) * Ts
    assert alert.mean() == pytest.approx(hc.expected_detection_s(Ts, k), abs=1e-3)
    assert alert.min() > (k - 1) * Ts - 1e-12 and alert.max() <= k * Ts + 1e-12


# ---------------------------------------------------------------------------
# 8. Loop jitter
# ---------------------------------------------------------------------------
def test_jitter_hand_value_and_limits():
    # a = 1 - exp(-1/3) = 0.28347; a / (2 - a) = 0.16514; std = 0.05 * 0.40638
    assert hc.first_order_loop_jitter_std(0.05, 0.1, 0.3) == pytest.approx(0.020319, abs=2e-6)
    assert hc.first_order_loop_jitter_std(0.0, 0.1, 0.3) == 0.0
    # very slow fixes: the vehicle fully reaches each noisy setpoint, std -> sigma
    assert hc.first_order_loop_jitter_std(0.05, 100.0, 0.3) == pytest.approx(0.05, rel=1e-9)
    # fast fixes: sigma * sqrt(Ts / (2 tau))
    assert hc.first_order_loop_jitter_std(0.05, 0.001, 0.3) == pytest.approx(
        0.05 * math.sqrt(0.001 / 0.6), rel=2e-3)


@pytest.mark.parametrize("Ts,tau,idx", [(0.1, 0.3, 6), (0.5, 0.3, 7)], ids=["fast", "slow"])
def test_jitter_monte_carlo(Ts, tau, idx):
    # Independent route: integrate dx/dt = (s - x) / tau with small Euler steps
    # between fixes instead of using the exact per-fix factor.
    sigma, chains, periods, burn, sub = 0.05, 4000, 300, 50, 100
    rng = _rng(idx)
    dt = Ts / sub
    x = np.zeros(chains)                            # goal at 0
    samples = []
    for k in range(periods):
        noise = rng.normal(0.0, sigma, chains)
        s = x - (x + noise)                         # setpoint = x + (goal - fix)
        for _ in range(sub):
            x = x + (s - x) * dt / tau
        if k >= burn:
            samples.append(x.copy())
    std = float(np.std(np.concatenate(samples)))
    assert std == pytest.approx(hc.first_order_loop_jitter_std(sigma, Ts, tau), rel=0.05)


# ---------------------------------------------------------------------------
# 9. Lag
# ---------------------------------------------------------------------------
def test_lag_hand_values():
    assert hc.lag_error_m(1.5, 0.15) == pytest.approx(0.225)
    assert hc.lag_error_m(0.0, 0.5) == 0.0
    assert hc.lag_error_m(0.5, 0.0) == 0.0
