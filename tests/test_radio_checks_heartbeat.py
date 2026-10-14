"""Independent checks for sim/radio/heartbeat.py, written from the math only.

Every expected value below is hand-derived from derivations D9 (heartbeat false
alarms and detection delay) and D3 (Gilbert-Elliott chain), not copied from the
module under test. Only the public signatures were read, never the bodies.

Setup (D9): heartbeats every 1/h seconds, alarm when the gap since the last
received heartbeat is strictly greater than T. After a received heartbeat, m
straight losses make a gap of (m + 1)/h, so an alarm needs m >= m* = floor(T h).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from sim.radio.channel import GilbertElliott
from sim.radio.heartbeat import (
    detection_delay_bounds,
    detection_delay_samples,
    false_alarm_rate_ge_per_hour,
    false_alarm_rate_iid_per_hour,
    max_misses_before_alarm,
    simulate_heartbeats,
)

# 10^7 slots at 10 Hz is about 278 hours. If this file ever runs too slowly,
# drop to 3 * 10^6 (the 4 sigma tolerance scales with the count anyway).
N_SLOTS = 10_000_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _count_episodes(lost: np.ndarray, m: int) -> int:
    """Independent episode counter, no use of times or floats.

    An episode is a run of >= m losses that follows a reception AND is followed
    by a later reception (the gap is closed, so it can be measured). Count the
    receptions k whose next m slots are all lost and that have a later
    reception. A longer run is counted once, because only the reception right
    before the run qualifies.
    """
    n = lost.size
    rec = ~lost
    # all_lost[k] = slots k+1..k+m all lost
    window = np.ones(n - m, dtype=bool)
    for j in range(1, m + 1):
        window &= lost[j : n - m + j]
    start = rec[: n - m] & window
    # Keep only runs that end with a later reception inside the array.
    idx = np.flatnonzero(start)
    if idx.size == 0:
        return 0
    # Next reception after the reception at k: first index > k where rec is True.
    rec_pos = np.flatnonzero(rec)
    nxt = np.searchsorted(rec_pos, idx, side="right")
    return int(np.count_nonzero(nxt < rec_pos.size))


# --------------------------------------------------------------------------
# 1. m* = floor(T h)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate_hz, timeout_s, expected",
    [
        (10.0, 0.3, 3),   # 3.0 -> 3
        (10.0, 0.25, 2),  # 2.5 -> 2
        (10.0, 0.35, 3),  # 3.5 -> 3
        (50.0, 0.1, 5),   # 5.0 -> 5
        (20.0, 0.3, 6),   # 6.0 -> 6
        (1.0, 0.99, 0),   # T < 1/h: a single miss already exceeds T
        (2.0, 1.0, 2),
    ],
)
def test_max_misses_hand_cases(rate_hz, timeout_s, expected):
    """D9: m* = floor(T h), by hand."""
    assert max_misses_before_alarm(timeout_s, rate_hz) == expected


def test_max_misses_float_edge():
    """True product is an exact integer but the float product is a hair off.

    0.29 * 100 = 28.999999999999996, 1.16 * 25 = 28.999999999999996 and
    0.28 * 25 = 7.000000000000001 in float; the hand answers are 29, 29, 7.
    A bare floor() gives 28 for the first two.
    """
    assert 0.29 * 100 < 29.0 and 1.16 * 25 < 29.0  # the trap exists in this Python
    assert max_misses_before_alarm(0.29, 100.0) == 29
    assert max_misses_before_alarm(1.16, 25.0) == 29
    assert max_misses_before_alarm(0.28, 25.0) == 7
    assert max_misses_before_alarm(2.28, 25.0) == 57
    assert max_misses_before_alarm(0.3, 10.0) == 3
    assert max_misses_before_alarm(0.3, 10) == 3


# --------------------------------------------------------------------------
# 2. Closed forms
# --------------------------------------------------------------------------


def test_iid_closed_form_hand_values():
    """D9: rate = 3600 h (1 - p) p^m*; 36000 * 0.9 * 1e-3 = 32.4 and 36000 * 0.99 * 1e-6."""
    assert false_alarm_rate_iid_per_hour(0.1, 10.0, 0.3) == pytest.approx(32.4, rel=1e-12)
    assert false_alarm_rate_iid_per_hour(0.01, 10.0, 0.3) == pytest.approx(0.03564, rel=1e-12)
    # m* = 2 (T = 0.25): 36000 * 0.9 * 0.01 = 324.
    assert false_alarm_rate_iid_per_hour(0.1, 10.0, 0.25) == pytest.approx(324.0, rel=1e-12)


def test_iid_limits():
    """No loss -> no alarms. Total loss -> (1 - p) = 0 so no reception to start from."""
    assert false_alarm_rate_iid_per_hour(0.0, 10.0, 0.3) == 0.0
    assert false_alarm_rate_iid_per_hour(1.0, 10.0, 0.3) == 0.0


def test_classic_gilbert_closed_form():
    """D3/D9: classic Gilbert (h_G=0, h_B=1) rate = 3600 h pi_G p (1-r)^(m*-1).

    p = 0.1 * 0.2 / 0.9 = 1/45, r = 0.2, m* = 3:
    36000 * 0.9 * (1/45) * 0.8^2 = 36000 * 0.9 * 0.64 / 45 = 460.8 per hour.
    """
    ge = GilbertElliott(1.0 / 45.0, 0.2)
    assert ge.stationary_bad() == pytest.approx(0.1, rel=1e-12)
    assert false_alarm_rate_ge_per_hour(ge, 10.0, 0.3) == pytest.approx(460.8, rel=1e-12)


def _brute_force_ge_rate(ge: GilbertElliott, rate_hz: float, m: int) -> float:
    """Sum over all 2^(m+1) state paths s_0..s_m of
    pi(s_0) (1 - h_{s_0}) * prod_{i=1..m} P(s_{i-1}, s_i) h_{s_i}, times 3600 h.

    Slot 0 is received in state s_0; slots 1..m are lost, each in the state the
    chain has then. This is the D9 matrix product written out as a plain sum.
    """
    p, r = ge.p_good_to_bad, ge.p_bad_to_good
    h = {0: ge.loss_good, 1: ge.loss_bad}  # 0 = Good, 1 = Bad
    pi = {0: r / (p + r), 1: p / (p + r)}
    trans = {(0, 0): 1 - p, (0, 1): p, (1, 0): r, (1, 1): 1 - r}
    total = 0.0
    for path in itertools.product((0, 1), repeat=m + 1):
        prob = pi[path[0]] * (1.0 - h[path[0]])
        for i in range(1, m + 1):
            prob *= trans[(path[i - 1], path[i])] * h[path[i]]
        total += prob
    return 3600.0 * rate_hz * total


@pytest.mark.parametrize(
    "ge, rate_hz, timeout_s",
    [
        (GilbertElliott(0.03, 0.25, 0.05, 0.8), 10.0, 0.3),   # m* = 3
        (GilbertElliott(0.1, 0.4, 0.05, 0.8), 10.0, 0.25),    # m* = 2
        (GilbertElliott(0.02, 0.1, 0.01, 0.9), 20.0, 0.3),    # m* = 6
        (GilbertElliott(1.0 / 45.0, 0.2), 10.0, 0.3),         # classic
    ],
)
def test_general_ge_matches_brute_force_enumeration(ge, rate_hz, timeout_s):
    """D9: matrix formula equals the explicit sum over state paths (rel 1e-12)."""
    m = int(math.floor(timeout_s * rate_hz + 1e-9))
    expected = _brute_force_ge_rate(ge, rate_hz, m)
    got = false_alarm_rate_ge_per_hour(ge, rate_hz, timeout_s)
    assert got == pytest.approx(expected, rel=1e-12)


def test_ge_with_equal_loss_in_both_states_reduces_to_iid():
    """If h_G = h_B = p the chain is irrelevant: the GE rate must equal the iid rate."""
    ge = GilbertElliott(0.05, 0.3, 0.1, 0.1)
    assert false_alarm_rate_ge_per_hour(ge, 10.0, 0.3) == pytest.approx(
        false_alarm_rate_iid_per_hour(0.1, 10.0, 0.3), rel=1e-12
    )


def test_bursts_raise_false_alarms_about_14x():
    """D9 example: 460.8 / 32.4 = 14.22, same mean loss 0.1."""
    ge = GilbertElliott(1.0 / 45.0, 0.2)
    ratio = false_alarm_rate_ge_per_hour(ge, 10.0, 0.3) / false_alarm_rate_iid_per_hour(0.1, 10.0, 0.3)
    assert ratio == pytest.approx(460.8 / 32.4, rel=1e-12)
    assert 14.0 < ratio < 14.5


# --------------------------------------------------------------------------
# 3. Simulation versus closed form (loss sequences generated here)
# --------------------------------------------------------------------------


def _check_sim_against_closed_form(lost, expected_per_hour, rate_hz=10.0, timeout_s=0.3):
    m = int(math.floor(timeout_s * rate_hz + 1e-9))
    res = simulate_heartbeats(lost, rate_hz, timeout_s)
    mine = _count_episodes(lost, m)
    expected = expected_per_hour * lost.size / rate_hz / 3600.0
    assert res.n_episodes == mine, (res.n_episodes, mine)
    assert res.alarm_start_times.size == res.n_episodes
    assert abs(mine - expected) <= 4.0 * math.sqrt(expected) + 1.0, (mine, expected)
    return res, mine, expected


def test_sim_iid_p01_counts_match_closed_form():
    """D9: ~9000 expected episodes in 10^7 slots; iid loss p = 0.1 generated here."""
    rng = _rng(101)
    lost = rng.random(N_SLOTS) < 0.1
    expected_rate = 36000.0 * 0.9 * 0.1**3  # hand value 32.4
    res, mine, expected = _check_sim_against_closed_form(lost, expected_rate)
    assert expected == pytest.approx(9000.0, rel=1e-9)
    # The module's own rate field should be episodes per simulated hour.
    hours = N_SLOTS / 10.0 / 3600.0
    assert res.false_alarm_rate_per_hour == pytest.approx(mine / hours, rel=1e-3)


def test_sim_classic_gilbert_counts_match_closed_form():
    """D9: classic Gilbert, mean loss 0.1, burst 5 (p = 1/45, r = 0.2): 460.8 per hour."""
    ge = GilbertElliott(1.0 / 45.0, 0.2)
    _, lost = ge.sample(N_SLOTS, _rng(202))
    lost = np.asarray(lost, dtype=bool)
    assert lost.mean() == pytest.approx(0.1, abs=4e-3)  # D3 tolerance scale
    _check_sim_against_closed_form(lost, 460.8)


def test_sim_total_time_in_alarm_hand_case():
    """Hand-built sequence: one reception, 5 losses, one reception, at 10 Hz, T = 0.3.

    Receptions at t = 0 and t = 0.6. Alarm fires at 0.3 (gap 0.3 is NOT > 0.3
    before 0.3, so it starts after), ends at 0.6: one episode, ~0.3 s in alarm.
    """
    lost = np.array([False, True, True, True, True, True, False, False])
    res = simulate_heartbeats(lost, 10.0, 0.3)
    assert res.n_episodes == 1
    assert res.total_time_in_alarm == pytest.approx(0.3, abs=1e-9)
    assert res.alarm_start_times[0] == pytest.approx(0.3, abs=1e-9)


def test_sim_exact_threshold_gap_is_not_an_alarm():
    """D9 says strictly greater: 2 losses (gap 0.3 s) at T = 0.3 must NOT alarm; 3 losses must."""
    two = np.array([False, True, True, False, False])
    three = np.array([False, True, True, True, False, False])
    assert simulate_heartbeats(two, 10.0, 0.3).n_episodes == 0
    assert simulate_heartbeats(three, 10.0, 0.3).n_episodes == 1


def test_sim_exact_threshold_gap_is_not_an_alarm_at_any_phase():
    """Same as above, but the gap sits at every offset in the stream.

    The true gap between receptions k and k+3 is exactly 0.3 s for every k. In
    float, (k+3)/10 - k/10 is sometimes 0.30000000000000004, so a naive
    gap > T comparison false-alarms on a fraction of offsets. Hand answer: 0.
    """
    bad = []
    for k in range(0, 400):
        lost = np.zeros(k + 4, dtype=bool)
        lost[k + 1] = lost[k + 2] = True  # receptions at k and k+3: two losses, gap 3/h = 0.3 s
        if simulate_heartbeats(lost, 10.0, 0.3).n_episodes != 0:
            bad.append(k)
    assert not bad, f"{len(bad)} of 400 offsets false-alarmed on a gap exactly equal to T, first: {bad[:5]}"


def test_sim_no_loss_no_alarm():
    res = simulate_heartbeats(np.zeros(5000, dtype=bool), 10.0, 0.3)
    assert res.n_episodes == 0
    assert res.total_time_in_alarm == 0.0


# --------------------------------------------------------------------------
# 4. Detection delay
# --------------------------------------------------------------------------


def test_detection_delay_bounds_hand_values():
    """D9: delay uniform on (T - 1/h, T]; mean T - 1/(2h). h = 10, T = 0.3: (0.2, 0.3], 0.25."""
    lo, hi, mean = detection_delay_bounds(10.0, 0.3)
    assert lo == pytest.approx(0.2, abs=1e-12)
    assert hi == pytest.approx(0.3, abs=1e-12)
    assert mean == pytest.approx(0.25, abs=1e-12)


def test_detection_delay_no_prior_loss_is_uniform():
    """D9: 20000 samples in (T - 1/h, T], mean within 4 se of T - 1/(2h)."""
    h, T, n = 10.0, 0.3, 20_000
    d = np.asarray(detection_delay_samples(h, T, n, _rng(303)))
    assert d.shape == (n,)
    eps = 1e-9
    assert d.max() <= T + eps
    assert d.min() >= T - 1.0 / h - eps
    se = (1.0 / h) / math.sqrt(12.0) / math.sqrt(n)  # uniform sd = width / sqrt(12)
    assert abs(d.mean() - (T - 1.0 / (2 * h))) <= 4.0 * se
    # Uniform shape: each quarter of the interval holds about a quarter of the samples.
    counts, _ = np.histogram(d, bins=4, range=(T - 1.0 / h, T))
    for c in counts:
        assert abs(c - n / 4) <= 4.0 * math.sqrt(n * 0.25 * 0.75)


def test_detection_delay_with_prior_loss_is_shorter():
    """D9: prior losses make the last heartbeat older, so the mean delay is strictly smaller."""
    h, T, n = 10.0, 0.3, 20_000
    d = np.asarray(detection_delay_samples(h, T, n, _rng(404), pre_loss=0.3))
    se = (1.0 / h) / math.sqrt(12.0) / math.sqrt(n)
    assert d.mean() < (T - 1.0 / (2 * h)) - 4.0 * se
    assert d.max() <= T + 1e-9
    assert d.min() >= 0.0
    # Hand estimate: each extra lost slot before the failure moves the last
    # heartbeat back by 1/h. Expected number of extra lost slots right before
    # the failure is p/(1-p) = 3/7, and the clipped-at-0 tail is negligible
    # (needs m* losses in a row, 0.3^3 = 2.7 percent of trials, delay 0 for those).
    # So the mean is far below 0.25 but still positive.
    assert d.mean() > 0.0


# --------------------------------------------------------------------------
# 5. Jitter
# --------------------------------------------------------------------------


def test_small_jitter_with_no_loss_gives_zero_episodes():
    """Gaps are 0.1 s +- about 1.4 ms: nowhere near T = 0.3 s, so no alarm at all."""
    lost = np.zeros(20_000, dtype=bool)
    res = simulate_heartbeats(lost, 10.0, 0.3, jitter_s=1e-3, rng=_rng(505))
    assert res.n_episodes == 0
    assert res.total_time_in_alarm == 0.0
    # Same with an explicit jitter array drawn here.
    jit = _rng(506).normal(0.0, 1e-3, lost.size)
    res2 = simulate_heartbeats(lost, 10.0, 0.3, jitter_s=jit)
    assert res2.n_episodes == 0
