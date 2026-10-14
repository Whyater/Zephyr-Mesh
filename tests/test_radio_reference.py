"""Checks of the standalone reference simulator (tests/radio_reference.py).

The reference is the yardstick for the MAC cross-checks, so it is itself checked
against closed forms derived by hand (see the Run C derivations, D4 and D5):

* Pure ALOHA: P(success) = exp(-2G), S = G exp(-2G), peak S = 1/(2e) at G = 1/2.
* Slotted ALOHA: P(success) = exp(-G), S = G exp(-G), peak S = 1/e at G = 1.
* M/D/1: Wq = rho / (2 mu (1 - rho)), mu = 1 / D.

Tolerance rule: 4 standard errors.  For pure ALOHA the plain binomial standard
error sqrt(p (1 - p) / n) is too small, because neighbouring frames share one gap
(their success indicators are positively correlated).  The corrected standard
error is derived in radio_reference.pure_aloha_fraction_se and is itself tested
below against the spread over repeated runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import radio_reference as ref  # noqa: E402

N_FRAMES = 200_000


# ---------------------------------------------------------------------------
# Pure ALOHA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [0.25, 0.5, 1.0])
def test_pure_aloha_throughput_matches_g_exp_minus_2g(G):
    r = ref.simulate_pure_aloha(G, N_FRAMES, ref.make_rng(100 + int(G * 100)))
    expected = G * np.exp(-2.0 * G)
    assert abs(r["throughput"] - expected) <= 4.0 * r["se_throughput"]
    # the same statement on the per-frame success probability
    assert abs(r["success_fraction"] - np.exp(-2.0 * G)) <= 4.0 * r["se_fraction"]


def test_pure_aloha_correlation_corrected_se_matches_repeat_spread():
    """The standard error formula (with neighbour covariance) matches the real spread."""
    G, n, reps = 0.5, 20_000, 80
    rng = ref.make_rng(7)
    fr = np.array([ref.simulate_pure_aloha(G, n, rng)["success_fraction"] for _ in range(reps)])
    emp = fr.std(ddof=1)
    formula = ref.pure_aloha_fraction_se(G, n)
    naive = np.sqrt(np.exp(-2 * G) * (1 - np.exp(-2 * G)) / n)
    # emp has relative error about 1/sqrt(2 (reps - 1)) = 8 %, so 30 % is about 3.5 sigma
    assert abs(emp / formula - 1.0) < 0.30
    assert formula > 1.2 * naive  # the correction matters (about 1.3x at this load)


def test_pure_aloha_peak_is_one_over_two_e():
    # closed form: the maximum of G exp(-2G) is at G = 1/2 and equals 1/(2e)
    g = np.linspace(0.01, 3.0, 29_901)
    s = g * np.exp(-2.0 * g)
    assert abs(g[np.argmax(s)] - 0.5) < 1e-3
    assert s.max() == pytest.approx(1.0 / (2.0 * np.e), abs=1e-6)
    # simulated: S at the peak matches 1/(2e) and is above S at half and double load
    rng = ref.make_rng(11)
    peak = ref.simulate_pure_aloha(0.5, N_FRAMES, rng)
    lo = ref.simulate_pure_aloha(0.25, N_FRAMES, rng)
    hi = ref.simulate_pure_aloha(1.0, N_FRAMES, rng)
    assert abs(peak["throughput"] - 1.0 / (2.0 * np.e)) <= 4.0 * peak["se_throughput"]
    for other in (lo, hi):
        gap = peak["throughput"] - other["throughput"]
        assert gap > 4.0 * np.hypot(peak["se_throughput"], other["se_throughput"])


# ---------------------------------------------------------------------------
# Slotted ALOHA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [0.5, 1.0, 2.0])
def test_slotted_aloha_throughput_matches_g_exp_minus_g(G):
    r = ref.simulate_slotted_aloha(G, N_FRAMES, ref.make_rng(200 + int(G * 100)))
    # S is a mean of iid Bernoulli slot indicators: exact binomial standard error
    assert abs(r["throughput"] - G * np.exp(-G)) <= 4.0 * r["se_throughput"]
    assert abs(r["success_fraction"] - np.exp(-G)) <= 4.0 * r["se_fraction"]
    assert abs(r["n"] - N_FRAMES) < 0.02 * N_FRAMES  # about the requested frame count


def test_slotted_aloha_peak_is_one_over_e():
    g = np.linspace(0.01, 5.0, 49_901)
    s = g * np.exp(-g)
    assert abs(g[np.argmax(s)] - 1.0) < 1e-3
    assert s.max() == pytest.approx(1.0 / np.e, abs=1e-6)
    rng = ref.make_rng(12)
    peak = ref.simulate_slotted_aloha(1.0, N_FRAMES, rng)
    lo = ref.simulate_slotted_aloha(0.5, N_FRAMES, rng)
    hi = ref.simulate_slotted_aloha(2.0, N_FRAMES, rng)
    assert abs(peak["throughput"] - 1.0 / np.e) <= 4.0 * peak["se_throughput"]
    for other in (lo, hi):
        gap = peak["throughput"] - other["throughput"]
        assert gap > 4.0 * np.hypot(peak["se_throughput"], other["se_throughput"])


def test_slotted_peak_is_twice_pure_peak():
    # 1/e divided by 1/(2e) is exactly 2 (a classic result); same simulations as above
    rng = ref.make_rng(13)
    sp = ref.simulate_slotted_aloha(1.0, N_FRAMES, rng)["throughput"]
    pp = ref.simulate_pure_aloha(0.5, N_FRAMES, rng)["throughput"]
    assert sp / pp == pytest.approx(2.0, rel=0.02)


# ---------------------------------------------------------------------------
# Classifiers on hand-made inputs
# ---------------------------------------------------------------------------

def test_classify_pure_hand_cases():
    T = 1.0
    # 0.0 and 0.5 overlap; 2.0 is alone; 3.0 and 4.0 touch (gap exactly T): both fine
    ok = ref.classify_pure([0.0, 0.5, 2.0, 3.0, 4.0], T)
    assert ok.tolist() == [False, False, True, True, True]
    # input order is preserved
    ok = ref.classify_pure([4.0, 0.5, 2.0, 0.0, 3.0], T)
    assert ok.tolist() == [True, False, True, False, True]
    # floating point rounding at exact adjacency must not count as overlap
    t = 0.000792
    assert ref.classify_pure([0.0, t * (1.0 - 1e-12)], t).tolist() == [True, True]
    assert ref.classify_pure([0.0, t * 0.999], t).tolist() == [False, False]


def test_classify_slotted_hand_cases():
    T = 0.5
    ok, slot = ref.classify_slotted([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 2.0], T)
    assert slot.tolist() == [0, 0, 1, 2, 2, 2, 4]
    assert ok.tolist() == [False, False, True, False, False, False, True]
    with pytest.raises(ValueError):
        ref.classify_slotted([0.0, 0.2], T)


# ---------------------------------------------------------------------------
# Vectorised queue recursions against slow sequential loops
# ---------------------------------------------------------------------------

def test_defer_functions_match_sequential_loops():
    rng = ref.make_rng(21)
    T = 1.0
    a = np.sort(rng.uniform(0, 30, size=60))  # heavy overlap on purpose
    s_pure, s_slot = [], []
    for k, x in enumerate(a):
        s_pure.append(x if k == 0 else max(x, s_pure[-1] + T))
        m = np.ceil(x / T)
        s_slot.append(m * T if k == 0 else max(m, s_slot[-1] / T + 1) * T)
    np.testing.assert_allclose(ref.defer_pure(a, T), s_pure, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ref.defer_slotted(a, T), s_slot, rtol=0, atol=1e-9)
    # back-to-back or later, never earlier than arrival, never closer than T
    assert np.all(ref.defer_pure(a, T) >= a)
    assert np.all(np.diff(ref.defer_pure(a, T)) >= T - 1e-9)
    assert np.all(np.diff(ref.defer_slotted(a, T)) >= T - 1e-9)


def test_md1_lindley_matches_sequential_loop():
    lam, D, n = 0.8, 1.0, 5_000
    w = ref.md1_waits(lam, D, n, ref.make_rng(22))
    a = ref.make_rng(22).exponential(1.0 / lam, size=n)  # same stream, same draw
    loop = [0.0]
    for k in range(1, n):
        loop.append(max(0.0, loop[-1] + D - a[k]))
    np.testing.assert_allclose(w, loop, rtol=0, atol=1e-9)


def test_single_sender_starts_and_hol_wait_match_event_loop():
    rng = ref.make_rng(23)
    T, difs = 0.792e-3, 50e-6
    arr = np.cumsum(rng.exponential(1.0 / 900.0, size=3_000))  # rho about 0.76
    starts = ref.single_sender_starts(arr, T, difs)
    hol = ref.head_of_line_waits(arr, starts, T)
    # slow event loop written from the rules, not from the recursion
    prev_end = None
    for k, a in enumerate(arr):
        head = a if prev_end is None else max(a, prev_end)       # frame k becomes head of line
        start = head + difs                                      # always DIFS of idle after head of line
        assert starts[k] == pytest.approx(start, abs=1e-9)
        assert hol[k] == pytest.approx(head - a, abs=1e-9)
        prev_end = start + T
    # algebra: H = max(W - difs, 0)
    np.testing.assert_allclose(hol, np.maximum(starts - arr - difs, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# M/D/1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_md1_mean_wait_matches_pollaczek_khinchine(rho):
    D = 1e-3  # seconds; example in the derivations: rho 0.5 -> 0.5 ms, rho 0.8 -> 2.0 ms
    lam = rho / D
    expected = rho / (2.0 * (1.0 / D) * (1.0 - rho))
    assert ref.md1_mean_wait(lam, D) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx({0.5: 0.5e-3, 0.8: 2.0e-3}[rho], rel=1e-12)
    n_batches, batch = 20, 50_000
    w = ref.md1_waits(lam, D, 5_000 + n_batches * batch, ref.make_rng(300 + int(rho * 10)))
    mean, se = ref.batch_means(w[5_000:], n_batches)  # drop the empty-start transient
    assert abs(mean - expected) <= 4.0 * se
    assert se < 0.05 * expected  # the check is not vacuous


def test_md1_rejects_unstable_load():
    with pytest.raises(ValueError):
        ref.md1_mean_wait(1.2e3, 1e-3)


# ---------------------------------------------------------------------------
# Finite population (what the MAC engine actually runs)
# ---------------------------------------------------------------------------

def test_finite_n_matches_closed_form_up_to_known_small_bias():
    # first-order bias of the finite-population success probability versus the
    # infinite-population closed form: a node never overlaps itself, so
    # pure p_N is about exp(-2G (1 - 1/N)), i.e. p (1 + 2G/N); slotted similar, G/N.
    N, G = 200, 1.0
    rng = ref.make_rng(31)
    res = ref.finite_n_aloha(G, N, 4_000.0, rng)
    mean, se, n = ref.window_success(res, 20.0, 3_980.0)
    p = np.exp(-2 * G)
    assert abs(mean - p) <= 4.0 * se + 2 * G * p / N
    res = ref.finite_n_aloha(2.0, N, 4_000.0, rng, slotted=True)
    mean, se, n = ref.window_success(res, 20.0, 3_980.0)
    assert abs(mean - np.exp(-2.0)) <= 4.0 * se + 2.0 * np.exp(-2.0) / N


def test_deterministic_per_seed():
    a = ref.simulate_pure_aloha(0.5, 1000, ref.make_rng(5))
    b = ref.simulate_pure_aloha(0.5, 1000, ref.make_rng(5))
    c = ref.simulate_pure_aloha(0.5, 1000, ref.make_rng(6))
    assert a == b and a != c
    x = ref.finite_n_aloha(1.0, 20, 200.0, ref.make_rng(5))
    y = ref.finite_n_aloha(1.0, 20, 200.0, ref.make_rng(5))
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])
