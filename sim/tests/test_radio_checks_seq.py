"""Independent checks for sim/radio/seq.py and sim/radio/replay.py, from the math only.

Every expected value below was derived by hand from derivation D11 (32-bit serial
arithmetic, diff(a, b) = ((a - b + 2^31) mod 2^32) - 2^31) and the status
documented definitions, NOT taken from the code under test.

The modules are imported lazily through fixtures so this file collects (and the
other tests skip) before the modules exist, while `test_module_exists` does plain
imports and fails loudly on a base commit.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

MOD = 2**32


@pytest.fixture(scope="module")
def sq():
    return pytest.importorskip("sim.radio.seq")


@pytest.fixture(scope="module")
def rp():
    return pytest.importorskip("sim.radio.replay")


def test_module_exists():
    """Plain imports: fail (not skip) when sim/radio/seq.py or replay.py is missing."""
    import sim.radio.replay  # noqa: F401
    import sim.radio.seq  # noqa: F401


# ---------------------------------------------------------------------------
# D11: serial-number arithmetic
# ---------------------------------------------------------------------------


def test_d11_seq_diff_hand_cases(sq):
    """D11: diff(0, 2^32-1) = ((0 - (2^32-1) + 2^31) mod 2^32) - 2^31 = 1 (0 is one ahead
    of the largest value). diff(2^32-1, 0) = -1. diff(5, 3) = 2. Hand arithmetic."""
    assert sq.seq_diff(0, MOD - 1) == 1
    assert sq.seq_diff(MOD - 1, 0) == -1
    assert sq.seq_diff(5, 3) == 2
    assert sq.seq_diff(3, 5) == -2
    assert sq.seq_diff(7, 7) == 0
    assert sq.seq_diff(2, MOD - 3) == 5  # across the wrap: 2 is 5 steps after 2^32-3


def test_d11_seq_diff_range_boundaries(sq):
    """D11: the result lies in [-2^31, 2^31). diff(2^31 - 1, 0) = 2^31 - 1 (largest value);
    diff(2^31, 0) = ((2^31 + 2^31) mod 2^32) - 2^31 = -2^31 (smallest value);
    diff(0, 2^31) is also -2^31. Hand arithmetic."""
    assert sq.seq_diff(2**31 - 1, 0) == 2**31 - 1
    assert sq.seq_diff(2**31, 0) == -(2**31)
    assert sq.seq_diff(0, 2**31) == -(2**31)
    assert sq.seq_diff(1, 2**31 + 1) == -(2**31)


def test_d11_seq_diff_matches_formula_on_random_pairs(sq):
    """D11: compare to the closed form ((a - b + 2^31) mod 2^32) - 2^31 written out here,
    on 2000 random pairs, and check diff(a, b) == -diff(b, a) away from the -2^31 point."""
    rng = np.random.default_rng(np.random.SeedSequence(3))
    a = rng.integers(0, MOD, size=2000, dtype=np.int64)
    b = rng.integers(0, MOD, size=2000, dtype=np.int64)
    for x, y in zip(a.tolist(), b.tolist()):
        want = ((x - y + 2**31) % MOD) - 2**31
        got = sq.seq_diff(x, y)
        assert got == want
        assert -(2**31) <= got < 2**31
        if want != -(2**31):
            assert sq.seq_diff(y, x) == -want


def test_d11_seq_next_wraps(sq):
    """D11: (s + 1) mod 2^32. seq_next(5) = 6, seq_next(2^32 - 1) = 0."""
    assert sq.seq_next(5) == 6
    assert sq.seq_next(MOD - 1) == 0
    assert sq.seq_next(MOD - 2) == MOD - 1


# ---------------------------------------------------------------------------
# D11: SequenceTracker
# ---------------------------------------------------------------------------


def test_d11_explicit_wrap_run_all_new_unwrapped_plus_one(sq):
    """D11: seq 2^32-3, -2, -1, 0, 1, 2 in order are all 'new', unwrapped strictly +1
    (first unwrapped equals its seq, then 2^32-2, 2^32-1, 2^32, 2^32+1, 2^32+2), gap 0,
    accept True, 0 duplicates. Expected values derived by hand."""
    tr = sq.SequenceTracker()
    seqs = [MOD - 3, MOD - 2, MOD - 1, 0, 1, 2]
    obs = [tr.observe("a", s) for s in seqs]
    assert [o.status for o in obs] == ["new"] * 6
    assert [o.unwrapped for o in obs] == [MOD - 3 + i for i in range(6)]
    assert [o.gap for o in obs] == [0] * 6
    assert all(o.accept for o in obs)
    assert tr.counts["new"] == 6
    assert tr.counts["duplicate"] == 0
    assert tr.counts["stale"] == 0
    assert tr.counts["reordered"] == 0
    assert tr.counts["gap_packets"] == 0


def test_d11_first_packet_unwrapped_equals_seq(sq):
    """Contract: a sender's first packet is 'new' with unwrapped equal to its seq."""
    tr = sq.SequenceTracker()
    o = tr.observe("a", 12345)
    assert (o.status, o.unwrapped, o.gap, o.accept) == ("new", 12345, 0, True)


def test_d11_million_in_order_packets_across_wrap(sq):
    """D11: 10^6 unique in-order packets crossing the 2^32 wrap: 0 duplicates, 0 stale,
    0 reordered, every packet 'new' with unwrapped +1 and gap 0.

    Timing guard: this should finish in under 20 s. If it is slower on this machine,
    that is the module's performance problem (observe() should be O(1)), not the check's.
    """
    n = 1_000_000
    start = MOD - 500_000
    tr = sq.SequenceTracker()
    not_new = 0
    bad_step = 0
    bad_gap = 0
    accepted = 0
    prev_unwrapped = None
    t0 = time.perf_counter()
    for i in range(n):
        o = tr.observe("a", (start + i) % MOD)
        if o.status != "new":
            not_new += 1
        if prev_unwrapped is not None and o.unwrapped != prev_unwrapped + 1:
            bad_step += 1
        if o.gap != 0:
            bad_gap += 1
        if o.accept:
            accepted += 1
        prev_unwrapped = o.unwrapped
    elapsed = time.perf_counter() - t0
    assert not_new == 0
    assert bad_step == 0
    assert bad_gap == 0
    assert accepted == n
    assert prev_unwrapped == start + n - 1
    assert tr.counts["new"] == n
    assert tr.counts["duplicate"] == 0
    assert tr.counts["stale"] == 0
    assert tr.counts["reordered"] == 0
    assert elapsed < 20.0, f"1e6 observations took {elapsed:.1f} s (module performance problem)"


@pytest.mark.parametrize("start", [0, 1000, MOD - 600])
def test_d11_k_adjacent_swaps_give_exactly_k_reordered(sq, start):
    """D11: swap k disjoint adjacent pairs inside the window of an otherwise in-order
    stream. In each swapped pair the later-sent packet arrives first (status 'new', skipping
    one packet: gap 1), then the earlier-sent one arrives behind the highest and is counted
    'reordered' (accept True). Exactly k reordered, 0 duplicates, 0 stale, total gap k,
    n - k 'new'. Every unwrapped value equals start + (send index). The case start =
    2^32 - 600 puts a swap across the wrap (stream of 1000 crosses it).
    """
    n, k = 1000, 40
    rng = np.random.default_rng(np.random.SeedSequence(7))
    slots = rng.choice(np.arange(1, n // 2), size=k, replace=False)  # slot j swaps (2j, 2j+1)
    order = list(range(n))
    for j in slots.tolist():
        order[2 * j], order[2 * j + 1] = order[2 * j + 1], order[2 * j]
    tr = sq.SequenceTracker()  # default window 64
    statuses = []
    total_gap = 0
    for idx in order:
        o = tr.observe("s", (start + idx) % MOD)
        statuses.append(o.status)
        total_gap += o.gap
        assert o.unwrapped == start + idx
        assert o.accept
    assert statuses.count("reordered") == k
    assert statuses.count("new") == n - k
    assert statuses.count("duplicate") == 0
    assert statuses.count("stale") == 0
    assert total_gap == k
    assert tr.counts["reordered"] == k
    assert tr.counts["new"] == n - k
    assert tr.counts["duplicate"] == 0
    assert tr.counts["stale"] == 0
    assert tr.counts["gap_packets"] == k


def test_d11_swap_straddling_the_wrap_explicitly(sq):
    """D11: arrivals 2^32-2, 0, 2^32-1, 1. Seq 0 skips 2^32-1 (new, gap 1), then 2^32-1
    arrives behind the highest (reordered), then 1 is new. Hand-derived."""
    tr = sq.SequenceTracker()
    o1 = tr.observe("a", MOD - 2)
    o2 = tr.observe("a", 0)
    o3 = tr.observe("a", MOD - 1)
    o4 = tr.observe("a", 1)
    assert [o.status for o in (o1, o2, o3, o4)] == ["new", "new", "reordered", "new"]
    assert o2.gap == 1 and o3.gap == 0 and o4.gap == 0
    assert [o.unwrapped for o in (o1, o2, o3, o4)] == [MOD - 2, MOD, MOD - 1, MOD + 1]
    assert [o.accept for o in (o1, o2, o3, o4)] == [True, True, True, True]


def test_d11_forward_jump_reports_gap(sq):
    """Contract: 'gap' for a new packet is the number of packets skipped since the previous
    highest. 10 then 15 skips 11..14, so gap = 4 (hand count). Consecutive gives 0."""
    tr = sq.SequenceTracker()
    assert tr.observe("a", 10).gap == 0
    o = tr.observe("a", 15)
    assert (o.status, o.gap, o.unwrapped) == ("new", 4, 15)
    assert tr.observe("a", 16).gap == 0
    assert tr.counts["gap_packets"] == 4


def test_d11_repeated_packet_inside_window_is_duplicate(sq):
    """Contract: a seen packet inside the window is 'duplicate' with accept False.
    Covers repeating a middle packet, the highest packet, and a packet that itself
    arrived as 'reordered'."""
    tr = sq.SequenceTracker()
    for s in (10, 11, 12):
        assert tr.observe("a", s).status == "new"
    mid = tr.observe("a", 11)
    assert (mid.status, mid.accept, mid.gap) == ("duplicate", False, 0)
    top = tr.observe("a", 12)
    assert (top.status, top.accept) == ("duplicate", False)
    assert tr.observe("a", 15).status == "new"
    assert tr.observe("a", 13).status == "reordered"
    again = tr.observe("a", 13)
    assert (again.status, again.accept) == ("duplicate", False)
    assert tr.counts["duplicate"] == 3
    assert tr.counts["reordered"] == 1


def test_d11_immediate_repeat_of_first_packet_is_duplicate(sq):
    tr = sq.SequenceTracker()
    assert tr.observe("a", 99).status == "new"
    o = tr.observe("a", 99)
    assert (o.status, o.accept) == ("duplicate", False)


def test_d11_duplicate_across_the_wrap(sq):
    """D11: after 2^32-1 then 0, repeating 2^32-1 is one behind the highest and seen:
    'duplicate'."""
    tr = sq.SequenceTracker()
    tr.observe("a", MOD - 1)
    tr.observe("a", 0)
    o = tr.observe("a", MOD - 1)
    assert (o.status, o.accept) == ("duplicate", False)


@pytest.mark.parametrize("window", [16, 64])
def test_d11_packet_a_window_or_more_behind_is_stale(sq, window):
    """Contract: 'stale' means behind the highest by >= window. Highest = 1000.
    Behind by window - 1 (unseen) -> 'reordered' (accept True); behind by exactly window
    -> 'stale' (accept False); far behind (unseen) -> 'stale'. A packet that WAS seen but
    is now window or more behind is also 'stale' (not 'duplicate')."""
    tr = sq.SequenceTracker(window=window)
    for s in range(990, 1001):  # 990..1000 in order, highest = 1000
        tr.observe("a", s)
    edge_ok = tr.observe("a", 1000 - (window - 1))
    assert (edge_ok.status, edge_ok.accept) == ("reordered", True)
    edge_stale = tr.observe("a", 1000 - window)
    assert (edge_stale.status, edge_stale.accept, edge_stale.gap) == ("stale", False, 0)
    far = tr.observe("a", 10)
    assert (far.status, far.accept) == ("stale", False)
    seen_old = tr.observe("a", 995)  # seen, behind by 5 (inside both windows) -> duplicate
    assert seen_old.status == "duplicate"
    for s in range(1001, 1001 + window + 5):
        tr.observe("a", s)
    seen_now_stale = tr.observe("a", 995)  # seen, now behind by >= window
    assert (seen_now_stale.status, seen_now_stale.accept) == ("stale", False)
    assert tr.counts["stale"] == 3


def test_d11_stale_does_not_move_highest(sq):
    """Contract: a stale or duplicate arrival must not change the highest-seen state, so the
    next in-order packet is still 'new' with gap 0."""
    tr = sq.SequenceTracker(window=8)
    for s in range(0, 100):
        tr.observe("a", s)
    assert tr.observe("a", 5).status == "stale"
    assert tr.observe("a", 99).status == "duplicate"
    o = tr.observe("a", 100)
    assert (o.status, o.gap, o.unwrapped) == ("new", 0, 100)


def test_d11_senders_are_independent(sq):
    """Contract: state is per sender. The same seq from two senders is 'new' for each, and
    a repeat is a duplicate only for the sender that already sent it."""
    tr = sq.SequenceTracker()
    assert tr.observe("a", 5).status == "new"
    assert tr.observe("b", 5).status == "new"
    assert tr.observe("a", 5).status == "duplicate"
    assert tr.observe("b", 6).status == "new"
    assert tr.observe("a", 6).status == "new"


def test_d11_counts_keys_present(sq):
    """Contract: counts has the keys new, duplicate, reordered, stale, gap_packets."""
    tr = sq.SequenceTracker()
    for key in ("new", "duplicate", "reordered", "stale", "gap_packets"):
        assert key in tr.counts


# ---------------------------------------------------------------------------
# D11: ReplayGuard
# ---------------------------------------------------------------------------


def test_d11_replay_guard_fresh_then_replay(rp):
    """A frame accepted once ('fresh') and presented again is rejected with reason 'replay',
    both for a middle frame and for the latest one."""
    g = rp.ReplayGuard()
    assert tuple(g.check("a", "s1", 1)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 2)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 3)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 1)) == (False, "replay")
    assert tuple(g.check("a", "s1", 3)) == (False, "replay")
    assert tuple(g.check("a", "s1", 4)) == (True, "fresh")


def test_d11_replay_guard_replay_across_wrap(rp):
    """Replay detection also works over the 2^32 wrap: 2^32-1, 0 fresh; 2^32-1 again replay."""
    g = rp.ReplayGuard()
    assert tuple(g.check("a", "s1", MOD - 1)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 0)) == (True, "fresh")
    assert tuple(g.check("a", "s1", MOD - 1)) == (False, "replay")
    assert tuple(g.check("a", "s1", 0)) == (False, "replay")


def test_d11_replay_guard_reordered_fresh(rp):
    """An unseen frame behind the highest but inside the window is accepted as
    'reordered_fresh'; presenting it again is a 'replay'."""
    g = rp.ReplayGuard(window=64)
    assert tuple(g.check("a", "s1", 10)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 8)) == (True, "reordered_fresh")
    assert tuple(g.check("a", "s1", 8)) == (False, "replay")


def test_d11_replay_guard_stale(rp):
    """An unseen frame window or more behind the highest is rejected as 'stale'.
    Window 8, highest 100: seq 50 is 50 behind (>= 8), seq 92 is exactly 8 behind
    (>= 8) so stale, seq 93 is 7 behind and unseen so 'reordered_fresh'."""
    g = rp.ReplayGuard(window=8)
    assert tuple(g.check("a", "s1", 100)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 50)) == (False, "stale")
    assert tuple(g.check("a", "s1", 92)) == (False, "stale")
    assert tuple(g.check("a", "s1", 93)) == (True, "reordered_fresh")


def test_d11_replay_guard_new_session_resets_and_old_session_rejected(rp):
    """A new session id for a sender resets its window (even to lower sequence numbers,
    accepted 'fresh'); frames from the previously seen, superseded session are rejected as
    'old_session'; later sessions supersede earlier ones; other senders are unaffected."""
    g = rp.ReplayGuard()
    for s in range(1, 51):
        assert tuple(g.check("a", "s1", s)) == (True, "fresh")
    assert tuple(g.check("a", "s2", 1)) == (True, "fresh")  # lower seq, new session
    assert tuple(g.check("a", "s2", 1)) == (False, "replay")
    assert tuple(g.check("a", "s1", 51)) == (False, "old_session")
    assert tuple(g.check("a", "s1", 3)) == (False, "old_session")
    assert tuple(g.check("a", "s2", 2)) == (True, "fresh")
    assert tuple(g.check("a", "s3", 1)) == (True, "fresh")
    assert tuple(g.check("a", "s2", 3)) == (False, "old_session")
    assert tuple(g.check("a", "s1", 52)) == (False, "old_session")
    assert tuple(g.check("b", "s1", 1)) == (True, "fresh")
    assert tuple(g.check("b", "s1", 2)) == (True, "fresh")


def test_d11_replay_guard_too_old(rp):
    """With max_age_s = 1.0, a frame whose age (now_s - sent_s) exceeds the limit is
    rejected as 'too_old'; a younger one is 'fresh'. Ages: 0.5 s ok, 1.5 s too old.
    Without max_age_s (None) the same age is accepted."""
    g = rp.ReplayGuard(max_age_s=1.0)
    assert tuple(g.check("a", "s1", 1, sent_s=10.0, now_s=10.5)) == (True, "fresh")
    assert tuple(g.check("a", "s1", 2, sent_s=10.0, now_s=11.5)) == (False, "too_old")
    assert tuple(g.check("a", "s1", 3, sent_s=20.0, now_s=20.2)) == (True, "fresh")
    lax = rp.ReplayGuard(max_age_s=None)
    assert tuple(lax.check("a", "s1", 1, sent_s=10.0, now_s=500.0)) == (True, "fresh")
