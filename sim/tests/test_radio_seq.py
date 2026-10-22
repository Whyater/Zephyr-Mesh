from sim.radio.seq import SEQ_MOD, SequenceTracker, seq_diff, seq_next


def test_serial_helpers_and_explicit_wrap():
    assert seq_next(SEQ_MOD - 1) == 0
    assert seq_diff(0, SEQ_MOD - 1) == 1
    tracker = SequenceTracker(window=64)
    values = [SEQ_MOD - 3, SEQ_MOD - 2, SEQ_MOD - 1, 0, 1, 2]
    observations = [tracker.observe("s", value) for value in values]
    assert [o.status for o in observations] == ["new"] * len(values)
    assert [o.unwrapped for o in observations] == list(range(SEQ_MOD - 3, SEQ_MOD + 3))


def test_million_in_order_packets_cross_wrap():
    tracker = SequenceTracker(window=64)
    start = SEQ_MOD - 500_000
    unwrapped = []
    for i in range(1_000_000):
        observation = tracker.observe("sender", (start + i) % SEQ_MOD)
        assert observation.status == "new"
        unwrapped.append(observation.unwrapped)
    assert tracker.counts["duplicate"] == 0
    assert tracker.counts["stale"] == 0
    assert tracker.counts["gap_packets"] == 0
    assert all(b == a + 1 for a, b in zip(unwrapped, unwrapped[1:]))


def test_reordering_duplicates_stale_and_gaps():
    tracker = SequenceTracker(window=8)
    for seq in range(10):
        assert tracker.observe("s", seq).status == "new"
    assert tracker.observe("s", 8).status == "duplicate"
    assert tracker.observe("s", 7).status == "duplicate"
    assert tracker.observe("s", 5).status == "duplicate"
    assert tracker.observe("s", 0).status == "stale"
    gap = tracker.observe("s", 15)
    assert gap.status == "new"
    assert gap.gap == 5
    assert tracker.counts["gap_packets"] == 5
    assert tracker.observe("s", 12).status == "reordered"
    assert tracker.observe("s", 12).status == "duplicate"


def test_adjacent_reordering_inside_window():
    tracker = SequenceTracker(window=32)
    for seq in range(20):
        assert tracker.observe("s", seq).status == "new"
    for first in (20, 22, 24, 26):
        assert tracker.observe("s", first + 1).status == "new"
        assert tracker.observe("s", first).status == "reordered"
    assert tracker.counts["reordered"] == 4
    assert tracker.counts["duplicate"] == 0

