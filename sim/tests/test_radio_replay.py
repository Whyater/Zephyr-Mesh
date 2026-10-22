from sim.radio.replay import ReplayGuard


def test_replay_and_reordering_reasons():
    guard = ReplayGuard(window=8)
    assert guard.check("a", "one", 10) == (True, "fresh")
    assert guard.check("a", "one", 12) == (True, "fresh")
    assert guard.check("a", "one", 11) == (True, "reordered_fresh")
    assert guard.check("a", "one", 11) == (False, "replay")
    assert guard.check("a", "one", 10) == (False, "replay")
    assert guard.check("a", "one", 1) == (False, "stale")


def test_session_reset_and_superseded_sessions():
    guard = ReplayGuard(window=8)
    assert guard.check("a", "s1", 100) == (True, "fresh")
    assert guard.check("a", "s2", 0) == (True, "fresh")
    assert guard.check("a", "s2", 1) == (True, "fresh")
    assert guard.check("a", "s1", 101) == (False, "old_session")
    assert guard.check("a", "s3", 0) == (True, "fresh")
    assert guard.check("a", "s2", 2) == (False, "old_session")


def test_too_old_freshness_and_missing_times():
    guard = ReplayGuard(window=8, max_age_s=2.0)
    assert guard.check("a", "s", 1, sent_s=8.0, now_s=10.0) == (True, "fresh")
    assert guard.check("a", "s", 2, sent_s=7.9, now_s=10.0) == (False, "too_old")
    assert guard.check("a", "s", 2, sent_s=None, now_s=10.0) == (True, "fresh")
    assert guard.check("a", "s", 3, sent_s=10.5, now_s=10.0) == (True, "fresh")

