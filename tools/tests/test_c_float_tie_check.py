"""Checks for the exact-integer loss comparison enumeration (tools/c_float_tie_check.py)."""
from __future__ import annotations

from tools import c_float_tie_check as ft


def test_runner_and_default_configs_unchanged():
    assert ft.enumerate_config(20, 0.3, 3)["old_vs_new"] == []
    assert ft.enumerate_config(10, 0.5, 3)["old_vs_new"] == []


def test_threshold_0_2_disagreements_match_hand_list():
    # hand list: 1/5, 2/10, 3/15, 4/20 trigger ties; 0.1 clear ties at n=10 (1 lost) and n=20 (2 lost)
    expected = {("trigger", 5, 1), ("trigger", 10, 2), ("trigger", 15, 3), ("trigger", 20, 4),
                ("clear", 10, 1), ("clear", 20, 2)}
    got = ft.enumerate_config(20, 0.2, 3)["old_vs_new"]
    assert set(got) == expected and len(got) == len(expected)


def test_new_matches_exact_everywhere():
    rep = ft.run()
    assert all(r["new_vs_exact"] == [] for r in rep["checked"] + rep["test_configs"])
    assert rep["sanity_ok"] and rep["ok"]


def test_grep_finds_runner_construction():
    where = [c["where"] for c in ft.find_constructions() if c["config"] == (20, 0.3, 3)]
    assert any(w.startswith("tools/c_exp016_heartbeat.py:") for w in where)


def test_unknown_or_keyword_construction_fails_the_check(monkeypatch):
    # A construction the scan cannot match to a checked configuration must fail the check, not pass silently.
    for entry in ({"where": "x.py:1", "config": (20, 0.2, 3)}, {"where": "x.py:2", "config": None}):
        monkeypatch.setattr(ft, "find_constructions", lambda e=entry: [e])
        assert ft.run()["ok"] is False
