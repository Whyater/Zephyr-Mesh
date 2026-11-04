"""Tests for tools/check_public_numbers.py.

The first tests build a fake repository (a README, a run registry and run
outputs) so each state is shown on a case where the right answer is known by
hand.  The last test runs the real mapping on the real public pages: a README
or docs edit that adds a number, or changes one away from its run output,
fails it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import check_public_numbers as cpn

REPO_ROOT = Path(__file__).resolve().parents[2]

README = """# Fake

The basic tracker fails at 200 ms of delay, error 0.36 m.
The edge is 0.34 s by hand.
Twelve drones keep the channel about two thirds busy.
New line with 7.5 cm.
Version 1.2 and step 3.
"""

SUMMARY = {
    "cells": [
        {"delay_s": 0.1, "label": "passes", "p95_m": 0.161},
        {"delay_s": 0.2, "label": "fails", "p95_m": 0.2614},
        {"delay_s": 0.3, "label": "fails", "p95_m": 0.3591},
    ],
    "edge_s": 0.338,
    "windows": [1.0, 0.73, 0.41],
}

CELLS_CSV = "n,mode,busy\n12,r5,0.6619\n12,r0,0.88\n4,r5,0.2\n"


def make_repo(tmp_path: Path, readme: str = README) -> Path:
    (tmp_path / "tools").mkdir()
    run = tmp_path / "sim" / "runs" / "fake"
    run.mkdir(parents=True)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    (run / "cells.csv").write_text(CELLS_CSV, encoding="utf-8")
    (run / "notes.json").write_text(json.dumps({"x": 7.5}), encoding="utf-8")
    registry = [{"run_dir": "sim/runs/fake", "command": "true",
                 "outputs": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                             for name in ("summary.json", "cells.csv")}}]
    (tmp_path / "tools" / "reproduce_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    return tmp_path


SRC = "sim/runs/fake/summary.json"


def reregister(root: Path) -> None:
    """Record new hashes after a test rewrites a run output on purpose."""
    reg_path = root / "tools" / "reproduce_registry.json"
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    run = root / "sim" / "runs" / "fake"
    registry[0]["outputs"] = {n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in registry[0]["outputs"]}
    reg_path.write_text(json.dumps(registry), encoding="utf-8")


def base_mapping() -> dict:
    return {
        "files": ["README.md"],
        "quotes": [
            {"file": "README.md", "quote": "fails at 200 ms of delay, error 0.36 m", "numbers": [
                {"text": "200", "kind": "performance", "source": {
                    "path": SRC, "select": ["cells", {"label": "fails"}],
                    "op": "min", "field": ["delay_s"], "scale": 1000}},
                {"text": "0.36", "kind": "performance", "source": {
                    "path": SRC, "select": ["cells", {"delay_s": 0.3}, "p95_m"]}},
            ]},
            {"file": "README.md", "quote": "The edge is 0.34 s", "numbers": [
                {"text": "0.34", "kind": "performance", "source": {"path": SRC, "select": ["edge_s"]}}]},
            {"file": "README.md", "quote": "Twelve drones keep the channel about two thirds busy",
             "numbers": [
                 {"text": "Twelve", "kind": "setting"},
                 {"text": "two thirds", "kind": "performance", "tolerance": 0.02,
                  "rounding_note": "the text says about", "source": {
                      "path": "sim/runs/fake/cells.csv",
                      "select": [{"n": 12, "mode": "r5"}, "busy"]}}]},
            {"file": "README.md", "quote": "New line with 7.5 cm", "numbers": [
                {"text": "7.5", "kind": "performance"}]},
            {"file": "README.md", "quote": "Version 1.2 and step 3", "kind": "not_a_claim",
             "numbers": ["1.2", "3"]},
        ],
    }


def states(results):
    return {(r.text): r.state for r in results}


def test_tokens_skip_names_and_find_number_words():
    toks = cpn.find_tokens("x", "s6-recovery x86_64 v0.2.4 c-exp017-r2 drop 0.25 m, Twelve, two thirds 1e-9")
    assert [t.text for t in toks] == ["0.25", "Twelve", "two thirds", "1e-9"]


def test_fake_repo_states(tmp_path):
    root = make_repo(tmp_path)
    res = states(cpn.check(base_mapping(), root))
    assert res == {"200": "matched", "0.36": "matched", "0.34": "matched", "Twelve": "not_checked",
                   "two thirds": "matched", "7.5": "unmatched", "1.2": "not_checked", "3": "not_checked"}


def test_rounding_uses_the_shown_precision(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    edge = m["quotes"][1]["numbers"][0]
    # 0.338 shown as 0.34 is within half of the last shown digit.
    assert states(cpn.check(m, root))["0.34"] == "matched"
    # 0.333 shown as 0.34 is not (0.007 > 0.005).
    SUMMARY_333 = dict(SUMMARY, edge_s=0.333)
    (root / SRC).write_text(json.dumps(SUMMARY_333), encoding="utf-8")
    reregister(root)
    assert states(cpn.check(m, root))["0.34"] == "mismatched"
    # A stated rounding can widen it, but only with a note saying where it is stated.
    edge["tolerance"] = 0.01
    with pytest.raises(cpn.MappingError, match="rounding_note"):
        cpn.check(m, root)
    edge["rounding_note"] = "the text says about"
    assert states(cpn.check(m, root))["0.34"] == "matched"


def test_mismatch_when_the_text_drifts(tmp_path):
    root = make_repo(tmp_path, README.replace("error 0.36 m", "error 0.31 m"))
    m = base_mapping()
    m["quotes"][0]["quote"] = "fails at 200 ms of delay, error 0.31 m"
    m["quotes"][0]["numbers"][1]["text"] = "0.31"
    res = cpn.check(m, root)
    assert states(res)["0.31"] == "mismatched"
    assert any(r.text == "0.31" for r in cpn.failures(res))


def test_a_new_number_with_no_quote_is_unmatched(tmp_path):
    root = make_repo(tmp_path, README + "It now settles in 2.5 s.\n")
    res = cpn.check(base_mapping(), root)
    new = [r for r in res if r.text == "2.5"]
    assert len(new) == 1 and new[0].state == "unmatched" and new[0].kind == "unknown"
    assert new[0] in cpn.failures(res)


def test_unregistered_source_is_unmatched(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    m["quotes"][3]["numbers"][0]["source"] = {"path": "sim/runs/fake/notes.json", "select": ["x"], "scale": 1}
    r = [x for x in cpn.check(m, root) if x.text == "7.5"][0]
    assert r.state == "unmatched" and "not a registered run output" in r.detail


def test_a_source_edited_after_registration_is_unmatched(tmp_path):
    root = make_repo(tmp_path)
    # Edit the run output by hand so its bytes no longer match the registry.
    (root / SRC).write_text(json.dumps(dict(SUMMARY, edge_s=0.3381)), encoding="utf-8")
    r = [x for x in cpn.check(base_mapping(), root) if x.text == "0.34"][0]
    assert r.state == "unmatched" and "sha256" in r.detail


def test_known_drift_passes_and_a_fixed_item_must_leave_the_list(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    m["quotes"][3]["numbers"][0]["known"] = {"status": "unmatched", "reason": "no run yet"}
    res = cpn.check(m, root)
    assert states(res)["7.5"] == "known"
    assert cpn.failures(res) == []
    # Once it has a matching source, the stale known entry fails.
    m["quotes"][3]["numbers"][0]["source"] = {"path": "sim/runs/fake/cells.csv",
                                              "select": [{"n": 12, "mode": "r0"}, "busy"], "scale": 1}
    m["quotes"][3]["numbers"][0]["value"] = 0.88
    assert states(cpn.check(m, root))["7.5"] == "stale_known"


def test_known_needs_a_reason(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    m["quotes"][3]["numbers"][0]["known"] = {"status": "unmatched"}
    with pytest.raises(cpn.MappingError, match="reason"):
        cpn.check(m, root)


def test_every_source_must_match(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    item = m["quotes"][0]["numbers"][1]
    item["sources"] = [item.pop("source"), {"path": SRC, "select": ["cells", {"delay_s": 0.2}, "p95_m"]}]
    assert states(cpn.check(m, root))["0.36"] == "mismatched"


def test_selector_index_count_and_max(tmp_path):
    root = make_repo(tmp_path)
    cache: dict = {}
    assert cpn.evaluate_source({"path": SRC, "select": ["windows", -1]}, root, cache) == 0.41
    assert cpn.evaluate_source({"path": SRC, "select": ["cells", {"label": "fails"}], "op": "count"},
                               root, cache) == 2
    assert cpn.evaluate_source({"path": SRC, "select": ["cells"], "op": "max", "field": ["p95_m"]},
                               root, cache) == 0.3591
    with pytest.raises(cpn.MappingError, match="exactly one row"):
        cpn.evaluate_source({"path": SRC, "select": ["cells", {"label": "fails"}, "p95_m"]}, root, cache)


def test_stale_quotes_are_errors(tmp_path):
    root = make_repo(tmp_path)
    m = base_mapping()
    m["quotes"][1]["quote"] = "The edge is 0.35 s"
    res = cpn.check(m, root)
    assert [r.state for r in res if r.quote == "The edge is 0.35 s"] == ["stale_quote"]
    # The real line's number is then not covered by any quote.
    assert states(res)["0.34"] == "unmatched"
    m = base_mapping()
    m["quotes"][4]["numbers"] = ["1.2"]
    with pytest.raises(cpn.MappingError, match="do not match"):
        cpn.check(m, root)


def test_real_public_numbers_match_their_runs():
    mapping = json.loads((REPO_ROOT / "tools" / "public_number_sources.json").read_text(encoding="utf-8"))
    results = cpn.check(mapping, REPO_ROOT)
    bad = cpn.failures(results)
    assert not bad, cpn.format_report(bad)
    # Every performance number either matches a registered run output or is listed as known.
    for r in results:
        if r.kind == "performance":
            assert r.state in {"matched", "known"}, (r.file, r.line, r.text, r.state)
