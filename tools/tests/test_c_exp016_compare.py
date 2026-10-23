"""Checks for the EXP-016 old-vs-new detector comparison script.

Each test copies the registered sim/runs/c5-heartbeat folder, breaks one thing
on purpose, and confirms the comparison flags exactly that kind of change.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from tools import c_exp016_compare as cmp

REGISTERED = Path(__file__).resolve().parents[2] / "sim" / "runs" / "c5-heartbeat"


@pytest.fixture()
def pair(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    shutil.copytree(REGISTERED, old)
    shutil.copytree(REGISTERED, new)
    return old, new


def _failed(report):
    return [c["check"] for c in report["checks"] if not c["ok"]]


def _edit_row(path: Path, index: int, **values):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols, rows = reader.fieldnames, list(reader)
    rows[index].update({k: str(v) for k, v in values.items()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def test_identical_folders_pass(pair):
    report = cmp.compare(*pair)
    assert report["ok"], _failed(report)
    assert all(r["changed_columns"] == [] for r in report["rows"])
    assert len(report["rows"]) == 18


def test_allowed_text_fields_may_change(pair):
    old, new = pair
    doc = json.loads((new / "summary.json").read_text())
    doc["interference_method"] += "; extra words"
    (new / "summary.json").write_text(json.dumps(doc))
    assert cmp.compare(old, new)["ok"]


def test_heartbeat_byte_change_fails(pair):
    old, new = pair
    with open(new / "false_alarms.csv", "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert _failed(cmp.compare(old, new)) == ["false_alarms.csv byte identical"]


def test_number_change_in_summary_fails(pair):
    old, new = pair
    doc = json.loads((new / "summary.json").read_text())
    doc["snr_at_half_loss_db"] += 1e-9
    (new / "summary.json").write_text(json.dumps(doc))
    assert len(_failed(cmp.compare(old, new))) == 1


def test_more_misses_and_false_triggers_fail(pair):
    old, new = pair
    # Row 1 (20 dB margin, 1 s, 20 dB rise) caught 17 of 68 episodes.
    _edit_row(new / "interference.csv", 1, missed_episodes=52, false_triggers_per_hour=0.1)
    assert sorted(_failed(cmp.compare(old, new))) == ["row 1: missed episodes did not grow", "row 1: no false triggers"]


def test_lower_fraction_detected_fails(pair):
    old, new = pair
    _edit_row(new / "interference.csv", 1, fraction_detected=0.2)
    assert _failed(cmp.compare(old, new)) == ["row 1: fraction detected did not drop"]


def test_false_triggers_in_the_old_run_fail(pair):
    old, new = pair
    _edit_row(old / "interference.csv", 2, false_triggers_per_hour=0.1)
    assert _failed(cmp.compare(old, new)) == ["row 2: no false triggers"]


def test_never_triggered_cell_that_triggers_fails(pair):
    old, new = pair
    # Row 0 (10 dB rise) missed all 80 episodes.
    _edit_row(new / "interference.csv", 0, missed_episodes=79, mean_trigger_s=0.5)
    assert _failed(cmp.compare(old, new)) == ["row 0: never-triggered cell stays never-triggered"]


def test_episode_count_change_fails(pair):
    old, new = pair
    _edit_row(new / "interference.csv", 4, n_episodes=69)
    assert "row 4: cell keys and episode count unchanged" in _failed(cmp.compare(old, new))


def test_json_diff_paths_reports_nested_keys():
    assert cmp.json_diff_paths({"a": {"b": [1, 2]}, "c": 1}, {"a": {"b": [1, 3]}, "c": 1}) == ["a.b[1]"]
    assert cmp.json_diff_paths({"a": 1}, {"a": 1.0}) == ["a"]
    assert cmp.json_diff_paths({"a": 1}, {"b": 1}) == ["a", "b"]
