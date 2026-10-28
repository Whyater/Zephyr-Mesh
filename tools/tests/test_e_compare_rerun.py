"""Checks for the rerun comparison tool."""
from __future__ import annotations

import json

from tools.e_compare_rerun import compare, json_differences, main


def _write(folder, files):
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


def test_json_differences_finds_added_removed_and_changed_paths():
    old = {"a": 1, "b": [1, 2], "c": {"x": "same", "y": 2}}
    new = {"a": 1, "b": [1, 3, 4], "c": {"x": "same"}, "d": True}
    diff = json_differences(old, new)
    assert diff["added"] == ["/b[2]", "/d"]
    assert diff["removed"] == ["/c/y"]
    assert diff["changed"] == ["/b[1]"]
    # An int that became a float of the same value still counts as changed.
    assert json_differences({"v": 1}, {"v": 1.0})["changed"] == ["/v"]


def test_compare_passes_only_when_expectations_hold(tmp_path):
    registered, rerun = tmp_path / "reg", tmp_path / "new"
    _write(registered, {"t.csv": b"a,b\n1,2\n", "s.json": {"x": 1}, "f.png": b"old"})
    _write(rerun, {"t.csv": b"a,b\n1,2\n", "s.json": {"x": 1, "extra": [1]}, "f.png": b"new"})
    result = compare(registered, rerun, identical=["t.csv"], additive=["s.json"], changed=["f.png"])
    assert result["ok"] is True
    summary = next(item for item in result["files"] if item["file"] == "s.json")
    assert summary["added"] == ["/extra"]

    # A file expected to change that came out identical fails.
    _write(rerun, {"f.png": b"old"})
    result = compare(registered, rerun, identical=[], additive=[], changed=["f.png"])
    assert result["ok"] is False

    _write(rerun, {"t.csv": b"a,b\n1,3\n", "s.json": {"x": 2}})
    result = compare(registered, rerun, identical=["t.csv"], additive=["s.json"], changed=[])
    assert result["ok"] is False
    assert [item["ok"] for item in result["files"]] == [False, False]


def test_missing_file_fails_and_main_writes_report(tmp_path):
    registered, rerun = tmp_path / "reg", tmp_path / "new"
    _write(registered, {"t.csv": b"1\n"})
    rerun.mkdir()
    report = tmp_path / "report.json"
    assert main([str(registered), str(rerun), "--identical", "t.csv", "--report", str(report)]) == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert data["files"][0]["missing"] == [str(rerun / "t.csv")]
