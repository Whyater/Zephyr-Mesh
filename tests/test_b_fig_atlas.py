"""Build the atlas figures from the pilot output and check they are stable."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import make_s6_atlas_figure

PILOT = Path("/tmp/runb-pilots/atlas")
EXPECTED = ("atlas_overview.png", "atlas_which_method.png")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    if not PILOT.is_dir():
        pytest.skip(f"pilot output {PILOT} is missing; run the atlas pilot first")
    target = tmp_path / "run"
    shutil.copytree(PILOT, target)
    return target


def test_atlas_figures_build_and_are_deterministic(run_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for out in (first, second):
        out.mkdir()
        shutil.copy(run_dir / "manifest.json", out / "manifest.json")

    written_first = make_s6_atlas_figure.build(run_dir, first)
    written_second = make_s6_atlas_figure.build(run_dir, second)

    names = [p.name for p in written_first]
    for name in EXPECTED:
        assert name in names
    assert names == [p.name for p in written_second]
    for a, b in zip(written_first, written_second):
        assert a.exists() and a.stat().st_size > 20_000, a
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs between two builds"

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for path in written_first:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["outputs"][path.name] == digest
    # Existing manifest entries are kept.
    original = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key, value in original["outputs"].items():
        assert manifest["outputs"][key] == value


def _entry(value, censored=None, fails=False):
    return {"loss_model": "independent", "loss_probability": 0.1,
            "max_tolerated_delay_s": {"value": value, "censored": censored,
                                      "fails_at_zero": fails, "interval_s": [0, 0]}}


def test_clear_winner_needs_strictly_larger_delay() -> None:
    arms = {"a": _entry(0.1), "b": _entry(0.1), "c": _entry(None, ">0.3"),
            "d": _entry(None, None, True)}
    kind, top, entry = make_s6_atlas_figure.classify_cell(arms)
    assert (kind, top) == ("winner", ["c"])
    assert entry is arms["c"]
    del arms["c"]
    kind, top, _ = make_s6_atlas_figure.classify_cell(arms)
    assert (kind, top) == ("tie", ["a", "b"])  # equal delay is a tie, never an arm pick


def test_ties_listed_in_arm_order_and_none_when_all_fail() -> None:
    arms = {"adaptive": _entry(None, ">0.3"), "naive": _entry(None, ">0.3"),
            "imm": _entry(0.2)}
    kind, top, _ = make_s6_atlas_figure.classify_cell(arms)
    assert kind == "tie" and top == ["naive", "adaptive"]
    assert make_s6_atlas_figure.classify_cell({"d": _entry(None, None, True)}) == (
        "none", [], None)


def test_tie_text_names_arms_or_counts_them() -> None:
    assert make_s6_atlas_figure.tie_text(["naive", "adaptive"]) == "Tie: Naive, Adapt."
    assert make_s6_atlas_figure.tie_text(["naive", "recovery", "imm", "ct"]) == "Tie (4 arms)"


def test_count_line_is_computed_from_cells() -> None:
    cells = [("tie", ["a", "b"], None)] * 3 + [("winner", ["a"], None)] * 2 + [
        ("none", [], None)]
    counts = make_s6_atlas_figure.cell_counts(cells)
    assert counts == {"cells": 6, "none": 1, "winner": 2, "tie": 3, "survive": 5}
    assert make_s6_atlas_figure.count_line(counts) == (
        "Ties: 3 of 5 cells where some arm survives; clear winner: 2; "
        "no arm tolerates 0 ms: 1.")


def test_footnotes_and_context_lines(run_dir: Path) -> None:
    atlas = make_s6_atlas_figure.load_atlas(run_dir / "atlas.json")
    summary = make_s6_atlas_figure.load_summary(run_dir / "summary.json")
    for stratum in atlas["strata"]:
        text = make_s6_atlas_figure.footnote(atlas, stratum, summary)
        flat = " ".join(text.split())
        assert "median p95 with 95" not in flat
        assert "largest tested delay before the first failing delay" in flat
        assert "rule proposed, not yet adopted" in flat
        assert "\u2014" not in text
    central = " ".join(make_s6_atlas_figure.footnote(atlas, "central", summary).split())
    assert "assumes" not in central and "burst" not in central.lower().replace("burst loss", "")
    for stratum, info in atlas["strata"].items():
        text = " ".join(make_s6_atlas_figure.footnote(atlas, stratum, summary).split())
        if stratum.startswith("noise"):
            assert "still assumes 5 cm measurement noise" in text
        if stratum.startswith("rate"):
            assert "0.5 s mean bursts at every measurement rate" in text
        if info["jitter_rule"] != "none":
            assert info["jitter_rule"].replace("_s", "") in text


def test_which_method_counts_match_the_data(run_dir: Path) -> None:
    atlas = make_s6_atlas_figure.load_atlas(run_dir / "atlas.json")
    groups = make_s6_atlas_figure.group_entries(atlas)["central"]
    kinds = [make_s6_atlas_figure.classify_cell(arms)[0] for arms in groups.values()]
    counts = make_s6_atlas_figure.cell_counts(
        [make_s6_atlas_figure.classify_cell(arms) for arms in groups.values()])
    assert counts["tie"] == kinds.count("tie") and counts["none"] == kinds.count("none")
    assert counts["tie"] + counts["winner"] + counts["none"] == len(groups)
