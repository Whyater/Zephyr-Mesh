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


def test_winner_prefers_larger_delay_then_smaller_p95() -> None:
    def entry(value, censored=None, fails=False):
        return {"loss_model": "independent", "loss_probability": 0.1,
                "max_tolerated_delay_s": {"value": value, "censored": censored,
                                          "fails_at_zero": fails, "interval_s": [0, 0]}}

    arms = {"a": entry(0.1), "b": entry(0.1), "c": entry(None, ">0.3"), "d": entry(None, None, True)}
    table = {}
    assert make_s6_atlas_figure.pick_winner(arms, table, "s", "v", None)[0] == "c"
    del arms["c"]
    table[("s", "v", "independent", 0.1, 0.1, "a")] = 0.20
    table[("s", "v", "independent", 0.1, 0.1, "b")] = 0.15
    assert make_s6_atlas_figure.pick_winner(arms, table, "s", "v", None)[0] == "b"
    assert make_s6_atlas_figure.pick_winner({"d": entry(None, None, True)}, table, "s", "v",
                                            None) == (None, None)
