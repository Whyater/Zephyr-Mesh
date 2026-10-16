from __future__ import annotations

import json
from pathlib import Path

from tools.e_exp030_tag import (PREDICTION_CORRECTION_2, PREDICTION_TEXT, build_jobs, main)
from tests.platform_compare import same_tree


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 4*5*5*4*50 + calm 4*2*5*50 + followers 4*2*3*2*50 + physics 4*2*2*20.
    assert len(jobs) == 20000 + 2000 + 2400 + 320
    assert jobs[0]["labels"]["target_class"] == "straight"
    assert jobs[0]["labels"]["preset"] == "clean"
    assert any(j["labels"]["slice"] == "calm" and j["labels"]["policy_name"] == "calm" for j in jobs)
    assert any(j["labels"]["slice"] == "physics" and j["scenario"]["adapter"] == "physics" for j in jobs)
    assert len(build_jobs(pilot=True)) == 26


def test_pilot_outputs_are_deterministic(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    main(["--output", str(first), "--pilot", "--processes", "1"])
    main(["--output", str(second), "--pilot", "--processes", "1"])
    expected = {"tag.csv", "summary.json", "tag_success_delay.png", "tag_success_default_calm.png", "manifest.json"}
    assert {p.name for p in first.iterdir()} == expected
    assert (first / "tag.csv").read_bytes() == (second / "tag.csv").read_bytes()
    left = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert same_tree(left, right)
    assert left["pilot"] is True
    assert left["hand_check"]["observed_median_zone_entry_s"] == "not a metric"
    assert left["prediction"] == PREDICTION_TEXT
    assert left["prediction_correction_2"] == PREDICTION_CORRECTION_2


def test_prediction_constants_match_prereg():
    prereg = Path("/Users/akashi0281/.zm-worktrees/run-e-briefs/prereg.md")
    if not prereg.exists():
        import pytest
        pytest.skip("prereg is outside the repository")
    text = prereg.read_text(encoding="utf-8")
    assert PREDICTION_TEXT in text
    assert PREDICTION_CORRECTION_2 in text
