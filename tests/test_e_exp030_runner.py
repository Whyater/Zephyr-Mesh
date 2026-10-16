from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.e_exp030_tag import (PREDICTION_CORRECTION_2, PREDICTION_SOURCE, PREDICTION_TEXT,
                                build_jobs, main)
from tests.platform_compare import same_tree
from tests.reference_e.hand_checks import tail_follow_time_s


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 4*5*5*4*50 + calm 4*2*5*50 + followers 4*2*3*2*50 + physics 4*2*2*20.
    assert len(jobs) == 20000 + 2000 + 2400 + 320
    assert jobs[0]["labels"]["target_class"] == "straight"
    assert jobs[0]["labels"]["preset"] == "clean"
    assert any(j["labels"]["slice"] == "calm" and j["labels"]["policy_name"] == "calm" for j in jobs)
    assert any(j["labels"]["slice"] == "physics" and j["scenario"]["adapter"] == "physics" for j in jobs)
    assert len(build_jobs(pilot=True)) == 26


@pytest.fixture(scope="module")
def pilot_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp030")
    first, second = base / "first", base / "second"
    main(["--output", str(first), "--pilot", "--processes", "1"])
    main(["--output", str(second), "--pilot", "--processes", "1"])
    return first, second


def test_pilot_outputs_are_deterministic(pilot_dirs):
    first, second = pilot_dirs
    # The pilot has no calm slice, so the default versus calm figure is not drawn.
    expected = {"tag.csv", "summary.json", "tag_success_delay.png", "manifest.json"}
    assert {p.name for p in first.iterdir()} == expected
    assert (first / "tag.csv").read_bytes() == (second / "tag.csv").read_bytes()
    left = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert same_tree(left, right)
    assert left["pilot"] is True
    assert left["figures_not_drawn"][0]["file"] == "tag_success_default_calm.png"
    assert left["prediction"] == PREDICTION_TEXT
    assert left["prediction_source"] == PREDICTION_SOURCE
    assert left["prediction_correction_2"] == PREDICTION_CORRECTION_2
    assert "prediction:" not in PREDICTION_TEXT[:40]
    assert left["grid_counts"] == {"main": 20000, "calm": 2000, "followers": 2400, "physics": 320}
    assert left["missing_metrics"]


def test_hand_check_band_comes_from_reference_function(pilot_dirs):
    # Zone entry (4 - 1) / (1.5 - 0.5) = 3.0 s; the pre-registered entry band is 3.0 to 3.6 s, plus the 2.0 s hold.
    zone_entry = tail_follow_time_s(4.0, 1.0, 1.5, 0.5)
    summary = json.loads((pilot_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    hand = summary["hand_check"]
    assert hand["zone_entry_reference_s"] == zone_entry
    assert hand["predicted_time_to_tag_band_s"] == [zone_entry + 0.0 + 2.0, zone_entry + 0.6 + 2.0]
    median = hand["observed_time_to_tag_s"]["median"]
    assert hand["inside_predicted_band"] == (median is not None and
                                              hand["predicted_time_to_tag_band_s"][0] <= median
                                              <= hand["predicted_time_to_tag_band_s"][1])
