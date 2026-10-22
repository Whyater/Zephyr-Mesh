from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.e_exp030_tag import (PREDICTION_CORRECTION_2, PREDICTION_SOURCE, PREDICTION_TEXT,
                                build_jobs, main)
from sim.tests.platform_compare import same_tree
from sim.tests.reference_e.hand_checks import tail_follow_time_s


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 4*5*5*4*50 + calm 4*2*5*50 + followers 4*2*3*2*2 rules*50 + physics 4*2*2*20.
    assert len(jobs) == 20000 + 2000 + 4800 + 320
    followers = [j for j in jobs if j["labels"]["slice"] == "followers"]
    assert {j["labels"]["separation_rule"] for j in followers} == {"coop", "local"}
    assert all(j["scenario"]["params"]["separation_rule"] == j["labels"]["separation_rule"] for j in followers)
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
    assert left["grid_counts"] == {"main": 20000, "calm": 2000, "followers": 4800, "physics": 320}
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


def _fake_follower_row(rule: str, n_followers: int, *, tagged: bool, seed_index: int) -> dict:
    # Synthetic followers-slice rows with every column the summary reads.
    return {"slice": "followers", "target_class": "jink", "measurement_delay_s": 0.0, "preset": "clean",
            "arm": "recovery", "n_followers": n_followers, "separation_rule": rule, "adapter": "point_mass",
            "policy_name": "default", "seed_index": seed_index, "tagged": tagged,
            "time_to_tag_s": 5.0 if tagged else None, "min_separation_m": 1.0, "separation_violations": 0,
            "holds": 0, "landings": 0, "end_reason": "tagged" if tagged else "timeout",
            "guard_activations": 0, "follower_follower_min_m": 1.0}


def test_prediction_rates_tell_follower_cells_apart():
    from tools.e_exp030_tag import summarize

    rows = []
    for n_followers in (2, 3):
        for rule in ("coop", "local"):
            for seed_index in range(2):
                rows.append(_fake_follower_row(rule, n_followers, tagged=(rule == "coop"), seed_index=seed_index))
    rates = summarize(rows, pilot=False)["prediction_observed_rates"]
    assert len(rates) == 4
    keys = {(r["n_followers"], r["separation_rule"]) for r in rates}
    assert keys == {(2, "coop"), (2, "local"), (3, "coop"), (3, "local")}
    for entry in rates:
        assert entry["tag_success"]["rate"] == (1.0 if entry["separation_rule"] == "coop" else 0.0)


def test_deviations_and_limits_describe_the_rule_switch(pilot_dirs):
    from sim.missions.tag import TagMission
    from tools.e_exp030_tag import summarize

    pilot_summary = json.loads((pilot_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    for summary in (pilot_summary, summarize([], pilot=False)):
        text = " ".join(summary["deviations"])
        assert "D-123" in text
        assert "after the pre-registration" in text
        assert "for comparison only" in text
        assert "changed the velocity" in text
        for line in TagMission.limits:
            assert line in summary["limits"]
