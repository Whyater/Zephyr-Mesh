"""Checks for the EXP-042 hold-in-place comparison runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sim.missions.api import Scenario
from tools import e_exp042_hold as exp
from tools.e_common import seeds_for
from tools.e_exp029_part_b import scenario_for as part_b_scenario
from tools.e_exp031_sweep import _scenario as exp031_scenario

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "sim" / "runs" / "e-exp042-hold"


def test_grid_has_ten_cells_two_arms_and_the_registered_seeds():
    jobs = exp.build_jobs()
    # 6 tag cells x 20 seeds + 4 sweep cells x 20 seeds = 200 per arm, 400 in all.
    assert len(jobs) == 400
    assert sum(j["labels"]["arm"] == "hold" for j in jobs) == 200
    assert sum(j["labels"]["part"] == "tag" for j in jobs) == 240
    cells = {(j["labels"]["arm"], exp.cell_of(j["labels"])) for j in jobs}
    assert cells == {(arm, cell) for arm in exp.ARMS for cell in exp.EXPECTED_CELLS}
    assert len(set(exp.EXPECTED_CELLS)) == 10
    tag_seeds = {j["scenario"]["seed"] for j in jobs if j["labels"]["part"] == "tag"}
    sweep_seeds = {j["scenario"]["seed"] for j in jobs if j["labels"]["part"] == "sweep"}
    assert tag_seeds == set(seeds_for(29, 20))
    assert sweep_seeds == set(seeds_for(31, 50)[:20])
    assert len(exp.build_jobs(pilot=True)) == 40


def test_default_arm_uses_the_registered_scenarios_and_hold_only_adds_two_fields():
    for job in exp.build_jobs(pilot=True):
        labels, scenario = job["labels"], job["scenario"]
        if labels["part"] == "tag":
            registered = part_b_scenario(target_class=labels["target_class"], length_s=2.0, preset=labels["preset"],
                                         policy=exp.TAG_POLICY, seed=scenario["seed"])
        else:
            registered = exp031_scenario(scenario["seed"], preset=labels["preset"], dropout=labels["dropout"],
                                         message_mode="delta", duration_s=120.0)
        if labels["arm"] == "default":
            assert scenario == registered
        else:
            assert scenario["policy"] == {**registered["policy"], "lost_link_action": "hold",
                                          "hold_land_after_s": 5.0}
            assert {k: v for k, v in scenario.items() if k != "policy"} == {
                k: v for k, v in registered.items() if k != "policy"}


def test_registered_reference_has_one_row_per_cell_and_seed():
    reference = exp.registered_rows()
    tag = [k for k in reference if k[0] == "tag"]
    sweep = [k for k in reference if k[0] == "sweep"]
    assert len(tag) == 6 * 20
    assert len(sweep) == 4 * 50  # all 50 registered sweep seeds; the run uses the first 20


def _stats(part: str, *, landings: int, success: int, violations: int = 0, min_sep: float | None = 0.6) -> dict:
    count = {"count": success, "n": 20}
    if part == "tag":
        return {"runs": 20, "landings": landings, "landings_per_run": landings / 20, "tagged": count,
                "separation_violation_steps": violations, "min_separation_m": min_sep}
    return {"runs": 20, "landings": landings, "landings_per_run": landings / 20, "completed": count,
            "t100_s_median_of_completed": 50.0}


def _cells(**hold_overrides) -> list[dict]:
    cells = []
    for part, key, preset in exp.EXPECTED_CELLS:
        name = "target_class" if part == "tag" else "dropout"
        hold = {"landings": 0, "success": 20}
        hold.update(hold_overrides.get(f"{key}/{preset}", {}))
        cells.append({"part": part, name: key, "preset": preset,
                      "default": _stats(part, landings=20, success=10),
                      "hold": _stats(part, **hold)})
    return cells


def _valid(cells: list[dict]) -> dict:
    return exp.validity(cells, {"reproduced": True}, 20)


def test_no_cells_is_void_not_pass():
    valid = exp.validity([], {"reproduced": False}, 20)
    result = exp.verdict([], valid)
    assert result["status"] == "void" and result["pass"] is None
    assert valid["cells_present"] == 0 and len(valid["missing_cells"]) == 10


def test_a_missing_cell_a_short_cell_or_a_failed_reproduction_is_void():
    cells = _cells()
    assert exp.verdict(cells, _valid(cells))["status"] == "PASS"
    assert exp.verdict(cells[1:], _valid(cells[1:]))["status"] == "void"
    short = json.loads(json.dumps(cells))
    short[0]["hold"]["runs"] = 19
    assert exp.verdict(short, _valid(short))["status"] == "void"
    one_arm = json.loads(json.dumps(cells))
    one_arm[-1]["hold"] = None
    assert exp.verdict(one_arm, _valid(one_arm))["status"] == "void"
    assert exp.verdict(cells, exp.validity(cells, {"reproduced": False}, 20))["status"] == "void"


def test_rule_one_needs_every_cell_and_the_ten_percent_pool():
    # Default pools 10 cells x 20 = 200 landings, so the hold pool limit is 20.
    within = _cells(**{"jink/burst_30_0.5s": {"landings": 20}})
    assert exp.verdict(within, _valid(within))["rule_1_landings"]["pass"] is True
    over_pool = _cells(**{"jink/burst_30_0.5s": {"landings": 11}, "turn/burst_30_0.5s": {"landings": 10}})
    assert exp.verdict(over_pool, _valid(over_pool))["rule_1_landings"]["pass"] is False
    over_cell = _cells(**{"jink/burst_30_0.5s": {"landings": 21}})
    result = exp.verdict(over_cell, _valid(over_cell))["rule_1_landings"]
    assert result["pass"] is False and result["every_cell_hold_at_most_default"] is False


def test_rule_two_is_pooled_success():
    fewer_tags = _cells(**{"turn/burst_10_0.5s": {"success": 0}, "turn/burst_30_0.5s": {"success": 0},
                           "jink/burst_10_0.5s": {"success": 0}, "jink/burst_30_0.5s": {"success": 0},
                           "figure8/burst_10_0.5s": {"success": 9}})
    # Hold tags: 20 + 9 = 29 against default 6 x 10 = 60.
    assert exp.verdict(fewer_tags, _valid(fewer_tags))["rule_2_success"]["pass"] is False


def test_rule_three_separation_tolerance_is_five_centimetres():
    # Default minimum separation is 0.6 m in every tag cell.
    near = _cells(**{"jink/burst_30_0.5s": {"min_sep": 0.56}})
    assert exp.verdict(near, _valid(near))["rule_3_safety"]["pass"] is True
    far = _cells(**{"jink/burst_30_0.5s": {"min_sep": 0.54}})
    result = exp.verdict(far, _valid(far))
    assert result["rule_3_safety"]["pass"] is False and result["status"] == "FAIL"
    assert len(result["rule_3_safety"]["tag_cells_min_separation_more_than_0.05_m_below_default"]) == 1
    steps = _cells(**{"turn/burst_10_0.5s": {"violations": 1}})
    assert exp.verdict(steps, _valid(steps))["rule_3_safety"]["pass"] is False


def _row(**metrics) -> dict:
    return {"part": "sweep", "arm": "default", "preset": "burst_10_0.5s", "dropout": "none",
            "message_mode": "delta", "seed_index": 0, "seed": 7, "adapter": "point_mass", **metrics}


def _reference(**metrics) -> dict:
    row = {"slice": "main", "preset": "burst_10_0.5s", "dropout": "none", "message_mode": "delta",
           "adapter": "point_mass", "policy_name": "", "seed_index": "0", "seed": "7", **metrics}
    return {("sweep", "none", "burst_10_0.5s", "7"): row}


@pytest.mark.parametrize("row, reference, reproduced", [
    (_row(t100_s=None, replans=2, end_reason="timeout"), _reference(t100_s="", replans="2", end_reason="timeout"),
     True),
    (_row(t100_s=None, replans=3, end_reason="timeout"), _reference(t100_s="", replans="2", end_reason="timeout"),
     False),
    (_row(replans=2), _reference(t100_s="", replans="2"), False),  # a registered metric is missing
    (_row(t100_s=None, replans=2, extra=1), _reference(t100_s="", replans="2"), False),  # an unregistered metric
    (_row(replans=2, adapter="physics"), _reference(replans="2"), False),
])
def test_reproduction_compares_the_full_metric_set(row, reference, reproduced):
    hold_row = {**row, "arm": "hold", "replans": 99}
    result = exp.default_reproduction([row, hold_row], reference)
    assert result["reproduced"] is reproduced
    assert result["rows_checked"] == 1  # hold rows are not compared


def test_a_row_without_a_registered_twin_fails_reproduction():
    result = exp.default_reproduction([_row(replans=2)], {})
    assert result == {"rows_checked": 0, "rows_mismatched": 0, "rows_without_reference": 1, "examples": [],
                      "reproduced": False}


def test_small_run_writes_hashed_outputs_and_is_void(tmp_path: Path):
    jobs = [j for j in exp.build_jobs(pilot=True)
            if j["labels"]["part"] == "tag" and j["labels"]["target_class"] == "turn"
            and j["labels"]["preset"] == "burst_10_0.5s" and j["labels"]["seed_index"] == 0]
    assert len(jobs) == 2
    first = exp.run_experiment(tmp_path / "a", jobs=jobs, argv=["--output", "x"])
    exp.run_experiment(tmp_path / "b", jobs=jobs, argv=["--output", "x"])
    assert first["verdict"]["status"] == "void"
    assert first["default_arm_reproduces_registered_rows"]["reproduced"] is True
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    for name in exp.OUTPUTS:
        data = (tmp_path / "a" / name).read_bytes()
        assert manifest["outputs"][name] == hashlib.sha256(data).hexdigest()
        if name.endswith((".csv", ".json")):
            assert data == (tmp_path / "b" / name).read_bytes()


def test_committed_run_matches_its_manifest_and_registry_entry():
    if not RUN_DIR.exists():
        pytest.skip("EXP-042 run folder not present")
    manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((RUN_DIR / name).read_bytes()).hexdigest() == digest
    assert summary["pilot"] is False and summary["jobs"] == 400
    assert summary["validity"]["valid"] is True and summary["validity"]["cells_present"] == 10
    assert summary["default_arm_reproduces_registered_rows"]["rows_checked"] == 200
    assert summary["verdict"]["status"] in {"PASS", "FAIL"}
    registry = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    entries = [e for e in registry if e["run_dir"] == "sim/runs/e-exp042-hold"]
    assert len(entries) == 1 and entries[0]["outputs"] == manifest["outputs"]


def test_hold_fields_reach_the_failsafe_config():
    for job in exp.build_jobs(pilot=True):
        config = Scenario.from_dict(job["scenario"]).failsafe_config
        if job["labels"]["arm"] == "hold":
            assert (config.lost_link_action, config.hold_land_after_s) == ("hold", 5.0)
        else:
            assert config.lost_link_action == "land"
        assert (config.degraded_after_s, config.land_after_s) == (0.15, 0.3)


def test_tag_reproduction_checks_the_blackout_start():
    row = {"part": "tag", "arm": "default", "target_class": "jink", "preset": "burst_10_0.5s", "seed": 11,
           "adapter": "point_mass", "landings": 1}
    start = repr(exp.draw_outage_start(11))
    reference = {("tag", "jink", "burst_10_0.5s", "11"): {"adapter": "point_mass", "blackout_start_s": start,
                                                          "landings": "1"}}
    assert exp.default_reproduction([row], reference)["reproduced"] is True
    reference[("tag", "jink", "burst_10_0.5s", "11")]["blackout_start_s"] = "9.5"
    result = exp.default_reproduction([row], reference)
    assert result["reproduced"] is False and result["examples"][0]["fields"] == ["blackout_start_s"]


def test_cell_stats_from_hand_rows():
    tag_rows = [
        {"duration_s": 30.0, "landings": 1, "landings_before_outage": 1, "tagged": False, "min_separation_m": None,
         "separation_violations": 0, "time_degraded_s": 0.5},
        {"duration_s": 30.0, "landings": 0, "landings_before_outage": 0, "tagged": True, "min_separation_m": 0.4,
         "separation_violations": 3, "time_degraded_s": 1.5},
    ]
    tag = exp.cell_stats(tag_rows, "tag")
    assert tag["landings"] == 1 and tag["landings_per_run"] == 0.5
    assert tag["landings_per_drone_hour"] == pytest.approx(1 / (60.0 / 3600.0))  # 60 s of one follower
    assert tag["tagged"]["count"] == 1 and tag["min_separation_m"] == 0.4  # None is skipped, not zero
    assert (tag["separation_violation_steps"], tag["runs_with_violation"]) == (3, 1)
    sweep_rows = [
        {"duration_s": 120.0, "failsafe_landings": 3, "end_reason": "timeout", "t100_s": None,
         "false_loss_declarations": 2, "duplicated_fraction": 0.1},
        {"duration_s": 40.0, "failsafe_landings": 0, "end_reason": "complete", "t100_s": 40.0,
         "false_loss_declarations": 0, "duplicated_fraction": 0.3},
        {"duration_s": 60.0, "failsafe_landings": 0, "end_reason": "complete", "t100_s": 60.0,
         "false_loss_declarations": 1, "duplicated_fraction": 0.2},
    ]
    sweep = exp.cell_stats(sweep_rows, "sweep")
    assert sweep["completed"]["count"] == 2 and sweep["t100_s_median_of_completed"] == 60.0
    assert sweep["landings_per_drone_hour"] == pytest.approx(3 / (220.0 * 3 / 3600.0))
    assert sweep["false_loss_declarations"] == 3 and sweep["duplicated_fraction_mean"] == pytest.approx(0.2)


def test_prediction_check_reads_the_observed_cells():
    cells = _cells(**{"jink/burst_30_0.5s": {"violations": 5, "min_sep": 0.3}})
    checks = exp.prediction_check(cells, exp.verdict(cells, _valid(cells)))
    held = [c["held"] for c in checks]
    # Hand cells: default lands 20 times per 20 runs (tag 1.0 per drone, sweep a third per drone), default
    # tags and sweeps 10 of 20, hold 0 landings, 20 of 20 tags and sweeps, t100 50 s, violations only on jink.
    assert held == [
        False,  # about once per drone per run: sweep cells are a third
        True,   # hold 0 landings
        True,   # rule 1
        False,  # 20 of 20 tags is outside the 12 to 18 band for burst_30
        False,  # default sweeps completed 10, not 0
        True,   # hold sweeps at least 15 of 20
        False,  # t100 50 s is outside 60 to 90 s for dropout one
        True,   # rule 2
        True,   # rule 3 fails, all steps on jink, one cell more than 0.05 m below
        True,   # overall FAIL
    ]
    assert checks[8]["observed"]["hold_steps_on_jink"] == 5
    assert exp.prediction_check([], exp.verdict([], exp.validity([], {"reproduced": False}, 20))) == []
