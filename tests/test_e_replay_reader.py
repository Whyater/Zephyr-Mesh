"""Tests for the independent simulated mission log replay reader."""
from __future__ import annotations

import json

import math

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import write_log
from sim.missions.replay_reader import compare, recompute


def _mini_log():
    return {
        "mission": "hover",
        "scenario": {"params": {"goal_m": [1.0, 0.0, 0.0]}, "outages": []},
        "frames": {"time_s": [0.0, 0.05], "target_position_m": [None, None], "agents": {"d1": {
            "position_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "velocity_mps": [[0, 0, 0], [0, 0, 0]],
            "failsafe_state": ["nominal", "landing"], "authority": ["mission", "mission"],
            "link_age_s": [0.1, 0.2], "estimate_position_m": None, "estimate_age_s": None,
        }}},
        "events": [{"kind": "mission_end", "time_s": 0.05, "detail": {"reason": "timeout"}}],
        "metrics": {"end_reason": "timeout", "duration_s": 0.05, "final_error_m": 0.0,
                    "final_failsafe_state": "landing", "max_link_age_s": 0.2, "unknown": 3},
    }


def test_recompute_hand_built_log():
    result = recompute(_mini_log())
    assert result["final_error_m"]["value"] == 0.0
    assert result["final_error_m"]["exact"] is True
    assert result["max_link_age_s"]["value"] == 0.2
    assert result["unknown"]["value"] is None


def test_compare_marks_unknown_as_unchecked():
    rows = {row["metric"]: row for row in compare(_mini_log())}
    assert rows["final_error_m"]["agrees"] is True
    assert rows["unknown"]["exact"] is False
    assert rows["unknown"]["agrees"] is None


@pytest.mark.parametrize(
    ("mission", "params", "adapter", "duration_s"),
    [
        ("hover", {"goal_m": [0.0, 0.0, 1.5]}, "point_mass", 1.0),
        ("tag", {}, "point_mass", 1.0),
        ("sweep", {"message_mode": "delta"}, "point_mass", 1.0),
        ("relay", {}, "point_mass", 1.0),
        ("monitor", {}, "point_mass", 1.0),
        ("formation", {}, "point_mass_offboard", 6.0),
    ],
)
def test_replay_each_simulated_mission(tmp_path, mission, params, adapter, duration_s):
    scenario = Scenario(mission=mission, seed=35, duration_s=duration_s,
                        adapter=adapter, params=params)
    result = run_mission(scenario, log=True)
    path = tmp_path / f"{mission}.json"
    write_log(path, result.log)
    log = json.loads(path.read_text(encoding="utf-8"))
    replay = recompute(log)
    assert set(result.metrics).issubset(replay)
    assert replay["end_reason"]["exact"] is True
    assert replay["duration_s"]["exact"] is True
    assert all(set(row) == {"metric", "mission_value", "reader_value", "exact", "agrees", "reason"}
               for row in compare(log))


def test_direct_relay_has_no_per_hop_delivery(tmp_path):
    # EXP-035 finding: the first reader spec defined hop1_delivery for direct
    # links too; the mission defines per-hop delivery only with a relay.  The
    # spec was corrected to the mission's definition (one hop: use delivery_ratio).
    scenario = Scenario(mission="relay", seed=35, duration_s=1.0,
                        adapter="point_mass", params={})
    result = run_mission(scenario, log=True)
    path = tmp_path / "relay.json"
    write_log(path, result.log)
    log = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in compare(log) if row["metric"] == "hop1_delivery")
    assert row["agrees"] is True


def test_monitor_continuity_counts_no_estimate_as_a_gap(tmp_path):
    # EXP-035 finding: before the first fix arrives the estimate age is
    # infinite, which the log stores as null.  The first reader skipped those
    # checks and reported continuity 1.0 against the mission's 0.990.
    scenario = Scenario(mission="monitor", seed=1, duration_s=4.0, measurement_delay_s=0.2,
                        params={"path": "crossing"})
    result = run_mission(scenario, log=True)
    path = tmp_path / "monitor.json"
    write_log(path, result.log)
    log = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in compare(log) if row["metric"] == "continuity")
    assert row["exact"] is True
    assert row["mission_value"] < 1.0
    assert row["agrees"] is True
