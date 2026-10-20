"""Run D's separation rule inside the Run E tag and formation missions (simulated)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sim.missions.tag as tag
from sim.coop.avoidance import AvoidanceConfig
from sim.missions.api import Scenario
from sim.missions.engine import run_mission

BUFFER_FILE = Path(__file__).resolve().parents[1] / "runs" / "d4-separation" / "buffer.json"


def test_buffer_table_is_run_d_dead_reckoning_envelope():
    data = json.loads(BUFFER_FILE.read_text(encoding="utf-8"))
    envelope = {
        (float(row["speed_mps"]), float(row["delay_s"])): float(row["required_buffer_m"])
        for row in data["envelope"] if row["dead_reckoning"]
    }
    assert envelope == tag.COOP_BUFFER_M
    assert data["min_separation_m"] == tag.COOP_FLOOR_M
    assert data["time_horizon_s"] == tag.COOP_TIME_HORIZON_S
    assert data["avoidance"]["safety_margin_m"] == 0.0
    # Run D's floor is also AvoidanceConfig's default radius.
    assert AvoidanceConfig().radius_m == tag.COOP_FLOOR_M


def test_buffer_choice_takes_the_worst_covered_row():
    # Read off the table by hand: telemetry 0.002 s old is covered by the
    # 0.05 s column, a 1.5 m/s drone by the 2 m/s row; the worst row at or
    # below both is (1.0 m/s, 0.0 s) = 0.2 m.
    assert tag.coop_buffer_m(0.002, 1.5) == 0.2
    # Fix latency 0.2 s: the worst of every row up to 2 m/s and 0.2 s is
    # (1.0 m/s, 0.2 s) = 0.3 m.
    assert tag.coop_buffer_m(0.2, 1.5) == 0.3
    # A slow drone with fresh data uses only the 0.5 m/s, 0 s cell.
    assert tag.coop_buffer_m(0.0, 0.5) == 0.05
    config = tag.coop_avoidance_config(0.002, 1.5)
    assert config.radius_m == pytest.approx(0.55)
    assert config.max_speed_mps == 1.5
    assert config.time_horizon_s == 2.0
    assert config.responsibility == 0.5


def test_coop_goal_leaves_an_unconstrained_goal_alone():
    config = tag.coop_avoidance_config(0.002, 1.5)
    own = (np.array([2.0, 2.0, 1.5]), np.zeros(3))
    far = [(np.array([12.0, 8.0, 1.5]), np.zeros(3))]
    goal = np.array([5.0, 2.0, 1.5])
    sent, changed = tag.coop_goal(goal, own[0], own, far, config)
    assert changed is False
    assert np.array_equal(sent, goal)


def _head_on(monkeypatch, rule):
    """Two followers 1 m apart on one line, each sent to the far end of the other.

    The slots are moved to (2, 7.5) and (2, 2.5) and the assignment is pinned
    so f1 (starting at y 4.5) must go to y 7.5 and f2 (y 5.5) to y 2.5: an
    exact head-on swap at up to 1.5 m/s each.  The target stands still 2 m to
    the side, so the target guard stays out of it.
    """
    monkeypatch.setattr(tag.TagMission, "_slots", lambda self, estimate, velocity, n, standoff: [
        np.array((2.0, 7.5, 1.5)), np.array((2.0, 2.5, 1.5))])
    monkeypatch.setattr(tag, "linear_sum_assignment",
                        lambda costs: (np.arange(costs.shape[0]), np.arange(costs.shape[1])))
    return run_mission(Scenario(
        mission="tag", seed=20261030, duration_s=8.0, adapter="point_mass", preset="clean",
        params={"target_class": "straight", "target_speed_mps": 0.0, "n_followers": 2,
                "initial_gap_m": 2.0, "separation_rule": rule},
    ), log=True)


def test_head_on_followers_keep_run_d_floor_in_coop_mode(monkeypatch):
    coop = _head_on(monkeypatch, "coop")
    local = _head_on(monkeypatch, "local")
    print("coop", coop.metrics["follower_follower_min_m"], coop.metrics["guard_activations"],
          "local", local.metrics["follower_follower_min_m"], local.metrics["guard_activations"])
    # Run D's floor is 0.35 m (buffer.json min_separation_m).
    assert coop.metrics["follower_follower_min_m"] > tag.COOP_FLOOR_M
    assert coop.metrics["guard_activations"] > 0
    # Both followers still get past each other to their slots.
    agents = coop.log["frames"]["agents"]
    assert agents["f1"]["position_m"][-1][1] == pytest.approx(7.5, abs=0.1)
    assert agents["f2"]["position_m"][-1][1] == pytest.approx(2.5, abs=0.1)
    # The check has teeth: the old 0.6 m goal guard lets the pair close far
    # below the floor in the same case.
    assert local.metrics["follower_follower_min_m"] < tag.COOP_FLOOR_M


@pytest.mark.parametrize("target_class", ["straight", "turn", "figure8", "jink"])
def test_one_follower_is_the_same_in_both_modes(target_class):
    def run(rule):
        return run_mission(Scenario(
            mission="tag", seed=20261030, duration_s=30.0, adapter="point_mass", preset="clean",
            params={"target_class": target_class, "separation_rule": rule},
        ), log=True)
    coop, local = run("coop"), run("local")
    assert coop.metrics == local.metrics
    assert coop.log["frames"] == local.log["frames"]
    assert coop.log["events"] == local.log["events"]
    assert coop.metrics["guard_activations"] == 0


def test_separation_rule_is_validated():
    with pytest.raises(ValueError):
        run_mission(Scenario(mission="tag", seed=1, duration_s=1.0, adapter="point_mass", preset="clean",
                             params={"separation_rule": "run_d"}))
    with pytest.raises(ValueError):
        run_mission(Scenario(mission="formation", seed=1, duration_s=1.0, adapter="point_mass_offboard",
                             preset="clean", params={"separation_rule": "none"}))
