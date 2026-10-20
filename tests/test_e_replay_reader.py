"""Tests for the independent simulated mission log replay reader."""
from __future__ import annotations

import json

import math

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import write_log
from sim.missions.replay_reader import _interpolate_entry, compare, recompute


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
    for row in compare(log):
        if row["exact"]:
            assert row["agrees"] is not False


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


def _event_log(mission, events, *, params=None, outages=None, times=None,
               targets=None, agents=None, metrics=None, link_stats=None):
    return {
        "mission": mission,
        "scenario": {"params": params or {}, "outages": outages or []},
        "frames": {"time_s": times or [], "target_position_m": targets or [],
                   "agents": agents or {}},
        "events": events,
        "metrics": metrics or {},
        **({"link_stats": link_stats} if link_stats is not None else {}),
    }


def test_tag_landings_before_outage_uses_earliest_start():
    log = _event_log(
        "tag",
        [{"kind": "failsafe_transition", "time_s": 0.9, "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 1.0, "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 2.0, "detail": {"state": "landing"}}],
        outages=[{"start_s": 3.0}, {"start_s": 1.0}],
    )
    assert recompute(log)["landings_before_outage"]["value"] == 1


def test_tag_landings_before_outage_without_outage_counts_all():
    log = _event_log(
        "tag",
        [{"kind": "failsafe_transition", "time_s": 0.9, "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 2.0, "detail": {"state": "landing"}}],
    )
    assert recompute(log)["landings_before_outage"]["value"] == 2


def test_tag_holds_counts_degraded_transitions():
    log = _event_log(
        "tag",
        [{"kind": "failsafe_transition", "time_s": 0.2, "detail": {"state": "degraded"}},
         {"kind": "failsafe_transition", "time_s": 0.4, "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 0.6, "detail": {"state": "degraded"}}],
    )
    assert recompute(log)["holds"]["value"] == 2


def test_tag_one_follower_has_no_follower_pair_distance():
    log = _event_log(
        "tag",
        [],
        times=[0.0],
        targets=[[0.0, 0.0, 0.0]],
        agents={"f1": {"position_m": [[1.0, 0.0, 0.0]], "estimate_age_s": [0.0],
                        "failsafe_state": ["nominal"]}},
    )
    result = recompute(log)["follower_follower_min_m"]
    assert result["value"] is None


def test_sweep_failsafe_landings_excludes_d2_at_dropout():
    log = _event_log(
        "sweep",
        [{"kind": "drone_dropout", "time_s": 10.0, "agent": "d2"},
         {"kind": "failsafe_transition", "time_s": 9.99, "agent": "d2", "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 10.0, "agent": "d2", "detail": {"state": "landing"}},
         {"kind": "failsafe_transition", "time_s": 11.0, "agent": "d1", "detail": {"state": "landing"}}],
    )
    assert recompute(log)["failsafe_landings"]["value"] == 2


def test_sweep_t100_and_belief_error_are_exact_types():
    log = _event_log(
        "sweep",
        [{"kind": "replan", "time_s": 1.0, "detail": {"belief_error_cells": 2}},
         {"kind": "replan", "time_s": 2.0, "detail": {"belief_error_cells": 3}},
         {"kind": "mission_end", "time_s": 12.0, "detail": {"reason": "complete"}}],
    )
    result = recompute(log)
    assert result["belief_error_cells"]["value"] == 5
    assert isinstance(result["belief_error_cells"]["value"], int)
    assert result["t100_s"]["value"] == 12.0


def test_relay_delivery_values_with_relay_fraction():
    log = _event_log(
        "relay", [], params={"relay_fraction": 0.5},
        link_stats={"tel:far": {"sent": 4, "delivered": 3},
                    "tel:relay": {"sent": 2, "delivered": 1}},
    )
    result = recompute(log)
    assert result["hop1_delivery"]["value"] == 0.75
    assert result["hop2_delivery"]["value"] == 0.5
    assert result["delivery_ratio"]["value"] is None


def test_relay_delivery_values_without_relay_fraction():
    log = _event_log(
        "relay", [], params={"relay_fraction": None},
        link_stats={"tel:far": {"sent": 5, "delivered": 4},
                    "tel:relay": {"sent": 2, "delivered": 2}},
    )
    result = recompute(log)
    assert result["hop1_delivery"]["value"] is None
    assert result["hop2_delivery"]["value"] is None
    assert result["delivery_ratio"]["value"] == 0.8
    assert "counters are complete" not in result["hop2_delivery"]["reason"]


def test_monitor_missed_alert_requires_two_seconds_and_entry_window():
    # Entry is max((20 - 10) / 15, (12 - 10) / 5) * 1 s = 2/3 s.
    # The alert at 0.5 s is before entry, and the in-zone dwell reaches
    # 3 - 2/3 = 7/3 s, so the alert is missed.
    log = _event_log(
        "monitor",
        [{"kind": "alert_raised", "time_s": 0.5}],
        params={"path": "crossing"},
        times=[0.0, 1.0, 2.0, 3.0, 4.0],
        targets=[[10.0, 10.0, 0.0], [25.0, 15.0, 0.0],
                 [26.0, 16.0, 0.0], [27.0, 17.0, 0.0], [10.0, 10.0, 0.0]],
    )
    assert recompute(log)["missed_alert"]["value"] is True


def test_monitor_false_alert_uses_interpolated_truth_outside_zone():
    # At 0.5 s the interpolated truth is (45, 20), 5 m outside x = 40.
    log = _event_log(
        "monitor",
        [{"kind": "alert_raised", "time_s": 0.5}],
        params={"path": "crossing"},
        times=[0.0, 1.0],
        targets=[[45.0, 20.0, 0.0], [45.0, 20.0, 0.0]],
    )
    assert recompute(log)["false_alert"]["value"] is True


def test_monitor_entry_from_high_x_side_uses_slab_entry():
    # x enters at (40 - 50) / (38 - 50) = 5/6; y is already inside.
    assert _interpolate_entry(0.0, 1.0, [50.0, 20.0, 0.0], [38.0, 20.0, 0.0]) == pytest.approx(5 / 6)


def test_monitor_diagonal_entry_uses_later_axis_fraction():
    # x enters at (20 - 10) / 20 = 1/2; y enters at (12 - 5) / 10 = 7/10.
    assert _interpolate_entry(0.0, 10.0, [10.0, 5.0, 0.0], [30.0, 15.0, 0.0]) == pytest.approx(7.0)


def test_hover_last_frame_before_duration_is_not_exact():
    log = _mini_log()
    log["events"][0]["time_s"] = 0.10
    result = recompute(log)
    assert result["final_error_m"]["value"] == 0.0
    assert result["final_error_m"]["exact"] is False


def test_monitor_missed_alert_requires_two_seconds_of_dwell():
    # The in-zone interval is 2 - 1 = 1 s, so it is below the required 2 s.
    log = _event_log(
        "monitor", [], params={"path": "crossing"},
        times=[0.0, 1.0, 2.0, 2.1],
        targets=[[10.0, 10.0, 0.0], [25.0, 15.0, 0.0],
                 [25.0, 15.0, 0.0], [10.0, 10.0, 0.0]],
    )
    assert recompute(log)["missed_alert"]["value"] is False


def test_monitor_continuity_hand_built_null_age_is_a_gap():
    # Checks are false, true, false, so continuity is 1 / 3.
    ages = [None, 0.1, 0.6]
    log = _event_log(
        "monitor", [], params={"path": "crossing"},
        times=[0.0, 0.1, 0.2],
        targets=[[30.0, 20.0, 0.0]] * 3,
        agents={"o1": {"estimate_age_s": ages}},
    )
    result = recompute(log)["continuity"]
    assert result["value"] == pytest.approx(1 / 3)
    assert result["exact"] is True


def test_tag_blind_in_plane_uses_earliest_outage_start():
    log = _event_log(
        "tag", [],
        outages=[{"start_s": 5.0, "end_s": 6.0}, {"start_s": 2.0, "end_s": 4.0}],
        times=[2.0, 3.0, 4.0],
        targets=[[0.0, 0.0, 0.0]] * 3,
        agents={"f1": {"position_m": [[0.0, 0.0, 0.0]] * 3,
                        "failsafe_state": ["nominal"] * 3}},
    )
    # All three frames lie in the earliest outage window, so 3 * 0.05 = 0.15 s.
    assert recompute(log)["blind_in_plane_s"]["value"] == pytest.approx(0.15)


def test_tag_empty_follower_positions_do_not_reuse_prior_distance():
    log = _event_log(
        "tag", [], params={"min_separation_m": 0.5},
        times=[0.0, 0.05],
        targets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        agents={"f1": {"position_m": [[0.1, 0.0, 0.0]]}},
    )
    # Only the first frame has a follower position, so one violation * 5 = 5.
    assert recompute(log)["separation_violations"]["value"] == 5


def test_formation_hand_built_errors_and_separation():
    slots = {"f1": [7.0, 4.0, 1.5], "f2": [9.0, 4.0, 1.5], "f3": [9.0, 6.0, 1.5], "f4": [7.0, 6.0, 1.5]}
    agents = {name: {"position_m": [list(slot)] * 3} for name, slot in slots.items()}
    # t = 4.95 s (before the 5 s cut): f1 sits 0.4 m from f2, the closest pair of the run.
    # t = 5.00 s: f1 is 0.5 m off its slot along x.  t = 5.05 s: f1 is back on its slot.
    agents["f1"]["position_m"] = [[8.6, 4.0, 1.5], [7.5, 4.0, 1.5], [7.0, 4.0, 1.5]]
    log = _event_log("formation", [], times=[4.95, 5.0, 5.05], targets=[None] * 3, agents=agents)
    result = recompute(log)
    # Errors over t >= 5 s: 8 drone-frames, one of them 0.5 m: rms = sqrt(0.25 / 8).
    assert result["rms_error_m"]["value"] == pytest.approx(math.sqrt(0.25 / 8))
    # f1's x offsets are [0.5, 0], std 0.25; every other axis and drone has std 0:
    # the mean over 4 drones x 3 axes is 0.25 / 12.
    assert result["per_axis_std_m"]["value"] == pytest.approx(0.25 / 12)
    # The closest pair over all frames, including the one before 5 s.
    assert result["min_separation_m"]["value"] == pytest.approx(0.4)
    assert result["rms_error_m"]["exact"] is False
