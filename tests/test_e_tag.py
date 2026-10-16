"""Run E tag mission checks for the point-mass and physics adapters."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log
from sim.missions.tag import TagMission
from tests.reference_e.hand_checks import tail_follow_time_s


def _scenario(seed=20261030, **changes):
    values = dict(mission="tag", seed=seed, duration_s=30.0, adapter="point_mass", preset="clean")
    values.update(changes)
    return Scenario(**values)


def _first_zone_entry_s(log, agent="f1"):
    return next(e["time_s"] for e in log["events"] if e["kind"] == "tag_zone_enter" and e["agent"] == agent)


def test_hand_check_straight_recovery():
    result = run_mission(_scenario(params={"target_class": "straight", "arm": "recovery"}), log=True)
    entry = _first_zone_entry_s(result.log)
    reference = tail_follow_time_s(4.0, 1.0, 1.5, 0.5)
    print(f"zone entry={entry:.3f}s reference={reference:.3f}s tag={result.metrics['time_to_tag_s']}")
    assert reference == 3.0
    # Prereg EXP-030 band: 3.0 s plus the speed ramp (about v / a = 0.4 s).
    assert 3.0 <= entry <= 3.6
    # The point-mass follower stays inside from its first entry, so the tag
    # completes one hold time after entry.
    assert result.metrics["time_to_tag_s"] == pytest.approx(entry + 2.0, abs=0.02)


@pytest.mark.parametrize("target_class", ["straight", "turn", "figure8", "jink"])
def test_clean_all_classes_success(target_class):
    results = [run_mission(_scenario(seed=20261030 + k, params={"target_class": target_class})) for k in range(5)]
    tagged = sum(bool(item.metrics["tagged"]) for item in results)
    violations = sum(int(item.metrics["separation_violations"]) for item in results)
    print(target_class, "tagged", tagged, "/ 5", "separation violations", violations)
    assert tagged / 5.0 >= 0.8
    assert violations == 0


def test_separation_violation_is_reported():
    result = run_mission(_scenario(duration_s=3.0, params={
        "target_class": "straight", "target_speed_mps": 0.0,
        "standoff_m": 0.3, "min_separation_m": 0.5,
        "follower_guard_m": 0.0, "separation_guard": False,
    }), log=True)
    assert result.metrics["separation_violations"] > 0
    assert result.metrics["separation_episodes"] > 0
    assert any(event["kind"] == "separation_violation" for event in result.log["events"])


def test_hold_blackout_reactions_and_calm_policy():
    outage = ({"start_s": 10.0, "end_s": None, "streams": None},)
    default = run_mission(_scenario(duration_s=12.0, outages=outage, params={"mode": "hold"}))
    calm = run_mission(_scenario(duration_s=12.0, outages=outage, policy={"degraded_after_s": 0.4, "land_after_s": 1.0}, params={"mode": "hold"}))
    assert default.metrics["outage_reaction_hold_s"] == pytest.approx(0.15, abs=0.06)
    assert default.metrics["outage_reaction_land_s"] == pytest.approx(0.30, abs=0.06)
    assert default.metrics["nominal_at_age_0_33"] is False
    assert calm.metrics["nominal_at_age_0_33"] is True
    assert calm.metrics["outage_reaction_hold_s"] == pytest.approx(0.4, abs=0.06)
    assert calm.metrics["outage_reaction_land_s"] == pytest.approx(1.0, abs=0.06)


def test_determinism_and_log_validation():
    scenario = _scenario(seed=20261031, duration_s=8.0, params={"target_class": "figure8"})
    first = run_mission(scenario, log=True)
    second = run_mission(scenario, log=True)
    validate_log(first.log)
    assert first.metrics == second.metrics
    assert json.dumps(first.log, sort_keys=True, separators=(",", ":")) == json.dumps(second.log, sort_keys=True, separators=(",", ":"))


def test_three_follower_slots_match_prereg_geometry():
    mission = TagMission()
    estimate = np.array([8.0, 5.0, 1.5])
    heading_rad = math.radians(30.0)
    velocity = 0.5 * np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    mission._heading = np.array([1.0, 0.0, 0.0])
    slots = mission._slots(estimate, velocity, 3, 0.75)
    angles = []
    for slot in slots:
        offset = slot - estimate
        assert offset[2] == pytest.approx(0.0)
        assert math.hypot(offset[0], offset[1]) == pytest.approx(0.75)
        angle = math.degrees(math.atan2(offset[1], offset[0]) - heading_rad) % 360.0
        angles.append(angle)
    # Prereg: behind, then 120 and 240 degrees from the heading.
    assert angles == pytest.approx([180.0, 120.0, 240.0])
    chords = sorted(float(np.linalg.norm(slots[a] - slots[b])) for a, b in ((0, 1), (0, 2), (1, 2)))
    # Neighbours 60 degrees apart: 2 * 0.75 * sin(30 deg) = 0.75 m; the
    # 120 and 240 degree slots: 2 * 0.75 * sin(60 deg) = 1.299 m.
    assert chords == pytest.approx([0.75, 0.75, 2 * 0.75 * math.sin(math.radians(60.0))])
    assert chords[0] > 0.6


def test_landings_before_outage_clean_counts_none():
    outage = ({"start_s": 6.0, "end_s": None, "streams": None},)
    result = run_mission(_scenario(duration_s=9.0, outages=outage, params={"mode": "hold"}))
    no_outage = run_mission(_scenario(duration_s=6.0, params={"mode": "hold"}))
    # The clean link never drops a packet, so the only landing is the one the
    # blackout causes, and it comes after the blackout starts.
    assert result.metrics["landings"] == 1
    assert result.metrics["landings_before_outage"] == 0
    assert no_outage.metrics["landings_before_outage"] == no_outage.metrics["landings"] == 0


def test_landings_before_outage_matches_a_run_cut_at_the_outage():
    # Independent check: before the blackout the outage run and a run that
    # simply stops at the blackout start see the same seeded loss, so the
    # landings counted before the outage must equal all landings of the
    # shorter run.  Bursty synthetic loss gives both outcomes over these seeds.
    outage = ({"start_s": 10.0, "end_s": None, "streams": None},)
    seen = set()
    for k in range(6):
        seed = 20261029 + k
        with_outage = run_mission(_scenario(seed=seed, duration_s=13.0, preset="burst_10_0.5s",
                                            outages=outage, params={"mode": "hold"}))
        cut = run_mission(_scenario(seed=seed, duration_s=10.0, preset="burst_10_0.5s", params={"mode": "hold"}))
        assert with_outage.metrics["landings_before_outage"] == cut.metrics["landings"]
        assert cut.metrics["landings_before_outage"] == cut.metrics["landings"]
        assert with_outage.metrics["landings_before_outage"] <= with_outage.metrics["landings"]
        seen.add(with_outage.metrics["landings_before_outage"])
    assert seen == {0, 1}


# Measured reason (clean, delay 0, seed 20261030): the separation guard fires
# whenever the estimated distance is below 0.5 m + buffer, and with v = 1.5 m/s
# the buffer is 0.36 to 0.51 m, so it fires across most of the 1.0 m zone.  Its
# retreat point is the aim point without the 0.75 s lead.  The physics position
# loop lags a moving setpoint by Kd/Kp = 0.75 s, 0.375 m at 0.5 m/s, so behind
# a straight target the follower settles near 1.125 m (outside the zone) on the
# retreat point and near 0.75 m on the aim point.  It cycles across the zone
# edge with in-zone stints of about 1.8 to 2.1 s, first entry 4.09 s, and tags
# only at 12.08 s.  The point mass lags only 0.125 m and stays inside.
@pytest.mark.xfail(strict=True, reason="physics follower cycles across the zone edge: retreat point lacks the lead, PD lag 0.375 m")
def test_physics_holds_zone_from_first_entry():
    started = time.perf_counter()
    result = run_mission(_scenario(seed=20261030, duration_s=30.0, adapter="physics",
                                   params={"target_class": "straight"}), log=True)
    elapsed = time.perf_counter() - started
    entry = _first_zone_entry_s(result.log)
    print(f"physics straight wall={elapsed:.3f}s entry={entry:.2f}s tag={result.metrics['time_to_tag_s']}")
    assert result.metrics["tagged"] is True
    assert result.metrics["time_to_tag_s"] <= entry + 2.0 + 0.05


# Measured reason (clean, delay 0, seed 20261030): the jink target turns back
# toward the follower, closing at about 2.3 m/s.  The guard's retreat point
# sits 0.75 m from the estimate on the follower's side; the PD loop then trails
# that point by about 0.375 m toward the target and brakes more slowly than the
# buffer's 4 m/s^2, so the true distance falls to 0.023 m.  Every violation is
# reported in the metrics; this test documents that the pre-registered design
# does not keep 0.5 m on the physics model.
@pytest.mark.xfail(strict=True, reason="physics follower breaks 0.5 m on a jink reversal: PD lag and braking slower than the buffer")
def test_physics_jink_keeps_min_separation():
    result = run_mission(_scenario(seed=20261030, duration_s=30.0, adapter="physics",
                                   params={"target_class": "jink"}))
    print("physics jink min separation", result.metrics["min_separation_m"],
          "violation steps", result.metrics["separation_violations"])
    assert result.metrics["separation_violations"] == 0


def test_hold_mode_runs_through_the_blackout_and_times_from_its_start():
    # A 2 s blackout from t = 8 s with the lost-link timeout at 1.0 s: the
    # follower must hold within about T_w after the blackout starts, start to
    # land about T after it starts (one 20 Hz command period of slack, since the
    # newest command before the blackout can be up to 0.05 s old), and the run
    # must continue to the scenario duration instead of ending when the
    # landing begins (the 3 s window after the blackout must be observed).
    policy = {"degraded_after_s": 0.15, "land_after_s": 1.0}
    outage = ({"start_s": 8.0, "end_s": 10.0, "streams": None},)
    result = run_mission(_scenario(duration_s=16.0, outages=outage, policy=policy,
                                   params={"target_class": "turn", "mode": "hold"}))
    m = result.metrics
    assert result.end_reason == "timeout"
    assert m["duration_s"] == 16.0
    assert 0.15 - 1e-9 <= m["outage_reaction_hold_s"] <= 0.15 + 0.05 + 0.011
    assert 1.0 - 1e-9 <= m["outage_reaction_land_s"] <= 1.0 + 0.05 + 0.011
    assert m["landings_before_outage"] == 0
    # Blind in the target's plane: from 8.0 s until the 1.5 m high follower has
    # left the +-0.3 m band after the landing starts at about 9.0 s.  The point
    # mass LAND is the original SimAdapter law (first order toward the floor,
    # capped at the 1.5 m/s speed limit), so the 0.3 m drop takes at least
    # 0.3 / 1.5 = 0.2 s and, with the 0.25 s response ramp, well under 0.5 s.
    assert 1.0 + 0.2 <= m["blind_in_plane_s"] <= 1.0 + 0.05 + 0.011 + 0.5


def test_hold_mode_reports_landing_before_the_blackout():
    # On 0.5 s bursts with the default 0.30 s rule a follower often lands before
    # the scripted blackout; it must be reported as not airborne at the
    # blackout, with its first landing time, and a run on clean stays airborne.
    outage = ({"start_s": 15.0, "end_s": 17.0, "streams": None},)
    seen_grounded = False
    for seed in range(20261030, 20261036):
        result = run_mission(_scenario(seed=seed, duration_s=20.0, preset="burst_10_0.5s", outages=outage,
                                       params={"target_class": "turn", "mode": "hold"}))
        m = result.metrics
        if m["landings_before_outage"] > 0:
            seen_grounded = True
            assert m["airborne_at_blackout"] is False
            assert m["first_landing_s"] < 15.0
        else:
            assert m["airborne_at_blackout"] is True
    assert seen_grounded
    clean = run_mission(_scenario(duration_s=20.0, outages=outage, params={"target_class": "turn", "mode": "hold"}))
    assert clean.metrics["airborne_at_blackout"] is True
    assert 0.15 - 1e-9 <= clean.metrics["outage_hold_link_age_s"] <= clean.metrics["outage_reaction_hold_s"] + 1e-9
