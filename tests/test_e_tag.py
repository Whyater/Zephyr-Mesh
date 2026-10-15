"""Run E tag mission checks for the point-mass and physics adapters."""
from __future__ import annotations

import json
import math
import time

import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log
from tests.reference_e.hand_checks import tail_follow_time_s


def _scenario(seed=20261030, **changes):
    values = dict(mission="tag", seed=seed, duration_s=30.0, adapter="point_mass", preset="clean")
    values.update(changes)
    return Scenario(**values)


def test_hand_check_straight_recovery():
    result = run_mission(_scenario(params={"target_class": "straight", "arm": "recovery"}))
    entry = result.metrics["time_to_tag_s"] - 2.0
    reference = tail_follow_time_s(4.0, 1.0, 1.5, 0.5)
    print(f"zone entry={entry:.3f}s reference={reference:.3f}s tag={result.metrics['time_to_tag_s']}")
    assert reference == 3.0
    assert 3.0 <= entry <= 4.2
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


@pytest.mark.xfail(strict=True, reason="default physics controller does not sustain the straight tag zone for two seconds")
def test_physics_adapter_clean_straight():
    started = time.perf_counter()
    result = run_mission(_scenario(seed=20261030, duration_s=10.0, adapter="physics", params={"target_class": "straight"}))
    elapsed = time.perf_counter() - started
    print(f"physics straight wall={elapsed:.3f}s tagged={result.metrics['tagged']} time={result.metrics['time_to_tag_s']}")
    assert result.metrics["tagged"] is True
    assert math.isfinite(elapsed)
