"""Engine plumbing checks with a tiny hover mission defined here."""
import json

import numpy as np
import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission, stopping_buffer_m, target_path
from sim.missions.log import validate_log


import sim.missions.hover  # noqa: F401  (registers "hover")


def test_hover_reaches_goal_and_logs_validate():
    result = run_mission(Scenario(mission="hover", seed=1, duration_s=5.0), log=True)
    assert result.metrics["final_error_m"] < 0.05
    assert result.metrics["final_failsafe_state"] == "nominal"
    validate_log(result.log)
    assert len(result.log["frames"]["time_s"]) == 101  # 5 s at 20 Hz, both ends
    again = run_mission(Scenario(mission="hover", seed=1, duration_s=5.0), log=True)
    assert json.dumps(result.log, sort_keys=True) == json.dumps(again.log, sort_keys=True)


def test_permanent_blackout_lands_with_default_policy():
    scenario = Scenario(mission="hover", seed=1, duration_s=8.0,
                        outages=({"start_s": 2.0, "end_s": None, "streams": None},))
    result = run_mission(scenario, log=True)
    assert result.metrics["final_failsafe_state"] in {"landing", "landed"}
    kinds = [e["kind"] for e in result.log["events"]]
    assert "outage_start" in kinds and "failsafe_transition" in kinds
    first_land = next(e for e in result.log["events"]
                      if e["kind"] == "failsafe_transition" and e["detail"].get("state") == "landing")
    # The last command before 2.0 s left at 1.95 s and arrives 2 ms later;
    # with T = 0.30 s the landing starts at about 2.25 s (one 10 ms step later at most).
    assert 2.24 <= first_land["time_s"] <= 2.27


def test_unknown_params_and_mission_rejected():
    with pytest.raises(ValueError):
        run_mission(Scenario(mission="hover", seed=1, duration_s=1.0, params={"nope": 1}))
    with pytest.raises(ValueError):
        run_mission(Scenario(mission="no_such_mission", seed=1, duration_s=1.0))


def test_stopping_buffer_hand_value():
    # 1.5 m/s for (0.3 + 0.05) s = 0.525 m, plus 1.5^2 / 8 = 0.28125 m braking.
    assert stopping_buffer_m(1.5, 0.3) == pytest.approx(0.80625)


def test_target_path_is_placed_and_level():
    path = target_path("straight", 3, origin=(4.0, 5.0), heading_rad=0.0, altitude_m=1.5, speed_mps=0.5)
    p = path.position([0.0, 2.0])
    assert np.allclose(p[0], [4.0, 5.0, 1.5]) and np.allclose(p[1], [5.0, 5.0, 1.5])
    turn = target_path("turn", 3, origin=(8.0, 5.0), speed_mps=0.5)
    speeds = np.linalg.norm(turn.velocity(np.linspace(0, 10, 11)), axis=1)
    assert np.allclose(speeds, 0.5)
