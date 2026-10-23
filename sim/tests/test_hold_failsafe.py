"""Opt-in hold-in-place lost-link failsafe.

With ``lost_link_action="hold"`` a link that stays silent past
``land_after_s`` keeps the vehicle hovering in place (DEGRADED, HOLD) instead
of latching a landing.  It only lands if the silence reaches
``hold_land_after_s`` or the battery is low.  The default (``"land"``) must
behave exactly as before.
"""
import numpy as np
import pytest

from sim.adapters import SimAdapter
from sim.failsafe import (
    FailsafeAction,
    FailsafeConfig,
    FailsafeController,
    FailsafeState,
)

DT = 0.05
HOLD = {"lost_link_action": "hold", "hold_land_after_s": 5.0}


def _ages(blackout_start_s: float, blackout_s: float, end_s: float):
    """(time, link age) at 20 Hz: age 0 except during one scripted blackout."""
    steps = round(end_s / DT)
    out = []
    for i in range(1, steps + 1):
        t = round(i * DT, 10)
        in_blackout = blackout_start_s < t <= blackout_start_s + blackout_s + 1e-9
        age = round(t - blackout_start_s, 10) if in_blackout else 0.0
        out.append((t, age))
    return out


def test_default_config_is_unchanged():
    cfg = FailsafeConfig()
    assert cfg.lost_link_action == "land"
    # Registered runs log as_dict(); the default must not grow new keys.
    assert set(cfg.as_dict()) == {
        "degraded_action", "degraded_after_s", "land_after_s", "loss_window_packets",
        "low_battery_pct", "slow_speed_mps", "weak_loss_fraction", "weak_rule",
    }
    hold = FailsafeConfig(**HOLD)
    assert hold.as_dict()["lost_link_action"] == "hold"
    assert hold.as_dict()["hold_land_after_s"] == 5.0
    assert FailsafeConfig.from_dict(hold.as_dict()) == hold
    assert FailsafeConfig.from_dict(cfg.as_dict()) == cfg


def test_validation():
    with pytest.raises(ValueError):
        FailsafeConfig(lost_link_action="hover")
    with pytest.raises(ValueError):
        FailsafeConfig(lost_link_action="hold", hold_land_after_s=0.2)  # below land_after_s
    with pytest.raises(ValueError):
        FailsafeConfig(lost_link_action="hold", hold_land_after_s=float("inf"))


@pytest.mark.parametrize("ages", [
    [0.0] * 50,
    [0.0, 0.05, 0.1, 0.0, 0.12, 0.0],
    [0.0, 0.16, 0.29, 0.0],          # degraded but never reaches 0.30 s
])
def test_limiting_case_no_long_silence_is_identical(ages):
    """No silence of 0.30 s or more: hold mode makes exactly the same decisions."""
    for degraded_action in ("hold", "slow"):
        a = FailsafeController(FailsafeConfig(degraded_action=degraded_action))
        b = FailsafeController(FailsafeConfig(degraded_action=degraded_action, **HOLD))
        for i, age in enumerate(ages):
            da = a.observe(i * DT, link_age_s=age, battery_pct=90.0)
            db = b.observe(i * DT, link_age_s=age, battery_pct=90.0)
            assert da == db


def test_hand_check_one_1p5_s_blackout_controller():
    """Blackout from t = 2.0 to 3.5 s at 20 Hz.

    By hand: the link age reaches 0.15 s at t = 2.15 (degraded) and 0.30 s at
    t = 2.30.  Today the drone latches LANDING at t = 2.30 and never leaves it.
    With hold it stays DEGRADED with a HOLD action from 2.15 to 3.50, and the
    first fresh packet (t = 3.55) returns it to NOMINAL.
    """
    trace = _ages(2.0, 1.5, 5.0)
    land, hold = FailsafeController(), FailsafeController(FailsafeConfig(**HOLD))
    land_states, hold_states = {}, {}
    for t, age in trace:
        land_states[t] = land.observe(t, link_age_s=age, battery_pct=90.0)
        hold_states[t] = hold.observe(t, link_age_s=age, battery_pct=90.0)

    assert land_states[2.25].state is FailsafeState.DEGRADED
    assert land_states[2.3].state is FailsafeState.LANDING
    assert land_states[2.3].reason == "link_timeout"
    assert land_states[5.0].state is FailsafeState.LANDING   # latched, never resumes

    assert hold_states[2.1].state is FailsafeState.NOMINAL
    for t in (2.15, 2.25):
        assert hold_states[t].state is FailsafeState.DEGRADED
        assert hold_states[t].reason == "link_degraded"
    for t in (2.3, 3.0, 3.5):
        assert hold_states[t].state is FailsafeState.DEGRADED
        assert hold_states[t].reason == "link_lost_hold"
        assert hold_states[t].action is FailsafeAction.HOLD
    assert hold_states[3.55].state is FailsafeState.NOMINAL
    assert hold_states[3.55].changed
    assert all(d.state is not FailsafeState.LANDING for d in hold_states.values())


def test_hold_is_a_hold_even_when_degraded_action_is_slow():
    c = FailsafeController(FailsafeConfig(degraded_action="slow", **HOLD))
    assert c.observe(0.0, link_age_s=0.2, battery_pct=90.0).action is FailsafeAction.SLOW
    lost = c.observe(0.1, link_age_s=0.4, battery_pct=90.0)
    assert lost.reason == "link_lost_hold"
    assert lost.action is FailsafeAction.HOLD


def test_hold_ceiling_and_battery_still_land():
    c = FailsafeController(FailsafeConfig(**HOLD))
    assert c.observe(0.0, link_age_s=4.95, battery_pct=90.0).reason == "link_lost_hold"
    ceiling = c.observe(0.05, link_age_s=5.0, battery_pct=90.0)
    assert ceiling.state is FailsafeState.LANDING
    assert ceiling.reason == "hold_timeout"
    # latched: a fresh packet does not resume a landing
    assert c.observe(0.1, link_age_s=0.0, battery_pct=90.0).state is FailsafeState.LANDING

    c = FailsafeController(FailsafeConfig(**HOLD))
    low = c.observe(0.0, link_age_s=1.0, battery_pct=10.0)
    assert low.state is FailsafeState.LANDING and low.reason == "low_battery"

    c = FailsafeController(FailsafeConfig(**HOLD))
    assert c.observe(0.0, link_age_s=None, battery_pct=90.0).reason == "hold_timeout"


def _fly(config, *, blackout=True):
    """A point-mass drone flies toward x = 10 m at 1 m height; the ground
    station sends a setpoint every step while the link is up.  Optional
    1.5 s blackout from t = 2.0 s."""
    drone = SimAdapter("d1", initial_position_m=(0.0, 0.0, 1.0), dt_s=DT, failsafe_config=config)
    drone.connect()
    trace = _ages(2.0, 1.5 if blackout else 0.0, 8.0)
    log = []
    for t, age in trace:
        if age == 0.0:
            drone.send_setpoint(np.array([10.0, 0.0, 1.0]))
        state = drone.step(link_age_s=age)
        log.append((t, state))
    return log


def test_hand_check_one_1p5_s_blackout_adapter():
    """Same blackout through the point-mass adapter.

    Today: the drone starts landing at t = 2.30 and is on the floor by the end.
    Hold: it stops and stays at 1 m height during the blackout, then resumes
    toward x = 10 m once packets return.
    """
    land = dict(_fly(FailsafeConfig()))
    assert land[8.0].failsafe_state in {"landing", "landed"}
    assert land[8.0].position[2] < 0.05

    hold = dict(_fly(FailsafeConfig(**HOLD)))
    during = [hold[round(2.3 + k * DT, 10)] for k in range(round(1.2 / DT) + 1)]
    assert all(s.failsafe_state == "degraded" for s in during)
    assert all(abs(s.position[2] - 1.0) < 1e-6 for s in during)
    assert np.linalg.norm(hold[3.5].velocity) < 0.05            # stopped by the end of the hold
    x_at_end_of_blackout = hold[3.5].position[0]
    assert hold[8.0].failsafe_state == "nominal"
    assert hold[8.0].position[0] > x_at_end_of_blackout + 1.0   # resumed the mission
    assert not any(s.landed for s in hold.values())


def test_adapter_limiting_case_no_blackout_identical():
    a = _fly(FailsafeConfig(), blackout=False)
    b = _fly(FailsafeConfig(**HOLD), blackout=False)
    for (ta, sa), (tb, sb) in zip(a, b):
        assert ta == tb
        assert sa.as_dict() == sb.as_dict()


def test_hold_policy_mission_log_validates():
    """A mission run with the hold policy writes a log that passes the log schema."""
    from sim.missions.api import Scenario
    from sim.missions.engine import run_mission
    from sim.missions.log import validate_log

    scenario = Scenario.from_dict({"mission": "hover", "seed": 1, "duration_s": 0.5, "adapter": "point_mass",
                                   "preset": "clean", "policy": dict(HOLD), "params": {}})
    log = run_mission(scenario, log=True).log
    assert log["policy"]["lost_link_action"] == "hold"
    validate_log(log)
