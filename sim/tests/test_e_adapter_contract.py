"""DroneAdapter contract tests shared by every simulated adapter.

Every adapter (the point mass ``SimAdapter`` and the rigid-body
``PhysicsAdapter``) must pass the same behaviour checks: holding and landing,
the kill switch, the geofence, command authority, honest link age, the weak
link SLOW policy, determinism, and the low battery landing.  All vehicles here
are simulated.  Expected values come from the written contract and from hand
arithmetic (free fall time sqrt(2 h / g), the clipped geofence corner, the
failsafe thresholds), never from the adapter code.

The two adapters differ on purpose after a kill switch press: the point mass
freezes in place (it has no gravity) and the physics adapter falls.  The shared
tests only assert what both must do; the fall is checked in a physics-only test.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.authority import CommandSource, ControlCommand
from sim.failsafe import FailsafeConfig, FailsafeState

DT = 0.01
START = (2.0, 2.0, 1.5)
G = 9.81
ADAPTER_KINDS = ("point_mass", "physics")
ADAPTER_EVENT_KINDS = {
    "failsafe_transition",
    "authority_change",
    "geofence_clip",
    "geofence_breach",
    "kill",
    "landed",
}
LEGACY_STATE_KEYS = {
    "position",
    "velocity",
    "battery_pct",
    "link_delay_ms",
    "link_loss_pct",
    "timestamp_s",
    "connected",
    "command_authority",
    "failsafe_state",
}


def steps(seconds: float) -> int:
    return int(round(seconds / DT))


def arena_fence():
    from sim.adapters import Geofence

    return Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)


def make_adapter(kind, adapter_id="d1", *, position=START, geofence=None, failsafe_config=None):
    """Build a connected simulated adapter of the requested kind."""
    kwargs = {
        "dt_s": DT,
        "initial_position_m": position,
        "failsafe_config": failsafe_config,
    }
    if geofence is not None:
        kwargs["geofence"] = geofence
    if kind == "point_mass":
        from sim.adapters import SimAdapter

        adapter = SimAdapter(adapter_id, **kwargs)
    elif kind == "physics":
        from sim.missions.physics_adapter import PhysicsAdapter

        adapter = PhysicsAdapter(adapter_id, **kwargs)
    else:  # pragma: no cover - guards a typo in the parameter list
        raise ValueError(kind)
    adapter.connect()
    return adapter


def fly(adapter, n_steps, *, setpoint=None, link_age_s=0.0, **step_kwargs):
    """Resend the mission setpoint every step (its ttl is 0.5 s) and step."""
    states = []
    for _ in range(n_steps):
        if setpoint is not None:
            adapter.send_setpoint(np.asarray(setpoint, dtype=float))
        states.append(adapter.step(DT, link_age_s=link_age_s, **step_kwargs))
    return states


@pytest.fixture(params=ADAPTER_KINDS)
def kind(request):
    return request.param


# 1. hold, move, land ---------------------------------------------------------

def test_position_hold_and_land(kind):
    adapter = make_adapter(kind)
    goal = np.array((4.0, 2.0, 1.5))
    state = fly(adapter, steps(10.0), setpoint=goal)[-1]
    assert np.linalg.norm(state.position - goal) <= 0.15

    decision = adapter.land()
    assert decision.state is FailsafeState.LANDING
    assert adapter.get_state().failsafe_state == "landing"
    assert adapter.get_state().command_authority == "failsafe"
    # Landing is latched: it cannot be reset before touchdown.
    with pytest.raises(ValueError):
        adapter.reset_failsafe()

    # A mission setpoint somewhere else must not steer the landing.
    elsewhere = (8.0, 5.0, 2.0)
    xy_at_land = state.position[:2].copy()
    landed_state = None
    for _ in range(steps(15.0)):
        adapter.send_setpoint(np.asarray(elsewhere))
        current = adapter.step(DT, link_age_s=0.0)
        assert np.linalg.norm(current.position[:2] - xy_at_land) < 0.15
        if current.failsafe_state == "landed":
            landed_state = current
            break
    assert landed_state is not None, "vehicle did not report landed within 15 s"
    assert landed_state.position[2] <= 0.05
    assert adapter.get_state().landed is True

    rest = landed_state.position.copy()
    after = fly(adapter, steps(2.0), setpoint=elsewhere)
    for current in after:
        assert np.linalg.norm(current.position - rest) <= 0.05
        assert current.failsafe_state == "landed"
        assert current.landed is True


# 2. kill switch --------------------------------------------------------------

@pytest.mark.parametrize("trigger", ("call", "flag"))
def test_kill_switch_stops_and_latches(kind, trigger):
    adapter = make_adapter(kind)
    hover = fly(adapter, steps(1.0), setpoint=START)[-1]

    if trigger == "call":
        decision = adapter.kill()
        assert decision.state is FailsafeState.KILLED
        assert adapter.get_state().failsafe_state == "killed"
        assert adapter.get_state().command_authority == "failsafe"
        first = adapter.step(DT, link_age_s=0.0)
    else:
        first = adapter.step(DT, link_age_s=0.0, kill_requested=True)
    assert first.failsafe_state == "killed"

    # Mission setpoints, even upward ones, are ignored after the kill switch.
    upward = (2.0, 2.0, 3.0)
    z_prev = first.position[2]
    z_ceiling = hover.position[2]
    assert z_prev <= z_ceiling + 1e-9
    states = fly(adapter, steps(2.0), setpoint=upward)
    for current in states:
        assert current.failsafe_state == "killed"
        assert current.command_authority == "failsafe"
        assert current.position[2] <= z_prev + 1e-9, "vehicle moved up after the kill switch"
        z_prev = current.position[2]
        horizontal = np.linalg.norm(current.position[:2] - hover.position[:2])
        assert horizontal < 0.5

    # Latched until an explicit reset with a healthy link.
    adapter.step(DT, link_age_s=0.2)
    assert adapter.get_state().failsafe_state == "killed"
    with pytest.raises(ValueError):
        adapter.reset_failsafe()
    assert adapter.get_state().failsafe_state == "killed"
    adapter.step(DT, link_age_s=0.0)
    assert adapter.get_state().failsafe_state == "killed"
    reset = adapter.reset_failsafe()
    assert reset.state is FailsafeState.NOMINAL
    assert adapter.get_state().failsafe_state == "nominal"


def test_physics_kill_switch_falls_like_free_fall():
    adapter = make_adapter("physics")
    hover = fly(adapter, steps(1.0), setpoint=START)[-1]
    z0 = float(hover.position[2])
    vz0 = float(hover.velocity[2])
    t0 = adapter.time_s
    # Hand check: z(t) = z0 + vz0 t - g t^2 / 2 reaches 0 at
    # t = (vz0 + sqrt(vz0^2 + 2 g z0)) / g; for z0 = 1.5, vz0 = 0 that is 0.553 s.
    expected_fall_s = (vz0 + math.sqrt(vz0 * vz0 + 2.0 * G * z0)) / G
    assert abs(expected_fall_s - math.sqrt(2.0 * 1.5 / G)) < 0.01

    adapter.kill()
    floor_time = None
    checked_mid = False
    for _ in range(steps(1.5)):
        state = adapter.step(DT, link_age_s=0.0)
        elapsed = adapter.time_s - t0
        if not checked_mid and elapsed >= 0.3 - 1e-9:
            expected_z = z0 + vz0 * elapsed - 0.5 * G * elapsed * elapsed
            assert abs(state.position[2] - expected_z) < 0.03
            checked_mid = True
        if floor_time is None and state.position[2] <= 1e-6:
            floor_time = elapsed
        if floor_time is not None:
            assert state.position[2] <= 1e-6
            assert state.position[2] >= -1e-9
    assert checked_mid
    assert floor_time is not None, "vehicle never reached the floor after the kill switch"
    assert expected_fall_s - 0.02 <= floor_time <= expected_fall_s + 0.05


# 3. geofence -----------------------------------------------------------------

def test_geofence_box_contains_and_clips():
    fence = arena_fence()
    assert fence.contains((0.0, 0.0, 0.0))
    assert fence.contains((16.0, 10.0, 4.0))
    assert not fence.contains((16.001, 5.0, 2.0))
    assert not fence.contains((8.0, 5.0, -0.001))
    # The clip box is the arena shrunk by 0.5 m on every face:
    # x [0.5, 15.5], y [0.5, 9.5], z [0.5, 3.5].
    assert np.allclose(fence.clip((40.0, -20.0, 9.0)), (15.5, 0.5, 3.5))
    assert np.allclose(fence.clip((8.0, 5.0, 2.0)), (8.0, 5.0, 2.0))
    assert np.allclose(fence.clip((0.2, 9.9, 0.1)), (0.5, 9.5, 0.5))
    json.dumps(fence.as_dict(), allow_nan=False)

    from sim.adapters import Geofence

    with pytest.raises(ValueError):
        Geofence((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), margin_m=0.5)


def test_geofence_keeps_vehicle_inside(kind):
    fence = arena_fence()
    adapter = make_adapter(kind, geofence=fence)
    far = np.array((40.0, -20.0, 9.0))
    clipped = np.array((15.5, 0.5, 3.5))

    decision = adapter.send_setpoint(far)
    assert decision.accepted is True
    assert np.allclose(decision.command.position_m, clipped)

    states = fly(adapter, steps(20.0), setpoint=far)
    for state in states:
        assert fence.contains(state.position), f"left the fence at {state.position}"
    assert np.linalg.norm(states[-1].position - clipped) <= 0.2

    kinds = [event["kind"] for event in adapter.drain_events()]
    assert "geofence_clip" in kinds
    assert "geofence_breach" not in kinds


def test_geofence_breach_holds_until_reset(kind):
    # The vehicle starts outside the arena, so the first step is a breach.
    fence = arena_fence()
    outside = (17.0, 5.0, 1.5)
    adapter = make_adapter(kind, position=outside, geofence=fence)
    inside_goal = (8.0, 5.0, 1.5)
    states = fly(adapter, steps(2.0), setpoint=inside_goal)
    for state in states:
        assert state.command_authority == "failsafe"
        assert np.linalg.norm(state.position - np.asarray(outside)) < 0.1
    kinds = [event["kind"] for event in adapter.drain_events()]
    assert "geofence_breach" in kinds


# 4. command authority --------------------------------------------------------

def test_manual_hold_overrides_mission_until_ttl(kind):
    adapter = make_adapter(kind)
    goal = np.array((4.0, 2.0, 1.5))
    adapter.send_setpoint(goal)
    manual = adapter.request_command(ControlCommand.hold(), source=CommandSource.MANUAL, ttl_s=1.0)
    assert manual.accepted is True
    assert manual.source is CommandSource.MANUAL

    held = fly(adapter, steps(0.9), setpoint=goal)
    for state in held:
        assert state.command_authority == "manual"
        assert np.linalg.norm(state.position - np.asarray(START)) < 0.1

    resumed = fly(adapter, steps(10.0), setpoint=goal)
    assert resumed[-1].command_authority == "mission"
    assert np.linalg.norm(resumed[-1].position - goal) <= 0.15


@pytest.mark.parametrize("command", ("land", "kill"))
def test_mission_cannot_send_safety_commands(kind, command):
    adapter = make_adapter(kind)
    fly(adapter, steps(0.2), setpoint=START)
    safety = ControlCommand.land() if command == "land" else ControlCommand.kill()
    decision = adapter.request_command(safety, source=CommandSource.MISSION)
    assert decision.accepted is False
    assert decision.reason == "safety_command_requires_failsafe"
    state = fly(adapter, steps(0.5), setpoint=START)[-1]
    assert state.failsafe_state == "nominal"
    assert state.command_authority == "mission"
    assert abs(state.position[2] - START[2]) < 0.1


# 5. telemetry age ------------------------------------------------------------

@pytest.mark.parametrize(
    "age,expected",
    (
        (0.0, "nominal"),
        (0.14, "nominal"),
        (0.15, "degraded"),
        (0.29, "degraded"),
        (0.30, "landing"),
        (0.5, "landing"),
    ),
    ids=("a000", "a014", "a015", "a029", "a030", "a050"),
)
def test_link_age_is_reported_and_mapped(kind, age, expected):
    adapter = make_adapter(kind)
    fly(adapter, 5, setpoint=START)
    adapter.send_setpoint(np.asarray(START))
    state = adapter.step(DT, link_age_s=age)
    assert state.link_age_s == age
    assert adapter.get_state().link_age_s == age
    assert state.failsafe_state == expected
    if expected != "nominal":
        assert state.command_authority == "failsafe"


def test_unknown_link_age_means_landing(kind):
    adapter = make_adapter(kind)
    fly(adapter, 5, setpoint=START)
    adapter.set_link_metrics(delay_ms=0.0, loss_pct=0.0, age_s=None)
    state = adapter.step(DT)
    assert state.failsafe_state == "landing"
    assert state.link_age_s is None


def test_degraded_link_recovers(kind):
    adapter = make_adapter(kind)
    fly(adapter, 5, setpoint=START)
    assert fly(adapter, 1, setpoint=START, link_age_s=0.2)[-1].failsafe_state == "degraded"
    assert fly(adapter, 1, setpoint=START, link_age_s=0.0)[-1].failsafe_state == "nominal"


def test_timestamps_follow_adapter_clock(kind):
    adapter = make_adapter(kind)
    previous = adapter.get_state().timestamp_s
    for i in range(1, 51):
        adapter.send_setpoint(np.asarray(START))
        state = adapter.step(DT, link_age_s=0.0)
        assert state.timestamp_s >= previous
        assert state.timestamp_s == adapter.time_s
        assert adapter.get_state().timestamp_s == adapter.time_s
        assert abs(state.timestamp_s - i * DT) < 1e-9
        previous = state.timestamp_s
    with pytest.raises(ValueError):
        adapter.request_command(
            ControlCommand.hold(),
            source=CommandSource.MANUAL,
            timestamp_s=adapter.time_s - 0.05,
        )


def test_state_dict_keys(kind):
    adapter = make_adapter(kind)
    state = fly(adapter, 3, setpoint=START)[-1]
    assert set(state.as_dict()) == LEGACY_STATE_KEYS
    mission = state.as_mission_dict()
    assert set(mission) == LEGACY_STATE_KEYS | {"link_age_s", "landed"}
    assert mission["link_age_s"] == 0.0
    assert mission["landed"] is False
    json.dumps(mission, allow_nan=False)


# 6. weak link SLOW -----------------------------------------------------------

def test_weak_link_slow_caps_speed(kind):
    config = FailsafeConfig(degraded_action="slow", slow_speed_mps=0.3)
    adapter = make_adapter(kind, failsafe_config=config)
    far = np.array((12.0, 2.0, 1.5))
    start = adapter.get_state().position.copy()
    previous = start.copy()
    # The contract window is the first 1 s; the cap is checked for 4 s so the
    # progress check below has room for a vehicle that needs time to tilt.
    states = fly(adapter, steps(4.0), setpoint=far, link_age_s=0.2)
    for state in states:
        assert state.failsafe_state == "degraded"
        # SLOW leaves the mission in charge; only the speed is capped.
        assert state.command_authority == "mission"
        assert np.linalg.norm(state.velocity) <= 0.3 + 0.05
        assert np.linalg.norm(state.position - previous) / DT <= 0.3 + 0.05
        previous = state.position.copy()
    # It still makes progress toward the goal, unlike a hold: at 0.3 m/s the
    # allowed travel in 4 s is 1.2 m, and a hold would move about 0 m.  Ask
    # for at least 0.5 m so a lagging but moving vehicle passes.
    assert states[-1].position[0] - start[0] >= 0.5


# 7. determinism --------------------------------------------------------------

def _scripted_run(kind):
    adapter = make_adapter(kind, "det")
    goal = np.array((4.0, 3.0, 2.0))
    trace = []
    for i in range(300):
        if i == 200:
            adapter.request_command(ControlCommand.hold(), source=CommandSource.MANUAL, ttl_s=0.3)
        adapter.send_setpoint(goal)
        age = 0.2 if 100 <= i < 150 else 0.0
        trace.append(adapter.step(DT, link_age_s=age).as_mission_dict())
    return trace, adapter.drain_events()


def test_same_inputs_same_states(kind):
    first_states, first_events = _scripted_run(kind)
    second_states, second_events = _scripted_run(kind)
    assert first_states == second_states
    assert first_events == second_events


# 8. low battery --------------------------------------------------------------

def test_low_battery_lands(kind):
    adapter = make_adapter(kind)
    fly(adapter, 5, setpoint=START)
    assert fly(adapter, 1, setpoint=START, battery_pct=15.1)[-1].failsafe_state == "nominal"
    state = fly(adapter, 1, setpoint=START, battery_pct=15.0)[-1]
    assert state.failsafe_state == "landing"
    assert state.battery_pct == 15.0
    assert state.command_authority == "failsafe"
    # Latched: a recovered battery reading does not resume the mission.
    assert fly(adapter, 1, setpoint=START, battery_pct=100.0)[-1].failsafe_state == "landing"
    with pytest.raises(ValueError):
        adapter.reset_failsafe()


# Event records ---------------------------------------------------------------

def _check_event_shape(events, adapter):
    previous = -math.inf
    for event in events:
        assert set(event) == {"time_s", "kind", "agent", "detail"}
        assert event["kind"] in ADAPTER_EVENT_KINDS
        assert event["agent"] == adapter.adapter_id
        assert isinstance(event["detail"], dict)
        assert math.isfinite(event["time_s"])
        assert previous <= event["time_s"] <= adapter.time_s + 1e-12
        previous = event["time_s"]
        json.dumps(event, allow_nan=False, sort_keys=True)


def test_events_record_authority_and_kill(kind):
    adapter = make_adapter(kind, "ev1")
    fly(adapter, steps(0.5), setpoint=START)
    adapter.request_command(ControlCommand.hold(), source=CommandSource.MANUAL, ttl_s=0.2)
    fly(adapter, steps(0.1), setpoint=START)
    adapter.kill()
    fly(adapter, steps(0.2), setpoint=START)
    events = adapter.drain_events()
    _check_event_shape(events, adapter)
    kinds = {event["kind"] for event in events}
    assert {"authority_change", "failsafe_transition", "kill"} <= kinds
    transitions = [e["detail"].get("state") for e in events if e["kind"] == "failsafe_transition"]
    assert "killed" in transitions
    assert adapter.drain_events() == []


def test_events_record_landing(kind):
    low = (2.0, 2.0, 0.3)
    adapter = make_adapter(kind, "ev2", position=low)
    fly(adapter, steps(0.2), setpoint=low)
    adapter.land()
    for _ in range(steps(10.0)):
        if adapter.step(DT, link_age_s=0.0).failsafe_state == "landed":
            break
    assert adapter.get_state().failsafe_state == "landed"
    events = adapter.drain_events()
    _check_event_shape(events, adapter)
    assert "landed" in {event["kind"] for event in events}
    transitions = [e["detail"].get("state") for e in events if e["kind"] == "failsafe_transition"]
    assert "landing" in transitions
    assert "landed" in transitions
