"""Regression checks for the simulated adapter seam (geofence latch, floor,
offboard fixes, landing, kill switch, SLOW carrot, shared safety code).

Every vehicle here is simulated.  Expected values come from the written
contract and from hand arithmetic, not from the adapter code.  Measured
numbers (speed overshoot, fall time, landing time) are printed with ``-s``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.adapters import Geofence, OffboardPointMassAdapter, PositionFixHold, SimAdapter, SimulatedAdapterCore
from sim.authority import CommandKind
from sim.failsafe import FailsafeConfig, FailsafeController, FailsafeDecision, FailsafeState, SequenceLossWindow
from sim.hardware import load_hardware_profile
from sim.missions.physics_adapter import DEFAULT_PROFILE, PhysicsAdapter
from sim.vehicle import ActuatedQuadrotor

DT = 0.01
G = 9.81
GOAL_IN = (8.0, 5.0, 1.5)


def steps(seconds: float) -> int:
    return int(round(seconds / DT))


def arena() -> Geofence:
    return Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)


def make(kind: str, *, position=(2.0, 2.0, 1.5), **kwargs):
    """Build a connected simulated adapter of the requested kind."""
    if kind == "point_mass":
        adapter = SimAdapter("a", dt_s=DT, initial_position_m=position, **kwargs)
    elif kind == "point_mass_offboard":
        adapter = OffboardPointMassAdapter("a", dt_s=DT, initial_position_m=position, **kwargs)
    elif kind == "physics":
        adapter = PhysicsAdapter("a", dt_s=DT, initial_position_m=position, **kwargs)
    elif kind == "physics_offboard":
        adapter = PhysicsAdapter("a", dt_s=DT, initial_position_m=position, feedback="offboard", **kwargs)
    else:  # pragma: no cover - guards a typo in a parameter list
        raise ValueError(kind)
    adapter.connect()
    return adapter


def truth_velocity(adapter):
    if isinstance(adapter, PhysicsAdapter):
        return adapter.vehicle.drone.velocity
    return adapter.velocity


def fly(adapter, n, *, setpoint=None, link_age_s=0.0, **kwargs):
    """Resend the mission setpoint every step (its ttl is 0.5 s) and step."""
    states = []
    for _ in range(n):
        if setpoint is not None:
            adapter.send_setpoint(np.asarray(setpoint, dtype=float))
        states.append(adapter.step(DT, link_age_s=link_age_s, **kwargs))
    return states


def kinds(events, kind):
    return [event for event in events if event["kind"] == kind]


# 1. Geofence breach hold latches ---------------------------------------------

@pytest.mark.parametrize("kind", ("point_mass", "physics"))
def test_breach_hold_survives_return_inside_until_reset(kind):
    # The vehicle enters the first step 0.3 m outside the +x face, already
    # moving inward.  The breach hold brakes it, and the coast carries it back
    # inside the fence.  The hold must stay latched while it is inside.
    adapter = make(kind, position=(16.3, 5.0, 1.5), geofence=arena())
    truth_velocity(adapter)[:] = (-2.0, 0.0, 0.0)
    fence = arena()

    first = fly(adapter, 1, setpoint=GOAL_IN)[0]
    assert first.command_authority == "failsafe"
    assert adapter.geofence_latched is True

    inside_steps = 0
    reset_done = False
    for _ in range(steps(4.0)):
        state = fly(adapter, 1, setpoint=GOAL_IN)[0]
        assert state.command_authority == "failsafe", "breach hold released without a reset"
        assert adapter.authority.select(adapter.time_s).command.kind is CommandKind.HOLD
        if fence.contains(state.position):
            inside_steps += 1
        # Reset at the deepest point inside (velocity has turned around or stopped).
        if inside_steps >= 20 and truth_velocity(adapter)[0] >= 0.0:
            reset_done = True
            break
    assert inside_steps >= 20, "the vehicle never coasted back inside"
    assert reset_done

    events = adapter.drain_events()
    assert len(kinds(events, "geofence_breach")) == 1, "breach must be recorded once per episode"
    # The authority change to the breach hold is recorded in the breach step.
    changes = kinds(events, "authority_change")
    assert changes[0]["detail"]["source"] == "failsafe"
    assert changes[0]["time_s"] == pytest.approx(DT)
    assert [event["detail"]["source"] for event in changes] == ["failsafe"]

    adapter.reset_failsafe()
    assert adapter.geofence_latched is False
    after = fly(adapter, steps(1.0), setpoint=GOAL_IN)
    assert all(state.command_authority == "mission" for state in after)
    assert all(fence.contains(state.position) for state in after)
    assert after[-1].position[0] < 15.5
    assert kinds(adapter.drain_events(), "geofence_breach") == []


@pytest.mark.parametrize("kind", ("point_mass", "physics"))
def test_breach_after_reset_is_a_new_episode(kind):
    adapter = make(kind, position=(17.0, 5.0, 1.5), geofence=arena())
    fly(adapter, steps(1.0), setpoint=GOAL_IN)
    assert len(kinds(adapter.drain_events(), "geofence_breach")) == 1
    adapter.reset_failsafe()
    # Still outside: the next step starts a second episode.
    state = fly(adapter, 1, setpoint=GOAL_IN)[0]
    assert state.command_authority == "failsafe"
    fly(adapter, steps(0.5), setpoint=GOAL_IN)
    assert len(kinds(adapter.drain_events(), "geofence_breach")) == 1


def test_landing_outranks_a_latched_breach_hold():
    adapter = make("point_mass", position=(17.0, 5.0, 1.5), geofence=arena())
    fly(adapter, 5, setpoint=GOAL_IN)
    adapter.land()
    assert adapter.authority.select(adapter.time_s).command.kind is CommandKind.LAND
    for _ in range(steps(5.0)):
        if adapter.step(DT, link_age_s=0.0).failsafe_state == "landed":
            break
    assert adapter.get_state().landed is True


# 2 and 3. Offboard point mass: truth has the floor and drives safety checks --

def test_offboard_point_mass_truth_has_the_floor():
    adapter = make("point_mass_offboard", position=(2.0, 2.0, 1.0))
    adapter.set_position_fix((2.0, 2.0, 1.0), 0.0)  # one fix, then the camera goes quiet
    adapter.land()
    landed = False
    for _ in range(steps(3.0)):
        state = adapter.step(DT, link_age_s=0.0)
        assert state.position[2] >= 0.0, "true position went through the floor"
        if state.position[2] == 0.0:
            assert state.velocity[2] >= 0.0
        if state.landed:
            landed = True
            break
    # The stale fix still says 1 m up, so touchdown can only come from truth.
    assert landed
    assert adapter.get_state().failsafe_state == "landed"


@pytest.mark.parametrize("kind", ("point_mass_offboard", "physics_offboard"))
def test_offboard_breach_check_uses_truth_not_the_fix(kind):
    adapter = make(kind, position=(17.0, 5.0, 1.5), geofence=arena())
    adapter.set_position_fix(GOAL_IN, 0.0)  # the camera wrongly says it is inside
    state = fly(adapter, 1, setpoint=GOAL_IN)[0]
    assert state.command_authority == "failsafe"
    assert len(kinds(adapter.drain_events(), "geofence_breach")) == 1


def test_offboard_physics_touchdown_uses_truth():
    adapter = make("physics_offboard", position=(2.0, 2.0, 0.6))
    adapter.set_position_fix((2.0, 2.0, 0.6), 0.0)
    adapter.land()
    for _ in range(steps(10.0)):
        if adapter.step(DT, link_age_s=0.0).landed:
            break
    state = adapter.get_state()
    assert state.landed is True
    assert state.position[2] == 0.0
    assert adapter.position_fix_m[2] == pytest.approx(0.6)


# 4. Offboard before the first fix uses the initial position ---------------

@pytest.mark.parametrize("kind,seconds,bound", (("point_mass_offboard", 2.0, 3.5), ("physics_offboard", 3.0, 3.5)))
def test_offboard_without_fix_steers_from_initial_position(kind, seconds, bound):
    start = (2.0, 2.0, 1.5)
    adapter = make(kind, position=start)
    assert np.array_equal(adapter.position_fix_m, start)
    states = fly(adapter, steps(seconds), setpoint=(3.0, 2.0, 1.5))
    # The position loop believes it is still at x = 2, 1 m short of the goal,
    # so it keeps pushing +x and the true vehicle runs past x = 3.  A loop
    # that used truth would settle at x = 3.
    assert states[-1].position[0] > bound
    assert np.array_equal(adapter.position_fix_m, start)


def test_fix_velocity_comes_from_the_last_two_fixes():
    hold = PositionFixHold((0.0, 0.0, 1.0))
    assert np.array_equal(hold.velocity, np.zeros(3))
    hold.update((2.0, 2.0, 1.5), 0.0)
    # The initial position is not a fix, so one fix still means zero velocity.
    assert np.array_equal(hold.velocity, np.zeros(3))
    hold.update((2.1, 2.0, 1.4), 0.2)
    assert np.allclose(hold.velocity, (0.5, 0.0, -0.5))
    with pytest.raises(ValueError):
        hold.update((2.0, 2.0, 1.5), 0.1)


# 5. Physics after LANDED: rotors stopped, no controller, at rest on the floor -

class _CountingController:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def update_control(self, *args, **kwargs):
        self.calls += 1
        return self.inner.update_control(*args, **kwargs)


def test_physics_landed_is_unpowered_and_at_rest():
    adapter = make("physics", position=(2.0, 2.0, 0.4))
    adapter.land()
    for _ in range(steps(10.0)):
        if adapter.step(DT, link_age_s=0.0).landed:
            break
    assert adapter.get_state().failsafe_state == "landed"
    counter = _CountingController(adapter.vehicle.controller)
    adapter.vehicle.controller = counter
    rest = adapter.get_state().position.copy()
    assert rest[2] == 0.0
    for state in fly(adapter, steps(2.0), setpoint=(3.0, 3.0, 2.0)):
        assert state.failsafe_state == "landed"
        assert np.array_equal(adapter.vehicle.actuator.speed_rad_s, np.zeros(4))
        assert np.array_equal(state.velocity, np.zeros(3))
        assert np.array_equal(adapter.vehicle.drone.angular_velocity, np.zeros(3))
        assert np.array_equal(state.position, rest)
    assert counter.calls == 0


# 6. Physics LAND: carrot xy frozen where landing began -----------------------

def test_physics_land_carrot_xy_frozen_while_moving():
    adapter = make("physics")
    fly(adapter, steps(2.0), setpoint=(12.0, 2.0, 1.5))
    moving = adapter.get_state()
    assert moving.velocity[0] > 1.0  # landing starts at speed
    xy_at_land = moving.position[:2].copy()
    adapter.land()
    landed = None
    for _ in range(steps(15.0)):
        state = adapter.step(DT, link_age_s=0.0)
        if not state.landed:
            assert np.array_equal(adapter.carrot_m[:2], xy_at_land)
        if state.landed:
            landed = state
            break
    assert landed is not None
    # A carrot that followed the vehicle would let it coast about v / Kd
    # (1.5 / 1.5 = 1 m) past the landing point.
    assert np.linalg.norm(landed.position[:2] - xy_at_land) < 0.1


# 7. Floor contact while KILLED: inelastic, no slide ------------------------

def test_physics_kill_while_moving_stops_on_floor_without_sliding():
    adapter = make("physics")
    fly(adapter, steps(2.0), setpoint=(12.0, 2.0, 1.5))
    assert adapter.get_state().velocity[0] > 1.0
    adapter.kill()
    touch = None
    for _ in range(steps(1.0)):
        state = adapter.step(DT, link_age_s=0.0)
        if state.position[2] == 0.0:
            touch = state
            break
    assert touch is not None
    for state in fly(adapter, steps(1.0), setpoint=(12.0, 2.0, 1.5)):
        assert np.array_equal(state.velocity, np.zeros(3))
        assert np.array_equal(adapter.vehicle.drone.angular_velocity, np.zeros(3))
        assert np.array_equal(state.position, touch.position)


def test_physics_kill_keeps_vehicle_clock_in_step():
    adapter = make("physics")
    fly(adapter, steps(0.5), setpoint=(2.0, 2.0, 1.5))
    adapter.kill()
    fly(adapter, steps(1.0))
    assert adapter.vehicle.time_s == pytest.approx(adapter.time_s, abs=1e-9)


def test_physics_kill_fall_time_matches_free_fall():
    adapter = make("physics")
    fly(adapter, steps(1.0), setpoint=(2.0, 2.0, 1.5))
    z0 = float(adapter.get_state().position[2])
    t0 = adapter.time_s
    adapter.kill()
    while adapter.step(DT, link_age_s=0.0).position[2] > 0.0:
        assert adapter.time_s - t0 < 1.0
    fall = adapter.time_s - t0
    expected = math.sqrt(2.0 * z0 / G)  # 0.553 s from 1.5 m
    print(f"kill fall time {fall:.3f} s, hand value {expected:.3f} s")
    # The floor is checked once per 0.01 s step, so the first step at or
    # below the floor ends up to one step after the exact crossing.
    assert expected <= fall + 1e-9 <= expected + DT + 1e-9


# 8. SLOW limits the carrot only; the rigid body is never edited -------------

def _bare_vehicle(start):
    profile = load_hardware_profile(DEFAULT_PROFILE)
    vehicle = ActuatedQuadrotor(profile, initial_position=start, substeps=1)
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    return vehicle


def test_slow_does_not_touch_the_rigid_body():
    start = (2.0, 2.0, 1.5)
    adapter = make("physics", position=start, failsafe_config=FailsafeConfig(degraded_action="slow", slow_speed_mps=0.3))
    bare = _bare_vehicle(start)
    for _ in range(steps(4.0)):
        adapter.send_setpoint(np.array((12.0, 2.0, 1.5)))
        state = adapter.step(DT, link_age_s=0.2)
        assert state.failsafe_state == "degraded"
        bare.step(adapter.carrot_m, DT)
        # Same carrot sequence into an untouched vehicle gives the same truth.
        assert np.array_equal(state.position, bare.drone.position)
        assert np.array_equal(state.velocity, bare.drone.velocity)


def _max_speed(adapter, goal, n, **step_kwargs):
    peak = 0.0
    carrot_prev = adapter.carrot_m
    carrot_rate = 0.0
    for _ in range(n):
        adapter.send_setpoint(np.asarray(goal, dtype=float))
        state = adapter.step(DT, **step_kwargs)
        peak = max(peak, float(np.linalg.norm(state.velocity)))
        carrot_rate = max(carrot_rate, float(np.linalg.norm(adapter.carrot_m - carrot_prev)) / DT)
        carrot_prev = adapter.carrot_m
    return peak, carrot_rate


def test_true_speed_overshoot_is_measured():
    nominal = make("physics")
    peak, rate = _max_speed(nominal, (12.0, 2.0, 1.5), steps(6.0), link_age_s=0.0)
    print(f"nominal: carrot limit 1.5 m/s, carrot max {rate:.4f} m/s, true peak {peak:.4f} m/s, ratio {peak / 1.5:.3f}")
    assert rate <= 1.5 + 1e-9
    # True speed overshoots the carrot limit (about 21 % here); it is not
    # clipped.  The bound only flags a large change in the vehicle model.
    assert 1.5 < peak <= 1.3 * 1.5

    slow = make("physics", failsafe_config=FailsafeConfig(degraded_action="slow", slow_speed_mps=0.3))
    peak, rate = _max_speed(slow, (12.0, 2.0, 1.5), steps(6.0), link_age_s=0.2)
    print(f"slow: carrot limit 0.3 m/s, carrot max {rate:.4f} m/s, true peak {peak:.4f} m/s, ratio {peak / 0.3:.3f}")
    assert rate <= 0.3 + 1e-9
    # Lead limit 0.3 * min(Kd / Kp) = 0.3 * 2 / 3 = 0.2 m.  With the lead
    # saturated the x loop settles at Kp * lead / Kd = 2 * 0.2 / 1.5 = 0.267 m/s.
    assert peak <= 0.3 + 1e-6
    assert peak == pytest.approx(2.0 * 0.2 / 1.5, abs=0.01)


def test_slow_vertical_speed_respects_the_cap():
    slow = make("physics", failsafe_config=FailsafeConfig(degraded_action="slow", slow_speed_mps=0.3))
    peak, _ = _max_speed(slow, (2.0, 2.0, 3.5), steps(5.0), link_age_s=0.2)
    print(f"slow vertical: true peak {peak:.6f} m/s against 0.3 m/s")
    # z loop: Kp * lead / Kd = 3 * 0.2 / 2 = 0.3 m/s, approached from below.
    assert peak <= 0.3 + 1e-3


# Landing time --------------------------------------------------------------

def test_physics_landing_time_from_1_5_m():
    adapter = make("physics", position=(0.0, 0.0, 1.5))
    adapter.land()
    for _ in range(steps(15.0)):
        if adapter.step(DT, link_age_s=0.0).landed:
            break
    assert adapter.get_state().landed
    # Hand value: the carrot reaches the floor after 1.5 / 0.3 = 5 s; the z
    # loop then closes the remaining lag (Kd / Kp * 0.3 = 0.2 m) within a
    # couple of seconds.
    print(f"landing time from 1.5 m: {adapter.time_s:.2f} s")
    assert 5.0 <= adapter.time_s <= 7.5


# Shared safety code and small API checks -------------------------------------

def test_safety_methods_are_shared_not_copied():
    # restates implementation: a structural check that both adapters use one copy.
    for name in ("request_command", "send_setpoint", "land", "kill", "reset_failsafe",
                 "set_link_metrics", "set_battery_pct", "drain_events", "_sync_failsafe_authority",
                 "_observe_failsafe", "_event", "_record_authority"):
        assert name not in vars(SimAdapter) and name not in vars(PhysicsAdapter), name
        assert getattr(SimAdapter, name) is getattr(SimulatedAdapterCore, name)
        assert getattr(PhysicsAdapter, name) is getattr(SimulatedAdapterCore, name)


def test_physics_failsafe_properties_and_geofence_type():
    adapter = make("physics")
    assert adapter.failsafe_state is FailsafeState.NOMINAL
    assert isinstance(adapter.last_failsafe, FailsafeDecision)
    adapter.step(DT, link_age_s=0.2)
    assert adapter.failsafe_state is FailsafeState.DEGRADED
    assert adapter.last_failsafe.state is FailsafeState.DEGRADED
    with pytest.raises(ValueError):
        PhysicsAdapter("bad", geofence=((0, 0, 0), (4, 4, 4)))
    with pytest.raises(ValueError):
        SimAdapter("bad", geofence="arena")


def test_kill_event_recorded_once():
    for kind in ("point_mass", "physics"):
        adapter = make(kind)
        fly(adapter, 5, setpoint=(2.0, 2.0, 1.5))
        adapter.kill()
        adapter.kill()
        fly(adapter, 5, setpoint=(2.0, 2.0, 1.5), kill_requested=True)
        assert len(kinds(adapter.drain_events(), "kill")) == 1, kind


# Failsafe helper fixes -------------------------------------------------------

def test_loss_window_rejects_an_old_window_after_pruning():
    window = SequenceLossWindow(20)
    for seq in range(100):
        window.record(seq)
    assert window.loss_fraction(100) == 0.0
    with pytest.raises(ValueError):
        window.loss_fraction(50)  # numbers below 59 were already pruned
    with pytest.raises(ValueError):
        window.loss_fraction(99)  # equal to the newest: window would end at 98


def test_loss_window_reordered_packets_count_as_lost_and_startup_is_partial():
    window = SequenceLossWindow(20)
    window.record(0)
    window.record(2)
    window.record(1)  # arrives after 2, ignored
    # Expected 0, 1, 2 so far; 1 is counted missing: 1 / 3, not 1 / 20.
    assert window.loss_fraction() == pytest.approx(1.0 / 3.0)
    assert window.loss_fraction(5) == pytest.approx(3.0 / 5.0)


def test_recent_loss_fraction_rejects_bool():
    controller = FailsafeController(FailsafeConfig(weak_rule="age_or_loss"))
    with pytest.raises(ValueError):
        controller.observe(0.0, link_age_s=0.0, battery_pct=100.0, recent_loss_fraction=True)
    assert controller.observe(0.0, link_age_s=0.0, battery_pct=100.0, recent_loss_fraction=1.0).state is FailsafeState.DEGRADED
