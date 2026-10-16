"""Checks for the simulated adapter seam added after a second code review.

Every vehicle here is simulated.  Expected values come from the written
contract and from hand arithmetic (PD loop response, carrot step sizes),
not from the adapter code.  Measured numbers are printed with ``-s``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.adapters import Geofence, OffboardPointMassAdapter, SimAdapter
from sim.authority import CommandKind, CommandSource, ControlCommand
from sim.failsafe import FailsafeConfig
from sim.missions.physics_adapter import PhysicsAdapter

DT = 0.01


def steps(seconds: float) -> int:
    return int(round(seconds / DT))


def arena() -> Geofence:
    return Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)


def make(kind: str, *, position=(2.0, 2.0, 1.5), **kwargs):
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


def truth(adapter) -> np.ndarray:
    if isinstance(adapter, PhysicsAdapter):
        return adapter.vehicle.drone.position
    return adapter.position


def truth_velocity(adapter) -> np.ndarray:
    if isinstance(adapter, PhysicsAdapter):
        return adapter.vehicle.drone.velocity
    return adapter.velocity


def raw_mission_goal(adapter, goal) -> None:
    """A mission position request that skips send_setpoint (and so the clip)."""
    adapter.request_command(ControlCommand.position(goal), source=CommandSource.MISSION, ttl_s=0.5)


# 1. Geofence: the step that crosses a face latches the hold ------------------

FACES = {
    # start, goal far past the face, axis, face value, outward sign
    "plus_x": ((14.5, 5.0, 1.5), (40.0, 5.0, 1.5), 0, 16.0, 1.0),
    "minus_y": ((8.0, 1.5, 1.5), (8.0, -30.0, 1.5), 1, 0.0, -1.0),
    "plus_z": ((8.0, 5.0, 2.5), (8.0, 5.0, 30.0), 2, 4.0, 1.0),
}


@pytest.mark.parametrize("kind", ("point_mass", "physics"))
@pytest.mark.parametrize("face", sorted(FACES))
def test_breach_latches_on_the_step_that_crosses(kind, face):
    start, goal, axis, value, sign = FACES[face]
    fence = arena()
    adapter = make(kind, position=start, geofence=fence)
    crossing = None
    for _ in range(steps(5.0)):
        if not adapter.geofence_latched:
            raw_mission_goal(adapter, goal)
        state = adapter.step(DT, link_age_s=0.0)
        if not fence.contains(state.position):
            crossing = state
            break
    assert crossing is not None, "the vehicle never left the box"
    speed = float(np.linalg.norm(crossing.velocity))
    print(f"{kind} {face}: crossed at t={crossing.timestamp_s:.2f} s, "
          f"{sign * (crossing.position[axis] - value):.4f} m past the face at {speed:.3f} m/s")
    # Driven at full speed: the box is crossed well above walking pace.
    assert speed > 1.0
    # Same step: latched, safety-owned, and the event names the true position.
    assert adapter.geofence_latched
    assert crossing.command_authority == "failsafe"
    breaches = [event for event in adapter.drain_events() if event["kind"] == "geofence_breach"]
    assert len(breaches) == 1
    assert breaches[0]["time_s"] == pytest.approx(crossing.timestamp_s)
    assert np.allclose(breaches[0]["detail"]["position_m"], crossing.position)


def test_geofence_outside_means_outside_the_box_not_the_margin():
    fence = arena()
    # Boundary inclusive: a point on a face is inside.
    assert fence.contains((16.0, 10.0, 4.0)) and fence.contains((0.0, 0.0, 0.0))
    assert not fence.contains((16.0 + 1e-9, 5.0, 1.5))
    # The margin only shrinks the clip box: 15.8 is inside but past the clip.
    assert fence.contains((15.8, 5.0, 1.5))
    assert np.allclose(fence.clip((15.8, 5.0, 1.5)), (15.5, 5.0, 1.5))
    adapter = make("point_mass", position=(15.8, 5.0, 1.5), geofence=fence)
    adapter.step(DT, link_age_s=0.0)
    assert not adapter.geofence_latched


# 2. LAND -----------------------------------------------------------------------

def test_offboard_physics_land_without_fixes_stays_where_it_began():
    """Fixes stop (the usual reason for a landing) while the vehicle moves.

    The landing is the vehicle's own failsafe, so it must not depend on a
    laptop loop that has gone quiet.  Before the fix the held fix and its
    held velocity drove the body about 260 m along the floor.
    """
    adapter = make("physics_offboard")
    for _ in range(steps(2.0)):
        adapter.set_position_fix(truth(adapter).copy(), adapter.time_s)
        adapter.send_setpoint(np.array((12.0, 2.0, 1.5)))
        adapter.step(DT, link_age_s=0.0)
    entry_speed = float(np.linalg.norm(truth_velocity(adapter)))
    start_xy = truth(adapter)[:2].copy()
    assert entry_speed > 1.0
    adapter.land()
    excursion = 0.0
    landed = None
    for _ in range(steps(15.0)):
        state = adapter.step(DT, link_age_s=0.0)
        excursion = max(excursion, float(np.linalg.norm(state.position[:2] - start_xy)))
        if state.landed:
            landed = state
            break
    print(f"offboard land, no fixes: entry {entry_speed:.3f} m/s, max excursion {excursion:.3f} m")
    assert landed is not None
    # Same bounds as the onboard landing below (PD braking, then back).
    assert excursion <= 1.5 * 0.376 * entry_speed
    assert np.linalg.norm(landed.position[:2] - start_xy) < 0.1


def test_offboard_point_mass_land_flies_like_the_onboard_one():
    """A stale fix must not change the landing: it is flown on the true state.

    Before the fix the held fix kept the height error at 1.5 m, so the point
    mass came down at its 2 m/s speed limit all the way to the floor.
    """
    onboard = make("point_mass")
    offboard = make("point_mass_offboard")
    offboard.set_position_fix((3.0, 2.0, 1.5), 0.0)  # last fix, 1 m off, then silence
    for adapter in (onboard, offboard):
        adapter.land()
    for _ in range(steps(3.0)):
        a = onboard.step(DT, link_age_s=0.0)
        b = offboard.step(DT, link_age_s=0.0)
        assert np.array_equal(a.position, b.position)
        assert np.array_equal(a.velocity, b.velocity)
    assert b.landed and np.allclose(b.position, (2.0, 2.0, 0.0))


def test_physics_land_while_moving_brakes_like_the_pd_loop():
    """A moving body cannot stop in place; it brakes back to the frozen carrot.

    Hand value for the x loop alone (Kp = 2, Kd = 1.5, carrot frozen, start
    speed v0): x(t) = v0 / wd * exp(-zeta w t) * sin(wd t) with w = sqrt(2),
    zeta = 1.5 / (2 w) = 0.530, wd = w sqrt(1 - zeta^2) = 1.199.  The peak is
    at t = atan(wd / (zeta w)) / wd = 0.844 s and equals 0.376 * v0.  The
    attitude loop adds lag, so the real peak is somewhat larger.
    """
    adapter = make("physics")
    for _ in range(steps(2.0)):
        adapter.send_setpoint(np.array((12.0, 2.0, 1.5)))
        adapter.step(DT, link_age_s=0.0)
    v0 = float(truth_velocity(adapter)[0])
    x0 = float(truth(adapter)[0])
    adapter.land()
    xs = []
    for _ in range(steps(15.0)):
        state = adapter.step(DT, link_age_s=0.0)
        xs.append(float(state.position[0]) - x0)
        if state.landed:
            break
    xs = np.asarray(xs)
    peak = float(xs.max())
    peak_t = (int(xs.argmax()) + 1) * DT
    hand = 0.376 * v0
    print(f"land at {v0:.3f} m/s: peak {peak:.3f} m at {peak_t:.2f} s, hand {hand:.3f} m at 0.84 s, final {xs[-1]:.4f} m")
    assert abs(peak_t - 0.844) <= 0.15
    assert hand <= peak <= 1.5 * hand
    assert abs(xs[-1]) < 0.1


# 3. Floor contact: no sliding --------------------------------------------------

@pytest.mark.parametrize("kind", ("point_mass", "point_mass_offboard", "physics", "physics_offboard"))
def test_powered_vehicle_on_the_floor_does_not_slide(kind):
    start = (2.0, 2.0, 0.0)  # exactly on the floor
    adapter = make(kind, position=start)
    for _ in range(steps(1.0)):
        adapter.send_setpoint(np.array((6.0, 2.0, -1.0)))  # sideways and into the floor
        state = adapter.step(DT, link_age_s=0.0)
        assert state.position[2] == 0.0
        assert np.array_equal(state.position[:2], np.asarray(start[:2])), state.position
        assert np.array_equal(state.velocity, np.zeros(3)), state.velocity


@pytest.mark.parametrize("kind", ("point_mass", "physics"))
def test_vehicle_that_reaches_the_floor_stops_there(kind):
    adapter = make(kind, position=(2.0, 2.0, 0.05))
    touch = None
    for _ in range(steps(2.0)):
        adapter.send_setpoint(np.array((12.0, 2.0, -1.0)))
        state = adapter.step(DT, link_age_s=0.0)
        if touch is None and state.position[2] == 0.0:
            touch = state
        if touch is not None:
            assert np.array_equal(state.position, touch.position)
            assert np.array_equal(state.velocity, np.zeros(3))
    assert touch is not None


@pytest.mark.parametrize("kind", ("point_mass", "physics"))
def test_take_off_from_the_floor_still_works(kind):
    adapter = make(kind, position=(2.0, 2.0, 0.0))
    goal = np.array((4.0, 2.0, 1.5))
    for _ in range(steps(8.0)):
        adapter.send_setpoint(goal)
        state = adapter.step(DT, link_age_s=0.0)
    assert np.linalg.norm(state.position - goal) < 0.05


def test_physics_kill_at_exact_ground_does_not_slide():
    adapter = make("physics", position=(2.0, 2.0, 0.0))
    adapter.vehicle.drone.velocity[:] = (1.0, 0.0, 0.0)  # sliding when the switch is thrown
    adapter.kill()
    for _ in range(steps(0.5)):
        state = adapter.step(DT, link_age_s=0.0)
        assert np.array_equal(state.position, np.array((2.0, 2.0, 0.0))), state.position
        assert np.array_equal(state.velocity, np.zeros(3))


# 4. SLOW: the carrot never jumps -----------------------------------------------

SLOW = FailsafeConfig(degraded_action="slow", slow_speed_mps=0.3)


def _carrot_steps(adapter, n, goal, *, link_age_s, fix=False):
    largest = 0.0
    speeds = []
    for _ in range(n):
        before = adapter.carrot_m
        if fix:
            adapter.set_position_fix(truth(adapter).copy(), adapter.time_s)
        adapter.send_setpoint(np.asarray(goal, dtype=float))
        state = adapter.step(DT, link_age_s=link_age_s)
        largest = max(largest, float(np.linalg.norm(adapter.carrot_m - before)))
        speeds.append(float(np.linalg.norm(state.velocity)))
    return largest, np.asarray(speeds)


def test_slow_entry_while_moving_moves_the_carrot_at_most_one_slow_step():
    adapter = make("physics", failsafe_config=SLOW)
    twin = make("physics", failsafe_config=SLOW)  # same flight, link stays healthy
    for vehicle in (adapter, twin):
        _carrot_steps(vehicle, steps(2.0), (12.0, 2.0, 1.5), link_age_s=0.0)
    entry = float(np.linalg.norm(truth_velocity(adapter)))
    largest, speeds = _carrot_steps(adapter, steps(8.0), (12.0, 2.0, 1.5), link_age_s=0.2)
    _, nominal = _carrot_steps(twin, steps(8.0), (12.0, 2.0, 1.5), link_age_s=0.0)
    under = int(np.argmax(speeds <= 0.3)) + 1
    peak_at = (int(speeds.argmax()) + 1) * DT
    print(f"slow entry at {entry:.3f} m/s: largest carrot step {largest:.5f} m (limit 0.003 m), "
          f"peak {speeds.max():.3f} m/s at {peak_at:.2f} s (no SLOW: {nominal.max():.3f} m/s), "
          f"under 0.3 m/s after {under * DT:.2f} s, peak after 4 s {speeds[steps(4.0):].max():.4f} m/s")
    assert largest <= 0.3 * DT + 1e-12
    # The body was still speeding up when SLOW began; it cannot turn that
    # around at once (it must tilt back first), but SLOW must never leave it
    # faster than flying on would, the rise must end within the attitude
    # response (0.3 s), and the speed must settle under the cap.
    assert speeds.max() < nominal.max()
    assert peak_at <= 0.3
    assert speeds[steps(4.0):].max() <= 0.3 + 1e-3


@pytest.mark.parametrize("slow", (False, True))
def test_offboard_fix_jump_does_not_move_the_carrot_more_than_one_step(slow):
    config = SLOW if slow else None
    limit = (0.3 if slow else 1.5) * DT
    adapter = make("physics_offboard", failsafe_config=config)
    age = 0.2 if slow else 0.0
    _carrot_steps(adapter, steps(0.5), (12.0, 2.0, 1.5), link_age_s=age, fix=True)
    before = adapter.carrot_m
    adapter.set_position_fix(truth(adapter) + np.array((6.0, 0.0, 0.0)), adapter.time_s)  # a 6 m outlier
    adapter.send_setpoint(np.array((12.0, 2.0, 1.5)))
    adapter.step(DT, link_age_s=age)
    jump = float(np.linalg.norm(adapter.carrot_m - before))
    print(f"slow={slow}: carrot moved {jump:.5f} m after a 6 m fix jump (limit {limit:.3f} m)")
    assert jump <= limit + 1e-12


# 8. Kill switch: rotors stop at once ------------------------------------------

def test_physics_kill_zeroes_rotors_immediately():
    adapter = make("physics")
    adapter.step(DT, link_age_s=0.0)
    assert np.all(adapter.vehicle.actuator.speed_rad_s > 0.0)
    adapter.kill()
    assert np.array_equal(adapter.vehicle.actuator.speed_rad_s, np.zeros(4))
    for _ in range(steps(0.3)):
        adapter.send_setpoint(np.array((2.0, 2.0, 1.5)))
        adapter.step(DT, link_age_s=0.0)
        assert np.array_equal(adapter.vehicle.actuator.speed_rad_s, np.zeros(4))


def test_physics_kill_requested_in_step_zeroes_rotors():
    adapter = make("physics")
    adapter.step(DT, link_age_s=0.0)
    adapter.step(DT, link_age_s=0.0, kill_requested=True)
    assert np.array_equal(adapter.vehicle.actuator.speed_rad_s, np.zeros(4))


# 7. Both adapters follow one safety script -------------------------------------

def _safety_script(adapter):
    """Run the same command script and return what an observer can see."""
    seen = []
    for kind in (CommandKind.LAND, CommandKind.KILL):
        command = ControlCommand.land() if kind is CommandKind.LAND else ControlCommand.kill()
        for source in (CommandSource.MISSION, CommandSource.MANUAL):
            decision = adapter.request_command(command, source=source)
            seen.append(("request", kind.value, source.value, decision.accepted, decision.reason))
    for _ in range(5):
        adapter.send_setpoint(np.array((2.0, 2.0, 1.5)))
        seen.append(("step", adapter.step(DT, link_age_s=0.0).command_authority))
    adapter.request_command(ControlCommand.hold(), source=CommandSource.MANUAL, ttl_s=0.025)
    for _ in range(5):
        adapter.send_setpoint(np.array((2.0, 2.0, 1.5)))
        seen.append(("step", adapter.step(DT, link_age_s=0.0).command_authority))
    for age in (0.2, 0.2, 0.0):
        state = adapter.step(DT, link_age_s=age)
        seen.append(("age", age, state.failsafe_state, state.command_authority))
    seen.append(("land", adapter.land().state.value, adapter.step(DT, link_age_s=0.0).command_authority))
    with pytest.raises(ValueError):
        adapter.reset_failsafe()  # a landing must be acknowledged before a reset
    seen.append(("kill", adapter.kill().state.value, adapter.step(DT, link_age_s=0.0).failsafe_state))
    seen.append(("events", [event["kind"] for event in adapter.drain_events()]))
    return seen


def test_point_mass_and_physics_follow_the_same_safety_rules():
    point_mass = _safety_script(make("point_mass"))
    physics = _safety_script(make("physics"))
    assert point_mass == physics
    # Spot checks against the contract, independent of either adapter.
    requests = [item for item in point_mass if item[0] == "request"]
    assert all(item[3] is False and item[4] == "safety_command_requires_failsafe" for item in requests)
    authorities = [item[1] for item in point_mass if item[0] == "step"]
    assert authorities[:5] == ["mission"] * 5
    # Manual hold sent at 0.05 s with ttl 0.025 s: it owns the 0.06 and 0.07 s steps.
    assert authorities[5:] == ["manual", "manual", "mission", "mission", "mission"]
    ages = [item for item in point_mass if item[0] == "age"]
    assert ages[0][2] == "degraded" and ages[0][3] == "failsafe"
    assert ages[-1][2] == "nominal" and ages[-1][3] == "mission"
    events = point_mass[-1][1]
    assert events.count("kill") == 1
    assert "failsafe_transition" in events and "authority_change" in events
