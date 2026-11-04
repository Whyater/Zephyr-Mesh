"""Independent checks for the opt-in ground plane.

The expected numbers come from hand physics (free-fall kinematics and the
closed-form fall with quadratic drag), not from the ground code itself.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from sim.drone import SixDOFInterceptor
from sim.environment import AIR_DENSITY, GRAVITY
from sim.ground import GroundPlane
from sim.hardware import default_hardware_profile
from sim.vehicle import ActuatedQuadrotor

DT = 0.01


def _fall(height_m, *, drag_coefficient, ground):
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, height_m), drag_coefficient=drag_coefficient, ground=ground)
    for _ in range(1000):
        drone.step_physics(np.zeros(3), np.zeros(3), DT)
        if ground.touchdowns:
            return drone
    raise AssertionError("the body never reached the ground")


def test_free_fall_from_two_metres_touches_down_at_hand_speed_and_time():
    ground = GroundPlane()
    drone = _fall(2.0, drag_coefficient=0.0, ground=ground)
    event = ground.touchdowns[0]
    hand_speed = math.sqrt(2.0 * GRAVITY * 2.0)  # 6.264 m/s
    hand_time = math.sqrt(2.0 * 2.0 / GRAVITY)  # 0.6386 s
    # Free-fall acceleration is constant, so RK4 is exact and the in-step
    # crossing estimate is exact up to rounding.
    assert event.vertical_speed_mps == pytest.approx(hand_speed, abs=1e-9)
    assert event.time_s == pytest.approx(hand_time, abs=1e-9)
    # The step that crosses z = 0 ends after the real touchdown, by less than
    # one dt, so its raw end speed is too fast by at most g * dt.
    assert hand_speed <= event.end_of_step_vertical_speed_mps <= hand_speed + GRAVITY * DT + 1e-12
    assert event.horizontal_speed_mps == 0.0
    assert event.crashed
    assert drone.position[2] == 0.0
    assert drone.velocity[2] == 0.0


def test_free_fall_with_default_drag_matches_closed_form():
    ground = GroundPlane()
    drone = _fall(2.0, drag_coefficient=1.3, ground=ground)
    k = 0.5 * AIR_DENSITY * 1.3 * 0.05
    terminal_sq = drone.mass * GRAVITY / k
    hand_speed = math.sqrt(terminal_sq * (1.0 - math.exp(-2.0 * k * 2.0 / drone.mass)))  # 5.749 m/s
    # Drag makes the acceleration change inside the crossing step, so the
    # constant-acceleration estimate is only approximate there.
    assert ground.touchdowns[0].vertical_speed_mps == pytest.approx(hand_speed, abs=1e-3)
    assert ground.touchdowns[0].vertical_speed_mps < math.sqrt(2.0 * GRAVITY * 2.0)


def test_short_drop_is_a_touchdown_but_not_a_crash():
    ground = GroundPlane(crash_speed_mps=2.0)
    _fall(0.1, drag_coefficient=0.0, ground=ground)
    event = ground.touchdowns[0]
    assert event.vertical_speed_mps == pytest.approx(math.sqrt(2.0 * GRAVITY * 0.1), abs=1e-9)  # 1.401 m/s
    assert not event.crashed
    assert not ground.crashed
    assert ground.first_crash is None


@pytest.mark.parametrize("thrust_fraction", [0.0, 0.5])
def test_body_resting_on_ground_with_low_thrust_stays_put(thrust_fraction):
    ground = GroundPlane()
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, 0.0), ground=ground)
    force = np.array([0.0, 0.0, thrust_fraction * drone.mass * GRAVITY])
    for _ in range(500):
        drone.step_physics(force, np.zeros(3), DT)
        assert drone.position[2] == 0.0
        assert np.all(drone.velocity == 0.0)
    assert np.all(drone.position == 0.0)
    assert ground.touchdowns == []
    assert ground.in_contact


def test_thrust_above_weight_lifts_off_the_ground():
    ground = GroundPlane()
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, 0.0), drag_coefficient=0.0, ground=ground)
    force = np.array([0.0, 0.0, 1.5 * drone.mass * GRAVITY])
    for _ in range(100):
        drone.step_physics(force, np.zeros(3), DT)
    # Net upward acceleration 0.5 g for 1 s: z = 0.25 g t^2.
    assert drone.position[2] == pytest.approx(0.25 * GRAVITY, rel=1e-9)
    assert not ground.in_contact
    assert ground.touchdowns == []


def test_hovering_vehicle_never_touches_the_ground():
    ground = GroundPlane()
    vehicle = ActuatedQuadrotor(default_hardware_profile("ground-plane-test"), initial_position=(0.0, 0.0, 1.0), ground=ground)
    heights = [vehicle.step((0.0, 0.0, 1.0), DT).position_m[2] for _ in range(300)]
    assert ground.touchdowns == []
    assert not ground.in_contact
    # The motors start from rest, so the vehicle sags (about 0.23 m with the
    # default profile) before the controller catches it. It must stay well
    # clear of the floor.
    assert min(heights) > 0.5


def test_vehicle_takeoff_from_floor_never_goes_below_it():
    ground = GroundPlane()
    vehicle = ActuatedQuadrotor(default_hardware_profile("ground-plane-test"), ground=ground)
    heights = [vehicle.step((0.3, -0.2, 1.0), DT).position_m[2] for _ in range(300)]
    assert min(heights) >= 0.0
    assert ground.touchdowns == []
    assert heights[-1] == pytest.approx(1.0, abs=0.15)


def test_vehicle_reset_clears_ground_history():
    ground = GroundPlane()
    vehicle = ActuatedQuadrotor(default_hardware_profile("ground-plane-test"), initial_position=(0.0, 0.0, 0.2), ground=ground)
    for _ in range(100):
        vehicle.drone.step_physics(np.zeros(3), np.zeros(3), DT)
    assert ground.touchdowns
    vehicle.reset((0.0, 0.0, 1.0))
    assert ground.touchdowns == []
    assert ground.elapsed_s == 0.0


@pytest.mark.parametrize("kwargs", [{"crash_speed_mps": 0.0}, {"crash_speed_mps": float("nan")}, {"height_m": float("inf")}])
def test_ground_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        GroundPlane(**kwargs)


def test_provenance_labels_crash_threshold_as_placeholder():
    record = GroundPlane().model_provenance()
    assert record["crash_speed_mps"] == 2.0
    assert "placeholder" in record["crash_speed_basis"]


def test_provenance_records_ground_only_when_attached():
    assert "ground" not in SixDOFInterceptor().model_provenance()
    record = ActuatedQuadrotor(default_hardware_profile("ground-plane-test"), ground=GroundPlane()).model_provenance()
    assert record["dynamics"]["ground"]["model"] == "ground_plane_clamp"


# Ground off: the default path must be unchanged. The final states below were
# recorded on Windows by running these same two histories on the untouched
# files at commit 77be94f, before sim/ground.py existed. They are compared
# within a relative 1e-9, not by hash: NumPy picks different SIMD code paths
# on different CPUs, so the exact bits can change between two Windows
# machines running the same code (seen on CI runners).

BARE_FINAL_77BE94F = [
    -0.7374494163461934, 0.032596799679150315, 2.497957145982772, -0.5388586261749942,
    0.04672314605762598, 0.13445044520007465, 0.999912746437068, -0.0023205469877040897,
    -0.0053556400487807, -0.011850387927227842, -0.007177642820294551, 7.3423211373149e-05,
    -0.0039793266946356615,
]
VEHICLE_FINAL_77BE94F = [
    0.3292674170788611, -0.2195115988796134, 1.0578023571089574, -0.019904601239917247,
    0.013269749445199869, -0.058155091441546375, 0.9999949882314096, -0.001756180259885758,
    -0.00263426237895951, 2.1625520736049708e-06, 0.008920719988458082, 0.013380961108410182,
    3.098309439524692e-11,
]


def _bare_history(**drone_kwargs):
    rng = np.random.default_rng(20261103)
    drone = SixDOFInterceptor(init_pos=(0.0, 0.0, 2.0), **drone_kwargs)
    rows = []
    for _ in range(500):
        force = np.array([0.0, 0.0, drone.mass * 9.81]) + rng.normal(0.0, 0.5, 3)
        torque = rng.normal(0.0, 1e-4, 3)
        drone.step_physics(force, torque, 0.01)
        rows.append(drone._state_vector())
    return np.array(rows)


def _vehicle_history(**vehicle_kwargs):
    vehicle = ActuatedQuadrotor(default_hardware_profile("ground-plane-test"), **vehicle_kwargs)
    rows = []
    for _ in range(300):
        vehicle.step((0.3, -0.2, 1.0), 0.01)
        rows.append(vehicle.drone._state_vector())
    return np.array(rows)


def _hash(history):
    return hashlib.sha256(np.ascontiguousarray(history, dtype="<f8").tobytes()).hexdigest()


@pytest.mark.parametrize(
    "make_history, saved_final",
    [(_bare_history, BARE_FINAL_77BE94F), (_vehicle_history, VEHICLE_FINAL_77BE94F)],
)
def test_ground_off_matches_saved_77be94f_history(make_history, saved_final):
    history = make_history()
    assert np.allclose(history[-1], saved_final, rtol=1e-9, atol=1e-12)


def test_ground_that_is_never_reached_changes_nothing():
    # Same platform, same run: a floor far below must not touch a single bit.
    assert _hash(_bare_history(ground=GroundPlane(height_m=-1e6))) == _hash(_bare_history())
    assert _hash(_vehicle_history(ground=GroundPlane(height_m=-1e6))) == _hash(_vehicle_history())


def test_saved_vehicle_history_really_sinks_through_the_floor_without_ground():
    # The reason the ground exists: with it off, the default takeoff dips
    # below z = 0 before it climbs.
    assert _vehicle_history()[:, 2].min() < -0.05
