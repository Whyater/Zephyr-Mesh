"""Unit checks for the simulated adapters: geofence clip, SLOW cap, offboard
fixes, stale link, hover, carrot speed limit, kill switch fall, landing.

All vehicles are simulated.  Expected values are hand arithmetic.
"""
import math

import numpy as np

from sim.adapters import Capabilities, Geofence, OffboardPointMassAdapter, SimAdapter
from sim.failsafe import FailsafeConfig, FailsafeState
from sim.missions.physics_adapter import PhysicsAdapter

DT = 0.01


def test_geofence_clip_keeps_point_mass_inside():
    fence = Geofence((0, 0, 0), (4, 4, 4), 0.5)
    adapter = SimAdapter("f", geofence=fence, dt_s=DT)
    adapter.connect()
    for _ in range(3000):
        adapter.send_setpoint((20, 20, 20))
        adapter.step()
    assert fence.contains(adapter.position)
    # The clip box corner is (3.5, 3.5, 3.5).
    assert np.allclose(adapter.position, (3.5, 3.5, 3.5), atol=1e-6)


def test_point_mass_slow_caps_speed():
    config = FailsafeConfig(weak_rule="age_or_loss", degraded_action="slow", slow_speed_mps=0.3)
    adapter = SimAdapter("s", dt_s=DT, failsafe_config=config)
    adapter.connect()
    for _ in range(100):
        adapter.send_setpoint((10, 0, 1))
        state = adapter.step(recent_loss_fraction=0.4)
        assert state.failsafe_state == "degraded"
        assert np.linalg.norm(adapter.velocity) <= 0.3 + 1e-9


def test_offboard_point_mass_steers_from_the_fix():
    adapter = OffboardPointMassAdapter("o", initial_position_m=(0, 0, 1))
    adapter.connect()
    # The fix says x = 1 while the truth is x = 0, and no newer fix arrives,
    # so the loop keeps driving toward -x past the commanded x = 0.
    adapter.set_position_fix((1, 0, 1), 0.0)
    for _ in range(20):
        adapter.send_setpoint((0, 0, 1))
        adapter.step()
    assert adapter.position[0] < -0.5


def test_stale_link_lands_both_adapters():
    point_mass = SimAdapter("stale")
    point_mass.connect()
    assert point_mass.step(link_age_s=0.4).failsafe_state == FailsafeState.LANDING.value
    physics = PhysicsAdapter("physics-stale")
    physics.connect()
    assert physics.step(link_age_s=0.4).failsafe_state == FailsafeState.LANDING.value


def test_physics_hover_holds_position():
    hover = PhysicsAdapter("hover")
    hover.connect()
    for _ in range(500):
        hover.send_setpoint((0, 0, 1))
        hover.step()
    assert np.linalg.norm(hover.get_state().position - np.array((0, 0, 1))) < 0.02


def test_physics_carrot_speed_limit():
    max_speed = 1.0
    adapter = PhysicsAdapter("limited", capabilities=Capabilities(max_speed_mps=max_speed))
    adapter.connect()
    previous = adapter.carrot_m
    for _ in range(100):  # 1 s
        adapter.send_setpoint((5, 0, 1))
        adapter.step()
        carrot = adapter.carrot_m
        assert np.linalg.norm(carrot - previous) <= max_speed * DT + 1e-12
        previous = carrot
    # Hand bound: after 1 s the carrot is at most 1.0 m from the start, and
    # the PD loop (zero target velocity) lags a moving carrot, so the true
    # x is no further than the carrot.
    assert adapter.carrot_m[0] <= max_speed * 1.0 + 1e-9
    x = adapter.get_state().position[0]
    assert 0.0 < x <= max_speed * 1.0


def test_physics_kill_switch_fall_time():
    dropped = PhysicsAdapter("drop", initial_position_m=(0, 0, 1.5))
    dropped.connect()
    dropped.kill()
    while dropped.get_state().position[2] > 0:
        dropped.step()
    # Free fall from 1.5 m: sqrt(2 * 1.5 / 9.81) = 0.553 s, checked on a 0.01 s grid.
    assert abs(dropped.time_s - math.sqrt(3.0 / 9.81)) <= 0.04


def test_physics_landing_time():
    landing = PhysicsAdapter("land", initial_position_m=(0, 0, 1.5))
    landing.connect()
    landing.land()
    for _ in range(800):
        landing.step()
        if landing.get_state().landed:
            break
    assert landing.get_state().landed
    # The carrot alone needs 1.5 / 0.3 = 5 s to reach the floor.
    assert 4.5 <= landing.time_s <= 7.5
