import math

import numpy as np

from sim.adapters import Capabilities, Geofence, OffboardPointMassAdapter, SimAdapter
from sim.failsafe import FailsafeConfig, FailsafeState
from sim.missions.physics_adapter import PhysicsAdapter


def test_geofence_and_slow_cap():
    fence = Geofence((0, 0, 0), (4, 4, 4), 0.5)
    adapter = SimAdapter("f", geofence=fence, dt_s=0.01)
    adapter.connect()
    for _ in range(3000):
        adapter.send_setpoint((20, 20, 20))
        adapter.step()
    assert fence.contains(adapter.position)
    slow = SimAdapter("s", dt_s=0.01, failsafe_config=FailsafeConfig(weak_rule="age_or_loss", degraded_action="slow", slow_speed_mps=0.3))
    slow.connect()
    for _ in range(100):
        slow.send_setpoint((10, 0, 1))
        slow.step(recent_loss_fraction=0.4)
        assert np.linalg.norm(slow.velocity) <= 0.3 + 1e-9


def test_offboard_fix_and_stale_link():
    adapter = OffboardPointMassAdapter("o", initial_position_m=(0, 0, 1))
    adapter.connect()
    adapter.set_position_fix((1, 0, 1), 0.0)
    for _ in range(20):
        adapter.send_setpoint((0, 0, 1))
        adapter.step()
    assert abs(adapter.position[0]) > 0.5
    stale = SimAdapter("stale"); stale.connect(); assert stale.step(link_age_s=0.4).failsafe_state == FailsafeState.LANDING.value
    physics_stale = PhysicsAdapter("physics-stale"); physics_stale.connect()
    assert physics_stale.step(link_age_s=0.4).failsafe_state == FailsafeState.LANDING.value


def test_physics_hand_checks():
    hover = PhysicsAdapter("hover")
    hover.connect()
    hover.send_setpoint((0, 0, 1))
    for _ in range(500):
        hover.send_setpoint((0, 0, 1))
        hover.step()
    assert np.linalg.norm(hover.get_state().position - np.array((0, 0, 1))) < 0.02

    limited = PhysicsAdapter("limited", capabilities=Capabilities(max_speed_mps=1.0))
    limited.connect()
    for _ in range(200):
        limited.send_setpoint((2, 0, 1))
        limited.step()
    assert limited.time_s >= 2.0

    killed = PhysicsAdapter("drop", initial_position_m=(0, 0, 1.5))
    killed.connect(); killed.kill()
    while killed.get_state().position[2] > 0:
        killed.step()
    assert abs(killed.time_s - math.sqrt(3.0 / 9.81)) <= 0.04

    landed = PhysicsAdapter("land", initial_position_m=(0, 0, 1.5))
    landed.connect(); landed.land()
    for _ in range(800):
        landed.step()
        if landed.get_state().landed:
            break
    assert landed.get_state().landed
    assert 4.5 <= landed.time_s <= 7.5

