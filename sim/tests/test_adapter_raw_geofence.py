"""A raw POSITION command must meet the geofence the same way a mission setpoint does.

The clip used to live only in ``send_setpoint``; ``request_command`` with a
POSITION command went straight to the authority arbiter.  Expected values are
hand arithmetic on the box (0..16, 0..10, 0..4 with a 0.5 m margin, so the clip
box is 0.5..15.5, 0.5..9.5, 0.5..3.5), not adapter output.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.adapters import Geofence, SimAdapter
from sim.authority import CommandSource, ControlCommand
from sim.missions.physics_adapter import PhysicsAdapter

FENCE = Geofence((0.0, 0.0, 0.0), (16.0, 10.0, 4.0), margin_m=0.5)
START = (2.0, 2.0, 1.5)
OUTSIDE = np.array((40.0, -20.0, 9.0))
CLIPPED = np.array((15.5, 0.5, 3.5))


def make(kind, geofence=FENCE):
    cls = SimAdapter if kind == "point_mass" else PhysicsAdapter
    adapter = cls("d1", dt_s=0.01, initial_position_m=START, geofence=geofence)
    adapter.connect()
    return adapter


def clip_events(adapter):
    return [e for e in adapter.drain_events() if e["kind"] == "geofence_clip"]


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
@pytest.mark.parametrize("source", [CommandSource.MISSION, CommandSource.MANUAL, CommandSource.FORMATION])
def test_raw_position_command_is_clipped_and_logged_once(kind, source):
    adapter = make(kind)
    decision = adapter.request_command(ControlCommand.position(OUTSIDE), source=source)
    assert decision.accepted is True
    assert np.allclose(decision.command.position_m, CLIPPED)
    events = clip_events(adapter)
    assert len(events) == 1
    assert np.allclose(events[0]["detail"]["requested_m"], OUTSIDE)
    assert np.allclose(events[0]["detail"]["clipped_m"], CLIPPED)


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
def test_send_setpoint_logs_exactly_one_clip(kind):
    adapter = make(kind)
    decision = adapter.send_setpoint(OUTSIDE)
    assert np.allclose(decision.command.position_m, CLIPPED)
    assert len(clip_events(adapter)) == 1


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
def test_inside_position_is_untouched_and_silent(kind):
    adapter = make(kind)
    inside = (8.0, 5.0, 2.0)
    decision = adapter.request_command(ControlCommand.position(inside), source=CommandSource.MISSION)
    assert np.array_equal(decision.command.position_m, inside)
    assert clip_events(adapter) == []


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
def test_no_geofence_means_no_clip(kind):
    adapter = make(kind, geofence=None)
    decision = adapter.request_command(ControlCommand.position(OUTSIDE), source=CommandSource.MISSION)
    assert np.array_equal(decision.command.position_m, OUTSIDE)
    assert clip_events(adapter) == []


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
def test_non_position_commands_log_no_clip(kind):
    adapter = make(kind)
    adapter.request_command(ControlCommand.hold(), source=CommandSource.MISSION)
    assert clip_events(adapter) == []


@pytest.mark.parametrize("kind", ["point_mass", "physics"])
def test_raw_command_cannot_fly_the_vehicle_out(kind):
    adapter = make(kind)
    for _ in range(2000):
        adapter.request_command(ControlCommand.position(OUTSIDE), source=CommandSource.MISSION)
        state = adapter.step(0.01, link_age_s=0.0)
        assert FENCE.contains(state.position), f"left the fence at {state.position}"
    assert not adapter.geofence_latched
