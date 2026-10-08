"""Independent checks for the adapter, authority, and failsafe contracts."""

import numpy as np
import pytest

from sim.adapters import Capabilities, DroneState, SimAdapter
from sim.authority import (
    AuthorityArbiter,
    CommandKind,
    CommandSource,
    ControlCommand,
)
from sim.failsafe import (
    FailsafeAction,
    FailsafeConfig,
    FailsafeController,
    FailsafeState,
)
from sim.hardware import HardwareProfile, MotorProfile, PropellerProfile


def test_authority_priorities_and_expiry_are_deterministic():
    arbiter = AuthorityArbiter()
    mission = arbiter.submit(
        CommandSource.MISSION,
        ControlCommand.position((1, 0, 1)),
        timestamp_s=0.0,
        ttl_s=0.1,
    )
    assert mission.source is CommandSource.MISSION
    manual = arbiter.submit(
        CommandSource.MANUAL,
        ControlCommand.position((2, 0, 1)),
        timestamp_s=0.0,
        ttl_s=0.2,
    )
    assert manual.source is CommandSource.MANUAL
    safety = arbiter.submit(
        CommandSource.FAILSAFE,
        ControlCommand.hold(),
        timestamp_s=0.0,
        ttl_s=None,
    )
    assert safety.source is CommandSource.FAILSAFE
    assert safety.command.kind is CommandKind.HOLD
    assert safety.request_selected is True
    assert safety.contenders == ("failsafe", "manual", "mission")
    arbiter.release(CommandSource.FAILSAFE, timestamp_s=0.0)
    expired = arbiter.select(0.11)
    assert expired.source is CommandSource.MANUAL
    arbiter.release(CommandSource.MANUAL, timestamp_s=0.11)
    assert arbiter.select(0.21).reason == "no_active_command"


def test_lower_priority_submission_is_reported_as_preempted():
    arbiter = AuthorityArbiter()
    arbiter.submit(CommandSource.MANUAL, ControlCommand.hold(), timestamp_s=0.0, ttl_s=None)
    mission = arbiter.submit(
        CommandSource.MISSION,
        ControlCommand.position((3, 0, 1)),
        timestamp_s=0.1,
        ttl_s=None,
    )
    assert mission.accepted is False
    assert mission.request_selected is False
    assert mission.submitted_source is CommandSource.MISSION
    assert mission.source is CommandSource.MANUAL
    assert mission.reason == "request_preempted"


def test_authority_rejects_time_reversal_and_invalid_commands():
    arbiter = AuthorityArbiter()
    with pytest.raises(ValueError):
        arbiter.submit(CommandSource.MISSION, ControlCommand.hold(), timestamp_s=0.0, ttl_s=0.0)
    arbiter.submit(CommandSource.MISSION, ControlCommand.hold(), timestamp_s=0.2)
    with pytest.raises(ValueError):
        arbiter.select(0.1)
    assert arbiter.last_time_s == pytest.approx(0.2)
    with pytest.raises(ValueError):
        arbiter.submit(CommandSource.MISSION, ControlCommand.hold(), timestamp_s=0.4, ttl_s=0.0)
    assert arbiter.last_time_s == pytest.approx(0.2)
    with pytest.raises(ValueError):
        arbiter.submit(CommandSource.MISSION, ControlCommand.hold(), timestamp_s=0.4, request_id="")
    assert arbiter.last_time_s == pytest.approx(0.2)
    with pytest.raises(ValueError):
        ControlCommand(CommandKind.HOLD, position_m=(0, 0, 0))


def test_failsafe_degraded_recovers_but_landing_and_kill_latch():
    policy = FailsafeController(FailsafeConfig(degraded_after_s=0.1, land_after_s=0.3, low_battery_pct=20))
    nominal = policy.observe(0.0, link_age_s=0.0, battery_pct=100)
    assert nominal.action is FailsafeAction.NONE
    assert nominal.command is None
    degraded = policy.observe(0.0, link_age_s=0.15, battery_pct=100)
    assert degraded.state is FailsafeState.DEGRADED
    assert degraded.action is FailsafeAction.HOLD
    recovered = policy.observe(0.1, link_age_s=0.0, battery_pct=100)
    assert recovered.state is FailsafeState.NOMINAL
    landing = policy.observe(0.2, link_age_s=0.3, battery_pct=100)
    assert landing.state is FailsafeState.LANDING
    with pytest.raises(ValueError):
        policy.reset(0.21, link_age_s=0.0, battery_pct=100)
    with pytest.raises(ValueError):
        policy.reset(0.22, link_age_s=0.15, battery_pct=100)
    assert policy.last_time_s == pytest.approx(0.2)
    still_landing = policy.observe(0.3, link_age_s=0.0, battery_pct=100)
    assert still_landing.state is FailsafeState.LANDING
    landed = policy.observe(0.4, link_age_s=0.0, battery_pct=100, landed=True)
    assert landed.state is FailsafeState.LANDED
    killed = policy.force_kill(0.5)
    assert killed.state is FailsafeState.KILLED
    assert policy.observe(0.6, link_age_s=0.0, battery_pct=100).action is FailsafeAction.KILL
    reset = policy.reset(0.7, link_age_s=0.0, battery_pct=100)
    assert reset.state is FailsafeState.NOMINAL


def test_failsafe_low_battery_and_unknown_link_are_explicit():
    policy = FailsafeController(FailsafeConfig(low_battery_pct=15))
    low = policy.observe(0.0, link_age_s=0.0, battery_pct=15)
    assert low.reason == "low_battery"
    assert low.command.kind is CommandKind.LAND
    with pytest.raises(ValueError):
        policy.reset(0.1, link_age_s=0.0, battery_pct=15)
    other = FailsafeController()
    assert other.observe(0.0, link_age_s=None, battery_pct=100).state is FailsafeState.LANDING


def test_sim_adapter_normalizes_state_and_manual_override():
    adapter = SimAdapter(
        "sim-1",
        capabilities=Capabilities(max_speed_mps=1.0),
        initial_position_m=(0, 0, 1),
    )
    assert adapter.get_state().connected is False
    adapter.connect()
    assert adapter.send_setpoint(np.array((1, 0, 1))).source is CommandSource.MISSION
    adapter.request_command(
        ControlCommand.position((0, 1, 1)),
        source=CommandSource.MANUAL,
        ttl_s=None,
    )
    state = adapter.step()
    assert state.command_authority == "manual"
    assert np.linalg.norm(state.position - np.array((0, 0, 1))) > 0.0
    assert state.as_dict()["connected"] is True
    assert state.position_m is state.position


def test_sim_adapter_applies_profile_thrust_to_mass_limit():
    profile = HardwareProfile(
        "low-thrust",
        MotorProfile("low-motor", max_torque_nm=0.001, max_power_w=2, max_rpm=3_000, mass_kg=0.02),
        PropellerProfile("low-prop", diameter_m=0.05, pitch_m=0.02, max_rpm=3_000),
        frame_mass_kg=0.20,
    )
    adapter = SimAdapter("low", profile=profile, max_acceleration_mps2=4.0)
    assert adapter.profile_acceleration_limit_mps2 == pytest.approx(
        profile.estimated_max_thrust_n() / profile.total_mass_kg
    )
    assert adapter.profile_acceleration_limit_mps2 < 4.0


def test_profile_limit_changes_motion_under_the_same_setpoint():
    low_profile = HardwareProfile(
        "very-low-thrust",
        MotorProfile("low-motor", max_torque_nm=0.001, max_power_w=2, max_rpm=3_000, mass_kg=0.02),
        PropellerProfile("low-prop", diameter_m=0.05, pitch_m=0.02, max_rpm=3_000, thrust_coefficient=0.01),
        frame_mass_kg=0.20,
    )
    low = SimAdapter("low-motion", profile=low_profile)
    nominal = SimAdapter("nominal-motion")
    low.connect(); nominal.connect()
    low.send_setpoint((1.0, 0.0, 1.0)); nominal.send_setpoint((1.0, 0.0, 1.0))
    low_state = low.step(); nominal_state = nominal.step()
    assert low_state.velocity[0] < nominal_state.velocity[0]


def test_safety_commands_cannot_bypass_failsafe_policy():
    adapter = SimAdapter("sim-safety")
    adapter.connect()
    rejected = adapter.request_command(ControlCommand.kill(), source=CommandSource.MISSION)
    assert rejected.accepted is False
    assert rejected.reason == "safety_command_requires_failsafe"
    assert adapter.failsafe_state is FailsafeState.NOMINAL

    limited = SimAdapter("sim-limited", capabilities=Capabilities(supports_kill=False, supports_land=False))
    limited.connect()
    assert limited.request_command(ControlCommand.kill(), source=CommandSource.MISSION).accepted is False
    with pytest.raises(RuntimeError):
        limited.kill()
    with pytest.raises(RuntimeError):
        limited.land()


def test_sim_adapter_failsafe_preempts_mission_and_lands_synthetically():
    adapter = SimAdapter("sim-2", initial_position_m=(0, 0, 0.6))
    adapter.connect()
    adapter.send_setpoint((1, 0, 1))
    adapter.set_link_metrics(delay_ms=20, loss_pct=50, age_s=0.4)
    degraded_or_landing = adapter.step()
    assert degraded_or_landing.command_authority == "failsafe"
    assert degraded_or_landing.failsafe_state == "landing"
    adapter.set_link_metrics(delay_ms=20, loss_pct=50, age_s=0.0)
    # Landing is latched and the point-mass model reaches the ground deterministically.
    for _ in range(200):
        state = adapter.step()
    assert state.failsafe_state == "landed"
    assert state.position[2] == pytest.approx(0.0)
    assert adapter.reset_failsafe().state is FailsafeState.NOMINAL


def test_drone_state_validates_bounds_and_serializes():
    state = DroneState(np.zeros(3), np.zeros(3), 100, 2, 1, connected=True)
    assert state.as_dict()["position"] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        DroneState(np.zeros(3), np.zeros(3), 101, 0, 0)
