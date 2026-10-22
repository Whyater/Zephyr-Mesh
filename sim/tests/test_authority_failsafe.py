"""Independent checks for replay-time command authority and safety policy."""

import pytest

from sim.authority import AuthorityArbiter, CommandKind, CommandSource, ControlCommand
from sim.failsafe import FailsafeConfig, FailsafeController, FailsafeState


def test_manual_takeover_wins_then_expiry_returns_to_mission():
    arbiter = AuthorityArbiter()
    mission = arbiter.submit(
        CommandSource.MISSION,
        ControlCommand.position((1.0, 0.0, 1.0)),
        timestamp_s=0.0,
        ttl_s=None,
    )
    assert mission.source is CommandSource.MISSION
    manual = arbiter.submit(
        CommandSource.MANUAL,
        ControlCommand.hold(),
        timestamp_s=0.1,
        ttl_s=0.2,
    )
    assert manual.source is CommandSource.MANUAL
    assert manual.command is not None and manual.command.kind is CommandKind.HOLD
    after_expiry = arbiter.select(0.31)
    assert after_expiry.source is CommandSource.MISSION
    assert after_expiry.contenders == ("mission",)


def test_authority_rejects_clock_rewind():
    arbiter = AuthorityArbiter()
    arbiter.submit(CommandSource.MISSION, ControlCommand.hold(), timestamp_s=1.0)
    with pytest.raises(ValueError):
        arbiter.select(0.9)


def test_failsafe_degrades_recovers_and_latches_landing():
    policy = FailsafeController(FailsafeConfig(degraded_after_s=0.1, land_after_s=0.25))
    degraded = policy.observe(0.10, link_age_s=0.12, battery_pct=80.0)
    assert degraded.state is FailsafeState.DEGRADED
    healthy = policy.observe(0.20, link_age_s=0.02, battery_pct=80.0)
    assert healthy.state is FailsafeState.NOMINAL
    landing = policy.observe(0.30, link_age_s=0.40, battery_pct=80.0)
    assert landing.state is FailsafeState.LANDING
    still_landing = policy.observe(0.40, link_age_s=0.01, battery_pct=80.0)
    assert still_landing.state is FailsafeState.LANDING
    landed = policy.observe(0.50, link_age_s=0.01, battery_pct=80.0, landed=True)
    assert landed.state is FailsafeState.LANDED


def test_failsafe_reset_requires_healthy_link_and_battery():
    policy = FailsafeController()
    policy.force_land(0.1)
    with pytest.raises(ValueError):
        policy.reset(0.2, link_age_s=0.3, battery_pct=80.0)
    with pytest.raises(ValueError):
        policy.reset(0.3, link_age_s=0.01, battery_pct=10.0)
    policy.observe(0.35, link_age_s=0.01, battery_pct=80.0, landed=True)
    reset = policy.reset(0.4, link_age_s=0.01, battery_pct=80.0)
    assert reset.state is FailsafeState.NOMINAL
