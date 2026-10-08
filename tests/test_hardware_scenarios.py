"""Checks for SI-unit hardware profiles and the Stage 7 ring fixture."""

import pytest

from sim.hardware import HardwareProfile, MotorProfile, PropellerProfile
from sim.scenarios import default_hardware_profile, make_ring_swarm_config, scenario_manifest


def test_propeller_scaling_and_profile_limits_are_explicit():
    propeller = PropellerProfile("p", diameter_m=0.1, pitch_m=0.05, max_rpm=12000, thrust_coefficient=0.1)
    # Ct*rho*n^2*D^4, independently evaluated for 6000 RPM.
    assert propeller.estimated_static_thrust_n(6000) == pytest.approx(
        0.1 * 1.225 * (6000 / 60) ** 2 * 0.1 ** 4
    )
    motor = MotorProfile("m", max_torque_nm=0.004, max_power_w=20, max_rpm=10000)
    profile = HardwareProfile("h", motor, propeller, motor_count=4)
    assert profile.estimated_max_thrust_n() == pytest.approx(4 * propeller.estimated_static_thrust_n(10000))
    assert profile.total_mass_kg > 0


def test_hardware_profile_rejects_non_si_or_nonpositive_inputs():
    with pytest.raises(ValueError):
        MotorProfile("m", max_torque_nm=0, max_power_w=20, max_rpm=10000)
    with pytest.raises(ValueError):
        PropellerProfile("p", diameter_m=-0.1, pitch_m=0.05, max_rpm=10000)
    with pytest.raises(ValueError):
        HardwareProfile("h", default_hardware_profile(0).motor, default_hardware_profile(0).propeller, motor_count=0)


def test_fifty_agent_factory_has_unique_parts_and_manifest():
    config = make_ring_swarm_config(agent_count=50, seed=7)
    assert len(config.agents) == 50
    profile_ids = [agent.hardware_profile.profile_id for agent in config.agents]
    motor_ids = [agent.hardware_profile.motor.part_id for agent in config.agents]
    prop_ids = [agent.hardware_profile.propeller.part_id for agent in config.agents]
    assert len(set(profile_ids)) == 50
    assert len(set(motor_ids)) == 50
    assert len(set(prop_ids)) == 50
    manifest = scenario_manifest(config)
    assert manifest["schema"] == "zephyr-s7-scenario-manifest-1"
    assert manifest["agent_count"] == 50
    assert len(manifest["agents"]) == 50
    assert manifest["evidence_boundary"].startswith("Coordination")


def test_ring_factory_rejects_invalid_size():
    with pytest.raises(ValueError):
        make_ring_swarm_config(agent_count=0)
    with pytest.raises(ValueError):
        make_ring_swarm_config(radius_m=0)
