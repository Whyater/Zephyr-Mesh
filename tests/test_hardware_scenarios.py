"""Checks for SI-unit hardware profiles and the Stage 7 ring fixture."""

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from sim.hardware import HARDWARE_PROFILE_SCHEMA, HardwareProfile, MotorProfile, PropellerProfile
from sim.scenarios import default_hardware_profile, make_ring_swarm_config, scenario_manifest
from tools.validate_hardware_profile import main as validate_hardware_profile


def test_propeller_scaling_and_profile_limits_are_explicit():
    propeller = PropellerProfile("p", diameter_m=0.1, pitch_m=0.05, max_rpm=12000, thrust_coefficient=0.1)
    # Ct*rho*n^2*D^4, independently evaluated for 6000 RPM.
    assert propeller.estimated_static_thrust_n(6000) == pytest.approx(
        0.1 * 1.225 * (6000 / 60) ** 2 * 0.1 ** 4
    )
    motor = MotorProfile("m", max_torque_nm=0.004, max_power_w=20, max_rpm=10000)
    profile = HardwareProfile("h", motor, propeller, motor_count=4)
    assert profile.estimated_max_thrust_n() == pytest.approx(4 * propeller.estimated_static_thrust_n(10000))
    assert profile.estimated_thrust_at_rpm_limit_n() == pytest.approx(profile.estimated_max_thrust_n())
    assert propeller.estimated_power_w(6000) == pytest.approx(
        0.04 * 1.225 * (6000 / 60) ** 3 * 0.1 ** 5
    )
    assert profile.total_mass_kg > 0


def test_propeller_torque_is_power_over_angular_speed():
    propeller = PropellerProfile("p", diameter_m=0.1, pitch_m=0.05, max_rpm=12000, power_coefficient=0.04)
    rpm = 6000.0
    expected = (0.04 * 1.225 * (rpm / 60.0) ** 3 * 0.1 ** 5) / (2.0 * np.pi * (rpm / 60.0))
    assert propeller.estimated_torque_nm(rpm) == pytest.approx(expected)
    assert propeller.estimated_torque_nm(0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        propeller.estimated_torque_nm(0.0, air_density_kg_m3=0.0)


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


def test_hardware_profile_document_round_trip_and_schema_parity(tmp_path):
    profile = default_hardware_profile(2)
    document = profile.as_document()
    schema = json.loads((Path(__file__).parents[1] / "sim" / "hardware_profile_schema.json").read_text())
    Draft202012Validator(schema).validate(document)
    path = tmp_path / "profile.json"
    profile.write(path)
    decoded = HardwareProfile.from_path(path)
    assert decoded.as_document() == document
    assert decoded.motor.part_id == profile.motor.part_id
    assert decoded.propeller.diameter_m == pytest.approx(profile.propeller.diameter_m)
    assert document["schema"] == HARDWARE_PROFILE_SCHEMA


def test_hardware_profile_rejects_missing_schema_and_extra_fields():
    document = default_hardware_profile(0).as_document()
    missing_schema = dict(document)
    del missing_schema["schema"]
    with pytest.raises(ValueError, match="schema must be"):
        HardwareProfile.from_dict(missing_schema)
    extra = dict(document)
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        HardwareProfile.from_dict(extra)
    schema = json.loads((Path(__file__).parents[1] / "sim" / "hardware_profile_schema.json").read_text())
    assert not Draft202012Validator(schema).is_valid(extra)
    missing = dict(document)
    del missing["motor"]["mass_kg"]
    with pytest.raises(ValueError, match="motor missing required"):
        HardwareProfile.from_dict(missing)
    assert not Draft202012Validator(schema).is_valid(missing)
    document["motor"]["mass_kg"] = default_hardware_profile(0).motor.mass_kg
    nested_extra = dict(document)
    nested_extra["propeller"] = dict(document["propeller"], unexpected=1)
    with pytest.raises(ValueError, match="propeller contains unknown"):
        HardwareProfile.from_dict(nested_extra)
    assert not Draft202012Validator(schema).is_valid(nested_extra)


def test_hardware_profile_metadata_marks_measured_inputs_explicitly():
    document = default_hardware_profile(0).as_document()
    document.update({"status": "measured", "source": "bench-run-001", "calibration_id": "cal-001", "code_revision": "abc123"})
    profile = HardwareProfile.from_dict(document)
    assert profile.status == "measured"
    assert profile.source == "bench-run-001"
    bad = dict(document)
    del bad["calibration_id"]
    with pytest.raises(ValueError, match="calibration_id"):
        HardwareProfile.from_dict(bad)


def test_hardware_profile_cli_normalizes_and_rejects_same_path(tmp_path):
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(default_hardware_profile(1).as_document()), encoding="utf-8")
    output = tmp_path / "nested" / "normalized.json"
    assert validate_hardware_profile([str(source), "--output", str(output)]) == 0
    normalized = json.loads(output.read_text())
    assert normalized["schema"] == HARDWARE_PROFILE_SCHEMA
    with pytest.raises(SystemExit) as same_path:
        validate_hardware_profile([str(source), "--output", str(source)])
    assert same_path.value.code == 2
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as malformed_exit:
        validate_hardware_profile([str(malformed)])
    assert malformed_exit.value.code == 2
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema": HARDWARE_PROFILE_SCHEMA}), encoding="utf-8")
    with pytest.raises(SystemExit) as invalid_exit:
        validate_hardware_profile([str(invalid)])
    assert invalid_exit.value.code == 2
