"""Checks for the strict literature-profile schema and loader."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.hardware import (
    HARDWARE_PROFILE_SCHEMA_V2,
    RotorcraftProfile,
    load_hardware_profile,
)
from tools.validate_hardware_profile import main as validate_hardware_profile


def _parameter(value, *, status="CONFIRMED", note=None):
    if isinstance(value, list):
        bounds = [[item, item] for item in value]
    else:
        bounds = [value, value]
    result = {
        "value": value,
        "range": bounds,
        "source": "fixture",
        "url": "https://example.invalid/fixture",
        "accessed": "2026-01-01",
        "status": status,
    }
    if note is not None:
        result["note"] = note
    return result


def _document(*, status="literature-derived"):
    airframe = {
        "mass_kg": 0.030,
        "inertia_kg_m2": [1.0e-5, 1.0e-5, 2.0e-5],
        "arm_length_m": 0.040,
        "drag_coefficient": 0.0,
        "cross_sectional_area_m2": 0.010,
    }
    rotor = {
        "count": 4,
        "layout": "x",
        "thrust_coefficient_n_s2": 3.0e-8,
        "torque_to_thrust_m": 0.006,
        "max_speed_rad_s": 2500.0,
        "motor_time_constant_s": 0.030,
        "prop_radius_m": 0.020,
        "spin_directions": [1, -1, 1, -1],
    }
    parameters = {
        f"airframe.{name}": _parameter(value)
        for name, value in airframe.items()
    }
    parameters.update({f"rotor.{name}": _parameter(value) for name, value in rotor.items() if name != "layout"})
    return {
        "schema": HARDWARE_PROFILE_SCHEMA_V2,
        "profile_id": "fixture-v2",
        "status": status,
        "notes": "round-number fixture",
        "airframe": airframe,
        "rotor": rotor,
        "battery": None,
        "parameters": parameters,
    }


def test_v2_schema_and_derived_helpers(tmp_path):
    document = _document()
    schema = json.loads((Path(__file__).parents[2] / "sim" / "hardware_profile_schema_v2.json").read_text())
    Draft202012Validator(schema).validate(document)
    profile = RotorcraftProfile.from_dict(document)
    assert profile.hover_speed_rad_s() == pytest.approx((0.030 * 9.81 / (4 * 3.0e-8)) ** 0.5)
    assert profile.max_thrust_per_rotor_n() == pytest.approx(3.0e-8 * 2500.0**2)
    assert profile.thrust_to_weight() == pytest.approx(4 * profile.max_thrust_per_rotor_n() / (0.030 * 9.81))
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_hardware_profile(path)
    assert isinstance(loaded, RotorcraftProfile)
    assert loaded.as_document() == document


def test_literature_profile_requires_complete_non_unconfirmed_parameters():
    missing = _document()
    del missing["parameters"]["rotor.max_speed_rad_s"]
    with pytest.raises(ValueError, match="missing parameters"):
        RotorcraftProfile.from_dict(missing)
    unconfirmed = _document()
    unconfirmed["parameters"]["rotor.max_speed_rad_s"]["status"] = "UNCONFIRMED"
    with pytest.raises(ValueError, match="UNCONFIRMED"):
        RotorcraftProfile.from_dict(unconfirmed)

    broadcast_range = _document()
    broadcast_range["parameters"]["airframe.inertia_kg_m2"]["range"] = [0.0, 2.0e-5]
    assert RotorcraftProfile.from_dict(broadcast_range).inertia_kg_m2 == (1.0e-5, 1.0e-5, 2.0e-5)


def test_measured_v2_and_parameter_mismatch_are_rejected():
    measured = _document(status="measured")
    with pytest.raises(ValueError, match="measured v2 profiles need the provenance block"):
        RotorcraftProfile.from_dict(measured)
    mismatch = _document()
    mismatch["parameters"]["airframe.mass_kg"]["value"] = 0.031
    with pytest.raises(ValueError, match="must equal"):
        RotorcraftProfile.from_dict(mismatch)


def test_validator_prints_v2_derived_numbers(tmp_path, capsys):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")
    assert validate_hardware_profile([str(path)]) == 0
    output = capsys.readouterr().out
    assert "hover_speed_rad_s:" in output
    assert "hover_thrust_per_rotor_n:" in output
    assert "thrust_to_weight:" in output
    assert "max_vertical_acceleration_mps2:" in output


def _excluded_drag(document, drag=0.0):
    document["airframe"]["drag_coefficient"] = drag
    for name in ("drag_coefficient", "cross_sectional_area_m2"):
        entry = document["parameters"][f"airframe.{name}"]
        entry["value"] = document["airframe"][name]
        entry["range"] = [entry["value"], entry["value"]]
        entry["status"] = "EXCLUDED"
        entry["note"] = "no source, so switched off"
    return document


def test_excluded_body_drag_is_accepted_only_when_switched_off():
    """EXCLUDED lets an unsourced effect be turned off without inventing a number."""
    schema = json.loads((Path(__file__).parents[2] / "sim" / "hardware_profile_schema_v2.json").read_text())
    accepted = _excluded_drag(_document())
    Draft202012Validator(schema).validate(accepted)
    assert RotorcraftProfile.from_dict(accepted).drag_coefficient == 0.0

    nonzero = _excluded_drag(_document(), drag=1.2)
    with pytest.raises(ValueError, match="must be 0"):
        RotorcraftProfile.from_dict(nonzero)

    wrong_field = _document()
    wrong_field["parameters"]["airframe.mass_kg"]["status"] = "EXCLUDED"
    wrong_field["parameters"]["airframe.mass_kg"]["note"] = "not allowed"
    with pytest.raises(ValueError, match="cannot be EXCLUDED"):
        RotorcraftProfile.from_dict(wrong_field)

    no_note = _excluded_drag(_document())
    del no_note["parameters"]["airframe.drag_coefficient"]["note"]
    with pytest.raises(ValueError):
        RotorcraftProfile.from_dict(no_note)

    area_only = _document()
    area_only["parameters"]["airframe.cross_sectional_area_m2"]["status"] = "EXCLUDED"
    area_only["parameters"]["airframe.cross_sectional_area_m2"]["note"] = "unused"
    with pytest.raises(ValueError, match="only be EXCLUDED when drag_coefficient"):
        RotorcraftProfile.from_dict(area_only)
