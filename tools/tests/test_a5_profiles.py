"""Checks on the two literature-derived micro-quad profiles.

The derived numbers are worked by hand from the profile fields, so a typo in
a profile value or in the loader shows up here.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.hardware import RotorcraftProfile, load_hardware_profile

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "sim" / "physics" / "profiles"
SCHEMA = json.loads((ROOT / "sim" / "hardware_profile_schema_v2.json").read_text())
G = 9.81

# Hand numbers: hover speed = sqrt(m g / (4 k_f)), thrust-to-weight = 4 k_f w_max^2 / (m g).
EXPECTED = {
    # sqrt(0.028 * 9.81 / (4 * 1.9e-8)) = sqrt(3.6142e6) = 1901.1 rad/s
    # 4 * 1.9e-8 * 2846^2 / (0.028 * 9.81) = 0.61558 / 0.27468 = 2.241
    "crazyflie_class.json": {"hover": 1901.1, "tw": 2.241, "tau": 0.072},
    # sqrt(0.025 * 9.81 / (4 * 4.0e-9)) = sqrt(1.5328e7) = 3915.1 rad/s
    # 4 * 4.0e-9 * 6450^2 / (0.025 * 9.81) = 0.66564 / 0.24525 = 2.714
    "esp_fly_class.json": {"hover": 3915.1, "tw": 2.714, "tau": 0.07},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_profile_is_valid_literature_derived_and_matches_hand_numbers(name):
    path = PROFILE_DIR / name
    document = json.loads(path.read_text())
    Draft202012Validator(SCHEMA).validate(document)
    profile = load_hardware_profile(path)
    assert isinstance(profile, RotorcraftProfile)
    assert profile.status == "literature-derived"
    assert profile.hover_speed_rad_s(G) == pytest.approx(EXPECTED[name]["hover"], abs=0.1)
    assert profile.thrust_to_weight(G) == pytest.approx(EXPECTED[name]["tw"], abs=0.001)
    assert profile.motor_time_constant_s == EXPECTED[name]["tau"]
    # No UNCONFIRMED value may sit in a default, and only body drag may be EXCLUDED.
    statuses = {path: entry["status"] for path, entry in document["parameters"].items()}
    assert "UNCONFIRMED" not in statuses.values()
    assert {p for p, s in statuses.items() if s == "EXCLUDED"} == {
        "airframe.drag_coefficient", "airframe.cross_sectional_area_m2"}
    assert profile.drag_coefficient == 0.0
    for entry in document["parameters"].values():
        if entry["status"] != "EXCLUDED":
            assert entry["url"].startswith("http")
            assert entry["accessed"] == "2026-10-14"


def test_crazyflie_max_speed_reproduces_the_cross_checked_max_thrust():
    """w_max was chosen so k_f w_max^2 equals the cross-checked 0.1539 N per rotor."""
    profile = load_hardware_profile(PROFILE_DIR / "crazyflie_class.json")
    assert profile.max_thrust_per_rotor_n() == pytest.approx(0.1539, rel=1e-4)


def test_profiles_regenerate_from_the_builder(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools" / "a_build_profiles.py"), str(tmp_path)],
                   check=True, capture_output=True)
    for name in EXPECTED:
        assert (tmp_path / name).read_bytes() == (PROFILE_DIR / name).read_bytes()
