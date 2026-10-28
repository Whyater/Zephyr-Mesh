"""Hand check of the two thrust estimates for the example fixture profile.

The fixture declares a 3 inch (0.0762 m) prop rated to 36,000 rpm, a motor
rated to 38,000 rpm, 18 W and 0.0018 N m, and Ct = 0.10, Cp = 0.04.  The
numbers below are worked by hand from the rotor formulas, not taken from
the code:

* thrust per rotor  T = Ct * rho * n^2 * D^4     (n in rev/s)
* shaft power       P = Cp * rho * n^3 * D^5
* shaft torque      Q = P / (2 pi n) = Cp * rho * n^2 * D^5 / (2 pi)

At the prop limit, n = 36000 / 60 = 600 rev/s, the prop needs about
0.0072 N m and 27 W per motor, four times the declared torque. The motor
cannot reach that speed under load. Torque binds first, near 300 rev/s.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from sim.hardware import HardwareProfile

ROOT = Path(__file__).parents[2]
RHO = 1.225
G = 9.81


@pytest.fixture(scope="module")
def profile() -> HardwareProfile:
    return HardwareProfile.from_path(ROOT / "sim/hardware_profile_example.json")


def test_rpm_limit_thrust_matches_hand_value(profile):
    # D^4 = 0.0762^4 = 3.3717e-5 m^4; 0.1 * 1.225 * 600^2 * 3.3717e-5 = 1.4869 N per rotor.
    assert profile.estimated_thrust_at_rpm_limit_n() == pytest.approx(4 * 1.4869, rel=1e-3)
    assert profile.estimated_max_thrust_n() == profile.estimated_thrust_at_rpm_limit_n()


def test_declared_motor_cannot_drive_the_prop_to_its_rpm_limit(profile):
    # Q(600 rev/s) = 0.04 * 1.225 * 600^2 * 0.0762^5 / (2 pi) = 0.04 * 1.225 * 360000 * 2.5692e-6 / 6.2832
    torque_needed = 0.0072130
    power_needed = 0.04 * 1.225 * 600**3 * 2.5692e-6  # 27.19 W
    assert profile.propeller.estimated_torque_nm(36_000) == pytest.approx(torque_needed, rel=1e-3)
    assert profile.propeller.estimated_power_w(36_000) == pytest.approx(power_needed, rel=1e-3)
    assert torque_needed / profile.motor.max_torque_nm == pytest.approx(4.007, rel=1e-3)
    assert power_needed > profile.motor.max_power_w


def test_limited_thrust_matches_hand_value(profile):
    # Torque limit: n^2 = 2 pi * 0.0018 / (0.04 * 1.225 * 0.0762^5) = 0.011310 / 1.2589e-7 = 89,841,
    # n = 299.7 rev/s (17,984 rpm). Power limit: n = (18 / 1.2589e-7)^(1/3) = 522.9 rev/s. Torque binds.
    n = math.sqrt(2 * math.pi * 0.0018 / (0.04 * RHO * 0.0762**5))
    assert n == pytest.approx(299.7, abs=0.1)
    # T = 0.1 * 1.225 * 89,841 * 3.3717e-5 = 0.37107 N per rotor, 1.4843 N for four.
    assert profile.estimated_max_thrust_limited_n() == pytest.approx(4 * 0.37107, rel=1e-3)


def test_the_two_estimates_disagree_by_four_times_and_change_thrust_to_weight(profile):
    # Mass 0.03 + 4 * (0.01 + 0.002) = 0.078 kg, weight 0.765 N.
    weight = profile.total_mass_kg * G
    assert profile.total_mass_kg == pytest.approx(0.078)
    unlimited = profile.estimated_thrust_at_rpm_limit_n() / weight
    limited = profile.estimated_max_thrust_limited_n() / weight
    assert unlimited == pytest.approx(7.77, abs=0.01)
    assert limited == pytest.approx(1.94, abs=0.01)
    # The ratio is (600 / 299.7)^2 = 4.007: thrust grows with rpm squared.
    assert unlimited / limited == pytest.approx((600 / 299.7) ** 2, rel=1e-3)
