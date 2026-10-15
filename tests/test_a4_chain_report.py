"""Fast checks for the reproducible EXP-020 report tool."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools import a_make_vehicle_chain_report as report_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def crazyflie():
    return report_tool._profile("crazyflie-class")


def test_hover_matches_weight_split_over_four_rotors(crazyflie):
    result = report_tool.check_hover(crazyflie)
    # 0.028 kg * 9.81 / 4 is about 0.06867 N per rotor.
    assert result["per_rotor_thrust_n"]["hand_value"] == pytest.approx(0.028 * 9.81 / 4, rel=1e-3)
    assert result["per_rotor_thrust_n"]["pass"] and result["hover_speed_rad_s"]["pass"]


def test_motor_step_reaches_63_and_95_percent(crazyflie):
    result = report_tool.check_motor_step(crazyflie)
    assert result["speed_fraction_at_tau"]["hand_value"] == pytest.approx(1 - math.exp(-1))
    assert result["max_first_order_deviation"]["pass"]
    assert result["speed_fraction_at_3tau"]["pass"]


def test_full_throttle_climb_through_the_chain(crazyflie):
    result = report_tool.check_max_climb(crazyflie)
    # Hand: 4 * 1.9e-8 * 2846^2 / 0.028 - 9.81, about 12.17 m/s^2 (thrust to weight 2.24).
    assert result["vertical_acceleration_mps2"]["hand_value"] == pytest.approx(12.17, rel=2e-3)
    assert result["all_window_steps_saturated"]
    assert result["vertical_acceleration_mps2"]["pass"]


def test_battery_sag_keeps_speed_and_raises_duty(crazyflie):
    result = report_tool.check_battery_sag(crazyflie, duration=1.0)
    assert result["hover_speed_constant"] and result["hover_duty_monotonic"]
    assert all(row["terminal_voltage"]["pass"] for row in result["samples"])


def test_short_hover_hold_and_yaw_trim(crazyflie):
    hold = report_tool.check_hover_hold(crazyflie, durations=(2.0,))
    assert all(v["max_position_drift_m"]["pass"] for v in hold.values())
    yaw = report_tool.check_yaw_trim(crazyflie, duration=1.0)
    assert yaw["reaction_yaw_torque_nm"]["pass"]


def test_refuses_non_empty_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        report_tool.generate(output)
