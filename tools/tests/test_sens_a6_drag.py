"""Tests for the drag-on side run of the EXP-021 arm b delay limit."""
from __future__ import annotations

import json

import pytest

import sim.physics.control_tolerance as ct
from tools import sens_a6_drag as sens


def _config():
    return json.loads(sens.CONFIG_PATH.read_text(encoding="utf-8"))


def test_patch_sets_drag_on_harness_vehicles_and_restores():
    original = ct.ActuatedQuadrotor
    profile = ct._profile_obj(str(sens.PROFILES["crazyflie-class"]))
    with sens.drag_patch(1.0, 0.012) as built:
        vehicle = ct.ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0))
        assert vehicle.drone.drag_coefficient == 1.0
        assert vehicle.drone.cross_sectional_area == 0.012
    assert built == [(1.0, 0.012)]
    assert ct.ActuatedQuadrotor is original


def test_no_patch_values_keep_profile_drag():
    profile = ct._profile_obj(str(sens.PROFILES["esp-fly-class"]))
    with sens.drag_patch(None, None) as built:
        ct.ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0))
    # Both profiles set Cd = 0 and area = 0.001 m^2.
    assert built == [(0.0, 0.001)]


def test_patch_reaches_run_episode_and_changes_the_flight():
    kwargs = {"duration_s": 3.0, "scoring_windows": _config()["scoring_windows"]}
    path = str(sens.PROFILES["crazyflie-class"])
    with sens.drag_patch(None, None) as off_built:
        off = ct.run_episode(path, "b", "step", 0.1, 0.0, 0, **kwargs)
    with sens.drag_patch(1.0, 0.012) as on_built:
        on = ct.run_episode(path, "b", "step", 0.1, 0.0, 0, **kwargs)
    with sens.drag_patch(1.0, 0.012):
        again = ct.run_episode(path, "b", "step", 0.1, 0.0, 0, **kwargs)
    assert off_built == [(0.0, 0.001)] and on_built == [(1.0, 0.012)]
    assert on["rms_error_m"] != off["rms_error_m"]
    assert on == again


def test_first_failure_flags_a_pass_after_the_first_fail():
    rows = [{"delay_s": d, "bad": bad} for d, bad in
            [(0.3, False), (0.4, True), (0.5, False), (0.6, True)]]
    summary = sens._first_failure(rows, "bad")
    assert summary == {"first_fail_delay_s": 0.4, "monotone": False, "passes_after_first_fail_s": [0.5]}


def test_first_failure_with_no_failure():
    rows = [{"delay_s": 0.3, "bad": False}]
    assert sens._first_failure(rows, "bad") == {"first_fail_delay_s": None, "monotone": True}


def test_control_comparison_needs_every_registered_field():
    registered = sens.registered_arm_b("crazyflie-class")
    assert sens.control_matches(dict(registered), registered)
    changed = dict(registered, d_crit_s=0.37)
    assert not sens.control_matches(changed, registered)


def test_refuses_output_under_registered_runs():
    with pytest.raises(ValueError, match="sim/runs"):
        sens.run(sens.ROOT / "sim" / "runs" / "sens-a6-drag", areas=(), profiles=())


def test_scan_control_matches_with_float_grid_delays():
    # 35 * 0.01 is 0.35000000000000003, while the registered entry stores 0.35.
    registered = sens.registered_arm_b("crazyflie-class")
    rows = [
        {"delay_s": 35 * ct.CONTROL_DT_S, **registered["stable_side"]},
        {"delay_s": 36 * ct.CONTROL_DT_S, **registered["unstable_side"]},
    ]
    result = {"rows": rows, "exp021": {"first_fail_delay_s": 36 * ct.CONTROL_DT_S}}
    assert sens.scan_matches_registered(result, registered)


def test_scan_control_rejects_a_shifted_boundary_or_missing_row():
    registered = sens.registered_arm_b("crazyflie-class")
    rows = [
        {"delay_s": 0.35, **registered["stable_side"]},
        {"delay_s": 0.36, **registered["unstable_side"]},
    ]
    shifted = {"rows": rows, "exp021": {"first_fail_delay_s": 0.37}}
    missing = {"rows": rows[1:], "exp021": {"first_fail_delay_s": 0.36}}
    assert not sens.scan_matches_registered(shifted, registered)
    assert not sens.scan_matches_registered(missing, registered)
