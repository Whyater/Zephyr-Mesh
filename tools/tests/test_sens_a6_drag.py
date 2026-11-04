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


# --long-check ---------------------------------------------------------------

def _long_row(**overrides):
    row = {"growing": False, "stopped_at_s": None, "window_peak_tilt_deg": [5.0, None],
           "window_lowest_z_m": [0.9, None]}
    row.update(overrides)
    return row


def test_long_fail_flags_growth_tilt_and_floor_separately():
    assert sens.long_fail(_long_row()) == (False, [])
    assert sens.long_fail(_long_row(growing=True)) == (True, ["growing"])
    assert sens.long_fail(_long_row(growing=True, stopped_at_s=12.0)) == (True, ["left_box"])
    assert sens.long_fail(_long_row(window_peak_tilt_deg=[5.0, 45.5])) == (True, ["tilt_over_45_deg"])
    # Exactly 45 deg is not over the limit; exactly z = 0 is not below the floor.
    assert sens.long_fail(_long_row(window_peak_tilt_deg=[45.0], window_lowest_z_m=[0.0])) == (False, [])
    assert sens.long_fail(_long_row(window_lowest_z_m=[0.5, -0.01])) == (True, ["below_floor"])


def test_long_limit_reports_first_fail_and_non_monotone_passes():
    rows = [{"delay_s": d, "long_fail": f} for d, f in [(0.34, True), (0.32, False), (0.33, False), (0.35, False)]]
    assert sens.long_limit(rows) == {"limit_s": 0.34, "monotone": False, "passes_after_limit_s": [0.35]}
    assert sens.long_limit(rows[1:3]) == {"limit_s": None, "monotone": True, "passes_after_limit_s": []}


def test_long_report_serializer_reproduces_the_registered_file():
    # Feeding the registered rows back through the serializer must give the
    # registered bytes, and those bytes must carry the registry hash.
    import hashlib
    registered = sens.REGISTERED_LONG.read_bytes()
    rows = json.loads(registered)["rows"]
    rebuilt = sens.long_report_bytes(list(reversed(rows)))
    assert rebuilt == registered
    assert hashlib.sha256(rebuilt).hexdigest() == sens.registered_long_sha256()


def test_long_patch_reaches_fly_changes_the_flight_and_restores(monkeypatch):
    lc = sens._long_module()
    original = lc.ActuatedQuadrotor
    monkeypatch.setattr(lc, "DURATION_S", 30.0)  # three 10 s windows keeps the test short
    job = ("crazyflie-class", "b", 0.2, 1.0)
    off, off_built = sens._fly_rows([job], None, None)
    on, on_built = sens._fly_rows([job], 1.0, 0.012)
    assert off_built == [(0.0, 0.001)] and on_built == [(1.0, 0.012)]
    assert on[0]["window_peak_error_m"] != off[0]["window_peak_error_m"]
    assert lc.ActuatedQuadrotor is original


def test_long_check_refuses_registered_runs_folder():
    with pytest.raises(ValueError, match="sim/runs"):
        sens.run_long_check(sens.ROOT / "sim" / "runs" / "sens-a6-drag-long", levels=(), profiles=())


def test_long_check_stops_before_drag_levels_when_control_differs(tmp_path, monkeypatch):
    # A control file that does not match the registered bytes must block every drag level.
    (tmp_path / "control_long_check.json").write_bytes(b'{"rows": []}\n')
    lc = sens._long_module()
    monkeypatch.setattr(lc, "fly", lambda job: pytest.fail("no level may fly after a failed control"))
    result = sens.run_long_check(tmp_path, log=lambda msg: None)
    assert result["controls_pass"] is False
    assert result["levels"] == {}
    assert not list(tmp_path.glob("level_*.json"))


def test_long_check_reuses_a_finished_level(tmp_path, monkeypatch):
    # With the registered control in place and a finished level file, nothing flies again.
    (tmp_path / "control_long_check.json").write_bytes(sens.REGISTERED_LONG.read_bytes())
    level = {"config": {"level": "off", "drag_coefficient": None, "area_m2": None,
                        "profiles": ["crazyflie-class"], "delays_s": [0.37], "steps_m": [1.0, 0.01]},
             "effective_drag_coefficient": 0.0, "effective_area_m2": 0.001,
             "rows": [{"profile": "crazyflie-class", "arm": "b", "delay_s": 0.37, "step_m": s,
                       "long_fail": True, "long_fail_reasons": ["growing"]} for s in (1.0, 0.01)]}
    (tmp_path / "level_off.json").write_text(json.dumps(level), encoding="utf-8")
    lc = sens._long_module()
    monkeypatch.setattr(lc, "fly", lambda job: pytest.fail("finished level must not fly again"))
    result = sens.run_long_check(tmp_path, levels=(("off", None, None),), profiles=("crazyflie-class",),
                                 delays=(0.37,), log=lambda msg: None)
    assert result["controls_pass"] is True
    assert result["levels"]["off"]["limits"]["crazyflie-class step 1"]["limit_s"] == 0.37
    # The summary says both files were reused, so a reader can tell a stale folder from a fresh run.
    assert result["control"]["flown_this_invocation"] is False
    assert result["levels"]["off"]["flown_this_invocation"] is False
