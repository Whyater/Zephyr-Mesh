"""Tests for the slow-motor attitude-gain side run (tools/sens_slow_motor_gains.py)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import sim.physics.control_tolerance as ct
from tools import sens_slow_motor_gains as sm
from tools.a_profile_sensitivity import build_variant

SHORT_S = 2.0


def _short(profile, arm):
    return ct.run_episode(profile, arm, "step", 0.0, 0.0, 0, duration_s=SHORT_S, failed_to_fly_rule=True)


def test_gain_law_keeps_damping_ratio():
    kp, kd = sm.scaled_gains(0.5)
    assert kp == (25.0, 25.0, 25.0)
    assert kd == (9.0, 9.0, 9.0)
    for s in sm.SCALES:
        kp, kd = sm.scaled_gains(s)
        # zeta = K_d / (2 sqrt(K_p)) stays at the stock 0.9.
        assert kd[0] / (2.0 * math.sqrt(kp[0])) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        sm.scaled_gains(0.0)


def test_gain_patch_sets_logs_and_restores():
    with sm.gain_patch(0.5) as built:
        controller = ct.GeometricFlightController()
        np.testing.assert_array_equal(controller.K_p_att, [25.0, 25.0, 25.0])
        np.testing.assert_array_equal(controller.K_d_att, [9.0, 9.0, 9.0])
        np.testing.assert_array_equal(controller.K_p_pos, [2.0, 2.0, 3.0])
    assert built == [((25.0, 25.0, 25.0), (9.0, 9.0, 9.0))]
    assert ct.GeometricFlightController is sm.BASE_CONTROLLER


def test_gain_patch_restores_after_error():
    with pytest.raises(RuntimeError), sm.gain_patch(0.8):
        raise RuntimeError("boom")
    assert ct.GeometricFlightController is sm.BASE_CONTROLLER


def test_scale_one_flies_identically_to_unpatched():
    profile = sm.profile_with_tau(0.07)
    for arm in ("b", "c"):
        stock = _short(profile, arm)
        with sm.gain_patch(1.0) as built:
            patched = _short(profile, arm)
        assert built, "the harness must build its controllers through the patch"
        assert json.dumps(patched, sort_keys=True) == json.dumps(stock, sort_keys=True)


def test_patch_reaches_the_delayed_controllers():
    profile = sm.profile_with_tau(0.07)
    stock = _short(profile, "b")
    with sm.gain_patch(0.5) as built:
        slow = _short(profile, "b")
    # run_episode builds one plain controller, then the arm b wrapper builds its inner one.
    assert len(built) == 2
    assert slow["rms_error_m"] != stock["rms_error_m"]


def test_profile_with_tau():
    nominal = sm.profile_with_tau(0.07)
    assert nominal.motor_time_constant_s == 0.07
    assert nominal.as_document() == sm.load_hardware_profile(sm.PROFILE_PATH).as_document()
    slow = sm.profile_with_tau(0.16)
    a5_high = build_variant(nominal, sm.TAU_PARAM, "high")
    assert slow.motor_time_constant_s == a5_high.motor_time_constant_s == 0.16
    assert slow.as_document()["rotor"] == a5_high.as_document()["rotor"]
    fast = sm.profile_with_tau(0.03)
    entry = fast.as_document()["parameters"][sm.TAU_PARAM]
    assert fast.motor_time_constant_s == 0.03
    assert entry["range"] == [0.03, 0.16]
    assert "widened" in entry["note"]


def _row(d, unstable=False, ftf=False):
    return {"delay_s": d, "unstable": unstable, "failed_to_fly": ftf}


def test_first_failure_and_combined_limit():
    rows = [_row(0.0), _row(0.1), _row(0.2, ftf=True), _row(0.3, unstable=True), _row(0.4)]
    assert sm.first_failure(rows, "unstable") == {
        "first_fail_delay_s": 0.3, "monotone": False, "passes_after_first_fail_s": [0.4]}
    assert sm.first_failure(rows, "failed_to_fly")["first_fail_delay_s"] == 0.2
    # Pre-registered rule: the first delay that fails either verdict.
    assert sm._limit(rows) == 0.2
    assert sm._limit([_row(0.0), _row(0.1)]) is None


def _cell(stable, pm, limit):
    scan = None if limit is None else {"limit_s": limit}
    return {"stable_at_zero_delay": stable, "hand": {"attitude_pm_minus_hold_deg": pm}, "scan_b": scan}


def test_selection_rules():
    cells = {}
    for tau in sm.TAUS_S:
        for s, pm in zip(sm.SCALES, (-0.1, 3.0, 6.0, 11.5, 18.6)):
            if tau == sm.FINE_TAU_S:
                cells[(tau, s)] = _cell(s <= 0.8, pm, None if s > 0.8 else 0.2)
            else:
                cells[(tau, s)] = _cell(True, 40.0, {1.0: 0.36, 0.9: 0.33, 0.8: 0.30, 0.65: 0.25, 0.5: 0.18}[s])
    sel = sm.select(cells)
    assert sel["s_min"] == 0.8
    assert sel["s_margin"] == 0.65
    cost = sel["cost_arm_b"]["tau=0.07 s_margin=0.65"]
    assert cost["relative_change"] == pytest.approx((0.25 - 0.36) / 0.36)


def test_hand_margin_matches_closed_form_without_lag():
    # Independent check: with no motor lag the attitude open loop is (Kp + Kd s) / s^2.
    # |L| = 1 where w^4 = Kp^2 + Kd^2 w^2, and the phase margin is atan(Kd w / Kp).
    margins = []
    for s in (1.0, 0.65, 0.5):
        kp, kd = 100.0 * s * s, 18.0 * s
        w = math.sqrt((kd**2 + math.sqrt(kd**4 + 4.0 * kp**2)) / 2.0)
        pm_deg = math.degrees(math.atan(kd * w / kp))
        h = sm.hand(0.0, s)
        assert h["attitude_omega_c"] == pytest.approx(w, rel=1e-4)
        assert h["attitude_pm_deg"] == pytest.approx(pm_deg, abs=1e-3)
        assert h["attitude_pm_minus_hold_deg"] == pytest.approx(pm_deg - math.degrees(w * 0.005), abs=1e-3)
        margins.append(h["attitude_pm_deg"])
    # With damping kept and no lag, the lead angle does not depend on s: only a
    # motor lag (or the hold) makes a slower loop safer.
    assert max(margins) - min(margins) < 1e-3


def test_motor_lag_shrinks_hand_margin():
    margins = [sm.hand(tau, 1.0)["attitude_pm_deg"] for tau in sm.TAUS_S]
    assert margins == sorted(margins, reverse=True)
    assert sm.hand(0.16, 1.0)["attitude_pm_minus_hold_deg"] < 0.0 < sm.hand(0.16, 0.8)["attitude_pm_minus_hold_deg"]


def test_output_under_sim_runs_is_refused():
    with pytest.raises(ValueError):
        sm.run(sm.ROOT / "sim" / "runs" / "never-made")


def test_patch_reaches_the_whole_state_controller():
    profile = sm.profile_with_tau(0.07)
    stock = _short(profile, "c")
    with sm.gain_patch(0.5) as built:
        slow = _short(profile, "c")
    # One plain controller from run_episode, one inner controller in the slice c wrapper.
    assert len(built) == 2
    assert slow["rms_error_m"] != stock["rms_error_m"]


def test_failed_control_skips_every_variant(tmp_path, monkeypatch):
    ran = []

    def fake_cell(cells_dir, tau, scale, windows, scan=True):
        ran.append((tau, scale))
        return {"scan_b": None, "zero_delay": {"b": {"unstable": False}, "c": {"unstable": False}}}

    monkeypatch.setattr(sm, "control_dcrit", lambda windows: {
        "arm_b": {"matches_registered": False}, "slice_c": {"matches_registered": True}})
    monkeypatch.setattr(sm, "_cached_cell", fake_cell)
    # The other controls pass, so the arm b flag alone must close the gate.
    monkeypatch.setattr(sm, "scan_matches_registered", lambda cell: True)
    monkeypatch.setattr(sm, "control_a5_slow_motor", lambda cell: True)
    result = sm.run(tmp_path / "out")
    assert result["controls_pass"] is False
    assert "selection" not in result and "table" not in result
    # Only the two stock control cells ran.
    assert ran == [(0.07, 1.0), (0.16, 1.0)]
    assert not (tmp_path / "out" / "table.csv").exists()


# ---- motor-lag gap sweep (--mode lag-gap) ----

def _zero(b_fail=False, c_fail=False, c_ftf=False):
    return {"b": {"unstable": b_fail, "failed_to_fly": False},
            "c": {"unstable": c_fail, "failed_to_fly": c_ftf}}


def _gap_cell(pm, zero, b_limit, c_limit, s=1.0):
    def scan(lim):
        return None if lim == "x" else {"limit_s": lim, "scan_top_s": 0.5,
                                        "exp021": {"monotone": True}, "failed_to_fly": {"monotone": True}}
    kp, kd = sm.scaled_gains(s)
    return {"zero_delay": zero, "hand": {"attitude_pm_minus_hold_deg": pm},
            "scan_b": scan(b_limit), "scan_c": scan(c_limit),
            "effective_K_p_att": list(kp), "effective_K_d_att": list(kd)}


def test_gap_grid_is_the_preregistered_one():
    assert sm.GAP_TAUS_S == (0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
    assert len(sm.GAP_FINE_TAUS_S) == 31
    assert sm.GAP_FINE_TAUS_S[0] == 0.1 and sm.GAP_FINE_TAUS_S[-1] == sm.FINE_TAU_S
    # Every grid lag is also a fine-scan lag, so the fine scan reuses those cells.
    assert set(sm.GAP_TAUS_S) <= set(sm.GAP_FINE_TAUS_S)


def test_flies_at_zero_needs_both_arms_and_both_rules():
    assert sm.flies_at_zero({"zero_delay": _zero()})
    assert not sm.flies_at_zero({"zero_delay": _zero(b_fail=True)})
    assert not sm.flies_at_zero({"zero_delay": _zero(c_fail=True)})
    assert not sm.flies_at_zero({"zero_delay": _zero(c_ftf=True)})


def test_gap_selection_rules():
    cells = {}
    for s, pm in zip(sm.SCALES, (4.0, 7.0, 11.0, 16.0, 23.0)):
        # Stock flies only in arm b; slice c fails at zero, so stock must not count as flying.
        zero = _zero(c_fail=True) if s == 1.0 else _zero()
        cells[(0.13, s)] = _gap_cell(pm, zero, {1.0: 0.30, 0.9: 0.31, 0.8: None, 0.65: 0.27, 0.5: 0.19}[s],
                                     "x" if s == 1.0 else 0.02, s)
    (row,) = sm.gap_select(cells, 0.36, taus=(0.13,))
    assert row["stock_flies_at_zero_delay"] is False
    assert row["scales_flying_at_zero_delay"] == [0.9, 0.8, 0.65, 0.5]
    assert row["s_fly"]["scale"] == 0.9
    assert row["s_fly"]["arm_b_change_vs_headline"] == pytest.approx((0.31 - 0.36) / 0.36)
    assert row["s_fly"]["K_p_att"] == pytest.approx(81.0)
    assert row["s_margin"]["scale"] == 0.8
    # A limit above the scanned range has no change number and prints as "> top".
    assert row["s_margin"]["arm_b_change_vs_headline"] is None
    assert sm._fmt_limit(row["s_margin"]["arm_b"]) == "> 0.50"
    assert sm._fmt_limit(None) == "unstable at 0"


def test_stock_edge_reports_return_to_flying():
    fine = {t: {"zero_delay": _zero(b_fail=t in (0.152, 0.156))} for t in (0.15, 0.152, 0.154, 0.156)}
    assert sm.stock_edge(fine) == {"last_flying_s": 0.15, "first_failing_s": 0.152,
                                   "flies_again_after_first_fail_s": [0.154]}
    assert sm.stock_edge({0.1: {"zero_delay": _zero()}})["first_failing_s"] is None


def test_gap_mode_failed_control_skips_every_variant(tmp_path, monkeypatch):
    ran = []

    def fake_cell(cells_dir, tau, scale, windows, scan=True):
        ran.append((tau, scale))
        return {"scan_b": None, "zero_delay": _zero()}

    monkeypatch.setattr(sm, "control_dcrit", lambda windows: {
        "arm_b": {"matches_registered": True}, "slice_c": {"matches_registered": False}})
    monkeypatch.setattr(sm, "_cached_cell", fake_cell)
    monkeypatch.setattr(sm, "scan_matches_registered", lambda cell: True)
    monkeypatch.setattr(sm, "control_a5_slow_motor", lambda cell: True)
    result = sm.run_gap(tmp_path / "gap")
    assert result["controls_pass"] is False
    assert "per_lag" not in result
    assert ran == [(0.07, 1.0), (0.16, 1.0)]
    assert not (tmp_path / "gap" / "gap_table.csv").exists()


def test_gap_output_under_sim_runs_is_refused():
    with pytest.raises(ValueError):
        sm.run_gap(sm.ROOT / "sim" / "runs" / "never-made")


def test_stability_rule_text_has_no_decision_number():
    import re
    assert not re.search(r"\bD-\d", sm.STABILITY_RULE)
    assert "failed-to-fly rule" in sm.STABILITY_RULE
