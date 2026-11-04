"""Tests for the gated constant-acceleration study (tools/sens_gate_ca.py)."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from sim.estimation.base import EstimatorSettings
from sim.estimation.engine import deliver_cell, run_arm
from sim.estimation.factory import make_estimator
from sim.estimation.kinematics import ca_matrices
from tools import sens_gate_ca as ca
from tools import sens_gate_switching as sw
from tools import sens_outlier_gate as parent
from tools import sens_outlier_gate_turn as turn

# Independent hand case: constant 2 m/s^2 from rest, no noise, so x(1 s) = 0.5 * 2 * 1^2 = 1.0 m.

def test_ca_transition_hand_case_one_step_and_ten_steps():
    state = np.zeros(9)
    state[6] = 2.0  # x acceleration
    F1, _ = ca_matrices(1.0, 0.1)
    one = F1 @ state
    F01, _ = ca_matrices(0.1, 0.1)
    ten = state.copy()
    for _ in range(10):
        ten = F01 @ ten
    for s in (one, ten):
        assert s[0] == pytest.approx(1.0, abs=1e-12)
        assert s[3] == pytest.approx(2.0, abs=1e-12)
        assert s[6] == pytest.approx(2.0, abs=1e-12)


def test_ca_predict_positions_hand_case():
    est = make_estimator("ca", EstimatorSettings(), np.zeros(3))
    est.x[6] = 2.0
    p = est.predict_positions([1.0])[0]
    assert p == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert est.predict_at(1.0).position_m == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_ca_filter_learns_constant_acceleration_and_cv_does_not():
    """Noise-free x = t^2 (2 m/s^2 from rest) at 10 Hz for 5 s, then predict 1 s ahead (truth 36 m).
    CA should be within 1 cm; CV misses at least the 0.5 a dt^2 = 1 m term."""
    settings = EstimatorSettings()
    errors = {}
    for model in ("ca", "cv"):
        est = make_estimator(model, settings, np.zeros(3))
        for k in range(1, 51):
            t = 0.1 * k
            est.update(np.array([t * t, 0.0, 0.0]), t)
        errors[model] = abs(est.predict_at(6.0).position_m[0] - 36.0)
    assert errors["ca"] < 0.01
    assert errors["cv"] > 0.9


# Wiring and arms (restatements of the code, kept as guards).

def test_control_arms_are_the_switching_study_arms():
    want = {a.name: a for a in turn.ARMS}
    for arm in ca.CONTROL_ARMS:
        assert arm == want[arm.name]


def test_new_arms_differ_only_in_model_and_lockout():
    lockouts = {a.name: (a.model, a.gate_reset_after, a.gate_probability) for a in ca.NEW_ARMS}
    assert lockouts == {"ca": ("ca", 3, None), "ca_gated": ("ca", 3, 0.99), "ca_gated_l2": ("ca", 2, 0.99),
                        "ca_gated_l1": ("ca", 1, 0.99), "imm_gated_l1": ("imm", 1, 0.99)}
    assert all(a.stamping == "capture" and a.late_policy == "drop" for a in ca.NEW_ARMS)


def test_ca_rows_equal_a_direct_engine_call():
    spec = parent.OutlierSpec("jump", 0.03)
    seeds = [11, 12]
    rows = sw.cell_runs("start_stop", 0.1, spec, seeds, arms=ca.NEW_ARMS)
    cell = sw.switching_cell("start_stop", 0.1)
    for r in rows:
        bundle = parent.inject(deliver_cell(cell, r["seed"]), spec, r["seed"])
        arm = next(a for a in ca.NEW_ARMS if a.name == r["arm"])
        o = run_arm(bundle, cell, arm)
        assert (r["p95_error_m"], r["max_error_m"], r["gated_count"]) == (o.p95_error_m, o.max_error_m,
                                                                           o.gated_count)


def test_lockout_one_forced_accepts_never_exceed_rejects():
    """The forced-accept count can never exceed the reject count with lockout 1."""
    rows = sw.cell_runs("start_stop", 0.1, parent.OutlierSpec("jump", 0.03), [11, 12, 13],
                        arms=(ca.NEW_ARMS[3],))
    for r in rows:
        assert r["forced_accepts"] <= r["gated_count"]


# Paired cost and the control comparison, on hand-made rows.

def test_paired_gate_cost_hand_case():
    rows = [{"arm": "ca", "seed": s, "failed": f, "p95_error_m": p}
            for s, f, p in ((1, False, 0.10), (2, True, 0.30), (3, False, 0.20))]
    rows += [{"arm": "ca_gated", "seed": s, "failed": f, "p95_error_m": p}
             for s, f, p in ((1, True, 0.26), (2, False, 0.24), (3, False, 0.21))]
    (cost,) = ca.paired_gate_cost(rows, pairs=(("ca", "ca_gated"),))
    assert (cost["only_gated_fails"], cost["only_ungated_fails"]) == (1, 1)
    assert cost["median_p95_diff_m"] == pytest.approx(0.01)


def test_paired_gate_cost_refuses_mismatched_seeds():
    rows = [{"arm": "ca", "seed": 1, "failed": False, "p95_error_m": 0.1},
            {"arm": "ca_gated", "seed": 2, "failed": False, "p95_error_m": 0.1}]
    with pytest.raises(ValueError):
        ca.paired_gate_cost(rows, pairs=(("ca", "ca_gated"),))


def _write_reference(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_to_switching_finds_equal_and_changed_rows(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "_outcome"}
            for r in sw.cell_runs("start_stop", 0.1, parent.OutlierSpec("jump", 0.0), [11],
                                  arms=ca.CONTROL_ARMS)]
    ref = tmp_path / "per_seed.csv"
    _write_reference(ref, rows)
    same = ca.compare_to_switching(rows, ref)
    assert (same["compared"], same["missing"], same["differences"]) == (4, 0, 0)
    changed = [dict(r) for r in rows]
    changed[0]["p95_error_m"] += 1e-9
    assert ca.compare_to_switching(changed, ref)["differences"] == 1


def test_refuses_to_write_under_sim_runs():
    with pytest.raises(SystemExit):
        ca.main(["--output", str(ca.ROOT / "sim" / "runs" / "nope"), "--seeds", "1"])


def test_main_exits_nonzero_when_the_control_differs(tmp_path):
    """One seed: the reference holds the real control rows with one p95 changed, so the run must fail."""
    rows = [{k: v for k, v in r.items() if k != "_outcome"}
            for spec_cell in ca.plan()
            for r in sw.cell_runs(spec_cell[0], spec_cell[1], spec_cell[2], [ca.seed_list(1)[0]],
                                  arms=ca.CONTROL_ARMS)]
    rows[0]["p95_error_m"] += 1e-9
    ref = tmp_path / "ref.csv"
    _write_reference(ref, rows)
    assert ca.main(["--output", str(tmp_path / "out"), "--seeds", "1", "--compare", str(ref)]) == 1
