"""Checks for the EXP-037 runner (tools/run_s6_adaptive_r.py): the pre-registered rule is applied by code."""
from __future__ import annotations

from tools.run_s6_adaptive_r import evaluate_rules


def _fake_cell(motion, noise, delay, loss, medians, labels, cost=0.0):
    arms = {arm: {"p95_median_m": medians[arm], "label": labels.get(arm, "passes")} for arm in medians}
    return {"condition_id": f"{motion}/{noise}/{delay}/{loss}", "motion": motion, "noise_std_m": noise,
            "delay_s": delay, "loss": loss, "arms": arms, "paired_adaptive_minus_told_5cm": {"median_m": cost}}


def test_rule_verdicts():
    good_h1 = _fake_cell("constant_velocity", 0.10, 0.3, "none",
                         {"told_5cm": 0.30, "told_truth": 0.24, "adaptive": 0.25}, {"told_5cm": "fails"})
    small_gap = _fake_cell("constant_velocity", 0.10, 0.0, "none",
                           {"told_5cm": 0.19, "told_truth": 0.185, "adaptive": 0.30}, {})
    h2_ok = _fake_cell("constant_velocity", 0.05, 0.2, "none",
                       {"told_5cm": 0.12, "told_truth": 0.12, "adaptive": 0.125}, {}, cost=0.005)
    turn_flip = _fake_cell("constant_turn", 0.05, 0.2, "none",
                           {"told_5cm": 0.2, "told_truth": 0.2, "adaptive": 0.26}, {"adaptive": "fails"})
    assert evaluate_rules([good_h1, small_gap, h2_ok])["verdict"] == "worth proposing"
    assert evaluate_rules([good_h1, h2_ok, turn_flip])["verdict"] == "worth proposing only with a turn guard"
    bad_h1 = _fake_cell("constant_velocity", 0.10, 0.3, "none",
                        {"told_5cm": 0.30, "told_truth": 0.24, "adaptive": 0.29}, {"told_5cm": "fails"})
    assert evaluate_rules([bad_h1, h2_ok])["verdict"] == "not worth proposing as is"
    costly = _fake_cell("constant_velocity", 0.05, 0.2, "none",
                        {"told_5cm": 0.12, "told_truth": 0.12, "adaptive": 0.14}, {}, cost=0.02)
    assert evaluate_rules([good_h1, costly])["H2_no_harm"]["pass"] is False
