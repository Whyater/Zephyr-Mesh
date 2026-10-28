"""Tests for the EXP-040 process-noise (q) sweep (proposal, simulated)."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from sim.q_sweep import Q_GRID, REGISTERED_Q, simulate_q_condition
from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list
from tools.run_s6_q_sweep import best_q, evaluate_rules, schedule_comparison

CONFIG = RecoveryConfig()
SEED = seed_list(1)[0]


def test_grid_is_half_decades_and_contains_registered_q_exactly():
    assert len(Q_GRID) == 7
    assert REGISTERED_Q == 0.1 and REGISTERED_Q in Q_GRID
    assert Q_GRID[0] == 0.01 and Q_GRID[-1] == 10.0
    steps = np.diff(np.log10(Q_GRID))
    assert np.allclose(steps, 0.5, atol=1e-12)


@pytest.mark.parametrize("motion,delay,loss,model", [
    ("constant_velocity", 0.2, 0.0, "independent"),
    ("constant_turn", 0.3, 0.0, "independent"),
    ("constant_velocity", 0.0, 0.3, "burst"),
])
def test_registered_q_matches_the_registered_recovery_arm_bit_for_bit(motion, delay, loss, model):
    out = simulate_q_condition(motion, delay_s=delay, loss_probability=loss, noise_std_m=0.05, loss_model=model,
                               seed=SEED, config=CONFIG, q_values=(REGISTERED_Q,), told_stds={"told_5cm": 0.05})
    deliveries = deliver(motion, delay_s=delay, loss_probability=loss, noise_std_m=0.05, seed=SEED, config=CONFIG,
                         loss_model=model)
    reference = run_arms(deliveries, motion, CONFIG)["recovery"]
    got = out[(REGISTERED_Q, "told_5cm")]
    assert repr(got.p95_error_m) == repr(reference.p95_error_m)
    assert repr(got.max_error_m) == repr(reference.max_error_m)
    assert np.array_equal(got.errors_m, reference.errors_m)


def test_q_reaches_the_tracker_and_changes_the_answer():
    out = simulate_q_condition("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.05,
                               loss_model="independent", seed=SEED, config=CONFIG, q_values=(0.01, 0.1, 10.0),
                               told_stds={"told_5cm": 0.05})
    p = [out[(q, "told_5cm")].p95_error_m for q in (0.01, 0.1, 10.0)]
    assert len(set(p)) == 3
    # Same q through the plain recovery path with a replaced config gives the same answer.
    deliveries = deliver("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.05, seed=SEED,
                         config=CONFIG)
    ref = run_arms(deliveries, "constant_velocity", replace(CONFIG, tracker_acceleration_psd=0.01))["recovery"]
    assert repr(out[(0.01, "told_5cm")].p95_error_m) == repr(ref.p95_error_m)


def test_straight_line_without_delay_prefers_less_motion_noise_than_ten():
    # Limiting case: the target never accelerates, so q = 10 (velocity may wander about 3 m/s per
    # square-root second) must chase noise far more than q = 0.01.
    out = simulate_q_condition("constant_velocity", delay_s=0.3, loss_probability=0.0, noise_std_m=0.10,
                               loss_model="independent", seed=SEED, config=CONFIG, q_values=(0.01, 10.0),
                               told_stds={"told_5cm": 0.05})
    assert out[(0.01, "told_5cm")].p95_error_m < 0.5 * out[(10.0, "told_5cm")].p95_error_m


# ---- rule logic on hand-made summaries -------------------------------------------------------------

def _cell(motion, noise, delay, loss, medians, labels=None, truth_medians=None, truth_labels=None):
    labels = labels or {q: "passes" for q in medians}
    truth_medians = truth_medians or medians
    truth_labels = truth_labels or labels
    arms = {}
    for q in medians:
        arms[repr(q)] = {
            "told_5cm": {"p95_median_m": medians[q], "label": labels[q]},
            "told_truth": {"p95_median_m": truth_medians[q], "label": truth_labels[q]},
        }
    return {"motion": motion, "noise_std_m": noise, "delay_s": delay, "loss": loss,
            "condition_id": f"{motion}/n{noise:.2f}/d{delay:.1f}/{loss}", "by_q": arms}


def _grid(straight_best, turn_best, cost_per_decade=0.03):
    """Synthetic 36-cell grid where straight cells are best at straight_best and turn cells at turn_best."""
    cells = []
    for motion, best in (("constant_velocity", straight_best), ("constant_turn", turn_best)):
        for noise in (0.02, 0.05, 0.10):
            for delay in (0.0, 0.2, 0.3):
                for loss in ("none", "burst30"):
                    med = {q: 0.10 + cost_per_decade * abs(math.log10(q) - math.log10(best)) for q in Q_GRID}
                    cells.append(_cell(motion, noise, delay, loss, med))
    return cells


def test_best_q_picks_the_smallest_median():
    cell = _cell("constant_velocity", 0.05, 0.3, "none", {q: 1.0 + abs(math.log10(q) + 1.5) for q in Q_GRID})
    assert best_q(cell, "told_5cm") == Q_GRID[1]


def test_h2_passes_when_one_q_is_best_everywhere_and_fails_when_motions_disagree():
    same = evaluate_rules(_grid(0.1, 0.1))
    assert same["H2_one_fixed_q"]["pass"] is True
    assert 0.1 in same["H2_one_fixed_q"]["qualifying_q"]
    apart = evaluate_rules(_grid(0.01, 10.0))
    # Three decades apart at 3 cm per decade: any single q costs at least 4.5 cm somewhere.
    assert apart["H2_one_fixed_q"]["pass"] is False


def test_h3_schedule_beats_a_single_q_only_when_the_motions_disagree():
    assert schedule_comparison(_grid(0.1, 0.1))["pass"] is False
    result = schedule_comparison(_grid(0.01, 10.0))
    assert result["pass"] is True
    assert result["q_straight"] == 0.01 and result["q_turn"] == 10.0


def test_h1_needs_both_noises_and_the_ten_cm_pass():
    cells = _grid(0.01, 0.1)
    for c in cells:
        if c["motion"] == "constant_velocity" and c["loss"] == "none" and c["delay_s"] == 0.3:
            for q in Q_GRID:
                c["by_q"][repr(q)]["told_truth"]["p95_median_m"] = 0.12  # told truth at q = 0.1 is 0.12
    assert evaluate_rules(cells)["H1_q_is_the_knob"]["pass"] is True  # best told-5cm (0.10) <= 0.12, and it passes
    for c in cells:
        if (c["motion"], c["loss"], c["delay_s"], c["noise_std_m"]) == ("constant_velocity", "none", 0.3, 0.10):
            for q in Q_GRID:
                c["by_q"][repr(q)]["told_5cm"]["label"] = "fails"
    assert evaluate_rules(cells)["H1_q_is_the_knob"]["pass"] is False


def test_h2_counts_a_flip_against_the_registered_q():
    cells = _grid(0.1, 0.1)
    # Make q = 0.0316 the best everywhere by a hair, but have it fail one cell that passes at q = 0.1.
    for c in cells:
        c["by_q"][repr(Q_GRID[1])]["told_5cm"]["p95_median_m"] = 0.0999
    victim = cells[0]
    victim["by_q"][repr(Q_GRID[1])]["told_5cm"]["label"] = "fails"
    result = evaluate_rules(cells)["H2_one_fixed_q"]
    assert Q_GRID[1] not in result["qualifying_q"]
    assert 0.1 in result["qualifying_q"]


# ---- reproduction gate R0 on synthetic stored files ------------------------------------------------

def _fake_results(n_seeds=2):
    from tools.run_s6_q_sweep import TOLD, build_conditions
    conditions = build_conditions()
    results = []
    for ci, c in enumerate(conditions):
        rows = {}
        for q in Q_GRID:
            for ti, told in enumerate(TOLD):
                alias = 0 if c["noise_std_m"] == 0.05 else ti  # at 5 cm both names are one run
                rows[f"{q!r}|{told}"] = [{"p95_error_m": 0.1 + ci * 1e-3 + alias * 1e-5 + s * 1e-7,
                                          "max_error_m": 0.2 + ci * 1e-3 + alias * 1e-5 + s * 1e-7}
                                         for s in range(n_seeds)]
        results.append(rows)
    return conditions, results


def _write_stored(tmp_path, conditions, results, drop=None):
    """Write stored CSVs the way the two real files are laid out (only the columns the gate reads)."""
    import csv as _csv
    s6_path, exp_path = tmp_path / "s6.csv", tmp_path / "exp037.csv"
    s6_fields = ["slice", "motion", "delay_s", "loss_probability", "noise_std_m", "loss_model", "jitter_s",
                 "tracker_measurement_std_m", "arm", "seed_index", "p95_error_m", "max_error_m"]
    with s6_path.open("w", newline="") as h6, exp_path.open("w", newline="") as he:
        w6 = _csv.DictWriter(h6, fieldnames=s6_fields)
        we = _csv.DictWriter(he, fieldnames=["condition_id", "arm", "seed_index", "p95_error_m", "max_error_m"])
        w6.writeheader()
        we.writeheader()
        for c, rows in zip(conditions, results):
            stored_cells = []
            if c["loss"] == "none":
                stored_cells.append(("main", "told_5cm", 0.05))
                if c["noise_std_m"] == 0.05:
                    stored_cells.append(("main", "told_truth", 0.05))
                if c["noise_std_m"] == 0.10:
                    stored_cells.append(("oracle", "told_truth", 0.10))
            elif c["noise_std_m"] == 0.05:
                stored_cells += [("burst", "told_5cm", 0.05), ("burst", "told_truth", 0.05)]
            for slice_name, told, std in stored_cells:
                if drop == ("s6", c["condition_id"], told):
                    continue
                for s, r in enumerate(rows[f"{REGISTERED_Q!r}|{told}"]):
                    w6.writerow({"slice": slice_name, "motion": c["motion"], "delay_s": c["delay_s"],
                                 "loss_probability": c["loss_probability"], "noise_std_m": c["noise_std_m"],
                                 "loss_model": c["loss_model"], "jitter_s": 0.0, "tracker_measurement_std_m": std,
                                 "arm": "recovery", "seed_index": s, "p95_error_m": repr(r["p95_error_m"]),
                                 "max_error_m": repr(r["max_error_m"])})
            for told in ("told_5cm", "told_truth"):
                if drop == ("exp037", c["condition_id"], told):
                    continue
                for s, r in enumerate(rows[f"{REGISTERED_Q!r}|{told}"]):
                    we.writerow({"condition_id": c["condition_id"], "arm": told, "seed_index": s,
                                 "p95_error_m": repr(r["p95_error_m"]), "max_error_m": repr(r["max_error_m"])})
    return s6_path, exp_path


def test_r0_passes_with_full_coverage_and_counts_every_expected_cell(tmp_path):
    from tools.run_s6_q_sweep import reproduction_check
    conditions, results = _fake_results()
    s6, exp = _write_stored(tmp_path, conditions, results)
    r0 = reproduction_check(conditions, results, s6_path=s6, exp037_path=exp)
    assert r0["pass"] is True
    # 18 no-loss told-5cm + 6 no-loss told-truth at 5 cm + 6 oracle + 6 + 6 burst at 5 cm = 42; 36 x 2 = 72.
    assert r0["cells_compared"] == {"s6_registered": 42, "exp037": 72}
    assert r0["seeds_compared"] == {"s6_registered": 84, "exp037": 144}


def test_r0_catches_a_changed_value(tmp_path):
    from tools.run_s6_q_sweep import reproduction_check
    conditions, results = _fake_results()
    s6, exp = _write_stored(tmp_path, conditions, results)
    results[5][f"{REGISTERED_Q!r}|told_5cm"][1]["max_error_m"] += 1e-15
    r0 = reproduction_check(conditions, results, s6_path=s6, exp037_path=exp)
    assert r0["pass"] is False and r0["mismatch_count"] >= 1


@pytest.mark.parametrize("drop", [("s6", "constant_turn/n0.10/d0.3/none", "told_truth"),
                                  ("exp037", "constant_velocity/n0.02/d0.2/burst30", "told_5cm")])
def test_r0_fails_when_a_stored_cell_is_missing(tmp_path, drop):
    from tools.run_s6_q_sweep import reproduction_check
    conditions, results = _fake_results()
    s6, exp = _write_stored(tmp_path, conditions, results, drop=drop)
    r0 = reproduction_check(conditions, results, s6_path=s6, exp037_path=exp)
    assert r0["pass"] is False and r0["missing_count"] == 1


def test_r0_fails_when_a_stored_file_is_missing_or_short_of_seeds(tmp_path):
    from tools.run_s6_q_sweep import reproduction_check
    conditions, results = _fake_results()
    s6, exp = _write_stored(tmp_path, conditions, results)
    assert reproduction_check(conditions, results, s6_path=tmp_path / "nope.csv", exp037_path=exp)["pass"] is False
    more_seeds_than_stored = _fake_results(n_seeds=3)[1]
    assert reproduction_check(conditions, more_seeds_than_stored, s6_path=s6, exp037_path=exp)["pass"] is False


def test_told_truth_at_five_cm_is_the_same_run_as_told_5cm():
    from tools.run_s6_q_sweep import _run_condition, _told_stds, build_conditions
    assert _told_stds(0.05) == {"told_5cm": 0.05}
    assert _told_stds(0.10) == {"told_5cm": 0.05, "told_truth": 0.10}
    condition = next(c for c in build_conditions() if c["condition_id"] == "constant_velocity/n0.05/d0.0/none")
    rows = _run_condition(condition, [SEED], CONFIG)
    for q in (0.01, 10.0):
        assert rows[f"{q!r}|told_5cm"] == rows[f"{q!r}|told_truth"]


def test_h2_regret_ignores_loss_cells():
    cells = _grid(0.1, 0.1)
    for c in cells:
        if c["loss"] == "burst30":
            c["by_q"][repr(0.1)]["told_5cm"]["p95_median_m"] = 5.0  # terrible, but only in loss cells
    assert 0.1 in evaluate_rules(cells)["H2_one_fixed_q"]["qualifying_q"]
