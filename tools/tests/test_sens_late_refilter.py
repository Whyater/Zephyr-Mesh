"""Checks for the late-packet side run (tools/sens_late_refilter.py).

The registered rows come from sim/runs/s6-recovery and sim/runs/s6-retro (recorded
on macOS, so compared with the shared platform tolerance elsewhere), which
were produced by other code. The engine comparison uses the estimation engine's
own recovery and retrodict arms, a separately written implementation. The
late-fraction check compares the simulated link with a formula written from the
lognormal delay model.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from sim.estimation.base import EstimatorSettings, RECOVERY_V1
from sim.estimation.engine import deliver_cell, run_arm
from sim.s6_recovery import RecoveryConfig, seed_list
from sim.tests.platform_compare import same_tree
from tools.sens_late_refilter import (ARMS, ENGINE_RETRO, build_grid, hand_late_fraction, late_census,
                                      make_cell, run_late_arm)

RUNS = Path(__file__).resolve().parents[2] / "sim" / "runs"
CONFIG = RecoveryConfig()
SETTINGS = EstimatorSettings()


def _registered_s6_recovery(cell_id: str, arm: str) -> dict[int, float]:
    with (RUNS / "s6-recovery" / "per_seed.csv").open(encoding="utf-8") as handle:
        return {int(r["seed_index"]): float(r["p95_error_m"]) for r in csv.DictReader(handle)
                if r["cell_id"] == cell_id and r["arm"] == arm}


def _registered_s6_retro(cell_key: str, arm: str) -> dict[int, float]:
    with gzip.open(RUNS / "s6-retro" / "per_seed.csv.gz", "rt", encoding="utf-8") as handle:
        return {int(r["seed_index"]): float(r["p95_error_m"]) for r in csv.DictReader(handle)
                if r["cell_key"] == cell_key and r["arm"] == arm}


def test_grid_has_the_preregistered_32_points():
    grid = build_grid()
    assert len(grid) == 32
    assert {g["jitter_ratio"] for g in grid} == {0.0, 1.0, 2.0, 3.0}
    assert make_cell("ct", 0.3, 3.0, 0.3).jitter_s == pytest.approx(0.9)


def test_zero_jitter_arms_identical_and_drop_reproduces_registered_s6_row():
    registered = _registered_s6_recovery(
        "main/constant_velocity/d0.300/l0.30/n0.050/independent/j0.000/r0.050", "recovery")
    cell = make_cell("cv", 0.3, 0.0, 0.3)
    for index, seed in enumerate(seed_list(2)):
        bundle = deliver_cell(cell, seed, CONFIG, SETTINGS)
        results = {arm: run_late_arm(bundle, cell, arm, CONFIG, SETTINGS) for arm in ARMS}
        assert results["drop"] == results["apply_late"] == {**results["refilter"], "max_history": 0}
        assert results["drop"]["dropped_count"] == 0
        assert same_tree(results["drop"]["p95_error_m"], registered[index])  # macOS-recorded row


def test_jitter_equal_to_delay_reproduces_registered_retro_rows():
    cell = make_cell("cv", 0.3, 1.0, 0.0)
    recovery = _registered_s6_retro(cell.key(), "recovery")
    retro = _registered_s6_retro(cell.key(), "retro")
    assert recovery and retro
    for index, seed in enumerate(seed_list(2)):
        bundle = deliver_cell(cell, seed, CONFIG, SETTINGS)
        assert same_tree(run_late_arm(bundle, cell, "drop", CONFIG, SETTINGS)["p95_error_m"], recovery[index])
        assert same_tree(run_late_arm(bundle, cell, "refilter", CONFIG, SETTINGS)["p95_error_m"], retro[index])


def test_wrapper_arms_match_the_engine_at_three_times_jitter():
    cell = make_cell("ct", 0.3, 3.0, 0.3)
    bundle = deliver_cell(cell, seed_list(1)[0], CONFIG, SETTINGS)
    late, _ = late_census(bundle)
    assert late > 10  # the cell really reorders
    for arm, spec in (("drop", RECOVERY_V1), ("refilter", ENGINE_RETRO)):
        ours = run_late_arm(bundle, cell, arm, CONFIG, SETTINGS)
        engine = run_arm(bundle, cell, spec, CONFIG, SETTINGS)
        assert ours["p95_error_m"] == engine.p95_error_m
        assert ours["max_error_m"] == engine.max_error_m
        assert ours["fused_count"] == engine.fused_count
        assert ours["dropped_count"] == engine.dropped_count
    applied = run_late_arm(bundle, cell, "apply_late", CONFIG, SETTINGS)
    assert applied["late_applied_count"] == late  # every late packet is fused, none dropped
    assert applied["dropped_count"] == 0


def test_hand_formula_matches_the_registered_census():
    # Registered s6-retro census, 50 seeds: 0.3748 at 300 ms delay, 300 ms jitter, no loss.
    summary = json.loads((RUNS / "s6-retro" / "summary.json").read_text(encoding="utf-8"))
    key = make_cell("cv", 0.3, 1.0, 0.0).key()
    registered = next(c["reorder_census"]["fraction"] for c in summary["cells"] if c["key"] == key)
    assert hand_late_fraction(0.3, 0.3, 0.0) == pytest.approx(registered, abs=0.015)


@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_hand_formula_matches_simulated_late_fraction_at_heavy_jitter(ratio):
    cell = make_cell("cv", 0.3, ratio, 0.0)
    late = delivered = 0
    for seed in seed_list(30):
        n_late, n_delivered = late_census(deliver_cell(cell, seed, CONFIG, SETTINGS))
        late += n_late
        delivered += n_delivered
    assert late / delivered == pytest.approx(hand_late_fraction(0.3, 0.3 * ratio, 0.0), abs=0.03)
