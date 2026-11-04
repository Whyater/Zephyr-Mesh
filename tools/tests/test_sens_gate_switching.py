"""Tests for the switching-target gate study (tools/sens_gate_switching.py)."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from sim.estimation.engine import s6_cell
from sim.s6_recovery import seed_list
from tools import sens_gate_switching as sw
from tools import sens_outlier_gate as parent
from tools import sens_outlier_gate_turn as turn

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("p95_error_m", "max_error_m", "rmse_m", "failed", "fused_count", "gated_count",
        "gated_outliers", "gated_clean", "forced_accepts")


def test_circle_cell_is_the_s6_circle_cell():
    assert sw.switching_cell("ct", 0.2) == s6_cell("constant_turn", 0.2, 0.0, parent.NOISE_STD_M)


def test_cell_path_equals_the_turn_study_on_the_circle():
    """Control: on the circle this wrapper's cell path gives the turn study's rows bit for bit, all six arms."""
    spec = parent.OutlierSpec("jump", 0.03)
    seeds = [11, 12]
    mine = sw.cell_runs("ct", 0.2, spec, seeds)
    theirs = turn.cell_runs("constant_turn", 0.2, spec, seeds)
    assert [r["arm"] for r in mine] == [r["arm"] for r in theirs]
    for a, b in zip(mine, theirs):
        for key in KEYS:
            assert a[key] == b[key], (a["arm"], a["seed"], key)


@pytest.mark.parametrize("target", sw.TARGETS)
def test_cells_are_the_atlas_cells(target):
    assert sw.switching_cell(target, 0.2).key() == f"{target}/rate10/d0.2/j0/l0-independent/gaussian0.05"


@pytest.mark.parametrize("target", sw.TARGETS)
def test_ungated_rows_equal_the_registered_atlas_rows(target):
    """Independent of this code: the S6 atlas registered these cells. At 0 % outliers the
    ungated arms must give its per-seed p95, max and pass/fail exactly (first 2 seeds, 0.2 s)."""
    key = f"{target}/rate10/d0.2/j0/l0-independent/gaussian0.05"
    want = {}
    with gzip.open(ROOT / "sim" / "runs" / "s6-atlas" / "per_seed.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            if row["cell_key"] == key and row["arm"] in ("recovery", "ct", "imm") and int(row["seed_index"]) < 2:
                want[(row["arm"], int(row["seed"]))] = row
    arms = tuple(a for a in sw.ARMS if a.gate_probability is None)
    rows = sw.cell_runs(target, 0.2, parent.OutlierSpec("jump", 0.0), seed_list(2), arms=arms)
    assert len(want) == len(rows) == 6
    for r in rows:
        w = want[(r["arm"], r["seed"])]
        assert r["p95_error_m"] == pytest.approx(float(w["p95_error_m"]), rel=1e-12, abs=1e-15)
        assert r["max_error_m"] == pytest.approx(float(w["max_error_m"]), rel=1e-12, abs=1e-15)
        assert r["failed"] == (w["failed"] == "true")


def test_paired_gate_cost_hand_values():
    """Hand-built rows: CV gated fails seeds 2 and 3, ungated fails 3 and 4."""
    rows = []
    for seed, (fu, fg, pu, pg) in enumerate([(False, False, 0.10, 0.11), (False, False, 0.20, 0.20),
                                             (False, True, 0.24, 0.26), (True, True, 0.30, 0.31),
                                             (True, False, 0.26, 0.24)]):
        rows.append({"arm": "recovery", "seed": seed, "failed": fu, "p95_error_m": pu})
        rows.append({"arm": "recovery_gated", "seed": seed, "failed": fg, "p95_error_m": pg})
    (cost,) = sw.paired_gate_cost(rows)
    assert cost == {"model": "CV", "ungated_fails": 2, "gated_fails": 2, "only_gated_fails": 1,
                    "only_ungated_fails": 1, "median_p95_diff_m": pytest.approx(0.01)}


def test_plan_is_the_preregistered_grid():
    jobs = sw.plan()
    assert len(jobs) == 2 * 3 * 3
    assert {t for t, _, _ in jobs} == {"figure8", "start_stop"}
    assert sorted({d for _, d, _ in jobs}) == [0.0, 0.1, 0.2]
    assert sorted({s.probability for _, _, s in jobs}) == [0.0, 0.01, 0.03]
    assert {s.kind for _, _, s in jobs} == {"jump"}


def test_refuses_to_write_under_sim_runs():
    with pytest.raises(SystemExit, match="refusing"):
        sw.main(["--output", str(ROOT / "sim" / "runs" / "nope"), "--seeds", "1"])
