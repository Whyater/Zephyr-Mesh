"""Tests for the EXP-024 grid extension side run."""
from __future__ import annotations

import pytest

from tools import d_run_tolerance as registered
from tools import sens_d3_grid_extension as ext


@pytest.fixture(scope="module")
def reference():
    return ext.load_registered()


def test_extension_grid_is_the_n13_spec():
    cells = ext.extension_cells()
    assert len(cells) == 7 * 5
    assert {c for c in cells if c[1] == 0.0 and c[0] > 0} == {
        (d, 0.0, n) for d in (0.4, 0.5, 0.75, 1.0) for n in (1, 2, 3, 4, 6)}
    assert {c for c in cells if c[0] == 0.0} == {
        (0.0, p, n) for p in (0.7, 0.8, 0.9) for n in (1, 2, 3, 4, 6)}
    assert len(ext.jobs_for(cells, ext.seeds())) == 700


def test_new_cells_use_the_registered_worker_config():
    cell = registered._cell(ext.make_job(1.0, 0.0, 6, 1))
    assert cell.link.delay_s == 1.0 and cell.link.mode == "abstract"
    assert cell.rate_hz == 10.0 and cell.noise_std_m == 0.05 and cell.target == "ct"
    cell = registered._cell(ext.make_job(0.0, 0.9, 3, 1))
    assert cell.link.loss_probability == 0.9 and cell.link.loss_model == "independent"


def test_seeds_and_reference_cover_the_old_slice(reference):
    assert ext.seeds() == registered.seeds_for("tolerance", 20)
    # 7 delays x 5 loss conditions x 20 seeds x 13 arm-and-N combinations.
    assert len(reference) == 7 * 5 * 20 * 13
    assert {key[-1] for key in reference} == set(ext.seeds())


def test_one_registered_row_reproduces_as_text(reference):
    # Cheapest old-grid cell (N = 1), first seed: the rerun must equal the stored CSV line.
    result = ext.control(processes=1, cells=((0.3, 0.0, 1),), seed_list=ext.seeds()[:1],
                         reference=reference)
    assert result["rows"] == 1
    assert result["passed"], result["mismatches"]


def test_control_fails_when_a_registered_value_is_changed(reference):
    # Negative control: perturb the stored p95 of the row being rerun; the check must fail.
    # The nudge is relative 1e-6, far outside the 1e-9 tolerance used off macOS, so it fails on every OS.
    seed = ext.seeds()[0]
    key = ext.row_key("abstract", "ct_r1", "independent", 0.0, "single", 1, 0.3, seed)
    line, values = reference[key]
    p95 = values["p95"]
    altered = dict(reference)
    altered[key] = (line.replace(p95, repr(float(p95) * (1 + 1e-6) + 1e-9), 1), values)
    result = ext.control(processes=1, cells=((0.3, 0.0, 1),), seed_list=[seed], reference=altered)
    assert not result["passed"] and len(result["mismatches"]) == 1
    missing = {k: v for k, v in reference.items() if k != key}
    assert not ext.control(processes=1, cells=((0.3, 0.0, 1),), seed_list=[seed],
                           reference=missing)["passed"]


def test_fail_counts_keep_burst_rows_off_both_axes(reference):
    rows = [values for _, values in reference.values()]
    for axis in ("delay", "loss"):
        counts = ext.fail_counts(rows, axis)
        assert all(total == 20 for _, total in counts.values())
    # burst 0.3 at delay 0 must not be counted as independent 0.3 on the loss axis
    loss = ext.fail_counts(rows, "loss")
    assert loss[("single", 1, 0.3)] == (1, 20)
    assert {v for _, _, v in loss} == set(ext.OLD_LOSSES)


def test_row_text_matches_the_registered_writer(reference):
    line, values = next(iter(reference.values()))
    row = dict(values)
    for name in ("loss", "p95", "max", "rmse", "mean_nees", "delivery_ratio"):
        row[name] = float(row[name]) if row[name] != "" else None
    row["busy_fraction"] = None
    row["n"], row["seed"], row["delay"] = int(row["n"]), int(row["seed"]), float(row["delay"])
    row["failed"] = row["failed"] == "true"
    assert ext.row_text(row) == line


def test_edges_use_the_registered_rule_on_a_merged_axis():
    axis = (0.0, 0.1, 0.3)
    counts = {}
    for n in ext.NS:
        for arm in ext.arms_for(n):
            for value, failed in zip(axis, (0, 3, 15)):
                counts[(arm, n, value)] = (failed, 20)
    out = ext.edges(counts, axis)
    assert out["single|1"]["value"] == 0.1
    assert out["central|6"]["fail_counts"] == [0, 3, 15]
    counts[("single", 1, 0.0)] = (0, 19)  # a missing seed must be refused
    with pytest.raises(ValueError, match="19 seeds"):
        ext.edges(counts, axis)


def test_all_lost_fraction_at_loss_0_9_is_near_p_to_the_n():
    # Independent check of the link at a new loss value: with 6 observers at loss 0.9
    # the station gets nothing in a step with probability 0.9**6 = 0.531.
    rows, _ = ext.run_rows(ext.jobs_for([(0.0, 0.9, 6)], ext.seeds()[:2]), processes=1)
    summary = ext.gap_summary(rows)["0.9|6"]
    assert summary["hand_all_lost_fraction"] == pytest.approx(0.531441)
    # About 240 capture steps: one standard error is about 0.03.
    assert abs(summary["measured_all_lost_fraction"] - 0.531441) < 0.15


def test_output_under_sim_runs_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sim/runs"):
        ext.run(ext.ROOT / "sim" / "runs" / "d3-ext", processes=1, checkpoint=None)
    with pytest.raises(ValueError, match="checkpoint"):
        ext.run(tmp_path / "out", processes=1, checkpoint=tmp_path / "out" / "ck")
