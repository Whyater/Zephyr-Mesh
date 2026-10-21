"""Tests for the EXP-023 flag-on comparison helper (proposal tooling).

Rows go through the runner's real CSV writer, so the test uses the same text
the registered per_seed.csv uses (booleans are written as true/false).
"""
import pytest

from sim.estimation.sweep import write_csv
from tools.burn_divergence_wire_effect import _rows, compare
from tools.d_run_fusion import NEES_COLUMNS, ROW_COLUMNS


def _row(seed, failed, **extra):
    row = {c: 1.0 for c in ROW_COLUMNS}
    row.update({"arm": "peer_naive", "ci": "fast", "n_observers": 6, "seed": seed,
                "target": "turn", "failed": failed, "diverged": False})
    row.update(extra)
    return row


def _through_csv(tmp_path, name, rows, columns):
    path = tmp_path / name
    write_csv(path, rows, columns)
    return _rows(path)


def test_counts_pass_to_fail_and_spots_other_changes(tmp_path):
    old = _through_csv(tmp_path, "old.csv", [_row(1, False), _row(2, True), _row(3, False)],
                       ROW_COLUMNS)
    new_rows = [_row(1, True, nees_diverged=True, nees_inconsistent=True),
                _row(2, True, nees_diverged=False, nees_inconsistent=True),
                _row(3, False, nees_diverged=False, nees_inconsistent=False)]
    new = _through_csv(tmp_path, "new.csv", new_rows, ROW_COLUMNS + NEES_COLUMNS)
    assert old[0]["failed"] == "false"  # the writer's text, not Python's repr
    out = compare(old, new)
    assert out["totals"]["old_failed"] == 1
    assert out["totals"]["new_failed"] == 2
    assert out["totals"]["pass_to_fail"] == 1
    assert out["totals"]["fail_to_pass"] == 0
    assert out["totals"]["nees_diverged"] == 1
    assert out["totals"]["nees_inconsistent_warning"] == 2
    assert out["shared_columns_that_differ"] == []
    new[2]["p95_error_m"] = "2.0"
    assert compare(old, new)["shared_columns_that_differ"] == ["p95_error_m"]


def test_rejects_different_row_sets_and_unknown_booleans(tmp_path):
    old = _through_csv(tmp_path, "a.csv", [_row(1, False)], ROW_COLUMNS)
    new = _through_csv(tmp_path, "b.csv", [_row(2, False, nees_diverged=False,
                                                 nees_inconsistent=False)],
                       ROW_COLUMNS + NEES_COLUMNS)
    with pytest.raises(ValueError):
        compare(old, new)
    bad = [dict(old[0], failed="True")]
    with pytest.raises(ValueError):
        compare(bad, [dict(new[0], seed=old[0]["seed"])])
