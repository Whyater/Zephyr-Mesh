"""Small test for the EXP-023 flag-on comparison helper (proposal tooling)."""
import pytest

from tools.burn_divergence_wire_effect import compare
from tools.d_run_fusion import ROW_COLUMNS


def _row(seed, failed, **extra):
    row = {c: "1" for c in ROW_COLUMNS}
    row.update({"arm": "peer_naive", "ci": "fast", "n_observers": "6", "seed": str(seed),
                "failed": failed})
    row.update(extra)
    return row


def test_counts_pass_to_fail_and_spots_other_changes():
    old = [_row(1, "False"), _row(2, "True"), _row(3, "False")]
    new = [_row(1, "True", nees_diverged="True", nees_inconsistent="True"),
           _row(2, "True", nees_diverged="False", nees_inconsistent="True"),
           _row(3, "False", nees_diverged="False", nees_inconsistent="False")]
    out = compare(old, new)
    assert out["totals"]["pass_to_fail"] == 1
    assert out["totals"]["fail_to_pass"] == 0
    assert out["totals"]["nees_diverged"] == 1
    assert out["shared_columns_that_differ"] == []
    new[2]["p95_error_m"] = "2"
    assert compare(old, new)["shared_columns_that_differ"] == ["p95_error_m"]


def test_rejects_different_row_sets():
    with pytest.raises(ValueError):
        compare([_row(1, "False")], [_row(2, "False", nees_diverged="False",
                                            nees_inconsistent="False")])
