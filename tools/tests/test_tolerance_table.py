import pytest
import json
from pathlib import Path

from tools.tolerance_table import (
    boundary_for_rows,
    build_table,
    wilson_interval,
)


def test_wilson_interval_matches_hand_computed_value():
    # x=2 failures in n=10, z=1.96: Wilson interval is [0.0567, 0.5098].
    low, high = wilson_interval(2, 10)
    assert low == pytest.approx(0.0567, abs=0.001)
    assert high == pytest.approx(0.5098, abs=0.001)


def test_boundary_finds_largest_passing_axis_value():
    rows = [
        {"axis": 0.0, "failed": False, "p95": 0.10},
        {"axis": 0.0, "failed": False, "p95": 0.12},
        {"axis": 0.1, "failed": False, "p95": 0.20},
        {"axis": 0.1, "failed": True, "p95": 0.30},
        {"axis": 0.2, "failed": True, "p95": 0.30},
        {"axis": 0.2, "failed": True, "p95": 0.30},
    ]
    result = boundary_for_rows(rows, allowed_fail_fraction=0.5, p95_threshold=0.25)
    assert result["boundary"] == 0.0
    assert result["failing_seeds"] == 0
    assert result["seed_count"] == 2


def test_stricter_error_criterion_never_increases_tolerance():
    rows = [
        {"axis": 0.0, "failed": False, "p95": 0.10},
        {"axis": 0.1, "failed": False, "p95": 0.20},
        {"axis": 0.2, "failed": False, "p95": 0.30},
    ]
    loose = boundary_for_rows(rows, allowed_fail_fraction=0.5, p95_threshold=0.35)
    strict = boundary_for_rows(rows, allowed_fail_fraction=0.5, p95_threshold=0.15)
    assert strict["boundary"] <= loose["boundary"]


def test_failure_share_tie_fails_the_cell():
    rows = [
        {"axis": 0.0, "p95_error_m": 0.30, "max_error_m": 0.30},
        {"axis": 0.0, "p95_error_m": 0.10, "max_error_m": 0.10},
    ]
    result = boundary_for_rows(rows, allowed_fail_fraction=0.5, p95_threshold=0.25)
    assert result["boundary"] is None


def test_looser_criterion_recomputes_from_error_columns():
    rows = [{"axis": 0.0, "failed": True, "p95_error_m": 0.30, "max_error_m": 0.30}]
    result = boundary_for_rows(rows, allowed_fail_fraction=0.5, p95_threshold=0.35)
    assert result["boundary"] == 0.0


def test_default_s6_recovery_boundaries_match_published_summary():
    root = Path(__file__).resolve().parents[2]
    report = build_table(root, sources=("s6-recovery",))
    actual = {
        tuple(sorted(row["conditions"].items())): row["boundary"]
        for row in report["rows"]
        if row["cell_threshold"] == 0.5 and row["failure_criterion_m"] == 0.25
    }
    summary = json.loads((root / "sim/runs/s6-recovery/summary.json").read_text())
    by_group = {}
    for cell in summary["cells"]:
        conditions = {k: cell[k] for k in (
            "jitter_s", "loss_model", "loss_probability", "motion", "noise_std_m",
            "slice", "tracker_measurement_std_m")}
        for arm, values in cell["arms"].items():
            key = tuple(sorted({**conditions, "arm": arm}.items()))
            by_group.setdefault(key, []).append(
                cell["delay_s"] if values["label"] == "passes" else None
            )
    expected = {key: max((delay for delay in delays if delay is not None), default=None)
                for key, delays in by_group.items()}
    assert actual == expected
