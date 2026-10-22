"""Checks for the EXP-035 mission log integrity script (simulated)."""
from __future__ import annotations

from pathlib import Path

from tools.e_exp035_logs import EXAMPLES_DIR, canonical_cases, check_case


def test_one_case_per_example_and_adapter():
    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    cases = canonical_cases(seed=7)
    # every example runs on two adapters of its kind
    assert len(cases) == 2 * len(examples)
    assert {c["scenario"].mission for c in cases} >= {"hover", "tag", "sweep", "relay", "monitor", "formation"}
    formation = [c for c in cases if c["scenario"].mission == "formation"]
    assert {c["scenario"].adapter for c in formation} == {"point_mass_offboard", "physics_offboard"}
    assert all(c["scenario"].seed == 7 for c in cases)


def test_hover_case_regenerates_validates_and_replays(tmp_path: Path):
    case = next(c for c in canonical_cases(seed=7) if c["name"] == "hover_point_mass")
    summary, rows = check_case(case, tmp_path, "test-revision")
    assert summary["regenerates_byte_identical"] is True
    assert summary["schema_valid"] is True
    assert summary["exact_metrics"] >= 3
    assert summary["exact_disagreements"] == []
    assert (tmp_path / "hover_point_mass.json").exists()
    assert {r["metric"] for r in rows} >= {"final_error_m", "end_reason", "duration_s"}
