"""The EXP-042 verdict must never pass when it compared nothing."""
from __future__ import annotations

import pytest

from tools import hold_exp042_compare as compare


def _tag_cell(landings_default=4, landings_hold=0):
    def arm(landings, tagged, violations, min_sep):
        return {"landings": landings, "tagged": tagged, "separation_violation_steps": violations,
                "min_separation_m": min_sep}
    return {"part": "tag", "target_class": "turn", "preset": "burst_10_0.5s",
            "default": arm(landings_default, 3, 2, 0.60), "hold": arm(landings_hold, 3, 1, 0.62)}


def _sweep_cell(landings_default=5, landings_hold=0):
    def arm(landings, completed):
        return {"landings": landings, "completed": completed}
    return {"part": "sweep", "preset": "burst_10_0.5s", "dropout": "none",
            "default": arm(landings_default, 2), "hold": arm(landings_hold, 2)}


def test_verdict_with_no_cells_raises_instead_of_passing():
    with pytest.raises(ValueError, match="no complete tag cells"):
        compare.verdict([])


def test_summarize_with_no_rows_raises():
    with pytest.raises(ValueError, match="no complete"):
        compare.summarize([])


def test_verdict_with_only_one_arm_in_every_cell_raises():
    half = _tag_cell()
    half["hold"] = None
    with pytest.raises(ValueError, match="no complete tag cells"):
        compare.verdict([half, _sweep_cell()])


def test_verdict_without_sweep_cells_raises():
    # Rule 2 would compare 0 completed sweeps with 0 and pass.
    with pytest.raises(ValueError, match="no complete sweep cells"):
        compare.verdict([_tag_cell()])


def test_verdict_without_tag_cells_raises():
    # Rule 3 (safety) only reads tag cells, so it would pass on nothing.
    with pytest.raises(ValueError, match="no complete tag cells"):
        compare.verdict([_sweep_cell()])


def test_verdict_still_scores_real_cells():
    # Hand count: landings 4 + 5 = 9 default, 0 hold; tagged 3 vs 3; sweeps 2 vs 2;
    # violation steps 1 <= 2 and 0.62 m >= 0.60 - 0.05 m, so every rule passes.
    result = compare.verdict([_tag_cell(), _sweep_cell()])
    assert result["rule_1_landings"] == {"pass": True, "default": 9, "hold": 0}
    assert result["rule_2_success"]["pass"] is True
    assert result["rule_3_safety"]["pass"] is True
    assert result["pass"] is True
    # Hold lands 1 of 9 times: more than 10 percent of the default, so rule 1 fails.
    assert compare.verdict([_tag_cell(landings_hold=1), _sweep_cell()])["pass"] is False


def test_main_exits_nonzero_with_a_message_when_no_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "build_jobs", lambda *, pilot=False: [])
    monkeypatch.setattr(compare, "run_jobs", lambda jobs, n_processes=1: [])
    with pytest.raises(SystemExit) as exc:
        compare.main(["--output", str(tmp_path / "out"), "--pilot"])
    assert exc.value.code not in (0, None)
    assert "no complete" in str(exc.value.code)
    assert not (tmp_path / "out" / "summary.json").exists()
