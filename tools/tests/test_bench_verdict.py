"""Tests for tools/bench_verdict.py (no OpenCV, no camera, no radio).

The rule tests feed hand-built rows to the verdict function, one rule broken at a time. They restate the
written rules (PASS, MARGINAL, FAIL) on purpose, so they prove the code matches the rules, not that the
rules are good. The end-to-end tests run the real chain on the synthetic rehearsal inputs, which come from
generators written separately from every tool in the chain.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools import bench_verdict as tool


def arm(fails: int, n: int = 50) -> dict:
    return {"label": "fails" if 2 * fails >= n else "passes", "fail_count": fails, "n_seeds": n}


def rows(*fails: int, n: int = 50) -> list[dict]:
    motions = ("constant_velocity", "constant_turn")
    return [{"motion": m, "recovery": arm(f, n)} for m, f in zip(motions, fails)]


def grid(read=(0, 0), nxt=(0, 0), *, on_grid=True, noise_level=0.05, last=False) -> dict:
    return {"on_grid": on_grid, "total_delay_s": 0.12, "grid_point_s": 0.15 if on_grid else None,
            "noise_level_m": noise_level, "rows": rows(*read) if on_grid else [],
            "next_rows": None if last else rows(*nxt), "next_grid_point_s": None if last else 0.2}


def combined(*fails: int, n: int = 50) -> dict:
    motions = ("constant_velocity", "constant_turn")
    return {"cells": [{"motion": m, "arms": {"recovery": arm(f, n)}} for m, f in zip(motions, fails)]}


# ---------------------------------------------------------------- verdict rules


def test_pass_when_everything_has_headroom():
    verdict, reasons = tool.decide(grid(), combined(0, 0))
    assert verdict == "PASS", reasons


def test_fail_off_the_delay_grid():
    verdict, reasons = tool.decide(grid(on_grid=False), combined(0, 0))
    assert verdict == "FAIL"
    assert any("off the grid" in r for r in reasons)


def test_fail_noise_above_the_noise_grid():
    verdict, reasons = tool.decide(grid(noise_level=None), combined(0, 0))
    assert verdict == "FAIL"
    assert any("noise" in r for r in reasons)


def test_fail_grid_row_recovery_fails():
    verdict, reasons = tool.decide(grid(read=(0, 30)), combined(0, 0))
    assert verdict == "FAIL"
    assert any("constant_turn" in r for r in reasons)


def test_fail_combined_recovery_fails():
    verdict, _ = tool.decide(grid(), combined(25, 0))
    assert verdict == "FAIL"


def test_marginal_next_row_fails():
    verdict, reasons = tool.decide(grid(nxt=(50, 0)), combined(0, 0))
    assert verdict == "MARGINAL"
    assert any("next" in r for r in reasons)


def test_marginal_when_row_read_is_the_largest_delay():
    verdict, _ = tool.decide(grid(last=True), combined(0, 0))
    assert verdict == "MARGINAL"


def test_marginal_when_seeds_cannot_rule_out_failing():
    # 1 of 5 seeds fail: the cell passes, but the Wilson upper end is above one half.
    verdict, reasons = tool.decide(grid(), combined(1, 0, n=5))
    assert verdict == "MARGINAL"
    assert any("interval" in r for r in reasons)


def test_zero_of_five_is_not_marginal_on_its_own():
    # Hand check: Wilson 95% upper end for 0 of 5 is z^2/(n+z^2) = 3.84/8.84 = 0.43, under one half.
    assert tool.fail_rate_high(arm(0, 5)) == pytest.approx(0.4345, abs=1e-3)
    verdict, _ = tool.decide(grid(), combined(0, 0, n=5))
    assert verdict == "PASS"


def test_fail_beats_marginal():
    verdict, _ = tool.decide(grid(read=(40, 0), nxt=(50, 50)), combined(1, 0, n=5))
    assert verdict == "FAIL"


def test_noise_level_rounds_up():
    levels = (0.0, 0.02, 0.05, 0.10)
    assert tool.noise_level(0.0302, levels) == 0.05
    assert tool.noise_level(0.02, levels) == 0.02
    assert tool.noise_level(0.0, levels) == 0.0
    assert tool.noise_level(0.1001, levels) is None


# ---------------------------------------------------------------- data status


def test_status_synthetic_wins():
    status, _ = tool.data_status({"log a": "measured", "radio fit": "synthetic"}, radio_assumed=False)
    assert status == "synthetic"


def test_status_measured_needs_everything_measured():
    status, _ = tool.data_status({"log a": "measured", "radio fit": "measured"}, radio_assumed=False)
    assert status == "measured"


@pytest.mark.parametrize("statuses,assumed", [
    ({"log a": "measured", "radio fit": "measured"}, True),
    ({"log a": "measured", "latency": "unknown"}, False),
    ({"log a": "measured", "budget": "partly measured"}, False),
    ({"log a": "measured", "latency": None}, False),
])
def test_status_never_measured_when_something_is_not(statuses, assumed):
    status, why = tool.data_status(statuses, radio_assumed=assumed)
    assert status == "not fully measured"
    assert why


# ---------------------------------------------------------------- stopping early


def test_preflight_failure_stops_early(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    out = tmp_path / "verdict"
    assert tool.main([str(bench), "--output", str(out), "--pilot"]) == 1
    text = (out / "VERDICT.md").read_text(encoding="utf-8")
    assert "NO VERDICT" in text
    assert "RESULT: FAIL" in text            # the preflight's own message is copied in
    assert "preflight_bench_data.py" in text  # and the command it ran
    assert not (out / "vision-intake").exists()
    data = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    assert data["verdict"] is None
    assert data["stopped_at"] == "preflight"


def test_output_must_be_new_or_empty(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    out = tmp_path / "verdict"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    assert tool.main([str(bench), "--output", str(out)]) == 2


def test_output_never_under_sim_runs(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    assert tool.main([str(bench), "--output", str(tool.REPO_ROOT / "sim" / "runs" / "verdict-x")]) == 2
    assert not (tool.REPO_ROOT / "sim" / "runs" / "verdict-x").exists()


def test_radio_logs_need_a_status(tmp_path: Path):
    from tools import rehearse_measured_inputs as rehearsal

    inputs = rehearsal.build_inputs(tmp_path)
    out = tmp_path / "verdict"
    assert tool.main([str(inputs["log"].parent), "--output", str(out), "--pilot"]) == 1
    text = (out / "VERDICT.md").read_text(encoding="utf-8")
    assert "NO VERDICT" in text
    assert "--radio-status" in text
    assert not (out / "vision-intake").exists()


# ---------------------------------------------------------------- end to end (slow: real chain, pilot)

FIELDS = ("verdict", "reasons", "data_status", "data_status_reasons", "measured", "total_delay_s", "noise_std_m",
          "grid", "combined", "caveats", "commands", "inputs")


def _check_verdict_files(out: Path) -> tuple[dict, str]:
    data = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    text = (out / "VERDICT.md").read_text(encoding="utf-8")
    for key in FIELDS:
        assert key in data, key
    assert data["verdict"] in ("PASS", "MARGINAL", "FAIL")
    assert data["data_status"] == "synthetic"
    assert "Data status: synthetic" in text
    assert "rehearsal only, not a bench result" in text
    assert "Data status: measured" not in text
    assert "0 loss and 0 jitter" in text      # the lossless-row caveat always prints
    for command in data["commands"]:
        assert command["exit_code"] == 0
        assert command["command"] in text
    for entry in data["inputs"]:
        assert len(entry["sha256"]) == 64
    return data, text


def test_end_to_end_on_rehearsal_inputs_with_radio(tmp_path: Path):
    from tools import rehearse_measured_inputs as rehearsal

    inputs = rehearsal.build_inputs(tmp_path)
    out = tmp_path / "verdict"
    code = tool.main([str(inputs["log"].parent), "--output", str(out), "--pilot", "--radio-status", "synthetic"])
    assert code == 0
    data, text = _check_verdict_files(out)
    assert data["radio_assumed"] is False
    tools_run = [c["tool"] for c in data["commands"]]
    assert tools_run == ["preflight_bench_data.py", "intake_vision.py", "intake_espnow.py", "s6_combined_rerun.py",
                         "latency_budget.py"]
    # The budget row is read at the noise level at or above the fitted noise (3 cm fitted, so 5 cm).
    assert data["grid"]["noise_level_m"] == 0.05
    assert data["noise_std_m"] == pytest.approx(0.03, abs=0.002)


def test_end_to_end_without_radio_uses_the_assumed_term(tmp_path: Path):
    from tools import rehearse_measured_inputs as rehearsal

    inputs = rehearsal.build_inputs(tmp_path)
    shutil.rmtree(inputs["sender"].parent)
    out = tmp_path / "verdict"
    assert tool.main([str(inputs["log"].parent), "--output", str(out), "--pilot"]) == 0
    data, text = _check_verdict_files(out)
    assert data["radio_assumed"] is True
    assert "assumed" in text
    link = json.loads((out / "radio-assumed.link.json").read_text(encoding="utf-8"))
    assert link["delay_s"] == pytest.approx(0.040)
    assert link["status"] != "measured"
    # Same 40 ms in both steps: the combined delay is the vision link delay plus 40 ms.
    vision_link = json.loads(next((out / "vision-intake" / "links").glob("*.link.json")).read_text(encoding="utf-8"))
    assert data["combined"]["delay_s"] == pytest.approx(vision_link["delay_s"] + 0.040, abs=1e-9)
