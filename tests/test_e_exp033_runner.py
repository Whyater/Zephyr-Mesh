"""Checks for the EXP-033 simulated monitoring runner."""
from __future__ import annotations

import json
from pathlib import Path

from tools.e_exp033_monitor import PREDICTION, PREDICTION_SOURCE, build_jobs, run_experiment
from tests.reference_e.hand_checks import alert_latency_s


def test_preregistered_grid_count_and_named_cells():
    jobs = build_jobs()
    # 3 paths * 5 presets * 3 delays * 50 seeds + 3 paths * 2 physics presets * 20 seeds = 2370.
    assert len(jobs) == 2370
    labels = {(j["labels"]["adapter"], j["labels"]["path"], j["labels"]["preset"], j["labels"]["delay_s"]) for j in jobs}
    assert ("point_mass", "crossing", "clean", 0.0) in labels
    assert ("point_mass", "skimming", "burst_30_0.5s", 0.5) in labels
    assert ("physics", "loiter", "burst_10_0.5s", 0.0) in labels


def test_tiny_run_is_reproducible_and_reports_hand_check(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    # Keep this test short while exercising the same writer and summary paths.
    tiny_jobs = build_jobs(pilot=True)[:2]
    run_experiment(first, pilot=True, n_processes=1, jobs=tiny_jobs)
    run_experiment(second, pilot=True, n_processes=1, jobs=tiny_jobs)
    assert (first / "exp033.csv").read_bytes() == (second / "exp033.csv").read_bytes()
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["hand_check"]["expected_by_delay_s"][0]["value_s"] == alert_latency_s(0.25, 0.0, 0.0)  # (3 - 0.5) x 0.1 s
    assert summary["pilot"] is True
    assert "passed" in summary["safety_check"]
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["outputs"]) == sorted(["exp033.csv", "summary.json", "alert_latency_crossing.png",
                                                  "alert_latency_loiter.png", "continuity_simulated.png"])
    assert summary["safety_check"]["passed"] is True
    assert summary["prediction"] == PREDICTION
    assert "prediction:" not in PREDICTION[:40]
    assert summary["prediction_source"] == PREDICTION_SOURCE
    assert any("horizontal displacement" in text for text in summary["deviations"])
    assert any("Addendum D" in text for text in summary["design_choices"])
    # Called from code: no command line is recorded (not pytest's), but the replay command names --output.
    assert manifest["argv"] == []
    assert "--output" in manifest["replay_command"]
