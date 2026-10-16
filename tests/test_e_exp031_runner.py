"""Checks for the EXP-031 sweep runner."""
from __future__ import annotations

import json
from pathlib import Path

from tests.platform_compare import same_tree
from tools.e_exp031_sweep import build_jobs, main, run_experiment


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 5*2*2*50 + physics 2*1*2*20 + exploratory 2*2*2*50 = 1560.
    assert len(jobs) == 5 * 2 * 2 * 50 + 2 * 1 * 2 * 20 + 2 * 2 * 2 * 50
    assert jobs[0]["labels"] == {"slice": "main", "preset": "clean", "dropout": "none",
                                  "message_mode": "delta", "adapter": "point_mass", "seed_index": 0}
    assert any(j["labels"]["slice"] == "physics" and j["scenario"]["adapter"] == "physics" for j in jobs)
    assert any(j["labels"]["slice"] == "exploratory, added after the pilot, not pre-registered"
               and j["scenario"]["policy"]["land_after_s"] == 1.0 for j in jobs)
    assert len(build_jobs(pilot=True)) == 26


def test_pilot_outputs_are_deterministic(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    tiny_jobs = []
    for job in build_jobs(pilot=True)[:2]:
        tiny = dict(job)
        tiny["scenario"] = dict(job["scenario"], duration_s=0.1)
        tiny_jobs.append(tiny)
    run_experiment(first, pilot=True, n_processes=1, jobs=tiny_jobs)
    run_experiment(second, pilot=True, n_processes=1, jobs=tiny_jobs)
    expected = {"sweep.csv", "summary.json", "sweep_completion_simulated.png",
                "sweep_belief_error_simulated.png", "manifest.json"}
    assert {p.name for p in first.iterdir()} == expected
    assert (first / "sweep.csv").read_bytes() == (second / "sweep.csv").read_bytes()
    left = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert same_tree(left, right)
    assert left["pilot"] is True
    assert left["hand_check"]["reference_sweeping_s"] == 800 / (3 * 4 * 2)
    assert left["missing_metrics"] == []
    assert all("failsafe_landings" in cell["metrics"] for cell in left["cells"])

