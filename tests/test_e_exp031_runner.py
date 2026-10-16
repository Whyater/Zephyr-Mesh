"""Checks for the EXP-031 sweep runner."""
from __future__ import annotations

import json
from pathlib import Path

from tests.platform_compare import same_tree
from tools.e_exp031_sweep import EXPLORATORY_LABEL, PREDICTION, PREDICTION_NOTE, build_jobs, run_experiment


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 5 presets x 2 dropouts x 2 modes x 50 seeds = 1,000; physics 2 presets x 1 dropout x 2 modes
    # x 20 seeds = 80; exploratory calm 2 presets x 2 dropouts x 2 modes x 50 seeds = 400; total 1,480.
    slices = [job["labels"]["slice"] for job in jobs]
    assert slices.count("main") == 1000
    assert slices.count("physics") == 80
    assert slices.count(EXPLORATORY_LABEL) == 400
    assert len(jobs) == 1480
    assert jobs[0]["labels"] == {"slice": "main", "preset": "clean", "dropout": "none",
                                  "message_mode": "delta", "adapter": "point_mass", "seed_index": 0}
    assert any(j["labels"]["slice"] == "physics" and j["scenario"]["adapter"] == "physics" for j in jobs)
    assert any(j["labels"]["slice"] == EXPLORATORY_LABEL
               and j["scenario"]["policy"]["land_after_s"] == 1.0 for j in jobs)
    # Pilot: main 2 presets x 2 dropouts x 2 modes x 3 seeds = 24; physics 1 cell x 2 seeds = 2;
    # one calm cell (iid_30, dropout one, delta) x 2 seeds = 2; total 28.
    pilot_jobs = build_jobs(pilot=True)
    assert len(pilot_jobs) == 28
    calm = [j for j in pilot_jobs if j["labels"]["slice"] == EXPLORATORY_LABEL]
    assert len(calm) == 2
    assert all(j["labels"]["preset"] == "iid_30" and j["labels"]["dropout"] == "one"
               and j["labels"]["message_mode"] == "delta" for j in calm)
    assert {j["scenario"]["duration_s"] for j in pilot_jobs} == {60.0}


def test_pilot_outputs_are_deterministic(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    tiny_jobs = []
    for job in build_jobs(pilot=True)[:2]:
        tiny = dict(job)
        tiny["scenario"] = dict(job["scenario"], duration_s=0.1)
        tiny_jobs.append(tiny)
    run_experiment(first, pilot=True, n_processes=1, jobs=tiny_jobs, argv=["--output", "a folder/out"])
    run_experiment(second, pilot=True, n_processes=1, jobs=tiny_jobs)
    expected = {"sweep.csv", "summary.json", "sweep_completion_simulated.png",
                "sweep_belief_error_simulated.png", "manifest.json"}
    assert {p.name for p in first.iterdir()} == expected
    assert (first / "sweep.csv").read_bytes() == (second / "sweep.csv").read_bytes()
    left = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert same_tree(left, right)
    assert left["pilot"] is True
    assert left["duration_s"] == 0.1
    assert left["prediction"]["text"] == PREDICTION
    assert left["prediction"]["text"].startswith("without dropout")
    assert left["prediction"]["prediction_note"] == PREDICTION_NOTE
    assert left["prediction"]["prediction_source"].startswith("pre-registered before the runs")
    verdicts = left["hand_check"]["observed_clean_no_dropout"]
    assert [v["message_mode"] for v in verdicts] == ["delta", "cumulative"]
    assert all(v["verdict"] in {"inside", "outside", "not run"} for v in verdicts)
    assert verdicts[0]["n_runs"] == 2 and verdicts[0]["verdict"] == "outside"
    assert verdicts[1]["verdict"] == "not run"
    assert all("within_0.1_s" in key for item in left["atlas"].values() for key in item if "wilson" in key)
    assert "120" not in "".join(key for item in left["atlas"].values() for key in item)
    assert left["notes"][0].startswith("Mission changes before any full run:")
    assert any("keeps sharing" in line for line in left["limits"])
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["replay_command"] == "PYTHONPATH=. python tools/e_exp031_sweep.py --output 'a folder/out'"
    assert left["hand_check"]["reference_sweeping_s"] == 800 / (3 * 4 * 2)
    assert left["missing_metrics"] == []
    assert all("failsafe_landings" in cell["metrics"] for cell in left["cells"])

