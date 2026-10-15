import csv
import json
from pathlib import Path

import numpy as np

from sim.estimation.engine import CellSpec, deliver_cell, run_arm
from sim.estimation.sweep import compare_outputs
from sim.s6_recovery import RecoveryConfig, run_arms, seed_list, simulate_run
from tools.run_s6_turn import build_arms, build_cells, job_cell, run


def test_grid_counts_and_unique_keys():
    full = build_cells()
    assert len(full) == 399
    assert sum(c["slice"] == "main" for c in full) == 336
    assert sum(c["slice"] == "burst" for c in full) == 63
    assert len({CellSpec(target=c["target"], delay_s=c["delay_s"], loss_probability=c["loss_probability"],
                         loss_model=c["loss_model"], noise_std_m=c["noise_std_m"], jitter_s=0.0,
                         burst_mean_length_packets=c.get("burst_mean_length_packets", 5.0)).key() for c in full}) == 399
    assert len(build_cells(True)) == 21


def test_registered_arms_include_q10_ca_and_ct():
    arms = build_arms()
    assert [arm.name for arm in arms] == ["naive", "recovery", "ca", "ct", "imm", "ca_q10", "ct_q10"]
    assert [(arm.model, arm.q_factor) for arm in arms[-2:]] == [("ca", 10.0), ("ct", 10.0)]


def test_job_json_and_recovery_match_reference():
    cell = {"slice": "main", "target": "cv", "delay_s": .1, "loss_probability": .3,
            "loss_model": "independent", "noise_std_m": .05, "jitter_s": 0.0, "seeds": 3}
    result = job_cell(cell)
    json.dumps(result, allow_nan=False)
    spec = CellSpec(target="cv", delay_s=.1, loss_probability=.3, noise_std_m=.05)
    cfg = RecoveryConfig()
    for seed in seed_list(3):
        ref = simulate_run("constant_velocity", delay_s=.1, loss_probability=.3, noise_std_m=.05, seed=seed, config=cfg)
        got = run_arm(deliver_cell(spec, seed, cfg), spec, build_arms()[0], cfg)
        assert abs(got.p95_error_m - ref["naive"].p95_error_m) <= 1e-12
        got_r = run_arm(deliver_cell(spec, seed, cfg), spec, build_arms()[1], cfg)
        assert abs(got_r.p95_error_m - ref["recovery"].p95_error_m) <= 1e-12
    mixed = job_cell({**cell, "target": "mixed", "seeds": 2})
    assert "phase_p95" not in result
    assert set(mixed["phase_p95"]) == {a.name for a in build_arms()}
    assert all(set(v) == {"straight1", "turn", "straight2"} for v in mixed["phase_p95"].values())


def test_small_run_is_reproducible(tmp_path: Path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    import tools.run_s6_turn as runner
    small = build_cells(True)[:3]
    monkeypatch.setattr(runner, "build_cells", lambda pilot=False: small)
    run(a, pilot=True, seeds=2, processes=1, checkpoint_dir=None, no_checkpoint=True, argv=[])
    run(b, pilot=True, seeds=2, processes=1, checkpoint_dir=None, no_checkpoint=True, argv=[])
    assert all(compare_outputs(a, b, [name])[name] for name in ("summary.json", "cells.csv", "per_seed.csv.gz"))


def test_cells_csv_columns_have_values(tmp_path: Path, monkeypatch):
    import tools.run_s6_turn as runner
    cell = build_cells(True)[0]
    monkeypatch.setattr(runner, "build_cells", lambda pilot=False: [cell])
    run(tmp_path / "out", pilot=True, seeds=1, processes=1, checkpoint_dir=None, no_checkpoint=True, argv=[])
    with (tmp_path / "out" / "cells.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(any(row[column].strip() for row in rows) for column in rows[0])
