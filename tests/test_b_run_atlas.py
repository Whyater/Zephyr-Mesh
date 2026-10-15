"""Fast checks for the S6 atlas grid and deterministic output path."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sim.estimation.sweep import compare_outputs
import tools.run_s6_atlas as runner


def test_grid_size_slices_and_unique_keys():
    cells = runner.build_cells()
    assert len(cells) == 3969
    assert {c["slice"] for c in cells} == set(runner.STRATA)
    assert len({runner._cell_key(c) for c in cells}) == len(cells)
    assert len(runner.build_cells(pilot=True)) == 81


def test_speed_variants_central_only_reduces_grid():
    cells = runner.build_cells(speed_variants_central_only=True)
    # The two non-central speed variants are omitted from eight strata.
    assert len(cells) == (9 + 8 * 7) * 7 * 7
    assert {c["slice"] for c in cells if c["target"] == "cv" and c.get("target_params") in ({"speed_mps": 0.5}, {"speed_mps": 2.0})} == {"central"}


def test_imm_strata_restricts_only_imm_arms():
    # Cell enumeration is independent of which estimator arms are enabled.
    assert len(runner.build_cells(imm_strata="central")) == 3969
    assert [a.name for a in runner.build_arms("central", imm_strata="central")] == [
        "naive", "recovery", "imm", "ct", "adaptive"
    ]
    assert [a.name for a in runner.build_arms("rate5", imm_strata="central")] == [
        "naive", "recovery", "adaptive"
    ]


def test_arm_sets():
    assert [a.name for a in runner.build_arms("central")] == ["naive", "recovery", "imm", "ct", "adaptive"]
    assert [a.name for a in runner.build_arms("jitter")] == ["naive", "recovery", "imm", "ct", "adaptive", "retro"]
    assert [a.name for a in runner.build_arms("student_t")] == [
        "naive", "recovery", "imm", "ct", "adaptive", "recovery_gated", "imm_gated", "ct_gated"
    ]


def test_burst_mean_time_conversion_implementation_contract():
    """The registered 0.5 s burst duration is encoded in packet counts."""
    for cell in runner.build_cells():
        if cell["loss_model"] == "burst":
            assert cell["burst_mean_length_packets"] == 0.5 * cell["rate_hz"]


def test_tiny_deterministic_outputs(tmp_path, monkeypatch):
    cells = runner.build_cells(pilot=True)[:2]
    monkeypatch.setattr(runner, "build_cells", lambda pilot=False: cells)
    runner.run_sweep(tmp_path / "a", seeds_count=2, processes=1, no_checkpoint=True)
    runner.run_sweep(tmp_path / "b", seeds_count=2, processes=1, no_checkpoint=True)
    names = ["summary.json", "cells.csv", "per_seed.csv.gz", "counterfactuals.json", "atlas.json"]
    assert all(compare_outputs(tmp_path / "a", tmp_path / "b", names).values())


def test_runtime_options_are_recorded(tmp_path, monkeypatch):
    cells = runner.build_cells(pilot=True)[:1]
    monkeypatch.setattr(runner, "build_cells", lambda *args, **kwargs: cells)
    runner.run_sweep(tmp_path / "out", seeds_count=1, processes=1, no_checkpoint=True,
                     imm_strata="central", speed_variants_central_only=True)
    import json
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["burst_mean_s"] == 0.5
    design = summary["design"]
    assert "imm_strata" in json.dumps(design)
    assert "speed_variants_central_only" in json.dumps(design)
    atlas = json.loads((tmp_path / "out" / "atlas.json").read_text())
    assert "imm_strata" in json.dumps(atlas["design"])
    assert "speed_variants_central_only" in json.dumps(atlas["design"])


def test_summary_does_not_duplicate_per_seed_and_columns_are_fixed(tmp_path, monkeypatch):
    cells = runner.build_cells(pilot=True)[:1]
    monkeypatch.setattr(runner, "build_cells", lambda *args, **kwargs: cells)
    runner.run_sweep(tmp_path / "out", seeds_count=1, processes=1, no_checkpoint=True)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert all("per_seed" not in cell for cell in summary["cells"])
    with gzip.open(tmp_path / "out" / "per_seed.csv.gz", "rt", newline="") as handle:
        assert handle.readline().rstrip("\n\r") == ",".join(runner.PER_SEED_COLUMNS)
    assert (tmp_path / "out" / "cells.csv").read_text().splitlines()[0] == ",".join(runner.CELL_CSV_COLUMNS)


def test_manifest_replay_command_uses_actual_argv(tmp_path, monkeypatch):
    cells = runner.build_cells(pilot=True)[:1]
    monkeypatch.setattr(runner, "build_cells", lambda *args, **kwargs: cells)
    args = ["--output", "runs/space name", "--pilot", "--seeds", "1"]
    runner.run_sweep(tmp_path / "out", seeds_count=1, processes=1, no_checkpoint=True, argv=args)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["replay_command"] == (
        "PYTHONPATH=. ./.venv/bin/python tools/run_s6_atlas.py --output 'runs/space name' --pilot --seeds 1"
    )


def test_resume_after_deleted_checkpoints_is_byte_identical(tmp_path, monkeypatch):
    cells = runner.build_cells(pilot=True)[:2]
    monkeypatch.setattr(runner, "build_cells", lambda *args, **kwargs: cells)
    checkpoint_dir = tmp_path / "checkpoints"
    runner.run_sweep(tmp_path / "a", seeds_count=1, processes=1, checkpoint_dir=checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.json"))
    assert checkpoints
    for path in checkpoints[::2]:
        path.unlink()
    runner.run_sweep(tmp_path / "b", seeds_count=1, processes=2, checkpoint_dir=checkpoint_dir)
    names = ["summary.json", "cells.csv", "per_seed.csv.gz", "counterfactuals.json", "atlas.json"]
    assert all(compare_outputs(tmp_path / "a", tmp_path / "b", names).values())


def test_first_failures_rejects_missing_fail_rate_fields():
    cell = runner.build_cells(pilot=True)[0]
    item = {"slice": cell["slice"], "cell": cell,
            "arms": {"naive": {"fail_rate": 0.0, "fail_rate_ci_low": 0.0}}}
    with pytest.raises(ValueError, match="fail_rate_ci_high"):
        runner._first_failures([item])


def test_job_cell_result_is_json_serializable():
    import json

    cell = runner.build_cells(pilot=True)[0]
    result = runner.job_cell({"cell": cell, "slice": cell["slice"], "seeds": runner.seed_list(1)})
    json.dumps(result, allow_nan=False)
