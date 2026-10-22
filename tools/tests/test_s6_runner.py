from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.run_s6_recovery import build_cells, run_sweep


def test_build_cells_counts_and_labels():
    cells = build_cells()
    assert [sum(c["slice"] == name for c in cells) for name in ("main", "burst", "jitter", "oracle")] == [224, 42, 24, 28]
    assert len(cells) == 318
    assert len({c["cell_id"] for c in cells}) == 318
    assert all(c["loss_model"] == "burst" for c in cells if c["slice"] == "burst")
    assert all(c["jitter_s"] == c["delay_s"] / 2 for c in cells if c["slice"] == "jitter")
    assert all(c["tracker_measurement_std_m"] == 0.10 for c in cells if c["slice"] == "oracle")
    pilot = build_cells(pilot=True)
    assert len(pilot) == 24


def test_runner_outputs_are_process_deterministic(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    run_sweep(one, pilot=True, seeds_count=3, processes=1, overwrite=False, argv=["--pilot", "--seeds", "3"])
    run_sweep(two, pilot=True, seeds_count=3, processes=2, overwrite=False, argv=["--pilot", "--seeds", "3"])
    assert (one / "summary.json").read_bytes() == (two / "summary.json").read_bytes()
    assert (one / "per_seed.csv").read_bytes() == (two / "per_seed.csv").read_bytes()


def test_csv_recomputes_summary_and_manifest_hashes(tmp_path):
    out = tmp_path / "run"
    run_sweep(out, pilot=True, seeds_count=3, processes=1, argv=["--pilot", "--seeds", "3"])
    summary = json.loads((out / "summary.json").read_text())
    summary_text = json.dumps(summary)
    assert str(tmp_path) not in summary_text
    assert "git_revision" not in summary and "timestamp" not in summary_text.lower()
    with (out / "per_seed.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24 * 2 * 3
    for cell in summary["cells"]:
        for arm_name in ("naive", "recovery"):
            vals = [float(r["p95_error_m"]) for r in rows if r["cell_id"] == cell["cell_id"] and r["arm"] == arm_name]
            assert float(np.median(vals)) == cell["arms"][arm_name]["p95_median_m"]
    manifest = json.loads((out / "manifest.json").read_text())
    for name in ("summary.json", "per_seed.csv"):
        assert manifest["outputs"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_nonempty_output_requires_overwrite(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        run_sweep(out, pilot=True, seeds_count=1, processes=1)
