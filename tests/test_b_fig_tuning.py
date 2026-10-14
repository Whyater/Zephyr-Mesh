"""Tests for tools/make_s6_tuning_figure.py, built from the pilot tuning output."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import make_s6_tuning_figure as fig

PILOT = Path("/tmp/runb-pilots/tuning")
PNGS = (fig.MAIN_PNG, fig.MONITOR_PNG)

pytestmark = pytest.mark.skipif(
    not PILOT.is_dir(), reason="pilot output /tmp/runb-pilots/tuning is missing")


def copy_pilot(tmp_path: Path, name: str) -> Path:
    dest = tmp_path / name
    shutil.copytree(PILOT, dest)
    return dest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_deterministic_pngs_and_updates_manifest(tmp_path):
    run = copy_pilot(tmp_path, "run")
    first = fig.build(run, run)
    assert [p.name for p in first] == list(PNGS)
    hashes = {}
    for path in first:
        assert path.exists() and path.stat().st_size > 20_000
        hashes[path.name] = sha(path)
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in hashes.items():
        assert manifest["outputs"][name] == digest
    assert "summary.json" in manifest["outputs"]
    first_manifest_bytes = (run / "manifest.json").read_bytes()
    assert first_manifest_bytes.endswith(b"\n")

    other = tmp_path / "other"
    second = fig.build(run, other)
    assert {p.name: sha(p) for p in second} == hashes
    # Rebuilding in place leaves the manifest unchanged.
    fig.build(run, run)
    assert (run / "manifest.json").read_bytes() == first_manifest_bytes


def test_cli_default_output_dir_is_run_dir(tmp_path):
    run = copy_pilot(tmp_path, "run")
    assert fig.main([str(run)]) == 0
    for name in PNGS:
        assert (run / name).exists()


def test_full_grid_layout_from_expanded_pilot(tmp_path):
    """Expand the two pilot cells to the full 12-cell grid and check the figure still builds."""
    run = copy_pilot(tmp_path, "full")
    doc = json.loads((run / "summary.json").read_text())
    template = doc["cells"][0]
    cells = []
    for target in ("cv", "ct"):
        for delay in (0.0, 0.1, 0.3):
            for loss in (0.0, 0.3):
                cell = copy.deepcopy(template)
                cell["cell"].update(target=target, delay_s=delay, loss_probability=loss)
                cells.append(cell)
    doc["cells"] = cells
    (run / "summary.json").write_text(json.dumps(doc))
    written = fig.build(run, run)
    assert all(p.stat().st_size > 20_000 for p in written)
    classes = fig.classify_arms(doc)
    assert set(classes) == set(fig.MODELS)
    # Twins share the axis: Q 0.04 and R std 5x both sit at ratio 25 relative to R std 0.2x.
    cv = classes["cv"]
    assert (25.0, "cv_q25") in cv["q"] and (25.0, "cv_r0.2") in cv["r"]
    assert (0.04, "cv_q0.04") in cv["q"] and (0.04, "cv_r5") in cv["r"]
    assert cv["nominal"] == "recovery"


def test_bad_input_is_a_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert fig.main([str(empty)]) == 2
