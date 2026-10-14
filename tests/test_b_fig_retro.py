from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import make_s6_retro_figure as fig

PILOT = Path("/tmp/runb-pilots/retro")
NAMES = (fig.MAIN_NAME, fig.REORDER_NAME)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def pilot_copy(tmp_path):
    if not PILOT.is_dir():
        pytest.skip(f"pilot output {PILOT} is missing; run tools/run_s6_retro.py --pilot first")
    dest = tmp_path / "retro"
    shutil.copytree(PILOT, dest)
    return dest


def test_builds_deterministic_pngs_and_updates_manifest(pilot_copy, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(pilot_copy / "manifest.json", out / "manifest.json")
    before = json.loads((out / "manifest.json").read_text())

    first = fig.build(pilot_copy, out)
    assert [p.name for p in first] == list(NAMES)
    hashes = {}
    for name in NAMES:
        path = out / name
        assert path.exists()
        assert path.stat().st_size > 20_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        hashes[name] = _sha(path)

    fig.build(pilot_copy, out)
    for name in NAMES:
        assert _sha(out / name) == hashes[name], f"{name} is not byte-identical across builds"

    text = (out / "manifest.json").read_text()
    manifest = json.loads(text)
    assert text.endswith("\n")
    for name in NAMES:
        assert manifest["outputs"][name] == hashes[name]
    # Everything that was already there is kept.
    for key, value in before["outputs"].items():
        assert manifest["outputs"][key] == value
    assert {k: v for k, v in manifest.items() if k != "outputs"} == {
        k: v for k, v in before.items() if k != "outputs"
    }
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def test_default_output_dir_is_the_run_dir(pilot_copy):
    assert fig.main([str(pilot_copy)]) == 0
    for name in NAMES:
        assert (pilot_copy / name).exists()
    manifest = json.loads((pilot_copy / "manifest.json").read_text())
    assert set(NAMES) <= set(manifest["outputs"])


def test_missing_summary_gives_clear_error(tmp_path, capsys):
    assert fig.main([str(tmp_path)]) == 2
    assert "summary.json" in capsys.readouterr().err


def _full_doc_from(pilot: dict) -> dict:
    """Synthetic summary with every panel filled (test only, not simulation output)."""
    doc = copy.deepcopy(pilot)
    template = doc["cells"][1]
    cells = []
    for slice_name, ratio in (("J1", 0.5), ("J2", 1.0)):
        for target in ("cv", "ct"):
            for loss in (0.0, 0.3):
                for delay in (0.05, 0.15, 0.3):
                    cell = copy.deepcopy(template)
                    cell["slice"] = slice_name
                    cell["cell"].update(target=target, delay_s=delay, jitter_s=delay * ratio,
                                        loss_probability=loss)
                    for i, arm in enumerate(("naive", "recovery", "retro")):
                        base = 0.1 + delay * (2.0 - 0.3 * i) + (0.1 if target == "ct" else 0.0)
                        cell["arms"][arm].update(
                            p95_median_m=base, p95_ci_low_m=base * 0.9, p95_ci_high_m=base * 1.1,
                            label="fails" if base > 0.25 else "passes")
                    cells.append(cell)
    doc["cells"] = cells
    return doc


def test_full_grid_builds_with_every_panel(pilot_copy, tmp_path):
    pilot = json.loads((pilot_copy / "summary.json").read_text())
    run = tmp_path / "full"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(_full_doc_from(pilot)))
    out = tmp_path / "full_out"
    fig.build(run, out)
    for name in NAMES:
        assert (out / name).stat().st_size > 20_000
    assert not (out / "manifest.json").exists()
