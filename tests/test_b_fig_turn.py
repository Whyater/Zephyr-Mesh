from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import make_s6_turn_figure as fig

PILOT = Path("/tmp/runb-pilots/turn")
PNGS = ("s6_turn.png", "s6_turn_phases.png")


@pytest.fixture()
def run_copy(tmp_path):
    if not PILOT.is_dir():
        pytest.skip(f"pilot output {PILOT} is missing; run tools/run_s6_turn.py --pilot first")
    dest = tmp_path / "run"
    shutil.copytree(PILOT, dest)
    return dest


def test_figures_exist_and_are_deterministic(run_copy, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    written = fig.build(run_copy, first)
    fig.build(run_copy, second)
    assert [p.name for p in written] == list(PNGS)
    for name in PNGS:
        a = (first / name).read_bytes()
        b = (second / name).read_bytes()
        assert a.startswith(b"\x89PNG")
        assert len(a) > 20_000
        assert a == b


def test_manifest_gains_hashes_and_keeps_other_fields(run_copy):
    before = json.loads((run_copy / "manifest.json").read_text())
    fig.build(run_copy, run_copy)
    text = (run_copy / "manifest.json").read_text()
    after = json.loads(text)
    assert text.endswith("\n")
    assert text == json.dumps(after, indent=2, sort_keys=True) + "\n"
    for name in PNGS:
        digest = hashlib.sha256((run_copy / name).read_bytes()).hexdigest()
        assert after["outputs"][name] == digest
    for key, value in before["outputs"].items():
        assert after["outputs"][key] == value
    for key in before:
        if key != "outputs":
            assert after[key] == before[key]


def test_default_output_dir_is_run_dir(run_copy):
    assert fig.main([str(run_copy)]) == 0
    for name in PNGS:
        assert (run_copy / name).is_file()


def test_wrong_experiment_is_rejected(run_copy, capsys):
    path = run_copy / "summary.json"
    doc = json.loads(path.read_text())
    doc["experiment"] = "other"
    path.write_text(json.dumps(doc))
    assert fig.main([str(run_copy)]) == 2
    assert "unexpected experiment" in capsys.readouterr().err
