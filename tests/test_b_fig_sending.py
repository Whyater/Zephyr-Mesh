from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "make_s6_sending_figure.py"
PILOT = Path("/tmp/runb-pilots/sending")
NAMES = ("s6_sending_packets.png", "s6_sending_bytes.png")

pytestmark = pytest.mark.skipif(
    not (PILOT / "summary.json").is_file(),
    reason="pilot output /tmp/runb-pilots/sending is missing; run the sending pilot first",
)


def _build(run_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), "--output-dir", str(out_dir)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=180,
    )


def _copy_pilot(dest: Path) -> Path:
    shutil.copytree(PILOT, dest)
    return dest


def test_figures_exist_are_stable_and_hash_the_manifest(tmp_path):
    first = _copy_pilot(tmp_path / "first")
    second = _copy_pilot(tmp_path / "second")
    done = _build(first, first)
    assert done.returncode == 0, done.stderr
    assert _build(second, second).returncode == 0
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for name in NAMES:
        data = (first / name).read_bytes()
        assert data.startswith(b"\x89PNG")
        assert len(data) > 20_000
        assert data == (second / name).read_bytes()
        assert manifest["outputs"][name] == hashlib.sha256(data).hexdigest()
    # The original outputs stay in the manifest, and the file ends with a newline.
    original = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in original["outputs"].items():
        assert manifest["outputs"][name] == digest
    assert (first / "manifest.json").read_text(encoding="utf-8").endswith("}\n")


def test_separate_output_dir_and_wrong_experiment(tmp_path):
    run = _copy_pilot(tmp_path / "run")
    out = tmp_path / "out"
    assert _build(run, out).returncode == 0
    assert all((out / name).is_file() for name in NAMES)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["experiment"] = "something-else"
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    bad = _build(run, tmp_path / "bad")
    assert bad.returncode != 0
    assert "experiment" in bad.stderr


def test_script_text_rules():
    text = SCRIPT.read_text(encoding="utf-8")
    assert chr(0x2014) not in text
    import re

    assert re.search(r"#[0-9A-Fa-f]{6}", text) is None
