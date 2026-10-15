from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "make_s6_clocks_figure.py"
PILOT = Path("/tmp/runb-pilots/clocks")
NAMES = ("s6_clocks.png", "s6_clocks_sync.png")

pytestmark = pytest.mark.skipif(
    not (PILOT / "summary.json").exists(),
    reason="pilot output /tmp/runb-pilots/clocks is missing; run tools/run_s6_clocks.py --pilot first",
)


def _copy_pilot(dest: Path) -> Path:
    shutil.copytree(PILOT, dest)
    for name in NAMES:
        (dest / name).unlink(missing_ok=True)
    return dest


def _run(run_dir: Path, out_dir: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), str(run_dir)]
    if out_dir is not None:
        cmd += ["--output-dir", str(out_dir)]
    return subprocess.run(cmd, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=300)


def test_figures_exist_are_stable_and_update_manifest(tmp_path):
    run_dir = _copy_pilot(tmp_path / "run")
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        out.mkdir()
        shutil.copy(run_dir / "manifest.json", out / "manifest.json")
        result = _run(run_dir, out)
        assert result.returncode == 0, result.stderr
    for name in NAMES:
        data = (first / name).read_bytes()
        assert data.startswith(b"\x89PNG")
        assert len(data) > 20_000
        assert data == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in NAMES:
        assert manifest["outputs"][name] == hashlib.sha256((first / name).read_bytes()).hexdigest()
    for name, digest in original["outputs"].items():
        assert manifest["outputs"][name] == digest
    assert {k: v for k, v in manifest.items() if k != "outputs"} == \
        {k: v for k, v in original.items() if k != "outputs"}


def test_default_output_dir_is_the_run_dir(tmp_path):
    run_dir = _copy_pilot(tmp_path / "run")
    result = _run(run_dir)
    assert result.returncode == 0, result.stderr
    for name in NAMES:
        assert (run_dir / name).stat().st_size > 20_000


def test_wrong_experiment_exits_nonzero(tmp_path):
    run_dir = _copy_pilot(tmp_path / "run")
    doc = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    doc["experiment_id"] = "something-else"
    (run_dir / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    result = _run(run_dir, tmp_path / "out")
    assert result.returncode != 0
    assert "experiment" in result.stderr


def test_script_has_no_hex_colors_or_em_dashes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"#[0-9A-Fa-f]{6}", text) is None
    assert chr(0x2014) not in text
