"""Tests for tools/reproduce_all.py using isolated temporary source trees."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "reproduce_all.py"
PYTHON = sys.executable


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_registry(path: Path, entries: list[dict]) -> Path:
    registry = path / "registry.json"
    registry.write_text(json.dumps(entries), encoding="utf-8")
    return registry


def run_tool(source: Path, registry: Path, *args: str) -> subprocess.CompletedProcess[str]:
    command = [PYTHON, str(TOOL), "--source-dir", str(source), "--registry", str(registry), "--python", PYTHON, *args]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    return source


def basic_entry(name: str, expected: str, command: str = "{python} gen.py {out}/value.txt") -> dict:
    return {
        "run_dir": f"runs/{name}",
        "command": command,
        "outputs": {"value.txt": expected},
    }


def test_byte_identical_pass(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gen.py").write_text("from pathlib import Path\nPath(__file__).parent.joinpath('out').mkdir(exist_ok=True)\nPath(__import__('sys').argv[1]).write_bytes(b'stable\\n')\n", encoding="utf-8")
    expected = digest(b"stable\n")
    registry = write_registry(tmp_path, [basic_entry("stable", expected, "{python} gen.py {out}/value.txt")])

    result = run_tool(source, registry)

    assert result.returncode == 0, result.stderr
    assert "byte-identical" in result.stdout.lower()


def test_nondeterministic_generator_is_reported_as_differs(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gen.py").write_text(
        "import time\nfrom pathlib import Path\nPath(__import__('sys').argv[1]).write_text(str(time.time_ns()))\n",
        encoding="utf-8",
    )
    registry = write_registry(tmp_path, [basic_entry("changing", digest(b"fixed\n"))])

    result = run_tool(source, registry)

    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "differs" in output
    assert "value.txt" in output


def test_crashing_command_reports_exit_code_and_failed(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gen.py").write_text("import sys\nprint('generator problem', file=sys.stderr)\nsys.exit(7)\n", encoding="utf-8")
    registry = write_registry(tmp_path, [basic_entry("crash", digest(b"unused"))])

    result = run_tool(source, registry)

    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "failed" in output
    assert "7" in output
    assert "generator problem" in output


@pytest.mark.parametrize(
    ("label", "entries", "needle"),
    [
        ("extra key", [{"run_dir": "runs/x", "command": "echo {out}", "outputs": {}, "extra": 1}], "key"),
        ("missing token", [{"run_dir": "runs/x", "command": "echo hello", "outputs": {}}], "{out}"),
        ("bad hash", [{"run_dir": "runs/x", "command": "echo {out}", "outputs": {"x": "xyz"}}], "sha256"),
        (
            "duplicate run",
            [
                {"run_dir": "runs/x", "command": "echo {out}", "outputs": {}},
                {"run_dir": "runs/x", "command": "echo {out}", "outputs": {}},
            ],
            "duplicate",
        ),
    ],
)
def test_malformed_registry_is_rejected(tmp_path: Path, label: str, entries: list[dict], needle: str) -> None:
    source = make_source(tmp_path)
    registry = write_registry(tmp_path, entries)

    result = run_tool(source, registry)

    assert result.returncode != 0, label
    assert needle.lower() in (result.stdout + result.stderr).lower(), label


def test_real_runs_tree_is_unchanged(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gen.py").write_text("from pathlib import Path\nPath(__import__('sys').argv[1]).write_text('ok')\n", encoding="utf-8")
    expected = digest(b"ok")
    registry = write_registry(tmp_path, [basic_entry("isolated", expected)])

    before = {
        p.relative_to(ROOT / "runs"): (p.stat().st_mtime_ns, digest(p.read_bytes()))
        for p in (ROOT / "runs").rglob("*")
        if p.is_file()
    }
    result = run_tool(source, registry)
    after = {
        p.relative_to(ROOT / "runs"): (p.stat().st_mtime_ns, digest(p.read_bytes()))
        for p in (ROOT / "runs").rglob("*")
        if p.is_file()
    }

    assert result.returncode == 0, result.stderr
    assert before == after


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the fixture counts workers with fcntl, which Windows lacks")
def test_workers_cap_and_filters(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    state_path = tmp_path / "state"
    state_path.write_text("0 0", encoding="utf-8")
    (source / "gen.py").write_text(
        textwrap.dedent(
            """
            import fcntl
            import time
            from pathlib import Path
            state = Path(%r)
            with state.open('r+') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                current, peak = map(int, handle.read().split())
                current += 1
                peak = max(peak, current)
                handle.seek(0); handle.truncate(); handle.write(f'{current} {peak}')
                handle.flush(); fcntl.flock(handle, fcntl.LOCK_UN)
            time.sleep(0.25)
            with state.open('r+') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                current, peak = map(int, handle.read().split())
                handle.seek(0); handle.truncate(); handle.write(f'{current - 1} {peak}')
                handle.flush(); fcntl.flock(handle, fcntl.LOCK_UN)
            Path(__import__('sys').argv[1]).write_text('ok')
            """
        ) % str(state_path),
        encoding="utf-8",
    )
    expected = digest(b"ok")
    entries = [basic_entry(name, expected) for name in ("one", "two", "three")]
    registry = write_registry(tmp_path, entries)

    result = run_tool(source, registry, "--workers", "1", "--only", "one,two")

    assert result.returncode == 0, result.stderr
    output = result.stdout.lower()
    assert "one" in output and "two" in output
    assert "three" not in output
    assert state_path.read_text(encoding="utf-8").split()[1] == "1"

    skipped = run_tool(source, registry, "--workers", "1", "--skip", "one,two")
    assert skipped.returncode == 0, skipped.stderr
    assert "three" in skipped.stdout.lower()
    assert "one" not in skipped.stdout.lower()


def test_real_registry_has_strict_shape() -> None:
    data = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    seen: set[str] = set()
    for entry in data:
        assert set(entry) == {"run_dir", "command", "outputs"}
        assert isinstance(entry["run_dir"], str)
        assert entry["run_dir"].startswith("runs/")
        assert entry["run_dir"] not in seen
        seen.add(entry["run_dir"])
        assert "{out}" in entry["command"]
        assert isinstance(entry["outputs"], dict)
        for filename, checksum in entry["outputs"].items():
            assert isinstance(filename, str)
            assert isinstance(checksum, str) and len(checksum) == 64
            int(checksum, 16)
