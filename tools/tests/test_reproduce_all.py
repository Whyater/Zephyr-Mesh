"""Tests for tools/reproduce_all.py using isolated temporary source trees."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
        "run_dir": f"sim/runs/{name}",
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
        ("extra key", [{"run_dir": "sim/runs/x", "command": "echo {out}", "outputs": {}, "extra": 1}], "key"),
        ("missing token", [{"run_dir": "sim/runs/x", "command": "echo hello", "outputs": {}}], "{out}"),
        ("bad hash", [{"run_dir": "sim/runs/x", "command": "echo {out}", "outputs": {"x": "xyz"}}], "sha256"),
        (
            "duplicate run",
            [
                {"run_dir": "sim/runs/x", "command": "echo {out}", "outputs": {}},
                {"run_dir": "sim/runs/x", "command": "echo {out}", "outputs": {}},
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
        p.relative_to(ROOT / "sim" / "runs"): (p.stat().st_mtime_ns, digest(p.read_bytes()))
        for p in (ROOT / "sim" / "runs").rglob("*")
        if p.is_file()
    }
    result = run_tool(source, registry)
    after = {
        p.relative_to(ROOT / "sim" / "runs"): (p.stat().st_mtime_ns, digest(p.read_bytes()))
        for p in (ROOT / "sim" / "runs").rglob("*")
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


def test_timeout_kills_runner_and_grandchild(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    pid_file = tmp_path / "pids"
    (source / "orphan.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            pid_file = Path(%r)
            grandchild = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid_file.write_text(f"{os.getpid()} {grandchild.pid}")
            # Close the launcher pipes so the timeout test can return while
            # this intentionally stuck runner is still alive.
            os.close(1)
            os.close(2)
            while True:
                time.sleep(1)
            """
        ) % str(pid_file),
        encoding="utf-8",
    )
    registry = write_registry(
        tmp_path,
        [basic_entry("orphan", digest(b"unused"), "{python} orphan.py {out}/value.txt")],
    )

    recorded_pids: list[int] = []
    try:
        result = run_tool(source, registry, "--timeout", "0.1")
        assert result.returncode != 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.02)
        recorded_pids = [int(value) for value in pid_file.read_text().split()]

        def running(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return bool(state) and not state.startswith("Z")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(running(pid) for pid in recorded_pids):
            time.sleep(0.05)
        assert not any(running(pid) for pid in recorded_pids)
    finally:
        for pid in recorded_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_large_stderr_does_not_trigger_false_timeout(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gen.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.stderr.write('x' * 500_000)\n"
        "sys.stderr.flush()\n"
        "Path(sys.argv[1]).write_bytes(b'stable\\n')\n",
        encoding="utf-8",
    )
    registry = write_registry(
        tmp_path,
        [basic_entry("stderr", digest(b"stable\n"))],
    )

    started = time.monotonic()
    result = run_tool(source, registry, "--timeout", "1")
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert "byte-identical" in result.stdout.lower()
    assert elapsed < 2.0


def test_sigint_stops_queued_entries_and_descendants(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    launch_log = tmp_path / "launches"
    (source / "gen.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            launch_log = Path(%r)
            grandchild = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with launch_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()} {grandchild.pid}\\n")
                handle.flush()
            time.sleep(30)
            Path(sys.argv[1]).write_bytes(b"stable\\n")
            """
        ) % str(launch_log),
        encoding="utf-8",
    )
    registry = write_registry(
        tmp_path,
        [basic_entry(name, digest(b"stable\n")) for name in ("one", "two", "three", "four")],
    )
    command = [
        PYTHON,
        str(TOOL),
        "--source-dir",
        str(source),
        "--registry",
        str(registry),
        "--python",
        PYTHON,
        "--workers",
        "1",
        "--timeout",
        "30",
    ]
    tool_proc = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not launch_log.exists():
            time.sleep(0.02)
        assert launch_log.exists()
        pids = [int(value) for value in launch_log.read_text(encoding="utf-8").split()]
        assert len(pids) == 2

        tool_proc.send_signal(signal.SIGINT)
        assert tool_proc.wait(timeout=10) == -signal.SIGINT
        assert len(launch_log.read_text(encoding="utf-8").split()) == 2

        def running(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return bool(state) and not state.startswith("Z")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(running(pid) for pid in pids):
            time.sleep(0.05)
        assert not any(running(pid) for pid in pids)
    finally:
        if tool_proc.poll() is None:
            tool_proc.kill()
            tool_proc.wait()


def test_real_registry_has_strict_shape() -> None:
    data = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    seen: set[str] = set()
    for entry in data:
        assert set(entry) == {"run_dir", "command", "outputs"}
        assert isinstance(entry["run_dir"], str)
        assert entry["run_dir"].startswith("sim/runs/")
        assert entry["run_dir"] not in seen
        seen.add(entry["run_dir"])
        assert "{out}" in entry["command"]
        assert isinstance(entry["outputs"], dict)
        for filename, checksum in entry["outputs"].items():
            assert isinstance(filename, str)
            assert isinstance(checksum, str) and len(checksum) == 64
            int(checksum, 16)
