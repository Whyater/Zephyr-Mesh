"""Result writers must write LF line endings on every platform.

The run registry (tools/reproduce_registry.json) stores sha256 hashes recorded on
macOS, where text-mode writes produce LF. On Windows, a text-mode write without
``newline=`` turns every "\\n" into "\\r\\n", so a rerun there can never match the
registered bytes even when every number is identical.

Two checks:

1. A direct one: ``sim.estimation.sweep.write_json`` (the writer behind the
   ``s6-atlas`` JSON outputs and the other S6 runners) must write no carriage
   returns. It fails on Windows if the writer forgets ``newline="\\n"``; on macOS
   and Linux it passes either way, which is why check 2 exists.
2. A static one that runs on every platform: walk the import closure of every
   script a registry command runs, and fail on any text-mode write that does not
   pin ``newline=``. Binary writes and writes that pass ``newline=`` are fine
   (csv writers opened with ``newline=""`` keep their own line terminator).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "tools" / "reproduce_registry.json"

# Unpinned writes the scan still finds, with the number of call sites and why each
# is left alone. The test fails if a count no longer matches, so fixing one of these
# means deleting its line here.
KNOWN_UNPINNED = {
    # These three runners store the sha256 of their own source in the registered
    # summary.json (code_sha256), so any edit to them changes registered bytes.
    # Fix them together with a re-registration of those runs.
    "tools/run_s6_recovery.py": 2,
    "tools/run_s6_adaptive_r.py": 2,
    "tools/run_s6_q_sweep.py": 2,
    # Imported by registry scripts, but the unpinned writes are never called by a
    # registry command: trace and hardware-profile files, and the calibration
    # tool's own CLI output and synthetic temp-folder helper.
    "sim/trace.py": 1,
    "sim/hardware.py": 1,
    "tools/calibrate_radio_from_trace.py": 3,
}

# Calls that open a file for text writing and take a ``newline=`` keyword.
_OPENERS = {"open", "fdopen", "NamedTemporaryFile", "SpooledTemporaryFile", "TemporaryFile"}
_WRITE_FLAGS = set("wax+")


# ---------------------------------------------------------------------------
# Check 1: the s6-atlas JSON writer
# ---------------------------------------------------------------------------

def test_write_json_bytes_have_no_carriage_return(tmp_path: Path) -> None:
    from sim.estimation.sweep import write_json

    target = tmp_path / "atlas.json"
    write_json(target, {"a": 1, "b": [1.5, 2.5], "nested": {"c": "x"}})
    data = target.read_bytes()
    assert b"\r" not in data, data[:80]
    assert data == b'{\n  "a": 1,\n  "b": [\n    1.5,\n    2.5\n  ],\n  "nested": {\n    "c": "x"\n  }\n}\n'


def test_atomic_checkpoint_writer_has_no_carriage_return(tmp_path: Path) -> None:
    from sim.estimation.sweep import _write_atomic

    target = tmp_path / "job.json"
    _write_atomic(target, "{\n  \"a\": 1\n}\n")
    assert target.read_bytes() == b'{\n  "a": 1\n}\n'


# ---------------------------------------------------------------------------
# Check 2: static scan of every module a registry command imports
# ---------------------------------------------------------------------------

def _registry_scripts() -> list[Path]:
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))
    scripts: set[Path] = set()
    for entry in entries:
        for match in re.findall(r"(?:^|\s)((?:tools|sim)/[\w/]+\.py|main\.py)\b", entry["command"]):
            path = REPO / match
            if path.is_file():
                scripts.add(path)
    return sorted(scripts)


def _resolve(module: str, here: Path) -> list[Path]:
    """Repo files that importing ``module`` would load (package __init__ files too)."""
    parts = module.split(".")
    found: list[Path] = []
    for base in (REPO, here.parent):
        for i in range(1, len(parts) + 1):
            stem = base.joinpath(*parts[:i])
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file() and REPO in candidate.resolve().parents:
                    found.append(candidate.resolve())
    return found


def _imports(path: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[Path] = []
    package = path.parent
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out += _resolve(alias.name, path)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.parent
                rel = ".".join(base.relative_to(REPO).parts)
                module = ".".join(p for p in (rel, node.module or "") if p)
            else:
                module = node.module or ""
            out += _resolve(module, path)
            for alias in node.names:  # "from sim import drone" loads sim/drone.py
                out += _resolve(f"{module}.{alias.name}", path)
    return out


def _closure() -> list[Path]:
    seen: set[Path] = set()
    todo = [p.resolve() for p in _registry_scripts()]
    while todo:
        path = todo.pop()
        if path in seen or "tests" in path.relative_to(REPO).parts:
            continue
        seen.add(path)
        todo += _imports(path)
    return sorted(seen)


def _mode(call: ast.Call, position: int) -> str | None:
    """The literal mode string, ``""`` if absent, None if it is not a literal."""
    for kw in call.keywords:
        if kw.arg == "mode":
            node = kw.value
            break
    else:
        if len(call.args) > position:
            node = call.args[position]
        else:
            return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _text_writes_without_newline(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if any(kw.arg == "newline" for kw in node.keywords):
            continue
        shown = path.relative_to(REPO).as_posix() if REPO in path.resolve().parents else path.name
        where = f"{shown}:{node.lineno}"
        if name == "write_text":
            problems.append(f"{where} write_text without newline=")
        elif name == "TextIOWrapper":
            problems.append(f"{where} TextIOWrapper without newline=")
        elif name in {"savetxt", "to_csv"}:
            problems.append(f"{where} {name} (writes os.linesep)")
        elif name in _OPENERS:
            # builtin open(file, mode) and os.fdopen(fd, mode): mode is 2nd;
            # Path.open(mode) and gzip/bz2/lzma.open(file, mode): see below.
            is_method = isinstance(func, ast.Attribute)
            owner = func.value.id if is_method and isinstance(func.value, ast.Name) else ""
            if name in {"NamedTemporaryFile", "SpooledTemporaryFile", "TemporaryFile"}:
                mode = _mode(node, 0)
                mode = "w+b" if mode == "" else mode
            elif is_method and owner not in {"io", "os", "gzip", "bz2", "lzma", "codecs"}:
                mode = _mode(node, 0)  # Path.open(mode)
            else:
                mode = _mode(node, 1)
            if mode is None:
                problems.append(f"{where} {name} with a non-literal mode")
                continue
            if owner in {"gzip", "bz2", "lzma"} and "t" not in mode:
                continue  # these open in binary unless "t" is given
            if "b" in mode or not (_WRITE_FLAGS & set(mode)):
                continue
            problems.append(f"{where} {name}(mode={mode!r}) without newline=")
    return problems


# Line numbers in the comments are the line numbers inside the string.
SCANNER_CASES = r'''import gzip, io, os, tempfile
from pathlib import Path
p = Path("x")
p.write_text("a")                                  # 4 flagged
p.write_text("a", newline="\n")                    # 5 ok
open("x", "w")                                     # 6 flagged
open("x", "wb")                                    # 7 ok (binary)
open("x")                                          # 8 ok (read)
open("x", "w", newline="")                         # 9 ok
p.open("a", encoding="utf-8")                      # 10 flagged
p.open("rb")                                       # 11 ok
gzip.open("x", "wb")                               # 12 ok (binary)
gzip.open("x", "wt")                               # 13 flagged
gzip.open("x", "w")                                # 14 ok (gzip "w" is binary)
io.TextIOWrapper(raw)                              # 15 flagged
os.fdopen(fd, "w", encoding="utf-8")               # 16 flagged
open("x", mode)                                    # 17 flagged (unknown mode)
tempfile.NamedTemporaryFile("w", suffix=".json")   # 18 flagged
tempfile.NamedTemporaryFile(suffix=".json")        # 19 ok (w+b default)
'''


def test_scanner_flags_exactly_the_text_writes(tmp_path: Path) -> None:
    source = tmp_path / "cases.py"
    source.write_bytes(SCANNER_CASES.encode("utf-8"))
    flagged = sorted(int(p.split(":")[1].split()[0]) for p in _text_writes_without_newline(source))
    assert flagged == [4, 6, 10, 13, 15, 16, 17, 18]


def test_registry_closure_is_found() -> None:
    closure = {p.relative_to(REPO).as_posix() for p in _closure()}
    # Spot checks that the import walk reaches the shared writers.
    assert "tools/run_s6_atlas.py" in closure
    assert "sim/estimation/sweep.py" in closure
    assert len(closure) > 40


def test_registry_writers_pin_newline() -> None:
    problems = [p for path in _closure() for p in _text_writes_without_newline(path)]
    counts: dict[str, int] = {}
    for problem in problems:
        name = problem.split(":", 1)[0]
        counts[name] = counts.get(name, 0) + 1
    unexpected = [p for p in problems if p.split(":", 1)[0] not in KNOWN_UNPINNED]
    assert not unexpected, "text-mode writes without newline= (CRLF on Windows):\n" + "\n".join(unexpected)
    stale = {name: (want, counts.get(name, 0)) for name, want in KNOWN_UNPINNED.items()
             if counts.get(name, 0) != want}
    assert not stale, f"KNOWN_UNPINNED is out of date (expected, found): {stale}"


@pytest.mark.skipif(sys.platform != "win32", reason="the CRLF translation only happens on Windows")
def test_unpinned_write_text_really_writes_crlf_here(tmp_path: Path) -> None:
    """Shows the scan targets a real effect on this platform, not a style rule."""
    target = tmp_path / "probe.json"
    target.write_text("{\n}\n")
    assert target.read_bytes() == b"{\r\n}\r\n"
