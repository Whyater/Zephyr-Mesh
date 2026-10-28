"""Privacy scan of every tracked file in the public repository.

Each tracked text file is read line by line and checked for home-folder paths
(macOS, Linux and Windows), internal decision IDs, wording that calls the
pre-registration private, and the names of the AI tools used while writing
the code.  Tracked file names are checked for private notes and tool folders.

A few hits sit in committed run outputs, in the runners that write those
outputs or whose source bytes a run manifest hashes, and in the tests that
check the runner text word for word.  Changing them changes recorded hashes,
so they stay until the affected runs are regenerated.  KNOWN_HITS lists each
one by its exact matched text and count: a new hit anywhere fails (even one
that replaces a known hit of the same kind), and so does a known hit that has
been removed without updating the list.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Patterns are built in pieces so this file does not match itself.
# A home folder on macOS or Linux, with or without a trailing slash.
POSIX_HOME = re.compile(r"(?<![\w.])/(?:" + "Users|home" + r")/([A-Za-z0-9._-]+)")
# A Windows home folder: drive letter or UNC share, either slash, JSON-escaped or not.
WINDOWS_HOME = re.compile(
    r"(?:[A-Za-z]:|\\\\[\w.-]+\\+[\w$.-]+)[\\/]+" + "Users" + r"[\\/]+([\w. -]+)", re.IGNORECASE)
# Made-up user names used on purpose in tests of HOME handling.
PLACEHOLDER_USERS = {"someone", "user", "username", "name"}
DECISION_ID = re.compile(r"\b" + "D-" + r"\d{3}\b")
PRIVATE_WORDING = "private " + "pre-registration"
PRIVATE_PATTERN = re.compile(r"private\s+" + "pre-registration", re.IGNORECASE)
TOOL_WORDING = re.compile(
    r"\b(?:" + "|".join(["Cla" + "ude", "Co" + "dex", "Chat" + "GPT", "Gem" + "ini", "Anth" + "ropic",
                         "Open" + "AI", "orches" + "trator"]) + r")\b",
    re.IGNORECASE)
# Private notes and tool folders that must never be tracked.
PRIVATE_NAMES = re.compile(
    r"(?:^|/)(?:" + "|".join([
        "CLA" + r"UDE\.md", "AGE" + r"NTS\.md", "GEM" + r"INI\.md",
        r"\.cla" + "ude/", r"\.co" + "dex/", r"\.cur" + "sor/", r"\.obsi" + "dian/",
    ]) + ")",
    re.IGNORECASE)

HOME_PATH = "local home path"
DECISION = "internal decision ID"
PRIVATE = "private-source wording"
TOOL = "AI tool wording"


def _id(number: str) -> str:
    return "D-" + number


ORCH = "orches" + "trator"

# Hits that wait for a rerun of the run they belong to: {(kind, matched text): count}.
KNOWN_HITS: dict[str, dict[tuple[str, str], int]] = {
    # Registered run outputs (sha256 in tools/reproduce_registry.json).
    "sim/runs/e-exp029-part-b/summary.json": {(PRIVATE, PRIVATE_WORDING): 1},
    "sim/runs/e-exp030-tag/summary.json": {(PRIVATE, PRIVATE_WORDING): 1, (DECISION, _id("123")): 1},
    "sim/runs/e-exp031-sweep/summary.json": {(PRIVATE, PRIVATE_WORDING): 1},
    "sim/runs/e-exp032-relay/summary.json": {(PRIVATE, PRIVATE_WORDING): 1},
    "sim/runs/e-exp033-monitor/summary.json": {(PRIVATE, PRIVATE_WORDING): 1},
    "sim/runs/e-exp034-formation/summary.json": {
        (PRIVATE, PRIVATE_WORDING): 1, (DECISION, _id("126")): 1, (DECISION, _id("123")): 1},
    "sim/runs/d7-decisions/summary.json": {(TOOL, ORCH): 1},
    # Committed outputs without a registry entry; rerunning their runners with
    # new text would change them.
    "sim/runs/e-exp035-logs/summary.json": {(PRIVATE, PRIVATE_WORDING): 1},
    "sim/runs/missions-atlas/manifest.json": {(DECISION, _id("123")): 1, (DECISION, _id("126")): 1},
    "sim/runs/missions-atlas/missions_atlas.json": {(DECISION, _id("123")): 1, (DECISION, _id("126")): 1},
    "sim/runs/d088-exp021-diagnostic/diagnostic.json": {(DECISION, _id("088")): 1},
    # Runners whose text is copied into the outputs above, or whose source
    # bytes are hashed in a run manifest (code_sha256).
    "tools/e_exp029_part_b.py": {(PRIVATE, PRIVATE_WORDING): 2, (TOOL, ORCH): 1},
    "tools/e_exp030_tag.py": {(PRIVATE, PRIVATE_WORDING): 1, (DECISION, _id("123")): 2, (DECISION, _id("120")): 1},
    "tools/e_exp031_sweep.py": {(PRIVATE, PRIVATE_WORDING): 1},
    "tools/e_exp032_relay.py": {(PRIVATE, PRIVATE_WORDING): 1},
    "tools/e_exp033_monitor.py": {(PRIVATE, PRIVATE_WORDING): 1},
    "tools/e_exp034_formation.py": {
        (PRIVATE, PRIVATE_WORDING): 1, (DECISION, _id("126")): 3, (DECISION, _id("123")): 2},
    "tools/e_exp035_logs.py": {(PRIVATE, PRIVATE_WORDING): 1},
    "tools/e_missions_atlas.py": {(DECISION, _id("123")): 1, (DECISION, _id("126")): 1},
    "tools/d_run_decisions.py": {(TOOL, ORCH): 1},
    "tools/d088_exp021_diagnostic.py": {(DECISION, _id("088")): 1},
    # Tests that check the runner text above word for word.
    "tools/tests/test_e_exp029_part_b.py": {(PRIVATE, PRIVATE_WORDING): 1},
    "tools/tests/test_e_exp030_runner.py": {(DECISION, _id("123")): 1},
    "tools/tests/test_e_exp034_runner.py": {(DECISION, _id("123")): 2, (DECISION, _id("126")): 1},
}


def local_home_paths(line: str) -> list[str]:
    names = [m.group(1) for m in POSIX_HOME.finditer(line)] + [m.group(1) for m in WINDOWS_HOME.finditer(line)]
    return [name for name in names if name.strip().lower() not in PLACEHOLDER_USERS]


def line_hits(line: str) -> list[tuple[str, str]]:
    """Every privacy hit in one line as (kind, matched text)."""
    hits = [(HOME_PATH, name) for name in local_home_paths(line)]
    hits += [(DECISION, m.group(0)) for m in DECISION_ID.finditer(line)]
    hits += [(PRIVATE, PRIVATE_WORDING) for _ in PRIVATE_PATTERN.finditer(line)]
    hits += [(TOOL, m.group(0).lower()) for m in TOOL_WORDING.finditer(line)]
    return hits


def _tracked_names() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT)
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout")
    return [name for name in out.decode("utf-8").split("\0") if name]


def _tracked_text_files():
    for name in _tracked_names():
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            yield name, data.decode("utf-8")
        except UnicodeDecodeError:
            continue


def test_home_path_patterns():
    found = [
        "/" + "Users/jane/project", "/" + "Users/jane", "/" + "home/jane/x",
        "C:" + "\\" + "Users\\jane\\x", "C:/" + "Users/jane", "c:" + "\\\\" + "Users\\\\jane",
        "\\\\server\\share\\" + "Users\\jane",
    ]
    for text in found:
        assert local_home_paths(f'"{text}"'), text
    for text in ["runs/" + "Users/x", "https://example.com/" + "home/page", "/" + "Users/someone", "/usr/bin"]:
        assert not local_home_paths(text), text


def test_line_patterns():
    assert line_hits("decision " + _id("123") + " says, see " + _id("045")) == [
        (DECISION, _id("123")), (DECISION, _id("045"))]
    assert line_hits("EXP-021 and D-12 and XD-123") == []
    assert line_hits("the " + PRIVATE_WORDING + " text") == [(PRIVATE, PRIVATE_WORDING)]
    assert line_hits("the Private  " + "pre-registration text") == [(PRIVATE, PRIVATE_WORDING)]
    assert line_hits("the pre-registration text") == []
    assert line_hits("asked " + "Cla" + "ude") == [(TOOL, "cla" + "ude")]
    assert line_hits("per the " + "Orches" + "trator decision") == [(TOOL, ORCH)]
    assert line_hits("2 s burn-in before scoring") == []
    assert line_hits('"/' + 'Users/jane/x"') == [(HOME_PATH, "jane")]


def test_private_name_pattern():
    for name in ["CLA" + "UDE.md", "docs/AGE" + "NTS.md", ".cla" + "ude/settings.json",
                 "x/.obsi" + "dian/app.json", ".co" + "dex/config.toml"]:
        assert PRIVATE_NAMES.search(name), name
    for name in ["README.md", "docs/agents_overview.md", "sim/claudette.py"]:
        assert not PRIVATE_NAMES.search(name), name


def test_no_private_notes_or_tool_folders_are_tracked():
    bad = [name for name in _tracked_names() if PRIVATE_NAMES.search(name)]
    assert not bad, bad


def test_tracked_files_have_no_private_text_beyond_known_hits():
    found: dict[str, Counter] = {}
    where: dict[str, list[str]] = {}
    for name, text in _tracked_text_files():
        for number, line in enumerate(text.splitlines(), start=1):
            for kind, match in line_hits(line):
                found.setdefault(name, Counter())[(kind, match)] += 1
                where.setdefault(name, []).append(f"{name}:{number}: {kind} ({match})")
    changed = sorted(name for name in set(found) | set(KNOWN_HITS)
                     if dict(found.get(name, {})) != KNOWN_HITS.get(name, {}))
    report = []
    for name in changed:
        if name not in found:
            report.append(f"{name}: KNOWN_HITS entry with no hits left; remove it")
        else:
            report.append(f"{name}: expected {KNOWN_HITS.get(name, {})}")
            report.extend(where[name])
    assert not changed, "\n".join(report[:80])
