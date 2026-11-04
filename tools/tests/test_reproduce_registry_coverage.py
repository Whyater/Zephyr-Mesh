"""The run registry must not drift away from what is committed in sim/runs/.

tools/reproduce_registry.json lists every registered run: the folder, the command
that regenerates it, and the sha256 of each output file. tools/reproduce_all.py
reruns those commands, but that takes hours, so nothing cheap notices when the
registry and the committed folders stop agreeing. These checks read files only
(no simulation runs) and finish in a few seconds:

1. Every listed output file exists, is committed, and still has the recorded
   hash. Every script a command runs (and every repo path it reads) exists.
2. Every committed folder under sim/runs/ with result files has a registry entry,
   or is in UNREGISTERED_RUN_DIRS below with the reason it has none.
3. No two entries claim the same output file.
4. Every recorded hash is a lowercase 64-character sha256 hex digest, and not the
   digest of an empty file.

Each rule is a small function that returns a list of problems, so the last test
can plant problems in a scratch registry and show every rule catches them.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "reproduce_registry.json"
RUNS = "sim/runs"

# Committed folders under sim/runs/ that have no registry entry, with the reason.
# The coverage test fails if a folder is missing from both the registry and this
# list, and also if a folder listed here no longer exists or gets an entry, so
# this list cannot go stale. Registering a folder means deleting its line here.
UNREGISTERED_RUN_DIRS = {
    # found 2026-11-04, needs owner review: radio-model burst table from
    # tools/burst_bridge_radio_table.py; never registered.
    "burst-bridge-radio": "no entry yet (needs owner review)",
    # found 2026-11-04, needs owner review: EXP-025 ellipsoid diagnosis outputs; no
    # generating command is recorded anywhere in the repo.
    "d4-ellipsoid-diagnosis": "no entry yet (needs owner review)",
    # found 2026-11-04, needs owner review: EXP-035 mission logs from tools/e_exp035_logs.py.
    # The logs carry the git revision they were made with, so they cannot match
    # byte for byte from a different commit, which may be why it has no entry.
    "e-exp035-logs": "no entry yet (needs owner review)",
    # found 2026-11-04, needs owner review: summary built from other run folders by
    # tools/e_missions_atlas.py; never registered.
    "missions-atlas": "no entry yet (needs owner review)",
    # found 2026-11-04, needs owner review: EXP-038 (a proposal) from tools/retry_fades.py;
    # never registered.
    "retry-fades-exp038": "no entry yet (needs owner review)",
    # found 2026-11-04, needs owner review: S6 interaction analysis from tools/s6_interaction.py;
    # never registered.
    "s6-interaction": "no entry yet (needs owner review)",
}

# Files that do not count as results when deciding whether a folder needs an entry.
_NOT_RESULTS = {"README.md", ".gitkeep"}
_SHA256_LOWER = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_entries(path: Path = REGISTRY) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "registry must be a JSON array"
    return data


def committed_run_files(root: Path = ROOT) -> set[str] | None:
    """Repo-relative paths of committed files under sim/runs/, or None without git.

    git is used so stray local output folders do not count as committed.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", RUNS],
            cwd=root, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return {name for name in proc.stdout.decode("utf-8").split("\0") if name}


def run_files_on_disk(root: Path = ROOT) -> set[str]:
    base = root / RUNS
    return {p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file()}


def _run_files(root: Path = ROOT) -> set[str]:
    committed = committed_run_files(root)
    return committed if committed is not None else run_files_on_disk(root)


def _output_path(entry: dict, name: str) -> str:
    return (PurePosixPath(entry["run_dir"]) / name).as_posix()


# ---------------------------------------------------------------------------
# Rules (each returns a list of problems; empty means the rule holds)
# ---------------------------------------------------------------------------

def output_problems(entries: list[dict], root: Path, committed: set[str] | None) -> list[str]:
    """Rule 1a: listed outputs exist, are committed, and match the recorded hash."""
    problems = []
    for entry in entries:
        for name, digest in entry["outputs"].items():
            rel = _output_path(entry, name)
            path = root / rel
            if not path.is_file():
                problems.append(f"{rel}: listed output does not exist")
                continue
            if committed is not None and rel not in committed:
                problems.append(f"{rel}: listed output is not committed")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != str(digest).lower():
                problems.append(f"{rel}: committed bytes hash to {actual}, registry says {digest}")
    return problems


def command_problems(entries: list[dict], root: Path) -> list[str]:
    """Rule 1b: every command runs at least one script, and every repo path it names exists."""
    problems = []
    for entry in entries:
        tokens = shlex.split(entry["command"])
        scripts = [t for t in tokens if t.endswith(".py")]
        if not scripts:
            problems.append(f"{entry['run_dir']}: command runs no .py script")
        for token in tokens:
            # Paths built from {out} or {python} are made at run time; skip them,
            # along with options (--x), env assignments (A=b) and shell words.
            if "{" in token or token.startswith("-") or "=" in token:
                continue
            if token.endswith(".py") or "/" in token:
                if not (root / token).exists():
                    problems.append(f"{entry['run_dir']}: command names {token}, which does not exist")
    return problems


def coverage_problems(
    entries: list[dict], run_files: set[str], allowlist: dict[str, str]
) -> list[str]:
    """Rule 2: every committed result folder is registered or allowlisted, and the allowlist is current."""
    problems = []
    registered = set()
    for entry in entries:
        parts = PurePosixPath(entry["run_dir"]).parts
        if len(parts) != 3 or "/".join(parts[:2]) != RUNS:
            problems.append(f"{entry['run_dir']}: run_dir must be {RUNS}/<name>")
            continue
        registered.add(parts[2])
    result_dirs = set()
    for rel in run_files:
        parts = PurePosixPath(rel).parts
        # Files directly in sim/runs/ (a README, say) belong to no run folder.
        if len(parts) >= 4 and parts[-1] not in _NOT_RESULTS:
            result_dirs.add(parts[2])
    for name in sorted(result_dirs - registered - set(allowlist)):
        problems.append(f"{RUNS}/{name}: committed results with no registry entry and not in the allowlist")
    for name in sorted(set(allowlist) & registered):
        problems.append(f"{RUNS}/{name}: has a registry entry now, so delete it from the allowlist")
    for name in sorted(set(allowlist) - result_dirs):
        problems.append(f"{RUNS}/{name}: in the allowlist but holds no committed results")
    return problems


def duplicate_problems(entries: list[dict]) -> list[str]:
    """Rule 3: no two entries claim the same output (or the same run folder)."""
    problems = []
    seen_dirs: dict[str, int] = {}
    seen_outputs: dict[str, int] = {}
    for index, entry in enumerate(entries):
        # Case-folded because the macOS and Windows file systems ignore case.
        run_dir = PurePosixPath(entry["run_dir"]).as_posix().casefold()
        if run_dir in seen_dirs:
            problems.append(f"entries {seen_dirs[run_dir]} and {index} share run_dir {entry['run_dir']}")
        seen_dirs.setdefault(run_dir, index)
        for name in entry["outputs"]:
            key = _output_path(entry, name).casefold()
            if key in seen_outputs and seen_outputs[key] != index:
                problems.append(
                    f"entries {seen_outputs[key]} and {index} both claim {_output_path(entry, name)}"
                )
            seen_outputs.setdefault(key, index)
    return problems


def hash_problems(entries: list[dict]) -> list[str]:
    """Rule 4: hashes are lowercase sha256 hex, as sha256_file in reproduce_all.py writes them."""
    problems = []
    for entry in entries:
        for name, digest in entry["outputs"].items():
            rel = _output_path(entry, name)
            if not isinstance(digest, str) or not _SHA256_LOWER.fullmatch(digest):
                problems.append(f"{rel}: hash {digest!r} is not 64 lowercase hex characters")
            elif digest == _EMPTY_SHA256:
                problems.append(f"{rel}: hash is the sha256 of an empty file")
    return problems


# ---------------------------------------------------------------------------
# Tests on the real registry
# ---------------------------------------------------------------------------

def test_listed_outputs_exist_are_committed_and_match_their_hashes() -> None:
    assert output_problems(load_entries(), ROOT, committed_run_files()) == []


def test_every_command_names_scripts_and_paths_that_exist() -> None:
    assert command_problems(load_entries(), ROOT) == []


def test_every_committed_run_folder_is_registered_or_explained() -> None:
    assert all(reason.strip() for reason in UNREGISTERED_RUN_DIRS.values())
    assert coverage_problems(load_entries(), _run_files(), UNREGISTERED_RUN_DIRS) == []


def test_no_two_entries_claim_the_same_output() -> None:
    assert duplicate_problems(load_entries()) == []


def test_every_recorded_hash_is_well_formed() -> None:
    assert hash_problems(load_entries()) == []


# ---------------------------------------------------------------------------
# The rules catch planted problems (a scratch copy, never the real registry)
# ---------------------------------------------------------------------------

def test_each_rule_catches_a_planted_problem(tmp_path: Path) -> None:
    good = b"a,b\n1,2\n"
    good_hash = hashlib.sha256(good).hexdigest()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "make.py").write_text("print('x')\n", encoding="utf-8")
    for name in ("alpha", "beta", "stray"):
        folder = tmp_path / RUNS / name
        folder.mkdir(parents=True)
        (folder / "out.csv").write_bytes(good)
    clean = [
        {"run_dir": f"{RUNS}/alpha", "command": "{python} tools/make.py --output {out}",
         "outputs": {"out.csv": good_hash}},
        {"run_dir": f"{RUNS}/beta", "command": "{python} tools/make.py --output {out}",
         "outputs": {"out.csv": good_hash}},
    ]
    scratch = tmp_path / "registry.json"
    scratch.write_text(json.dumps(clean), encoding="utf-8")
    files = run_files_on_disk(tmp_path)
    entries = load_entries(scratch)
    assert output_problems(entries, tmp_path, None) == []
    assert command_problems(entries, tmp_path) == []
    assert coverage_problems(entries, files, {"stray": "test"}) == []
    assert duplicate_problems(entries) == []
    assert hash_problems(entries) == []

    # Plant one fake entry that breaks every rule at once.
    fake = {
        "run_dir": f"{RUNS}/alpha",                          # rule 3: same folder as entry 0
        "command": "{python} tools/missing.py --output {out}",  # rule 1b: no such script
        "outputs": {"out.csv": good_hash.upper(),            # rule 4: upper case
                    "ghost.json": "0" * 63},                 # rule 1a: no file; rule 4: 63 chars
    }
    scratch.write_text(json.dumps(clean + [fake]), encoding="utf-8")
    entries = load_entries(scratch)
    assert any("ghost.json: listed output does not exist" in p for p in output_problems(entries, tmp_path, None))
    assert any("tools/missing.py" in p for p in command_problems(entries, tmp_path))
    assert any("both claim" in p for p in duplicate_problems(entries))
    assert len(hash_problems(entries)) == 2

    # Rule 1a: a committed file whose bytes changed after registration.
    (tmp_path / RUNS / "beta" / "out.csv").write_bytes(good + b"3,4\n")
    assert any("beta/out.csv: committed bytes" in p for p in output_problems(load_entries(scratch), tmp_path, None))
    # Rule 1a: listed output present on disk but not committed.
    assert any("not committed" in p for p in output_problems(clean, tmp_path, set()))

    # Rule 2: an unexplained folder, a stale allowlist line, an allowlisted folder that got an entry.
    assert any("stray: committed results" in p for p in coverage_problems(clean, files, {}))
    assert any("holds no committed results" in p for p in coverage_problems(clean, files, {"stray": "t", "gone": "t"}))
    assert any("has a registry entry now" in p for p in coverage_problems(clean, files, {"stray": "t", "alpha": "t"}))

    # Rule 4: the empty-file digest.
    empty = [{**clean[0], "outputs": {"out.csv": _EMPTY_SHA256}}]
    assert any("empty file" in p for p in hash_problems(empty))
