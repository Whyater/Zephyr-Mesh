#!/usr/bin/env python3
"""Turn a finished intake folder into a PROPOSED registry entry (a proposal, not a registration).

    PYTHONPATH=. .venv/bin/python tools/propose_measured_entry.py INTAKE_DIR \\
        --run-dir sim/runs/<name> \\
        --intake-command "PYTHONPATH=. {python} tools/intake_espnow.py --sender data/x/sender.csv \\
            --echoer data/x/echoer.csv --status measured ... --pilot"

INTAKE_DIR is an output folder of tools/intake_espnow.py or tools/intake_vision.py. --intake-command is
the intake command that made it, with repo-relative input paths and without --output. The intake tools
refuse to write under sim/runs, so the proposed command runs the intake into a staging folder at the
root of the clean tree and copies it into {out}. reproduce_all always runs a command inside a fresh
temporary copy of the tree and deletes that copy afterwards, so the staging folder needs no rm.
The <name> in sim/runs/<name> must match [a-z0-9][a-z0-9._-]* with no '..', and the intake command is
rebuilt from its parsed words with each word quoted, so no shell text can ride in on either.
INTAKE_DIR must sit outside every sim/runs folder: this tool never writes under sim/runs.

Before it writes anything, it checks, and refuses on the first failure:

1. INTAKE.md is there and states the status, the inputs with their sha256, and the output files with
   their sha256 (the intake's own provenance record). Every listed output still has that hash.
2. Every input INTAKE.md lists is a file the command names, with the same sha256.
3. The run_dir is a plain sim/runs/<name>. The command has {python}, no {out} or --output of its
   own, and no absolute or '..' paths.
4. The run_dir is not already in the registry, and the registry plus the new entry passes the real
   loader in tools/reproduce_all.py.
5. The command, run the way tools/reproduce_all.py runs it (a clean copy of --rev, or --source-dir),
   reproduces every listed output byte for byte and makes exactly the files in INTAKE_DIR, no more and
   no fewer. Files INTAKE.md leaves out of its table are volatile (for example INTAKE.md itself) and
   are not hashed.

It then writes INTAKE_DIR/PROPOSED_REGISTRY_ENTRY.json, whose "entry" object has exactly run_dir,
command and outputs. It never edits tools/reproduce_registry.json; registering is a reviewed step.
Inputs must be committed for the default --rev HEAD replay to find them.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import reproduce_all

PROPOSAL_NAME = "PROPOSED_REGISTRY_ENTRY.json"
FORMAT = "zephyr-proposed-registry-entry-1"
STATUSES = ("measured", "synthetic", "fixture", "unknown")
_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9a-fA-F]{64})\s*\|")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
NOTE = ("PROPOSED, not registered. To register: review it, add the outputs under run_dir, append 'entry' "
        "to tools/reproduce_registry.json, and rerun tools/reproduce_all.py --only <name>. The hashes were "
        "recorded on the platform named under 'replay'.")


class ProposalError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return reproduce_all.sha256_file(path)


# ---------------------------------------------------------------- INTAKE.md

def read_intake(intake_dir: Path) -> dict[str, Any]:
    """Read status, inputs and outputs from INTAKE.md. Refuse a folder without that record."""
    path = intake_dir / "INTAKE.md"
    if not path.is_file():
        raise ProposalError(f"no provenance: {intake_dir.name} has no INTAKE.md (not an intake output folder)")
    status: str | None = None
    section: str | None = None
    inputs: list[tuple[str, str]] = []
    outputs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Overall status:"):
            status = line.split(":", 1)[1].strip()
        elif line.startswith("## "):
            section = line[3:].strip()
        else:
            match = _ROW.match(line)
            if match and section == "Inputs":
                inputs.append((match.group(1), match.group(2).lower()))
            elif match and section == "Output files":
                outputs[match.group(1)] = match.group(2).lower()
    if status not in STATUSES:
        raise ProposalError("no provenance: INTAKE.md does not state an overall status")
    if not inputs:
        raise ProposalError("no provenance: INTAKE.md lists no inputs with a sha256")
    if not outputs:
        raise ProposalError("no provenance: INTAKE.md lists no output files with a sha256")
    for name, digest in outputs.items():
        file = intake_dir / name
        if not file.is_file():
            raise ProposalError(f"INTAKE.md lists {name}, which is not in the folder")
        if sha256_file(file) != digest:
            raise ProposalError(f"{name} changed since the intake wrote INTAKE.md")
    return {"status": status, "inputs": inputs, "outputs": outputs}


def folder_files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*")
            if p.is_file() and p.relative_to(folder).as_posix() != PROPOSAL_NAME}


# ---------------------------------------------------------------- command

def _is_unsafe(token: str) -> bool:
    for piece in [token] + ([token.split("=", 1)[1]] if "=" in token else []):
        if piece.startswith("/") or _WINDOWS_ABS.match(piece) or ".." in Path(piece).parts:
            return True
    return False


def check_run_dir(run_dir: str) -> str:
    """Return <name> for a run_dir that is exactly sim/runs/<name>; refuse anything else."""
    prefix = "sim/runs/"
    name = run_dir[len(prefix):] if run_dir.startswith(prefix) else ""
    if not _RUN_NAME.fullmatch(name) or ".." in name:
        raise ProposalError(f"run_dir must be exactly sim/runs/<name>, with <name> made of a-z, 0-9, '.', "
                            f"'_' and '-', starting with a letter or digit, and no '..': {run_dir!r}")
    return name


def check_outside_runs(intake_dir: Path) -> None:
    """Refuse an intake folder inside any sim/runs folder, so the proposal never lands there."""
    parts = [part.lower() for part in intake_dir.parts]
    if any(parts[i:i + 2] == ["sim", "runs"] for i in range(len(parts) - 1)):
        raise ProposalError(f"{intake_dir} is under sim/runs. Keep the intake output outside sim/runs until "
                            "the entry is registered; this tool never writes under sim/runs")


def build_command(intake_command: str, run_dir: str) -> str:
    """Wrap the intake command so it writes into a staging folder and copies that into {out}."""
    name = check_run_dir(run_dir)
    if "{python}" not in intake_command:
        raise ProposalError("the intake command must run Python as {python}")
    if "{out}" in intake_command:
        raise ProposalError("leave {out} out of the intake command; this tool adds it")
    try:
        tokens = shlex.split(intake_command)
    except ValueError as exc:
        raise ProposalError(f"cannot parse the intake command: {exc}") from exc
    if any(t == "--output" or t.startswith("--output=") for t in tokens):
        raise ProposalError("leave --output out of the intake command; this tool adds a staging folder")
    unsafe = [t for t in tokens if _is_unsafe(t)]
    if unsafe:
        raise ProposalError(f"the command must use repo-relative paths only, without '..': {unsafe[0]}")
    # Rebuild from the parsed words so shell operators become plain arguments. {python} stays bare:
    # reproduce_all swaps it for a quoted path.
    words = " ".join(t if t == "{python}" else shlex.quote(t) for t in tokens)
    stage = shlex.quote(f".intake-staging-{name}")
    return f"{words} --output {stage} && cp -R {stage}/. {{out}}/"


def match_inputs(intake_command: str, inputs: list[tuple[str, str]], tree: Path) -> list[dict[str, str]]:
    """Every INTAKE.md input must be a file the command names (directly or inside a named folder)."""
    candidates: list[Path] = []
    for token in shlex.split(intake_command):
        value = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
        if value and (tree / value).exists():
            candidates.append(Path(value))
    matched: list[dict[str, str]] = []
    for label, digest in inputs:
        found = None
        for cand in candidates:
            full = tree / cand
            if full.is_file() and cand.name == label:
                path = cand
            elif full.is_dir() and label.startswith(cand.name + "/"):
                path = cand.parent / label
            else:
                continue
            if (tree / path).is_file() and sha256_file(tree / path) == digest:
                found = path
                break
        if found is None:
            raise ProposalError(f"INTAKE.md lists input {label} (sha256 {digest[:12]}...), but the command "
                                "names no file with that name and hash")
        matched.append({"label": label, "path": found.as_posix(), "sha256": digest})
    return matched


# ---------------------------------------------------------------- registry

def check_registry(registry: Path, entry: dict[str, Any]) -> None:
    try:
        existing = reproduce_all.load_registry(registry)
    except reproduce_all.RegistryError as exc:
        raise ProposalError(f"the registry does not load: {exc}") from exc
    if any(e["run_dir"] == entry["run_dir"] for e in existing):
        raise ProposalError(f"{entry['run_dir']} is already in the registry; pick a new run_dir")
    with tempfile.TemporaryDirectory(prefix="propose-entry-") as tmp:
        trial = Path(tmp) / "registry.json"
        trial.write_text(json.dumps([*existing, entry], indent=2) + "\n", encoding="utf-8")
        try:
            reproduce_all.load_registry(trial)
        except reproduce_all.RegistryError as exc:
            raise ProposalError(f"the proposed entry fails the registry loader: {exc}") from exc


# ---------------------------------------------------------------- replay

def replay(entry: dict[str, Any], intake_dir: Path, *, repo_root: Path, rev: str, source_dir: Path | None,
           python: str, timeout: float) -> dict[str, Any]:
    """Run the entry the way reproduce_all does and compare its file set with the intake folder."""
    result = reproduce_all.run_entry(entry, repo_root, rev, source_dir, python, timeout, keep_temp=True)
    temp = result.pop("temp_dir", None)
    try:
        status = result["status"]
        if status == "failed":
            detail = result.get("error") or " | ".join(result.get("stderr_tail", [])[-3:])
            raise ProposalError(f"the command failed in a clean tree: {detail}")
        if status == "missing-output":
            raise ProposalError(f"the command did not produce: {', '.join(result['missing'])}")
        if status == "differs":
            names = ", ".join(sorted(result["differences"]))
            raise ProposalError(f"the command did not reproduce these outputs byte for byte: {names}")
        if status != "byte-identical":
            raise ProposalError(f"replay ended with status {status}")
        made = folder_files(Path(temp) / "tree" / entry["run_dir"])
        have = folder_files(intake_dir)
        if made != have:
            parts = []
            if have - made:
                parts.append("in the folder but not made by the command: " + ", ".join(sorted(have - made)))
            if made - have:
                parts.append("made by the command but not in the folder: " + ", ".join(sorted(made - have)))
            raise ProposalError("the command and the folder disagree on files (" + "; ".join(parts) + ")")
        return result
    finally:
        if temp:
            shutil.rmtree(temp, ignore_errors=True)


def _revision(repo_root: Path, rev: str) -> str:
    proc = subprocess.run(["git", "rev-parse", rev], cwd=repo_root, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else rev


# ---------------------------------------------------------------- driver

def propose(intake_dir: Path, *, run_dir: str, intake_command: str, registry: Path, repo_root: Path,
            rev: str = "HEAD", source_dir: Path | None = None, python: str = sys.executable,
            timeout: float = 3600.0) -> Path:
    intake_dir = intake_dir.resolve()
    if not intake_dir.is_dir():
        raise ProposalError(f"not a folder: {intake_dir}")
    check_outside_runs(intake_dir)
    check_run_dir(run_dir)
    record = read_intake(intake_dir)
    command = build_command(intake_command, run_dir)
    tree = source_dir.resolve() if source_dir is not None else repo_root
    inputs = match_inputs(intake_command, record["inputs"], tree)
    entry = {"run_dir": run_dir, "command": command, "outputs": dict(sorted(record["outputs"].items()))}
    check_registry(registry, entry)
    replay(entry, intake_dir, repo_root=repo_root, rev=rev, source_dir=source_dir, python=python,
           timeout=timeout)
    proposal = {
        "format": FORMAT,
        "proposal_only": True,
        "note": NOTE,
        "intake_status": record["status"],
        "entry": entry,
        "inputs": inputs,
        "volatile_not_hashed": sorted(folder_files(intake_dir) - set(record["outputs"])),
        "replay": {"status": "byte-identical", "platform": sys.platform,
                   "tree": "source-dir copy" if source_dir is not None else _revision(repo_root, rev)},
    }
    out = intake_dir / PROPOSAL_NAME
    out.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("intake_dir", type=Path, help="output folder of an intake tool")
    parser.add_argument("--run-dir", required=True, help="sim/runs/<name> for the proposed entry")
    parser.add_argument("--intake-command", required=True,
                        help="the intake command with {python}, repo-relative inputs, and no --output")
    parser.add_argument("--registry", type=Path, default=Path("tools/reproduce_registry.json"),
                        help="read only; checked for duplicates and loader rules")
    parser.add_argument("--rev", default="HEAD", help="git revision for the clean replay")
    parser.add_argument("--source-dir", type=Path, help="replay from a copy of this plain folder instead")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    registry = args.registry if args.registry.is_absolute() else repo_root / args.registry
    try:
        path = propose(args.intake_dir, run_dir=args.run_dir, intake_command=args.intake_command,
                       registry=registry, repo_root=repo_root, rev=args.rev, source_dir=args.source_dir,
                       python=args.python, timeout=args.timeout)
    except ProposalError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {path} (a proposal; the registry was not changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
