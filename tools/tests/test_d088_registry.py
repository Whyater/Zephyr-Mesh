"""The EXP-021 failed-to-fly diagnostic is registered for byte reproduction.

tools/reproduce_all.py reruns every registry entry and compares sha256 hashes.
These checks make sure the diagnostic has an entry, that the entry runs the
diagnostic tool, and that the hash it pins is the hash of the committed file,
so a rerun that matches the registry also matches what is in sim/runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = "sim/runs/d088-exp021-diagnostic"


def _entry() -> dict:
    entries = json.loads((ROOT / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    matches = [e for e in entries if e["run_dir"] == RUN_DIR]
    assert len(matches) == 1, f"expected one registry entry for {RUN_DIR}, found {len(matches)}"
    return matches[0]


def test_diagnostic_has_one_registry_entry_that_runs_the_diagnostic_tool() -> None:
    entry = _entry()
    assert "tools/d088_exp021_diagnostic.py" in entry["command"]
    assert "--output-dir {out}" in entry["command"]
    assert (ROOT / "tools" / "d088_exp021_diagnostic.py").is_file()


def test_registered_hash_matches_the_committed_diagnostic() -> None:
    outputs = _entry()["outputs"]
    assert set(outputs) == {"diagnostic.json"}
    committed = (ROOT / RUN_DIR / "diagnostic.json").read_bytes()
    assert hashlib.sha256(committed).hexdigest() == outputs["diagnostic.json"]
