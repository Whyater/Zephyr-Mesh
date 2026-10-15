"""Checks for the shared results fixture."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MACOS = ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/Resources/results_fixture.json"
DESKTOP = ROOT / "desktop/results_fixture.json"


def test_regeneration_and_checked_in_parity(tmp_path):
    result = subprocess.run([sys.executable, "tools/build_results_fixture.py", "--out", str(tmp_path)], cwd=ROOT)
    assert result.returncode == 0
    generated = (tmp_path / "results_fixture.json").read_bytes()
    assert generated == MACOS.read_bytes()
    assert generated == DESKTOP.read_bytes()


def test_check_mode_is_clean():
    result = subprocess.run([sys.executable, "tools/build_results_fixture.py", "--check"], cwd=ROOT)
    assert result.returncode == 0


def test_findings_and_central_counts():
    fixture = json.loads(MACOS.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = (ROOT / "docs/RESULTS.md").read_text(encoding="utf-8")
    for finding in fixture["s6"]["findings"]:
        assert finding["text"] in readme
    assert fixture["atlas"]["findings"][0]["text"] in results
    central = next(s for s in fixture["atlas"]["strata"] if s["id"] == "central")
    assert central["counts"]["cells"] == 63
    assert central["counts"]["tie"] == 39
    assert central["counts"]["winner"] == 16
    assert central["counts"]["none"] == 8


def test_source_coverage_and_shape():
    fixture = json.loads(MACOS.read_text(encoding="utf-8"))
    atlas = json.loads((ROOT / "runs/s6-atlas/atlas.json").read_text(encoding="utf-8"))
    assert len(fixture["atlas"]["entries"]) == len(atlas["entries"]) == 1883
    assert len({(e["stratum"], e["variant"], e["loss_condition"], e["arm"]) for e in fixture["atlas"]["entries"]}) == 1883
    assert all(len(series["cell_ids"]) == 7 for series in fixture["s6"]["series"])


def test_json_is_finite_and_small():
    raw = MACOS.read_bytes()
    json.loads(raw)
    text = raw.decode("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    print(f"results_fixture.json size: {len(raw)} bytes")
    assert len(raw) < 3_000_000  # D-134: verbatim atlas differences
