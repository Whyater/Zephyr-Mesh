"""Checks for the shared results fixture."""
from __future__ import annotations

import json
from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_results_fixture import build_atlas, build_s6

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
    # size target: under 3 MB
    assert len(raw) < 3_000_000


def _sources():
    summary_path = ROOT / "runs/s6-recovery/summary.json"
    atlas_path = ROOT / "runs/s6-atlas/atlas.json"
    return (
        json.loads(summary_path.read_text(encoding="utf-8")),
        summary_path.read_bytes(),
        json.loads(atlas_path.read_text(encoding="utf-8")),
        atlas_path.read_bytes(),
    )


def _atlas_args(atlas):
    manifest_path = ROOT / "runs/s6-atlas/manifest.json"
    return atlas, b"{}", json.loads(manifest_path.read_text(encoding="utf-8")), "- The map: most cells are ties"


def _s6_args(summary):
    manifest_path = ROOT / "runs/s6-recovery/manifest.json"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return summary, b"{}", json.loads(manifest_path.read_text(encoding="utf-8")), readme


def test_builders_reject_source_shape_drift():
    summary, _, atlas, _ = _sources()

    mutated = deepcopy(atlas)
    mutated["strata"]["extra"] = {}
    with pytest.raises(ValueError):
        build_atlas(*_atlas_args(mutated))

    mutated = deepcopy(atlas)
    mutated["entries"][0]["loss_model"] = "unknown"
    with pytest.raises(ValueError):
        build_atlas(*_atlas_args(mutated))

    mutated = deepcopy(atlas)
    mutated["entries"][0]["variant"] = "new_variant"
    with pytest.raises(ValueError):
        build_atlas(*_atlas_args(mutated))

    mutated = deepcopy(summary)
    selected = [cell for cell in mutated["cells"] if cell["slice"] == "burst"]
    mutated["cells"].append(deepcopy(selected[0]))
    with pytest.raises(ValueError):
        build_s6(*_s6_args(mutated))

    mutated = deepcopy(summary)
    mutated["cells"] = [
        cell for cell in mutated["cells"]
        if not (cell["slice"] == "main" and cell["loss_model"] == "independent" and cell["loss_probability"] == 0.5)
    ]
    with pytest.raises(ValueError):
        build_s6(*_s6_args(mutated))

    mutated = deepcopy(atlas)
    next(entry for entry in mutated["entries"] if entry["arm"] != "naive")["improvement_vs_naive"] = [
        {"delay_s": 0, "median_p95_diff_m": 0, "interval_m": [0, 0]}
    ] * 6
    with pytest.raises(ValueError):
        build_atlas(*_atlas_args(mutated))
