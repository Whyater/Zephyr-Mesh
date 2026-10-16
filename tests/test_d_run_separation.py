"""Output and reproducibility checks for the separation-buffer runner."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import struct

import pytest
from jsonschema import Draft202012Validator

from sim.coop.separation import EncounterSpec, run_encounter
from tools.d_run_separation import run


ROOT = Path(__file__).resolve().parents[1]


def _pilot(path: Path) -> None:
    run(path, pilot=True, processes=6)


@pytest.fixture(scope="module")
def pilot_output(tmp_path_factory):
    first = tmp_path_factory.mktemp("separation-pilot") / "first"
    _pilot(first)
    return first


def test_pilot_is_byte_stable_and_figures_are_real(pilot_output, tmp_path):
    first, second = pilot_output, tmp_path / "second"
    _pilot(second)
    manifest_a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    volatile = {"runtime_s", "git_revision", "git_tree_state", "platform", "n_processes", "argv"}
    for key in volatile:
        manifest_a.pop(key, None)
        manifest_b.pop(key, None)
    assert manifest_a == manifest_b
    for name in manifest_a["outputs"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for name in ("separation_buffer.png", "separation_baseline.png"):
        content = (first / name).read_bytes()
        assert len(content) > 10_000
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", content[16:24])
        assert width > 100 and height > 100


def test_written_buffer_matches_schema_and_perfect_info_is_present(pilot_output):
    out = pilot_output
    value = json.loads((out / "buffer.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "sim/coop/buffer_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(value)) == []
    perfect = json.loads((out / "perfect_info.json").read_text(encoding="utf-8"))
    rows = perfect["rows"] if isinstance(perfect, dict) else perfect
    assert rows
    assert all({"kind", "speed_mps", "violations", "n_seeds", "zero_violations"} <= set(row) for row in rows)
    assert all(row["violations"] == 0 and row["zero_violations"] is True for row in rows)


def test_one_encounter_row_recomputes_from_simulation(pilot_output):
    out = pilot_output
    with (out / "encounters.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    row = next(item for item in rows if item.get("pass") in {"1", "pass1", "baseline"})
    spec = EncounterSpec(
        kind=row["kind"],
        speed_mps=float(row["speed_mps"]),
        delay_s=float(row["delay_s"]),
        loss_probability=float(row["loss_probability"]),
        noise_std_m=float(row["noise_std_m"]),
        dead_reckoning=row["dead_reckoning"].lower() == "true",
        avoidance_radius_m=0.35,
    )
    expected = run_encounter(spec, int(row["seed"]))
    assert float(row["min_center_distance_m"]) == expected["min_center_distance_m"]
    assert int(float(row["violated"])) == int(expected["violated"])

