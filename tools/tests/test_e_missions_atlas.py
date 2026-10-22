import csv
import json
import math
from pathlib import Path

import pytest

from tools import e_missions_atlas as atlas
from tools.c_common import write_json, write_manifest
from tools.e_common import write_rows

def make_input(root: Path, key: str, rows: list[dict], *, pilot=False, csv_name=None):
    spec = {k: (folder, name, exp) for k, folder, name, exp in atlas.INPUT_SPECS}[key]
    folder_name, default_csv, _ = spec
    folder = root / folder_name
    folder.mkdir(parents=True)
    csv_name = csv_name or default_csv
    write_rows(folder / csv_name, rows)
    write_json(folder / "summary.json", {"pilot": pilot})
    write_manifest(folder, schema="synthetic", argv=[], command="synthetic", config={"pilot": pilot},
                   seeds=[], outputs=[csv_name, "summary.json"], limits=[], runtime_s=0.0,
                   n_processes=1, hashed_sources=[])
    return folder


def tag_row(delay, *, target="car", preset="clean", arm="tag", tagged=True,
            separation=0, adapter="point_mass", n_followers=1, policy_name="default", slice="main"):
    return {"slice": slice, "target_class": target, "measurement_delay_s": delay, "preset": preset,
            "arm": arm, "n_followers": n_followers, "adapter": adapter, "policy_name": policy_name,
            "tagged": tagged, "separation_violations": separation}


def formation_row(latency, *, sigma=0.02, preset="clean", adapter="point_mass", rms=0.01,
                  separation=0.8, slice="main", separation_rule="coop"):
    return {"slice": slice, "sigma_m": sigma, "fix_latency_s": latency, "preset": preset,
            "adapter": adapter, "rms_error_m": rms, "min_separation_m": separation,
            "separation_rule": separation_rule}


def run_atlas(tmp_path, *, pilot_preview=False, exp030_rows=None, exp034_rows=None):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    if exp030_rows is not None:
        make_input(inputs, "exp030", exp030_rows, pilot=pilot_preview)
    if exp034_rows is not None:
        make_input(inputs, "exp034", exp034_rows)
    output = tmp_path / "output"
    atlas.run(output, [], inputs=inputs, pilot_preview=pilot_preview)
    return json.loads((output / atlas.OUTPUT_NAME).read_text(encoding="utf-8"))


def test_bracket_tolerance_rules_and_empty_cell():
    cells = [{"value": value, "met": met} for value, met in ((0.0, True), (0.1, True), (0.2, False), (0.3, True))]
    result = atlas.bracket(cells, "x", "s")
    assert result["tolerated"] == 0.1
    assert result["bracket"] == [0.1, 0.2]
    assert result["met_again_after_first_not_met"] is True
    assert atlas.bracket([{"value": 0.0, "met": True}, {"value": 0.1, "met": True}], "x", "s")["bracket"] == [0.1, None]
    assert atlas.bracket([{"value": 0.0, "met": False}, {"value": 0.1, "met": True}], "x", "s")["tolerated"] is None

    rows = [tag_row(0.0, target="a"), tag_row(0.1, target="a"), tag_row(0.2, target="a", tagged=False),
            tag_row(0.3, target="a"), tag_row(0.0, target="b")]
    entries = atlas.tag_entries(rows)
    entry_b = next(e for e in entries if e["target_class"] == "b")
    assert entry_b["cells"][1]["missing"] is True
    assert entry_b["cells"][1]["met"] is False


def test_wilson_hand_formula_and_schema(tmp_path):
    rows = [tag_row(0.0, tagged=i < 9) for i in range(10)]
    atlas_json = run_atlas(tmp_path, exp030_rows=rows)
    condition = atlas_json["missions"]["tag"]["entries"][0]["cells"][0]["conditions"][0]
    z = 1.959963984540054
    n, k = 10, 9
    p = k / n
    denom = 1 + z * z / n
    expected_low = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    assert condition["compared"] == pytest.approx(expected_low)
    assert atlas_json["schema"] == atlas.ATLAS_SCHEMA_ID
    atlas.validate_atlas(atlas_json)


def test_input_statuses_pilot_and_manifest_hash(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    pilot_rows = [tag_row(0.0)]
    folder = make_input(inputs, "exp030", pilot_rows, pilot=True)
    out = tmp_path / "out"
    atlas.run(out, [], inputs=inputs)
    data = json.loads((out / atlas.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert data["missions"]["tag"]["status"] == atlas.PILOT_ONLY
    assert data["missions"]["tag"]["entries"] == []

    out_preview = tmp_path / "out_preview"
    atlas.run(out_preview, [], inputs=inputs, pilot_preview=True)
    preview = json.loads((out_preview / atlas.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert preview["missions"]["tag"]["entries"] == []
    assert preview["missions"]["tag"]["pilot_entries"]

    (folder / "tag.csv").write_text((folder / "tag.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    out_bad = tmp_path / "out_bad"
    atlas.run(out_bad, [], inputs=inputs)
    bad = json.loads((out_bad / atlas.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert bad["inputs"]["exp030"]["status"] == atlas.NOT_AVAILABLE
    assert "does not match the sha256" in bad["inputs"]["exp030"]["reason"]

    missing = tmp_path / "missing"
    assert atlas.load_input("exp030", missing, "tag.csv", "EXP-030")["status"] == atlas.NOT_AVAILABLE
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    (unfinished / "tag.csv").write_text("slice\nmain\n", encoding="utf-8", newline="\n")
    assert atlas.load_input("exp030", unfinished, "tag.csv", "EXP-030")["status"] == atlas.NOT_AVAILABLE


def test_output_is_byte_identical(tmp_path):
    rows = [tag_row(0.0), tag_row(0.1, tagged=False)]
    first = run_atlas(tmp_path / "one", exp030_rows=rows)
    second = run_atlas(tmp_path / "two", exp030_rows=rows)
    assert first == second
    p1 = tmp_path / "one" / "output" / atlas.OUTPUT_NAME
    p2 = tmp_path / "two" / "output" / atlas.OUTPUT_NAME
    assert p1.read_bytes() == p2.read_bytes()


def test_formation_separation_rule_and_comparison_scope(tmp_path):
    rows = [
        formation_row(0.0, separation_rule="local", slice="main"),
        formation_row(0.1, separation_rule="local", slice="main"),
        formation_row(0.0, separation_rule="coop", slice="coop_comparison", separation=0.2),
        formation_row(0.1, separation_rule="coop", slice="coop_comparison", separation=0.2),
    ]
    data = run_atlas(tmp_path, exp034_rows=rows)
    entries = data["missions"]["formation"]["entries"]
    assert len(entries) == 2
    main = next(e for e in entries if e["slice"] == "main")
    comparison = next(e for e in entries if e["slice"] == "coop_comparison")
    assert all(c["met"] for c in main["cells"])
    assert all(not c["met"] for c in comparison["cells"])
    assert comparison["scope"] == atlas.SCOPE_COMPARISON
    assert any("comparison slice" in note for note in comparison["notes"])
    assert any("registered 0.8 m" in note for note in main["notes"])
    assert data["missions"]["tag"]["notes"][0] == atlas.FOLLOWERS_NOTE
    assert "Rows with separation_rule local are labelled as the comparison rule." in atlas.FOLLOWERS_NOTE
    assert "entries labelled local" not in " ".join(data["limits"])


def test_tag_separation_rule_groups_and_labels_local():
    rows = [
        tag_row(0.0, n_followers=2, separation=0, target="multi", policy_name="default"),
        tag_row(0.0, n_followers=2, separation=1, target="multi", policy_name="default"),
    ]
    rows[0]["separation_rule"] = "coop"
    rows[1]["separation_rule"] = "local"
    entries = atlas.tag_entries(rows)
    assert len(entries) == 2
    local = next(e for e in entries if any("comparison rule" in n for n in e["notes"]))
    assert local["cells"][0]["met"] is False


def test_seed_count_note_when_zero_events_cannot_pass():
    # 20 runs, 0 events: Wilson upper bound = z^2 / (n + z^2) = 3.8415 / 23.8415 = 0.161 > 0.10.
    z = 1.959963984540054
    assert z * z / (20 + z * z) > 0.10
    cell = {"conditions": [{"statistic": "wilson_high", "met": False, "summary": {"n": 20, "count": 0}}]}
    note = atlas.seed_count_note({"cells": [cell]})
    assert note is not None and "20 runs with zero events" in note
    passing = {"conditions": [{"statistic": "wilson_high", "met": False, "summary": {"n": 50, "count": 3}}]}
    assert atlas.seed_count_note({"cells": [passing]}) is None
