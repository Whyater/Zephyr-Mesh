import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.coop.net import LinkSpec
from sim.coop.observers import CoopCell, run_seed
from tools.d_make_tolerance_figure import make_figure
from tools.d_run_tolerance import _cell, main
from sim.coop.boundary import tolerated


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direct_cell(row):
    n = int(row["n"])
    positions = {"gs": (0.0, 0.0, 1.0)}
    for index in range(n):
        angle = 2.0 * np.pi * index / n
        positions[f"o{index + 1}"] = (2.0 + 3.0 * float(np.cos(angle)),
                                       3.0 * float(np.sin(angle)), 1.5)
    if row["link"] == "abstract":
        link = LinkSpec("abstract", float(row["delay"]), float(row["loss"]),
                        loss_model=row["loss_model"])
    else:
        link = LinkSpec("radio", float(row["delay"]), float(row["loss"]),
                        loss_model=row["loss_model"], positions=positions)
    return CoopCell(target="ct", target_params=(), n_observers=n, rate_hz=10.0,
                    noise_std_m=0.05, r_std_m=None, link=link, phase="sync", ci="fast")


@pytest.fixture(scope="module")
def pilot_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("tolerance-pilot")
    one, two = root / "one", root / "two"
    main(["--pilot", "--processes", "2", "--output", str(one)])
    main(["--pilot", "--processes", "2", "--output", str(two)])
    return one, two


def _stable_manifest(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    volatile = set(manifest["volatile_fields"])
    return {key: value for key, value in manifest.items() if key not in volatile}


def test_fresh_pilots_are_byte_identical(pilot_outputs):
    one, two = pilot_outputs
    assert _stable_manifest(one) == _stable_manifest(two)
    m1 = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    for name in m1["outputs"]:
        assert _digest(one / name) == _digest(two / name)
    assert (one / "tolerance_vs_n.png").stat().st_size > 10_000


def test_checkpoint_resume_reproduces_fresh_run(tmp_path):
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    checkpoint = tmp_path / "checkpoint"
    main(["--pilot", "--processes", "2", "--checkpoint", str(checkpoint),
          "--output", str(first)])
    files = sorted(checkpoint.glob("*.json"))
    for path in files[::2]:
        path.unlink()
    main(["--pilot", "--processes", "2", "--checkpoint", str(checkpoint),
          "--output", str(resumed)])
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest["outputs"]:
        assert _digest(first / name) == _digest(resumed / name)


def test_cell_keeps_atlas_measurement_covariance_and_registered_arms():
    cell = _cell({"link": "abstract", "target": "ct_r1", "loss_model": "independent",
                  "loss": 0.0, "n": 1, "delay": 0.0})
    assert cell.r_std_m is None
    assert cell.phase == "sync"
    assert _cell({"link": "abstract", "target": "ct_r1", "loss_model": "independent",
                  "loss": 0.0, "n": 2, "delay": 0.0}).r_std_m is None


def test_n1_single_reproduces_atlas_recovery_seed():
    cell = _cell({"link": "abstract", "target": "ct_r1", "loss_model": "independent",
                  "loss": 0.0, "n": 1, "delay": 0.0})
    result = run_seed(cell, 396089605, arms=("single",))["single"]
    with gzip.open("runs/s6-atlas/per_seed.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        expected = next(row for row in csv.DictReader(handle)
                        if row["arm"] == "recovery" and row["seed"] == "396089605"
                        and row["cell_key"] == "ct/rate10/d0/j0/l0-independent/gaussian0.05")
    assert result.p95_error_m == float(expected["p95_error_m"])


def test_written_row_matches_direct_simulation_and_failure_rule(pilot_outputs):
    output, _ = pilot_outputs
    with gzip.open(output / "per_seed.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    direct = run_seed(_direct_cell(row), int(row["seed"]), arms=(row["arm"],))[row["arm"]]
    assert float(row["p95"]) == direct.p95_error_m
    assert float(row["max"]) == direct.max_error_m
    assert float(row["rmse"]) == direct.rmse_m
    expected_failed = direct.p95_error_m > 0.25 or direct.max_error_m > 0.50
    assert row["failed"] == ("true" if expected_failed else "false")


def test_boundaries_rebuilt_from_per_seed_rows(pilot_outputs):
    output, _ = pilot_outputs
    selected = []
    with gzip.open(output / "per_seed.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["link"], row["target"], row["loss_model"], row["loss"], row["arm"], row["n"]) == \
                    ("abstract", "ct_r1", "independent", "0.5", "central", "2"):
                selected.append(row)
    delays = sorted({float(row["delay"]) for row in selected})
    counts = [sum(row["failed"] == "true" and float(row["delay"]) == delay for row in selected)
              for delay in delays]
    expected = tolerated(delays, counts, 4)
    boundaries = json.loads((output / "boundaries.json").read_text(encoding="utf-8"))
    actual = boundaries["abstract|ct_r1|independent:0.5|central|2"]
    assert actual["value"] == expected["value"]
    assert actual["fails_at_zero"] == expected["fails_at_zero"]
    assert actual["censored"] == expected["censored"]
    assert actual["fail_counts"] == counts


def test_gap_statistics_are_independently_recomputed():
    job = {"link": "abstract", "target": "ct_r1", "loss_model": "independent",
           "loss": 0.5, "n": 2, "delay": 0.0}
    cell = _cell(job)
    outcome = run_seed(cell, 396089605, arms=("central",))["central"]
    table = outcome.extras["delivery_table"]
    routes = [table[("o1", "gs")], table[("o2", "gs")]]
    statuses = []
    for index in range(min(len(routes[0]), len(routes[1]))):
        entries = (routes[0][index], routes[1][index])
        statuses.append(all(item.receive_time_s is None and item.loss_reason == "link"
                            for item in entries))
    success = [index for index, lost in enumerate(statuses) if not lost]
    own_mean = float(np.mean(np.diff(success))) / cell.rate_hz
    valid = [not any(route[index].loss_reason == "late_end" for route in routes)
             for index in range(len(statuses))]
    expected_fraction = sum(lost and valid[index] for index, lost in enumerate(statuses)) / sum(valid)
    assert abs(own_mean - 0.13333333333333333) < 0.03
    assert abs(expected_fraction - 0.25) < 0.12


def test_figure_changes_when_boundary_data_change(pilot_outputs, tmp_path):
    output, _ = pilot_outputs
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    make_figure(output / "boundaries.json", first, pilot=True)
    data = json.loads((output / "boundaries.json").read_text(encoding="utf-8"))
    key = "abstract|ct_r1|burst:0.3|central|2"
    data[key]["value"] = 0.123456
    data[key]["censored"] = None
    data[key]["fails_at_zero"] = False
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    make_figure(changed, second, pilot=True)
    assert first.stat().st_size > 10_000
    assert _digest(first) != _digest(second)
