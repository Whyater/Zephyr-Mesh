from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

from sim.coop.boundary import tolerated


def test_boundary_hand_cases() -> None:
    assert tolerated([0.0, 0.1, 0.2], [0, 2, 4], 4)["value"] == 0.0
    result = tolerated([0.0, 0.1], [4, 4], 4)
    assert result["fails_at_zero"] and result["value"] is None
    assert tolerated([0.0, 0.1], [0, 0], 4)["censored"] == ">0.1"


@pytest.mark.parametrize(
    ("delays", "counts", "expected"),
    [
        ([0.0, 0.1], [19, 19], {"value": None, "censored": ">0.1", "fails_at_zero": False,
                                "interval_s": ["<0", ">0.1"]}),
        ([0.0, 0.1], [25, 25], {"value": None, "censored": None, "fails_at_zero": True,
                                "interval_s": ["<0", ">0.1"]}),
    ],
)
def test_boundary_wilson_switches(delays, counts, expected) -> None:
    assert tolerated(delays, counts, 50) == expected


def test_boundary_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        tolerated([], [], 20)
    with pytest.raises(ValueError):
        tolerated([0.1, 0.0], [0, 0], 20)
    with pytest.raises(ValueError):
        tolerated([0.0], [21], 20)
    with pytest.raises(ValueError):
        tolerated([0.0], [0], 0)


def test_matches_three_hundred_real_atlas_series() -> None:
    root = Path("runs/s6-atlas")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    atlas = json.loads((root / "atlas.json").read_text(encoding="utf-8"))
    n_seeds = len(summary["seeds"])
    cell_labels = {cell["key"]: (cell["stratum"], cell["variant"])
                   for cell in summary["cells"]}
    grouped = defaultdict(list)
    with (root / "cells.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stratum, variant = cell_labels[row["key"]]
            loss = f"{row['loss_model']}:{float(row['loss_probability']):g}"
            count = int(round(float(row["fail_rate"]) * n_seeds))
            grouped[(stratum, variant, loss, row["arm"])].append(
                (float(row["delay_s"]), count)
            )

    # The atlas has one entry per series. Sampling the first 300 keeps this
    # regression check quick while exercising output states in the real run.
    for entry in atlas["entries"][:300]:
        key = (entry["stratum"], entry["variant"], entry["loss_condition"], entry["arm"])
        series = sorted(grouped[key])
        delays, counts = zip(*series)
        got = tolerated(delays, counts, n_seeds)
        assert got == entry["max_tolerated_delay_s"], key
