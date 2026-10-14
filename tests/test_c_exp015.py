"""Fast checks for the EXP-015 network runner.

The direct-range assertions are independent hand cases: they check the
stress-corner channel at one metre and at an obviously unusable distance.
The aggregation check uses a small hand-made sample with known medians and a
pooled delivery count.  The final check exercises the complete pilot twice and
compares output bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.radio.params import PARAMS, channel_preset
from tools.c_common import VOLATILE_MANIFEST_FIELDS
from tools import c_exp015_network as runner


def _stress_cfg():
    wavelength = 299_792_458.0 / 2.437e9
    return channel_preset(
        "outdoor_open",
        shadowing_sigma_db=PARAMS["sigma_outdoor_open_db"].high,
        breakpoint_m=4 * 1.5 * 1.5 / wavelength,
        exponent_far=4.0,
        orientation_loss_db=PARAMS["orientation_loss_db"].high,
    )


def test_direct_range_hand_cases():
    cfg = _stress_cfg()
    assert bool(runner.direct_link_in_range(1.0, cfg))
    assert not bool(runner.direct_link_in_range(5_000.0, cfg))


def test_cell_aggregation_hand_rows():
    rows = [
        {"topology": "star", "delivery_ratio": 0.50, "n_messages": 10,
         "n_delivered": 5, "latency_median_s": 0.10, "latency_p95_s": 0.20,
         "mean_hops": 1.0, "transmissions_per_message": 2.0, "busy_fraction": 0.10,
         "direct_range_fraction": 0.2},
        {"topology": "star", "delivery_ratio": 1.00, "n_messages": 30,
         "n_delivered": 30, "latency_median_s": 0.20, "latency_p95_s": 0.30,
         "mean_hops": 1.0, "transmissions_per_message": 3.0, "busy_fraction": 0.20,
         "direct_range_fraction": 0.4},
    ]
    out = runner.aggregate_cell(rows)
    assert out["topology"] == "star"
    assert out["delivery_pooled"] == pytest.approx(35 / 40)
    assert out["delivery_median"] == pytest.approx(0.75)
    assert out["latency_median_s"] == pytest.approx(0.15)
    assert out["transmissions_per_message"] == pytest.approx(2.5)


def test_answer_fields_use_paired_seed_differences():
    rows = []
    deliveries = {
        "star": [0.50, 0.50, 0.50],
        "relay": [0.70, 0.90, 0.80],
        "flood": [0.60, 0.70, 0.90],
        "gossip": [0.55, 0.65, 0.75],
        "broadcast": [0.40, 0.50, 0.60],
    }
    for topology, values in deliveries.items():
        for seed, ratio in enumerate(values, 1):
            rows.append({
                "area": "gym", "n": 4, "rate": 10.0, "topology": topology,
                "seed": seed, "delivery_ratio": ratio, "n_messages": 10,
                "n_delivered": int(ratio * 10), "latency_median_s": 0.1,
                "latency_p95_s": 0.2, "mean_hops": 1.0,
                "transmissions_per_message": 2.0, "busy_fraction": 0.1,
                "direct_range_fraction": 1.0,
            })
    cells = runner.aggregate(rows)
    answer = runner.answer_fields(rows, cells)
    assert len(answer) == 1
    result = answer[0]
    assert result["highest_delivery_topology"] == "relay"
    assert result["relay_gain"]["median"] == pytest.approx(0.30)
    assert result["flood_gain"]["median"] == pytest.approx(0.20)


def test_pilot_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("MPL_IGNORE_SYSTEM_FONTS", "1")
    first, second = tmp_path / "one", tmp_path / "two"
    argv = ["--output", str(first), "--pilot", "--processes", "1"]
    runner.main(argv)
    runner.main(["--output", str(second), "--pilot", "--processes", "1"])
    a = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    b = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert a == b
    for rel in a:
        if rel.name != "manifest.json":
            assert (first / rel).read_bytes() == (second / rel).read_bytes()
            continue
        first_manifest = json.loads((first / rel).read_text())
        second_manifest = json.loads((second / rel).read_text())
        for manifest in (first_manifest, second_manifest):
            for field in (*VOLATILE_MANIFEST_FIELDS, "argv"):
                manifest.pop(field, None)
        assert first_manifest == second_manifest
