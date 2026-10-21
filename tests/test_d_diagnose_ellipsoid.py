"""Checks for the EXP-025 ellipsoid diagnosis tool (simulated)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.coop import separation
from tools.d_common import seeds_for
from tools.d_diagnose_ellipsoid import (
    VARIANTS,
    diagnose_encounter,
    instrumented,
    min_vertical_offset_for_failure,
    select_conditions,
)
from tools.d_run_separation import N_SEEDS, _spec

ROOT = Path(__file__).resolve().parents[1]
BUFFER = ROOT / "runs" / "d4-separation" / "buffer.json"
FAILING = {"kind": "crossing", "speed_mps": 1.0, "delay_s": 0.0, "loss_probability": 0.1,
           "noise_std_m": 0.05, "dead_reckoning": False, "lateral_offset_m": 0.0}


def test_vertical_bound_matches_a_brute_force_search():
    # Independent check: walk round the 0.35 m sphere and find the smallest
    # vertical offset that lands inside the 0.24 m wide, 0.60 m tall ellipsoid.
    dz = np.linspace(0.0, 0.35, 350001)
    horizontal = np.sqrt(0.35 ** 2 - dz ** 2)
    inside = (horizontal / 0.24) ** 2 + (dz / 0.60) ** 2 < 1.0
    brute = float(dz[np.argmax(inside)])
    assert inside.any()
    assert min_vertical_offset_for_failure() == pytest.approx(brute, abs=2e-6)
    assert min_vertical_offset_for_failure() == pytest.approx(0.27796, abs=1e-5)
    # A ball of radius 0.24 m or less already sits inside the ellipsoid's waist.
    assert min_vertical_offset_for_failure(center_m=0.24) == 0.0


def test_patches_are_removed_after_each_encounter():
    before = {name: getattr(separation, name) for name in
              ("_linear_segment_minimum", "safe_velocity", "_message_noise", "AvoidanceConfig")}
    spec = _spec({**FAILING, "avoidance_radius_m": 0.4})
    for variant in VARIANTS:
        with instrumented(variant, spec):
            assert separation.safe_velocity is not before["safe_velocity"]
        for name, value in before.items():
            assert getattr(separation, name) is value
    with pytest.raises(RuntimeError), instrumented("as_run", spec):
        raise RuntimeError("boom")
    assert separation.safe_velocity is before["safe_velocity"]


@pytest.mark.parametrize("seed_index", [0, 7])
def test_as_run_matches_the_unpatched_encounter(seed_index):
    seed = seeds_for("separation", N_SEEDS)[seed_index]
    spec = _spec({**FAILING, "avoidance_radius_m": 0.4})
    plain = separation.run_encounter(spec, seed)
    record = diagnose_encounter(FAILING, 0.4, seed, "as_run")
    assert record["min_center_distance_m"] == plain["min_center_distance_m"]
    assert record["min_ellipsoid_separation"] == plain["min_ellipsoid_separation"]
    # The recorded offset at the minimum reproduces the scaled minimum itself.
    scaled = math.hypot(record["at_minimum_horizontal_m"] / 0.24, record["at_minimum_dz_m"] / 0.60)
    assert scaled == pytest.approx(plain["min_ellipsoid_separation"], rel=1e-9)


def test_as_run_reproduces_the_buffer_json_ellipsoid_count():
    buffer = json.loads(BUFFER.read_text(encoding="utf-8"))
    entry = next(e for e in buffer["entries"] if all(e[k] == v for k, v in FAILING.items()))
    radius = round(0.35 + entry["required_buffer_m"], 2)
    row = next(r for r in entry["tested_radii"] if abs(r["avoidance_radius_m"] - radius) < 1e-9)
    records = [diagnose_encounter(FAILING, radius, seed, "as_run") for seed in seeds_for("separation", N_SEEDS)]
    assert sum(r["ellipsoid_violated"] for r in records) == row["ellipsoid_violation_count"] > 0
    assert sum(r["violated"] for r in records) == row["violations"] == 0


def test_every_failure_needs_the_geometric_vertical_offset():
    seeds = seeds_for("separation", N_SEEDS)[:30]
    records = [diagnose_encounter(FAILING, 0.4, seed, "as_run") for seed in seeds]
    failing = [r for r in records if r["ellipsoid_violated"]]
    assert failing
    for record in failing:
        assert abs(record["at_minimum_dz_m"]) > min_vertical_offset_for_failure()
        assert record["at_minimum_horizontal_m"] < 0.24


def test_flat_avoidance_and_noise_free_runs_stay_at_one_altitude():
    seed = seeds_for("separation", N_SEEDS)[3]
    flat = diagnose_encounter(FAILING, 0.4, seed, "flat_avoidance")
    assert flat["max_altitude_drift_m"] == 0.0
    noise_free = diagnose_encounter({**FAILING, "noise_std_m": 0.0}, 0.4, seed, "as_run")
    assert noise_free["max_altitude_drift_m"] == 0.0
    assert noise_free["rms_error_z_m"] == 0.0


def test_no_vertical_noise_keeps_the_horizontal_draws():
    seed = seeds_for("separation", N_SEEDS)[3]
    spec = _spec({**FAILING, "avoidance_radius_m": 0.4})
    original = separation._message_noise(spec, seed, 2)
    with instrumented("no_vertical_noise", spec):
        wrapped = separation._message_noise(spec, seed, 2)
    for plain, patched in zip(original, wrapped):
        a = plain.normal(0.0, 0.05, size=3)
        b = patched.normal(0.0, 0.05, size=3)
        assert a[:2].tolist() == b[:2].tolist()
        assert b[2] == 0.0


def test_sample_is_reproducible_and_covers_each_group():
    entries = json.loads(BUFFER.read_text(encoding="utf-8"))["entries"]
    first = select_conditions(entries, per_kind=2, seed=25)
    assert first == select_conditions(entries, per_kind=2, seed=25)
    groups = {item["group"] for item in first}
    assert groups == {"fails_ellipsoid", "noisy_passes_ellipsoid", "noise_free", "most_failing"}
    assert all(item["buffer_ellipsoid_count"] > 0 for item in first if item["group"] == "fails_ellipsoid")
    assert all(item["noise_std_m"] == 0.0 for item in first if item["group"] == "noise_free")
    assert sum(item["kind"] == "ring_swap" for item in first if item["group"] == "fails_ellipsoid") <= 1


def test_saved_as_run_matches_buffer_json_in_every_sampled_condition():
    # The saved rerun must reproduce buffer.json's centre and ellipsoid counts
    # at the passing radius for all 16 sampled conditions, not just one.
    entries = json.loads(BUFFER.read_text(encoding="utf-8"))["entries"]
    saved = json.loads((ROOT / "runs" / "d4-ellipsoid-diagnosis" / "as_run.json").read_text(encoding="utf-8"))
    assert len(saved["results"]) == 16
    keys = ("kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m", "dead_reckoning", "lateral_offset_m")
    for result in saved["results"]:
        condition = result["condition"]
        entry = next(e for e in entries if all(e[k] == condition[k] for k in keys))
        row = next(r for r in entry["tested_radii"]
                   if abs(r["avoidance_radius_m"] - condition["radius_m"]) < 1e-9)
        summary = result["variants"]["as_run"]["summary"]
        assert summary["seeds"] == row["seeds_run"] == 75
        assert summary["ellipsoid_violations"] == row["ellipsoid_violation_count"]
        assert summary["center_violations"] == row["violations"] == 0
