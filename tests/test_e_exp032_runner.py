"""Tests for the EXP-032 runner."""
from __future__ import annotations

import json

from tools.e_exp032_relay import (PREDICTION_CORRECTION_3, PREDICTION_SOURCE, PREDICTION_TEXT,
                                  build_jobs, physics_matches_point_mass, run_experiment)
from tests.reference_e.hand_checks import max_range_dual_slope_m


def test_grid_builder_matches_preregistration():
    # 6 distances x 6 fractions x 50 point-mass seeds + 2 x 2 x 20 physics seeds = 1880.
    jobs, seeds = build_jobs()
    assert len(seeds) == 50
    assert len(jobs) == 1880
    assert jobs[0]["labels"] == {"adapter": "point_mass", "distance_m": 300.0, "relay_fraction": "none"}
    assert any(j["labels"] == {"adapter": "point_mass", "distance_m": 1300.0, "relay_fraction": 0.75} for j in jobs)
    assert any(j["labels"] == {"adapter": "physics", "distance_m": 900.0, "relay_fraction": 0.5} for j in jobs)


def test_pilot_is_small_and_hand_check_is_reference():
    jobs, seeds = build_jobs(pilot=True)
    assert len(seeds) == 3
    assert len(jobs) == 14  # 4 point-mass cells x 3 + 1 physics cell x 2.
    breakpoint_m = 4.0 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)
    reference = max_range_dual_slope_m(13.5, -98.0, 40.2, 2.0, breakpoint_m, 4.0, 0.0)
    assert 517.0 < reference < 519.0


def test_tiny_run_writes_byte_stable_outputs(tmp_path):
    first = run_experiment(tmp_path / "one", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    second = run_experiment(tmp_path / "two", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    assert first["missing_metrics"]
    for name in ("relay.csv", "summary.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    summary = json.loads((tmp_path / "one" / "summary.json").read_text(encoding="utf-8"))
    assert summary["hand_check"]["reference_median_signal_range_m"] > 500.0
    assert (tmp_path / "one" / "delivery_vs_distance.png").exists()
    assert (tmp_path / "one" / "latency_vs_distance.png").exists()
    assert summary["prediction"] == PREDICTION_TEXT
    assert "prediction:" not in PREDICTION_TEXT[:40]
    assert summary["prediction_source"] == PREDICTION_SOURCE
    assert summary["prediction_correction_3"] == PREDICTION_CORRECTION_3
    assert summary["physics_geometry_note"].startswith("Physics rows equal point-mass rows by construction")
    physics_cells = [cell for key, cell in summary["cells"].items() if key.startswith("physics|")]
    assert physics_cells
    assert all(isinstance(cell["physics_matches_point_mass"], bool) for cell in physics_cells)
    point_cells = [cell for key, cell in summary["cells"].items() if key.startswith("point_mass|")]
    assert all("physics_matches_point_mass" not in cell for cell in point_cells)
    # Called from code: no command line is recorded, but the replay command still names --output.
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["argv"] == []
    assert "--output" in manifest["replay_command"]


def test_physics_match_flag_compares_every_metric():
    point = {"adapter": "point_mass", "distance_m": 900.0, "relay_fraction": 0.5, "seed": 7,
             "delivery_ratio": 0.9, "median_latency_s": None, "p95_latency_s": float("nan")}
    same = dict(point, adapter="physics")
    assert physics_matches_point_mass([same], [point, same])
    different = dict(same, delivery_ratio=0.8)
    assert not physics_matches_point_mass([different], [point, different])
    other_seed = dict(same, seed=8)
    assert not physics_matches_point_mass([other_seed], [point, other_seed])
