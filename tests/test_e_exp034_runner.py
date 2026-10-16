"""Tests for the EXP-034 simulated formation runner."""
from __future__ import annotations

import json

from tools.e_exp034_formation import build_jobs, main
from tests.reference_e.hand_checks import first_order_loop_jitter_std


def test_full_grid_count_and_named_cells() -> None:
    jobs = build_jobs()
    assert len(jobs) == 1760  # 1600 point-mass cells plus 160 physics cells at the D-121 20-seed limit
    scenarios = {(j["labels"]["adapter"], j["labels"]["sigma_m"], j["labels"]["fix_latency_s"], j["labels"]["preset"])
                 for j in jobs}
    assert ("point_mass_offboard", 0.01, 0.0, "clean") in scenarios
    assert ("point_mass_offboard", 0.10, 0.2, "iid_10") in scenarios
    assert ("physics_offboard", 0.05, 0.1, "clean") in scenarios


def test_hand_check_reference() -> None:
    assert abs(first_order_loop_jitter_std(0.02, 1.0 / 30.0, 0.25) / 0.02 - 0.258) < 0.002


def test_tiny_run_is_reproducible(tmp_path, monkeypatch) -> None:
    tiny = [
        {"labels": {"slice": "tiny", "sigma_m": 0.02, "fix_latency_s": 0.0,
                     "preset": "clean", "adapter": "point_mass_offboard", "seed_index": 0},
         "scenario": {"mission": "formation", "seed": 1, "duration_s": 5.0,
                       "adapter": "point_mass_offboard", "preset": "clean",
                       "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": 0.02, "fix_latency_s": 0.0}}},
        {"labels": {"slice": "tiny", "sigma_m": 0.10, "fix_latency_s": 0.2,
                     "preset": "clean", "adapter": "point_mass_offboard", "seed_index": 1},
         "scenario": {"mission": "formation", "seed": 2, "duration_s": 5.0,
                       "adapter": "point_mass_offboard", "preset": "clean",
                       "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": 0.10, "fix_latency_s": 0.2}}},
    ]
    monkeypatch.setattr("tools.e_exp034_formation.build_jobs", lambda pilot=False: tiny)
    first = tmp_path / "one"
    second = tmp_path / "two"
    main(["--output", str(first), "--processes", "1"])
    main(["--output", str(second), "--processes", "1"])
    assert (first / "formation.csv").read_bytes() == (second / "formation.csv").read_bytes()
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    for name in ("formation.csv", "summary.json", "formation_rms_latency.png", "manifest.json"):
        assert (first / name).exists()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["pilot"] is False
