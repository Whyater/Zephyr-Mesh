"""Tests for the EXP-034 simulated formation runner."""
from __future__ import annotations

import json
from pathlib import Path

from tools.e_exp034_formation import PREDICTION_TEXT, build_jobs, main
from tests.reference_e.hand_checks import first_order_loop_jitter_std


def test_full_grid_count_and_named_cells() -> None:
    jobs = build_jobs()
    assert len(jobs) == 2000  # 1600 point-mass jobs plus 400 physics jobs, all cells at 50 seeds
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
    hand = summary["hand_check"]["zero_latency_clean"]
    assert hand
    assert all("observed_to_reference_rms" in entry and "observed_to_reference" in entry and "within_10_percent" in entry for entry in hand)
    assert any(entry["adapter"] == "point_mass_offboard" and entry["sigma_m"] == 0.02 for entry in hand)


def test_prediction_matches_preregistration_when_present() -> None:
    prereg = Path("/Users/akashi0281/.zm-worktrees/run-e-briefs/prereg.md")
    if not prereg.exists():
        return
    text = prereg.read_text(encoding="utf-8")
    expected = (
        "Claude's prediction: point mass at zero latency matches the hand formula within 10 %; latency up to 0.2 s raises the error by under 30 % "
        "(the first-order loop with delay stays stable below about (pi/2) tau = 0.39 s, but the margin shrinks); physics adapter errors larger than the point mass at the same sigma "
        "(PD loop with lower damping passes more noise) and grows faster with latency (Run A: position feedback edge 0.33 s). Separation never near 0.5 m (slots 2 m apart, errors of centimetres)."
    )
    assert PREDICTION_TEXT == expected
    assert PREDICTION_TEXT in text
