"""Tests for the EXP-034 simulated formation runner."""
from __future__ import annotations

import json
import math

from tools.e_exp034_formation import PREDICTION_SOURCE, PREDICTION_TEXT, build_jobs, main
from sim.tests.reference_e.hand_checks import first_order_loop_jitter_std


def test_full_grid_count_and_named_cells() -> None:
    jobs = build_jobs()
    # 1600 point-mass + 400 physics jobs on the registered 0.8 m guard ("local", D-126),
    # plus 800 Run D rule comparison jobs, all at 50 seeds
    assert len(jobs) == 2800
    rules = {}
    for j in jobs:
        rules.setdefault(j["labels"]["slice"], set()).add(j["labels"]["separation_rule"])
        assert j["scenario"]["params"]["separation_rule"] == j["labels"]["separation_rule"]
    assert rules == {"main": {"local"}, "physics": {"local"}, "coop_comparison": {"coop"}}
    scenarios = {(j["labels"]["adapter"], j["labels"]["sigma_m"], j["labels"]["fix_latency_s"], j["labels"]["preset"])
                 for j in jobs}
    assert ("point_mass_offboard", 0.01, 0.0, "clean") in scenarios
    assert ("point_mass_offboard", 0.10, 0.2, "iid_10") in scenarios
    assert ("physics_offboard", 0.05, 0.1, "clean") in scenarios


def test_hand_check_reference() -> None:
    assert abs(first_order_loop_jitter_std(0.02, 1.0 / 30.0, 0.25) / 0.02 - 0.258) < 0.002


def test_tiny_run_is_reproducible(tmp_path, monkeypatch) -> None:
    tiny = [
        {"labels": {"slice": "main", "sigma_m": 0.02, "fix_latency_s": 0.0,
                     "preset": "clean", "adapter": "point_mass_offboard", "separation_rule": "local", "seed_index": 0},
         "scenario": {"mission": "formation", "seed": 1, "duration_s": 6.0,
                       "adapter": "point_mass_offboard", "preset": "clean",
                       "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": 0.02, "fix_latency_s": 0.0,
                                  "separation_rule": "local"}}},
        {"labels": {"slice": "main", "sigma_m": 0.10, "fix_latency_s": 0.2,
                     "preset": "clean", "adapter": "point_mass_offboard", "separation_rule": "local", "seed_index": 1},
         "scenario": {"mission": "formation", "seed": 2, "duration_s": 6.0,
                       "adapter": "point_mass_offboard", "preset": "clean",
                       "params": {"fix_rate_hz": 30.0, "fix_noise_std_m": 0.10, "fix_latency_s": 0.2,
                                  "separation_rule": "local"}}},
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
    entry = next(e for e in hand if e["adapter"] == "point_mass_offboard" and e["sigma_m"] == 0.02)
    # Independent reference: 3 axes of first-order loop jitter at sigma 0.02 m, 30 Hz fixes, tau 0.25 s.
    expected_rms = math.sqrt(3.0) * first_order_loop_jitter_std(0.02, 1.0 / 30.0, 0.25)
    assert math.isclose(entry["reference_rms_error_m"], expected_rms, rel_tol=1e-12)
    for field in ("observed_to_reference", "observed_to_reference_rms"):
        assert math.isfinite(entry[field])
    assert "observed_to_reference_per_axis" not in entry
    assert summary["prediction"] == PREDICTION_TEXT
    assert "prediction:" not in PREDICTION_TEXT[:40]
    assert summary["prediction_source"] == PREDICTION_SOURCE
    assert "sqrt(2) x 30 x sigma at 30 Hz" in " ".join(summary["notes"])
    assert any("D-123" in text for text in summary["deviations"])


def test_pilot_summary_lists_the_shortened_runs() -> None:
    from tools.e_exp034_formation import summarize

    pilot = summarize([], pilot=True)
    assert any("10 s" in text for text in pilot["deviations"])
    assert not any("10 s" in text for text in summarize([], pilot=False)["deviations"])


def _fake_row(slice_name: str, rule: str, *, sigma: float, latency: float, value: float,
              adapter: str = "point_mass_offboard", seed_index: int = 0) -> dict:
    # Synthetic rows with every column the summary reads; value marks which rule a row came from.
    return {"slice": slice_name, "sigma_m": sigma, "fix_latency_s": latency, "preset": "clean",
            "adapter": adapter, "separation_rule": rule, "seed_index": seed_index,
            "rms_error_m": value, "p95_error_m": value, "per_axis_std_m": value,
            "min_separation_m": 2.0, "guard_activations": 0, "downwash_wind_max_mps": 0.0}


def _mixed_rows() -> list[dict]:
    # Same preset, adapter, sigma and latency in both rules: registered (local) rows read 0.01,
    # Run D rule comparison rows 1.0.
    rows = []
    for seed_index in range(3):
        rows.append(_fake_row("main", "local", sigma=0.02, latency=0.0, value=0.01, seed_index=seed_index))
        rows.append(_fake_row("coop_comparison", "coop", sigma=0.02, latency=0.0, value=1.0,
                              seed_index=seed_index))
    return rows


def test_hand_check_keeps_the_rules_apart() -> None:
    from tools.e_exp034_formation import summarize

    summary = summarize(_mixed_rows(), pilot=False)
    hand = summary["hand_check"]["zero_latency_clean"]
    entry = next(e for e in hand if e["adapter"] == "point_mass_offboard" and e["sigma_m"] == 0.02)
    assert entry["separation_rule"] == "local"
    assert entry["observed_per_axis_std_m"]["median"] == 0.01
    assert entry["observed_rms_error_m"]["median"] == 0.01
    comparison = summary["hand_check"]["zero_latency_clean_coop_rule_comparison"]
    comparison_entry = next(e for e in comparison if e["sigma_m"] == 0.02)
    assert comparison_entry["separation_rule"] == "coop"
    assert comparison_entry["slice"] == "coop_comparison"
    assert comparison_entry["observed_rms_error_m"]["median"] == 1.0


def test_figure_draws_the_comparison_rule_as_its_own_dashed_series(tmp_path, monkeypatch) -> None:
    import matplotlib.axes

    from tools.e_exp034_formation import _figures

    calls = []
    original = matplotlib.axes.Axes.plot

    def recording_plot(self, *args, **kwargs):
        calls.append((list(args[1]) if len(args) > 1 else [], dict(kwargs)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    _figures(tmp_path, _mixed_rows())
    series = [(y, kw) for y, kw in calls if kw.get("marker") == "o"]
    registered = [y for y, kw in series if kw.get("linestyle", "-") != "--"]
    comparison = [(y, kw) for y, kw in series if kw.get("linestyle") == "--"]
    assert registered == [[0.01]]
    assert [y for y, _ in comparison] == [[1.0]]
    assert "Run D rule" in comparison[0][1]["label"]


def test_grid_counts_come_from_the_job_list() -> None:
    from collections import Counter

    from tools.e_exp034_formation import summarize

    expected = dict(Counter(j["labels"]["slice"] for j in build_jobs()))
    counts = summarize([], pilot=False)["grid_counts"]
    assert counts == expected
    # 4*4*2*50 point mass, 2*2*2*50 physics, 4*4*1*50 Run D rule comparison.
    assert counts == {"main": 1600, "physics": 400, "coop_comparison": 800}


def test_deviations_and_limits_describe_the_rule_switch() -> None:
    from sim.missions.formation import FormationMission
    from tools.e_exp034_formation import summarize

    for pilot in (False, True):
        summary = summarize([], pilot=pilot)
        text = " ".join(summary["deviations"])
        assert "D-126" in text and "D-123" in text
        assert "as pre-registered" in text
        assert "after the pre-registration" in text
        assert "for comparison only" in text
        assert "changed the velocity" in text
        for line in FormationMission.limits:
            assert line in summary["limits"]
