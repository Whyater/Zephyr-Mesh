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


def _synthetic_rows(ratios_by_cell: dict, sent: int, latency_s: float = 0.002) -> list[dict]:
    """Point-mass rows with given per-seed delivery ratios; no delivered packet means no latency."""
    rows = []
    for (distance, fraction), ratios in ratios_by_cell.items():
        for index, ratio in enumerate(ratios):
            delivered = ratio > 0.0
            rows.append({"adapter": "point_mass", "distance_m": distance, "relay_fraction": fraction,
                         "seed": index, "delivery_ratio": ratio,
                         "median_latency_s": latency_s if delivered else None,
                         "p95_latency_s": latency_s if delivered else None,
                         "hop1_delivery": None, "hop2_delivery": None, "retries_mean": None})
    return rows


def _reference_wilson(k: int, n: int) -> tuple[float, float]:
    # Independent reference: the shared S6 Wilson implementation (scipy normal quantile).
    from sim.s6_recovery import wilson_interval
    return wilson_interval(k, n)


def test_cell_reports_pooled_wilson_on_delivered_over_sent():
    from tools.e_exp032_relay import cell_summary
    rows = _synthetic_rows({(900.0, "none"): [1.0, 0.5, 0.0]}, sent=2)
    cell = cell_summary(rows, packets_per_run=2)
    pooled = cell["delivery_pooled"]
    # 2 + 1 + 0 delivered out of 3 x 2 sent.
    assert pooled["sent"] == 6
    assert pooled["delivered"] == 3
    assert pooled["ratio"] == 0.5
    low, high = _reference_wilson(3, 6)
    assert abs(pooled["wilson_low"] - low) < 1e-12
    assert abs(pooled["wilson_high"] - high) < 1e-12
    # The per-seed median interval is still reported beside it.
    assert cell["delivery_ratio_interval"]["n"] == 3


def test_pooled_counts_refuse_a_ratio_that_is_not_a_whole_count():
    from tools.e_exp032_relay import cell_summary
    rows = _synthetic_rows({(900.0, "none"): [0.3]}, sent=2)
    pooled = cell_summary(rows, packets_per_run=2)["delivery_pooled"]
    assert pooled["delivered"] is None
    assert pooled["wilson_low"] is None


def test_send_schedule_count_matches_a_real_run():
    # A 2 s direct run at 900 m: the mission's ratio times 21 is a whole number, times 20 or 22 is not.
    from tools.e_common import make_job, run_job
    from tools.e_exp032_relay import delivered_count, packets_sent_per_run
    assert packets_sent_per_run(2.0, 10.0) == 21
    assert packets_sent_per_run(60.0, 10.0) == 601
    job = make_job({"distance_m": 900.0, "relay_fraction": "none", "adapter": "point_mass"},
                   {"mission": "relay", "seed": 1, "duration_s": 2.0, "adapter": "point_mass",
                    "params": {"distance_m": 900.0, "relay_fraction": None}})
    ratio = run_job(job)["delivery_ratio"]
    assert 0.0 < ratio < 1.0
    assert delivered_count(ratio, 21) is not None
    assert delivered_count(ratio, 20) is None
    assert delivered_count(ratio, 22) is None


def test_deviation_explains_both_delivery_intervals(tmp_path):
    summary = run_experiment(tmp_path / "one", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    assert any("Wilson" in item and "median interval" in item for item in summary["deviations"])
    assert all(cell["delivery_pooled"]["delivered"] is not None for cell in summary["cells"].values())


def test_latency_is_labelled_conditional_and_counts_silent_seeds(tmp_path):
    from tools.e_exp032_relay import LATENCY_NOTE, cell_summary, latency_points
    rows = _synthetic_rows({(1300.0, "none"): [0.0, 0.0, 0.5], (900.0, "none"): [1.0, 0.0, 1.0],
                            (1100.0, "none"): [0.5, 0.5, 0.0]}, sent=2)
    cells = {}
    for distance in (900.0, 1100.0, 1300.0):
        cell_rows = [r for r in rows if r["distance_m"] == distance]
        cells[f"point_mass|{distance}|none"] = cell_summary(cell_rows, packets_per_run=2)
    assert cells["point_mass|1300.0|none"]["seeds_without_delivered_packet"] == 2
    assert cells["point_mass|900.0|none"]["seeds_without_delivered_packet"] == 1
    points, omitted, minority = latency_points(cells, "none")
    assert [d for d, _, _, _ in points] == [900.0, 1100.0]
    assert omitted == [1300.0]
    # 900 m delivered 4 of 6 packets (drawn filled); 1100 m delivered 2 of 6 (drawn hollow).
    assert minority == [1100.0]
    assert "delivered packets only" in LATENCY_NOTE
    summary = run_experiment(tmp_path / "one", pilot=True, n_processes=1, duration_s=0.1, seed_count=1)
    assert any("conditional on delivery" in item for item in summary["limits"])


def test_every_relay_fraction_gets_its_own_visible_panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.e_exp032_relay import FRACTION_LABELS, cell_summary, delivery_figure, latency_figure
    # 0.4 and 0.6 identical, 0.25 and 0.75 identical: the case where single-axes lines hide each other.
    ratios = {}
    for fraction in ("none", 0.25, 0.4, 0.5, 0.6, 0.75):
        for distance in (300.0, 900.0):
            ratios[(distance, fraction)] = [1.0, 0.5]
    rows = _synthetic_rows(ratios, sent=2)
    cells = {}
    for (distance, fraction), _ in ratios.items():
        cell_rows = [r for r in rows if r["distance_m"] == distance and r["relay_fraction"] == fraction]
        cells[f"point_mass|{distance}|{fraction}"] = cell_summary(cell_rows, packets_per_run=2)
    for build in (delivery_figure, latency_figure):
        fig = build(cells)
        data_axes = [ax for ax in fig.axes if ax.get_title()]
        assert len(data_axes) == 6
        titles = [ax.get_title() for ax in data_axes]
        assert titles == [FRACTION_LABELS[f] for f in ("none", 0.25, 0.4, 0.5, 0.6, 0.75)]
        main_styles = set()
        for ax in data_axes:
            main = [line for line in ax.get_lines() if line.get_label() == ax.get_title()]
            assert len(main) == 1
            main_styles.add((main[0].get_marker(), main[0].get_linestyle()))
        assert len(main_styles) == 6
        assert "simulated" in fig._suptitle.get_text()
        plt.close(fig)


def test_prediction_observations_split_stored_text_and_use_row_values():
    from tools.e_exp032_relay import (cell_summary, prediction_observations, split_sentences)
    for text in (PREDICTION_TEXT, PREDICTION_CORRECTION_3):
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert " ".join(sentences) == text
    ratios = {}
    for fraction in ("none", 0.25, 0.5, 0.75):
        for distance, value in ((300.0, 1.0), (900.0, 0.5), (1100.0, 0.0)):
            ratios[(distance, fraction)] = [value, value]
    rows = _synthetic_rows(ratios, sent=2)
    cells = {}
    for (distance, fraction), _ in ratios.items():
        cell_rows = [r for r in rows if r["distance_m"] == distance and r["relay_fraction"] == fraction]
        cells[f"point_mass|{distance}|{fraction}"] = cell_summary(cell_rows, packets_per_run=2)
    hand_check = {"reference_median_signal_range_m": 518.0,
                  "direct_50_percent_crossing_m_interpolation": 900.0}
    observations = prediction_observations(cells, hand_check, rows)
    assert [o["sentence"] for o in observations] == (split_sentences(PREDICTION_TEXT)
                                                    + split_sentences(PREDICTION_CORRECTION_3))
    first = observations[0]["observed"]
    assert first["direct_delivery_at_300_m"]["mean"] == 1.0
    assert first["direct_delivery_at_900_m_and_beyond"]["1100.0"]["mean"] == 0.0
    assert first["f_0_5_minus_asymmetric_delivery"]["0.25"]["900.0"] == 0.0
    latency = observations[1]["observed"]
    assert latency["median_latency_ratio_relay_over_direct"]["0.5"]["300.0"] == 1.0
    assert latency["max_cell_median_latency_s"]["value"] == 0.002
    assert latency["largest_per_seed_median_latency_s"]["value"] == 0.002
    assert observations[3]["observed"]["f_0_5_delivery_from_1000_to_1300_m"]["1100.0"]["mean"] == 0.0
    verdict_words = ("confirmed", "contradicted", "verdict", "correct", "wrong", "met", "pass")
    text = json.dumps([o["observed"] for o in observations]).lower()
    assert not any(f'"{word}' in text or f" {word} " in text for word in verdict_words)


def test_longer_hop_follows_the_relay_geometry():
    from tools.e_exp032_relay import longer_hop

    # sim/missions/relay.py puts the relay at fraction * D from the ground
    # station, so hop 1 (far drone to relay) is (1 - fraction) * D long and
    # hop 2 (relay to ground station) is fraction * D long.
    distance = 1000.0
    for fraction in (0.25, 0.4, 0.6, 0.75):
        hop1 = (1.0 - fraction) * distance
        hop2 = fraction * distance
        assert longer_hop(fraction) == ("hop1" if hop1 > hop2 else "hop2")
    assert longer_hop(0.5) == "equal"
