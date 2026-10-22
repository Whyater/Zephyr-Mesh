"""Checks for the EXP-033 simulated monitoring runner."""
from __future__ import annotations

import json
from pathlib import Path

from tools.e_exp033_monitor import PREDICTION, PREDICTION_SOURCE, build_jobs, run_experiment
from sim.tests.reference_e.hand_checks import alert_latency_s


def test_preregistered_grid_count_and_named_cells():
    jobs = build_jobs()
    # 3 paths * 5 presets * 3 delays * 50 seeds + 3 paths * 2 physics presets * 20 seeds = 2370.
    assert len(jobs) == 2370
    labels = {(j["labels"]["adapter"], j["labels"]["path"], j["labels"]["preset"], j["labels"]["delay_s"]) for j in jobs}
    assert ("point_mass", "crossing", "clean", 0.0) in labels
    assert ("point_mass", "skimming", "burst_30_0.5s", 0.5) in labels
    assert ("physics", "loiter", "burst_10_0.5s", 0.0) in labels


def test_tiny_run_is_reproducible_and_reports_hand_check(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    # Keep this test short while exercising the same writer and summary paths.
    tiny_jobs = build_jobs(pilot=True)[:2]
    run_experiment(first, pilot=True, n_processes=1, jobs=tiny_jobs)
    run_experiment(second, pilot=True, n_processes=1, jobs=tiny_jobs)
    assert (first / "exp033.csv").read_bytes() == (second / "exp033.csv").read_bytes()
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["hand_check"]["expected_by_delay_s"][0]["value_s"] == alert_latency_s(0.25, 0.0, 0.0)  # (3 - 0.5) x 0.1 s
    assert summary["pilot"] is True
    assert "passed" in summary["safety_check"]
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["outputs"]) == sorted(["exp033.csv", "summary.json", "alert_latency_crossing.png",
                                                  "alert_latency_loiter.png", "continuity_simulated.png"])
    assert summary["safety_check"]["passed"] is True
    assert summary["prediction"] == PREDICTION
    assert "prediction:" not in PREDICTION[:40]
    assert summary["prediction_source"] == PREDICTION_SOURCE
    assert any("horizontal displacement" in text for text in summary["deviations"])
    assert any("Addendum D" in text for text in summary["design_choices"])
    # Called from code: no command line is recorded (not pytest's), but the replay command names --output.
    assert manifest["argv"] == []
    assert "--output" in manifest["replay_command"]


# Synthetic rows for the summary and figure checks below. Values are chosen by hand so the
# expected answers are known without running the summary code (for example a median of 3 values).
import re
import statistics

import matplotlib.figure
import pytest

import tools.e_exp033_monitor as runner


def _row(path, preset, delay, seed, *, latency=0.3, continuity=1.0, landings=0, missed=False,
         false_alert=False, adapter="point_mass", alerts=None):
    return {
        "adapter": adapter, "path": path, "preset": preset, "delay_s": delay, "seed": seed,
        "alert_latency_s": latency, "alerts_raised": (0 if latency is None else 1) if alerts is None else alerts,
        "continuity": continuity, "duration_s": 20.0, "end_reason": "timeout",
        "false_alert": false_alert, "max_estimate_age_s": 0.0, "missed_alert": missed,
        "observer_landings": landings, "observer_max_displacement_m": 12.0 if landings else 0.0,
        "observer_max_horizontal_displacement_m": 0.0,
    }


def _saved_figures(monkeypatch, tmp_path, rows):
    """Run the runner's figure step and keep each saved Figure object by file name."""
    saved = {}

    def record_savefig(self, fname, *args, **kwargs):
        saved[Path(fname).name] = self

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_savefig)
    runner._figures(tmp_path, rows)
    return saved


def _all_text(fig) -> str:
    parts = [fig._suptitle.get_text() if fig._suptitle else ""]
    parts += [text.get_text() for text in fig.texts]
    for ax in fig.axes:
        parts.append(ax.get_title())
        parts += [text.get_text() for text in ax.texts]
    return "\n".join(parts)


def test_continuity_figure_has_one_panel_per_delay(monkeypatch, tmp_path):
    rows = []
    for seed, (c0, c2, c5) in enumerate([(1.0, 0.99, 0.04)] * 3):
        rows += [_row("crossing", "clean", 0.0, seed, continuity=c0),
                 _row("crossing", "clean", 0.2, seed, continuity=c2),
                 _row("crossing", "clean", 0.5, seed, continuity=c5)]
    fig = _saved_figures(monkeypatch, tmp_path, rows)["continuity_simulated.png"]
    titles = {ax.get_title(): ax for ax in fig.axes}
    # One labelled panel per delay, and no pooled value: the 0.5 s panel shows 0.04, not the pooled 0.99.
    assert set(titles) == {"measurement delay 0 s", "measurement delay 0.2 s", "measurement delay 0.5 s"}
    heights = {title: [patch.get_height() for patch in ax.patches] for title, ax in titles.items()}
    assert heights["measurement delay 0 s"] == pytest.approx([1.0])
    assert heights["measurement delay 0.2 s"] == pytest.approx([0.99])
    assert heights["measurement delay 0.5 s"] == pytest.approx([0.04])
    assert "simulated" in _all_text(fig)


def test_latency_figure_separates_presets_and_names_cells_without_alerts(monkeypatch, tmp_path):
    rows = []
    for seed in range(3):
        for delay in (0.0, 0.5):
            # clean and iid_10 have identical latencies, burst_30_0.5s never raises an alert.
            rows.append(_row("crossing", "clean", delay, seed, latency=0.25))
            rows.append(_row("crossing", "iid_10", delay, seed, latency=0.25))
            rows.append(_row("crossing", "burst_30_0.5s", delay, seed, latency=None, missed=True, landings=2))
    fig = _saved_figures(monkeypatch, tmp_path, rows)["alert_latency_crossing.png"]
    panel_titles = [ax.get_title() for ax in fig.axes]
    assert panel_titles == ["clean", "iid_10", "burst_30_0.5s"]
    text = _all_text(fig)
    assert "no alerts raised" in text
    assert "No alerts raised in: burst_30_0.5s" in text
    assert "simulated" in text


def test_cells_and_top_block_relate_missed_alerts_to_landings():
    rows = [
        _row("crossing", "burst_10_0.5s", 0.0, 1, latency=None, missed=True, landings=2),
        _row("crossing", "burst_10_0.5s", 0.0, 2, latency=0.3, landings=1),
        _row("crossing", "burst_10_0.5s", 0.0, 3, latency=None, missed=True, landings=0),
        _row("crossing", "burst_10_0.5s", 0.0, 4, latency=0.3, landings=0),
    ]
    summary = runner._summary(rows, pilot=True)
    metrics = summary["cells"][0]["metrics"]
    assert metrics["n_runs_with_observer_landing"] == 2
    assert metrics["n_missed_alert_runs"] == 2
    assert metrics["n_missed_alert_runs_with_observer_landing"] == 1
    block = summary["missed_alerts_and_observer_landings"]
    entry = block["by_adapter_and_preset"][0]
    assert (entry["preset"], entry["n_missed_alert_runs_with_observer_landing"],
            entry["n_missed_alert_runs_without_observer_landing"]) == ("burst_10_0.5s", 1, 1)
    assert block["totals"]["n_runs"] == 4
    assert "straight down" in block["note"] and "stops sending fixes" in block["note"]
    assert not re.search(r"\d", block["note"])


def test_hand_check_has_delay_note_without_numbers():
    summary = runner._summary([_row("crossing", "clean", 0.0, 1)], pilot=True)
    note = summary["hand_check"]["delay_note"]
    assert "forward to the current time" in note
    assert "straight line" in note
    assert "manoeuvring drone would not be treated this way" in note
    assert not re.search(r"\d", note)


def test_physics_equality_flag_compares_alert_metrics():
    point = [_row("crossing", "clean", 0.0, seed, latency=0.2 + 0.1 * seed) for seed in (1, 2)]
    physics = [dict(row, adapter="physics", continuity=0.5) for row in point]
    summary = runner._summary(point + physics, pilot=True)
    block = summary["physics_alert_equality"]
    assert block["metrics_compared"] == ["alert_latency_s", "alerts_raised", "false_alert", "missed_alert"]
    assert block["physics_matches_point_mass"] is True
    physics[1]["alert_latency_s"] = 0.9
    changed = runner._summary(point + physics, pilot=True)
    assert changed["physics_alert_equality"]["physics_matches_point_mass"] is False
    assert changed["physics_alert_equality"]["cells"][0]["n_rows_matching"] == 1
    physics_cell = [c for c in changed["cells"] if c["adapter"] == "physics"][0]
    assert physics_cell["physics_matches_point_mass_on_alert_metrics"] is False


def test_prediction_observations_keep_sentences_and_report_row_numbers():
    rows = []
    for seed, latency in enumerate((0.2, 0.3, 0.4)):
        rows.append(_row("crossing", "clean", 0.0, seed, latency=latency))
        rows.append(_row("crossing", "iid_30", 0.0, seed, latency=latency + 0.5, continuity=0.8))
        rows.append(_row("skimming", "clean", 0.0, seed, latency=None, false_alert=(seed == 0), alerts=int(seed == 0)))
    rows.append(_row("crossing", "iid_30", 0.0, 9, latency=None, missed=True, continuity=0.8))
    observations = runner._summary(rows, pilot=True)["prediction_observations"]
    sentences = [item["sentence"] for item in observations]
    assert " ".join(sentences) == runner.PREDICTION
    assert len(sentences) == 2
    first, second = observations[0]["observed"], observations[1]["observed"]
    clean = first["clean_alert_latency_s"][0]
    assert (clean["path"], clean["delay_s"], clean["n_runs_with_alert"]) == ("crossing", 0.0, 3)
    assert clean["median_s"] == pytest.approx(statistics.median([0.2, 0.3, 0.4]))
    assert first["clean_missed_alerts"][0]["missed_alert_count"] == 0
    assert first["skimming_false_alerts"][0]["false_alert_count"] == 1
    lossy = [e for e in second["alert_latency_s_by_preset"] if e["preset"] == "iid_30"][0]
    assert lossy["median_s"] == pytest.approx(0.8)
    assert lossy["change_from_clean_median_s"] == pytest.approx(0.5)
    lossy_miss = [e for e in second["missed_alerts_by_preset_and_delay"] if e["preset"] == "iid_30"][0]
    assert (lossy_miss["missed_alert_count"], lossy_miss["n_runs"]) == (1, 4)
    lossy_cont = [e for e in second["continuity_at_delay_0_by_preset"]
                  if e["preset"] == "iid_30" and e["path"] == "crossing"][0]
    assert lossy_cont["median"] == pytest.approx(0.8)
    text = json.dumps(observations).lower()
    for word in ("agree", "confirmed", "contradicted", "pass", "fail", "verdict", "correct"):
        assert not re.search(rf"\b{word}", text)
