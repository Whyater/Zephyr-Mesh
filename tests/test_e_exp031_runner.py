"""Checks for the EXP-031 sweep runner."""
from __future__ import annotations

import json
from pathlib import Path

from tests.platform_compare import same_tree
from tools.e_exp031_sweep import EXPLORATORY_LABEL, PREDICTION, PREDICTION_NOTE, build_jobs, run_experiment


def test_grid_builder_counts_and_named_cells():
    jobs = build_jobs()
    # Main 5 presets x 2 dropouts x 2 modes x 50 seeds = 1,000; physics 2 presets x 1 dropout x 2 modes
    # x 20 seeds = 80; exploratory calm 2 presets x 2 dropouts x 2 modes x 50 seeds = 400; total 1,480.
    slices = [job["labels"]["slice"] for job in jobs]
    assert slices.count("main") == 1000
    assert slices.count("physics") == 80
    assert slices.count(EXPLORATORY_LABEL) == 400
    assert len(jobs) == 1480
    assert jobs[0]["labels"] == {"slice": "main", "preset": "clean", "dropout": "none",
                                  "message_mode": "delta", "adapter": "point_mass", "seed_index": 0}
    assert any(j["labels"]["slice"] == "physics" and j["scenario"]["adapter"] == "physics" for j in jobs)
    assert any(j["labels"]["slice"] == EXPLORATORY_LABEL
               and j["scenario"]["policy"]["land_after_s"] == 1.0 for j in jobs)
    # Pilot: main 2 presets x 2 dropouts x 2 modes x 3 seeds = 24; physics 1 cell x 2 seeds = 2;
    # one calm cell (iid_30, dropout one, delta) x 2 seeds = 2; total 28.
    pilot_jobs = build_jobs(pilot=True)
    assert len(pilot_jobs) == 28
    calm = [j for j in pilot_jobs if j["labels"]["slice"] == EXPLORATORY_LABEL]
    assert len(calm) == 2
    assert all(j["labels"]["preset"] == "iid_30" and j["labels"]["dropout"] == "one"
               and j["labels"]["message_mode"] == "delta" for j in calm)
    assert {j["scenario"]["duration_s"] for j in pilot_jobs} == {60.0}


def test_pilot_outputs_are_deterministic(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    tiny_jobs = []
    for job in build_jobs(pilot=True)[:2]:
        tiny = dict(job)
        tiny["scenario"] = dict(job["scenario"], duration_s=0.1)
        tiny_jobs.append(tiny)
    run_experiment(first, pilot=True, n_processes=1, jobs=tiny_jobs, argv=["--output", "a folder/out"])
    run_experiment(second, pilot=True, n_processes=1, jobs=tiny_jobs)
    expected = {"sweep.csv", "summary.json", "sweep_completion_simulated.png",
                "sweep_belief_error_simulated.png", "manifest.json"}
    assert {p.name for p in first.iterdir()} == expected
    assert (first / "sweep.csv").read_bytes() == (second / "sweep.csv").read_bytes()
    left = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert same_tree(left, right)
    assert left["pilot"] is True
    assert left["duration_s"] == 0.1
    assert left["prediction"]["text"] == PREDICTION
    assert left["prediction"]["text"].startswith("without dropout")
    assert left["prediction"]["prediction_note"] == PREDICTION_NOTE
    assert left["prediction"]["prediction_source"].startswith("pre-registered before the runs")
    verdicts = left["hand_check"]["observed_clean_no_dropout"]
    assert [v["message_mode"] for v in verdicts] == ["delta", "cumulative"]
    assert all(v["verdict"] in {"inside", "outside", "not run"} for v in verdicts)
    assert verdicts[0]["n_runs"] == 2 and verdicts[0]["verdict"] == "outside"
    assert verdicts[1]["verdict"] == "not run"
    assert all("within_0.1_s" in key for item in left["atlas"].values() for key in item if "wilson" in key)
    assert "120" not in "".join(key for item in left["atlas"].values() for key in item)
    assert left["notes"][0].startswith("Mission changes before any full run:")
    assert any("keeps sharing" in line for line in left["limits"])
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["replay_command"] == "PYTHONPATH=. python tools/e_exp031_sweep.py --output 'a folder/out'"
    assert left["hand_check"]["reference_sweeping_s"] == 800 / (3 * 4 * 2)
    assert left["missing_metrics"] == []
    assert all("failsafe_landings" in cell["metrics"] for cell in left["cells"])



def _row(preset, dropout, mode, k, *, t100=None, dup=0.03, belief=0, landings=0, slice_name="main"):
    """One synthetic sweep row with every column the summary reads."""
    return {"slice": slice_name, "preset": preset, "dropout": dropout, "message_mode": mode,
            "adapter": "point_mass", "seed_index": k, "seed": k, "t100_s": t100, "t95_s": t100,
            "duplicated_fraction": dup, "missed_cells": 0, "replans": 0, "belief_error_cells": belief,
            "false_loss_declarations": 0, "failsafe_landings": landings}


def _synthetic_rows():
    """Four runs per cell with numbers chosen so every expected value is easy to work out by hand."""
    rows = []
    for mode in ("delta", "cumulative"):
        for k in range(4):
            rows.append(_row("clean", "none", mode, k, t100=38.0))
            rows.append(_row("clean", "one", mode, k, t100=50.0 + 2 * k, dup=0.04))  # median 53
            rows.append(_row("iid_10", "none", mode, k, t100=38.0))
            rows.append(_row("iid_10", "one", mode, k, t100=56.0, belief=14 if mode == "delta" else 0))
            # iid_30: three of four runs land on the failsafe and never finish.
            lands = 1 if k < 3 else 0
            rows.append(_row("iid_30", "none", mode, k, t100=None if lands else 39.0, landings=lands))
            belief = (40, 50, 60, 70)[k] if mode == "delta" else (0, 1, 1, 2)[k]  # medians 55 and 1
            rows.append(_row("iid_30", "one", mode, k, t100=None if lands else 60.0,
                             dup=0.10 if mode == "delta" else 0.04, belief=belief, landings=lands))
            rows.append(_row("burst_10_0.5s", "none", mode, k, landings=2))
            rows.append(_row("burst_10_0.5s", "one", mode, k, dup=0.0,
                             belief=8 if mode == "delta" else 0, landings=1))
    return rows


def test_prediction_observations_follow_each_stored_sentence():
    from tools.e_exp031_sweep import _summarize

    summary = _summarize(_synthetic_rows(), pilot=False, horizon_s=120.0)
    observations = summary["prediction"]["prediction_observations"]
    # The stored text has four sentences; together they rebuild it exactly.
    assert len(observations) == 4
    assert " ".join(item["sentence"] for item in observations) == PREDICTION
    no_dropout, dropout_dup, bursts, dropout_time = (item["observed"] for item in observations)

    # Sentence 1: without dropout, per preset and mode, with failsafe landings per cell.
    assert no_dropout["dropout"] == "none"
    cells = {(c["preset"], c["message_mode"]): c for c in no_dropout["cells"]}
    assert [c["preset"] for c in no_dropout["cells"][::2]] == ["clean", "iid_10", "iid_30", "burst_10_0.5s"]
    iid30 = cells[("iid_30", "delta")]
    assert iid30["n_runs"] == 4
    assert iid30["p_t100_reached"]["rate"] == 0.25
    assert iid30["t100_s"]["median"] == 39.0
    assert iid30["runs_with_failsafe_landing"]["rate"] == 0.75
    assert iid30["t100_median_minus_clean_s"] == 1.0
    assert iid30["p_t100_minus_clean"] == -0.75
    assert cells[("burst_10_0.5s", "cumulative")]["t100_median_minus_clean_s"] is None

    # Sentence 2: duplication and belief error after one dropout, per preset and mode.
    cells = {(c["preset"], c["message_mode"]): c for c in dropout_dup["cells"]}
    assert dropout_dup["dropout"] == "one"
    assert cells[("iid_30", "delta")]["belief_error_cells"]["median"] == 55.0
    assert cells[("iid_30", "cumulative")]["belief_error_cells"]["median"] == 1.0
    assert cells[("iid_30", "delta")]["duplicated_fraction"]["median"] == 0.10
    diff = {d["preset"]: d for d in dropout_dup["cumulative_minus_delta"]}
    assert diff["iid_30"]["belief_error_cells_median"] == -54.0
    assert diff["iid_30"]["duplicated_fraction_median"] == -0.06

    # Sentence 3: each burst preset against the independent preset with the same loss probability.
    pairs = bursts["burst_vs_iid"]
    assert [(p["iid_preset"], p["burst_preset"]) for p in pairs] == [("iid_10", "burst_10_0.5s")]
    delta = {m["message_mode"]: m for m in pairs[0]["by_mode"]}["delta"]
    assert delta["burst_minus_iid"]["belief_error_cells_median"] == -6.0
    assert delta["burst"]["runs_with_failsafe_landing"]["rate"] == 1.0
    assert delta["iid"]["runs_with_failsafe_landing"]["rate"] == 0.0

    # Sentence 4: completion time with one dropout, and the band read from the stored text.
    assert dropout_time["band_s"] == [45.0, 55.0]
    cells = {(c["preset"], c["message_mode"]): c for c in dropout_time["cells"]}
    assert cells[("clean", "delta")]["t100_s"]["median"] == 53.0
    assert cells[("clean", "delta")]["t100_median_in_band"] is True
    assert cells[("iid_30", "delta")]["t100_median_in_band"] is False
    assert cells[("burst_10_0.5s", "delta")]["t100_median_in_band"] is None
    assert cells[("iid_30", "delta")]["runs_with_failsafe_landing"]["count"] == 3

    band = summary["prediction"]["dropout_completion_band_check"]
    assert band["predicted_t100_range_s"] == [45.0, 55.0]
    assert [b["message_mode"] for b in band["observed_clean_one_dropout"]] == ["delta", "cumulative"]
    assert all(b["observed_t100"]["median"] == 53.0 and b["median_in_band"] is True
               and b["n_reached_t100"] == 4 for b in band["observed_clean_one_dropout"])

    # Observed numbers only: no verdict words outside the stored sentences.
    import re
    text = json.dumps([item["observed"] for item in observations] + [band])
    assert not re.search(r"\b(verdict|confirmed|contradicted|supported|refuted|inside|outside|met|correct|wrong)\b",
                         text)


def test_dropout_band_comes_from_stored_prediction_text():
    from tools.e_exp031_sweep import DROPOUT_BAND_S, _prediction_sentences

    assert DROPOUT_BAND_S == (45.0, 55.0)
    assert PREDICTION.endswith(f"about {DROPOUT_BAND_S[0]:g} to {DROPOUT_BAND_S[1]:g} s.")
    assert len(_prediction_sentences(PREDICTION)) == 4


def _figure_text(fig) -> str:
    parts = [t.get_text() for t in fig.texts]
    if fig._suptitle is not None:
        parts.append(fig._suptitle.get_text())
    parts += [ax.get_title() for ax in fig.axes]
    return "\n".join(parts)


def _check_categorical_points(fig, presets):
    for ax in fig.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == list(presets)
        # Presets are categories: no line in the axes joins points (error bar caps included).
        assert all(line.get_linestyle() == "None" for line in ax.get_lines())
        points = []
        for container in ax.containers:
            data_line = container.lines[0]
            assert data_line.get_marker() not in (None, "None", "")
            points += list(zip(data_line.get_xdata(), data_line.get_ydata()))
        # Every plotted point has its own position, so equal values never hide each other.
        assert points and len(points) == len(set(points))


def test_figures_label_slice_and_keep_series_apart():
    import matplotlib.pyplot as plt
    from tools.e_exp031_sweep import _belief_figure, _completion_figure, _failsafe_caption

    rows = _synthetic_rows()
    presets = ["clean", "iid_10", "iid_30", "burst_10_0.5s"]
    caption = _failsafe_caption(rows, ("one",))
    # iid_30 (3 of 4) and burst_10_0.5s (4 of 4) landed in most runs; clean and iid_10 never did.
    assert "iid_30" in caption and "burst_10_0.5s" in caption
    assert "clean" not in caption and "iid_10" not in caption
    assert "most runs" in caption
    assert "No preset" in _failsafe_caption([r for r in rows if r["preset"] == "clean"], ("one",))

    # Rows from other slices are not drawn and do not change the runs-per-cell count.
    others = []
    for slice_name, adapter in (("physics", "physics"), (EXPLORATORY_LABEL, "point_mass")):
        for preset in presets:
            for dropout in ("none", "one"):
                for mode in ("delta", "cumulative"):
                    for k in range(3):
                        row = _row(preset, dropout, mode, k, t100=40.0, slice_name=slice_name)
                        others.append(dict(row, adapter=adapter))
    assert _failsafe_caption(rows + others, ("one",)) == caption
    completion = _completion_figure(rows + others, 120.0)
    belief = _belief_figure(rows + others)
    for fig, dropouts in ((completion, ("none", "one")), (belief, ("one",))):
        text = _figure_text(fig)
        assert "simulated" in text.lower()
        assert "main" in text and "point mass" in text and "default policy" in text
        assert "dropout" in text
        assert "4 runs per cell" in text
        assert _failsafe_caption(rows, dropouts) in text
        assert "calm policy slice and the physics slice are not shown" in text
        _check_categorical_points(fig, presets)
        plt.close(fig)
    assert len(completion.axes) == 2 and len(belief.axes) == 2
