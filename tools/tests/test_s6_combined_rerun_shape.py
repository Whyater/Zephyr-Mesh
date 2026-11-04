"""Tests for the noise shape, bias and dropout options of tools/s6_combined_rerun.py.

The exactness checks compare against sim.s6_recovery.deliver, the registered
S6 code that this tool does not change, so they do not restate the new code:
equal per-axis noise must give the same packets as the isotropic path, and a
zero dropout or a zero bias must change nothing. The other checks use hand
numbers: a constant bias b on a straight line leaves the recovery arm about |b|
away from the truth, a fraction p of dropped detections removes about p of the
121 captures, and noise put on one world axis shows up on that axis only.
"""
import csv
import itertools
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from sim.s6_recovery import RecoveryConfig, deliver, seed_list, truth_position
from tools import s6_combined_rerun as tool

METRICS = ("seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count",
           "dropped_count", "fused_out_of_order_count", "failed")
N_CAPTURES = 121  # 12 s at 10 Hz, both ends included


def _fit(path: Path, *, pooled=0.0, std=None, bias=(0.0, 0.0, 0.0), dropout=0.0, mode="independent",
         start=0.0, end=1.0, frame="camera") -> Path:
    data = {"schema": "zephyr-vision-noise-fit-1", "input": {"trace_id": "hand", "status": "synthetic"},
            "sensor_config": {"noise_std_m": pooled, "bias_m": list(bias), "random_walk_std_m": 0.0,
                              "dropout_probability": dropout, "burst_start_probability": start,
                              "burst_end_probability": end, "mode": mode},
            "per_axis": {"axes": ["x", "y", "z"], "std_m": list(std or [pooled] * 3), "position_frame": frame},
            "scenario_delay": None}
    path.write_text(json.dumps(data))
    return path


def _radio(path: Path, **link) -> Path:
    path.write_text(json.dumps({"delay_s": 0.0, "jitter_s": 0.0, "loss_probability": 0.0, "status": "synthetic",
                                **link}))
    return path


def _run(tmp_path: Path, name: str, fit: Path, **kwargs) -> tuple[dict, list[dict]]:
    out = tmp_path / name
    tool.run_rerun(out, vision_fit_path=fit, **kwargs)
    with (out / "per_seed.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return json.loads((out / "summary.json").read_text()), rows


def _metrics(rows):
    return [{m: row[m] for m in METRICS} | {"motion": row["motion"], "arm": row["arm"]} for row in rows]


# ---- packets: the shaped path against the registered deliver ----------------

LINKS = [
    {"delay_s": 0.0, "loss_probability": 0.0, "loss_model": "independent", "jitter_s": 0.0},
    {"delay_s": 0.08, "loss_probability": 0.2, "loss_model": "independent", "jitter_s": 0.01},
    {"delay_s": 0.05, "loss_probability": 0.3, "loss_model": "burst", "jitter_s": 0.0,
     "burst_mean_length_packets": 4.0},
]


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("motion", tool.MOTIONS)
def test_equal_axis_std_and_zero_bias_and_dropout_give_the_registered_packets(link, motion):
    config = RecoveryConfig()
    for seed in seed_list(3):
        for noise in (0.0, 0.05, 0.137):
            expected = deliver(motion, noise_std_m=noise, seed=seed, config=config, **link)
            shaped, dropped = tool.shaped_deliver(
                motion, seed=seed, config=config, axis_std_m=(noise, noise, noise), bias_m=(0.0, 0.0, 0.0),
                dropout={"mode": "independent", "dropout_probability": 0.0, "burst_start_probability": 0.0,
                         "burst_end_probability": 1.0}, **link)
            assert dropped == 0
            assert shaped == expected  # exact: same seq, times and measurement floats


def test_noise_on_one_world_axis_appears_on_that_axis_only():
    config = RecoveryConfig()
    errors = []
    for seed in seed_list(20):
        packets, _ = tool.shaped_deliver("constant_velocity", delay_s=0.0, loss_probability=0.0, seed=seed,
                                         config=config, loss_model="independent", jitter_s=0.0,
                                         axis_std_m=(0.0, 0.09, 0.0), bias_m=(0.0, 0.0, 0.0), dropout=None)
        truth = truth_position("constant_velocity", [p.capture_time_s for p in packets], config)
        errors.append(np.asarray([p.measurement_m for p in packets]) - truth)
    err = np.vstack(errors)
    assert np.all(err[:, 0] == 0.0) and np.all(err[:, 2] == 0.0)
    # 2420 draws: the sample std of N(0, 0.09) is within a few percent of 0.09.
    assert np.std(err[:, 1]) == pytest.approx(0.09, rel=0.05)


def test_dropout_removes_captures_and_keeps_the_link_draws_of_the_rest():
    config = RecoveryConfig()
    link = {"delay_s": 0.05, "loss_probability": 0.1, "loss_model": "independent", "jitter_s": 0.01}
    kept_fraction = []
    for seed in seed_list(10):
        full = deliver("constant_turn", noise_std_m=0.0, seed=seed, config=config, **link)
        shaped, dropped = tool.shaped_deliver(
            "constant_turn", seed=seed, config=config, axis_std_m=(0.0, 0.0, 0.0), bias_m=(0.0, 0.0, 0.0),
            dropout={"mode": "independent", "dropout_probability": 0.3}, **link)
        by_seq = {p.seq: p for p in full}
        # Every surviving packet arrives exactly when it would have without dropout.
        assert all(by_seq[p.seq] == p for p in shaped)
        assert len(shaped) + sum(1 for p in full if p.seq not in {s.seq for s in shaped}) == len(full)
        assert dropped >= len(full) - len(shaped)
        kept_fraction.append(1 - dropped / N_CAPTURES)
    assert statistics.mean(kept_fraction) == pytest.approx(0.7, abs=0.05)


def test_burst_dropout_drops_runs_of_neighbouring_captures():
    config = RecoveryConfig()
    lengths = []
    for seed in seed_list(10):
        shaped, _ = tool.shaped_deliver(
            "constant_velocity", delay_s=0.0, loss_probability=0.0, seed=seed, config=config,
            loss_model="independent", jitter_s=0.0, axis_std_m=(0.0, 0.0, 0.0), bias_m=(0.0, 0.0, 0.0),
            dropout={"mode": "burst", "burst_start_probability": 0.05, "burst_end_probability": 0.25})
        seqs = sorted(p.seq for p in shaped)
        lengths += [b - a - 1 for a, b in itertools.pairwise(seqs) if b - a > 1]
    # sensor.py burst: once a burst starts, each dropped frame ends it with probability 0.25,
    # so a burst lasts 1 / 0.25 = 4 frames on average (a geometric count).
    assert statistics.mean(lengths) == pytest.approx(4.0, rel=0.3)


# ---- whole runs ---------------------------------------------------------------

def test_default_call_adds_nothing_to_the_outputs(tmp_path):
    fit = _fit(tmp_path / "fit.json", pooled=0.04, std=[0.02, 0.02, 0.06], bias=(0.01, 0.02, 0.03), dropout=0.2)
    summary, _ = _run(tmp_path, "out", fit, seeds_count=2)
    assert set(summary["settings"]) == {"delay_s", "jitter_s", "loss_probability", "loss_model", "noise_std_m",
                                        "tracker_measurement_std_m", "tracker_r"}
    assert "shape_rule" not in summary
    assert "fitted bias, dropout and per-axis shape are not applied; noise is isotropic" in summary["limits"]
    assert all("/shape" not in cell["cell_id"] for cell in summary["cells"])


def test_equal_per_axis_std_reproduces_the_isotropic_run(tmp_path):
    fit = _fit(tmp_path / "fit.json", pooled=0.07, std=[0.07, 0.07, 0.07])
    radio = _radio(tmp_path / "radio.json", delay_s=0.06, loss_probability=0.1, jitter_s=0.01)
    iso, iso_rows = _run(tmp_path, "iso", fit, radio_path=radio, seeds_count=3)
    shaped, shaped_rows = _run(tmp_path, "shaped", fit, radio_path=radio, seeds_count=3, noise_shape="per-axis",
                               axis_map="z,-x,-y")
    assert _metrics(shaped_rows) == _metrics(iso_rows)
    assert [c["arms"] for c in shaped["cells"]] == [c["arms"] for c in iso["cells"]]
    assert shaped["settings"]["noise_std_axis_m"] == [0.07, 0.07, 0.07]
    assert all("/shape/" in cell["cell_id"] for cell in shaped["cells"])


def test_zero_dropout_and_zero_bias_equal_no_dropout_and_no_bias(tmp_path):
    fit = _fit(tmp_path / "fit.json", pooled=0.05)
    radio = _radio(tmp_path / "radio.json", delay_s=0.1, loss_probability=0.2)
    _, plain = _run(tmp_path, "plain", fit, radio_path=radio, seeds_count=3)
    _, flagged = _run(tmp_path, "flagged", fit, radio_path=radio, seeds_count=3, apply_dropout=True,
                      apply_bias=True, axis_map="x,y,z")
    assert _metrics(flagged) == _metrics(plain)


def test_constant_bias_leaves_the_recovery_arm_bias_length_away(tmp_path):
    # Camera bias (0.04, 0, 0.03); axis map z,-x,-y puts it at world (0.03, -0.04, -0.0): length 0.05 m.
    fit = _fit(tmp_path / "fit.json", bias=(0.04, 0.0, 0.03))
    summary, _ = _run(tmp_path, "out", fit, seeds_count=2, apply_bias=True, axis_map="z,-x,-y")
    assert summary["settings"]["bias_m"] == [0.03, -0.04, -0.0]
    cell = next(c for c in summary["cells"] if c["motion"] == "constant_velocity")
    # No noise and a constant velocity target: the filter tracks truth + bias exactly after warm-up.
    assert cell["arms"]["recovery"]["p95_median_m"] == pytest.approx(0.05, abs=1e-3)


def test_dropout_from_the_fit_lowers_fused_counts(tmp_path):
    fit = _fit(tmp_path / "fit.json", dropout=0.25)
    summary, rows = _run(tmp_path, "out", fit, seeds_count=4, apply_dropout=True)
    fused = [int(r["fused_count"]) for r in rows if r["arm"] == "recovery"]
    assert statistics.mean(fused) == pytest.approx(N_CAPTURES * 0.75, rel=0.1)
    assert summary["settings"]["sensor_dropout"]["dropout_probability"] == 0.25
    assert any("dropout applied" in line for line in summary["limits"])


def test_shape_labels_reach_summary_readme_and_manifest(tmp_path):
    fit = _fit(tmp_path / "fit.json", pooled=0.0383, std=[0.02, 0.02, 0.06], bias=(0.0, 0.0, 0.01), dropout=0.1)
    out = tmp_path / "out"
    manifest = tool.run_rerun(out, vision_fit_path=fit, seeds_count=1, noise_shape="per-axis", apply_bias=True,
                              apply_dropout=True, axis_map="z,-x,-y")
    summary = json.loads((out / "summary.json").read_text())
    readme = (out / "README.txt").read_text()
    assert summary["settings"]["noise_std_axis_m"] == [0.06, 0.02, 0.02]  # world x is camera depth (z)
    assert summary["settings"]["axis_map"] == ["z", "-x", "-y"]
    assert summary["shape_rule"] and manifest["shape_rule"] == summary["shape_rule"]
    assert "fitted bias, dropout and per-axis shape are not applied; noise is isotropic" not in summary["limits"]
    assert "noise shape: per-axis" in readme and "axis map" in readme
    assert "--noise-shape" in manifest["argv"] and "--apply-dropout" in manifest["argv"]


# ---- refusals -------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"noise_shape": "per-axis"}, "--axis-map"),
    ({"apply_bias": True}, "--axis-map"),
    ({"noise_shape": "per-axis", "axis_map": "z,x"}, "three"),
    ({"noise_shape": "per-axis", "axis_map": "z,x,z"}, "once"),
    ({"noise_shape": "per-axis", "axis_map": "z,x,w"}, "axis"),
    ({"noise_shape": "per-axis", "axis_map": "z,-x,-y", "noise_label": "depth"}, "pooled"),
    ({"noise_shape": "cubic", "axis_map": "x,y,z"}, "isotropic or per-axis"),
    ({"axis_map": "x,y,z"}, "only with"),
])
def test_bad_shape_options_are_refused(tmp_path, kwargs, message):
    fit = _fit(tmp_path / "fit.json", pooled=0.03)
    with pytest.raises(tool.CombinedRerunError, match=message):
        tool.run_rerun(tmp_path / "out", vision_fit_path=fit, seeds_count=1, **kwargs)


def test_shape_options_need_a_vision_fit(tmp_path):
    radio = _radio(tmp_path / "radio.json", delay_s=0.05)
    with pytest.raises(tool.CombinedRerunError, match="--vision-fit"):
        tool.run_rerun(tmp_path / "out", radio_path=radio, seeds_count=1, apply_dropout=True)


@pytest.mark.parametrize("field, value, message", [
    ("bias_m", [0.0, 0.0], "bias_m"),
    ("bias_m", [0.0, float("nan"), 0.0], "bias_m"),
    ("dropout_probability", 1.5, "dropout_probability"),
    ("mode", "sometimes", "mode"),
    ("mode", "burst", "not identifiable"),  # start 0.0 (the fit's fallback) with dropout 0.2 would drop nothing
])
def test_bad_fit_fields_are_refused(tmp_path, field, value, message):
    fit_path = _fit(tmp_path / "fit.json", pooled=0.03, dropout=0.2)
    data = json.loads(fit_path.read_text())
    data["sensor_config"][field] = value
    fit_path.write_text(json.dumps(data))
    with pytest.raises(tool.CombinedRerunError, match=message):
        tool.run_rerun(tmp_path / "out", vision_fit_path=fit_path, seeds_count=1, apply_bias=True,
                       apply_dropout=True, axis_map="x,y,z")


def test_cli_flags_reach_the_run(tmp_path, capsys):
    fit = _fit(tmp_path / "fit.json", pooled=0.0383, std=[0.02, 0.02, 0.06], dropout=0.1)
    out = tmp_path / "out"
    assert tool.main(["--vision-fit", str(fit), "--output", str(out), "--seeds", "1", "--noise-shape", "per-axis",
                      "--axis-map", "z,-x,-y", "--apply-dropout"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"]["noise_shape"] == "per-axis"
    assert summary["settings"]["sensor_dropout"]["dropout_probability"] == 0.1
    assert "noise shape per-axis" in capsys.readouterr().out


def test_default_calls_never_use_the_shaped_path(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("shaped_deliver must not run when every shape option is off")

    monkeypatch.setattr(tool, "shaped_deliver", refuse)
    fit = _fit(tmp_path / "fit.json", pooled=0.04, std=[0.02, 0.02, 0.06], bias=(0.01, 0.0, 0.0), dropout=0.3)
    radio = _radio(tmp_path / "radio.json", delay_s=0.05, loss_probability=0.2, loss_model="burst",
                   burst_mean_length_packets=3.0, jitter_s=0.01)
    _run(tmp_path, "a", fit, radio_path=radio, seeds_count=1)
    _run(tmp_path, "b", fit, seeds_count=1, tracker_r="measured", noise_label="depth")
