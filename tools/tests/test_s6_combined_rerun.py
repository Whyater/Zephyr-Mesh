"""Tests for tools/s6_combined_rerun.py.

The zero-input check compares against the registered S6 sweep output
(sim/runs/s6-recovery/per_seed.csv), which was written by a different tool, so
it does not restate this tool's own code. The direction checks use the
physics: a delayed packet is stale when the naive arm fuses it, so delay must
raise naive error on both motions; the recovery arm back-dates the packet, so
on constant velocity with no noise it should barely move, while on a turn the
constant velocity model drifts and error grows.
"""
import csv
import json
import statistics
import time
from pathlib import Path

import pytest

from sim.s6_recovery import RecoveryConfig, seed_list
from tools import s6_combined_rerun as tool

REGISTERED_CSV = Path(__file__).resolve().parents[2] / "sim" / "runs" / "s6-recovery" / "per_seed.csv"
METRICS = ("seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count",
           "dropped_count", "fused_out_of_order_count", "failed")


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _radio(tmp_path, delay=0.0, jitter=0.0, loss=0.0, status="synthetic", name="radio.json", **extra):
    return _write(tmp_path / name, {"delay_s": delay, "jitter_s": jitter, "loss_probability": loss,
                                    "status": status, **extra})


def _vision_link(tmp_path, delay=0.0, jitter=0.0, status="synthetic", name="vlink.json", **extra):
    return _write(tmp_path / name, {"schema": "zephyr-vision-latency-link-1", "delay_s": delay, "jitter_s": jitter,
                                    "loss_probability": 0.0, "status": status, **extra})


def _fit(tmp_path, noise=0.0, status="synthetic", name="fit.json", std_m=None):
    data = {"schema": "zephyr-vision-noise-fit-1", "input": {"trace_id": "hand", "status": status},
            "sensor_config": {"noise_std_m": noise, "bias_m": [0.0, 0.0, 0.0], "dropout_probability": 0.0},
            "per_axis": {"axes": ["x", "y", "z"], "std_m": std_m or [noise] * 3, "position_frame": "camera"},
            "scenario_delay": None}
    return _write(tmp_path / name, data)


def _all(tmp_path, radio=None, vision=None, noise=0.0, **kwargs):
    """Run with all three inputs; zero settings unless overridden."""
    radio_path = _radio(tmp_path, **(radio or {}))
    vision_path = _vision_link(tmp_path, **(vision or {}))
    fit_path = _fit(tmp_path, noise=noise)
    out = tmp_path / "out"
    tool.run_rerun(out, radio_path=radio_path, vision_link_path=vision_path, vision_fit_path=fit_path, **kwargs)
    return json.loads((out / "summary.json").read_text()), out


def _p95(summary, motion, arm):
    cell = next(c for c in summary["cells"] if c["motion"] == motion)
    return cell["arms"][arm]["p95_median_m"]


def test_combining_rule_by_hand(tmp_path):
    # 3-4-5 triangle: jitters of 3 ms and 4 ms in quadrature give exactly 5 ms.
    radio = tool.radio_component({"delay_s": 0.010, "jitter_s": 0.003, "loss_probability": 0.2,
                                  "status": "measured"}, b"r", "r.json")
    vision = tool.vision_link_component({"schema": "zephyr-vision-latency-link-1", "delay_s": 0.130,
                                         "jitter_s": 0.004, "loss_probability": 0.0, "status": "measured"},
                                        b"v", "v.json")
    combined = tool.combine(radio, vision, None)
    settings = combined["settings"]
    assert settings["delay_s"] == pytest.approx(0.140, abs=1e-15)
    assert settings["jitter_s"] == pytest.approx(0.005, abs=1e-15)
    assert settings["loss_probability"] == 0.2 and settings["loss_model"] == "independent"
    assert settings["noise_std_m"] == 0.0 and settings["tracker_measurement_std_m"] == 0.05
    assert combined["not_included"] == ["vision_fit"]


def _assert_matches_registered(out: Path, registered_suffix: str, n_seeds: int) -> None:
    """Every per-seed metric equals the registered sweep's row for the same motion, arm and seed."""
    with (out / "per_seed.csv").open(newline="") as handle:
        ours = list(csv.DictReader(handle))
    with REGISTERED_CSV.open(newline="") as handle:
        registered = [row for row in csv.DictReader(handle)
                      if row["cell_id"].startswith("main/") and row["cell_id"].endswith(registered_suffix)
                      and int(row["seed_index"]) < n_seeds]
    assert len(ours) == len(registered) == 2 * 2 * n_seeds  # 2 motions x 2 arms

    def key(row):
        return row["motion"], row["arm"], row["seed_index"]

    for mine, theirs in zip(sorted(ours, key=key), sorted(registered, key=key)):
        assert key(mine) == key(theirs)
        assert {m: mine[m] for m in METRICS} == {m: theirs[m] for m in METRICS}


needs_registered = pytest.mark.skipif(not REGISTERED_CSV.is_file(),
                                      reason=f"registered S6 output missing: {REGISTERED_CSV.name} not found")


@needs_registered
def test_all_zero_inputs_reproduce_registered_main_cell_per_seed(tmp_path):
    _, out = _all(tmp_path, seeds_count=2)
    _assert_matches_registered(out, "/d0.000/l0.00/n0.000/independent/j0.000/r0.050", 2)


@needs_registered
def test_split_delay_loss_and_noise_reproduce_a_registered_nonzero_cell(tmp_path):
    # 0.04 s radio + 0.06 s vision = the registered 0.1 s delay; loss 0.1 from the radio, noise 0.05 from the fit.
    _, out = _all(tmp_path, radio={"delay": 0.04, "loss": 0.10}, vision={"delay": 0.06}, noise=0.05, seeds_count=2)
    _assert_matches_registered(out, "/d0.100/l0.10/n0.050/independent/j0.000/r0.050", 2)


def test_vision_delay_alone_raises_naive_error_and_turn_recovery_error(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    base, _ = _all(tmp_path / "a", seeds_count=3)
    late, _ = _all(tmp_path / "b", vision={"delay": 0.10}, seeds_count=3)
    # Hand check: the naive arm fuses a 0.1 s old measurement as current, so on a 1 m/s
    # straight line it trails the target by speed x delay = 0.1 m.
    speed = RecoveryConfig().target_speed_mps
    assert _p95(late, "constant_velocity", "naive") == pytest.approx(speed * 0.10, abs=1e-3)
    assert _p95(late, "constant_turn", "naive") > _p95(base, "constant_turn", "naive") + 0.5 * speed * 0.10
    assert _p95(late, "constant_turn", "recovery") > _p95(base, "constant_turn", "recovery")
    # Constant velocity model on a constant velocity target with exact stamps and no noise: still exact.
    assert _p95(late, "constant_velocity", "recovery") < 1e-3


def test_radio_and_vision_delays_add(tmp_path):
    split_dir, whole_dir = tmp_path / "split", tmp_path / "whole"
    split_dir.mkdir(), whole_dir.mkdir()
    split, _ = _all(split_dir, radio={"delay": 0.04}, vision={"delay": 0.06}, seeds_count=2)
    whole, _ = _all(whole_dir, radio={"delay": 0.10}, seeds_count=2)
    assert split["settings"]["delay_s"] == pytest.approx(0.10, abs=1e-15)
    for motion in tool.MOTIONS:
        assert _p95(split, motion, "naive") == pytest.approx(_p95(whole, motion, "naive"), rel=1e-12)


def test_radio_loss_alone_drops_packets(tmp_path):
    _, out = _all(tmp_path, radio={"loss": 0.3}, seeds_count=3)
    with (out / "per_seed.csv").open(newline="") as handle:
        fused = [int(row["fused_count"]) for row in csv.DictReader(handle)]
    assert max(fused) < 121  # 121 captures in 12 s at 10 Hz; loss removes some
    assert statistics.mean(fused) == pytest.approx(121 * 0.7, rel=0.15)


def test_burst_radio_loss_is_passed_through(tmp_path):
    _, out = _all(tmp_path, radio={"loss": 0.3, "loss_model": "burst", "burst_mean_length_packets": 4.0},
                  seeds_count=5)
    summary = json.loads((out / "summary.json").read_text())
    cell = summary["cells"][0]
    assert cell["loss_model"] == "burst" and cell["cell_id"].endswith("/b4.0")
    with (out / "per_seed.csv").open(newline="") as handle:
        fused = [int(row["fused_count"]) for row in csv.DictReader(handle) if row["arm"] == "recovery"]
    # Long-run loss is 0.3 in bursts, so about 70 percent of 121 arrive; bursts make seeds vary more.
    assert statistics.mean(fused) == pytest.approx(121 * 0.7, rel=0.25)
    assert min(fused) < max(fused)


def test_radio_jitter_with_zero_radio_delay_rides_on_vision_delay(tmp_path):
    summary, _ = _all(tmp_path, radio={"jitter": 0.003}, vision={"delay": 0.1, "jitter": 0.004}, seeds_count=1)
    assert summary["settings"]["delay_s"] == 0.1
    assert summary["settings"]["jitter_s"] == pytest.approx(0.005, abs=1e-15)


def test_vision_link_warnings_reach_the_outputs(tmp_path):
    vision = _vision_link(tmp_path, delay=0.13, status="measured", source="camera latency upper bound",
                          warnings=["display lag is assumed"])
    out = tmp_path / "out"
    tool.run_rerun(out, vision_link_path=vision, seeds_count=1)
    summary = json.loads((out / "summary.json").read_text())
    assert "vision link warning: display lag is assumed" in summary["limits"]
    readme = (out / "README.txt").read_text()
    assert "warning: display lag is assumed" in readme and "camera latency upper bound" in readme


def test_noise_alone_raises_error_on_both_arms(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    clean, _ = _all(tmp_path / "a", seeds_count=3)
    noisy, _ = _all(tmp_path / "b", noise=0.03, seeds_count=3)
    for motion in tool.MOTIONS:
        for arm in ("naive", "recovery"):
            assert _p95(noisy, motion, arm) > _p95(clean, motion, arm)


def test_jitter_on_top_of_delay_raises_naive_error(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    steady, _ = _all(tmp_path / "a", vision={"delay": 0.05}, seeds_count=5)
    jittery, _ = _all(tmp_path / "b", vision={"delay": 0.05, "jitter": 0.03}, seeds_count=5)
    for motion in tool.MOTIONS:
        assert _p95(jittery, motion, "naive") > _p95(steady, motion, "naive")


def test_tracker_r_measured_uses_fitted_noise_and_moves_nis_toward_three(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    registered, _ = _all(tmp_path / "a", noise=0.03, seeds_count=5)
    matched, _ = _all(tmp_path / "b", noise=0.03, tracker_r="measured", seeds_count=5)
    assert registered["settings"]["tracker_measurement_std_m"] == 0.05
    assert matched["settings"]["tracker_measurement_std_m"] == 0.03

    def nis(summary):
        cell = next(c for c in summary["cells"] if c["motion"] == "constant_velocity")
        return cell["arms"]["recovery"]["mean_nis_median"]

    # NIS is the squared innovation over its predicted variance, summed over 3 axes:
    # about 3 when R matches the true noise, lower when R overstates it.
    assert nis(registered) < 2.0
    assert abs(nis(matched) - 3.0) < abs(nis(registered) - 3.0)


@pytest.mark.parametrize("statuses, expected", [
    (("measured", "measured", "measured"), "measured"),
    (("measured", "measured", "synthetic"), "synthetic"),
    (("fixture", "measured", "measured"), "synthetic"),
    (("measured", "unknown", "measured"), "unknown"),
    (("measured", None, "measured"), "unknown"),
])
def test_weakest_input_status_sets_output_status(tmp_path, statuses, expected):
    radio_status, vision_status, fit_status = statuses
    radio = _radio(tmp_path, status=radio_status)
    vision = _vision_link(tmp_path, status=vision_status)
    fit = _fit(tmp_path, status=fit_status)
    out = tmp_path / "out"
    manifest = tool.run_rerun(out, radio_path=radio, vision_link_path=vision, vision_fit_path=fit, seeds_count=1)
    summary = json.loads((out / "summary.json").read_text())
    assert manifest["input_status"] == summary["input_status"] == expected
    assert f"from {expected} inputs" in summary["status"]
    assert expected in (out / "README.txt").read_text()


def test_espnow_fit_status_comes_from_source_trace(tmp_path):
    fit = {"schema": "zephyr-espnow-fit-1", "source_trace": {"status": "measured", "trace_id": "t1"},
           "s6_deliver": {"delay_s": 0.004, "jitter_s": 0.001, "loss_probability": 0.05}}
    component = tool.radio_component(fit, b"x", "fit.json")
    assert component["status"] == "measured" and component["provenance"]["trace_id"] == "t1"


def test_missing_inputs_are_listed_as_not_included(tmp_path):
    out = tmp_path / "out"
    tool.run_rerun(out, vision_link_path=_vision_link(tmp_path, delay=0.1, status="measured"), seeds_count=1)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["included"] == ["vision_link"]
    assert summary["not_included"] == ["radio", "vision_fit"]
    assert "not included: radio, vision_fit" in summary["status"]
    assert summary["settings"]["noise_std_m"] == 0.0
    assert summary["complete"] is False and summary["input_status"] == "measured"


def test_repeat_run_is_byte_identical(tmp_path):
    outputs = []
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        _, out = _all(folder, radio={"delay": 0.01, "jitter": 0.004, "loss": 0.1}, vision={"delay": 0.12},
                      noise=0.02, seeds_count=3)
        outputs.append(out)
    for name in ("summary.json", "per_seed.csv"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    manifests = [json.loads((out / "manifest.json").read_text()) for out in outputs]
    volatile = {"runtime_s", "argv", "replay_command"}  # components hold only file names and hashes
    assert {k: v for k, v in manifests[0].items() if k not in volatile} == \
        {k: v for k, v in manifests[1].items() if k not in volatile}
    assert manifests[0]["runtime_s"] > 0


def test_cli_pilot_runs_in_seconds(tmp_path, capsys):
    radio = _radio(tmp_path, delay=0.005, jitter=0.002, loss=0.1)
    vision = _vision_link(tmp_path, delay=0.13, jitter=0.005)
    fit = _fit(tmp_path, noise=0.03)
    out = tmp_path / "out"
    start = time.perf_counter()
    assert tool.main(["--radio", str(radio), "--vision-link", str(vision), "--vision-fit", str(fit),
                      "--output", str(out), "--pilot", "--tracker-r", "measured"]) == 0
    assert time.perf_counter() - start < 30
    summary = json.loads((out / "summary.json").read_text())
    assert summary["design"]["n_seeds"] == tool.PILOT_SEEDS and summary["design"]["pilot"] is True
    assert summary["complete"] is True
    assert not any(path.name.startswith(".") for path in tmp_path.iterdir())  # no staging folder left
    assert summary["design"]["seeds"] == seed_list(tool.PILOT_SEEDS)
    assert summary["config"]["duration_s"] == RecoveryConfig().duration_s
    assert "recovery" in capsys.readouterr().out


@pytest.mark.parametrize("case, message", [
    ("none", "at least one"),
    ("vision_loss", "loss_probability must be 0"),
    ("vision_schema", "schema zephyr-vision-latency-link-1"),
    ("r_without_fit", "needs --vision-fit"),
    ("r_zero_noise", "positive fitted noise"),
    ("jitter_no_delay", "combined delay is zero"),
    ("bad_radio", "radio input"),
])
def test_rejects_bad_inputs(tmp_path, case, message):
    kwargs = {}
    if case == "vision_loss":
        kwargs["vision_link_path"] = _vision_link(tmp_path, delay=0.1, loss_probability=0.1)
    elif case == "vision_schema":
        kwargs["vision_link_path"] = _write(tmp_path / "v.json", {"schema": "zephyr-espnow-fit-1", "delay_s": 0.1})
    elif case == "r_without_fit":
        kwargs.update(radio_path=_radio(tmp_path), tracker_r="measured")
    elif case == "r_zero_noise":
        kwargs.update(vision_fit_path=_fit(tmp_path, noise=0.0), tracker_r="measured")
    elif case == "jitter_no_delay":
        kwargs["vision_link_path"] = _vision_link(tmp_path, delay=0.0, jitter=0.01)
    elif case == "bad_radio":
        kwargs["radio_path"] = _write(tmp_path / "r.json", {"jitter_s": 0.01})
    with pytest.raises(tool.CombinedRerunError, match=message):
        tool.run_rerun(tmp_path / "out", seeds_count=1, **kwargs)
    assert not (tmp_path / "out").exists()


def test_refuses_output_inside_sim_runs(tmp_path):
    runs = Path(tool._REPO_ROOT) / "sim" / "runs" / "combined-should-not-exist"
    with pytest.raises(tool.CombinedRerunError, match="must not be inside"):
        tool.run_rerun(runs, radio_path=_radio(tmp_path), seeds_count=1)
    assert not runs.exists()


def test_failed_write_leaves_no_staging_folder(tmp_path, monkeypatch):
    def broken_csv(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tool, "_write_csv", broken_csv)
    with pytest.raises(OSError, match="disk full"):
        _all(tmp_path, pilot=True)
    assert not (tmp_path / "out").exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".out.partial-")] == []
