"""Depth bracket of tools/s6_measured_rerun.py on fixtures with different noise per axis.

The fixture generator's isotropic noise cannot tell the depth axis from the
pooled value, so these logs give one axis three times the noise of the other
two. The truths each test recovers are the per-axis standard deviations given
to the generator (TRUE_AXIS_STD below) and the pooled value worked out from
them by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sim.sensor import SensorConfig
from tools.fit_vision_noise import main as fit_cli
from tools.make_detection_fixture import main as fixture_cli
from tools.make_detection_fixture import make_fixture
from tools.s6_measured_rerun import RerunError, resolve_brackets, run_rerun

# The --noise-std-axis given to the generator: z (the camera depth axis) is 3 times x and y.
TRUE_AXIS_STD = (0.02, 0.02, 0.06)
# Pooled truth worked by hand: sqrt((0.02^2 + 0.02^2 + 0.06^2) / 3) = sqrt(0.0044 / 3), about 0.03830 m.
TRUE_POOLED = math.sqrt(0.0044 / 3)


def _fixture_and_fit(folder: Path, axis_std: tuple[float, float, float]) -> dict:
    log = folder / "log.jsonl"
    fixture_cli([str(log), "--frames", "3000", "--rate", "30",
                 "--noise-std-axis", *(str(value) for value in axis_std),
                 "--bias", "0.01", "-0.02", "0.03", "--dropout", "0.1",
                 "--latency-mean", "0.05", "--latency-std", "0.01", "--seed", "11"])
    fit_path = folder / "fit.json"
    fit_cli([str(log), "--output", str(fit_path), "--bootstrap", "200", "--seed", "1"])
    return {"log": log, "fit_path": fit_path, "fit": json.loads(fit_path.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def deep_z(tmp_path_factory) -> dict:
    return _fixture_and_fit(tmp_path_factory.mktemp("deep_z"), TRUE_AXIS_STD)


@pytest.fixture(scope="module")
def deep_x(tmp_path_factory) -> dict:
    # Same noise values, but the large one is on x, which is not the camera depth axis.
    return _fixture_and_fit(tmp_path_factory.mktemp("deep_x"), (0.06, 0.02, 0.02))


def test_fit_recovers_each_axis_within_its_interval(deep_z):
    per_axis = deep_z["fit"]["per_axis"]
    assert per_axis["axes"] == ["x", "y", "z"]
    assert per_axis["position_frame"] == "camera"
    for i, truth in enumerate(TRUE_AXIS_STD):
        assert per_axis["std_ci_low_m"][i] <= truth <= per_axis["std_ci_high_m"][i]
        # About 2700 detected frames; the std of a Gaussian sample has standard error sigma / sqrt(2N).
        assert abs(per_axis["std_m"][i] - truth) < 5 * truth / math.sqrt(2 * 2700)


def test_depth_bracket_is_the_z_std_within_its_interval(deep_z):
    per_axis = deep_z["fit"]["per_axis"]
    entries = {entry["label"]: entry for entry in resolve_brackets(deep_z["fit"])}
    depth = entries["depth"]
    assert depth["source"] == "per_axis.std_m[z]"
    assert depth["noise_std_m"] == per_axis["std_m"][2]
    assert per_axis["std_ci_low_m"][2] <= TRUE_AXIS_STD[2] <= per_axis["std_ci_high_m"][2]
    assert per_axis["std_ci_low_m"][2] <= depth["noise_std_m"] <= per_axis["std_ci_high_m"][2]


def test_pooled_bracket_sits_between_the_axes(deep_z):
    entries = {entry["label"]: entry["noise_std_m"] for entry in resolve_brackets(deep_z["fit"])}
    std = deep_z["fit"]["per_axis"]["std_m"]
    assert max(std[0], std[1]) < entries["pooled"] < std[2]
    noise = deep_z["fit"]["uncertainty"]["noise_std_m"]
    assert noise["ci_low"] <= TRUE_POOLED <= noise["ci_high"]
    assert entries["pooled"] == pytest.approx(math.sqrt(sum(value**2 for value in std) / 3), rel=1e-12)
    # The depth bracket is at least 1.5 times the pooled one here (truth ratio 0.06 / 0.0383, about 1.57),
    # so the two brackets really differ.
    assert entries["depth"] > 1.5 * entries["pooled"]


def test_depth_bracket_follows_the_axis_not_the_largest_value(deep_x):
    # The large noise is on x. The default depth axis in the camera frame is z, so the depth
    # bracket must stay small; only an explicit --depth-axis x picks the large value.
    fit = deep_x["fit"]
    std = fit["per_axis"]["std_m"]
    default = {entry["label"]: entry["noise_std_m"] for entry in resolve_brackets(fit)}
    assert default["depth"] == std[2]
    assert abs(default["depth"] - 0.02) < 5 * 0.02 / math.sqrt(2 * 2700)
    assert default["depth"] < default["pooled"]
    chosen = resolve_brackets(fit, ("depth",), depth_axis="x")[0]
    assert chosen["source"] == "per_axis.std_m[x]"
    assert abs(chosen["noise_std_m"] - 0.06) < 5 * 0.06 / math.sqrt(2 * 2700)


def test_rerun_depth_folder_uses_the_z_std(deep_z, tmp_path):
    out = tmp_path / "out"
    manifest = run_rerun(deep_z["fit_path"], out, seeds_count=1, processes=1,
                         cell_filter=lambda cell: (cell["motion"], cell["delay_s"], cell["loss_probability"])
                         == ("constant_velocity", 0.1, 0.3))
    std = deep_z["fit"]["per_axis"]["std_m"]
    by_label = {item["label"]: item["noise_std_m"] for item in manifest["bracket"]}
    assert manifest["depth_axis"] == "z"
    assert by_label["depth"] == std[2]
    assert json.loads((out / "depth" / "summary.json").read_text(encoding="utf-8"))["noise_std_m"] == std[2]
    assert json.loads((out / "pooled" / "summary.json").read_text(encoding="utf-8"))["noise_std_m"] \
        == by_label["pooled"]


def test_equal_axis_values_match_the_single_value_path():
    # normal(0, s) in NumPy is s times a standard normal draw, so giving every axis the same s must
    # reproduce the old single-value positions exactly.
    common = {"n_frames": 300, "frame_rate_hz": 30, "seed": 5, "latency_mean_s": 0.03, "latency_std_s": 0.008}
    single = make_fixture(sensor_config=SensorConfig(noise_std_m=0.04, bias_m=(0.02, -0.01, 0.05),
                                                     dropout_probability=0.1), **common)
    per_axis = make_fixture(sensor_config=SensorConfig(noise_std_m=0.0, bias_m=(0.02, -0.01, 0.05),
                                                       dropout_probability=0.1),
                            noise_std_axis_m=(0.04, 0.04, 0.04), **common)
    assert [o.as_dict() for o in per_axis.observations] == [o.as_dict() for o in single.observations]


def test_cli_records_axis_noise_and_rejects_bad_values(tmp_path, capsys):
    output = tmp_path / "fixture.jsonl"
    base = ["--frames", "12", "--rate", "30", "--bias", "0", "0", "0", "--dropout", "0",
            "--latency-mean", "0.03", "--latency-std", "0.0", "--seed", "7"]
    fixture_cli([str(output), "--noise-std-axis", "0.02", "0.02", "0.06", *base])
    sidecar = json.loads(output.with_suffix(".params.json").read_text(encoding="utf-8"))
    assert sidecar["noise_std_axis_m"] == [0.02, 0.02, 0.06]
    assert sidecar["noise_std_m"] is None
    header = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(header["notes"].split("Parameters: ", 1)[1]) == sidecar
    assert header["status"] == "synthetic"

    old = tmp_path / "old.jsonl"
    fixture_cli([str(old), "--noise-std", "0.04", *base])
    assert "noise_std_axis_m" not in json.loads(old.with_suffix(".params.json").read_text(encoding="utf-8"))

    # Each bad command must fail for its own reason, so the argparse message is checked too.
    bad_commands = [
        (["--noise-std", "0.04", "--noise-std-axis", "0.02", "0.02", "0.06"], "not allowed with argument"),
        ([], "one of the arguments --noise-std --noise-std-axis is required"),
        (["--noise-std-axis", "0.02", "-0.02", "0.06"], "noise_std_axis_m must be three finite values"),
        (["--noise-std-axis", "0.02", "nan", "0.06"], "noise_std_axis_m must be three finite values"),
    ]
    for index, (noise_args, message) in enumerate(bad_commands):
        with pytest.raises(SystemExit):
            fixture_cli([str(tmp_path / f"bad{index}.jsonl"), *noise_args, *base])
        assert message in capsys.readouterr().err
    with pytest.raises(ValueError, match="not both"):
        make_fixture(n_frames=5, frame_rate_hz=30, sensor_config=SensorConfig(noise_std_m=0.04), seed=1,
                     latency_mean_s=0.03, latency_std_s=0.0, noise_std_axis_m=(0.02, 0.02, 0.06))


def test_depth_rerun_still_refuses_world_frame_without_axis(deep_z, tmp_path):
    fit = dict(deep_z["fit"])
    fit["per_axis"] = dict(fit["per_axis"], position_frame="world")
    with pytest.raises(RerunError):
        resolve_brackets(fit)
