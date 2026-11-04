"""Tests for tools/fit_camera_bias_field.py (synthetic logs only, no hardware)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools import fit_camera_bias_field as fcb

HEADER = {
    "schema": "zephyr-detection-log-1",
    "trace_id": "bias-grid-test",
    "status": "synthetic",
    "clock_domain": "shared_monotonic",
    "capture": {"source_type": "camera", "source_id": "test-camera", "device_ids": ["test-host"],
                "model": "test-detector"},
}

# 5 columns across the image (x / z from -0.5 to +0.5) at 2 ranges: 10 spots, camera frame
# (x right, y down, z out of the lens).
RANGES_M = (1.5, 2.5)
COLUMNS = (-0.5, -0.25, 0.0, 0.25, 0.5)
GRID = [(f"r{r}-c{i}", (u * r, 0.0, r)) for r in RANGES_M for i, u in enumerate(COLUMNS)]


def _write_log(path: Path, spots, error_fn, *, frames: int = 60, jitter_m: float = 0.002, seed: int = 0,
               ground_truth: bool = True, frame_gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Write a log with the marker still at each spot in turn. Returns each spot's capture-time window."""
    rng = np.random.default_rng(seed)
    lines = [json.dumps(HEADER)]
    windows = []
    t = 0.0
    n = 0
    for _, truth in spots:
        truth = np.asarray(truth, dtype=float)
        start = t
        bias = np.asarray(error_fn(truth), dtype=float)
        for _ in range(frames):
            n += 1
            measured = truth + bias + rng.normal(0.0, jitter_m, 3)
            obs = {"observation_id": f"obs-{n}", "frame_id": f"frame-{n}", "subject_id": "marker-0",
                   "capture_time_s": round(t, 6), "outcome": "detected",
                   "position_m": [float(v) for v in measured], "position_frame": "camera"}
            if ground_truth:
                obs["ground_truth"] = {"position_m": [float(v) for v in truth], "uncertainty_m": 0.01,
                                       "method": "tape"}
            lines.append(json.dumps(obs))
            t += 1.0 / 30.0
        windows.append((start - 1.0 / 60.0, t - 1.0 / 60.0))  # half a frame either side
        t += frame_gap_s
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return windows


def _write_spots(path: Path, spots, windows=None) -> None:
    rows = ["spot_id,x_m,y_m,z_m" + (",start_s,end_s" if windows else "")]
    for k, (spot_id, (x, y, z)) in enumerate(spots):
        row = f"{spot_id},{x!r},{y!r},{z!r}"
        if windows:
            row += f",{windows[k][0]!r},{windows[k][1]!r}"
        rows.append(row)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fit(tmp_path: Path, spots=GRID, error_fn=lambda p: (0.0, 0.0, 0.0), *, windows_in_csv=False, **kw):
    log = tmp_path / "run.jsonl"
    windows = _write_log(log, spots, error_fn, ground_truth=not windows_in_csv, **kw)
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, spots, windows if windows_in_csv else None)
    return fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))


# ---------------------------------------------------------------- the decomposition, by hand


def test_three_spot_hand_example() -> None:
    # Per-spot mean errors (m): A (0.01, 0, 0.03), B (0.03, 0, 0.06), C (0.02, 0, 0.00).
    # Each spot has 2 frames whose x error is the mean -0.002 and +0.002; y and z have no jitter.
    means = {"A": (0.01, 0.0, 0.03), "B": (0.03, 0.0, 0.06), "C": (0.02, 0.0, 0.00)}
    errors = {k: np.array([[m[0] - 0.002, m[1], m[2]], [m[0] + 0.002, m[1], m[2]]]) for k, m in means.items()}
    out = fcb.decompose_bias_field(errors, depth_axis="z")
    ax = out["per_axis"]
    # x: common = (0.01 + 0.03 + 0.02) / 3 = 0.02. Deviations -0.01, +0.01, 0.
    #    varying = sqrt((0.0001 + 0.0001 + 0) / 3) = sqrt(0.0000667) = 0.0081650.
    #    share = 0.0000667 / (0.0004 + 0.0000667) = 1/7 = 0.142857.
    assert ax["x"]["common_m"] == pytest.approx(0.02, abs=1e-12)
    assert ax["x"]["varying_m"] == pytest.approx(0.0081650, abs=1e-7)
    assert ax["x"]["varying_share"] == pytest.approx(1 / 7, abs=1e-9)
    # x jitter: two frames 0.004 apart, sample std = sqrt(2 * 0.002^2 / 1) = 0.0028284 at every spot.
    assert ax["x"]["jitter_rms_m"] == pytest.approx(0.0028284, abs=1e-7)
    # y: no error at all, so the share is undefined.
    assert ax["y"]["common_m"] == 0.0 and ax["y"]["varying_m"] == 0.0
    assert ax["y"]["varying_share"] is None
    # z: common = 0.03. Deviations 0, +0.03, -0.03. varying^2 = 0.0018 / 3 = 0.0006, varying = 0.0244949.
    #    bias mean square = 0.0009 + 0.0006 = 0.0015, share = 0.4.
    assert ax["z"]["common_m"] == pytest.approx(0.03, abs=1e-12)
    assert ax["z"]["varying_m"] == pytest.approx(0.0244949, abs=1e-7)
    assert ax["z"]["varying_share"] == pytest.approx(0.4, abs=1e-12)
    # 3D: mean of |per-spot mean error|^2 = (0.001 + 0.0045 + 0.0004) / 3 = 0.0019667, sqrt = 0.0443471.
    #     share = (0.0000667 + 0.0006) / 0.0019667 = 20/59 = 0.338983.
    #     total adds the x jitter mean square 0.000008: sqrt(0.0019747) = 0.0444372.
    assert out["accuracy_rms_3d_m"] == pytest.approx(0.0443471, abs=1e-7)
    assert out["varying_share_3d"] == pytest.approx(20 / 59, abs=1e-9)
    assert out["total_rms_3d_m"] == pytest.approx(0.0444372, abs=1e-7)
    assert out["depth_axis"] == "z"
    assert out["varying_share_depth"] == pytest.approx(0.4, abs=1e-12)
    spot_b = next(s for s in out["spots"] if s["spot_id"] == "B")
    assert spot_b["mean_error_m"] == pytest.approx([0.03, 0.0, 0.06], abs=1e-12)
    assert spot_b["frames"] == 2


def test_common_and_varying_add_up_to_the_bias_mean_square() -> None:
    rng = np.random.default_rng(3)
    errors = {f"s{i}": rng.normal(0.01 * i, 0.003, size=(7, 3)) for i in range(6)}
    out = fcb.decompose_bias_field(errors)
    means = np.array([e.mean(axis=0) for e in errors.values()])
    for k, name in enumerate("xyz"):
        a = out["per_axis"][name]
        assert a["common_m"] ** 2 + a["varying_m"] ** 2 == pytest.approx(float(np.mean(means[:, k] ** 2)), rel=1e-12)


def test_spots_are_weighted_equally_not_by_frame_count() -> None:
    # Spot A: 2 frames, z error 0. Spot B: 4 frames, z error 0.06 m.
    # Equal weight per spot: common = (0 + 0.06) / 2 = 0.03, varying = 0.03, share = 0.5.
    # (Weighting by frames would give common = 0.24 / 6 = 0.04 instead.)
    errors = {"A": np.zeros((2, 3)), "B": np.tile([0.0, 0.0, 0.06], (4, 1))}
    z = fcb.decompose_bias_field(errors)["per_axis"]["z"]
    assert z["common_m"] == pytest.approx(0.03, abs=1e-12)
    assert z["varying_m"] == pytest.approx(0.03, abs=1e-12)
    assert z["varying_share"] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------- whole logs


def test_pure_offset_gives_share_near_zero(tmp_path: Path) -> None:
    out = _fit(tmp_path, error_fn=lambda p: (0.02, -0.01, 0.05))
    ax = out["per_axis"]
    assert ax["x"]["common_m"] == pytest.approx(0.02, abs=0.001)
    assert ax["y"]["common_m"] == pytest.approx(-0.01, abs=0.001)
    assert ax["z"]["common_m"] == pytest.approx(0.05, abs=0.001)
    # Only jitter moves the per-spot means: 2 mm / sqrt(60) = 0.26 mm, against a 5 cm offset.
    assert out["varying_share_depth"] < 0.01
    assert out["varying_share_3d"] < 0.01
    assert out["spot_count"] == 10
    assert all(s["frames"] == 60 for s in out["spots"])


def test_pure_field_gives_share_near_one(tmp_path: Path) -> None:
    # Depth error grows across the image (8 cm per unit of x / z), lateral error with range. Both average to
    # zero over this symmetric grid, so nothing is a common offset.
    out = _fit(tmp_path, error_fn=lambda p: (0.03 * (p[2] - 2.0), 0.0, 0.08 * p[0] / p[2]))
    assert abs(out["per_axis"]["z"]["common_m"]) < 0.001
    assert out["per_axis"]["z"]["varying_m"] == pytest.approx(0.08 * math.sqrt(np.mean(np.square(COLUMNS))), abs=0.001)
    assert out["varying_share_depth"] > 0.99
    assert out["per_axis"]["x"]["varying_share"] > 0.99


def test_per_spot_mean_and_jitter(tmp_path: Path) -> None:
    out = _fit(tmp_path, error_fn=lambda p: (0.0, 0.0, 0.04), jitter_m=0.003, frames=400)
    for spot in out["spots"]:
        assert spot["mean_error_m"][2] == pytest.approx(0.04, abs=0.0006)
        assert spot["jitter_m"] == pytest.approx([0.003] * 3, abs=0.0004)


def test_time_windows_give_the_same_answer_as_ground_truth(tmp_path: Path) -> None:
    field = lambda p: (0.01, 0.0, 0.02 + 0.05 * p[0] / p[2])  # noqa: E731
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    by_truth = _fit(tmp_path / "a", error_fn=field)
    by_time = _fit(tmp_path / "b", error_fn=field, windows_in_csv=True)
    assert by_time["assignment"] == "time_windows" and by_truth["assignment"] == "ground_truth"
    assert by_time["per_axis"] == by_truth["per_axis"]


def test_several_logs_one_per_spot(tmp_path: Path) -> None:
    logs = []
    for k, spot in enumerate(GRID):
        path = tmp_path / f"spot{k}.jsonl"
        _write_log(path, [spot], lambda p: (0.0, 0.0, 0.03), seed=k)
        logs.append(path)
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID)
    out = fcb.fit_camera_bias_field(logs, fcb.read_spots_csv(spot_file))
    assert out["spot_count"] == 10
    assert out["per_axis"]["z"]["common_m"] == pytest.approx(0.03, abs=0.001)


# ---------------------------------------------------------------- verdict


def _v(total, share, p95=0.10, over=0, spots=10, share_is_noise=False):
    return fcb.verdict({"total_rms_3d_m": total, "varying_share_depth": share, "pooled_raw_p95_3d_m": p95,
                        "spots_over_p95_line": over, "spot_count": spots, "share_is_noise": share_is_noise})


def test_verdict_regimes() -> None:
    small = _v(0.07, 0.9)
    assert small["size_band"] == "at_or_below_5cm_cell" and small["regime"] == "shape_does_not_matter"
    assert "shape does not matter" in small["text"]
    mostly_offset = _v(0.17, 0.2)
    assert mostly_offset["regime"] == "mostly_offset" and mostly_offset["size_band"] == "between_cells"
    mostly_varying = _v(0.17, 0.85)
    assert mostly_varying["regime"] == "mostly_varying"
    between = _v(0.17, 0.5)
    assert between["regime"] == "share_not_simulated"
    big = _v(0.25, 0.2)
    assert big["size_band"] == "above_10cm_cell"
    no_bias = _v(0.12, None)
    assert no_bias["regime"] == "no_bias"
    noise = _v(0.12, 0.6, share_is_noise=True)
    assert noise["regime"] == "share_is_noise"
    for v in (small, mostly_offset, mostly_varying, between, big, no_bias, noise):
        assert "simulation" in v["text"] and "not measured" in v["text"]
        # The verdict leads with this camera's own numbers.
        assert v["text"].startswith("Measured (this session):")
        assert "expect worse" not in v["text"]
    # The study's seed shares are described as random simulated cameras, not as this camera's chance.
    assert "random simulated cameras" in mostly_offset["text"] and "one such draw" in mostly_offset["text"]
    assert "15 of 30" in mostly_varying["text"]
    # The thresholds are the study's cell sizes: sqrt(3) x 5 cm and sqrt(3) x 10 cm.
    assert fcb.CELL_5CM_TOTAL_RMS_M == pytest.approx(0.0866025, abs=1e-7)
    assert fcb.CELL_10CM_TOTAL_RMS_M == pytest.approx(0.1732051, abs=1e-7)


def test_verdict_expectation_uses_raw_p95_against_25_cm() -> None:
    under = _v(0.25, 0.2, p95=0.20, over=0)
    assert under["expectation"] == "pass_on_bias"
    over = _v(0.17, 0.2, p95=0.31, over=4)
    assert over["expectation"] == "fail_on_bias"
    close = _v(0.17, 0.2, p95=0.24, over=1)
    assert close["expectation"] == "near_the_line"
    for v in (under, over, close):
        assert "Simulation-based expectation, not a measured result" in v["text"]
        assert "0.25 m" in v["text"]


def test_weighted_p95_by_hand() -> None:
    # Values 1..20 with equal weight: the 95th percentile (first value whose cumulative weight reaches 95 %) is 19.
    assert fcb.weighted_p95(np.arange(1.0, 21.0), np.ones(20)) == 19.0
    # Two spots weighted equally: spot A has 2 frames at 0.1, spot B 8 frames at 0.3. Each frame of A weighs 1/2
    # of a spot, each of B 1/8. Sorted: 0.1, 0.1 (cumulative 0.5 of 1.0), then 0.3 frames reach 0.95 at the 8th.
    values = np.array([0.1, 0.1] + [0.3] * 8)
    weights = np.array([0.5, 0.5] + [0.125] * 8) / 2.0
    assert fcb.weighted_p95(values, weights) == 0.3


def test_raw_p95_per_spot_and_pooled(tmp_path: Path) -> None:
    # Half the columns (x > 0 and x = 0 alike) read 30 cm off in depth, the others 0: with jitter 2 mm the
    # spots over the line are exactly the 30 cm ones, and the pooled p95 is about 0.30 m.
    out = _fit(tmp_path, error_fn=lambda p: (0.0, 0.0, 0.30 if p[0] >= 0 else 0.0))
    over = [s["spot_id"] for s in out["spots"] if s["raw_p95_3d_m"] > 0.25]
    assert len(over) == 6 and out["spots_over_p95_line"] == 6
    assert out["pooled_raw_p95_3d_m"] == pytest.approx(0.30, abs=0.01)
    assert out["verdict"]["expectation"] == "fail_on_bias"


def test_notes_flag_a_share_inside_the_jitter_floor(tmp_path: Path) -> None:
    # Pure depth offset: x and y have no bias, so their shares are jitter noise and must be flagged.
    out = _fit(tmp_path, error_fn=lambda p: (0.0, 0.0, 0.05))
    assert any("x" in note and "noise floor" in note for note in out["notes"])
    assert not any(note.startswith("z ") for note in out["notes"])


def test_cli_writes_json_and_text(tmp_path: Path, capsys) -> None:
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID, lambda p: (0.0, 0.0, 0.05))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID)
    out_path = tmp_path / "fit.json"
    assert fcb.main([str(log), "--spots", str(spot_file), "--output", str(out_path)]) == 0
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["schema"] == "zephyr-camera-bias-field-1"
    assert result["verdict"]["size_band"] == "at_or_below_5cm_cell"
    text = capsys.readouterr().out
    assert "varying share" in text and "Verdict" in text


# ---------------------------------------------------------------- refusals


def test_refuses_fewer_than_five_spots(tmp_path: Path) -> None:
    with pytest.raises(fcb.BiasFieldError, match="at least 5"):
        _fit(tmp_path, spots=GRID[:4])


def test_refuses_too_few_frames_at_a_spot(tmp_path: Path) -> None:
    with pytest.raises(fcb.BiasFieldError, match="at least 60"):
        _fit(tmp_path, frames=59)


def test_refuses_spots_too_close_together(tmp_path: Path) -> None:
    close = GRID[:5] + [("too-close", (GRID[0][1][0] + 0.03, 0.0, GRID[0][1][2]))]
    with pytest.raises(fcb.BiasFieldError, match="not distinguishable"):
        _fit(tmp_path, spots=close)


def test_refuses_overlapping_time_windows(tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    windows = _write_log(log, GRID, lambda p: (0.0, 0.0, 0.0), ground_truth=False)
    windows[3] = (windows[2][0], windows[3][1])
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID, windows)
    with pytest.raises(fcb.BiasFieldError, match="overlap"):
        fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))


def test_refuses_measured_clusters_that_overlap(tmp_path: Path) -> None:
    # Spots 6 cm apart pass the position check, but 3 cm of jitter smears the measurements into each other.
    tight = [(f"c{i}", (-0.12 + 0.06 * i, 0.0, 2.0)) for i in range(5)]
    with pytest.raises(fcb.BiasFieldError, match="not distinguishable"):
        _fit(tmp_path, spots=tight, jitter_m=0.03)


def test_refuses_ground_truth_not_in_spot_file(tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID, lambda p: (0.0, 0.0, 0.0))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID[:-1] + [("moved", (0.9, 0.0, 2.5))])
    with pytest.raises(fcb.BiasFieldError, match="no spot"):
        fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))


def test_refuses_a_log_in_millimetres(tmp_path: Path) -> None:
    # The detector was given the marker length as 150 instead of 0.150: every position is 1000 x too big,
    # while the spots file (and the ground truth) are in metres. Matched by time so the ground truth cannot
    # catch it first.
    log = tmp_path / "run.jsonl"
    windows = _write_log(log, GRID, lambda p: 999.0 * np.asarray(p), ground_truth=False)
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID, windows)
    with pytest.raises(fcb.BiasFieldError, match="not in meters"):
        fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))


def test_refuses_log_and_spots_both_in_millimetres(tmp_path: Path) -> None:
    mm_grid = [(k, tuple(1000.0 * v for v in p)) for k, p in GRID]
    with pytest.raises(fcb.BiasFieldError, match="millimet"):
        _fit(tmp_path, spots=mm_grid, jitter_m=2.0)


def test_refuses_pose_flips(tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID, lambda p: (0.0, 0.0, 0.0))
    lines = log.read_text(encoding="utf-8").splitlines()
    for k in (5, 6, 7):  # three frames of the first spot come back behind the camera
        obs = json.loads(lines[k])
        obs["position_m"][2] = -obs["position_m"][2]
        lines[k] = json.dumps(obs)
    log.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID)
    with pytest.raises(fcb.BiasFieldError, match="pose flip"):
        fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))


def test_quirk_warnings_reach_the_notes(tmp_path: Path) -> None:
    # A frozen stream: every frame at one spot repeats the same position_m exactly.
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID, lambda p: (0.0, 0.0, 0.02))
    lines = log.read_text(encoding="utf-8").splitlines()
    frozen = json.loads(lines[1])["position_m"]
    for k in range(1, 61):
        obs = json.loads(lines[k])
        obs["position_m"] = frozen
        lines[k] = json.dumps(obs)
    log.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID)
    out = fcb.fit_camera_bias_field([log], fcb.read_spots_csv(spot_file))
    assert any(note.startswith(f"spot {GRID[0][0]}:") and "repeat the previous position_m" in note
               for note in out["notes"])


def test_note_when_all_spots_at_one_range(tmp_path: Path) -> None:
    one_range = [(k, p) for k, p in GRID if p[2] == 1.5]
    out = _fit(tmp_path, spots=one_range, error_fn=lambda p: (0.0, 0.0, 0.03))
    assert any("one range" in note for note in out["notes"])


def test_cli_refusal_exits_2(tmp_path: Path, capsys) -> None:
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID[:4], lambda p: (0.0, 0.0, 0.0))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID[:4])
    assert fcb.main([str(log), "--spots", str(spot_file)]) == 2
    assert "at least 5" in capsys.readouterr().err
