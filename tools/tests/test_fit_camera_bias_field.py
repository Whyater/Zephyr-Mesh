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


def test_verdict_regimes() -> None:
    small = fcb.verdict({"total_rms_3d_m": 0.07, "varying_share_depth": 0.9})
    assert small["size_band"] == "at_or_below_5cm_cell" and small["regime"] == "shape_does_not_matter"
    assert "shape does not matter" in small["text"]
    mostly_offset = fcb.verdict({"total_rms_3d_m": 0.17, "varying_share_depth": 0.2})
    assert mostly_offset["regime"] == "mostly_offset" and mostly_offset["size_band"] == "between_cells"
    assert "20 to 33 %" in mostly_offset["text"]
    mostly_varying = fcb.verdict({"total_rms_3d_m": 0.17, "varying_share_depth": 0.85})
    assert mostly_varying["regime"] == "mostly_varying"
    assert "15 of 30" in mostly_varying["text"]
    between = fcb.verdict({"total_rms_3d_m": 0.17, "varying_share_depth": 0.5})
    assert between["regime"] == "share_not_simulated"
    big = fcb.verdict({"total_rms_3d_m": 0.25, "varying_share_depth": 0.2})
    assert big["size_band"] == "above_10cm_cell"
    no_bias = fcb.verdict({"total_rms_3d_m": 0.12, "varying_share_depth": None})
    assert no_bias["regime"] == "no_bias"
    for v in (small, mostly_offset, mostly_varying, between, big, no_bias):
        assert "simulation" in v["text"] and "not measured" in v["text"]
    # The thresholds are the study's cell sizes: sqrt(3) x 5 cm and sqrt(3) x 10 cm.
    assert fcb.CELL_5CM_TOTAL_RMS_M == pytest.approx(0.0866025, abs=1e-7)
    assert fcb.CELL_10CM_TOTAL_RMS_M == pytest.approx(0.1732051, abs=1e-7)


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


def test_cli_refusal_exits_2(tmp_path: Path, capsys) -> None:
    log = tmp_path / "run.jsonl"
    _write_log(log, GRID[:4], lambda p: (0.0, 0.0, 0.0))
    spot_file = tmp_path / "spots.csv"
    _write_spots(spot_file, GRID[:4])
    assert fcb.main([str(log), "--spots", str(spot_file)]) == 2
    assert "at least 5" in capsys.readouterr().err
