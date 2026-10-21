"""Checks for tools/s6_measured_rerun.py.

All fit files here come from synthetic logs or are edited by hand. Comparisons
use the registered S6 files and the generator's known inputs, not values
recomputed by the tool under test.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

import tools.s6_measured_rerun as rerun_module
from tools.fit_vision_noise import main as fit_cli
from tools.make_detection_fixture import main as fixture_cli
from tools.run_s6_recovery import build_cells
from tools.s6_measured_rerun import (
    HASHED_SOURCES,
    RerunError,
    _status_text,
    build_rerun_cells,
    fit_status,
    main,
    protected_runs_dirs,
    resolve_brackets,
    run_rerun,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_DIR = REPO_ROOT / "runs" / "s6-recovery"
HAVE_REGISTERED = (REGISTERED_DIR / "summary.json").is_file() and (REGISTERED_DIR / "per_seed.csv").is_file()
ONE_CELL = "main/constant_velocity/d0.100/l0.30/n0.050/independent/j0.000/r0.050"
TRUE_NOISE = 0.05  # the --noise-std given to the fixture generator below


def _only_cell(motion: str, delay: float, loss: float):
    return lambda cell: (cell["motion"], cell["delay_s"], cell["loss_probability"]) == (motion, delay, loss)


@pytest.fixture(scope="module")
def fit_file(tmp_path_factory) -> Path:
    """A fit built from the synthetic generator: noise 0.05 m, 10 percent dropout."""
    folder = tmp_path_factory.mktemp("fit")
    log = folder / "log.jsonl"
    fixture_cli([str(log), "--frames", "300", "--rate", "10", "--noise-std", "0.05",
                 "--bias", "0.01", "0.02", "0.03", "--dropout", "0.1",
                 "--latency-mean", "0.05", "--latency-std", "0.01", "--seed", "7"])
    fit = folder / "fit.json"
    fit_cli([str(log), "--output", str(fit), "--bootstrap", "20", "--seed", "1"])
    return fit


def _edited(fit_file: Path, tmp_path: Path, edit, name: str = "edited.json") -> Path:
    data = json.loads(fit_file.read_text())
    edit(data)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _hand_fit(tmp_path: Path, noise: float = 0.05, axis_std: float = 0.05, frame: str = "camera",
              std_m: list[float] | None = None, status: str | None = "synthetic") -> Path:
    data = {
        "schema": "zephyr-vision-noise-fit-1",
        "input": {"trace_id": "hand", "input_sha256": "0" * 64},
        "sensor_config": {"noise_std_m": noise, "bias_m": [0.0, 0.0, 0.0], "dropout_probability": 0.0,
                          "mode": "independent"},
        "per_axis": {"axes": ["x", "y", "z"], "std_m": std_m or [axis_std] * 3,
                     "position_frame": frame},
        "scenario_delay": None,
    }
    if status is not None:
        data["input"]["status"] = status
    path = tmp_path / "hand_fit.json"
    path.write_text(json.dumps(data))
    return path


def test_rerun_from_generated_fit(fit_file, tmp_path):
    out = tmp_path / "out"
    manifest = run_rerun(fit_file, out, seeds_count=3, processes=2, cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    fit = json.loads(fit_file.read_text())
    assert manifest["fit_sha256"] == hashlib.sha256(fit_file.read_bytes()).hexdigest()
    assert manifest["git_tree_state"] in {"clean", "dirty", "unknown"}
    assert manifest["status"].startswith("proposal:") and "synthetic" in manifest["status"]
    assert fit["input"]["status"] == "synthetic"
    by_label = {item["label"]: item for item in manifest["bracket"]}
    assert by_label["pooled"]["noise_std_m"] == fit["sensor_config"]["noise_std_m"]
    assert by_label["depth"]["noise_std_m"] == fit["per_axis"]["std_m"][2]
    assert manifest["fit_input"]["input_sha256"] == fit["input"]["input_sha256"]
    assert manifest["n_cells"] == {"pooled": 1, "depth": 1}
    for label in ("pooled", "depth"):
        summary = json.loads((out / label / "summary.json").read_text())
        assert summary["bracket_label"] == label
        assert summary["noise_std_m"] == by_label[label]["noise_std_m"]
        assert summary["status"].startswith("proposal:") and "synthetic" in summary["status"]
        assert summary["design"]["n_seeds"] == 3
        assert (out / label / "per_seed.csv").is_file()
    assert (out / "comparison.csv").is_file()
    stored = json.loads((out / "manifest.json").read_text())
    assert stored["outputs"]["pooled/summary.json"] == hashlib.sha256((out / "pooled" / "summary.json").read_bytes()).hexdigest()
    assert set(stored["outputs"]) == {"pooled/summary.json", "pooled/per_seed.csv", "depth/summary.json",
                                      "depth/per_seed.csv", "comparison.csv", "README.txt"}
    # The fitted bias is recorded, not applied.
    assert stored["not_applied"]["sensor_config"]["bias_m"] == fit["sensor_config"]["bias_m"]


def test_depth_bracket_uses_camera_z(fit_file):
    fit = json.loads(fit_file.read_text())
    assert fit["per_axis"]["position_frame"] == "camera"
    entries = resolve_brackets(fit)
    depth = next(item for item in entries if item["label"] == "depth")
    assert depth["noise_std_m"] == fit["per_axis"]["std_m"][2]
    assert depth["source"] == "per_axis.std_m[z]"


def test_world_frame_needs_depth_axis(fit_file, tmp_path):
    def edit(data):
        data["per_axis"]["position_frame"] = "world"
        data["per_axis"]["std_m"] = [0.01, 0.02, 0.03]

    world = _edited(fit_file, tmp_path, edit)
    with pytest.raises(RerunError, match="depth"):
        run_rerun(world, tmp_path / "out", seeds_count=1, processes=1)
    assert not (tmp_path / "out").exists()
    manifest = run_rerun(world, tmp_path / "out_y", bracket=("depth",), depth_axis="y", seeds_count=2, processes=1,
                         cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert manifest["bracket"] == [{"label": "depth", "noise_std_m": 0.02, "source": "per_axis.std_m[y]"}]
    assert manifest["depth_axis"] == "y"
    # The pooled bracket does not need an axis.
    assert resolve_brackets(json.loads(world.read_text()), ("pooled",))[0]["label"] == "pooled"


def test_refuses_output_under_runs(tmp_path):
    fit = _hand_fit(tmp_path)
    target = REPO_ROOT / "runs" / "zz-rerun-should-not-exist"
    with pytest.raises(RerunError, match="runs"):
        run_rerun(fit, target, seeds_count=1, processes=1)
    assert not target.exists()
    with pytest.raises(SystemExit) as exit_info:
        main([str(fit), "--output", str(target)])
    assert exit_info.value.code == 2
    assert not target.exists()


def test_refuses_non_empty_output(tmp_path):
    fit = _hand_fit(tmp_path)
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(RerunError, match="not empty"):
        run_rerun(fit, out, seeds_count=1, processes=1)
    assert [path.name for path in out.iterdir()] == ["keep.txt"]
    assert (out / "keep.txt").read_text() == "keep"
    with pytest.raises(RerunError, match="inside the output"):
        run_rerun(out / "keep.txt", out, seeds_count=1, processes=1)


def test_rejects_wrong_schema_and_bad_noise(fit_file, tmp_path):
    def schema(data):
        data["schema"] = "zephyr-vision-noise-fit-2"

    def negative(data):
        data["sensor_config"]["noise_std_m"] = -0.01

    def infinite(data):
        data["sensor_config"]["noise_std_m"] = float("inf")

    def nan_axis(data):
        data["per_axis"]["std_m"][1] = float("nan")

    def negative_axis(data):
        data["per_axis"]["std_m"][2] = -1.0

    def short_axes(data):
        data["per_axis"]["axes"] = ["x", "y"]

    for index, edit in enumerate((schema, negative, infinite, nan_axis, negative_axis, short_axes)):
        path = _edited(fit_file, tmp_path, edit, name=f"bad{index}.json")
        out = tmp_path / f"out{index}"
        with pytest.raises(RerunError):
            run_rerun(path, out, seeds_count=1, processes=1)
        assert not out.exists()


def test_cells_match_registered_grid():
    cells = build_rerun_cells(0.05)
    assert len(cells) == 56
    registered = json.loads((REGISTERED_DIR / "summary.json").read_text()) if HAVE_REGISTERED else None
    if registered is None:
        pytest.skip("runs/s6-recovery is not present")
    expected = [{key: value for key, value in cell.items() if key not in {"arms", "paired"}}
                for cell in registered["cells"] if cell["slice"] == "main" and cell["noise_std_m"] == 0.05]
    assert [cell["cell_id"] for cell in cells] == [cell["cell_id"] for cell in expected]
    # Same keys, same key order (the registered runner's own order), same value types.
    runner_order = [list(cell) for cell in build_cells(False) if cell["slice"] == "main" and cell["noise_std_m"] == 0.05]
    for cell, want, keys in zip(cells, expected, runner_order):
        assert cell == want
        assert list(cell) == keys
        assert {k: type(v) for k, v in cell.items()} == {k: type(v) for k, v in want.items()}
    assert {cell["noise_std_m"] for cell in build_rerun_cells(0.0123)} == {0.0123}


@pytest.mark.skipif(not HAVE_REGISTERED, reason="runs/s6-recovery is not present")
def test_one_cell_matches_registered_bytes(tmp_path):
    fit = _hand_fit(tmp_path, noise=0.05, axis_std=0.05)
    out = tmp_path / "out"
    run_rerun(fit, out, bracket=("pooled",), seeds_count=50, processes=1,
              cell_filter=lambda cell: cell["cell_id"] == ONE_CELL)
    prefix = ONE_CELL + ","
    ours = [line for line in (out / "pooled" / "per_seed.csv").read_text().splitlines(keepends=True)
            if line.startswith(prefix)]
    registered = [line for line in (REGISTERED_DIR / "per_seed.csv").read_text().splitlines(keepends=True)
                  if line.startswith(prefix)]
    assert len(registered) == 100
    assert ours == registered
    new_entry = next(c for c in json.loads((out / "pooled" / "summary.json").read_text())["cells"]
                     if c["cell_id"] == ONE_CELL)
    old_entry = next(c for c in json.loads((REGISTERED_DIR / "summary.json").read_text())["cells"]
                     if c["cell_id"] == ONE_CELL)
    assert json.dumps(new_entry, indent=2, sort_keys=True) == json.dumps(old_entry, indent=2, sort_keys=True)
    # The comparison file carries the same cell's registered value at noise 0.05.
    header, *rows = (out / "comparison.csv").read_text().splitlines()
    columns = header.split(",")
    row = next(r.split(",") for r in rows if r.startswith("constant_velocity,0.1,0.3,"))
    assert row[columns.index("registered_n0.050_recovery_p95_median_m")] == repr(old_entry["arms"]["recovery"]["p95_median_m"])
    assert row[columns.index("pooled_recovery_p95_median_m")] == repr(old_entry["arms"]["recovery"]["p95_median_m"])


def test_bad_options(tmp_path):
    fit = _hand_fit(tmp_path)
    with pytest.raises(RerunError):
        run_rerun(fit, tmp_path / "a", bracket=("sideways",), seeds_count=1, processes=1)
    with pytest.raises(RerunError):
        run_rerun(fit, tmp_path / "b", seeds_count=0, processes=1)
    with pytest.raises(RerunError):
        run_rerun(fit, tmp_path / "c", seeds_count=1, processes=1, cell_filter=lambda cell: False)
    assert not any((tmp_path / name).exists() for name in "abc")


def _git_porcelain() -> str | None:
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
                                       cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_hand_made_camera_fit_brackets(tmp_path):
    std = [0.01, 0.02, 0.07]
    pooled = math.sqrt((0.01**2 + 0.02**2 + 0.07**2) / 3)  # worked by hand: sqrt(0.0054 / 3), about 0.04243
    fit = _hand_fit(tmp_path, noise=pooled, std_m=std)
    out = tmp_path / "out"
    manifest = run_rerun(fit, out, seeds_count=1, processes=1, cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    by_label = {item["label"]: item["noise_std_m"] for item in manifest["bracket"]}
    assert by_label["pooled"] == pytest.approx(0.0424264, abs=1e-7)
    assert by_label["pooled"] == pytest.approx(pooled, abs=1e-15)
    assert by_label["depth"] == 0.07
    assert manifest["depth_axis"] == "z"
    assert json.loads((out / "depth" / "summary.json").read_text())["noise_std_m"] == 0.07


def test_generator_fit_pooled_noise_near_truth(fit_file):
    fit = json.loads(fit_file.read_text())
    pooled = resolve_brackets(fit)[0]["noise_std_m"]
    # 300 frames at 10 percent dropout leave about 270 detected frames, so about 810 residuals. The standard error
    # of a sample std is sigma / sqrt(2 N), about 0.0012 m here; four standard errors is 0.005 m.
    assert abs(pooled - TRUE_NOISE) < 0.005
    # The generator's noise is isotropic, so the depth axis should agree with the pooled value too.
    depth = resolve_brackets(fit)[1]["noise_std_m"]
    assert abs(depth - TRUE_NOISE) < 0.01


def test_code_hashes_match_files(tmp_path):
    fit = _hand_fit(tmp_path)
    manifest = run_rerun(fit, tmp_path / "out", bracket=("pooled",), seeds_count=1, processes=1,
                         cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert set(manifest["code_sha256"]) == {"sim/tracker.py", "sim/s6_recovery.py", "sim/link.py", "sim/sensor.py",
                                            "tools/run_s6_recovery.py", "tools/s6_measured_rerun.py"}
    assert set(manifest["code_sha256"]) == set(HASHED_SOURCES)
    for relative, digest in manifest["code_sha256"].items():
        assert digest == hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_tree_state_matches_porcelain(tmp_path):
    before = _git_porcelain()
    if before is None:
        pytest.skip("git cannot read this checkout")
    fit = _hand_fit(tmp_path)
    manifest = run_rerun(fit, tmp_path / "out", bracket=("pooled",), seeds_count=1, processes=1,
                         cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert manifest["git_tree_state"] == ("dirty" if before else "clean")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    assert manifest["git_revision"] == head


def test_duplicate_and_empty_bracket_rejected(tmp_path):
    fit = _hand_fit(tmp_path)
    with pytest.raises(RerunError, match="once"):
        run_rerun(fit, tmp_path / "dup", bracket=("pooled", "pooled"), seeds_count=1, processes=1)
    with pytest.raises(RerunError, match="at least one"):
        run_rerun(fit, tmp_path / "empty", bracket=(), seeds_count=1, processes=1)
    assert not (tmp_path / "dup").exists() and not (tmp_path / "empty").exists()


@pytest.mark.parametrize("bad", [5, None, ["measured"], "bogus", "Measured", ""])
def test_bad_input_status_rejected(tmp_path, bad):
    fit = _hand_fit(tmp_path)
    data = json.loads(fit.read_text())
    data["input"]["status"] = bad
    fit.write_text(json.dumps(data))
    out = tmp_path / "out"
    with pytest.raises(RerunError, match="status"):
        run_rerun(fit, out, seeds_count=1, processes=1)
    assert not out.exists()


def test_missing_status_is_unknown_and_grammar(tmp_path):
    fit = _hand_fit(tmp_path, status=None)
    manifest = run_rerun(fit, tmp_path / "out", bracket=("pooled",), seeds_count=1, processes=1,
                         cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert "from an unknown fit;" in manifest["status"]
    assert fit_status({"input": {}}) == "unknown"
    assert fit_status({}) == "unknown"
    for status, phrase in (("synthetic", "from a synthetic fit;"), ("measured", "from a measured fit;"),
                           ("fixture", "from a fixture fit;"), ("unknown", "from an unknown fit;")):
        assert phrase in _status_text(status)


def test_readme_is_written_and_hashed(fit_file, tmp_path):
    out = tmp_path / "out"
    manifest = run_rerun(fit_file, out, seeds_count=1, processes=1, cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    text = (out / "README.txt").read_text()
    stored = json.loads((out / "manifest.json").read_text())
    assert text.splitlines()[0] == manifest["status"] == stored["status"]
    assert manifest["fit_sha256"] == hashlib.sha256(fit_file.read_bytes()).hexdigest()
    assert manifest["fit_sha256"] in text and str(fit_file) in text
    for item in manifest["bracket"]:
        assert item["label"] in text and repr(item["noise_std_m"]) in text
    assert "manifest.json" in text and "no label of their own" in text
    assert stored["outputs"]["README.txt"] == hashlib.sha256((out / "README.txt").read_bytes()).hexdigest()
    # The fit's own caveats travel with the outputs, word for word.
    fit = json.loads(fit_file.read_text())
    assert stored["assumptions"] == fit["assumptions"] and stored["assumptions"]
    assert stored["evidence_boundary"] == fit["evidence_boundary"]
    assert "camera" in stored["depth_axis_rule"] and stored["depth_axis"] == "z"


def test_caveats_null_when_absent_and_depth_axis_only_when_asked(tmp_path):
    fit = _hand_fit(tmp_path)
    manifest = run_rerun(fit, tmp_path / "out", bracket=("pooled",), seeds_count=1, processes=1,
                         cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert manifest["assumptions"] is None and manifest["evidence_boundary"] is None
    assert manifest["depth_axis"] is None
    assert isinstance(manifest["depth_axis_rule"], str)


def test_second_runs_root_is_refused(tmp_path, monkeypatch):
    fit = _hand_fit(tmp_path)
    main_checkout = tmp_path / "main-checkout"
    monkeypatch.setattr(rerun_module, "_git_common_dir", lambda: main_checkout / ".git")
    assert (main_checkout / "runs").resolve() in protected_runs_dirs()
    target = main_checkout / "runs" / "zz-rerun"
    with pytest.raises(RerunError, match="runs"):
        run_rerun(fit, target, seeds_count=1, processes=1)
    assert not main_checkout.exists()
    # Without git only this checkout's runs/ is protected.
    monkeypatch.setattr(rerun_module, "_git_common_dir", lambda: None)
    assert protected_runs_dirs() == [(REPO_ROOT / "runs").resolve()]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_real_main_checkout_runs_is_protected():
    try:
        common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                         text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git cannot read this checkout")
    assert (Path(common).parent / "runs").resolve() in protected_runs_dirs()


def test_off_grid_noise_gets_its_own_cell_id():
    registered_ids = {cell["cell_id"] for cell in build_cells(False)}
    grid = [cell["cell_id"] for cell in build_rerun_cells(0.05)]
    assert set(grid) <= registered_ids
    off_grid = [cell["cell_id"] for cell in build_rerun_cells(0.0504)]
    assert not set(off_grid) & registered_ids
    assert all("/n0.0504/" in cell_id for cell_id in off_grid)
    assert len(set(off_grid)) == len(off_grid) == 56
    long_value = 0.010020132273963072
    assert all(f"/n{long_value!r}/" in cell["cell_id"] for cell in build_rerun_cells(long_value))
    # Every registered noise level keeps its registered id.
    for noise in (0.0, 0.02, 0.05, 0.10):
        assert {cell["cell_id"] for cell in build_rerun_cells(noise)} <= registered_ids


def test_failed_run_leaves_no_folder(tmp_path, monkeypatch):
    fit = _hand_fit(tmp_path)
    out = tmp_path / "out"

    def boom(cell, result):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(rerun_module, "_cell_summary", boom)
    with pytest.raises(RuntimeError, match="summary failed"):
        run_rerun(fit, out, seeds_count=1, processes=1, cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert not out.exists()


def test_folder_filled_during_run_is_not_overwritten(tmp_path, monkeypatch):
    fit = _hand_fit(tmp_path)
    out = tmp_path / "out"
    real = rerun_module._cell_summary

    def late_writer(cell, result):
        out.mkdir(exist_ok=True)
        (out / "keep.txt").write_text("keep")
        return real(cell, result)

    monkeypatch.setattr(rerun_module, "_cell_summary", late_writer)
    with pytest.raises(RerunError, match="not empty"):
        run_rerun(fit, out, seeds_count=1, processes=1, cell_filter=_only_cell("constant_velocity", 0.1, 0.3))
    assert [path.name for path in out.iterdir()] == ["keep.txt"]
