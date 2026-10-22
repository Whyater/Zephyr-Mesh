"""Rerun the S6 main slice with the noise from a vision-noise fit (a proposal, not a result).

Reads a "zephyr-vision-noise-fit-1" file, takes its fitted position noise, and
reruns the S6 recovery main slice (independent loss, no jitter, tracker
measurement std 0.05 m) with that noise in place of the registered noise
levels. Two noise values can be bracketed: the pooled value and the depth-axis
value. Output goes to a new folder; registered outputs are never overwritten.

The link and the tracker are simulated, and only the noise is taken from the
fit. Run it with:

    PYTHONPATH=. ./.venv/bin/python tools/s6_measured_rerun.py FIT.json --output DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.s6_recovery import RecoveryConfig, seed_list
from sim.vision_trace import TRACE_STATUSES
from tools.run_s6_recovery import (
    MAIN_DELAYS,
    MAIN_LOSSES,
    MAIN_NOISES,
    _cell_summary,
    _csv_value,
    _git_value,
    _run_cell,
    _sha256,
    _write_csv,
    build_cells,
)

FIT_SCHEMA = "zephyr-vision-noise-fit-1"
SUMMARY_SCHEMA = "zephyr-s6-measured-rerun-1"
MANIFEST_SCHEMA = "zephyr-s6-measured-rerun-manifest-1"
HASHED_SOURCES = ("sim/tracker.py", "sim/s6_recovery.py", "sim/link.py", "sim/sensor.py",
                  "tools/run_s6_recovery.py", "tools/s6_measured_rerun.py")
BRACKET_LABELS = ("pooled", "depth")
REGISTERED_SUMMARY = Path("sim", "runs") / "s6-recovery" / "summary.json"
BASE_SEED = 20261013
CAMERA_DEPTH_AXIS = "z"
DEPTH_AXIS_RULE = (
    'The default depth axis is z only when position_frame is "camera" (the OpenCV camera frame, z out of the '
    "lens); for any other frame the depth axis must be given."
)


class RerunError(ValueError):
    """The fit file, the options, or the output folder cannot be used."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _noise_value(value: Any, name: str) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise RerunError(f"{name} must be a finite number that is zero or more, got {value!r}")
    return float(value)


def validate_fit(fit: Any) -> None:
    """Check that a fit file has the pieces this tool reads."""
    if not isinstance(fit, dict):
        raise RerunError("fit file must hold a JSON object")
    if fit.get("schema") != FIT_SCHEMA:
        raise RerunError(f"fit schema must be {FIT_SCHEMA!r}, got {fit.get('schema')!r}")
    sensor_config = fit.get("sensor_config")
    if not isinstance(sensor_config, dict) or "noise_std_m" not in sensor_config:
        raise RerunError("fit file has no sensor_config.noise_std_m")
    _noise_value(sensor_config["noise_std_m"], "sensor_config.noise_std_m")
    per_axis = fit.get("per_axis")
    if not isinstance(per_axis, dict):
        raise RerunError("fit file has no per_axis block")
    axes = per_axis.get("axes")
    std = per_axis.get("std_m")
    if not isinstance(axes, list) or len(axes) != 3 or not all(isinstance(name, str) for name in axes):
        raise RerunError("per_axis.axes must list three axis names")
    if len(set(axes)) != 3:
        raise RerunError("per_axis.axes names must be distinct")
    if not isinstance(std, list) or len(std) != 3:
        raise RerunError("per_axis.std_m must list three values")
    for name, value in zip(axes, std):
        _noise_value(value, f"per_axis.std_m[{name}]")
    input_block = fit.get("input", {})
    if not isinstance(input_block, dict):
        raise RerunError("fit input block must be an object")
    if "status" in input_block and (not isinstance(input_block["status"], str)
                                    or input_block["status"] not in TRACE_STATUSES):
        raise RerunError(f"fit input.status must be one of {sorted(TRACE_STATUSES)}, got {input_block['status']!r}")


def resolve_brackets(fit: dict[str, Any], bracket: tuple[str, ...] = BRACKET_LABELS,
                     depth_axis: str | None = None) -> list[dict[str, Any]]:
    """Return one {label, noise_std_m, source} entry per requested bracket."""
    validate_fit(fit)
    labels = tuple(bracket)
    if not labels:
        raise RerunError("choose at least one bracket")
    unknown = [label for label in labels if label not in BRACKET_LABELS]
    if unknown:
        raise RerunError(f"unknown bracket {unknown[0]!r}; choose from {', '.join(BRACKET_LABELS)}")
    if len(set(labels)) != len(labels):
        raise RerunError("each bracket can be named once")
    entries = []
    for label in labels:
        if label == "pooled":
            entries.append({"label": "pooled", "noise_std_m": float(fit["sensor_config"]["noise_std_m"]),
                            "source": "sensor_config.noise_std_m"})
            continue
        axes = fit["per_axis"]["axes"]
        frame = fit["per_axis"].get("position_frame")
        axis = depth_axis
        if axis is None:
            if frame != "camera":
                raise RerunError(
                    f"the depth axis is only known for a camera-frame fit (z points out of the lens), but this fit "
                    f"has position_frame {frame!r}; pass --depth-axis x, y or z")
            axis = CAMERA_DEPTH_AXIS
        if axis not in axes:
            raise RerunError(f"depth axis {axis!r} is not one of the fit axes {axes}")
        entries.append({"label": "depth", "noise_std_m": float(fit["per_axis"]["std_m"][axes.index(axis)]),
                        "source": f"per_axis.std_m[{axis}]"})
    return entries


def _noise_part(noise: float) -> str:
    """The registered three-decimal text when it is exact, else the full value."""
    text = f"{noise:.3f}"
    return text if float(text) == noise else repr(float(noise))


def build_rerun_cells(noise_std_m: float) -> list[dict[str, Any]]:
    """The registered S6 main slice, with its noise replaced by ``noise_std_m``.

    One cell per motion, delay and loss, in registered order. Every key, value
    type and the cell_id format are copied from the registered grid.
    """
    base = [cell for cell in build_cells(False) if cell["slice"] == "main" and cell["noise_std_m"] == 0.0]
    cells = []
    for cell in base:
        new = dict(cell)
        new["noise_std_m"] = float(noise_std_m)
        new["cell_id"] = (
            f"{new['slice']}/{new['motion']}/d{new['delay_s']:.3f}/l{new['loss_probability']:.2f}/"
            f"n{_noise_part(new['noise_std_m'])}/{new['loss_model']}/j{new['jitter_s']:.3f}/r{new['tracker_measurement_std_m']:.3f}"
        )
        cells.append(new)
    expected = 2 * len(MAIN_DELAYS) * len(MAIN_LOSSES)
    if len(cells) != expected:
        raise RerunError(f"expected {expected} main-slice cells, built {len(cells)}")
    return cells


def _git_common_dir() -> Path | None:
    """The shared .git folder (the main checkout's, inside a worktree), or None without git."""
    try:
        text = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                       text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(text) if text else None


def protected_runs_dirs() -> list[Path]:
    """The sim/runs/ folders that hold registered outputs: this checkout's and the main checkout's."""
    dirs = [(REPO_ROOT / "sim" / "runs").resolve()]
    common = _git_common_dir()
    if common is not None:
        main_runs = (common.parent / "sim" / "runs").resolve()
        if main_runs not in dirs:
            dirs.append(main_runs)
    return dirs


def _require_fresh_folder(resolved: Path) -> None:
    if resolved.exists():
        if not resolved.is_dir():
            raise RerunError(f"output path is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise RerunError(f"output folder is not empty: {resolved}; pick a new folder")


def check_paths(fit_path: Path, output: Path) -> Path:
    """Refuse unsafe output folders. Returns the resolved output path."""
    resolved = output.resolve()
    for runs_dir in protected_runs_dirs():
        if resolved == runs_dir or resolved.is_relative_to(runs_dir):
            raise RerunError(f"output must not be inside {runs_dir}; registered outputs live there")
    if fit_path.resolve().is_relative_to(resolved):
        raise RerunError("the fit file must not be inside the output folder")
    _require_fresh_folder(resolved)
    return resolved


def _read_registered(noises: tuple[float, ...]) -> tuple[dict[tuple, dict[str, Any]], dict[str, Any]]:
    """Registered main-slice arms keyed by (motion, delay, loss, noise)."""
    path = REPO_ROOT / REGISTERED_SUMMARY
    if not path.is_file():
        return {}, {"path": REGISTERED_SUMMARY.as_posix(), "status": "missing; registered columns are blank"}
    data = json.loads(path.read_text())
    table = {}
    for cell in data.get("cells", []):
        if cell.get("slice") == "main" and cell.get("loss_model") == "independent" and cell["noise_std_m"] in noises:
            table[(cell["motion"], cell["delay_s"], cell["loss_probability"], cell["noise_std_m"])] = cell["arms"]
    return table, {"path": REGISTERED_SUMMARY.as_posix(), "status": "found", "sha256": _sha256(path)}


def _write_comparison(path: Path, labels: list[str], noises: dict[str, float],
                      summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    registered, info = _read_registered(MAIN_NOISES)
    grid: list[tuple[str, float, float]] = []
    by_label: dict[str, dict[tuple, dict[str, Any]]] = {}
    for label in labels:
        by_label[label] = {}
        for entry in summaries[label]:
            key = (entry["motion"], entry["delay_s"], entry["loss_probability"])
            by_label[label][key] = entry["arms"]
            if key not in grid:
                grid.append(key)
    fields = ["motion", "delay_s", "loss_probability"]
    for label in labels:
        fields += [f"{label}_noise_std_m", f"{label}_recovery_p95_median_m", f"{label}_naive_p95_median_m"]
    for noise in MAIN_NOISES:
        fields += [f"registered_n{noise:.3f}_recovery_p95_median_m", f"registered_n{noise:.3f}_naive_p95_median_m"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in grid:
            row: dict[str, Any] = {"motion": key[0], "delay_s": key[1], "loss_probability": key[2]}
            for label in labels:
                arms = by_label[label].get(key)
                row[f"{label}_noise_std_m"] = noises[label]
                row[f"{label}_recovery_p95_median_m"] = None if arms is None else arms["recovery"]["p95_median_m"]
                row[f"{label}_naive_p95_median_m"] = None if arms is None else arms["naive"]["p95_median_m"]
            for noise in MAIN_NOISES:
                arms = registered.get((*key, noise))
                row[f"registered_n{noise:.3f}_recovery_p95_median_m"] = None if arms is None else arms["recovery"]["p95_median_m"]
                row[f"registered_n{noise:.3f}_naive_p95_median_m"] = None if arms is None else arms["naive"]["p95_median_m"]
            writer.writerow({name: _csv_value(row.get(name)) for name in fields})
    return info


def fit_status(fit: dict[str, Any]) -> str:
    """The fit's input status, or "unknown" when the fit does not give one."""
    block = fit.get("input", {})
    return block.get("status", "unknown")


def _status_text(input_status: str) -> str:
    article = "an" if input_status[:1] in "aeiou" else "a"
    return (f"proposal: S6 main slice rerun with noise from {article} {input_status} fit; "
            "simulated link and tracker, not flight performance")


def _readme_text(status: str, fit_path: Path, fit_sha256: str, entries: list[dict[str, Any]]) -> str:
    lines = [status, "", f"input fit: {fit_path}", f"input fit sha256: {fit_sha256}", "", "bracket values:"]
    lines += [f"  {item['label']}: noise {item['noise_std_m']!r} m ({item['source']})" for item in entries]
    lines += [
        "",
        "files:",
        "  <bracket>/summary.json: the S6 main-slice cell summaries for that noise value, with status and limits.",
        "  <bracket>/per_seed.csv: one row per cell, arm and seed, in the registered column layout.",
        "  comparison.csv: median p95 error per cell for each bracket next to the registered noise levels.",
        "  manifest.json: fit hash, code hashes, git state, options, and a hash of every output file.",
        "",
        "comparison.csv and per_seed.csv carry no label of their own. This README and manifest.json are the label:",
        "every number here comes from a simulated link and tracker with the noise taken from the fit above,",
        "and none of it is flight performance.",
        "",
    ]
    return "\n".join(lines)


def run_rerun(fit_path: Path, output: Path, *, bracket: tuple[str, ...] = BRACKET_LABELS,
              depth_axis: str | None = None, seeds_count: int = 50, base_seed: int = BASE_SEED,
              processes: int = 2, cell_filter: Callable[[dict[str, Any]], bool] | None = None,
              argv: list[str] | None = None) -> dict[str, Any]:
    """Run the rerun and write all outputs. Returns the manifest."""
    fit_path = Path(fit_path)
    output = Path(output)
    if seeds_count < 1:
        raise RerunError("seeds must be positive")
    if processes < 1:
        raise RerunError("processes must be positive")
    resolved_output = check_paths(fit_path, output)
    try:
        fit_bytes = fit_path.read_bytes()
        fit = json.loads(fit_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RerunError(f"cannot read fit file: {exc}") from exc
    entries = resolve_brackets(fit, tuple(bracket), depth_axis)

    labels = [entry["label"] for entry in entries]
    noises = {entry["label"]: entry["noise_std_m"] for entry in entries}
    cells_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        cells = build_rerun_cells(noises[label])
        if cell_filter is not None:
            cells = [cell for cell in cells if cell_filter(cell)]
        if not cells:
            raise RerunError("the cell filter selected no cells")
        cells_by_label[label] = cells
    seeds = seed_list(seeds_count, base_seed=base_seed)
    config = RecoveryConfig()
    input_block = fit.get("input", {})
    status = _status_text(fit_status(fit))

    # Provenance is captured before anything is written so the new output
    # files cannot make the tree look dirty.
    code_sha256 = {relative: _sha256(REPO_ROOT / relative) for relative in HASHED_SOURCES}
    git_revision = _git_value(["rev-parse", "HEAD"]) or None
    porcelain = _git_value(["status", "--porcelain"])
    git_tree_state = "unknown" if porcelain is None else ("dirty" if porcelain else "clean")
    if argv is None:
        argv = [str(fit_path), "--output", str(output), "--bracket", ",".join(labels), "--seeds", str(seeds_count),
                "--base-seed", str(base_seed), "--processes", str(processes)]
        if depth_axis is not None:
            argv += ["--depth-axis", depth_axis]

    start = time.perf_counter()
    results: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {label: [pool.submit(_run_cell, cell, seeds, config) for cell in cells_by_label[label]]
                   for label in labels}
        for label in labels:
            results[label] = [future.result() for future in futures[label]]
    runtime = time.perf_counter() - start

    limits = [
        "tracker tuning stays at 0.05 m as registered",
        "only noise is applied; fitted bias, dropout, burst and latency are recorded but not applied",
        "the isotropic noise model applies the chosen value to all three axes",
        "simulated link and tracker, with a noise value fitted from one log (see the manifest for the log status)",
        "main slice only: independent loss, no jitter",
        f"{seeds_count} seeds per cell",
    ]
    summaries: dict[str, list[dict[str, Any]]] = {}
    summary_texts: dict[str, str] = {}
    for entry in entries:
        label = entry["label"]
        summaries[label] = [_cell_summary(cell, result) for cell, result in zip(cells_by_label[label], results[label])]
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": status,
            "bracket_label": label,
            "noise_std_m": entry["noise_std_m"],
            "noise_source": entry["source"],
            "generator": "tools/s6_measured_rerun.py",
            "config": asdict(config),
            "design": {"n_seeds": seeds_count, "base_seed": base_seed, "seeds": seeds},
            "cells": summaries[label],
            "limits": limits,
        }
        summary_texts[label] = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    # Nothing is created until every result is in, so a failed run leaves no folder behind.
    _require_fresh_folder(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for entry in entries:
        label = entry["label"]
        folder = resolved_output / label
        folder.mkdir()
        summary_path = folder / "summary.json"
        summary_path.write_text(summary_texts[label])
        csv_path = folder / "per_seed.csv"
        _write_csv(csv_path, cells_by_label[label], results[label], seeds)
        written += [summary_path, csv_path]
    comparison_path = resolved_output / "comparison.csv"
    registered_info = _write_comparison(comparison_path, labels, noises, summaries)
    written.append(comparison_path)
    fit_sha256 = hashlib.sha256(fit_bytes).hexdigest()
    readme_path = resolved_output / "README.txt"
    readme_path.write_text(_readme_text(status, fit_path, fit_sha256, entries))
    written.append(readme_path)

    sensor_config = fit["sensor_config"]
    not_applied = {
        "sensor_config": {key: value for key, value in sensor_config.items() if key != "noise_std_m"},
        "scenario_delay": fit.get("scenario_delay"),
        "note": "the S6 deliver() path takes only noise, so none of these values change the rerun",
    }
    uncertainty = fit.get("uncertainty")
    noise_interval = uncertainty.get("noise_std_m") if isinstance(uncertainty, dict) else None
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "fit_path": str(fit_path),
        "fit_sha256": fit_sha256,
        "fit_schema": fit["schema"],
        "fit_input": input_block,
        "assumptions": fit.get("assumptions"),
        "evidence_boundary": fit.get("evidence_boundary"),
        "bracket": entries,
        "noise_interval": noise_interval,
        "depth_axis": (depth_axis if depth_axis is not None else CAMERA_DEPTH_AXIS) if "depth" in labels else None,
        "depth_axis_rule": DEPTH_AXIS_RULE,
        "position_frame": fit["per_axis"].get("position_frame"),
        "not_applied": not_applied,
        "registered_summary": registered_info,
        "cell_filter_used": cell_filter is not None,
        "git_revision": git_revision,
        "git_tree_state": git_tree_state,
        "code_sha256": code_sha256,
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "platform": platform.platform(),
        "seeds_count": seeds_count,
        "base_seed": base_seed,
        "processes": processes,
        "n_cells": {label: len(cells_by_label[label]) for label in labels},
        "runtime_s": runtime,
        "argv": argv,
        "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/s6_measured_rerun.py " + shlex.join(argv),
        "outputs": {path.relative_to(resolved_output).as_posix(): _sha256(path) for path in written},
    }
    (resolved_output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _print_summary(entries, summaries, resolved_output)
    return manifest


def _print_summary(entries: list[dict[str, Any]], summaries: dict[str, list[dict[str, Any]]], output: Path) -> None:
    for entry in entries:
        label = entry["label"]
        p95 = [item["arms"]["recovery"]["p95_median_m"] for item in summaries[label]]
        print(f"{label}: noise {entry['noise_std_m']!r} m ({entry['source']}), {len(p95)} cells, "
              f"recovery median p95 error {min(p95):.4f} to {max(p95):.4f} m", file=sys.stderr)
    print(f"output: {output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fit", type=Path, help="zephyr-vision-noise-fit-1 JSON file")
    parser.add_argument("--output", type=Path, required=True, help="new or empty folder; never under sim/runs/")
    parser.add_argument("--bracket", default=",".join(BRACKET_LABELS), help="comma list from: pooled, depth")
    parser.add_argument("--depth-axis", choices=("x", "y", "z"), help="axis to use for the depth bracket")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args(argv)
    labels = tuple(part.strip() for part in args.bracket.split(",") if part.strip())
    try:
        run_rerun(args.fit, args.output, bracket=labels, depth_axis=args.depth_axis, seeds_count=args.seeds,
                  base_seed=args.base_seed, processes=args.processes,
                  argv=list(sys.argv[1:] if argv is None else argv))
    except (RerunError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
