"""Rerun the S6 recovery cell with a radio link, a vision latency and a vision noise fit applied together.

Inputs (each optional, at least one needed):

--radio LINK.json         a zephyr-espnow-fit-1 file (its s6_deliver block) or a small JSON with
                          delay_s, jitter_s and loss_probability, read the same way as
                          tools/s6_link_rerun.py reads it.
--vision-link LINK.json   a zephyr-vision-latency-link-1 file from tools/vision_latency_to_link.py.
--vision-fit FIT.json     a zephyr-vision-noise-fit-1 file; --noise picks the pooled value or the
                          depth-axis value, the same way tools/s6_measured_rerun.py does.

Combining rule (an assumption, written into every output):

- delays add: the measurement waits for the camera and detector, then for the radio;
- jitters add in quadrature, which assumes the two delay spreads are independent;
- loss and the loss model come from the radio (the vision link never drops packets);
- noise comes from the fit; without a fit the sensor has no noise;
- tracker measurement std stays 0.05 m as registered, unless --tracker-r measured sets it to the
  fitted noise.

The S6 link draws a lognormal delay with the combined mean and std. The sum of two delay
distributions is not lognormal in general, so only its mean and std are kept.

The recovery arm stamps each packet with its capture time. Treating the vision latency as link
delay therefore assumes the capture stamp is the true exposure time (the camera latency is
compensated in the stamp). A stamp taken at frame read without that correction is a stamp bias,
which this tool does not model.

The output status is the weakest input status: synthetic (or fixture) if any input is synthetic,
measured only if every given input is measured, otherwise unknown. Missing inputs are listed as
not included, and the "complete" field is true only when all three inputs were given. Motion, the link timing model and the tracker are always simulated. Output goes to a
new or empty folder, never under sim/runs/.

    PYTHONPATH=. .venv/bin/python tools/s6_combined_rerun.py --radio radio.json \\
        --vision-link link.json --vision-fit fit.json --output DIR [--tracker-r measured] [--pilot]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list
from tools.run_s6_recovery import _cell_summary, _git_value, _sha256, _write_csv
from tools.s6_link_rerun import LinkRerunError, describe_input, parse_link
from tools.s6_measured_rerun import RerunError, protected_runs_dirs, resolve_brackets
from tools.vision_latency_to_link import SCHEMA as VISION_LINK_SCHEMA
from tools.vision_latency_to_link import combine_status

SUMMARY_SCHEMA = "zephyr-s6-combined-rerun-1"
MANIFEST_SCHEMA = "zephyr-s6-combined-rerun-manifest-1"
REGISTERED_TRACKER_STD_M = 0.05
BASE_SEED = 20261013
PILOT_SEEDS = 5
MOTIONS = ("constant_velocity", "constant_turn")
HASHED_SOURCES = ("sim/tracker.py", "sim/s6_recovery.py", "sim/link.py", "sim/sensor.py",
                  "tools/run_s6_recovery.py", "tools/s6_link_rerun.py", "tools/s6_measured_rerun.py",
                  "tools/s6_combined_rerun.py")
COMBINING_RULE = [
    "delays add (camera and detector first, then radio)",
    "jitters add in quadrature (the two delay spreads are assumed independent)",
    "loss and loss model come from the radio; the vision link drops nothing",
    "noise comes from the vision fit; without a fit the sensor has no noise",
    "the combined delay is drawn lognormal with the combined mean and std; the true sum is not lognormal",
    "the capture stamp is assumed to be the true exposure time (camera latency compensated)",
]


class CombinedRerunError(ValueError):
    """An input, an option, or the output folder cannot be used."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
            or float(value) < 0:
        raise CombinedRerunError(f"{name} must be a finite number that is zero or more")
    return float(value)


def _status(value: Any) -> str:
    """Map a declared status onto measured, synthetic or unknown (fixture counts as synthetic)."""
    if value == "fixture":
        return "synthetic"
    return value if value in ("measured", "synthetic") else "unknown"


def radio_component(data: Any, raw: bytes, name: str) -> dict[str, Any]:
    try:
        link = parse_link(data)
    except LinkRerunError as exc:
        raise CombinedRerunError(f"radio input: {exc}") from exc
    provenance = describe_input(data, raw, name)
    return {"included": True, "link": link, "declared_status": provenance["input_status"],
            "status": _status(provenance["input_status"]), "provenance": provenance}


def vision_link_component(data: Any, raw: bytes, name: str) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != VISION_LINK_SCHEMA:
        schema = data.get("schema") if isinstance(data, dict) else None
        raise CombinedRerunError(f"vision link must have schema {VISION_LINK_SCHEMA}, got {schema!r}")
    delay = _number(data.get("delay_s"), "vision link delay_s")
    jitter = _number(data.get("jitter_s", 0.0), "vision link jitter_s")
    if _number(data.get("loss_probability", 0.0), "vision link loss_probability") != 0.0:
        raise CombinedRerunError("vision link loss_probability must be 0; only the radio drops packets here")
    warnings = data.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise CombinedRerunError("vision link warnings must be a list of strings")
    return {"included": True, "delay_s": delay, "jitter_s": jitter, "declared_status": data.get("status"),
            "status": _status(data.get("status")), "source": data.get("source"), "warnings": list(warnings),
            "file": name, "sha256": hashlib.sha256(raw).hexdigest()}


def noise_component(fit: Any, raw: bytes, name: str, label: str = "pooled",
                    depth_axis: str | None = None) -> dict[str, Any]:
    try:
        (entry,) = resolve_brackets(fit, (label,), depth_axis)
    except RerunError as exc:
        raise CombinedRerunError(f"vision fit: {exc}") from exc
    declared = fit.get("input", {}).get("status")
    return {"included": True, "noise_std_m": entry["noise_std_m"], "label": entry["label"],
            "source": entry["source"], "declared_status": declared, "status": _status(declared),
            "not_applied": {"scenario_delay": fit.get("scenario_delay"),
                            "note": "the fit's detector delay is not added here; use it through the vision "
                                    "link (vision_latency_to_link.py --vision-fit) so it is counted once"},
            "file": name, "sha256": hashlib.sha256(raw).hexdigest()}


def combine(radio: dict[str, Any] | None, vision_link: dict[str, Any] | None,
            noise: dict[str, Any] | None, tracker_r: str = "registered") -> dict[str, Any]:
    """Apply the combining rule. Returns the deliver and tracker settings plus the combined status."""
    parts = {"radio": radio, "vision_link": vision_link, "vision_fit": noise}
    if all(part is None for part in parts.values()):
        raise CombinedRerunError("give at least one of --radio, --vision-link and --vision-fit")
    radio_link = radio["link"] if radio else {"delay_s": 0.0, "jitter_s": 0.0, "loss_probability": 0.0,
                                              "loss_model": "independent"}
    vision_delay = vision_link["delay_s"] if vision_link else 0.0
    vision_jitter = vision_link["jitter_s"] if vision_link else 0.0
    delay = radio_link["delay_s"] + vision_delay
    jitter = math.hypot(radio_link["jitter_s"], vision_jitter)
    if jitter > 0 and delay == 0:
        raise CombinedRerunError("the combined jitter is positive but the combined delay is zero")
    noise_std = noise["noise_std_m"] if noise else 0.0
    if tracker_r == "measured":
        if noise is None:
            raise CombinedRerunError("--tracker-r measured needs --vision-fit")
        if noise_std <= 0:
            raise CombinedRerunError("--tracker-r measured needs a positive fitted noise")
        tracker_std = noise_std
    elif tracker_r == "registered":
        tracker_std = REGISTERED_TRACKER_STD_M
    else:
        raise CombinedRerunError("--tracker-r must be registered or measured")
    settings = {"delay_s": delay, "jitter_s": jitter, "loss_probability": radio_link["loss_probability"],
                "loss_model": radio_link["loss_model"], "noise_std_m": noise_std,
                "tracker_measurement_std_m": tracker_std, "tracker_r": tracker_r}
    if radio_link["loss_model"] == "burst":
        settings["burst_mean_length_packets"] = radio_link["burst_mean_length_packets"]
    included = [name for name, part in parts.items() if part is not None]
    status = combine_status([part["status"] for part in parts.values() if part is not None])
    return {"settings": settings, "status": status, "included": included,
            "not_included": [name for name in parts if name not in included],
            "complete": len(included) == len(parts)}


def status_text(combined: dict[str, Any]) -> str:
    missing = combined["not_included"]
    text = (f"investigation: combined S6 rerun from {combined['status']} inputs "
            f"({', '.join(combined['included'])})")
    if missing:
        text += f"; not included: {', '.join(missing)}"
    return text + "; motion, link timing model and tracker simulated, not flight performance"


def _id_part(value: float) -> str:
    return repr(float(value))


def build_cells(settings: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for motion in MOTIONS:
        cell = {"slice": "combined", "motion": motion, "delay_s": settings["delay_s"],
                "loss_probability": settings["loss_probability"], "noise_std_m": settings["noise_std_m"],
                "loss_model": settings["loss_model"], "jitter_s": settings["jitter_s"],
                "tracker_measurement_std_m": settings["tracker_measurement_std_m"]}
        cell_id = (f"combined/{motion}/d{_id_part(cell['delay_s'])}/l{_id_part(cell['loss_probability'])}/"
                   f"n{_id_part(cell['noise_std_m'])}/{cell['loss_model']}/j{_id_part(cell['jitter_s'])}/"
                   f"r{_id_part(cell['tracker_measurement_std_m'])}")
        if settings["loss_model"] == "burst":
            cell["burst_mean_length_packets"] = settings["burst_mean_length_packets"]
            cell_id += f"/b{_id_part(cell['burst_mean_length_packets'])}"
        cell["cell_id"] = cell_id
        cells.append(cell)
    return cells


def run_cell(cell: dict[str, Any], seeds: list[int], config: RecoveryConfig) -> dict[str, list]:
    naive, recovery = [], []
    for seed in seeds:
        kwargs = {"delay_s": cell["delay_s"], "loss_probability": cell["loss_probability"],
                  "noise_std_m": cell["noise_std_m"], "seed": seed, "config": config,
                  "loss_model": cell["loss_model"], "jitter_s": cell["jitter_s"]}
        if cell["loss_model"] == "burst":
            kwargs["burst_mean_length_packets"] = cell["burst_mean_length_packets"]
        arms = run_arms(deliver(cell["motion"], **kwargs), cell["motion"], config,
                        tracker_measurement_std_m=cell["tracker_measurement_std_m"])
        naive.append(replace(arms["naive"], query_times_s=None, errors_m=None))
        recovery.append(replace(arms["recovery"], query_times_s=None, errors_m=None))
    return {"naive": naive, "recovery": recovery}


def _read(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8")), raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CombinedRerunError(f"cannot read {path.name}: {exc}") from exc


def check_output(output: Path, inputs: list[Path]) -> Path:
    resolved = output.resolve()
    for runs_dir in protected_runs_dirs():
        if resolved == runs_dir or resolved.is_relative_to(runs_dir):
            raise CombinedRerunError(f"output must not be inside {runs_dir}; registered outputs live there")
    for path in inputs:
        if path.resolve().is_relative_to(resolved):
            raise CombinedRerunError(f"input {path.name} must not be inside the output folder")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise CombinedRerunError("output must be a new or empty directory")
    return resolved


def _readme(status: str, combined: dict[str, Any], components: dict[str, Any]) -> str:
    settings = combined["settings"]
    lines = [status, "", "applied settings:"]
    lines += [f"  {key}: {settings[key]!r}" for key in sorted(settings)]
    lines += ["", "inputs:"]
    for name, part in components.items():
        if part is None:
            lines.append(f"  {name}: not included")
        else:
            sha = part.get("sha256") or part["provenance"]["input_sha256"]
            lines.append(f"  {name}: status {part['status']} (declared {part['declared_status']!r}), sha256 {sha}")
            if part.get("source"):
                lines.append(f"    source: {part['source']}")
            lines += [f"    warning: {warning}" for warning in part.get("warnings", [])]
    lines += ["", "combining rule (assumptions):"] + [f"  - {rule}" for rule in COMBINING_RULE]
    lines += ["", "per_seed.csv carries no label of its own; this README and manifest.json are the label.", ""]
    return "\n".join(lines)


def run_rerun(output: Path, *, radio_path: Path | None = None, vision_link_path: Path | None = None,
              vision_fit_path: Path | None = None, noise_label: str = "pooled", depth_axis: str | None = None,
              tracker_r: str = "registered", seeds_count: int = 50, base_seed: int = BASE_SEED,
              pilot: bool = False, argv: list[str] | None = None) -> dict[str, Any]:
    """Run both motions with the combined settings and write the outputs. Returns the manifest."""
    if pilot:
        seeds_count = PILOT_SEEDS
    if seeds_count < 1:
        raise CombinedRerunError("seeds must be positive")
    given = [Path(p) for p in (radio_path, vision_link_path, vision_fit_path) if p is not None]
    resolved = check_output(Path(output), given)
    components: dict[str, Any] = {"radio": None, "vision_link": None, "vision_fit": None}
    if radio_path is not None:
        data, raw = _read(Path(radio_path))
        components["radio"] = radio_component(data, raw, Path(radio_path).name)
    if vision_link_path is not None:
        data, raw = _read(Path(vision_link_path))
        components["vision_link"] = vision_link_component(data, raw, Path(vision_link_path).name)
    if vision_fit_path is not None:
        data, raw = _read(Path(vision_fit_path))
        components["vision_fit"] = noise_component(data, raw, Path(vision_fit_path).name, noise_label, depth_axis)
    combined = combine(components["radio"], components["vision_link"], components["vision_fit"], tracker_r)
    status = status_text(combined)
    cells = build_cells(combined["settings"])
    seeds = seed_list(seeds_count, base_seed=base_seed)
    config = RecoveryConfig()
    code_sha256 = {relative: _sha256(_REPO_ROOT / relative) for relative in HASHED_SOURCES}
    git_revision = _git_value(["rev-parse", "HEAD"]) or None
    porcelain = _git_value(["status", "--porcelain"])
    git_tree_state = "unknown" if porcelain is None else ("dirty" if porcelain else "clean")

    start = time.perf_counter()
    results = [run_cell(cell, seeds, config) for cell in cells]
    runtime = time.perf_counter() - start

    limits = ["combined settings are an assumption-based sum, see combining_rule",
              (f"tracker measurement std {combined['settings']['tracker_measurement_std_m']!r} m "
               f"({'fitted noise' if tracker_r == 'measured' else 'registered value'})"),
              "fitted bias, dropout and per-axis shape are not applied; noise is isotropic",
              f"{seeds_count} seeds per motion"]
    limits += [f"{name} not included: contributes no delay, jitter, loss or noise" for name in combined["not_included"]]
    if components["vision_link"] is not None:
        limits += [f"vision link warning: {warning}" for warning in components["vision_link"]["warnings"]]
    summary = {"schema": SUMMARY_SCHEMA, "status": status, "input_status": combined["status"],
               "complete": combined["complete"],
               "generator": "tools/s6_combined_rerun.py", "settings": combined["settings"],
               "included": combined["included"], "not_included": combined["not_included"],
               "combining_rule": COMBINING_RULE, "config": {"duration_s": config.duration_s,
                                                             "observation_period_s": config.observation_period_s},
               "design": {"pilot": pilot, "n_seeds": seeds_count, "base_seed": base_seed, "seeds": seeds},
               "cells": [_cell_summary(cell, result) for cell, result in zip(cells, results)], "limits": limits}

    # Write into a hidden sibling folder and rename it at the end, so a failed write never
    # leaves a half-filled output folder behind.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    staging = resolved.parent / f".{resolved.name}.partial-{os.getpid()}"
    staging.mkdir()
    summary_path = staging / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = staging / "per_seed.csv"
    _write_csv(csv_path, cells, results, seeds)
    readme_path = staging / "README.txt"
    readme_path.write_text(_readme(status, combined, components))
    if argv is None:
        argv = ["--output", str(output)]
        for flag, path in (("--radio", radio_path), ("--vision-link", vision_link_path),
                           ("--vision-fit", vision_fit_path)):
            if path is not None:
                argv += [flag, str(path)]
        argv += ["--noise", noise_label, "--tracker-r", tracker_r, "--seeds", str(seeds_count),
                 "--base-seed", str(base_seed)]
        if depth_axis is not None:
            argv += ["--depth-axis", depth_axis]
    manifest = {"schema": MANIFEST_SCHEMA, "status": status, "input_status": combined["status"],
                "complete": combined["complete"],
                "components": components, "settings": combined["settings"], "combining_rule": COMBINING_RULE,
                "git_revision": git_revision, "git_tree_state": git_tree_state, "code_sha256": code_sha256,
                "python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform(), "seeds_count": seeds_count, "base_seed": base_seed,
                "pilot": pilot, "runtime_s": runtime, "argv": argv,
                "replay_command": "PYTHONPATH=. .venv/bin/python tools/s6_combined_rerun.py " + shlex.join(argv),
                "outputs": {path.name: _sha256(path) for path in (summary_path, csv_path, readme_path)}}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if resolved.exists():
        resolved.rmdir()  # check_output already made sure it is empty
    staging.rename(resolved)
    return manifest


def summary_lines(summary: dict[str, Any]) -> list[str]:
    s = summary["settings"]
    lines = [summary["status"],
             (f"delay {1000 * s['delay_s']:.1f} ms, jitter {1000 * s['jitter_s']:.1f} ms, loss {s['loss_probability']:.3f} "
              f"({s['loss_model']}), noise {s['noise_std_m']:.4f} m, tracker R {s['tracker_measurement_std_m']:.4f} m")]
    for cell in summary["cells"]:
        arms = cell["arms"]
        lines.append(f"  {cell['motion']}: median p95 error naive {arms['naive']['p95_median_m']:.4f} m, "
                     f"recovery {arms['recovery']['p95_median_m']:.4f} m")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radio", type=Path, help="espnow fit or small link JSON")
    parser.add_argument("--vision-link", type=Path, help="zephyr-vision-latency-link-1 JSON")
    parser.add_argument("--vision-fit", type=Path, help="zephyr-vision-noise-fit-1 JSON")
    parser.add_argument("--noise", choices=("pooled", "depth"), default="pooled",
                        help="which fitted noise value to apply (default pooled)")
    parser.add_argument("--depth-axis", choices=("x", "y", "z"), help="axis for --noise depth")
    parser.add_argument("--tracker-r", choices=("registered", "measured"), default="registered",
                        help="tracker measurement std: registered 0.05 m or the fitted noise")
    parser.add_argument("--output", type=Path, required=True, help="new or empty folder, never under sim/runs/")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--pilot", action="store_true", help=f"run {PILOT_SEEDS} seeds per motion")
    args = parser.parse_args(argv)
    try:
        run_rerun(args.output, radio_path=args.radio, vision_link_path=args.vision_link,
                  vision_fit_path=args.vision_fit, noise_label=args.noise, depth_axis=args.depth_axis,
                  tracker_r=args.tracker_r, seeds_count=args.seeds, base_seed=args.base_seed, pilot=args.pilot,
                  argv=list(sys.argv[1:] if argv is None else argv))
    except (CombinedRerunError, OSError) as exc:
        parser.error(str(exc))
    summary = json.loads((args.output / "summary.json").read_text())
    print("\n".join(summary_lines(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
