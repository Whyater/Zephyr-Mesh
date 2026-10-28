"""Rerun the S6 main slice with one fitted link configuration.

The input is either a fit JSON produced by ``fit_espnow_log.py`` (using its
``s6_deliver`` block) or a small JSON object with ``delay_s``, ``jitter_s``
and ``loss_probability``. Results are always written to a new folder outside
``sim/runs``.

The summary label follows the input's declared status (``source_trace.status``
in a fit, ``status`` in a small JSON). Only ``measured`` gives a measured link
label; ``synthetic`` and ``fixture`` stay synthetic, and a missing or other
value is labelled unknown. The sensor is always simulated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import replace

from sim.s6_recovery import RecoveryConfig, deliver, run_arms, seed_list
from tools.run_s6_recovery import _cell_summary, _csv_value, _sha256

SCHEMA = "zephyr-s6-link-rerun-1"
MANIFEST_SCHEMA = "zephyr-s6-link-rerun-manifest-1"
FIT_SCHEMA = "zephyr-espnow-fit-1"
NOISES = (0.0, 0.02, 0.05, 0.10)
KNOWN_STATUSES = ("measured", "synthetic", "fixture")
SYNTHETIC_STATUS = "synthetic investigation: simulated link and sensor"
SYNTHETIC_LIMITS = ["simulated link and sensor", "main slice only", "no hardware data"]


class LinkRerunError(ValueError):
    pass


def _number(value: Any, name: str, *, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LinkRerunError(f"{name} must be a finite number")
    result = float(value)
    if result < 0 or (upper is not None and result > upper):
        bound = f" and at most {upper}" if upper is not None else ""
        raise LinkRerunError(f"{name} must be zero or more{bound}")
    return result


def parse_link(data: Any) -> dict[str, Any]:
    """Extract and validate the S6 deliver settings from a JSON object."""
    if not isinstance(data, dict):
        raise LinkRerunError("input must be a JSON object")
    if data.get("schema") == FIT_SCHEMA:
        block = data.get("s6_deliver")
        if not isinstance(block, dict):
            raise LinkRerunError("fit input has no s6_deliver object")
    else:
        block = data
    required = ("delay_s",)
    missing = [key for key in required if key not in block]
    if missing:
        raise LinkRerunError("link input is missing " + ", ".join(missing))
    delay = _number(block["delay_s"], "delay_s")
    jitter = _number(block.get("jitter_s", block.get("delay_jitter_s", 0.0)), "jitter_s")
    if "loss_probability" not in block and "loss" not in block:
        raise LinkRerunError("link input is missing loss_probability")
    loss = _number(block.get("loss_probability", block.get("loss")), "loss_probability", upper=1.0)
    model = block.get("loss_model", "independent")
    if model not in {"independent", "burst"}:
        raise LinkRerunError("loss_model must be independent or burst")
    result: dict[str, Any] = {"delay_s": delay, "jitter_s": jitter,
                              "loss_probability": loss, "loss_model": model}
    if model == "burst":
        mean = _number(block.get("burst_mean_length_packets"), "burst_mean_length_packets")
        if mean <= 0:
            raise LinkRerunError("burst_mean_length_packets must be positive")
        result["burst_mean_length_packets"] = mean
    return result


def describe_input(data: Any, raw: bytes, input_file: str) -> dict[str, Any]:
    """Record where the link settings came from: kind, declared status, trace id and input hash."""
    if isinstance(data, dict) and data.get("schema") == FIT_SCHEMA:
        kind = "espnow-fit"
        source = data.get("source_trace")
        source = source if isinstance(source, dict) else {}
    else:
        kind = "link-json"
        source = data if isinstance(data, dict) else {}
    status = source.get("status")
    trace_id = source.get("trace_id")
    return {"input_kind": kind, "input_status": status if status in KNOWN_STATUSES else "unknown",
            "trace_id": trace_id if isinstance(trace_id, str) else None,
            "input_file": input_file, "input_sha256": hashlib.sha256(raw).hexdigest()}


def status_label(provenance: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the summary status text and limits for one input description."""
    status = provenance["input_status"]
    if status == "measured":
        what = "a measured fit" if provenance["input_kind"] == "espnow-fit" else "measured input"
        return (f"investigation: link from {what}, sensor simulated",
                [f"link from {what}", "sensor simulated", "main slice only"])
    if status in ("synthetic", "fixture"):
        return SYNTHETIC_STATUS, list(SYNTHETIC_LIMITS)
    return ("investigation: link input status unknown, sensor simulated",
            ["link input status unknown, not shown to be measured", "sensor simulated", "main slice only"])


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LinkRerunError(f"cannot read input JSON: {exc}") from exc


def _check_output(output: Path) -> Path:
    resolved = output.resolve()
    runs = (_REPO_ROOT / "sim" / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        raise LinkRerunError("output must not be inside sim/runs")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise LinkRerunError("output must be a new or empty directory")
    return resolved


def build_cells(link: dict[str, Any], pilot: bool = False) -> list[dict[str, Any]]:
    noises = (0.0, 0.05) if pilot else NOISES
    cells: list[dict[str, Any]] = []
    for motion in ("constant_velocity", "constant_turn"):
        for noise in noises:
            cell = {"slice": "main", "motion": motion, "delay_s": link["delay_s"],
                    "loss_probability": link["loss_probability"], "noise_std_m": noise,
                    "loss_model": link["loss_model"], "jitter_s": link["jitter_s"],
                    "tracker_measurement_std_m": 0.05}
            if link["loss_model"] == "burst":
                cell["burst_mean_length_packets"] = link["burst_mean_length_packets"]
            cell["cell_id"] = (f"main/{motion}/measured/d{link['delay_s']:.9g}/l{link['loss_probability']:.9g}/"
                                f"n{noise:.3f}/{link['loss_model']}/j{link['jitter_s']:.9g}/r0.050")
            cells.append(cell)
    return cells


def _run_cell_link(cell: dict[str, Any], seeds: list[int], config: RecoveryConfig) -> dict[str, Any]:
    naive, recovery = [], []
    for seed in seeds:
        kwargs = {"delay_s": cell["delay_s"], "loss_probability": cell["loss_probability"],
                  "noise_std_m": cell["noise_std_m"], "seed": seed, "config": config,
                  "loss_model": cell["loss_model"], "jitter_s": cell["jitter_s"]}
        if cell["loss_model"] == "burst":
            kwargs["burst_mean_length_packets"] = cell["burst_mean_length_packets"]
        deliveries = deliver(cell["motion"], **kwargs)
        arms = run_arms(deliveries, cell["motion"], config,
                        tracker_measurement_std_m=cell["tracker_measurement_std_m"])
        naive.append(replace(arms["naive"], query_times_s=None, errors_m=None))
        recovery.append(replace(arms["recovery"], query_times_s=None, errors_m=None))
    return {"naive": naive, "recovery": recovery}


def run_rerun(input_path: Path, output: Path, *, pilot: bool = False, seeds_count: int = 50,
              base_seed: int = 20261013, argv: list[str] | None = None) -> dict[str, Any]:
    if seeds_count < 1:
        raise LinkRerunError("seeds must be positive")
    resolved_output = _check_output(output)
    raw = input_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    link = parse_link(data)
    provenance = describe_input(data, raw, input_path.name)
    status, limits = status_label(provenance)
    cells = build_cells(link, pilot=pilot)
    seeds = seed_list(seeds_count, base_seed=base_seed)
    config = RecoveryConfig()
    results = [_run_cell_link(cell, seeds, config) for cell in cells]
    summaries = [_cell_summary(cell, result) for cell, result in zip(cells, results)]
    summary = {"schema": SCHEMA, "status": status, "provenance": provenance,
               "generator": "tools/s6_link_rerun.py", "link": link,
               "config": {"duration_s": config.duration_s, "observation_period_s": config.observation_period_s},
               "design": {"pilot": pilot, "n_seeds": seeds_count, "base_seed": base_seed, "seeds": seeds},
               "cells": summaries, "limits": limits}
    resolved_output.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = resolved_output / "per_seed.csv"
    fields = ["cell_id", "slice", "motion", "delay_s", "loss_probability", "noise_std_m", "loss_model", "jitter_s", "tracker_measurement_std_m", "arm", "seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count", "fused_out_of_order_count", "failed"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for cell, result in zip(cells, results):
            for arm_name in ("naive", "recovery"):
                for index, (seed, arm) in enumerate(zip(seeds, result[arm_name])):
                    row = {key: cell.get(key) for key in fields[:9]}; row.update({"arm": arm_name, "seed_index": index, "seed": seed})
                    row.update({key: getattr(arm, key) for key in fields[12:]})
                    writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    manifest = {"schema": MANIFEST_SCHEMA, "input_file": input_path.name,
                "input_sha256": provenance["input_sha256"], "provenance": provenance,
                "link": link, "pilot": pilot, "seeds_count": seeds_count, "base_seed": base_seed,
                "n_cells": len(cells), "argv": argv or [],
                "replay_command": "PYTHONPATH=. .venv/bin/python tools/s6_link_rerun.py " + shlex.join(argv or []),
                "outputs": {"summary.json": _sha256(summary_path), "per_seed.csv": _sha256(csv_path)}}
    (resolved_output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true"); parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20261013)
    args = parser.parse_args(argv)
    try:
        run_rerun(args.input, args.output, pilot=args.pilot, seeds_count=args.seeds, base_seed=args.base_seed,
                  argv=list(sys.argv[1:] if argv is None else argv))
    except (LinkRerunError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
