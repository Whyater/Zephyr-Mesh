"""Run the recovery atlas over simulated link and sensor conditions.

The experiment asks how delay, loss, noise, target motion, and estimator choice
change the first failing delay. Every result is simulated link and simulated
sensor output, and nothing is measured. Run it with::

    PYTHONPATH=. ./.venv/bin/python tools/run_s6_atlas.py --output runs/s6-atlas

The atlas uses a mean burst length of 0.5 seconds and adds optional IMM-stratum
filtering plus a central-only speed grid; the default grid is 3969 cells and
the central-only speed grid is 3185 cells.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from sim.estimation.atlas import build_atlas, validate_atlas
from sim.estimation.base import ArmSpec, EstimatorSettings, NAIVE, RECOVERY_V1
from sim.estimation.engine import CellSpec, run_cell
from sim.estimation.sweep import (
    capture_provenance,
    default_processes,
    run_jobs,
    sha256_file,
    sha256_json,
    write_csv,
    write_csv_gz,
    write_json,
    write_manifest,
)
from sim.s6_recovery import RecoveryConfig, seed_list


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261013
EXPERIMENT = "s6-atlas"
STRATA = ("central", "rate5", "rate20", "rate50", "noise0", "noise2", "noise10", "student_t", "jitter")
VARIANTS = (
    ("cv_0.5", "cv", (("speed_mps", 0.5),)),
    ("cv_1", "cv", ()),
    ("cv_2", "cv", (("speed_mps", 2.0),)),
    ("ct_r0.5", "ct", (("radius_m", 0.5),)),
    ("ct_r1", "ct", ()),
    ("ct_r3", "ct", (("radius_m", 3.0),)),
    ("jink", "jink", ()),
    ("figure8", "figure8", ()),
    ("start_stop", "start_stop", ()),
)
LOSSES = (
    ("independent", 0.0),
    ("independent", 0.10),
    ("independent", 0.30),
    ("independent", 0.50),
    ("burst", 0.10),
    ("burst", 0.30),
    ("burst", 0.50),
)
DELAYS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
PILOT_LOSSES = (("independent", 0.0), ("independent", 0.30), ("burst", 0.30))
PILOT_DELAYS = (0.0, 0.10, 0.30)

CELL_CSV_COLUMNS = (
    "slice", "key", "arm", "target", "delay_s", "loss_model", "loss_probability",
    "p95_median_m", "p95_ci_low_m", "p95_ci_high_m", "fail_rate", "fail_rate_ci_low",
    "fail_rate_ci_high", "max_error_median_m", "mean_nis_median", "fused_total",
    "dropped_total", "refiltered_total", "gated_total", "paired_naive_median_m",
    "paired_naive_ci_low_m", "paired_naive_ci_high_m", "paired_recovery_median_m",
    "paired_recovery_ci_low_m", "paired_recovery_ci_high_m",
)
PER_SEED_COLUMNS = (
    "cell_key", "arm", "seed_index", "seed", "p95_error_m", "max_error_m", "rmse_m",
    "mean_nis", "fused_count", "dropped_count", "fused_out_of_order_count", "gated_count",
    "refiltered_count", "failed", "x_final_q_scale", "x_final_r_scale",
    "x_gate_checked_clean", "x_gate_checked_outliers", "x_gate_forced_accepts", "x_gated_clean",
    "x_gated_outliers", "x_imm_ct_prob_mean", "x_monitor_first_warning_s",
    "x_monitor_warn_fraction",
)


def _normalise_imm_strata(value: str | list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
    """Return the selected strata in the registered order."""
    if value is None or value == "all":
        return STRATA
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = sorted(set(names) - set(STRATA))
    if unknown:
        raise ValueError(f"unknown IMM strata: {', '.join(unknown)}")
    return tuple(name for name in STRATA if name in names)


def _arm_specs(stratum: str, imm_strata: str | list[str] | tuple[str, ...] | set[str] | None = None) -> list[ArmSpec]:
    """Return the pre-registered arm list for one stratum."""
    selected_imm_strata = _normalise_imm_strata(imm_strata)
    arms = [
        NAIVE,
        RECOVERY_V1,
        ArmSpec("adaptive", model="cv", stamping="capture", late_policy="drop", adaptive="q"),
    ]
    if stratum in selected_imm_strata:
        arms.insert(2, ArmSpec("imm", model="imm", stamping="capture", late_policy="drop"))
    if stratum == "jitter":
        arms.append(ArmSpec("retro", model="cv", stamping="capture", late_policy="retrodict"))
    if stratum == "student_t":
        arms.append(ArmSpec("recovery_gated", model="cv", stamping="capture", late_policy="drop", gate_probability=0.99))
        if stratum in selected_imm_strata:
            arms.append(ArmSpec("imm_gated", model="imm", stamping="capture", late_policy="drop", gate_probability=0.99))
    return arms


def build_arms(stratum: str | dict[str, Any] | None = None,
               imm_strata: str | list[str] | tuple[str, ...] | set[str] | None = None) -> list[ArmSpec]:
    """Build arms, optionally for one named stratum."""
    if isinstance(stratum, dict):
        imm_strata = stratum.get("imm_strata", imm_strata)
        stratum = stratum.get("slice", stratum.get("stratum", "central"))
    if stratum is not None and stratum not in STRATA:
        raise ValueError(f"unknown stratum: {stratum}")
    return _arm_specs(stratum or "central", imm_strata)


def _stratum_settings(name: str) -> dict[str, Any]:
    if name == "central":
        return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.05}
    if name == "rate5":
        return {"rate_hz": 5.0, "noise_model": "gaussian", "noise_std_m": 0.05}
    if name == "rate20":
        return {"rate_hz": 20.0, "noise_model": "gaussian", "noise_std_m": 0.05}
    if name == "rate50":
        return {"rate_hz": 50.0, "noise_model": "gaussian", "noise_std_m": 0.05}
    if name == "noise0":
        return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.0}
    if name == "noise2":
        return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.02}
    if name == "noise10":
        return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.10}
    if name == "student_t":
        return {"rate_hz": 10.0, "noise_model": "student_t", "noise_std_m": 0.05, "noise_dof": 3.0}
    if name == "jitter":
        return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.05}
    raise ValueError(f"unknown stratum: {name}")


def build_cells(pilot: bool = False, *, speed_variants_central_only: bool = False,
                imm_strata: str | list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Build the deterministic 3969-cell grid, or the central pilot grid."""
    _normalise_imm_strata(imm_strata)
    strata = ("central",) if pilot else STRATA
    losses = PILOT_LOSSES if pilot else LOSSES
    delays = PILOT_DELAYS if pilot else DELAYS
    cells: list[dict[str, Any]] = []
    for stratum in strata:
        settings = _stratum_settings(stratum)
        variants = VARIANTS
        if speed_variants_central_only and stratum != "central":
            variants = tuple(item for item in VARIANTS if item[0] not in {"cv_0.5", "cv_2"})
        for variant, target, target_params in variants:
            for loss_model, loss_probability in losses:
                for delay_s in delays:
                    jitter_s = delay_s / 2.0 if stratum == "jitter" else 0.0
                    cell = {
                        "slice": stratum,
                        "target": target,
                        "target_params": dict(target_params),
                        **settings,
                        "delay_s": float(delay_s),
                        "jitter_s": float(jitter_s),
                        "loss_probability": float(loss_probability),
                        "loss_model": loss_model,
                        "burst_mean_length_packets": 0.5 * float(settings["rate_hz"]),
                    }
                    cell["cell_id"] = f"{stratum}/{variant}/{loss_model}:{loss_probability:g}/d{delay_s:g}"
                    cell["key"] = cell["cell_id"]
                    cells.append(cell)
    expected = 81 if pilot else (3969 if not speed_variants_central_only else 3185)
    if len(cells) != expected:
        raise AssertionError(f"expected {expected} cells, got {len(cells)}")
    return cells


def _cell_from_job(job: dict[str, Any]) -> tuple[str, CellSpec]:
    source = job.get("cell", job)
    if not isinstance(source, dict):
        raise ValueError("job cell must be a mapping")
    stratum = str(job.get("slice", source.get("slice", "central")))
    fields = {f.name for f in dataclasses.fields(CellSpec)}
    values = {key: value for key, value in source.items() if key in fields}
    return stratum, CellSpec(**values)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def job_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Run one atlas cell with common random-number seeds."""
    stratum, cell = _cell_from_job(job)
    seeds = job.get("seeds")
    if seeds is None:
        seeds = seed_list(int(job.get("n_seeds", 50)))
    elif isinstance(seeds, (int, float)):
        seeds = seed_list(int(seeds))
    names = job.get("arm_names")
    arms = build_arms(stratum, job.get("imm_strata"))
    if names is not None:
        wanted = set(names)
        arms = [arm for arm in arms if arm.name in wanted]
    if not arms:
        raise ValueError("job selected no arms")
    config = RecoveryConfig(**job["config"]) if isinstance(job.get("config"), dict) else RecoveryConfig()
    settings = EstimatorSettings(**job["settings"]) if isinstance(job.get("settings"), dict) else EstimatorSettings()
    result = _json_safe(run_cell(cell, arms, [int(seed) for seed in seeds], config, settings))
    result["slice"] = stratum
    result["variant"] = _variant_name(cell.as_dict())
    result["key"] = f"{stratum}/{result['key']}"
    return result


def _variant_name(cell: dict[str, Any]) -> str:
    target = cell["target"]
    params = cell.get("target_params") or {}
    for name, wanted_target, wanted_params in VARIANTS:
        if target == wanted_target and dict(wanted_params) == dict(params):
            return name
    return target


def _cell_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (cell["slice"], _variant_name(cell), cell["loss_model"], float(cell["loss_probability"]), float(cell["delay_s"]))


def _main_jobs(cells: list[dict[str, Any]], seeds: list[int], config: RecoveryConfig, settings: EstimatorSettings,
               imm_strata: str | list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    return [{"cell": cell, "slice": cell["slice"], "seeds": seeds, "config": dataclasses.asdict(config),
             "settings": dataclasses.asdict(settings),
             "imm_strata": list(_normalise_imm_strata(imm_strata))} for cell in cells]


def _summary_cell(cell: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"slice": cell["slice"], "stratum": cell["slice"], "variant": _variant_name(cell),
            "cell_id": cell.get("cell_id"),
            "cell": result["cell"], "key": result["key"],
            "arms": result["arms"], "paired_vs_naive": result.get("paired_vs_naive", {}),
            "paired_vs_recovery": result.get("paired_vs_recovery", {})}


def _first_failures(summary_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in summary_cells:
        cell = item["cell"]
        key = (item["slice"], _variant_name(cell), cell["loss_model"], float(cell["loss_probability"]))
        groups.setdefault(key, []).append(item)
    out = []
    for group_key in sorted(groups, key=repr):
        items = sorted(groups[group_key], key=lambda item: float(item["cell"]["delay_s"]))
        arms = sorted({name for item in items for name in item["arms"]})
        for arm_name in arms:
            for item in items:
                arm_summary = item["arms"].get(arm_name)
                if not isinstance(arm_summary, dict):
                    raise ValueError(f"missing arm summary for {arm_name}")
                for field in ("fail_rate", "fail_rate_ci_low", "fail_rate_ci_high"):
                    value = arm_summary.get(field)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                        raise ValueError(f"missing or non-finite {field} for {arm_name}")
            failing = next((item for item in items if item["arms"][arm_name]["fail_rate"] >= 0.5), None)
            if failing is None:
                continue
            out.append({"group": group_key, "arm": arm_name, "item": failing})
    return out


def _counterfactual_jobs(failures: list[dict[str, Any]], seeds: list[int], config: RecoveryConfig,
                         settings: EstimatorSettings, imm_strata: str | list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    jobs = []
    for failure in failures:
        original = dict(failure["item"]["cell"])
        original["slice"] = failure["item"]["slice"]
        kinds = ["all_removed"]
        if float(original.get("delay_s", 0.0)) != 0.0 or float(original.get("jitter_s", 0.0)) != 0.0:
            kinds.insert(0, "delay_removed")
        if float(original.get("loss_probability", 0.0)) != 0.0:
            kinds.insert(-1, "loss_removed")
        if float(original.get("noise_std_m", 0.0)) != 0.0:
            kinds.insert(-1, "noise_removed")
        for kind in kinds:
            changed = dict(original)
            if kind in ("delay_removed", "all_removed"):
                changed["delay_s"] = 0.0
                changed["jitter_s"] = 0.0
            if kind in ("loss_removed", "all_removed"):
                changed["loss_probability"] = 0.0
            if kind in ("noise_removed", "all_removed"):
                changed["noise_std_m"] = 0.0
            jobs.append({"cell": changed, "slice": changed["slice"], "seeds": seeds,
                         "config": dataclasses.asdict(config), "settings": dataclasses.asdict(settings),
                         "imm_strata": list(_normalise_imm_strata(imm_strata)),
                         "arm_names": [failure["arm"]], "counterfactual": kind,
                         "counterfactual_key": _cf_key(failure, kind)})
    return jobs


def _cf_key(failure: dict[str, Any], kind: str) -> str:
    group = failure["group"]
    return "/".join([str(group[0]), str(group[1]), str(group[2]), f"{float(group[3]):g}", str(failure["arm"]), kind])


def _build_counterfactual_doc(failures: list[dict[str, Any]], jobs: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {}
    for job, result in zip(jobs, results):
        key = str(job["counterfactual_key"])
        arm = next(iter(result["arms"]))
        values[key] = {"stratum": job["slice"], "variant": _variant_name(job["cell"]),
                       "loss_condition": {"model": job["cell"]["loss_model"], "probability": job["cell"]["loss_probability"]},
                       "arm": arm, "delay_s": job["cell"]["delay_s"], "kind": job["counterfactual"],
                       "median_p95_m": result["arms"][arm]["p95_median_m"]}
    for failure in failures:
        group = failure["group"]
        base = "/".join([str(group[0]), str(group[1]), str(group[2]), f"{float(group[3]):g}", str(failure["arm"])])
        selected = {kind: values.get(f"{base}/{kind}", {}).get("median_p95_m")
                    for kind in ("delay_removed", "loss_removed", "noise_removed", "all_removed")}
        original = failure["item"]["arms"][failure["arm"]]["p95_median_m"]
        reductions = {kind: original - median for kind, median in selected.items()
                      if kind != "all_removed" and median is not None}
        mode = max(reductions, key=reductions.get) if reductions else None
        label = {"delay_removed": "lag-dominated", "loss_removed": "gap-dominated", "noise_removed": "noise-dominated"}.get(mode)
        if selected["all_removed"] is not None and (not reductions or selected["all_removed"] >= max(reductions.values())):
            label = "model-mismatch"
        record = {"stratum": group[0], "variant": group[1],
                  "loss_condition": f"{group[2]}:{float(group[3]):g}",
                  "arm": failure["arm"], "delay_s": failure["item"]["cell"]["delay_s"],
                  "original_median_p95_m": original, "medians": selected,
                  "drops": {kind.removesuffix("_removed"): value for kind, value in reductions.items()},
                  "failure_mode": label}
        values[f"{base}/summary"] = record
        values[f"{group[0]}|{group[1]}|{group[2]}:{float(group[3]):g}|{failure['arm']}"] = record
    return values


def _fingerprint(seeds_count: int) -> str:
    paths = sorted((REPO_ROOT / "sim/estimation").glob("*.py"))
    paths += [REPO_ROOT / name for name in ("sim/timesync.py", "sim/link.py", "sim/sensor.py", "sim/tracker.py", "sim/s6_recovery.py", "tools/run_s6_atlas.py")]
    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths if path.is_file()}
    return sha256_json({"sources": hashes, "seeds": int(seeds_count)})


def _ordered_arm_names(item: dict[str, Any]) -> list[str]:
    canonical = [arm.name for arm in _arm_specs(str(item.get("slice", "central")))]
    present = item.get("arms", {})
    return [name for name in canonical if name in present] + sorted(set(present) - set(canonical))


def _csv_rows(cells: list[dict[str, Any]], per_seed_by_key: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    per_seed_by_key = per_seed_by_key or {}
    rows = []
    for item in cells:
        cell = item["cell"]
        for arm in _ordered_arm_names(item):
            summary = item["arms"][arm]
            paired_n = item.get("paired_vs_naive", {}).get(arm, {})
            paired_r = item.get("paired_vs_recovery", {}).get(arm, {})
            per_seed = [row for row in per_seed_by_key.get(str(item["key"]), []) if row.get("arm") == arm]
            rows.append({"slice": item["slice"], "key": item["key"], "arm": arm, "target": cell["target"],
                         "delay_s": cell["delay_s"], "loss_model": cell["loss_model"], "loss_probability": cell["loss_probability"],
                         "p95_median_m": summary.get("p95_median_m"), "p95_ci_low_m": summary.get("p95_ci_low_m"),
                         "p95_ci_high_m": summary.get("p95_ci_high_m"), "fail_rate": summary.get("fail_rate"),
                         "fail_rate_ci_low": summary.get("fail_rate_ci_low"), "fail_rate_ci_high": summary.get("fail_rate_ci_high"),
                         "max_error_median_m": summary.get("max_error_median_m"), "mean_nis_median": summary.get("mean_nis_median"),
                         "fused_total": summary.get("fused_total"), "dropped_total": summary.get("dropped_total"),
                         "refiltered_total": sum(int(row.get("refiltered_count") or 0) for row in per_seed),
                         "gated_total": sum(int(row.get("gated_count") or 0) for row in per_seed),
                         "paired_naive_median_m": paired_n.get("p95_diff_median_m"), "paired_naive_ci_low_m": paired_n.get("p95_diff_ci_low_m"),
                         "paired_naive_ci_high_m": paired_n.get("p95_diff_ci_high_m"), "paired_recovery_median_m": paired_r.get("p95_diff_median_m"),
                         "paired_recovery_ci_low_m": paired_r.get("p95_diff_ci_low_m"), "paired_recovery_ci_high_m": paired_r.get("p95_diff_ci_high_m")})
    return rows


def _strata_document() -> dict[str, dict[str, Any]]:
    result = {}
    for name in STRATA:
        settings = dict(_stratum_settings(name))
        settings["noise_dof"] = settings.get("noise_dof")
        settings["jitter_rule"] = "jitter_s = delay_s / 2" if name == "jitter" else "none"
        result[name] = settings
    return result


def run_sweep(output: Path, *, pilot: bool = False, seeds_count: int = 50, processes: int | None = None,
              checkpoint_dir: Path | None = None, no_checkpoint: bool = False, argv: list[str] | None = None,
              imm_strata: str | list[str] | tuple[str, ...] | None = None,
              speed_variants_central_only: bool = False) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if seeds_count < 1:
        raise ValueError("--seeds must be positive")
    selected_imm_strata = _normalise_imm_strata(imm_strata)
    cells = (build_cells(pilot, speed_variants_central_only=True)
             if speed_variants_central_only else build_cells(pilot))
    seeds = seed_list(seeds_count, base_seed=BASE_SEED)
    n_processes = default_processes() if processes is None else int(processes)
    if n_processes < 1:
        raise ValueError("--processes must be positive")
    config, settings = RecoveryConfig(), EstimatorSettings()
    jobs = _main_jobs(cells, seeds, config, settings, selected_imm_strata)
    provenance = capture_provenance()
    start = time.perf_counter()
    cache = None if no_checkpoint else (checkpoint_dir or Path.home() / "zm-runb-cache" / "atlas")
    main_start = time.perf_counter()
    results = run_jobs(jobs, "tools.run_s6_atlas:job_cell", processes=n_processes, checkpoint_dir=cache,
                       fingerprint=_fingerprint(seeds_count))
    main_runtime = time.perf_counter() - main_start
    summary_cells = [_summary_cell(cell, result) for cell, result in zip(cells, results)]
    failures = _first_failures(summary_cells)
    cf_jobs = _counterfactual_jobs(failures, seeds, config, settings, selected_imm_strata)
    cf_start = time.perf_counter()
    cf_results = run_jobs(cf_jobs, "tools.run_s6_atlas:job_cell", processes=n_processes, checkpoint_dir=cache,
                          fingerprint=_fingerprint(seeds_count) + ":phase2") if cf_jobs else []
    cf_runtime = time.perf_counter() - cf_start
    counterfactuals = _build_counterfactual_doc(failures, cf_jobs, cf_results)
    all_arms = {arm.name: arm for stratum in STRATA for arm in _arm_specs(stratum, selected_imm_strata)}
    design = {
        "name": "stratified one-factor strata around a central plane",
        "reason": "a full factorial is about 2 million runs, so the design is stratified",
        "burst_mean_s": 0.5,
        "imm_strata": list(selected_imm_strata),
        "speed_variants_central_only": bool(speed_variants_central_only),
        "arms_by_stratum": {
            name: [arm.name for arm in _arm_specs(name, selected_imm_strata)] for name in STRATA
        },
        "options": {
            "imm_strata": list(selected_imm_strata),
            "speed_variants_central_only": bool(speed_variants_central_only),
        },
    }
    summary = {"experiment": EXPERIMENT, "experiment_id": EXPERIMENT,
               "question": "How much delay can each recovery arm tolerate across target and link strata?",
               "burst_mean_s": 0.5,
               "simulated": True, "failure_rule": {"p95_error_m": config.failure_p95_m, "max_error_m": config.failure_max_m,
               "p95_m": config.failure_p95_m, "max_m": config.failure_max_m, "fail_rate_threshold": 0.5,
               "per_seed": {"p95_error_m": config.failure_p95_m, "max_error_m": config.failure_max_m,
                            "scored_window_s": [config.burn_in_s, config.duration_s]},
               "cell_rule": "fail rate >= 0.5", "cell_rule_status": "proposed, not yet adopted"}, "config": dataclasses.asdict(config),
               "seeds": seeds, "arms": [arm.as_dict() for arm in all_arms.values()],
               "design": design, "design_reason": design["reason"],
               "strata": _strata_document(), "cells": summary_cells,
               "limits": ["simulated link and sensor only", "failure labels are research criteria", "no hardware measurements",
                          "imm is a placeholder for the best maneuver arm from the turn experiment"]}
    write_json(output / "summary.json", summary)
    per_seed_by_key = {str(result["key"]): list(result.get("per_seed", [])) for result in results}
    cell_rows = _csv_rows(summary_cells, per_seed_by_key)
    write_csv(output / "cells.csv", cell_rows, CELL_CSV_COLUMNS)
    per_seed = []
    for item in summary_cells:
        rows = per_seed_by_key.get(str(item["key"]), [])
        arm_rank = {name: index for index, name in enumerate(_ordered_arm_names(item))}
        per_seed.extend(sorted(rows, key=lambda row: (arm_rank.get(str(row.get("arm")), len(arm_rank)),
                                                       int(row.get("seed_index", 0)))))
    write_csv_gz(output / "per_seed.csv.gz", per_seed, PER_SEED_COLUMNS)
    write_json(output / "counterfactuals.json", counterfactuals)
    atlas_doc = build_atlas(summary, counterfactuals)
    validate_atlas(atlas_doc)
    write_json(output / "atlas.json", atlas_doc)
    runtime = time.perf_counter() - start
    config_sha = sha256_json(dataclasses.asdict(config))
    used_argv = list(sys.argv[1:] if argv is None else argv)
    write_manifest(output, outputs=["summary.json", "cells.csv", "per_seed.csv.gz", "counterfactuals.json", "atlas.json"],
                   argv=used_argv, n_processes=n_processes, runtime_s=runtime,
                   extra={"n_cells": len(cells), "n_runs": len(cells) * seeds_count, "n_seeds": seeds_count,
                          "config_sha256": config_sha, "seeds": seeds, "limits": summary["limits"],
                          "replay_command": "PYTHONPATH=. ./.venv/bin/python tools/run_s6_atlas.py " + shlex.join(used_argv)},
                   schema="s6-atlas-manifest-1", provenance=provenance)
    if pilot:
        full_grid = 3185 if speed_variants_central_only else 3969
        scale = full_grid / max(1, len(cells))
        seed_scale_50 = 50 / max(1, seeds_count)
        main_estimate = main_runtime * scale * seed_scale_50
        phase2_estimate = cf_runtime * scale * seed_scale_50
        print(f"pilot cells: {len(cells)}", file=sys.stderr)
        print(f"pilot seeds: {seeds_count}", file=sys.stderr)
        print(f"pilot wall time s: {runtime!r}", file=sys.stderr)
        print(f"estimated full wall time s at 50 seeds, {n_processes} processes (main): {main_estimate!r}", file=sys.stderr)
        print(f"estimated full wall time s at 50 seeds, {n_processes} processes (phase 2): {phase2_estimate!r}", file=sys.stderr)
        print(f"estimated full wall time s at 50 seeds, {n_processes} processes (main + phase 2): {main_estimate + phase2_estimate!r}", file=sys.stderr)
        for name in ("summary.json", "cells.csv", "per_seed.csv.gz"):
            pilot_size = (output / name).stat().st_size
            size_scale = scale * (seed_scale_50 if name == "per_seed.csv.gz" else 1.0)
            print(f"estimated full {name} bytes at 50 seeds: {pilot_size * size_scale!r}", file=sys.stderr)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--imm-strata", default="all", help="comma-separated strata that run IMM arms")
    parser.add_argument("--speed-variants-central-only", action="store_true",
                        help="run cv_0.5 and cv_2 only in the central stratum")
    args = parser.parse_args(argv)
    run_sweep(args.output, pilot=args.pilot, seeds_count=args.seeds, processes=args.processes,
              checkpoint_dir=args.checkpoint_dir, no_checkpoint=args.no_checkpoint,
              imm_strata=args.imm_strata, speed_variants_central_only=args.speed_variants_central_only,
              argv=list(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
