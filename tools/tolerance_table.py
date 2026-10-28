"""Report tolerance boundaries from registered per-seed result files.

This module deliberately reads result files only.  It does not run simulations or
modify any registered output.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

Z = 1.96
CELL_THRESHOLDS = (0.3, 0.5, 0.7)
FAILURE_CRITERIA_M = (0.15, 0.25, 0.35)


def wilson_interval(failures: int, trials: int, z: float = Z) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a failure proportion."""
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("failures must be between zero and trials, with trials positive")
    n = float(trials)
    p = float(failures) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _failure(row: Mapping, criterion: float) -> bool:
    """Apply the selected p95 criterion and fixed 0.5 m max-error rule."""
    p95 = row.get("p95_error_m", row.get("p95"))
    maximum = row.get("max_error_m", row.get("max"))
    return ((p95 is not None and float(p95) > criterion)
            or (maximum is not None and float(maximum) > 0.5))


def boundary_for_rows(rows: Iterable[Mapping], allowed_fail_fraction: float = 0.5,
                      p95_threshold: float = 0.25) -> dict:
    """Find the largest passing ``axis`` value and its failing-flight count.

    Rows at each axis value are treated as independent flights.  A value passes
    when its observed failing share is strictly below ``allowed_fail_fraction``.
    """
    grouped: dict[float, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[float(row["axis"])].append(row)
    candidates = []
    for axis in sorted(grouped):
        subset = grouped[axis]
        failures = sum(_failure(row, p95_threshold) for row in subset)
        if failures / len(subset) < allowed_fail_fraction:
            candidates.append((axis, failures, len(subset)))
    if not candidates:
        return {"boundary": None, "failing_seeds": None, "seed_count": 0}
    axis, failures, count = candidates[-1]
    low, high = wilson_interval(failures, count)
    return {"boundary": axis, "failing_seeds": failures, "seed_count": count,
            "failing_share": failures / count, "wilson_low": low, "wilson_high": high}


def tolerance_rows(rows: Iterable[Mapping], cell_thresholds: Iterable[float] = CELL_THRESHOLDS,
                   failure_criteria: Iterable[float] = FAILURE_CRITERIA_M) -> list[dict]:
    """Return one compact result row for every threshold and criterion pair."""
    materialized = list(rows)
    return [{"cell_threshold": float(threshold), "failure_criterion_m": float(criterion),
             **boundary_for_rows(materialized, float(threshold), float(criterion))}
            for threshold in cell_thresholds for criterion in failure_criteria]


def _read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _atlas_rows(run: Path) -> list[dict]:
    """Return per-flight atlas rows, one set per atlas group.

    An atlas group is one stratum, motion variant, loss model, loss rate and method
    arm.  The per-seed export drops the leading stratum from the cell key, and a cell
    shared by two strata is written once per stratum with the same seeds.  Each row is
    therefore attached to every (stratum, variant) that owns the cell, kept only for
    the arms that stratum ran, and counted once.
    """
    summary = json.loads((run / "summary.json").read_text())
    owners: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for entry in summary["cells"]:
        key = entry["key"]
        if key.split("/", 1)[0] == entry["stratum"]:
            key = key.split("/", 1)[1]
        owners[key].append((entry, entry["cell"]))
    rows = []
    seen = set()
    for row in _read_csv(run / "per_seed.csv.gz"):
        for entry, cell in owners[row["cell_key"]]:
            if row["arm"] not in entry["arms"]:
                continue
            signature = (entry["stratum"], entry["variant"], row["cell_key"], row["arm"],
                         row["seed"])
            if signature in seen:
                continue
            seen.add(signature)
            rows.append({**row, "stratum": entry["stratum"], "variant": entry["variant"],
                         "delay_s": cell["delay_s"], "loss_model": cell["loss_model"],
                         "loss_probability": cell["loss_probability"],
                         "noise_std_m": cell["noise_std_m"], "rate_hz": cell["rate_hz"]})
    return rows


def _source_rows(root: Path, source: str) -> tuple[list[dict], tuple[str, ...]]:
    run = root / "sim" / "runs" / source
    if source == "s6-atlas":
        return _atlas_rows(run), ("delay_s",)
    if source == "s6-recovery":
        return _read_csv(run / "per_seed.csv"), ("delay_s",)
    if source == "d3-tolerance":
        return _read_csv(run / "per_seed.csv.gz"), ("delay",)
    raise ValueError(f"unknown source: {source}")


def _clean(value):
    return _num(value)


def build_table(root: Path, sources: tuple[str, ...] = ("s6-recovery", "s6-atlas", "d3-tolerance")) -> dict:
    """Build the complete deterministic report as a JSON-compatible object."""
    output = []
    for source in sources:
        rows, axes = _source_rows(root, source)
        for axis in axes:
            usable = [r for r in rows if r.get(axis) not in (None, "")]
            metric_names = {axis, "p95", "p95_error_m", "failed", "seed", "seed_index",
                            "max", "max_error_m", "rmse", "rmse_m", "gap_check",
                            "cell_id", "cell_key", "key", "fused_count", "dropped_count",
                            "fused_out_of_order_count", "gated_count", "refiltered_count",
                            "mean_nis", "mean_nees", "busy_fraction", "delivery_ratio"}
            dimensions = sorted(k for k in {k for r in usable for k in r}
                                if k not in metric_names and not k.startswith("x_"))
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for row in usable:
                key = tuple((k, _clean(row.get(k))) for k in dimensions)
                groups[key].append(row)
            for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
                for cell_threshold in CELL_THRESHOLDS:
                    for criterion in FAILURE_CRITERIA_M:
                        grouped = [{"axis": r[axis], "failed": r.get("failed"),
                                     "p95_error_m": r.get("p95_error_m", r.get("p95")),
                                     "max_error_m": r.get("max_error_m", r.get("max"))}
                                    for r in group]
                        result = boundary_for_rows(grouped, cell_threshold, criterion)
                        output.append({"source": source, "axis": axis,
                                       "conditions": dict(key),
                                       "cell_threshold": cell_threshold,
                                       "failure_criterion_m": criterion, **result})
    return {"schema": "tolerance-table-v1", "cell_thresholds": list(CELL_THRESHOLDS),
            "failure_criteria_m": list(FAILURE_CRITERIA_M), "rows": output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_table(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
