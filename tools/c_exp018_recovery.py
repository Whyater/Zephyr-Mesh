#!/usr/bin/env python3
"""EXP-018 simulated parameter recovery and measurement design.

Usage: ``PYTHONPATH=. ./.venv/bin/python tools/c_exp018_recovery.py --output runs/c-exp018-recovery``.
All numbers are simulated from literature-derived radio parameters.  The RSSI
value at 1 m is reported as ``a_dbm_at_1m`` by the calibration tool.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.ordered_pool import effective_workers, ordered_map
from sim.radio.channel import GilbertElliott
from sim.radio.params import PARAMS, channel_preset
from sim.s6_recovery import wilson_interval
from sim.trace import EspNowTrace
from tools.calibrate_radio_from_trace import (PositionInput, calibrate, packets_for_wilson_halfwidth,
                                               sizing_table, synthetic_trace_dicts)
from tools.c_common import (argv_or_default, default_processes, prepare_output, seeds_from, write_json,
                            write_manifest)

BASE_SEED = 20261014 + 18
DISTANCES_M = (1.0, 5.0, 10.0, 20.0, 40.0)
N_VALUES = (1.8, 2.0, 2.6)
PARAMETERS = ("n", "a_dbm", "sigma_db", "p", "r")
BURST_P_GOOD_TO_BAD = 0.02
BURST_R_BAD_TO_GOOD = 0.2
BURST_AWARE_SIZING_RULE = (
    "For each delivery probability p and Wilson half-width, packets_bursty = "
    "ceil(F * packets_independent), where rho = 1 - burst_p - burst_r and "
    "F = (1 + rho) / (1 - rho); this is a large-n approximation for the "
    "two-state chain. Here p is the delivery probability used by the Wilson "
    "table, while burst_p is the Good-to-Bad transition probability. With "
    "loss_good=0 and loss_bad=1, the chain's stationary delivery is "
    "burst_r / (burst_p + burst_r) = 0.9090909. The table's delivery p "
    "values do not equal this stationary delivery; the inflation factor is "
    "still the chain's own."
)


def coverage_stat(values: list[float], intervals: list[list[float]], truth: float, confidence: float = 0.95) -> dict:
    """Return coverage, its Wilson interval, mean interval width, and bias."""
    pairs = [(value, interval) for value, interval in zip(values, intervals)
             if value is not None and interval is not None
             and interval[0] is not None and interval[1] is not None]
    if not pairs:
        return {"coverage": None, "coverage_wilson": None, "mean_interval_width": None,
                "bias": None, "n": len(values), "n_identifiable": 0,
                "n_unidentifiable": len(values)}
    identifiable_values = [value for value, _ in pairs]
    identifiable_intervals = [interval for _, interval in pairs]
    inside = [lo <= truth <= hi for lo, hi in identifiable_intervals]
    count = sum(inside)
    lo, hi = wilson_interval(count, len(pairs), confidence)
    return {"coverage": count / len(pairs), "coverage_wilson": [float(lo), float(hi)],
            "mean_interval_width": float(np.mean([b - a for a, b in identifiable_intervals])),
            "bias": float(np.mean(identifiable_values) - truth), "n": len(values),
            "n_identifiable": len(pairs), "n_unidentifiable": len(values) - len(pairs)}


def _trace_with_provenance(doc: dict) -> EspNowTrace:
    doc["provenance"] = {
        "instrument": {"kind": "simulated", "model": "synthetic radio trace", "settings": {}},
        "calibration_ref": None, "calibration_ref_null_reason": "simulated known parameters",
        "raw_sha256": "0" * 64, "code_revision": "simulated", "captured_at": "2026-01-01T00:00:00Z",
        "operator": "simulation", "ground_truth": None, "ground_truth_null_reason": "known simulation inputs",
    }
    return EspNowTrace.from_dict(doc)


def recover_one(seed: int, n_true: float, packets: int = 1000, distances: tuple[float, ...] = DISTANCES_M) -> dict:
    """Generate one multi-position trace set and fit it with the preregistered calibrator."""
    preset = channel_preset("indoor_gym")
    a_true = preset.tx_power_dbm - preset.pl0_db
    ge = GilbertElliott(BURST_P_GOOD_TO_BAD, BURST_R_BAD_TO_GOOD)
    docs = synthetic_trace_dicts(n=n_true, a_dbm=a_true, sigma_db=3.0, distances_m=distances,
                                 packets_per_distance=packets, ge=ge, fast_sigma_db=0.0, seed=seed)
    positions = [PositionInput(pid, _trace_with_provenance(doc), d) for pid, d, doc in docs]
    result = calibrate(positions)
    path = result["path_loss"]
    gil = result["gilbert_pooled"]
    estimates = {"n": path["path_loss_exponent"], "a_dbm": path["a_dbm_at_1m"],
                 "sigma_db": path["shadowing_sigma_db"], "p": gil["p_good_to_bad"], "r": gil["p_bad_to_good"]}
    intervals = {"n": path["path_loss_exponent_ci"], "a_dbm": path["a_ci"], "sigma_db": path["shadowing_sigma_ci"],
                 "p": gil["p_good_to_bad_ci"], "r": gil["p_bad_to_good_ci"]}
    row = {"seed": int(seed), "true_n": n_true, "packets_per_distance": packets, "n_distances": len(distances)}
    for key in PARAMETERS:
        row[f"true_{key}"] = {"n": n_true, "a_dbm": a_true, "sigma_db": 3.0,
                               "p": BURST_P_GOOD_TO_BAD, "r": BURST_R_BAD_TO_GOOD}[key]
        row[f"fit_{key}"] = estimates[key]
        row[f"{key}_lo"] = intervals[key][0] if intervals[key] else None
        row[f"{key}_hi"] = intervals[key][1] if intervals[key] else None
    return row


def _run_cells(cells: list[tuple[int, float, int, tuple[float, ...]]], processes: int,
               workers: int | None = None) -> list[dict]:
    if workers is not None:
        return ordered_map("tools.c_exp018_recovery:_recover_cell", cells, workers=workers)
    if processes > 1 and len(cells) > 1:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            return list(pool.map(_recover_cell, cells))
    return [recover_one(*cell) for cell in cells]


def _recover_cell(cell: tuple[int, float, int, tuple[float, ...]]) -> dict:
    return recover_one(*cell)


def _summary(rows: list[dict]) -> list[dict]:
    out = []
    for n_true in N_VALUES:
        subset = [r for r in rows if r["true_n"] == n_true]
        for key in PARAMETERS:
            identifiable = [r for r in subset
                            if r[f"fit_{key}"] is not None
                            and r[f"{key}_lo"] is not None
                            and r[f"{key}_hi"] is not None]
            n_unidentifiable = len(subset) - len(identifiable)
            vals = [r[f"fit_{key}"] for r in identifiable]
            ints = [[r[f"{key}_lo"], r[f"{key}_hi"]] for r in identifiable]
            item = {"coverage": None, "coverage_wilson": None,
                    "mean_interval_width": None, "bias": None,
                    "n": len(subset), "n_identifiable": len(identifiable),
                    "n_unidentifiable": n_unidentifiable}
            if identifiable:
                item.update(coverage_stat(vals, ints, subset[0][f"true_{key}"]))
            out.append({"true_n": n_true, "parameter": key, **item})
    return out


def _sweep_summary(rows: list[dict]) -> list[dict]:
    """Summarize interval half-widths without treating missing fits as zero."""
    out = []
    for sweep in ("packets", "distances"):
        settings = sorted({r["setting"] for r in rows if r["sweep"] == sweep})
        for setting in settings:
            subset = [r for r in rows if r["sweep"] == sweep and r["setting"] == setting]
            n_rows = [r for r in subset if r["n_lo"] is not None and r["n_hi"] is not None]
            r_rows = [r for r in subset if r["r_lo"] is not None and r["r_hi"] is not None]
            out.append({"sweep": sweep, "setting": setting,
                        "n_unidentifiable": len(subset) - len(n_rows),
                        "r_unidentifiable": len(subset) - len(r_rows),
                        "n_mean_half_width": (float(np.mean([(r["n_hi"] - r["n_lo"]) / 2 for r in n_rows]))
                                               if n_rows else None),
                        "r_mean_half_width": (float(np.mean([(r["r_hi"] - r["r_lo"]) / 2 for r in r_rows]))
                                               if r_rows else None)})
    return out


def _sweep_groups(seeds: list[int], pilot: bool) -> list[tuple[str, int, list[tuple[int, float, int, tuple[float, ...]]]]]:
    reps = 10 if pilot else 200
    groups = [("packets", packets, [(seeds[i], 2.0, packets, DISTANCES_M) for i in range(reps)])
              for packets in (100, 300, 1000, 3000)]
    for k in (3, 5, 8, 12):
        ds = tuple(float(x) for x in np.geomspace(1.0, 40.0, k))
        groups.append(("distances", k, [(seeds[reps + i], 2.0, 1000, ds) for i in range(reps)]))
    return groups


def _sweeps(seeds: list[int], pilot: bool, processes: int, workers: int | None = None) -> list[dict]:
    groups = _sweep_groups(seeds, pilot)
    if workers is None:
        # Old path: one pool per sweep setting, as in the registered run.
        results = [_run_cells(cells, processes) for _, _, cells in groups]
    else:
        # Opt-in path: one pool for all eight settings, then split back in order.
        flat = _run_cells([cell for _, _, cells in groups for cell in cells], processes, workers)
        results, start = [], 0
        for _, _, cells in groups:
            results.append(flat[start:start + len(cells)]); start += len(cells)
    rows = []
    for (sweep, setting, _), group_rows in zip(groups, results):
        for r in group_rows:
            r["sweep"] = sweep; r["setting"] = setting; rows.append(r)
    return rows


def _falsification() -> list[dict]:
    p = PARAMS
    a_range = [p["tx_power_dbm"].low - p["pl0_1m_db"].high,
               p["tx_power_dbm"].high - p["pl0_1m_db"].low]
    n_range = [p["exponent_indoor_large_room"].low, p["exponent_indoor_large_room"].high]
    outdoor_n_range = [p["exponent_outdoor_open"].low, p["exponent_outdoor_open"].high]
    sigma_range = [p["sigma_indoor_large_room_db"].low, p["sigma_indoor_large_room_db"].high]
    return [
        {"parameter": "a_dbm", "measurement": "EXP-002 RSSI versus distance", "literature_range": a_range,
         "packets_needed": 1000, "limiting_factor": "number of positions",
         "falsified_if": "95% interval excludes the literature-derived 1 m RSSI range"},
        {"parameter": "n", "measurement": "EXP-002 RSSI versus distance", "literature_range": n_range,
         "packets_needed": 1000, "limiting_factor": "number of positions",
         "falsified_if": "interval excludes the indoor literature-derived exponent range"},
        {"parameter": "n_outdoor", "measurement": "EXP-002 RSSI versus distance", "literature_range": outdoor_n_range,
         "packets_needed": 1000, "limiting_factor": "number of positions",
         "falsified_if": "interval excludes the outdoor literature-derived exponent range"},
        {"parameter": "sigma_db", "measurement": "EXP-002 RSSI versus distance", "literature_range": sigma_range,
         "packets_needed": 1000, "limiting_factor": "number of positions",
         "falsified_if": "residuals are non-Gaussian; this cannot be tested with 5 positions"},
        {"parameter": "p/r", "measurement": "EXP-002 delivery and burst structure versus distance", "literature_range": [0.0, 1.0],
         "packets_needed": packets_for_wilson_halfwidth(0.95, 0.01), "limiting_factor": "packets per position",
         "falsified_if": "burst ratio is near 1 when the model needs bursts"},
        {"parameter": "PER waterfall", "measurement": "EXP-002 delivery versus distance", "literature_range": [0.0, 1.0],
         "packets_needed": 1000, "limiting_factor": "distance grid",
         "falsified_if": "waterfall location is outside the calibrated model interval"},
        {"parameter": "MAC contention and queueing", "measurement": "EXP-003 loss versus send rate", "literature_range": [0.0, 1.0],
         "packets_needed": 1000, "limiting_factor": "send-rate sweep",
         "falsified_if": "loss versus send rate is inconsistent with simulated queueing"},
    ]


def burst_aware_sizing_table(confidence: float = 0.95) -> list[dict]:
    """Adjust independent Wilson packet counts for the runner's burst chain."""
    rho = 1.0 - BURST_P_GOOD_TO_BAD - BURST_R_BAD_TO_GOOD
    inflation_factor = (1.0 + rho) / (1.0 - rho)
    rows = []
    for independent in sizing_table(confidence):
        packets_independent = independent["packets"]
        rows.append({
            "p": independent["p"],
            "halfwidth": independent["halfwidth"],
            "packets_independent": packets_independent,
            "inflation_factor": inflation_factor,
            "packets_bursty": math.ceil(inflation_factor * packets_independent),
            "burst_p": BURST_P_GOOD_TO_BAD,
            "burst_r": BURST_R_BAD_TO_GOOD,
        })
    return rows


def run(output: Path, argv: list[str], pilot: bool, processes: int, workers: int | None = None) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/zephyr-mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    start = time.perf_counter(); output = prepare_output(output)
    reps = 10 if pilot else 200
    seeds = seeds_from(BASE_SEED, 3 * reps + 2 * reps)
    cells = [(seeds[i * reps + j], n, 1000, DISTANCES_M) for i, n in enumerate(N_VALUES) for j in range(reps)]
    recovery = _run_cells(cells, processes, workers)
    sweeps = _sweeps(seeds[3 * reps:], pilot, processes, workers)
    with open(output / "recovery.csv", "w", newline="") as f:
        fields = list(recovery[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in recovery: w.writerow({k: ("" if v is None else f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    sweep_summary = _sweep_summary(sweeps)
    unidentifiable_by_setting = {(r["sweep"], r["setting"]): r["n_unidentifiable"] for r in sweep_summary}
    with open(output / "sweeps.csv", "w", newline="") as f:
        fields = ["sweep", "setting", "packets_per_distance", "n_distances", "seed", "fit_n", "n_lo", "n_hi", "r_lo", "r_hi", "n_unidentifiable"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in sweeps:
            row = {k: r.get(k) for k in fields}
            row["n_unidentifiable"] = unidentifiable_by_setting[(r["sweep"], r["setting"])]
            w.writerow(row)
    colors = apply_style("light")
    recovery_summary = _summary(recovery)
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(PARAMETERS)); width = 0.22
    for i, n in enumerate(N_VALUES):
        xpos = x + (i - 1) * width
        entries = [next(z for z in recovery_summary if z["true_n"] == n and z["parameter"] == p) for p in PARAMETERS]
        for j, entry in enumerate(entries):
            if entry["coverage"] is None:
                continue
            lo, hi = entry["coverage_wilson"]
            value = entry["coverage"]
            ax.bar(xpos[j], value, width, color=[colors["accent"], colors["ok"], colors["textSecondary"]][i],
                   label=f"n={n:g}" if j == 0 else None)
            ax.errorbar(xpos[j], value, yerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]], fmt="none",
                        ecolor=colors["textSecondary"], capsize=3, lw=1)
    ax.axhline(0.95, color=colors["caution"], ls="--", lw=1); ax.set_ylim(0, 1.05); ax.set_ylabel("coverage (simulated)"); ax.set_xticks(x, PARAMETERS); ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(output / "exp018_coverage.png", dpi=150, metadata={"Software": None}); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4));
    for ax, sweep in zip(axes, ("packets", "distances")):
        pts = [(r["setting"], r["n_mean_half_width"], r["r_mean_half_width"])
               for r in sweep_summary if r["sweep"] == sweep]
        pts = [p for p in pts if p[1] is not None or p[2] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colors["accent"], label="n")
        ax.plot([p[0] for p in pts], [p[2] for p in pts], marker="s", color=colors["ok"], label="r")
        ax.set_xlabel("packets per distance" if sweep == "packets" else "number of distances"); ax.set_ylabel("mean half-width"); ax.legend(frameon=False)
    fig.suptitle("Simulated design sweeps"); fig.tight_layout(); fig.savefig(output / "exp018_design.png", dpi=150, metadata={"Software": None}); plt.close(fig)
    prediction = []
    for n in N_VALUES:
        for parameter in PARAMETERS:
            item = next(z for z in recovery_summary
                        if z["true_n"] == n and z["parameter"] == parameter)
            half_width = (item["mean_interval_width"] / 2
                          if parameter == "n" and item["mean_interval_width"] is not None else None)
            prediction.append({
                "true_n": n,
                "parameter": parameter,
                "coverage_within_0.92_0.98": bool(item["coverage"] is not None
                                                    and 0.92 <= item["coverage"] <= 0.98),
                "observed_n_half_width": half_width,
                "predicted_n_half_width_range": [0.15, 0.2] if parameter == "n" else None,
                "within_prediction": (bool(half_width is not None and 0.15 <= half_width <= 0.2)
                                       if parameter == "n" else None),
            })
    summary = {"experiment": "EXP-018", "status": "simulated; literature-derived parameters", "pilot": pilot,
               "recovery": recovery_summary, "sweeps": sweep_summary,
               "prediction_check": prediction,
               "why_n_interval_is_wide": "5 positions with one static shadowing draw each give 3 residual degrees of freedom; the t multiplier (about 3.18) and sigma / sqrt(sum of squared deviations of 10 log10 d) (about 3 / 12.3 dB) set the half-width; more packets per position do not shrink it",
               "packet_note": "With no fast RSSI noise the n half-width is flat in packets by construction (fast_sigma_db = 0); seeds are reused across packet settings (unpaired design).",
               "sizing_table": sizing_table(),
               "burst_aware_sizing_table": burst_aware_sizing_table(),
               "burst_aware_sizing_rule": BURST_AWARE_SIZING_RULE,
               "falsification_criteria": _falsification(),
               "wilson_checks": {"p095_halfwidth001": packets_for_wilson_halfwidth(0.95, 0.01), "p099_halfwidth0005": packets_for_wilson_halfwidth(0.99, 0.005)}}
    write_json(output / "summary.json", summary)
    write_manifest(output, schema="zephyr-run-c-manifest-1", argv=argv, command="PYTHONPATH=. ./.venv/bin/python tools/c_exp018_recovery.py --output {out}",
                   config={"pilot": pilot, "repetitions": reps, "packets_per_distance": 1000,
                           "distances_m": DISTANCES_M, "n_values": N_VALUES,
                           "sweep_packet_settings": [100, 300, 1000, 3000]}, seeds=seeds,
                   outputs=["summary.json", "recovery.csv", "sweeps.csv", "exp018_coverage.png", "exp018_design.png"],
                   limits=["Static shadowing is one draw per distance per trace; no fast RSSI noise is added.", "Synthetic traces are not radio measurements."], runtime_s=time.perf_counter()-start,
                   n_processes=(processes if workers is None else effective_workers(workers, len(cells))),
                   hashed_sources=["tools/c_exp018_recovery.py", "tools/calibrate_radio_from_trace.py", "sim/trace.py", "sim/radio/channel.py"])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv); p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); p.add_argument("--output", type=Path, required=True); p.add_argument("--pilot", action="store_true"); p.add_argument("--processes", type=int, default=default_processes()); p.add_argument("--workers", type=int, default=None, help="worker processes through sim.ordered_pool, at most 4 (opt-in; when given, --processes is ignored; default: the old --processes pool); outputs matched the old path at 1 and 4 workers in tests"); a = p.parse_args(argv); run(a.output, argv, a.pilot, a.processes, a.workers); return 0


if __name__ == "__main__":
    raise SystemExit(main())
