"""EXP-038 (proposal): EXP-017's distance cells rerun with the fading model as a new factor.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/retry_fades.py --output sim/runs/retry-fades-exp038

EXP-017 (registered) drew a fresh shadowing fade for every radio attempt
(``iid``), so a retry a few milliseconds after a failed try rolled new dice.
This runner repeats EXP-017's Scenario A field cells (outdoor stress corner,
processing delay 0, unicast with 5 retries and broadcast, the same 50 seeds)
with five fading settings: ``iid`` and ``static`` (both registered in EXP-017)
and ``time_correlated`` with correlation times 0.1, 0.5 and 2 s.

Nothing in EXP-017 is copied or changed: each (cell, seed) goes through
``c_exp017_tolerance.run_one``.  The only change is that, for the new cells,
the channel gets ``shadowing_corr_time_s`` (see ``_scenario_config`` below).
The ``iid`` and ``static`` rows must equal the registered
``sim/runs/c-exp017-tolerance-r2/per_seed.csv`` rows text for text (the sanity
gate); if they do not, the run is void.  Everything is simulated; radio
parameters are literature-derived.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim import s6_recovery
from tools import c_exp017_tolerance as exp017
from tools import retry_fades_hand as hand
from tools.c_common import (
    REPO_ROOT,
    argv_or_default,
    default_processes,
    prepare_output,
    sha256_file,
    write_json,
    write_manifest,
)
from tools.horizon_mpl import apply_style

REGISTERED_DIR = "sim/runs/c-exp017-tolerance-r2"
# (label, EXP-017 shadowing mode, correlation time in s or None)
FADINGS = (("iid", "iid", None), ("static", "static", None), ("tau_0.1s", "time_correlated", 0.1),
           ("tau_0.5s", "time_correlated", 0.5), ("tau_2s", "time_correlated", 2.0))
MODES = ("unicast", "broadcast")
DISTANCES_M = exp017.DISTANCES_M
N_SEEDS = 50
CHECK_DISTANCES_M = (DISTANCES_M[8], DISTANCES_M[9], DISTANCES_M[10])  # 524, 574, 630 m
P4_DISTANCES_M = (DISTANCES_M[9], DISTANCES_M[10], DISTANCES_M[11])  # 574, 630, 691 m
P3_LIMIT_M = 630.0
CONTRADICT_FROM_M = DISTANCES_M[12]  # 758 m
_ORIGINAL_SCENARIO_CONFIG = exp017.scenario_config


def _scenario_config(cell: dict, seed: int):
    """EXP-017's scenario_config, plus the correlation time for the new cells only.

    Cells without a ``corr_time_s`` (all iid and static cells) get EXP-017's
    configuration object unchanged, so their rows can be compared with the
    registered run.
    """
    cfg = _ORIGINAL_SCENARIO_CONFIG(cell, seed)
    tau = cell.get("corr_time_s")
    if tau is None:
        return cfg
    channel = dataclasses.replace(cfg.radio.channel, shadowing_corr_time_s=float(tau))
    return dataclasses.replace(cfg, radio=dataclasses.replace(cfg.radio, channel=channel))


# run_one looks scenario_config up in its own module at call time; this module is imported
# by every worker process, so the wrapper is in place wherever run_one runs.
exp017.scenario_config = _scenario_config


def build_cells() -> list[dict]:
    cells = []
    for label, shadowing, tau in FADINGS:
        for mode in MODES:
            for d in DISTANCES_M:
                cell = {"scenario": "A", "area": "field", "distance_m": d, "n": 1, "rate_hz": None, "mode": mode,
                        "processing_s": 0.0, "shadowing": shadowing}
                if tau is not None:
                    cell["corr_time_s"] = tau
                cell["fading"] = label
                cells.append(cell)
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


def run_job(job: tuple[dict, int]) -> list[dict]:
    cell, seed = job
    return exp017.run_one((cell, seed))


# ---- sanity gate ---------------------------------------------------------------

def _csv_text(row: dict, fields: list[str]) -> str:
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n").writerow(row)
    return buf.getvalue()


def _same_text(ours: str, registered: str) -> bool:
    """How the gate compares a value with the registered text: character for character."""
    return ours == registered


def sanity_gate(cells: list[dict], rows: list[dict], registered_dir: Path) -> dict:
    """Every iid and static row must match the registered EXP-017 row for the same cell, seed, motion and arm."""
    with open(registered_dir / "cells.csv", newline="") as handle:
        reg_cells = {(r["mode"], r["shadowing"], float(r["distance_m"])): int(r["cell"])
                     for r in csv.DictReader(handle)
                     if r["scenario"] == "A" and r["area"] == "field" and r["processing_s"] == "0.0"}
    with open(registered_dir / "per_seed.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [f for f in reader.fieldnames or [] if f != "cell"]
        reg_rows = {(int(r["cell"]), r["seed"], r["motion"], r["arm"]): r for r in reader}
    by_cell = {c["cell"]: c for c in cells}
    compared = mismatched = 0
    first_mismatch = None
    seen = set()
    for r in rows:
        cell = by_cell[r["cell"]]
        if cell["fading"] not in ("iid", "static"):
            continue
        reg_cell = reg_cells[(cell["mode"], cell["shadowing"], float(cell["distance_m"]))]
        key = (reg_cell, str(r["seed"]), r["motion"], r["arm"])
        seen.add(key)
        reg = reg_rows.get(key)
        ok = reg is not None and _same_text(_csv_text(r, fields), _csv_text(reg, fields))
        compared += 1
        if not ok:
            mismatched += 1
            if first_mismatch is None:
                first_mismatch = {"cell": r["cell"], "seed": r["seed"], "motion": r["motion"], "arm": r["arm"]}
    expected = sum(1 for k in reg_rows if k[0] in set(reg_cells.values()))
    missing = expected - len(seen)
    return {"rows_compared": compared, "rows_mismatched": mismatched, "registered_rows_not_compared": missing,
            "columns_compared": fields, "first_mismatch": first_mismatch,
            "passed": bool(compared > 0 and mismatched == 0 and missing == 0)}


# ---- summaries -------------------------------------------------------------------

def summarize(cells: list[dict], rows: list[dict]) -> list[dict]:
    by_cell: dict[int, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    hand_by_d = {d: hand.hand_cell(d) for d in DISTANCES_M}
    out = []
    for cell in cells:
        rs = by_cell[cell["cell"]]
        per_seed = {r["seed"]: r for r in rs}
        # Every seed must appear once with all four motion and arm rows; never deduplicate silently.
        if len(rs) != 4 * len(per_seed) or any(sum(1 for r in rs if r["seed"] == s) != 4 for s in per_seed):
            raise ValueError(f"cell {cell['cell']}: expected 4 rows per seed, got {len(rs)} rows for {len(per_seed)} seeds")
        sent = sum(r["sent"] for r in per_seed.values())
        received = sum(r["received"] for r in per_seed.values())
        frac = np.array([r["received"] / r["sent"] for r in per_seed.values()])
        half = 1.96 * float(frac.std(ddof=1)) / math.sqrt(len(frac))
        wlo, whi = s6_recovery.wilson_interval(received, sent)
        h = hand_by_d[cell["distance_m"]]
        base = {"cell": cell["cell"], "fading": cell["fading"], "shadowing": cell["shadowing"],
                "corr_time_s": cell.get("corr_time_s"), "mode": cell["mode"], "distance_m": cell["distance_m"],
                "n_seeds": len(per_seed), "delivery_pooled": received / sent,
                "delivery_seed_mean": float(frac.mean()), "delivery_seed_low": float(frac.mean()) - half,
                "delivery_seed_high": float(frac.mean()) + half,
                "delivery_wilson_low": wlo, "delivery_wilson_high": whi,
                "wilson_valid": cell["fading"] == "iid",
                "hand_one_attempt": h["broadcast_delivery"], "hand_iid_unicast": h["unicast_iid_delivery"],
                "hand_frozen_unicast": h["unicast_frozen_delivery"]}
        for motion in s6_recovery.MOTIONS:
            for arm in s6_recovery.ARMS:
                sub = [r for r in rs if r["motion"] == motion and r["arm"] == arm]
                fails = sum(r["failed"] for r in sub)
                flo, fhi = s6_recovery.wilson_interval(fails, len(sub))
                key = f"{arm}_{motion}"
                base[f"{key}_fail_fraction"] = fails / len(sub)
                base[f"{key}_fail_wilson_low"] = flo
                base[f"{key}_fail_wilson_high"] = fhi
                base[f"{key}_cell_failed"] = bool(2 * fails >= len(sub))  # EXP-006 cell rule, still a proposal
        out.append(base)
    return out


def _get(summary: list[dict], fading: str, mode: str, d: float) -> dict:
    return next(c for c in summary if c["fading"] == fading and c["mode"] == mode and c["distance_m"] == d)


def first_failing(summary: list[dict], fading: str, mode: str, key: str) -> dict:
    cs = sorted((c for c in summary if c["fading"] == fading and c["mode"] == mode), key=lambda c: c["distance_m"])
    first = next((c for c in cs if c[f"{key}_cell_failed"]), None)
    if first is None:
        return {"first_failing_m": None, "delivery_there": None, "last_passing_below_m": cs[-1]["distance_m"]}
    below = [c["distance_m"] for c in cs if c["distance_m"] < first["distance_m"] and not c[f"{key}_cell_failed"]]
    return {"first_failing_m": first["distance_m"], "delivery_there": first["delivery_pooled"],
            "fail_fraction_there": first[f"{key}_fail_fraction"],
            "last_passing_below_m": max(below) if below else None}


def evaluate(summary: list[dict]) -> dict:
    """The pre-registered predictions and verdict (packet: 2026-10-21 retry behavior under slow fades)."""
    preds = {}
    # P1: tau 0.5 and 2 s unicast delivery within 0.05 of the hand frozen-fade value where that is in [0.05, 0.95]
    p1 = {}
    for fading in ("tau_0.5s", "tau_2s"):
        diffs = [(c["distance_m"], c["delivery_pooled"] - c["hand_frozen_unicast"]) for c in summary
                 if c["fading"] == fading and c["mode"] == "unicast" and 0.05 <= c["hand_frozen_unicast"] <= 0.95]
        worst = max(diffs, key=lambda x: abs(x[1]))
        p1[fading] = {"distances_checked": len(diffs), "worst_distance_m": worst[0], "worst_difference": worst[1],
                      "holds": all(abs(x) <= 0.05 for _, x in diffs)}
    preds["P1"] = {"rule": "tau 0.5 and 2 s unicast delivery within 0.05 of hand E_z[1 - PER^6]", "by_fading": p1,
                   "holds": all(v["holds"] for v in p1.values())}
    # P2: at tau 2 s the retry gain is at most a third of the iid gain at 524, 574, 630 m
    p2 = []
    for d in CHECK_DISTANCES_M:
        g_iid = _get(summary, "iid", "unicast", d)["delivery_pooled"] - _get(summary, "iid", "broadcast", d)["delivery_pooled"]
        g_tau = (_get(summary, "tau_2s", "unicast", d)["delivery_pooled"]
                 - _get(summary, "tau_2s", "broadcast", d)["delivery_pooled"])
        p2.append({"distance_m": d, "gain_iid": g_iid, "gain_tau_2s": g_tau, "holds": g_tau <= g_iid / 3.0})
    preds["P2"] = {"rule": "tau 2 s retry gain <= iid retry gain / 3 at 524, 574, 630 m", "by_distance": p2,
                   "holds": all(x["holds"] for x in p2)}
    # P3: recovery arm, 1 m turn, unicast first failing distance <= 630 m at tau 0.5 and 2 s
    p3 = {f: first_failing(summary, f, "unicast", "recovery_constant_turn") for f in ("tau_0.5s", "tau_2s")}
    for v in p3.values():
        v["holds"] = v["first_failing_m"] is not None and v["first_failing_m"] <= P3_LIMIT_M
    preds["P3"] = {"rule": "recovery 1 m turn unicast first failing distance <= 630 m at tau 0.5 and 2 s",
                   "by_fading": p3, "holds": all(v["holds"] for v in p3.values())}
    # P4 (not blind): tau 0.1 s unicast strictly between the hand frozen value and the iid simulated value
    p4 = []
    for d in P4_DISTANCES_M:
        c = _get(summary, "tau_0.1s", "unicast", d)
        lo, hi = c["hand_frozen_unicast"], _get(summary, "iid", "unicast", d)["delivery_pooled"]
        p4.append({"distance_m": d, "frozen_hand": lo, "tau_0.1s": c["delivery_pooled"], "iid_sim": hi,
                   "holds": lo < c["delivery_pooled"] < hi})
    preds["P4"] = {"rule": "tau 0.1 s unicast strictly between hand frozen and simulated iid at 574, 630, 691 m",
                   "blind": False, "by_distance": p4, "holds": all(x["holds"] for x in p4)}
    # P5 control: broadcast delivery does not depend on the fading (seed-level intervals overlap the iid one)
    p5 = []
    for d in DISTANCES_M:
        ref = _get(summary, "iid", "broadcast", d)
        for fading, _, _ in FADINGS[1:]:
            c = _get(summary, fading, "broadcast", d)
            gap = abs(c["delivery_seed_mean"] - ref["delivery_seed_mean"])
            allow = (c["delivery_seed_high"] - c["delivery_seed_mean"]) + (ref["delivery_seed_high"] - ref["delivery_seed_mean"])
            p5.append({"distance_m": d, "fading": fading, "gap": gap, "allowed": allow, "holds": gap <= allow})
    preds["P5"] = {"rule": "broadcast delivery agrees with iid within the seed-level intervals at every distance",
                   "failures": [x for x in p5 if not x["holds"]], "checked": len(p5),
                   "holds": all(x["holds"] for x in p5)}
    turn = {f: first_failing(summary, f, "unicast", "recovery_constant_turn")["first_failing_m"]
            for f in ("tau_0.5s", "tau_2s")}
    if preds["P2"]["holds"] and preds["P3"]["holds"]:
        verdict = "SUPPORTED"
    # Amendment A1 (written before the full run): "never fails on the grid" (passes to 1000 m)
    # counts as "758 m or more", since it is further out than any grid distance.
    elif all(v is None or v >= CONTRADICT_FROM_M for v in turn.values()):
        verdict = "CONTRADICTED"
    else:
        verdict = "MIXED"
    return {"predictions": preds, "verdict": verdict,
            "verdict_rule": ("SUPPORTED if P2 and P3 hold; CONTRADICTED if the recovery 1 m turn unicast first failing "
                             "distance is >= 758 m at both tau 0.5 and 2 s, where never failing on the grid counts as "
                             ">= 758 m (amendment A1, before the full run); MIXED otherwise")}


def headline(summary: list[dict]) -> dict:
    out = {}
    for mode in MODES:
        for motion in s6_recovery.MOTIONS:
            for arm in s6_recovery.ARMS:
                key = f"{arm}_{motion}"
                out[f"{mode}_{key}"] = {f: first_failing(summary, f, mode, key) for f, _, _ in FADINGS}
    return out


# ---- figure ------------------------------------------------------------------------

def make_figure(summary: list[dict], output: Path) -> None:
    colors = apply_style("light")
    palette = {"iid": colors["accent"], "static": colors["textSecondary"], "tau_0.1s": colors["ok"],
               "tau_0.5s": colors["keepOut"], "tau_2s": colors["text"]}
    labels = {"iid": "new fade every attempt (EXP-017)", "static": "one fade per seed (EXP-017)",
              "tau_0.1s": "drifting fade, 0.1 s", "tau_0.5s": "drifting fade, 0.5 s", "tau_2s": "drifting fade, 2 s"}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for col, mode in enumerate(MODES):
        ax, axf = axes[0][col], axes[1][col]
        for fading, _, _ in FADINGS:
            cs = sorted((c for c in summary if c["fading"] == fading and c["mode"] == mode), key=lambda c: c["distance_m"])
            d = [c["distance_m"] for c in cs]
            ax.plot(d, [c["delivery_pooled"] for c in cs], color=palette[fading], marker="o", ms=3, label=labels[fading])
            axf.plot(d, [c["recovery_constant_turn_fail_fraction"] for c in cs], color=palette[fading], marker="o",
                     ms=3, label=labels[fading])
        hd = [hand.hand_cell(x) for x in DISTANCES_M]
        if mode == "unicast":
            ax.plot(DISTANCES_M, [h["unicast_frozen_delivery"] for h in hd], color=colors["alert"], ls=":", lw=1.2,
                    label="hand: fade frozen over retries")
            ax.plot(DISTANCES_M, [h["unicast_iid_delivery"] for h in hd], color=colors["alert"], ls="--", lw=0.9,
                    label="hand: 1 - p^6")
        else:
            ax.plot(DISTANCES_M, [h["broadcast_delivery"] for h in hd], color=colors["alert"], ls="--", lw=0.9,
                    label="hand: 1 - p")
        ax.set_title(f"{mode}{' with 5 retries' if mode == 'unicast' else ', no retries'}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        axf.axhline(0.5, color=colors["alert"], lw=1.0, ls="--")
        axf.set_ylim(-0.02, 1.02)
        axf.set_xscale("log")
        ticks = [250, 300, 400, 500, 600, 800, 1000]
        axf.set_xticks(ticks, [str(t) for t in ticks])
        axf.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axf.set_xlabel("distance to ground station (m, log scale)")
    axes[0][0].set_ylabel("tracker packets delivered")
    axes[1][0].set_ylabel("fraction of 50 seeds failing\n(recovery arm, 1 m turn)")
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    axes[0][1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Simulated (EXP-038 proposal): retries under slow fades, outdoor stress corner, 10 Hz tracker",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "exp038_retry_fades.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


# ---- run -----------------------------------------------------------------------------

def run(output: Path, argv: list[str], *, processes: int | None = None, distances=None, n_seeds: int = N_SEEDS) -> dict:
    start = time.perf_counter()
    output = prepare_output(output)
    seeds = s6_recovery.seed_list(n_seeds)
    cells = build_cells()
    if distances is not None:  # pilot only: a subset of the registered distances
        keep = {float(d) for d in distances}
        cells = [c for c in cells if c["distance_m"] in keep]
    jobs = [(c, s) for c in cells for s in seeds]
    proc = default_processes() if processes is None else int(processes)
    if proc > 1:
        with ProcessPoolExecutor(max_workers=proc) as pool:
            rows = [r for chunk in pool.map(run_job, jobs, chunksize=4) for r in chunk]
    else:
        rows = [r for job in jobs for r in run_job(job)]
    gate = sanity_gate(cells, rows, REPO_ROOT / REGISTERED_DIR) if n_seeds == N_SEEDS else {"passed": None,
                                                                                            "note": "pilot seeds"}
    summary_cells = summarize(cells, rows)
    full = distances is None and n_seeds == N_SEEDS
    if full and any(c["n_seeds"] != N_SEEDS for c in summary_cells):
        raise ValueError("a full-run cell does not have exactly 50 seeds")
    void = full and not gate["passed"]  # pre-registered: one mismatch voids the run
    by_cell = {c["cell"]: c for c in cells}
    with open(output / "per_seed.csv", "w", newline="") as handle:
        fields = ["fading", "corr_time_s", "mode", "distance_m", *rows[0]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            c = by_cell[r["cell"]]
            writer.writerow({"fading": c["fading"], "corr_time_s": c.get("corr_time_s"), "mode": c["mode"],
                             "distance_m": c["distance_m"], **r})
    with open(output / "cells.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_cells[0]))
        writer.writeheader()
        writer.writerows(summary_cells)
    status = "proposal; simulated; radio parameters literature-derived"
    summary = {"experiment": "EXP-038", "status": f"VOID (sanity gate failed); {status}" if void else status,
               "full_run": full, "n_seeds": len(seeds), "sanity_gate": gate,
               "evaluation": evaluate(summary_cells) if full and not void else None,
               "headline_first_failing_distance": headline(summary_cells) if full and not void else None,
               "cells": summary_cells,
               "registered_exp017_per_seed_sha256": sha256_file(REPO_ROOT / REGISTERED_DIR / "per_seed.csv"),
               "rules": {"seed_fails": "p95 error > 0.25 m or max error > 0.50 m after the 2 s settling period (EXP-006)",
                         "cell_fails": "at least 50 % of seeds fail (EXP-006 cell rule, still proposed)",
                         "delivery": "received over sent, all 121 tracker packets, any arrival time",
                         "delivery_interval": ("seed-level: mean of per-seed delivery +/- 1.96 sd / sqrt(n); the pooled "
                                               "Wilson interval is valid only for iid fading"),
                         "retry_gain": "unicast delivery minus broadcast delivery at the same distance and fading"}}
    write_json(output / "summary.json", summary)
    if void:
        raise SystemExit(f"EXP-038 VOID: sanity gate failed ({gate['rows_mismatched']} mismatched rows, "
                         f"{gate['registered_rows_not_compared']} registered rows not compared); see summary.json")
    if full:
        make_figure(summary_cells, output)
    runtime = time.perf_counter() - start
    print(f"runtime {runtime:.1f} s for {len(jobs)} radio runs on {proc} processes; gate passed: {gate['passed']}")
    outputs = ["summary.json", "per_seed.csv", "cells.csv"] + (["exp038_retry_fades.png"] if full else [])
    config = {"fadings": [list(f) for f in FADINGS], "modes": list(MODES), "distances_m": [c for c in DISTANCES_M
              if distances is None or c in {float(d) for d in distances}], "processing_s": 0.0,
              "n_cells": len(cells), "n_seeds": len(seeds), "registered_dir": REGISTERED_DIR}
    return write_manifest(
        output, schema="zephyr-run-c-manifest-1", argv=argv,
        command="PYTHONPATH=. ./.venv/bin/python tools/retry_fades.py --output {out}", config=config,
        seeds=seeds, outputs=outputs,
        limits=["Proposal (EXP-038), not Adopted; simulated only, radio parameters literature-derived.",
                "Correlation times 0.1, 0.5 and 2 s are assumptions; no ESP-NOW log has measured one.",
                "One drone at a fixed distance, outdoor stress corner (sigma 4 dB, 6 dB orientation loss).",
                "The EXP-006 cell rule (50 % of seeds) is still a proposal, not an adopted rule."],
        runtime_s=runtime, n_processes=proc,
        hashed_sources=["sim/s6_recovery.py", "sim/tracker.py", "sim/sensor.py", "sim/radio/channel.py",
                        "sim/radio/mac.py", "sim/radio/model.py", "sim/radio/params.py", "sim/radio/seq.py",
                        "tools/c_exp017_tolerance.py", "tools/retry_fades.py", "tools/retry_fades_hand.py",
                        exp017.EXP006_SUMMARY, f"{REGISTERED_DIR}/per_seed.csv", f"{REGISTERED_DIR}/cells.csv"])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=default_processes())
    parser.add_argument("--pilot", action="store_true", help="3 seeds at 524 and 630 m (no gate, no evaluation)")
    args = parser.parse_args(argv)
    if args.pilot:
        run(args.output, argv, processes=args.processes, distances=(DISTANCES_M[8], DISTANCES_M[10]), n_seeds=3)
    else:
        run(args.output, argv, processes=args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
