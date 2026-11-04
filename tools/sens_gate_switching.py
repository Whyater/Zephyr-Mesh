#!/usr/bin/env python3
"""Gated CV, CT and IMM trackers on targets that switch between kinds of motion.

Follow-up to ``tools/sens_outlier_gate_turn.py``. That study ran the 0.99 NIS
gate with CV, CT and IMM trackers on the S6 straight line and circle, which are
the exact models of CV and CT. Here the same six arms, the same packets
pipeline and the same 1 m outliers run on two targets that change how they move:

- ``figure8``: a figure 8 whose turn direction flips twice per loop;
- ``start_stop``: a straight line that keeps accelerating, cruising, braking
  and standing still.

Both are the estimation engine's existing targets with their default
parameters, the same cells the S6 atlas registered. The only difference from
the parent study is the cell (``switching_cell``); outliers come from the
parent's ``inject`` and the arms and per-arm summary from the turn study.

Each gated arm is also compared seed by seed with its ungated twin
(``paired_gate_cost``), because on start-stop at 0.2 s about half the seeds
already fail with no gate and no outliers.

``--control`` runs only the parent's control cell (CV on the circle, 0.2 s, no
outliers) through this wrapper's cell path and prints the failure counts.

Nothing under ``sim/`` is changed or registered. Outputs go to ``--output``
and must stay out of ``sim/runs/``.

Run (pilot first):
  PYTHONPATH=. python tools/sens_gate_switching.py --output sens_runs/switching-pilot --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sim.estimation.base import RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm
from sim.s6_recovery import seed_list
from tools.sens_outlier_gate import (
    GATED,
    NOISE_STD_M,
    OUTLIER_STREAM,
    OutlierSpec,
    _git_revision,
    inject,
)
from tools.sens_outlier_gate_turn import (
    ARMS,
    DELAYS_S,
    MODEL_LABEL,
    PAIRS,
    PROBABILITIES,
    predicted_clean_reject_rate,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("figure8", "start_stop")
TARGET_TITLE = {"figure8": "figure 8", "start_stop": "start-stop line", "ct": "circle, 1 m radius"}
CONTROL_DELAY_S = 0.2


def switching_cell(target: str, delay_s: float) -> CellSpec:
    """One cell: the target with its default parameters, S6 central settings otherwise."""
    return CellSpec(target=target, delay_s=float(delay_s), noise_std_m=NOISE_STD_M)


def cell_runs(target: str, delay_s: float, spec: OutlierSpec, seeds, arms=ARMS) -> list[dict]:
    """Every arm on every seed of one cell, all arms on the same packets. One row per (seed, arm).

    Same row layout as the turn study's ``cell_runs`` with ``motion`` holding the target name.
    """
    cell = switching_cell(target, delay_s)
    rows = []
    for index, seed in enumerate(seeds):
        bundle = inject(deliver_cell(cell, int(seed)), spec, int(seed))
        for arm in arms:
            o = run_arm(bundle, cell, arm)
            x = o.extras
            rows.append({
                "motion": target, "delay_s": delay_s, "outlier": spec.label(), "kind": spec.kind,
                "probability": spec.probability, "arm": arm.name, "model": arm.model,
                "gated": arm.gate_probability is not None, "seed_index": index, "seed": int(seed),
                "p95_error_m": o.p95_error_m, "max_error_m": o.max_error_m, "rmse_m": o.rmse_m,
                "mean_nis": o.mean_nis, "failed": o.failed, "fused_count": o.fused_count,
                "gated_count": o.gated_count, "outlier_count": len(bundle.outlier_seqs),
                "gated_outliers": x.get("gated_outliers"), "gated_clean": x.get("gated_clean"),
                "checked_outliers": x.get("gate_checked_outliers"),
                "checked_clean": x.get("gate_checked_clean"),
                "forced_accepts": x.get("gate_forced_accepts"),
                "_outcome": o,
            })
    return rows


def paired_gate_cost(rows: list[dict]) -> list[dict]:
    """For each (ungated, gated) pair: seeds only the gated arm fails, only the ungated one fails,
    and the median of (gated p95 - ungated p95) over seeds. Rows must come from one cell."""
    out = []
    for plain, gated in PAIRS:
        a = {r["seed"]: r for r in rows if r["arm"] == plain}
        b = {r["seed"]: r for r in rows if r["arm"] == gated}
        if not a or set(a) != set(b):
            continue
        seeds = sorted(a)
        out.append({
            "model": MODEL_LABEL[plain], "ungated_fails": sum(bool(a[s]["failed"]) for s in seeds),
            "gated_fails": sum(bool(b[s]["failed"]) for s in seeds),
            "only_gated_fails": sum(bool(b[s]["failed"]) and not bool(a[s]["failed"]) for s in seeds),
            "only_ungated_fails": sum(bool(a[s]["failed"]) and not bool(b[s]["failed"]) for s in seeds),
            "median_p95_diff_m": statistics.median(float(b[s]["p95_error_m"]) - float(a[s]["p95_error_m"])
                                                   for s in seeds),
        })
    return out


def plan() -> list[tuple]:
    """Every (target, delay, spec) in the pre-registered sweep."""
    return [(t, d, OutlierSpec("jump", p)) for t in TARGETS for d in DELAYS_S for p in PROBABILITIES]


def run_control(n_seeds: int) -> dict:
    """The parent's control cell (CV on the circle, 0.2 s, no outliers) through this wrapper."""
    rows = cell_runs("ct", CONTROL_DELAY_S, OutlierSpec("jump", 0.0), seed_list(n_seeds),
                     arms=(RECOVERY_V1, GATED))
    return {arm: sum(bool(r["failed"]) for r in rows if r["arm"] == arm) for arm in ("recovery", "recovery_gated")}


FIGURE_PANELS = (("figure8", 0.2), ("start_stop", 0.1), ("start_stop", 0.2))


def figure(summary: list[dict], path: Path, n_seeds: int, panels=FIGURE_PANELS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 6.6), sharex=True,
                             constrained_layout=True)
    xs = [100 * p for p in PROBABILITIES]
    dodge = np.linspace(-0.12, 0.12, len(ARMS))  # keeps equal values visible; ticks stay at the true rates
    for col, (target, delay) in enumerate(panels):
        for k, arm in enumerate(ARMS):
            xd = [x + dodge[k] for x in xs]
            style = horizon_mpl.arm_style(arm.name, pal)
            rows = [next(r for r in summary if r["motion"] == target and r["delay_s"] == delay
                         and r["probability"] == p and r["arm"] == arm.name) for p in PROBABILITIES]
            med = np.array([r["p95_median_m"] for r in rows])
            lo = np.array([r["p95_ci_low_m"] for r in rows])
            hi = np.array([r["p95_ci_high_m"] for r in rows])
            label = MODEL_LABEL.get(arm.name.split("_")[0], arm.name) + (", gate on" if arm.gate_probability else "")
            kw = {"color": style["color"], "linestyle": style["linestyle"], "marker": style["marker"],
                  "label": label, "linewidth": 1.6}
            axes[0, col].plot(xd, [r["fail_count"] for r in rows], **kw)
            axes[1, col].errorbar(xd, med, yerr=[med - lo, hi - med], capsize=2, **kw)
        axes[0, col].set_title(f"{TARGET_TITLE[target]}, delay {delay:g} s")
        axes[0, col].set_ylabel(f"failing seeds (of {n_seeds})")
        axes[0, col].set_ylim(-0.04 * n_seeds, 1.04 * n_seeds)
        axes[1, col].axhline(0.25, color=pal["lineStrong"], linewidth=0.8, linestyle="--")
        axes[1, col].annotate("S6 fail line (p95 0.25 m)", (1.0, 0.25), xycoords=("axes fraction", "data"),
                              xytext=(-2, -3), textcoords="offset points", ha="right", va="top", fontsize=8,
                              color=pal["textSecondary"])
        axes[1, col].set_ylabel("median p95 error (m)")
        axes[1, col].set_xlabel("outlier frames (%)")
        axes[1, col].set_xticks(xs)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(ARMS), fontsize=8)
    fig.suptitle("1 m outliers on switching targets: CV, CT and IMM with and without the 0.99 NIS gate\n"
                 "(synthetic; points nudged sideways so equal values stay visible)", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--control", action="store_true", help="run only the parent's CV circle control cell")
    ap.add_argument("--shard", default="0/1",
                    help="i/k: run only cells with index %% k == i (separate processes share the checkpoint)")
    args = ap.parse_args(argv)
    if args.control:
        print(json.dumps({"control": "ct circle, 0.2 s, no outliers", "n_seeds": args.seeds,
                          "fail_counts": run_control(args.seeds)}))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --control")
    out = args.output.resolve()
    if (ROOT / "sim" / "runs") in [out, *out.parents]:
        raise SystemExit("refusing to write under sim/runs/: this study is not registered")
    shard_i, shard_k = (int(v) for v in args.shard.split("/"))
    if not 0 <= shard_i < shard_k:
        raise SystemExit("--shard must be i/k with 0 <= i < k")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    seeds = seed_list(args.seeds)
    jobs = plan()
    # One checkpoint file per finished cell; rerunning the same command resumes.
    ckpt = out.with_name(out.name + ".ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    stamp = {"n_seeds": args.seeds, "n_cells": len(jobs), "arms": [a.name for a in ARMS], "targets": list(TARGETS)}
    stamp_path = ckpt / "plan.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise SystemExit(f"{ckpt} holds a different plan; delete it to start over")
    stamp_path.write_text(json.dumps(stamp))
    start = time.time()
    for i, (target, delay, spec) in enumerate(jobs):
        if i % shard_k != shard_i or (ckpt / f"cell{i:03d}.json").exists():
            continue
        cpu = time.process_time()
        rows = cell_runs(target, delay, spec, seeds)
        body = {"cpu_s": time.process_time() - cpu, "summary": summarize(target, delay, spec, rows),
                "paired": [dict(p, motion=target, delay_s=delay, probability=spec.probability)
                           for p in paired_gate_cost(rows)],
                "per_seed": [{k: v for k, v in r.items() if k != "_outcome"} for r in rows]}
        part = ckpt / f"cell{i:03d}.json.part"
        part.write_text(json.dumps(body, default=float))
        os.replace(part, ckpt / f"cell{i:03d}.json")
        print(f"cell {i + 1}/{len(jobs)} done", flush=True)
    missing = [i for i in range(len(jobs)) if not (ckpt / f"cell{i:03d}.json").exists()]
    if missing:
        print(f"{len(missing)} cells still missing (other shards?); rerun without --shard to assemble")
        return 0
    per_seed, summary, paired, worker_cpu_s = [], [], [], 0.0
    for i in range(len(jobs)):
        body = json.loads((ckpt / f"cell{i:03d}.json").read_text())
        worker_cpu_s += body["cpu_s"]
        summary.extend(body["summary"])
        paired.extend(body["paired"])
        per_seed.extend(body["per_seed"])
    # Hand check: predicted clean reject rate from each ungated twin's median mean NIS at 0 % outliers.
    for row in summary:
        if row["gated"] and row["probability"] == 0.0:
            twin = next(r for r in summary if r["motion"] == row["motion"] and r["delay_s"] == row["delay_s"]
                        and r["probability"] == 0.0 and r["arm"] == row["arm"].replace("_gated", ""))
            row["predicted_clean_reject_rate"] = predicted_clean_reject_rate(twin["mean_nis_median"])
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with (tmp / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    meta = {"schema": "zephyr-sens-gate-switching-1", "argv": sys.argv[1:] if argv is None else list(argv),
            "git_revision": _git_revision(), "n_seeds": args.seeds, "seeds": seeds, "targets": list(TARGETS),
            "cells": {t: switching_cell(t, 0.0).as_dict() for t in TARGETS},
            "runtime_s_this_session": time.time() - start, "worker_cpu_s": worker_cpu_s,
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": sys.platform,
            "arms": [a.as_dict() for a in ARMS], "outlier_stream": OUTLIER_STREAM, "registered": False,
            "summary": summary, "paired_gate_cost": paired}
    (tmp / "summary.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    figure(summary, tmp / "switching_gate.png", args.seeds)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    shutil.rmtree(ckpt)
    print(f"{len(jobs)} cells x {args.seeds} seeds x {len(ARMS)} arms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
