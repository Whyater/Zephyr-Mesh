#!/usr/bin/env python3
"""Outlier sensitivity with turn-aware trackers: CV, CT and IMM, each with and without the NIS gate.

Follow-up to ``tools/sens_outlier_gate.py``. That study found the 0.99 NIS gate
removes 1 m outliers on the straight line, but on the 1 m circle the constant
velocity (CV) filter lags the turn, about 2 % of clean packets look like
outliers, and the gate makes things worse with no outliers at all. This wrapper
asks whether a gated turn-aware tracker keeps the protection without that cost.

It reuses the parent's packets and outliers unchanged (``deliver_cell`` and the
parent's ``inject``), so the ``recovery`` and ``recovery_gated`` rows here must
equal the parent's rows for the same cells and seeds. Six arms, all with
capture stamps and late packets dropped:

- ``recovery`` / ``recovery_gated``: CV (the parent's two arms);
- ``ct`` / ``ct_gated``: coordinated turn;
- ``imm`` / ``imm_gated``: the CV/CT interacting multiple model.

Gated arms use the 0.99 chi-square gate (11.34 for 3 axes) with the default
lockout guard (after 3 rejections in a row the next packet is fused anyway).
The gated CT and IMM specs equal ``ct_gated`` and ``imm_gated`` in the S6 atlas.

Nothing under ``sim/`` is changed or registered. Outputs go to ``--output``
and must stay out of ``sim/runs/``.

Run (pilot first):
  PYTHONPATH=. python tools/sens_outlier_gate_turn.py --output sens_runs/turn-gate-pilot --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from sim.estimation.base import RECOVERY_V1, ArmSpec
from sim.estimation.engine import deliver_cell, run_arm, s6_cell
from sim.s6_recovery import seed_list, summarize_arm
from tools.sens_outlier_gate import (GATE_PROBABILITY, GATED, NOISE_STD_M, OUTLIER_STREAM, OutlierSpec,
                                     _git_revision, inject)

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ("constant_velocity", "constant_turn")
DELAYS_S = (0.0, 0.1, 0.2)
PROBABILITIES = (0.0, 0.01, 0.03)
ARMS = (
    RECOVERY_V1,
    GATED,
    ArmSpec("ct", model="ct", stamping="capture", late_policy="drop"),
    ArmSpec("ct_gated", model="ct", stamping="capture", late_policy="drop", gate_probability=GATE_PROBABILITY),
    ArmSpec("imm", model="imm", stamping="capture", late_policy="drop"),
    ArmSpec("imm_gated", model="imm", stamping="capture", late_policy="drop", gate_probability=GATE_PROBABILITY),
)
PAIRS = (("recovery", "recovery_gated"), ("ct", "ct_gated"), ("imm", "imm_gated"))
MODEL_LABEL = {"recovery": "CV", "ct": "CT", "imm": "IMM"}


def cell_runs(motion: str, delay_s: float, spec: OutlierSpec, seeds, arms=ARMS) -> list[dict]:
    """Every arm on every seed of one cell, all arms on the same packets. One row per (seed, arm)."""
    cell = s6_cell(motion, delay_s, 0.0, NOISE_STD_M)
    rows = []
    for index, seed in enumerate(seeds):
        bundle = inject(deliver_cell(cell, int(seed)), spec, int(seed))
        for arm in arms:
            o = run_arm(bundle, cell, arm)
            x = o.extras
            rows.append({
                "motion": motion, "delay_s": delay_s, "outlier": spec.label(), "kind": spec.kind,
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


def plan() -> list[tuple]:
    """Every (motion, delay, spec) in the pre-registered sweep. Only the parent's jump kind."""
    return [(m, d, OutlierSpec("jump", p)) for m in MOTIONS for d in DELAYS_S for p in PROBABILITIES]


def summarize(motion, delay, spec, rows, arms=ARMS) -> list[dict]:
    out = []
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm.name]
        s = summarize_arm([r["_outcome"] for r in mine])
        total = lambda key: sum(int(r[key] or 0) for r in mine)  # noqa: E731
        gated = arm.gate_probability is not None
        checked_clean, checked_out = total("checked_clean"), total("checked_outliers")
        out.append({
            "motion": motion, "delay_s": delay, "outlier": spec.label(), "kind": spec.kind,
            "probability": spec.probability, "arm": arm.name, "model": arm.model, "gated": gated,
            "n_seeds": s["n_seeds"], "p95_median_m": s["p95_median_m"], "p95_ci_low_m": s["p95_ci_low_m"],
            "p95_ci_high_m": s["p95_ci_high_m"], "max_error_median_m": s["max_error_median_m"],
            "fail_count": s["fail_count"], "fail_rate_ci_low": s["fail_rate_ci_low"],
            "fail_rate_ci_high": s["fail_rate_ci_high"], "mean_nis_median": s["mean_nis_median"],
            "outliers_total": total("outlier_count"),
            "clean_reject_rate": (total("gated_clean") / checked_clean) if gated and checked_clean else None,
            "outlier_reject_rate": (total("gated_outliers") / checked_out) if gated and checked_out else None,
            "checked_clean": checked_clean if gated else None,
            "checked_outliers": checked_out if gated else None,
            "forced_accepts": total("forced_accepts") if gated else None,
        })
    return out


def predicted_clean_reject_rate(mean_nis: float, probability: float = GATE_PROBABILITY, dof: int = 3) -> float:
    """Clean reject rate if NIS were a chi-square with ``dof`` degrees of freedom scaled to mean ``mean_nis``.

    A consistent filter has mean NIS = dof and rejects 1 - probability. This is the
    hand model the packet compares the measured rates against.
    """
    from scipy.stats import chi2

    threshold = float(chi2.ppf(probability, dof))
    return float(chi2.sf(threshold * dof / float(mean_nis), dof))


def figure(summary: list[dict], path: Path, n_seeds: int, delay: float = 0.2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, constrained_layout=True)
    xs = [100 * p for p in PROBABILITIES]
    for col, motion in enumerate(MOTIONS):
        for arm in ARMS:
            style = horizon_mpl.arm_style(arm.name, pal)
            rows = [next(r for r in summary if r["motion"] == motion and r["delay_s"] == delay
                         and r["probability"] == p and r["arm"] == arm.name) for p in PROBABILITIES]
            med = np.array([r["p95_median_m"] for r in rows])
            lo = np.array([r["p95_ci_low_m"] for r in rows])
            hi = np.array([r["p95_ci_high_m"] for r in rows])
            label = MODEL_LABEL.get(arm.name.split("_")[0], arm.name) + (", gate on" if arm.gate_probability else "")
            kw = dict(color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                      label=label, linewidth=1.6)
            axes[0, col].plot(xs, [r["fail_count"] for r in rows], **kw)
            axes[1, col].errorbar(xs, med, yerr=[med - lo, hi - med], capsize=2, **kw)
        title = "straight line, 1 m/s" if motion == "constant_velocity" else "circle, 1 m radius"
        axes[0, col].set_title(f"{title}, delay {delay:g} s")
        axes[0, col].set_ylabel(f"failing seeds (of {n_seeds})")
        axes[0, col].set_ylim(-0.04 * n_seeds, 1.04 * n_seeds)
        axes[1, col].axhline(0.25, color=pal["lineStrong"], linewidth=0.8, linestyle="--")
        axes[1, col].annotate("S6 fail line (p95 0.25 m)", (xs[-1], 0.25), xytext=(0, 3),
                              textcoords="offset points", ha="right", va="bottom", fontsize=8,
                              color=pal["textSecondary"])
        axes[1, col].set_ylabel("median p95 error (m)")
        axes[1, col].set_xlabel("outlier frames (%)")
        axes[1, col].set_xticks(xs)
    axes[0, 0].legend(loc="center right", fontsize=8)
    fig.suptitle("1 m outliers: CV, CT and IMM trackers with and without the 0.99 NIS gate (synthetic)",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", default="0/1",
                    help="i/k: run only cells with index %% k == i (separate processes share the checkpoint)")
    args = ap.parse_args(argv)
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
    stamp = {"n_seeds": args.seeds, "n_cells": len(jobs), "arms": [a.name for a in ARMS]}
    stamp_path = ckpt / "plan.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise SystemExit(f"{ckpt} holds a different plan; delete it to start over")
    stamp_path.write_text(json.dumps(stamp))
    start = time.time()
    for i, (motion, delay, spec) in enumerate(jobs):
        if i % shard_k != shard_i or (ckpt / f"cell{i:03d}.json").exists():
            continue
        cpu = time.process_time()
        rows = cell_runs(motion, delay, spec, seeds)
        body = {"cpu_s": time.process_time() - cpu, "summary": summarize(motion, delay, spec, rows),
                "per_seed": [{k: v for k, v in r.items() if k != "_outcome"} for r in rows]}
        part = ckpt / f"cell{i:03d}.json.part"
        part.write_text(json.dumps(body, default=float))
        os.replace(part, ckpt / f"cell{i:03d}.json")
        print(f"cell {i + 1}/{len(jobs)} done", flush=True)
    missing = [i for i in range(len(jobs)) if not (ckpt / f"cell{i:03d}.json").exists()]
    if missing:
        print(f"{len(missing)} cells still missing (other shards?); rerun without --shard to assemble")
        return 0
    per_seed, summary, worker_cpu_s = [], [], 0.0
    for i in range(len(jobs)):
        body = json.loads((ckpt / f"cell{i:03d}.json").read_text())
        worker_cpu_s += body["cpu_s"]
        summary.extend(body["summary"])
        per_seed.extend(body["per_seed"])
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with (tmp / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    meta = {"schema": "zephyr-sens-outlier-gate-turn-1", "argv": sys.argv[1:] if argv is None else list(argv),
            "git_revision": _git_revision(), "n_seeds": args.seeds, "seeds": seeds,
            "runtime_s_this_session": time.time() - start, "worker_cpu_s": worker_cpu_s,
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": sys.platform,
            "arms": [a.as_dict() for a in ARMS], "gate_probability": GATE_PROBABILITY,
            "outlier_stream": OUTLIER_STREAM, "registered": False, "summary": summary}
    (tmp / "summary.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    figure(summary, tmp / "turn_gate.png", args.seeds)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    shutil.rmtree(ckpt)
    print(f"{len(jobs)} cells x {args.seeds} seeds x {len(ARMS)} arms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
