#!/usr/bin/env python3
"""Gated constant-acceleration tracker and shorter gate lockouts on switching targets.

Follow-up to ``tools/sens_gate_switching.py``. That study found the 0.99 NIS
gate adds failures on the start-stop target with clean data, because good
packets right after each 2 m/s^2 acceleration ends get rejected and no tracker
in its set (CV, CT, IMM) predicts acceleration. This tool runs the same cells,
packets and outliers with:

- the four CV and IMM arms of that study, unchanged (the control);
- the engine's constant-acceleration model (``sim/estimation/ca.py``), ungated
  and gated with lockout guards 3 (the default), 2 and 1;
- gated IMM with lockout 1, which separates the lockout effect from the model.

The lockout guard N fuses a packet anyway after N rejections in a row.

``--compare`` checks the control arms against the switching study's
``per_seed.csv`` row by row.

Nothing under ``sim/`` is changed or registered. Outputs go to ``--output``
and must stay out of ``sim/runs/``.

Run:
  PYTHONPATH=. python tools/sens_gate_ca.py --output sens_runs/gate-ca --seeds 50
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

from sim.estimation.base import RECOVERY_V1, ArmSpec
from sim.s6_recovery import seed_list
from tools.sens_gate_switching import TARGETS, cell_runs, plan, switching_cell
from tools.sens_outlier_gate import (
    GATE_PROBABILITY,
    GATED,
    OUTLIER_STREAM,
    _git_revision,
)
from tools.sens_outlier_gate_turn import (
    PROBABILITIES,
    predicted_clean_reject_rate,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def _gated(name: str, model: str, lockout: int) -> ArmSpec:
    return ArmSpec(name, model=model, stamping="capture", late_policy="drop",
                   gate_probability=GATE_PROBABILITY, gate_reset_after=lockout)


CONTROL_ARMS = (
    RECOVERY_V1,
    GATED,
    ArmSpec("imm", model="imm", stamping="capture", late_policy="drop"),
    _gated("imm_gated", "imm", 3),
)
NEW_ARMS = (
    ArmSpec("ca", model="ca", stamping="capture", late_policy="drop"),
    _gated("ca_gated", "ca", 3),
    _gated("ca_gated_l2", "ca", 2),
    _gated("ca_gated_l1", "ca", 1),
    _gated("imm_gated_l1", "imm", 1),
)
ARMS = CONTROL_ARMS + NEW_ARMS
CONTROL_NAMES = tuple(a.name for a in CONTROL_ARMS)
# (ungated twin, gated arm) for the paired gate cost.
PAIRS = (("recovery", "recovery_gated"), ("imm", "imm_gated"), ("imm", "imm_gated_l1"),
         ("ca", "ca_gated"), ("ca", "ca_gated_l2"), ("ca", "ca_gated_l1"))
VARIANT_STYLE = {"": (":", "o", ", no gate"), "gated": ("-", "s", ", gate"),
                 "gated_l1": ("--", "^", ", gate, lockout 1")}
COMPARE_KEYS = ("p95_error_m", "max_error_m", "failed", "fused_count", "gated_count")


def paired_gate_cost(rows: list[dict], pairs=PAIRS) -> list[dict]:
    """Per (ungated, gated) pair of one cell: seeds only the gated arm fails, only the ungated one
    fails, and the median over seeds of (gated p95 - ungated p95)."""
    out = []
    for plain, gated in pairs:
        a = {r["seed"]: r for r in rows if r["arm"] == plain}
        b = {r["seed"]: r for r in rows if r["arm"] == gated}
        if not a and not b:
            continue
        if set(a) != set(b):
            raise ValueError(f"{plain} and {gated} rows cover different seeds")
        seeds = sorted(a)
        out.append({
            "ungated": plain, "gated": gated,
            "ungated_fails": sum(bool(a[s]["failed"]) for s in seeds),
            "gated_fails": sum(bool(b[s]["failed"]) for s in seeds),
            "only_gated_fails": sum(bool(b[s]["failed"]) and not bool(a[s]["failed"]) for s in seeds),
            "only_ungated_fails": sum(bool(a[s]["failed"]) and not bool(b[s]["failed"]) for s in seeds),
            "median_p95_diff_m": statistics.median(float(b[s]["p95_error_m"]) - float(a[s]["p95_error_m"])
                                                   for s in seeds),
        })
    return out


def _key(row: dict) -> tuple:
    return (row["motion"], float(row["delay_s"]), float(row["probability"]), row["arm"], int(row["seed"]))


def _same(mine, theirs) -> bool:
    """Exact match between a value from this run and its CSV text in the switching study."""
    if isinstance(mine, bool):
        return str(mine) == theirs
    if isinstance(mine, int):
        return str(mine) == theirs
    return float(mine) == float(theirs)


def compare_to_switching(rows: list[dict], path: Path) -> dict:
    """Control: every control-arm row must equal the switching study's row exactly."""
    want = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            want[_key(row)] = row
    mine = [r for r in rows if r["arm"] in CONTROL_NAMES]
    missing = [k for k in (_key(r) for r in mine) if k not in want]
    diffs = []
    for r in mine:
        w = want.get(_key(r))
        if w is None:
            continue
        for key in COMPARE_KEYS:
            if not _same(r[key], w[key]):
                diffs.append({"row": list(_key(r)), "key": key, "mine": r[key], "theirs": w[key]})
    return {"compared": len(mine) - len(missing), "missing": len(missing), "differences": len(diffs),
            "first_differences": diffs[:5], "reference_rows": len(want)}


def figure(summary: list[dict], path: Path, n_seeds: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    horizon_mpl.apply_style("light")
    panels = (("start_stop", 0.0), ("start_stop", 0.1), ("start_stop", 0.2), ("figure8", 0.2))
    shown = ("imm", "imm_gated", "imm_gated_l1", "ca", "ca_gated", "ca_gated_l1")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), sharey=True,
                             constrained_layout=True)
    xs = [100 * p for p in PROBABILITIES]
    dodge = np.linspace(-0.12, 0.12, len(shown))  # keeps equal values visible; ticks stay at the true rates
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for col, (target, delay) in enumerate(panels):
        ax = axes[col]
        for k, name in enumerate(shown):
            rows = [next(r for r in summary if r["motion"] == target and r["delay_s"] == delay
                         and r["probability"] == p and r["arm"] == name) for p in PROBABILITIES]
            model, variant = name.split("_", 1) if "_" in name else (name, "")
            ax.plot([x + dodge[k] for x in xs], [r["fail_count"] for r in rows],
                    color=colors[0 if model == "imm" else 1], linestyle=VARIANT_STYLE[variant][0],
                    marker=VARIANT_STYLE[variant][1], label=f"{model.upper()}{VARIANT_STYLE[variant][2]}",
                    linewidth=1.6)
        ax.set_title(f"{'start-stop' if target == 'start_stop' else 'figure 8'}, delay {delay:g} s")
        ax.set_xticks(xs)
        ax.set_xlabel("outlier frames (%)")
        ax.set_ylim(-0.04 * n_seeds, 1.04 * n_seeds)
    axes[0].set_ylabel(f"failing seeds (of {n_seeds})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(shown), fontsize=8)
    fig.suptitle("1 m outliers: IMM and constant-acceleration trackers, gate off, gate on, gate with lockout 1\n"
                 "(synthetic; points nudged sideways so equal values stay visible)", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare", type=Path, help="switching study per_seed.csv to check the control arms against")
    args = ap.parse_args(argv)
    out = args.output.resolve()
    if (ROOT / "sim" / "runs") in [out, *out.parents]:
        raise SystemExit("refusing to write under sim/runs/: this study is not registered")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    seeds = seed_list(args.seeds)
    jobs = plan()
    ckpt = out.with_name(out.name + ".ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    stamp = {"n_seeds": args.seeds, "arms": [a.as_dict() for a in ARMS],
             "jobs": [[t, d, s.kind, s.probability] for t, d, s in jobs], "git_revision": _git_revision()}
    stamp_path = ckpt / "plan.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != json.loads(json.dumps(stamp, default=float)):
        raise SystemExit(f"{ckpt} holds a different plan; delete it to start over")
    stamp_path.write_text(json.dumps(stamp, default=float))
    start = time.time()
    for i, (target, delay, spec) in enumerate(jobs):
        if (ckpt / f"cell{i:03d}.json").exists():
            continue
        cpu = time.process_time()
        rows = cell_runs(target, delay, spec, seeds, arms=ARMS)
        body = {"cpu_s": time.process_time() - cpu, "summary": summarize(target, delay, spec, rows, arms=ARMS),
                "paired": [dict(p, motion=target, delay_s=delay, probability=spec.probability)
                           for p in paired_gate_cost(rows)],
                "per_seed": [{k: v for k, v in r.items() if k != "_outcome"} for r in rows]}
        part = ckpt / f"cell{i:03d}.json.part"
        part.write_text(json.dumps(body, default=float))
        os.replace(part, ckpt / f"cell{i:03d}.json")
        print(f"cell {i + 1}/{len(jobs)} done", flush=True)
    per_seed, summary, paired, worker_cpu_s = [], [], [], 0.0
    for i in range(len(jobs)):
        body = json.loads((ckpt / f"cell{i:03d}.json").read_text())
        worker_cpu_s += body["cpu_s"]
        summary.extend(body["summary"])
        paired.extend(body["paired"])
        per_seed.extend(body["per_seed"])
    # Hand check: predicted clean reject rate from each ungated twin's median mean NIS at 0 % outliers.
    twin_of = {g: p for p, g in PAIRS}
    for row in summary:
        if row["gated"] and row["probability"] == 0.0:
            twin = next(r for r in summary if r["motion"] == row["motion"] and r["delay_s"] == row["delay_s"]
                        and r["probability"] == 0.0 and r["arm"] == twin_of[row["arm"]])
            row["predicted_clean_reject_rate"] = predicted_clean_reject_rate(twin["mean_nis_median"])
    control = compare_to_switching(per_seed, args.compare) if args.compare else None
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with (tmp / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    meta = {"schema": "zephyr-sens-gate-ca-1", "argv": sys.argv[1:] if argv is None else list(argv),
            "git_revision": _git_revision(), "n_seeds": args.seeds, "seeds": seeds, "targets": list(TARGETS),
            "cells": {t: switching_cell(t, 0.0).as_dict() for t in TARGETS},
            "runtime_s_this_session": time.time() - start, "worker_cpu_s": worker_cpu_s,
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": sys.platform,
            "arms": [a.as_dict() for a in ARMS], "outlier_stream": OUTLIER_STREAM, "registered": False,
            "control_vs_switching_study": control, "summary": summary, "paired_gate_cost": paired}
    (tmp / "summary.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    figure(summary, tmp / "gate_ca.png", args.seeds)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    shutil.rmtree(ckpt)
    print(f"{len(jobs)} cells x {args.seeds} seeds x {len(ARMS)} arms -> {out}")
    if control is not None:
        print(json.dumps({"control_vs_switching_study": {k: control[k] for k in
                                                         ("compared", "missing", "differences")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
