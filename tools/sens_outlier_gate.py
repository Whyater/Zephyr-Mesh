#!/usr/bin/env python3
"""Outlier sensitivity: rare large vision errors against the S6 tracker, with and without a NIS gate.

Real marker detections now and then jump or flip, which Gaussian noise never
does. This wrapper takes the packets the S6 engine delivers for one cell
(``sim.estimation.engine.deliver_cell``), adds outliers to some packet values,
and scores two arms with ``run_arm``:

- ``recovery``: recovery-v1 (capture stamps, no gate);
- ``recovery_gated``: the same with the engine's chi-square gate at 0.99
  (threshold 11.34 for 3 axes) and its default lockout guard (after 3
  rejections in a row the next packet is fused anyway).

Nothing under ``sim/`` is changed or registered; this is a side study. Its
outputs go to the folder given with ``--output`` and must stay out of
``sim/runs/``.

Outlier kinds (each frame is hit independently with chance p, or, for the
"episode" variants, in runs with a mean length of ``episode_mean_frames``):

- ``jump``: add ``magnitude_m`` in a uniformly random 3D direction.
- ``zflip``: a stand-in for an ArUco pose flip. Every target here flies at
  z = 0, so a literal z -> -z flip would only flip the sign of the noise. The
  model instead assumes the tracked center sits magnitude_m / 2 from the
  marker along the marker normal (pointing along z); a flip mirrors that
  normal, so the reported center moves magnitude_m along +z. The offset is a
  modelling choice, not a measured number.

Outlier draws use their own stream, SeedSequence([seed, 105]), so the link
and sensor draws are exactly the S6 ones. A coin and a direction are drawn
for every capture whatever p is, so raising p only adds outliers.

Run (pilot first):
  PYTHONPATH=. python tools/sens_outlier_gate.py --output sens_runs/outlier-gate --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sim.estimation.base import RECOVERY_V1, ArmSpec
from sim.estimation.engine import Bundle, deliver_cell, run_arm, s6_cell
from sim.s6_recovery import seed_list, summarize_arm

ROOT = Path(__file__).resolve().parents[1]
OUTLIER_STREAM = 105
KINDS = ("jump", "zflip")
MOTIONS = ("constant_velocity", "constant_turn")
DELAYS_S = (0.0, 0.1, 0.2)
PROBABILITIES = (0.0, 0.01, 0.03, 0.05)
NOISE_STD_M = 0.05
MAGNITUDE_M = 1.0
EPISODE_MEAN_FRAMES = 5.0
EPISODE_PROBABILITIES = (0.03, 0.05)
EPISODE_DELAY_S = 0.1
GATE_PROBABILITY = 0.99
GATED = ArmSpec("recovery_gated", model="cv", stamping="capture", late_policy="drop",
                gate_probability=GATE_PROBABILITY)
ARMS = (RECOVERY_V1, GATED)


@dataclasses.dataclass(frozen=True)
class OutlierSpec:
    """What gets added to the packets. probability is the long-run share of frames hit."""

    kind: str = "jump"
    probability: float = 0.0
    magnitude_m: float = MAGNITUDE_M
    episode_mean_frames: float | None = None  # None: each frame independent

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not 0.0 <= self.probability < 1.0:
            raise ValueError("probability must be in [0, 1)")
        if not np.isfinite(self.magnitude_m) or self.magnitude_m < 0:
            raise ValueError("magnitude_m must be finite and nonnegative")
        if self.episode_mean_frames is not None and not self.episode_mean_frames >= 1.0:
            raise ValueError("episode_mean_frames must be at least 1")

    def label(self) -> str:
        if self.probability == 0.0:
            return "none"
        tail = "" if self.episode_mean_frames is None else f"-episodes{self.episode_mean_frames:g}"
        return f"{self.kind}{tail}"


def outlier_frames(n: int, spec: OutlierSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Which of n captures are outliers (bool array) and the offset added to each (n, 3).

    The draws are the same for every probability: u and the direction are drawn
    for all n captures first, then compared with the probability.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), OUTLIER_STREAM]))
    u = rng.random(n)
    direction = rng.normal(size=(n, 3))
    p = float(spec.probability)
    if spec.episode_mean_frames is None:
        hit = u < p
    else:
        # Two-state chain. Leave an episode with chance e each frame (mean length
        # 1 / e); enter with chance s chosen so the long-run share s / (s + e) is p.
        e = 1.0 / float(spec.episode_mean_frames)
        s = p * e / (1.0 - p)
        hit = np.zeros(n, dtype=bool)
        inside = False
        for k in range(n):
            inside = (u[k] >= e) if inside else (u[k] < s)
            hit[k] = inside
    offsets = np.zeros((n, 3))
    if spec.kind == "jump":
        unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        offsets[hit] = spec.magnitude_m * unit[hit]
    else:
        offsets[hit, 2] = spec.magnitude_m
    return hit, offsets


def inject(bundle: Bundle, spec: OutlierSpec, seed: int) -> Bundle:
    """A copy of the bundle with outliers added to the delivered values.

    The outlier set is by capture index (seq), so a frame's fate does not depend
    on whether the link drops it. The returned bundle lists every hit seq in
    ``outlier_seqs`` (an empty set at p = 0), which turns on the engine's
    per-class gate counts.
    """
    hit, offsets = outlier_frames(bundle.capture_count, spec, seed)
    deliveries = []
    for d in bundle.deliveries:
        if hit[d.seq]:
            value = tuple(float(v) for v in np.asarray(d.measurement_m, dtype=float) + offsets[d.seq])
            d = dataclasses.replace(d, measurement_m=value)
        deliveries.append(d)
    return dataclasses.replace(bundle, deliveries=deliveries,
                               outlier_seqs=frozenset(int(i) for i in np.flatnonzero(hit)))


def cell_runs(motion: str, delay_s: float, spec: OutlierSpec, seeds) -> list[dict]:
    """Both arms on every seed of one cell. One row per (seed, arm)."""
    cell = s6_cell(motion, delay_s, 0.0, NOISE_STD_M)
    rows = []
    for index, seed in enumerate(seeds):
        bundle = inject(deliver_cell(cell, int(seed)), spec, int(seed))
        for arm in ARMS:
            o = run_arm(bundle, cell, arm)
            x = o.extras
            rows.append({
                "motion": motion, "delay_s": delay_s, "outlier": spec.label(), "kind": spec.kind,
                "probability": spec.probability, "episode_mean_frames": spec.episode_mean_frames,
                "arm": arm.name, "seed_index": index, "seed": int(seed),
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


def plan(pilot_seeds: int | None = None) -> list[tuple]:
    """Every (motion, delay, spec) in the pre-registered sweep, primary cells first."""
    out = []
    for motion in MOTIONS:
        for delay in DELAYS_S:
            out.append((motion, delay, OutlierSpec("jump", 0.0)))
            for kind in KINDS:
                for p in PROBABILITIES[1:]:
                    out.append((motion, delay, OutlierSpec(kind, p)))
    for motion in MOTIONS:
        for p in EPISODE_PROBABILITIES:
            out.append((motion, EPISODE_DELAY_S, OutlierSpec("zflip", p, episode_mean_frames=EPISODE_MEAN_FRAMES)))
    return out


def _job(args):
    motion, delay, spec, seeds = args
    cpu = time.process_time()
    rows = cell_runs(motion, delay, spec, seeds)
    return motion, delay, spec, rows, time.process_time() - cpu


def summarize(motion, delay, spec, rows) -> list[dict]:
    out = []
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm.name]
        s = summarize_arm([r["_outcome"] for r in mine])
        total = lambda key: sum(int(r[key] or 0) for r in mine)  # noqa: E731
        checked_clean, checked_out = total("checked_clean"), total("checked_outliers")
        out.append({
            "motion": motion, "delay_s": delay, "outlier": spec.label(), "kind": spec.kind,
            "probability": spec.probability, "episode_mean_frames": spec.episode_mean_frames, "arm": arm.name,
            "n_seeds": s["n_seeds"], "p95_median_m": s["p95_median_m"], "p95_ci_low_m": s["p95_ci_low_m"],
            "p95_ci_high_m": s["p95_ci_high_m"], "max_error_median_m": s["max_error_median_m"],
            "fail_count": s["fail_count"], "fail_rate_ci_low": s["fail_rate_ci_low"],
            "fail_rate_ci_high": s["fail_rate_ci_high"], "mean_nis_median": s["mean_nis_median"],
            "outliers_total": total("outlier_count"),
            "clean_reject_rate": (total("gated_clean") / checked_clean) if arm.gate_probability and checked_clean else None,
            "outlier_reject_rate": (total("gated_outliers") / checked_out) if arm.gate_probability and checked_out else None,
            "checked_clean": checked_clean if arm.gate_probability else None,
            "checked_outliers": checked_out if arm.gate_probability else None,
            "forced_accepts": total("forced_accepts") if arm.gate_probability else None,
        })
    return out


def _git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "sim", "tools"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def figure(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, constrained_layout=True)
    delay = 0.1
    for col, motion in enumerate(MOTIONS):
        for arm in ARMS:
            style = horizon_mpl.arm_style(arm.name, pal)
            for kind, fill in (("jump", True), ("zflip", False)):
                pts = []
                for p in PROBABILITIES:
                    label = "none" if p == 0 else kind
                    row = next(r for r in summary if r["motion"] == motion and r["delay_s"] == delay
                               and r["outlier"] == label and r["probability"] == p and r["arm"] == arm.name)
                    pts.append((100 * p, row["p95_median_m"], row["p95_ci_low_m"], row["p95_ci_high_m"],
                                row["fail_count"]))
                xs = np.array([q[0] for q in pts])
                name = ("gate on" if arm.gate_probability else "no gate") + f", {kind}"
                kw = dict(color=style["color"], linestyle=style["linestyle"] if fill else ":",
                          marker=style["marker"], markerfacecolor=style["color"] if fill else pal["surface"],
                          label=name, linewidth=1.6)
                med = np.array([q[1] for q in pts])
                axes[0, col].errorbar(xs, med, yerr=[med - np.array([q[2] for q in pts]),
                                                     np.array([q[3] for q in pts]) - med], capsize=2, **kw)
                axes[1, col].plot(xs, [q[4] for q in pts], **kw)
        title = "straight line, 1 m/s" if motion == "constant_velocity" else "circle, 1 m radius"
        axes[0, col].set_title(f"{title}, delay {delay:g} s")
        axes[0, col].axhline(0.25, color=pal["lineStrong"], linewidth=0.8, linestyle="--")
        axes[0, col].set_ylabel("median p95 error (m)")
        axes[1, col].set_ylabel("failing seeds (of 50)")
        axes[1, col].set_xlabel("outlier frames (%)")
        axes[1, col].set_ylim(-2, 52)
        axes[1, col].set_xticks([100 * p for p in PROBABILITIES])
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("1 m outliers vs the 0.99 NIS gate (synthetic; zflip uses an assumed 0.5 m marker offset)",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
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
    jobs = [(m, d, s, seeds) for m, d, s in plan()]
    # One checkpoint file per finished cell, so a killed worker or a stop costs at
    # most the cells in flight. Rerunning the same command resumes.
    ckpt = out.with_name(out.name + ".ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    stamp = {"n_seeds": args.seeds, "n_cells": len(jobs)}
    stamp_path = ckpt / "plan.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise SystemExit(f"{ckpt} holds a different plan; delete it to start over")
    stamp_path.write_text(json.dumps(stamp))
    todo = [i for i in range(len(jobs))
            if i % shard_k == shard_i and not (ckpt / f"cell{i:03d}.json").exists()]
    start = time.time()

    def finished():
        if args.workers <= 1:  # in-process: no pool (a pool started from a detached shell can fail on Windows)
            for i in todo:
                yield i, _job(jobs[i])
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_job, jobs[i]): i for i in todo}
            for future in as_completed(futures):
                yield futures[future], future.result()

    for i, result in finished():
        motion, delay, spec, rows, cpu_s = result
        body = {"cpu_s": cpu_s, "summary": summarize(motion, delay, spec, rows),
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
    runtime = time.time() - start
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with (tmp / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    meta = {"schema": "zephyr-sens-outlier-gate-1", "argv": sys.argv[1:] if argv is None else list(argv),
            "git_revision": _git_revision(), "n_seeds": args.seeds, "seeds": seeds,
            "runtime_s_this_session": runtime, "worker_cpu_s": worker_cpu_s,
            "workers": args.workers, "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": sys.platform, "gate_probability": GATE_PROBABILITY,
            "gate_reset_after": GATED.gate_reset_after, "outlier_stream": OUTLIER_STREAM,
            "registered": False, "summary": summary}
    (tmp / "summary.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    figure(summary, tmp / "outlier_gate.png")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    shutil.rmtree(ckpt)
    print(f"{len(jobs)} cells x {args.seeds} seeds x {len(ARMS)} arms in {runtime:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
