"""Late packets under heavy jitter: drop them, fuse them late, or re-filter.

A side run, not a registered result. Simulated link and sensor only. When the
delay jitters, some packets arrive after a packet with a newer capture time.
This script compares three ways to handle such a late packet, all with
capture-time stamping and the S6 constant-velocity filter:

- drop: throw it away (the S6 recovery rule).
- apply_late: fuse it as if it were the newest measurement (stamped one
  microsecond after the filter's current time, because the tracker refuses a
  repeated timestamp).
- refilter: roll the filter back to just before the first newer packet, fuse
  the late one, and fuse the newer ones again in capture-time order. A short
  history of (stamp, measurement, filter before) is kept for packets inside a
  window: the 0.999 quantile of the lognormal delay, the same rule as the
  estimation engine's retrodiction, so the two can be compared.

All three arms are implemented here; sim/ is used read-only. Every seed also
runs the engine's own recovery and retrodict arms and records whether they
match this file's drop and refilter arms exactly.

    PYTHONPATH=. python tools/sens_late_refilter.py --output DIR [--seeds 30]
    PYTHONPATH=. python tools/sens_late_refilter.py --figure DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from sim.estimation.base import ArmSpec, EstimatorSettings, RECOVERY_V1
from sim.estimation.engine import (BURN_IN_TOLERANCE_S, QUERY_DT_S, TIME_TOLERANCE_S, CellSpec,
                                   deliver_cell, retro_window_for, run_arm)
from sim.estimation.factory import make_estimator
from sim.estimation.sweep import run_jobs, sha256_file, write_json
from sim.s6_recovery import RecoveryConfig, median_interval, seed_list, wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261013
ARMS = ("drop", "apply_late", "refilter")
TARGETS = ("cv", "ct")
DELAYS_S = (0.10, 0.30)
JITTER_RATIOS = (0.0, 1.0, 2.0, 3.0)
LOSSES = (0.0, 0.3)
NOISE_STD_M = 0.05
RATE_HZ = 10.0
APPLY_LATE_EPS_S = 1e-6
ENGINE_RETRO = ArmSpec("retro", model="cv", stamping="capture", late_policy="retrodict")
SEED_FIELDS = ("p95_error_m", "max_error_m", "rmse_m", "fused_count", "dropped_count",
               "late_applied_count", "refiltered_count", "max_history", "failed")


def make_cell(target: str, delay_s: float, jitter_ratio: float, loss: float) -> CellSpec:
    """The cell for one grid point: S6 settings with jitter std = ratio x delay."""
    return CellSpec(target=target, rate_hz=RATE_HZ, delay_s=float(delay_s),
                    jitter_s=float(jitter_ratio) * float(delay_s), loss_probability=float(loss),
                    loss_model="independent", noise_model="gaussian", noise_std_m=NOISE_STD_M)


def build_grid() -> list[dict[str, Any]]:
    """The 32 grid points (target, delay, jitter ratio, loss) in a fixed order."""
    grid = [{"target": t, "delay_s": d, "jitter_ratio": r, "loss_probability": p}
            for t in TARGETS for d in DELAYS_S for r in JITTER_RATIOS for p in LOSSES]
    assert len(grid) == 32
    return grid


def run_late_arm(bundle, cell: CellSpec, arm: str, config: RecoveryConfig = RecoveryConfig(),
                 settings: EstimatorSettings = EstimatorSettings(),
                 window_s: float | None = None) -> dict[str, Any]:
    """Score one late-packet arm on one delivered bundle.

    Queries every 0.02 s; a packet is handled at the first query time at or
    after its arrival; queries before a fusion are answered by the filter as it
    was. The scored window starts at the burn-in. Returns per-seed numbers.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    window = retro_window_for(cell) if window_s is None else float(window_s)
    est = make_estimator("cv", settings, bundle.target.start_position)
    query_times = np.arange(0.0, config.duration_s + QUERY_DT_S / 2, QUERY_DT_S)
    truth = np.asarray(bundle.target.position(query_times), dtype=float)
    scored = query_times >= config.burn_in_s - BURN_IN_TOLERANCE_S
    positions = np.empty((query_times.size, 3))
    ready_at = np.searchsorted(query_times + TIME_TOLERANCE_S,
                               np.array([d.receive_time_s for d in bundle.deliveries], dtype=float),
                               side="left")
    pending = 0
    newest = -np.inf
    counts = {"fused_count": 0, "dropped_count": 0, "late_applied_count": 0, "refiltered_count": 0}
    history: list[list] = []  # refilter only: [stamp, measurement, filter before this fusion]
    max_history = 0

    def flush(upto: int) -> None:
        nonlocal pending
        if upto > pending:
            times = query_times[pending:upto]
            positions[pending:upto] = est.predict_positions(np.maximum(times, est.timestamp_s))
            pending = upto

    def prune() -> None:
        keep_from = 0
        while keep_from < len(history) - 1 and history[keep_from][0] < history[-1][0] - window:
            keep_from += 1
        del history[:keep_from]

    for packet, qi in zip(bundle.deliveries, ready_at):
        qi = int(qi)
        if qi >= query_times.size:
            break
        z = np.asarray(packet.measurement_m, dtype=float)
        stamp = float(packet.stamp_s)
        if stamp > newest:
            if stamp < est.timestamp_s:
                counts["dropped_count"] += 1
                continue
            flush(qi)
            if arm == "refilter":
                history.append([stamp, z, est.copy()])
            est.update(z, stamp)
            newest = stamp
            counts["fused_count"] += 1
            if arm == "refilter":
                prune()
        elif arm == "apply_late":
            flush(qi)
            est.update(z, est.timestamp_s + APPLY_LATE_EPS_S)
            counts["fused_count"] += 1
            counts["late_applied_count"] += 1
        elif arm == "refilter" and stamp >= newest - window:
            stamps = [entry[0] for entry in history]
            i = bisect_right(stamps, stamp)
            if i > 0 and stamps[i - 1] == stamp:
                counts["dropped_count"] += 1
                continue
            snapshot = history[i][2]
            fresh = snapshot.update_count == 0 and stamp == snapshot.timestamp_s
            if not (stamp > snapshot.timestamp_s or fresh):
                counts["dropped_count"] += 1
                continue
            flush(qi)
            filt = snapshot.copy()
            filt.update(z, stamp)
            rebuilt = [[stamp, z, snapshot]]
            for entry in history[i:]:
                before = filt.copy()
                filt.update(entry[1], entry[0])
                rebuilt.append([entry[0], entry[1], before])
            history[i:] = rebuilt
            est = filt
            counts["fused_count"] += 1
            counts["refiltered_count"] += 1
            prune()
        else:
            counts["dropped_count"] += 1
        max_history = max(max_history, len(history))
    flush(query_times.size)

    errors = np.linalg.norm(positions - truth, axis=1)[scored]
    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    return {"p95_error_m": p95, "max_error_m": maximum, "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            **counts, "max_history": max_history,
            "failed": bool(p95 > config.failure_p95_m or maximum > config.failure_max_m)}


def late_census(bundle) -> tuple[int, int]:
    """(late, delivered): delivered packets that arrive after a newer capture."""
    late = 0
    newest = -np.inf
    for delivery in bundle.deliveries:
        capture = float(delivery.capture_time_s)
        if capture < newest:
            late += 1
        newest = max(newest, capture)
    return late, len(bundle.deliveries)


def hand_late_fraction(delay_s: float, jitter_s: float, loss: float, period_s: float = 0.1,
                       duration_s: float = 12.0, grid: int = 4000) -> float:
    """Expected pooled late fraction for a finite run, from the lognormal formula.

    Written from the delay model's definition, not from the simulation code.
    Packet k is captured at k x period and has delay x. It is delivered if it
    is not lost and x <= duration - k x period. It is late if a later packet
    k + m got through and beat it: its delay is under x - m x period. So
    P(late | x) = 1 - product over m of (1 - (1 - loss) F(x - m x period)).
    The result is E[late and delivered] / E[delivered], summed over k.
    """
    if jitter_s == 0:
        return 0.0
    sigma = float(np.sqrt(np.log1p((jitter_s / delay_s) ** 2)))
    mu = float(np.log(delay_s) - sigma ** 2 / 2.0)
    x = np.exp(mu + sigma * ndtri((np.arange(grid) + 0.5) / grid))  # equal-probability delays

    def cdf(values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values)
        positive = values > 0
        out[positive] = ndtr((np.log(values[positive]) - mu) / sigma)
        return out

    n = int(round(duration_s / period_s))
    late_sum = delivered_sum = 0.0
    for k in range(n + 1):
        delivered = x <= duration_s - k * period_s + 1e-9
        delivered_sum += (1.0 - loss) * delivered.mean()
        m = np.arange(1, n - k + 1)[:, None]
        survive = np.prod(1.0 - (1.0 - loss) * cdf(x[None, :] - m * period_s), axis=0) if n > k else 1.0
        late_sum += (1.0 - loss) * np.mean(delivered * (1.0 - survive))
    return late_sum / delivered_sum


def _same(a: dict[str, Any], outcome) -> bool:
    return (a["p95_error_m"] == outcome.p95_error_m and a["max_error_m"] == outcome.max_error_m
            and a["fused_count"] == outcome.fused_count and a["dropped_count"] == outcome.dropped_count)


def job_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Every arm on every seed of one grid point (one delivery per seed for all arms)."""
    point = {k: job[k] for k in ("target", "delay_s", "jitter_ratio", "loss_probability")}
    cell = make_cell(point["target"], point["delay_s"], point["jitter_ratio"], point["loss_probability"])
    config, settings = RecoveryConfig(), EstimatorSettings()
    rows = []
    for index, seed in enumerate(job["seeds"]):
        bundle = deliver_cell(cell, int(seed), config, settings)
        late, delivered = late_census(bundle)
        engine_drop = run_arm(bundle, cell, RECOVERY_V1, config, settings)
        engine_retro = run_arm(bundle, cell, ENGINE_RETRO, config, settings)
        for arm in ARMS:
            result = run_late_arm(bundle, cell, arm, config, settings)
            match = {"drop": _same(result, engine_drop), "refilter": _same(result, engine_retro)}.get(arm)
            rows.append({**point, "cell_key": cell.key(), "arm": arm, "seed_index": index, "seed": int(seed),
                         **result, "late_count": late, "delivered_count": delivered, "engine_match": match})
    return {"point": point, "cell_key": cell.key(), "retro_window_s": retro_window_for(cell),
            "hand_late_fraction": hand_late_fraction(cell.delay_s, cell.jitter_s, cell.loss_probability),
            "rows": rows}


def summarize(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One table row per arm: medians, intervals, fails, and paired gains of refilter."""
    rows = result["rows"]
    by_arm = {arm: sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed_index"]) for arm in ARMS}
    refilter_p95 = np.array([r["p95_error_m"] for r in by_arm["refilter"]])
    late = sum(r["late_count"] for r in by_arm["drop"])
    delivered = sum(r["delivered_count"] for r in by_arm["drop"])
    out = []
    for arm in ARMS:
        arm_rows = by_arm[arm]
        p95 = np.array([r["p95_error_m"] for r in arm_rows])
        median, low, high = median_interval(p95)
        fails = sum(1 for r in arm_rows if r["failed"])
        fail_low, fail_high = wilson_interval(fails, len(arm_rows))
        gain = p95 - refilter_p95  # positive: refilter is better than this arm
        g_median, g_low, g_high = median_interval(gain)
        matches = [r["engine_match"] for r in arm_rows if r["engine_match"] is not None]
        out.append({**result["point"], "arm": arm, "n_seeds": len(arm_rows),
                    "p95_median_m": median, "p95_ci_low_m": low, "p95_ci_high_m": high,
                    "max_error_median_m": float(np.median([r["max_error_m"] for r in arm_rows])),
                    "fail_count": fails, "fail_rate_ci_low": fail_low, "fail_rate_ci_high": fail_high,
                    "label": "fails" if 2 * fails >= len(arm_rows) else "passes",
                    "refilter_gain_median_m": g_median, "refilter_gain_ci_low_m": g_low,
                    "refilter_gain_ci_high_m": g_high,
                    "refilter_wins": int(np.sum(gain > 0)), "refilter_losses": int(np.sum(gain < 0)),
                    "dropped_total": sum(r["dropped_count"] for r in arm_rows),
                    "late_applied_total": sum(r["late_applied_count"] for r in arm_rows),
                    "refiltered_total": sum(r["refiltered_count"] for r in arm_rows),
                    "max_history": max(r["max_history"] for r in arm_rows),
                    "engine_match_all": (all(matches) if matches else None),
                    "late_fraction_sim": late / delivered if delivered else 0.0,
                    "late_fraction_hand": result["hand_late_fraction"],
                    "retro_window_s": result["retro_window_s"]})
    return out


def _fingerprint(n_seeds: int) -> str:
    digest = hashlib.sha256()
    paths = sorted((REPO_ROOT / "sim/estimation").glob("*.py"))
    paths += [REPO_ROOT / p for p in ("sim/link.py", "sim/sensor.py", "sim/tracker.py", "sim/s6_recovery.py",
                                      "tools/sens_late_refilter.py")]
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(sha256_file(path).encode())
    digest.update(str(int(n_seeds)).encode())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def run_sweep(output: Path, n_seeds: int, processes: int, checkpoint_dir: Path | None) -> dict[str, Any]:
    """Run the grid and write table.csv, per_seed.csv and summary.json to a fresh folder."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output}")
    seeds = seed_list(n_seeds, base_seed=BASE_SEED)
    jobs = [dict(point, seeds=seeds) for point in build_grid()]
    start = time.perf_counter()
    results = run_jobs(jobs, "tools.sens_late_refilter:job_cell", processes=processes,
                       checkpoint_dir=checkpoint_dir, fingerprint=_fingerprint(n_seeds))
    runtime = time.perf_counter() - start
    table = [row for result in results for row in summarize(result)]
    per_seed = [row for result in results for row in result["rows"]]
    tmp = output.with_name(output.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    _write_csv(tmp / "table.csv", table)
    _write_csv(tmp / "per_seed.csv", per_seed)
    write_json(tmp / "summary.json", {
        "question": "With jitter 2x and 3x the delay, does re-filtering late packets beat dropping them "
                    "or fusing them late?",
        "simulated": True, "registered": False, "arms": list(ARMS), "n_seeds": n_seeds, "seeds": seeds,
        "grid": build_grid(), "config": asdict(RecoveryConfig()), "apply_late_eps_s": APPLY_LATE_EPS_S,
        "failure_rule": "seed fails if p95 > 0.25 m or max > 0.50 m; cell label uses the proposed 50 % rule",
        "runtime_s": runtime, "processes": processes, "fingerprint": _fingerprint(n_seeds)})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.rmdir()
    tmp.rename(output)
    return {"runtime_s": runtime, "rows": len(per_seed)}


def make_figure(output: Path) -> Path:
    """Refilter's paired gain over drop and over apply_late, against jitter ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools import horizon_mpl

    with (output / "table.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    pal = horizon_mpl.apply_style("light")
    styles = {"drop": horizon_mpl.arm_style("recovery", pal),
              "apply_late": {"color": pal["target"], "linestyle": "--", "marker": "^"}}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for col, target in enumerate(TARGETS):
        for row, delay in enumerate(DELAYS_S):
            ax = axes[row][col]
            for arm in ("drop", "apply_late"):
                for loss, fill in ((0.0, True), (0.3, False)):
                    pts = sorted((float(r["jitter_ratio"]), 100 * float(r["refilter_gain_median_m"]),
                                  100 * float(r["refilter_gain_ci_low_m"]), 100 * float(r["refilter_gain_ci_high_m"]))
                                 for r in table if r["arm"] == arm and r["target"] == target
                                 and float(r["delay_s"]) == delay and float(r["loss_probability"]) == loss)
                    x = np.array([p[0] for p in pts]) + (0.04 if loss else -0.04)
                    y = np.array([p[1] for p in pts])
                    err = np.array([[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]])
                    st = styles[arm]
                    ax.errorbar(x, y, yerr=err, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                                markerfacecolor=st["color"] if fill else pal["surface"], markersize=5,
                                linewidth=1.3, capsize=2,
                                label=f"vs {arm.replace('_', ' ')}, loss {int(loss * 100)} %")
            ax.axhline(0.0, color=pal["lineStrong"], linewidth=0.8)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(-3.0, 150.0)
            ax.set_yticks([-1, 0, 1, 10, 100])
            ax.set_yticklabels(["-1", "0", "1", "10", "100"])
            ax.set_title(f"{'line' if target == 'cv' else 'turn'}, mean delay {int(delay * 1000)} ms")
            if row == 1:
                ax.set_xlabel("jitter std / mean delay")
            ax.set_xticks(JITTER_RATIOS)
    fig.supylabel("refilter p95 gain (cm, paired median)", fontsize=9.5, color=pal["textSecondary"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Re-filtering late packets: gain over dropping them and over fusing them late (simulated)",
                 fontsize=10.5)
    fig.text(0.5, 0.008, "Above 0 means re-filtering had the lower p95 error. Bars: order-statistic 95 % interval "
             "of the paired median. Log scale above 1 cm.", ha="center", fontsize=8, color=pal["textSecondary"])
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output / "late_refilter.png"
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 1) // 4))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--figure", type=Path, help="draw the figure from an existing output folder")
    args = parser.parse_args(argv)
    if args.figure:
        print(make_figure(args.figure))
        return 0
    if args.output is None:
        parser.error("--output is required unless --figure is given")
    info = run_sweep(args.output, args.seeds, args.processes, args.checkpoint_dir)
    print(json.dumps(info), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
