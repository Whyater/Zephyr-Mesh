"""EXP-036 (proposal): the turn-aware filter inside cooperative tracking, and shared fading.

Simulated only. One cell family: the 1 m turn at 1 m/s, 10 Hz capture, 5 cm
noise, 30 % loss in 0.5 s Gilbert-Elliott bursts on the abstract link, the
same cell as the S6 atlas ``ct`` result and the EXP-024 cooperation result.
Ten conditions cross the filter (cv or ct), the number of observers (1 or 3
with central fusion), and whether the links lose packets independently or in
the same capture slots ("shared"). Three reference conditions bound what
noise averaging alone can do. See ``CONDITIONS``.

Outputs (all regenerate byte for byte from the same command):
per_seed.csv.gz, cells.csv, boundaries.json, comparisons.json, checks.json,
fail_rate_vs_delay.png and manifest.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.coop.boundary import tolerated
from sim.coop.net import LinkSpec
from sim.coop.observers import CoopCell, run_seed
from sim.estimation.sweep import run_jobs, sha256_json, write_csv, write_csv_gz
from sim.s6_recovery import median_interval, wilson_interval
from tools import horizon_mpl
from tools.d_common import (
    MOVING_AGENT_LIMITS,
    argv_or_default,
    prepare_output,
    seeds_for,
    sha256_file,
    write_json,
    write_manifest,
)

DELAYS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
PILOT_DELAYS = (0.0, 0.20, 0.30)
N_SEEDS = 200
PILOT_SEEDS = 10
NOISE_M = 0.05
AVERAGED_NOISE_M = NOISE_M / float(np.sqrt(3.0))
LOSS = 0.3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261021
DECISION_DELAY = 0.30

# id: (filter, observers, loss across links, simulated and assumed noise, capture phase, label)
CONDITIONS = {
    "A": ("cv", 1, "independent", NOISE_M, "random", "cv, 1 observer"),
    "B": ("ct", 1, "independent", NOISE_M, "random", "ct, 1 observer"),
    "C": ("cv", 3, "independent", NOISE_M, "random", "cv, 3 central, independent bursts"),
    "D": ("ct", 3, "independent", NOISE_M, "random", "ct, 3 central, independent bursts"),
    "E": ("cv", 3, "shared", NOISE_M, "random", "cv, 3 central, shared bursts"),
    "F": ("ct", 3, "shared", NOISE_M, "random", "ct, 3 central, shared bursts"),
    "G": ("ct", 1, "independent", AVERAGED_NOISE_M, "random", "ct, 1 observer, noise / sqrt(3)"),
    "H": ("cv", 1, "independent", AVERAGED_NOISE_M, "random", "cv, 1 observer, noise / sqrt(3)"),
    "I": ("ct", 3, "shared", NOISE_M, "sync", "ct, 3 central, shared bursts, synchronous"),
    "J": ("cv", 3, "shared", NOISE_M, "sync", "cv, 3 central, shared bursts, synchronous"),
}
# (left, right): every comparison is the fail rate of left minus the fail rate of right.
COMPARISONS = (("B", "D"), ("D", "F"), ("F", "G"), ("B", "F"), ("A", "C"), ("C", "E"),
               ("A", "E"), ("E", "H"), ("I", "G"), ("J", "H"))


def _arm(condition: str) -> str:
    return "single" if CONDITIONS[condition][1] == 1 else "central"


def _cell(job: dict) -> CoopCell:
    estimator, n, correlation, noise, phase, _ = CONDITIONS[str(job["condition"])]
    link = LinkSpec(mode="abstract", delay_s=float(job["delay"]), loss_probability=LOSS,
                    loss_model="burst", loss_correlation=correlation)
    return CoopCell(target="ct", n_observers=n, rate_hz=10.0, noise_std_m=noise, r_std_m=noise,
                    link=link, phase=phase, ci="fast", estimator=estimator)


def _gap_counts(table, n_observers: int) -> dict:
    """Capture slots where every observer's packet was lost on the link, from the delivery table."""
    routes = [table[(f"o{i + 1}", "gs")] for i in range(n_observers)]
    steps = min(len(route) for route in routes)
    lost = [[route[k].loss_reason == "link" for k in range(steps)] for route in routes]
    return {"steps": int(steps),
            "all_lost": int(sum(all(column) for column in zip(*lost))),
            "o1_lost": int(sum(lost[0]))}


def _job(job: dict) -> dict:
    cpu_start = time.process_time()
    cell = _cell(job)
    arm = _arm(str(job["condition"]))
    outcome = run_seed(cell, int(job["seed"]), arms=(arm,))[arm]
    table = outcome.extras["delivery_table"]
    row = {"condition": str(job["condition"]), "delay": float(job["delay"]),
           "seed_index": int(job["seed_index"]), "seed": int(job["seed"]),
           "failed": bool(outcome.failed), "p95": float(outcome.p95_error_m),
           "max": float(outcome.max_error_m), "rmse": float(outcome.rmse_m),
           "mean_nees": outcome.mean_nees, "delivery_ratio": table.summary()["delivery_ratio"]}
    row.update(_gap_counts(table, cell.n_observers))
    return {"row": row, "worker_cpu_s": time.process_time() - cpu_start}


def _bootstrap_indices(n_seeds: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(BOOTSTRAP_SEED))
    return rng.integers(0, n_seeds, size=(BOOTSTRAP_RESAMPLES, n_seeds))


def _difference(left: np.ndarray, right: np.ndarray, indices: np.ndarray) -> dict:
    """Fail-rate difference left minus right with a seed-level paired percentile bootstrap CI."""
    diff = left.astype(float) - right.astype(float)
    resampled = diff[indices].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {"difference": float(diff.mean()), "ci_low": float(low), "ci_high": float(high)}


def _verdicts(boundaries: dict, comparisons: dict) -> dict:
    """The decision rules written in the pre-registration, applied as written."""
    def at(left, right):
        return comparisons[f"{left}-{right}"][f"{DECISION_DELAY:g}"]

    def rank(condition):
        b = boundaries[condition]
        if b["fails_at_zero"]:
            return -1.0
        if b["censored"]:
            return float("inf")
        return float(b["value"])

    q1 = at("B", "D")
    longer = rank("D") > rank("B")
    if q1["ci_low"] > 0 and longer:
        q1_verdict = "YES"
    elif q1["ci_low"] <= 0 <= q1["ci_high"] and not longer:
        q1_verdict = "NO"
    else:
        q1_verdict = "MIXED"
    q2 = at("D", "F")
    if q2["ci_low"] > 0:
        q2_verdict = "COLLAPSES"
        noise_only = at("F", "G")["ci_low"] <= 0 <= at("F", "G")["ci_high"]
    else:
        # The CI includes 0 or lies below it.
        q2_verdict = "SURVIVES" if rank("F") >= rank("D") else "MIXED"
        noise_only = None
    e_zero = boundaries["E"]["fails_at_zero"]
    return {
        "decision_delay_s": DECISION_DELAY,
        "q1_cooperation_beyond_filter": {"verdict": q1_verdict, "B_minus_D": q1,
                                         "D_tolerates_longer_than_B": bool(longer)},
        "q2_gain_survives_shared_fading": {"verdict": q2_verdict, "D_minus_F": q2,
                                           "only_noise_averaging_left": noise_only},
        "cv_shared_fails_at_zero": bool(e_zero),
    }


def _checks(rows: list[dict], comparisons: dict, delays: tuple[float, ...]) -> dict:
    shared = [r for r in rows if CONDITIONS[r["condition"]][2] == "shared"]
    independent = [r for r in rows if CONDITIONS[r["condition"]][2] == "independent"
                   and CONDITIONS[r["condition"]][1] > 1]
    mismatched = sum(r["all_lost"] != r["o1_lost"] for r in shared)
    ind_steps = sum(r["steps"] for r in independent)
    shared_steps = sum(r["steps"] for r in shared)
    i_vs_g = [comparisons["I-G"][f"{d:g}"] for d in delays]
    covered = sum(c["ci_low"] <= 0 <= c["ci_high"] for c in i_vs_g)
    needed = len(delays) - 1
    return {
        "shared_rows": len(shared),
        "shared_rows_where_all_lost_differs_from_o1_lost": int(mismatched),
        "shared_all_lost_fraction": (sum(r["all_lost"] for r in shared) / shared_steps
                                     if shared_steps else None),
        "independent_all_lost_fraction": (sum(r["all_lost"] for r in independent) / ind_steps
                                          if ind_steps else None),
        "independent_all_lost_hand_value": LOSS ** 3,
        "sync_shared_vs_averaged_delays_with_ci_covering_zero": int(covered),
        "sync_shared_vs_averaged_delays_needed": int(needed),
        "passed": bool(mismatched == 0 and covered >= needed),
    }


def _write_cells(output: Path, grouped: dict, n_seeds: int) -> None:
    columns = ("condition", "label", "filter", "observers", "loss_across_links", "noise_m", "phase",
               "delay", "n_seeds", "fail_count", "fail_rate", "fail_rate_wilson_lo", "fail_rate_wilson_hi",
               "p95_median", "p95_median_lo", "p95_median_hi", "max_median", "rmse_median",
               "delivery_ratio_mean", "all_lost_fraction")
    out = []
    for (condition, delay), selected in sorted(grouped.items()):
        estimator, n, correlation, noise, phase, label = CONDITIONS[condition]
        failed = sum(bool(r["failed"]) for r in selected)
        median, low, high = median_interval([r["p95"] for r in selected])
        w_low, w_high = wilson_interval(failed, n_seeds)
        steps = sum(r["steps"] for r in selected)
        out.append(dict(zip(columns, (
            condition, label, estimator, n, correlation, noise, phase, delay, n_seeds, failed,
            failed / n_seeds, w_low, w_high, median, low, high,
            float(np.median([r["max"] for r in selected])),
            float(np.median([r["rmse"] for r in selected])),
            float(np.mean([r["delivery_ratio"] for r in selected])),
            sum(r["all_lost"] for r in selected) / steps))))
    write_csv(output / "cells.csv", out, columns)


def _figure(path: Path, grouped: dict, delays: tuple[float, ...], n_seeds: int) -> None:
    pal = horizon_mpl.apply_style("light")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    styles = {"1": "-", "3i": "-", "3s": "--", "avg": ":", "sync": "-."}
    colors = {"A": pal["lineStrong"], "B": pal["lineStrong"], "C": pal["accent"], "D": pal["accent"],
              "E": pal["keepOut"], "F": pal["keepOut"], "G": pal["ok"], "H": pal["ok"],
              "I": pal["target"], "J": pal["target"]}
    kinds = {"A": "1", "B": "1", "C": "3i", "D": "3i", "E": "3s", "F": "3s", "G": "avg", "H": "avg",
             "I": "sync", "J": "sync"}
    x = [1000.0 * d for d in delays]
    for axis, model in zip(axes, ("ct", "cv")):
        for condition, spec in CONDITIONS.items():
            if spec[0] != model:
                continue
            rates = [sum(r["failed"] for r in grouped[(condition, d)]) / n_seeds for d in delays]
            axis.plot(x, rates, linestyle=styles[kinds[condition]], marker="o", markersize=3.5,
                      color=colors[condition], label=f"{condition}: {spec[5]}")
        axis.axhline(0.5, color=pal["textSecondary"], linewidth=0.8, linestyle=(0, (2, 2)))
        axis.set_title(f"{model} filter, 1 m turn, 30 % loss in 0.5 s bursts (simulated)")
        axis.set_xlabel("added delay (ms)")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel(f"fraction of {n_seeds} seeds that fail")
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _config(delays, seeds) -> dict:
    return {
        "experiment": "EXP-036 (proposal)",
        "target": {"name": "ct_r1", "target": "ct", "params": {}},
        "link": "abstract", "rate_hz": 10.0, "loss_model": "burst", "loss_probability": LOSS,
        "burst_mean_packets": 5, "shared_fading": "one burst chain on the 0.1 s capture slots, "
        "the lone observer's chain; packet k of every route is lost when slot k fades",
        "conditions": {k: {"filter": v[0], "observers": v[1], "loss_across_links": v[2],
                           "noise_and_r_m": v[3], "phase": v[4], "label": v[5]}
                       for k, v in CONDITIONS.items()},
        "fusion": {"1": "single (local filter at the ground station)", "3": "central sequential Kalman"},
        "stamping": "capture time, predict to now; stale packets dropped",
        "delays_s": list(delays), "duration_s": 12.0, "burn_in_s": 2.0, "query_dt_s": 0.02,
        "failure_rule": {"p95_error_m": 0.25, "max_error_m": 0.50, "cell_fail_rate_at_least": 0.5},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95,
                      "unit": "seed index, pairs kept together"},
        "decision_delay_s": DECISION_DELAY,
        "comparisons": [f"{a}-{b}" for a, b in COMPARISONS],
        "seeds": list(seeds),
    }


# Imported lazily by sim/estimation/factory.py, so pinned here: without the pin the
# hashed list would depend on whether the parent process ran a job itself.
PINNED_SOURCES = ("sim/estimation/cv.py", "sim/estimation/ct.py", "sim/estimation/kinematics.py",
                  "sim/coop/boundary.py", "sim/coop/net.py", "sim/coop/observers.py", "tools/horizon_mpl.py")


def _source_paths() -> list[str]:
    """Repo files behind the outputs: the pinned list plus every repo module imported so far."""
    root = Path(__file__).resolve().parents[1]
    paths = set(PINNED_SOURCES)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(root)))
        except ValueError:
            continue
    return sorted(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true", help="10 seeds and 3 delays, for a smoke run")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 4:
        raise ValueError("--processes must be between 1 and 4")
    output = prepare_output(Path(args.output))
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and (checkpoint.resolve() == output.resolve()
                                   or output.resolve() in checkpoint.resolve().parents):
        raise ValueError("--checkpoint must be outside --output")
    seeds = seeds_for("coop-ct", PILOT_SEEDS if args.pilot else N_SEEDS)
    delays = PILOT_DELAYS if args.pilot else DELAYS
    config = _config(delays, seeds)
    jobs = [{"condition": c, "delay": d, "seed_index": i, "seed": s}
            for c in CONDITIONS for d in delays for i, s in enumerate(seeds)]
    root = Path(__file__).resolve().parents[1]
    sources = _source_paths()
    fingerprint = "coop-ct-" + sha256_json({"config": config, "sources": [
        (p, sha256_file(root / p)) for p in sources]})
    start = time.perf_counter()
    raw = run_jobs(jobs, "tools.d_run_coop_ct:_job", processes=args.processes,
                   checkpoint_dir=checkpoint, fingerprint=fingerprint, progress=False)
    runtime_s = time.perf_counter() - start
    rows = [result["row"] for result in raw]
    expected = {(j["condition"], j["delay"], j["seed_index"]) for j in jobs}
    if {(r["condition"], r["delay"], r["seed_index"]) for r in rows} != expected or len(rows) != len(jobs):
        raise RuntimeError("runner rows do not match the grid")
    columns = ("condition", "delay", "seed_index", "seed", "failed", "p95", "max", "rmse", "mean_nees",
               "delivery_ratio", "steps", "all_lost", "o1_lost")
    write_csv_gz(output / "per_seed.csv.gz", rows, columns)
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["delay"]), []).append(row)
    for selected in grouped.values():
        selected.sort(key=lambda r: r["seed_index"])
    n_seeds = len(seeds)
    _write_cells(output, grouped, n_seeds)
    boundaries = {}
    for condition in CONDITIONS:
        counts = [sum(r["failed"] for r in grouped[(condition, d)]) for d in delays]
        boundaries[condition] = dict(tolerated(delays, counts, n_seeds), fail_counts=counts)
    write_json(output / "boundaries.json", boundaries)
    indices = _bootstrap_indices(n_seeds)
    comparisons = {}
    for left, right in COMPARISONS:
        comparisons[f"{left}-{right}"] = {
            f"{d:g}": _difference(np.array([r["failed"] for r in grouped[(left, d)]]),
                                  np.array([r["failed"] for r in grouped[(right, d)]]), indices)
            for d in delays}
    verdicts = _verdicts(boundaries, comparisons) if DECISION_DELAY in delays else None
    write_json(output / "comparisons.json", {"comparisons": comparisons, "verdicts": verdicts})
    write_json(output / "checks.json", _checks(rows, comparisons, delays))
    _figure(output / "fail_rate_vs_delay.png", grouped, delays, n_seeds)
    worker_cpu_s = float(sum(result["worker_cpu_s"] for result in raw))
    print(f"runtime_s={runtime_s:.1f} worker_cpu_s={worker_cpu_s:.1f}")
    command = "PYTHONPATH=. {python} tools/d_run_coop_ct.py --output {out}"
    if args.pilot:
        command += " --pilot"
    outputs = ["per_seed.csv.gz", "cells.csv", "boundaries.json", "comparisons.json", "checks.json",
               "fail_rate_vs_delay.png"]
    write_manifest(output, schema="zephyr-d-coop-ct-manifest-1", argv=argv_or_default(argv),
                   command=command, config=config, seeds=seeds, outputs=outputs,
                   limits=list(MOVING_AGENT_LIMITS) + [
                       "Links are the abstract simulated link; bursts are a synthetic Gilbert-Elliott chain.",
                       ("Shared fading is the extreme case: every link fades in the same capture slots. "
                        "Real links would sit somewhere between shared and independent."),
                       "One target path (1 m turn), one loss level, 3 observers only.",
                       "The half-the-seeds cell rule is still a proposal.",
                       "EXP-036 is a proposal, not an adopted experiment."],
                   runtime_s=runtime_s, n_processes=args.processes, hashed_sources=sources,
                   extra={"volatile_fields": ["runtime_s", "git_revision", "git_tree_state", "platform",
                                               "n_processes", "argv"],
                          "n_runs": len(jobs)})


if __name__ == "__main__":
    main()
