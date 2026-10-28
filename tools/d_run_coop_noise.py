"""EXP-039 (proposal): EXP-024's cooperation cells rerun with camera noise as a new factor.

EXP-024 (tools/d_run_tolerance.py, sim/runs/d3-tolerance) fixes the simulated camera noise at
5 cm and tells the filter 5 cm. This runner keeps everything else the same and varies the
true noise (0, 2, 5, 10 cm) and what the filter is told (5 cm, as registered, or the truth),
for 1 observer and 3 observers with central fusion, at the loss levels where EXP-024's
verdict changes. It is opt-in: it changes no registered code or output.

The first 20 seeds are the registered EXP-024 seeds, so the 5 cm rows are checked against
sim/runs/d3-tolerance row by row (the reproduction gate) before any verdict is written.
All results are simulated. Nothing is measured.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim.estimation.cv  # noqa: F401
from sim.coop.boundary import tolerated
from sim.coop.observers import run_seed
from sim.estimation.sweep import _csv_value, run_jobs, write_csv, write_csv_gz
from sim.s6_recovery import wilson_interval
from tools import horizon_mpl
from tools.d_common import (
    BASE_SEED,
    EXPERIMENT_OFFSETS,
    _seeds_from,
    argv_or_default,
    prepare_output,
    seeds_for,
    write_json,
    write_manifest,
)
from tools.d_run_tolerance import DELAYS, _cell

EXPERIMENT_OFFSET = 39  # EXP-039; not in EXPERIMENT_OFFSETS on purpose (tools/d_common.py is hashed)
N_REGISTERED = 20
TARGETS = ("ct_r1", "jink")
LOSSES = (("independent", 0.3), ("independent", 0.5), ("burst", 0.3))
NS = (1, 3)
ARM_FOR_N = {1: "single", 3: "central"}
# (true noise in m, what the filter is told). "told5" is r_std_m=None, exactly as registered.
NOISE_PAIRS = ((0.0, "told5"), (0.02, "told5"), (0.02, "truth"), (0.05, "told5"),
               (0.10, "told5"), (0.10, "truth"))
CHECK_TARGETS = ("cv_1", "ct_r1")
CHECK_ARMS = ((1, "random"), (3, "random"), (3, "sync"))
VERDICT_PAIRS = ("2|told5", "2|truth", "10|told5", "10|truth")

PILOT_DELAYS = (0.0, 0.2)
PILOT_TARGETS = ("ct_r1",)
PILOT_LOSSES = (("burst", 0.3),)
PILOT_PAIRS = ((0.05, "told5"), (0.10, "told5"), (0.10, "truth"))
PILOT_CHECK_PAIRS = ((0.0, "told5"), (0.05, "told5"))

GATE_FIELDS = ("failed", "p95", "max", "rmse", "mean_nees")
COLUMNS = ("slice", "target", "loss_model", "loss", "n", "arm", "phase", "delay",
           "noise_std_m", "told", "r_std_m", "seed", "failed", "p95", "max", "rmse", "mean_nees")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 39
REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_RUN = REPO_ROOT / "sim" / "runs" / "d3-tolerance"
HAND_SCRIPT = REPO_ROOT / "tools" / "coop_noise_hand.py"


def csv_text(value) -> str:
    """The exact text a value gets in the registered per_seed.csv.gz."""
    return _csv_value(value)


def experiment_seeds(n: int) -> list[int]:
    """The 20 registered EXP-024 seeds first, then new EXP-039 seeds."""
    registered = seeds_for("tolerance", N_REGISTERED)
    if n <= N_REGISTERED:
        return registered[:n]
    new = _seeds_from(BASE_SEED + EXPERIMENT_OFFSET, n - N_REGISTERED)
    taken = set(registered)
    for name in EXPERIMENT_OFFSETS:
        taken.update(seeds_for(name, 2000))
    if set(new) & taken or len(set(new)) != len(new):
        raise RuntimeError("new EXP-039 seeds overlap a registered seed list")
    return registered + new


def pilot_seeds() -> list[int]:
    seeds = experiment_seeds(N_REGISTERED + 2)
    return seeds[:2] + seeds[N_REGISTERED:]


def make_job(slice_name, target, loss_model, loss, n, phase, delay, noise, told, seed) -> dict:
    if told not in ("told5", "truth"):
        raise ValueError("told must be told5 or truth")
    if told == "truth" and float(noise) <= 0.0:
        raise ValueError("a filter told zero noise is a different experiment; use told5 at 0 cm")
    return {"slice": str(slice_name), "target": str(target), "loss_model": str(loss_model),
            "loss": float(loss), "n": int(n), "phase": str(phase), "delay": float(delay),
            "noise_std_m": float(noise), "told": str(told), "seed": int(seed)}


def _job(job: dict) -> dict:
    """Run one simulated cell and seed for one arm."""
    cpu_start = time.process_time()
    n = int(job["n"])
    arm = ARM_FOR_N[n]
    base = _cell({"link": "abstract", "target": job["target"], "loss_model": job["loss_model"],
                  "loss": job["loss"], "n": n, "delay": job["delay"], "seed": job["seed"]})
    r_std = None if job["told"] == "told5" else float(job["noise_std_m"])
    cell = replace(base, noise_std_m=float(job["noise_std_m"]), r_std_m=r_std, phase=job["phase"])
    outcome = run_seed(cell, int(job["seed"]), arms=(arm,))[arm]
    told_value = 0.05 if r_std is None else r_std
    row = {key: job[key] for key in ("slice", "target", "loss_model", "loss", "n", "phase",
                                     "delay", "noise_std_m", "told", "seed")}
    row.update({"arm": arm, "r_std_m": told_value, "failed": bool(outcome.failed),
                "p95": float(outcome.p95_error_m), "max": float(outcome.max_error_m),
                "rmse": float(outcome.rmse_m), "mean_nees": outcome.mean_nees})
    return {"rows": [row], "worker_cpu_s": time.process_time() - cpu_start}


def build_jobs(pilot: bool) -> tuple[list[dict], list[int], tuple[float, ...]]:
    seeds = pilot_seeds() if pilot else experiment_seeds(60)
    delays = PILOT_DELAYS if pilot else DELAYS
    targets = PILOT_TARGETS if pilot else TARGETS
    losses = PILOT_LOSSES if pilot else LOSSES
    pairs = PILOT_PAIRS if pilot else NOISE_PAIRS
    check_targets = ("cv_1",) if pilot else CHECK_TARGETS
    check_pairs = PILOT_CHECK_PAIRS if pilot else NOISE_PAIRS
    jobs = [make_job("main", target, model, loss, n, "random", delay, noise, told, seed)
            for target in targets for model, loss in losses for noise, told in pairs
            for n in NS for delay in delays for seed in seeds]
    jobs += [make_job("check", target, "independent", 0.0, n, phase, 0.0, noise, told, seed)
             for target in check_targets for noise, told in check_pairs
             for n, phase in CHECK_ARMS for seed in seeds]
    return jobs, seeds, delays


def expected_row_count(pilot: bool) -> int:
    return len(build_jobs(pilot)[0])


def pair_label(noise: float, told: str) -> str:
    return f"{round(float(noise) * 100)}|{told}"


def condition_key(target, loss_model, loss, noise, told, n) -> str:
    return f"{target}|{loss_model}:{float(loss):g}|{float(noise):g}|{told}|{int(n)}"


def step_index(boundary: dict) -> int:
    """fails at 0 = 0; tolerates grid delay number i (0 ms is 1, 200 ms is 6) = i; never fails = 8.

    There is no 7: tolerating 300 ms (the last grid delay) and never failing are the same
    observation, so both score 8.
    """
    if boundary["fails_at_zero"]:
        return 0
    if boundary["censored"]:
        return len(DELAYS) + 1
    value = float(boundary["value"])
    matches = [i for i, delay in enumerate(DELAYS) if abs(delay - value) < 1e-12]
    if not matches or matches[0] == len(DELAYS) - 1:
        raise ValueError(f"tolerated delay {value} is not an interior grid delay")
    return matches[0] + 1


def bootstrap_mean(values, resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean and percentile 95 % interval, resampling the per-seed values with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[picks].mean(axis=1)
    return (float(np.mean(values)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def verdict(holds: dict[str, bool]) -> str:
    if all(holds[label] for label in VERDICT_PAIRS):
        return "holds across noise"
    if holds["2|truth"] and holds["10|truth"]:
        return "holds only when told the truth"
    if not holds["10|told5"] and not holds["10|truth"]:
        return "breaks at 10 cm"
    return "mixed"


def _fail_matrix(rows, seeds, delays) -> np.ndarray:
    """seeds x delays array of 0/1 seed failures for one condition."""
    lookup = {(int(r["seed"]), float(r["delay"])): float(bool(r["failed"])) for r in rows}
    return np.array([[lookup[(seed, delay)] for delay in delays] for seed in seeds])


def _group_main(rows) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["slice"] == "main":
            key = condition_key(row["target"], row["loss_model"], row["loss"], row["noise_std_m"],
                                row["told"], row["n"])
            grouped.setdefault(key, []).append(row)
    return grouped


def _boundaries(grouped, seeds, delays) -> dict:
    out = {}
    wanted = set(seeds)
    for key, rows in sorted(grouped.items()):
        selected = [r for r in rows if int(r["seed"]) in wanted]
        counts = [sum(bool(r["failed"]) for r in selected if float(r["delay"]) == delay)
                  for delay in delays]
        result = tolerated(delays, counts, len(seeds))
        result["fail_counts"] = counts
        out[key] = result
    return out


def _gate(rows, registered_seeds, delays, registered_boundaries, n_conditions: int) -> dict:
    """Compare the 5 cm, told 5 cm rows with the registered EXP-024 rows.

    Fails closed: every expected row must exist once on each side, and with the full delay
    grid every 5 cm boundary (one per target, loss and N) must be compared.
    """
    wanted = {}
    duplicates = 0
    for row in rows:
        if (row["slice"] == "main" and row["noise_std_m"] == 0.05 and row["told"] == "told5"
                and int(row["seed"]) in registered_seeds):
            key = (row["target"], row["loss_model"], row["loss"], row["arm"], int(row["n"]),
                   row["delay"], int(row["seed"]))
            duplicates += key in wanted
            wanted[key] = row
    expected_rows = n_conditions * len(NS) * len(delays) * len(registered_seeds)
    mismatches = []
    found = 0
    seen = set()
    with gzip.open(REGISTERED_RUN / "per_seed.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for reg in csv.DictReader(handle):
            if reg["link"] != "abstract":
                continue
            key = (reg["target"], reg["loss_model"], float(reg["loss"]), reg["arm"], int(reg["n"]),
                   float(reg["delay"]), int(reg["seed"]))
            if key not in wanted:
                continue
            found += 1
            duplicates += key in seen
            seen.add(key)
            for field in GATE_FIELDS:
                if csv_text(wanted[key][field]) != reg[field]:
                    mismatches.append({"key": list(key), "field": field})
    boundary_checks = 0
    boundary_mismatches = []
    full_grid = tuple(delays) == tuple(DELAYS) and len(registered_seeds) == N_REGISTERED
    expected_boundaries = n_conditions * len(NS) if full_grid else 0
    if full_grid:
        registered = json.loads((REGISTERED_RUN / "boundaries.json").read_text(encoding="utf-8"))
        for key, ours in registered_boundaries.items():
            target, loss, noise, told, n = key.split("|")
            if float(noise) != 0.05 or told != "told5":
                continue
            reg_key = f"abstract|{target}|{loss}|{ARM_FOR_N[int(n)]}|{n}"
            boundary_checks += 1
            if registered[reg_key] != ours:
                boundary_mismatches.append(key)
    passed = (found == len(wanted) == expected_rows > 0 and duplicates == 0 and not mismatches
              and boundary_checks == expected_boundaries and not boundary_mismatches)
    return {"passed": bool(passed), "rows_compared": found, "rows_expected": expected_rows,
            "duplicate_keys": duplicates, "fields": list(GATE_FIELDS),
            "row_mismatches": mismatches[:20], "boundaries_compared": boundary_checks,
            "boundaries_expected": expected_boundaries, "boundary_mismatches": boundary_mismatches}


def _conditions(grouped):
    return sorted({tuple(key.split("|")[:2]) for key in grouped})


def _verdict(grouped, boundaries, seeds, delays, pilot: bool) -> dict:
    pairs = PILOT_PAIRS if pilot else NOISE_PAIRS
    conditions = _conditions(grouped)
    per_pair = {}
    holds = {}
    for noise, told in pairs:
        label = pair_label(noise, told)
        entries = {}
        for target, loss in conditions:
            model, probability = loss.split(":")
            keys = [condition_key(target, model, probability, noise, told, n) for n in NS]
            one, three = (boundaries[key] for key in keys)
            diff = (_fail_matrix(grouped[keys[0]], seeds, delays).mean(axis=1)
                    - _fail_matrix(grouped[keys[1]], seeds, delays).mean(axis=1))
            mean, low, high = bootstrap_mean(diff)
            entries[f"{target}|{loss}"] = {
                "index_1": step_index(one), "index_3": step_index(three),
                "gain_steps": step_index(three) - step_index(one),
                "pooled_fail_rate_1_minus_3": {"mean": mean, "lo": low, "hi": high},
                "supported": bool(low > 0.0)}
        holds[label] = all(entry["gain_steps"] >= 1 for entry in entries.values())
        per_pair[label] = entries
    told_effect = {}
    for noise in sorted({noise for noise, told in pairs if told == "truth"}):
        for target, loss in conditions:
            model, probability = loss.split(":")
            for n in NS:
                five = grouped[condition_key(target, model, probability, noise, "told5", n)]
                truth = grouped[condition_key(target, model, probability, noise, "truth", n)]
                diff = (_fail_matrix(five, seeds, delays).mean(axis=1)
                        - _fail_matrix(truth, seeds, delays).mean(axis=1))
                mean, low, high = bootstrap_mean(diff)
                told_effect[f"{target}|{loss}|{round(noise * 100)}|{n}"] = {
                    "pooled_fail_rate_told5_minus_truth": {"mean": mean, "lo": low, "hi": high}}
    return {"pilot": pilot, "holds": holds,
            "verdict": None if pilot else verdict(holds),
            "per_pair": per_pair, "told_effect": told_effect,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "rng": "numpy default_rng(39), fresh "
                          "for every comparison", "unit": "seed (all delays of a seed kept together)",
                          "interval": "percentile 95 %"},
            "rule": "gain holds at a noise pair if index(3 central) - index(1 single) >= 1 in every "
                    "condition; index: fails at 0 = 0, tolerates grid delay i = i (0 ms = 1, 200 ms = 6), "
                    "never fails = 8",
            "index_note": "There is no index 7 on this grid: tolerating 300 ms and never failing are the "
                          "same observation, both scored 8 as pre-registered. A gain that involves a "
                          "never-failing cell is therefore one step larger than the grid distance. Quote "
                          "tolerated delays in ms (boundaries.json); use gain_steps only for gain >= 1."}


def _check_cells(rows) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["slice"] == "check":
            key = f"{row['target']}|{row['noise_std_m']:g}|{row['told']}|{row['n']}|{row['phase']}"
            grouped.setdefault(key, []).append(row)
    return {key: {"n_seeds": len(values), "fail_count": sum(bool(r["failed"]) for r in values),
                  "rmse_median": float(np.median([r["rmse"] for r in values])),
                  "p95_median": float(np.median([r["p95"] for r in values]))}
            for key, values in sorted(grouped.items())}


def _hand_check(check: dict) -> dict:
    """Compare the no-loss, zero-delay slice with the independent reference script."""
    completed = subprocess.run([sys.executable, "-I", str(HAND_SCRIPT)], check=True,
                               capture_output=True, text=True)
    ref = json.loads(completed.stdout)
    out = {"reference_script": "tools/coop_noise_hand.py", "items": [], "missing": [],
           "ideal_ratio": ref["part_a"]["ideal_ratio_1_over_sqrt3"]}
    for noise, told in NOISE_PAIRS:
        if noise == 0.0:
            continue
        stem = f"cv_1|{noise:g}|{told}"
        keys = [f"{stem}|1|random", f"{stem}|3|random", f"{stem}|3|sync"]
        missing = [key for key in keys if key not in check]
        if missing:
            out["missing"].extend(missing)
            continue
        r_told = 0.05 if told == "told5" else noise
        reference = ref["part_a"][f"true{noise:.2f}_told{r_told:.2f}"]
        sync_ref = reference["ratio_N3sync_over_N1"]
        stag_ref = reference["ratio_N3stag_over_N1"]
        one = check[keys[0]]["rmse_median"]
        random_ratio = check[keys[1]]["rmse_median"] / one
        sync_ratio = check[keys[2]]["rmse_median"] / one
        low, high = min(sync_ref, stag_ref) - 0.03, max(sync_ref, stag_ref) + 0.03
        out["items"].append({"pair": pair_label(noise, told), "measured_random_ratio": random_ratio,
                             "measured_sync_ratio": sync_ratio, "reference_sync_ratio": sync_ref,
                             "reference_staggered_ratio": stag_ref,
                             "measured_rmse_1": one, "reference_rmse_1": reference["N1"]["rms3d_noise"],
                             "random_in_band": bool(low <= random_ratio <= high),
                             "sync_within_0.03": bool(abs(sync_ratio - sync_ref) <= 0.03)})
    zero = {}
    for target, ref_key, field in (("cv_1", "straight_line_1mps", "p95_abs"),
                                   ("ct_r1", "circle_r1_v1", "p95")):
        key = f"{target}|0|told5|1|random"
        if key not in check:
            out["missing"].append(key)
        else:
            measured = check[key]["p95_median"]
            expected = ref["part_b"][ref_key]["N1"][field]
            zero[target] = {"measured_p95": measured, "reference_p95": expected,
                            "within_1mm": bool(abs(measured - expected) <= 0.001)}
    out["zero_noise_single"] = zero
    flags = [item["random_in_band"] and item["sync_within_0.03"] for item in out["items"]]
    flags += [entry["within_1mm"] for entry in zero.values()]
    out["passed"] = bool(flags) and all(flags) and not out["missing"]
    return out


def _cells_csv(output: Path, grouped, seeds, delays) -> None:
    columns = ("target", "loss_model", "loss", "noise_std_m", "told", "n", "delay", "n_seeds",
               "fail_count", "fail_rate", "fail_rate_wilson_lo", "fail_rate_wilson_hi",
               "p95_median", "rmse_median", "max_median")
    out = []
    for key, rows in sorted(grouped.items()):
        target, loss, noise, told, n = key.split("|")
        model, probability = loss.split(":")
        for delay in delays:
            selected = [r for r in rows if float(r["delay"]) == delay]
            failed = sum(bool(r["failed"]) for r in selected)
            low, high = wilson_interval(failed, len(seeds))
            out.append({"target": target, "loss_model": model, "loss": float(probability),
                        "noise_std_m": float(noise), "told": told, "n": int(n), "delay": delay,
                        "n_seeds": len(selected), "fail_count": failed,
                        "fail_rate": failed / len(selected), "fail_rate_wilson_lo": low,
                        "fail_rate_wilson_hi": high,
                        "p95_median": float(np.median([r["p95"] for r in selected])),
                        "rmse_median": float(np.median([r["rmse"] for r in selected])),
                        "max_median": float(np.median([r["max"] for r in selected]))})
    write_csv(output / "cells.csv", out, columns)


def _figure(boundaries: dict, path: Path, pilot: bool) -> None:
    pal = horizon_mpl.apply_style("light")
    conditions = sorted({tuple(key.split("|")[:2]) for key in boundaries})
    noises = sorted({float(key.split("|")[2]) for key in boundaries})
    fig, axes = plt.subplots(len(TARGETS) if not pilot else 1, max(1, len(conditions) // (1 if pilot else 2)),
                             figsize=(10.5, 6.4 if not pilot else 3.4), squeeze=False)
    colors = {1: pal["lineStrong"], 3: pal["accent"]}
    for index, (target, loss) in enumerate(conditions):
        axis = axes.flat[index]
        for n in NS:
            for told, style in (("told5", "-"), ("truth", "--")):
                xs, ys = [], []
                for noise in noises:
                    key = f"{target}|{loss}|{noise:g}|{told}|{n}"
                    if told == "truth" and noise == 0.05:
                        # told the truth at 5 cm is the same filter as told 5 cm (run once)
                        key = f"{target}|{loss}|{noise:g}|told5|{n}"
                    if key not in boundaries:
                        continue
                    xs.append(noise * 100.0 + (0.15 if told == "truth" else 0.0))
                    ys.append(step_index(boundaries[key]))
                axis.plot(xs, ys, linestyle=style, marker="o" if n == 1 else "s", color=colors[n],
                          markerfacecolor="white", linewidth=1.2, markersize=4.5,
                          label=f"{n} observer{'s, central' if n == 3 else ''}, "
                                f"{'told 5 cm' if told == 'told5' else 'told the truth'}")
        model, probability = loss.split(":")
        name = "1 m turn" if target == "ct_r1" else "jinking"
        loss_name = f"{float(probability) * 100:g} % {'bursts' if model == 'burst' else 'independent'}"
        axis.set_title(f"{name}, {loss_name}")
        axis.set_xticks([noise * 100.0 for noise in noises])
        axis.set_ylim(-0.4, 8.4)
        axis.set_yticks(range(9))
        axis.set_yticklabels(["fails at 0"] + [f"{d * 1000:g} ms" for d in DELAYS] + ["> 300 ms"])
        axis.set_xlabel("true camera noise (cm)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("Simulated (EXP-039 proposal): largest delay that still passes, 60 seeds, abstract link")
    fig.text(0.5, 0.005, "The vertical axis steps through the delay grid and is not to scale. "
             "Told the truth is not run at 0 cm; at 5 cm it is the same filter as told 5 cm.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.1 if not pilot else 0.22, 1, 0.95))
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _source_paths() -> list[str]:
    """Every repository .py file this process has imported, plus the hand reference script."""
    names = {"tools/coop_noise_hand.py"}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(REPO_ROOT):
            names.add(str(path.relative_to(REPO_ROOT)))
    return sorted(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 3:
        raise ValueError("--processes must be between 1 and 3")
    output = prepare_output(Path(args.output))
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    pilot = bool(args.pilot)
    jobs, seeds, delays = build_jobs(pilot)
    registered_seeds = set(seeds_for("tolerance", N_REGISTERED)) & set(seeds)
    config = {"experiment": "EXP-039 (proposal)", "pilot": pilot, "link": "abstract", "rate_hz": 10.0,
              "phase": "random (o1 at 0) except the sync check arm", "targets_main": list(
                  PILOT_TARGETS if pilot else TARGETS),
              "losses": [list(item) for item in (PILOT_LOSSES if pilot else LOSSES)],
              "noise_pairs": [list(item) for item in (PILOT_PAIRS if pilot else NOISE_PAIRS)],
              "told5_means": "r_std_m=None (filter default 0.05 m), as registered",
              "ns": {str(n): ARM_FOR_N[n] for n in NS}, "delays_s": list(delays),
              "check_slice": {"targets": ["cv_1"] if pilot else list(CHECK_TARGETS),
                              "loss": 0.0, "delay": 0.0, "arms": [list(a) for a in CHECK_ARMS]},
              "seeds": seeds, "n_registered_seeds": len(registered_seeds),
              "failure_rule": {"p95_error_m": 0.25, "max_error_m": 0.50},
              "cell_rule": "fails when at least half the seeds fail (Proposed)"}
    sources = _source_paths()
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    for name in sources:
        digest.update(name.encode("utf-8"))
        digest.update((REPO_ROOT / name).read_bytes())
    fingerprint = "coop-noise-" + digest.hexdigest()
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    raw = run_jobs(jobs, "tools.d_run_coop_noise:_job", processes=args.processes,
                   checkpoint_dir=checkpoint, fingerprint=fingerprint, progress=False)
    runtime_s = time.perf_counter() - start
    rows = [row for result in raw for row in result["rows"]]
    if len(rows) != len(jobs):
        raise RuntimeError("runner rows do not match the job list")
    write_csv_gz(output / "per_seed.csv.gz", rows, COLUMNS)
    grouped = _group_main(rows)
    boundaries = {"all_seeds": _boundaries(grouped, seeds, delays),
                  "registered_seeds": _boundaries(grouped, sorted(registered_seeds), delays)}
    write_json(output / "boundaries.json", boundaries)
    n_conditions = len(PILOT_TARGETS if pilot else TARGETS) * len(PILOT_LOSSES if pilot else LOSSES)
    gate = _gate(rows, registered_seeds, delays, boundaries["registered_seeds"], n_conditions)
    write_json(output / "reproduction.json", gate)
    if not gate["passed"]:
        raise RuntimeError("reproduction gate failed: the 5 cm rows do not match EXP-024")
    _cells_csv(output, grouped, seeds, delays)
    write_json(output / "verdict.json", _verdict(grouped, boundaries["all_seeds"], seeds, delays, pilot))
    check = _check_cells(rows)
    write_json(output / "check_cells.json", check)
    outputs = ["per_seed.csv.gz", "boundaries.json", "reproduction.json", "cells.csv", "verdict.json",
               "check_cells.json"]
    if not pilot:
        write_json(output / "hand_check.json", _hand_check(check))
        outputs.append("hand_check.json")
    _figure(boundaries["all_seeds"], output / "tolerance_vs_noise.png", pilot)
    outputs.append("tolerance_vs_noise.png")
    worker_cpu_s = float(sum(float(result["worker_cpu_s"]) for result in raw))
    write_json(output / "timing.json", {"runtime_s": runtime_s, "worker_cpu_s": worker_cpu_s,
                                        "n_jobs": len(jobs)})
    command = "PYTHONPATH=. {python} tools/d_run_coop_noise.py --output {out}"
    if pilot:
        command += " --pilot"
    if args.processes != 2:
        command += f" --processes {args.processes}"
    write_manifest(output, schema="zephyr-d-coop-noise-manifest-1", argv=argv_or_default(argv),
                   command=command, config=config, seeds=seeds, outputs=outputs,
                   limits=["Proposal (EXP-039), not Adopted.",
                           "Agents move as kinematic point masses in the tracking simulation.",
                           "Sensor samples are simulated Gaussian noise on the true target position.",
                           "The link is the abstract model with synthetic loss; nothing is measured.",
                           "Sixty seeds; one-step boundary differences are still weak.",
                           "The cell rule (half the seeds fail) is Proposed, not Adopted."],
                   runtime_s=runtime_s, n_processes=args.processes, hashed_sources=sources,
                   extra={"volatile_fields": ["runtime_s", "git_revision", "git_tree_state", "platform",
                                              "n_processes", "argv"],
                          "n_runs": len(jobs)})


if __name__ == "__main__":
    main()
