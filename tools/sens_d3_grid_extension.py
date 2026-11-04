"""Side run: where does the EXP-024 cooperative tracker fail past the registered grid?

The registered tolerance grid (``sim/runs/d3-tolerance``) stops at 0.30 s of delay
and independent loss 0.5, so many cells end as "300+" (censored).  This wrapper
extends one slice of it, turn target ``ct_r1`` on the abstract link, along two
axes:

* delay 0.4, 0.5, 0.75 and 1.0 s at independent loss 0;
* independent loss 0.7, 0.8 and 0.9 at zero delay.

Every new cell is run by the registered worker ``tools.d_run_tolerance._job``
(imported read-only), with the registered seeds, so only the delay or loss
changes.  Before the extension runs, a few old-grid cells are rerun and must
match the registered ``per_seed.csv.gz`` rows as CSV text (exact on macOS, the
``sim/tests/platform_compare.py`` rule elsewhere).

The edge on each axis is the registered ``sim.coop.boundary.tolerated`` rule over
the merged axis: registered fail counts for the old values, new counts for the
new ones.  On the loss axis the rule runs over loss values instead of delays.

Outputs are simulated only and must go to a folder outside ``sim/runs``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim.coop.boundary import tolerated
from sim.estimation.sweep import _csv_value, run_jobs, write_csv, write_csv_gz
from sim.s6_recovery import median_interval
from sim.tests.platform_compare import same_csv_row
from tools import d_run_tolerance as registered
from tools.d_common import seeds_for

REGISTERED_PER_SEED = ROOT / "sim" / "runs" / "d3-tolerance" / "per_seed.csv.gz"
WORKER = "tools.d_run_tolerance:_job"
LINK = "abstract"
TARGET = "ct_r1"
NS = registered.NS
N_SEEDS = 20
NEW_DELAYS = (0.4, 0.5, 0.75, 1.0)
NEW_LOSSES = (0.7, 0.8, 0.9)
OLD_DELAYS = registered.DELAYS
OLD_LOSSES = tuple(p for model, p in registered.LOSSES if model == "independent")
DELAY_AXIS = OLD_DELAYS + NEW_DELAYS
LOSS_AXIS = OLD_LOSSES + NEW_LOSSES
# (delay, loss, n): old-grid cells rerun as the control.
CONTROL_CELLS = ((0.3, 0.5, 2), (0.3, 0.0, 1), (0.0, 0.5, 6))
COLUMNS = ("link", "target", "loss_model", "loss", "arm", "n", "delay", "seed",
           "failed", "p95", "max", "rmse", "mean_nees", "busy_fraction",
           "delivery_ratio", "gap_check")


def seeds() -> list[int]:
    """The registered EXP-024 seed list."""
    return seeds_for("tolerance", N_SEEDS)


def make_job(delay: float, loss: float, n: int, seed: int) -> dict:
    """One worker job in the registered key order."""
    return {"link": LINK, "target": TARGET, "loss_model": "independent", "loss": float(loss),
            "n": int(n), "delay": float(delay), "seed": int(seed)}


def extension_cells() -> list[tuple[float, float, int]]:
    """(delay, loss, n) for every new cell: the delay axis at loss 0, the loss axis at delay 0."""
    cells = [(delay, 0.0, n) for delay in NEW_DELAYS for n in NS]
    cells += [(0.0, loss, n) for loss in NEW_LOSSES for n in NS]
    return cells


def jobs_for(cells, seed_list) -> list[dict]:
    return [make_job(delay, loss, n, seed) for delay, loss, n in cells for seed in seed_list]


def encode_row(row: dict) -> dict:
    """Same encoding as the registered runner uses for per_seed.csv.gz."""
    encoded = dict(row)
    encoded["gap_check"] = (json.dumps(row["gap_check"], sort_keys=True)
                            if row["gap_check"] is not None else None)
    return encoded


def row_text(row: dict) -> str:
    """CSV text of one encoded row, written exactly as the registered writer does."""
    handle = io.StringIO()
    csv.writer(handle, lineterminator="\n").writerow(
        [_csv_value(row.get(column)) for column in COLUMNS])
    return handle.getvalue().rstrip("\n")


def row_key(link, target, loss_model, loss, arm, n, delay, seed) -> tuple:
    return (str(link), str(target), str(loss_model), float(loss), str(arm), int(n),
            float(delay), int(seed))


def load_registered(path: Path = REGISTERED_PER_SEED, link: str = LINK,
                    target: str = TARGET) -> dict[tuple, tuple[str, dict]]:
    """Registered rows for one link and target: key -> (raw CSV line, parsed row)."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        lines = handle.read().split("\n")
    header = next(csv.reader([lines[0]]))
    if tuple(header) != COLUMNS:
        raise ValueError("registered per_seed.csv.gz has unexpected columns")
    out = {}
    for line in lines[1:]:
        if not line:
            continue
        values = dict(zip(COLUMNS, next(csv.reader([line]))))
        if values["link"] != link or values["target"] != target:
            continue
        key = row_key(values["link"], values["target"], values["loss_model"], values["loss"],
                      values["arm"], values["n"], values["delay"], values["seed"])
        out[key] = (line, values)
    return out


def run_rows(jobs: list[dict], processes: int, checkpoint: Path | None = None,
             fingerprint: str = "") -> tuple[list[dict], float]:
    raw = run_jobs(jobs, WORKER, processes=processes, checkpoint_dir=checkpoint,
                   fingerprint=fingerprint, progress=False)
    rows = [row for result in raw for row in result["rows"]]
    cpu = float(sum(float(result["worker_cpu_s"]) for result in raw))
    return rows, cpu


def control(processes: int = 1, cells=CONTROL_CELLS, seed_list=None,
            reference: dict | None = None) -> dict:
    """Rerun old-grid cells and compare each row with the registered CSV text."""
    seed_list = seeds() if seed_list is None else list(seed_list)
    reference = load_registered() if reference is None else reference
    rows, cpu = run_rows(jobs_for(cells, seed_list), processes)
    exact = tolerant = 0
    mismatches = []
    for row in rows:
        key = row_key(row["link"], row["target"], row["loss_model"], row["loss"], row["arm"],
                      row["n"], row["delay"], row["seed"])
        if key not in reference:
            mismatches.append({"key": list(key), "reason": "not in registered rows"})
            continue
        expected = reference[key][0]
        actual = row_text(encode_row(row))
        if actual == expected:
            exact += 1
        elif same_csv_row(actual, expected):
            tolerant += 1
        else:
            mismatches.append({"key": list(key), "actual": actual, "expected": expected})
    expected_rows = sum(len(arms_for(n)) for _, _, n in cells) * len(seed_list)
    return {"cells": [list(cell) for cell in cells], "rows": len(rows),
            "expected_rows": expected_rows,
            "exact_text_matches": exact, "platform_tolerance_matches": tolerant,
            "mismatches": mismatches,
            "passed": not mismatches and len(rows) == expected_rows > 0,
            "worker_cpu_s": cpu, "platform": sys.platform}


def arms_for(n: int) -> tuple[str, ...]:
    return registered._registered_arms(n)


def fail_counts(rows, axis: str) -> dict[tuple[str, int, float], tuple[int, int]]:
    """(arm, n, axis value) -> (failed seeds, seeds) for rows on one axis."""
    counts: dict[tuple[str, int, float], list[int]] = {}
    for row in rows:
        loss, delay = float(row["loss"]), float(row["delay"])
        if axis == "delay" and loss == 0.0:
            value = delay
        elif axis == "loss" and delay == 0.0 and row["loss_model"] == "independent":
            value = loss
        else:
            continue
        failed = row["failed"]
        failed = failed if isinstance(failed, bool) else failed == "true"
        entry = counts.setdefault((str(row["arm"]), int(row["n"]), value), [0, 0])
        entry[0] += int(failed)
        entry[1] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def edges(counts: dict, axis_values: tuple[float, ...], n_seeds: int = N_SEEDS) -> dict:
    """Registered boundary rule per arm and N over one merged axis."""
    out = {}
    for n in NS:
        for arm in arms_for(n):
            per_value = []
            for value in axis_values:
                failed, total = counts[(arm, n, value)]
                if total != n_seeds:
                    raise ValueError(f"{arm} N={n} at {value:g} has {total} seeds, not {n_seeds}")
                per_value.append(failed)
            result = tolerated(axis_values, per_value, n_seeds)
            result["fail_counts"] = per_value
            out[f"{arm}|{n}"] = result
    return out


def cell_table(registered_rows, new_rows) -> list[dict]:
    """One line per arm, N and axis value on both merged axes, with its source."""
    lines = []
    for axis, values, new_values in (("delay", DELAY_AXIS, NEW_DELAYS),
                                     ("loss", LOSS_AXIS, NEW_LOSSES)):
        for value in values:
            source = "extension" if value in new_values else "registered"
            pool = new_rows if source == "extension" else registered_rows
            for n in NS:
                for arm in arms_for(n):
                    selected = [r for r in pool if str(r["arm"]) == arm and int(r["n"]) == n
                                and float(r["loss"]) == (0.0 if axis == "delay" else value)
                                and float(r["delay"]) == (value if axis == "delay" else 0.0)
                                and str(r["loss_model"]) == "independent"]
                    failed = sum(bool(r["failed"]) if isinstance(r["failed"], bool)
                                 else r["failed"] == "true" for r in selected)
                    p95 = [float(r["p95"]) for r in selected]
                    median, low, high = median_interval(p95)
                    lines.append({"axis": axis, "value": value, "source": source, "arm": arm,
                                  "n": n, "n_seeds": len(selected), "fail_count": failed,
                                  "cell_fails": failed * 2 >= len(selected),
                                  "p95_median": median, "p95_lo": low, "p95_hi": high,
                                  "max_median": float(np.median([float(r["max"]) for r in selected]))})
    return lines


def gap_summary(rows) -> dict:
    """Measured all-lost fraction against p^N for the new loss cells (an independent check)."""
    out = {}
    for loss in NEW_LOSSES:
        for n in NS:
            checks = [r["gap_check"] for r in rows if float(r["loss"]) == loss
                      and int(r["n"]) == n and r["gap_check"] is not None
                      and str(r["arm"]) in ("single", "central")]
            lost = sum(c["all_lost_steps"] for c in checks)
            steps = sum(c["valid_capture_steps"] for c in checks)
            out[f"{loss:g}|{n}"] = {"hand_all_lost_fraction": loss ** n,
                                    "measured_all_lost_fraction": lost / steps if steps else None,
                                    "capture_steps": steps}
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fingerprint() -> str:
    """Checkpoint fingerprint: the registered fingerprint scheme plus this wrapper's source."""
    digest = hashlib.sha256()
    digest.update(registered._checkpoint_fingerprint({"wrapper": "sens_d3_grid_extension"}).encode())
    digest.update(Path(__file__).read_bytes())
    return "d3-ext-" + digest.hexdigest()


def make_figure(table: list[dict], path: Path) -> None:
    """Heatmap of fail count per arm and N over both merged axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    cmap = ListedColormap(horizon_mpl.sequential_ramp(
        ["surface", "atlasStop1", "atlasStop2", "atlasStop3", "atlasStop4", "accent"], 21))
    labels = [(arm, n) for n in NS for arm in arms_for(n)]
    labels.sort(key=lambda item: (("single", "central", "star_avg", "peer_ci").index(item[0]), item[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2),
                             gridspec_kw={"width_ratios": [len(DELAY_AXIS), len(LOSS_AXIS)]})
    for ax, axis, values, new_values, xlabel in (
            (axes[0], "delay", DELAY_AXIS, NEW_DELAYS, "added delay (ms), independent loss 0"),
            (axes[1], "loss", LOSS_AXIS, NEW_LOSSES, "independent loss probability, delay 0")):
        grid = np.zeros((len(labels), len(values)))
        for i, (arm, n) in enumerate(labels):
            for j, value in enumerate(values):
                line = next(r for r in table if r["axis"] == axis and r["arm"] == arm
                            and r["n"] == n and r["value"] == value)
                grid[i, j] = line["fail_count"]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=N_SEEDS, aspect="auto")
        for i in range(len(labels)):
            for j in range(len(values)):
                count = int(grid[i, j])
                ax.text(j, i, str(count), ha="center", va="center", fontsize=8,
                        color=pal["surface"] if count >= 14 else pal["text"])
                if count * 2 >= N_SEEDS:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=pal["alert"], linewidth=1.2))
        split = len(values) - len(new_values) - 0.5
        ax.axvline(split, color=pal["text"], linewidth=1.5)
        ax.text(split + 0.1, -0.8, "new cells", fontsize=8, color=pal["textSecondary"], va="bottom")
        ticks = [f"{v * 1000:g}" for v in values] if axis == "delay" else [f"{v:g}" for v in values]
        ax.set_xticks(range(len(values)), ticks)
        ax.set_yticks(range(len(labels)), [f"{arm} N={n}" for arm, n in labels])
        ax.set_xlabel(xlabel)
        ax.grid(False)
    axes[1].set_yticklabels([])
    fig.suptitle("SIMULATED. Failed seeds out of 20, turn r = 1 m, abstract link, 10 Hz. "
                 "Red outline: cell fails (10 or more).", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def run(output: Path, processes: int, checkpoint: Path | None, skip_control: bool = False) -> dict:
    output = output.resolve()
    runs_root = (ROOT / "sim" / "runs").resolve()
    if output == runs_root or runs_root in output.parents:
        raise ValueError("side-run output must not go under sim/runs")
    if checkpoint is not None and (checkpoint.resolve() == output
                                   or output in checkpoint.resolve().parents):
        raise ValueError("--checkpoint must be outside --output")
    output.mkdir(parents=True, exist_ok=True)
    seed_list = seeds()
    reference = load_registered()
    start = time.perf_counter()
    control_result = None
    if not skip_control:
        control_result = control(processes, seed_list=seed_list, reference=reference)
        (output / "control.json").write_text(json.dumps(control_result, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        if not control_result["passed"]:
            raise RuntimeError("control rows do not match the registered d3-tolerance rows")
    jobs = jobs_for(extension_cells(), seed_list)
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    rows, cpu = run_rows(jobs, processes, checkpoint, fingerprint())
    registered._validate_rows(rows, jobs)
    write_csv_gz(output / "per_seed.csv.gz", [encode_row(r) for r in rows], COLUMNS)
    registered_rows = [values for _, values in reference.values()]
    table = cell_table(registered_rows, rows)
    write_csv(output / "cells.csv", table, tuple(table[0].keys()))
    merged_delay = fail_counts(registered_rows, "delay") | fail_counts(rows, "delay")
    merged_loss = fail_counts(registered_rows, "loss") | fail_counts(rows, "loss")
    summary = {"delay_axis_s": list(DELAY_AXIS), "loss_axis": list(LOSS_AXIS),
               "delay_edges_at_loss_0": edges(merged_delay, DELAY_AXIS),
               "loss_edges_at_delay_0": edges(merged_loss, LOSS_AXIS),
               "gap_check_new_losses": gap_summary(rows)}
    (output / "edges.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    make_figure(table, output / "grid_extension_heatmap.png")
    outputs = ("per_seed.csv.gz", "cells.csv", "edges.json", "grid_extension_heatmap.png")
    manifest = {
        "schema": "zephyr-sens-d3-grid-extension-1", "simulated": True,
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "command": "PYTHONPATH=. python tools/sens_d3_grid_extension.py --output <dir>"
                   f" --processes {processes}" + (" --checkpoint <dir>" if checkpoint else ""),
        "link": LINK, "target": TARGET, "n_observers": list(NS), "seeds": seed_list,
        "new_delays_s": list(NEW_DELAYS), "new_losses": list(NEW_LOSSES),
        "control_cells": [list(c) for c in CONTROL_CELLS], "control_passed": (
            control_result["passed"] if control_result else None),
        "failure_rule": {"p95_error_m": 0.25, "max_error_m": 0.50, "cell_fail_fraction": 0.5},
        "n_jobs": len(jobs), "n_rows": len(rows), "worker_cpu_s": cpu,
        "runtime_s": time.perf_counter() - start, "platform": sys.platform,
        "registered_per_seed_sha256": _sha256(REGISTERED_PER_SEED),
        "registered_runner_sha256": _sha256(ROOT / "tools" / "d_run_tolerance.py"),
        "outputs": {name: _sha256(output / name) for name in outputs},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.processes <= 4:
        raise ValueError("--processes must be between 1 and 4")
    if args.control_only:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = control(args.processes)
        (output / "control.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("rows", "exact_text_matches",
                                                  "platform_tolerance_matches", "passed")}))
        return 0 if result["passed"] else 1
    run(Path(args.output), args.processes, Path(args.checkpoint) if args.checkpoint else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
