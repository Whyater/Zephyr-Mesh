"""Compare an EXP-023 run made with --nees-divergence against the registered one.

Proposal tooling (simulated results only). It answers: if the opt-in check had
been on, which rows would change from passed to failed, by arm and observer
count? Rows are matched on (n_observers, noise, target, loss, arm, ci, seed).

Usage (full flag-on run already made):
    PYTHONPATH=. python tools/burn_divergence_wire_effect.py \
        runs/d2-fusion/per_seed.csv /tmp/d2-on/per_seed.csv --json /tmp/effect.json

Usage (sample: rerun the first K registered seeds of every cell with the flag on,
one process, and compare with the matching registered rows):
    PYTHONPATH=. python tools/burn_divergence_wire_effect.py \
        runs/d2-fusion/per_seed.csv --sample-seeds 4 --json /tmp/effect_sample.json
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from sim.estimation.sweep import write_csv
from tools import d_run_fusion as runner
from tools.d_common import seeds_for
from tools.d_run_fusion import NEES_COLUMNS, ROW_COLUMNS

KEY = ("n_observers", "noise_std_m", "target", "loss_probability", "arm", "ci", "seed")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _true(value: str) -> bool:
    """CSV booleans from sim.estimation.sweep.write_csv are 'true'/'false'."""
    if value not in ("true", "false"):
        raise ValueError(f"not a CSV boolean: {value!r}")
    return value == "true"


def compare(registered: list[dict], flagged: list[dict]) -> dict:
    old = {tuple(r[k] for k in KEY): r for r in registered}
    new = {tuple(r[k] for k in KEY): r for r in flagged}
    if set(old) != set(new) or len(old) != len(registered) or len(new) != len(flagged):
        raise ValueError("the two runs do not have the same rows")
    shared = [c for c in ROW_COLUMNS if c not in ("failed",)]
    mismatched = sorted({c for key in old for c in shared if old[key][c] != new[key][c]})
    groups: dict[tuple, Counter] = {}
    for key, n in new.items():
        o = old[key]
        g = groups.setdefault((n["arm"], n["ci"], int(n["n_observers"])), Counter())
        g["runs"] += 1
        g["old_failed"] += _true(o["failed"])
        g["new_failed"] += _true(n["failed"])
        g["nees_diverged"] += _true(n["nees_diverged"])
        g["nees_inconsistent_warning"] += _true(n["nees_inconsistent"])
        g["pass_to_fail"] += not _true(o["failed"]) and _true(n["failed"])
        g["fail_to_pass"] += _true(o["failed"]) and not _true(n["failed"])
    table = [{"arm": a, "ci": ci, "n_observers": n, **dict(sorted(c.items()))}
             for (a, ci, n), c in sorted(groups.items())]
    totals = Counter()
    for c in groups.values():
        totals.update(c)
    return {"rows": len(new), "columns_added": list(NEES_COLUMNS),
            "shared_columns_that_differ": mismatched,
            "totals": dict(sorted(totals.items())), "by_arm": table}


def sample_rows(k: int) -> list[dict]:
    """Flag-on rows for the first k registered seeds of every registered cell."""
    seeds = seeds_for("fusion", 50)[:k]
    jobs = runner._jobs((2, 3, 4, 6), (0.05, 0.10), runner.TARGETS, (0.0, 0.3), seeds,
                        "fast", True)
    exact = runner._jobs((2, 3, 6), (0.05,), (runner.TARGETS[1],), (0.0, 0.3), seeds,
                         "exact", True)
    for job in exact:
        job["arms"] = ("peer_ci",)
    rows = [row for job in jobs + exact for row in runner._job(job)["rows"]]
    # Round-trip through the runner's own CSV writer so values compare as text.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.csv"
        write_csv(path, rows, ROW_COLUMNS + NEES_COLUMNS)
        return _rows(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registered", type=Path)
    parser.add_argument("flagged", type=Path, nargs="?")
    parser.add_argument("--sample-seeds", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    registered = _rows(args.registered)
    if args.sample_seeds:
        flagged = sample_rows(args.sample_seeds)
        wanted = {tuple(r[k] for k in KEY) for r in flagged}
        registered = [r for r in registered if tuple(r[k] for k in KEY) in wanted]
    elif args.flagged:
        flagged = _rows(args.flagged)
    else:
        parser.error("give a flagged per_seed.csv or --sample-seeds")
    result = compare(registered, flagged)
    if args.sample_seeds:
        result["sample_seeds_per_cell"] = args.sample_seeds
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not result["shared_columns_that_differ"] and result["totals"].get(
        "fail_to_pass", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
