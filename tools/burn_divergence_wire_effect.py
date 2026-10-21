"""Compare an EXP-023 run made with --nees-divergence against the registered one.

Proposal tooling (simulated results only). It answers: if the opt-in check had
been on, which rows would change from passed to failed, by arm and observer
count? Rows are matched on (n_observers, noise, target, loss, arm, ci, seed).

Usage:
    PYTHONPATH=. python tools/burn_divergence_wire_effect.py \
        runs/d2-fusion/per_seed.csv /tmp/d2-on/per_seed.csv --json /tmp/effect.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tools.d_run_fusion import NEES_COLUMNS, ROW_COLUMNS

KEY = ("n_observers", "noise_std_m", "target", "loss_probability", "arm", "ci", "seed")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        g["old_failed"] += o["failed"] == "True"
        g["new_failed"] += n["failed"] == "True"
        g["nees_diverged"] += n["nees_diverged"] == "True"
        g["nees_inconsistent_warning"] += n["nees_inconsistent"] == "True"
        g["pass_to_fail"] += o["failed"] == "False" and n["failed"] == "True"
        g["fail_to_pass"] += o["failed"] == "True" and n["failed"] == "False"
    table = [{"arm": a, "ci": ci, "n_observers": n, **dict(sorted(c.items()))}
             for (a, ci, n), c in sorted(groups.items())]
    totals = Counter()
    for c in groups.values():
        totals.update(c)
    return {"rows": len(new), "columns_added": list(NEES_COLUMNS),
            "shared_columns_that_differ": mismatched,
            "totals": dict(sorted(totals.items())), "by_arm": table}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registered", type=Path)
    parser.add_argument("flagged", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = compare(_rows(args.registered), _rows(args.flagged))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not result["shared_columns_that_differ"] and result["totals"].get(
        "fail_to_pass", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
