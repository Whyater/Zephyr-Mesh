"""Diagnostic only (proposal): how many EXP-023 seeds would the windowed NEES check flag?

Reruns the first ``--seeds`` seeds of every main-grid EXP-023 cell from
sim/runs/d2-fusion/per_seed.csv, one arm at a time, checks each rerun matches the
stored p95 error exactly, and applies sim.estimation.divergence to the
outcome's 10-query NEES window means. Writes nothing into sim/runs/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
from pathlib import Path

from sim.coop.observers import run_seed
from sim.estimation.divergence import DivergenceRule, check_window_means
from sim.s6_recovery import RecoveryConfig
from tools.d_run_fusion import TARGETS, _cell

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    params = {label: (kind, extra) for label, kind, extra in TARGETS}
    with open(ROOT / "sim/runs/d2-fusion/per_seed.csv", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["ci"] == "fast"]
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["n_observers"], r["noise_std_m"], r["target"], r["loss_probability"], r["arm"])].append(r)
    rule = DivergenceRule()
    tally = collections.defaultdict(lambda: collections.Counter())
    examples = []
    per_run = []
    mismatches = 0
    started = time.process_time()
    for key in sorted(by_cell):
        for r in by_cell[key][: args.seeds]:
            kind, extra = params[r["target"]]
            job = {"target": kind, "target_params": extra, "n_observers": r["n_observers"],
                   "noise_std_m": r["noise_std_m"], "loss_probability": r["loss_probability"],
                   "ci": "fast"}
            out = run_seed(_cell(job), int(r["seed"]), arms=(r["arm"],), config=RecoveryConfig())[r["arm"]]
            stored = float(r["p95_error_m"]) if r["p95_error_m"] else None
            rerun = out.p95_error_m if math.isfinite(out.p95_error_m) else None
            if stored != rerun:  # both absent counts as a match; numbers must be exactly equal
                mismatches += 1
            metrics_finite = all(v is not None and math.isfinite(v)
                                 for v in (out.p95_error_m, out.max_error_m, out.rmse_m, out.mean_nees))
            res = check_window_means(out.nees_window_means, rule)
            old = bool((out.extras or {}).get("diverged", False))
            blown = out.max_error_m > 1.0
            per_run.append({"arm": r["arm"], "n": int(r["n_observers"]), "target": r["target"],
                            "noise": r["noise_std_m"], "loss": r["loss_probability"],
                            "seed": int(r["seed"]), "max_m": out.max_error_m,
                            "p95_m": out.p95_error_m, "failed": bool(out.failed), "old_flag": old,
                            "metrics_finite": metrics_finite,
                            "window_means": list(out.nees_window_means)})
            t = tally[(r["arm"], int(r["n_observers"]), r["noise_std_m"])]
            t["seeds"] += 1
            t["new_inconsistent"] += res.inconsistent
            t["new_diverged"] += res.diverged
            t["old_flag"] += old
            t["registered_fail"] += bool(out.failed)
            t["max_over_1m"] += blown
            t["max_over_1m_and_new_diverged"] += blown and res.diverged
            if blown and metrics_finite and not old and len(examples) < 6:
                examples.append({"arm": r["arm"], "n": int(r["n_observers"]), "target": r["target"],
                                 "noise": r["noise_std_m"], "loss": r["loss_probability"],
                                 "seed": int(r["seed"]), "p95_m": out.p95_error_m,
                                 "max_m": out.max_error_m, "mean_nees": out.mean_nees,
                                 "first_diverged_window": res.first_window,
                                 "first_diverged_time_s": None if res.first_window is None
                                 else RecoveryConfig().burn_in_s + res.first_window * 10 * RecoveryConfig().query_dt_s,
                                 "old_flag": old})
    report = {"rule": {"dof": rule.dof, "window": rule.window,
                       "window_false_alarm": rule.window_false_alarm,
                       "consecutive": rule.consecutive, "threshold": rule.threshold,
                       "gross_nees": rule.gross_nees},
              "seeds_per_cell": args.seeds, "p95_mismatches_vs_stored": mismatches,
              "cpu_s": time.process_time() - started,
              "tally": {f"{a}|N={n}|noise={z}": dict(c) for (a, n, z), c in sorted(tally.items())},
              "examples_max_over_1m_all_metrics_finite_old_flag_false": examples}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        args.out.with_suffix(".runs.json").write_text(json.dumps(per_run) + "\n", encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
