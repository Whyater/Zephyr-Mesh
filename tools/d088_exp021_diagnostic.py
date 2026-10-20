"""Diagnostic only: how EXP-021 reads under the stricter D-088 "failed to fly" rule.

EXP-021 is reported under its pre-registered rules (tilt above 80 degrees,
error above 5 m, x-only growth).  This tool does NOT replace that result.  It
answers a side question: which EXP-021 runs would the stricter rule (tilt
above 45 degrees, height below 0 m, growth on x, y or z) call failed?

Part 1 reads the committed ``per_run.csv``.  Its whole-run maximum tilt and
lowest height cover the same recorded flight the new rule would see, so the
tilt and height parts are exact.  The y and z growth part cannot be checked
from summary columns, so these counts are a lower bound on D-088 failures.

Part 2 reruns the four loss-0 d_crit searches (arm b and slice c, both
profiles, seed 0, 20 s) with ``failed_to_fly_rule=True``.  These are exact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sim.physics.control_tolerance import CONTROL_DT_S, FAILED_TO_FLY_RULES, find_d_crit
from sim.physics.run_manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]
EXP021 = ROOT / "runs" / "a6-control-tolerance"
TILT_LIMIT_DEG = 45.0
MIN_HEIGHT_M = 0.0
LABEL = ("DIAGNOSTIC ONLY. EXP-021 stays reported under its pre-registered rules; "
         "these numbers do not replace that result (D-088).")


def _partial_reclassify(rows: list[dict]) -> dict:
    """Count runs per cell under the old rule and under the tilt and height parts of D-088."""
    cells: dict[tuple, dict] = defaultdict(lambda: {"runs": 0, "old_unstable": 0, "d088_failed_lower_bound": 0})
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (row["profile_id"], row["reference"], row["arm"], float(row["delay_s"]), float(row["loss_probability"]))
        old = row["unstable"] == "True"
        tilt_fail = float(row["whole_run_max_tilt_deg"]) > TILT_LIMIT_DEG
        height_fail = float(row["whole_run_min_z_m"]) < MIN_HEIGHT_M
        cell = cells[key]
        cell["runs"] += 1
        cell["old_unstable"] += old
        if old or tilt_fail or height_fail:
            cell["d088_failed_lower_bound"] += 1
        if not old and (tilt_fail or height_fail):
            reasons["tilt_above_45_deg and height_below_0_m" if tilt_fail and height_fail
                    else "tilt_above_45_deg" if tilt_fail else "height_below_0_m"] += 1
    changed = [
        {"profile_id": k[0], "reference": k[1], "arm": k[2], "delay_s": k[3], "loss_probability": k[4], **v}
        for k, v in sorted(cells.items()) if v["d088_failed_lower_bound"] != v["old_unstable"]
    ]
    return {
        "runs": len(rows),
        "cells": len(cells),
        "old_unstable_runs": sum(v["old_unstable"] for v in cells.values()),
        "d088_failed_runs_lower_bound": sum(v["d088_failed_lower_bound"] for v in cells.values()),
        "newly_failed_by_thresholds": dict(sorted(reasons.items())),
        "cells_that_change": changed,
        "not_checked": "growth of the y or z error (needs the full trajectory, not stored in per_run.csv)",
    }


def _dcrit_compare(config: dict, committed: dict) -> list[dict]:
    kwargs = {"duration_s": 20.0, "scoring_windows": config["scoring_windows"]}
    out = []
    for profile in config["profiles"]:
        for arm, cfg_key in (("b", "arm_b"), ("c", "slice_c")):
            strict = find_d_crit(ROOT / profile, arm=arm, resolution_s=CONTROL_DT_S,
                                 bracket=tuple(config["d_crit"][cfg_key]["bracket"]),
                                 failed_to_fly_rule=True, **kwargs)
            pid = strict["profile"]
            old = committed["profiles"][pid]["arm_b"]["p0"] if arm == "b" else committed["profiles"][pid]["slice_c"]
            out.append({
                "profile": pid,
                "arm": arm,
                "old_rule_stable_delay_s": old["stable_delay_s"],
                "old_rule_unstable_delay_s": old["unstable_delay_s"],
                "d088_stable_delay_s": strict["stable_delay_s"],
                "d088_failed_delay_s": strict["unstable_delay_s"],
                "d088_stable_side": strict["stable_side"],
                "d088_failed_side": strict["unstable_side"],
            })
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="runs/d088-exp021-diagnostic")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "tools" / "a_control_tolerance_config.json").read_text(encoding="utf-8"))
    committed = json.loads((EXP021 / "d_crit.json").read_text(encoding="utf-8"))
    with (EXP021 / "per_run.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = {
        "schema": "zephyr-d088-exp021-diagnostic-1",
        "label": LABEL,
        "rule_set": FAILED_TO_FLY_RULES,
        "per_run_partial": _partial_reclassify(rows),
        "d_crit_loss0_seed0_20s": _dcrit_compare(config, committed),
    }
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", schema="zephyr-d088-exp021-diagnostic-manifest-1", argv=sys.argv[1:] if argv is None else argv,
                   config={"tilt_limit_deg": TILT_LIMIT_DEG, "min_height_m": MIN_HEIGHT_M, "source": "runs/a6-control-tolerance"},
                   seeds=[0], limits=["diagnostic only, does not replace EXP-021", "simulated only",
                                      "literature-derived profiles", "per-run part omits y and z growth"],
                   outputs={"diagnostic.json": out / "diagnostic.json"}, output_dir=out,
                   replay_command=f"PYTHONPATH=. python tools/d088_exp021_diagnostic.py --output-dir {args.output_dir}",
                   runtime_s=time.perf_counter() - started, n_processes=1, repo=ROOT)


if __name__ == "__main__":
    main()
