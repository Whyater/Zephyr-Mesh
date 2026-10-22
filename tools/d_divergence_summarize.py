"""Summarize the saved per-run NEES windows from d_divergence_diagnostic (proposal, diagnostic only)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

from sim.estimation.divergence import DivergenceRule, check_window_means


def main(path: str) -> dict:
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    rule = DivergenceRule()
    out = collections.defaultdict(collections.Counter)
    single_windows = []
    for r in runs:
        res = check_window_means(r["window_means"], rule)
        blown = r["max_m"] > 1.0
        key = f'{r["arm"]}|N={r["n"]}'
        c = out[key]
        c["runs"] += 1
        c["old_flag"] += r["old_flag"]
        c["registered_fail"] += r["failed"]
        c["max_over_1m"] += blown
        c["inconsistent"] += res.inconsistent
        c["diverged"] += res.diverged
        c["diverged_but_max_under_1m"] += res.diverged and not blown
        c["max_over_1m_not_diverged"] += blown and not res.diverged
        c["max_over_1m_all_finite_old_flag_false"] += blown and r["metrics_finite"] and not r["old_flag"]
        if r["arm"] == "single" and r["noise"] == "0.05":
            single_windows.extend(r["window_means"])
    w = np.asarray(single_windows, dtype=float)
    report = {
        "runs": len(runs),
        "by_arm_and_n": {k: dict(v) for k, v in sorted(out.items())},
        "healthy_single_noise_0.05": {
            "windows": int(w.size),
            "mean_window_nees": float(np.mean(w)),
            "frac_above_chi2_threshold": float(np.mean(w > rule.threshold)),
            "chi2_predicted_if_independent": rule.window_false_alarm,
            "frac_above_gross": float(np.mean(w > rule.gross_nees)),
        },
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main(sys.argv[1])
