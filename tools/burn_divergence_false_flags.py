"""False-flag and detection study for the windowed NEES divergence check.

All results come from the simulator; nothing here is measured hardware data.

Question 1: when the filter's models match the simulated world, how often does the ``diverged`` flag
(3 windows in a row of 10-query mean NEES above 100) or the ``inconsistent``
flag (same test at the chi-square 0.1 % level) fire?
Question 2: when the filter is told its sensor is much better than it is, how
often does each flag fire, and does the old error rule notice?

Strata
- matched: target ``cv`` (a straight line, which the constant-velocity
  filter models exactly), sensor noise 0.05 m equal to the assumed 0.05 m.
  This filter is model-matched but mildly conservative, not exactly
  calibrated: its process noise allows for turns the straight target never
  makes, so its median mean NEES is about 2, not the 3 of an exactly
  calibrated 3-axis filter. The false-flag bound applies to this filter.
  Arms ``single`` (1 observer) and ``central`` (2 and 3 observers), link loss
  0.0 and 0.3, link delay 0.05 s (the same link as tools/d_run_fusion.py).
- mismodelled: targets ``ct`` and ``jink``, arms ``single``, ``central`` and
  ``peer_ci`` (3 observers), noise 0.05 m, loss 0. The filter is honest about
  the sensor but its motion model is wrong for a turning target.
- overconfident: target ``cv``, ``single``, noise 0.05 m, filter measurement
  std ``r_std_m`` in (0.002, 0.005, 0.01, 0.02) m, so the filter believes its
  sensor is 25x, 10x, 5x or 2.5x better than it is.

Seeds: stratum number k (0, 1, 2, ... in ``build_strata`` order) uses seeds
k * SEED_BLOCK + 1 .. k * SEED_BLOCK + N, so no two strata share a seed and the
pooled matched count is a count of separately seeded runs. Each run is done once with no
rule (the old behaviour, giving the old ``failed`` value) and the windowed
check is then applied to that same outcome with ``apply_divergence_rule``,
which is exactly what ``run_seed(..., divergence_rule=...)`` does.

Output JSON is deterministic for identical arguments (sorted keys, no
timestamps). CPU seconds are printed, not stored.

Usage:
    TMPDIR=/tmp PYTHONPATH=. python tools/burn_divergence_false_flags.py \
        --seeds 500 --out /tmp/false_flags.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

from sim.coop.net import LinkSpec
from sim.coop.observers import CoopCell, apply_divergence_rule, run_seed
from sim.estimation.divergence import DivergenceRule
from sim.s6_recovery import wilson_interval

# The 97.5 % normal quantile, 1.959964 to six places. The standard library gives
# it independently of scipy, so the k = 0 check does not reuse wilson_interval.
Z95 = NormalDist().inv_cdf(0.975)
DEFAULT_SEEDS = 500
DEFAULT_MISMODEL_SEEDS = 100
OVERCONFIDENT_R_STD_M = (0.002, 0.005, 0.01, 0.02)
TRUE_NOISE_M = 0.05


def wilson_zero_upper(n: int, z: float = Z95) -> float:
    """Closed-form Wilson 95 % upper bound when k = 0: z^2 / (n + z^2)."""
    return z * z / (n + z * z)


def rule_of_three(n: int) -> float:
    """Plain-language upper bound for k = 0 events in n trials: 3 / n."""
    return 3.0 / n


def _stratum(name, group, target, arm, n_obs, loss, r_std=None):
    return {"name": name, "group": group, "target": target, "arm": arm,
            "n_observers": n_obs, "loss": loss, "r_std_m": r_std}


def build_strata() -> list[dict]:
    strata = []
    for loss in (0.0, 0.3):
        strata.append(_stratum(f"matched cv single N=1 loss={loss}", "matched",
                               "cv", "single", 1, loss))
        for n_obs in (2, 3):
            strata.append(_stratum(f"matched cv central N={n_obs} loss={loss}",
                                   "matched", "cv", "central", n_obs, loss))
    for target in ("ct", "jink"):
        for arm in ("single", "central", "peer_ci"):
            n_obs = 1 if arm == "single" else 3
            strata.append(_stratum(f"mismodelled {target} {arm} N={n_obs}", "mismodelled",
                                   target, arm, n_obs, 0.0))
    for r_std in OVERCONFIDENT_R_STD_M:
        strata.append(_stratum(f"overconfident cv single r_std={r_std}", "overconfident",
                               "cv", "single", 1, 0.0, r_std))
    return strata


def _cell(stratum: dict) -> CoopCell:
    return CoopCell(
        target=stratum["target"], n_observers=stratum["n_observers"],
        noise_std_m=TRUE_NOISE_M, r_std_m=stratum["r_std_m"],
        link=LinkSpec(mode="abstract", delay_s=0.05,
                      loss_probability=float(stratum["loss"])))


SEED_BLOCK = 10_000


def stratum_seeds(index: int, n_seeds: int) -> range:
    """Seeds for stratum ``index``: a block of SEED_BLOCK that no other stratum uses."""
    if not 1 <= int(n_seeds) < SEED_BLOCK:
        raise ValueError(f"n_seeds must be between 1 and {SEED_BLOCK - 1}")
    start = int(index) * SEED_BLOCK + 1
    return range(start, start + int(n_seeds))


def run_stratum(stratum: dict, n_seeds: int, rule: DivergenceRule | None = None,
                index: int = 0) -> dict:
    """Run this stratum's seed block and return per-seed flag lists."""
    rule = rule if rule is not None else DivergenceRule()
    cell = _cell(stratum)
    arm = stratum["arm"]
    rows = []
    for seed in stratum_seeds(index, n_seeds):
        old = run_seed(cell, seed, arms=(arm,))[arm]
        new = apply_divergence_rule(old, rule)
        check = new.extras["nees_check"]
        rows.append({
            "seed": seed,
            "diverged": bool(check["diverged"]),
            "inconsistent": bool(check["inconsistent"]),
            "old_failed": bool(old.failed),
            "new_failed": bool(new.failed),
            "mean_nees": None if old.mean_nees is None or not np.isfinite(old.mean_nees)
            else float(old.mean_nees),
        })
    return summarize(stratum, rows)


def _rate(k: int, n: int) -> dict:
    low, high = wilson_interval(k, n)
    return {"k": int(k), "rate": k / n, "wilson95_low": low, "wilson95_high": high}


def summarize(stratum: dict, rows: list[dict]) -> dict:
    n = len(rows)
    nees = [r["mean_nees"] for r in rows if r["mean_nees"] is not None]
    old_pass_new_fail = [r["seed"] for r in rows if r["diverged"] and not r["old_failed"]]
    return {
        **stratum,
        "n": n,
        "diverged": _rate(sum(r["diverged"] for r in rows), n),
        "inconsistent": _rate(sum(r["inconsistent"] for r in rows), n),
        "old_rule_failed": _rate(sum(r["old_failed"] for r in rows), n),
        "new_rule_failed": _rate(sum(r["new_failed"] for r in rows), n),
        "old_pass_but_diverged_count": len(old_pass_new_fail),
        "old_pass_but_diverged_seeds": old_pass_new_fail[:20],
        "diverged_seeds": [r["seed"] for r in rows if r["diverged"]][:20],
        "median_mean_nees": float(np.median(nees)) if nees else None,
        "_mean_nees_values": nees,
    }


def pool(rows: list[dict]) -> dict:
    """Pool the counts of several strata (each stratum has its own seed block)."""
    n = sum(r["n"] for r in rows)
    out = {"n": n, "strata": len(rows)}
    for key in ("diverged", "inconsistent", "old_rule_failed"):
        out[key] = _rate(sum(r[key]["k"] for r in rows), n)
    out["diverged_zero_check"] = zero_check(out["diverged"]["k"], n)
    out["inconsistent_zero_check"] = zero_check(out["inconsistent"]["k"], n)
    return out


def zero_check(k: int, n: int) -> dict:
    """Compare wilson_interval to the closed form and rule of three when k = 0."""
    result = {"rule_of_three_upper": rule_of_three(n)}
    if k == 0:
        closed = wilson_zero_upper(n)
        library = wilson_interval(0, n)[1]
        assert abs(closed - library) < 1e-12, (closed, library)
        result.update({"closed_form_upper": closed, "wilson_interval_upper": library,
                       "difference": abs(closed - library)})
    else:
        result["note"] = "k > 0, closed form for k = 0 not applicable"
    return result


def run_study(n_seeds: int, n_mismodel_seeds: int) -> dict:
    results = []
    for index, stratum in enumerate(build_strata()):
        seeds = n_mismodel_seeds if stratum["group"] == "mismodelled" else n_seeds
        results.append(run_stratum(stratum, seeds, index=index))
    matched = [r for r in results if r["group"] == "matched"]
    pooled_nees = sorted(v for r in matched for v in r.pop("_mean_nees_values"))
    for r in results:
        r.pop("_mean_nees_values", None)
    return {
        "note": "All results are simulated. Stratum k uses seeds k * 10000 + 1 .. k * 10000 + n.",
        "seeds": int(n_seeds),
        "mismodel_seeds": int(n_mismodel_seeds),
        "rule": {"window": 10, "consecutive": 3, "gross_nees": 100.0,
                 "chi2_window_threshold": DivergenceRule().threshold},
        "matched_pool": {**pool(matched),
                         "median_mean_nees": float(np.median(pooled_nees)) if pooled_nees else None},
        "strata": results,
    }


def _fmt(rate: dict) -> str:
    return f"{rate['k']:>4d} [{rate['wilson95_low']:.4f},{rate['wilson95_high']:.4f}]"


def print_table(study: dict) -> None:
    head = (f"{'stratum':<42}{'n':>5}  {'diverged k [95% Wilson]':<24}"
            f"{'inconsistent k [95% Wilson]':<30}{'old fail k':>10}{'old ok/div':>11}"
            f"{'med NEES':>10}")
    print(head)
    for r in study["strata"]:
        med = "n/a" if r["median_mean_nees"] is None else f"{r['median_mean_nees']:.2f}"
        print(f"{r['name']:<42}{r['n']:>5}  {_fmt(r['diverged']):<24}"
              f"{_fmt(r['inconsistent']):<30}{r['old_rule_failed']['k']:>10d}"
              f"{r['old_pass_but_diverged_count']:>11d}{med:>10}")
    p = study["matched_pool"]
    med = "n/a" if p["median_mean_nees"] is None else f"{p['median_mean_nees']:.2f}"
    print(f"{'MATCHED POOL (mildly conservative)':<42}{p['n']:>5}  {_fmt(p['diverged']):<24}"
          f"{_fmt(p['inconsistent']):<30}{p['old_rule_failed']['k']:>10d}{'':>11}{med:>10}")
    z = p["diverged_zero_check"]
    print(f"pool diverged k={p['diverged']['k']}: rule of three upper = {z['rule_of_three_upper']:.6f}"
          + (f", closed form {z['closed_form_upper']:.6f}, wilson_interval "
             f"{z['wilson_interval_upper']:.6f}, diff {z['difference']:.2e}"
             if "closed_form_upper" in z else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="seeds per matched and overconfident stratum")
    parser.add_argument("--mismodel-seeds", type=int, default=DEFAULT_MISMODEL_SEEDS,
                        help="seeds per mismodelled stratum")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.mismodel_seeds < 1:
        parser.error("seed counts must be at least 1")
    started = time.process_time()
    study = run_study(args.seeds, args.mismodel_seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
    print_table(study)
    print(f"CPU seconds: {time.process_time() - started:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
