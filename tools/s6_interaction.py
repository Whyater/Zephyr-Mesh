"""Balanced ANOVA-style variance decomposition for the S6 main slice."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

FACTORS = ("motion", "delay_s", "loss_probability", "noise_std_m")
PAIR_TERMS = tuple((a, b) for i, a in enumerate(FACTORS) for b in FACTORS[i + 1 :])
DEFAULT_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 20261024
RESIDUAL_DEFINITION = "Residual combines within-cell seed variance plus all 3-way and 4-way interactions."


def _factor_values(rows: Sequence[dict[str, str]], factor: str) -> list[str]:
    return sorted({row[factor] for row in rows}, key=lambda value: float(value) if factor != "motion" else value)


def _numeric(rows: Sequence[dict[str, str]], metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{metric} must contain finite positive values")
    return values


def _tensor(rows: Sequence[dict[str, str]], metric: str) -> tuple[np.ndarray, list[list[str]], int]:
    if not rows:
        raise ValueError("no rows")
    levels = [_factor_values(rows, factor) for factor in FACTORS]
    indices = [{value: index for index, value in enumerate(values)} for values in levels]
    seeds = sorted({row["seed"] for row in rows})
    seed_index = {value: index for index, value in enumerate(seeds)}
    shape = tuple(len(values) for values in levels) + (len(seeds),)
    data = np.full(shape, np.nan, dtype=float)
    for row, value in zip(rows, _numeric(rows, metric)):
        key = tuple(indices[i][row[factor]] for i, factor in enumerate(FACTORS)) + (seed_index[row["seed"]],)
        if not np.isnan(data[key]):
            raise ValueError(f"duplicate row for factor/seed combination: {key}")
        data[key] = value
    if np.isnan(data).any():
        raise ValueError("main slice is not balanced across factors and seeds")
    return data, levels, len(seeds)


def _project_ss(data: np.ndarray, axes: tuple[int, ...]) -> float:
    """SS of an orthogonal balanced ANOVA projection, including replicate axis."""
    marginal = data.mean(axis=tuple(i for i in range(data.ndim) if i not in axes))
    for axis in range(len(axes)):
        marginal = marginal - marginal.mean(axis=axis, keepdims=True)
    n_total = data.size
    multiplier = n_total / marginal.size
    return float(multiplier * np.sum(marginal * marginal))


def decompose(data: np.ndarray) -> dict[str, float]:
    """Return normalized main, two-way, and residual SS shares.

    The last axis is replicate/seed. Terms are orthogonal projections, so the
    residual is the remaining sum of squares and includes higher-order terms
    and within-cell seed variation.
    """
    if data.ndim != 5 or min(data.shape) < 1:
        raise ValueError("data must have four factor axes and one seed axis")
    centered = data - data.mean()
    total = float(np.sum(centered * centered))
    if total == 0:
        names = [f"main:{factor}" for factor in FACTORS] + [f"interaction:{a}*{b}" for a, b in PAIR_TERMS]
        return {name: 0.0 for name in names} | {"residual": 1.0}
    ss: dict[str, float] = {}
    for index, factor in enumerate(FACTORS):
        ss[f"main:{factor}"] = _project_ss(data, (index,))
    for a, b in PAIR_TERMS:
        ss[f"interaction:{a}*{b}"] = _project_ss(data, (FACTORS.index(a), FACTORS.index(b)))
    explained = sum(ss.values())
    ss["residual"] = max(0.0, total - explained)
    shares = {name: value / total for name, value in ss.items()}
    # Floating point roundoff is corrected only in the residual, preserving all
    # projected terms and making the reported decomposition exactly normalized.
    shares["residual"] += 1.0 - sum(shares.values())
    return shares


def _bootstrap(data: np.ndarray, samples: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n_seeds = data.shape[-1]
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        draws.append(decompose(data[..., indices]))
    names = list(draws[0])
    return {name: (float(np.quantile([draw[name] for draw in draws], 0.025)),
                   float(np.quantile([draw[name] for draw in draws], 0.975))) for name in names}


def analyze_rows(rows: Sequence[dict[str, str]], *, bootstrap_samples: int = DEFAULT_BOOTSTRAP,
                 bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, object]:
    if any(row.get("slice") != "main" for row in rows):
        raise ValueError("input must contain only slice=main rows")
    if {row.get("arm") for row in rows} != {"naive", "recovery"}:
        raise ValueError("input must contain both naive and recovery arms")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    result: dict[str, object] = {"schema": "zephyr-s6-interaction-1", "residual_definition": RESIDUAL_DEFINITION,
                                  "factors": list(FACTORS),
                                  "pair_terms": [list(pair) for pair in PAIR_TERMS],
                                  "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
                                  "arms": {}}
    for arm_index, arm in enumerate(("naive", "recovery")):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_result: dict[str, object] = {}
        for metric, transform in (("p95_error_m", "p95"), ("p95_error_m", "log_p95")):
            values, levels, n_seeds = _tensor(arm_rows, metric)
            if transform == "log_p95":
                values = np.log(values)
            shares = decompose(values)
            intervals = _bootstrap(values, bootstrap_samples, bootstrap_seed + arm_index * 100003 + (1 if transform == "log_p95" else 0))
            arm_result[transform] = {"shares": shares, "bootstrap_95": {name: list(interval) for name, interval in intervals.items()},
                                     "levels": {factor: levels[i] for i, factor in enumerate(FACTORS)}, "n_seeds": n_seeds,
                                     "n_rows": len(arm_rows)}
        result["arms"][arm] = arm_result
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    rows = [row for row in load_csv(args.input) if row.get("slice") == "main"]
    result = analyze_rows(rows, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    result["input"] = {"path": str(args.input), "sha256": sha256(args.input), "rows": len(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    manifest = {
        "schema": "zephyr-s6-interaction-manifest-1",
        "input": {"path": str(args.input), "sha256": result["input"]["sha256"], "rows": len(rows)},
        "output": {"analysis.json": sha256(args.output)},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
