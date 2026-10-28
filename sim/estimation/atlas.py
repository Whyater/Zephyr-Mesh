"""Build and validate the S6 recovery atlas.

The atlas is a summary of simulated link and simulated sensor results. It does
not contain measurements from hardware. Use ``build_atlas(summary,
counterfactuals)`` after an atlas sweep, then ``validate_atlas(document)``.

The builder is strict on purpose. A missing fail-rate number, a missing
counterfactual record for a failing series, or a counterfactual record that
belongs to a different delay raises ``ValueError`` instead of turning into a
quiet default. Counterfactual records are found only by the exact key the
runner writes, ``"stratum|variant|loss_model:probability|arm"``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import validate


_SCHEMA_PATH = Path(__file__).with_name("atlas_schema.json")
FAIL_RATE_THRESHOLD = 0.5
# Fixed tie order for the single-removal labels.
_REMOVALS = ("delay", "loss", "noise")
_LABELS = {"delay": "lag-dominated", "loss": "gap-dominated", "noise": "noise-dominated"}
_ARM_FIELDS = ("name", "model", "stamping", "late_policy", "retro_window_s", "gate_probability",
               "gate_reset_after", "adaptive", "r_std_m", "q_factor", "model_options")
_STRATUM_FIELDS = ("rate_hz", "noise_model", "noise_std_m", "noise_dof", "jitter_rule")
_DESIGN_FIELDS = ("name", "reason", "burst_mean_s", "imm_strata", "speed_variants_central_only",
                  "arms_by_stratum", "options")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _need(mapping: Any, name: str, where: str) -> Any:
    if not isinstance(mapping, dict) or name not in mapping:
        raise ValueError(f"{where} is missing required field {name!r}")
    return mapping[name]


def _need_number(mapping: Any, name: str, where: str) -> float:
    value = _need(mapping, name, where)
    if not _finite(value):
        raise ValueError(f"{where} field {name!r} must be a finite number, got {value!r}")
    return float(value)


def loss_label(model: str, probability: float) -> str:
    """The loss condition as the runner writes it, for example ``independent:0.1``."""
    return f"{model}:{float(probability):g}"


def counterfactual_key(stratum: str, variant: str, loss: str, arm: str) -> str:
    """The exact key of one record in the runner's counterfactuals.json."""
    return "|".join((stratum, variant, loss, arm))


def _removal_changes_cell(name: str, cell: dict) -> bool:
    """Whether a single-removal counterfactual changes this cell (same rule as the runner)."""
    if name == "delay":
        return (_need_number(cell, "delay_s", "cell") != 0.0
                or _need_number(cell, "jitter_s", "cell") != 0.0)
    if name == "loss":
        return _need_number(cell, "loss_probability", "cell") != 0.0
    if name == "noise":
        return _need_number(cell, "noise_std_m", "cell") != 0.0
    raise ValueError(f"unknown removal {name!r}")


def label_failure(original: float, medians: dict[str, float | None], cell: dict) -> str:
    """Label a failing cell with the median-p95 rule.

    ``medians`` holds the counterfactual median p95 for ``delay``, ``loss``,
    ``noise`` and ``all``; a removal that does not change the cell is None.
    D_x = original - median_x is the drop in p95 from removing x, for the
    removals that apply. The label is the removal with the largest D_x (ties:
    delay, loss, noise), unless the p95 still left with everything removed
    (``medians["all"]``) is at least as large as that largest drop. Then even a
    perfect link and sensor would leave more error than any one cause explains,
    and the label is "model-mismatch". When no removal applies (the cell is
    already clean in delay, loss and noise) and it still fails, that is also
    "model-mismatch".
    """
    all_removed = medians.get("all")
    if not _finite(all_removed):
        raise ValueError("counterfactual record has no finite all-removed median")
    drops: dict[str, float] = {}
    for name in _REMOVALS:
        if not _removal_changes_cell(name, cell):
            continue
        value = medians.get(name)
        if not _finite(value):
            raise ValueError(f"counterfactual record has no finite {name}-removed median "
                             f"although removing {name} changes the cell")
        drops[name] = float(original) - float(value)
    if not drops:
        return "model-mismatch"
    best = max(drops.values())
    if float(all_removed) >= best:
        return "model-mismatch"
    winner = max(drops, key=lambda name: (drops[name], -_REMOVALS.index(name)))
    return _LABELS[winner]


def _counterfactual_for(counterfactuals: Any, key: str, delay_s: float, original: float,
                        cell: dict) -> tuple[dict[str, float | None], str | None]:
    if not isinstance(counterfactuals, dict) or not isinstance(counterfactuals.get(key), dict):
        raise ValueError(f"no counterfactual record under the exact key {key!r} for a failing series")
    record = counterfactuals[key]
    if _need_number(record, "delay_s", f"counterfactual {key!r}") != float(delay_s):
        raise ValueError(f"counterfactual {key!r} was made at delay {record['delay_s']!r}, "
                         f"but the series first fails at {delay_s!r}")
    recorded_original = _need_number(record, "original_median_p95_m", f"counterfactual {key!r}")
    if not math.isclose(recorded_original, original, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"counterfactual {key!r} original median {recorded_original!r} "
                         f"does not match the cell median {original!r}")
    raw = _need(record, "medians", f"counterfactual {key!r}")
    medians: dict[str, float | None] = {}
    for name in (*_REMOVALS, "all"):
        value = _need(raw, f"{name}_removed", f"counterfactual {key!r} medians")
        if value is not None and not _finite(value):
            raise ValueError(f"counterfactual {key!r} {name}_removed median must be finite or null")
        medians[name] = None if value is None else float(value)
    return medians, record.get("failure_mode")


def _improvement(row: dict, arm: str, delay: float) -> dict | None:
    if arm == "naive":
        return None
    paired = _need(row, "paired_vs_naive", "cell summary")
    value = paired.get(arm) if isinstance(paired, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"cell summary has no paired_vs_naive entry for arm {arm!r}")
    where = f"paired_vs_naive[{arm!r}]"
    return {
        "delay_s": float(delay),
        "median_p95_diff_m": _need_number(value, "p95_diff_median_m", where),
        "interval_m": [_need_number(value, "p95_diff_ci_low_m", where),
                       _need_number(value, "p95_diff_ci_high_m", where)],
    }


def _censor_label(max_delay: float) -> str:
    return f">{max_delay:g}"


def _interval_end(hit: int | None, delays: list[float]) -> float | str:
    """One end of the tolerated-delay interval.

    hit is the first index whose Wilson bound reaches 0.5. No hit means the
    end is above every tested delay (">0.3"). A hit at index 0 means even 0 s
    may not be tolerated ("<0"); it is never stored as the number 0.0, which
    is a real answer ("0 s is tolerated, 0.025 s is not").
    """
    if hit is None:
        return _censor_label(delays[-1])
    if hit == 0:
        return "<0"
    return float(delays[hit - 1])


def _tolerated(delays: list[float], rates: list[float], low: list[float],
               high: list[float]) -> tuple[dict[str, Any], int | None]:
    failing = next((i for i, rate in enumerate(rates) if rate >= FAIL_RATE_THRESHOLD), None)
    upper_hit = next((i for i, value in enumerate(high) if value >= FAIL_RATE_THRESHOLD), None)
    lower_hit = next((i for i, value in enumerate(low) if value >= FAIL_RATE_THRESHOLD), None)
    result: dict[str, Any] = {"value": None, "censored": None, "fails_at_zero": False,
                              "interval_s": [_interval_end(upper_hit, delays),
                                             _interval_end(lower_hit, delays)]}
    if failing is None:
        result["censored"] = _censor_label(delays[-1])
    elif failing == 0:
        result["fails_at_zero"] = True
    else:
        result["value"] = float(delays[failing - 1])
    return result, failing


def _failure_rule(summary: dict) -> dict:
    rule = _need(summary, "failure_rule", "summary")
    per_seed = _need(rule, "per_seed", "failure_rule")
    window = _need(per_seed, "scored_window_s", "failure_rule.per_seed")
    return {
        "per_seed": {
            "p95_error_m": _need_number(per_seed, "p95_error_m", "failure_rule.per_seed"),
            "max_error_m": _need_number(per_seed, "max_error_m", "failure_rule.per_seed"),
            "scored_window_s": [float(value) for value in window],
        },
        "cell_rule": _need(rule, "cell_rule", "failure_rule"),
        "fail_rate_threshold": _need_number(rule, "fail_rate_threshold", "failure_rule"),
        "cell_rule_status": _need(rule, "cell_rule_status", "failure_rule"),
    }


def _design(summary: dict, strata: dict) -> dict:
    design = _need(summary, "design", "summary")
    out = {name: _need(design, name, "design") for name in _DESIGN_FIELDS}
    if set(out["arms_by_stratum"]) != set(strata):
        raise ValueError("design.arms_by_stratum and strata must name the same strata")
    return out


def _strata(summary: dict) -> dict:
    strata = _need(summary, "strata", "summary")
    if not isinstance(strata, dict) or not strata:
        raise ValueError("summary strata must be a non-empty mapping of stratum definitions")
    return {name: {field: _need(definition, field, f"stratum {name!r}") for field in _STRATUM_FIELDS}
            for name, definition in strata.items()}


def build_atlas(summary: dict, counterfactuals: dict) -> dict:
    """Convert cell summaries and counterfactuals into a validated atlas."""
    if not isinstance(summary, dict) or not isinstance(summary.get("cells"), list):
        raise ValueError("summary must contain a cells list")
    strata = _strata(summary)
    design = _design(summary, strata)
    grouped: dict[tuple[str, str, str, str], list[tuple[float, dict, dict]]] = {}
    for row in summary["cells"]:
        cell = _need(row, "cell", "cell summary")
        stratum = str(_need(row, "stratum", "cell summary"))
        variant = str(_need(row, "variant", "cell summary"))
        loss = loss_label(str(_need(cell, "loss_model", "cell")),
                          _need_number(cell, "loss_probability", "cell"))
        delay = _need_number(cell, "delay_s", "cell")
        arms = _need(row, "arms", "cell summary")
        for arm, arm_summary in arms.items():
            grouped.setdefault((stratum, variant, loss, str(arm)), []).append((delay, row, arm_summary))

    entries = []
    for key in sorted(grouped):
        stratum, variant, loss, arm = key
        series = sorted(grouped[key], key=lambda item: item[0])
        delays = [item[0] for item in series]
        if len(set(delays)) != len(delays):
            raise ValueError(f"series {'|'.join(key)} has a repeated delay")
        where = f"series {'|'.join(key)}"
        rates = [_need_number(item[2], "fail_rate", where) for item in series]
        low = [_need_number(item[2], "fail_rate_ci_low", where) for item in series]
        high = [_need_number(item[2], "fail_rate_ci_high", where) for item in series]
        tolerated, failing = _tolerated(delays, rates, low, high)
        model, _, probability = loss.partition(":")
        entry: dict[str, Any] = {
            "stratum": stratum, "variant": variant, "loss_condition": loss,
            "loss_model": model, "loss_probability": float(probability), "arm": arm,
            "max_tolerated_delay_s": tolerated, "failure_mode": None,
            "improvement_vs_naive": [],
        }
        if failing is not None:
            failure_delay, failure_row, failure_summary = series[failing]
            original = _need_number(failure_summary, "p95_median_m", where)
            medians, runner_label = _counterfactual_for(
                counterfactuals, counterfactual_key(*key), failure_delay, original, failure_row["cell"])
            label = label_failure(original, medians, failure_row["cell"])
            if runner_label is not None and runner_label != label:
                raise ValueError(f"atlas label {label!r} disagrees with counterfactual record "
                                 f"label {runner_label!r} for {'|'.join(key)}")
            entry["failure_mode"] = label
            entry["counterfactual_medians"] = medians
        for delay, row, _ in series:
            item = _improvement(row, arm, delay)
            if item is not None:
                entry["improvement_vs_naive"].append(item)
        entries.append(entry)

    limits = [str(limit) for limit in _need(summary, "limits", "summary")]
    for needle, text in (
        ("max-error rule", "A cell can fail on the max-error rule even when its median p95 does not exceed the p95 threshold."),
        ("model-mismatch", "A failing cell where no removal applies (delay, loss and noise already 0) is labelled model-mismatch."),
        ("<0", "An interval end of \"<0\" means the Wilson bound already reaches 0.5 at delay 0, so even 0 s may not be tolerated."),
    ):
        if not any(needle in limit for limit in limits):
            limits.append(text)
    doc = {
        "schema": "1.0",
        "experiment": "s6-atlas",
        "simulated": True,
        "failure_rule": _failure_rule(summary),
        "design": design,
        "design_reason": _need(summary, "design_reason", "summary"),
        "seeds": [int(seed) for seed in _need(summary, "seeds", "summary")],
        "arms": [{field: _need(arm, field, "arm") for field in _ARM_FIELDS}
                 for arm in _need(summary, "arms", "summary")],
        "strata": strata,
        "limits": limits,
        "entries": entries,
    }
    validate_atlas(doc)
    return doc


def validate_atlas(doc: dict) -> bool:
    """Validate an atlas document against the strict bundled JSON schema."""
    if not isinstance(doc, dict):
        raise ValueError("atlas document must be a dictionary")
    with _SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    validate(instance=doc, schema=schema)
    return True
