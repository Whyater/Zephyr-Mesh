"""EXP-031 area sweep with lossy coverage sharing.

Usage (repo root):
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.venv/bin/python tools/e_exp031_sweep.py --output runs/e-exp031-sweep --pilot --processes 1

All scenarios and figures are simulated.  The mission models three volunteer
search drones sweeping a field and sharing coverage over synthetic links.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.failsafe import FailsafeConfig
from sim.missions.engine import COMMON_LIMITS
from sim.missions.presets import get_preset
from sim.missions.sweep import SweepMission
from tests.reference_e.hand_checks import coverage_time_s
from tools.c_common import prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    apply_mission_style,
    group_rows,
    make_job,
    median_summary,
    processes,
    rate_summary,
    replay_command,
    run_jobs,
    seeds_for,
    series_colors,
    write_rows,
)


EXP_NUMBER = 31
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
PHYSICS_PRESETS = ("clean", "iid_30")
MESSAGE_MODES = ("delta", "cumulative")
DROPOUTS = ("none", "one")
DURATION_S = 120.0
PILOT_DURATION_S = 60.0  # long enough for clean runs to finish (hand check 35 to 42 s)
CALM_POLICY = {"degraded_after_s": 0.25, "land_after_s": 1.0}
EXPLORATORY_LABEL = "exploratory, added after the pilot, not pre-registered"
PREDICTION = (
    "without dropout, loss changes nothing (each drone sweeps its own strip). With a dropout and delta "
    "messages, loss makes survivors believe covered cells are uncovered, so duplication rises roughly with "
    "the loss rate (iid_30: about 30 % of the dead drone's reported cells unknown, duplicated); cumulative "
    "messages remove almost all of it (any single message carries the whole map). Bursts hurt delta most. "
    "Completion time with dropout: the dead drone's remaining two thirds of a strip (about 177 m^2) shared "
    "by two survivors adds about 177/(2 x 4 x 2) = 11 s plus transit, so about 45 to 55 s."
)
PREDICTION_SOURCE = "pre-registered before the runs (EXP-031 section of the private pre-registration)"
PREDICTION_NOTE = (
    'Prediction-before-run check: EXP-031 prediction "without dropout, loss changes nothing" was already '
    "contradicted by the pilot on iid_30 under the default policy (failsafe landings); this is reported as "
    "a miss of the prediction, not reworded."
)
HAND_CHECK_BAND_S = (35.0, 42.0)


def _prediction_sentences(text: str) -> list[str]:
    """The stored prediction split into sentences (a full stop followed by a capital letter)."""
    return re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())


def _band_from_text(text: str) -> tuple[float, float]:
    """The completion band with a dropout, read from the last sentence of the stored prediction."""
    found = re.search(r"about (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) s\.$", _prediction_sentences(text)[-1])
    if found is None:
        raise ValueError("the stored prediction has no dropout completion band")
    return float(found.group(1)), float(found.group(2))


# Completion time with one dropout, as pre-registered in the EXP-031 prediction ("about 45 to 55 s").
DROPOUT_BAND_S = _band_from_text(PREDICTION)
EXPLORATORY_REASON = (
    "the default 0.30 s lost-link rule landed drones on iid_30 and burst presets and the field was never "
    "finished, because a landed drone that keeps sharing is never declared lost"
)
MISSION_NOTE = (
    "Mission changes before any full run: a drone that is landing or landed covers no cells; a drone that "
    "lands on its own failsafe keeps sharing and is never declared lost, so on lossy links P(complete) "
    "measures the failsafe (see the mission limits); duplicated_fraction counts cells seen by two or more "
    "different drones (contract definition)."
)


def _scenario(seed: int, *, preset: str, dropout: str, message_mode: str,
              adapter: str = "point_mass", policy: dict | None = None,
              duration_s: float = DURATION_S) -> dict:
    return {
        "mission": "sweep", "seed": int(seed), "duration_s": float(duration_s),
        "adapter": adapter, "preset": preset, "policy": dict(policy or {}),
        "params": {"message_mode": message_mode, "dropout": dropout},
    }


def _calm_job(seed: int, seed_index: int, *, preset: str, dropout: str, mode: str,
              duration_s: float) -> dict:
    labels = {"slice": EXPLORATORY_LABEL, "preset": preset, "dropout": dropout,
              "message_mode": mode, "adapter": "point_mass", "policy_name": "calm",
              "seed_index": seed_index}
    return make_job(labels, _scenario(seed, preset=preset, dropout=dropout, message_mode=mode,
                                      policy=CALM_POLICY, duration_s=duration_s))


def build_jobs(*, pilot: bool = False) -> list[dict]:
    seeds = seeds_for(EXP_NUMBER, 3 if pilot else 50)
    duration_s = PILOT_DURATION_S if pilot else DURATION_S
    jobs: list[dict] = []
    main_presets = ("clean", "iid_30") if pilot else PRESETS
    for preset in main_presets:
        for dropout in DROPOUTS:
            for mode in MESSAGE_MODES:
                for seed_index, seed in enumerate(seeds):
                    labels = {"slice": "main", "preset": preset, "dropout": dropout,
                              "message_mode": mode, "adapter": "point_mass", "seed_index": seed_index}
                    jobs.append(make_job(labels, _scenario(seed, preset=preset, dropout=dropout,
                                                           message_mode=mode,
                                                           duration_s=duration_s)))
    physics_seeds = seeds[:2] if pilot else seeds[:20]
    physics_cells = (("clean", "one", "delta"),) if pilot else tuple(
        (preset, "one", mode) for preset in PHYSICS_PRESETS for mode in MESSAGE_MODES
    )
    for preset, dropout, mode in physics_cells:
        for seed_index, seed in enumerate(physics_seeds):
            labels = {"slice": "physics", "preset": preset, "dropout": dropout,
                      "message_mode": mode, "adapter": "physics", "seed_index": seed_index}
            jobs.append(make_job(labels, _scenario(seed, preset=preset, dropout=dropout,
                                                   message_mode=mode, adapter="physics",
                                                   duration_s=duration_s)))
    if pilot:
        for seed_index, seed in enumerate(seeds[:2]):
            jobs.append(_calm_job(seed, seed_index, preset="iid_30", dropout="one", mode="delta",
                                  duration_s=duration_s))
    else:
        for preset in ("iid_30", "burst_10_0.5s"):
            for dropout in DROPOUTS:
                for mode in MESSAGE_MODES:
                    for seed_index, seed in enumerate(seeds):
                        jobs.append(_calm_job(seed, seed_index, preset=preset, dropout=dropout,
                                              mode=mode, duration_s=duration_s))
    return jobs



def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n_runs": len(rows),
        "p_t100_reached": rate_summary([row["t100_s"] is not None for row in rows]),
        "t95_s": median_summary([row["t95_s"] for row in rows]),
        "t100_s": median_summary([row["t100_s"] for row in rows]),
        "duplicated_fraction": median_summary([row["duplicated_fraction"] for row in rows]),
        "missed_cells": {"median": median_summary([row["missed_cells"] for row in rows]),
                         "max": max(int(row["missed_cells"]) for row in rows)},
        "replans": {"mean": sum(float(row["replans"]) for row in rows) / len(rows)},
        "belief_error_cells": median_summary([row["belief_error_cells"] for row in rows]),
        "false_loss_declarations": {
            "total": sum(int(row["false_loss_declarations"]) for row in rows),
            "runs_with_any": rate_summary([int(row["false_loss_declarations"]) > 0 for row in rows]),
        },
        "failsafe_landings": {
            "total": sum(int(row["failsafe_landings"]) for row in rows),
            "runs_with_any": rate_summary([int(row["failsafe_landings"]) > 0 for row in rows]),
        },
    }


def _hand_check_verdicts(rows: list[dict]) -> list[dict]:
    """Per message mode: how many clean, no-dropout runs reached t100 and whether the median is in the band."""
    low, high = HAND_CHECK_BAND_S
    verdicts = []
    for mode in MESSAGE_MODES:
        selected = [r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"
                    and r["preset"] == "clean" and r["dropout"] == "none" and r["message_mode"] == mode]
        item = {"message_mode": mode, "reference_sweeping_s": coverage_time_s(800, 3, 4.0, 2.0),
                "predicted_t100_range_s": [low, high], "n_runs": len(selected)}
        if not selected:
            item.update({"n_reached_t100": 0, "observed_t100": None, "verdict": "not run"})
        else:
            observed = median_summary([r["t100_s"] for r in selected])
            median = observed["median"]
            # A cell where no run reached t100 has no median, so it cannot be inside the band.
            inside = median is not None and low <= median <= high
            item.update({"n_reached_t100": observed["n"], "observed_t100": observed,
                         "verdict": "inside" if inside else "outside"})
        verdicts.append(item)
    return verdicts


def _main_rows(rows: list[dict], slice_name: str = "main") -> list[dict]:
    return [r for r in rows if r["slice"] == slice_name and r["adapter"] == "point_mass"]


def _diff(left: float | None, right: float | None) -> float | None:
    """left minus right, rounded to 6 decimals; None when either side is missing."""
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _cell_rows(rows: list[dict], preset: str, dropout: str, mode: str) -> list[dict]:
    return [r for r in rows if r["preset"] == preset and r["dropout"] == dropout and r["message_mode"] == mode]


def _presets_present(rows: list[dict]) -> list[str]:
    return [preset for preset in PRESETS if any(r["preset"] == preset for r in rows)]


def _cell_numbers(grouped: list[dict], fields: tuple[str, ...]) -> dict:
    """Observed numbers for one cell: the requested metrics plus runs with a failsafe landing."""
    item: dict = {"n_runs": len(grouped)}
    for field in fields:
        if field == "p_t100_reached":
            item[field] = rate_summary([r["t100_s"] is not None for r in grouped])
        else:
            item[field] = median_summary([r[field] for r in grouped])
    item["runs_with_failsafe_landing"] = rate_summary([int(r["failsafe_landings"]) > 0 for r in grouped])
    return item


def _cells(rows: list[dict], dropout: str, fields: tuple[str, ...]) -> list[dict]:
    """One entry per preset and message mode that has rows, in grid order."""
    cells = []
    for preset in _presets_present(rows):
        link = get_preset(preset)
        for mode in MESSAGE_MODES:
            grouped = _cell_rows(rows, preset, dropout, mode)
            if not grouped:
                continue
            item = {"preset": preset, "message_mode": mode, "loss_probability": link.loss_probability,
                    "burst_mean_s": link.burst_mean_s}
            item.update(_cell_numbers(grouped, fields))
            cells.append(item)
    return cells


def _observe_no_dropout(rows: list[dict]) -> dict:
    cells = _cells(rows, "none", ("p_t100_reached", "t100_s"))
    clean = {c["message_mode"]: c for c in cells if c["preset"] == "clean"}
    for cell in cells:
        base = clean.get(cell["message_mode"])
        cell["t100_median_minus_clean_s"] = _diff(cell["t100_s"]["median"],
                                                  base["t100_s"]["median"] if base else None)
        cell["p_t100_minus_clean"] = _diff(cell["p_t100_reached"]["rate"],
                                           base["p_t100_reached"]["rate"] if base else None)
    return {"what": "completion within the horizon and t100 per preset and message mode, no dropout",
            "dropout": "none", "cells": cells}


def _observe_dropout_duplication(rows: list[dict]) -> dict:
    cells = _cells(rows, "one", ("duplicated_fraction", "belief_error_cells"))
    by_key = {(c["preset"], c["message_mode"]): c for c in cells}
    differences = []
    for preset in _presets_present(rows):
        delta, cumulative = by_key.get((preset, "delta")), by_key.get((preset, "cumulative"))
        if delta is None or cumulative is None:
            continue
        differences.append({
            "preset": preset,
            "duplicated_fraction_median": _diff(cumulative["duplicated_fraction"]["median"],
                                                delta["duplicated_fraction"]["median"]),
            "belief_error_cells_median": _diff(cumulative["belief_error_cells"]["median"],
                                               delta["belief_error_cells"]["median"]),
        })
    return {"what": "duplicated fraction and belief error cells per preset and message mode, one dropout",
            "dropout": "one", "cells": cells, "cumulative_minus_delta": differences}


def _burst_iid_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """Each burst preset with the independent-loss preset of the same long-run loss probability."""
    present = _presets_present(rows)
    pairs = []
    for burst in present:
        if get_preset(burst).loss_model != "burst":
            continue
        for iid in present:
            link = get_preset(iid)
            if link.loss_model == "independent" and link.loss_probability == get_preset(burst).loss_probability:
                pairs.append((iid, burst))
    return pairs


def _observe_bursts(rows: list[dict]) -> dict:
    fields = ("p_t100_reached", "duplicated_fraction", "belief_error_cells")
    pairs = []
    for iid, burst in _burst_iid_pairs(rows):
        by_mode = []
        for mode in MESSAGE_MODES:
            iid_rows, burst_rows = _cell_rows(rows, iid, "one", mode), _cell_rows(rows, burst, "one", mode)
            if not iid_rows or not burst_rows:
                continue
            left, right = _cell_numbers(iid_rows, fields), _cell_numbers(burst_rows, fields)
            by_mode.append({
                "message_mode": mode, "iid": left, "burst": right,
                "burst_minus_iid": {
                    "duplicated_fraction_median": _diff(right["duplicated_fraction"]["median"],
                                                        left["duplicated_fraction"]["median"]),
                    "belief_error_cells_median": _diff(right["belief_error_cells"]["median"],
                                                       left["belief_error_cells"]["median"]),
                    "p_t100_reached": _diff(right["p_t100_reached"]["rate"], left["p_t100_reached"]["rate"]),
                },
            })
        if by_mode:
            link = get_preset(burst)
            pairs.append({"iid_preset": iid, "burst_preset": burst, "loss_probability": link.loss_probability,
                          "burst_mean_s": link.burst_mean_s, "by_mode": by_mode})
    return {"what": "burst presets against independent-loss presets of the same loss probability, one dropout",
            "dropout": "one", "burst_vs_iid": pairs}


def _observe_dropout_time(rows: list[dict]) -> dict:
    low, high = DROPOUT_BAND_S
    cells = _cells(rows, "one", ("p_t100_reached", "t100_s"))
    for cell in cells:
        median = cell["t100_s"]["median"]
        cell["t100_median_in_band"] = None if median is None else bool(low <= median <= high)
    return {"what": "t100 with one dropout per preset and message mode (median over runs that reached t100)",
            "dropout": "one", "band_s": [low, high], "cells": cells}


# Each stored prediction sentence, matched by its opening words, and the observation it refers to.
SENTENCE_OBSERVERS = (
    ("without dropout", _observe_no_dropout),
    ("With a dropout and delta", _observe_dropout_duplication),
    ("Bursts hurt delta", _observe_bursts),
    ("Completion time with dropout", _observe_dropout_time),
)


def _prediction_observations(rows: list[dict]) -> list[dict]:
    """For each pre-registered prediction sentence, the observed numbers it refers to (no judgement)."""
    default_policy = FailsafeConfig()
    calm_rows = _main_rows(rows, EXPLORATORY_LABEL)
    observations = []
    for sentence in _prediction_sentences(PREDICTION):
        observers = [fn for opening, fn in SENTENCE_OBSERVERS if sentence.startswith(opening)]
        if len(observers) != 1:
            raise ValueError(f"no single observation for prediction sentence: {sentence!r}")
        observed = {"slice": "main", "adapter": "point_mass",
                    "policy": {"name": "default", "degraded_after_s": default_policy.degraded_after_s,
                               "land_after_s": default_policy.land_after_s}}
        observed.update(observers[0](_main_rows(rows)))
        calm = None
        if calm_rows:
            calm = {"slice": EXPLORATORY_LABEL, "adapter": "point_mass",
                    "policy": dict(CALM_POLICY, name="calm")}
            calm.update(observers[0](calm_rows))
        observations.append({"sentence": sentence, "observed": observed, "exploratory_calm_policy": calm})
    return observations


def _dropout_band_check(rows: list[dict]) -> dict:
    """Clean link, one dropout, per message mode: the observed median t100 against the predicted band."""
    low, high = DROPOUT_BAND_S
    items = []
    for mode in MESSAGE_MODES:
        selected = _cell_rows(_main_rows(rows), "clean", "one", mode)
        item = {"message_mode": mode, "n_runs": len(selected)}
        observed = median_summary([r["t100_s"] for r in selected])
        median = observed["median"]
        item.update({"n_reached_t100": observed["n"], "observed_t100": observed,
                     "median_in_band": None if median is None else bool(low <= median <= high)})
        items.append(item)
    return {"predicted_t100_range_s": [low, high],
            "band_source": "last sentence of the stored EXP-031 prediction",
            "observed_clean_one_dropout": items}


def _summarize(rows: list[dict], *, pilot: bool, horizon_s: float) -> dict:
    keys = ("slice", "preset", "dropout", "message_mode", "adapter")
    cells = []
    for coords, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, coords))
        item["metrics"] = _cell_summary(grouped)
        if any("policy_name" in row for row in grouped):
            item["policy_name"] = "calm"
        cells.append(item)
    hand_reference = coverage_time_s(800, 3, 4.0, 2.0)
    hand = _hand_check_verdicts(rows)
    atlas = {}
    for coords, grouped in group_rows([r for r in rows if r["slice"] == "main" and r["adapter"] == "point_mass"],
                                      ("preset", "dropout", "message_mode")).items():
        preset, dropout, mode = coords
        atlas[f"{preset}|{dropout}|{mode}"] = {
            f"wilson_lower_p_t100_within_{horizon_s:g}_s":
                rate_summary([r["t100_s"] is not None for r in grouped])["wilson_low"],
            "median_duplicated_fraction": median_summary([r["duplicated_fraction"] for r in grouped])["median"],
        }
    exploratory = any(r["slice"] == EXPLORATORY_LABEL for r in rows)
    deviations = []
    if pilot:
        deviations.append(f"Pilot uses a {horizon_s:g} s horizon; the preregistered grid horizon is {DURATION_S:g} s.")
    if exploratory:
        calm_seeds = "2 seeds, one cell" if pilot else "both dropouts and message modes, 50 seeds, 400 runs"
        deviations.append(
            "Exploratory slice, not pre-registered (Addendum B, written after the pilots): the calm policy "
            f"(degraded after {CALM_POLICY['degraded_after_s']} s, land after {CALM_POLICY['land_after_s']} s) "
            f"on iid_30 and burst_10_0.5s ({calm_seeds}). Reason: {EXPLORATORY_REASON}. "
            "It is reported separately after the pre-registered grid."
        )
    return {
        "experiment": "EXP-031", "simulated": True, "pilot": bool(pilot),
        "duration_s": horizon_s, "cells": cells, "atlas": atlas,
        "hand_check": {"reference_sweeping_s": hand_reference, "predicted_t100_range_s": list(HAND_CHECK_BAND_S),
                       "observed_clean_no_dropout": hand},
        "prediction": {"text": PREDICTION, "prediction_source": PREDICTION_SOURCE,
                       "prediction_note": PREDICTION_NOTE, "observed_clean_no_dropout": hand,
                       "prediction_observations": _prediction_observations(rows),
                       "dropout_completion_band_check": _dropout_band_check(rows)},
        "exploratory_slice": EXPLORATORY_LABEL if exploratory else None,
        "deviations": deviations,
        "missing_metrics": [],
        "notes": [MISSION_NOTE],
        "limits": list(MISSION_LIMITS) + list(COMMON_LIMITS) + list(SweepMission.limits),
    }


MODE_MARKERS = {"delta": "o", "cumulative": "s"}
FAILSAFE_MARKERS = {"delta": "v", "cumulative": "^"}


def _runs_per_cell_text(rows: list[dict], dropouts: tuple[str, ...]) -> str:
    rows = _main_rows(rows)
    counts = sorted({len(_cell_rows(rows, preset, dropout, mode))
                     for preset in _presets_present(rows) for dropout in dropouts for mode in MESSAGE_MODES
                     if _cell_rows(rows, preset, dropout, mode)})
    if not counts:
        return "no runs"
    if len(counts) == 1:
        return f"{counts[0]} runs per cell"
    return f"{counts[0]} to {counts[-1]} runs per cell"


def _slice_text(rows: list[dict], dropouts: tuple[str, ...]) -> str:
    """Which runs a figure shows, built from the rows and the mission's default failsafe policy."""
    policy = FailsafeConfig()
    shown = " and ".join(dropouts)
    return (f"Slice: main (pre-registered grid), point mass adapter, default policy (hold after "
            f"{policy.degraded_after_s:g} s, land after {policy.land_after_s:g} s), dropout: {shown}, "
            f"{_runs_per_cell_text(rows, dropouts)}.")


def _failsafe_caption(rows: list[dict], dropouts: tuple[str, ...]) -> str:
    """Names the presets where more than half of the shown runs had a lost-link failsafe landing."""
    shown = [r for r in _main_rows(rows) if r["dropout"] in dropouts]
    heavy = []
    for preset in _presets_present(shown):
        summary = rate_summary([int(r["failsafe_landings"]) > 0 for r in shown if r["preset"] == preset])
        if summary["count"] * 2 > summary["n"]:
            heavy.append(f"{preset} ({summary['count']} of {summary['n']} runs)")
    if not heavy:
        return "No preset had a lost-link failsafe landing in most runs."
    return ("Lost-link failsafe landings in most runs: " + ", ".join(heavy)
            + "; on these presets the results mostly reflect those landings, not coverage sharing.")


def _caption(fig, rows: list[dict], dropouts: tuple[str, ...], extra: str) -> None:
    lines = [_slice_text(rows, dropouts), _failsafe_caption(rows, dropouts), extra]
    if any(r["slice"] == EXPLORATORY_LABEL for r in rows):
        lines.append("The exploratory calm policy slice and the physics slice are not shown.")
    elif any(r["slice"] == "physics" for r in rows):
        lines.append("The physics slice is not shown.")
    fig.text(0.01, 0.01, "\n".join(lines), ha="left", va="bottom", fontsize=7, wrap=True)


def _categorical_axis(ax, presets: list[str]) -> None:
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels(presets, fontsize=7)
    ax.set_xlim(-0.6, len(presets) - 0.4)
    ax.set_xlabel("link preset")


def _plot_rate(ax, xs: list[float], summaries: list[dict], **style) -> None:
    rates = [s["rate"] for s in summaries]
    # Interval ends can sit a rounding error past the rate at 0 or 1; error bar lengths cannot be negative.
    below = [max(0.0, s["rate"] - s["wilson_low"]) for s in summaries]
    above = [max(0.0, s["wilson_high"] - s["rate"]) for s in summaries]
    ax.errorbar(xs, rates, yerr=[below, above], linestyle="none", capsize=2, **style)


def _plot_median(ax, xs: list[float], summaries: list[dict], **style) -> None:
    kept = [(x, s) for x, s in zip(xs, summaries) if s["median"] is not None]
    if not kept:
        return
    medians = [s["median"] for _, s in kept]
    below = [max(0.0, s["median"] - s["low"]) for _, s in kept]
    above = [max(0.0, s["high"] - s["median"]) for _, s in kept]
    ax.errorbar([x for x, _ in kept], medians, yerr=[below, above], linestyle="none", capsize=2, **style)


def _completion_figure(rows: list[dict], horizon_s: float):
    pal = apply_mission_style("light")
    colors = dict(zip(MESSAGE_MODES, series_colors(len(MESSAGE_MODES))))
    main = _main_rows(rows)
    presets = _presets_present(main)
    # Four series per preset, each at its own sideways offset so equal values stay visible.
    offsets = {("complete", "delta"): -0.27, ("complete", "cumulative"): -0.09,
               ("failsafe", "delta"): 0.09, ("failsafe", "cumulative"): 0.27}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, dropout in zip(axes, DROPOUTS):
        for mode in MESSAGE_MODES:
            for kind in ("complete", "failsafe"):
                xs, summaries = [], []
                for index, preset in enumerate(presets):
                    grouped = _cell_rows(main, preset, dropout, mode)
                    if not grouped:
                        continue
                    xs.append(index + offsets[(kind, mode)])
                    if kind == "complete":
                        summaries.append(rate_summary([r["t100_s"] is not None for r in grouped]))
                    else:
                        summaries.append(rate_summary([int(r["failsafe_landings"]) > 0 for r in grouped]))
                if not xs:
                    continue
                if kind == "complete":
                    _plot_rate(ax, xs, summaries, marker=MODE_MARKERS[mode], color=colors[mode],
                               label=f"complete within {horizon_s:g} s, {mode}")
                else:
                    _plot_rate(ax, xs, summaries, marker=FAILSAFE_MARKERS[mode], color=pal["caution"],
                               markerfacecolor="none", label=f"runs with a failsafe landing, {mode}")
        _categorical_axis(ax, presets)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"dropout: {dropout}", fontsize=9)
    axes[0].set_ylabel("fraction of runs (95 % Wilson interval)")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.115))
    fig.suptitle(f"Sweep completion within {horizon_s:g} s and failsafe landings, simulated", fontsize=11)
    _caption(fig, rows, DROPOUTS, "Points are offset sideways by series; presets are categories, not a scale.")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    return fig


def _belief_figure(rows: list[dict]):
    apply_mission_style("light")
    colors = dict(zip(MESSAGE_MODES, series_colors(len(MESSAGE_MODES))))
    main = _main_rows(rows)
    presets = _presets_present(main)
    offsets = {"delta": -0.12, "cumulative": 0.12}
    panels = (("belief_error_cells", "belief error cells (median, 95 % interval)"),
              ("duplicated_fraction", "duplicated fraction of area (median, 95 % interval)"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (field, ylabel) in zip(axes, panels):
        for mode in MESSAGE_MODES:
            xs, summaries = [], []
            for index, preset in enumerate(presets):
                grouped = _cell_rows(main, preset, "one", mode)
                if grouped:
                    xs.append(index + offsets[mode])
                    summaries.append(median_summary([r[field] for r in grouped]))
            if xs:
                _plot_median(ax, xs, summaries, marker=MODE_MARKERS[mode], color=colors[mode], label=mode)
        _categorical_axis(ax, presets)
        ax.set_ylabel(ylabel, fontsize=8)
        if ax.containers:
            ax.legend(fontsize=7, title="message mode", title_fontsize=7)
    axes[0].set_title("belief error cells, dropout: one", fontsize=9)
    axes[1].set_title("duplicated fraction, dropout: one", fontsize=9)
    fig.suptitle("Coverage belief and duplication after one dropout, simulated", fontsize=11)
    _caption(fig, rows, ("one",), "Message modes are offset sideways; presets are categories, not a scale.")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    return fig


def _figures(output: Path, rows: list[dict], horizon_s: float) -> list[str]:
    names: list[str] = []
    for name, fig in (("sweep_completion_simulated.png", _completion_figure(rows, horizon_s)),
                      ("sweep_belief_error_simulated.png", _belief_figure(rows))):
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        names.append(name)
    return names


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   argv: list[str] | None = None, jobs: list[dict] | None = None) -> dict:
    output = prepare_output(Path(output))
    started = time.perf_counter()
    run_list = build_jobs(pilot=pilot) if jobs is None else list(jobs)
    rows = run_jobs(run_list, n_processes=processes(n_processes))
    csv_name = "sweep.csv"
    write_rows(output / csv_name, rows, first_columns=("slice", "preset", "dropout", "message_mode", "adapter", "policy_name", "seed_index", "seed"))
    horizon_s = max(float(job["scenario"]["duration_s"]) for job in run_list)
    summary = _summarize(rows, pilot=pilot, horizon_s=horizon_s)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows, horizon_s)
    outputs = [csv_name, "summary.json", *figures]
    actual_argv = list(argv or [])
    write_manifest(output, schema="zephyr-exp031-1", argv=actual_argv,
                   command=replay_command("tools/e_exp031_sweep.py", actual_argv),
                   config={"experiment": "EXP-031", "pilot": pilot,
                           "duration_s": horizon_s},
                   seeds=seeds_for(EXP_NUMBER, 3 if pilot else 50), outputs=outputs,
                   limits=list(MISSION_LIMITS) + list(SweepMission.limits), runtime_s=time.perf_counter() - started,
                   n_processes=processes(n_processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/sweep.py", "tools/e_exp031_sweep.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args(argv)
    recorded_argv = list(sys.argv[1:] if argv is None else argv)
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes,
                   argv=recorded_argv)


if __name__ == "__main__":
    main()
