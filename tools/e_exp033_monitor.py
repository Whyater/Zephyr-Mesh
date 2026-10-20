"""EXP-033 simulated monitoring runner.

Usage: ``PYTHONPATH=. python tools/e_exp033_monitor.py --output runs/e-exp033``

The scenario is simulated only. Observers hold fixed posts and alert humans
when the estimated unknown drone is in the alert zone.
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.missions.engine import COMMON_LIMITS
from tests.reference_e.hand_checks import alert_latency_s, expected_detection_s
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (
    MISSION_LIMITS,
    MISSION_SOURCES,
    apply_mission_style,
    group_rows,
    make_job,
    median_summary,
    processes,
    replay_command,
    rate_summary,
    run_jobs,
    seeds_for,
    series_colors,
    write_rows,
)


PATHS = ("crossing", "skimming", "loiter")
PRESETS = ("clean", "iid_10", "iid_30", "burst_10_0.5s", "burst_30_0.5s")
DELAYS = (0.0, 0.2, 0.5)
# The unknown drone flies at 3 m/s across the 60 m venue, so 20 s covers the
# whole crossing and skimming pass (zone edge at x = 20 to 40 m, t = 6.7 to
# 13.3 s) and about 6.7 s of the loiter circle (it starts circling at 11.3 s).
DURATION_S = 20.0
FIX_PERIOD_S = 0.1
DEBOUNCE_K = 3
SAFETY_LIMIT_M = 0.5
PREDICTION_SOURCE = "pre-registered before the runs (EXP-033 section of the private pre-registration)"
PREDICTION = (
    "clean: latency about 0.25 to 0.4 s, no misses; "
    "skimming at 2 m outside with 0.5 m noise (fused about 0.35 m) gives a few "
    "percent false alerts at most (3 consecutive needed). Loss raises latency by "
    "about the mean gap; bursts raise it by about the burst length (0.5 s) and "
    "cut continuity; misses only on burst_30 at long delay."
)
# Paths where the unknown drone enters the zone, so an alert is expected.
ALERT_PATHS = ("crossing", "loiter")
# Row columns compared between physics rows and point-mass rows of the same cell and seed.
ALERT_METRICS = ("alert_latency_s", "alerts_raised", "false_alert", "missed_alert")
SERIES_MARKERS = ("o", "s", "^", "D", "v")
LANDING_NOTE = (
    "A lost-link failsafe landing is straight down, so the observer never leaves its post sideways. "
    "A landed observer stops sending fixes, so monitoring from that post stops; when both observers "
    "have landed the ground station has no fresh fixes and cannot raise an alert."
)
DELAY_NOTE = (
    "Alert latency does not grow with measurement delay in this scenario. The ground station predicts "
    "its constant-velocity estimate forward to the current time at every check, and every path enters "
    "the zone along a straight line at constant speed, so the forward prediction removes the delay. "
    "A manoeuvring drone would not be treated this way: the forward prediction would be wrong by the "
    "manoeuvre made during the delay. The hand check adds the delay because it assumes no forward prediction."
)
PREDICTION_NOT_IN_ROWS = (
    "The rows do not carry the mean gap between delivered fixes or the burst lengths seen, so latency "
    "per preset is listed next to the clean latency instead of being compared with a gap length."
)


def split_sentences(text: str) -> list[str]:
    """Split a stored text into sentences without changing any character inside them."""
    return re.split(r"(?<=\.)\s+(?=[A-Z])", text)


def build_jobs(*, pilot: bool = False) -> list[dict]:
    """Build the preregistered point mass jobs and physics subset."""
    seeds = seeds_for(33, 3 if pilot else 50)
    jobs: list[dict] = []
    if pilot:
        paths, presets, delays = ("crossing", "skimming"), ("clean", "burst_10_0.5s"), (0.0, 0.5)
    else:
        paths, presets, delays = PATHS, PRESETS, DELAYS
    for path in paths:
        for preset in presets:
            for delay in delays:
                for seed in seeds:
                    labels = {"path": path, "preset": preset, "delay_s": delay, "adapter": "point_mass"}
                    scenario = {
                        "mission": "monitor", "seed": seed, "duration_s": DURATION_S,
                        "adapter": "point_mass", "preset": preset,
                        "measurement_delay_s": delay, "params": {"path": path},
                    }
                    jobs.append(make_job(labels, scenario))
    physics_seeds = seeds[:2] if pilot else seeds[:20]
    physics_cells = (("crossing", "clean"),) if pilot else tuple((p, preset) for p in PATHS for preset in ("clean", "burst_10_0.5s"))
    for path, preset in physics_cells:
        for seed in physics_seeds:
            labels = {"path": path, "preset": preset, "delay_s": 0.0, "adapter": "physics"}
            scenario = {
                "mission": "monitor", "seed": seed, "duration_s": DURATION_S,
                "adapter": "physics", "preset": preset,
                "measurement_delay_s": 0.0, "params": {"path": path},
            }
            jobs.append(make_job(labels, scenario))
    return jobs


build_grid = build_jobs


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n_runs": len(rows),
        "continuity": median_summary([row["continuity"] for row in rows]),
        "alert_latency_s": median_summary([row["alert_latency_s"] for row in rows]),
        "false_alert": rate_summary([bool(row["false_alert"]) for row in rows]),
        "missed_alert": rate_summary([bool(row["missed_alert"]) for row in rows]),
        "max_estimate_age_s": median_summary([row["max_estimate_age_s"] for row in rows]),
        "alerts_raised": median_summary([float(row["alerts_raised"]) for row in rows]),
        "observer_max_horizontal_displacement_m": {
            "maximum": max(float(row["observer_max_horizontal_displacement_m"]) for row in rows),
            "under_limit": all(float(row["observer_max_horizontal_displacement_m"]) < SAFETY_LIMIT_M for row in rows),
        },
        "observer_max_displacement_m": max(float(row["observer_max_displacement_m"]) for row in rows),
        "runs_with_observer_landing": rate_summary([_landed(row) for row in rows]),
        "n_runs_with_observer_landing": sum(1 for row in rows if _landed(row)),
        "n_missed_alert_runs": sum(1 for row in rows if bool(row["missed_alert"])),
        "n_missed_alert_runs_with_observer_landing": sum(
            1 for row in rows if bool(row["missed_alert"]) and _landed(row)
        ),
    }


def _landed(row: dict) -> bool:
    """True when at least one observer made a failsafe landing during the run."""
    return int(row["observer_landings"]) > 0


def _point_rows(rows: list[dict], **labels) -> list[dict]:
    """Point-mass rows whose labels equal every given value."""
    return [row for row in rows if row["adapter"] == "point_mass"
            and all(row[key] == value for key, value in labels.items())]


def _present(values: tuple, rows: list[dict], key: str) -> list:
    """The values of ``key`` that occur in ``rows``, in the given order."""
    return [value for value in values if any(row[key] == value for row in rows)]


def _values_equal(left, right) -> bool:
    """Equality for metric values where two missing numbers (NaN) count as equal."""
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True
    return left == right


def _physics_cell_equality(cell_rows: list[dict], all_rows: list[dict]) -> dict:
    """Compare each physics row with the point-mass row of the same path, preset, delay and seed."""
    partners = {(row["path"], row["preset"], row["delay_s"], row["seed"]): row
                for row in all_rows if row["adapter"] == "point_mass"}
    matching = 0
    without_partner = 0
    for row in cell_rows:
        partner = partners.get((row["path"], row["preset"], row["delay_s"], row["seed"]))
        if partner is None:
            without_partner += 1
            continue
        if all(_values_equal(row[key], partner[key]) for key in ALERT_METRICS):
            matching += 1
    return {
        "n_physics_rows": len(cell_rows),
        "n_rows_matching": matching,
        "n_rows_without_point_mass_partner": without_partner,
        "matches": bool(cell_rows) and matching == len(cell_rows),
    }


def physics_alert_equality(rows: list[dict]) -> dict:
    """Whether physics rows equal point-mass rows on every alert metric, cell by cell."""
    keys = ("path", "preset", "delay_s")
    physics_rows = [row for row in rows if row["adapter"] == "physics"]
    cells = []
    for key, grouped in group_rows(physics_rows, keys).items():
        item = dict(zip(keys, key))
        item.update(_physics_cell_equality(grouped, rows))
        cells.append(item)
    return {
        "metrics_compared": list(ALERT_METRICS),
        "compared_on": "same path, preset, measurement delay and seed",
        "physics_matches_point_mass": (all(cell["matches"] for cell in cells) if cells else None),
        "cells": cells,
    }


def missed_alerts_and_landings(rows: list[dict]) -> dict:
    """Counts relating missed alerts to observer landings, per adapter and preset."""
    entries = []
    totals = {"n_runs": 0, "n_missed_alert_runs": 0, "n_runs_with_observer_landing": 0,
              "n_missed_alert_runs_with_observer_landing": 0,
              "n_missed_alert_runs_without_observer_landing": 0}
    for adapter in ("point_mass", "physics"):
        for preset in PRESETS:
            selected = [row for row in rows if row["adapter"] == adapter and row["preset"] == preset
                        and row["path"] in ALERT_PATHS]
            if not selected:
                continue
            missed = [row for row in selected if bool(row["missed_alert"])]
            counts = {
                "n_runs": len(selected),
                "n_missed_alert_runs": len(missed),
                "n_runs_with_observer_landing": sum(1 for row in selected if _landed(row)),
                "n_missed_alert_runs_with_observer_landing": sum(1 for row in missed if _landed(row)),
                "n_missed_alert_runs_without_observer_landing": sum(1 for row in missed if not _landed(row)),
            }
            for name, value in counts.items():
                totals[name] += value
            entries.append({"adapter": adapter, "preset": preset, **counts})
    return {
        "paths": list(ALERT_PATHS),
        "pooled_over": ["path (crossing and loiter)", "measurement delay"],
        "by_adapter_and_preset": entries,
        "totals": totals,
        "note": LANDING_NOTE,
    }


def _latency_entry(cell: list[dict]) -> dict:
    values = [row["alert_latency_s"] for row in cell if row["alert_latency_s"] is not None]
    interval = median_summary(values)
    return {"n_runs": len(cell), "n_runs_with_alert": len(values), "median_s": interval["median"],
            "low_s": interval["low"], "high_s": interval["high"]}


def _miss_entry(cell: list[dict]) -> dict:
    summary = rate_summary([bool(row["missed_alert"]) for row in cell])
    return {"n_runs": summary["n"], "missed_alert_count": summary["count"], "rate": summary["rate"],
            "wilson_high": summary["wilson_high"]}


def prediction_observations(rows: list[dict]) -> list[dict]:
    """For each stored prediction sentence, the point-mass numbers it refers to."""
    sentences = split_sentences(PREDICTION)
    point = _point_rows(rows)
    presets = _present(PRESETS, point, "preset")
    delays = _present(DELAYS, point, "delay_s")
    alert_paths = _present(ALERT_PATHS, point, "path")

    clean_latency = []
    clean_misses = []
    for path in alert_paths:
        for delay in delays:
            cell = _point_rows(rows, path=path, preset="clean", delay_s=delay)
            if cell:
                clean_latency.append({"path": path, "delay_s": delay, **_latency_entry(cell)})
                clean_misses.append({"path": path, "delay_s": delay, **_miss_entry(cell)})
    skimming_false = []
    for preset in presets:
        for delay in delays:
            cell = _point_rows(rows, path="skimming", preset=preset, delay_s=delay)
            if cell:
                summary = rate_summary([bool(row["false_alert"]) for row in cell])
                skimming_false.append({"preset": preset, "delay_s": delay, "n_runs": summary["n"],
                                       "false_alert_count": summary["count"], "rate": summary["rate"],
                                       "wilson_high": summary["wilson_high"]})
    first = {"clean_alert_latency_s": clean_latency, "clean_missed_alerts": clean_misses,
             "skimming_false_alerts": skimming_false}

    latency_by_preset = []
    misses_by_preset = []
    continuity_at_zero = []
    for path in alert_paths:
        for preset in presets:
            for delay in delays:
                cell = _point_rows(rows, path=path, preset=preset, delay_s=delay)
                if not cell:
                    continue
                entry = {"path": path, "preset": preset, "delay_s": delay, **_latency_entry(cell)}
                clean_cell = _point_rows(rows, path=path, preset="clean", delay_s=delay)
                clean_median = _latency_entry(clean_cell)["median_s"] if clean_cell else None
                entry["clean_median_s"] = clean_median
                entry["change_from_clean_median_s"] = (
                    None if clean_median is None or entry["median_s"] is None else entry["median_s"] - clean_median
                )
                latency_by_preset.append(entry)
                misses_by_preset.append({"path": path, "preset": preset, "delay_s": delay, **_miss_entry(cell)})
    for path in _present(PATHS, point, "path"):
        for preset in presets:
            cell = _point_rows(rows, path=path, preset=preset, delay_s=0.0)
            if cell:
                interval = median_summary([row["continuity"] for row in cell])
                continuity_at_zero.append({"path": path, "preset": preset, "n_runs": len(cell),
                                           "median": interval["median"], "low": interval["low"],
                                           "high": interval["high"]})
    second = {"alert_latency_s_by_preset": latency_by_preset,
              "continuity_at_delay_0_by_preset": continuity_at_zero,
              "missed_alerts_by_preset_and_delay": misses_by_preset,
              "not_in_rows": PREDICTION_NOT_IN_ROWS}

    observed_by_sentence = [first, second]
    result = []
    for index, sentence in enumerate(sentences):
        observed = observed_by_sentence[index] if index < len(observed_by_sentence) else {}
        result.append({"sentence": sentence, "adapter": "point_mass", "observed": observed})
    return result


def _summary(rows: list[dict], *, pilot: bool) -> dict:
    keys = ("adapter", "path", "preset", "delay_s")
    cells = []
    for key, grouped in group_rows(rows, keys).items():
        item = dict(zip(keys, key))
        item["metrics"] = _cell_summary(grouped)
        if item["adapter"] == "physics":
            item["physics_matches_point_mass_on_alert_metrics"] = _physics_cell_equality(grouped, rows)["matches"]
        cells.append(item)
    safety_max = max(float(row["observer_max_horizontal_displacement_m"]) for row in rows)
    detection_s = expected_detection_s(FIX_PERIOD_S, DEBOUNCE_K)
    observed = []
    for delay in DELAYS:
        selected = [r["alert_latency_s"] for r in rows
                    if r["adapter"] == "point_mass" and r["path"] == "crossing"
                    and r["preset"] == "clean" and r["delay_s"] == delay and r["alert_latency_s"] is not None]
        if selected:
            observed_interval = median_summary(selected)
            hand_value = alert_latency_s(detection_s, delay, 0.0)
            observed.append({"delay_s": delay, "observed_median_s": median_summary(selected)["median"],
                             "observed_median_interval_s": {
                                 "low": observed_interval["low"],
                                 "high": observed_interval["high"],
                             },
                             "hand_check_s": hand_value,
                             "agrees_with_hand_check": (
                                 observed_interval["low"] <= hand_value <= observed_interval["high"]
                             )})
    observer_landing_runs = sum(1 for row in rows if _landed(row))
    return {
        "experiment": "EXP-033",
        "pilot": bool(pilot),
        "simulated": True,
        "cells": cells,
        "design_choices": [
            "20 s mission duration: The unknown drone flies at 3 m/s across the 60 m venue, so 20 s covers the whole crossing and skimming pass (zone edge at x = 20 to 40 m, t = 6.7 to 13.3 s) and about 6.7 s of the loiter circle (it starts circling at 11.3 s).",
            "The preregistration gives no duration, so 20 s is recorded as a runner design choice.",
            "Mission change before any full run (prereg Addendum D): an estimate older than 1.0 s cannot raise or hold an alert.",
        ],
        "hand_check": {"formula": "(k - 0.5) * 0.1 + delay + estimator_lag (lag taken as 0)", "k": DEBOUNCE_K,
                        "expected_by_delay_s": [{"delay_s": d, "value_s": alert_latency_s(detection_s, d, 0.0)} for d in DELAYS],
                        "observed_clean_crossing": observed,
                        "delay_note": DELAY_NOTE},
        "prediction": PREDICTION,
        "prediction_source": PREDICTION_SOURCE,
        "prediction_observations": prediction_observations(rows),
        "missed_alerts_and_observer_landings": missed_alerts_and_landings(rows),
        "physics_alert_equality": physics_alert_equality(rows),
        "safety_check": {"limit_m": SAFETY_LIMIT_M,
                         "quantity": "maximum horizontal distance of any observer from its post, over every run",
                         "maximum_observer_horizontal_displacement_m": safety_max,
                         "runs_with_observer_landing": observer_landing_runs,
                         "passed": safety_max < SAFETY_LIMIT_M,
                         "flag": "OBSERVER LEFT ITS POST SIDEWAYS" if safety_max >= SAFETY_LIMIT_M else None,
                         "note": ("A lost-link failsafe landing is straight down; it shows in observer_max_displacement_m "
                                  "and runs_with_observer_landing, and a landing observer stops sending fixes.")},
        "continuity_note": ("Continuity counts checks with estimate age under 0.5 s. At 0.5 s measurement delay it is near zero by definition; this is a property of the pre-registered metric."),
        "deviations": [
            "Observer post check uses horizontal displacement (< 0.5 m), not 3-D displacement, because a failsafe landing "
            "drops 12 m straight down; 3-D displacement and runs with an observer landing are reported per cell.",
        ],
        "missing_metrics": [],
        "limits": list(MISSION_LIMITS + COMMON_LIMITS),
    }


def _empty_figure(title: str, message: str, palette: dict):
    """A figure that says plainly that the output holds no runs for it."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=palette["textSecondary"])
    fig.suptitle(title)
    return fig


def _latency_figure(rows: list[dict], path: str, palette: dict):
    """One panel per preset: median alert latency against measurement delay."""
    point = _point_rows(rows, path=path)
    presets = _present(PRESETS, point, "preset")
    delays = _present(DELAYS, point, "delay_s")
    if not presets:
        return _empty_figure(f"Alert latency, simulated: {path} path (point mass)",
                             f"No point-mass {path} runs in this output.", palette)
    colors = series_colors(len(PRESETS))
    detection_s = expected_detection_s(FIX_PERIOD_S, DEBOUNCE_K)
    hand = [alert_latency_s(detection_s, delay, 0.0) for delay in delays]
    fig, axes = plt.subplots(1, len(presets), figsize=(2.4 * len(presets) + 1.0, 4.4),
                             sharey=True, squeeze=False)
    axes = list(axes[0])
    no_alert = {}
    handles = {}
    for ax, preset in zip(axes, presets):
        index = PRESETS.index(preset)
        (hand_line,) = ax.plot(delays, hand, linestyle="--", linewidth=1.0, color=palette["textSecondary"])
        handles.setdefault("hand check", hand_line)
        xs, medians, below, above = [], [], [], []
        for delay in delays:
            cell = _point_rows(rows, path=path, preset=preset, delay_s=delay)
            if not cell:
                continue
            entry = _latency_entry(cell)
            if entry["n_runs_with_alert"] == 0:
                no_alert.setdefault(preset, []).append(delay)
                continue
            xs.append(delay)
            medians.append(entry["median_s"])
            below.append(entry["median_s"] - entry["low_s"])
            above.append(entry["high_s"] - entry["median_s"])
            ax.annotate(f"{entry['n_runs_with_alert']}/{entry['n_runs']}", (delay, entry["median_s"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6,
                        color=palette["textSecondary"])
        if xs:
            series = ax.errorbar(xs, medians, yerr=[below, above], marker=SERIES_MARKERS[index],
                                 color=colors[index], capsize=3, linewidth=1.2)
            handles.setdefault("median with 95 % interval", series)
        if preset in no_alert:
            text = "no alerts raised" if not xs else "no alerts raised at " + ", ".join(
                f"{delay:g} s" for delay in no_alert[preset])
            ax.text(0.5, 0.9, text, transform=ax.transAxes, ha="center", va="center", fontsize=7,
                    color=palette["textSecondary"], wrap=True)
        ax.set_title(preset, fontsize=9)
        ax.set_xticks(delays)
        # Room on the right for the run-count label of the last point.
        ax.margins(x=0.25)
        ax.set_xlabel("measurement delay (s)", fontsize=8)
    axes[0].set_ylabel("alert latency (s)")
    fig.suptitle(f"Alert latency, simulated: {path} path (point mass)")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=7)
    if no_alert:
        listed = "; ".join(f"{preset} (delay " + ", ".join(f"{d:g}" for d in delays_missing) + " s)"
                           for preset, delays_missing in no_alert.items())
        missing_text = f"No alerts raised in: {listed}."
    else:
        missing_text = "Every cell raised at least one alert."
    fig.text(0.01, 0.02,
             f"Labels: runs with an alert / runs in the cell. Dashed line: hand check, "
             f"(k - 0.5) x fix period + delay with k = {DEBOUNCE_K}.\n{missing_text}",
             fontsize=7, color=palette["textSecondary"])
    fig.subplots_adjust(bottom=0.27, top=0.82, wspace=0.12)
    return fig


def _continuity_figure(rows: list[dict], palette: dict):
    """One panel per measurement delay: median continuity per preset, one bar per path."""
    point = _point_rows(rows)
    delays = _present(DELAYS, point, "delay_s")
    presets = _present(PRESETS, point, "preset")
    paths = _present(PATHS, point, "path")
    if not delays:
        return _empty_figure("Track continuity, simulated (point mass)",
                             "No point-mass runs in this output.", palette)
    colors = series_colors(len(paths))
    width = 0.8 / len(paths)
    fig, axes = plt.subplots(1, len(delays), figsize=(4.0 * len(delays) + 0.6, 4.6),
                             sharey=True, squeeze=False)
    axes = list(axes[0])
    for ax, delay in zip(axes, delays):
        for path_index, path in enumerate(paths):
            positions, heights, below, above = [], [], [], []
            for preset_index, preset in enumerate(presets):
                cell = _point_rows(rows, path=path, preset=preset, delay_s=delay)
                if not cell:
                    continue
                interval = median_summary([row["continuity"] for row in cell])
                positions.append(preset_index - 0.4 + width * (path_index + 0.5))
                heights.append(interval["median"])
                below.append(interval["median"] - interval["low"])
                above.append(interval["high"] - interval["median"])
            if positions:
                ax.bar(positions, heights, width, yerr=[below, above], capsize=2,
                       color=colors[path_index], label=path,
                       error_kw={"ecolor": palette["textSecondary"], "elinewidth": 0.8})
        ax.set_xticks(range(len(presets)))
        ax.set_xticklabels(presets, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"measurement delay {delay:g} s", fontsize=9)
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("continuity (median, 95 % interval)")
    axes[0].legend(fontsize=7, title="path", title_fontsize=7)
    fig.suptitle("Track continuity, simulated (point mass), one panel per measurement delay")
    fig.text(0.01, 0.02,
             "Continuity: fraction of checks while the unknown drone is in the venue with estimate age "
             "under 0.5 s. Cells are not pooled across delays.",
             fontsize=7, color=palette["textSecondary"])
    fig.subplots_adjust(bottom=0.27, top=0.86, wspace=0.08)
    return fig


def _figures(output: Path, rows: list[dict]) -> list[str]:
    palette = apply_mission_style()
    names = []
    for path in ALERT_PATHS:
        fig = _latency_figure(rows, path, palette)
        name = f"alert_latency_{path}.png"
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        names.append(name)
    fig = _continuity_figure(rows, palette)
    name = "continuity_simulated.png"
    fig.savefig(output / name, dpi=140)
    plt.close(fig)
    names.append(name)
    return names


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   jobs: list[dict] | None = None, argv: list[str] | None = None) -> dict:
    output_arg = output
    output = prepare_output(Path(output))
    start = time.perf_counter()
    run_list = build_jobs(pilot=pilot) if jobs is None else list(jobs)
    rows = run_jobs(run_list, n_processes=processes(n_processes))
    write_rows(output / "exp033.csv", rows, first_columns=("adapter", "path", "preset", "delay_s", "seed"))
    summary = _summary(rows, pilot=pilot)
    write_json(output / "summary.json", summary)
    figures = _figures(output, rows)
    outputs = ["exp033.csv", "summary.json", *figures]
    elapsed = time.perf_counter() - start
    if argv is None:
        # Called from code: there is no command line, so record none and name a replay command.
        manifest_argv = []
        replay_argv = ["--output", str(output_arg)] + (["--pilot"] if pilot else [])
    else:
        manifest_argv = list(argv)
        replay_argv = manifest_argv
    write_manifest(output, schema="zephyr-exp033-1", argv=manifest_argv,
                   command=replay_command("tools/e_exp033_monitor.py", replay_argv),
                   config={"pilot": pilot, "duration_s": DURATION_S},
                   seeds=seeds_for(33, 3 if pilot else 50), outputs=outputs,
                   limits=list(MISSION_LIMITS), runtime_s=elapsed,
                   n_processes=processes(n_processes),
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/monitor.py", "tools/e_exp033_monitor.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = argv_or_default(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args(raw_argv)
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes, argv=raw_argv)


if __name__ == "__main__":
    main()
