"""EXP-032 relay range runner for the simulated relay mission.

Usage: PYTHONPATH=. ./.venv/bin/python tools/e_exp032_relay.py --output runs/e-exp032 --pilot

The mission owns radio physics, retries, forwarding, and telemetry timing. This
runner only constructs the preregistered cells, aggregates returned metrics,
and writes reproducible tables, summaries, and figures.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sim.missions.api import Scenario
from sim.missions.relay import RelayMission
from tests.reference_e.hand_checks import max_range_dual_slope_m
from tools.c_common import argv_or_default, prepare_output, write_json, write_manifest
from tools.e_common import (MISSION_LIMITS, MISSION_SOURCES, group_rows, make_job,
                            apply_mission_style, median_summary, processes, replay_command,
                            run_jobs, seeds_for, series_colors, write_rows)

DISTANCES = (300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0)
FRACTIONS = (None, 0.25, 0.4, 0.5, 0.6, 0.75)
FULL_SEEDS = 50
PILOT_SEEDS = 3
PHYSICS_DISTANCES = (500.0, 900.0)
PHYSICS_FRACTIONS = (None, 0.5)

PREDICTION_SOURCE = "pre-registered before the runs (EXP-032 section of the private pre-registration)"
PREDICTION_TEXT = ("d_max with 0 dB margin is about 600 to 700 m (Run C: d90 409 m, "
                   "d10 746 m for the stress corner); direct delivery near 100 % at 300 m, near 0 beyond "
                   "about 900 m; f = 0.5 keeps delivery above 99 % to about 1100 m; asymmetric relays "
                   "(0.25, 0.75) fail first on the long hop. Latency roughly doubles with the relay (two hops, "
                   "plus retries on the weaker hop) but stays under 10 ms median.")
PREDICTION_CORRECTION_3 = ("3. EXP-032 prediction: the median-signal range for the stress corner with 0 dB margin "
                           "and -98 dBm sensitivity is about 518 m (reference computation), not 600 to 700 m. "
                           "Corrected: direct delivery falls through 50 % somewhere around 500 to 750 m "
                           "(shadowing and 5 retries push it outward); a relay at f = 0.5 holds high delivery to "
                           "about 1000 to 1300 m.")

# Distances named in the stored prediction text (not results): the direct link
# "near 0 beyond about 900 m", f = 0.5 "above 99 % to about 1100 m", and the
# corrected "high delivery to about 1000 to 1300 m".
PREDICTION_DIRECT_NEAR_M = 300.0
PREDICTION_DIRECT_FAR_FROM_M = 900.0
PREDICTION_MIDPOINT_UP_TO_M = 1100.0
PREDICTION_CORRECTED_WINDOW_M = (1000.0, 1300.0)
ASYMMETRIC_FRACTIONS = (0.25, 0.75)

# 97.5 % quantile of the standard normal distribution, for a two-sided 95 % interval.
WILSON_Z_95 = 1.959963984540054
# A delivery ratio times the packets sent must be this close to a whole number.
COUNT_TOLERANCE = 1e-6

FRACTION_ORDER = ("none", 0.25, 0.4, 0.5, 0.6, 0.75)
FRACTION_LABELS = {"none": "direct link (no relay)", 0.25: "relay at f = 0.25", 0.4: "relay at f = 0.4",
                   0.5: "relay at f = 0.5", 0.6: "relay at f = 0.6", 0.75: "relay at f = 0.75"}
FRACTION_STYLES = {"none": ("o", "-"), 0.25: ("s", "--"), 0.4: ("^", "-."),
                   0.5: ("D", "-"), 0.6: ("v", ":"), 0.75: ("P", "--")}
LATENCY_NOTE = ("Latency is over delivered packets only (conditional on delivery). Hollow markers: under half "
                "of the sent packets arrived, so the value describes the minority that got through. Cells where "
                "fewer than half the seeds delivered any packet are left out and listed in their panel.")
LATENCY_LIMIT = ("Latency (median and p95) is computed over delivered packets only, so it is conditional on "
                 "delivery: at long range it describes the packets that got through, not the lost ones. "
                 "Each cell reports seeds_without_delivered_packet.")


def build_jobs(*, pilot: bool = False, duration_s: float = 60.0,
               seed_count: int | None = None) -> tuple[list[dict], list[int]]:
    """Build point-mass jobs followed by the preregistered physics subset."""
    n = PILOT_SEEDS if seed_count is None and pilot else FULL_SEEDS if seed_count is None else int(seed_count)
    seeds = seeds_for(32, n)
    jobs: list[dict] = []
    distances = (500.0, 900.0) if pilot else DISTANCES
    fractions = (None, 0.5) if pilot else FRACTIONS
    for distance in distances:
        for fraction in fractions:
            for seed in seeds:
                params = {"distance_m": distance, "relay_fraction": fraction}
                labels = {"distance_m": distance, "relay_fraction": "none" if fraction is None else fraction,
                          "adapter": "point_mass"}
                scenario = {"mission": "relay", "seed": seed, "duration_s": float(duration_s),
                            "adapter": "point_mass", "params": params}
                jobs.append(make_job(labels, scenario))
    if pilot:
        physics_seeds = seeds[:2]
        physics_cells = ((900.0, 0.5),)
    else:
        physics_seeds = seeds[:20]
        physics_cells = tuple((d, f) for d in PHYSICS_DISTANCES for f in PHYSICS_FRACTIONS)
    for distance, fraction in physics_cells:
        for seed in physics_seeds:
            params = {"distance_m": distance, "relay_fraction": fraction}
            labels = {"distance_m": distance, "relay_fraction": "none" if fraction is None else fraction,
                      "adapter": "physics"}
            scenario = {"mission": "relay", "seed": seed, "duration_s": float(duration_s),
                        "adapter": "physics", "params": params}
            jobs.append(make_job(labels, scenario))
    return jobs, seeds


def packets_sent_per_run(duration_s: float, rate_hz: float) -> int:
    """Packets the far drone sends in one run.

    sim/missions/relay.py sends one packet at t = 0 and one every 1 / rate_hz s
    while the send time is at most duration_s (the engine ticks include
    t = duration_s), so a 60 s run at 10 Hz sends 601 packets.
    """
    return int(math.floor(float(duration_s) * float(rate_hz) + 1e-9)) + 1


def delivered_count(ratio: float, sent: int) -> int | None:
    """Delivered packets behind a ratio, or None when ratio x sent is not a whole number."""
    raw = float(ratio) * int(sent)
    whole = round(raw)
    if abs(raw - whole) > COUNT_TOLERANCE:
        return None
    return int(whole)


def wilson_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson score 95 % interval for successes out of n.

    centre = (p + z^2 / (2n)) / (1 + z^2 / n)
    half   = z * sqrt(p (1 - p) / n + z^2 / (4 n^2)) / (1 + z^2 / n)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = WILSON_Z_95
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def pooled_delivery(rows: list[dict], packets_per_run: int) -> dict:
    """Total delivered over total sent across the seeds of one cell, with a Wilson 95 % interval."""
    sent = int(packets_per_run) * len(rows)
    counts = [delivered_count(row["delivery_ratio"], packets_per_run) for row in rows]
    if not rows or any(count is None for count in counts):
        return {"sent": sent, "delivered": None, "ratio": None, "wilson_low": None, "wilson_high": None}
    delivered = int(sum(counts))
    low, high = wilson_95(delivered, sent)
    return {"sent": sent, "delivered": delivered, "ratio": delivered / sent,
            "wilson_low": low, "wilson_high": high}


def cell_summary(rows: list[dict], packets_per_run: int) -> dict:
    values = [float(row["delivery_ratio"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "delivery_ratio_mean": float(np.mean(values)),
        "delivery_ratio_interval": median_summary(values),
        "delivery_pooled": pooled_delivery(rows, packets_per_run),
        "seeds_without_delivered_packet": sum(1 for row in rows if row["median_latency_s"] is None),
        "median_latency_s": median_summary([row["median_latency_s"] for row in rows]),
        "p95_latency_s": median_summary([row["p95_latency_s"] for row in rows]),
        "hop1_delivery": median_summary([row["hop1_delivery"] for row in rows]),
        "hop2_delivery": median_summary([row["hop2_delivery"] for row in rows]),
        "retries_mean": median_summary([row["retries_mean"] for row in rows]),
    }


def _values_equal(left, right) -> bool:
    """Equality for metric values where two missing numbers (NaN) count as equal."""
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True
    return left == right


def physics_matches_point_mass(cell_rows: list[dict], all_rows: list[dict]) -> bool:
    """True when every physics row equals the point-mass row of the same D, f and seed on every metric."""
    label_keys = {"adapter", "distance_m", "relay_fraction", "seed"}
    point_by_key = {(r["distance_m"], r["relay_fraction"], r["seed"]): r
                    for r in all_rows if r["adapter"] == "point_mass"}
    for row in cell_rows:
        partner = point_by_key.get((row["distance_m"], row["relay_fraction"], row["seed"]))
        if partner is None:
            return False
        metric_keys = set(row) | set(partner)
        for key in sorted(metric_keys - label_keys):
            if key not in row or key not in partner or not _values_equal(row[key], partner[key]):
                return False
    return True


def _interpolated_crossing(cell_values: dict[float, float]) -> float | None:
    ordered = sorted(cell_values.items())
    for (d0, v0), (d1, v1) in zip(ordered, ordered[1:]):
        if (v0 - 0.5) * (v1 - 0.5) <= 0.0 and v0 != v1:
            return float(d0 + (0.5 - v0) * (d1 - d0) / (v1 - v0))
    return None


def _point_cell(cells: dict, distance: float, fraction) -> dict | None:
    return cells.get(f"point_mass|{float(distance)}|{fraction}")


def _delivery_observation(cell: dict | None) -> dict | None:
    if cell is None:
        return None
    pooled = cell["delivery_pooled"]
    return {"mean": cell["delivery_ratio_mean"], "pooled_wilson_low": pooled["wilson_low"],
            "pooled_wilson_high": pooled["wilson_high"], "n_seeds": cell["n_seeds"]}


def _cell_median_latency(cell: dict | None) -> float | None:
    if cell is None:
        return None
    return cell["median_latency_s"]["median"]


def split_sentences(text: str) -> list[str]:
    """Split stored prediction text at a full stop followed by a capital letter.

    A full stop after a digit ("3. EXP-032") or inside a number ("0.5") is not a
    sentence end. Joining the pieces with single spaces gives the text back.
    """
    return re.split(r"(?<=[a-z%)]\.)\s+(?=[A-Z])", text)


def longer_hop(fraction: float) -> str:
    """Which hop is longer: the relay sits at fraction * D from the ground
    station (sim/missions/relay.py), so hop 1 (far drone to relay) is
    (1 - fraction) * D and hop 2 (relay to ground station) is fraction * D."""
    if fraction < 0.5:
        return "hop1"
    if fraction > 0.5:
        return "hop2"
    return "equal"


def prediction_observations(cells: dict, hand_check: dict, rows: list[dict]) -> list[dict]:
    """For every sentence of the stored prediction and its correction, the observed numbers it refers to."""
    distances = sorted({float(key.split("|")[1]) for key in cells if key.startswith("point_mass|")})

    def delivery_by_distance(fraction, keep) -> dict:
        return {str(d): _delivery_observation(_point_cell(cells, d, fraction))
                for d in distances if keep(d) and _point_cell(cells, d, fraction) is not None}

    asymmetric_gap: dict = {}
    hops: dict = {}
    for fraction in ASYMMETRIC_FRACTIONS:
        gap_row = {}
        hop_row = {}
        for d in distances:
            mid = _point_cell(cells, d, 0.5)
            side = _point_cell(cells, d, fraction)
            if mid is None or side is None:
                continue
            gap_row[str(d)] = mid["delivery_ratio_mean"] - side["delivery_ratio_mean"]
            hop_row[str(d)] = {"hop1_delivery_median": side["hop1_delivery"]["median"],
                               "hop2_delivery_median": side["hop2_delivery"]["median"],
                               "longer_hop": longer_hop(fraction)}
        asymmetric_gap[str(fraction)] = gap_row
        hops[str(fraction)] = hop_row

    latency_ratio: dict = {}
    for fraction in FRACTION_ORDER[1:]:
        ratio_row = {}
        for d in distances:
            relay = _cell_median_latency(_point_cell(cells, d, fraction))
            direct = _cell_median_latency(_point_cell(cells, d, "none"))
            if relay is not None and direct is not None and direct > 0.0:
                ratio_row[str(d)] = relay / direct
        if ratio_row:
            latency_ratio[str(fraction)] = ratio_row
    largest = None
    for key, cell in cells.items():
        value = cell["median_latency_s"]["median"]
        if key.startswith("point_mass|") and value is not None and (largest is None or value > largest[1]):
            largest = (key, value, cell["n_seeds"] - cell["seeds_without_delivered_packet"], cell["n_seeds"])
    largest_seed = None
    for row in rows:
        value = row["median_latency_s"]
        if row["adapter"] == "point_mass" and value is not None and (largest_seed is None
                                                                      or value > largest_seed["value"]):
            largest_seed = {"value": value, "distance_m": row["distance_m"],
                            "relay_fraction": row["relay_fraction"], "seed": row["seed"]}
    silent = {key: cell["seeds_without_delivered_packet"] for key, cell in cells.items()
              if key.startswith("point_mass|") and cell["seeds_without_delivered_packet"] > 0}

    original = split_sentences(PREDICTION_TEXT)
    corrected = split_sentences(PREDICTION_CORRECTION_3)
    blocks = [
        {"source": "prediction", "sentence": original[0], "observed": {
            "reference_median_signal_range_m": hand_check["reference_median_signal_range_m"],
            "direct_50_percent_crossing_m": hand_check["direct_50_percent_crossing_m_interpolation"],
            "direct_delivery_at_300_m": _delivery_observation(
                _point_cell(cells, PREDICTION_DIRECT_NEAR_M, "none")),
            "direct_delivery_at_900_m_and_beyond": delivery_by_distance(
                "none", lambda d: d >= PREDICTION_DIRECT_FAR_FROM_M),
            "f_0_5_delivery_up_to_1100_m": delivery_by_distance(
                0.5, lambda d: d <= PREDICTION_MIDPOINT_UP_TO_M),
            "f_0_5_minus_asymmetric_delivery": asymmetric_gap,
            "asymmetric_hop_delivery": hops,
        }},
        {"source": "prediction", "sentence": original[1], "observed": {
            "median_latency_ratio_relay_over_direct": latency_ratio,
            "max_cell_median_latency_s": (None if largest is None else
                                          {"value": largest[1], "cell": largest[0],
                                           "seeds_with_a_delivered_packet": largest[2], "n_seeds": largest[3]}),
            "largest_per_seed_median_latency_s": largest_seed,
            "cells_with_seeds_without_delivered_packet": silent,
            "note": "Latency is over delivered packets only.",
        }},
        {"source": "prediction_correction_3", "sentence": corrected[0], "observed": {
            "reference_median_signal_range_m": hand_check["reference_median_signal_range_m"],
        }},
        {"source": "prediction_correction_3", "sentence": corrected[1], "observed": {
            "direct_50_percent_crossing_m": hand_check["direct_50_percent_crossing_m_interpolation"],
            "direct_delivery_by_distance": delivery_by_distance("none", lambda d: True),
            "f_0_5_delivery_from_1000_to_1300_m": delivery_by_distance(
                0.5, lambda d: PREDICTION_CORRECTED_WINDOW_M[0] <= d <= PREDICTION_CORRECTED_WINDOW_M[1]),
        }},
    ]
    return blocks


def _fraction_colors() -> dict:
    colors = series_colors(len(FRACTION_ORDER))
    return {fraction: colors[index] for index, fraction in enumerate(FRACTION_ORDER)}


def delivery_points(cells: dict, fraction) -> tuple[list[tuple[float, float, float, float]], list, list]:
    """(distance, pooled ratio, Wilson low, Wilson high) per point-mass cell of one fraction.

    The other two items (distances left out, distances drawn hollow) are always
    empty for delivery; they keep the same shape as latency_points.
    """
    points = []
    for distance in DISTANCES:
        cell = _point_cell(cells, distance, fraction)
        if cell is None or cell["delivery_pooled"]["ratio"] is None:
            continue
        pooled = cell["delivery_pooled"]
        points.append((distance, pooled["ratio"], pooled["wilson_low"], pooled["wilson_high"]))
    return points, [], []


def latency_points(cells: dict, fraction) -> tuple[list[tuple[float, float, float, float]], list[float],
                                                  list[float]]:
    """Median latency points in ms (distance, median, low, high), distances left out, distances drawn hollow.

    A cell is left out when fewer than half of its seeds delivered any packet,
    and drawn hollow when under half of its sent packets arrived: in both cases
    the latency describes only the packets that survived.
    """
    points = []
    omitted = []
    minority = []
    for distance in DISTANCES:
        cell = _point_cell(cells, distance, fraction)
        if cell is None:
            continue
        delivering = cell["n_seeds"] - cell["seeds_without_delivered_packet"]
        summary = cell["median_latency_s"]
        if 2 * delivering < cell["n_seeds"] or summary["median"] is None:
            omitted.append(distance)
            continue
        points.append((distance, 1000.0 * summary["median"], 1000.0 * summary["low"], 1000.0 * summary["high"]))
        pooled_ratio = cell["delivery_pooled"]["ratio"]
        if pooled_ratio is None or pooled_ratio < 0.5:
            minority.append(distance)
    return points, omitted, minority


def _faceted_figure(cells: dict, point_builder, ylabel: str, title: str, note: str, ylim=None):
    colors = _fraction_colors()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
    direct_points = point_builder(cells, "none")[0]
    for ax, fraction in zip(axes.flat, FRACTION_ORDER):
        points, omitted, minority = point_builder(cells, fraction)
        label = FRACTION_LABELS[fraction]
        marker, linestyle = FRACTION_STYLES[fraction]
        if fraction != "none" and direct_points:
            ax.plot([p[0] for p in direct_points], [p[1] for p in direct_points], color=colors["none"],
                    linestyle="-", linewidth=1.0, alpha=0.35, label="direct link, for reference")
        if points:
            xs = [p[0] for p in points]
            ax.fill_between(xs, [p[2] for p in points], [p[3] for p in points], color=colors[fraction],
                            alpha=0.18, linewidth=0.0)
            ax.plot(xs, [p[1] for p in points], color=colors[fraction], marker=marker, linestyle=linestyle,
                    label=label)
            hollow = [p for p in points if p[0] in minority]
            if hollow:
                ax.plot([p[0] for p in hollow], [p[1] for p in hollow], linestyle="none", marker=marker,
                        markersize=8, markerfacecolor=ax.get_facecolor(), markeredgecolor=colors[fraction],
                        markeredgewidth=1.6, label="_hollow")
        else:
            ax.plot([], [], color=colors[fraction], marker=marker, linestyle=linestyle, label=label)
            ax.text(0.5, 0.5, "not in this run", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        if omitted:
            ax.text(0.03, 0.95, "left out: " + ", ".join(f"{d:.0f} m" for d in omitted), transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
        ax.set_title(label)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes[1, :]:
        ax.set_xlabel("far-drone distance (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=colors["none"], linestyle="-", linewidth=1.0, alpha=0.35,
                      label="direct link, for reference")]
    if any(point_builder(cells, fraction)[2] for fraction in FRACTION_ORDER):
        handles.append(Line2D([], [], color=colors["none"], linestyle="none", marker="o",
                              markerfacecolor="none", label="under half of sent packets arrived"))
    fig.suptitle(title)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(handles),
               fontsize=8.5, frameon=False)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    return fig


def delivery_figure(cells: dict):
    note = ("Pooled over seeds: total delivered over total sent; band is the Wilson 95 % interval "
            "(it can be thinner than the line). "
            "f is the relay position as a fraction of the line from the ground station.")
    return _faceted_figure(cells, delivery_points, "end-to-end delivery ratio",
                           "Relay delivery ratio vs distance, simulated", note, ylim=(0.0, 1.05))


def latency_figure(cells: dict):
    note = LATENCY_NOTE + " Line: median over seeds of each run's median latency; band: its 95 % interval."
    return _faceted_figure(cells, latency_points, "median latency (ms)",
                           "Relay median latency vs distance, simulated", note)


def _make_figures(output: Path, cells: dict) -> None:
    apply_mission_style("light")
    fig = delivery_figure(cells)
    fig.savefig(output / "delivery_vs_distance.png", dpi=160)
    plt.close(fig)
    fig = latency_figure(cells)
    fig.savefig(output / "latency_vs_distance.png", dpi=160)
    plt.close(fig)


def run_experiment(output: Path, *, pilot: bool = False, n_processes: int | None = None,
                   duration_s: float = 60.0, seed_count: int | None = None,
                   argv: list[str] | None = None) -> dict:
    output_arg = output
    output = prepare_output(Path(output))
    jobs, seeds = build_jobs(pilot=pilot, duration_s=duration_s, seed_count=seed_count)
    nproc = processes(n_processes)
    started = time.perf_counter()
    rows = run_jobs(jobs, n_processes=nproc)
    runtime = time.perf_counter() - started
    write_rows(output / "relay.csv", rows,
               first_columns=("adapter", "distance_m", "relay_fraction", "seed"))
    # The jobs never override rate_hz, so every run uses the mission default send rate.
    packets_per_run = packets_sent_per_run(duration_s, RelayMission.defaults["rate_hz"])
    cells = {}
    for key, cell_rows in group_rows(rows, ("adapter", "distance_m", "relay_fraction")).items():
        cell = cell_summary(cell_rows, packets_per_run)
        if key[0] == "physics":
            cell["physics_matches_point_mass"] = physics_matches_point_mass(cell_rows, rows)
        cells["|".join(str(v) for v in key)] = cell
    breakpoint_m = 4.0 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)
    reference = max_range_dual_slope_m(19.5 - 6.0, -98.0, 40.2, 2.0, breakpoint_m, 4.0, 0.0)
    direct = {float(d): float(np.mean([r["delivery_ratio"] for r in rows
                                        if r["adapter"] == "point_mass" and r["relay_fraction"] == "none" and r["distance_m"] == d]))
              for d in DISTANCES if any(r["adapter"] == "point_mass" and r["relay_fraction"] == "none" and r["distance_m"] == d for r in rows)}
    midpoint = {float(d): float(np.mean([r["delivery_ratio"] for r in rows
                                          if r["adapter"] == "point_mass" and r["relay_fraction"] == 0.5 and r["distance_m"] == d]))
                for d in DISTANCES if any(r["adapter"] == "point_mass" and r["relay_fraction"] == 0.5 and r["distance_m"] == d for r in rows)}
    point_cell_keys = {(row["distance_m"], row["relay_fraction"]) for row in rows if row["adapter"] == "point_mass"}
    physics_cell_keys = {(row["distance_m"], row["relay_fraction"]) for row in rows if row["adapter"] == "physics"}
    point_seed_count = (sum(1 for row in rows if row["adapter"] == "point_mass") // len(point_cell_keys)
                        if point_cell_keys else 0)
    physics_seed_count = (sum(1 for row in rows if row["adapter"] == "physics") // len(physics_cell_keys)
                          if physics_cell_keys else 0)
    deviations = []
    if pilot:
        deviations.append("Pilot uses the preregistered subset and is not the full grid.")
    if duration_s != 60.0:
        deviations.append("Run duration was shortened for a test invocation.")
    deviations.append(
        "The pre-registration names a Wilson interval for end-to-end delivery. Each cell reports "
        "delivery_pooled: total delivered over total sent across its seeds, with a Wilson 95 % interval that "
        "treats every packet as an independent trial. The rows carry only delivery_ratio, so the counts are "
        "rebuilt from the mission's fixed send schedule (one packet at t = 0 and one every 1 / rate_hz s up to "
        "the run duration) and accepted only where ratio x sent is a whole number. The order-statistic median "
        "interval of the per-seed ratios (delivery_ratio_interval) is also reported, because it does not assume "
        "packets are independent within a run and because the Addendum A relay goal rule is written on it.")
    uncounted = sorted(key for key, cell in cells.items() if cell["delivery_pooled"]["delivered"] is None)
    if uncounted:
        deviations.append("Delivered counts could not be rebuilt for these cells, so delivery_pooled is null: "
                          + ", ".join(uncounted) + ".")
    hand_check = {"reference_median_signal_range_m": reference,
                  "two_d_max_m": 2.0 * reference,
                  "direct_50_percent_crossing_m_interpolation": _interpolated_crossing(direct),
                  "relay_f_0_5_50_percent_crossing_m_interpolation": _interpolated_crossing(midpoint)}
    summary = {
        "experiment": "EXP-032",
        "pilot": bool(pilot),
        "duration_s": float(duration_s),
        "grid": {"distances_m": list(DISTANCES), "relay_fractions": ["none", *FRACTIONS[1:]],
                 "point_mass_seeds": FULL_SEEDS, "physics_distances_m": list(PHYSICS_DISTANCES),
                 "physics_fractions": ["none", 0.5], "physics_seeds": 20,
                 "actual_point_mass_seeds": point_seed_count, "actual_physics_seeds": physics_seed_count},
        "mission_defaults": dict(RelayMission.defaults),
        "simulated": True,
        "limits": list(MISSION_LIMITS) + list(RelayMission.limits) + [LATENCY_LIMIT],
        "cells": cells,
        "hand_check": hand_check,
        "prediction": PREDICTION_TEXT,
        "prediction_source": PREDICTION_SOURCE,
        "prediction_correction_3": PREDICTION_CORRECTION_3,
        "prediction_observed": {"text": PREDICTION_CORRECTION_3,
                       "observed_direct": direct, "observed_f_0_5": midpoint},
        "prediction_observations": prediction_observations(cells, hand_check, rows),
        "physics_geometry_note": ("Physics rows equal point-mass rows by construction: path loss uses the fixed "
                                  "commanded positions (sim/missions/relay.py limits), so the physics adapter "
                                  "only changes the vehicles, not the radio geometry."),
        "missing_metrics": ["retries_mean: RadioMissionLink does not expose per-packet retry counts; mission returns null."],
        "deviations": deviations,
    }
    write_json(output / "summary.json", summary)
    _make_figures(output, cells)
    if argv is None:
        # Called from code: there is no command line, so record none and name a replay command.
        actual_argv = []
        replay_argv = ["--output", str(output_arg)] + (["--pilot"] if pilot else [])
    else:
        actual_argv = list(argv)
        replay_argv = actual_argv
    write_manifest(output, schema="zephyr-exp032-runner-1", argv=actual_argv,
                   command=replay_command("tools/e_exp032_relay.py", replay_argv), config=summary["grid"],
                   seeds=seeds, outputs=["relay.csv", "summary.json", "delivery_vs_distance.png", "latency_vs_distance.png"],
                   limits=list(MISSION_LIMITS), n_processes=nproc, runtime_s=runtime,
                   hashed_sources=list(MISSION_SOURCES) + ["sim/missions/relay.py", "tools/e_exp032_relay.py"])
    return summary


def main(argv: list[str] | None = None) -> None:
    actual_argv = argv_or_default(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args(actual_argv)
    run_experiment(args.output, pilot=args.pilot, n_processes=args.processes, argv=actual_argv)


if __name__ == "__main__":
    main()
