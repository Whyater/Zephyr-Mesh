"""Sensitivity of the S6 tolerance result to the shape of the vision error.

S6 (``sim/s6_recovery.py``) draws fresh white noise for every vision frame.
Published webcam ArUco data shows a different shape: frame-to-frame jitter of
well under a centimetre, plus a larger accuracy error (a bias) that changes
slowly with position and lighting. A Kalman filter averages white noise down
but cannot average a bias away, so the shape may matter even at equal size.

This tool reruns the S6 main slice with the vision error swapped for one of
these models, all at the same total RMS as the S6 noise cell ``s``
(3D mean-square error ``3 s^2``):

- ``white_iso``: today's model, fresh N(0, s^2) on every axis every frame,
  generated with ``SensorModel`` exactly like S6 (the control).
- ``white_aniso``: fresh white noise with depth (y) 3x lateral.
- ``jitter_const_bias``: small per-frame jitter plus one constant bias per
  seed and axis.
- ``jitter_ou_bias``: the same jitter plus a bias that drifts as a
  Gauss-Markov (Ornstein-Uhlenbeck) process with correlation time ``tau_s``.
- ``jitter_position_bias`` (added after the pilot): the same jitter plus a
  bias that is a smooth function of where the target is, built from random
  plane waves of wavelength ``wavelength_m``. A still target keeps a fixed
  bias; the bias changes only as the target moves across the view.

The link, the tracker, the scoring and the failure rule are S6's, reused
read-only. Everything is synthetic; nothing here is measured on hardware.

    PYTHONPATH=. python tools/sens_vision_error_shape.py --output <dir> --seeds 10 --control
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from math import exp, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from sim.link import LinkConfig, SimulatedLink
from sim.s6_recovery import (
    TIME_TOLERANCE_S,
    Delivery,
    RecoveryConfig,
    _burst_start_probability,
    paired_summary,
    run_arms,
    seed_list,
    summarize_arm,
    truth_position,
)
from sim.sensor import SensorConfig, SensorModel

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CSV = REPO_ROOT / "sim" / "runs" / "s6-recovery" / "per_seed.csv"

# The camera sits beside the straight-line path and looks across it, so the
# depth axis is y. Depth error is DEPTH_RATIO times lateral, for both the
# jitter and the bias, in every anisotropic model.
DEPTH_AXIS = 1
DEPTH_RATIO = 3.0
# Per-frame jitter, about the size Meyer et al. 2026 report for a 16 cm marker
# at 3 m (0.3 mm lateral, 2.1 mm depth), with lateral raised to keep the 3x ratio.
JITTER_LATERAL_M = 0.0007
JITTER_DEPTH_M = JITTER_LATERAL_M * DEPTH_RATIO
MAIN_TAU_S = 3.0
MAIN_WAVELENGTH_M = 2.0
POSITION_WAVES = 8

WHITE_ISO = "white_iso"
WHITE_ANISO = "white_aniso"
CONST_BIAS = "jitter_const_bias"
OU_BIAS = "jitter_ou_bias"
POS_BIAS = "jitter_position_bias"
MODELS = (WHITE_ISO, WHITE_ANISO, CONST_BIAS, OU_BIAS, POS_BIAS)
BIASED = (CONST_BIAS, OU_BIAS, POS_BIAS)

DELAYS = (0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
LOSSES = (0, 0.10, 0.30, 0.50)
NOISES = (0, 0.02, 0.05, 0.10)
MOTIONS = ("constant_velocity", "constant_turn")
TRACKER_STD_M = 0.05


def axis_totals(model: str, noise_std_m: float) -> np.ndarray:
    """Per-axis total RMS error (jitter and bias together) for one model.

    Every model keeps the 3D mean-square error at 3 s^2. The anisotropic ones
    split it as lateral^2 + (3 lateral)^2 + lateral^2 = 11 lateral^2, so
    lateral = s sqrt(3 / 11) and depth = 3 s sqrt(3 / 11).
    """
    s = float(noise_std_m)
    if model == WHITE_ISO:
        return np.full(3, s)
    if model in (WHITE_ANISO, *BIASED):
        lateral = s * sqrt(3.0 / (2.0 + DEPTH_RATIO**2))
        out = np.full(3, lateral)
        out[DEPTH_AXIS] = DEPTH_RATIO * lateral
        return out
    raise ValueError(f"model must be one of {MODELS}")


def jitter_and_bias_std(model: str, noise_std_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Split each axis total into a jitter std and a bias std (root-sum-square)."""
    total = axis_totals(model, noise_std_m)
    if model in (WHITE_ISO, WHITE_ANISO):
        return total, np.zeros(3)
    if noise_std_m == 0:
        return np.zeros(3), np.zeros(3)
    jitter = np.full(3, JITTER_LATERAL_M)
    jitter[DEPTH_AXIS] = JITTER_DEPTH_M
    if np.any(jitter > total):
        raise ValueError("jitter is larger than the total error for this noise cell")
    return jitter, np.sqrt(total**2 - jitter**2)


def error_series(model: str, noise_std_m: float, n: int, dt_s: float, rng: np.random.Generator,
                 tau_s: float = MAIN_TAU_S, truth_m: np.ndarray | None = None,
                 wavelength_m: float = MAIN_WAVELENGTH_M) -> np.ndarray:
    """Vision error for n frames, shape (n, 3), for every model except white_iso.

    Order of draws (fixed so runs repeat): the bias (start (3) then steps
    ((n - 1) x 3) for Gauss-Markov; directions then phases (3 x waves each) for
    the position model), then jitter (n x 3).
    """
    if model == WHITE_ISO:
        raise ValueError("white_iso goes through SensorModel, not error_series")
    jitter_std, bias_std = jitter_and_bias_std(model, noise_std_m)
    if model == WHITE_ANISO:
        return rng.normal(0.0, 1.0, size=(n, 3)) * jitter_std
    bias = np.empty((n, 3))
    if model != POS_BIAS:
        bias[0] = rng.normal(0.0, 1.0, size=3) * bias_std
    if model == CONST_BIAS:
        bias[1:] = bias[0]
    elif model == OU_BIAS:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        # Each frame keeps a fraction phi of the old bias and adds just enough
        # fresh noise that the bias std stays at bias_std forever.
        phi = exp(-dt_s / tau_s)
        steps = rng.normal(0.0, 1.0, size=(n - 1, 3)) * bias_std * sqrt(1.0 - phi * phi)
        for k in range(1, n):
            bias[k] = phi * bias[k - 1] + steps[k - 1]
    elif model == POS_BIAS:
        bias = position_bias(np.asarray(truth_m, dtype=float), bias_std, rng, wavelength_m)
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return bias + rng.normal(0.0, 1.0, size=(n, 3)) * jitter_std


def position_bias(positions_m: np.ndarray, bias_std: np.ndarray, rng: np.random.Generator,
                  wavelength_m: float = MAIN_WAVELENGTH_M, waves: int = POSITION_WAVES) -> np.ndarray:
    """A bias field over the ground plane, read at each position, shape (n, 3).

    Each axis is a sum of ``waves`` cosines in (x, y) with one wavelength,
    random directions and random phases, scaled by sqrt(2 / waves). One cosine
    with a random phase has variance 1/2, so the sum has variance 1 and the
    bias std over random fields is ``bias_std``. Near one spot the field is
    almost flat, so a still target sees a fixed bias.
    """
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    xy = positions_m[:, :2]
    k = 2.0 * np.pi / wavelength_m
    out = np.empty((xy.shape[0], 3))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(3, waves))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(3, waves))
    for axis in range(3):
        directions = np.stack([np.cos(angles[axis]), np.sin(angles[axis])], axis=1) * k
        out[:, axis] = (np.cos(xy @ directions.T + phases[axis]).sum(axis=1)
                        * sqrt(2.0 / waves) * bias_std[axis])
    return out


def deliver_shaped(motion: str, model: str, *, delay_s: float, loss_probability: float, noise_std_m: float,
                   seed: int, config: RecoveryConfig, tau_s: float = MAIN_TAU_S,
                   wavelength_m: float = MAIN_WAVELENGTH_M, loss_model: str = "independent", jitter_s: float = 0.0,
                   burst_mean_length_packets: float = 5.0) -> list[Delivery]:
    """Same as ``sim.s6_recovery.deliver`` with the vision error model swapped.

    The seed split, the link and the capture times are S6's, so for white_iso
    this returns exactly what S6 returns, and every model sees the same
    delivered/lost packets for the same seed.
    """
    link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
    if loss_model == "burst":
        end = 1.0 / float(burst_mean_length_packets)
        link_config = LinkConfig(delay_s=float(delay_s), delay_jitter_s=float(jitter_s), loss_model="burst",
                                 burst_start_probability=_burst_start_probability(float(loss_probability), end),
                                 burst_end_probability=end)
    elif loss_model == "independent":
        link_config = LinkConfig(delay_s=float(delay_s), delay_jitter_s=float(jitter_s), loss_model="independent",
                                 loss_probability=float(loss_probability))
    else:
        raise ValueError("loss_model must be independent or burst")
    link = SimulatedLink(link_config, seed=link_seed)
    n_captures = round(config.duration_s / config.observation_period_s)
    capture_times = np.arange(n_captures + 1) * config.observation_period_s
    truth = truth_position(motion, capture_times, config)
    if model == WHITE_ISO:
        sensor = SensorModel(SensorConfig(noise_std_m=float(noise_std_m)), seed=sensor_seed)
        rows = [np.asarray(sensor.sample(p, float(t)).measurement_m) for t, p in zip(capture_times, truth)]
        measured = np.asarray(rows)
    else:
        rng = np.random.default_rng(sensor_seed)
        measured = truth + error_series(model, noise_std_m, capture_times.size, config.observation_period_s,
                                        rng, tau_s, truth, wavelength_m)
    measurements = {}
    for seq, capture_time in enumerate(capture_times):
        link.send("target", "tracker", seq, float(capture_time))
        measurements[seq] = tuple(float(v) for v in measured[seq])
    deliveries = [
        Delivery(event.seq, float(event.send_time), float(event.receive_time), measurements[event.seq])
        for event in link.deliver_all()
        if event.receive_time <= config.duration_s + TIME_TOLERANCE_S
    ]
    deliveries.sort(key=lambda d: (d.receive_time_s, d.seq))
    return deliveries


# ---- grid --------------------------------------------------------------------

def build_cells(taus: tuple[float, ...] = (MAIN_TAU_S,), models: tuple[str, ...] = MODELS,
                wavelengths: tuple[float, ...] = (MAIN_WAVELENGTH_M,)) -> list[dict[str, Any]]:
    """The grid. Noise 0 is the same in every model, so it runs once (as white_iso).

    ``tau_s`` holds the Gauss-Markov correlation time or the position-bias wavelength.
    """
    variants = []
    for model in models:
        if model == OU_BIAS:
            variants.extend((model, float(t), f"/tau{t:g}") for t in taus)
        elif model == POS_BIAS:
            variants.extend((model, float(w), f"/L{w:g}") for w in wavelengths)
        else:
            variants.append((model, None, ""))
    cells = []
    for model, tau, tau_txt in variants:
        for motion in MOTIONS:
            for delay in DELAYS:
                for loss in LOSSES:
                    for noise in NOISES:
                        if noise == 0 and model != WHITE_ISO:
                            continue
                        cells.append({
                            "model": model,
                            "tau_s": tau,
                            "variant": model + tau_txt.replace("/", "_"),
                            "motion": motion,
                            "delay_s": float(delay),
                            "loss_probability": float(loss),
                            "noise_std_m": float(noise),
                            "cell_id": f"{model}{tau_txt}/{motion}/d{delay:.3f}/l{loss:.2f}/n{noise:.3f}",
                            # The S6 cell this one mirrors, for the control comparison.
                            "s6_cell_id": (f"main/{motion}/d{delay:.3f}/l{loss:.2f}/n{noise:.3f}/independent/"
                                           f"j0.000/r{TRACKER_STD_M:.3f}"),
                        })
    return cells


PER_SEED_KEYS = ("p95_error_m", "max_error_m", "rmse_m", "mean_nis", "fused_count", "dropped_count",
                 "fused_out_of_order_count", "failed")


def run_cell(cell: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    config = RecoveryConfig()
    arms_by_name: dict[str, list] = {"naive": [], "recovery": []}
    for seed in seeds:
        deliveries = deliver_shaped(cell["motion"], cell["model"], delay_s=cell["delay_s"],
                                    loss_probability=cell["loss_probability"], noise_std_m=cell["noise_std_m"],
                                    seed=seed, config=config,
                                    tau_s=cell["tau_s"] if cell["model"] == OU_BIAS else MAIN_TAU_S,
                                    wavelength_m=cell["tau_s"] if cell["model"] == POS_BIAS else MAIN_WAVELENGTH_M)
        arms = run_arms(deliveries, cell["motion"], config, tracker_measurement_std_m=TRACKER_STD_M)
        for name, collected in arms_by_name.items():
            collected.append(arms[name])
    per_seed = {name: [{key: getattr(r, key) for key in PER_SEED_KEYS} for r in results]
                for name, results in arms_by_name.items()}
    return {
        "cell_id": cell["cell_id"],
        "arms": {name: summarize_arm(results) for name, results in arms_by_name.items()},
        "paired": paired_summary(arms_by_name["naive"], arms_by_name["recovery"]),
        "per_seed": per_seed,
    }


# ---- control and summaries ---------------------------------------------------

def control_check(cells: list[dict[str, Any]], results: dict[str, dict], seeds: list[int],
                  registered_csv: Path = REGISTERED_CSV) -> dict[str, Any]:
    """Compare every white_iso cell with the registered S6 rows for the same seeds, value for value."""
    wanted = {c["s6_cell_id"]: c["cell_id"] for c in cells if c["model"] == WHITE_ISO}
    registered: dict[tuple[str, str, int], dict[str, str]] = {}
    with registered_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["cell_id"] in wanted and int(row["seed_index"]) < len(seeds):
                registered[(row["cell_id"], row["arm"], int(row["seed_index"]))] = row
    compared = mismatched = 0
    examples = []
    for s6_id, cell_id in wanted.items():
        for arm in ("naive", "recovery"):
            for index, seed in enumerate(seeds):
                reg = registered.get((s6_id, arm, index))
                if reg is None:
                    mismatched += 1
                    examples.append(f"missing registered row {s6_id} {arm} {index}")
                    continue
                if int(reg["seed"]) != seed:
                    mismatched += 1
                    examples.append(f"seed differs at {s6_id} {index}")
                    continue
                mine = results[cell_id]["per_seed"][arm][index]
                for key in PER_SEED_KEYS:
                    compared += 1
                    value = mine[key]
                    text = "" if value is None else ("true" if value is True else "false" if value is False
                                                     else repr(float(value)) if isinstance(value, float)
                                                     else str(value))
                    if text != reg[key]:
                        mismatched += 1
                        if len(examples) < 10:
                            examples.append(f"{s6_id} {arm} seed {index} {key}: {text} vs {reg[key]}")
    return {"cells": len(wanted), "values_compared": compared, "mismatches": mismatched, "examples": examples}


def tolerance_rows(cells: list[dict[str, Any]], results: dict[str, dict]) -> list[dict[str, Any]]:
    """Largest grid delay that passes together with every smaller delay, per model/motion/loss/noise/arm.

    Noise-0 rows are shared: they come from the single white_iso run.
    """
    by_key: dict[tuple, dict[float, dict]] = {}
    variants = sorted({(c["variant"], c["model"]) for c in cells})
    for cell in cells:
        res = results[cell["cell_id"]]
        targets = [cell["variant"]] if cell["noise_std_m"] > 0 else [v for v, _ in variants]
        for variant in targets:
            key = (variant, cell["motion"], cell["loss_probability"], cell["noise_std_m"])
            by_key.setdefault(key, {})[cell["delay_s"]] = res
    rows = []
    for (variant, motion, loss, noise), by_delay in sorted(by_key.items()):
        for arm in ("naive", "recovery"):
            tolerated = None
            for delay in DELAYS:
                if delay not in by_delay or by_delay[delay]["arms"][arm]["label"] != "passes":
                    break
                tolerated = float(delay)
            zero = by_delay.get(0.0)
            rows.append({
                "variant": variant, "motion": motion, "loss_probability": loss, "noise_std_m": noise, "arm": arm,
                "tolerated_delay_s": "" if tolerated is None else tolerated,
                "fail_rate_at_0ms": zero["arms"][arm]["fail_rate"] if zero else "",
                "p95_median_at_0ms_m": zero["arms"][arm]["p95_median_m"] if zero else "",
            })
    return rows


def bias_floor(model: str, noise_std_m: float, threshold_m: float = 0.25, n: int = 200_000,
               seed: int = 0) -> dict[str, float]:
    """Hand check of the error a filter cannot remove: the bias alone.

    Draw many bias vectors with this model's per-axis bias std and look at
    their 3D length. If the bias wanders across its whole range during a run,
    a seed's p95 error is at least about the 95th percentile of this length.
    For a constant bias, the share of draws longer than ``threshold_m`` is
    roughly the share of seeds that fail at zero delay.
    """
    _, bias_std = jitter_and_bias_std(model, noise_std_m)
    lengths = np.linalg.norm(np.random.default_rng(seed).normal(size=(n, 3)) * bias_std, axis=1)
    return {"p50_m": float(np.percentile(lengths, 50)), "p95_m": float(np.percentile(lengths, 95)),
            "share_over_threshold": float(np.mean(lengths > threshold_m))}


def seed_error_floor(variant: str, noise_std_m: float, motion: str, seeds: list[int],
                     threshold_m: float = 0.25) -> dict[str, float]:
    """Each seed's own p95 of the raw vision error over the scored window, before any filter.

    For the biased models it uses the same seed split, capture times and draws
    as ``deliver_shaped``, so the error is exactly what the tracker was fed.
    For white_iso the draws are statistically the same but not identical
    (``SensorModel`` interleaves a dropout draw). No filter removes a bias,
    so a seed whose raw p95 is above ``threshold_m`` is expected to fail at any
    delay; the filter's own error usually adds to it (on a turn it can
    occasionally pull a seed's p95 a few centimetres below the raw value).
    Raw errors are taken at the 10 Hz capture times; S6 scores 20 ms queries.
    """
    config = RecoveryConfig()
    model, _, extra = variant.partition("_tau") if "_tau" in variant else variant.partition("_L")
    value = float(extra) if extra else None
    n_captures = round(config.duration_s / config.observation_period_s)
    times = np.arange(n_captures + 1) * config.observation_period_s
    truth = truth_position(motion, times, config)
    scored = times >= config.burn_in_s - 1e-9
    p95s = []
    for seed in seeds:
        _, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
        rng = np.random.default_rng(sensor_seed)
        if model == WHITE_ISO:
            raw = rng.normal(0.0, float(noise_std_m), size=truth.shape)
        else:
            raw = error_series(model, noise_std_m, times.size, config.observation_period_s, rng,
                               value if model == OU_BIAS else MAIN_TAU_S, truth,
                               value if model == POS_BIAS else MAIN_WAVELENGTH_M)
        p95s.append(float(np.percentile(np.linalg.norm(raw[scored], axis=1), 95)))
    arr = np.asarray(p95s)
    return {"median_seed_p95_m": float(np.median(arr)), "share_seeds_over_threshold": float(np.mean(arr > threshold_m))}


FIGURE_VARIANTS = (
    (WHITE_ISO, "white noise, isotropic (S6 today)", "textSecondary", "-"),
    (WHITE_ANISO, "white noise, depth 3x", "textSecondary", "--"),
    (CONST_BIAS, "jitter + constant bias", "accent", "-"),
    (OU_BIAS + "_tau3", "jitter + Gauss-Markov bias, tau 3 s", "keepOut", "-"),
    (POS_BIAS + "_L2", "jitter + position bias, 2 m waves (post hoc)", "target", "-"),
)


def make_figure(summary_path: Path, out_png: Path, loss: float = 0.0, arm: str = "recovery") -> None:
    """Fail rate against delay at one loss, one panel per motion and noise cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    summary = json.loads(summary_path.read_text())
    index = {}
    for cell in summary["cells"]:
        if cell["loss_probability"] == loss:
            index[(cell["variant"], cell["motion"], cell["noise_std_m"], cell["delay_s"])] = cell
    noises = [n for n in NOISES if n > 0]
    fig, axes = plt.subplots(2, len(noises), figsize=(11, 6.2), sharex=True, sharey=True)
    for row, motion in enumerate(MOTIONS):
        for col, noise in enumerate(noises):
            ax = axes[row, col]
            ax.axhline(0.5, color=pal["alert"], linewidth=0.8, linestyle=":")
            for variant, label, color, style in FIGURE_VARIANTS:
                xs, ys = [], []
                for delay in DELAYS:
                    cell = index.get((variant, motion, noise, delay))
                    if cell is not None:
                        xs.append(delay * 1000)
                        ys.append(cell["arms"][arm]["fail_rate"])
                if xs:
                    ax.plot(xs, ys, linestyle=style, marker="o", markersize=3, linewidth=1.4,
                            color=pal[color], label=label)
            title_motion = "straight line" if motion == "constant_velocity" else "turn, r = 1 m"
            ax.set_title(f"{title_motion}, S6 noise cell {noise * 100:.0f} cm")
            ax.set_ylim(-0.03, 1.03)
            if row == 1:
                ax.set_xlabel("one-way delay (ms)")
            if col == 0:
                ax.set_ylabel(f"share of seeds failing ({arm})")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, frameon=False)
    n_seeds = summary["design"]["n_seeds"]
    fig.suptitle(f"Same total vision error, different shape: S6 failure share vs delay "
                 f"(loss {loss * 100:.0f} %, {n_seeds} seeds, synthetic). Dotted line: cell labelled fails at or above it.",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, *, seeds_count: int, taus: tuple[float, ...], processes: int, control: bool,
        argv: list[str], wavelengths: tuple[float, ...] = (MAIN_WAVELENGTH_M,)) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cells = build_cells(taus, wavelengths=wavelengths)
    seeds = seed_list(seeds_count)
    checkpoint = output / "checkpoint.jsonl"
    # A checkpoint line is reused only if it came from this exact tool file.
    code_sha = _sha256(Path(__file__))
    results: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("n_seeds") == seeds_count and record.get("code_sha256") == code_sha:
                    results[record["cell_id"]] = record
    todo = [c for c in cells if c["cell_id"] not in results]
    print(f"cells: {len(cells)} (done {len(results)}, to run {len(todo)}), seeds {seeds_count}", file=sys.stderr)
    start = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=processes) as pool, checkpoint.open("a") as sink:
            futures = {pool.submit(run_cell, cell, seeds): cell for cell in todo}
            for done, future in enumerate(as_completed(futures), 1):
                record = future.result()
                record["n_seeds"] = seeds_count
                record["code_sha256"] = code_sha
                results[record["cell_id"]] = record
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)} cells, {time.perf_counter() - start:.0f} s", file=sys.stderr)
    runtime = time.perf_counter() - start

    control_result = control_check(cells, results, seeds) if control else None
    tol = tolerance_rows(cells, results)
    with (output / "tolerance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tol[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tol)
    summary = {
        "schema": "zephyr-vision-error-shape-1",
        "status": "synthetic sensitivity study: simulated link and sensor, not flight performance",
        "generator": "tools/sens_vision_error_shape.py",
        "config": asdict(RecoveryConfig()),
        "design": {
            "n_seeds": seeds_count, "seeds": seeds, "taus_s": list(taus), "models": list(MODELS),
            "wavelengths_m": list(wavelengths), "position_waves": POSITION_WAVES,
            "depth_axis": DEPTH_AXIS, "depth_ratio": DEPTH_RATIO,
            "jitter_lateral_m": JITTER_LATERAL_M, "jitter_depth_m": JITTER_DEPTH_M,
            "tracker_measurement_std_m": TRACKER_STD_M,
            "equal_size_rule": "every model has 3D mean-square vision error 3 s^2 for S6 noise cell s",
            "criterion": "S6: a seed fails if p95 > 0.25 m or max > 0.50 m; a cell fails if at least half its seeds fail",
        },
        "control": control_result,
        "cells": [{**cell, "arms": results[cell["cell_id"]]["arms"], "paired": results[cell["cell_id"]]["paired"]}
                  for cell in cells],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "argv": argv, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "runtime_s_this_call": runtime, "tool_sha256": code_sha, "n_cells": len(cells), "n_seeds": seeds_count, "processes": processes,
        "outputs": {name: _sha256(output / name) for name in ("summary.json", "tolerance.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if control_result is not None:
        print(f"control: {control_result['values_compared']} values compared, "
              f"{control_result['mismatches']} mismatches", file=sys.stderr)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-only", action="store_true",
                        help="only draw the figure and bias-floor table from <output>/summary.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=[MAIN_TAU_S])
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[MAIN_WAVELENGTH_M])
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--control", action="store_true", help="compare white_iso with the registered S6 rows")
    args = parser.parse_args(argv)
    if args.figure_only:
        make_figure(args.output / "summary.json", args.output / "vision_error_shape.png")
        # The jitter/bias split is the same for every biased model, so one bias floor covers b, c and d.
        floors = {f"biased_models/n{n:.3f}": bias_floor(CONST_BIAS, n) for n in NOISES if n > 0}
        (args.output / "bias_floor.json").write_text(json.dumps(floors, indent=2, sort_keys=True) + "\n")
        summary = json.loads((args.output / "summary.json").read_text())
        seeds = summary["design"]["seeds"]
        variants = sorted({c["variant"] for c in summary["cells"]})
        seed_floor = {f"{v}/{m}/n{n:.3f}": seed_error_floor(v, n, m, seeds)
                      for v in variants for m in MOTIONS for n in (0.05, 0.10)}
        (args.output / "seed_floor.json").write_text(json.dumps(seed_floor, indent=2, sort_keys=True) + "\n")
        print(json.dumps(floors, indent=2))
        return 0
    summary = run(args.output, seeds_count=args.seeds, taus=tuple(args.taus), processes=args.processes,
                  control=args.control, wavelengths=tuple(args.wavelengths), argv=list(sys.argv[1:] if argv is None else argv))
    control = summary["control"]
    return 1 if control is not None and control["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
