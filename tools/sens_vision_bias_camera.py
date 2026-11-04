"""Camera-shaped vision bias for the S6 vision error shape study.

``tools/sens_vision_error_shape.py`` found that at the 10 cm S6 noise cell a
bias that drifts during the run makes the tracker fail even at zero delay. Its
drifting models (Gauss-Markov, random plane waves) change by centimetres per
frame on the simulated paths, which is not how a webcam behaves. A webcam's
accuracy error comes mostly from calibration and lens-distortion residuals,
so it is a smooth function of where the marker sits in the image and how far
away it is, and it changes by millimetres per frame as the marker moves.

This tool adds that shape and reruns the 5 cm and 10 cm cells:

- ``camera_field``: per axis, a random low-order polynomial in the marker's
  normalised image column ``u`` and relative range ``rho``
  (terms 1, u, rho, u^2, u rho, rho^2; coefficients N(0, 1)), scaled so its
  mean square over the path and over seeds is the bias variance of model b,
  plus model b's mm-level jitter.
- ``camera_field_noconst``: the same field with the constant term removed,
  so all of the bias varies across the image (the harsher bracket).

Camera geometry: a 70 degree horizontal field of view, the camera at the
target's height on the -y side looking along +y (y is the depth axis, as in
the merged tool). For the straight line the camera is centred on the 12 m
path and far enough back (8.57 m) to keep all of it in view, so the path
sweeps the whole image width (the fastest sweep a fixed camera allows). For
the turn the camera sits 3 m from the circle's centre. The target stays at
camera height, so the image row is 0 and drops out of the polynomial.

Models a, a2 and b are run through the merged tool's own ``run_cell``, so
their outputs can be compared byte for byte with the merged run. The merged
tool is imported read-only and not changed.

    PYTHONPATH=. python tools/sens_vision_bias_camera.py --output <dir> --seeds 30 --control \
        --merged <merged run dir>
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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from math import exp, radians, sqrt, tan
from pathlib import Path
from typing import Any

import numpy as np

from sim.link import LinkConfig, SimulatedLink
from sim.s6_recovery import (
    TIME_TOLERANCE_S,
    Delivery,
    RecoveryConfig,
    paired_summary,
    run_arms,
    seed_list,
    summarize_arm,
    truth_position,
)
from tools import sens_vision_error_shape as ves

CAMERA = "camera_field"
CAMERA_NC = "camera_field_noconst"
CAMERA_MODELS = (CAMERA, CAMERA_NC)
REFERENCE_MODELS = (ves.WHITE_ISO, ves.WHITE_ANISO, ves.CONST_BIAS)
NOISES = (0.05, 0.10)

HFOV_DEG = 70.0
TAN_HALF_FOV = tan(radians(HFOV_DEG / 2.0))
LINE_LENGTH_M = RecoveryConfig().target_speed_mps * RecoveryConfig().duration_s
# Camera position per motion (x, y, z). Line: centred on x = 0..12 m, back far
# enough that the path just fills the field of view. Turn: 3 m from the centre
# of the circle (0, 1, 0), so the range runs from 2 m to 4 m.
CAMERA_POSITIONS = {
    "constant_velocity": np.array([LINE_LENGTH_M / 2.0, -(LINE_LENGTH_M / 2.0) / TAN_HALF_FOV, 0.0]),
    "constant_turn": np.array([0.0, -2.0, 0.0]),
}
TERMS = ("1", "u", "rho", "u^2", "u*rho", "rho^2")


def image_coords(positions_m: np.ndarray, motion: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalised image column u (1 at the edge of the view) and relative range rho.

    u = (lateral offset / depth) / tan(half field of view). rho = range / R - 1,
    with R the camera's distance to the path centre, so a 1 % focal-length
    error, which scales depth by 1 %, is a term linear in rho.
    """
    cam = CAMERA_POSITIONS[motion]
    rel = np.asarray(positions_m, dtype=float) - cam
    u = (rel[:, 0] / rel[:, ves.DEPTH_AXIS]) / TAN_HALF_FOV
    centre = np.array([LINE_LENGTH_M / 2.0, 0.0, 0.0]) if motion == "constant_velocity" else np.array([0.0, 1.0, 0.0])
    ref = float(np.linalg.norm(centre - cam))
    rho = np.linalg.norm(rel, axis=1) / ref - 1.0
    return u, rho


def basis(u: np.ndarray, rho: np.ndarray, constant: bool = True) -> np.ndarray:
    """The polynomial terms (1, u, rho, u^2, u rho, rho^2), shape (n, 6)."""
    out = np.stack([np.ones_like(u), u, rho, u * u, u * rho, rho * rho], axis=1)
    if not constant:
        out[:, 0] = 0.0
    return out


def camera_bias(model: str, noise_std_m: float, truth_m: np.ndarray, motion: str,
                rng: np.random.Generator) -> np.ndarray:
    """The bias field read along the path, shape (n, 3). Draws 3 x 6 coefficients.

    With coefficients N(0, 1), the bias variance at a frame is the sum of the
    squared terms there. Dividing by the path average of that sum makes the
    mean square over the path and over seeds equal to model b's bias variance.
    Both camera models draw the same coefficients, so they pair seed by seed.
    """
    if model not in CAMERA_MODELS:
        raise ValueError(f"model must be one of {CAMERA_MODELS}")
    _, bias_std = ves.jitter_and_bias_std(ves.CONST_BIAS, noise_std_m)
    coeffs = rng.normal(0.0, 1.0, size=(3, len(TERMS)))
    phi = basis(*image_coords(truth_m, motion), constant=(model == CAMERA))
    norm = sqrt(float(np.mean(np.sum(phi * phi, axis=1))))
    return (phi @ coeffs.T) / norm * bias_std


def camera_error_series(model: str, noise_std_m: float, truth_m: np.ndarray, motion: str,
                        rng: np.random.Generator) -> np.ndarray:
    """Camera bias plus model b's jitter. Draw order: coefficients, then jitter (n x 3)."""
    jitter_std, _ = ves.jitter_and_bias_std(ves.CONST_BIAS, noise_std_m)
    bias = camera_bias(model, noise_std_m, truth_m, motion, rng)
    return bias + rng.normal(0.0, 1.0, size=bias.shape) * jitter_std


def deliver_with_error(motion: str, error_fn: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray], *,
                       delay_s: float, loss_probability: float, seed: int, config: RecoveryConfig) -> list[Delivery]:
    """``ves.deliver_shaped`` (independent loss, no delay jitter) with any error function.

    ``error_fn(truth, capture_times, rng)`` returns the (n, 3) error. The seed
    split and the link are the same, so every model sees the same packets.
    """
    link_seed, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
    link = SimulatedLink(LinkConfig(delay_s=float(delay_s), delay_jitter_s=0.0, loss_model="independent",
                                    loss_probability=float(loss_probability)), seed=link_seed)
    n_captures = round(config.duration_s / config.observation_period_s)
    capture_times = np.arange(n_captures + 1) * config.observation_period_s
    truth = truth_position(motion, capture_times, config)
    measured = truth + error_fn(truth, capture_times, np.random.default_rng(sensor_seed))
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

def build_cells() -> list[dict[str, Any]]:
    """Reference cells (a, a2, b) from the merged tool's grid, plus the camera cells, 5 and 10 cm only."""
    cells = [c for c in ves.build_cells(models=REFERENCE_MODELS) if c["noise_std_m"] in NOISES]
    for model in CAMERA_MODELS:
        for motion in ves.MOTIONS:
            for delay in ves.DELAYS:
                for loss in ves.LOSSES:
                    for noise in NOISES:
                        cells.append({
                            "model": model, "tau_s": None, "variant": model, "motion": motion,
                            "delay_s": float(delay), "loss_probability": float(loss), "noise_std_m": float(noise),
                            "cell_id": f"{model}/{motion}/d{delay:.3f}/l{loss:.2f}/n{noise:.3f}",
                            "s6_cell_id": (f"main/{motion}/d{delay:.3f}/l{loss:.2f}/n{noise:.3f}/independent/"
                                           f"j0.000/r{ves.TRACKER_STD_M:.3f}"),
                        })
    return cells


def run_camera_cell(cell: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Same as ``ves.run_cell`` for a camera model."""
    config = RecoveryConfig()
    model, motion, noise = cell["model"], cell["motion"], cell["noise_std_m"]
    arms_by_name: dict[str, list] = {"naive": [], "recovery": []}
    for seed in seeds:
        deliveries = deliver_with_error(
            motion, lambda truth, _times, rng: camera_error_series(model, noise, truth, motion, rng),
            delay_s=cell["delay_s"], loss_probability=cell["loss_probability"], seed=seed, config=config)
        arms = run_arms(deliveries, motion, config, tracker_measurement_std_m=ves.TRACKER_STD_M)
        for name, collected in arms_by_name.items():
            collected.append(arms[name])
    per_seed = {name: [{key: getattr(r, key) for key in ves.PER_SEED_KEYS} for r in results]
                for name, results in arms_by_name.items()}
    return {
        "cell_id": cell["cell_id"],
        "arms": {name: summarize_arm(results) for name, results in arms_by_name.items()},
        "paired": paired_summary(arms_by_name["naive"], arms_by_name["recovery"]),
        "per_seed": per_seed,
    }


def run_any_cell(cell: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    return run_camera_cell(cell, seeds) if cell["model"] in CAMERA_MODELS else ves.run_cell(cell, seeds)


# ---- checks --------------------------------------------------------------------

RECORD_KEYS = ("cell_id", "arms", "paired", "per_seed")


def _record_text(record: dict[str, Any]) -> str:
    return json.dumps({k: record[k] for k in RECORD_KEYS}, sort_keys=True)


def merged_identity_check(cells: list[dict[str, Any]], results: dict[str, dict], merged_dir: Path) -> dict[str, Any]:
    """Byte-for-byte check of models a, a2, b against the merged run's checkpoint and tolerance table."""
    wanted = {c["cell_id"] for c in cells if c["model"] in REFERENCE_MODELS}
    merged: dict[str, str] = {}
    with (merged_dir / "checkpoint.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["cell_id"] in wanted:
                    merged[record["cell_id"]] = _record_text(record)
    cell_mismatch = [cid for cid in sorted(wanted) if merged.get(cid) != _record_text(results[cid])]

    def tol_lines(path: Path) -> list[str]:
        keep = {"white_iso", "white_aniso", "jitter_const_bias"}
        lines = path.read_text().splitlines()
        out = [ln for ln in lines[1:] if ln.split(",")[0] in keep and float(ln.split(",")[3]) in NOISES]
        return [lines[0], *out]

    mine, theirs = tol_lines(Path(results["__tolerance_csv__"])), tol_lines(merged_dir / "tolerance.csv")
    return {"cells_compared": len(wanted), "cells_byte_identical": len(wanted) - len(cell_mismatch),
            "cell_mismatches": cell_mismatch[:10],
            "tolerance_rows_compared": len(theirs) - 1, "tolerance_rows_identical": mine == theirs}


def bias_change_stats(model: str, noise_std_m: float, motion: str, draws: int = 4000, seed: int = 7) -> dict[str, float]:
    """How fast the bias moves: RMS frame-to-frame change and RMS change across the run, per axis.

    Depth (y) is the axis that matters; lateral is listed for completeness.
    For comparison it also gives the merged run's drifting models on the
    same path: Gauss-Markov tau 3 s (exact formula) and 2 m plane waves
    (sampled).
    """
    config = RecoveryConfig()
    times = np.arange(round(config.duration_s / config.observation_period_s) + 1) * config.observation_period_s
    truth = truth_position(motion, times, config)
    rng = np.random.default_rng(seed)
    if model in CAMERA_MODELS:
        fields = np.stack([camera_bias(model, noise_std_m, truth, motion, rng) for _ in range(draws)])
    elif model == ves.POS_BIAS:
        _, bias_std = ves.jitter_and_bias_std(ves.CONST_BIAS, noise_std_m)
        fields = np.stack([ves.position_bias(truth, bias_std, rng) for _ in range(draws)])
    elif model == ves.OU_BIAS:
        _, bias_std = ves.jitter_and_bias_std(ves.CONST_BIAS, noise_std_m)
        phi = exp(-config.observation_period_s / ves.MAIN_TAU_S)
        step = bias_std * sqrt(2.0 * (1.0 - phi))
        return {"frame_step_rms_depth_m": float(step[ves.DEPTH_AXIS]), "frame_step_rms_lateral_m": float(step[0]),
                "note": "exact: bias_std sqrt(2 (1 - exp(-dt / tau)))"}
    else:
        raise ValueError("bias_change_stats covers the camera models, the position bias and Gauss-Markov")
    steps = np.diff(fields, axis=1)
    step_rms = np.sqrt(np.mean(steps**2, axis=(0, 1)))
    spread = fields - fields.mean(axis=1, keepdims=True)
    spread_rms = np.sqrt(np.mean(spread**2, axis=(0, 1)))
    total_rms = np.sqrt(np.mean(fields**2, axis=(0, 1)))
    return {"frame_step_rms_depth_m": float(step_rms[ves.DEPTH_AXIS]), "frame_step_rms_lateral_m": float(step_rms[0]),
            "within_run_spread_rms_depth_m": float(spread_rms[ves.DEPTH_AXIS]),
            "bias_rms_depth_m": float(total_rms[ves.DEPTH_AXIS]),
            "varying_share_of_variance_depth": float((spread_rms[ves.DEPTH_AXIS] / total_rms[ves.DEPTH_AXIS]) ** 2)}


def seed_error_floor(model: str, noise_std_m: float, motion: str, seeds: list[int],
                     threshold_m: float = 0.25) -> dict[str, float]:
    """Each seed's own p95 of the raw vision error over the scored window (same draws the tracker saw)."""
    if model not in CAMERA_MODELS:
        return ves.seed_error_floor(model, noise_std_m, motion, seeds, threshold_m)
    config = RecoveryConfig()
    times = np.arange(round(config.duration_s / config.observation_period_s) + 1) * config.observation_period_s
    truth = truth_position(motion, times, config)
    scored = times >= config.burn_in_s - 1e-9
    p95s = []
    for seed in seeds:
        _, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
        raw = camera_error_series(model, noise_std_m, truth, motion, np.random.default_rng(sensor_seed))
        p95s.append(float(np.percentile(np.linalg.norm(raw[scored], axis=1), 95)))
    arr = np.asarray(p95s)
    return {"median_seed_p95_m": float(np.median(arr)), "share_seeds_over_threshold": float(np.mean(arr > threshold_m))}


# ---- figure --------------------------------------------------------------------

FIGURE_VARIANTS = (
    (ves.WHITE_ANISO, "white noise, depth 3x (a2)", "textSecondary", "--", 1.3),
    (ves.CONST_BIAS, "jitter + constant bias (b)", "accent", "-", 1.6),
    (CAMERA, "jitter + camera field (new)", "ok", "-", 2.0),
    (CAMERA_NC, "jitter + camera field, no constant (new)", "target", "-", 2.0),
    (ves.OU_BIAS + "_tau3", "Gauss-Markov tau 3 s (merged run)", "keepOut", ":", 1.1),
    (ves.POS_BIAS + "_L2", "plane waves 2 m (merged run)", "lineStrong", ":", 1.1),
)


def make_figure(summary_path: Path, merged_summary_path: Path, out_png: Path, loss: float = 0.0,
                arm: str = "recovery") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    index = {}
    for path in (merged_summary_path, summary_path):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["loss_probability"] == loss and cell["noise_std_m"] in NOISES:
                index[(cell["variant"], cell["motion"], cell["noise_std_m"], cell["delay_s"])] = cell
    fig, axes = plt.subplots(2, len(NOISES), figsize=(9.5, 6.4), sharex=True, sharey=True)
    for row, motion in enumerate(ves.MOTIONS):
        for col, noise in enumerate(NOISES):
            ax = axes[row, col]
            ax.axhline(0.5, color=pal["alert"], linewidth=0.8, linestyle=":")
            for variant, label, color, style, width in FIGURE_VARIANTS:
                pts = [(d * 1000, index[(variant, motion, noise, d)]["arms"][arm]["fail_rate"])
                       for d in ves.DELAYS if (variant, motion, noise, d) in index]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linestyle=style, marker="o", markersize=3, linewidth=width,
                            color=pal[color], label=label)
            title_motion = "straight line" if motion == "constant_velocity" else "turn, r = 1 m"
            ax.set_title(f"{title_motion}, S6 noise cell {noise * 100:.0f} cm")
            ax.set_ylim(-0.03, 1.03)
            if row == 1:
                ax.set_xlabel("one-way delay (ms)")
            if col == 0:
                ax.set_ylabel(f"share of seeds failing ({arm})")
    axes[0, 0].legend(loc="upper left", fontsize=7.2, frameon=False)
    fig.suptitle(f"Camera-shaped bias vs other error shapes at equal total RMS (loss {loss * 100:.0f} %, 30 seeds, "
                 f"synthetic). Dotted red: cell labelled fails at or above it.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---- driver --------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, *, seeds_count: int, processes: int, control: bool, merged_dir: Path | None,
        argv: list[str]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    seeds = seed_list(seeds_count)
    checkpoint = output / "checkpoint.jsonl"
    code_sha = _sha256(Path(__file__)) + ":" + _sha256(Path(ves.__file__))
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
            futures = {pool.submit(run_any_cell, cell, seeds): cell for cell in todo}
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

    tol = ves.tolerance_rows(cells, results)
    tol_path = output / "tolerance.csv"
    with tol_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tol[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tol)
    controls: dict[str, Any] = {}
    if control:
        controls["s6_registered_rows"] = ves.control_check(cells, results, seeds)
        if merged_dir is not None:
            controls["merged_run_identity"] = merged_identity_check(cells, {**results, "__tolerance_csv__": str(tol_path)},
                                                                    merged_dir)
    summary = {
        "schema": "zephyr-vision-bias-camera-1",
        "status": "synthetic sensitivity study: simulated link and sensor, not flight performance",
        "generator": "tools/sens_vision_bias_camera.py (imports tools/sens_vision_error_shape.py)",
        "config": asdict(RecoveryConfig()),
        "design": {
            "n_seeds": seeds_count, "seeds": seeds, "noises_m": list(NOISES),
            "reference_models": list(REFERENCE_MODELS), "camera_models": list(CAMERA_MODELS),
            "terms": list(TERMS), "hfov_deg": HFOV_DEG,
            "camera_positions_m": {k: v.tolist() for k, v in CAMERA_POSITIONS.items()},
            "depth_axis": ves.DEPTH_AXIS, "depth_ratio": ves.DEPTH_RATIO,
            "jitter_lateral_m": ves.JITTER_LATERAL_M, "jitter_depth_m": ves.JITTER_DEPTH_M,
            "equal_size_rule": "every model has 3D mean-square vision error 3 s^2 for S6 noise cell s",
            "criterion": "S6: a seed fails if p95 > 0.25 m or max > 0.50 m; a cell fails if at least half its seeds fail",
        },
        "control": controls or None,
        "cells": [{**cell, "arms": results[cell["cell_id"]]["arms"], "paired": results[cell["cell_id"]]["paired"]}
                  for cell in cells],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "argv": argv, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "runtime_s_this_call": runtime, "code_sha256": code_sha, "n_cells": len(cells), "n_seeds": seeds_count,
        "processes": processes, "outputs": {name: _sha256(output / name) for name in ("summary.json", "tolerance.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(controls, indent=2), file=sys.stderr)
    return summary


def write_checks(output: Path, seeds: list[int]) -> dict[str, Any]:
    """Bias speed and per-seed raw error floors (cheap; no tracker runs)."""
    speeds = {}
    for motion in ves.MOTIONS:
        for noise in NOISES:
            for model in (*CAMERA_MODELS, ves.POS_BIAS, ves.OU_BIAS):
                speeds[f"{model}/{motion}/n{noise:.3f}"] = bias_change_stats(model, noise, motion)
    floors = {f"{m}/{motion}/n{n:.3f}": seed_error_floor(m, n, motion, seeds)
              for m in (*REFERENCE_MODELS, *CAMERA_MODELS) for motion in ves.MOTIONS for n in NOISES}
    out = {"bias_change": speeds, "seed_floor": floors}
    (output / "checks.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--merged", type=Path, default=None, help="merged vision error shape run folder")
    parser.add_argument("--checks-only", action="store_true", help="only write checks.json (bias speed, seed floors)")
    parser.add_argument("--figure-only", action="store_true", help="only draw the figure (needs --merged)")
    args = parser.parse_args(argv)
    if args.checks_only:
        print(json.dumps(write_checks(args.output, seed_list(args.seeds)), indent=2))
        return 0
    if args.figure_only:
        make_figure(args.output / "summary.json", args.merged / "summary.json", args.output / "vision_bias_camera.png")
        return 0
    summary = run(args.output, seeds_count=args.seeds, processes=args.processes, control=args.control,
                  merged_dir=args.merged, argv=list(sys.argv[1:] if argv is None else argv))
    control = summary["control"] or {}
    bad = control.get("s6_registered_rows", {}).get("mismatches", 0)
    ident = control.get("merged_run_identity")
    if ident is not None and (ident["cells_byte_identical"] != ident["cells_compared"] or not ident["tolerance_rows_identical"]):
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
