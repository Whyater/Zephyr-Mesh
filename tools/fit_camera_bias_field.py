"""Split a camera's position bias into a common offset and a part that varies across the image.

Bench test: put the marker still at known spots across the image (at least 5
columns, at 2 or 3 ranges), record a detection log, and run

    PYTHONPATH=. python tools/fit_camera_bias_field.py run.jsonl --spots spots.csv --output fit.json

It prints a short text summary and writes the numbers as JSON
(``zephyr-camera-bias-field-1``). The verdict line places the result on the
regimes found by the camera-shaped bias simulation
(``tools/sens_vision_bias_camera.py``). Those regimes are simulation results
on synthetic data, not measurements.

STATUS: a proposal, tested on synthetic logs only. It has never seen a real
camera log.

Inputs
------
Detection logs in the ``zephyr-detection-log-1`` format (docs/DETECTION_LOG.md),
read with ``sim.detection_log.read_detection_log``. Give one log for the whole
session, or one log per spot (as ``tools/capture_aruco.py`` writes them).

The spots file is a CSV with a header row:

    spot_id,x_m,y_m,z_m              (match lines to spots by ground truth)
    spot_id,x_m,y_m,z_m,start_s,end_s (match lines to spots by capture time)

``x_m, y_m, z_m`` is the known (taped) marker centre, in the same frame as the
log's ``position_m``. For ``tools/capture_aruco.py`` that is the camera frame:
x to the right, y down, z straight out of the lens.

- Without time columns, every detection must carry ``ground_truth.position_m``,
  and each one must be within ``GT_MATCH_TOL_M`` (5 mm) of exactly one spot.
- With ``start_s`` and ``end_s`` (capture times, inclusive), a detection
  belongs to the spot whose window holds its ``capture_time_s``; detections
  outside every window (the marker being moved) are skipped. This mode needs
  exactly one log, because separate logs do not share a clock. If a detection
  also has ground truth, it must match its window's spot.

Only ``detected`` lines with a ``position_m`` are used. ``missed`` and
``unknown`` lines are counted and skipped.

Definitions (all in metres, each spot weighted equally)
-------------------------------------------------------
For spot i with n_i frames, per axis k (x, y, z), the error of one frame is
measured position minus the spot's known position. Then:

- ``mean_error_m[i, k]`` e_ik: the mean of the frame errors at spot i. This is
  the bias at that spot.
- ``jitter_m[i, k]``: the sample standard deviation (n - 1 in the denominator)
  of the frame errors at spot i.
- ``common_m[k]`` c_k = (1 / S) * sum over spots of e_ik, where S is the
  number of spots. The common offset.
- ``varying_m[k]`` v_k = sqrt((1 / S) * sum over spots of (e_ik - c_k)^2).
  The RMS over spots of each spot's bias minus the common offset.
- ``bias_rms_m[k]`` = sqrt(c_k^2 + v_k^2), which equals the RMS over spots of
  e_ik (the two parts add in squares, exactly).
- ``varying_share[k]`` = v_k^2 / (c_k^2 + v_k^2): the share of the bias mean
  square that varies across the spots. It is ``None`` when the axis has no
  bias at all. This is the same split the simulation reported ("share of the
  depth bias variance that varies within a run").
- ``varying_share_3d`` = sum_k v_k^2 / sum_k (c_k^2 + v_k^2), and
  ``varying_share_depth`` = ``varying_share`` on the depth axis (z by default,
  ``--depth-axis`` to change). The verdict uses the depth share, as the
  simulation did.
- ``jitter_rms_m[k]`` = sqrt((1 / S) * sum over spots of jitter_ik^2).
- ``accuracy_rms_3d_m`` = sqrt(sum_k (c_k^2 + v_k^2)): the RMS over spots of
  the length of the per-spot bias vector. Bias only.
- ``total_rms_3d_m`` = sqrt(sum_k (c_k^2 + v_k^2 + jitter_rms_k^2)): bias and
  jitter together. The simulation's noise cells were sized on this total
  (3D mean square = 3 s^2 for a cell of size s, jitter included), so the
  verdict compares this number with sqrt(3) x 5 cm = 8.66 cm and
  sqrt(3) x 10 cm = 17.32 cm.
- ``varying_noise_floor_m[k]`` = sqrt((1 / S) * sum over spots of
  jitter_ik^2 / n_i): about how much varying part jitter alone would put into
  the per-spot means if frames were independent. Still frames are usually
  correlated, so the real floor is higher; treat a varying part near this
  number as zero.

Refusals (exit code 2, with the reason)
---------------------------------------
- Fewer than ``MIN_SPOTS`` = 5 spots. The bench design asks for at least 5
  columns across the image, and the varying part needs several spots to mean
  anything (with 2 spots it is just half their difference).
- Any spot with fewer than ``MIN_FRAMES_PER_SPOT`` = 60 usable frames (2 s at
  30 frames per second). Why 60: the per-spot mean must be limited by the
  tape measure (about 1 cm), not by camera jitter. The simulation used 0.7 mm
  lateral and 2.1 mm depth jitter (published webcam ArUco numbers). Even at
  5 mm jitter, with neighbouring frames correlated enough to cut the
  effective count by 4, 60 frames give a standard error of
  5 mm / sqrt(15) = 1.3 mm, well under the tape. The smallest varying part the
  verdict separates is centimetres (about 8 cm in depth at the 10 cm cell for
  a 28 % share), so 60 frames is enough; fewer lets a few bad frames move a
  spot's mean.
- Spots that are not distinguishable:
  - two known positions closer than ``MIN_SPOT_SEPARATION_M`` = 5 cm (they
    sample the same patch of the image; at 1.5 m, 5 cm is under 2 degrees);
  - in time mode, windows that overlap or run backwards;
  - two spots whose measured mean positions are closer than
    ``CLUSTER_SEPARATION_JITTERS`` = 4 times the larger of their 3D jitter
    RMS (the two clouds of frames run into each other, so the marker moved or
    the spots are mixed up);
  - in ground-truth mode, a detection whose ground truth matches no spot, or
    a detection with no ground truth.
- More than one ``subject_id`` without ``--subject``, mixed ``position_frame``
  values, or time windows with more than one log.

Limits
------
- No intervals. The tool reports point values and the jitter noise floor, and
  adds a ``notes`` line when an axis's bias is within 3 x that floor (its
  share is then noise), when the depth varying part is not above 2 x the
  ground-truth uncertainty, or when jitter is larger than the bias.
- Tape error at each spot lands in the varying part (it looks like a bias that
  changes from spot to spot). A varying part not well above the tape
  uncertainty (``truth_uncertainty_m``, the largest ``uncertainty_m`` in the
  lines used) cannot be told apart from tape error.
- The split depends on which part of the image and which ranges the spots
  cover. Cover the area the tracker will use, and no more.
- The simulation's regimes come from one tracker, one failure rule (S6) and
  30 seeds; its varying share was a modelling choice. The verdict says which
  simulated case the bench numbers resemble, nothing more.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.detection_log import DetectionLogError, read_detection_log

SCHEMA = "zephyr-camera-bias-field-1"
AXES = ("x", "y", "z")

MIN_SPOTS = 5
MIN_FRAMES_PER_SPOT = 60
MIN_SPOT_SEPARATION_M = 0.05
GT_MATCH_TOL_M = 0.005
CLUSTER_SEPARATION_JITTERS = 4.0

# Simulation regimes, from tools/sens_vision_bias_camera.py (synthetic, 30 seeds, S6 failure rule).
CELL_5CM_TOTAL_RMS_M = math.sqrt(3.0) * 0.05
CELL_10CM_TOTAL_RMS_M = math.sqrt(3.0) * 0.10
SIM_MAIN_SHARE_MAX = 0.28      # main camera model: 16 % (turn) to 28 % (line) of depth variance varying
SIM_BRACKET_SHARE_MIN = 0.79   # bracket model: 79 % (line) to 90 % (turn)

SIM_SOURCE = ("simulation results from tools/sens_vision_bias_camera.py on synthetic data, 30 seeds, "
              "S6 failure rule; not measured")


class BiasFieldError(ValueError):
    """The log or spots file cannot be scored. The message says why."""


@dataclass(frozen=True)
class Spot:
    spot_id: str
    position_m: tuple[float, float, float]
    window_s: tuple[float, float] | None = None


# ---------------------------------------------------------------- spots file


def read_spots_csv(path: str | Path) -> list[Spot]:
    """Read the known spot positions (and optional capture-time windows)."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise BiasFieldError(f"{path}: empty spots file")
    header = [cell.strip() for cell in rows[0]]
    base = ["spot_id", "x_m", "y_m", "z_m"]
    if header not in (base, base + ["start_s", "end_s"]):
        raise BiasFieldError(f"{path}: header must be {','.join(base)} or {','.join(base)},start_s,end_s; "
                             f"got {','.join(header)}")
    timed = len(header) == 6
    spots: list[Spot] = []
    seen: set[str] = set()
    for line, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise BiasFieldError(f"{path}:{line}: expected {len(header)} columns, got {len(row)}")
        spot_id = row[0].strip()
        if not spot_id:
            raise BiasFieldError(f"{path}:{line}: empty spot_id")
        if spot_id in seen:
            raise BiasFieldError(f"{path}:{line}: spot_id {spot_id!r} appears twice")
        seen.add(spot_id)
        try:
            numbers = [float(cell) for cell in row[1:]]
        except ValueError as exc:
            raise BiasFieldError(f"{path}:{line}: {exc}") from exc
        if not all(math.isfinite(v) for v in numbers):
            raise BiasFieldError(f"{path}:{line}: numbers must be finite")
        window = None
        if timed:
            window = (numbers[3], numbers[4])
            if window[1] < window[0]:
                raise BiasFieldError(f"{path}:{line}: spot {spot_id!r} window ends before it starts")
        spots.append(Spot(spot_id, (numbers[0], numbers[1], numbers[2]), window))
    return spots


# ---------------------------------------------------------------- the decomposition


def _share(varying_sq: float, total_sq: float) -> float | None:
    return None if total_sq == 0.0 else varying_sq / total_sq


def decompose_bias_field(errors_by_spot: Mapping[str, np.ndarray], *, depth_axis: str = "z") -> dict[str, Any]:
    """Per-spot mean error and jitter, then the common offset and varying part per axis.

    ``errors_by_spot`` maps a spot id to an (n, 3) array of frame errors (measured minus known position).
    Definitions are in the module docstring. No minimum checks here; ``fit_camera_bias_field`` does them.
    """
    if depth_axis not in AXES:
        raise BiasFieldError(f"depth axis must be one of {AXES}")
    if not errors_by_spot:
        raise BiasFieldError("no spots")
    spots = []
    means = []
    jitters = []
    counts = []
    for spot_id, errors in errors_by_spot.items():
        errors = np.asarray(errors, dtype=float)
        if errors.ndim != 2 or errors.shape[1] != 3 or errors.shape[0] < 2:
            raise BiasFieldError(f"spot {spot_id!r}: need an (n, 3) error array with n >= 2")
        mean = errors.mean(axis=0)
        jitter = errors.std(axis=0, ddof=1)
        means.append(mean)
        jitters.append(jitter)
        counts.append(errors.shape[0])
        spots.append({"spot_id": spot_id, "frames": int(errors.shape[0]),
                      "mean_error_m": [float(v) for v in mean], "jitter_m": [float(v) for v in jitter]})
    means_a = np.array(means)
    jitters_a = np.array(jitters)
    counts_a = np.array(counts, dtype=float)

    common = means_a.mean(axis=0)
    varying_sq = np.mean((means_a - common) ** 2, axis=0)
    bias_sq = common**2 + varying_sq
    jitter_sq = np.mean(jitters_a**2, axis=0)
    floor_sq = np.mean(jitters_a**2 / counts_a[:, None], axis=0)

    per_axis = {}
    for k, name in enumerate(AXES):
        per_axis[name] = {
            "common_m": float(common[k]),
            "varying_m": float(math.sqrt(varying_sq[k])),
            "bias_rms_m": float(math.sqrt(bias_sq[k])),
            "varying_share": _share(float(varying_sq[k]), float(bias_sq[k])),
            "jitter_rms_m": float(math.sqrt(jitter_sq[k])),
            "varying_noise_floor_m": float(math.sqrt(floor_sq[k])),
        }
    return {
        "spots": spots,
        "spot_count": len(spots),
        "per_axis": per_axis,
        "depth_axis": depth_axis,
        "varying_share_depth": per_axis[depth_axis]["varying_share"],
        "varying_share_3d": _share(float(varying_sq.sum()), float(bias_sq.sum())),
        "accuracy_rms_3d_m": float(math.sqrt(bias_sq.sum())),
        "total_rms_3d_m": float(math.sqrt(bias_sq.sum() + jitter_sq.sum())),
    }


# ---------------------------------------------------------------- verdict


def verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Place the bench numbers on the camera-shaped bias simulation's regimes."""
    total = float(result["total_rms_3d_m"])
    share = result["varying_share_depth"]
    cm = f"{100 * total:.1f} cm"
    lead = f"Verdict ({SIM_SOURCE}): total 3D error RMS {cm}"
    if total <= CELL_5CM_TOTAL_RMS_M:
        text = (f"{lead} is at or below the simulated 5 cm cell (8.7 cm). There, every camera-shaped bias "
                "version passed S6's majority rule with the same tolerated delay as a constant bias to within "
                "one step (300 ms line, 200 ms turn, recovery arm, no loss), so the shape does not matter at "
                "this size.")
        return {"size_band": "at_or_below_5cm_cell", "regime": "shape_does_not_matter", "text": text}

    if total <= CELL_10CM_TOTAL_RMS_M:
        size_band = "between_cells"
        size = ("is between the simulated 5 cm cell (8.7 cm) and 10 cm cell (17.3 cm); this size was not "
                "simulated, so read the 10 cm results below as a worst case")
    else:
        size_band = "above_10cm_cell"
        size = ("is above the simulated 10 cm cell (17.3 cm); this size was not simulated, so expect worse "
                "than the 10 cm results below")

    if share is None:
        text = (f"{lead} {size}. The spots show no bias at all, so the error is jitter only; the camera-shaped "
                "bias simulation does not cover that case (the white-noise rows of the S6 grid do).")
        return {"size_band": size_band, "regime": "no_bias", "text": text}
    pct = f"{100 * share:.0f} %"
    if share <= SIM_MAIN_SHARE_MAX:
        regime = "mostly_offset"
        shape = (f"The depth bias is {pct} varying, in the mostly-offset range (0 to 28 %). At the 10 cm cell a "
                 "constant bias and the main camera field (16 to 28 % varying) both passed S6's majority rule at "
                 "0 ms delay, but 20 to 33 % of seeds failed at every delay on bias alone.")
    elif share >= SIM_BRACKET_SHARE_MIN:
        regime = "mostly_varying"
        shape = (f"The depth bias is {pct} varying, in the bracket's range (79 to 90 %). At the 10 cm cell that "
                 "field failed at 0 ms delay on a coin flip: 15 of 30 seeds on the line, 16 of 30 on the turn.")
    else:
        regime = "share_not_simulated"
        shape = (f"The depth bias is {pct} varying, between the simulated shares (28 % and 79 %). That share "
                 "was not simulated, so at the 10 cm cell the pass or fail label could go either way.")
    return {"size_band": size_band, "regime": regime, "text": f"{lead} {size}. {shape}"}


# ---------------------------------------------------------------- whole logs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_camera_bias_field(log_paths: Sequence[str | Path], spots: Sequence[Spot], *, subject: str | None = None,
                          depth_axis: str = "z") -> dict[str, Any]:
    """Score a bench session. Raises BiasFieldError (or DetectionLogError) when it cannot be scored."""
    log_paths = [Path(p) for p in log_paths]
    if not log_paths:
        raise BiasFieldError("give at least one detection log")
    if len(spots) < MIN_SPOTS:
        raise BiasFieldError(f"need at least {MIN_SPOTS} distinct spots, the spots file has {len(spots)}")
    positions = {s.spot_id: np.array(s.position_m) for s in spots}
    for a in range(len(spots)):
        for b in range(a + 1, len(spots)):
            gap = float(np.linalg.norm(positions[spots[a].spot_id] - positions[spots[b].spot_id]))
            if gap < MIN_SPOT_SEPARATION_M:
                raise BiasFieldError(
                    f"spots {spots[a].spot_id!r} and {spots[b].spot_id!r} are {100 * gap:.1f} cm apart, under "
                    f"{100 * MIN_SPOT_SEPARATION_M:.0f} cm: not distinguishable")

    timed = [s.window_s is not None for s in spots]
    by_time = all(timed)
    if any(timed) and not by_time:
        raise BiasFieldError("either every spot has a time window or none does")
    if by_time:
        if len(log_paths) != 1:
            raise BiasFieldError("time windows need exactly one log (separate logs do not share a clock)")
        ordered = sorted(spots, key=lambda s: s.window_s[0])
        for before, after in zip(ordered, ordered[1:]):
            if after.window_s[0] <= before.window_s[1]:
                raise BiasFieldError(f"time windows of {before.spot_id!r} and {after.spot_id!r} overlap: "
                                     "not distinguishable")

    errors: dict[str, list[np.ndarray]] = {s.spot_id: [] for s in spots}
    skipped = {"not_detected": 0, "no_position": 0, "outside_windows": 0, "other_subject": 0}
    frames_seen: set[str] = set()
    truth_uncertainty = None
    logs_meta = []
    traces = []
    for path in log_paths:
        trace = read_detection_log(path)
        traces.append((path, trace))
        logs_meta.append({"path": path.name, "sha256": _sha256(path), "trace_id": trace.trace_id,
                          "status": trace.status})
    subjects = sorted({o.subject_id for _, t in traces for o in t.observations})
    if subject is None:
        if len(subjects) > 1:
            raise BiasFieldError(f"the logs have several subject_id values {subjects}; choose one with --subject")
        subject = subjects[0] if subjects else None
    elif subject not in subjects:
        raise BiasFieldError(f"subject {subject!r} not in the logs (found {subjects})")

    for path, trace in traces:
        for obs in trace.observations:
            if obs.subject_id != subject:
                skipped["other_subject"] += 1
                continue
            if obs.outcome != "detected":
                skipped["not_detected"] += 1
                continue
            if obs.position_m is None:
                skipped["no_position"] += 1
                continue
            frames_seen.add(obs.position_frame or "")
            truth = None
            if obs.ground_truth is not None:
                truth = np.array(obs.ground_truth["position_m"], dtype=float)
                u = float(obs.ground_truth["uncertainty_m"])
                truth_uncertainty = u if truth_uncertainty is None else max(truth_uncertainty, u)
            where = f"{path.name} observation {obs.observation_id!r}"
            if by_time:
                hits = [s for s in spots if s.window_s[0] <= obs.capture_time_s <= s.window_s[1]]
                if not hits:
                    skipped["outside_windows"] += 1
                    continue
                spot = hits[0]
                if truth is not None and np.linalg.norm(truth - positions[spot.spot_id]) > GT_MATCH_TOL_M:
                    raise BiasFieldError(f"{where}: its ground truth is not spot {spot.spot_id!r}, whose window "
                                         "holds it")
            else:
                if truth is None:
                    raise BiasFieldError(f"{where}: no ground_truth; give time windows in the spots file "
                                         "or record ground truth on every detection")
                hits = [s for s in spots if np.linalg.norm(truth - positions[s.spot_id]) <= GT_MATCH_TOL_M]
                if not hits:
                    raise BiasFieldError(f"{where}: ground truth {truth.tolist()} matches no spot in the spots "
                                         f"file (within {1000 * GT_MATCH_TOL_M:.0f} mm)")
                spot = hits[0]
            errors[spot.spot_id].append(np.array(obs.position_m, dtype=float) - positions[spot.spot_id])

    if len(frames_seen) > 1:
        raise BiasFieldError(f"detections use several position_frame values {sorted(frames_seen)}")
    short = {k: len(v) for k, v in errors.items() if len(v) < MIN_FRAMES_PER_SPOT}
    if short:
        listed = ", ".join(f"{k} ({n})" for k, n in short.items())
        raise BiasFieldError(f"every spot needs at least {MIN_FRAMES_PER_SPOT} detected frames; too few at: {listed}")

    result = decompose_bias_field({k: np.array(v) for k, v in errors.items()}, depth_axis=depth_axis)

    by_id = {s["spot_id"]: s for s in result["spots"]}
    ids = list(by_id)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            sa, sb = by_id[ids[a]], by_id[ids[b]]
            mean_a = positions[ids[a]] + np.array(sa["mean_error_m"])
            mean_b = positions[ids[b]] + np.array(sb["mean_error_m"])
            spread = max(float(np.linalg.norm(sa["jitter_m"])), float(np.linalg.norm(sb["jitter_m"])))
            gap = float(np.linalg.norm(mean_a - mean_b))
            if gap < CLUSTER_SEPARATION_JITTERS * spread:
                raise BiasFieldError(
                    f"measured positions at spots {ids[a]!r} and {ids[b]!r} are {100 * gap:.1f} cm apart, under "
                    f"{CLUSTER_SEPARATION_JITTERS:g} x their {100 * spread:.1f} cm jitter: not distinguishable")

    for s in result["spots"]:
        s["known_position_m"] = [float(v) for v in positions[s["spot_id"]]]
    result.update({
        "schema": SCHEMA,
        "status": "proposal; scored by a tool tested on synthetic logs only",
        "logs": logs_meta,
        "subject_id": subject,
        "position_frame": next(iter(frames_seen)) if frames_seen else None,
        "assignment": "time_windows" if by_time else "ground_truth",
        "skipped_lines": skipped,
        "truth_uncertainty_m": truth_uncertainty,
        "rules": {"min_spots": MIN_SPOTS, "min_frames_per_spot": MIN_FRAMES_PER_SPOT,
                  "min_spot_separation_m": MIN_SPOT_SEPARATION_M, "gt_match_tol_m": GT_MATCH_TOL_M,
                  "cluster_separation_jitters": CLUSTER_SEPARATION_JITTERS},
    })
    result["notes"] = _notes(result)
    result["verdict"] = verdict(result)
    return result


def _notes(result: Mapping[str, Any]) -> list[str]:
    """Warnings about numbers that look precise but are not."""
    notes = []
    for name in AXES:
        a = result["per_axis"][name]
        if a["bias_rms_m"] <= 3.0 * a["varying_noise_floor_m"]:
            notes.append(f"{name} axis: the bias ({_mm(a['bias_rms_m'])} mm RMS) is within 3 x the jitter noise "
                         f"floor ({_mm(a['varying_noise_floor_m'])} mm), so its varying share means nothing")
    depth = result["per_axis"][result["depth_axis"]]
    u = result["truth_uncertainty_m"]
    if u is not None and depth["varying_m"] <= 2.0 * u:
        notes.append(f"depth varying part ({_mm(depth['varying_m'])} mm) is not above 2 x the ground-truth "
                     f"uncertainty ({_mm(u)} mm), so it could be mostly tape error")
    jitter_sq = sum(result["per_axis"][n]["jitter_rms_m"] ** 2 for n in AXES)
    if jitter_sq > result["accuracy_rms_3d_m"] ** 2:
        notes.append("jitter is larger than the bias; the camera-shaped bias simulation assumed mm-level jitter, "
                     "so the S6 white-noise rows may describe this camera better than the verdict below")
    return notes


# ---------------------------------------------------------------- text and command line


def _mm(value: float | None) -> str:
    return "n/a" if value is None else f"{1000 * value:.1f}"


def summary_text(result: Mapping[str, Any]) -> str:
    statuses = sorted({log["status"] for log in result["logs"]})
    lines = [f"Camera bias field: {result['spot_count']} spots, logs {', '.join(statuses)} "
             f"(assignment by {result['assignment'].replace('_', ' ')}), frame {result['position_frame']}",
             "", "spot              frames   mean error x y z (mm)        jitter x y z (mm)"]
    for s in result["spots"]:
        mean = " ".join(f"{1000 * v:7.1f}" for v in s["mean_error_m"])
        jit = " ".join(f"{1000 * v:6.1f}" for v in s["jitter_m"])
        lines.append(f"{s['spot_id']:<17} {s['frames']:>6}   {mean}   {jit}")
    lines += ["", "axis  common (mm)  varying (mm)  varying share  jitter RMS (mm)  noise floor (mm)"]
    for name in AXES:
        a = result["per_axis"][name]
        share = "n/a" if a["varying_share"] is None else f"{100 * a['varying_share']:.0f} %"
        lines.append(f"{name:<5} {_mm(a['common_m']):>11}  {_mm(a['varying_m']):>12}  {share:>13}  "
                     f"{_mm(a['jitter_rms_m']):>15}  {_mm(a['varying_noise_floor_m']):>16}")
    share3 = result["varying_share_3d"]
    lines += ["",
              f"3D accuracy RMS (bias only): {_mm(result['accuracy_rms_3d_m'])} mm",
              f"3D total RMS (bias and jitter): {_mm(result['total_rms_3d_m'])} mm",
              f"varying share, 3D: {'n/a' if share3 is None else f'{100 * share3:.0f} %'}; "
              f"depth axis {result['depth_axis']} used for the verdict",
              f"largest ground-truth uncertainty in the lines: {_mm(result['truth_uncertainty_m'])} mm "
              "(tape error lands in the varying part)",
              *(f"note: {note}" for note in result["notes"]),
              "", result["verdict"]["text"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="detection log(s), .jsonl")
    parser.add_argument("--spots", required=True, type=Path, help="CSV of known spot positions")
    parser.add_argument("--subject", help="subject_id to score, needed when the logs have more than one")
    parser.add_argument("--depth-axis", choices=AXES, default="z", help="depth axis for the verdict (default z)")
    parser.add_argument("--output", type=Path, help="write the JSON result here")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            out = args.output.resolve()
            if out in {p.resolve() for p in args.logs} or out == args.spots.resolve():
                raise BiasFieldError("--output must not be one of the input files")
        result = fit_camera_bias_field(args.logs, read_spots_csv(args.spots), subject=args.subject,
                                       depth_axis=args.depth_axis)
    except (BiasFieldError, DetectionLogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output is not None:
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.write_bytes(text.encode("utf-8"))
    sys.stdout.write(summary_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
