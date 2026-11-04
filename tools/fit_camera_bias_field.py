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
- ``raw_p95_3d_m`` (per spot): the 95th percentile over that spot's frames of
  the length of the 3D frame error (bias and jitter, before any filter).
  ``pooled_raw_p95_3d_m``: the same over all frames, each spot weighted
  equally (each frame weighs 1 / n_i). The 95th percentile here is the first
  value, in sorted order, whose cumulative weight reaches 95 %.

The verdict
-----------
It reports this session's numbers first, then names the study regime they fall
in (by total size and depth share), then gives a simulation-based expectation.
The study found that which simulated runs failed S6 was set by each run's own
raw error p95 against S6's 0.25 m line, not by the ensemble's share of failing
seeds. A measured camera is one such run with a known field, so the
expectation compares ``pooled_raw_p95_3d_m`` with 0.25 m (within 10 % counts
as "either way"). It is an expectation from simulation, not a measured result,
and it assumes a run sweeps the area the spots cover.

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
- Wrong units: at any one spot in any one log, the measured distance is over
  10 or under 0.1 times the known distance (``sim.vision_noise_fit._check_units``,
  reused read-only, run per spot and log; a factor near 1000 means millimetres), or the spots' median distance is over 100 m
  (the spots file itself is in millimetres or pixels).
- Pose flips: any detection with z <= 0 in the camera frame.

The other camera-log checks of ``sim.vision_noise_fit`` (``_quirk_warnings``:
gaps, repeated time stamps, frozen frames, outliers) run on each spot's frames
and their warnings are added to ``notes``, prefixed with the spot id. Their
text was written for the noise fit, so it names that fit's outputs.

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

from sim import vision_noise_fit as vnf
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

S6_P95_LINE_M = 0.25           # S6 failure rule: a run fails when its error p95 is over 0.25 m
NEAR_LINE_FRACTION = 0.10      # pooled p95 within 10 % of the line: called 'either way'
# Spread of x / z the study's camera swept, assuming its 70 degree field of view (tan 35 deg = 0.70).
SIM_COVERAGE_TEXT = "about -0.70 to +0.70 on the line and -0.35 to +0.35 on the turn"

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


def weighted_p95(values: np.ndarray, weights: np.ndarray) -> float:
    """The first value, in sorted order, whose cumulative weight reaches 95 % of the total weight."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.95 * cumulative[-1] * (1.0 - 1e-12)))
    return float(values[order][index])


# ---------------------------------------------------------------- verdict


def _expectation(result: Mapping[str, Any]) -> tuple[str | None, str]:
    p95 = result.get("pooled_raw_p95_3d_m")
    if p95 is None:
        return None, ""
    line = S6_P95_LINE_M
    if p95 > line * (1.0 + NEAR_LINE_FRACTION):
        key, outcome, where = "fail_on_bias", "fail S6 at every delay on bias alone", "over"
    elif p95 < line * (1.0 - NEAR_LINE_FRACTION):
        key, outcome, where = "pass_on_bias", "pass on bias alone (delay and loss limits still apply)", "under"
    else:
        key, outcome, where = "near_the_line", "go either way", f"within {100 * NEAR_LINE_FRACTION:.0f} % of"
    text = (f"Simulation-based expectation, not a measured result: the pooled raw 3D error p95 is "
            f"{p95:.3f} m, {where} S6's {line:.2f} m p95 line. In the study that number decided which runs "
            "failed: on the line every seed whose raw p95 was over 0.25 m failed and none under it did; on the "
            "turn 9 of 10 over it failed, plus 1 under it. So a tracked run across the area these spots cover "
            f"would be expected to {outcome}, assuming the study's tracker and a run that sweeps that area.")
    return key, text


def verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Report this session's numbers, the study regime they fall in, and a labelled p95 expectation."""
    total = float(result["total_rms_3d_m"])
    share = result["varying_share_depth"]
    p95 = result.get("pooled_raw_p95_3d_m")
    measured = f"Measured (this session): total 3D error RMS {100 * total:.1f} cm (bias and jitter)"
    measured += ", depth varying share " + ("n/a" if share is None else f"{100 * share:.0f} %")
    if p95 is not None:
        measured += (f", pooled raw 3D error p95 {100 * p95:.1f} cm, {result.get('spots_over_p95_line', 0)} of "
                     f"{result.get('spot_count', 0)} spots over 25 cm")
    coverage = result.get("coverage_x_over_z")
    if coverage is not None:
        measured += (f"; the spots span x / z from {coverage[0]:+.2f} to {coverage[1]:+.2f} (the study swept "
                     f"{SIM_COVERAGE_TEXT})")
    measured += "."
    key, expectation = _expectation(result)
    regime_lead = f"Regime ({SIM_SOURCE}):"
    draw = ("This camera is one such draw with a known field, so that share is not its chance of failing; the "
            "expectation below is the per-run test.")

    def done(size_band: str, regime: str, regime_text: str) -> dict[str, Any]:
        text = " ".join(part for part in (measured, f"{regime_lead} {regime_text}", expectation) if part)
        return {"size_band": size_band, "regime": regime, "expectation": key, "text": text}

    if total <= CELL_5CM_TOTAL_RMS_M:
        return done("at_or_below_5cm_cell", "shape_does_not_matter",
                    "at or below the simulated 5 cm cell (8.7 cm). There every camera-shaped version passed S6 "
                    "with the same tolerated delay as a constant bias to within one step (300 ms line, 200 ms "
                    "turn, recovery arm, no loss), so the shape does not matter at this size.")
    if total <= CELL_10CM_TOTAL_RMS_M:
        size_band = "between_cells"
        size = ("between the simulated 5 cm cell (8.7 cm) and 10 cm cell (17.3 cm); this size was not simulated, "
                "so the 10 cm results are an upper reference.")
    else:
        size_band = "above_10cm_cell"
        size = ("above the largest simulated cell (17.3 cm); the study has no regime for this size, so only the "
                "p95 expectation applies.")

    if share is None:
        return done(size_band, "no_bias", f"{size} The spots show no bias at all, so the error is jitter only; "
                    "the camera-shaped bias study does not cover that case (the S6 white-noise rows do).")
    if result.get("share_is_noise"):
        return done(size_band, "share_is_noise", f"{size} The depth bias is within the jitter noise floor, so its "
                    "share is noise and no shape regime applies.")
    if share <= SIM_MAIN_SHARE_MAX:
        return done(size_band, "mostly_offset", f"{size} Shape: mostly offset (0 to 28 % varying). In the study, "
                    "random simulated cameras of the 10 cm size with this shape (a constant bias, or 16 to 28 % "
                    "varying) passed S6's majority rule at 0 ms, while 20 to 33 % of them failed at every delay on "
                    f"bias alone. {draw}")
    if share >= SIM_BRACKET_SHARE_MIN:
        return done(size_band, "mostly_varying", f"{size} Shape: mostly varying (79 to 90 % in the study's "
                    "bracket). In the study, random simulated cameras of the 10 cm size with this shape failed the "
                    f"majority rule at 0 ms on a coin flip (15 of 30 on the line, 16 of 30 on the turn). {draw}")
    return done(size_band, "share_not_simulated", f"{size} Shape: between the simulated shares (28 % and 79 %); "
                "that share was not simulated.")


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
    median_distance = float(np.median([np.linalg.norm(v) for v in positions.values()]))
    if median_distance > vnf._LARGE_DISTANCE_M:
        raise BiasFieldError(f"the spots' median distance is {median_distance:.4g}, far beyond a bench: the spots "
                             "file (and probably the log) is in millimetres or pixels, not metres")
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
    records: dict[tuple[str, int], list[Any]] = {}
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

    for log_index, (path, trace) in enumerate(traces):
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
                                         f"file (within {1000 * GT_MATCH_TOL_M:.0f} mm); if the numbers are "
                                         "about 1000 times apart, one of them is in millimetres, not metres")
                spot = hits[0]
            errors[spot.spot_id].append(np.array(obs.position_m, dtype=float) - positions[spot.spot_id])
            records.setdefault((spot.spot_id, log_index), []).append(obs)

    if len(frames_seen) > 1:
        raise BiasFieldError(f"detections use several position_frame values {sorted(frames_seen)}")
    used = [k for k, v in errors.items() if v]
    if used:
        known = np.concatenate([np.tile(positions[k], (len(errors[k]), 1)) for k in used])
        measured_all = known + np.concatenate([np.array(errors[k]) for k in used])
        # One check per spot and log, so a single wrong-units placement cannot be outvoted by the others
        # (the check takes a median).
        for (spot_id, log_index), recs in records.items():
            group = np.array([o.position_m for o in recs], dtype=float)
            try:
                vnf._check_units(group, np.tile(positions[spot_id], (len(recs), 1)))
            except ValueError as exc:
                raise BiasFieldError(f"spot {spot_id} in {traces[log_index][0].name}: {exc}; the spots file must "
                                     "be in metres too") from exc
        if frames_seen == {"camera"}:
            behind = int(np.count_nonzero(measured_all[:, 2] <= 0.0))
            if behind:
                raise BiasFieldError(
                    f"{behind} detections have z <= 0 in the camera frame (behind the lens): pose flips. They "
                    "would move the spot means by centimetres. Check the frames file, drop or re-record those "
                    "frames (a flatter, matte marker and better light make flips rarer)")
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

    pooled_values = []
    pooled_weights = []
    for s in result["spots"]:
        s["known_position_m"] = [float(v) for v in positions[s["spot_id"]]]
        norms = np.linalg.norm(np.array(errors[s["spot_id"]]), axis=1)
        s["raw_p95_3d_m"] = weighted_p95(norms, np.ones(norms.size))
        pooled_values.append(norms)
        pooled_weights.append(np.full(norms.size, 1.0 / norms.size))
    result["pooled_raw_p95_3d_m"] = weighted_p95(np.concatenate(pooled_values), np.concatenate(pooled_weights))
    result["spots_over_p95_line"] = int(sum(s["raw_p95_3d_m"] > S6_P95_LINE_M for s in result["spots"]))
    result["coverage_x_over_z"] = None
    if depth_axis == "z" and all(v[2] > 0 for v in positions.values()):
        ratios = [v[0] / v[2] for v in positions.values()]
        result["coverage_x_over_z"] = [float(min(ratios)), float(max(ratios))]
    depth = result["per_axis"][depth_axis]
    result["share_is_noise"] = bool(depth["bias_rms_m"] <= 3.0 * depth["varying_noise_floor_m"])
    quirks = []
    for (spot_id, log_index), recs in records.items():
        trace = traces[log_index][1]
        recs = sorted(recs, key=lambda o: (o.capture_time_s, o.observation_id))
        measured = np.array([o.position_m for o in recs], dtype=float)
        known = np.tile(positions[spot_id], (len(recs), 1))
        res = measured - known
        std = res.std(axis=0, ddof=1) if len(recs) > 1 else np.zeros(3)
        robust = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
        for warning in vnf._quirk_warnings(trace, subject, recs, res, std, robust, measured, known,
                                           sorted(frames_seen)):
            quirks.append(f"spot {spot_id}: {warning}")
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
    result["notes"] = quirks + _notes(result)
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
    depth_k = AXES.index(result["depth_axis"])
    ranges = [s["known_position_m"][depth_k] for s in result["spots"]]
    if max(ranges) - min(ranges) < MIN_SPOT_SEPARATION_M:
        notes.append("all spots are at one range (within 5 cm), so the part of the field that changes with range "
                     "is not sampled; add a second range")
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
             "", "spot              frames   mean error x y z (mm)        jitter x y z (mm)    raw p95 3D (mm)"]
    for s in result["spots"]:
        mean = " ".join(f"{1000 * v:7.1f}" for v in s["mean_error_m"])
        jit = " ".join(f"{1000 * v:6.1f}" for v in s["jitter_m"])
        lines.append(f"{s['spot_id']:<17} {s['frames']:>6}   {mean}   {jit}   {1000 * s['raw_p95_3d_m']:8.1f}")
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
              "", "Verdict:", result["verdict"]["text"]]
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
