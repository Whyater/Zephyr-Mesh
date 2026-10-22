"""Diagnose why EXP-025 passes the 0.35 m centre rule but fails the ellipsoid check.

Proposal tool, simulated only.  It reruns a small sample of
EXP-025 conditions at the radius ``buffer.json`` reports as passing, with the
same 75 seeds, and records where the two closest drones were when the Run A
ellipsoid check (radii 0.12, 0.12, 0.30 m per drone) came closest to failing.

It does not change ``sim/coop/separation.py``.  Instead it wraps three of that
module's helpers while an encounter runs:

* ``_linear_segment_minimum`` to find the pair, time and offset at the
  ellipsoid minimum,
* ``safe_velocity`` to record true and perceived positions, and to apply the
  "flat" and "altitude hold" variants,
* ``_message_noise`` to remove vertical noise in the "no vertical noise"
  variant (the x and y draws stay identical),

and swaps ``AvoidanceConfig`` for the "ellipsoid avoidance" variant.

Variants:

``as_run``               EXP-025 unchanged (must reproduce buffer.json counts).
``no_vertical_noise``    position and velocity noise on z set to zero.
``flat_avoidance``       the vertical part of every chosen velocity set to zero.
``altitude_hold``        chosen vertical velocity plus 2.0 * (start altitude - z).
``ellipsoid_avoidance``  avoidance uses the same ellipsoid as the check, with the
                         condition's buffer as its safety margin.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from sim.coop import separation
from tools.d_common import seeds_for
from tools.d_run_separation import MIN_SEPARATION_M, N_SEEDS, _spec

ELLIPSOID_RADII_M = (0.12, 0.12, 0.30)
ALTITUDE_HOLD_GAIN_PER_S = 2.0
VARIANTS = ("as_run", "no_vertical_noise", "flat_avoidance", "altitude_hold", "ellipsoid_avoidance")
CONDITION_KEYS = ("kind", "speed_mps", "delay_s", "loss_probability", "noise_std_m",
                  "dead_reckoning", "lateral_offset_m")


def min_vertical_offset_for_failure(radii=ELLIPSOID_RADII_M, center_m: float = MIN_SEPARATION_M) -> float:
    """Smallest vertical offset at which a pair can fail the ellipsoid but pass the centre rule.

    With combined semi-axes a = 2 * r_xy and c = 2 * r_z, a pair at centre
    distance d >= ``center_m`` is inside the ellipsoid only if
    h^2 / a^2 + dz^2 / c^2 < 1 with h^2 = d^2 - dz^2.  The worst case is
    d = ``center_m``, which gives dz^2 > (d^2 / a^2 - 1) / (1 / a^2 - 1 / c^2).
    Returns 0 when the sphere already lies inside the ellipsoid's waist.
    """
    a = 2.0 * float(radii[0])
    c = 2.0 * float(radii[2])
    if c <= a:
        raise ValueError("this bound assumes a taller-than-wide ellipsoid")
    numerator = center_m * center_m / (a * a) - 1.0
    if numerator <= 0.0:
        return 0.0
    return math.sqrt(numerator / (1.0 / (a * a) - 1.0 / (c * c)))


def _closest_point(start: np.ndarray, end: np.ndarray, scales: np.ndarray | None) -> tuple[float, np.ndarray]:
    """Same clamped projection as ``separation._linear_segment_minimum``; also returns the unscaled offset."""
    s = start if scales is None else start / scales
    e = end if scales is None else end / scales
    step = e - s
    denominator = float(np.dot(step, step))
    fraction = 0.0 if denominator <= 1.0e-12 else float(np.clip(-np.dot(s, step) / denominator, 0.0, 1.0))
    return fraction, start + fraction * (end - start)


class _NoVerticalNoise:
    """Wrap a Generator so 3-vector normal draws keep x and y and zero z."""

    def __init__(self, generator: np.random.Generator) -> None:
        self._generator = generator

    def normal(self, loc=0.0, scale=1.0, size=None):
        draw = self._generator.normal(loc, scale, size=size)
        if size == 3:
            draw = np.array(draw, dtype=float)
            draw[2] = 0.0
        return draw


@contextmanager
def instrumented(variant: str, spec) -> Iterator[dict[str, Any]]:
    """Patch ``sim.coop.separation`` for one encounter and collect a trace."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    n = 6 if spec.kind == "ring_swap" else 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    trace: dict[str, Any] = {
        "best": None, "z": [], "dz_series": {}, "errors": [], "segment_calls": 0, "pairs": pairs,
    }
    original = {name: getattr(separation, name) for name in
                ("_linear_segment_minimum", "safe_velocity", "_message_noise", "AvoidanceConfig")}
    pending: list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = []

    def segment_minimum(start_delta, end_delta, scales=None):
        value = original["_linear_segment_minimum"](start_delta, end_delta, scales)
        start = np.asarray(start_delta, dtype=float)
        end = np.asarray(end_delta, dtype=float)
        call = trace["segment_calls"]
        trace["segment_calls"] += 1
        step, slot = divmod(call, 2 * len(pairs))
        pair = pairs[slot // 2]
        if scales is None:
            trace["dz_series"].setdefault(pair, []).append(
                (step, float(np.hypot(start[0], start[1])), float(start[2])))
            return value
        best = trace["best"]
        if best is None or value < best["min_ellipsoid"]:
            fraction, offset = _closest_point(start, end, np.asarray(scales, dtype=float))
            trace["best"] = {
                "min_ellipsoid": float(value), "step": int(step), "fraction": fraction, "pair": pair,
                "dz_m": float(offset[2]), "horizontal_m": float(np.hypot(offset[0], offset[1])),
                "center_m": float(np.linalg.norm(offset)),
                "dvz_mps": float((end[2] - start[2]) / spec.control_dt_s),
            }
        return value

    def chosen_velocity(own_pos, own_vel, preferred, neighbors, config, **kwargs):
        value, flags = original["safe_velocity"](own_pos, own_vel, preferred, neighbors, config, **kwargs)
        position = np.asarray(own_pos, dtype=float)
        pending.append((position.copy(), [(np.asarray(p, dtype=float), np.asarray(v, dtype=float))
                                          for p, v in neighbors]))
        if len(pending) == n:
            truth = np.array([item[0] for item in pending])
            trace["z"].append(truth[:, 2].copy())
            for receiver, (_, seen) in enumerate(pending):
                if len(seen) != n - 1:
                    continue
                senders = [agent for agent in range(n) if agent != receiver]
                for sender, (perceived, _) in zip(senders, seen):
                    trace["errors"].append(perceived - truth[sender])
            pending.clear()
        value = np.array(value, dtype=float)
        if variant == "flat_avoidance":
            value[2] = 0.0
        elif variant == "altitude_hold":
            value[2] += ALTITUDE_HOLD_GAIN_PER_S * (float(spec.altitude_m) - float(position[2]))
        return value, flags

    def message_noise(spec_arg, seed, count):
        generators = original["_message_noise"](spec_arg, seed, count)
        if variant == "no_vertical_noise":
            return [_NoVerticalNoise(generator) for generator in generators]
        return generators

    def avoidance_config(**kwargs):
        if variant == "ellipsoid_avoidance":
            margin = max(0.0, float(kwargs["radius_m"]) - MIN_SEPARATION_M)
            kwargs = {**kwargs, "ellipsoid_radii": ELLIPSOID_RADII_M, "safety_margin_m": margin}
        return original["AvoidanceConfig"](**kwargs)

    separation._linear_segment_minimum = segment_minimum
    separation.safe_velocity = chosen_velocity
    separation._message_noise = message_noise
    separation.AvoidanceConfig = avoidance_config
    try:
        yield trace
    finally:
        for name, value in original.items():
            setattr(separation, name, value)


def diagnose_encounter(condition: dict[str, Any], radius_m: float, seed: int, variant: str) -> dict[str, Any]:
    """Run one encounter under ``variant`` and summarise where the ellipsoid minimum happened."""
    spec = _spec({**condition, "avoidance_radius_m": radius_m})
    with instrumented(variant, spec) as trace:
        result = separation.run_encounter(spec, seed)
    best = trace["best"]
    z = np.array(trace["z"]) if trace["z"] else np.zeros((1, 1))
    errors = np.array(trace["errors"]) if trace["errors"] else np.zeros((0, 3))
    record = {
        "seed": int(seed),
        "violated": bool(result["violated"]),
        "ellipsoid_violated": bool(result["ellipsoid_violated"]),
        "min_center_distance_m": float(result["min_center_distance_m"]),
        "min_ellipsoid_separation": float(result["min_ellipsoid_separation"]),
        "max_altitude_drift_m": float(np.max(np.abs(z - float(spec.altitude_m)))),
        "rms_error_x_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))) if len(errors) else 0.0,
        "rms_error_y_m": float(np.sqrt(np.mean(errors[:, 1] ** 2))) if len(errors) else 0.0,
        "rms_error_z_m": float(np.sqrt(np.mean(errors[:, 2] ** 2))) if len(errors) else 0.0,
    }
    if best is not None:
        series = trace["dz_series"][best["pair"]]
        # Vertical offset of that pair when it first came within 1.0 m horizontally,
        # which separates drift that built up before the approach from a dodge during it.
        onset = next((dz for _, horizontal, dz in series if horizontal < 1.0), series[-1][2])
        record.update({
            "at_minimum_dz_m": best["dz_m"], "at_minimum_horizontal_m": best["horizontal_m"],
            "at_minimum_center_m": best["center_m"], "at_minimum_step": best["step"],
            "dz_at_1m_approach_m": float(onset),
            "pair_max_abs_dz_m": float(max(abs(dz) for _, _, dz in series)),
        })
    return record


def first_vertical_commands(condition: dict[str, Any], radius_m: float, seed: int, variant: str,
                            limit: int = 3) -> list[dict[str, Any]]:
    """Return the first control steps where avoidance commanded any vertical speed.

    Used on ``no_vertical_noise`` runs that still drift, to find what starts
    the vertical motion when nothing in the inputs is vertical.
    """
    spec = _spec({**condition, "avoidance_radius_m": radius_m})
    events: list[dict[str, Any]] = []
    with instrumented(variant, spec) as trace:
        patched = separation.safe_velocity

        def spy(own_pos, own_vel, preferred, neighbors, config, **kwargs):
            value, flags = patched(own_pos, own_vel, preferred, neighbors, config, **kwargs)
            if value[2] != 0.0 and len(events) < limit:
                events.append({"step": len(trace["z"]), "flags": sorted(flags), "vz_mps": float(value[2]),
                               "z_offset_m": float(own_pos[2]) - float(spec.altitude_m)})
            return value, flags

        separation.safe_velocity = spy
        separation.run_encounter(spec, seed)
    return events


def passing_radius(entry: dict[str, Any]) -> float:
    if entry.get("censored") or entry.get("required_buffer_m") is None:
        raise ValueError("censored condition has no passing radius")
    return round(MIN_SEPARATION_M + float(entry["required_buffer_m"]), 2)


def ellipsoid_count_at_pass(entry: dict[str, Any]) -> int:
    radius = passing_radius(entry)
    row = next(item for item in entry["tested_radii"] if abs(float(item["avoidance_radius_m"]) - radius) < 1e-9)
    return int(row["ellipsoid_violation_count"])


def select_conditions(entries: list[dict[str, Any]], *, per_kind: int, seed: int) -> list[dict[str, Any]]:
    """Pick a reproducible sample stratified by group and encounter kind.

    Groups: conditions that fail the ellipsoid at their passing radius, noisy
    conditions that pass it, noise-free conditions, and the worst failing
    head-on and crossing conditions.  Within each group up
    to ``per_kind`` conditions per kind are drawn, but at most one ring swap
    (ring swaps cost about 15 times a pair encounter).
    """
    usable = [entry for entry in entries if not entry.get("censored")]
    groups = {
        "fails_ellipsoid": [e for e in usable if ellipsoid_count_at_pass(e) > 0],
        "noisy_passes_ellipsoid": [e for e in usable if ellipsoid_count_at_pass(e) == 0
                                   and float(e["noise_std_m"]) > 0.0],
        "noise_free": [e for e in usable if float(e["noise_std_m"]) == 0.0],
    }
    rng = np.random.default_rng(seed)
    chosen: list[dict[str, Any]] = []
    for name, members in groups.items():
        for kind in ("head_on", "crossing", "ring_swap"):
            of_kind = [entry for entry in members if entry["kind"] == kind]
            limit = min(per_kind, 1) if kind == "ring_swap" else per_kind
            for index in rng.permutation(len(of_kind))[:limit]:
                entry = of_kind[int(index)]
                chosen.append({"group": name, **{key: entry[key] for key in CONDITION_KEYS},
                               "radius_m": passing_radius(entry),
                               "buffer_ellipsoid_count": ellipsoid_count_at_pass(entry)})
    # Plus the single worst pair condition of each kind, so the counterfactuals
    # are tested where failures are common, not only where they are rare.
    for kind in ("head_on", "crossing"):
        worst = max((e for e in groups["fails_ellipsoid"] if e["kind"] == kind),
                    key=lambda e: (ellipsoid_count_at_pass(e), -passing_radius(e)))
        chosen.append({"group": "most_failing", **{key: worst[key] for key in CONDITION_KEYS},
                       "radius_m": passing_radius(worst), "buffer_ellipsoid_count": ellipsoid_count_at_pass(worst)})
    return chosen


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    failing = [r for r in records if r["ellipsoid_violated"]]
    out: dict[str, Any] = {
        "seeds": len(records),
        "center_violations": sum(r["violated"] for r in records),
        "ellipsoid_violations": len(failing),
        "max_altitude_drift_m": max(r["max_altitude_drift_m"] for r in records),
        "rms_error_xy_m": float(np.sqrt(np.mean([(r["rms_error_x_m"] ** 2 + r["rms_error_y_m"] ** 2) / 2.0
                                                  for r in records]))),
        "rms_error_z_m": float(np.sqrt(np.mean([r["rms_error_z_m"] ** 2 for r in records]))),
    }
    if failing:
        out.update({
            "failing_min_abs_dz_m": min(abs(r["at_minimum_dz_m"]) for r in failing),
            "failing_median_abs_dz_m": float(np.median([abs(r["at_minimum_dz_m"]) for r in failing])),
            "failing_max_horizontal_m": max(r["at_minimum_horizontal_m"] for r in failing),
            "failing_median_abs_dz_at_1m_approach_m": float(np.median(
                [abs(r["dz_at_1m_approach_m"]) for r in failing])),
        })
    return out


def run(buffer_path: Path, output: Path, *, variants: list[str], per_kind: int, sample_seed: int,
        seeds_n: int, only: list[int] | None) -> dict[str, Any]:
    buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
    conditions = select_conditions(buffer["entries"], per_kind=per_kind, seed=sample_seed)
    if only is not None:
        conditions = [conditions[i] for i in only]
    seeds = seeds_for("separation", N_SEEDS)[:seeds_n]
    started = time.process_time()
    results = []
    for condition in conditions:
        base = {key: condition[key] for key in CONDITION_KEYS}
        entry = {"condition": condition, "variants": {}}
        for variant in variants:
            records = [diagnose_encounter(base, condition["radius_m"], seed, variant) for seed in seeds]
            entry["variants"][variant] = {"summary": summarise(records), "records": records}
        results.append(entry)
    report = {
        "simulated": True,
        "tool": "tools/d_diagnose_ellipsoid.py",
        "buffer": str(buffer_path),
        "seeds_n": seeds_n,
        "sample_seed": sample_seed,
        "per_kind": per_kind,
        "variants": variants,
        "ellipsoid_radii_m": list(ELLIPSOID_RADII_M),
        "min_vertical_offset_for_failure_m": min_vertical_offset_for_failure(),
        "altitude_hold_gain_per_s": ALTITUDE_HOLD_GAIN_PER_S,
        "cpu_seconds": time.process_time() - started,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


def comparison_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per sampled condition with each variant's centre and ellipsoid counts."""
    rows: dict[tuple, dict[str, Any]] = {}
    for report in reports:
        for entry in report["results"]:
            condition = entry["condition"]
            key = (condition["group"], *(condition[k] for k in CONDITION_KEYS), condition["radius_m"])
            row = rows.setdefault(key, {"condition": condition})
            for variant, data in entry["variants"].items():
                summary = data["summary"]
                row[variant] = {"center": summary["center_violations"], "ellipsoid": summary["ellipsoid_violations"],
                                "seeds": summary["seeds"]}
    return list(rows.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--buffer", type=Path, default=Path("runs/d4-separation/buffer.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--per-kind", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--only", default=None, help="comma-separated indices into the selected sample")
    parser.add_argument("--list", action="store_true", help="print the selected sample and exit")
    parser.add_argument("--compare", nargs="+", type=Path, help="print a table from earlier --output files")
    parser.add_argument("--vertical-origin", type=Path,
                        help="for each drifting condition in an earlier --output file, print the first vertical commands")
    args = parser.parse_args(argv)
    if args.vertical_origin:
        report = json.loads(args.vertical_origin.read_text(encoding="utf-8"))
        for entry in report["results"]:
            condition = entry["condition"]
            for variant, data in entry["variants"].items():
                drifting = [r for r in data["records"] if r["max_altitude_drift_m"] > 0.0]
                if not drifting:
                    continue
                worst = max(r["max_altitude_drift_m"] for r in drifting)
                events = first_vertical_commands({k: condition[k] for k in CONDITION_KEYS}, condition["radius_m"],
                                                 drifting[0]["seed"], variant)
                print(f"{variant} {condition['group']} {condition['kind']} delay={condition['delay_s']}: "
                      f"{len(drifting)} of {len(data['records'])} seeds drift (worst {worst:.3f} m); "
                      f"first vertical commands in seed {drifting[0]['seed']}: {json.dumps(events)}")
        return
    if args.compare:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        rows = comparison_rows(reports)
        names = [v for v in VARIANTS if any(v in row for row in rows)]
        print("group,kind,speed,delay,loss,dr,offset,radius," + ",".join(f"{v} centre/ellipsoid" for v in names))
        totals = {v: [0, 0, 0] for v in names}
        for row in rows:
            c = row["condition"]
            cells = []
            for v in names:
                if v in row:
                    cells.append(f"{row[v]['center']}/{row[v]['ellipsoid']} of {row[v]['seeds']}")
                    totals[v] = [totals[v][0] + row[v]["center"], totals[v][1] + row[v]["ellipsoid"],
                                 totals[v][2] + row[v]["seeds"]]
                else:
                    cells.append("-")
            print(f"{c['group']},{c['kind']},{c['speed_mps']},{c['delay_s']},{c['loss_probability']},"
                  f"{c['dead_reckoning']},{c['lateral_offset_m']},{c['radius_m']}," + ",".join(cells))
        print("total,,,,,,,," + ",".join(f"{t[0]}/{t[1]} of {t[2]}" for t in totals.values()))
        return
    if args.list:
        buffer = json.loads(args.buffer.read_text(encoding="utf-8"))
        for index, item in enumerate(select_conditions(buffer["entries"], per_kind=args.per_kind,
                                                       seed=args.sample_seed)):
            print(index, json.dumps(item, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --list is given")
    variants = [v for v in args.variants.split(",") if v]
    only = None if args.only is None else [int(x) for x in args.only.split(",")]
    report = run(args.buffer, args.output, variants=variants, per_kind=args.per_kind,
                 sample_seed=args.sample_seed, seeds_n=args.seeds, only=only)
    for entry in report["results"]:
        c = entry["condition"]
        label = (f"{c['group']:<22} {c['kind']:<9} v={c['speed_mps']} d={c['delay_s']} "
                 f"loss={c['loss_probability']} n={c['noise_std_m']} dr={c['dead_reckoning']} "
                 f"off={c['lateral_offset_m']} r={c['radius_m']} buffer.json ell={c['buffer_ellipsoid_count']}")
        print(label)
        for variant, data in entry["variants"].items():
            print(f"    {variant:<20} {json.dumps(data['summary'], sort_keys=True)}")
    print(f"cpu {report['cpu_seconds']:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
