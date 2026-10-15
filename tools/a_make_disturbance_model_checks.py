"""Write reproducible hand checks for the EXP-022 disturbance models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from sim.environment import AIR_DENSITY, GRAVITY
from sim.hardware import load_hardware_profile
from sim.interaction import downwash_wind_at
from sim.physics.disturbances import GroundEffect, OUGust
from sim.physics.run_manifest import write_manifest
from tools.a_run_disturbance_response import _DownwashSource, _profile_data, _trimmed_vehicle

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("crazyflie-class", "esp-fly-class")
PROFILE_PATHS = {
    "crazyflie-class": ROOT / "sim/physics/profiles/crazyflie_class.json",
    "esp-fly-class": ROOT / "sim/physics/profiles/esp_fly_class.json",
}


def _round(value):
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.9g}")
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _ou_one(sigma: float, seed: int, n: int = 100_000, dt: float = 0.01, time_constant: float = 1.0):
    gust = OUGust(np.zeros(3), np.full(3, sigma), time_constant, seed=seed)
    samples = np.array([gust.sample(0.0, dt, np.zeros(3), np.zeros(3)).wind_velocity_mps for _ in range(n)])
    n_eff = n * dt / (2.0 * time_constant)
    lags = (0.5, 1.0, 2.0)
    axes = []
    for axis in range(3):
        x = samples[:, axis]
        mean = float(np.mean(x)); mean_se = sigma / math.sqrt(n_eff)
        variance = float(np.var(x)); var_se = sigma * sigma * math.sqrt(2.0 / n_eff)
        corr = {}
        for lag_s in lags:
            lag = int(round(lag_s / dt))
            centered = x - mean
            rho = float(np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered * centered))
            expected = math.exp(-lag_s / time_constant)
            rho_se = math.sqrt((1.0 - rho * rho) ** 2 / n_eff)
            corr[str(lag_s)] = {"value": rho, "expected": expected, "se": rho_se,
                                "z": (rho - expected) / rho_se, "pass": abs(rho - expected) <= 3.0 * rho_se}
        axis_result = {
            "mean": mean, "mean_se": mean_se, "mean_z": mean / mean_se,
            "mean_pass": abs(mean) <= 3.0 * mean_se,
            "variance": variance, "expected_variance": sigma * sigma, "variance_se": var_se,
            "variance_z": (variance - sigma * sigma) / var_se,
            "variance_pass": abs(variance - sigma * sigma) <= 3.0 * var_se,
            "autocorrelation": corr,
        }
        axis_result["pass"] = bool(axis_result["mean_pass"] and axis_result["variance_pass"] and all(v["pass"] for v in corr.values()))
        axes.append(axis_result)
    return {"sigma_mps": sigma, "seed": seed, "n": n, "dt_s": dt, "time_constant_s": time_constant,
            "n_eff": n_eff, "autocorrelation_se_method": "sqrt((1-rho^2)^2/n_eff)", "axes": axes,
            "pass": all(a["pass"] for a in axes)}


def ou_checks(sigmas=(0.25, 0.5, 1.0), n=100_000, dt=0.01, seeds=None):
    seeds = tuple(range(len(sigmas))) if seeds is None else tuple(seeds)
    cells = [_ou_one(float(s), int(seed), int(n), float(dt)) for s, seed in zip(sigmas, seeds)]
    return {"T_s": 1.0, "cells": cells, "pass": all(c["pass"] for c in cells)}


def ground_effect_checks():
    cells = []
    for profile_id in PROFILES:
        profile = _profile_data(profile_id); radius = profile.rotor.prop_radius_m
        for multiple in (1.0, 2.0, 4.0):
            z = multiple * radius
            actual = GroundEffect(radius).sample(0, 0.01, (0, 0, z), (0, 0, 0)).thrust_scale
            expected = 1.0 / (1.0 - (radius / (4.0 * z)) ** 2)
            cells.append({"profile": profile_id, "height_R": multiple, "rotor_radius_m": radius,
                          "value": actual, "expected": expected, "difference": actual - expected,
                          "pass": abs(actual - expected) < 1e-12})
    return {"formula": "1 / (1 - (R / (4 z))^2)", "cells": cells, "pass": all(c["pass"] for c in cells)}


def downwash_checks():
    cells = []
    for profile_id in PROFILES:
        profile = _profile_data(profile_id); radius = profile.rotor.prop_radius_m
        thrust = profile.mass_kg * GRAVITY
        source = np.array((0.0, 0.0, 1.2)); target = np.array((0.0, 0.0, 1.0))
        wind = downwash_wind_at(source, thrust, profile.rotor_count, radius, target)
        area = 4.0 * math.pi * radius * radius
        hover = math.sqrt(thrust / (2.0 * AIR_DENSITY * area))
        vh = abs(float(wind[2])) / 2.0
        rel_vh = abs(vh - hover) / hover
        rel_far = abs(abs(float(wind[2])) - 2.0 * hover) / (2.0 * hover)
        cells.append({"profile": profile_id, "mass_kg": profile.mass_kg, "rotor_radius_m": radius,
                      "source_thrust_n": thrust, "area_m2": area, "v_h_mps": vh,
                      "expected_v_h_mps": hover, "far_wake_mps": abs(float(wind[2])),
                      "expected_far_wake_mps": 2.0 * hover, "v_h_relative_error": rel_vh,
                      "far_wake_relative_error": rel_far, "pass": rel_vh < 0.01 and rel_far < 0.01})
    return {"formula": "v_h = sqrt(T / (2 rho A)), A = 4 pi R^2; far wake = 2 v_h",
            "column_radius": "equivalent radius sqrt((N pi R^2)/pi) divided by sqrt(2), with N=4; max depth is 1 m",
            "cells": cells, "pass": all(c["pass"] for c in cells)}


def _downwash_cell(job):
    profile_id, drag_area, gap = job
    profile = _profile_data(profile_id); dt = 0.01; n = 1000
    target = np.array((0.0, 0.0, 1.0)); source = np.array((0.0, 0.0, 1.0 + gap))
    thrust = profile.mass_kg * GRAVITY
    vehicle = _trimmed_vehicle(profile, target, drag_area,
                               _DownwashSource(source, thrust, profile.rotor_count, profile.rotor.prop_radius_m))
    saturated_steps = 0; heights = []
    for i in range(n):
        step = vehicle.step(target, dt)
        saturated_steps += int(step.mixer.saturated)
        if i + 1 > n - 300:
            heights.append(float(vehicle.drone.position[2]))
    return {"profile": profile_id, "drag_area_m2": drag_area, "gap_m": gap, "saturated_steps": saturated_steps,
            "steps": n, "last_3s_min_z_m": min(heights), "last_3s_max_z_m": max(heights)}


def downwash_thrust_margin(workers=None):
    """Compare the wake push on a vehicle directly below with the thrust it has to spare."""
    rows = []
    for profile_id in PROFILES:
        profile = _profile_data(profile_id); radius = profile.rotor.prop_radius_m
        weight = profile.mass_kg * GRAVITY
        # Written out here rather than read from the profile helper: 4 rotors at full speed.
        max_thrust = profile.rotor_count * profile.rotor.thrust_coefficient_n_s2 * profile.rotor.max_speed_rad_s ** 2
        far_wake = 2.0 * math.sqrt(weight / (2.0 * AIR_DENSITY * 4.0 * math.pi * radius * radius))
        for drag_area in (0.002, 0.005, 0.012):
            push = 0.5 * AIR_DENSITY * drag_area * far_wake ** 2
            rows.append({"profile": profile_id, "drag_area_m2": drag_area, "weight_n": weight,
                         "max_thrust_n": max_thrust, "spare_thrust_n": max_thrust - weight,
                         "wake_push_n": push, "push_over_weight": push / weight,
                         "push_exceeds_spare_thrust": bool(push > max_thrust - weight)})
    jobs = [(p, a, g) for p in PROFILES for a in (0.002, 0.005, 0.012) for g in (0.2, 0.4, 0.6)]
    if workers is None: workers = max(1, (os.cpu_count() or 1) // 3)
    if workers <= 1:
        cells = [_downwash_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cells = list(pool.map(_downwash_cell, jobs))
    cells.sort(key=lambda r: (r["profile"], r["drag_area_m2"], r["gap_m"]))
    return {"note": "on-axis follower, same 10 s run as the main downwash cells; no floor is modeled",
            "margins": rows, "on_axis_cells": cells}


def _gust_monte_carlo(sigma, seed, n):
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, sigma, size=(n, 3))
    force = np.linalg.norm(g, axis=1)[:, None] * g
    w = np.array((1.5, 0.0, 0.0)); shifted = np.linalg.norm(g + w, axis=1)[:, None] * (g + w)
    centered0 = force - np.mean(force, axis=0); centered1 = shifted - np.mean(shifted, axis=0)
    gains = np.array((0.5, 0.5, 1.0 / 3.0))
    return {"sigma_mps": sigma, "n": n, "seed": seed,
            "rms_ratio_unweighted": float(np.sqrt(np.mean(centered1 ** 2)) / np.sqrt(np.mean(centered0 ** 2))),
            "rms_ratio_weighted": float(np.sqrt(np.mean((centered1 * gains) ** 2)) / np.sqrt(np.mean((centered0 * gains) ** 2))),
            "one_axis_formula": math.sqrt(1.0 + 2.0 * 1.5 * 1.5 / (sigma * sigma))}


def _gust_trajectory(job):
    profile_id, sigma, mean_wind, seed = job
    profile = _profile_data(profile_id); target = np.array((0.0, 0.0, 1.0)); dt = 0.01; n = 3000
    disturbance = OUGust(np.array((mean_wind, 0.0, 0.0)), np.full(3, sigma), 1.0, seed=seed)
    vehicle = _trimmed_vehicle(profile, target, 0.005, disturbance)
    positions = []
    for i in range(n):
        vehicle.step(target, dt)
        if i + 1 >= 500:
            positions.append(vehicle.drone.position.copy())
    positions = np.asarray(positions); offset = np.mean(positions, axis=0); centered = positions - offset
    return {"profile": profile_id, "sigma_mps": sigma, "mean_wind_mps": mean_wind, "seed": seed,
            "rms_about_mean_m": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
            "mean_offset_m": offset.tolist(), "plain_rms_m": float(np.sqrt(np.mean(np.sum((positions - target) ** 2, axis=1))))}


def _paired_ratio_interval(windy, still, seed=20261014, samples=4000):
    """95 % percentile bootstrap of median(windy) / median(still), resampling seeds in pairs.

    Both lists are in seed order and share random numbers seed by seed, so a seed is
    resampled together with its partner.
    """
    windy = np.asarray(windy, dtype=float); still = np.asarray(still, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(windy), size=(samples, len(windy)))
    ratios = np.median(windy[picks], axis=1) / np.median(still[picks], axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def gust_ratio_explanation(mc_n=2_000_000, mc_seed=20261014, run_seeds=range(50), workers=None):
    monte_carlo = [_gust_monte_carlo(float(sigma), mc_seed, int(mc_n)) for sigma in (0.25, 0.5)]
    jobs = [(p, float(s), float(w), int(seed)) for p in PROFILES for s in (0.25, 0.5) for w in (0.0, 1.5) for seed in run_seeds]
    if workers is None: workers = max(1, (os.cpu_count() or 1) // 3)
    if workers <= 1:
        rows = [_gust_trajectory(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_gust_trajectory, jobs))
    rows.sort(key=lambda r: (r["profile"], r["sigma_mps"], r["mean_wind_mps"], r["seed"]))
    groups = []
    for profile in PROFILES:
        for sigma in (0.25, 0.5):
            cells = {w: [r for r in rows if r["profile"] == profile and r["sigma_mps"] == sigma and r["mean_wind_mps"] == w] for w in (0.0, 1.5)}
            med = {w: float(np.median([r["rms_about_mean_m"] for r in cells[w]])) for w in cells}
            offsets = {w: np.mean([r["mean_offset_m"] for r in cells[w]], axis=0).tolist() for w in cells}
            low, high = _paired_ratio_interval(
                [r["rms_about_mean_m"] for r in cells[1.5]], [r["rms_about_mean_m"] for r in cells[0.0]])
            groups.append({"profile": profile, "sigma_mps": sigma, "median_rms_about_mean_m": med,
                           "mean_offset_m": offsets, "rms_ratio": med[1.5] / med[0.0],
                           "rms_ratio_ci95": [low, high], "n_seeds": len(cells[0.0])})
    committed = ROOT / "runs/a7-disturbance/per_run.csv"
    checks = []
    if committed.is_file():
        with committed.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("part") == "b" and row.get("seed") and row.get("rms_error_m"):
                    match = next((x for x in rows if x["profile"] == row["profile"] and x["sigma_mps"] == float(row["sigma_mps"]) and x["mean_wind_mps"] == float(row["mean_wind_mps"]) and x["seed"] == int(row["seed"])), None)
                    if match:
                        checks.append(abs(match["plain_rms_m"] - float(row["rms_error_m"])) <= 1e-9)
    return {"monte_carlo": monte_carlo, "run_seeds": list(run_seeds), "run_summary": groups,
            "committed_plain_rms_check": {"rows_checked": len(checks), "pass": bool(checks) and all(checks)}}


def build_checks(*, mc_n=2_000_000, mc_seed=20261014, ou_n=100_000, workers=None):
    return {"ou": ou_checks(n=ou_n), "ground_effect": ground_effect_checks(), "downwash": downwash_checks(),
            "gust_ratio_explanation": gust_ratio_explanation(mc_n=mc_n, mc_seed=mc_seed, workers=workers),
            "downwash_thrust_margin": downwash_thrust_margin(workers=workers)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    checks = _round(build_checks(workers=args.workers))
    (out / "model_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", schema="a7-disturbance-model-checks-manifest-v1", argv=sys.argv[1:],
                   config={"ou_n": 100000, "mc_n": 2000000, "mc_seed": 20261014}, seeds=[0, 1, 2, 20261014] + list(range(50)),
                   limits=["simulated only", "literature-derived profiles", "assumed drag area"],
                   outputs={"model_checks.json": out / "model_checks.json"}, output_dir=out,
                   runtime_s=perf_counter() - started, n_processes=args.workers or max(1, (os.cpu_count() or 1) // 3), repo=ROOT)


if __name__ == "__main__":
    main()
