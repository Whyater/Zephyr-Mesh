"""EXP-022 disturbance response sweep for the synthetic quadrotor chain."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from sim.environment import AIR_DENSITY, GRAVITY
from sim.hardware import load_hardware_profile
from sim.interaction import downwash_wind_at, ellipsoid_separation, is_separated
from sim.physics.disturbance_api import DisturbanceSample
from sim.physics.disturbances import ConstantWind, GroundEffect, OUGust
from sim.vehicle import ActuatedQuadrotor
from sim.physics.run_manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("a_disturbance_config.json")
PROFILE_PATHS = {
    "crazyflie-class": ROOT / "sim/physics/profiles/crazyflie_class.json",
    "esp-fly-class": ROOT / "sim/physics/profiles/esp_fly_class.json",
}

def _profile_data(profile_id: str):
    return load_hardware_profile(PROFILE_PATHS[profile_id])


def _trimmed_vehicle(profile, position, drag_area, disturbance=None):
    vehicle = ActuatedQuadrotor(
        profile,
        initial_position=position,
        drag_coefficient=1.0,
        cross_sectional_area=float(drag_area),
        substeps=1,
        disturbance=disturbance,
    )
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    vehicle.actuator.last_state = vehicle.actuator._state(saturated=False, thrust_scale=1.0)
    return vehicle


def _simulate(profile_id, part, params):
    profile = _profile_data(profile_id)
    dt = float(params["dt_s"])
    target = np.asarray(params.get("target", (0.0, 0.0, 1.0)), dtype=float)
    area = float(params.get("drag_area_m2", 0.005))
    if part == "steady":
        wind = float(params["wind_mps"])
        vehicle = _trimmed_vehicle(profile, target, area, ConstantWind((wind, 0.0, 0.0)))
        samples = []
        for _ in range(int(round(params["duration_s"] / dt))):
            vehicle.step(target, dt)
            samples.append(vehicle.drone.position.copy())
        tail = np.asarray(samples)[-max(1, min(len(samples), int(round(5.0 / dt)))) :]
        x = float(np.mean(tail[:, 0]))
        hand = 0.5 * AIR_DENSITY * area * wind * wind / (profile.mass_kg * 2.0)
        return {"part": "a", "profile": profile_id, "wind_mps": wind, "drag_area_m2": area,
                "steady_x_m": x, "steady_z_m": float(np.mean(tail[:, 2]) - target[2]),
                "hand_x_m": hand, "sim_hand_ratio": x / hand if hand else None}
    if part == "gust":
        mean_wind = float(params["mean_wind_mps"])
        sigma = float(params["sigma_mps"])
        seed = int(params["seed"])
        disturbance = OUGust(np.array((mean_wind, 0.0, 0.0)), np.full(3, sigma), float(params["time_constant_s"]), seed=seed)
        vehicle = _trimmed_vehicle(profile, target, area, disturbance)
        positions = []
        n = int(round(params["duration_s"] / dt))
        for index in range(n):
            vehicle.step(target, dt)
            if index + 1 >= max(1, int(round(5.0 / dt))) or n < int(round(5.0 / dt)):
                positions.append(vehicle.drone.position.copy())
        errors = np.linalg.norm(np.asarray(positions) - target, axis=1)
        return {"part": "b", "profile": profile_id, "mean_wind_mps": mean_wind, "sigma_mps": sigma,
                "seed": seed, "rms_error_m": float(np.sqrt(np.mean(errors * errors))),
                "max_error_m": float(np.max(errors))}
    if part == "ground":
        z = float(params["height_m"])
        disturbance = GroundEffect(profile.rotor.prop_radius_m)
        vehicle = _trimmed_vehicle(profile, (0.0, 0.0, z), area, disturbance)
        positions = []
        n = int(round(params["duration_s"] / dt))
        for _ in range(n):
            vehicle.step((0.0, 0.0, z), dt)
            positions.append(vehicle.drone.position.copy())
        tail = np.asarray(positions)[-max(1, min(len(positions), int(round(3.0 / dt)))) :]
        R = profile.rotor.prop_radius_m
        s = 1.0 / (1.0 - (R / (4.0 * z)) ** 2)
        # At steady state, the controller asks for m*g + m*Kp*(target-z),
        # while the actuator supplies s(z_actual) times that command. Thus
        # delta = g*(s(z+delta)-1)/(Kp*s(z+delta)); solve the resulting
        # one-dimensional fixed point because the disturbance reads current z.
        # f(e) = e - g (R / (4 (z + e)))^2 / Kp rises monotonically in e, so
        # bisection always finds the single root (plain fixed-point iteration
        # oscillates near z = R).
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid - GRAVITY * (R / (4.0 * (z + mid))) ** 2 / 3.0 > 0.0:
                hi = mid
            else:
                lo = mid
        hand = 0.5 * (lo + hi)
        return {"part": "c", "profile": profile_id, "height_m": z, "height_R": z / profile.rotor.prop_radius_m,
                "z_offset_m": float(np.mean(tail[:, 2]) - z), "hand_z_offset_m": hand,
                "sim_hand_ratio": (float(np.mean(tail[:, 2]) - z) / hand if hand else None)}
    if part == "ground_transition":
        mode = str(params["mode"])
        R = profile.rotor.prop_radius_m
        start_z, target_z = ((R, 0.5) if mode == "takeoff" else (0.5, R))
        max_errors = {}
        for label, disturbance in (("ground_effect", GroundEffect(R)), ("no_ground_effect", None)):
            vehicle = _trimmed_vehicle(profile, (0.0, 0.0, start_z), area, disturbance)
            n = int(round(params["duration_s"] / dt))
            errors = []
            for _ in range(n):
                vehicle.step((0.0, 0.0, target_z), dt)
                errors.append(abs(float(vehicle.drone.position[2]) - target_z))
            max_errors[label] = float(np.max(errors))
        return {"part": "c_transition", "profile": profile_id, "mode": mode,
                "start_z_m": start_z, "target_z_m": target_z,
                "max_z_error_ground_effect_m": max_errors["ground_effect"],
                "max_z_error_no_ground_effect_m": max_errors["no_ground_effect"],
                "max_z_error_difference_m": max_errors["ground_effect"] - max_errors["no_ground_effect"]}
    if part == "downwash":
        offset = float(params["offset_m"])
        gap = float(params["gap_m"])
        source = np.array((offset, 0.0, 1.0 + gap))
        target = np.array((0.0, 0.0, 1.0))
        thrust = profile.mass_kg * GRAVITY
        disturbance = _DownwashSource(source, thrust, profile.rotor_count, profile.rotor.prop_radius_m)
        vehicle = _trimmed_vehicle(profile, target, area, disturbance)
        positions = []
        n = int(round(params["duration_s"] / dt))
        for _ in range(n):
            vehicle.step(target, dt)
            positions.append(vehicle.drone.position.copy())
        tail = np.asarray(positions)[-max(1, min(len(positions), int(round(3.0 / dt)))) :]
        sag = float(target[2] - np.mean(tail[:, 2]))
        wake = downwash_wind_at(source, thrust, profile.rotor_count, profile.rotor.prop_radius_m, target)
        hand_force = 0.5 * AIR_DENSITY * area * float(wake[2] ** 2)
        hand_sag = hand_force / (profile.mass_kg * 3.0)
        separated = is_separated(source, target)
        return {"part": "d", "profile": profile_id, "offset_m": offset, "gap_m": gap, "drag_area_m2": area,
                "z_sag_m": sag, "hand_z_sag_m": hand_sag, "sim_hand_ratio": sag / hand_sag if hand_sag else None,
                "separation_normalized": ellipsoid_separation(source, target), "is_separated": separated,
                "pushed_out": bool(np.min(tail[:, 2]) < 0.5), "wind_z_mps": float(wake[2])}
    raise ValueError(part)


class _DownwashSource:
    def __init__(self, position, thrust, rotor_count, rotor_radius):
        self.position = np.asarray(position, dtype=float)
        self.thrust = float(thrust)
        self.rotor_count = int(rotor_count)
        self.rotor_radius = float(rotor_radius)

    def sample(self, t_s, dt_s, position_m, velocity_mps):
        del t_s, dt_s, velocity_mps
        wind = downwash_wind_at(self.position, self.thrust, self.rotor_count, self.rotor_radius, position_m)
        return DisturbanceSample(wind_velocity_mps=wind)


def _run_jobs(jobs, workers):
    if workers <= 1:
        return [_simulate(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_simulate_job, jobs))


def _simulate_job(job):
    return _simulate(*job)


def _percentile_bootstrap(values, seed, samples):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.median(values[rng.integers(0, len(values), size=(int(samples), len(values)))], axis=1)
    return float(np.median(values)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            def rendered(value):
                if value is None:
                    return ""
                if isinstance(value, (float, np.floating)):
                    return f"{float(value):.9g}"
                return value
            writer.writerow({key: rendered(value) for key, value in row.items()})


def _plot(summary, path):
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style, palette
    pal = apply_style("light")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = [pal.get("accent", "#28698E"), pal.get("textSecondary", "#5C6770"), pal.get("text", "#222222")]
    ax = axes[0, 0]
    for pi, profile in enumerate(summary["profiles"]):
        for ai, area in enumerate(summary["steady"][profile]):
            cells = summary["steady"][profile][area]
            wind = np.array([c["wind_mps"] for c in cells]); sim = np.array([c["steady_x_m"] for c in cells])
            hand = np.array([c["hand_x_m"] for c in cells])
            ax.plot(wind, sim, color=colors[pi], linestyle=("-", "--", ":")[ai], label=f"{profile}, CdA={area:g}")
            ax.plot(wind, hand, color=colors[pi], linestyle="--", alpha=0.35)
    ax.set(title="(a) Steady wind offset", xlabel="W (m/s)", ylabel="mean x offset (m)")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[0, 1]
    for pi, profile in enumerate(summary["profiles"]):
        for wi, mean_wind in enumerate((0.0, 1.5)):
            cells = [c for c in summary["gust"][profile] if c["mean_wind_mps"] == mean_wind]
            if not cells: continue
            x = [c["sigma_mps"] for c in cells]; y = [c["rms_median_m"] for c in cells]
            lo = [c["rms_ci_low_m"] for c in cells]; hi = [c["rms_ci_high_m"] for c in cells]
            ax.errorbar(x, y, yerr=[np.array(y)-lo, np.array(hi)-np.array(y)], color=colors[pi], linestyle="-" if wi == 0 else "--", marker="o", label=f"{profile}, W={mean_wind:g}")
    ax.set(title="(b) Gust RMS error", xlabel="sigma (m/s)", ylabel="RMS 3-D error (m)"); ax.legend(fontsize=6)
    ax = axes[1, 0]
    for pi, profile in enumerate(summary["profiles"]):
        cells = summary["ground"][profile]
        x = np.array([c["height_R"] for c in cells]); y = np.array([c["z_offset_m"] for c in cells]); h = np.array([c["hand_z_offset_m"] for c in cells])
        ax.plot(x, y, color=colors[pi], marker="o", label=profile); ax.plot(x, h, color=colors[pi], linestyle="--", alpha=0.45)
    ax.set(title="(c) Ground effect", xlabel="height / R", ylabel="z offset (m)"); ax.legend(fontsize=7)
    ax = axes[1, 1]
    for pi, profile in enumerate(summary["profiles"]):
        for ai, area in enumerate(summary["drag_areas_m2"]):
            cells = [c for c in summary["downwash"][profile] if c["drag_area_m2"] == area and c["offset_m"] == 0.0]
            cells.sort(key=lambda c: c["gap_m"])
            ax.plot([c["gap_m"] for c in cells], [c["z_sag_m"] for c in cells], color=colors[pi], linestyle=("-", "--", ":")[ai], label=f"{profile}, CdA={area:g}")
    ax.axvline(0.60, color=pal.get("caution", "#B27A00"), linestyle="--", linewidth=1)
    ax.set(title="(d) Downwash sag, on axis", xlabel="gap (m)", ylabel="z sag (m)"); ax.legend(fontsize=6, ncol=2)
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def run(output_dir, workers=None, pilot=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if pilot:
        cfg["steady_duration_s"] = 0.2; cfg["gust_duration_s"] = 0.2; cfg["ground_duration_s"] = 0.2; cfg["downwash_duration_s"] = 0.2
        cfg["steady_winds_mps"] = [1.0]; cfg["drag_areas_m2"] = [0.005]
        cfg["gust_sigmas_mps"] = [0.5]; cfg["gust_mean_winds_mps"] = [0.0, 1.5]; cfg["gust_seeds"] = [0, 1]
        cfg["ground_heights_m"] = ["2R"]
        cfg["downwash_offsets_m"] = [0.0]; cfg["downwash_gaps_m"] = [0.2]
    profiles = ["crazyflie-class", "esp-fly-class"]
    jobs = []
    for profile in profiles:
        for wind in cfg["steady_winds_mps"]:
            for area in cfg["drag_areas_m2"]:
                jobs.append((profile, "steady", {"dt_s": cfg["dt_s"], "duration_s": cfg["steady_duration_s"], "wind_mps": wind, "drag_area_m2": area}))
    for profile in profiles:
        for mean_wind in cfg["gust_mean_winds_mps"]:
            for sigma in cfg["gust_sigmas_mps"]:
                for seed in cfg["gust_seeds"]:
                    jobs.append((profile, "gust", {"dt_s": cfg["dt_s"], "duration_s": cfg["gust_duration_s"], "mean_wind_mps": mean_wind, "sigma_mps": sigma, "time_constant_s": cfg["gust_time_constant_s"], "seed": seed}))
    for profile in profiles:
        p = _profile_data(profile); R = p.rotor.prop_radius_m
        for value in cfg["ground_heights_m"]:
            if isinstance(value, str):
                factor = 1.0 if value == "R" else float(value[:-1])
                z = factor * R
            else:
                z = float(value)
            jobs.append((profile, "ground", {"dt_s": cfg["dt_s"], "duration_s": cfg["ground_duration_s"], "height_m": z}))
        for mode in ("takeoff", "landing"):
            jobs.append((profile, "ground_transition", {"dt_s": cfg["dt_s"], "duration_s": cfg["ground_duration_s"], "mode": mode}))
    for profile in profiles:
        for offset in cfg["downwash_offsets_m"]:
            for gap in cfg["downwash_gaps_m"]:
                for area in cfg["drag_areas_m2"]:
                    jobs.append((profile, "downwash", {"dt_s": cfg["dt_s"], "duration_s": cfg["downwash_duration_s"], "offset_m": offset, "gap_m": gap, "drag_area_m2": area}))
    start = perf_counter()
    rows = _run_jobs(jobs, int(workers or max(1, (os.cpu_count() or 1) // 3)))
    rows.sort(key=lambda r: (r["part"], r["profile"], r.get("wind_mps", -1), r.get("mean_wind_mps", -1), r.get("sigma_mps", -1), r.get("seed", -1), r.get("height_m", -1), r.get("offset_m", -1), r.get("gap_m", -1), r.get("drag_area_m2", -1)))
    _write_csv(output_dir / "per_run.csv", rows)
    summary = {"schema": "a7-disturbance-response-summary-v1", "profiles": profiles, "drag_areas_m2": cfg["drag_areas_m2"], "steady": {}, "gust": {}, "ground": {}, "downwash": {}}
    for profile in profiles:
        summary["steady"][profile] = {}
        for area in cfg["drag_areas_m2"]:
            summary["steady"][profile][area] = [r for r in rows if r["part"] == "a" and r["profile"] == profile and r["drag_area_m2"] == area]
        summary["gust"][profile] = []
        for mean_wind in cfg["gust_mean_winds_mps"]:
            for sigma in cfg["gust_sigmas_mps"]:
                cells = [r for r in rows if r["part"] == "b" and r["profile"] == profile and r["mean_wind_mps"] == mean_wind and r["sigma_mps"] == sigma]
                rms = [r["rms_error_m"] for r in cells]; mx = [r["max_error_m"] for r in cells]
                med, lo, hi = _percentile_bootstrap(rms, cfg["bootstrap_seed"], cfg["bootstrap_samples"])
                mm, ml, mh = _percentile_bootstrap(mx, cfg["bootstrap_seed"], cfg["bootstrap_samples"])
                if pilot:
                    order_lo = order_hi = med
                else:
                    from sim.s6_recovery import median_interval
                    _, order_lo, order_hi = median_interval(rms)
                ratio = math.sqrt(1.0 + 2.0 * mean_wind * mean_wind / (sigma * sigma)) if sigma else None
                summary["gust"][profile].append({"mean_wind_mps": mean_wind, "sigma_mps": sigma, "rms_median_m": med, "rms_ci_low_m": lo, "rms_ci_high_m": hi, "rms_order_interval_low_m": order_lo, "rms_order_interval_high_m": order_hi, "max_median_m": mm, "max_ci_low_m": ml, "max_ci_high_m": mh, "hand_force_variance_ratio": 1.0 + 2.0 * mean_wind * mean_wind / (sigma * sigma), "hand_rms_ratio": ratio})
        for cell in summary["gust"][profile]:
            baseline = next((x for x in summary["gust"][profile] if x["sigma_mps"] == cell["sigma_mps"] and x["mean_wind_mps"] == 0), None)
            cell["sim_rms_ratio_vs_zero"] = (cell["rms_median_m"] / baseline["rms_median_m"] if baseline and baseline["rms_median_m"] else None)
        summary["ground"][profile] = [r for r in rows if r["part"] == "c" and r["profile"] == profile]
        summary.setdefault("ground_transitions", {})[profile] = [r for r in rows if r["part"] == "c_transition" and r["profile"] == profile]
        summary["downwash"][profile] = [r for r in rows if r["part"] == "d" and r["profile"] == profile]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(summary, output_dir / "a7_disturbance.png")
    limits = ["simulated only", "literature-derived profiles", "drag area is an assumption sweep (UNCONFIRMED)", "downwash wake has no decay (conservative)", "single-disk wake lumps four rotors"]
    runtime = perf_counter() - start
    write_manifest(output_dir / "manifest.json", schema="a7-disturbance-response-manifest-v1", config=cfg, seeds=cfg["gust_seeds"], limits=limits, outputs={"per_run.csv": output_dir / "per_run.csv", "summary.json": output_dir / "summary.json", "a7_disturbance.png": output_dir / "a7_disturbance.png"}, output_dir=output_dir, runtime_s=runtime, n_processes=int(workers or max(1, (os.cpu_count() or 1) // 3)), repo=ROOT)
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.workers, args.pilot)


if __name__ == "__main__":
    main()
