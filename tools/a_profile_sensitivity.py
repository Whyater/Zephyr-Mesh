"""One-at-a-time sensitivity sweep for the two literature-derived profiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.environment import AIR_DENSITY, GRAVITY
from sim.hardware import RotorcraftProfile, load_hardware_profile
from sim.physics.control_tolerance import find_d_crit, run_episode
from sim.physics.run_manifest import write_manifest
from sim.vehicle import ActuatedQuadrotor
from tools.a_dcrit_hand import estimate

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "sim" / "physics" / "profiles"
PARAMETERS = (
    "airframe.mass_kg", "airframe.inertia_kg_m2", "airframe.arm_length_m",
    "rotor.thrust_coefficient_n_s2", "rotor.torque_to_thrust_m", "rotor.max_speed_rad_s",
    "rotor.motor_time_constant_s", "rotor.prop_radius_m",
)
OUTPUTS = ("hover_speed_rad_s", "thrust_to_weight", "p_mech_w", "p_ideal_w", "power_ratio",
           "max_climb_hand_mps2", "max_climb_sim_mps2", "arm_b_dcrit_s", "slice_c_dcrit_s",
           "hand_xy_dcrit_s", "hand_whole_state_dcrit_s")


def _set_path(doc, path, value):
    section, field = path.split(".")
    doc[section][field] = value


def build_variant(profile: RotorcraftProfile, parameter: str, level: str) -> RotorcraftProfile:
    """Build a low/high variant by editing one document field."""
    doc = profile.as_document()
    entry = profile.parameters[parameter]
    bounds = entry["range"]
    if level in {"low", "high"}:
        endpoint = 0 if level == "low" else 1
        if parameter == "airframe.inertia_kg_m2":
            value = [float(x[endpoint]) for x in bounds]
        else:
            value = float(bounds[endpoint])
    else:
        value = level
    _set_path(doc, parameter, value)
    doc["parameters"][parameter]["value"] = value
    doc["status"] = "literature-derived"
    doc["notes"] = profile.notes + " Sensitivity variant: one parameter moved to its declared range endpoint."
    return RotorcraftProfile.from_dict(doc)


def _climb(profile):
    dt, window = 0.01, 0.2
    settle = 10.0 * profile.rotor.motor_time_constant_s
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=1)
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    for _ in range(int(round(settle / dt))):
        vehicle.step((0.0, 0.0, 50.0), dt)
    v0 = float(vehicle.drone.velocity[2])
    for _ in range(int(round(window / dt))):
        vehicle.step((0.0, 0.0, 50.0), dt)
    return (float(vehicle.drone.velocity[2]) - v0) / window


def _dcrit(profile, arm, bracket):
    """Delay limit on the 10 ms grid, or 0 when the loop is unstable with no delay at all."""
    zero = run_episode(profile, arm, "step", 0.0, 0.0, 0)
    if zero["unstable"]:
        # Some range ends (for example a slow motor) leave no margin even before any delay.
        return {"stable_delay_s": None, "unstable_delay_s": 0.0, "d_crit_s": 0.0,
                "unstable_at_zero_delay": True, "zero_delay_reason": zero["unstable_reason"]}
    try:
        return find_d_crit(profile, arm=arm, bracket=bracket)
    except ValueError as exc:
        if "upper bracket" not in str(exc):
            raise
    widened = (bracket[0], bracket[1] * 2.0)
    out = find_d_crit(profile, arm=arm, bracket=widened)
    out["bracket_widened"] = True
    out["bracket_used"] = list(widened)
    return out


def _metrics(profile):
    m, kf, ratio, w = profile.mass_kg, profile.thrust_coefficient_n_s2, profile.torque_to_thrust_m, profile.hover_speed_rad_s()
    km = ratio * kf
    p_mech = 4.0 * km * w**3
    area = 4.0 * math.pi * profile.rotor.prop_radius_m**2
    p_ideal = (m * GRAVITY) ** 1.5 / math.sqrt(2.0 * AIR_DENSITY * area)
    hand_climb = 4.0 * kf * profile.max_speed_rad_s**2 / m - GRAVITY
    arm = _dcrit(profile, "b", (0.0, 0.6))
    whole = _dcrit(profile, "c", (0.0, 0.1))
    hand = estimate(tau=profile.motor_time_constant_s)
    return {
        "hover_speed_rad_s": w, "thrust_to_weight": profile.thrust_to_weight(),
        "p_mech_w": p_mech, "p_ideal_w": p_ideal, "power_ratio": p_ideal / p_mech,
        "max_climb_hand_mps2": hand_climb, "max_climb_sim_mps2": _climb(profile),
        "arm_b_dcrit_s": arm.get("d_crit_s"), "arm_b_stable_delay_s": arm.get("stable_delay_s"),
        "arm_b_unstable_delay_s": arm.get("unstable_delay_s"), "arm_b_bracket_widened": arm.get("bracket_widened", False),
        "arm_b_unstable_at_zero_delay": arm.get("unstable_at_zero_delay", False),
        "slice_c_dcrit_s": whole.get("d_crit_s"), "slice_c_stable_delay_s": whole.get("stable_delay_s"),
        "slice_c_unstable_delay_s": whole.get("unstable_delay_s"), "slice_c_bracket_widened": whole.get("bracket_widened", False),
        "slice_c_unstable_at_zero_delay": whole.get("unstable_at_zero_delay", False),
        "hand_xy_dcrit_s": hand["xy"]["d_crit_s"], "hand_whole_state_dcrit_s": hand["attitude_whole_state"]["d_crit_s"],
    }


def _worker(payload):
    name, doc, parameter, level = payload
    profile = RotorcraftProfile.from_dict(doc)
    return {"profile": name, "parameter": parameter, "level": level, "metrics": _metrics(profile)}


def _rel(value, nominal):
    if value is None or nominal is None or nominal == 0:
        return None
    return (float(value) - float(nominal)) / float(nominal)


def _round(v):
    if isinstance(v, float): return float(f"{v:.9g}")
    if isinstance(v, dict): return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list): return [_round(x) for x in v]
    return v


def _figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools import horizon_mpl
    horizon_mpl.apply_style("light"); pal = horizon_mpl.palette("light")
    # Accent and textSecondary are ordinary data colors, rather than status colors.
    low_color, high_color = pal["accent"], pal["textSecondary"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    panels = (("p_mech_w", "hover power (mechanical W)"), ("max_climb_sim_mps2", "max climb (m/s²)"),
              ("arm_b_dcrit_s", "arm b d_crit (s)"), ("slice_c_dcrit_s", "slice c d_crit (s)"))
    for row, name in enumerate(sorted(data)):
        for col, (metric, title) in enumerate(panels):
            ax = axes[row, col]; vals = []
            for item in data[name]["parameters"]:
                p = item["parameter"]
                vals.append((p, max(abs(x or 0) for x in (item["relative"][metric]["low"], item["relative"][metric]["high"]))))
            vals.sort(key=lambda x: x[1]); labels = [x[0] for x in vals]
            y = np.arange(len(labels)); lo = [data[name]["parameter_results"][x]["relative"][metric]["low"] or 0 for x in labels]; hi = [data[name]["parameter_results"][x]["relative"][metric]["high"] or 0 for x in labels]
            ax.barh(y - .18, lo, .34, color=low_color, label="low" if row == 0 and col == 0 else None)
            ax.barh(y + .18, hi, .34, color=high_color, label="high" if row == 0 and col == 0 else None)
            ax.set_yticks(y, [x.split(".")[-1] for x in labels]); ax.set_title(title); ax.axvline(0, color=pal["lineStrong"], lw=.6)
            if row == 1: ax.set_xlabel("relative change")
    axes[0,0].legend(); fig.suptitle("Sensitivity of simulated, literature-derived ranges", y=1.02)
    fig.savefig(path, dpi=150, metadata={"Software": None}); plt.close(fig)


def generate(output_dir: Path, workers=None):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    profiles = {name: load_hardware_profile(PROFILE_DIR / f"{name.replace('-', '_')}.json") for name in ("crazyflie-class", "esp-fly-class")}
    payloads = []
    for name, p in profiles.items():
        payloads.append((name, p.as_document(), "nominal", "nominal"))
        for param in PARAMETERS:
            for level in ("low", "high"):
                payloads.append((name, build_variant(p, param, level).as_document(), param, level))
    n_workers = workers or max(1, (os.cpu_count() or 1) // 3)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(_worker, payloads))
    rows.sort(key=lambda r: (r["profile"], r["parameter"], r["level"]))
    nominal = {r["profile"]: r["metrics"] for r in rows if r["parameter"] == "nominal"}
    result = {"schema": "zephyr-a5-profile-sensitivity-1", "threshold_relative": 0.05,
              "dcrit_grid_resolution_s": 0.01,
              "dcrit_resolution_note": "Changes within one 10 ms control-grid step are not resolvable.",
              "excluded": ["rotor.count", "rotor.spin_directions", "airframe.drag_coefficient", "airframe.cross_sectional_area_m2"], "profiles": {}}
    csv_rows = []
    for name in profiles:
        pr = [r for r in rows if r["profile"] == name and r["parameter"] != "nominal"]
        param_results = {}
        for param in PARAMETERS:
            levels = {r["level"]: r for r in pr if r["parameter"] == param}
            rel = {metric: {lev: _rel(levels[lev]["metrics"].get(metric), nominal[name].get(metric)) for lev in ("low", "high")} for metric in OUTPUTS}
            param_results[param] = {"low": levels["low"]["metrics"], "high": levels["high"]["metrics"], "relative": rel}
            for lev in ("low", "high"):
                csv_rows.append({"profile": name, "parameter": param, "level": lev, **{f"{m}_relative": rel[m][lev] for m in OUTPUTS}})
        ranking = {}
        for metric in OUTPUTS:
            ranking[metric] = sorted(({"parameter": p, "max_abs_relative": max(abs(param_results[p]["relative"][metric][x] or 0) for x in ("low", "high")), "matters": max(abs(param_results[p]["relative"][metric][x] or 0) for x in ("low", "high")) > .05} for p in PARAMETERS), key=lambda x: (-x["max_abs_relative"], x["parameter"]))
        result["profiles"][name] = {"nominal": nominal[name], "parameter_results": param_results, "parameters": [{"parameter": p, "relative": param_results[p]["relative"]} for p in PARAMETERS], "ranking": ranking}
    (output_dir / "sensitivity.json").write_text(json.dumps(_round(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["profile", "parameter", "level"] + [f"{m}_relative" for m in OUTPUTS]
    with (output_dir / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in csv_rows: writer.writerow({k: ("" if v is None else f"{v:.9g}" if isinstance(v, float) else v) for k, v in row.items()})
    _figure(result["profiles"], output_dir / "a5_sensitivity.png")
    runtime = time.perf_counter() - started
    write_manifest(output_dir / "manifest.json", schema="zephyr-a5-profile-sensitivity-manifest-1", argv=sys.argv[1:], config={"profiles": list(profiles), "parameters": PARAMETERS}, seeds=[], limits=["simulated only", "literature-derived profiles and ranges", "one parameter at a time, no interactions", "d_crit on the 10 ms control grid"], outputs=[output_dir / "sensitivity.json", output_dir / "sensitivity.csv", output_dir / "a5_sensitivity.png"], output_dir=output_dir, runtime_s=runtime, n_processes=n_workers)
    print(f"runtime_s={runtime:.3f}")
    print(json.dumps({name: result["profiles"][name]["ranking"] for name in result["profiles"]}, indent=2, sort_keys=True))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True, type=Path); ap.add_argument("--workers", type=int); args = ap.parse_args(argv); generate(args.output_dir, args.workers); return 0


if __name__ == "__main__":
    raise SystemExit(main())
