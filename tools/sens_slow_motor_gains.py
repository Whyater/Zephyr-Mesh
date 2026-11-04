"""Side run: which attitude-gain scale keeps a slow motor stable, and what it costs.

The A5 sensitivity run (sim/runs/a5-sensitivity) found that the ESP-FLY-class
profile with a 0.16 s motor time constant, the top of its listed range, is
unstable even with no link delay.  A slow motor delivers the requested torque
late, and that lateness eats the attitude loop's phase margin.  The usual fix
is a slower attitude loop.  This wrapper scales the attitude gains with one
number ``s`` while keeping the damping ratio:

    K_p_att = 100 * s**2,   K_d_att = 18 * s

so the attitude loop's natural frequency is 10 * s rad/s and its damping
ratio stays 0.9.  Position gains are not touched.

For every (motor lag, s) cell it records the zero-delay verdict and scans
the arm b (position feedback) and slice c (whole-state) delays on the 10 ms
grid.  It scans every grid delay instead of bisecting, because the EXP-021
verdict is not always monotone in delay (see tools/sens_a6_drag.py).  Each
flight records both the EXP-021 verdict and the stricter "failed to fly"
verdict (the failed-to-fly rule, still Proposed); the flight itself is the same
for both.

The gains are patched only here: ``sim.physics.control_tolerance``'s
``GeometricFlightController`` name is replaced by a subclass for the duration
of one run, and every controller built is logged so the output proves which
gains flew.  ``sim/controller.py`` is not edited.  The motor lag is set on an
in-memory copy of the profile.  Everything runs in one process.

Controls run first: with s = 1.0 the registered arm b and slice c d_crit
(sim/runs/a6-control-tolerance) and the A5 slow-motor failure must reproduce,
or no variant runs.  Cells are checkpointed, so a rerun with the same output
folder resumes.  Outputs are simulated only and go outside sim/runs.

``--mode lag-gap`` runs the motor-lag gap sweep instead: lags 0.10 to 0.15 s
on the same scale grid, plus a zero-delay stock-gain scan from 0.100 to
0.160 s in 2 ms steps, and reports for each lag the largest grid scale that
flies and the largest that keeps a 10 degree hand margin, with the delay
limits at each.  The default mode (``--mode n18``) is unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sim.physics.control_tolerance as ct
from sim.hardware import RotorcraftProfile, load_hardware_profile
from tools.a_dcrit_hand import estimate

PROFILE_ID = "esp-fly-class"
PROFILE_PATH = ROOT / "sim" / "physics" / "profiles" / "esp_fly_class.json"
CONFIG_PATH = ROOT / "tools" / "a_control_tolerance_config.json"
REGISTERED_DCRIT = ROOT / "sim" / "runs" / "a6-control-tolerance" / "d_crit.json"
REGISTERED_A5 = ROOT / "sim" / "runs" / "a5-sensitivity" / "sensitivity.json"
TAU_PARAM = "rotor.motor_time_constant_s"

# The controller the harness would build without this wrapper.
BASE_CONTROLLER = ct.GeometricFlightController
_BASE = BASE_CONTROLLER()
BASE_KP_ATT = tuple(float(x) for x in _BASE.K_p_att)
BASE_KD_ATT = tuple(float(x) for x in _BASE.K_d_att)
del _BASE

TAUS_S = (0.03, 0.07, 0.10, 0.16)
SCALES = (1.0, 0.9, 0.8, 0.65, 0.5)
FINE_TAU_S = 0.16
FINE_SCALES = tuple(round(0.50 + 0.02 * i, 2) for i in range(26))  # 0.50 to 1.00
ARM_B_TICKS = tuple(range(51))   # 0.00 to 0.50 s, the registered arm b bracket
ARM_C_TICKS = tuple(range(21))   # 0.00 to 0.20 s, wider than the registered 0.08 s
EPISODE_DURATION_S = 20.0
SEED = 0
HOLD_S = ct.CONTROL_DT_S / 2.0      # average extra age from the 10 ms torque hold
MARGIN_TARGET_DEG = 10.0
STABILITY_RULE = "stable = EXP-021 'unstable' false AND the failed-to-fly rule's 'failed_to_fly' false"
GAP_TAUS_S = tuple(round(0.10 + 0.01 * i, 2) for i in range(6))           # 0.10 to 0.15
GAP_FINE_TAUS_S = tuple(round(0.100 + 0.002 * i, 3) for i in range(31))   # 0.100 to 0.160
ROW_KEYS = ("rms_error_m", "max_error_m", "whole_run_max_error_m", "whole_run_max_tilt_deg",
            "whole_run_min_z_m", "unstable", "unstable_reason", "unstable_time_s",
            "failed_to_fly", "failed_to_fly_reason", "failed_to_fly_time_s")


def scaled_gains(scale: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Attitude gains for one scale, with the damping ratio kept."""
    s = float(scale)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("scale must be finite and positive")
    return tuple(k * s * s for k in BASE_KP_ATT), tuple(k * s for k in BASE_KD_ATT)


@contextlib.contextmanager
def gain_patch(scale: float) -> Iterator[list[tuple[tuple[float, ...], tuple[float, ...]]]]:
    """Build every harness controller with scaled attitude gains, and log them."""
    kp, kd = scaled_gains(scale)
    built: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    class ScaledController(BASE_CONTROLLER):
        def __init__(self):
            super().__init__()
            self.K_p_att = np.array(kp, dtype=float)
            self.K_d_att = np.array(kd, dtype=float)
            built.append((tuple(float(x) for x in self.K_p_att), tuple(float(x) for x in self.K_d_att)))

    original = ct.GeometricFlightController
    ct.GeometricFlightController = ScaledController
    try:
        yield built
    finally:
        ct.GeometricFlightController = original


def profile_with_tau(tau_s: float) -> RotorcraftProfile:
    """The ESP-FLY-class profile with one motor lag, built in memory.

    The profile's own nominal lag returns the file profile unchanged.  A lag
    outside the declared range widens that range in memory (it is a side-run
    variant, not a new literature value) and says so in the notes.
    """
    base = load_hardware_profile(PROFILE_PATH)
    tau = float(tau_s)
    if tau == base.motor_time_constant_s:
        return base
    doc = base.as_document()
    doc["rotor"]["motor_time_constant_s"] = tau
    entry = doc["parameters"][TAU_PARAM]
    entry["value"] = tau
    lo, hi = entry["range"]
    if not lo <= tau <= hi:
        entry["range"] = [min(lo, tau), max(hi, tau)]
        entry["note"] = (entry.get("note", "") + " Side run: range widened in memory to hold "
                         f"{tau:g} s; no source gives this value.").strip()
    doc["status"] = "literature-derived"
    doc["notes"] = base.notes + f" Side-run variant: motor time constant set to {tau:g} s."
    return RotorcraftProfile.from_dict(doc)


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _episode(profile, arm: str, tick: int, scoring_windows) -> dict[str, Any]:
    out = ct.run_episode(profile, arm, "step", tick * ct.CONTROL_DT_S, 0.0, SEED,
                         duration_s=EPISODE_DURATION_S, scoring_windows=scoring_windows,
                         failed_to_fly_rule=True)
    return {"delay_s": round(tick * ct.CONTROL_DT_S, 10), **{k: out[k] for k in ROW_KEYS}}


def first_failure(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """First failing grid delay for one verdict, and any later delay that passes again."""
    failing = [row["delay_s"] for row in rows if row[key]]
    if not failing:
        return {"first_fail_delay_s": None, "monotone": True, "passes_after_first_fail_s": []}
    first = failing[0]
    again = [row["delay_s"] for row in rows if row["delay_s"] > first and not row[key]]
    return {"first_fail_delay_s": first, "monotone": not again, "passes_after_first_fail_s": again}


def _limit(rows: list[dict[str, Any]]) -> float | None:
    """First grid delay that fails either verdict (the pre-registered rule)."""
    both = [dict(row, either=bool(row["unstable"] or row["failed_to_fly"])) for row in rows]
    return first_failure(both, "either")["first_fail_delay_s"]


def hand(tau_s: float, scale: float) -> dict[str, float]:
    """Linearized hand numbers (tools/a_dcrit_hand.py) for one cell."""
    kp, kd = scaled_gains(scale)
    e = estimate(kp_att=kp[0], kd_att=kd[0], tau=tau_s)
    att = e["attitude_whole_state"]
    return {
        "attitude_omega_c": att["omega_c"],
        "attitude_pm_deg": att["pm_deg"],
        "attitude_pm_minus_hold_deg": att["pm_deg"] - math.degrees(att["omega_c"] * HOLD_S),
        "xy_dcrit_sim_expected_s": e["xy"]["d_crit_sim_expected_s"],
        "whole_state_dcrit_sim_expected_s": att["d_crit_sim_expected_s"],
    }


def run_cell(tau_s: float, scale: float, scoring_windows, *, scan: bool = True) -> dict[str, Any]:
    """Zero-delay verdicts, then (if stable) full arm b and slice c delay scans."""
    profile = profile_with_tau(tau_s)
    with gain_patch(scale) as built:
        zero_b = _episode(profile, "b", 0, scoring_windows)
        zero_c = _episode(profile, "c", 0, scoring_windows)
        out: dict[str, Any] = {
            "tau_s": float(tau_s), "scale": float(scale),
            "effective_motor_time_constant_s": profile.motor_time_constant_s,
            "zero_delay": {"b": zero_b, "c": zero_c},
            "stable_at_zero_delay": not (zero_b["unstable"] or zero_b["failed_to_fly"]),
            "hand": hand(tau_s, scale),
        }
        if scan:
            for arm, ticks, zero in (("b", ARM_B_TICKS, zero_b), ("c", ARM_C_TICKS, zero_c)):
                if zero["unstable"] or zero["failed_to_fly"]:
                    out[f"scan_{arm}"] = None
                    continue
                rows = [zero] + [_episode(profile, arm, t, scoring_windows) for t in ticks[1:]]
                out[f"scan_{arm}"] = {
                    "rows": rows,
                    "exp021": first_failure(rows, "unstable"),
                    "failed_to_fly": first_failure(rows, "failed_to_fly"),
                    "limit_s": _limit(rows),
                    "scan_top_s": rows[-1]["delay_s"],
                }
    gains = sorted(set(built))
    if len(gains) != 1 or gains[0] != scaled_gains(scale):
        raise RuntimeError(f"expected one gain setting {scaled_gains(scale)}, got {gains}")
    out["effective_K_p_att"], out["effective_K_d_att"] = map(list, gains[0])
    out["controllers_built"] = len(built)
    return out


def control_dcrit(scoring_windows) -> dict[str, Any]:
    """Registered find_d_crit searches, run through the patch at s = 1.0."""
    config = _config()
    registered = json.loads(REGISTERED_DCRIT.read_text(encoding="utf-8"))["profiles"][PROFILE_ID]
    out = {}
    for arm, key, reg in (("b", "arm_b", registered["arm_b"]["p0"]), ("c", "slice_c", registered["slice_c"])):
        with gain_patch(1.0) as built:
            answer = ct.find_d_crit(str(PROFILE_PATH), arm=arm, resolution_s=config["d_crit"][key]["resolution_s"],
                                    bracket=tuple(config["d_crit"][key]["bracket"]),
                                    duration_s=EPISODE_DURATION_S, scoring_windows=scoring_windows)
        answer["episode_duration_s"] = EPISODE_DURATION_S
        compared = {k: v for k, v in reg.items() if k != "hand_attitude_whole_state"}
        out[key] = {
            "d_crit_s": answer["d_crit_s"],
            "matches_registered": all(answer.get(k) == v for k, v in compared.items()),
            "effective_gains": sorted(set(built)),
        }
    return out


def control_a5_slow_motor(cell: dict[str, Any]) -> bool:
    """Stock gains at tau 0.16 must fail at zero delay in arm b and slice c, as A5 recorded."""
    a5 = json.loads(REGISTERED_A5.read_text(encoding="utf-8"))
    high = a5["profiles"][PROFILE_ID]["parameter_results"][TAU_PARAM]["high"]
    return (high["arm_b_unstable_at_zero_delay"] is True and high["slice_c_unstable_at_zero_delay"] is True
            and cell["zero_delay"]["b"]["unstable"] and cell["zero_delay"]["c"]["unstable"])


def scan_matches_registered(cell: dict[str, Any]) -> bool:
    """The s = 1.0, nominal-lag arm b scan must fail first at the registered d_crit."""
    reg = json.loads(REGISTERED_DCRIT.read_text(encoding="utf-8"))["profiles"][PROFILE_ID]["arm_b"]["p0"]
    scan = cell["scan_b"]
    if scan is None:
        return False
    tick = lambda d: None if d is None else round(d / ct.CONTROL_DT_S)
    return tick(scan["exp021"]["first_fail_delay_s"]) == tick(reg["d_crit_s"])


def select(cells: dict[tuple[float, float], dict[str, Any]], scales=SCALES) -> dict[str, Any]:
    """Apply the pre-registered selection rules to finished cells."""
    stable = [s for s in scales if cells[(FINE_TAU_S, s)]["stable_at_zero_delay"]]
    s_min = max(stable) if stable else None
    margin_ok = [s for s in scales
                 if cells[(FINE_TAU_S, s)]["hand"]["attitude_pm_minus_hold_deg"] >= MARGIN_TARGET_DEG]
    s_margin = max(margin_ok) if margin_ok else None
    cost = {}
    for tau in (t for t in TAUS_S if t != FINE_TAU_S):
        base = cells[(tau, 1.0)]["scan_b"]["limit_s"]
        for name, s in (("s_min", s_min), ("s_margin", s_margin), ("s_0.5", 0.5)):
            if s is None or (tau, s) not in cells or cells[(tau, s)]["scan_b"] is None:
                continue
            lim = cells[(tau, s)]["scan_b"]["limit_s"]
            cost[f"tau={tau:g} {name}={s:g}"] = {
                "stock_limit_s": base, "scaled_limit_s": lim,
                "relative_change": None if (base is None or lim is None) else (lim - base) / base,
            }
    return {"s_min": s_min, "s_margin": s_margin, "cost_arm_b": cost}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cell_path(cells_dir: Path, tau: float, scale: float, scan: bool) -> Path:
    kind = "cell" if scan else "zero"
    return cells_dir / f"{kind}_tau{tau:.3f}_s{scale:.2f}.json"


def _cached_cell(cells_dir: Path, tau: float, scale: float, scoring_windows, scan: bool = True):
    """Run one cell, or load it from a checkpoint written by an earlier run."""
    path = _cell_path(cells_dir, tau, scale, scan)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cell = run_cell(tau, scale, scoring_windows, scan=scan)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return cell


def _limit_str(cell, arm):
    scan = cell.get(f"scan_{arm}")
    if scan is None:
        return "unstable at 0"
    lim = scan["limit_s"]
    return f"> {scan['scan_top_s']:.2f}" if lim is None else f"{lim:.2f}"


def write_table(cells, path: Path) -> list[dict[str, Any]]:
    rows = []
    for (tau, s), cell in sorted(cells.items()):
        h = cell["hand"]
        rows.append({
            "tau_s": tau, "scale": s, "K_p_att": cell["effective_K_p_att"][0], "K_d_att": cell["effective_K_d_att"][0],
            "stable_at_zero_delay": cell["stable_at_zero_delay"],
            "zero_delay_rms_error_m": cell["zero_delay"]["b"]["rms_error_m"],
            "zero_delay_max_tilt_deg": cell["zero_delay"]["b"]["whole_run_max_tilt_deg"],
            "arm_b_limit_s": _limit_str(cell, "b"),
            "arm_b_exp021_monotone": None if cell["scan_b"] is None else cell["scan_b"]["exp021"]["monotone"],
            "slice_c_limit_s": _limit_str(cell, "c"),
            "hand_attitude_pm_minus_hold_deg": round(h["attitude_pm_minus_hold_deg"], 2),
            "hand_arm_b_dcrit_sim_expected_s": round(h["xy_dcrit_sim_expected_s"], 3),
            "hand_whole_state_dcrit_sim_expected_s": round(h["whole_state_dcrit_sim_expected_s"], 3),
            "hand_agrees_in_sign": (h["attitude_pm_minus_hold_deg"] > 0) == cell["stable_at_zero_delay"],
        })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_figure(cells, fine, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl
    horizon_mpl.apply_style("light")
    pal = horizon_mpl.palette("light")
    colors = horizon_mpl.sequential_ramp(["atlasStop2", "atlasStop4", "accent", "text"], len(TAUS_S))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, arm, title in ((axes[0], "b", "position-feedback delay limit (arm b)"),
                           (axes[1], "c", "whole-state delay limit (slice c)")):
        for tau, color in zip(TAUS_S, colors):
            xs, ys = [], []
            for s in sorted(SCALES):
                scan = cells[(tau, s)].get(f"scan_{arm}")
                if scan is not None and scan["limit_s"] is not None:
                    xs.append(s); ys.append(scan["limit_s"])
            ax.plot(xs, ys, "o-", color=color, lw=1.6, ms=4, label=f"motor lag {tau:g} s")
            gone = [s for s in SCALES if cells[(tau, s)].get(f"scan_{arm}") is None]
            ax.plot(gone, [0.0] * len(gone), "x", color=color, ms=7, mew=1.6)
        ax.set_xlabel("attitude gain scale s  (K_p x s², K_d x s)")
        ax.set_ylabel("first failing delay (s)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.45, 1.05)
        ax.set_ylim(bottom=-0.02)
    axes[0].legend(fontsize=8, loc="upper left")
    xs = [s for s in FINE_SCALES]
    ok = [fine[s]["stable_at_zero_delay"] for s in xs]
    tilt = [fine[s]["zero_delay"]["b"]["whole_run_max_tilt_deg"] for s in xs]
    ax = axes[2]
    ax.plot(xs, tilt, "-", color=pal["lineStrong"], lw=1.2)
    ax.plot([x for x, k in zip(xs, ok) if k], [t for t, k in zip(tilt, ok) if k], "o", color=pal["accent"], ms=5,
            label="flies (both rules pass)")
    ax.plot([x for x, k in zip(xs, ok) if not k], [t for t, k in zip(tilt, ok) if not k], "x", color=pal["textSecondary"],
            ms=7, mew=1.6, label="fails")
    ax.set_xlabel("attitude gain scale s")
    ax.set_ylabel("peak tilt in 20 s run (deg)")
    ax.set_title("motor lag 0.16 s, zero delay", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Simulated only: ESP-FLY-class, 1 m step in x, seed 0. Limit = first delay failing the EXP-021 or "
                 "the stricter failed-to-fly rule; x on the axis = fails with no delay.", fontsize=10)
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _prepare(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to((ROOT / "sim" / "runs").resolve()):
        raise ValueError("side-run output must not go under sim/runs")
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, cells_dir


def _run_controls(output_dir: Path, cells_dir: Path, windows):
    """The registered d_crit, nominal scan and A5 slow-motor controls (shared by both modes)."""
    nominal = load_hardware_profile(PROFILE_PATH).motor_time_constant_s
    controls_path = output_dir / "controls.json"
    if controls_path.exists():
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
    else:
        controls = control_dcrit(windows)
        controls_path.write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stock_nominal = _cached_cell(cells_dir, nominal, 1.0, windows)
    stock_slow = _cached_cell(cells_dir, FINE_TAU_S, 1.0, windows)
    controls_pass = (controls["arm_b"]["matches_registered"] and controls["slice_c"]["matches_registered"]
                     and scan_matches_registered(stock_nominal) and control_a5_slow_motor(stock_slow))
    return controls, stock_nominal, stock_slow, controls_pass


def run(output_dir: Path, *, taus=TAUS_S, scales=SCALES, fine_scales=FINE_SCALES) -> dict[str, Any]:
    output_dir, cells_dir = _prepare(output_dir)
    started = time.process_time()
    windows = _config().get("scoring_windows")

    # Controls first; nothing else runs unless all of them pass.
    controls, stock_nominal, stock_slow, controls_pass = _run_controls(output_dir, cells_dir, windows)
    result: dict[str, Any] = {
        "schema": "zephyr-mesh-sens-slow-motor-gains-1", "simulated_only": True,
        "profile": PROFILE_ID, "reference": "step", "loss_p": 0.0, "seed": SEED,
        "episode_duration_s": EPISODE_DURATION_S, "base_K_p_att": BASE_KP_ATT, "base_K_d_att": BASE_KD_ATT,
        "gain_law": "K_p_att = base * s^2, K_d_att = base * s (damping ratio kept)",
        "stability_rule": STABILITY_RULE,
        "controls": {**controls, "scan_matches_registered": scan_matches_registered(stock_nominal),
                     "a5_slow_motor_reproduced": control_a5_slow_motor(stock_slow)},
        "controls_pass": controls_pass,
    }
    if controls_pass:
        cells = {(t, s): _cached_cell(cells_dir, t, s, windows) for t in taus for s in scales}
        fine = {s: (cells[(FINE_TAU_S, s)] if (FINE_TAU_S, s) in cells
                    else _cached_cell(cells_dir, FINE_TAU_S, s, windows, scan=False)) for s in fine_scales}
        result["selection"] = select(cells, scales)
        result["fine_scan_tau_0.16"] = [{"scale": s, "stable_at_zero_delay": fine[s]["stable_at_zero_delay"],
                                         "exp021_unstable": fine[s]["zero_delay"]["b"]["unstable"],
                                         "failed_to_fly": fine[s]["zero_delay"]["b"]["failed_to_fly"],
                                         "max_tilt_deg": fine[s]["zero_delay"]["b"]["whole_run_max_tilt_deg"],
                                         "hand_attitude_pm_minus_hold_deg": fine[s]["hand"]["attitude_pm_minus_hold_deg"],
                                         "hand_agrees_in_sign": ((fine[s]["hand"]["attitude_pm_minus_hold_deg"] > 0)
                                                                 == fine[s]["stable_at_zero_delay"])}
                                        for s in fine_scales]
        result["table"] = write_table(cells, output_dir / "table.csv")
        write_figure(cells, fine, output_dir / "slow_motor_gains.png")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inputs = [Path(__file__).resolve(), CONFIG_PATH, REGISTERED_DCRIT, REGISTERED_A5, PROFILE_PATH,
              ROOT / "sim" / "controller.py", ROOT / "sim" / "physics" / "control_tolerance.py",
              ROOT / "tools" / "a_dcrit_hand.py"]
    outputs = [p for p in (output_dir / "result.json", output_dir / "table.csv", output_dir / "slow_motor_gains.png")
               if p.exists()]
    manifest = {
        "command": [sys.executable, "tools/sens_slow_motor_gains.py", *sys.argv[1:]],
        "git_commit": _git_commit(), "seeds": [SEED], "workers": 1,
        "taus_s": list(taus), "scales": list(scales), "fine_scales": list(fine_scales),
        "inputs_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in inputs},
        "outputs_sha256": {p.name: _sha256(p) for p in outputs},
        "cpu_s_this_invocation": round(time.process_time() - started, 1),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def flies_at_zero(cell: dict[str, Any]) -> bool:
    """Zero-delay flight passes both rules in arm b and in slice c."""
    return not any(cell["zero_delay"][arm]["unstable"] or cell["zero_delay"][arm]["failed_to_fly"]
                   for arm in ("b", "c"))


def _scan_summary(cell: dict[str, Any], arm: str) -> dict[str, Any] | None:
    scan = cell.get(f"scan_{arm}")
    if scan is None:
        return None
    return {"limit_s": scan["limit_s"], "scan_top_s": scan["scan_top_s"],
            "exp021_monotone": scan["exp021"]["monotone"],
            "failed_to_fly_monotone": scan["failed_to_fly"]["monotone"]}


def gap_select(cells: dict[tuple[float, float], dict[str, Any]], headline_s: float,
               taus=GAP_TAUS_S, scales=SCALES) -> list[dict[str, Any]]:
    """Per lag: the largest grid scale that flies, the largest with a 10 deg hand margin, and their limits."""
    rows = []
    for tau in taus:
        flying = [s for s in scales if flies_at_zero(cells[(tau, s)])]
        margin = [s for s in scales if cells[(tau, s)]["hand"]["attitude_pm_minus_hold_deg"] >= MARGIN_TARGET_DEG]
        row: dict[str, Any] = {"tau_s": tau, "stock_flies_at_zero_delay": flies_at_zero(cells[(tau, 1.0)]),
                               "scales_flying_at_zero_delay": flying}
        for name, pick in (("s_fly", max(flying) if flying else None),
                           ("s_margin", max(margin) if margin else None)):
            entry: dict[str, Any] = {"scale": pick}
            if pick is not None:
                cell = cells[(tau, pick)]
                b, c = _scan_summary(cell, "b"), _scan_summary(cell, "c")
                lim = None if b is None else b["limit_s"]
                entry.update({
                    "K_p_att": cell["effective_K_p_att"][0], "K_d_att": cell["effective_K_d_att"][0],
                    "flies_at_zero_delay": flies_at_zero(cell),
                    "arm_b": b, "slice_c": c,
                    "arm_b_change_vs_headline": None if lim is None else (lim - headline_s) / headline_s,
                    "hand_attitude_pm_minus_hold_deg": cell["hand"]["attitude_pm_minus_hold_deg"],
                })
            row[name] = entry
        rows.append(row)
    return rows


def _fmt_limit(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "unstable at 0"
    lim = summary["limit_s"]
    return f"> {summary['scan_top_s']:.2f}" if lim is None else f"{lim:.2f}"


def write_gap_table(rows: list[dict[str, Any]], path: Path) -> None:
    out = []
    for row in rows:
        line: dict[str, Any] = {"tau_s": row["tau_s"], "stock_flies_at_zero_delay": row["stock_flies_at_zero_delay"]}
        for name in ("s_fly", "s_margin"):
            e = row[name]
            line[f"{name}"] = e["scale"]
            line[f"{name}_arm_b_limit_s"] = _fmt_limit(e.get("arm_b")) if e["scale"] is not None else ""
            line[f"{name}_slice_c_limit_s"] = _fmt_limit(e.get("slice_c")) if e["scale"] is not None else ""
            change = e.get("arm_b_change_vs_headline")
            line[f"{name}_arm_b_change_vs_headline"] = "" if change is None else round(change, 4)
            pm = e.get("hand_attitude_pm_minus_hold_deg")
            line[f"{name}_hand_pm_minus_hold_deg"] = "" if pm is None else round(pm, 2)
        out.append(line)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)


def stock_edge(fine: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """Last fine-scan lag where stock gains fly before the first one where they fail."""
    taus = sorted(fine)
    fails = [t for t in taus if not flies_at_zero(fine[t])]
    first_fail = fails[0] if fails else None
    passes = [t for t in taus if first_fail is None or t < first_fail]
    again = [t for t in taus if first_fail is not None and t > first_fail and flies_at_zero(fine[t])]
    return {"last_flying_s": passes[-1] if passes else None, "first_failing_s": first_fail,
            "flies_again_after_first_fail_s": again}


def write_gap_figure(cells, rows, fine, headline_s: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl
    horizon_mpl.apply_style("light")
    pal = horizon_mpl.palette("light")
    taus = [r["tau_s"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    edge = stock_edge(fine)

    ax = axes[0]
    for name, color, marker, label in (("s_fly", pal["accent"], "o", "largest grid scale that flies"),
                                       ("s_margin", pal["text"], "s", "largest with 10 deg hand margin")):
        ys = [r[name]["scale"] for r in rows]
        ax.plot(taus, ys, marker + "-", color=color, lw=1.6, ms=5, label=label, drawstyle="steps-mid")
    if edge["first_failing_s"] is not None:
        ax.axvline(edge["first_failing_s"], color=pal["textSecondary"], lw=1, ls="--",
                   label=f"stock gains first fail at {edge['first_failing_s']:.3f} s")
    ax.set_ylabel("attitude gain scale s  (K_p x s², K_d x s)")
    ax.set_title("recommended gain scale", fontsize=10)
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    for ax, key, title, ref in ((axes[1], "arm_b", "position-feedback delay limit (arm b)", headline_s),
                                (axes[2], "slice_c", "whole-state delay limit (slice c)", None)):
        for name, color, marker, label in (("s_fly", pal["accent"], "o", "at largest flying scale"),
                                           ("s_margin", pal["text"], "s", "at 10 deg margin scale")):
            xs, ys = [], []
            for r in rows:
                summ = r[name].get(key)
                if summ is not None and summ["limit_s"] is not None:
                    xs.append(r["tau_s"]); ys.append(summ["limit_s"])
            ax.plot(xs, ys, marker + "-", color=color, lw=1.6, ms=5, label=label)
        if ref is not None:
            ax.axhline(ref, color=pal["textSecondary"], lw=1, ls=":",
                       label=f"registered stock limit at 0.07 s lag ({ref:.2f} s)")
        ax.set_ylabel("first failing delay (s)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xlabel("motor time constant (s)")
    fig.suptitle("Simulated only: ESP-FLY-class, 1 m step in x, seed 0. Limit = first delay failing the EXP-021 or "
                 "the stricter failed-to-fly rule.", fontsize=10)
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def run_gap(output_dir: Path, *, taus=GAP_TAUS_S, scales=SCALES, fine_taus=GAP_FINE_TAUS_S) -> dict[str, Any]:
    """The motor-lag gap sweep (``--mode lag-gap``)."""
    output_dir, cells_dir = _prepare(output_dir)
    started = time.process_time()
    windows = _config().get("scoring_windows")
    controls, stock_nominal, stock_slow, controls_pass = _run_controls(output_dir, cells_dir, windows)
    headline = json.loads(REGISTERED_DCRIT.read_text(encoding="utf-8"))["profiles"][PROFILE_ID]["arm_b"]["p0"]["d_crit_s"]
    result: dict[str, Any] = {
        "schema": "zephyr-mesh-sens-slow-motor-gains-lag-gap-1", "simulated_only": True,
        "profile": PROFILE_ID, "reference": "step", "loss_p": 0.0, "seed": SEED,
        "episode_duration_s": EPISODE_DURATION_S, "base_K_p_att": BASE_KP_ATT, "base_K_d_att": BASE_KD_ATT,
        "gain_law": "K_p_att = base * s^2, K_d_att = base * s (damping ratio kept)",
        "stability_rule": STABILITY_RULE,
        "flies_rule": "zero-delay flight passes both rules in arm b and in slice c",
        "margin_rule": f"hand attitude phase margin minus half-step hold >= {MARGIN_TARGET_DEG:g} deg",
        "headline_arm_b_limit_s": headline,
        "controls": {**controls, "scan_matches_registered": scan_matches_registered(stock_nominal),
                     "a5_slow_motor_reproduced": control_a5_slow_motor(stock_slow)},
        "controls_pass": controls_pass,
    }
    if controls_pass:
        cells = {(t, s): _cached_cell(cells_dir, t, s, windows) for t in taus for s in scales}
        fine = {}
        for t in fine_taus:
            if (t, 1.0) in cells:
                fine[t] = cells[(t, 1.0)]
            elif t == FINE_TAU_S:
                fine[t] = stock_slow
            else:
                fine[t] = _cached_cell(cells_dir, t, 1.0, windows, scan=False)
        rows = gap_select(cells, headline, taus, scales)
        result["per_lag"] = rows
        result["stock_edge"] = stock_edge(fine)
        result["fine_stock_scan"] = [{
            "tau_s": t, "flies_at_zero_delay": flies_at_zero(fine[t]),
            "arm_b_exp021_unstable": fine[t]["zero_delay"]["b"]["unstable"],
            "arm_b_failed_to_fly": fine[t]["zero_delay"]["b"]["failed_to_fly"],
            "slice_c_exp021_unstable": fine[t]["zero_delay"]["c"]["unstable"],
            "slice_c_failed_to_fly": fine[t]["zero_delay"]["c"]["failed_to_fly"],
            "max_tilt_deg": fine[t]["zero_delay"]["b"]["whole_run_max_tilt_deg"],
            "hand_attitude_pm_minus_hold_deg": fine[t]["hand"]["attitude_pm_minus_hold_deg"],
        } for t in sorted(fine)]
        result["table"] = write_table(cells, output_dir / "table.csv")
        write_gap_table(rows, output_dir / "gap_table.csv")
        write_gap_figure(cells, rows, fine, headline, output_dir / "motor_lag_gap.png")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inputs = [Path(__file__).resolve(), CONFIG_PATH, REGISTERED_DCRIT, REGISTERED_A5, PROFILE_PATH,
              ROOT / "sim" / "controller.py", ROOT / "sim" / "physics" / "control_tolerance.py",
              ROOT / "tools" / "a_dcrit_hand.py"]
    outputs = [output_dir / n for n in ("result.json", "table.csv", "gap_table.csv", "motor_lag_gap.png")
               if (output_dir / n).exists()]
    manifest = {
        "command": [sys.executable, "tools/sens_slow_motor_gains.py", *sys.argv[1:]],
        "git_commit": _git_commit(), "seeds": [SEED], "workers": 1, "mode": "lag-gap",
        "taus_s": list(taus), "scales": list(scales), "fine_taus_s": list(fine_taus),
        "inputs_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in inputs},
        "outputs_sha256": {p.name: _sha256(p) for p in outputs},
        "cpu_s_this_invocation": round(time.process_time() - started, 1),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("n18", "lag-gap"), default="n18",
                        help="n18: the original scale sweep (default); lag-gap: lags 0.10 to 0.15 s")
    args = parser.parse_args(argv)
    if args.mode == "lag-gap":
        result = run_gap(args.output_dir)
        print("controls pass:", result["controls_pass"])
        if result["controls_pass"]:
            print("stock edge:", json.dumps(result["stock_edge"]))
            for row in result["per_lag"]:
                f, m = row["s_fly"], row["s_margin"]
                print(f"tau {row['tau_s']:.2f} s: stock flies {row['stock_flies_at_zero_delay']}, "
                      f"s_fly {f['scale']} (arm b {_fmt_limit(f.get('arm_b'))}, slice c {_fmt_limit(f.get('slice_c'))}), "
                      f"s_margin {m['scale']} (arm b {_fmt_limit(m.get('arm_b'))}, slice c {_fmt_limit(m.get('slice_c'))})")
        return 0 if result["controls_pass"] else 1
    result = run(args.output_dir)
    print(json.dumps(result["controls"], indent=1, sort_keys=True))
    print("controls pass:", result["controls_pass"])
    if result["controls_pass"]:
        print(json.dumps(result["selection"], indent=1, sort_keys=True))
        for row in result["table"]:
            print(f"tau {row['tau_s']:.2f} s {row['scale']:.2f}: zero-delay stable {row['stable_at_zero_delay']}, "
                  f"arm b {row['arm_b_limit_s']}, slice c {row['slice_c_limit_s']}, "
                  f"hand PM-hold {row['hand_attitude_pm_minus_hold_deg']} deg, "
                  f"hand arm b {row['hand_arm_b_dcrit_sim_expected_s']} s")
    return 0 if result["controls_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
