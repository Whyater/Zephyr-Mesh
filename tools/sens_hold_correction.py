"""Fit the sample-and-hold term of the d_crit hand formula from simulated d_crit.

The hand formula in tools/a_dcrit_hand.py gives the delay edge of a continuous
position loop.  The simulator sends feedback packets only every ``P`` seconds
(the stream period) while the controller runs every ``c = 10 ms`` and reuses
the newest packet on every tick.  That reuse is a zero-order hold: the packet
the controller uses is d, d + c, ..., d + P - c old, so on average it is
d + (P - c) / 2 old.  Each extra millisecond of stream period should therefore
cost about half a millisecond of delay tolerance (slope 0.5).

This tool measures that directly.  For each stream period it scans every 10 ms
delay in a bracket (a full scan, not bisection, so a non-monotone verdict
shows up), takes the first unstable delay as d_crit, and fits

    d_crit_sim(P) = A - m * P,   hold(P) = d_hand - d_crit_sim(P) = m * P + b.

The stock controllers send on a hardcoded ``step_index % 2`` (every 20 ms), so
changing the module's sample-period constant alone would not change the send
rate.  This tool swaps in feedback and setpoint controllers that send every
``P / c`` ticks, at runtime only.  At P = 20 ms they match the stock
controllers exactly.

All runs are synthetic simulator runs.  Outputs go to the folder passed with
``--output-dir``; nothing under sim/runs is written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from sim.hardware import load_hardware_profile
from sim.physics import control_tolerance as ct
from sim.physics.delay_buffer import DelayLossChannel
from tools.a_dcrit_hand import estimate

CONTROL_DT_S = ct.CONTROL_DT_S
DEFAULT_PERIODS_S = tuple(round(0.01 * k, 10) for k in range(1, 11))
DEFAULT_PROFILES = ("sim/physics/profiles/esp_fly_class.json", "sim/physics/profiles/crazyflie_class.json")
DEFAULT_BRACKET_S = (0.20, 0.45)
SCHEMA = "zephyr-mesh-hold-correction-1"


def ticks_per_packet(period_s: float) -> int:
    """Number of 10 ms control ticks between packets; the period must be a whole multiple."""
    k = round(float(period_s) / CONTROL_DT_S)
    if k < 1 or abs(k * CONTROL_DT_S - float(period_s)) > 1e-9:
        raise ValueError("stream period must be a positive whole multiple of the 10 ms control tick")
    return k


def mean_hold_age_s(period_s: float, control_dt_s: float = CONTROL_DT_S) -> float:
    """Average extra age from reusing a packet sent every period_s, read every control_dt_s."""
    return (float(period_s) - float(control_dt_s)) / 2.0


class PeriodFeedbackDelayController(ct.FeedbackDelayController):
    """Arm b with a chosen stream period.  At 20 ms this is the stock controller."""

    def __init__(self, delay_s, loss_p, rng, *, period_s, initial_state=None, inner=None):
        ct._DelayControllerBase.__init__(self, inner)
        self.delay_s = float(delay_s)
        self.loss_p = float(loss_p)
        self.period_s = float(period_s)
        self.send_every = ticks_per_packet(period_s)
        self.channel = DelayLossChannel(self.delay_s, self.loss_p, self.period_s, rng, initial_value=initial_state)
        # Optional list of (time, true x position); recording does not change the flight.
        self.trace: list[tuple[float, float]] | None = None

    def update_control(self, drone, target_position, dt, **kwargs):
        t = self._time_s
        if self.trace is not None:
            self.trace.append((t, float(drone.position[0])))
        if self._step_index % self.send_every == 0:
            self.channel.offer(t, (drone.position.copy(), drone.velocity.copy()))
        if self.delay_s == 0.0 and self.loss_p == 0.0:
            delayed_pos, delayed_vel = drone.position, drone.velocity
        else:
            delayed_pos, delayed_vel = self.channel.latest(t)
        proxy = ct._ProxyDrone(drone, (delayed_pos, delayed_vel, drone.quaternion, drone.angular_velocity))
        out = self.inner.update_control(proxy, target_position, dt, **kwargs)
        self._advance(dt)
        return out


class PeriodSetpointDelayController(ct.SetpointDelayController):
    """Arm a with a chosen stream period.  At 20 ms this is the stock controller."""

    def __init__(self, delay_s, loss_p, rng, *, period_s, initial_reference=(0.0, 0.0, 1.0), inner=None):
        ct._DelayControllerBase.__init__(self, inner)
        self.delay_s = float(delay_s)
        self.loss_p = float(loss_p)
        self.period_s = float(period_s)
        self.send_every = ticks_per_packet(period_s)
        self.channel = DelayLossChannel(self.delay_s, self.loss_p, self.period_s, rng,
                                        initial_value=np.asarray(initial_reference, dtype=float))

    def update_control(self, drone, target_position, dt, **kwargs):
        t = self._time_s
        if self._step_index % self.send_every == 0:
            self.channel.offer(t, np.asarray(target_position, dtype=float).copy())
        delayed = (np.asarray(target_position, dtype=float).copy()
                   if self.delay_s == 0.0 and self.loss_p == 0.0
                   else np.asarray(self.channel.latest(t), dtype=float))
        out = self.inner.update_control(drone, delayed, dt, **kwargs)
        self._advance(dt)
        return out


@contextlib.contextmanager
def stream_period(period_s: float, made: list | None = None):
    """Inside this block, run_episode builds arm a and b controllers with the given stream period.

    Controllers built in the block are appended to ``made`` when it is given.
    """
    ticks_per_packet(period_s)
    original = ct._controller_for

    def _patched(arm, delay_s, loss_p, rng, drone):
        hover = np.array([0.0, 0.0, 1.0], dtype=float)
        if arm == "a":
            ctrl = PeriodSetpointDelayController(delay_s, loss_p, rng, period_s=period_s, initial_reference=hover)
        elif arm == "b":
            state = (drone.position.copy(), drone.velocity.copy())
            ctrl = PeriodFeedbackDelayController(delay_s, loss_p, rng, period_s=period_s, initial_state=state)
            if made is not None:
                ctrl.trace = []
        else:
            ctrl = original(arm, delay_s, loss_p, rng, drone)
        if made is not None:
            made.append(ctrl)
        return ctrl

    ct._controller_for = _patched
    try:
        yield
    finally:
        ct._controller_for = original


def run_at_period(profile, arm: str, delay_s: float, period_s: float, *, duration_s: float = 20.0,
                  seed: int = 0) -> dict[str, Any]:
    with stream_period(period_s):
        return ct.run_episode(profile, arm, "step", delay_s, 0.0, seed, duration_s=duration_s)


SMALL_STEP_M = 0.01      # small enough to stay in the linear range (as the 80 s long check does)
GROWTH_DURATION_S = 80.0
GROWTH_WINDOW_S = 10.0
GROWTH_WINDOWS = (4, 8)  # 1-based windows used for the fit, after the step transient


def window_growth_rate_per_s(errors_m, *, window_s: float = GROWTH_WINDOW_S, windows=GROWTH_WINDOWS) -> float:
    """Growth rate from the peak error in each 10 s window: slope of log(peak) vs time.

    Positive means the wobble grows, negative means it dies out.  ``errors_m`` is one
    sample per 10 ms control tick from t = 0.
    """
    e = np.asarray(errors_m, dtype=float)
    per = round(window_s / CONTROL_DT_S)
    first, last = windows
    if e.size < last * per:
        raise ValueError("run ended before the last growth window")
    peaks = [float(np.max(e[(w - 1) * per:w * per])) for w in range(first, last + 1)]
    centers = [(w - 0.5) * window_s for w in range(first, last + 1)]
    slope, _ = np.polyfit(centers, np.log(peaks), 1)
    return float(slope)


def small_step_errors(profile, delay_s: float, period_s: float, *, duration_s: float = GROWTH_DURATION_S,
                      step_m: float = SMALL_STEP_M) -> np.ndarray:
    """Fly arm b with a small x step and return the 3-D error every tick (the long check's loop)."""
    from sim.controller import GeometricFlightController
    from sim.vehicle import ActuatedQuadrotor

    p = ct._profile_obj(profile)
    vehicle = ActuatedQuadrotor(p, initial_position=(0.0, 0.0, 1.0), controller=GeometricFlightController(), substeps=1)
    vehicle.actuator.speed_rad_s[:] = p.hover_speed_rad_s()
    drone = vehicle.drone
    vehicle.controller = PeriodFeedbackDelayController(delay_s, 0.0, np.random.default_rng(0), period_s=period_s,
                                                       initial_state=(drone.position.copy(), drone.velocity.copy()))
    errors = []
    for k in range(round(duration_s / CONTROL_DT_S)):
        t = k * CONTROL_DT_S
        goal = np.array((step_m if t >= 1.0 else 0.0, 0.0, 1.0))
        vehicle.step(tuple(goal), CONTROL_DT_S)
        err = float(np.linalg.norm(vehicle.drone.position - goal))
        if not np.isfinite(err):
            break
        errors.append(err)
    return np.asarray(errors)


def growth_rate_at(profile, delay_s: float, period_s: float, *, duration_s: float = GROWTH_DURATION_S) -> float:
    return window_growth_rate_per_s(small_step_errors(profile, delay_s, period_s, duration_s=duration_s))


def zero_crossing_edge(delays_s, rates_per_s) -> dict[str, Any]:
    """Delay where the growth rate crosses zero.

    ``edge_s`` interpolates linearly between the two neighbouring grid delays where the
    sign changes (the long-check reading).  ``edge_line_fit_s`` fits one line through all points.
    """
    d = np.asarray(list(delays_s), dtype=float)
    r = np.asarray(list(rates_per_s), dtype=float)
    order = np.argsort(d)
    d, r = d[order], r[order]
    crossings = [i for i in range(len(d) - 1) if r[i] < 0.0 <= r[i + 1]]
    edge = None
    if crossings:
        i = crossings[0]
        edge = float(d[i] + (d[i + 1] - d[i]) * (-r[i]) / (r[i + 1] - r[i]))
    slope, intercept = np.polyfit(d, r, 1)
    return {"edge_s": edge, "edge_line_fit_s": float(-intercept / slope) if slope > 0 else None,
            "n_sign_changes": int(np.sum(np.diff(np.sign(r)) != 0))}


def run_growth_edges(output_dir, hold_scan: dict[str, Any], *, profile=DEFAULT_PROFILES[0],
                     duration_s: float = GROWTH_DURATION_S, offsets_ticks=(-5, -4, -3, -2, -1), tag="growth") -> dict[str, Any]:
    """Small-step growth-rate edge per period, on grid delays just below the 20 s first-unstable delay."""
    out = Path(output_dir)
    ckpt = out / f"runs_{tag}.jsonl"
    done = _load_checkpoint(ckpt)
    loaded = load_hardware_profile(profile)
    pid = loaded.profile_id
    d_hand = float(estimate(tau=loaded.motor_time_constant_s)["xy"]["d_crit_s"])
    rows = []
    with ckpt.open("a", encoding="utf-8") as handle:
        for scan_row in hold_scan["profiles"][pid]["per_period"]:
            period = scan_row["period_s"]
            base = round(scan_row["d_crit_s"] / CONTROL_DT_S)
            delays = [round((base + k) * CONTROL_DT_S, 10) for k in offsets_ticks]
            rates = []
            for delay in delays:
                key = _key(pid, "b-growth", period, delay, duration_s)
                if key not in done:
                    done[key] = {"key": key, "period_s": period, "delay_s": delay,
                                 "growth_rate_per_s": growth_rate_at(profile, delay, period, duration_s=duration_s)}
                    handle.write(json.dumps(done[key]) + "\n")
                    handle.flush()
                rates.append(done[key]["growth_rate_per_s"])
            rows.append({"period_s": period, "delays_s": delays, "growth_rates_per_s": rates,
                         **zero_crossing_edge(delays, rates)})
    usable = [r for r in rows if r["edge_s"] is not None]
    ps = [r["period_s"] for r in usable]
    es = [r["edge_s"] for r in usable]
    result = {"schema": SCHEMA, "profile": pid, "duration_s": duration_s, "d_hand_continuous_s": d_hand,
              "synthetic": True, "per_period": rows, "fit": fit_hold(ps, es, d_hand),
              "formula_residuals": formula_residuals(ps, es, d_hand)}
    (out / f"hold_{tag}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def scan_delays(bracket_s=DEFAULT_BRACKET_S) -> list[float]:
    lo = round(bracket_s[0] / CONTROL_DT_S)
    hi = round(bracket_s[1] / CONTROL_DT_S)
    return [round(k * CONTROL_DT_S, 10) for k in range(lo, hi + 1)]


def summarize_scan(verdicts: dict[float, bool]) -> dict[str, Any]:
    """First unstable delay, last stable delay, and whether the verdicts are monotone."""
    delays = sorted(verdicts)
    unstable = [d for d in delays if verdicts[d]]
    if not unstable:
        return {"d_crit_s": None, "last_stable_s": delays[-1] if delays else None, "monotone": True,
                "stable_above_first_unstable_s": []}
    first = unstable[0]
    islands = [d for d in delays if d > first and not verdicts[d]]
    below = [d for d in delays if d < first]
    return {"d_crit_s": first, "last_stable_s": below[-1] if below else None, "monotone": not islands,
            "stable_above_first_unstable_s": islands}


def fit_hold(periods_s: Iterable[float], d_crit_s: Iterable[float], d_hand_s: float) -> dict[str, float]:
    """Least-squares line d_crit = A - m P, reported as hold(P) = m P + b with b = d_hand - A."""
    p = np.asarray(list(periods_s), dtype=float)
    d = np.asarray(list(d_crit_s), dtype=float)
    if p.size < 3 or p.size != d.size:
        raise ValueError("need at least three (period, d_crit) pairs")
    slope, intercept = np.polyfit(p, d, 1)
    resid = d - (intercept + slope * p)
    dof = p.size - 2
    s2 = float(np.sum(resid ** 2) / dof) if dof > 0 else float("nan")
    se_slope = math.sqrt(s2 / float(np.sum((p - p.mean()) ** 2)))
    # Rounding to the 10 ms grid alone (uniform, width 10 ms) would give this slope error.
    se_slope_grid = (CONTROL_DT_S / math.sqrt(12.0)) / math.sqrt(float(np.sum((p - p.mean()) ** 2)))
    return {
        "hold_slope": float(-slope),
        "hold_slope_se": float(se_slope),
        "hold_slope_se_from_grid_only": float(se_slope_grid),
        "hold_intercept_s": float(d_hand_s - intercept),
        "d_crit_at_zero_period_s": float(intercept),
        "max_abs_residual_s": float(np.max(np.abs(resid))),
    }


def formula_residuals(periods_s, d_crit_s, d_hand_s) -> dict[str, dict[str, float]]:
    """How far each candidate hand formula sits from the sim d_crit (sim minus prediction)."""
    p = np.asarray(list(periods_s), dtype=float)
    d = np.asarray(list(d_crit_s), dtype=float)
    candidates = {
        "main_P_over_2_plus_c_over_2": d_hand_s - (p / 2.0 + CONTROL_DT_S / 2.0),
        "dc_fix_P_minus_c_over_2": d_hand_s - (p - CONTROL_DT_S) / 2.0,
        "no_hold": np.full_like(p, d_hand_s),
    }
    out = {}
    for name, pred in candidates.items():
        diff = d - pred
        out[name] = {
            "mean_s": float(np.mean(diff)),
            "max_abs_s": float(np.max(np.abs(diff))),
            "max_abs_percent": float(np.max(np.abs(100.0 * diff / pred))),
            "spread_s": float(np.max(diff) - np.min(diff)),
        }
    return out


def _key(profile_id: str, arm: str, period_s: float, delay_s: float, duration_s: float) -> str:
    return f"{profile_id}|{arm}|{period_s:.3f}|{delay_s:.3f}|{duration_s:g}"


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    return done


def run_sweep(output_dir, *, profiles=DEFAULT_PROFILES, periods_s=DEFAULT_PERIODS_S, duration_s=20.0,
              bracket_s=DEFAULT_BRACKET_S, arm_a_delays_s=(0.45, 0.50), tag="20s") -> dict[str, Any]:
    """Scan, checkpointing every run to runs_<tag>.jsonl, then fit and write hold_<tag>.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"runs_{tag}.jsonl"
    done = _load_checkpoint(ckpt)
    delays = scan_delays(bracket_s)
    result: dict[str, Any] = {"schema": SCHEMA, "duration_s": duration_s, "bracket_s": list(bracket_s),
                              "periods_s": list(periods_s), "control_dt_s": CONTROL_DT_S,
                              "synthetic": True, "profiles": {}}
    with ckpt.open("a", encoding="utf-8") as handle:
        def _get(profile, pid, arm, period, delay):
            key = _key(pid, arm, period, delay, duration_s)
            if key not in done:
                r = run_at_period(profile, arm, delay, period, duration_s=duration_s)
                row = {"key": key, "profile": pid, "arm": arm, "period_s": period, "delay_s": delay,
                       "duration_s": duration_s, "unstable": bool(r["unstable"]),
                       "unstable_reason": r["unstable_reason"], "unstable_time_s": r["unstable_time_s"],
                       "whole_run_max_error_m": r["whole_run_max_error_m"]}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done[key] = row
            return done[key]

        for profile in profiles:
            loaded = load_hardware_profile(profile)
            pid = loaded.profile_id
            d_hand = float(estimate(tau=loaded.motor_time_constant_s)["xy"]["d_crit_s"])
            per_period = []
            for period in periods_s:
                verdicts = {d: _get(profile, pid, "b", period, d)["unstable"] for d in delays}
                summary = summarize_scan(verdicts)
                arm_a = {f"{d:.2f}": _get(profile, pid, "a", period, d)["unstable"] for d in arm_a_delays_s}
                per_period.append({"period_s": period, **summary,
                                   "unstable_delays_s": [d for d in delays if verdicts[d]],
                                   "arm_a_unstable_at": arm_a})
            usable = [row for row in per_period if row["d_crit_s"] is not None]
            entry = {"d_hand_continuous_s": d_hand, "per_period": per_period}
            if len(usable) >= 3:
                ps = [row["period_s"] for row in usable]
                ds = [row["d_crit_s"] for row in usable]
                entry["fit"] = fit_hold(ps, ds, d_hand)
                entry["formula_residuals"] = formula_residuals(ps, ds, d_hand)
            result["profiles"][pid] = entry
    (out / f"hold_{tag}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def control_check(profile=DEFAULT_PROFILES[0]) -> dict[str, Any]:
    """Stock find_d_crit, plus patched-vs-stock equality at the stock 20 ms period."""
    stock = ct.find_d_crit(profile, arm="b", resolution_s=CONTROL_DT_S, bracket=(0.0, 0.5), duration_s=20.0)
    equal = {}
    for delay in (0.30, 0.36):
        a = ct.run_episode(profile, "b", "step", delay, 0.0, 0, duration_s=20.0)
        b = run_at_period(profile, "b", delay, 0.02, duration_s=20.0)
        equal[f"{delay:.2f}"] = a == b
    return {"find_d_crit_s": stock["d_crit_s"], "patched_equals_stock_at_20ms": equal}


def _series(results, profile_id):
    """(label, periods_s, edges_s, fit) for each result file, one profile."""
    out = []
    for res in results:
        if "profiles" in res:
            entry = res["profiles"].get(profile_id)
            if entry is None:
                continue
            rows = [r for r in entry["per_period"] if r["d_crit_s"] is not None]
            label = f"first unstable delay, 1 m step, {res['duration_s']:g} s runs"
            out.append((label, [r["period_s"] for r in rows], [r["d_crit_s"] for r in rows], entry.get("fit"),
                        entry["d_hand_continuous_s"]))
        elif res.get("profile") == profile_id:
            rows = [r for r in res["per_period"] if r["edge_s"] is not None]
            out.append(("growth-rate zero crossing, 1 cm step, 80 s runs", [r["period_s"] for r in rows],
                        [r["edge_s"] for r in rows], res.get("fit"), res["d_hand_continuous_s"]))
    return out


def write_figure(output_dir, results: list[dict[str, Any]], *, profile_id="esp-fly-class",
                 path_name="hold_correction.png") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools import horizon_mpl

    pal = horizon_mpl.apply_style("light")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = [("o", pal["keepOut"], -1.2), ("s", pal["target"], 1.2), ("D", pal["accent"], 0.0)]
    d_hand = None
    for i, (label, p, d, fit, d_hand) in enumerate(_series(results, profile_id)):
        marker, color, jitter = styles[i % len(styles)]
        ax.plot(np.asarray(p) * 1000 + jitter, np.asarray(d) * 1000, linestyle="none", marker=marker,
                color=color, markersize=4.5, label=label)
    if d_hand is not None:
        xs = np.array([0.0, 105.0])
        ax.plot(xs, (d_hand - (xs / 1000 - CONTROL_DT_S) / 2) * 1000, color=pal["text"], linewidth=1.0,
                label="hand edge, hold (P - 10 ms)/2")
        ax.plot(xs, (d_hand - (xs / 1000 + CONTROL_DT_S) / 2) * 1000, color=pal["lineStrong"], linestyle=":",
                linewidth=1.2, label="hand edge, hold P/2 + 5 ms (current file)")
    ax.set_xlabel("feedback send period P (ms)")
    ax.set_ylabel("delay limit (ms)")
    ax.set_title(f"Delay limit vs feedback send period ({profile_id}, synthetic sim, arm b)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    path = Path(output_dir) / path_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--profile", action="append", help="profile json path (repeatable)")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--control", action="store_true", help="run the stock-vs-patched control check only")
    parser.add_argument("--figure-from", action="append", help="hold_<tag>.json files to plot")
    parser.add_argument("--growth-from", help="hold_<tag>.json scan to center the growth-rate runs on")
    args = parser.parse_args(argv)
    if args.control:
        print(json.dumps(control_check(), indent=2))
        return
    if args.growth_from:
        scan = json.loads(Path(args.growth_from).read_text(encoding="utf-8"))
        profile = (args.profile or [DEFAULT_PROFILES[0]])[0]
        res = run_growth_edges(args.output_dir, scan, profile=profile, duration_s=args.duration)
        print(json.dumps(res["fit"], indent=1))
        return
    if args.figure_from:
        results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.figure_from]
        print(write_figure(args.output_dir, results))
        return
    tag = args.tag or f"{args.duration:g}s"
    res = run_sweep(args.output_dir, profiles=tuple(args.profile or DEFAULT_PROFILES), duration_s=args.duration, tag=tag)
    for pid, entry in res["profiles"].items():
        print(pid, json.dumps(entry.get("fit"), indent=1))


if __name__ == "__main__":
    main()
