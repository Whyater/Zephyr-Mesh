"""Numeric readiness check for the constant-velocity tracker: float32 and a fixed tick.

A tracker on a small board (single-precision floats in hardware, a fixed
control tick) does the same arithmetic as ``sim.tracker.ConstantVelocityTracker``
with fewer digits and coarser time. This tool asks whether that changes the
answers. It does not edit ``sim/tracker.py``: ``NumericTracker`` below is a
copy of the same update equations with a ``dtype`` parameter, and a test
checks that the float64 copy matches ``sim/tracker.py`` bit for bit.

Variants (all fed the same measurements for a given scenario and seed):

- ``f64_exact``: float64, exact capture times. The reference, same as the sim.
- ``f32_exact``: float32 everywhere, including the clock in seconds.
- ``f64_tick``: float64 on a fixed tick; each measurement is fused at the
  nearest tick.
- ``f32_tick``: float32 on a fixed tick with an integer tick counter and a
  fixed dt; nearest tick.
- ``f32_tick_next``: like ``f32_tick`` but each measurement is fused at the
  first tick at or after its capture time (causal, adds about half a tick of
  lag on average).
- ``f32_simple``: float32, exact times, the short covariance update
  P = (I - K H) P and no symmetrize step. A contrast, not a proposal.

Everything is synthetic (seeded Gaussian noise on an analytic truth path). No
hardware is involved and the numbers are not flight performance.

Run:  PYTHONPATH=. python tools/sens_tracker_numeric.py --out <dir> [--pilot]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_SEED = 20261103
TICK_S = 0.01
OBS_PERIOD_S = 0.10
CAPTURE_PHASE_S = 0.0043
CAPTURE_JITTER_S = 0.003
BURN_IN_S = 2.0
TARGET_SPEED_MPS = 1.0
TURN_RADIUS_M = 1.0
JINK_HALF_PERIOD_S = 1.5
TRACKER_Q_PSD = 0.1
INITIAL_POSITION_VAR_M2 = 1.0
INITIAL_VELOCITY_VAR_M2PS2 = 1.0

SCENARIOS = {
    # name: (motion, noise std in metres, filter's assumed std in metres)
    "straight": ("straight", 0.05, 0.05),
    "turn": ("turn", 0.05, 0.05),
    "jink": ("jink", 0.05, 0.05),
    "stress_straight": ("straight", 0.001, 0.001),
}
VARIANTS = ("f64_exact", "f32_exact", "f64_tick", "f32_tick", "f32_tick_next", "f32_simple")
REFERENCE = "f64_exact"

# Pass lines fixed before the first run.
F32_REL_P95 = 0.01
F32_REL_NIS = 0.01
F32_MAX_POS_DIFF_M = 0.001
TICK_REL_P95 = 0.05


class NumericTracker:
    """The ``sim/tracker.py`` filter with a chosen float type and covariance form.

    ``form="joseph"`` with ``symmetrize=True`` is exactly the sim's update.
    ``form="simple"`` uses P = (I - K H) P. ``symmetrize=False`` skips the
    (P + P^T) / 2 step after predict and after update.
    """

    def __init__(self, position_m, velocity_mps, timestamp_s=0.0, *, position_variance_m2=1.0,
                 velocity_variance_m2ps2=1.0, acceleration_spectral_density=0.1, measurement_std_m=0.05,
                 dtype=np.float64, form="joseph", symmetrize=True):
        if form not in ("joseph", "simple"):
            raise ValueError("form must be joseph or simple")
        self.dtype = np.dtype(dtype).type
        d = self.dtype
        self.form = form
        self.symmetrize = bool(symmetrize)
        self.x = np.r_[np.asarray(position_m, dtype=d), np.asarray(velocity_mps, dtype=d)]
        self.timestamp_s = d(timestamp_s)
        self.q_acc = d(acceleration_spectral_density)
        self.measurement_variance = d(measurement_std_m) ** 2
        self.P = np.diag(np.array([position_variance_m2] * 3 + [velocity_variance_m2ps2] * 3, dtype=d))
        self.I3 = np.eye(3, dtype=d)
        self.I6 = np.eye(6, dtype=d)
        self.H = np.c_[np.eye(3, dtype=d), np.zeros((3, 3), dtype=d)]
        self.R = np.eye(3, dtype=d) * self.measurement_variance
        self.update_count = 0
        self.max_asymmetry = 0.0
        self.last_nis = None

    def _transition(self, dt):
        d = self.dtype
        I3 = self.I3
        F = np.eye(6, dtype=d)
        F[:3, 3:] = I3 * dt
        q = self.q_acc
        Q = np.zeros((6, 6), dtype=d)
        Q[:3, :3] = I3 * (dt**3 / d(3.0)) * q
        Q[:3, 3:] = I3 * (dt**2 / d(2.0)) * q
        Q[3:, :3] = Q[:3, 3:].T
        Q[3:, 3:] = I3 * dt * q
        return F, Q

    def _note_asymmetry(self, P):
        scale = float(np.max(np.abs(P)))
        if scale > 0:
            self.max_asymmetry = max(self.max_asymmetry, float(np.max(np.abs(P - P.T))) / scale)

    def _finish(self, P):
        self._note_asymmetry(P)
        if self.symmetrize:
            P = (P + P.T) / self.dtype(2.0)
        return P

    def advance(self, dt):
        """Predict forward by ``dt`` (already in this tracker's float type)."""
        F, Q = self._transition(dt)
        self.x = F @ self.x
        self.P = self._finish(F @ self.P @ F.T + Q)

    def predict(self, timestamp_s):
        t = self.dtype(timestamp_s)
        if t < self.timestamp_s:
            raise ValueError("prediction timestamp must be nondecreasing")
        self.advance(t - self.timestamp_s)
        self.timestamp_s = t

    def predict_at(self, timestamp_s):
        """Predicted (position, covariance) at ``timestamp_s``; the filter is unchanged."""
        t = self.dtype(timestamp_s)
        F, Q = self._transition(t - self.timestamp_s)
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        if self.symmetrize:
            P = (P + P.T) / self.dtype(2.0)
        return x, P

    def update(self, measurement_m, timestamp_s=None):
        """Fuse one position measurement.

        With ``timestamp_s`` the filter first predicts to it, as the sim does.
        Without it (fixed-tick mode) the filter is already at the right tick.
        """
        z = np.asarray(measurement_m, dtype=self.dtype)
        if timestamp_s is not None:
            self.predict(timestamp_s)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.last_nis = float(innovation @ np.linalg.solve(S, innovation))
        self.x = self.x + K @ innovation
        if self.form == "joseph":
            A = self.I6 - K @ self.H
            P = A @ self.P @ A.T + K @ self.R @ K.T
        else:
            P = (self.I6 - K @ self.H) @ self.P
        self.P = self._finish(P)
        self.update_count += 1


# --- truth, measurements, tick snapping ---------------------------------

def truth_position(motion: str, times_s) -> np.ndarray:
    """True position, shape (N, 3), in float64."""
    t = np.asarray(times_s, dtype=float)
    v = TARGET_SPEED_MPS
    out = np.zeros((t.size, 3))
    if motion == "straight":
        out[:, 0] = v * t
    elif motion == "turn":
        r = TURN_RADIUS_M
        w = v / r
        out[:, 0] = r * np.sin(w * t)
        out[:, 1] = r * (1.0 - np.cos(w * t))
    elif motion == "jink":
        out[:, 0] = v * t
        period = 2.0 * JINK_HALF_PERIOD_S
        phase = np.mod(t, period)
        out[:, 1] = v * np.where(phase < JINK_HALF_PERIOD_S, phase, period - phase)
    else:
        raise ValueError(f"unknown motion {motion}")
    return out


def seed_list(n_seeds: int, base_seed: int = BASE_SEED) -> list[int]:
    return [int(v) for v in np.random.SeedSequence(base_seed).generate_state(int(n_seeds))]


def make_measurements(motion: str, noise_std_m: float, seed: int, duration_s: float):
    """Irregular capture times and noisy positions (float64), the same for every variant."""
    rng = np.random.default_rng(seed)
    n = int(math.floor((duration_s - CAPTURE_PHASE_S - CAPTURE_JITTER_S) / OBS_PERIOD_S)) + 1
    times = np.arange(n) * OBS_PERIOD_S + CAPTURE_PHASE_S + rng.uniform(-CAPTURE_JITTER_S, CAPTURE_JITTER_S, n)
    times[0] = max(times[0], 0.0)
    z = truth_position(motion, times) + rng.normal(0.0, noise_std_m, (n, 3))
    return times, z


def snap_tick(t_s: float, mode: str) -> int:
    """Tick index a measurement captured at ``t_s`` is fused on."""
    x = t_s / TICK_S
    if mode == "nearest":
        return int(math.floor(x + 0.5))
    if mode == "next":
        return int(math.ceil(x - 1e-9))
    raise ValueError("mode must be nearest or next")


# --- one run --------------------------------------------------------------

@dataclass(frozen=True)
class VariantSpec:
    dtype: type
    clock: str  # "exact" or "tick"
    snap: str | None
    form: str
    symmetrize: bool


SPECS = {
    "f64_exact": VariantSpec(np.float64, "exact", None, "joseph", True),
    "f32_exact": VariantSpec(np.float32, "exact", None, "joseph", True),
    "f64_tick": VariantSpec(np.float64, "tick", "nearest", "joseph", True),
    "f32_tick": VariantSpec(np.float32, "tick", "nearest", "joseph", True),
    "f32_tick_next": VariantSpec(np.float32, "tick", "next", "joseph", True),
    "f32_simple": VariantSpec(np.float32, "exact", None, "simple", False),
}


def _cov_health(P, state):
    P64 = np.asarray(P, dtype=np.float64)
    eig = float(np.min(np.linalg.eigvalsh((P64 + P64.T) / 2.0)))
    state["min_eig"] = min(state["min_eig"], eig)
    try:
        np.linalg.cholesky(P64)
    except np.linalg.LinAlgError:
        state["chol_fail"] += 1
    if not np.all(np.isfinite(P64)):
        state["nonfinite"] = True


def run_variant(scenario: str, variant: str, seed: int, duration_s: float, *, keep_trace: bool = False) -> dict:
    motion, noise_std, told_std = SCENARIOS[scenario]
    spec = SPECS[variant]
    times, z = make_measurements(motion, noise_std, seed, duration_s)
    x0 = truth_position(motion, [0.0])[0]
    trk = NumericTracker(x0, (0.0, 0.0, 0.0), 0.0, position_variance_m2=INITIAL_POSITION_VAR_M2,
                         velocity_variance_m2ps2=INITIAL_VELOCITY_VAR_M2PS2,
                         acceleration_spectral_density=TRACKER_Q_PSD, measurement_std_m=told_std,
                         dtype=spec.dtype, form=spec.form, symmetrize=spec.symmetrize)
    n_ticks = int(round(duration_s / TICK_S))
    tick_times = np.arange(n_ticks + 1) * TICK_S
    truth = truth_position(motion, tick_times)
    est = np.empty((n_ticks + 1, 3))
    health = {"min_eig": math.inf, "chol_fail": 0, "nonfinite": False}
    nis = []
    k = 0
    if spec.clock == "exact":
        for q in range(n_ticks + 1):
            t_q = tick_times[q]
            while k < len(times) and times[k] <= t_q:
                trk.predict(times[k])
                _cov_health(trk.P, health)
                trk.update(z[k], times[k])
                _cov_health(trk.P, health)
                if times[k] >= BURN_IN_S:
                    nis.append(trk.last_nis)
                k += 1
            x, _ = trk.predict_at(t_q)
            est[q] = x[:3]
    else:
        dt = trk.dtype(TICK_S)
        fuse_tick = [snap_tick(float(t), spec.snap) for t in times]
        for q in range(n_ticks + 1):
            if q > 0:
                trk.advance(dt)
            while k < len(times) and fuse_tick[k] <= q:
                _cov_health(trk.P, health)
                trk.update(z[k])
                _cov_health(trk.P, health)
                if times[k] >= BURN_IN_S:
                    nis.append(trk.last_nis)
                k += 1
            est[q] = trk.x[:3]
    scored = tick_times >= BURN_IN_S - 1e-9
    err = np.linalg.norm(est[scored] - truth[scored], axis=1)
    out = {
        "scenario": scenario, "variant": variant, "seed": int(seed), "duration_s": float(duration_s),
        "p95_error_m": float(np.percentile(err, 95)), "rmse_m": float(np.sqrt(np.mean(err**2))),
        "max_error_m": float(np.max(err)), "mean_nis": float(np.mean(nis)) if nis else None,
        "updates": int(trk.update_count), "max_asymmetry": float(trk.max_asymmetry),
        "min_eigenvalue": float(health["min_eig"]), "cholesky_failures": int(health["chol_fail"]),
        "nonfinite": bool(health["nonfinite"]),
        "final_P_dtype": str(trk.P.dtype),
    }
    if keep_trace:
        out["_est"] = est
    return out


def run_seed(scenario: str, seed: int, duration_s: float) -> list[dict]:
    """All variants for one scenario and seed, with each one's position difference from the reference."""
    rows, traces = [], {}
    for variant in VARIANTS:
        row = run_variant(scenario, variant, seed, duration_s, keep_trace=True)
        traces[variant] = row.pop("_est")
        rows.append(row)
    ref = traces[REFERENCE]
    scored = np.arange(ref.shape[0]) * TICK_S >= BURN_IN_S - 1e-9
    for row in rows:
        diff = np.linalg.norm(traces[row["variant"]] - ref, axis=1)
        # Scored window (after burn-in), like the error metrics. The whole-run
        # value is kept too: it includes the start-up transient, where a snapped
        # first measurement lands one tick earlier or later than the exact one.
        row["max_pos_diff_vs_ref_m"] = float(np.max(diff[scored]))
        row["max_pos_diff_vs_ref_full_run_m"] = float(np.max(diff))
    return rows


# --- summary --------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault((r["scenario"], r["seed"]), {})[r["variant"]] = r
    table = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            rel_p95, rel_nis, diffs, asym, eig, chol, p95s, nis_vals = [], [], [], [], [], 0, [], []
            for (sc, _seed), group in sorted(by.items()):
                if sc != scenario or variant not in group or REFERENCE not in group:
                    continue
                r, ref = group[variant], group[REFERENCE]
                rel_p95.append((r["p95_error_m"] - ref["p95_error_m"]) / ref["p95_error_m"])
                if r["mean_nis"] is not None and ref["mean_nis"]:
                    rel_nis.append((r["mean_nis"] - ref["mean_nis"]) / ref["mean_nis"])
                    nis_vals.append(r["mean_nis"])
                diffs.append(r["max_pos_diff_vs_ref_m"])
                asym.append(r["max_asymmetry"])
                eig.append(r["min_eigenvalue"])
                chol += r["cholesky_failures"] + (1 if r["nonfinite"] else 0)
                p95s.append(r["p95_error_m"])
            if not p95s:
                continue
            pd_ok = chol == 0 and min(eig) > 0
            worst_rel = max(rel_p95, key=abs)
            worst_nis = max(rel_nis, key=abs) if rel_nis else None
            if variant.startswith("f32") and SPECS[variant].clock == "exact":
                rule = "rule1"
                passed = (abs(worst_rel) <= F32_REL_P95 and worst_nis is not None and abs(worst_nis) <= F32_REL_NIS
                          and max(diffs) < F32_MAX_POS_DIFF_M and pd_ok)
            elif SPECS[variant].clock == "tick":
                rule = "rule2"
                passed = abs(worst_rel) <= TICK_REL_P95 and pd_ok
            else:
                rule, passed = "reference", None
            table.append({
                "scenario": scenario, "variant": variant, "n_seeds": len(p95s),
                "median_p95_m": float(np.median(p95s)), "worst_rel_p95": float(worst_rel),
                "median_rel_p95": float(np.median(rel_p95)),
                "worst_rel_nis": None if worst_nis is None else float(worst_nis),
                "median_mean_nis": float(np.median(nis_vals)) if nis_vals else None,
                "max_pos_diff_m": float(max(diffs)), "max_asymmetry": float(max(asym)),
                "min_eigenvalue": float(min(eig)), "cholesky_failures": int(chol),
                "positive_definite": bool(pd_ok), "rule": rule, "passed": passed,
            })
    return {"rows": table}


def write_table(summary: dict, path: Path) -> None:
    lines = ["| scenario | variant | median p95 (cm) | worst p95 change | max pos diff vs f64 (mm) | "
             "worst NIS change | max asymmetry | min eigenvalue | PD all run | rule | pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        nis = "n/a" if r["worst_rel_nis"] is None else f"{r['worst_rel_nis'] * 100:+.3f} %"
        passed = "" if r["passed"] is None else ("yes" if r["passed"] else "NO")
        lines.append(
            f"| {r['scenario']} | {r['variant']} | {r['median_p95_m'] * 100:.3f} | {r['worst_rel_p95'] * 100:+.3f} % | "
            f"{r['max_pos_diff_m'] * 1000:.4f} | {nis} | {r['max_asymmetry']:.1e} | {r['min_eigenvalue']:.2e} | "
            f"{'yes' if r['positive_definite'] else 'NO'} | {r['rule']} | {passed} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.e_common import apply_mission_style, series_colors

    pal = apply_mission_style("light")
    variants = [v for v in VARIANTS if v != REFERENCE]
    colors = series_colors(len(variants))
    by = {}
    for r in rows:
        by.setdefault((r["scenario"], r["seed"]), {})[r["variant"]] = r
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    scen = list(SCENARIOS)
    width = 0.8 / len(variants)
    for j, v in enumerate(variants):
        for i, sc in enumerate(scen):
            rel = [(g[v]["p95_error_m"] - g[REFERENCE]["p95_error_m"]) / g[REFERENCE]["p95_error_m"] * 100
                   for (s, _), g in by.items() if s == sc and v in g]
            asym = [g[v]["max_asymmetry"] for (s, _), g in by.items() if s == sc and v in g]
            xs = np.full(len(rel), i + (j - (len(variants) - 1) / 2) * width)
            ax1.scatter(xs, rel, s=14, color=colors[j], label=v if i == 0 else None, zorder=3)
            ax2.scatter(xs, np.maximum(asym, 1e-18), s=14, color=colors[j], label=v if i == 0 else None, zorder=3)
    ax1.axhspan(-TICK_REL_P95 * 100, TICK_REL_P95 * 100, color=pal.get("surfaceRaised", "#eeeeee"), alpha=0.5, zorder=1)
    ax1.axhline(0, color=pal.get("lineStrong", "#888888"), lw=0.8)
    ax1.set_yscale("symlog", linthresh=0.01)
    ax1.set_xticks(range(len(scen)), scen)
    ax1.set_ylabel("p95 error change vs float64 exact time (%)")
    ax1.set_title("Tracking error: float32 and fixed tick vs the sim")
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(scen)), scen)
    ax2.set_ylabel("worst covariance asymmetry |P - P^T| / max|P|")
    ax2.set_title("Covariance symmetry over 10 simulated minutes")
    ax1.legend(fontsize=7, loc="upper left")
    fig.text(0.01, 0.01, "Synthetic runs, seeded; shaded band is the pre-registered 5 % fixed-tick line.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- driver ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--pilot", action="store_true", help="1 seed, 60 s")
    args = ap.parse_args(argv)
    n_seeds, duration = (1, 60.0) if args.pilot else (args.seeds, args.duration_s)
    final = args.out
    work = final.with_name(final.name + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    ckpt = work / "runs.jsonl"
    done = set()
    rows = []
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["duration_s"] == duration:
                rows.append(r)
                done.add((r["scenario"], r["seed"]))
    seeds = seed_list(n_seeds)
    for scenario in SCENARIOS:
        for seed in seeds:
            if (scenario, seed) in done:
                continue
            t0 = time.perf_counter()
            new = run_seed(scenario, seed, duration)
            with ckpt.open("a", encoding="utf-8") as fh:
                for r in new:
                    fh.write(json.dumps(r) + "\n")
            rows.extend(new)
            print(f"{scenario} seed {seed}: {time.perf_counter() - t0:.1f} s", flush=True)
    summary = summarize(rows)
    summary["config"] = {"seeds": seeds, "duration_s": duration, "tick_s": TICK_S, "obs_period_s": OBS_PERIOD_S,
                         "capture_phase_s": CAPTURE_PHASE_S, "capture_jitter_s": CAPTURE_JITTER_S,
                         "burn_in_s": BURN_IN_S, "scenarios": SCENARIOS, "q_psd": TRACKER_Q_PSD,
                         "command": " ".join(["PYTHONPATH=.", "python", "tools/sens_tracker_numeric.py"]
                                             + [str(a) for a in (argv if argv is not None else sys.argv[1:])])}
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(summary, work / "table.md")
    make_figure(rows, work / "numeric_readiness.png")
    if final.exists():
        shutil.rmtree(final)
    os.replace(work, final)
    print((final / "table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
