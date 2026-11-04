"""Why does fusing at the next 100 Hz tick look slightly better than exact time?

``tools/sens_tracker_numeric.py`` found that the causal next-tick snap gave
about 1 % lower p95 error than fusing each measurement at its exact capture
time. This tool tests four explanations with paired runs (same seed, same
measurements):

- X0 (bug): a separate, fast filter reproduces that tool's exact-time trace and
  its next-tick trace. The fast filter keeps one 2x2 covariance per axis (the
  three axes share it because the noise and starting variances are the same on
  every axis) and does the math in plain Python floats.
- X1 (seed noise): paired changes across many seeds, with a 95 % interval.
- X2 (scoring times): score on a 1 ms grid instead of the 10 ms ticks. Each
  estimate coasts (predicts forward) from its own last fusion.
- X3 (process noise): a straight line with smaller filter q, and a truth drawn
  from the filter's own model (white-noise acceleration, matched q), where the
  exact-time filter is the minimum mean-square estimator.

It does not change ``sim/tracker.py`` or the numeric tool. Everything is
synthetic and seeded; the numbers are not flight performance.

Run:  PYTHONPATH=. python tools/sens_tracker_tick_explain.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

from tools import sens_tracker_numeric as w20

GRID_TICK_S = w20.TICK_S
GRID_MS_S = 0.001
# Captures sit CAPTURE_PHASE_S past a 10 ms boundary on average, so the next
# tick is on average TICK_S - CAPTURE_PHASE_S later (5.7 ms).
MEAN_LAG_S = w20.TICK_S - w20.CAPTURE_PHASE_S
MATCHED_Q_TRUE = 0.1
MATCHED_V0 = (1.0, 0.0, 0.0)


# --- the fast filter -------------------------------------------------------

def fast_filter(fuse_t, z, *, q, r_var, x0, p0=w20.INITIAL_POSITION_VAR_M2, v0var=w20.INITIAL_VELOCITY_VAR_M2PS2):
    """Constant-velocity Kalman filter fused at the times ``fuse_t``.

    Same equations as ``sim/tracker.py`` (exact discretized Q, Joseph update),
    written per axis. Returns (fusion times, posterior positions, posterior
    velocities), each with the starting state at t = 0 as row 0.
    """
    n = len(fuse_t)
    tf = np.empty(n + 1)
    pos = np.empty((n + 1, 3))
    vel = np.empty((n + 1, 3))
    px = [float(v) for v in x0]
    vx = [0.0, 0.0, 0.0]
    tf[0], pos[0], vel[0] = 0.0, px, vx
    a, b, c = float(p0), 0.0, float(v0var)
    t = 0.0
    for k in range(n):
        tk = float(fuse_t[k])
        dt = tk - t
        if dt < 0:
            raise ValueError("fusion times must be nondecreasing")
        a, b, c = (a + 2.0 * dt * b + dt * dt * c + q * dt**3 / 3.0, b + dt * c + q * dt * dt / 2.0, c + q * dt)
        zk = z[k]
        s = a + r_var
        k1, k2 = a / s, b / s
        for i in range(3):
            px[i] += vx[i] * dt
            innov = float(zk[i]) - px[i]
            px[i] += k1 * innov
            vx[i] += k2 * innov
        a, b, c = ((1.0 - k1) ** 2 * a + k1 * k1 * r_var,
                   (1.0 - k1) * (b - k2 * a) + k1 * k2 * r_var,
                   k2 * k2 * a - 2.0 * k2 * b + c + k2 * k2 * r_var)
        t = tk
        tf[k + 1], pos[k + 1], vel[k + 1] = tk, px, vx
    return tf, pos, vel


def estimates_at(grid_t, tf, pos, vel):
    """Each grid time's estimate: the last fusion at or before it, coasted forward."""
    idx = np.searchsorted(tf, grid_t, side="right") - 1
    return pos[idx] + vel[idx] * (grid_t - tf[idx])[:, None]


def next_tick_times(times):
    return np.array([w20.snap_tick(float(t), "next") for t in times]) * GRID_TICK_S


# --- matched-model truth ---------------------------------------------------

def matched_truth(times, seed, q_true=MATCHED_Q_TRUE, v0=MATCHED_V0):
    """White-noise-acceleration truth sampled exactly at the sorted ``times`` (from t = 0)."""
    rng = np.random.default_rng([int(seed), 7])
    t_all = np.asarray(times, dtype=float)
    if np.any(np.diff(t_all) < 0) or t_all[0] < 0:
        raise ValueError("times must be sorted and nonnegative")
    dt = np.diff(np.r_[0.0, t_all])
    # Exact one-step sample of (position, velocity) noise: Cholesky factor of
    # q [[dt^3/3, dt^2/2], [dt^2/2, dt]], which is l11 = sqrt(q dt^3/3),
    # l21 = sqrt(3 q dt)/2, l22 = sqrt(q dt)/2. Zero for dt = 0.
    l11 = np.sqrt(q_true * dt**3 / 3.0)[:, None]
    l21 = (np.sqrt(3.0 * q_true * dt) / 2.0)[:, None]
    l22 = (np.sqrt(q_true * dt) / 2.0)[:, None]
    e1 = rng.standard_normal((t_all.size, 3))
    e2 = rng.standard_normal((t_all.size, 3))
    v = np.asarray(v0, dtype=float) + np.cumsum(l21 * e1 + l22 * e2, axis=0)
    v_before = np.vstack([np.asarray(v0, dtype=float), v[:-1]])
    return np.cumsum(v_before * dt[:, None] + l11 * e1, axis=0)


def capture_times(seed, duration_s):
    """Same capture times as the numeric tool (its RNG draws them first)."""
    rng = np.random.default_rng(seed)
    n = int(math.floor((duration_s - w20.CAPTURE_PHASE_S - w20.CAPTURE_JITTER_S) / w20.OBS_PERIOD_S)) + 1
    times = (np.arange(n) * w20.OBS_PERIOD_S + w20.CAPTURE_PHASE_S
             + rng.uniform(-w20.CAPTURE_JITTER_S, w20.CAPTURE_JITTER_S, n))
    times[0] = max(times[0], 0.0)
    return times


def make_case(truth_kind, seed, duration_s, noise_std=0.05):
    """Capture times, measurements, and a truth function for a scenario or the matched model."""
    if truth_kind in w20.SCENARIOS:
        motion, noise, _ = w20.SCENARIOS[truth_kind]
        times, z = w20.make_measurements(motion, noise, seed, duration_s)
        return times, z, (lambda g: w20.truth_position(motion, g)), w20.truth_position(motion, [0.0])[0]
    if truth_kind != "matched":
        raise ValueError(truth_kind)
    times = capture_times(seed, duration_s)
    grid_ms = np.arange(int(round(duration_s / GRID_MS_S)) + 1) * GRID_MS_S
    all_t = np.unique(np.concatenate([times, grid_ms, np.arange(int(round(duration_s / GRID_TICK_S)) + 1) * GRID_TICK_S]))
    path = matched_truth(all_t, seed)

    def at(g):
        idx = np.searchsorted(all_t, g)
        if not np.array_equal(all_t[idx], g):
            raise ValueError("truth asked at a time it was not sampled at")
        return path[idx]

    noise_rng = np.random.default_rng([int(seed), 11])
    z = at(times) + noise_rng.normal(0.0, noise_std, (times.size, 3))
    return times, z, at, np.zeros(3)


# --- one paired run --------------------------------------------------------

def grid(kind, duration_s):
    step = GRID_TICK_S if kind == "tick" else GRID_MS_S
    g = np.arange(int(round(duration_s / step)) + 1) * step
    return g[g >= w20.BURN_IN_S - 1e-9]


def run_case(truth_kind, seed, duration_s, *, q=w20.TRACKER_Q_PSD, variants=("exact", "next", "lag_const"),
             scorings=("tick", "ms")):
    times, z, truth_fn, x0 = make_case(truth_kind, seed, duration_s)
    fuse = {"exact": times, "next": next_tick_times(times), "lag_const": times + MEAN_LAG_S}
    out = []
    grids = {s: grid(s, duration_s) for s in scorings}
    truths = {s: truth_fn(g) for s, g in grids.items()}
    for v in variants:
        tf, pos, vel = fast_filter(fuse[v], z, q=q, r_var=0.05**2, x0=x0)
        for s in scorings:
            err = np.linalg.norm(estimates_at(grids[s], tf, pos, vel) - truths[s], axis=1)
            out.append({"truth": truth_kind, "seed": int(seed), "q": float(q), "variant": v, "scoring": s,
                        "duration_s": float(duration_s), "p95_m": float(np.percentile(err, 95)),
                        "rmse_m": float(np.sqrt(np.mean(err**2)))})
    return out


def paired(rows, *, truth, q, scoring, variant, metric, ref="exact"):
    by = {}
    for r in rows:
        if r["truth"] == truth and r["q"] == q and r["scoring"] == scoring:
            by.setdefault(r["seed"], {})[r["variant"]] = r
    rel = np.array([(g[variant][metric] - g[ref][metric]) / g[ref][metric]
                    for g in by.values() if variant in g and ref in g])
    return interval(rel)


def interval(rel):
    n = rel.size
    mean = float(np.mean(rel))
    half = float(stats.t.ppf(0.975, n - 1) * np.std(rel, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"n": int(n), "mean": mean, "lo": mean - half, "hi": mean + half, "n_negative": int(np.sum(rel < 0))}


# --- X0: does the fast filter reproduce the numeric tool? ------------------

def w20_tick_next_f64(scenario, seed, duration_s):
    """The numeric tool's fixed-tick next-snap loop, run in float64 (its own run uses float32)."""
    motion, noise_std, told_std = w20.SCENARIOS[scenario]
    times, z = w20.make_measurements(motion, noise_std, seed, duration_s)
    trk = w20.NumericTracker(w20.truth_position(motion, [0.0])[0], (0.0, 0.0, 0.0), 0.0,
                             acceleration_spectral_density=w20.TRACKER_Q_PSD, measurement_std_m=told_std)
    n_ticks = int(round(duration_s / w20.TICK_S))
    est = np.empty((n_ticks + 1, 3))
    fuse_tick = [w20.snap_tick(float(t), "next") for t in times]
    k = 0
    for qi in range(n_ticks + 1):
        if qi > 0:
            trk.advance(w20.TICK_S)
        while k < len(times) and fuse_tick[k] <= qi:
            trk.update(z[k])
            k += 1
        est[qi] = trk.x[:3]
    return est


def x0_check(scenario, seed, duration_s):
    motion, noise_std, told_std = w20.SCENARIOS[scenario]
    times, z = w20.make_measurements(motion, noise_std, seed, duration_s)
    x0 = w20.truth_position(motion, [0.0])[0]
    ticks = np.arange(int(round(duration_s / w20.TICK_S)) + 1) * w20.TICK_S
    fast_exact = estimates_at(ticks, *fast_filter(times, z, q=w20.TRACKER_Q_PSD, r_var=told_std**2, x0=x0))
    fast_next = estimates_at(ticks, *fast_filter(next_tick_times(times), z, q=w20.TRACKER_Q_PSD,
                                                 r_var=told_std**2, x0=x0))
    w20_exact = w20.run_variant(scenario, "f64_exact", seed, duration_s, keep_trace=True)["_est"]
    w20_next32 = w20.run_variant(scenario, "f32_tick_next", seed, duration_s, keep_trace=True)["_est"]
    w20_next64 = w20_tick_next_f64(scenario, seed, duration_s)
    return {
        "scenario": scenario, "seed": int(seed), "duration_s": float(duration_s),
        "fast_exact_vs_w20_f64_exact_m": float(np.max(np.abs(fast_exact - w20_exact))),
        "fast_next_vs_w20_tick_next_f64_m": float(np.max(np.abs(fast_next - w20_next64))),
        "fast_next_vs_w20_f32_tick_next_m": float(np.max(np.abs(fast_next - w20_next32))),
    }


# --- X1 from the numeric tool's own rows ------------------------------------

def w20_paired(runs_jsonl: Path):
    rows = [json.loads(line) for line in runs_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {}
    for scenario in ("straight", "turn", "jink"):
        by = {}
        for r in rows:
            if r["scenario"] == scenario:
                by.setdefault(r["seed"], {})[r["variant"]] = r
        for metric, key in (("p95", "p95_error_m"), ("rmse", "rmse_m")):
            rel = np.array([(g["f32_tick_next"][key] - g["f64_exact"][key]) / g["f64_exact"][key]
                            for g in by.values() if "f32_tick_next" in g and "f64_exact" in g])
            out[f"{scenario}/{metric}"] = interval(rel)
    return out


# --- driver ----------------------------------------------------------------

def run_all(n_seeds, duration_s, w20_runs=None, log=print):
    seeds = w20.seed_list(n_seeds)
    res = {"config": {"seeds": seeds, "duration_s": duration_s, "mean_lag_s": MEAN_LAG_S,
                      "matched_q_true": MATCHED_Q_TRUE, "filter_q": w20.TRACKER_Q_PSD}}
    res["x0"] = [x0_check(sc, seeds[0], min(duration_s, 60.0)) for sc in ("straight", "turn")]
    log("x0 done")
    if w20_runs is not None and Path(w20_runs).exists():
        res["x1_w20_rows"] = w20_paired(Path(w20_runs))
    rows = []
    for seed in seeds:
        for sc in ("straight", "turn"):
            rows += run_case(sc, seed, duration_s)
        for q in (0.01, 0.001):
            rows += run_case("straight", seed, duration_s, q=q, variants=("exact", "next"))
        rows += run_case("matched", seed, duration_s, variants=("exact", "next"))
        log(f"seed {seed} done")
    res["rows"] = rows
    summary = []
    combos = ([(sc, w20.TRACKER_Q_PSD, v) for sc in ("straight", "turn") for v in ("next", "lag_const")]
              + [("straight", q, "next") for q in (0.01, 0.001)] + [("matched", w20.TRACKER_Q_PSD, "next")])
    for truth, q, v in combos:
        for s in ("tick", "ms"):
            for metric in ("p95_m", "rmse_m"):
                ci = paired(rows, truth=truth, q=q, scoring=s, variant=v, metric=metric)
                summary.append({"truth": truth, "q": q, "variant": v, "scoring": s, "metric": metric, **ci})
    res["summary"] = summary
    return res


def write_table(res, path: Path):
    lines = ["| truth | filter q | variant vs exact | scoring | metric | mean change | 95 % interval | seeds negative |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["summary"]:
        lines.append(f"| {r['truth']} | {r['q']:g} | {r['variant']} | {r['scoring']} | {r['metric'][:-2]} | "
                     f"{r['mean'] * 100:+.3f} % | {r['lo'] * 100:+.3f} to {r['hi'] * 100:+.3f} % | "
                     f"{r['n_negative']}/{r['n']} |")
    lines += ["", "X0 (max abs difference over the tick trace, metres):", ""]
    for r in res["x0"]:
        lines.append(f"- {r['scenario']} seed {r['seed']} {r['duration_s']:g} s: fast exact vs W20 f64_exact "
                     f"{r['fast_exact_vs_w20_f64_exact_m']:.2e}; fast next vs W20 tick loop in f64 "
                     f"{r['fast_next_vs_w20_tick_next_f64_m']:.2e}; fast next vs W20 f32_tick_next "
                     f"{r['fast_next_vs_w20_f32_tick_next_m']:.2e}")
    if "x1_w20_rows" in res:
        lines += ["", "X1 from the numeric tool's own 10-seed rows (f32_tick_next vs f64_exact, tick scoring):", ""]
        for k, r in res["x1_w20_rows"].items():
            lines.append(f"- {k}: mean {r['mean'] * 100:+.3f} %, 95 % interval {r['lo'] * 100:+.3f} to "
                         f"{r['hi'] * 100:+.3f} %, negative {r['n_negative']}/{r['n']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--w20-runs", type=Path, default=None, help="runs.jsonl from sens_tracker_numeric.py")
    args = ap.parse_args(argv)
    work = args.out.with_name(args.out.name + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    res = run_all(args.seeds, args.duration_s, args.w20_runs, log=lambda m: print(m, flush=True))
    (work / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_table(res, work / "table.md")
    if args.out.exists():
        shutil.rmtree(args.out)
    os.replace(work, args.out)
    print((args.out / "table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
