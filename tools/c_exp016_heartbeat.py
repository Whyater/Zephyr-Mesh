"""EXP-016: simulated heartbeat timeout and degraded-link thresholds.

Usage: ``PYTHONPATH=. ./.venv/bin/python tools/c_exp016_heartbeat.py --output runs/c5-heartbeat``
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import math
import time
from pathlib import Path

import numpy as np
import jsonschema

from sim.estimation.sweep import run_jobs
from sim.radio.channel import GilbertElliott, packet_error_rate, mean_rssi_dbm
from sim.radio.heartbeat import (detection_delay_bounds, detection_delay_samples,
                                  false_alarm_rate_ge_per_hour, false_alarm_rate_iid_per_hour,
                                  simulate_heartbeats)
from sim.radio.interference import (DegradedLinkDetector, InterferenceConfig,
                                    evaluate_detector, simulate_link_with_interference)
from sim.radio.mac import frame_bytes
from sim.radio.params import channel_preset, default_phy
from tools.c_common import (argv_or_default, default_processes, prepare_output, seeds_from,
                            write_json, write_manifest)

BASE_SEED = 20261016
HOURS = 10.0
RATES = (5, 10, 20, 50)
TIMEOUTS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
LOSSES = (0.01, 0.05, 0.1, 0.2, 0.3)
DURATIONS = (0.05, 0.2, 0.5)


def expected_rate(condition: dict | str, rate_hz: float, timeout_s: float, *, p: float | None = None,
                  duration_s: float | None = None) -> float:
    """Return the D9 expected false alarm rate in alarms per hour."""
    if isinstance(condition, str):
        if condition == "independent":
            return false_alarm_rate_iid_per_hour(float(p), rate_hz, timeout_s)
        ge = GilbertElliott.from_loss_and_burst(float(p), max(float(duration_s) * rate_hz, 1.0))
        return false_alarm_rate_ge_per_hour(ge, rate_hz, timeout_s)
    if condition["kind"] == "independent":
        return false_alarm_rate_iid_per_hour(float(condition["p"]), rate_hz, timeout_s)
    ge = GilbertElliott.from_loss_and_burst(float(condition["p"]), max(float(condition["duration_s"] * rate_hz), 1.0))
    return false_alarm_rate_ge_per_hour(ge, rate_hz, timeout_s)


def _conditions(pilot: bool) -> list[dict]:
    ps = (0.1,) if pilot else LOSSES
    out = [{"kind": "independent", "p": p, "label": f"independent_p_{p:g}"} for p in ps]
    ds = (0.5,) if pilot else DURATIONS
    out += [{"kind": "bursty", "p": p, "duration_s": d,
             "label": f"bursty_p_{p:g}_duration_{d:g}s"} for p in ps for d in ds]
    return out


# CSV column order (run_jobs returns rows with sorted keys, so the order is fixed here).
FALSE_ALARM_COLUMNS = ("h_hz", "timeout_s", "loss_condition", "kind", "p", "duration_s", "expected_count",
                       "simulated_count", "expected_rate_per_hour", "simulated_rate_per_hour", "z")
DELAY_COLUMNS = ("h_hz", "timeout_s", "loss_condition", "mean_s", "p95_s", "max_s",
                 "bound_low_s", "bound_high_s", "bound_mean_s")
INTERFERENCE_COLUMNS = ("margin_db", "distance_m", "mean_duration_s", "noise_rise_db", "mean_trigger_s",
                        "p95_trigger_s", "fraction_detected", "false_triggers_per_hour", "missed_episodes",
                        "n_episodes", "mean_clear_after_episode_s")


def _rng(*key: int) -> np.random.Generator:
    """One independent stream per (purpose, cell): SeedSequence([BASE_SEED, *key])."""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *[int(k) for k in key]]))


def _loss_sequence(cond: dict, n: int, rng: np.random.Generator, h: float) -> np.ndarray:
    if cond["kind"] == "independent":
        return rng.random(n) < cond["p"]
    ge = GilbertElliott.from_loss_and_burst(cond["p"], max(cond["duration_s"] * h, 1.0))
    return ge.sample(n, rng)[1]


def select_thresholds(records: list[dict], timeout_grid=TIMEOUTS) -> list[dict]:
    """Pick the first grid timeout meeting simulated and formula rate limits."""
    groups = sorted({(r["h_hz"], r["loss_condition"]) for r in records})
    out = []
    for h, loss in groups:
        rows = sorted((r for r in records if r["h_hz"] == h and r["loss_condition"] == loss),
                      key=lambda r: r["timeout_s"])
        result = {"h_hz": h, "loss_condition": loss}
        for key, limit, basis in (("false_alarm_below_1_per_hour", 1.0, "simulated_and_closed_form"),
                                  ("false_alarm_below_0_01_per_hour", 0.01, "closed_form")):
            found = next((r for r in rows if r["expected_rate_per_hour"] < limit and
                          (key == "false_alarm_below_0_01_per_hour" or r["simulated_rate_per_hour"] < limit)), None)
            result[key] = None if found is None else {"T_s": found["timeout_s"],
                "mean_detection_delay_s": found.get("mean_detection_delay_s"),
                "p95_detection_delay_s": found.get("p95_detection_delay_s"), "basis": basis}
        out.append(result)
    return out


def select_threshold(rows: list[dict], metric: str = "below_1_per_hour", threshold_per_hour: float | None = None):
    """Return the smallest timeout in a hand-made rate table meeting a limit.

    This small helper accepts rows with ``timeout_s`` and either ``simulated_rate_per_hour``
    or ``expected_rate_per_hour``.  It is also useful for checking threshold logic without
    running the experiment.
    """
    limit = float(threshold_per_hour) if threshold_per_hour is not None else (1.0 if metric in ("below_1_per_hour", "1", "one") else 0.01)
    ok = [r for r in rows if float(r.get("simulated_rate_per_hour", math.inf)) < limit and float(r.get("expected_rate_per_hour", math.inf)) < limit]
    return min(ok, key=lambda r: float(r["timeout_s"])) if ok else None


def simulate_false_alarm_cell(*, h: float, timeout_s: float, condition: str, p: float,
                              duration_hours: float, seed: int, duration_s: float = 0.5) -> dict:
    n = int(round(duration_hours * 3600.0 * h))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    lost = _loss_sequence({"kind": condition, "p": p, "duration_s": duration_s}, n, rng, h)
    result = simulate_heartbeats(lost, h, timeout_s)
    return {"simulated_count": int(result.n_episodes), "simulated_rate_per_hour": float(result.n_episodes / duration_hours)}


def _snr_at_half_loss(cfg, frame_b: int) -> float:
    """SNR (dB) where a frame of frame_b bytes on air is lost half the time (bisection)."""
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(packet_error_rate(mid, frame_b, cfg)) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _margin_distance(cfg, margin_db: float, frame_b: int) -> float:
    """Distance whose mean SNR sits margin_db above the 50 % loss point (pre-registered definition)."""
    target = _snr_at_half_loss(cfg, frame_b) + margin_db
    lo, hi = 0.1, 1.0e6
    for _ in range(100):
        d = math.sqrt(lo * hi)  # bisect in log distance
        if float(mean_rssi_dbm(d, cfg)) - cfg.noise_floor_dbm > target:
            lo = d
        else:
            hi = d
    return math.sqrt(lo * hi)


def _clear_delays(episodes, transitions) -> list[float]:
    """For episodes during which the detector was degraded at the episode end:
    time from the end of the episode to the next return to ok."""
    out = []
    for _start, end in episodes:
        state, nxt = "ok", None
        for when, new_state in transitions:
            if when <= end:
                state = new_state
            elif new_state == "ok":
                nxt = when
                break
        if state == "degraded" and nxt is not None:
            out.append(nxt - end)
    return out


def _write_figures(out: Path, rows: list[dict], rule_rows: list[dict]) -> None:
    """Closed form as lines, simulation as markers; a cell with no false alarm in
    10 simulated hours is drawn as an open triangle on the bottom edge (it is a
    zero count, not a value)."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zephyr-mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    colors = apply_style("light")
    floor, top = 1e-3, 3e4
    series = [("independent", None, colors["accent"], "independent loss"),
              ("bursty", 0.05, colors["ok"], "bursts of 0.05 s"),
              ("bursty", 0.2, colors["keepOut"], "bursts of 0.2 s"),
              ("bursty", 0.5, colors["textSecondary"], "bursts of 0.5 s")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, h in zip(axes.flat, RATES):
        for kind, dur, color, label in series:
            rr = sorted((r for r in rows if r["h_hz"] == h and r["kind"] == kind and abs(r["p"] - 0.1) < 1e-12
                         and (dur is None or abs(r["duration_s"] - dur) < 1e-12)), key=lambda r: r["timeout_s"])
            if not rr:
                continue
            t = [r["timeout_s"] for r in rr]
            ax.plot(t, [min(max(r["expected_rate_per_hour"], floor / 10), top) for r in rr], color=color, lw=1.4,
                    label=label)
            hit = [r for r in rr if r["simulated_count"] > 0]
            zero = [r for r in rr if r["simulated_count"] == 0]
            ax.plot([r["timeout_s"] for r in hit], [r["simulated_rate_per_hour"] for r in hit], ls="none", marker="o",
                    ms=4.5, color=color)
            ax.plot([r["timeout_s"] for r in zero], [floor * 1.25] * len(zero), ls="none", marker="v", ms=5,
                    mfc="none", color=color)
        ax.axhline(1, color=colors["caution"], ls="--", lw=1)
        ax.axvline(0.3, color=colors["line"], lw=1)
        ax.set_yscale("log"); ax.set_ylim(floor, top)
        ax.set_title(f"heartbeat {h} Hz, mean loss 10 %", fontsize=10)
        ax.set_xlabel("timeout T (s)"); ax.set_ylabel("false 'link lost' per hour")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Simulated (10 h per point, dots) vs closed form (lines). Open triangles: none in 10 h. "
                 "Dashed: 1 per hour. Grey line: 300 ms.", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "exp016_false_alarms.png", dpi=150, metadata={"Software": None}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    rr = [r for r in rule_rows if r["h_hz"] == 10]
    x = np.arange(len(rr))
    sim = [r["simulated_rate_per_hour"] for r in rr]
    ax.bar(x, [max(v, floor) for v in sim], color=[colors["accent"] if v > 0 else colors["surface"] for v in sim],
           edgecolor=colors["accent"])
    ax.bar([0], [0], color=colors["accent"], label="simulated, 10 h per bar")  # legend swatch only
    ax.plot(x, [max(r["expected_rate_per_hour"], floor) for r in rr], ls="none", marker="_", ms=14, mew=2,
            color=colors["text"], label="closed form (D9)")
    for xi, v in zip(x, sim):
        if v == 0:
            ax.text(xi, floor * 1.4, "0", ha="center", va="bottom", fontsize=7, color=colors["textSecondary"])
    ax.axhline(1, color=colors["caution"], ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylim(floor, top); ax.legend(frameon=False, fontsize=8, loc="upper left")

    def short(r):
        if r["kind"] == "independent":
            return f"indep. {r['p'] * 100:g} %"
        return f"bursts {r['p'] * 100:g} %, {r['duration_s']:g} s"
    ax.set_xticks(x, [short(r) for r in rr], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("false 'link lost' per hour (log)")
    ax.set_title("Planned 'lost link over 300 ms' rule with 10 Hz heartbeats (simulated; 0 = none in 10 h)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "exp016_rule_300ms.png", dpi=150, metadata={"Software": None}); plt.close(fig)


def _false_alarm_job(job: dict) -> dict:
    """One (loss condition, heartbeat rate) cell: false-alarm and detection-delay rows for every timeout."""
    ci, cond, hi, h, pilot = int(job["ci"]), job["cond"], int(job["hi"]), job["h"], bool(job["pilot"])
    false_rows, delay_rows = [], []
    n = int(round(HOURS * 3600 * h * (0.2 / 10 if pilot else 1.0)))
    seq = _loss_sequence(cond, n, _rng(1, ci, hi), h)
    for ti, T in enumerate(job["timeouts"]):
        sim = simulate_heartbeats(seq, h, T)
        exp_rate = expected_rate(cond, h, T); exp_count = exp_rate * (n / h / 3600)
        sim_count = sim.n_episodes; z = (sim_count - exp_count) / math.sqrt(max(exp_count, 1.0))
        false_rows.append({"h_hz": h, "timeout_s": T, "loss_condition": cond["label"], "kind": cond["kind"],
            "p": cond["p"], "duration_s": cond.get("duration_s"), "expected_count": exp_count,
            "simulated_count": sim_count, "expected_rate_per_hour": exp_rate,
            "simulated_rate_per_hour": sim_count / (n / h / 3600), "z": z})
        nd = 500 if pilot else 10000
        rng = _rng(2, ci, hi, ti)
        pre = None if cond["kind"] == "independent" and cond["p"] == 0 else (cond["p"] if cond["kind"] == "independent" else GilbertElliott.from_loss_and_burst(cond["p"], max(cond["duration_s"] * h, 1.0)))
        d = detection_delay_samples(h, T, nd, rng, pre_loss=pre)
        delay_rows.append({"h_hz": h, "timeout_s": T, "loss_condition": cond["label"], "mean_s": float(np.mean(d)),
            "p95_s": float(np.percentile(d, 95)), "max_s": float(np.max(d)),
            "bound_low_s": detection_delay_bounds(h, T)[0], "bound_high_s": detection_delay_bounds(h, T)[1],
            "bound_mean_s": detection_delay_bounds(h, T)[2]})
    return {"false_rows": _plain(false_rows), "delay_rows": _plain(delay_rows)}


def _plain(value):
    """numpy scalars to Python numbers, so run_jobs can pass results between processes as JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _interference_job(job: dict) -> dict:
    """One C6 interference cell: margin, mean episode duration and noise rise."""
    margin, dur, rise, pilot = job["margin"], job["dur"], job["rise"], bool(job["pilot"])
    frame_b = frame_bytes(32, default_phy()); cfg = channel_preset("indoor_gym")
    dist = _margin_distance(cfg, margin, frame_b)
    ic = InterferenceConfig(6, dur, rise)
    # Same episode timeline for both margins (common random numbers).
    run_seed = int(np.random.SeedSequence([BASE_SEED, 3, dur, rise]).generate_state(1)[0])
    run_i = simulate_link_with_interference(cfg, dist, 10, 3600 * (0.2 if pilot else 10), frame_b, ic, seed=run_seed)
    ev = evaluate_detector(run_i["times"], run_i["received"], run_i["snr_db"], run_i["episodes"], DegradedLinkDetector(20, .3, 6, 3), grace_s=2.0)
    delays = ev["trigger_delays_s"]
    return _plain({"margin_db": margin, "distance_m": dist, "mean_duration_s": dur, "noise_rise_db": rise,
        "mean_trigger_s": float(np.mean(delays)) if delays else None, "p95_trigger_s": float(np.percentile(delays, 95)) if delays else None,
        "fraction_detected": len(delays) / max(len(run_i["episodes"]), 1), "false_triggers_per_hour": ev["false_triggers_per_hour"],
        "missed_episodes": ev["missed"], "n_episodes": len(run_i["episodes"]),
        "mean_clear_after_episode_s": (float(np.mean(cl)) if (cl := _clear_delays(run_i["episodes"], ev["transitions"])) else None)})


def run(output: Path, argv: list[str], pilot: bool = False, processes: int | None = None, workers: int = 1) -> dict:
    start = time.perf_counter(); output = prepare_output(output)
    rates = (10,) if pilot else RATES; timeouts = (0.3, 1.0) if pilot else TIMEOUTS
    conditions = _conditions(pilot)
    false_rows, delay_rows = [], []
    for hi, h in enumerate(rates):
        for ti, T in enumerate(timeouts):
            d = detection_delay_samples(h, T, 500 if pilot else 10000, _rng(2, 99, hi, ti), pre_loss=None)
            lo_b, hi_b, mean_b = detection_delay_bounds(h, T)
            delay_rows.append({"h_hz": h, "timeout_s": T, "loss_condition": "no_prior_loss", "mean_s": float(np.mean(d)),
                               "p95_s": float(np.percentile(d, 95)), "max_s": float(np.max(d)),
                               "bound_low_s": lo_b, "bound_high_s": hi_b, "bound_mean_s": mean_b})
    # Each (condition, rate) cell draws from its own seeded streams, so cells are
    # independent and can run in any process; run_jobs returns them in job order.
    cell_jobs = [{"ci": ci, "cond": cond, "hi": hi, "h": h, "timeouts": list(timeouts), "pilot": pilot}
                 for ci, cond in enumerate(conditions) for hi, h in enumerate(rates)]
    for cell in run_jobs(cell_jobs, "tools.c_exp016_heartbeat:_false_alarm_job", processes=workers, progress=False):
        false_rows += cell["false_rows"]; delay_rows += cell["delay_rows"]
    thresholds = select_thresholds(false_rows)
    # Add delays to threshold records where available.
    for t in thresholds:
        for key in ("false_alarm_below_1_per_hour", "false_alarm_below_0_01_per_hour"):
            if t[key]:
                d = next((x for x in delay_rows if x["h_hz"] == t["h_hz"] and x["loss_condition"] == t["loss_condition"] and x["timeout_s"] == t[key]["T_s"]), None)
                if d: t[key].update(mean_detection_delay_s=d["mean_s"], p95_detection_delay_s=d["p95_s"])
    # C6 interference cells, with a small pilot subset.
    int_rows = []; margins = (20, 30); durations = (1, 5, 30); rises = (10, 20, 30)
    if pilot: durations, rises, margins = (1,), (10,), (20,)
    frame_b = frame_bytes(32, default_phy()); cfg = channel_preset("indoor_gym")
    int_jobs = [{"margin": margin, "dur": dur, "rise": rise, "pilot": pilot}
                for margin in margins for dur in durations for rise in rises]
    int_rows = run_jobs(int_jobs, "tools.c_exp016_heartbeat:_interference_job", processes=workers, progress=False)
    rule_rows = [r for r in false_rows if r["timeout_s"] == 0.3 and r["h_hz"] in (10, 20)]
    with open(output / "false_alarms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FALSE_ALARM_COLUMNS); w.writeheader(); w.writerows(false_rows)
    with open(output / "detection_delay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DELAY_COLUMNS); w.writeheader(); w.writerows(delay_rows)
    with open(output / "interference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INTERFERENCE_COLUMNS); w.writeheader(); w.writerows(int_rows)
    threshold_doc = {"status": "simulated", "parameters_basis": "literature-derived",
                     "generating_command": "PYTHONPATH=. ./.venv/bin/python tools/c_exp016_heartbeat.py --output {out}",
                     "records": thresholds}
    schema_path = Path("sim/radio/thresholds_schema.json")
    if schema_path.exists(): jsonschema.validate(threshold_doc, json.loads(schema_path.read_text(encoding="utf-8")))
    write_json(output / "thresholds.json", threshold_doc)
    rule_summary = [{"h_hz": r["h_hz"], "loss_condition": r["loss_condition"],
                     "false_alarms_per_hour_simulated": r["simulated_rate_per_hour"],
                     "false_alarms_per_hour_closed_form": r["expected_rate_per_hour"],
                     "mean_time_between_false_alarms_s_closed_form": (3600.0 / r["expected_rate_per_hour"] if r["expected_rate_per_hour"] > 0 else None)}
                    for r in rule_rows]
    write_json(output / "summary.json", {"experiment": "EXP-016", "status": "simulated", "hours_per_cell": HOURS if not pilot else .2,
        "false_alarm_cells": len(false_rows), "z_abs_gt_4": sum(abs(r["z"]) > 4 for r in false_rows), "thresholds": thresholds,
        "product_rule_300ms": rule_summary,
        "interference_method": "link margin set by distance in the indoor_gym preset: mean SNR = (SNR where a 75 B frame on air is lost half the time) + margin; detector window 20 packets, loss threshold 0.3, SNR threshold 6 dB, at least 3 packets, grace 2 s after an episode ends",
        "snr_at_half_loss_db": _snr_at_half_loss(cfg, frame_b)})
    _write_figures(output, false_rows, rule_rows)
    outputs = ["summary.json", "false_alarms.csv", "detection_delay.csv", "interference.csv", "thresholds.json", "exp016_false_alarms.png", "exp016_rule_300ms.png"]
    stable_argv = ["--pilot" if pilot else "--full", "--processes", str(processes or 1)]
    return write_manifest(output, schema="zephyr-run-c-manifest-1", argv=stable_argv,
        command="PYTHONPATH=. ./.venv/bin/python tools/c_exp016_heartbeat.py --output {out}", config={"pilot": pilot, "rates": rates, "timeouts": timeouts},
        seeds=[BASE_SEED], outputs=outputs, limits=["All values are simulated; radio parameters are literature-derived.",
        "Heartbeats have zero jitter; bursts are classic Gilbert chains with burst length set in seconds (r = 1 / (duration x h)).",
        "The 1 per 100 hours threshold uses the closed form only: 10 simulated hours cannot resolve it.",
        "When burst duration x heartbeat rate < 1 (e.g. 0.05 s or 0.2 s at 5 Hz) a burst is a single packet, so those cases coincide."],
        runtime_s=time.perf_counter() - start,
        n_processes=workers, hashed_sources=["sim/radio/heartbeat.py", "sim/radio/interference.py", "sim/radio/channel.py", "tools/c_exp016_heartbeat.py"])


def main(argv=None):
    argv = argv_or_default(argv); p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); p.add_argument("--output", type=Path, required=True); p.add_argument("--pilot", action="store_true"); p.add_argument("--processes", type=int, default=default_processes(), help="kept for old commands; ignored"); p.add_argument("--workers", type=int, default=1, help="worker processes for the independent cells (default 1 = serial); outputs are byte-identical for any value"); a = p.parse_args(argv); run(a.output, argv, a.pilot, a.processes, a.workers); return 0


if __name__ == "__main__": raise SystemExit(main())
