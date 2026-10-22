"""EXP-017: the EXP-006 tracker fed through the radio model instead of the abstract link.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/c_exp017_tolerance.py --output sim/runs/c-exp017-tolerance

Two scenarios turn physical settings into delay and loss:
- A: one drone at a fixed distance from the ground station, outdoor "stress
  corner" channel from EXP-013 (plus a 30 m gym control);
- B: the tracker's 10 Hz stream competing with N - 1 other drones in the gym.
The tracker arms, motions, seeds and failure rule are EXP-006's, imported
read-only from sim/s6_recovery.py.  The radio's delivery times do not depend
on the motion or the arm, so the radio runs once per (cell, seed) and both
motions and both arms reuse it.  Everything is simulated; radio parameters are
literature-derived.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim import s6_recovery  # noqa: E402
from sim.radio.model import BackgroundSource, NetworkModelConfig, RadioLinkConfig, make_network_model  # noqa: E402
from sim.radio.params import PARAMS, channel_preset, default_mac  # noqa: E402
from sim.sensor import SensorConfig, SensorModel  # noqa: E402
from tools.c_common import (REPO_ROOT, argv_or_default, default_processes, prepare_output, sha256_file,  # noqa: E402
                            write_json, write_manifest)
from tools.horizon_mpl import apply_style  # noqa: E402

BREAKPOINT_M = 4 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)  # same dual-slope breakpoint as EXP-013
NOISE_STD_M = 0.05  # EXP-006 main-slice sensor noise used for the mapping
EXP006_SUMMARY = "sim/runs/s6-recovery/summary.json"
DISTANCES_M = tuple(float(d) for d in np.geomspace(250.0, 1000.0, 16))
GYM_CONTROL_M = 30.0
N_VALUES = (1, 4, 8, 12)
BACKGROUND_RATES_HZ = (10.0, 20.0, 50.0)
MODES = ("unicast", "broadcast")
PROCESSING_S = (0.0, 0.0015)
# Scenario B (N up to 12 at 50 Hz) dominates the runtime: it uses the first 20 of the
# same 50 EXP-006 seeds (a recorded deviation; pre-registered 50).  Scenario A uses all 50.
SEEDS_A, SEEDS_B = 50, 20


# ---- delivery ------------------------------------------------------------------

def radio_events(radio_cfg: NetworkModelConfig, *, seed: int, config: s6_recovery.RecoveryConfig):
    """Send the tracker's captures through the network model; return (all sent events, model).

    Same seed split as s6_recovery.deliver: the link gets the first value of
    SeedSequence(seed).generate_state(2).  deliver_all() returns only the
    received packets, so the full list (every sent packet with its final
    outcome, "received" or "lost") comes from model.events.
    """
    link_seed = int(np.random.SeedSequence(seed).generate_state(2)[0])
    model = make_network_model(radio_cfg, seed=link_seed)
    n_captures = int(round(config.duration_s / config.observation_period_s))
    for seq, t in enumerate(np.arange(n_captures + 1) * config.observation_period_s):
        model.send("target", "gs", seq, float(t))
    model.deliver_all()
    assert all(e.outcome in ("received", "lost") for e in model.events)
    return list(model.events), model


def measurements(motion: str, *, noise_std_m: float, seed: int, config: s6_recovery.RecoveryConfig) -> dict:
    """The sensor half of s6_recovery.deliver: same seed, same capture times, same draws."""
    sensor_seed = int(np.random.SeedSequence(seed).generate_state(2)[1])
    sensor = SensorModel(SensorConfig(noise_std_m=float(noise_std_m)), seed=sensor_seed)
    n_captures = int(round(config.duration_s / config.observation_period_s))
    times = np.arange(n_captures + 1) * config.observation_period_s
    truth = s6_recovery.truth_position(motion, times, config)
    return {seq: tuple(float(v) for v in sensor.sample(pos, float(t)).measurement_m)
            for seq, (t, pos) in enumerate(zip(times, truth))}


def to_deliveries(events, meas: dict, config: s6_recovery.RecoveryConfig) -> list:
    """Received packets that land inside the run, sorted like s6_recovery.deliver."""
    out = [s6_recovery.Delivery(e.seq, float(e.send_time), float(e.receive_time), meas[e.seq])
           for e in events
           if e.outcome == "received" and e.receive_time <= config.duration_s + s6_recovery.TIME_TOLERANCE_S]
    out.sort(key=lambda d: (d.receive_time_s, d.seq))
    return out


def deliver_radio(motion: str, *, radio_cfg: NetworkModelConfig, noise_std_m: float, seed: int,
                  config: s6_recovery.RecoveryConfig) -> list:
    """Drop-in for s6_recovery.deliver with the link replaced by the network model."""
    events, _ = radio_events(radio_cfg, seed=seed, config=config)
    return to_deliveries(events, measurements(motion, noise_std_m=noise_std_m, seed=seed, config=config), config)


# ---- scenarios -----------------------------------------------------------------

def stress_corner(shadowing_mode: str):
    """EXP-013's outdoor stress corner (d90 409 m, d10 746 m there)."""
    return channel_preset("outdoor_open", shadowing_sigma_db=PARAMS["sigma_outdoor_open_db"].high,
                          breakpoint_m=BREAKPOINT_M, exponent_far=4.0,
                          orientation_loss_db=PARAMS["orientation_loss_db"].high, shadowing_mode=shadowing_mode)


def scenario_config(cell: dict, seed: int) -> NetworkModelConfig:
    unicast = cell["mode"] == "unicast"
    mac = default_mac(unicast_retry_limit=5 if unicast else 0, processing_delay_s=float(cell["processing_s"]))
    if cell["scenario"] == "A":
        positions = {"gs": (0.0, 0.0, 1.5), "target": (float(cell["distance_m"]), 0.0, 1.5)}
        channel = (channel_preset("indoor_gym", shadowing_mode=cell["shadowing"]) if cell["area"] == "gym"
                   else stress_corner(cell["shadowing"]))
        background = ()
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 17017]))  # positions only
        positions = {"gs": (0.0, 10.0, 1.0), "target": (15.0, 10.0, 2.0)}
        for i in range(1, int(cell["n"])):
            positions[f"d{i:02d}"] = (float(rng.uniform(0, 30)), float(rng.uniform(0, 20)), float(rng.uniform(1, 3)))
        channel = channel_preset("indoor_gym", shadowing_mode=cell["shadowing"])
        dst = "gs" if unicast else "*"
        background = tuple(BackgroundSource(f"d{i:02d}", dst, float(cell["rate_hz"]), payload_bytes=32,
                                            start_s=0.0, stop_s=s6_recovery.RecoveryConfig().duration_s)
                           for i in range(1, int(cell["n"])))
    return NetworkModelConfig(mode="radio", radio=RadioLinkConfig(positions=positions, channel=channel, mac=mac,
                                                                  unicast=unicast, background=background,
                                                                  payload_bytes=32))


def build_cells(pilot: bool = False) -> list[dict]:
    cells = []
    distances = (250.0, 750.0) if pilot else DISTANCES_M
    modes = ("unicast",) if pilot else MODES
    processing = (0.0,) if pilot else PROCESSING_S
    for mode in modes:
        for proc in processing:
            for shadowing in (("iid",) if pilot else ("iid", "static")):
                for d in distances:
                    cells.append(dict(scenario="A", area="field", distance_m=d, n=1, rate_hz=None, mode=mode,
                                      processing_s=proc, shadowing=shadowing))
            cells.append(dict(scenario="A", area="gym", distance_m=GYM_CONTROL_M, n=1, rate_hz=None, mode=mode,
                              processing_s=proc, shadowing="iid"))
            for n in ((1, 8) if pilot else N_VALUES):
                for rate in ((None,) if n == 1 else ((50.0,) if pilot else BACKGROUND_RATES_HZ)):
                    cells.append(dict(scenario="B", area="gym", distance_m=None, n=n, rate_hz=rate, mode=mode,
                                      processing_s=proc, shadowing="iid"))
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


# ---- one (cell, seed) ------------------------------------------------------------

def run_one(job: tuple[dict, int]) -> list[dict]:
    """One radio run, then both motions and both arms on its deliveries."""
    cell, seed = job
    config = s6_recovery.RecoveryConfig()
    radio_cfg = scenario_config(cell, seed)
    events, model = radio_events(radio_cfg, seed=seed, config=config)
    sent = len(events)
    received = [e for e in events if e.outcome == "received"]
    received_in_run = [e for e in received
                       if e.receive_time <= config.duration_s + s6_recovery.TIME_TOLERANCE_S]
    lat = np.array([e.receive_time - e.send_time for e in received])
    radio = {"sent": sent, "received": len(received), "received_in_run": len(received_in_run),
             "background_senders": len(radio_cfg.radio.background),
             "latency_median_s": float(np.median(lat)) if lat.size else None,
             "latency_p95_s": float(np.percentile(lat, 95)) if lat.size else None,
             "duplicates_suppressed": int(model.duplicates_suppressed)}
    rows = []
    for motion in s6_recovery.MOTIONS:
        deliveries = to_deliveries(events, measurements(motion, noise_std_m=NOISE_STD_M, seed=seed, config=config),
                                   config)
        arms = s6_recovery.run_arms(deliveries, motion, config)
        for arm in s6_recovery.ARMS:
            r = arms[arm]
            rows.append({"cell": cell["cell"], "seed": int(seed), "motion": motion, "arm": arm,
                         "failed": bool(r.failed), "p95_m": r.p95_error_m, "max_m": r.max_error_m, "rmse_m": r.rmse_m,
                         "fused": r.fused_count, "dropped": r.dropped_count, **radio})
    return rows


# ---- EXP-006 mapping ---------------------------------------------------------------

def nearest_exp006(exp006: dict, *, delay_s: float, loss: float, motion: str, arm: str) -> dict:
    """Nearest EXP-006 main-slice cell (independent loss, noise 0.05 m) in (delay, loss).

    Distances are measured in grid steps so delay (s) and loss (fraction) are
    comparable: each axis is divided by its grid spacing near the point.
    """
    main = [c for c in exp006["cells"] if c["slice"] == "main" and c["loss_model"] == "independent"
            and c["motion"] == motion and c["noise_std_m"] == NOISE_STD_M]
    if not main:
        raise ValueError("no EXP-006 main-slice cells for this motion")
    delays = sorted({c["delay_s"] for c in main})
    losses = sorted({c["loss_probability"] for c in main})
    d0 = min(delays, key=lambda d: (abs(d - delay_s), d))
    l0 = min(losses, key=lambda x: (abs(x - loss), x))
    hit = next(c for c in main if c["delay_s"] == d0 and c["loss_probability"] == l0)
    return {"cell_id": hit["cell_id"], "delay_s": d0, "loss_probability": l0, "label": hit["arms"][arm]["label"],
            "fail_rate": hit["arms"][arm]["fail_rate"]}


# ---- summaries -------------------------------------------------------------------

def summarize(cells: list[dict], rows: list[dict], exp006: dict) -> list[dict]:
    by_cell: dict[int, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    out = []
    for cell in cells:
        rs = by_cell[cell["cell"]]
        # radio statistics: one value per seed (the first row of each seed)
        per_seed = {r["seed"]: r for r in rs}
        sent = sum(r["sent"] for r in per_seed.values())
        received = sum(r["received"] for r in per_seed.values())
        received_in_run = sum(r["received_in_run"] for r in per_seed.values())
        dlo, dhi = s6_recovery.wilson_interval(received, sent)
        meds = [r["latency_median_s"] for r in per_seed.values() if r["latency_median_s"] is not None]
        p95s = [r["latency_p95_s"] for r in per_seed.values() if r["latency_p95_s"] is not None]
        delivery = received / sent if sent else None
        delivery_in_run = received_in_run / sent if sent else None
        latency = float(np.median(meds)) if meds else None
        base = {**{k: cell[k] for k in ("cell", "scenario", "area", "distance_m", "n", "rate_hz", "mode",
                                       "processing_s", "shadowing")},
                "n_seeds": len(per_seed), "delivery_pooled": delivery,
                "delivery_in_run_pooled": delivery_in_run, "delivery_wilson_low": dlo,
                "delivery_wilson_high": dhi, "latency_median_s": latency,
                "latency_p95_median_s": float(np.median(p95s)) if p95s else None,
                "seeds_with_no_delivery": sum(1 for r in per_seed.values() if r["received"] == 0),
                "duplicates_suppressed": sum(r["duplicates_suppressed"] for r in per_seed.values())}
        for motion in s6_recovery.MOTIONS:
            for arm in s6_recovery.ARMS:
                sub = [r for r in rs if r["motion"] == motion and r["arm"] == arm]
                fails = sum(r["failed"] for r in sub)
                flo, fhi = s6_recovery.wilson_interval(fails, len(sub))
                med, plo, phi = s6_recovery.median_interval([r["p95_m"] for r in sub])
                key = f"{arm}_{motion}"
                base[f"{key}_fail_fraction"] = fails / len(sub)
                base[f"{key}_fail_wilson_low"] = flo
                base[f"{key}_fail_wilson_high"] = fhi
                base[f"{key}_cell_failed"] = bool(2 * fails >= len(sub))  # EXP-006 cell rule, still a proposal
                base[f"{key}_p95_median_m"] = med
                base[f"{key}_p95_interval_low_m"] = plo
                base[f"{key}_p95_interval_high_m"] = phi
                if delivery is not None:
                    near = nearest_exp006(exp006, delay_s=latency or 0.0, loss=1.0 - delivery, motion=motion, arm=arm)
                    base[f"{key}_exp006_nearest_cell"] = near["cell_id"]
                    base[f"{key}_exp006_nearest_label"] = near["label"]
        out.append(base)
    return out


def headline(summary_cells: list[dict]) -> dict:
    """Data-derived only: where cells first fail (primary settings: unicast, processing 0, iid shadowing)."""
    out = {}
    primary = [c for c in summary_cells if c["mode"] == "unicast" and c["processing_s"] == 0.0
               and c["shadowing"] == "iid"]
    a = sorted((c for c in primary if c["scenario"] == "A" and c["area"] == "field"), key=lambda c: c["distance_m"])
    b = [c for c in primary if c["scenario"] == "B"]
    for motion in s6_recovery.MOTIONS:
        for arm in s6_recovery.ARMS:
            key = f"{arm}_{motion}"
            first_a = next((c for c in a if c[f"{key}_cell_failed"]), None)
            last_ok = [c for c in a if not c[f"{key}_cell_failed"]]
            by_rate = {}
            for rate in BACKGROUND_RATES_HZ:
                bs = sorted((c for c in b if c["rate_hz"] == rate), key=lambda c: c["n"])
                first_b = next((c for c in bs if c[f"{key}_cell_failed"]), None)
                by_rate[f"{rate:g}_hz"] = None if first_b is None else first_b["n"]
            out[key] = {
                "scenario_a_smallest_failing_distance_m": None if first_a is None else first_a["distance_m"],
                "scenario_a_delivery_at_that_distance": None if first_a is None else first_a["delivery_pooled"],
                "scenario_a_largest_passing_distance_below_it_m": (
                    None if first_a is None else max((c["distance_m"] for c in last_ok
                                                      if c["distance_m"] < first_a["distance_m"]), default=None)),
                "scenario_b_smallest_failing_n_by_background_rate": by_rate,
            }
    lat = [c["latency_median_s"] for c in summary_cells if c["latency_median_s"] is not None]
    p95 = [c["latency_p95_median_s"] for c in summary_cells if c["latency_p95_median_s"] is not None]
    out["max_cell_median_latency_s"] = max(lat) if lat else None
    out["max_cell_p95_latency_s"] = max(p95) if p95 else None
    return out


# ---- figures -----------------------------------------------------------------------

def make_figures(summary_cells: list[dict], exp006: dict, output: Path) -> None:
    colors = apply_style("light")
    series = [("recovery_constant_turn", colors["accent"], "recovery arm, 1 m turn"),
              ("recovery_constant_velocity", colors["ok"], "recovery arm, straight line"),
              ("naive_constant_turn", colors["keepOut"], "naive arm, 1 m turn"),
              ("naive_constant_velocity", colors["textSecondary"], "naive arm, straight line")]
    # Distance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    a = sorted((c for c in summary_cells if c["scenario"] == "A" and c["area"] == "field" and c["mode"] == "unicast"
                and c["processing_s"] == 0.0 and c["shadowing"] == "iid"), key=lambda c: c["distance_m"])
    a_static = sorted((c for c in summary_cells if c["scenario"] == "A" and c["area"] == "field"
                       and c["mode"] == "unicast" and c["processing_s"] == 0.0 and c["shadowing"] == "static"),
                      key=lambda c: c["distance_m"])
    d = [c["distance_m"] for c in a]
    ax1.plot(d, [c["delivery_pooled"] for c in a], color=colors["accent"], marker="o", ms=3,
             label="new shadowing draw per packet")
    ax1.plot([c["distance_m"] for c in a_static], [c["delivery_pooled"] for c in a_static], color=colors["textSecondary"],
             marker="s", ms=3, ls="--", label="one shadowing draw per seed")
    ax1.set_ylabel("tracker packets delivered")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.set_title("Simulated: one drone at a fixed distance, outdoor stress corner, unicast with 5 retries", fontsize=9)
    for key, color, label in series:
        ax2.plot(d, [c[f"{key}_fail_fraction"] for c in a], color=color, marker="o", ms=3, label=label)
    ax2.axhline(0.5, color=colors["alert"], lw=1.0, ls="--")
    ax2.text(d[0], 0.53, "cell fails at 50 % of seeds", color=colors["alert"], fontsize=8)
    ax2.set_xscale("log")
    ticks = [250, 300, 400, 500, 600, 800, 1000]
    ax2.set_xticks(ticks, [str(t) for t in ticks])
    ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.set_xlabel("distance to ground station (m, log scale)")
    ax2.set_ylabel("fraction of seeds failing (50)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "exp017_distance.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    # Contention
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        for rate, color in zip(BACKGROUND_RATES_HZ, (colors["ok"], colors["accent"], colors["keepOut"])):
            bs = sorted((c for c in summary_cells if c["scenario"] == "B" and c["mode"] == mode
                         and c["processing_s"] == 0.0 and (c["rate_hz"] == rate or c["n"] == 1)), key=lambda c: c["n"])
            ax.plot([c["n"] for c in bs], [c["recovery_constant_turn_fail_fraction"] for c in bs], color=color,
                    marker="o", ms=3, label=f"others send at {rate:g} Hz")
        ax.axhline(0.5, color=colors["alert"], lw=1.0, ls="--")
        ax.set_title(f"{mode} (recovery arm, 1 m turn)", fontsize=9)
        ax.set_xlabel("drones sharing the channel, N")
        ax.set_xticks(N_VALUES)
    axes[0].set_ylabel("fraction of seeds failing (20)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Simulated: tracker stream at 10 Hz in the gym with N - 1 other senders", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "exp017_contention.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    # EXP-006 boundary with the radio cells
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    main = [c for c in exp006["cells"] if c["slice"] == "main" and c["loss_model"] == "independent"
            and c["motion"] == "constant_turn" and c["noise_std_m"] == NOISE_STD_M]
    for label, color, marker in (("passes", colors["ok"], "o"), ("fails", colors["alert"], "o")):
        pts = [c for c in main if c["arms"]["recovery"]["label"] == label]
        ax.scatter([c["delay_s"] for c in pts], [c["loss_probability"] for c in pts], color=color, marker=marker,
                   s=36, alpha=0.35, label=f"EXP-006 cell {label}")
    for scen, color, marker, label in (("A", colors["accent"], "x", "radio, scenario A (distance)"),
                                       ("B", colors["text"], "+", "radio, scenario B (contention)")):
        pts = [c for c in summary_cells if c["scenario"] == scen and c["delivery_pooled"] is not None]
        ax.scatter([c["latency_median_s"] or 0.0 for c in pts], [1.0 - c["delivery_pooled"] for c in pts],
                   color=color, marker=marker, s=28, label=label)
    ax.set_xlabel("median delay (s)")
    ax.set_ylabel("loss fraction")
    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Simulated: where the radio cells land on EXP-006's map (recovery arm, turn)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "exp017_boundary.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


# ---- run -----------------------------------------------------------------------------

def run(output: Path, argv: list[str], *, pilot: bool = False, processes: int | None = None) -> dict:
    start = time.perf_counter()
    output = prepare_output(output)
    exp006_path = REPO_ROOT / EXP006_SUMMARY
    exp006 = json.loads(exp006_path.read_text())  # missing file is an error, never silently empty
    seeds = s6_recovery.seed_list(3 if pilot else SEEDS_A)
    n_b = 3 if pilot else SEEDS_B
    cells = build_cells(pilot)
    jobs = [(c, s) for c in cells for s in (seeds if c["scenario"] == "A" else seeds[:n_b])]
    proc = default_processes() if processes is None else int(processes)
    if proc > 1:
        with ProcessPoolExecutor(max_workers=proc) as pool:
            rows = [r for chunk in pool.map(run_one, jobs, chunksize=2) for r in chunk]
    else:
        rows = [r for job in jobs for r in run_one(job)]
    summary_cells = summarize(cells, rows, exp006)
    with open(output / "per_seed.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "cells.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_cells[0]))
        writer.writeheader()
        writer.writerows(summary_cells)
    summary = {"experiment": "EXP-017", "status": "simulated; radio parameters literature-derived", "pilot": pilot,
               "n_seeds_scenario_a": len(seeds), "n_seeds_scenario_b": n_b, "pre_registered_n_seeds": 50,
               "headline": headline(summary_cells), "cells": summary_cells,
               "exp006_summary_sha256": sha256_file(exp006_path),
               "rules": {"seed_fails": "p95 error > 0.25 m or max error > 0.50 m after 2 s burn-in (EXP-006)",
                         "cell_fails": "at least 50 % of seeds fail (the EXP-006 cell rule, still a proposal)",
                         "delivery": "received over sent, all 121 tracker packets, any arrival time",
                         "delivery_in_run": "received at or before the end of the 12 s run, over sent; the tracker only sees these",
                         "mapping": "nearest EXP-006 main-slice cell (independent loss, noise 0.05 m) by grid value"}}
    write_json(output / "summary.json", summary)
    make_figures(summary_cells, exp006, output)
    runtime = time.perf_counter() - start
    print(f"runtime {runtime:.1f} s for {len(jobs)} radio runs on {proc} processes")
    outputs = ["summary.json", "per_seed.csv", "cells.csv", "exp017_distance.png", "exp017_contention.png",
               "exp017_boundary.png"]
    config = {"pilot": pilot, "distances_m": list(DISTANCES_M), "gym_control_m": GYM_CONTROL_M, "n_values": list(N_VALUES),
              "background_rates_hz": list(BACKGROUND_RATES_HZ), "modes": list(MODES), "processing_s": list(PROCESSING_S),
              "noise_std_m": NOISE_STD_M, "n_cells": len(cells), "seeds_a": len(seeds), "seeds_b": n_b}
    return write_manifest(
        output, schema="zephyr-run-c-manifest-1", argv=argv,
        command="PYTHONPATH=. ./.venv/bin/python tools/c_exp017_tolerance.py --output {out}", config=config,
        seeds=seeds, outputs=outputs,
        limits=["Scenario B uses 20 seeds (the first 20 of EXP-006's 50) to fit the runtime budget; pre-registered 50.",
                "Static shadowing (one draw per node pair per seed) is an addition to the pre-registered design.",
                "Scenario A holds the radio distance fixed while the tracked motion stays near the origin.",
                "Gym contention uses periodic background senders with random phase; no hidden terminals in the gym.",
                "The EXP-006 cell rule (50 % of seeds) is still a proposal, not an adopted rule."],
        runtime_s=runtime, n_processes=proc,
        hashed_sources=["sim/s6_recovery.py", "sim/tracker.py", "sim/sensor.py", "sim/link.py", "sim/radio/channel.py",
                        "sim/radio/mac.py", "sim/radio/model.py", "sim/radio/params.py", "sim/radio/seq.py",
                        "tools/c_exp017_tolerance.py", EXP006_SUMMARY])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=default_processes())
    args = parser.parse_args(argv)
    run(args.output, argv, pilot=args.pilot, processes=args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
