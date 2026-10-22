#!/usr/bin/env python3
"""Which radio-model settings give loss that looks like the tracker's bursts?

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/burst_bridge_radio_table.py --output sim/runs/burst-bridge-radio

One drone sends 10 Hz, 32-byte packets to the ground station through the
existing radio model (sim/radio), using EXP-013/EXP-017's outdoor stress-corner
channel.  Nothing new is modelled: the sweep only turns existing knobs:

- shadowing mode: ``iid`` (new fade every packet), ``static`` (one fade per
  run), ``time_correlated`` with correlation time 0.1, 0.5, 2 and 10 s;
- MAC: unicast with 5 retries, or broadcast (no retries);
- distance: 12 values from 380 to 860 m (where loss crosses 30 % in these settings).

Each (setting, distance, seed) run is analysed with tools/burst_bridge.py.
summary.json lists, for each (MAC, shadowing) pair, the distance whose pooled
loss is closest to 30 %, with its loss runs, verdict and nearest tracker cell.
Everything is simulated with literature-derived radio parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.radio.model import NetworkModelConfig, RadioLinkConfig, make_network_model
from sim.radio.params import default_mac
from tools.burst_bridge import (
    VERDICT_BURSTY,
    VERDICT_IID,
    VERDICT_UNKNOWN,
    Piece,
    bridge,
)
from tools.c_exp017_tolerance import stress_corner

BASE_SEED = 20261021
RATE_HZ = 10.0
PAYLOAD_BYTES = 32
DURATION_S = 200.0
N_SEEDS = 8
DISTANCES_M = tuple(float(d) for d in np.round(np.geomspace(380.0, 860.0, 12), 1))
SHADOWING = (("iid", None), ("static", None), ("time_correlated", 0.1), ("time_correlated", 0.5),
             ("time_correlated", 2.0), ("time_correlated", 10.0))
MACS = ("unicast", "broadcast")
TARGET_LOSS = 0.30
OFF_TARGET_WARN = 0.05  # closest distance still this far from 30 % loss: flagged in summary.json
CSV_COLUMNS = ("mac", "shadowing", "corr_time_s", "distance_m", "n_seeds", "n_packets", "loss_rate",
               "seed_loss_min", "seed_loss_max", "seeds_bursty", "seeds_independent", "seeds_too_short", "runs_complete", "mean_run_packets", "p95_run_packets",
               "mean_run_s", "p95_run_s", "gilbert_p", "gilbert_r", "approx_z", "verdict", "nearest_cell",
               "nearest_distance", "loss_off_target", "note")


def run_seed(mac: str, mode: str, tau: float | None, distance_m: float, seed: int) -> Piece:
    channel = stress_corner(mode)
    if tau is not None:
        channel = replace(channel, shadowing_corr_time_s=tau)
    unicast = mac == "unicast"
    cfg = NetworkModelConfig(mode="radio", radio=RadioLinkConfig(
        positions={"gs": (0.0, 0.0, 1.5), "target": (distance_m, 0.0, 1.5)}, channel=channel,
        mac=default_mac(unicast_retry_limit=5 if unicast else 0), unicast=unicast, payload_bytes=PAYLOAD_BYTES))
    model = make_network_model(cfg, seed=seed)
    n = round(DURATION_S * RATE_HZ)
    for i in range(n):
        model.send("target", "gs", i, i / RATE_HZ)
    model.deliver_all()
    events = sorted(model.events, key=lambda e: e.seq)
    if any(e.outcome not in ("received", "lost") for e in events):
        raise RuntimeError("radio model left a packet unresolved")
    return Piece(tuple(float(e.send_time) for e in events), tuple(e.outcome == "lost" for e in events))


def seeds_for(cell_index: int) -> list[int]:
    return [int(v) for v in np.random.SeedSequence([BASE_SEED, cell_index]).generate_state(N_SEEDS)]


def build_rows() -> list[dict]:
    rows = []
    idx = 0
    for mac in MACS:
        for mode, tau in SHADOWING:
            for d in DISTANCES_M:
                pieces = [run_seed(mac, mode, tau, d, s) for s in seeds_for(idx)]
                idx += 1
                res = bridge(pieces, {"format": "radio-sim", "loss_kind": "one_way"})
                seed_loss = [float(np.mean(p.lost)) for p in pieces]
                near = res["nearest_cells"][0] if res["nearest_cells"] else None
                r, g = res["runs"], res["gilbert"]
                rows.append({
                    "mac": mac, "shadowing": mode, "corr_time_s": tau, "distance_m": d, "n_seeds": N_SEEDS,
                    "n_packets": res["n_packets"], "loss_rate": res["loss_rate"],
                    "seed_loss_min": min(seed_loss), "seed_loss_max": max(seed_loss),
                    "seeds_bursty": res["per_piece"]["verdict_counts"][VERDICT_BURSTY],
                    "seeds_independent": res["per_piece"]["verdict_counts"][VERDICT_IID],
                    "seeds_too_short": res["per_piece"]["verdict_counts"][VERDICT_UNKNOWN],
                    "runs_complete": r["n_complete"], "mean_run_packets": r["mean_packets"],
                    "p95_run_packets": r["p95_packets"], "mean_run_s": r["mean_s"], "p95_run_s": r["p95_s"],
                    "gilbert_p": g["p_good_to_bad"], "gilbert_r": g["p_bad_to_good"],
                    "approx_z": g["burstiness"]["approx_z"], "verdict": res["verdict"],
                    "nearest_cell": near["cell"] if near else None,
                    "nearest_distance": near["distance"] if near else None,
                    "loss_off_target": abs(res["loss_rate"] - TARGET_LOSS) if res["loss_rate"] is not None else None,
                    "note": ("static: one fade per seed, so each seed is all-or-nothing independent loss; the "
                             "pooled verdict and nearest cell mix seeds" if mode == "static" else ""),
                })
    return rows


def closest_to_target(rows: list[dict]) -> list[dict]:
    out = []
    for mac in MACS:
        for mode, tau in SHADOWING:
            group = [r for r in rows if r["mac"] == mac and r["shadowing"] == mode and r["corr_time_s"] == tau
                     and r["loss_rate"] is not None]
            if group:
                best = dict(min(group, key=lambda r: abs(r["loss_rate"] - TARGET_LOSS)))
                best["off_target_warning"] = best["loss_off_target"] > OFF_TARGET_WARN
                out.append(best)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", default="sim/runs/burst-bridge-radio")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = build_rows()
    with (out / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in CSV_COLUMNS})
    summary = {
        "status": "simulated (literature-derived radio parameters, no measured radio data)",
        "command": "PYTHONPATH=. ./.venv/bin/python tools/burst_bridge_radio_table.py --output " + args.output,
        "base_seed": BASE_SEED, "rate_hz": RATE_HZ, "payload_bytes": PAYLOAD_BYTES, "duration_s": DURATION_S,
        "n_seeds": N_SEEDS, "distances_m": list(DISTANCES_M), "target_loss": TARGET_LOSS,
        "channel": "EXP-013/EXP-017 outdoor stress corner (tools/c_exp017_tolerance.py stress_corner)",
        "closest_to_target_loss": closest_to_target(rows),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows in {time.time() - t0:.0f} s")
    for r in summary["closest_to_target_loss"]:
        tau = "" if r["corr_time_s"] is None else f" {r['corr_time_s']} s"
        print(f"{r['mac']:9s} {r['shadowing']}{tau:6s} d={r['distance_m']:6.1f} m loss={r['loss_rate']:.3f} "
              f"run={r['mean_run_packets'] or float('nan'):.2f} pk p95={r['p95_run_s'] or float('nan'):.2f} s "
              f"seeds bursty/iid/short {r['seeds_bursty']}/{r['seeds_independent']}/{r['seeds_too_short']} "
              f"-> {r['nearest_cell']} ({r['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
