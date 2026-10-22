"""EXP-014: how many simulated drones before the air gets crowded?

Usage: ``PYTHONPATH=. ./.venv/bin/python tools/c_exp014_mac.py --output sim/runs/c-exp014-mac``
"""
from __future__ import annotations

import argparse
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.radio.channel import Channel
from sim.radio.mac import BROADCAST, MacSimulator, ack_airtime_s, data_airtime_s
from sim.radio.params import channel_preset, default_mac, default_phy
from sim.s6_recovery import median_interval, wilson_interval
from tools.c_common import default_processes, prepare_output, seeds_from, write_json, write_manifest

BASE_SEED = 20261014 + 14
N_VALUES = (1, 2, 4, 6, 8, 12)
RATES_HZ = (10, 20, 50)
MODES = ("unicast_r0", "unicast_r5", "unicast_r10", "broadcast")
PAYLOADS = (32, 64)
WARMUP_S = 1.0
AGE_START_S = 2.0


def predicted_channel_use(n: int, rate_hz: float, payload_bytes: int, mode: str, phy=None) -> float:
    """Predicted offered channel fraction from D8 airtimes and mean backoff."""
    phy = default_phy() if phy is None else phy
    t = phy.difs_s + data_airtime_s(payload_bytes, phy) + 15.5 * phy.slot_s
    if mode != "broadcast":
        t += phy.sifs_s + ack_airtime_s(phy)
    return float(n * rate_hz * t)


def age_of_information(deliveries: list[tuple[float, float, str]], drones: list[str], *, t0: float = 2.0,
                       t1: float = 31.0, dt: float = 0.001) -> float:
    """Mean age in seconds, averaged over drones and 1 ms samples.

    A drone with no delivery has age equal to elapsed time since zero, as pre-registered.
    """
    if t1 <= t0 or dt <= 0:
        raise ValueError("invalid age interval")
    by = {d: [] for d in drones}
    for deliver_t, created_t, d in deliveries:
        if d in by and deliver_t <= t1:
            by[d].append((float(deliver_t), float(created_t)))
    for vals in by.values():
        vals.sort()
    times = np.arange(t0, t1, dt, dtype=float)
    total = 0.0
    for d in drones:
        latest = 0.0
        j = 0
        vals = by[d]
        for t in times:
            while j < len(vals) and vals[j][0] <= t + 1e-12:
                latest = max(latest, vals[j][1])
                j += 1
            total += t - latest
    return float(total / (len(drones) * len(times)))


LOSS_KINDS = ("collision", "channel_error", "below_sensitivity", "queue_drop", "retry_exhausted")


def loss_counts(frame_records, rx_records, scored_ids, *, destination="gs", broadcast=False) -> dict[str, int]:
    """Count undelivered scored frames by cause.

    Queue drops come from the frame record.  Otherwise the cause is the outcome
    of the last try recorded at the destination.  A broadcast frame with no
    record at the destination was below sensitivity there (the MAC writes no
    broadcast record for out-of-range nodes); a unicast frame with no record is
    counted as retry_exhausted.  One pass over the records (no rescans).
    """
    cats = {k: 0 for k in LOSS_KINDS}
    last: dict[int, str] = {}
    delivered: set[int] = set()
    for r in rx_records:
        if r.rx != destination:
            continue
        last[r.frame_id] = r.outcome
        if r.outcome == "delivered":
            delivered.add(r.frame_id)
    for fid in scored_ids:
        if fid in delivered:
            continue
        if frame_records[fid].status == "queue_drop":
            cats["queue_drop"] += 1
        elif fid in last:
            cats[last[fid] if last[fid] in cats else "retry_exhausted"] += 1
        else:
            cats["below_sensitivity" if broadcast else "retry_exhausted"] += 1
    return cats


def loss_decomposition(frame_records, rx_records, scored_ids, *, destination="gs", broadcast=False) -> dict[str, float]:
    """Fractions of undelivered scored frames per cause (they sum to 1, or all 0 if nothing was lost)."""
    cats = loss_counts(frame_records, rx_records, scored_ids, destination=destination, broadcast=broadcast)
    n = sum(cats.values())
    return {k: (v / n if n else 0.0) for k, v in cats.items()}


def _positions(seed: int) -> dict[str, tuple[float, float, float]]:
    a, _ = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(a)
    xy = rng.uniform([0.0, 0.0], [30.0, 20.0], size=(12, 2))
    z = rng.uniform(1.0, 3.0, size=12)
    out = {"gs": (0.0, 10.0, 1.0)}
    out.update({f"d{i:02d}": tuple(float(x) for x in (*xy[i], z[i])) for i in range(12)})
    return out


def _cell(cell: dict) -> dict:
    seed = int(cell["seed"])
    positions = _positions(seed)
    active = {"gs", *(f"d{i:02d}" for i in range(cell["n"]))}
    positions = {k: v for k, v in positions.items() if k in active}
    phy = default_phy()
    retry = int(cell["mode"].split("r")[1]) if cell["mode"].startswith("unicast") else 0
    mac0 = default_mac(unicast_retry_limit=retry)
    sim_seed = int(np.random.SeedSequence(seed).spawn(2)[1].generate_state(1)[0])
    channel = Channel(channel_preset("indoor_gym"), np.random.default_rng(np.random.SeedSequence(sim_seed)))
    sim = MacSimulator(positions, channel, phy, mac0, seed=sim_seed)
    broadcast = cell["mode"] == "broadcast"
    dst = BROADCAST if broadcast else "gs"
    drones = [f"d{i:02d}" for i in range(cell["n"])]
    t_stop, t_end = float(cell["t_stop"]), float(cell["t_stop"]) + 0.5
    for d in drones:
        sim.add_periodic_source(d, dst, cell["rate"], cell["payload"], start_s=0.0, stop_s=t_stop)
    sim.run_until(t_end)
    scored = [fid for fid, f in sorted(sim.frame_records.items()) if WARMUP_S <= f.created_s < t_stop]
    byid = set(scored)
    first: dict[int, object] = {}
    copies: dict[int, int] = {}
    pairs: set[tuple[int, str]] = set()
    for r in sim.rx_records:
        if r.frame_id not in byid or r.outcome != "delivered":
            continue
        pairs.add((r.frame_id, r.rx))
        if r.rx == "gs":
            first.setdefault(r.frame_id, r)
            copies[r.frame_id] = copies.get(r.frame_id, 0) + 1
    lat = sorted(r.deliver_s - r.created_s for r in first.values())
    deliveries = [(r.deliver_s, r.created_s, r.src) for r in first.values()]
    counts = loss_counts(sim.frame_records, sim.rx_records, scored, broadcast=broadcast)
    all_pairs_den = len(scored) * cell["n"]  # every other drone plus gs hears each broadcast
    return {**cell, "scored_frames": len(scored), "delivered": len(first),
            "delivery_ratio": len(first) / len(scored) if scored else None,
            "broadcast_all_pairs_ratio": len(pairs) / all_pairs_den if broadcast and all_pairs_den else None,
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "age_s": age_of_information(deliveries, drones, t0=AGE_START_S, t1=t_stop),
            "busy_fraction": sim.channel_busy_fraction(WARMUP_S, t_stop),
            "duplicate_frames": sum(1 for c in copies.values() if c > 1),
            "predicted_U": predicted_channel_use(cell["n"], cell["rate"], cell["payload"], cell["mode"], phy),
            "undelivered": len(scored) - len(first),
            **{f"lost_{k}": v for k, v in counts.items()}}


def _summary(rows: list[dict]) -> tuple[list[dict], dict]:
    cells = []
    for key in sorted({(r["n"], r["rate"], r["mode"], r["payload"]) for r in rows}):
        n, rate, mode, payload = key
        rs = [r for r in rows if (r["n"], r["rate"], r["mode"], r["payload"]) == key]
        med, lo, hi = median_interval([r["delivery_ratio"] for r in rs])
        succ, total = sum(r["delivered"] for r in rs), sum(r["scored_frames"] for r in rs)
        wlo, whi = wilson_interval(succ, total)
        p95s = [r["latency_p95_s"] for r in rs if r["latency_p95_s"] is not None]
        meds = [r["latency_median_s"] for r in rs if r["latency_median_s"] is not None]
        pmed, pl, ph = median_interval(p95s) if p95s else (None, None, None)
        lost = sum(r["undelivered"] for r in rs)
        # Pooled over seeds: share of all undelivered frames in the cell, per cause.
        shares = {k: (sum(r[f"lost_{k}"] for r in rs) / lost if lost else 0.0) for k in LOSS_KINDS}
        pairs = [r["broadcast_all_pairs_ratio"] for r in rs if r["broadcast_all_pairs_ratio"] is not None]
        cells.append({"n": n, "rate_hz": rate, "mode": mode, "payload_bytes": payload, "n_seeds": len(rs),
                      "delivery_median": med, "delivery_interval_low": lo, "delivery_interval_high": hi,
                      "delivery_pooled": succ / total if total else None, "delivery_wilson_low": wlo, "delivery_wilson_high": whi,
                      "broadcast_all_pairs_median": float(np.median(pairs)) if pairs else None,
                      "latency_median_median_s": float(np.median(meds)) if meds else None,
                      "latency_p95_median_s": pmed, "latency_p95_interval_low_s": pl, "latency_p95_interval_high_s": ph,
                      "age_median_s": float(np.median([r["age_s"] for r in rs])),
                      "busy_fraction_median": float(np.median([r["busy_fraction"] for r in rs])),
                      "predicted_U": rs[0]["predicted_U"], "undelivered_frames": lost,
                      "duplicate_frames": sum(r["duplicate_frames"] for r in rs),
                      "contention_loss_share": shares["collision"] + shares["queue_drop"] + shares["retry_exhausted"],
                      **{f"loss_share_{k}": v for k, v in shares.items()}})
    answer = {}
    for rate in sorted({c["rate_hz"] for c in cells}):
        for mode in MODES:
            base = sorted((c for c in cells if c["rate_hz"] == rate and c["mode"] == mode and c["payload_bytes"] == 32),
                          key=lambda c: c["n"])
            if not base:
                continue
            answer[f"rate_{rate}_{mode}"] = {
                "smallest_N_delivery_below_0.95": next((c["n"] for c in base if c["delivery_pooled"] < 0.95), None),
                "smallest_N_p95_latency_above_10ms": next((c["n"] for c in base if c["latency_p95_median_s"] is not None
                                                           and c["latency_p95_median_s"] > 0.010), None),
                "contention_dominant_at_N": [c["n"] for c in base if c["delivery_pooled"] < 0.99 and c["contention_loss_share"] > 0.5]}
    return cells, answer


def _cross_check(seed: int, duration_s: float) -> dict:
    """One saturated unicast sender, 250 B payload at 1 Mbps, retry limit 5: delivered packets per second.

    Hand budget per packet (no contention, mean backoff): DIFS 50 us + mean
    backoff 15.5 x 20 us = 310 us + data (250 + 43 B = 2344 bits + 192 us
    preamble = 2536 us) + SIFS 10 us + ACK 304 us = 3210 us, i.e. about 312
    packets/s.  With immediate access and post-transmission backoff only when
    a frame is queued, every packet after the first backs off.
    """
    phy = default_phy()
    positions = {"tx": (0.0, 0.0, 1.0), "gs": (5.0, 0.0, 1.0)}
    channel = Channel(channel_preset("indoor_gym"), np.random.default_rng(np.random.SeedSequence([seed, 1])))
    sim = MacSimulator(positions, channel, phy, default_mac(unicast_retry_limit=5, queue_capacity=10_000), seed=seed)
    # Keep the queue full: offer far more than the channel can carry.
    sim.add_periodic_source("tx", "gs", 2000.0, 250, start_s=0.0, stop_s=duration_s)
    sim.run_until(duration_s)
    t0 = 1.0
    delivered = sum(1 for r in sim.rx_records if r.rx == "gs" and r.outcome == "delivered" and r.attempt == 0
                    and t0 <= r.deliver_s < duration_s)
    hand = phy.difs_s + 15.5 * phy.slot_s + data_airtime_s(250, phy) + phy.sifs_s + ack_airtime_s(phy)
    rate = delivered / (duration_s - t0)
    return {"simulated_packets_per_s": rate, "window_s": [t0, duration_s], "hand_packet_time_s": hand,
            "hand_packets_per_s": 1.0 / hand, "pre_registered_hand_range_packets_per_s": [312, 345],
            "published_measurement_packets_per_s": 357, "published_measurement_strength": "weak (one blog post)",
            "gap_simulated_minus_published": rate - 357.0}


def _figure(cells, path1: Path, path2: Path, *, pilot: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    colors = apply_style("light"); fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    # One distinct colour, marker and line style per mode: unicast r5 and r10 can give
    # identical data, so the dashed r10 line must stay visible on top of r5.
    palette = [colors["accent"], colors["ok"], colors["keepOut"], colors["target"]]
    styles = [dict(marker="o", ls="-"), dict(marker="s", ls="-"), dict(marker="^", ls="--", mfc="none", ms=9, zorder=3),
              dict(marker="D", ls="-")]
    for ax, rate in zip(axes, RATES_HZ):
        for mode, color, style in zip(MODES, palette, styles):
            cc = [c for c in cells if c["rate_hz"] == rate and c["mode"] == mode and c["payload_bytes"] == 32]
            ax.plot([c["n"] for c in cc], [c["delivery_pooled"] for c in cc], color=color, label=mode, **style)
            ax.fill_between([c["n"] for c in cc], [c["delivery_wilson_low"] for c in cc], [c["delivery_wilson_high"] for c in cc], color=color, alpha=.12)
        ax.axhline(.95, color=colors["caution"], ls="--", lw=1); ax.set_title(f"{rate} Hz"); ax.set_xlabel("drones N")
    axes[0].set_ylabel("pooled delivery ratio at gs (simulated)"); axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path1, dpi=150, metadata={"Software": None}); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, rate in zip(axes, RATES_HZ):
        for mode, color, style in zip(MODES, palette, styles):
            cc = [c for c in cells if c["rate_hz"] == rate and c["mode"] == mode and c["payload_bytes"] == 32 and c["latency_p95_median_s"] is not None]
            ax.plot([c["n"] for c in cc], [c["latency_p95_median_s"] for c in cc], color=color, label=mode, **style)
        ax.set_yscale("log"); ax.set_title(f"{rate} Hz"); ax.set_xlabel("drones N")
    axes[0].set_ylabel("median p95 latency (s, log scale)"); axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path2, dpi=150, metadata={"Software": None}); plt.close(fig)


def run(output: Path, argv: list[str], *, pilot=False, processes=None, n_seeds: int = 50) -> dict:
    start = time.perf_counter(); output = prepare_output(output)
    seeds = seeds_from(BASE_SEED, 3 if pilot else int(n_seeds))
    ns = (1, 4) if pilot else N_VALUES; rates = (50,) if pilot else RATES_HZ; modes = ("unicast_r5", "broadcast") if pilot else MODES
    t_stop = 5.0 if pilot else 31.0
    cells = [{"n": n, "rate": rate, "mode": mode, "payload": payload, "seed": seed, "t_stop": t_stop}
             for n in ns for rate in rates for mode in modes for payload in ((32,) if n != 12 or pilot else PAYLOADS)
             for seed in seeds]
    proc = default_processes() if processes is None else int(processes)
    if proc > 1 and len(cells) > 1:
        with ProcessPoolExecutor(max_workers=proc) as ex:
            rows = list(ex.map(_cell, cells, chunksize=4))
    else:
        rows = [_cell(c) for c in cells]
    cell_rows, answer = _summary(rows)
    cross = _cross_check(seeds[0], 3.0 if pilot else 11.0)
    summary = {"experiment": "EXP-014", "status": "simulated; literature-derived parameters", "pilot": pilot,
               "n_seeds": len(seeds), "pre_registered_n_seeds": 50,
               "question": answer, "cross_check": cross,
               "scoring": {"warmup_s": WARMUP_S, "t_stop_s": t_stop, "age_start_s": AGE_START_S,
                           "age_no_delivery_rule": "a drone with no delivered frame yet has age equal to time since 0"},
               "default_mac": {k: getattr(default_mac(), k) for k in sorted(vars(default_mac()))}}
    write_json(output / "summary.json", summary)
    with open(output / "per_seed.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    with open(output / "cells.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=sorted(cell_rows[0])); w.writeheader(); w.writerows(cell_rows)
    _figure(cell_rows, output / "exp014_delivery.png", output / "exp014_latency.png")
    config = {"base_seed": BASE_SEED, "n_values": list(ns), "rates_hz": list(rates), "modes": list(modes),
              "payloads_bytes": list(PAYLOADS), "t_stop_s": t_stop, "pilot": pilot}
    runtime = time.perf_counter() - start
    print(f"runtime {runtime:.1f} s for {len(cells)} runs on {proc} processes")
    return write_manifest(output, schema="zephyr-run-c-manifest-1", argv=list(argv),
                          command="PYTHONPATH=. ./.venv/bin/python tools/c_exp014_mac.py --output {out}"
                          + (f" --seeds {len(seeds)}" if not pilot and len(seeds) != 50 else ""), config=config,
                          seeds=seeds, outputs=["summary.json", "per_seed.csv", "cells.csv", "exp014_delivery.png", "exp014_latency.png"],
                          limits=["Gym preset only: distance loss is about zero, so losses here are contention.",
                                  "Fixed positions per seed, no motion; periodic senders with random phase, no jitter."],
                          runtime_s=runtime, n_processes=proc,
                          hashed_sources=["sim/radio/mac.py", "sim/radio/channel.py", "sim/radio/params.py",
                                          "sim/s6_recovery.py", "tools/c_exp014_mac.py"])


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--processes", type=int, default=default_processes())
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds (pre-registered: 50)")
    args = parser.parse_args(argv)
    run(args.output, argv, pilot=args.pilot, processes=args.processes, n_seeds=args.seeds)
    return 0


if __name__ == "__main__": raise SystemExit(main())
