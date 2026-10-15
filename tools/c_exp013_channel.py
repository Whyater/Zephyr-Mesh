"""EXP-013: distance, obstacles and packet error for a 32 B ESP-NOW state frame.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/c_exp013_channel.py --output runs/c-exp013-channel

Writes summary.json, curves.csv, exp013_delivery_vs_distance.png and
manifest.json.  Everything is simulated with literature-derived parameters.
The shadowing draws are common random numbers: the same 20 000 standard
normal draws are reused at every distance and in every case, so curves are
smooth and directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim.radio.channel import mean_rssi_dbm, packet_error_rate  # noqa: E402
from sim.radio.mac import frame_bytes  # noqa: E402
from sim.radio.params import PARAMS, channel_preset, default_phy  # noqa: E402
from tools.c_common import (argv_or_default, prepare_output, seeds_from, write_json,  # noqa: E402
                            write_manifest)
from tools.horizon_mpl import apply_style  # noqa: E402

BASE_SEED = 20261014
N_DRAWS = 20_000
PAYLOAD_BYTES = 32
# Pre-registered grid was geomspace(1, 3000, 60).  Extended to 30 km because the
# nominal presets keep delivering past 3 km (a recorded deviation, not a change of question).
DISTANCES_M = np.geomspace(1.0, 30_000.0, 121)
GYM_SIZE_M = 50.0
BREAKPOINT_M = 4 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)  # 4 h_t h_r / wavelength, both antennas at 1.5 m


def build_cases() -> list[dict]:
    p = PARAMS
    cases = []
    for preset, n_key, s_key in (("indoor_gym", "exponent_indoor_large_room", "sigma_indoor_large_room_db"),
                                 ("outdoor_open", "exponent_outdoor_open", "sigma_outdoor_open_db")):
        n_low, n_nom, n_high = p[n_key].low, p[n_key].value, p[n_key].high
        sigma_nom, sigma_high = p[s_key].value, p[s_key].high
        for label, n in (("n low", n_low), ("n nominal", n_nom), ("n high", n_high)):
            cases.append(dict(preset=preset, case=label, exponent=n, sigma=sigma_nom))
        cases.append(dict(preset=preset, case="n high, sigma high", exponent=n_high, sigma=sigma_high))
        for label, extra in (("1 body", dict(bodies=1)), ("1 body worst (21 dB)", dict(bodies=1, body_loss_db=21.0)),
                             ("1 brick wall", dict(walls=1, wall_loss_db=p["wall_brick_db"].value)),
                             ("orientation 6 dB", dict(orientation_loss_db=p["orientation_loss_db"].high)),
                             ("body + orientation", dict(bodies=1, orientation_loss_db=p["orientation_loss_db"].high))):
            cases.append(dict(preset=preset, case=label, exponent=n_nom, sigma=sigma_nom, **extra))
    for n_far in (2.0, 3.0, 4.0):
        cases.append(dict(preset="outdoor_open", case=f"ground bounce, n_far {n_far:g}", exponent=p["exponent_outdoor_open"].value,
                          sigma=p["sigma_outdoor_open_db"].value, breakpoint_m=BREAKPOINT_M, exponent_far=n_far))
    cases.append(dict(preset="outdoor_open", case="stress corner", exponent=p["exponent_outdoor_open"].value,
                      sigma=p["sigma_outdoor_open_db"].high, breakpoint_m=BREAKPOINT_M, exponent_far=4.0,
                      orientation_loss_db=p["orientation_loss_db"].high))
    return cases


def evaluate(case: dict, z: np.ndarray, n_bytes: int) -> dict:
    overrides = dict(exponent=case["exponent"], shadowing_sigma_db=case["sigma"], orientation_loss_db=case.get("orientation_loss_db", 0.0))
    for key in ("body_loss_db", "wall_loss_db", "breakpoint_m", "exponent_far"):
        if key in case:
            overrides[key] = case[key]
    cfg = channel_preset(case["preset"], **overrides)
    mean = np.asarray(mean_rssi_dbm(DISTANCES_M, cfg, walls=case.get("walls", 0), bodies=case.get("bodies", 0)))
    rssi = mean[:, None] + cfg.shadowing_sigma_db * z[None, :]
    delivery = 1.0 - np.asarray(packet_error_rate(rssi - cfg.noise_floor_dbm, n_bytes, cfg))
    mean_delivery = delivery.mean(axis=1)
    band = np.percentile(delivery, [5, 95], axis=1)

    def last_at_least(level):
        ok = np.flatnonzero(mean_delivery >= level)
        return float(DISTANCES_M[ok[-1]]) if ok.size else None

    d90, d10 = last_at_least(0.9), last_at_least(0.1)
    at = {f"{d:g}": float(np.interp(np.log(d), np.log(DISTANCES_M), mean_delivery)) for d in (30.0, 50.0, 100.0, 300.0)}
    return dict(preset=case["preset"], case=case["case"], exponent=case["exponent"], sigma_db=cfg.shadowing_sigma_db,
                d90_m=d90, d10_m=d10, cliff_ratio_d10_over_d90=None if not (d90 and d10) else d10 / d90,
                delivery_at_m=at, beyond_grid=bool(mean_delivery[-1] >= 0.9),
                mean_delivery=mean_delivery, band=band, mean_rssi=mean)


def make_figure(results: list[dict], path: Path) -> None:
    colors = apply_style("light")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    picks = {
        "indoor_gym": ["n nominal", "n high, sigma high", "body + orientation", "1 body worst (21 dB)"],
        "outdoor_open": ["n nominal", "n high", "ground bounce, n_far 4", "stress corner"],
    }
    line_colors = [colors["accent"], colors["keepOut"], colors["ok"], colors["textSecondary"]]
    for ax, preset in zip(axes, ("indoor_gym", "outdoor_open")):
        ax.axvspan(1, GYM_SIZE_M, color=colors["selectionFill"], lw=0)
        ax.text(1.3, 0.84, "gym-sized\n(under 50 m)", color=colors["textSecondary"], fontsize=8, va="top")
        for name, color in zip(picks[preset], line_colors):
            r = next(x for x in results if x["preset"] == preset and x["case"] == name)
            if name == "n nominal":
                ax.fill_between(DISTANCES_M, r["band"][0], r["band"][1], color=color, alpha=0.15, lw=0)
            ax.plot(DISTANCES_M, r["mean_delivery"], color=color, lw=1.8, label=name)
        ax.axhline(0.9, color=colors["caution"], lw=1.0, ls="--")
        ax.text(DISTANCES_M[-1] * 0.8, 0.915, "90 %", color=colors["caution"], fontsize=8, ha="right")
        ax.set_xscale("log")
        ax.set_xlim(1, DISTANCES_M[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("distance (m, log scale)")
        ax.set_title("indoor gym preset" if preset == "indoor_gym" else "outdoor open preset")
        ax.legend(loc="center left", fontsize=8, frameon=False)
    axes[0].set_ylabel("delivery ratio, 32 B ESP-NOW frame")
    fig.suptitle("Simulated delivery vs distance (literature-derived parameters; shaded: 5 to 95 % over shadowing)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def run(output: Path, argv: list[str]) -> dict:
    start = time.perf_counter()
    output = prepare_output(output)
    seed = seeds_from(BASE_SEED, 1)[0]
    z = np.random.default_rng(seed).standard_normal(N_DRAWS)
    n_bytes = frame_bytes(PAYLOAD_BYTES, default_phy())
    results = [evaluate(case, z, n_bytes) for case in build_cases()]
    summary = {
        "experiment": "EXP-013",
        "status": "simulated; literature-derived parameters",
        "payload_bytes": PAYLOAD_BYTES, "frame_bytes": n_bytes, "n_shadow_draws": N_DRAWS,
        "breakpoint_m": BREAKPOINT_M,
        "grid_deviation": "distance grid extended from 3 km (pre-registered) to 30 km because nominal presets deliver past 3 km",
        "cases": [{k: v for k, v in r.items() if k not in ("mean_delivery", "band", "mean_rssi")} for r in results],
        "gym_min_delivery_under_50m": min(
            float(r["mean_delivery"][DISTANCES_M <= GYM_SIZE_M].min()) for r in results if r["preset"] == "indoor_gym"),
    }
    write_json(output / "summary.json", summary)
    with open(output / "curves.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["preset", "case", "distance_m", "mean_delivery", "delivery_p05", "delivery_p95", "mean_rssi_dbm"])
        for r in results:
            for i, d in enumerate(DISTANCES_M):
                writer.writerow([r["preset"], r["case"], f"{d:.6g}", f"{r['mean_delivery'][i]:.6f}",
                                 f"{r['band'][0][i]:.6f}", f"{r['band'][1][i]:.6f}", f"{r['mean_rssi'][i]:.4f}"])
    make_figure(results, output / "exp013_delivery_vs_distance.png")
    config = dict(base_seed=BASE_SEED, n_draws=N_DRAWS, payload_bytes=PAYLOAD_BYTES,
                  distances=[float(DISTANCES_M[0]), float(DISTANCES_M[-1]), len(DISTANCES_M)], cases=build_cases())
    return write_manifest(output, schema="zephyr-run-c-manifest-1", argv=argv,
                          command="PYTHONPATH=. ./.venv/bin/python tools/c_exp013_channel.py --output {out}",
                          config=config, seeds=[seed], outputs=["summary.json", "curves.csv", "exp013_delivery_vs_distance.png"],
                          limits=["Free-space-like log-distance model; no fast fading term; one PER curve (DBPSK) anchored at the datasheet sensitivity.",
                                  "No measured gym or ESP32 path-loss exponent exists; presets come from ITU, 3GPP and UAV survey values."],
                          runtime_s=time.perf_counter() - start, n_processes=1,
                          hashed_sources=["sim/radio/channel.py", "sim/radio/params.py", "tools/c_exp013_channel.py"])


def main(argv: list[str] | None = None) -> int:
    argv = argv_or_default(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
