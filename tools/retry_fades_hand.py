"""EXP-038 (proposal) hand check: delivery with k retries, by direct integration over the fade.

Usage (repo root):

    PYTHONPATH=. ./.venv/bin/python tools/retry_fades_hand.py --output sim/runs/retry-fades-hand

This does not call the channel or MAC code.  It writes down the EXP-017
outdoor stress corner by hand (dual-slope path loss, a 4 dB normal fade, the
DBPSK bit-error curve anchored at the datasheet sensitivity) and integrates
over the fade z on a fine grid.  The only import from sim/ is PARAMS, used to
assert that the hand-typed inputs still match the presets.

For one attempt at distance d, PER(z) is the chance the 75-byte frame fails
when the fade is z standard deviations.  Then:

- p = E_z[PER(z)], the per-attempt loss averaged over the fade;
- broadcast (one attempt): delivery = 1 - p;
- unicast, fresh fade per attempt (iid): every attempt fails with chance p on
  its own, so all 6 fail with chance p^6 and delivery = 1 - p^6;
- unicast, fade frozen for the whole retry span: all 6 attempts see the same
  z, so delivery = E_z[1 - PER(z)^6].  When PER(z) is close to 0 or 1 for
  almost every z (a steep curve), this is close to 1 - p: retries add little.

The run also compares one cell (630 m, 50 seeds) against the registered
EXP-017 per-seed file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# Hand-typed inputs (sim/radio/params.py and tools/c_exp017_tolerance.py, 2026-10-21).
TX_POWER_DBM = 19.5
PL0_DB = 40.2  # free space at 1 m
EXPONENT_NEAR = 2.0
EXPONENT_FAR = 4.0
BREAKPOINT_M = 4 * 1.5 * 1.5 / (299_792_458.0 / 2.437e9)  # 4 h_t h_r / wavelength, 73.2 m
ORIENTATION_DB = 6.0
SIGMA_DB = 4.0
NOISE_FLOOR_DBM = -96.5
SENSITIVITY_DBM = -98.0  # 8 % PER for a 1024-byte frame at this RSSI
SENSITIVITY_PER = 0.08
SENSITIVITY_BYTES = 1024
FRAME_BYTES = 32 + 43  # payload + MAC overhead
BELOW_SENSITIVITY_MARGIN_DB = 10.0  # the MAC drops frames more than 10 dB under sensitivity
ATTEMPTS = 6  # 1 + 5 retries
REGISTERED_PER_SEED = "sim/runs/c-exp017-tolerance-r2/per_seed.csv"
REGISTERED_CELLS = "sim/runs/c-exp017-tolerance-r2/cells.csv"
CHECK_DISTANCE_M = 629.9605249474363  # the registered grid point near 630 m
DISTANCES_M = tuple(float(d) for d in np.geomspace(250.0, 1000.0, 16))


def mean_rssi_dbm(d: float) -> float:
    near = min(d, BREAKPOINT_M)
    far = max(d, BREAKPOINT_M)
    loss = PL0_DB + 10 * EXPONENT_NEAR * math.log10(near) + 10 * EXPONENT_FAR * math.log10(far / BREAKPOINT_M)
    return TX_POWER_DBM - loss - ORIENTATION_DB


def per_of_snr(snr_db: np.ndarray) -> np.ndarray:
    # Anchor: a 1024-byte frame has 8 % PER at the sensitivity SNR, so the bit error there is
    # ber_ref = 1 - 0.92^(1 / 8192).  DBPSK: BER = 0.5 exp(-a * snr_linear); solve a at the anchor.
    snr_ref = SENSITIVITY_DBM - NOISE_FLOOR_DBM
    ber_ref = 1.0 - (1.0 - SENSITIVITY_PER) ** (1.0 / (8 * SENSITIVITY_BYTES))
    a = math.log(0.5 / ber_ref) / 10 ** (snr_ref / 10)
    ber = np.minimum(0.5 * np.exp(-a * 10 ** (snr_db / 10)), 0.5)
    return 1.0 - (1.0 - ber) ** (8 * FRAME_BYTES)


def fade_grid(n: int = 160_001, zmax: float = 8.0):
    z = np.linspace(-zmax, zmax, n)
    w = np.exp(-0.5 * z * z)
    return z, w / w.sum()  # normal weights on the grid, summing to 1


def hand_cell(d: float) -> dict:
    z, w = fade_grid()
    rssi = mean_rssi_dbm(d) - SIGMA_DB * z  # a positive z is a deeper fade
    per = np.where(rssi < SENSITIVITY_DBM - BELOW_SENSITIVITY_MARGIN_DB, 1.0, per_of_snr(rssi - NOISE_FLOOR_DBM))
    p = float(np.sum(w * per))
    return {"distance_m": d, "mean_rssi_dbm": mean_rssi_dbm(d), "p_attempt_loss": p,
            "broadcast_delivery": 1.0 - p,
            "unicast_iid_delivery": 1.0 - p ** ATTEMPTS,
            "unicast_frozen_delivery": float(np.sum(w * (1.0 - per ** ATTEMPTS))),
            "crude_frozen_limit": 1.0 - p}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    phat = k / n
    den = 1 + z * z / n
    mid = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def registered_check(repo: Path) -> list[dict]:
    """The 630 m cell from the registered EXP-017 run, against the hand values."""
    with open(repo / REGISTERED_CELLS, newline="") as handle:
        cells = {int(r["cell"]): r for r in csv.DictReader(handle)}
    want = {c: r for c, r in cells.items() if r["scenario"] == "A" and r["area"] == "field"
            and r["processing_s"] == "0.0" and math.isclose(float(r["distance_m"]), CHECK_DISTANCE_M)}
    per_seed: dict[int, dict[int, tuple[int, int]]] = {c: {} for c in want}
    with open(repo / REGISTERED_PER_SEED, newline="") as handle:
        for r in csv.DictReader(handle):
            c = int(r["cell"])
            if c in want:
                per_seed[c][int(r["seed"])] = (int(r["received"]), int(r["sent"]))
    hand = hand_cell(CHECK_DISTANCE_M)
    out = []
    for c, r in sorted(want.items()):
        seeds = per_seed[c]
        k = sum(v[0] for v in seeds.values())
        n = sum(v[1] for v in seeds.values())
        frac = np.array([v[0] / v[1] for v in seeds.values()])
        half = 1.96 * float(frac.std(ddof=1)) / math.sqrt(len(frac))
        key = {("unicast", "iid"): "unicast_iid_delivery", ("unicast", "static"): "unicast_frozen_delivery",
               ("broadcast", "iid"): "broadcast_delivery", ("broadcast", "static"): "broadcast_delivery"}[
            (r["mode"], r["shadowing"])]
        lo, hi = wilson(k, n)
        valid = "pooled_wilson" if r["shadowing"] == "iid" else "seed_level"
        use_lo, use_hi = (lo, hi) if valid == "pooled_wilson" else (float(frac.mean()) - half, float(frac.mean()) + half)
        out.append({"mode": r["mode"], "shadowing": r["shadowing"], "n_seeds": len(seeds), "received": k, "sent": n,
                    "sim_delivery": k / n, "pooled_wilson": [lo, hi],
                    "seed_level_interval": [float(frac.mean()) - half, float(frac.mean()) + half],
                    "hand_key": key, "hand_value": hand[key], "interval_used": valid,
                    "hand_inside_interval": bool(use_lo <= hand[key] <= use_hi)})
    return out


def static_fades(seeds: list[int]) -> dict[int, float]:
    """The one fade (in standard deviations) each EXP-017 seed drew under static shadowing.

    This repeats only the seed-splitting rule (c_exp017_tolerance.radio_events, then
    RadioNetworkModel: SeedSequence(link_seed).spawn(2)[0] feeds the channel); the
    first normal draw is the pair's fade for the whole run.
    """
    out = {}
    for seed in seeds:
        link_seed = int(np.random.SeedSequence(seed).generate_state(2)[0])
        channel_seq = np.random.SeedSequence(link_seed).spawn(2)[0]
        out[seed] = float(np.random.default_rng(channel_seq).standard_normal(1024)[0])
    return out


def per_at(d: float, z: float) -> float:
    rssi = mean_rssi_dbm(d) - SIGMA_DB * z
    if rssi < SENSITIVITY_DBM - BELOW_SENSITIVITY_MARGIN_DB:
        return 1.0
    return float(per_of_snr(np.array(rssi - NOISE_FLOOR_DBM)))


def static_conditional_check(repo: Path) -> list[dict]:
    """Registered static cells against the hand value for the 50 fades actually drawn.

    With the fade known, each seed's 121 packets are independent tries: unicast
    delivers each with chance 1 - PER^6, broadcast with 1 - PER.  So the pooled
    count is a sum of binomials and its spread is known exactly.
    """
    with open(repo / REGISTERED_CELLS, newline="") as handle:
        cells = {int(r["cell"]): r for r in csv.DictReader(handle)
                 if r["scenario"] == "A" and r["area"] == "field" and r["processing_s"] == "0.0"
                 and r["shadowing"] == "static"}
    per_seed: dict[int, dict[int, tuple[int, int]]] = {c: {} for c in cells}
    with open(repo / REGISTERED_PER_SEED, newline="") as handle:
        for r in csv.DictReader(handle):
            c = int(r["cell"])
            if c in cells:
                per_seed[c][int(r["seed"])] = (int(r["received"]), int(r["sent"]))
    out = []
    for c, r in sorted(cells.items()):
        seeds = per_seed[c]
        fades = static_fades(sorted(seeds))
        d = float(r["distance_m"])
        q = {s: 1.0 - per_at(d, z) ** (ATTEMPTS if r["mode"] == "unicast" else 1) for s, z in fades.items()}
        k = sum(v[0] for v in seeds.values())
        n = sum(v[1] for v in seeds.values())
        expected = sum(q[s] * seeds[s][1] for s in seeds)
        var = sum(q[s] * (1 - q[s]) * seeds[s][1] for s in seeds)
        zscore = (k - expected) / math.sqrt(var) if var > 0 else (0.0 if k == expected else math.inf)
        out.append({"mode": r["mode"], "distance_m": d, "received": k, "sent": n, "sim_delivery": k / n,
                    "hand_given_drawn_fades": expected / n, "hand_unconditional": hand_cell(d)[
                        "unicast_frozen_delivery" if r["mode"] == "unicast" else "broadcast_delivery"],
                    "z_score": zscore, "mean_drawn_fade_sd": float(np.mean(list(fades.values())))})
    return out


def check_inputs() -> None:
    from sim.radio.params import PARAMS
    assert PARAMS["tx_power_dbm"].value == TX_POWER_DBM
    assert PARAMS["pl0_1m_db"].value == PL0_DB
    assert PARAMS["sigma_outdoor_open_db"].high == SIGMA_DB
    assert PARAMS["orientation_loss_db"].high == ORIENTATION_DB
    assert PARAMS["noise_floor_dbm"].value == NOISE_FLOOR_DBM
    assert PARAMS["sensitivity_1mbps_dbm"].value == SENSITIVITY_DBM
    assert PARAMS["sensitivity_per"].value == SENSITIVITY_PER
    assert PARAMS["exponent_outdoor_open"].value == EXPONENT_NEAR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check_inputs()
    repo = Path(__file__).resolve().parents[1]
    result = {"experiment": "EXP-038 (proposal) hand check", "status": "hand calculation; simulated comparison",
              "attempts": ATTEMPTS, "sigma_db": SIGMA_DB, "frame_bytes": FRAME_BYTES,
              "cells": [hand_cell(d) for d in DISTANCES_M], "registered_630m_check": registered_check(repo),
              "registered_static_given_fades": static_conditional_check(repo)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hand.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{'d (m)':>7} {'p':>7} {'bcast 1-p':>9} {'iid 1-p^6':>9} {'frozen':>7}")
    for c in result["cells"]:
        print(f"{c['distance_m']:7.1f} {c['p_attempt_loss']:7.4f} {c['broadcast_delivery']:9.4f} "
              f"{c['unicast_iid_delivery']:9.4f} {c['unicast_frozen_delivery']:7.4f}")
    for r in result["registered_630m_check"]:
        print(f"630 m {r['mode']:9} {r['shadowing']:6} sim {r['sim_delivery']:.4f} hand {r['hand_value']:.4f} "
              f"inside {r['interval_used']}: {r['hand_inside_interval']}")
    for r in result["registered_static_given_fades"]:
        print(f"static {r['mode']:9} {r['distance_m']:7.1f} sim {r['sim_delivery']:.4f} "
              f"hand(drawn fades) {r['hand_given_drawn_fades']:.4f} z {r['z_score']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
