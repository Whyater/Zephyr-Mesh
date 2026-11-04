"""ESP-NOW airtime budget for the 161-byte drone state packet (sim/radio/packet.py).

Question: how much of one 2.4 GHz channel does the state message use, for a given
number of drones and send rate, and how often do two frames land on the air at the
same time?

Everything here is worked out from frame sizes and 802.11 timing constants, written
out below with a source for each one.  It does not import the airtime functions in
sim/radio/mac.py; those are used only as a cross-check (``crosscheck_sim_radio``).
The collision estimates come in two kinds:

- ``aloha``: a closed-form pure-ALOHA estimate.  Each drone sends once per period at
  a random point in the period, nobody listens before talking, and a frame is lost if
  any other frame overlaps it.  This is a pessimistic bound, because ESP-NOW senders
  do carrier sense.
- ``dcf_sim``: the existing event-driven 802.11 DCF engine in sim/radio/mac.py
  (carrier sense, DIFS, random backoff), run on a flat channel where every node hears
  every other node and no frame is lost to noise.  Only overlap losses remain.

No number here is measured on Zephyr hardware.

Run:  PYTHONPATH=. python tools/espnow_airtime_budget.py [--sim-seconds 20] [--out budget.json]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

from sim.radio.packet import STATE_PACKET_BYTES

ESPNOW_PAGE = "https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html"
IDF_RATES = ("https://github.com/espressif/esp-idf/blob/master/components/esp_wifi/include/"
             "esp_wifi_types_generic.h (wifi_phy_rate_t)")
LINUX = "https://raw.githubusercontent.com/torvalds/linux/master/net/mac80211/util.c (ieee80211_frame_duration)"
FREEBSD_H = "https://cgit.freebsd.org/src/plain/sys/net80211/ieee80211_phy.h"
FREEBSD_C = "https://cgit.freebsd.org/src/plain/sys/net80211/ieee80211_phy.c (ieee80211_compute_duration)"
IEEE = "IEEE 802.11-2020 (text not opened; value as implemented in the Linux and FreeBSD code cited)"
CHECKED = "2026-11-04"


@dataclass(frozen=True)
class Constant:
    value: float
    units: str
    source: str
    status: str  # "CONFIRMED", "standard", "modelling-choice", or "UNCONFIRMED"
    note: str = ""


C: dict[str, Constant] = {
    # ---- ESP-NOW frame (Espressif ESP-NOW guide, "Frame Format") ----
    "mac_header_bytes": Constant(24, "bytes", ESPNOW_PAGE, "CONFIRMED", "802.11 management header"),
    "category_bytes": Constant(1, "bytes", ESPNOW_PAGE, "CONFIRMED", "category code 127, vendor specific"),
    "org_id_bytes": Constant(3, "bytes", ESPNOW_PAGE, "CONFIRMED", "0x18fe34"),
    "random_bytes": Constant(4, "bytes", ESPNOW_PAGE, "CONFIRMED", "anti-replay random value"),
    "vendor_element_header_bytes": Constant(7, "bytes", ESPNOW_PAGE, "CONFIRMED",
                                            "element id 1 + length 1 + OUI 3 + type 1 + version 1"),
    "fcs_bytes": Constant(4, "bytes", ESPNOW_PAGE, "CONFIRMED"),
    "default_rate_mbps": Constant(1.0, "Mbit/s", ESPNOW_PAGE, "CONFIRMED", "'The default ESP-NOW bit rate is 1 Mbps.'"),
    # ---- DSSS / CCK PHY (802.11b) ----
    "dsss_long_preamble_us": Constant(144 + 48, "us", LINUX + "; " + FREEBSD_C, "standard",
                                      "144 us SYNC+SFD + 48 us PLCP header, both sent at 1 Mbit/s"),
    "dsss_short_preamble_us": Constant(72 + 24, "us", LINUX + "; " + FREEBSD_C, "standard",
                                       "2, 5.5 and 11 Mbit/s only; ESP-IDF exposes it as *_S rates"),
    "dsss_slot_us": Constant(20, "us", FREEBSD_H, "standard", "DS/CCK slot, ERP long slot"),
    "sifs_24ghz_us": Constant(10, "us", FREEBSD_H + "; " + LINUX, "standard"),
    "dsss_cw_min": Constant(31, "slots", FREEBSD_H, "standard", "DS/CCK aCWmin"),
    # ---- OFDM PHY in 2.4 GHz (802.11g ERP-OFDM) ----
    "ofdm_preamble_signal_us": Constant(20, "us", FREEBSD_C + "; " + LINUX, "standard",
                                        "16 us training + 4 us SIGNAL field"),
    "ofdm_symbol_us": Constant(4, "us", FREEBSD_C, "standard"),
    "ofdm_service_tail_bits": Constant(22, "bits", FREEBSD_C, "standard", "16 SERVICE + 6 tail"),
    "erp_signal_extension_us": Constant(6, "us", LINUX, "standard",
                                        "ERP-OFDM idle tail; Linux counts 'SIFS + signal ext' as 16 us"),
    "ofdm_cw_min": Constant(15, "slots", FREEBSD_H, "standard",
                            "ERP aCWmin(1); which value ESP32 uses for ESP-NOW is UNCONFIRMED"),
    "ofdm_slot_us": Constant(20, "us", FREEBSD_H, "modelling-choice",
                             "ERP long slot. A 9 us short slot needs every station to support it; "
                             "what ESP32 uses for ESP-NOW is UNCONFIRMED"),
    # ---- ACK ----
    "ack_bytes": Constant(14, "bytes", IEEE, "standard", "frame control 2 + duration 2 + RA 6 + FCS 4"),
    "ack_rate_rule": Constant(0, "-", IEEE, "UNCONFIRMED",
                              "802.11 sends an ACK at the highest basic rate not above the data rate. ESP-NOW "
                              "has no access point, so its basic rate set is UNCONFIRMED; this tool assumes the "
                              "mandatory sets {1, 2, 5.5, 11} (DSSS/CCK) and {6, 12, 24} (OFDM)"),
    "broadcast_acked": Constant(0, "bool", IEEE, "standard",
                                "802.11 group-addressed frames get no ACK and no retry; Espressif's page is silent"),
}

MANDATORY_DSSS = (1.0, 2.0, 5.5, 11.0)
MANDATORY_OFDM = (6.0, 12.0, 24.0)


def espnow_overhead_bytes() -> int:
    """Bytes ESP-NOW adds around the payload, summed from the frame format table."""
    parts = ("mac_header_bytes", "category_bytes", "org_id_bytes", "random_bytes",
             "vendor_element_header_bytes", "fcs_bytes")
    return int(sum(C[p].value for p in parts))


@dataclass(frozen=True)
class PhyMode:
    name: str
    modulation: str  # "dsss" or "ofdm"
    rate_mbps: float
    short_preamble: bool = False

    def __post_init__(self) -> None:
        if self.modulation not in ("dsss", "ofdm"):
            raise ValueError("modulation must be dsss or ofdm")
        if self.modulation == "dsss" and self.rate_mbps == 1.0 and self.short_preamble:
            raise ValueError("1 Mbit/s DSSS always uses the long preamble")

    @property
    def ack_rate_mbps(self) -> float:
        pool = MANDATORY_DSSS if self.modulation == "dsss" else MANDATORY_OFDM
        below = [r for r in pool if r <= self.rate_mbps]
        return max(below) if below else min(pool)

    @property
    def slot_us(self) -> float:
        return C["dsss_slot_us"].value if self.modulation == "dsss" else C["ofdm_slot_us"].value

    @property
    def cw_min(self) -> int:
        return int(C["dsss_cw_min"].value if self.modulation == "dsss" else C["ofdm_cw_min"].value)


# The ESP-NOW default plus a few rates that ESP-IDF's wifi_phy_rate_t offers
# (WIFI_PHY_RATE_1M_L, _2M_S, _11M_S, _6M, _24M, _54M).  Long range (LORA_250K and
# LORA_500K) is not modelled: its frame timing is not published (UNCONFIRMED).
MODES = (
    PhyMode("1M_L (default)", "dsss", 1.0, False),
    PhyMode("2M_S", "dsss", 2.0, True),
    PhyMode("11M_S", "dsss", 11.0, True),
    PhyMode("6M OFDM", "ofdm", 6.0),
    PhyMode("24M OFDM", "ofdm", 24.0),
    PhyMode("54M OFDM", "ofdm", 54.0),
)
DEFAULT_MODE = MODES[0]

DRONES = (2, 3, 5, 8)
RATES_HZ = (10.0, 25.0, 50.0)
CASTS = ("broadcast", "unicast_gs")


def ppdu_us(n_bytes: int, rate_mbps: float, modulation: str, short_preamble: bool) -> float:
    """Time one PHY frame of ``n_bytes`` (MAC header to FCS) holds the air, in us."""
    if int(n_bytes) != n_bytes or n_bytes < 1:
        raise ValueError("n_bytes must be a positive integer")
    if modulation == "dsss":
        pre = C["dsss_short_preamble_us"].value if short_preamble else C["dsss_long_preamble_us"].value
        # 802.11b TXTIME rounds the payload time up to a whole microsecond (Ceiling in
        # the standard's formula, as quoted in the Linux code cited for this constant)
        return pre + math.ceil(8 * n_bytes / rate_mbps)
    bits_per_symbol = rate_mbps * C["ofdm_symbol_us"].value
    symbols = math.ceil((C["ofdm_service_tail_bits"].value + 8 * n_bytes) / bits_per_symbol)
    return (C["ofdm_preamble_signal_us"].value + symbols * C["ofdm_symbol_us"].value
            + C["erp_signal_extension_us"].value)


def data_us(payload_bytes: int, mode: PhyMode) -> float:
    return ppdu_us(payload_bytes + espnow_overhead_bytes(), mode.rate_mbps, mode.modulation, mode.short_preamble)


def ack_us(mode: PhyMode) -> float:
    rate = mode.ack_rate_mbps
    short = mode.short_preamble and mode.modulation == "dsss" and rate > 1.0
    return ppdu_us(int(C["ack_bytes"].value), rate, mode.modulation, short)


def difs_us(mode: PhyMode) -> float:
    return C["sifs_24ghz_us"].value + 2 * mode.slot_us


@dataclass(frozen=True)
class Exchange:
    payload_bytes: int
    mpdu_bytes: int
    data_us: float
    sifs_ack_us: float      # 0 for broadcast
    busy_us: float          # time the medium is held: data (+ SIFS + ACK for unicast)
    access_us: float        # DIFS + mean backoff (CWmin / 2 slots) spent before the frame


def exchange(payload_bytes: int, mode: PhyMode, cast: str) -> Exchange:
    if cast not in CASTS:
        raise ValueError(f"cast must be one of {CASTS}")
    d = data_us(payload_bytes, mode)
    tail = (C["sifs_24ghz_us"].value + ack_us(mode)) if cast == "unicast_gs" else 0.0
    access = difs_us(mode) + mode.cw_min / 2.0 * mode.slot_us
    return Exchange(payload_bytes, payload_bytes + espnow_overhead_bytes(), d, tail, d + tail, access)


def aloha_collision(n_senders: int, rate_hz: float, busy_us: float) -> dict[str, float]:
    """Chance a frame overlaps another sender's frame, nobody listening first.

    Two equal-length exchanges of D seconds overlap if their starts are within D of
    each other: a window 2D wide.  One other drone sending once per period P at a
    random time lands in that window with probability 2D / P = 2 D f.  With n - 1
    independent other drones the frame survives with (1 - 2 D f) ** (n - 1).
    ``poisson`` is the textbook pure-ALOHA form exp(-2 G) for the same load.
    """
    if n_senders < 1 or rate_hz <= 0:
        raise ValueError("need at least one sender and a positive rate")
    d = busy_us * 1e-6
    p_one = min(1.0, 2.0 * d * rate_hz)
    periodic = 1.0 - (1.0 - p_one) ** (n_senders - 1)
    poisson = 1.0 - math.exp(-2.0 * (n_senders - 1) * rate_hz * d)
    return {"periodic": periodic, "poisson": poisson}


def cell(n_drones: int, rate_hz: float, cast: str, mode: PhyMode = DEFAULT_MODE,
         payload_bytes: int = STATE_PACKET_BYTES) -> dict:
    ex = exchange(payload_bytes, mode, cast)
    frames_per_s = n_drones * rate_hz  # broadcast: one frame reaches every peer; unicast: one frame to the gs
    busy = frames_per_s * ex.busy_us * 1e-6
    with_access = frames_per_s * (ex.busy_us + ex.access_us) * 1e-6
    al = aloha_collision(n_drones, rate_hz, ex.busy_us)
    return {"drones": n_drones, "rate_hz": rate_hz, "cast": cast, "mode": mode.name,
            "frames_per_s": frames_per_s, "busy_us_per_frame": ex.busy_us,
            "utilization": busy, "utilization_with_access": with_access,
            "aloha_collision": al["periodic"], "aloha_collision_poisson": al["poisson"]}


# ---------------------------------------------------------------------------
# DCF cross-check with the existing engine (sim/radio/mac.py)
# ---------------------------------------------------------------------------

class _FlatChannel:
    """Every node hears every other node at -40 dBm and no frame is lost to noise."""

    sensitivity_dbm = -98.0

    def __init__(self):
        self.cfg = SimpleNamespace(noise_floor_dbm=-95.0, sensitivity_dbm=-98.0)
        self._good = SimpleNamespace(rssi_dbm=-40.0, snr_db=55.0, per=0.0, shadow_db=0.0, path_loss_db=50.0)

    def sample(self, tx, rx, distance_m, t_s, *, frame_bytes=0, **kwargs):
        return self._good

    def received_power_dbm(self, tx, rx, distance_m, t_s, **kwargs):
        return -40.0


def _engine_phy(mode: PhyMode):
    from sim.radio.mac import PhyConfig

    return PhyConfig(rate_mbps=mode.rate_mbps, modulation=mode.modulation,
                     preamble="short" if mode.short_preamble else "long", slot_s=mode.slot_us * 1e-6,
                     sifs_s=C["sifs_24ghz_us"].value * 1e-6, cw_min=mode.cw_min, cw_max=1023,
                     mac_overhead_bytes=espnow_overhead_bytes(), ack_bytes=int(C["ack_bytes"].value),
                     ack_rate_mbps=mode.ack_rate_mbps, max_payload_bytes=250)


def simulate_collision(n_drones: int, rate_hz: float, cast: str, *, carrier_sense: bool, sim_seconds: float,
                       seed: int = 37, mode: PhyMode = DEFAULT_MODE,
                       payload_bytes: int = STATE_PACKET_BYTES) -> dict:
    """Fraction of state frames lost to overlap in the sim/radio MAC engine.

    Each drone sends once per period at a uniformly random time inside that period
    (``jitter_s`` = one period), so relative timing between drones keeps changing, as
    it would with free-running clocks.  Unicast uses no retries, so the number is the
    chance one attempt fails.  Frames from the first and last 0.5 s are not scored.
    ``wait_*`` is the time from a frame being handed to the radio to its first
    transmission starting (DIFS, deferral behind other frames, backoff, any queue).
    """
    from sim.radio.mac import BROADCAST, MacConfig, MacSimulator

    phy = _engine_phy(mode)
    cfg = MacConfig(carrier_sense=carrier_sense, backoff=carrier_sense, unicast_retry_limit=0,
                    queue_capacity=64)
    names = [f"d{i}" for i in range(n_drones)]
    pos = {name: (float(i), 0.0, 1.0) for i, name in enumerate(names)}
    pos["gs"] = (0.0, 2.0, 1.0)
    sim = MacSimulator(pos, _FlatChannel(), phy, cfg, seed=seed)
    period = 1.0 / rate_hz
    dst = BROADCAST if cast == "broadcast" else "gs"
    for name in names:
        sim.add_periodic_source(name, dst, rate_hz, payload_bytes, start_s=0.0, stop_s=sim_seconds,
                                phase_s=0.0, jitter_s=period)
    sim.run_until(sim_seconds + 1.0)
    lo, hi = 0.5, sim_seconds - 0.5
    collided_ids = {r.frame_id for r in sim.rx_records if r.outcome == "collision"}
    scored = lost = dropped = 0
    waits = []
    for fid, rec in sim.frame_records.items():
        if not (lo <= rec.created_s < hi):
            continue
        scored += 1
        if rec.first_tx_s is not None:
            waits.append(rec.first_tx_s - rec.created_s)
        if rec.status == "queue_drop":
            dropped += 1
            lost += 1
        elif cast == "broadcast":
            lost += fid in collided_ids
        else:
            lost += rec.status != "acked"
    busy = sim.channel_busy_fraction(lo, hi)
    waits.sort()
    p95 = waits[min(len(waits) - 1, int(0.95 * len(waits)))] if waits else float("nan")
    return {"frames": scored, "lost": lost, "queue_drops": dropped,
            "loss_fraction": lost / scored if scored else float("nan"), "busy_fraction": busy,
            "wait_mean_ms": 1e3 * sum(waits) / len(waits) if waits else float("nan"), "wait_p95_ms": 1e3 * p95}


def crosscheck_sim_radio(payload_bytes: int = STATE_PACKET_BYTES) -> list[dict]:
    """Compare this file's frame times with sim/radio/mac.py's airtime functions."""
    from sim.radio.mac import ack_airtime_s, data_airtime_s

    rows = []
    for mode in MODES:
        phy = _engine_phy(mode)
        rows.append({"mode": mode.name, "data_us_here": data_us(payload_bytes, mode),
                     "data_us_sim_radio": data_airtime_s(payload_bytes, phy) * 1e6,
                     "ack_us_here": ack_us(mode), "ack_us_sim_radio": ack_airtime_s(phy) * 1e6})
    return rows


def build(sim_seconds: float = 0.0, seed: int = 37) -> dict:
    per_mode = []
    for mode in MODES:
        b = exchange(STATE_PACKET_BYTES, mode, "broadcast")
        u = exchange(STATE_PACKET_BYTES, mode, "unicast_gs")
        per_mode.append({"mode": mode.name, "ack_rate_mbps": mode.ack_rate_mbps, "mpdu_bytes": b.mpdu_bytes,
                         "data_us": b.data_us, "sifs_ack_us": u.sifs_ack_us, "access_us": b.access_us,
                         "max_frames_per_s_broadcast": 1e6 / (b.busy_us + b.access_us),
                         "max_frames_per_s_unicast": 1e6 / (u.busy_us + u.access_us)})
    table = []
    for cast in CASTS:
        for n in DRONES:
            for f in RATES_HZ:
                row = cell(n, f, cast)
                if sim_seconds > 0:
                    a = simulate_collision(n, f, cast, carrier_sense=False, sim_seconds=sim_seconds, seed=seed)
                    d = simulate_collision(n, f, cast, carrier_sense=True, sim_seconds=sim_seconds, seed=seed)
                    row.update(aloha_sim=a["loss_fraction"], dcf_sim=d["loss_fraction"],
                               dcf_queue_drops=d["queue_drops"], dcf_frames=d["frames"],
                               dcf_busy_fraction=d["busy_fraction"], dcf_wait_mean_ms=d["wait_mean_ms"],
                               dcf_wait_p95_ms=d["wait_p95_ms"])
                table.append(row)
    fast = {}
    for mode in MODES[1:]:
        r = cell(8, 50.0, "broadcast", mode)
        fast[mode.name] = {"utilization": r["utilization"], "aloha_collision": r["aloha_collision"]}
    return {"payload_bytes": STATE_PACKET_BYTES, "espnow_overhead_bytes": espnow_overhead_bytes(),
            "checked": CHECKED, "sim_seconds": sim_seconds, "seed": seed,
            "constants": {k: asdict(v) for k, v in C.items()}, "per_mode": per_mode,
            "table_1M": table, "eight_at_50hz_broadcast_other_rates": fast,
            "crosscheck_sim_radio": crosscheck_sim_radio()}


def markdown(result: dict) -> str:
    lines = ["| Mode | Drones | Rate (Hz) | Frames/s | Busy (%) | Busy + access (%) | ALOHA collision (%) |"
             + (" ALOHA sim (%) | DCF sim (%) | DCF wait mean / p95 (ms) |" if result["sim_seconds"] > 0 else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|---|" if result["sim_seconds"] > 0 else "")]
    for r in result["table_1M"]:
        line = (f"| {r['cast']} | {r['drones']} | {r['rate_hz']:.0f} | {r['frames_per_s']:.0f} | "
                f"{100 * r['utilization']:.1f} | {100 * r['utilization_with_access']:.1f} | "
                f"{100 * r['aloha_collision']:.1f} |")
        if "dcf_sim" in r:
            line += (f" {100 * r['aloha_sim']:.1f} | {100 * r['dcf_sim']:.1f} | "
                     f"{r['dcf_wait_mean_ms']:.2f} / {r['dcf_wait_p95_ms']:.2f} |")
        lines.append(line)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim-seconds", type=float, default=0.0,
                    help="also run the sim/radio MAC engine for this many simulated seconds per cell")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", type=Path, help="write the full result as JSON")
    args = ap.parse_args(argv)
    if args.sim_seconds and args.sim_seconds <= 1.0:
        ap.error("--sim-seconds must be more than 1 (0.5 s is trimmed at each end)")
    result = build(args.sim_seconds, args.seed)
    print(f"state payload {result['payload_bytes']} B + ESP-NOW overhead {result['espnow_overhead_bytes']} B")
    for m in result["per_mode"]:
        print(f"  {m['mode']:>15}: data {m['data_us']:7.1f} us, SIFS+ACK {m['sifs_ack_us']:6.1f} us "
              f"(ACK at {m['ack_rate_mbps']:g} Mbit/s), DIFS+mean backoff {m['access_us']:5.0f} us, "
              f"one sender max {m['max_frames_per_s_broadcast']:6.0f} bcast / {m['max_frames_per_s_unicast']:6.0f} "
              "unicast frames/s")
    print("\n1 Mbit/s long preamble (ESP-NOW default):")
    print(markdown(result))
    print("\n8 drones at 50 Hz, broadcast, other rates:")
    for name, v in result["eight_at_50hz_broadcast_other_rates"].items():
        print(f"  {name:>9}: busy {100 * v['utilization']:5.1f} %, ALOHA collision {100 * v['aloha_collision']:5.1f} %")
    print("\nCross-check against sim/radio/mac.py (us):")
    for r in result["crosscheck_sim_radio"]:
        print(f"  {r['mode']:>15}: data {r['data_us_here']:7.1f} here vs {r['data_us_sim_radio']:7.1f}, "
              f"ACK {r['ack_us_here']:6.1f} here vs {r['ack_us_sim_radio']:6.1f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
