"""EXP-027 airtime check: does "ground sent more bytes" survive as "ground used more air"?

Reads the registered per-kind traffic from runs/d7-decisions/summary.json (read only),
turns each message kind into radio frames, and prices the frames with the existing
sim/radio airtime functions (frame_bytes, data_airtime_s, ack_airtime_s, PhyConfig.difs_s).
The timing model is not reimplemented here, except one independent hand formula that is
used only as a cross-check.

Two readings of "one message" are computed because the coop simulator counts per directed
route (sim/coop/simulator.py:333-351), while a real ESP-NOW radio could send one broadcast
frame to everyone:

  A_per_route_unicast  every counted message is one unicast frame (what the simulator counts).
  B_broadcast_shares   a state share and a ground track are each ONE broadcast frame (no ACK),
                       raw reports and ground commands stay unicast (commands differ per drone).

Run:  PYTHONPATH=. python tools/burn_rund_exp027_airtime.py --out <dir>/airtime.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sim.radio.mac import ack_airtime_s, data_airtime_s, frame_bytes
from sim.radio.params import PARAMS, default_phy

KINDS = ("state", "raw", "track", "command", "leaving")
DURATION_S = 20.0
N_RECEIVERS_PER_SHARE = 4  # 3 peers + gs for state; 4 agents for track (simulator.py:545-547, 553-556)

# Interpretation table: kind -> (cast, receivers collapsed into one frame)
INTERPRETATIONS = {
    "A_per_route_unicast": {k: ("unicast", 1) for k in KINDS},
    "B_broadcast_shares": {
        "state": ("broadcast", N_RECEIVERS_PER_SHARE),
        "raw": ("unicast", 1),
        "track": ("broadcast", N_RECEIVERS_PER_SHARE),
        "command": ("unicast", 1),
        "leaving": ("unicast", 1),
    },
}

# PHY settings that sim/radio/params.py documents as valid (PARAMS ranges and notes).
PHYS = {
    "dsss_1M_long (default)": {},
    "dsss_2M_short": {"rate_mbps": 2.0, "preamble": "short", "ack_rate_mbps": 2.0},
    "dsss_5.5M_short": {"rate_mbps": 5.5, "preamble": "short", "ack_rate_mbps": 2.0},
    "dsss_11M_short": {"rate_mbps": 11.0, "preamble": "short", "ack_rate_mbps": 2.0},
    "dsss_11M_long": {"rate_mbps": 11.0, "preamble": "long", "ack_rate_mbps": 2.0},
    "ofdm_54M": {"rate_mbps": 54.0, "modulation": "ofdm", "ack_rate_mbps": 24.0, "slot_s": 9e-6, "cw_min": 15},
}

# Screening threshold chosen for this check only; it has no source in the project.
CHANNEL_SHARE_FLAG = 0.3


def kind_rows(cell: dict, sizes: dict, scheme: dict, phy) -> list[dict]:
    rows = []
    for kind in KINDS:
        size = int(sizes[kind])
        bytes_ps = float(cell[f"{kind}_bytes_per_second"]["median"])
        msgs_ps = bytes_ps / size
        cast, group = scheme[kind]
        frames_ps = msgs_ps / group
        unicast = cast == "unicast"
        data = data_airtime_s(size, phy) if frames_ps > 0 else 0.0
        tail = (phy.sifs_s + ack_airtime_s(phy)) if unicast else 0.0
        access = phy.difs_s + (phy.cw_min / 2.0) * phy.slot_s
        rows.append({
            "kind": kind, "cast": cast, "payload_bytes": size, "frame_bytes": frame_bytes(size, phy),
            "messages_per_s": msgs_ps, "frames_per_s": frames_ps, "payload_bytes_per_s": bytes_ps,
            "data_airtime_us": data * 1e6, "sifs_plus_ack_us": tail * 1e6, "difs_plus_mean_backoff_us": access * 1e6,
            "air_bare_s_per_s": frames_ps * data,
            "air_with_ack_s_per_s": frames_ps * (data + tail),
            "air_with_ack_difs_backoff_s_per_s": frames_ps * (data + tail + access),
        })
    return rows


def totals(rows: list[dict]) -> dict:
    return {
        "payload_bytes_per_s": sum(r["payload_bytes_per_s"] for r in rows),
        "frames_per_s": sum(r["frames_per_s"] for r in rows),
        "wire_bytes_per_s": sum(r["frames_per_s"] * r["frame_bytes"] for r in rows if r["frames_per_s"] > 0),
        "air_bare": sum(r["air_bare_s_per_s"] for r in rows),
        "air_with_ack": sum(r["air_with_ack_s_per_s"] for r in rows),
        "air_with_ack_difs_backoff": sum(r["air_with_ack_difs_backoff_s_per_s"] for r in rows),
    }


def find_cell(cells: list[dict], mode: str, loss: float, outage: str) -> dict:
    for c in cells:
        if c["mode"] == mode and c["loss"] == loss and c["outage"] == outage:
            return c
    raise KeyError((mode, loss, outage))


def ratios(g: dict, o: dict) -> dict:
    out = {}
    for key in ("payload_bytes_per_s", "frames_per_s", "wire_bytes_per_s", "air_bare", "air_with_ack",
                "air_with_ack_difs_backoff"):
        out[key] = g[key] / o[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="runs/d7-decisions/summary.json")
    ap.add_argument("--out", required=True, help="output JSON path, outside runs/")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    cells = summary["cells"]
    sizes = {k: v["message_bytes"] for k, v in summary["traffic"].items()}

    # Step 1 check: size x rate reproduces the registered byte totals.
    reproduce = []
    for c in cells:
        total_msgs = sum(float(c[f"{k}_bytes_per_second"]["median"]) / sizes[k] for k in KINDS)
        total_bytes = sum(float(c[f"{k}_bytes_per_second"]["median"]) for k in KINDS)
        reproduce.append({
            "mode": c["mode"], "loss": c["loss"], "outage": c["outage"],
            "sum_kind_bytes": total_bytes, "registered_bytes_per_second": c["bytes_per_second"]["median"],
            "sum_kind_messages": total_msgs, "registered_messages_per_second": c["messages_per_second"]["median"],
            "bytes_match": math.isclose(total_bytes, c["bytes_per_second"]["median"], rel_tol=0, abs_tol=1e-6),
            "messages_match": math.isclose(total_msgs, c["messages_per_second"]["median"], rel_tol=0, abs_tol=1e-6),
        })
    assert all(r["bytes_match"] and r["messages_match"] for r in reproduce), reproduce

    # Independent hand formula for the state frame at the default PHY (802.11b, 1 Mbps, long preamble).
    hand_state_s = 192e-6 + (240 + 43) * 8 / 1e6
    phy0 = default_phy()
    lib_state_s = data_airtime_s(240, phy0)
    hand_check = {
        "formula": "192 us + (240 payload + 43 overhead) bytes * 8 bits / 1e6 bit/s",
        "hand_us": hand_state_s * 1e6, "library_us": lib_state_s * 1e6,
        "match": math.isclose(hand_state_s, lib_state_s, rel_tol=0, abs_tol=1e-12),
    }
    assert hand_check["match"], hand_check

    results = {}
    for phy_name, overrides in PHYS.items():
        phy = default_phy(**overrides)
        phy_block = {
            "phy": {"rate_mbps": phy.rate_mbps, "modulation": phy.modulation, "preamble": phy.preamble,
                    "slot_us": phy.slot_s * 1e6, "sifs_us": phy.sifs_s * 1e6, "difs_us": phy.difs_s * 1e6,
                    "cw_min": phy.cw_min, "mean_backoff_slots": phy.cw_min / 2.0,
                    "ack_rate_mbps": phy.ack_rate_mbps, "ack_airtime_us": ack_airtime_s(phy) * 1e6},
            "interpretations": {},
        }
        for interp, scheme in INTERPRETATIONS.items():
            per_cell = []
            for loss in (0.0, 0.1, 0.3):
                for outage in ("none", "10_15"):
                    g_rows = kind_rows(find_cell(cells, "ground", loss, outage), sizes, scheme, phy)
                    o_rows = kind_rows(find_cell(cells, "onboard", loss, outage), sizes, scheme, phy)
                    gt, ot = totals(g_rows), totals(o_rows)
                    per_cell.append({
                        "loss": loss, "outage": outage,
                        "ground": {"kinds": g_rows, "totals": gt},
                        "onboard": {"kinds": o_rows, "totals": ot},
                        "ground_over_onboard": ratios(gt, ot),
                        "flag_airtime_over_0p3": {
                            arm: {k: t[k] > CHANNEL_SHARE_FLAG for k in ("air_bare", "air_with_ack", "air_with_ack_difs_backoff")}
                            for arm, t in (("ground", gt), ("onboard", ot))
                        },
                    })
            phy_block["interpretations"][interp] = per_cell
        results[phy_name] = phy_block

    out = {
        "schema": "burn-rund-exp027-airtime-1",
        "note": "Computed from registered per-kind traffic; simulation not rerun. Timing from sim/radio/mac.py and params.py.",
        "prediction_on_record": "ground/onboard airtime ratio stays above 1, between 1.1 and 1.6",
        "source_summary": args.summary,
        "message_bytes": sizes,
        "n_receivers_per_share": N_RECEIVERS_PER_SHARE,
        "shared_channel": ("MacSimulator (sim/coop/net.py:_radio_table) puts all nodes plus gs on ONE channel; the d7 coop "
                           "simulator itself uses independent abstract SimulatedLink per directed route, so no contention "
                           "is modelled in EXP-027 itself."),
        "params_used": {k: PARAMS[k].value for k in ("espnow_overhead_bytes", "slot_s", "sifs_s", "cw_min", "ack_bytes")},
        "reproduce_registered_bytes": reproduce,
        "hand_check_state_frame": hand_check,
        "results": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")

    # Short console summary for the primary cell.
    for phy_name in PHYS:
        for interp in INTERPRETATIONS:
            cell = results[phy_name]["interpretations"][interp][0]
            g, o, r = cell["ground"]["totals"], cell["onboard"]["totals"], cell["ground_over_onboard"]
            print(f"{phy_name:26s} {interp:20s} air(bare) g={g['air_bare']:.3f} o={o['air_bare']:.3f} r={r['air_bare']:.3f} | "
                  f"air(full) g={g['air_with_ack_difs_backoff']:.3f} o={o['air_with_ack_difs_backoff']:.3f} "
                  f"r={r['air_with_ack_difs_backoff']:.3f} | bytes r={r['payload_bytes_per_s']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
