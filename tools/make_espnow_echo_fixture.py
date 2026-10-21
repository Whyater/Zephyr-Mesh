"""Make a synthetic ESP-NOW echo log pair (sender.csv, echoer.csv) from known settings.

This generator is written from the physical story in docs/ESPNOW_ECHO_LOG.md
with plain NumPy. It deliberately does not call sim.link.SimulatedLink or
tools/fit_espnow_log.py, so a fit that recovers these settings is checked
against something written separately, not against itself.

The story, one packet at a time (all times in true seconds first):

1. Board A sends packet i at ``t_tx = i / rate_hz``.
2. The outbound leg (A to B) either loses it or delivers it after a one-way
   delay ``d_out``. Loss on this leg follows the same burst rule as the
   simulator's link: when no burst is running, a packet starts a burst with
   probability ``burst_start``; every packet inside a burst is lost; after
   each lost packet the burst ends with probability ``burst_end``.
3. Board B holds the packet for ``h`` seconds (uniform between
   ``hold_min_s`` and ``hold_max_s``) and sends the echo.
4. The return leg (B to A) loses the echo with probability ``return_loss``
   (independently per packet) or delivers it after ``d_back``.
5. Optionally B sees a second copy of the packet (a duplicate arrival) a
   little later; it logs that row and does not echo it.

Each one-way delay is a lognormal with the stated mean and standard deviation
(or a constant when the deviation is zero). The outbound and return legs have
their own settings, so a lopsided link can be made on purpose.

Clocks: each board counts whole microseconds from its own boot, so a stamp is
``floor(offset_us + true_us * (1 + drift_ppm * 1e-6))``. Rounding down to a
whole microsecond adds an error between 0 and 1 us to every stamp. The round
trip uses four stamps (two differences), so its rounding error is the sum of
two triangle-shaped errors on (-1, 1) us: mean zero, standard deviation
sqrt(1/3) us (about 0.58 us), and never more than 2 us in size.

Every random quantity comes from its own stream spawned from one seed, so
changing (for example) the hold time cannot change which packets are lost or
how long any leg took.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_LINE = "# zephyr-espnow-echo-log-1"
SENDER_HEADER = "seq,tx_us,echo_us"
ECHOER_HEADER = "seq,rx_us,echo_tx_us,rssi_dbm"


@dataclass(frozen=True)
class EchoParams:
    """Every setting the synthetic echo test is made from.

    Delays are in seconds. ``*_delay_std_s`` is the standard deviation of one
    leg's one-way delay (zero means every packet takes exactly the mean).
    Drift is the board clock's rate error in parts per million (50 ppm means
    the board's clock gains 50 us every real second).
    """

    n_packets: int = 3000
    rate_hz: float = 50.0
    seq_start: int = 0
    out_delay_mean_s: float = 0.0025
    out_delay_std_s: float = 0.0008
    back_delay_mean_s: float = 0.0025
    back_delay_std_s: float = 0.0008
    burst_start: float = 0.02
    burst_end: float = 0.3
    return_loss: float = 0.01
    hold_min_s: float = 20e-6
    hold_max_s: float = 60e-6
    duplicate_probability: float = 0.0
    duplicate_extra_delay_s: float = 0.001
    sender_offset_us: int = 1_000_123
    echoer_offset_us: int = 553_001
    sender_drift_ppm: float = 0.0
    echoer_drift_ppm: float = 0.0
    payload_bytes: int = 32
    channel: int = 1

    def __post_init__(self) -> None:
        if int(self.n_packets) != self.n_packets or self.n_packets < 1:
            raise ValueError("n_packets must be a positive integer")
        if not self.rate_hz > 0:
            raise ValueError("rate_hz must be positive")
        for name in ("out_delay_mean_s", "back_delay_mean_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        for name in ("out_delay_std_s", "back_delay_std_s", "hold_min_s", "duplicate_extra_delay_s"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.hold_max_s < self.hold_min_s:
            raise ValueError("hold_max_s must be at least hold_min_s")
        for name in ("burst_start", "burst_end", "return_loss", "duplicate_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.sender_offset_us < 0 or self.echoer_offset_us < 0 or self.seq_start < 0:
            raise ValueError("offsets and seq_start must be nonnegative")


@dataclass
class EchoLogs:
    """What the two boards would have printed, plus the hidden truth."""

    sender_rows: list[tuple[int, int, int | None]]
    echoer_rows: list[tuple[int, int, int | None, int]]
    sender_text: str
    echoer_text: str
    truth: dict[str, Any] = field(default_factory=dict)


def _lognormal(rng: np.random.Generator, mean: float, std: float, n: int) -> np.ndarray:
    """Delays with exactly this mean and standard deviation (lognormal shape).

    For a lognormal, mean = exp(mu + s^2/2) and std^2 / mean^2 = exp(s^2) - 1,
    so s^2 = ln(1 + (std/mean)^2) and mu = ln(mean) - s^2/2.
    """
    if std == 0:
        return np.full(n, float(mean))
    s2 = math.log(1.0 + (std / mean) ** 2)
    return rng.lognormal(math.log(mean) - s2 / 2.0, math.sqrt(s2), size=n)


def _burst_losses(rng: np.random.Generator, start: float, end: float, n: int) -> np.ndarray:
    """True where the outbound packet is lost, using the start/end burst rule.

    The loop is written out plainly on purpose (it is the rule in words, not a
    copy of the simulator's code). Two uniform numbers are drawn per packet so
    the stream stays aligned whatever happens.
    """
    u_start = rng.random(n)
    u_end = rng.random(n)
    lost = np.zeros(n, dtype=bool)
    in_burst = False
    for i in range(n):
        if not in_burst and u_start[i] < start:
            in_burst = True
        if in_burst:
            lost[i] = True
            if u_end[i] < end:
                in_burst = False
    return lost


def _stamp(offset_us: int, true_s: np.ndarray, drift_ppm: float) -> np.ndarray:
    """A board's whole-microsecond reading at true time ``true_s`` (rounded down)."""
    return np.floor(offset_us + np.asarray(true_s) * 1e6 * (1.0 + drift_ppm * 1e-6)).astype(np.int64)


def truth_summary(p: EchoParams) -> dict[str, Any]:
    """What a correct echo fit should find, worked out from the settings by hand.

    The echo can only see the average of the two legs, so the delay a fit can
    recover is (out + back) / 2, and the jitter it recovers is the root mean
    square of the two legs' spreads. Burst loss: after a received packet the
    next is lost with probability start; after a lost packet the next gets
    through with probability end * (1 - start) (the burst must end AND the
    next packet must not start a new one). The long-run lost fraction is
    start / (start + end * (1 - start)).
    """
    s, e = p.burst_start, p.burst_end
    p_rl = s
    p_lr = e * (1.0 - s)
    denom = p_rl + p_lr
    stationary = s / denom if denom > 0 else None
    return {
        "echo_visible_delay_s": (p.out_delay_mean_s + p.back_delay_mean_s) / 2.0,
        "echo_visible_jitter_s": math.sqrt((p.out_delay_std_s ** 2 + p.back_delay_std_s ** 2) / 2.0),
        "true_outbound_delay_s": p.out_delay_mean_s,
        "asymmetry_bias_s": (p.back_delay_mean_s - p.out_delay_mean_s) / 2.0,
        "outbound_p_received_to_lost": p_rl,
        "outbound_p_lost_to_received": p_lr,
        "outbound_stationary_loss": stationary,
        "outbound_mean_loss_run_packets": (1.0 / p_lr) if p_lr > 0 else None,
        "outbound_mean_burst_length_packets": (1.0 / e) if e > 0 else None,
        "return_loss": p.return_loss,
    }


def generate_echo_logs(params: EchoParams, seed: int) -> EchoLogs:
    """Run the synthetic echo test and return both logs as rows and as text."""
    p = params
    n = int(p.n_packets)
    streams = np.random.SeedSequence(int(seed)).spawn(7)
    rng_loss_out, rng_loss_back, rng_d_out, rng_d_back, rng_hold, rng_rssi, rng_dup = (
        np.random.default_rng(s) for s in streams)
    t_tx = np.arange(n) / float(p.rate_hz)
    lost_out = _burst_losses(rng_loss_out, p.burst_start, p.burst_end, n)
    lost_back = rng_loss_back.random(n) < p.return_loss
    d_out = _lognormal(rng_d_out, p.out_delay_mean_s, p.out_delay_std_s, n)
    d_back = _lognormal(rng_d_back, p.back_delay_mean_s, p.back_delay_std_s, n)
    hold = rng_hold.uniform(p.hold_min_s, p.hold_max_s, size=n)
    rssi = np.clip(np.round(rng_rssi.normal(-50.0, 2.0, size=n)), -128, 0).astype(int)
    dup = rng_dup.random(n) < p.duplicate_probability

    t_rx = t_tx + d_out
    t_etx = t_rx + hold
    t_echo = t_etx + d_back
    t_dup = t_rx + p.duplicate_extra_delay_s

    tx_us = _stamp(p.sender_offset_us, t_tx, p.sender_drift_ppm)
    echo_us = _stamp(p.sender_offset_us, t_echo, p.sender_drift_ppm)
    rx_us = _stamp(p.echoer_offset_us, t_rx, p.echoer_drift_ppm)
    etx_us = _stamp(p.echoer_offset_us, t_etx, p.echoer_drift_ppm)
    dup_us = _stamp(p.echoer_offset_us, t_dup, p.echoer_drift_ppm)

    sender_rows: list[tuple[int, int, int | None]] = []
    arrivals: list[tuple[float, int, int, int | None, int]] = []  # (true time, order, seq, ...)
    for i in range(n):
        seq = p.seq_start + i
        got_echo = (not lost_out[i]) and (not lost_back[i])
        sender_rows.append((seq, int(tx_us[i]), int(echo_us[i]) if got_echo else None))
        if not lost_out[i]:
            arrivals.append((float(t_rx[i]), 0, seq, int(rx_us[i]), int(etx_us[i])))
            if dup[i]:
                arrivals.append((float(t_dup[i]), 1, seq, int(dup_us[i]), None))
    arrivals.sort(key=lambda a: (a[0], a[1], a[2]))
    echoer_rows = [(a[2], a[3], a[4], int(rssi[a[2] - p.seq_start])) for a in arrivals]

    sender_text = _sender_text(p, sender_rows)
    echoer_text = _echoer_text(echoer_rows)
    truth = {
        "schema": "zephyr-espnow-echo-truth-1",
        "status": "synthetic",
        "note": "Made by tools/make_espnow_echo_fixture.py from the settings below. Not a radio measurement.",
        "seed": int(seed),
        "params": asdict(p),
        "expected": truth_summary(p),
        "realized": {
            "n_outbound_lost": int(lost_out.sum()),
            "n_return_lost_after_arrival": int((lost_back & ~lost_out).sum()),
            "n_echoes": int(sum(1 for r in sender_rows if r[2] is not None)),
            "n_duplicate_arrivals": int((dup & ~lost_out).sum()),
            "mean_out_delay_s": float(d_out.mean()),
            "mean_back_delay_s": float(d_back.mean()),
        },
        "stamp_rounding": ("Stamps are rounded down to whole microseconds. The round trip's rounding error has "
                           "mean 0, standard deviation sqrt(1/3) us, and is never larger than 2 us."),
    }
    return EchoLogs(sender_rows, echoer_rows, sender_text, echoer_text, truth)


def _sender_text(p: EchoParams, rows: list[tuple[int, int, int | None]]) -> str:
    lines = [FORMAT_LINE, "# role=sender", "# board=A", "# peer=B", "# session=a-boot-synthetic",
             "# firmware=synthetic-echo-generator", f"# payload_bytes={p.payload_bytes}",
             f"# channel={p.channel}", f"# rate_hz={p.rate_hz:g}", SENDER_HEADER]
    lines += [f"{seq},{tx},{'' if echo is None else echo}" for seq, tx, echo in rows]
    return "\n".join(lines) + "\n"


def _echoer_text(rows: list[tuple[int, int, int | None, int]]) -> str:
    lines = [FORMAT_LINE, "# role=echoer", "# board=B", "# peer=A", "# session=b-boot-synthetic", ECHOER_HEADER]
    lines += [f"{seq},{rx},{'' if etx is None else etx},{rssi}" for seq, rx, etx, rssi in rows]
    return "\n".join(lines) + "\n"


def write_fixture(out_dir: str | Path, params: EchoParams, seed: int) -> EchoLogs:
    """Write sender.csv, echoer.csv and truth.json into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = generate_echo_logs(params, seed)
    with (out / "sender.csv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(logs.sender_text)
    with (out / "echoer.csv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(logs.echoer_text)
    with (out / "truth.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(logs.truth, indent=2, allow_nan=False) + "\n")
    return logs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic ESP-NOW echo log pair from known settings.")
    parser.add_argument("out_dir", type=Path, help="folder for sender.csv, echoer.csv, truth.json")
    parser.add_argument("--seed", type=int, default=20261020)
    defaults = EchoParams()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    values = {name: getattr(args, name) for name in asdict(defaults)}
    write_fixture(args.out_dir, EchoParams(**values), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
