"""Radio channel model: distance and obstacles -> signal strength -> packet errors.

Simulation only.  No ESP32 hardware has been measured; every default here is a
placeholder, and the sourced presets live in ``sim.radio.params``
(literature-derived, never measured).

The chain, in plain words:

- **Path loss** (dB) is how much weaker the signal gets on its way.  It grows
  by ``10 * n`` dB every time the distance is multiplied by 10.  ``n`` is the
  path-loss exponent: 2 in empty space, a bit less in some halls, more when
  things are in the way.
- **Shadowing** is a random extra loss (or gain) of a few dB from whatever
  happens to be around: people, walls, the floor.  It is drawn from a normal
  distribution with spread ``shadowing_sigma_db``.
- **RSSI** (received power, dBm) = transmit power minus all the losses.
- **SNR** (dB) = RSSI minus the noise floor.  More SNR, fewer bit errors.
- **PER** (packet error rate) is the chance that a frame arrives with at least
  one wrong bit, so the receiver throws it away.  Longer frames fail more.

The PER curve is anchored at the one point a datasheet gives: the receiver
sensitivity, the RSSI at which a frame of ``sensitivity_ref_bytes`` bytes has
``sensitivity_per`` chance of failing.  The shape around that point is the
textbook differential-BPSK bit error curve (``per_model="dbpsk"``), or a
logistic curve of chosen width (``per_model="logistic"``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    """Channel parameters.  Defaults are placeholders, not a preset."""

    tx_power_dbm: float = 10.0
    pl0_db: float = 40.05  # path loss at d0; free space at 2.437 GHz and 1 m is about 40.2 dB
    d0_m: float = 1.0
    exponent: float = 2.0
    min_distance_m: float = 0.1  # closer than this is treated as this distance
    shadowing_sigma_db: float = 0.0
    shadowing_mode: str = "iid"  # iid, static, or time_correlated
    shadowing_corr_time_s: float = 1.0
    noise_floor_dbm: float = -95.0  # noise floor in the simulated environment
    reference_noise_dbm: float = -95.0  # noise floor under which the sensitivity point holds
    sensitivity_dbm: float = -98.0
    sensitivity_per: float = 0.08
    sensitivity_ref_bytes: int = 1024
    per_model: str = "dbpsk"  # dbpsk or logistic
    logistic_width_db: float = 1.0
    body_loss_db: float = 0.0  # per blocking body
    wall_loss_db: float = 0.0  # per wall
    orientation_loss_db: float = 0.0  # constant antenna-orientation penalty
    breakpoint_m: float | None = None  # dual-slope option: beyond this distance the exponent becomes exponent_far
    exponent_far: float = 4.0  # ground-bounce region; the ideal two-ray value 4 is pessimistic for ESP32 data

    def __post_init__(self) -> None:
        _validate_cfg(self)


def _validate_cfg(cfg: ChannelConfig) -> None:
    for name in ("tx_power_dbm", "pl0_db", "d0_m", "exponent", "min_distance_m", "shadowing_sigma_db",
                 "shadowing_corr_time_s", "noise_floor_dbm", "reference_noise_dbm", "sensitivity_dbm",
                 "logistic_width_db", "body_loss_db", "wall_loss_db", "orientation_loss_db"):
        if not math.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite")
    if cfg.d0_m <= 0 or cfg.min_distance_m <= 0:
        raise ValueError("d0_m and min_distance_m must be positive")
    if cfg.shadowing_sigma_db < 0:
        raise ValueError("shadowing_sigma_db must be nonnegative")
    if cfg.shadowing_mode not in ("iid", "static", "time_correlated"):
        raise ValueError("shadowing_mode must be iid, static, or time_correlated")
    if cfg.shadowing_corr_time_s <= 0:
        raise ValueError("shadowing_corr_time_s must be positive")
    if cfg.per_model not in ("dbpsk", "logistic"):
        raise ValueError("per_model must be dbpsk or logistic")
    if not 0.0 < cfg.sensitivity_per < 1.0:
        raise ValueError("sensitivity_per must be strictly between 0 and 1")
    if int(cfg.sensitivity_ref_bytes) != cfg.sensitivity_ref_bytes or cfg.sensitivity_ref_bytes < 1:
        raise ValueError("sensitivity_ref_bytes must be a positive integer")
    if cfg.logistic_width_db <= 0:
        raise ValueError("logistic_width_db must be positive")
    if cfg.breakpoint_m is not None and not (math.isfinite(cfg.breakpoint_m) and cfg.breakpoint_m > cfg.d0_m):
        raise ValueError("breakpoint_m must be finite and beyond d0_m")
    if not math.isfinite(cfg.exponent_far) or cfg.exponent_far <= 0:
        raise ValueError("exponent_far must be positive")
    if min(cfg.body_loss_db, cfg.wall_loss_db, cfg.orientation_loss_db) < 0:
        raise ValueError("body, wall and orientation losses must be nonnegative")


def _out(x):
    return float(x) if np.ndim(x) == 0 else x


def _distance(distance_m, cfg: ChannelConfig) -> np.ndarray:
    d = np.asarray(distance_m, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise ValueError("distance must be finite and nonnegative")
    return np.maximum(d, cfg.min_distance_m)


def path_loss_db(distance_m, cfg: ChannelConfig):
    """Mean path loss PL0 + 10 n log10(d / d0), no shadowing.

    With ``breakpoint_m`` set (dual-slope model), distances beyond the
    breakpoint add 10 n_far log10(d / d_break) instead, so the curve is
    continuous and only its steepness changes.  Near the ground the reflected
    ray starts cancelling the direct one past about 4 h_t h_r / wavelength.
    """
    d = _distance(distance_m, cfg)
    if cfg.breakpoint_m is None:
        return _out(cfg.pl0_db + 10.0 * cfg.exponent * np.log10(d / cfg.d0_m))
    near = np.minimum(d, cfg.breakpoint_m)
    far = np.maximum(d, cfg.breakpoint_m)
    return _out(cfg.pl0_db + 10.0 * cfg.exponent * np.log10(near / cfg.d0_m)
                + 10.0 * cfg.exponent_far * np.log10(far / cfg.breakpoint_m))


def mean_rssi_dbm(distance_m, cfg: ChannelConfig, *, walls: int = 0, bodies: int = 0):
    """Transmit power minus path loss and the wall, body and orientation terms."""
    if walls < 0 or bodies < 0:
        raise ValueError("walls and bodies must be nonnegative")
    return _out(cfg.tx_power_dbm - np.asarray(path_loss_db(distance_m, cfg)) - walls * cfg.wall_loss_db
                - bodies * cfg.body_loss_db - cfg.orientation_loss_db)


def snr_db(rssi_dbm, noise_floor_dbm):
    return _out(np.asarray(rssi_dbm, dtype=float) - float(noise_floor_dbm))


def _sensitivity_snr_db(cfg: ChannelConfig) -> float:
    return cfg.sensitivity_dbm - cfg.reference_noise_dbm


def _reference_ber(cfg: ChannelConfig) -> float:
    # Per-bit error rate that gives sensitivity_per for the reference frame:
    # 1 - (1 - BER)^(8 L) = P  =>  BER = 1 - (1 - P)^(1 / (8 L)).
    return float(-np.expm1(np.log1p(-cfg.sensitivity_per) / (8 * cfg.sensitivity_ref_bytes)))


def _dbpsk_a(cfg: ChannelConfig) -> float:
    # BER = 0.5 exp(-a gamma); solve a so the anchor point holds exactly.
    gamma_s = 10.0 ** (_sensitivity_snr_db(cfg) / 10.0)
    return math.log(0.5 / _reference_ber(cfg)) / gamma_s


def _logistic_per(snr, n_bytes: int, cfg: ChannelConfig) -> np.ndarray:
    """Logistic PER whose midpoint shifts with frame length.

    For the reference frame, PER = 1 / (1 + exp((snr - s50) / w)), with s50 set
    so the anchor holds.  When errors are rare, PER is close to (bits) x BER,
    so it scales with frame length; in the logistic tail that is the same as
    moving the midpoint by w ln(L / L_ref).  This keeps short frames failing
    at very low SNR instead of letting them survive.
    """
    w = cfg.logistic_width_db
    s50_ref = _sensitivity_snr_db(cfg) - w * math.log(1.0 / cfg.sensitivity_per - 1.0)
    s50 = s50_ref + w * math.log(n_bytes / cfg.sensitivity_ref_bytes)
    z = (np.asarray(snr, dtype=float) - s50) / w
    return 0.5 * (1.0 - np.tanh(z / 2.0))  # = 1 / (1 + exp(z)), without overflow


def bit_error_rate(snr_db, cfg: ChannelConfig):
    """Per-bit error rate.  For the logistic model this is the value implied by the reference frame."""
    s = np.asarray(snr_db, dtype=float)
    if cfg.per_model == "dbpsk":
        return _out(0.5 * np.exp(-_dbpsk_a(cfg) * np.power(10.0, s / 10.0)))
    per_ref = np.clip(_logistic_per(s, cfg.sensitivity_ref_bytes, cfg), 0.0, 1.0 - 1e-16)
    return _out(-np.expm1(np.log1p(-per_ref) / (8 * cfg.sensitivity_ref_bytes)))


def packet_error_rate(snr_db, frame_bytes: int, cfg: ChannelConfig):
    """Chance a frame of ``frame_bytes`` bytes (whole MPDU) is lost to bit errors."""
    if int(frame_bytes) != frame_bytes or frame_bytes < 1:
        raise ValueError("frame_bytes must be a positive integer")
    n_bytes = int(frame_bytes)
    if cfg.per_model == "logistic":
        per = _logistic_per(snr_db, n_bytes, cfg)
    else:
        ber = np.clip(np.asarray(bit_error_rate(snr_db, cfg), dtype=float), 0.0, 0.5)
        # 1 - (1 - BER)^(8 L), written so tiny BERs do not round to zero.
        per = -np.expm1(8 * n_bytes * np.log1p(-ber))
    return _out(np.clip(per, 0.0, 1.0))


@dataclass
class LinkSample:
    rssi_dbm: float
    snr_db: float
    per: float
    shadow_db: float
    path_loss_db: float


class Channel:
    """Stateful channel that keeps the shadowing for each pair of nodes.

    Shadowing is reciprocal: a to b and b to a share one value.  Modes:
    ``iid`` draws a new value for every new time, ``static`` keeps one value
    per pair forever, and ``time_correlated`` lets it drift with correlation
    exp(-dt / tau) between calls.  The last (time, value) per pair is cached,
    so ``sample`` and ``received_power_dbm`` at the same time agree.
    """

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        _validate_cfg(cfg)
        self.cfg = cfg
        self.rng = rng
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        # Speed: the MAC asks for the same node pairs over and over, so the mean
        # terms are computed once per (distance, walls, bodies) with the same
        # vectorized functions (identical values), and the DBPSK packet error
        # rate uses plain math on single numbers.
        self._mean_cache: dict[tuple[float, int, int], tuple[float, float]] = {}
        self._a = _dbpsk_a(cfg) if cfg.per_model == "dbpsk" else None
        # Standard normals drawn in blocks.  A Generator gives the same sequence
        # whether drawn one at a time or in a block, so the values match
        # rng.normal(0, sigma) = sigma * z call for call.
        self._z = np.empty(0)
        self._zi = 0

    def _next_z(self) -> float:
        if self._zi >= self._z.size:
            self._z = self.rng.standard_normal(1024)
            self._zi = 0
        z = float(self._z[self._zi])
        self._zi += 1
        return z

    def _mean_terms(self, distance_m: float, walls: int, bodies: int) -> tuple[float, float]:
        key = (float(distance_m), int(walls), int(bodies))
        hit = self._mean_cache.get(key)
        if hit is None:
            hit = (float(path_loss_db(distance_m, self.cfg)),
                   float(mean_rssi_dbm(distance_m, self.cfg, walls=walls, bodies=bodies)))
            self._mean_cache[key] = hit
        return hit

    def _per(self, snr: float, frame_bytes: int) -> float:
        if self._a is None:
            return float(packet_error_rate(snr, frame_bytes, self.cfg))
        ber = min(max(0.5 * math.exp(-self._a * 10.0 ** (snr / 10.0)), 0.0), 0.5)
        return min(max(-math.expm1(8 * int(frame_bytes) * math.log1p(-ber)), 0.0), 1.0)

    @property
    def sensitivity_dbm(self) -> float:
        return self.cfg.sensitivity_dbm

    def _shadow_value(self, tx: str, rx: str, t_s: float) -> float:
        t_s = float(t_s)
        if not math.isfinite(t_s):
            raise ValueError("time must be finite")
        tx, rx = str(tx), str(rx)
        pair = (tx, rx) if tx <= rx else (rx, tx)
        sigma = self.cfg.shadowing_sigma_db
        mode = self.cfg.shadowing_mode
        last = self._last.get(pair)
        if last is not None and mode == "time_correlated" and t_s < last[0]:
            raise ValueError("time went backwards for this pair in time_correlated mode")
        if sigma == 0.0:
            return 0.0
        if last is not None and (last[0] == t_s or mode == "static"):
            return last[1]
        if last is None or mode == "iid":
            value = sigma * self._next_z()
        else:
            rho = math.exp(-(t_s - last[0]) / self.cfg.shadowing_corr_time_s)
            value = float(rho * last[1] + math.sqrt(max(0.0, 1.0 - rho * rho)) * sigma * self._next_z())
        self._last[pair] = (t_s, value)
        return value

    def sample(self, tx: str, rx: str, distance_m: float, t_s: float, *, frame_bytes: int, walls: int = 0,
               bodies: int = 0, noise_floor_dbm: float | None = None, extra_loss_db: float = 0.0) -> LinkSample:
        if int(frame_bytes) != frame_bytes or frame_bytes < 1:
            raise ValueError("frame_bytes must be a positive integer")
        shadow = self._shadow_value(tx, rx, t_s)
        mean_loss, mean_rssi = self._mean_terms(distance_m, walls, bodies)
        rssi = mean_rssi - shadow - extra_loss_db
        noise = self.cfg.noise_floor_dbm if noise_floor_dbm is None else float(noise_floor_dbm)
        snr = rssi - noise
        return LinkSample(rssi, snr, self._per(snr, frame_bytes), shadow, mean_loss + shadow)

    def received_power_dbm(self, tx: str, rx: str, distance_m: float, t_s: float, *, walls: int = 0,
                           bodies: int = 0, extra_loss_db: float = 0.0) -> float:
        shadow = self._shadow_value(tx, rx, t_s)
        return self._mean_terms(distance_m, walls, bodies)[1] - shadow - extra_loss_db


@dataclass(frozen=True)
class GilbertElliott:
    """Two-state bursty loss: a Good state and a Bad state.

    Each step the chain moves Good -> Bad with chance p and Bad -> Good with
    chance r.  Long run, it spends p / (p + r) of the time in Bad, and a stay
    in Bad lasts 1 / r steps on average.  A packet is lost with chance
    ``loss_good`` or ``loss_bad`` depending on the current state.
    """

    p_good_to_bad: float
    p_bad_to_good: float
    loss_good: float = 0.0
    loss_bad: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.p_good_to_bad, self.p_bad_to_good, self.loss_good, self.loss_bad):
            if not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must be in [0, 1]")
        if self.p_good_to_bad + self.p_bad_to_good <= 0.0:
            raise ValueError("p_good_to_bad and p_bad_to_good cannot both be zero")

    def stationary_bad(self) -> float:
        return self.p_good_to_bad / (self.p_good_to_bad + self.p_bad_to_good)

    def mean_bad_burst(self) -> float:
        return math.inf if self.p_bad_to_good == 0.0 else 1.0 / self.p_bad_to_good

    def mean_loss(self) -> float:
        bad = self.stationary_bad()
        return (1.0 - bad) * self.loss_good + bad * self.loss_bad

    @classmethod
    def from_loss_and_burst(cls, mean_loss: float, mean_burst: float) -> "GilbertElliott":
        """Classic Gilbert chain (no loss in Good, all lost in Bad): r = 1 / b, p = m r / (1 - m)."""
        if not 0.0 <= mean_loss < 1.0 or not mean_burst >= 1.0:
            raise ValueError("mean_loss must be in [0, 1) and mean_burst at least 1")
        r = 1.0 / mean_burst
        p = mean_loss * r / (1.0 - mean_loss)
        if p > 1.0:
            raise ValueError("mean_loss and mean_burst are incompatible")
        return cls(p, r)

    def sample(self, n: int, rng: np.random.Generator, *, start_state: str | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """Return (in_bad_state, lost) for n steps.  Loss uses the current state, then the chain moves."""
        if int(n) != n or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if start_state not in (None, "good", "bad"):
            raise ValueError("start_state must be None, good, or bad")
        n = int(n)
        bad = (rng.random() < self.stationary_bad()) if start_state is None else (start_state == "bad")
        loss_u = rng.random(n)
        move_u = rng.random(n)
        states = np.empty(n, dtype=bool)
        p, r = self.p_good_to_bad, self.p_bad_to_good
        for i in range(n):
            states[i] = bad
            bad = (move_u[i] >= r) if bad else (move_u[i] < p)
        lost = loss_u < np.where(states, self.loss_bad, self.loss_good)
        return states, lost


def delivery_vs_distance(cfg: ChannelConfig, distances_m, *, frame_bytes: int, n_draws: int,
                         rng: np.random.Generator) -> dict:
    """Delivery ratio and RSSI versus distance, averaged over shadowing draws.

    The same shadowing draws are reused at every distance (common random
    numbers), so the curve is smooth and nonincreasing.  Returns arrays keyed
    by name: ``mean_delivery`` and its 5/50/95 percentiles across draws,
    ``mean_rssi_dbm`` and its 5/95 percentiles.
    """
    d = np.asarray(distances_m, dtype=float)
    _distance(d, cfg)
    if int(n_draws) != n_draws or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    shadow = rng.normal(0.0, cfg.shadowing_sigma_db, size=int(n_draws)) if cfg.shadowing_sigma_db else np.zeros(int(n_draws))
    rssi = np.asarray(mean_rssi_dbm(d, cfg), dtype=float).reshape(-1, 1) - shadow.reshape(1, -1)
    delivery = 1.0 - np.asarray(packet_error_rate(rssi - cfg.noise_floor_dbm, frame_bytes, cfg), dtype=float)
    d_pct = np.percentile(delivery, [5, 50, 95], axis=1)
    r_pct = np.percentile(rssi, [5, 95], axis=1)
    return {
        "distances_m": d,
        "mean_delivery": delivery.mean(axis=1),
        "delivery_p05": d_pct[0], "delivery_p50": d_pct[1], "delivery_p95": d_pct[2],
        "mean_rssi_dbm": rssi.mean(axis=1),
        "rssi_p05": r_pct[0], "rssi_p95": r_pct[1],
    }
