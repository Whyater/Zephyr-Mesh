"""Independent checks for sim/radio/channel.py, written from the math only.

Every expected value below was derived by hand (the arithmetic is in the comments) and
NOT taken from the module under test. The tests use only this public API:
ChannelConfig, path_loss_db, mean_rssi_dbm, snr_db,
bit_error_rate, packet_error_rate, Channel, LinkSample, GilbertElliott and
delivery_vs_distance.

The module is imported lazily through the `rc` fixture so this file collects
(and the other tests skip) before the module exists, while
`test_module_exists` does a plain import and fails loudly on a base commit.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


@pytest.fixture(scope="module")
def rc():
    return pytest.importorskip("sim.radio.channel")


def test_module_exists():
    """Plain import: fails (not skips) when sim/radio/channel.py is missing."""
    import sim.radio.channel  # noqa: F401


# ---------------------------------------------------------------------------
# D1: log-distance path loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1.6, 2.0, 2.7, 3.5, 4.2])
@pytest.mark.parametrize("d", [0.5, 1.0, 3.0, 17.0, 120.0])
def test_d1_path_loss_slope_is_10n_per_decade(rc, n, d):
    """D1: with sigma 0, PL(10 d) - PL(d) = 10 n log10(10) = 10 n dB exactly.

    Expected value (10 n) derived by hand, tolerance 1e-9.
    """
    cfg = rc.ChannelConfig(exponent=n, shadowing_sigma_db=0.0)
    diff = float(rc.path_loss_db(10.0 * d, cfg)) - float(rc.path_loss_db(d, cfg))
    assert diff == pytest.approx(10.0 * n, abs=1e-9)


def test_d1_path_loss_value_by_hand(rc):
    """D1: PL(d) = PL0 + 10 n log10(d / d0). Hand value for n = 3, d0 = 2 m, d = 50 m.

    PL = 41.0 + 30 * log10(25) = 41.0 + 30 * 1.39794 = 82.9382 dB (hand).
    """
    cfg = rc.ChannelConfig(pl0_db=41.0, d0_m=2.0, exponent=3.0)
    expected = 41.0 + 30.0 * math.log10(25.0)
    assert float(rc.path_loss_db(50.0, cfg)) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(82.9382, abs=1e-3)  # my hand value


def test_d1_path_loss_clamps_below_min_distance(rc):
    """Distances below min_distance_m are clamped to it."""
    cfg = rc.ChannelConfig(min_distance_m=0.5, exponent=2.5)
    at_min = float(rc.path_loss_db(0.5, cfg))
    assert float(rc.path_loss_db(0.01, cfg)) == pytest.approx(at_min, abs=1e-9)
    # Design choice: distance 0 (co-located nodes) clamps too; negative distance is an error.
    assert float(rc.path_loss_db(0.0, cfg)) == pytest.approx(at_min, abs=1e-9)
    with pytest.raises(ValueError):
        rc.path_loss_db(-1.0, cfg)


def test_d1_free_space_pl0_constant_check():
    """D1: PL0 = 20 log10(4 pi d0 f / c) at 2.437 GHz, 1 m is 40.18 dB.

    This checks my hand value (4 pi f / c = 102.15, 20 log10(102.15) = 40.18),
    not the module. Kept as a documented constant check.
    """
    f_hz = 2.437e9
    c_mps = 299792458.0
    ratio = 4.0 * math.pi * f_hz / c_mps
    assert ratio == pytest.approx(102.15, abs=0.01)
    assert 20.0 * math.log10(ratio) == pytest.approx(40.18, abs=0.005)


def test_d1_mean_rssi_composition(rc):
    """D1: mean RSSI = tx - PL - walls*wall - bodies*body - orientation (hand sum).

    tx 10, PL0 40.05, n 2.5, d 20 m: PL = 40.05 + 25 log10(20) = 72.5757 dB.
    With 2 walls (5 dB), 1 body (3 dB), orientation 1.5 dB:
    RSSI = 10 - 72.5757 - 10 - 3 - 1.5 = -77.0757 dBm.
    """
    cfg = rc.ChannelConfig(
        tx_power_dbm=10.0,
        pl0_db=40.05,
        exponent=2.5,
        wall_loss_db=5.0,
        body_loss_db=3.0,
        orientation_loss_db=1.5,
    )
    pl = 40.05 + 25.0 * math.log10(20.0)
    assert pl == pytest.approx(72.5757, abs=1e-3)
    expected = 10.0 - pl - 2 * 5.0 - 1 * 3.0 - 1.5
    got = float(rc.mean_rssi_dbm(20.0, cfg, walls=2, bodies=1))
    assert got == pytest.approx(expected, abs=1e-9)
    # No obstructions: only the constant orientation penalty remains.
    assert float(rc.mean_rssi_dbm(20.0, cfg)) == pytest.approx(10.0 - pl - 1.5, abs=1e-9)


def test_d1_rssi_falls_10n_per_decade(rc):
    """D1: RSSI = Ptx - PL, so it falls by 10 n dB per decade (hand: n=3.1 -> 31 dB)."""
    cfg = rc.ChannelConfig(exponent=3.1, wall_loss_db=4.0, orientation_loss_db=2.0)
    drop = float(rc.mean_rssi_dbm(5.0, cfg, walls=1)) - float(rc.mean_rssi_dbm(50.0, cfg, walls=1))
    assert drop == pytest.approx(31.0, abs=1e-9)


def test_d1_snr_is_rssi_minus_noise(rc):
    """Definition: SNR in dB = RSSI dBm - noise floor dBm (hand: -80 - (-95) = 15)."""
    assert float(rc.snr_db(-80.0, -95.0)) == pytest.approx(15.0, abs=1e-12)
    assert float(rc.snr_db(-100.0, -90.0)) == pytest.approx(-10.0, abs=1e-12)


# ---------------------------------------------------------------------------
# D1: shadowing
# ---------------------------------------------------------------------------


def _deviation_series(rc, cfg, rng, n, dist_m, dt_s, frame_bytes=75, tx="a", rx="b"):
    """RSSI minus the no-shadowing mean RSSI at a fixed distance, plus shadow_db."""
    ch = rc.Channel(cfg, rng)
    mean = float(rc.mean_rssi_dbm(dist_m, cfg))
    dev = np.empty(n)
    shadow = np.empty(n)
    for k in range(n):
        s = ch.sample(tx, rx, dist_m, k * dt_s, frame_bytes=frame_bytes)
        dev[k] = s.rssi_dbm - mean
        shadow[k] = s.shadow_db
    return dev, shadow


def test_d1_iid_shadowing_std_and_mean(rc):
    """D1: iid N(0, sigma=6 dB) shadowing, N = 1e5 draws at a fixed distance.

    Std of draws has standard error sigma / sqrt(2 N) = 0.22 % of sigma, so the
    tolerance is 1 % of sigma. Mean within 4 sigma / sqrt(N). Expected sigma and
    mean 0 are the definition of the model (hand derived).
    """
    sigma = 6.0
    n = 100_000
    cfg = rc.ChannelConfig(shadowing_sigma_db=sigma, shadowing_mode="iid")
    dev, shadow = _deviation_series(rc, cfg, _rng(11), n, dist_m=30.0, dt_s=0.01)
    assert dev.std(ddof=1) == pytest.approx(sigma, rel=0.01)
    assert abs(dev.mean()) < 4.0 * sigma / math.sqrt(n)
    # shadow_db reports the same excursion (sign convention not fixed by the documented API).
    assert np.allclose(np.abs(dev), np.abs(shadow), atol=1e-9)
    assert shadow.std(ddof=1) == pytest.approx(sigma, rel=0.01)


def test_d1_iid_shadowing_is_gaussian_shaped(rc):
    """D1: Gaussian tails. P(|X| > 1.96 sigma) = 0.05 (hand: standard normal table).

    N = 1e5, sd of the fraction = sqrt(0.05*0.95/N) = 6.9e-4, tolerance 4 sd = 2.8e-3.
    """
    sigma = 4.0
    n = 100_000
    cfg = rc.ChannelConfig(shadowing_sigma_db=sigma, shadowing_mode="iid")
    dev, _ = _deviation_series(rc, cfg, _rng(12), n, dist_m=10.0, dt_s=0.01)
    frac = float(np.mean(np.abs(dev) > 1.96 * sigma))
    assert abs(frac - 0.05) < 4.0 * math.sqrt(0.05 * 0.95 / n)


def test_d1_zero_sigma_draws_no_random_numbers(rc):
    """Contract: with sigma == 0 no random numbers are drawn (generator state unchanged)."""
    cfg = rc.ChannelConfig(shadowing_sigma_db=0.0, shadowing_mode="iid")
    rng = _rng(5)
    before = rng.bit_generator.state
    ch = rc.Channel(cfg, rng)
    for k in range(20):
        s = ch.sample("a", "b", 25.0, k * 0.1, frame_bytes=75)
        assert s.shadow_db == pytest.approx(0.0, abs=1e-12)
        assert s.rssi_dbm == pytest.approx(float(rc.mean_rssi_dbm(25.0, cfg)), abs=1e-9)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize(
    ("dt_s", "tau_s"),
    [(0.5, 2.0), (1.0, 1.0), (2.0, 1.0)],
)
def test_d1_time_correlated_lag1_autocorrelation(rc, dt_s, tau_s):
    """D1: AR(1) shadowing. Lag-1 autocorrelation = rho = exp(-dt / tau).

    Hand values: dt/tau = 0.25 -> 0.7788, 1 -> 0.3679, 2 -> 0.1353.
    N = 2e4 steps, tolerance 4 / sqrt(N) = 0.0283. Stationary std equals sigma
    (effective sample size N (1-rho)/(1+rho) is at least 2000, std sd about 1.6 %,
    so 6 % is about 3.7 sd at the worst case).
    """
    sigma = 6.0
    n = 20_000
    cfg = rc.ChannelConfig(
        shadowing_sigma_db=sigma,
        shadowing_mode="time_correlated",
        shadowing_corr_time_s=tau_s,
    )
    dev, _ = _deviation_series(rc, cfg, _rng(21), n, dist_m=15.0, dt_s=dt_s)
    rho = math.exp(-dt_s / tau_s)
    r1 = float(np.corrcoef(dev[:-1], dev[1:])[0, 1])
    assert abs(r1 - rho) < 4.0 / math.sqrt(n)
    assert dev.std(ddof=1) == pytest.approx(sigma, rel=0.06)


def test_d1_time_correlated_equal_time_reuses_state(rc):
    """Contract: dt = 0 gives rho = 1, so the shadow is unchanged within one time step."""
    cfg = rc.ChannelConfig(
        shadowing_sigma_db=6.0, shadowing_mode="time_correlated", shadowing_corr_time_s=3.0
    )
    ch = rc.Channel(cfg, _rng(22))
    first = ch.sample("a", "b", 20.0, 1.0, frame_bytes=75)
    again = ch.sample("a", "b", 20.0, 1.0, frame_bytes=75)
    assert again.shadow_db == pytest.approx(first.shadow_db, abs=1e-9)


def test_d1_static_shadow_is_reciprocal_and_fixed(rc):
    """In 'static' mode sample(a,b) and sample(b,a) at the same time share
    one shadow draw per unordered pair, fixed over time. Different pairs differ.
    """
    cfg = rc.ChannelConfig(shadowing_sigma_db=6.0, shadowing_mode="static")
    ch = rc.Channel(cfg, _rng(31))
    ab = ch.sample("a", "b", 20.0, 0.0, frame_bytes=75)
    ba = ch.sample("b", "a", 20.0, 0.0, frame_bytes=75)
    assert ab.shadow_db == pytest.approx(ba.shadow_db, abs=1e-12)
    assert ab.rssi_dbm == pytest.approx(ba.rssi_dbm, abs=1e-9)
    later = ch.sample("a", "b", 20.0, 123.4, frame_bytes=75)
    assert later.shadow_db == pytest.approx(ab.shadow_db, abs=1e-12)
    # Many distinct pairs: reciprocal each time and not all identical.
    shadows = []
    for k in range(30):
        x, y = f"n{k}", f"m{k}"
        s1 = ch.sample(x, y, 20.0, 1.0, frame_bytes=75).shadow_db
        s2 = ch.sample(y, x, 20.0, 1.0, frame_bytes=75).shadow_db
        assert s1 == pytest.approx(s2, abs=1e-12)
        shadows.append(s1)
    assert np.std(shadows) > 0.5  # 30 independent N(0, 6) draws: std near 6, never near 0


def test_d1_static_shadow_std_across_pairs(rc):
    """D1: static draws across many pairs are N(0, sigma). 4000 pairs, sigma 5 dB:
    std standard error sigma / sqrt(2 N) = 1.1 %, tolerance 5 %; mean within 4 sigma / sqrt(N).
    """
    sigma = 5.0
    n = 4000
    cfg = rc.ChannelConfig(shadowing_sigma_db=sigma, shadowing_mode="static")
    ch = rc.Channel(cfg, _rng(32))
    sh = np.array(
        [ch.sample(f"x{k}", f"y{k}", 10.0, 0.0, frame_bytes=75).shadow_db for k in range(n)]
    )
    assert sh.std(ddof=1) == pytest.approx(sigma, rel=0.05)
    assert abs(sh.mean()) < 4.0 * sigma / math.sqrt(n)


def test_d1_time_correlated_reciprocal_same_time(rc):
    """Contract: pair key is the sorted pair, so a->b and b->a at the same time agree."""
    cfg = rc.ChannelConfig(
        shadowing_sigma_db=6.0, shadowing_mode="time_correlated", shadowing_corr_time_s=2.0
    )
    ch = rc.Channel(cfg, _rng(33))
    for k in range(10):
        t = 0.7 * k
        s1 = ch.sample("a", "b", 20.0, t, frame_bytes=75)
        s2 = ch.sample("b", "a", 20.0, t, frame_bytes=75)
        assert s1.shadow_db == pytest.approx(s2.shadow_db, abs=1e-9)


@pytest.mark.parametrize("mode", ["iid", "static", "time_correlated"])
def test_d1_received_power_matches_sample_shadow(rc, mode):
    """Contract: received_power_dbm and sample at the same (pair, time) see the same shadow."""
    cfg = rc.ChannelConfig(
        shadowing_sigma_db=6.0, shadowing_mode=mode, shadowing_corr_time_s=2.0
    )
    ch = rc.Channel(cfg, _rng(34))
    for k in range(5):
        t = 1.0 * k
        p = ch.received_power_dbm("a", "b", 25.0, t)
        s = ch.sample("a", "b", 25.0, t, frame_bytes=75)
        assert p == pytest.approx(s.rssi_dbm, abs=1e-9)


def test_d1_sample_applies_walls_bodies_extra_and_noise_override(rc):
    """With sigma 0 the sample equals the hand sum and SNR uses the override.

    tx 10, PL0 40.05, n 2, d 100 m: PL = 40.05 + 40 = 80.05. RSSI with 1 wall (6),
    1 body (2), extra 1.5: 10 - 80.05 - 6 - 2 - 1.5 = -79.55 dBm. SNR with noise
    override -90 dBm is 10.45 dB.
    """
    cfg = rc.ChannelConfig(wall_loss_db=6.0, body_loss_db=2.0, shadowing_sigma_db=0.0)
    ch = rc.Channel(cfg, _rng(35))
    s = ch.sample(
        "a", "b", 100.0, 0.0, frame_bytes=75, walls=1, bodies=1,
        noise_floor_dbm=-90.0, extra_loss_db=1.5,
    )
    assert s.rssi_dbm == pytest.approx(-79.55, abs=1e-9)
    assert s.snr_db == pytest.approx(10.45, abs=1e-9)


# ---------------------------------------------------------------------------
# D2: DBPSK PER anchored at the sensitivity point
# ---------------------------------------------------------------------------


def _anchor_cfg(rc, **kw):
    base = dict(
        sensitivity_dbm=-98.0,
        reference_noise_dbm=-95.0,
        sensitivity_per=0.08,
        sensitivity_ref_bytes=1024,
        per_model="dbpsk",
    )
    base.update(kw)
    return rc.ChannelConfig(**base)


def _hand_dbpsk_per(snr_db: float, frame_bytes: int) -> float:
    """Hand derivation D2 written out with math only (not from the module)."""
    p_s, l_ref, snr_s = 0.08, 1024, -3.0
    ber_ref = -math.expm1(math.log1p(-p_s) / (8 * l_ref))  # 1.0178e-5
    a = math.log(0.5 / ber_ref) / 10 ** (snr_s / 10)  # 21.55
    ber = 0.5 * math.exp(-a * 10 ** (snr_db / 10))
    return -math.expm1(8 * frame_bytes * math.log1p(-ber))


def test_d2_hand_constants_are_self_consistent():
    """D2 sanity: the hand constants (BER_ref 1.0178e-5, a 21.55) reproduce the anchor."""
    p_s, l_ref = 0.08, 1024
    ber_ref = 1.0 - (1.0 - p_s) ** (1.0 / (8 * l_ref))
    assert ber_ref == pytest.approx(1.0178e-5, rel=1e-3)
    gamma_s = 10 ** (-3.0 / 10)
    assert gamma_s == pytest.approx(0.5012, rel=1e-3)
    a = math.log(0.5 / ber_ref) / gamma_s
    assert a == pytest.approx(21.55, rel=1e-3)
    assert 1.0 - (1.0 - 0.5 * math.exp(-a * gamma_s)) ** (8 * l_ref) == pytest.approx(0.08, abs=1e-9)


def test_d2_dbpsk_anchor_is_exact(rc):
    """D2: PER(SNR_s = -98 - (-95) = -3 dB, 1024 B) = 0.08 exactly (tolerance 1e-9)."""
    cfg = _anchor_cfg(rc)
    assert float(rc.packet_error_rate(-3.0, 1024, cfg)) == pytest.approx(0.08, abs=1e-9)


def test_d2_dbpsk_waterfall_hand_values(rc):
    """D2: PER at SNR_s +1 dB = 5.07e-3 and SNR_s -1 dB = 0.537 (1024 B), and
    PER(75 B) at SNR_s = 6.09e-3. Hand numbers from the derivation, rel 1e-3;
    the full-precision hand recomputation agrees to rel 1e-6.
    """
    cfg = _anchor_cfg(rc)
    up = float(rc.packet_error_rate(-2.0, 1024, cfg))
    down = float(rc.packet_error_rate(-4.0, 1024, cfg))
    short = float(rc.packet_error_rate(-3.0, 75, cfg))
    assert up == pytest.approx(5.07e-3, rel=1e-3)
    assert down == pytest.approx(0.537, rel=1e-3)
    assert short == pytest.approx(6.09e-3, rel=1e-3)
    assert up == pytest.approx(_hand_dbpsk_per(-2.0, 1024), rel=1e-6)
    assert down == pytest.approx(_hand_dbpsk_per(-4.0, 1024), rel=1e-6)
    assert short == pytest.approx(_hand_dbpsk_per(-3.0, 75), rel=1e-6)


def test_d2_bit_error_rate_hand_values(rc):
    """D2: BER = 0.5 exp(-a gamma). At SNR_s BER = 1.0178e-5 (anchor), at +1 dB 6.21e-7,
    at -1 dB 9.39e-5 (hand numbers, rel 2e-3)."""
    cfg = _anchor_cfg(rc)
    assert float(rc.bit_error_rate(-3.0, cfg)) == pytest.approx(1.0178e-5, rel=1e-3)
    assert float(rc.bit_error_rate(-2.0, cfg)) == pytest.approx(6.21e-7, rel=2e-3)
    assert float(rc.bit_error_rate(-4.0, cfg)) == pytest.approx(9.39e-5, rel=2e-3)


def test_d2_per_depends_on_snr_not_noise_floor_setting(rc):
    """D2: the anchor is defined at SNR_s = sensitivity - reference noise, so changing
    the actual noise floor in the config does not move packet_error_rate(snr)."""
    cfg = _anchor_cfg(rc, noise_floor_dbm=-85.0)
    assert float(rc.packet_error_rate(-3.0, 1024, cfg)) == pytest.approx(0.08, abs=1e-9)


def test_d2_anchor_moves_with_reference_noise(rc):
    """D2: with sensitivity -92 and reference noise -90, SNR_s = -2 dB; the anchor
    must sit there (hand arithmetic)."""
    cfg = _anchor_cfg(rc, sensitivity_dbm=-92.0, reference_noise_dbm=-90.0)
    assert float(rc.packet_error_rate(-2.0, 1024, cfg)) == pytest.approx(0.08, abs=1e-9)


def test_d2_channel_sample_at_sensitivity_gives_anchor_per(rc):
    """D2 end to end: place the receiver where mean RSSI = -98 dBm (sigma 0).
    tx 10, PL0 40.05, n 2: need PL = 108, so 20 log10(d) = 67.95, d = 10^3.3975 m.
    The sample has SNR -3 dB vs the -95 dBm floor and PER 0.08 for 1024 B.
    A noise override of -90 dBm gives SNR -8 dB, and PER then must be larger.
    """
    cfg = _anchor_cfg(rc, tx_power_dbm=10.0, pl0_db=40.05, exponent=2.0,
                      noise_floor_dbm=-95.0, shadowing_sigma_db=0.0)
    d = 10 ** ((108.0 - 40.05) / 20.0)
    ch = rc.Channel(cfg, _rng(41))
    s = ch.sample("a", "b", d, 0.0, frame_bytes=1024)
    assert s.rssi_dbm == pytest.approx(-98.0, abs=1e-6)
    assert s.snr_db == pytest.approx(-3.0, abs=1e-6)
    assert s.per == pytest.approx(0.08, abs=1e-6)
    noisy = ch.sample("a", "b", d, 1.0, frame_bytes=1024, noise_floor_dbm=-90.0)
    assert noisy.snr_db == pytest.approx(-8.0, abs=1e-6)
    assert noisy.per > s.per


@pytest.mark.parametrize("model", ["dbpsk", "logistic"])
def test_d2_per_properties_over_grid(rc, model):
    """D2: PER in [0, 1], nonincreasing in SNR, nondecreasing in frame bytes, over a grid.

    PER -> 0 as SNR -> +inf and PER -> ~1 as SNR -> -inf (checked at -120 dB) (frames of 75 B or more).
    """
    cfg = _anchor_cfg(rc, per_model=model)
    snrs = np.arange(-30.0, 30.0 + 1e-9, 0.5)
    sizes = [1, 10, 75, 250, 1024, 2000]
    grid = np.array(
        [np.asarray(rc.packet_error_rate(snrs, L, cfg), dtype=float) for L in sizes]
    )  # shape (sizes, snrs)
    assert grid.shape == (len(sizes), len(snrs))
    assert np.all(np.isfinite(grid))
    assert grid.min() >= 0.0 and grid.max() <= 1.0
    assert np.all(np.diff(grid, axis=1) <= 1e-12), "PER must be nonincreasing in SNR"
    assert np.all(np.diff(grid, axis=0) >= -1e-12), "PER must be nondecreasing in frame bytes"
    hi = np.asarray(rc.packet_error_rate(40.0, 1024, cfg), dtype=float)
    assert float(hi) < 1e-6
    for L in (75, 1024):
        lo = float(rc.packet_error_rate(-120.0, L, cfg))
        assert lo > 0.99


def test_d2_dbpsk_far_below_floor_hand_limit(rc):
    """D2 limit: PER(-inf) -> 1 - (1 - 0.5)^(8 L). At -80 dB, BER = 0.5 exp(-a 1e-8)
    is 0.5 to 8 digits, so a 1-byte frame has PER = 1 - 0.5^8 = 0.99609375."""
    cfg = _anchor_cfg(rc)
    assert float(rc.packet_error_rate(-80.0, 1, cfg)) == pytest.approx(0.99609375, rel=1e-6)


def test_d2_logistic_passes_through_anchor(rc):
    """D2: the logistic model uses the same anchor, PER(SNR_s, 1024 B) = 0.08 (1e-9)."""
    for w in (0.5, 1.0, 2.5):
        cfg = _anchor_cfg(rc, per_model="logistic", logistic_width_db=w)
        assert float(rc.packet_error_rate(-3.0, 1024, cfg)) == pytest.approx(0.08, abs=1e-9)


def test_d2_logistic_hand_shape(rc):
    """D2 (logistic): PER_ref = 1 / (1 + exp((snr - s50) / w)). Anchor 0.08 gives
    (snr_s - s50) / w = ln(0.92 / 0.08) = ln(11.5) = 2.4423, so s50 = -3 - 2.4423 w
    and PER(s50) = 0.5 exactly. For w = 1, at SNR_s + 1 dB the exponent is 3.4423:
    PER = 1 / (1 + e^3.4423) = 0.03100 (hand). Other frame sizes use the same
    sizes move the midpoint by w ln(L / L_ref) (design revision: the
    rare-error tail scales with frame length, so short frames still fail at very low
    SNR). At SNR_s for 75 B: exponent = ln(11.5) - ln(75/1024) = ln(157.01), so
    PER = 1 / (1 + 157.01) = 6.329e-3 (hand).
    """
    ln115 = math.log(11.5)
    assert ln115 == pytest.approx(2.4423, abs=1e-4)
    for w in (1.0, 2.0):
        cfg = _anchor_cfg(rc, per_model="logistic", logistic_width_db=w)
        s50 = -3.0 - w * ln115
        assert float(rc.packet_error_rate(s50, 1024, cfg)) == pytest.approx(0.5, abs=1e-9)
    cfg1 = _anchor_cfg(rc, per_model="logistic", logistic_width_db=1.0)
    expected_up = 1.0 / (1.0 + math.exp(1.0 + ln115))
    assert expected_up == pytest.approx(0.03100, abs=2e-4)
    assert float(rc.packet_error_rate(-2.0, 1024, cfg1)) == pytest.approx(expected_up, rel=1e-6)
    short = 1.0 / (1.0 + 11.5 * 1024.0 / 75.0)
    assert short == pytest.approx(6.329e-3, abs=1e-6)
    assert float(rc.packet_error_rate(-3.0, 75, cfg1)) == pytest.approx(short, rel=1e-6)


def test_d2_logistic_tiny_ber_does_not_round_to_zero(rc):
    """Contract: stable forms so tiny BERs survive. Far above the anchor the logistic
    reference PER is about exp(-(snr - s50)/w) ~ 1e-9 or smaller, and a 1-byte frame
    then has PER of about (1/8192) of that, which must still be a positive number."""
    cfg = _anchor_cfg(rc, per_model="logistic", logistic_width_db=1.0)
    snr = -3.0 + 12.0  # reference PER about 0.08 * exp(-12) ~ 4.9e-7
    got = float(rc.packet_error_rate(snr, 1, cfg))
    ref = 1.0 / (1.0 + math.exp(12.0 + math.log(11.5)))
    expected = 1.0 - (1.0 - ref) ** (1.0 / 1024.0)  # 1 B of 1024 B, per-bit conversion
    assert got > 0.0
    assert got == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# D1: expected delivery is monotone in distance
# ---------------------------------------------------------------------------


def test_d1_expected_delivery_monotone_with_common_random_numbers(rc):
    """D1: E[delivery](d) = mean over X of (1 - PER(SNR(d) - X)). PER is nonincreasing in
    SNR and SNR(d) decreases in d, so with ONE fixed array of normal draws reused at every
    distance the estimates must be nonincreasing (float slack 1e-12).

    The estimate is computed here from mean_rssi_dbm and packet_error_rate, not from
    Channel or delivery_vs_distance.
    """
    for model in ("dbpsk", "logistic"):
        cfg = _anchor_cfg(rc, per_model=model, exponent=2.7, shadowing_sigma_db=6.0)
        z = _rng(51).standard_normal(2000)
        dists = np.geomspace(1.0, 3000.0, 40)
        est = []
        for d in dists:
            rssi = float(rc.mean_rssi_dbm(d, cfg)) + cfg.shadowing_sigma_db * z
            snr = rssi - cfg.noise_floor_dbm
            per = np.asarray(rc.packet_error_rate(snr, 75, cfg), dtype=float)
            est.append(float(np.mean(1.0 - per)))
        est = np.array(est)
        assert np.all(np.diff(est) <= 1e-12), model
        # The range really spans the transition. (Logistic short frames keep a small tail:
        # the per-bit conversion gives delivery = 0.92-style power law, so only demand < 0.5.)
        assert est[0] > 0.99 and est[-1] < 0.01  # the range really spans the transition (both PER models)


def _find_keys(result: dict, *needles: str) -> list[str]:
    return [k for k in result if all(n in str(k).lower() for n in needles)]


def _percentile_key(keys, number: int):
    """Find a key naming percentile `number` (5, 50 or 95), tolerating p5/p05/pct_5 etc."""
    pat = re.compile(rf"(?<!\d)0?{number}(?!\d)")
    hits = [k for k in keys if pat.search(str(k))]
    return hits[0] if hits else None


def test_d1_delivery_vs_distance_structure_and_monotone(rc):
    """delivery_vs_distance returns, per distance, mean delivery, 5/50/95
    percentiles of delivery, mean RSSI and its 5/95 percentiles (key names are not fixed
    by the documented API, so keys are located by substring: 'deliver', 'rssi', 'mean', 5/50/95).

    Expected values computed by hand in the test: mean RSSI at each distance equals
    mean_rssi_dbm within 4 sigma / sqrt(n_draws); mean delivery is nonincreasing in
    distance up to 4 combined standard errors of Monte Carlo noise
    (se_i^2 <= m_i (1 - m_i) / n because delivery is in [0, 1]).
    """
    sigma, n_draws = 6.0, 4000
    cfg = _anchor_cfg(rc, exponent=2.7, shadowing_sigma_db=sigma)
    dists = np.geomspace(1.0, 3000.0, 30)
    res = rc.delivery_vs_distance(cfg, dists, frame_bytes=75, n_draws=n_draws, rng=_rng(52))
    assert isinstance(res, dict)

    deliver_keys = _find_keys(res, "deliver")
    rssi_keys = _find_keys(res, "rssi")
    assert deliver_keys, f"no delivery keys in {list(res)}"
    assert rssi_keys, f"no rssi keys in {list(res)}"
    mean_del = [k for k in deliver_keys if "mean" in str(k).lower()]
    mean_rssi = [k for k in rssi_keys if "mean" in str(k).lower()]
    assert mean_del, f"no mean-delivery key in {list(res)}"
    assert mean_rssi, f"no mean-rssi key in {list(res)}"
    # Aliases (several keys naming the same quantity) must agree with each other.
    for group in (mean_del, mean_rssi):
        for other in group[1:]:
            assert np.array_equal(np.asarray(res[group[0]]), np.asarray(res[other])), (group[0], other)
    non_mean_del = [k for k in deliver_keys if k not in mean_del]
    non_mean_rssi = [k for k in rssi_keys if k not in mean_rssi]
    for pct in (5, 50, 95):
        assert _percentile_key(non_mean_del, pct) is not None, f"no p{pct} delivery key in {list(res)}"
    for pct in (5, 95):
        assert _percentile_key(non_mean_rssi, pct) is not None, f"no p{pct} rssi key in {list(res)}"

    m = np.asarray(res[mean_del[0]], dtype=float)
    assert m.shape == (len(dists),)
    assert np.all((m >= 0.0) & (m <= 1.0))
    assert m[0] > 0.99 and m[-1] < 0.01

    se = np.sqrt(np.maximum(m * (1.0 - m), 1e-12) / n_draws)
    tol = 4.0 * np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
    assert np.all(np.diff(m) <= tol), "mean delivery rises with distance beyond Monte Carlo noise"

    p5 = np.asarray(res[_percentile_key(non_mean_del, 5)], dtype=float)
    p50 = np.asarray(res[_percentile_key(non_mean_del, 50)], dtype=float)
    p95 = np.asarray(res[_percentile_key(non_mean_del, 95)], dtype=float)
    assert np.all(p5 <= p50 + 1e-12) and np.all(p50 <= p95 + 1e-12)
    assert np.all((p5 >= 0.0) & (p95 <= 1.0))

    r_mean = np.asarray(res[mean_rssi[0]], dtype=float)
    expected_rssi = np.array([float(rc.mean_rssi_dbm(d, cfg)) for d in dists])
    assert np.all(np.abs(r_mean - expected_rssi) < 4.0 * sigma / math.sqrt(n_draws))
    r5 = np.asarray(res[_percentile_key(non_mean_rssi, 5)], dtype=float)
    r95 = np.asarray(res[_percentile_key(non_mean_rssi, 95)], dtype=float)
    assert np.all(r5 <= r95)
    # Hand value: for N(0, 6) the 5 % and 95 % points are -/+ 1.645 sigma = -/+ 9.87 dB.
    assert np.all(np.abs((r95 - expected_rssi) - 1.645 * sigma) < 1.0)
    assert np.all(np.abs((r5 - expected_rssi) + 1.645 * sigma) < 1.0)


def test_d1_delivery_vs_distance_deterministic_per_seed(rc):
    """Contract: same seed and inputs give identical outputs."""
    cfg = _anchor_cfg(rc, exponent=2.7, shadowing_sigma_db=6.0)
    dists = np.geomspace(1.0, 500.0, 8)
    a = rc.delivery_vs_distance(cfg, dists, frame_bytes=75, n_draws=300, rng=_rng(53))
    b = rc.delivery_vs_distance(cfg, dists, frame_bytes=75, n_draws=300, rng=_rng(53))
    assert list(a) == list(b)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


# ---------------------------------------------------------------------------
# D3: Gilbert-Elliott two-state chain
# ---------------------------------------------------------------------------


def _bad_runs(bad: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of True, dropping runs that touch either end of the array
    (those are truncated and would bias the mean)."""
    b = np.concatenate(([False], bad.astype(bool), [False]))
    d = np.diff(b.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    keep = (starts > 0) & (ends < len(bad))
    return lengths[keep]


def test_d3_closed_forms_hand_values(rc):
    """D3: pi_B = p / (p + r), E[burst] = 1 / r, mean loss = pi_G h_G + pi_B h_B.

    p = 0.02, r = 0.2: pi_B = 0.02 / 0.22 = 0.090909, burst 5.
    With h_G = 0.01, h_B = 0.6: 0.909091 * 0.01 + 0.090909 * 0.6 = 0.063636.
    """
    ge = rc.GilbertElliott(0.02, 0.2)
    assert ge.stationary_bad() == pytest.approx(0.02 / 0.22, abs=1e-12)
    assert ge.mean_bad_burst() == pytest.approx(5.0, abs=1e-12)
    assert ge.mean_loss() == pytest.approx(0.02 / 0.22, abs=1e-12)  # classic Gilbert
    gen = rc.GilbertElliott(0.02, 0.2, loss_good=0.01, loss_bad=0.6)
    assert gen.stationary_bad() == pytest.approx(0.090909090909, abs=1e-9)
    assert gen.mean_loss() == pytest.approx(0.0636363636, abs=1e-9)
    assert gen.mean_bad_burst() == pytest.approx(5.0, abs=1e-12)


def test_d3_from_loss_and_burst_hand_values(rc):
    """D3: r = 1 / b, p = m r / (1 - m). (m, b) = (0.1, 5): r = 0.2, p = 0.1*0.2/0.9 = 0.022222.
    (0.3, 4): r = 0.25, p = 0.3*0.25/0.7 = 0.107142857. Round trip recovers m and b."""
    ge = rc.GilbertElliott.from_loss_and_burst(0.1, 5.0)
    assert ge.p_bad_to_good == pytest.approx(0.2, abs=1e-12)
    assert ge.p_good_to_bad == pytest.approx(0.1 * 0.2 / 0.9, abs=1e-12)
    assert ge.p_good_to_bad == pytest.approx(0.0222222, abs=1e-6)
    assert ge.loss_good == 0.0 and ge.loss_bad == 1.0
    assert ge.mean_loss() == pytest.approx(0.1, abs=1e-12)
    assert ge.mean_bad_burst() == pytest.approx(5.0, abs=1e-12)
    ge2 = rc.GilbertElliott.from_loss_and_burst(0.3, 4.0)
    assert ge2.p_bad_to_good == pytest.approx(0.25, abs=1e-12)
    assert ge2.p_good_to_bad == pytest.approx(0.3 * 0.25 / 0.7, abs=1e-12)
    assert ge2.mean_loss() == pytest.approx(0.3, abs=1e-12)


def test_d3_sample_shapes_and_classic_loss_equals_bad(rc):
    """Contract: sample returns (bad bool[n], lost bool[n]); classic Gilbert loses
    exactly in the bad state (loss_good 0, loss_bad 1)."""
    ge = rc.GilbertElliott(0.02, 0.2)
    bad, lost = ge.sample(5000, _rng(61))
    assert bad.shape == (5000,) and lost.shape == (5000,)
    assert bad.dtype == bool and lost.dtype == bool
    assert np.array_equal(bad, lost)


def test_d3_stationary_fraction_and_mean_burst(rc):
    """D3: p = 0.02, r = 0.2, n = 1e6.

    Bad fraction -> pi_B = 0.090909 with sd 8.2e-4 (hand, from the chain's eigenvalue
    0.78), tolerance 4 sd = 3.3e-3. Mean bad run length -> 1/r = 5.0, from about
    18 000 bursts: sd 0.033, tolerance 0.14. Run lengths are counted here from the
    returned array.
    """
    ge = rc.GilbertElliott(0.02, 0.2)
    bad, lost = ge.sample(1_000_000, _rng(62))
    assert abs(bad.mean() - 0.02 / 0.22) < 3.3e-3
    runs = _bad_runs(bad)
    assert len(runs) > 15_000
    assert abs(runs.mean() - 5.0) < 0.14
    assert abs(lost.mean() - 0.02 / 0.22) < 3.3e-3


def test_d3_geometric_burst_distribution(rc):
    """D3: bad sojourn K is geometric, P(K = k) = (1 - r)^(k-1) r. For r = 0.2:
    P(K = 1) = 0.2, P(K <= 3) = 1 - 0.8^3 = 0.488. About 18 000 bursts, binomial sd
    <= 0.0038, tolerance 0.015."""
    ge = rc.GilbertElliott(0.02, 0.2)
    bad, _ = ge.sample(1_000_000, _rng(63))
    runs = _bad_runs(bad)
    assert abs(np.mean(runs == 1) - 0.2) < 0.015
    assert abs(np.mean(runs <= 3) - 0.488) < 0.015


def test_d3_transition_frequencies(rc):
    """D3: P(G -> B) = p and P(B -> G) = r, counted from the returned states.
    n = 1e6: about 909 000 good steps give sd 1.5e-4 for p (tolerance 6e-4); about
    91 000 bad steps give sd 1.3e-3 for r (tolerance 5.3e-3)."""
    ge = rc.GilbertElliott(0.02, 0.2)
    bad, _ = ge.sample(1_000_000, _rng(64))
    cur, nxt = bad[:-1], bad[1:]
    p_hat = float(np.mean(nxt[~cur]))
    r_hat = float(np.mean(~nxt[cur]))
    assert abs(p_hat - 0.02) < 6e-4
    assert abs(r_hat - 0.2) < 5.3e-3


def test_d3_general_loss_probabilities_follow_current_state(rc):
    """D3: with h_G = 0.05, h_B = 0.7, p = 0.02, r = 0.2, the loss decision uses the
    CURRENT state: P(lost | state good now) = 0.05, P(lost | bad now) = 0.7.

    Overall mean loss = 0.909091*0.05 + 0.090909*0.7 = 0.109091 (hand), sd about 6.2e-4,
    tolerance 3e-3. Conditional rates: bad steps (~91 000) sd 1.5e-3, tolerance 8e-3;
    good steps (~909 000) sd 2.3e-4, tolerance 1.5e-3.
    """
    ge = rc.GilbertElliott(0.02, 0.2, loss_good=0.05, loss_bad=0.7)
    bad, lost = ge.sample(1_000_000, _rng(65))
    assert ge.mean_loss() == pytest.approx(0.1090909, abs=1e-6)
    assert abs(lost.mean() - 0.109091) < 3e-3
    assert abs(lost[bad].mean() - 0.7) < 8e-3
    assert abs(lost[~bad].mean() - 0.05) < 1.5e-3


def test_d3_start_state_and_stationary_initial_draw(rc):
    """Contract: start_state 'good' or 'bad' fixes the first state; None draws it from
    the stationary law. p = r = 0.2 gives pi_B = 0.5; over 4000 seeds the fraction of
    bad first states has sd sqrt(0.25 / 4000) = 7.9e-3, tolerance 4 sd = 0.032."""
    ge = rc.GilbertElliott(0.2, 0.2)
    for seed in range(5):
        bad_g, _ = ge.sample(50, _rng(seed), start_state="good")
        bad_b, _ = ge.sample(50, _rng(seed), start_state="bad")
        assert not bad_g[0]
        assert bad_b[0]
    firsts = np.array([ge.sample(2, _rng(1000 + s))[0][0] for s in range(4000)])
    assert abs(firsts.mean() - 0.5) < 4.0 * math.sqrt(0.25 / 4000)


def test_d3_determinism_per_seed(rc):
    """Contract: same seed and inputs give identical arrays; a different seed differs
    (two 2000-step chains agreeing everywhere has probability far below 1e-100)."""
    ge = rc.GilbertElliott(0.05, 0.3, loss_good=0.02, loss_bad=0.8)
    b1, l1 = ge.sample(2000, _rng(77))
    b2, l2 = ge.sample(2000, _rng(77))
    b3, l3 = ge.sample(2000, _rng(78))
    assert np.array_equal(b1, b2) and np.array_equal(l1, l2)
    assert not (np.array_equal(b1, b3) and np.array_equal(l1, l3))
