"""Independent checks of the radio link model in sim/radio/channel.py.

Most expected values here come from outside the module under test: the
free-space (Friis) formula typed from c and the carrier frequency, algebraic
identities of the textbook differential-BPSK bit error curve, a numerical
integral over the shadowing bell curve, and statistics of the two-state burst
chain worked out by hand. One exception is stated where it is used: the
packet error curve inside the shadowing test (_hand_dbpsk_per) is a second
typing of the model's own anchor rule (8 percent loss of a 1024 byte frame at
3 dB), so on its own it only checks transcription. What is independent in
that test is the hand SNR budget, the integral over shadowing and the
sampling bounds. Only the library objects under test are imported.

Existing tests already check the anchor numbers, the 10 n dB per decade slope
and the chain's long-run bad fraction. This file adds what they leave open:
the absolute path loss against physics, the shape of the bit error curve, the
bits-per-byte factor, the full distance -> packet delivery chain under random
shadowing, and the memory (correlation) of the burst chain.

Fixed seeds make every run repeat. Every band is at least about 4 standard
errors wide, so a correct model fails by chance only very rarely.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.radio.channel import (
    Channel,
    ChannelConfig,
    GilbertElliott,
    bit_error_rate,
    mean_rssi_dbm,
    packet_error_rate,
    path_loss_db,
)

C_MPS = 299_792_458.0  # speed of light, exact by definition of the metre
F_HZ = 2.437e9  # Wi-Fi channel 6 carrier, the frequency the default placeholder assumes


def _friis_db(d_m: float) -> float:
    """Free-space path loss 20 log10(4 pi d f / c), typed from the textbook."""
    return 20.0 * math.log10(4.0 * math.pi * d_m * F_HZ / C_MPS)


def test_default_path_loss_and_rssi_track_free_space_at_every_distance():
    """Independent fact: Friis free-space loss from c and f, over three decades.

    Free space has exponent 2, so the default config (n = 2) should follow
    20 log10(4 pi d f / c) at every distance, not only at 1 m. The default
    reference loss is 40.05 dB and Friis gives 40.18 dB, so the model is
    allowed to sit at most 0.15 dB below the physical value everywhere (a
    constant offset, because the slope is the same). The default 40.05 dB is a
    placeholder; if it is ever changed, this band must be revisited. A wrong factor in the
    slope (20 instead of 10 per decade for n = 2, or 10 log instead of
    10 n log) changes the gap by 20 dB per decade and is caught at 1 km.
    """
    cfg = ChannelConfig()  # tx 10 dBm, n = 2, no walls
    assert cfg.exponent == 2.0 and cfg.d0_m == 1.0
    for d in (1.0, 3.0, 10.0, 47.0, 100.0, 450.0, 1000.0):
        gap = _friis_db(d) - path_loss_db(d, cfg)
        assert 0.0 <= gap <= 0.15, (d, gap)
        # Received power is the transmit power minus that loss (nothing else is switched on).
        assert mean_rssi_dbm(d, cfg) == pytest.approx(10.0 - _friis_db(d), abs=0.15)


def test_dbpsk_ber_squares_when_the_linear_snr_doubles():
    """Independent fact: DBPSK error rate is BER = 0.5 exp(-a * gamma), gamma = linear SNR.

    Write u = 2 * BER = exp(-a * gamma). Doubling gamma (adding 10 log10(2) =
    3.0103 dB) gives exp(-2 a gamma) = u * u, whatever the constant a is. So
    this checks the exponential shape without knowing a: a coherent-style
    exp(-a * sqrt(gamma)) or a plain power law would not square. Tolerance
    1e-9 relative: both sides are exp of numbers up to about 70 with 1e-16 rounding.
    Limits: at very low SNR a bit is a coin flip (BER -> 0.5), at very high
    SNR the error rate is far below anything a frame could see.
    """
    cfg = ChannelConfig()
    up = 10.0 * math.log10(2.0)
    for snr in (-3.0, -1.0, 0.0, 1.0, 2.0):
        u = 2.0 * bit_error_rate(snr, cfg)
        u_doubled = 2.0 * bit_error_rate(snr + up, cfg)
        assert u_doubled == pytest.approx(u * u, rel=1e-9), snr
    assert bit_error_rate(-80.0, cfg) == pytest.approx(0.5, abs=1e-6)
    assert bit_error_rate(20.0, cfg) < 1e-100
    # Worse SNR never gives fewer bit errors.
    grid = bit_error_rate(np.linspace(-30.0, 10.0, 81), cfg)
    assert np.all(np.diff(grid) <= 0.0)


@pytest.mark.parametrize("model", ["dbpsk", "logistic"])
def test_rare_error_packet_loss_is_proportional_to_frame_length(model):
    """Independent fact: when errors are rare, P(loss) is about (number of trials) x (chance each).

    A frame of L bytes has 8 L bits, and for small q, 1 - (1 - q)^m = m q to
    first order. So at high SNR PER / BER -> 8 L for the dbpsk model. This
    checks the 8 bits per byte (a bytes-for-bits slip changes it by 8x).
    At 0 dB the BER is about 2e-10, so the next-order term is below 1e-6
    relative; tolerance 1e-4. For the logistic model the code gives no
    bit error rate per frame, only the rule that loss in the tail scales with
    length, so we check PER(L) / PER(1024) = L / 1024 deep in the tail
    (SNR = +10 dB, where the neglected term is about e^-15).
    """
    cfg = ChannelConfig(per_model=model)
    if model == "dbpsk":
        for n_bytes in (14, 75, 1024):
            per = packet_error_rate(0.0, n_bytes, cfg)
            assert per / bit_error_rate(0.0, cfg) == pytest.approx(8.0 * n_bytes, rel=1e-4), n_bytes
    else:
        ref = packet_error_rate(10.0, 1024, cfg)
        assert 0.0 < ref < 1e-4
        for n_bytes in (14, 75, 512):
            assert packet_error_rate(10.0, n_bytes, cfg) / ref == pytest.approx(n_bytes / 1024.0, rel=1e-4), n_bytes


def _hand_dbpsk_per(snr_db: float, n_bytes: int) -> float:
    """PER for the 3 dB / 8 percent / 1024 byte anchor rule.

    This is a second typing of the model's own anchor rule, not an outside
    reference: it only guards against transcription slips. Per-bit error ber_ref solves 1 - (1 - ber_ref)^(8 * 1024) = 0.08. DBPSK
    has BER = 0.5 exp(-a gamma); a is fixed by ber_ref at gamma = 10^(-0.3).
    """
    ber_ref = 1.0 - (1.0 - 0.08) ** (1.0 / (8 * 1024))
    a = math.log(0.5 / ber_ref) / 10.0 ** (-3.0 / 10.0)
    ber = 0.5 * math.exp(-a * 10.0 ** (snr_db / 10.0))
    return 1.0 - (1.0 - ber) ** (8 * n_bytes)


def test_distance_to_delivery_chain_under_random_shadowing():
    """Independent fact: delivery rate = 1 - integral of PER over the shadowing bell curve.

    Setup: 10 dBm, n = 3, reference loss 40.05 dB, two walls of 5 dB each,
    noise floor -95 dBm, 200-byte frames at 85 m. The mean SNR is
    10 - (40.05 + 30 log10 85) - 10 + 95 = -2.93 dB, right at the sensitivity
    point where PER changes fastest. Shadowing is a normal with sigma 4 dB,
    drawn fresh at each of 100,000 distinct times. The expected PER is the
    integral of PER(SNR - x) * pdf(x) dx, done here by a trapezoid sum on
    +-8 sigma with 8001 points (error far below the bounds). The realized
    delivery fraction (drawn here with a separate generator, because
    Channel.sample returns only the PER) is a binomial count, standard error
    sqrt(p(1-p)/N) (bound 4.5 SE). The mean of the per-packet PER values is tighter, with
    bound 4.5 * std / sqrt(N). With sigma = 0 the answer is exact:
    the sample must equal the hand value to 1e-9 relative.
    """
    n_pkts, sigma, n_bytes = 100_000, 4.0, 200
    base = {"tx_power_dbm": 10.0, "pl0_db": 40.05, "exponent": 3.0, "noise_floor_dbm": -95.0, "wall_loss_db": 5.0}
    mean_snr = 10.0 - (40.05 + 30.0 * math.log10(85.0)) - 2 * 5.0 + 95.0
    assert mean_snr == pytest.approx(-2.93, abs=0.005)

    # Exact case: no shadowing.
    quiet = Channel(ChannelConfig(**base), np.random.default_rng(1))
    s0 = quiet.sample("a", "b", 85.0, 0.0, frame_bytes=n_bytes, walls=2)
    assert s0.snr_db == pytest.approx(mean_snr, abs=1e-9)
    assert s0.per == pytest.approx(_hand_dbpsk_per(mean_snr, n_bytes), rel=1e-9)

    # Random case: integrate PER over the bell curve.
    x = np.linspace(-8.0 * sigma, 8.0 * sigma, 8001)
    pdf = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    per_x = np.array([_hand_dbpsk_per(mean_snr - xi, n_bytes) for xi in x])
    expected_per = float(np.sum(per_x * pdf) * (x[1] - x[0]))
    assert 0.02 < expected_per < 0.5  # a real mix of good and bad packets

    ch = Channel(ChannelConfig(shadowing_sigma_db=sigma, shadowing_mode="iid", **base), np.random.default_rng(2026))
    pers = np.array([ch.sample("a", "b", 85.0, float(i), frame_bytes=n_bytes, walls=2).per for i in range(n_pkts)])
    assert abs(pers.mean() - expected_per) <= 4.5 * pers.std() / math.sqrt(n_pkts)
    lost = np.random.default_rng(7).random(n_pkts) < pers
    assert abs(lost.mean() - expected_per) <= 4.5 * math.sqrt(expected_per * (1 - expected_per) / n_pkts)


def _autocorr(x: np.ndarray, lag: int) -> float:
    y = x.astype(float) - x.mean()
    return float(np.dot(y[:-lag], y[lag:]) / np.dot(y, y))


def test_burst_chain_loss_memory_decays_as_lambda_to_the_k():
    """Independent fact: autocorrelation of a two-state chain at lag k is lambda^k.

    With p = 0.05 (good to bad) and r = 0.25 (bad to good), lambda = 1 - p - r
    = 0.7. Classic Gilbert loses exactly in the bad state, so the loss series
    has autocorrelation 0.7 at lag 1, 0.49 at lag 2, 0.343 at lag 3. An
    independent coin flip would give 0 at every lag. Standard error for a
    series like this is sqrt((1 - lambda^2) / n) = 0.0011 at lag 1 (about
    0.0022 at lag 3) for n = 400,000, so the 0.01 band is more than 4.5 SE.
    Swapping the roles of p and r keeps lambda but moves the bad fraction;
    that is caught by the next two tests and by the check on the bad fraction here:
    pi_bad = p / (p + r) = 1/6.
    """
    ge = GilbertElliott(0.05, 0.25)
    _, lost = ge.sample(400_000, np.random.default_rng(31))
    lam = 1.0 - 0.05 - 0.25
    for lag in (1, 2, 3):
        assert abs(_autocorr(lost, lag) - lam ** lag) <= 0.01, lag
    assert abs(lost.mean() - 1.0 / 6.0) <= 0.01  # 4.5 SE: sqrt(1/6*5/6/400000*17/3)=0.0014


def test_burst_chain_fraction_wobbles_more_than_coin_flips_by_the_hand_factor():
    """Independent fact: variance of a chain average is inflated by (1 + lambda) / (1 - lambda).

    300 independent chains of n = 2000 steps (each started from the chain's
    long-run mix), p = 0.05, r = 0.25, lambda = 0.7. The loss fraction of one
    chain has mean pi = 1/6 and variance
    pi (1 - pi) / n * (1 + lambda) / (1 - lambda) = 0.1389 / 2000 * 5.667
    = 3.93e-4 (the finite-n correction is below 0.2 percent). Coin flips would
    give 6.9e-5, six times smaller. The variance of 300 values is itself
    uncertain by sqrt(2 / 299) = 8 percent, so accept 35 percent either way
    (over 4 SE). The mean of the 300 fractions must be pi within 4.5 SE
    = 4.5 * sqrt(3.93e-4 / 300) = 0.005.
    """
    ge = GilbertElliott(0.05, 0.25)
    n, chains, pi, lam = 2000, 300, 1.0 / 6.0, 0.7
    rng = np.random.default_rng(32)
    fractions = np.array([ge.sample(n, rng)[1].mean() for _ in range(chains)])
    predicted_var = pi * (1 - pi) / n * (1 + lam) / (1 - lam)
    assert abs(fractions.mean() - pi) <= 4.5 * math.sqrt(predicted_var / chains)
    assert 0.65 * predicted_var <= fractions.var(ddof=1) <= 1.35 * predicted_var


def test_burst_chain_built_from_loss_and_burst_length_delivers_those_numbers():
    """Independent fact: the user's request, 10 percent loss in bursts of 5, is the spec.

    from_loss_and_burst(0.1, 5) is asked for a long-run loss of 0.1 and a mean
    lost run of 5 packets; whatever the internal p and r are, a long sample must
    show those two numbers. n = 1,000,000 steps. Loss fraction: lambda is
    about 0.778, so the variance is the coin-flip value inflated by
    (1 + lambda) / (1 - lambda) = 8; SE = sqrt(0.09 / 1e6 * 8) = 0.00085,
    band 0.004. Mean run: about n * pi_good * p = 1e6 * 0.9 * 0.0222 = 20,000
    completed runs (so require at least 18,000), each geometric with
    standard deviation sqrt(1 - r) / r = 4.47, so SE 0.03, band 0.15.
    """
    ge = GilbertElliott.from_loss_and_burst(0.1, 5.0)
    _, lost = ge.sample(1_000_000, np.random.default_rng(33))
    assert abs(lost.mean() - 0.1) <= 0.004
    padded = np.concatenate(([False], lost, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (starts > 0) & (ends < lost.size)  # drop runs cut off by either end
    runs = (ends - starts)[keep]
    assert runs.size > 18_000
    assert abs(runs.mean() - 5.0) <= 0.15
