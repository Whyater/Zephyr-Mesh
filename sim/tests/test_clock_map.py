"""Checks for mapping a device boot clock onto sim time (``sim.clock_map``).

Tolerances come from the hand formulas for a straight-line least-squares fit,
written out next to the tests that use them.
"""
import math

import numpy as np
import pytest

from sim import clock_map
from sim.timesync import ClockModel, sync_log

OFFSET_S = 2.0
DRIFT_PPM = 50.0


def device_reading(sim_s, offset_s=OFFSET_S, drift_ppm=DRIFT_PPM):
    # Written out by hand so the test does not depend on the module under test.
    return np.asarray(sim_s, dtype=float) * (1.0 + drift_ppm * 1e-6) + offset_s


def test_hand_point_after_1000_seconds():
    # 50 ppm fast means 50 microseconds gained per second, so 0.05 s after 1000 s.
    # The device reads 1000 + 0.05 + 2 = 1002.05 when the sim reads 1000.
    m = clock_map.ClockMap(offset_s=OFFSET_S, drift_ppm=DRIFT_PPM)
    assert m.to_device(1000.0) == pytest.approx(1002.05, abs=1e-12)
    assert m.to_sim(1002.05) == pytest.approx(1000.0, abs=1e-9)


def test_convention_matches_timesync_clock_model():
    times = np.linspace(0.0, 3600.0, 37)
    model = ClockModel(offset_s=OFFSET_S, drift_ppm=DRIFT_PPM)
    m = clock_map.ClockMap(offset_s=OFFSET_S, drift_ppm=DRIFT_PPM)
    assert np.allclose(m.to_device(times), model.read(times), atol=1e-12, rtol=0)


def test_round_trip_is_identity():
    m = clock_map.ClockMap(offset_s=-12.5, drift_ppm=-30.0)
    sim_s = np.random.default_rng(0).uniform(0.0, 7200.0, size=500)
    assert np.max(np.abs(m.to_sim(m.to_device(sim_s)) - sim_s)) < 1e-9


def test_noise_free_pairs_recover_offset_and_drift():
    sim_s = np.linspace(0.0, 100.0, 201)
    fit = clock_map.fit_clock_map(device_reading(sim_s), sim_s)
    assert fit.offset_s == pytest.approx(OFFSET_S, abs=1e-12)
    assert fit.drift_ppm == pytest.approx(DRIFT_PPM, abs=1e-6)
    assert fit.residual_rms_s < 1e-12
    assert fit.n_pairs == 201
    assert fit.span_s == pytest.approx(100.0)


# Hand tolerance for the noisy case. 201 pairs, one every 0.5 s over 100 s, and
# 1 ms of noise (standard deviation sigma) on each device stamp.
#   spread of the sim times: S = sum (t - mean)^2, about N * T^2 / 12 = 167500 s^2.
#     Exactly, with t = 0.5 k: S = 0.25 * 2 * (1^2 + ... + 100^2)
#     = 0.5 * (100 * 101 * 201 / 6) = 169175 s^2
#   slope error:  sigma / sqrt(S) = 0.001 / 411.3 = 2.43e-6 = 2.43 ppm
#   offset error at t = 0: sigma * sqrt(1/N + mean^2 / S)
#     = 0.001 * sqrt(1/201 + 50^2 / 169175) = 0.001 * 0.1405 = 0.14 ms
# The test allows 4 of those standard errors.
N_PAIRS = 201
SPAN_S = 100.0
SIGMA_S = 1e-3
SPREAD = float(np.sum((np.linspace(0.0, SPAN_S, N_PAIRS) - SPAN_S / 2) ** 2))
SE_DRIFT_PPM = SIGMA_S / math.sqrt(SPREAD) * 1e6
SE_OFFSET_S = SIGMA_S * math.sqrt(1.0 / N_PAIRS + (SPAN_S / 2) ** 2 / SPREAD)


def test_hand_standard_errors_are_what_the_comment_says():
    assert SPREAD == pytest.approx(169175.0, abs=1e-6)
    assert SE_DRIFT_PPM == pytest.approx(2.43, abs=0.005)
    assert SE_OFFSET_S == pytest.approx(1.405e-4, abs=0.005e-4)


def noisy_pairs(seed):
    sim_s = np.linspace(0.0, SPAN_S, N_PAIRS)
    noise = np.random.default_rng(seed).normal(0.0, SIGMA_S, size=N_PAIRS)
    return device_reading(sim_s) + noise, sim_s


def test_noisy_pairs_recover_offset_and_drift_within_hand_tolerance():
    device_s, sim_s = noisy_pairs(seed=11)
    fit = clock_map.fit_clock_map(device_s, sim_s)
    assert abs(fit.drift_ppm - DRIFT_PPM) < 4 * SE_DRIFT_PPM
    assert abs(fit.offset_s - OFFSET_S) < 4 * SE_OFFSET_S
    # The leftover scatter is the stamp noise itself.
    assert fit.residual_rms_s == pytest.approx(SIGMA_S, rel=0.15)
    # Mapped back, the device stamps land on sim time to within the noise.
    assert np.sqrt(np.mean((fit.to_sim(device_s) - sim_s) ** 2)) < 1.2 * SIGMA_S


def test_spread_of_fits_over_many_seeds_matches_the_hand_formula():
    # An independent check on the hand formula: refit 400 noise draws and
    # compare the scatter of the answers with the predicted standard errors.
    drifts = []
    offsets = []
    for seed in range(400):
        fit = clock_map.fit_clock_map(*noisy_pairs(seed))
        drifts.append(fit.drift_ppm)
        offsets.append(fit.offset_s)
    assert np.std(drifts) == pytest.approx(SE_DRIFT_PPM, rel=0.15)
    assert np.std(offsets) == pytest.approx(SE_OFFSET_S, rel=0.15)
    assert np.mean(drifts) == pytest.approx(DRIFT_PPM, abs=4 * SE_DRIFT_PPM / math.sqrt(400))


def test_ignoring_drift_leaves_a_growing_error():
    # Why drift matters: correct only the offset and the error after 1000 s is
    # 50 ppm * 1000 s = 0.05 s, which at 2 m/s is a 10 cm position error.
    offset_only = clock_map.ClockMap(offset_s=OFFSET_S, drift_ppm=0.0)
    assert 1000.0 - offset_only.to_sim(device_reading(1000.0)) == pytest.approx(-0.05, abs=1e-9)


def test_pairs_from_symmetric_ntp_exchanges_recover_the_clock():
    log = sync_log(60.0, {"mode": "ntp", "offset_s": OFFSET_S, "drift_ppm": DRIFT_PPM,
                          "sync_delay_s": 0.005, "processing_s": 0.001},
                   {"sync_up": 1, "sync_down": 2})
    device_s, sim_s = clock_map.pairs_from_sync_log(log)
    assert len(device_s) == 61
    fit = clock_map.fit_clock_map(device_s, sim_s)
    assert fit.offset_s == pytest.approx(OFFSET_S, abs=1e-9)
    assert fit.drift_ppm == pytest.approx(DRIFT_PPM, abs=1e-4)


def test_path_asymmetry_shows_up_as_half_its_size_in_the_offset():
    # Uplink 7 ms, downlink 3 ms (asymmetry 4 ms). The sensor midpoint happens
    # (7 - 3) / 2 = 2 ms later than the tracker midpoint, so the fitted offset is
    # 2 ms too big, times (1 + 50e-6), so 0.0020001 s.
    log = sync_log(60.0, {"mode": "ntp", "offset_s": OFFSET_S, "drift_ppm": DRIFT_PPM,
                          "sync_delay_s": 0.005, "sync_asymmetry_s": 0.004, "processing_s": 0.001},
                   {"sync_up": 1, "sync_down": 2})
    fit = clock_map.fit_clock_map(*clock_map.pairs_from_sync_log(log))
    assert fit.offset_s - OFFSET_S == pytest.approx(0.002 * (1 + 50e-6), abs=1e-9)


def test_lost_exchanges_are_skipped():
    log = sync_log(60.0, {"mode": "ntp", "offset_s": OFFSET_S, "drift_ppm": DRIFT_PPM,
                          "sync_loss_probability": 0.3},
                   {"sync_up": 3, "sync_down": 4})
    device_s, sim_s = clock_map.pairs_from_sync_log(log)
    kept = sum(1 for row in log if not row["lost"])
    assert 0 < kept < len(log)
    assert len(device_s) == len(sim_s) == kept
    assert np.all(np.isfinite(device_s)) and np.all(np.isfinite(sim_s))


def test_unwrap_32_bit_microsecond_counter_by_hand():
    # A uint32 microsecond counter rolls over every 2^32 us = 4294.967296 s.
    modulus = 2**32
    ticks = np.array([modulus - 2_000_000, modulus - 1, 5, 1_000_000], dtype=np.int64)
    unwrapped = clock_map.unwrap_counter(ticks, modulus)
    assert unwrapped.tolist() == [modulus - 2_000_000, modulus - 1, modulus + 5, modulus + 1_000_000]
    assert clock_map.ticks_to_seconds(unwrapped, tick_hz=1_000_000)[2] == pytest.approx(4294.967301)


def test_unwrap_handles_two_rollovers_and_leaves_a_clean_counter_alone():
    modulus = 1000
    raw = np.array([900, 100, 500, 950, 10, 20])
    assert clock_map.unwrap_counter(raw, modulus).tolist() == [900, 1100, 1500, 1950, 2010, 2020]
    clean = np.arange(0, 900, 100)
    assert clock_map.unwrap_counter(clean, modulus).tolist() == clean.tolist()


@pytest.mark.parametrize(
    ("device_s", "sim_s"),
    [
        ([1.0], [1.0]),                      # one pair cannot give a drift
        ([1.0, 2.0], [5.0, 5.0]),            # no time span
        ([1.0, 2.0, 3.0], [1.0, 2.0]),       # length mismatch
        ([1.0, float("nan")], [1.0, 2.0]),   # not finite
    ],
)
def test_fit_rejects_unusable_pairs(device_s, sim_s):
    with pytest.raises(ValueError):
        clock_map.fit_clock_map(device_s, sim_s)


def test_clock_map_rejects_impossible_drift():
    with pytest.raises(ValueError):
        clock_map.ClockMap(offset_s=0.0, drift_ppm=-1e6)


def test_small_backward_step_is_not_a_wrap():
    # A reordered packet can arrive with a slightly older stamp. That is not a
    # rollover, so nothing is added (the drop of 10 is within the gap of 250).
    raw = np.array([100, 200, 190, 300])
    assert clock_map.unwrap_counter(raw, 1000).tolist() == [100, 200, 190, 300]


# Reboots versus rollovers. An ESP32 running Arduino micros() has a 32-bit
# microsecond counter. If it reboots after more than half a wrap period
# (2^31 us = 35.8 min), the counter drops from above 2^31 to near zero. The
# old rule (any drop bigger than half the modulus is a wrap) read that as a
# rollover and silently moved time forward by (2^32 - previous) ticks.
US_32 = 2**32


@pytest.mark.parametrize("before_reboot", [2_100_000_000, 2_200_000_000, 3_000_000_000])
def test_reboot_after_half_a_wrap_period_is_not_read_as_a_rollover(before_reboot):
    # 2.1e9 us is 35 min (just under half, 2^31 = 2.147e9: the old rule let
    # time run backward with no error), 2.2e9 us is 36.7 min and 3.0e9 us is
    # 50 min (the old rule jumped time forward 2095 s and 1295 s). After the
    # reboot the counter restarts near zero. A real rollover would need the
    # counter to be within a few seconds of 2^32 first, and here it is at
    # least 1295 s short of it, so every case must raise.
    ticks = np.array([before_reboot - 100_000, before_reboot, 1_000, 2_000], dtype=np.int64)
    with pytest.raises(ValueError, match="reboot"):
        clock_map.unwrap_counter(ticks, US_32)


def test_wrap_just_inside_the_gap_unwraps_and_reboot_just_outside_raises():
    # The rule: a drop counts as a wrap only when the step forward across the
    # wrap, (modulus - previous) + next, is at most max_gap_ticks. With the
    # default of 5,000,000 ticks (5 s of a microsecond counter):
    #   3,000,000 ticks before the top, then 1,999,999: step 4,999,999, a wrap.
    #   3,000,000 ticks before the top, then 2,000,001: step 5,000,001, refused.
    top = US_32 - 3_000_000
    inside = clock_map.unwrap_counter(np.array([top, 1_999_999]), US_32)
    assert inside.tolist() == [top, US_32 + 1_999_999]
    assert (inside[1] - inside[0]) == 4_999_999
    with pytest.raises(ValueError, match="reboot"):
        clock_map.unwrap_counter(np.array([top, 2_000_001]), US_32)
    # Passing the default explicitly changes nothing.
    same = clock_map.unwrap_counter(np.array([top, 1_999_999]), US_32, max_gap_ticks=5_000_000)
    assert same.tolist() == inside.tolist()
    # A caller with sparser samples can widen the gap, and then it is a wrap.
    wide = clock_map.unwrap_counter(np.array([top, 2_000_001]), US_32, max_gap_ticks=6_000_000)
    assert wide.tolist() == [top, US_32 + 2_000_001]


def test_large_backward_step_that_is_not_a_wrap_raises():
    # Modulus 1000, default gap = 1000 // 4 = 250 ticks. A drop of 300 is too
    # big to be a reordered packet and the forward step across the wrap,
    # (1000 - 600) + 300 = 700, is too big to be a rollover.
    with pytest.raises(ValueError, match="index 2"):
        clock_map.unwrap_counter(np.array([500, 600, 300]), 1000)


@pytest.mark.parametrize("gap", [0, -5, 500, 600])
def test_unwrap_rejects_a_gap_that_makes_wraps_and_backsteps_overlap(gap):
    # The gap must be positive and under half the modulus, so that a small
    # backward step and a rollover can never both explain the same drop.
    with pytest.raises(ValueError):
        clock_map.unwrap_counter(np.array([1, 2]), 1000, max_gap_ticks=gap)


# Reordered packets across the rollover. Near the wrap a late packet stamped
# just before the top can arrive after one stamped just after zero. Going
# from the new sample (near zero) to the late one (near the top) is a forward
# step of almost the whole modulus. Read naively, that adds a whole extra
# wrap (2^32 us = 4295 s) to the late packet and to every sample after it.


def test_reorder_straddling_the_wrap_keeps_each_packet_in_its_own_epoch():
    # Worked by hand: 500 and 1500 come after the wrap (add M), the late
    # packet M-500 comes from before it (add nothing).
    M = US_32
    raw = np.array([M - 2000, M - 1000, 500, M - 500, 1500], dtype=np.int64)
    assert clock_map.unwrap_counter(raw, M).tolist() == [M - 2000, M - 1000, M + 500, M - 500, M + 1500]


def test_late_pre_wrap_packet_after_the_first_post_wrap_sample():
    M = US_32
    raw = np.array([M - 2000, 10, M - 1000, 2000], dtype=np.int64)
    assert clock_map.unwrap_counter(raw, M).tolist() == [M - 2000, M + 10, M - 1000, M + 2000]


@pytest.mark.parametrize(
    "true_ticks",
    [
        # Two packets swapped across the wrap.
        [2**32 - 3000, 2**32 - 2000, 2**32 + 500, 2**32 - 500, 2**32 + 1500],
        # Two late packets in a row, then the stream catches up.
        [2**32 - 3000, 2**32 + 200, 2**32 - 900, 2**32 - 800, 2**32 + 1500, 2**32 + 2500],
        # Reorders before and after the wrap, but none across it.
        [2**32 - 3000, 2**32 - 1000, 2**32 - 1200, 2**32 + 1000, 2**32 + 700, 2**32 + 2000],
        # The very first sample is already past the wrap and the late one is
        # from before it. Values are counted from the first sample's epoch,
        # so the late packet comes out just below zero.
        [2**32 + 100, 2**32 - 100, 2**32 + 300],
    ],
)
def test_unwrap_recovers_the_true_spacing_of_a_reordered_stream(true_ticks):
    # Independent check: start from a known continuous counter (the truth),
    # wrap it with % by hand, unwrap it, and compare. The first sample may be
    # in any epoch, so the result may differ from the truth by a whole number
    # of wraps, but by the same number for every sample.
    M = US_32
    truth = np.array(true_ticks, dtype=np.int64)
    unwrapped = clock_map.unwrap_counter(truth % M, M)
    shift = unwrapped - truth
    assert np.all(shift == shift[0])
    assert shift[0] % M == 0


def test_late_packet_boundary_matches_the_reorder_limit():
    # Modulus 1000, default gap 250. A late packet across the wrap is at most
    # 250 ticks older than the sample before it, the same limit as an
    # ordinary reorder. From 100 to 850 is 250 ticks back across the wrap
    # (850 = -150 in the earlier epoch), so it is late. From 100 to 849 is
    # 251 ticks back, so it is not read as a late packet.
    assert clock_map.unwrap_counter(np.array([100, 850]), 1000).tolist() == [100, -150]
    assert clock_map.unwrap_counter(np.array([100, 849]), 1000).tolist() == [100, 849]


def test_two_wraps_with_a_late_packet_across_each():
    # Small counter so two wraps fit in a short list. Modulus 1000, default
    # gap 250. The truth counts up by 100 from 50; around each wrap the two
    # packets 950/1050 and 1950/2050 arrive swapped.
    truth = np.arange(50, 2600, 100)
    for a, b in ((950, 1050), (1950, 2050)):
        i, j = int(np.flatnonzero(truth == a)[0]), int(np.flatnonzero(truth == b)[0])
        truth[i], truth[j] = truth[j], truth[i]
    assert clock_map.unwrap_counter(truth % 1000, 1000).tolist() == truth.tolist()
