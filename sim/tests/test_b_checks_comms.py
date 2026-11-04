"""Independent checks for comms, sync, sending and screening, written from the model equations before the implementation existed."""

import numpy as np
import pytest
from scipy.stats import chi2

from sim.estimation.base import ArmSpec, EstimatorSettings, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm, run_cell
from sim.estimation.morris import elementary_effects, morris_screen
from sim.estimation.policies import apply_sender, packet_bytes, quantize
from sim.link import LinkConfig, SimulatedLink
from sim.timesync import ClockModel, ntp_exchange, offset_from_timestamps, stamp_errors


def _summary_median_p95(result, arm_name="recovery"):
    """Accept the contract's summary shape while keeping the assertion about p95 explicit."""
    value = result["arms"][arm_name]["p95_median_m"]
    if isinstance(value, dict):
        if "median" in value:
            return float(value["median"])
        if "q50" in value:
            return float(value["q50"])
        return float(value["value"])
    return float(value)


def _rowwise(fn):
    """morris_screen evaluates fn on one design row (a 1D array of k inputs) at a time."""
    return lambda row: float(fn(np.atleast_2d(row))[0])


def _linear_y(x):
    return 4 * x[:, 0] + 2 * x[:, 1] + x[:, 3]


def _sobol_g(x):
    a = np.array([0.0, 1.0, 4.5, 9.0, 99.0])
    return np.prod((np.abs(4 * x - 2) + a) / (1 + a), axis=1)


# History: this check first used t4 = 0.071, which wrongly added the 0.05 s sensor offset to a
# TRACKER clock reading; it ran as a strict xfail until the hand input was fixed.
def test_b4_symmetric_clock_sync_is_exact():
    """With equal ten millisecond delays, the hand calculation gives offset 0.05 seconds exactly.

    Tracker clock = true time, sensor clock = true time + 0.05. The request leaves at t1 = 0
    (tracker clock), lands 0.01 s later, so the sensor stamps t2 = 0.01 + 0.05 = 0.06. The sensor
    waits 1 ms and replies at t3 = 0.061 (sensor clock). The reply lands at true time
    0 + 0.01 + 0.001 + 0.01 = 0.021, and t4 is read on the TRACKER clock, so t4 = 0.021.
    theta = ((t2 - t1) + (t3 - t4)) / 2 = (0.06 + 0.04) / 2 = 0.05, the true offset.
    """
    assert offset_from_timestamps(0.0, 0.06, 0.061, 0.021) == pytest.approx(0.05, abs=1e-12)
    link = LinkConfig(delay_s=0.01, delay_jitter_s=0.0, loss_model="none", loss_probability=0.0)
    sample = ntp_exchange(0.0, ClockModel(offset_s=0.05), SimulatedLink(link, seed=1), SimulatedLink(link, seed=2),
                          seq=0, processing_s=0.001)
    assert sample is not None
    assert (sample.t1, sample.t2, sample.t3, sample.t4) == pytest.approx((0.0, 0.06, 0.061, 0.021), abs=1e-12)
    assert sample.theta_s == pytest.approx(0.05, abs=1e-12)


def test_b4_asymmetric_delay_has_half_difference_error():
    """The hand formula is theta = e0 plus half the delay difference, so the two errors are plus and minus 0.005 seconds."""
    e0 = 0.05
    for d_up, d_down, expected in ((0.015, 0.005, 0.055), (0.005, 0.015, 0.045)):
        t1, p = 0.0, 0.001
        t4 = d_up + p + d_down
        t2, t3 = d_up + e0, d_up + e0 + p
        theta = offset_from_timestamps(t1, t2, t3, t4)
        assert theta == pytest.approx(expected, abs=1e-12)


# History: the NTP half first expected 100e-6 * (t - time of the last exchange START). That skips
# two things: a capture can only use a reply that has already ARRIVED, and the sync itself measures
# the clock at the exchange midpoint, not at its start. It ran as a strict xfail until fixed.
def test_b4_clock_drift_unsynced_and_ntp():
    """A one hundred ppm unsynced clock has error 100e-6 times t, while NTP error is drift times time since the clock was last measured.

    Drift r = 100e-6 s per s, so the sensor clock reads t (1 + r). Exchanges start at T = 0, 2, 4,
    ... (tracker clock = true time), 10 ms each way, 1 ms processing (real time on the sensor):
      t1 = T, t2 = (T + 0.01)(1 + r), t3 = (T + 0.011)(1 + r), t4 = T + 0.021.
      theta = ((t2 - t1) + (t3 - t4)) / 2 = (0.01 + 0.011 - 0.021 + r (2T + 0.021)) / 2 = r (T + 0.0105).
    So a sync measures the clock error at the midpoint T + 0.0105 of its two sensor stamps.
    A capture at t uses the newest exchange whose reply arrived by t (arrival T + 0.021).
      error(t) = r t - theta = r (t - T - 0.0105), or r t if no reply has arrived yet.
    t = 0: no reply yet, error 0. t = 1: uses T = 0, error 100e-6 * 0.9895 = 9.895e-5.
    t = 2: the T = 2 reply only lands at 2.021, so still T = 0, error 100e-6 * 1.9895 = 1.9895e-4.
    """
    r = 100e-6
    times = np.arange(0.0, 10.0 + 1.0, 1.0)
    unsynced = stamp_errors(times, {"mode": "unsynced", "offset_s": 0.0, "drift_ppm": 100.0}, {})
    np.testing.assert_allclose(unsynced, r * times, atol=1e-12)
    ntp = stamp_errors(
        times,
        {"mode": "ntp", "offset_s": 0.0, "drift_ppm": 100.0, "sync_interval_s": 2.0, "sync_delay_s": 0.01, "sync_jitter_s": 0.0, "sync_loss_probability": 0.0},
        {"sync_up": 11, "sync_down": 12},
    )
    expected = []
    for t in times:
        # Captures are whole seconds and arrivals end in .021, so these float comparisons have no ties.
        arrived = [start for start in np.arange(0.0, 11.0, 2.0) if start + 0.021 <= t]
        expected.append(r * t if not arrived else r * (t - max(arrived) - 0.0105))
    # Errors are about 1e-4 s and rounding in them is about 1e-20 s, so 1e-12 is a loose bound
    # that still fails on the old 1.05e-6 slip by a factor of a million.
    np.testing.assert_allclose(ntp, expected, atol=1e-12)
    assert ntp[0] == 0.0
    assert ntp[1] == pytest.approx(9.895e-5, abs=1e-12)
    assert ntp[2] == pytest.approx(1.9895e-4, abs=1e-12)


# History: this check first used the default 0.05 m sensor noise and expected the WHOLE p95 error
# to be the clock shift 0.05 m; the noise alone already gives about 0.11 m. It ran as a strict xfail
# until the noise was taken out so only the clock shift is left.
def test_b4_prediction_error_matches_speed_times_clock_offset():
    """A one metre per second target with a 0.05 second clock offset should have median p95 position error about 0.05 metres.

    The cv target moves in a straight line at 1 m/s. A stamp that is wrong by e seconds tells the
    tracker the target was there e seconds earlier or later than it really was, so on a straight line
    the estimate slides by v e = 1 * 0.05 = 0.05 m along the path, for either sign of e.
    Sensor noise is set to 1 micrometre so that shift is the only error left. Limiting case: with
    e = 0 the error should be almost nothing, which rules out the 0.05 m coming from somewhere else
    (the 0.1 s link delay times 1 m/s divided by 2 is also 0.05, for example).
    Tolerance: 1 mm, a thousand times the noise; the filter's start-up samples are a small share of
    each run, so they do not move the 95th percentile.
    """
    from sim.s6_recovery import RecoveryConfig

    config = RecoveryConfig()
    for offset, expected in ((0.05, 0.05), (-0.05, 0.05), (0.0, 0.0)):
        cell = CellSpec(target="cv", delay_s=0.1, noise_std_m=1e-6, clock=(("mode", "unsynced"), ("offset_s", offset)))
        result = run_cell(cell, [RECOVERY_V1], [1, 2, 3], config)
        assert _summary_median_p95(result) == pytest.approx(expected, abs=0.001)


def test_b5_quantization_uniform_error_and_int16_codes():
    """Uniform samples across many quantizer steps have per-axis error standard deviation q divided by square root of twelve, 2.887e-4 metres."""
    rng = np.random.default_rng(12345)
    z = rng.uniform(-5.0, 5.0, size=(100_000, 3))
    codes, z_hat = quantize(z, 0.001)
    assert codes.dtype == np.int16
    np.testing.assert_allclose(np.std(z_hat - z, axis=0), 0.001 / np.sqrt(12), rtol=0.02)


def test_b5_zero_threshold_event_sender_sends_every_sample():
    """A zero metre event threshold sends every sample, including noisy samples and exact repeats, so the expected indices are all indices."""
    settings = EstimatorSettings()
    times = np.arange(20, dtype=float) * 0.1
    noisy = np.column_stack((times, np.sin(times), np.zeros_like(times)))
    repeats = np.zeros((times.size, 3))
    for measurements in (noisy, repeats):
        out = apply_sender(times, measurements, {"kind": "event", "delta_m": 0.0}, settings, seed=7)
        np.testing.assert_array_equal(out.send_indices, np.arange(times.size))


def test_b5_periodic_sender_and_packet_data_sizes():
    """A ten hertz sender on a fifty hertz stream sends every fifth sample, and int16 positions save exactly six bytes."""
    times = np.arange(50, dtype=float) / 50.0
    measurements = np.zeros((times.size, 3))
    out = apply_sender(times, measurements, {"kind": "periodic", "rate_hz": 10.0}, EstimatorSettings(), seed=0)
    np.testing.assert_array_equal(out.send_indices, np.arange(0, 50, 5))
    assert packet_bytes(False)["data"] - packet_bytes(True)["data"] == 6


def test_b6_gate_rejects_outliers_and_keeps_most_clean_packets():
    """A one metre jump has NIS about (1/0.06)^2, above chi-square 0.99 at 3 degrees of freedom, so over 95 percent of outliers reject and clean rejection is 0.2 to 3 percent."""
    from sim.s6_recovery import RecoveryConfig

    config = RecoveryConfig()
    cell = CellSpec(target="cv", noise_model="gaussian_outliers", outlier_probability=0.05, outlier_m=1.0, noise_std_m=0.05)
    arm = ArmSpec("gated", gate_probability=0.99, r_std_m=0.05)
    gated_outliers = gated_clean = outliers = clean = 0
    for seed in range(20):
        bundle = deliver_cell(cell, seed, config)
        outcome = run_arm(bundle, cell, arm, config)
        gated_outliers += outcome.extras["gated_outliers"]
        gated_clean += outcome.extras["gated_clean"]
        outliers += len(bundle.outlier_seqs)
        clean += bundle.capture_count - len(bundle.outlier_seqs)
    assert outliers > 0 and clean > 0
    assert gated_outliers / outliers > 0.95
    assert 0.002 <= gated_clean / clean <= 0.03
    assert chi2.ppf(0.99, 3) == pytest.approx(11.34, abs=0.01)


# History: this check first demanded sigma == 0 bit for bit. In exact arithmetic it is 0, but the
# grid values 1/3 and 2/3 are not exact in binary, so single effects differ in the last digit.
# It ran as a strict xfail until the exact compare got a rounding tolerance.
def test_b7_morris_linear_effects_are_exact():
    """For y = 4x0 + 2x1 + x3, Morris gives mu star four, two, zero, one, all sigma zero, ranked zero one three two.

    Every elementary effect of input i is (y(x + Delta e_i) - y(x)) / Delta, which for a straight-line
    y is exactly its slope, so mu star = (4, 2, 0, 1) and sigma = 0. Tolerance: y is at most 7, where
    one rounding step is about 9e-16; a difference of two rounded values divided by Delta = 2/3
    (levels 4) is off by at most about 1e-14. 1e-12 covers that and is still a trillion times smaller
    than the smallest nonzero effect (1).
    """
    result = morris_screen(_rowwise(_linear_y), k=4, r=20, levels=4, seed=0)
    np.testing.assert_allclose(result["mu_star"], [4.0, 2.0, 0.0, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["sigma"], np.zeros(4), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.argsort(-result["mu_star"]), [0, 1, 3, 2])


def test_b7_morris_product_has_symmetric_interactions():
    """For y = x0 times x1, both interaction effects have positive sigma and equal mu star within sampling noise."""
    result = morris_screen(_rowwise(lambda x: x[:, 0] * x[:, 1]), k=2, r=200, levels=4, seed=0)
    assert np.all(result["sigma"] > 0)
    assert result["mu_star"][0] == pytest.approx(result["mu_star"][1], rel=0.10)


def test_b7_morris_sobol_g_function_ranking():
    """For the Sobol g function with a equal to zero, one, 4.5, 9, 99, the mu star ranking is zero one two three four."""
    result = morris_screen(_rowwise(_sobol_g), k=5, r=100, levels=4, seed=0)
    np.testing.assert_array_equal(np.argsort(-result["mu_star"]), [0, 1, 2, 3, 4])
