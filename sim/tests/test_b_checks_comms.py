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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Hand input slip: t4 is the TRACKER clock at reply arrival, so with t1 = 0, 10 ms each way and 1 ms processing t4 = 0.021, not 0.071 (0.071 wrongly adds the 0.05 s sensor offset). With t4 = 0.071 the formula correctly gives ((0.06 - 0) + (0.061 - 0.071)) / 2 = 0.025, i.e. a 60 ms downlink. With t4 = 0.021 it gives 0.05, and ntp_exchange on the same setup returns t4 = 0.021 and theta_s = 0.05 exactly.')
def test_b4_symmetric_clock_sync_is_exact():
    """With equal ten millisecond delays, the hand calculation gives offset 0.05 seconds exactly."""
    assert offset_from_timestamps(0.0, 0.06, 0.061, 0.071) == pytest.approx(0.05, abs=1e-12)
    sample = ntp_exchange(
        0.0,
        ClockModel(offset_s=0.05),
        SimulatedLink(LinkConfig(delay_s=0.01, delay_jitter_s=0.0, loss_model="none", loss_probability=0.0), seed=1),
        SimulatedLink(LinkConfig(delay_s=0.01, delay_jitter_s=0.0, loss_model="none", loss_probability=0.0), seed=2),
        seq=0,
        processing_s=0.001,
    )
    assert sample is not None
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="NTP expectation ignores two things the clock model requires: (1) a capture at t uses the latest exchange whose reply ARRIVED by t; the exchange started at t = 2k arrives at 2k + 0.021, so captures at t = 2k still use the previous sync and have error about 2e-4, not 0; (2) the sync's own error: theta = r (T + 0.0105) for an exchange started at T (offset at the exchange midpoint), so error = r (t - T - 0.0105), off from r (t - T) by 1.05e-6 > atol 1e-6. Correct values: 0 at t = 0, then 9.895e-5 at odd t and 1.9895e-4 at even t. The unsynced half passes.")
def test_b4_clock_drift_unsynced_and_ntp():
    """A one hundred ppm unsynced clock has error 100e-6 times t, while NTP error is drift times time since the last sync."""
    times = np.arange(0.0, 10.0 + 1.0, 1.0)
    unsynced = stamp_errors(times, {"mode": "unsynced", "offset_s": 0.0, "drift_ppm": 100.0}, {})
    np.testing.assert_allclose(unsynced, 100e-6 * times, atol=1e-12)
    ntp = stamp_errors(
        times,
        {"mode": "ntp", "offset_s": 0.0, "drift_ppm": 100.0, "sync_interval_s": 2.0, "sync_delay_s": 0.01, "sync_jitter_s": 0.0, "sync_loss_probability": 0.0},
        {"sync_up": 11, "sync_down": 12},
    )
    expected = 100e-6 * (times - 2.0 * np.floor(times / 2.0))
    np.testing.assert_allclose(ntp, expected, atol=1e-6)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Expectation equates the WHOLE p95 error with the clock shift v e = 0.05 m and ignores the noise floor: with the default 0.05 m sensor noise and perfect clocks this cell already has median p95 0.110 m, and with offset +/-0.05 s it is 0.128 / 0.140 m. the clock model says the shift is about v e, which holds: with noise_std_m = 1e-6 the median p95 is 0.050009 (+0.05 s) and 0.050001 (-0.05 s), and the noisy RMSE (0.0804, 0.0929) averages to sqrt(0.0706^2 + 0.05^2) = 0.0866.')
def test_b4_prediction_error_matches_speed_times_clock_offset():
    """A one metre per second target with a 0.05 second clock offset should have median p95 position error about 0.05 metres."""
    from sim.s6_recovery import RecoveryConfig

    config = RecoveryConfig()
    for offset in (0.05, -0.05):
        cell = CellSpec(target="cv", delay_s=0.1, clock=(("mode", "unsynced"), ("offset_s", offset)))
        result = run_cell(cell, [RECOVERY_V1], [1, 2, 3], config)
        assert _summary_median_p95(result) == pytest.approx(0.05, rel=0.10)


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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="sigma is zero in exact arithmetic but not bitwise in IEEE doubles: the p = 4 grid values 1/3 and 2/3 are not representable and fn's sum rounds, so single elementary effects differ by 1 to 2 ulp (x0: 3.999999999999999 to 4.000000000000002), giving sigma about 6e-16, 5e-16, 0, 3e-16. assert_array_equal demands exact 0. mu_star equals [4, 2, 0, 1] exactly and the ranking assertion passes.")
def test_b7_morris_linear_effects_are_exact():
    """For y = 4x0 + 2x1 + x3, Morris gives mu star four, two, zero, one, all sigma zero, ranked zero one three two."""
    result = morris_screen(_rowwise(_linear_y), k=4, r=20, levels=4, seed=0)
    np.testing.assert_array_equal(result["mu_star"], [4.0, 2.0, 0.0, 1.0])
    np.testing.assert_array_equal(result["sigma"], np.zeros(4))
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



# ---- Corrected versions of the four checks above whose hand inputs were wrong ----
# The originals stay as strict xfails (the record of the slip). These siblings keep the same
# physical question and the same tolerances, with the hand derivation fixed.

def test_b4_symmetric_clock_sync_is_exact_corrected():
    """t4 is the TRACKER clock at reply arrival: 0 + 0.01 up + 0.001 processing + 0.01 down = 0.021.
    Then theta = ((0.06 - 0) + (0.061 - 0.021)) / 2 = 0.05 exactly, the true offset."""
    assert offset_from_timestamps(0.0, 0.06, 0.061, 0.021) == pytest.approx(0.05, abs=1e-12)
    link = LinkConfig(delay_s=0.01, delay_jitter_s=0.0, loss_model="none", loss_probability=0.0)
    sample = ntp_exchange(0.0, ClockModel(offset_s=0.05), SimulatedLink(link, seed=1), SimulatedLink(link, seed=2),
                          seq=0, processing_s=0.001)
    assert sample is not None
    assert sample.theta_s == pytest.approx(0.05, abs=1e-12)


def test_b4_clock_drift_ntp_corrected():
    """Drift r = 100 ppm, exchanges started at T = 0, 2, 4, ..., 10 ms each way, 1 ms processing.
    The reply of the exchange started at T arrives at T + 0.021, and the offset it measures is the
    sensor offset at the midpoint of the two sensor stamps: theta = r (T + 0.0105). A capture at t uses
    the latest exchange whose reply arrived by t, so its error is r (t - T - 0.0105) (0 before any reply).
    Hand values: t = 1 gives 9.895e-5; t = 2 still uses T = 0 (reply only at 2.021), giving 1.9895e-4."""
    r = 100e-6
    times = np.arange(0.0, 10.0 + 1.0, 1.0)
    ntp = stamp_errors(
        times,
        {"mode": "ntp", "offset_s": 0.0, "drift_ppm": 100.0, "sync_interval_s": 2.0, "sync_delay_s": 0.01,
         "sync_jitter_s": 0.0, "sync_loss_probability": 0.0},
        {"sync_up": 11, "sync_down": 12},
    )
    expected = []
    for t in times:
        arrived = [T for T in np.arange(0.0, t + 1e-9, 2.0) if T + 0.021 <= t + 1e-12]
        expected.append(0.0 if not arrived else r * (t - max(arrived) - 0.0105))
    np.testing.assert_allclose(ntp, expected, atol=1e-12)
    assert ntp[1] == pytest.approx(9.895e-5, abs=1e-12) and ntp[2] == pytest.approx(1.9895e-4, abs=1e-12)


def test_b4_prediction_error_matches_speed_times_clock_offset_corrected():
    """With sensor noise removed (1 micrometre), the only error left on a 1 m/s straight line is the
    clock shift: a stamp off by e = +-0.05 s moves the predict-forward estimate by v e = 0.05 m."""
    from sim.s6_recovery import RecoveryConfig

    config = RecoveryConfig()
    for offset in (0.05, -0.05):
        cell = CellSpec(target="cv", delay_s=0.1, noise_std_m=1e-6, clock=(("mode", "unsynced"), ("offset_s", offset)))
        result = run_cell(cell, [RECOVERY_V1], [1, 2, 3], config)
        assert _summary_median_p95(result) == pytest.approx(0.05, rel=0.02)


def test_b7_morris_linear_effects_corrected():
    """For y = 4 x0 + 2 x1 + x3 every elementary effect of input i is exactly its slope, so mu star is
    (4, 2, 0, 1) and sigma is 0. Levels like 1/3 are not exact in binary, so sigma is zero only to
    rounding (about 1e-15), hence a 1e-12 tolerance instead of exact equality."""
    result = morris_screen(_rowwise(_linear_y), k=4, r=20, levels=4, seed=0)
    np.testing.assert_allclose(result["mu_star"], [4.0, 2.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result["sigma"], np.zeros(4), atol=1e-12)
    np.testing.assert_array_equal(np.argsort(-result["mu_star"]), [0, 1, 3, 2])
