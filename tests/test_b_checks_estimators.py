"""Independent checks for estimators, written from the model equations before the implementation existed."""
# Independent checks for estimators, written from the model equations before the implementation existed.
import numpy as np
import pytest

from sim.estimation.base import ArmSpec, EstimatorSettings, NAIVE, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm
from sim.estimation.factory import make_estimator
from sim.estimation.consistency import AdaptiveEstimator, nis_band
from sim.s6_recovery import RecoveryConfig, seed_list
from sim.tracker import ConstantVelocityTracker
from sim.estimation.targets import make_target


def _config():
    return RecoveryConfig()


def test_b1_retrodiction_matches_offline_cv():
    """A fixed lag re-filter is the same as sorting received packets by capture time; no packet is dropped and every scored error matches to 1e-9 m."""
    cell = CellSpec(target="cv", rate_hz=10.0, delay_s=0.3, jitter_s=0.15,
                    loss_probability=0.0, noise_std_m=0.0)
    bundle = deliver_cell(cell, seed=7, config=_config())
    arm = ArmSpec("retro", stamping="capture", late_policy="retrodict")
    out = run_arm(bundle, cell, arm, _config())
    assert out.refiltered_count > 0
    assert out.dropped_count == 0

    tracker = ConstantVelocityTracker((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0,
                                       position_variance_m2=1.0,
                                       velocity_variance_m2ps2=1.0,
                                       acceleration_spectral_density=0.1,
                                       measurement_std_m=0.05)
    qs = np.asarray(out.query_times_s)
    scored = qs >= _config().burn_in_s - 1e-9
    expected = []
    for t in qs[scored]:
        # Rebuild the offline filter at each query so the capture-time sort is exact.
        offline = ConstantVelocityTracker((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0,
                                          position_variance_m2=1.0,
                                          velocity_variance_m2ps2=1.0,
                                          acceleration_spectral_density=0.1,
                                          measurement_std_m=0.05)
        ready = [d for d in bundle.deliveries if d.receive_time_s <= t + 1e-12]
        ready.sort(key=lambda d: d.capture_time_s)
        for d in ready:
            offline.update(d.measurement_m, d.capture_time_s)
        p = offline.predict_at(float(t)).position_m
        expected.append(float(np.linalg.norm(p - np.array([t, 0.0, 0.0]))))
    actual = np.asarray(out.errors_m)
    if actual.size == qs.size:
        actual = actual[scored]
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-9)


def test_b1_retrodiction_equals_recovery_without_jitter():
    """With zero jitter, retrodiction and recovery-v1 see the same ordered history, so p95, max, and RMSE are exactly equal."""
    cell = CellSpec(target="cv", rate_hz=10.0, delay_s=0.1, jitter_s=0.0,
                    loss_probability=0.3)
    bundle = deliver_cell(cell, seed=11, config=_config())
    retro = run_arm(bundle, cell, ArmSpec("retro", stamping="capture", late_policy="retrodict"), _config())
    rec = run_arm(bundle, cell, RECOVERY_V1, _config())
    assert retro.p95_error_m == rec.p95_error_m
    assert retro.max_error_m == rec.max_error_m
    assert retro.rmse_m == rec.rmse_m


def test_b1_zero_window_equals_recovery():
    """A zero retrodiction window drops every late packet just like recovery-v1, so all three error summaries are exactly equal."""
    cell = CellSpec(target="cv", rate_hz=10.0, delay_s=0.1, jitter_s=0.0,
                    loss_probability=0.3)
    bundle = deliver_cell(cell, seed=12, config=_config())
    retro = run_arm(bundle, cell, ArmSpec("retro", stamping="capture", late_policy="retrodict", retro_window_s=0.0), _config())
    rec = run_arm(bundle, cell, RECOVERY_V1, _config())
    assert retro.p95_error_m == rec.p95_error_m
    assert retro.max_error_m == rec.max_error_m
    assert retro.rmse_m == rec.rmse_m


def test_b2_ct_known_turn_rate_is_exact():
    """A known 1 rad/s circular turn has an exact CT transition, so p95 position error stays below 1e-3 m at all tested delays."""
    for delay in (0.0, 0.1, 0.3):
        cell = CellSpec(target="ct", rate_hz=10.0, delay_s=delay, jitter_s=0.0,
                        loss_probability=0.0, noise_std_m=0.0)
        arm = ArmSpec("ct", model="ct", stamping="capture", late_policy="drop",
                      model_options=(("known_turn_rate", 1.0),))
        out = run_arm(deliver_cell(cell, seed=3, config=_config()), cell, arm, _config())
        assert out.p95_error_m < 1e-3


def test_b2_ca_zero_lag_and_cv_alpha_beta_lag():
    """A CA model fits constant acceleration with less than 1e-4 m error, while CV has the hand-computed 0.0364 m alpha-beta lag within 5 percent."""
    T, a, q, R = 0.1, 1.0, 0.1, 0.05**2
    F = np.array([[1., T], [0., 1.]])
    Q = q * np.array([[T**3/3, T**2/2], [T**2/2, T]])
    P = np.eye(2)
    for _ in range(20000):
        Pm = F @ P @ F.T + Q
        K = Pm[:, 0] / (Pm[0, 0] + R)
        H = np.array([[1., 0.]])
        I = np.eye(2)
        Pn = (I - np.outer(K, H)) @ Pm @ (I - np.outer(K, H)).T + R * np.outer(K, K)
        P = (Pn + Pn.T) / 2
    alpha, beta = K[0], K[1] * T
    assert alpha == pytest.approx(0.469, abs=0.0005)
    assert beta == pytest.approx(0.146, abs=0.0005)

    settings = EstimatorSettings(measurement_std_m=0.05, q_acc_psd=0.1, q_jerk_psd=0.1)
    ca = make_estimator("ca", settings, (0., 0., 0.))
    cv = make_estimator("cv", settings, (0., 0., 0.))
    for k in range(151):
        z = np.array([a * (k*T)**2 / 2, 0., 0.])
        ca.update(z, k*T)
        cv.update(z, k*T)
    truth = a * (150*T)**2 / 2
    ca_err = abs(ca.predict_at(150*T).position_m[0] - truth)
    cv_err = abs(cv.predict_at(150*T).position_m[0] - truth)
    bx = (1-alpha) * a * T**2 / beta
    assert ca_err < 1e-4
    assert cv_err == pytest.approx(bx, rel=0.05)


def test_b2_imm_small_penalty_on_straight_line():
    """On a straight noisy line, IMM p95 should be within 0.02 m of recovery-v1 across ten common seeds."""
    cell = CellSpec(target="cv", rate_hz=10.0, delay_s=0.1, jitter_s=0.0,
                    loss_probability=0.0, noise_std_m=0.05)
    imm = ArmSpec("imm", model="imm", stamping="capture", late_policy="drop")
    imm_p95, rec_p95 = [], []
    for seed in seed_list(10):
        bundle = deliver_cell(cell, seed=seed, config=_config())
        imm_p95.append(run_arm(bundle, cell, imm, _config()).p95_error_m)
        rec_p95.append(run_arm(bundle, cell, RECOVERY_V1, _config()).p95_error_m)
    diff = float(np.median(imm_p95) - np.median(rec_p95))
    assert diff < 0.02, f"IMM median p95 penalty was {diff:.6f} m"


def test_b2_imm_turn_probability_rises():
    """The IMM CT probability should rise by at least 0.3 during the pi turn relative to the first straight segment."""
    target = make_target("mixed")
    T = 0.1
    times = np.arange(0.0, 8.0 + T/2, T)
    rng = np.random.default_rng(19)
    z = target.position(times) + rng.normal(0.0, 0.02, size=(times.size, 3))
    est = make_estimator("imm", EstimatorSettings(measurement_std_m=0.02), target.start_position, r_std_m=0.02)
    probs = []
    for t, measurement in zip(times, z):
        est.update(measurement, float(t))
        probs.append(est.mode_probabilities[1])
    probs = np.asarray(probs)
    straight = probs[(times >= 1.0) & (times <= 4.0)]
    turn = probs[(times >= 5.0) & (times <= 4.0 + np.pi)]
    assert float(turn.mean() - straight.mean()) >= 0.3


def _nis_series(r_std, seed=23):
    """Generate NIS values for the seeded CWNA truth under a selected assumed noise."""
    target = make_target("cwna", seed=seed)
    times = np.arange(0.0, 400.0 + 0.05, 0.1)
    truth = target.position(times)
    rng = np.random.default_rng(seed + 1000)
    measurements = truth + rng.normal(0.0, 0.05, size=truth.shape)
    est = make_estimator("cv", EstimatorSettings(measurement_std_m=r_std), target.start_position,
                         r_std_m=r_std)
    nis = []
    for t, m in zip(times, measurements):
        nis.append(est.nis(m, float(t)))
        est.update(m, float(t))
    return np.asarray(nis), times


def test_b3_correctly_tuned_nis_and_windows():
    """Correct q and R give mean NIS near the chi-square-3 mean of 3, with 10-sample window coverage between 0.92 and 0.98."""
    nis, times = _nis_series(0.05)
    post = nis[times >= 5.0]
    assert 2.85 <= float(post.mean()) <= 3.15
    low, high = nis_band(10)
    means = np.array([post[i:i+10].mean() for i in range(0, len(post)-9, 10)])
    fraction = float(np.mean((means >= low) & (means <= high)))
    assert 0.92 <= fraction <= 0.98


def test_b3_nis_scales_with_assumed_measurement_noise():
    """Assuming R four times too small makes mean NIS exceed 10, while assuming R twenty-five times too large makes it below 1."""
    small, times = _nis_series(0.2 * 0.05)
    large, _ = _nis_series(5.0 * 0.05)
    assert float(small[times >= 5.0].mean()) > 10.0
    assert float(large[times >= 5.0].mean()) < 1.0


def test_b3_adaptive_q_recovers_ct():
    """A CV filter with q 100x too small cannot follow a CT turn; q adaptation raises q_scale above 1 and lowers p95 error."""
    cell = CellSpec(target="ct", rate_hz=10.0, delay_s=0.0, jitter_s=0.0,
                    loss_probability=0.0, noise_std_m=0.05)
    config = _config()
    bundle = deliver_cell(cell, seed=41, config=config)
    adapted_arm = ArmSpec("adapt", model="cv", stamping="capture", late_policy="drop",
                          adaptive="q", q_factor=0.01)
    plain_arm = ArmSpec("plain", model="cv", stamping="capture", late_policy="drop",
                        adaptive=None, q_factor=0.01)
    adapted = run_arm(bundle, cell, adapted_arm, config)
    plain = run_arm(bundle, cell, plain_arm, config)
    assert adapted.extras["final_q_scale"] > 1.0
    assert adapted.p95_error_m < plain.p95_error_m

