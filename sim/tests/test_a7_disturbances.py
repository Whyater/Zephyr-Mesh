import numpy as np

from sim.physics.disturbances import ConstantWind, GroundEffect, OUGust


def test_constant_wind_and_ground_effect_checks():
    wind = ConstantWind((1.0, -2.0, 0.5)).sample(0.0, 0.01, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(wind.wind_velocity_mps, [1.0, -2.0, 0.5])
    assert wind.thrust_scale == 1.0
    ground = GroundEffect(0.2)
    assert abs(ground.sample(0, 0.01, [0, 0, 20], [0, 0, 0]).thrust_scale - 1.0) < 1e-5
    assert ground.sample(0, 0.01, [0, 0, 0.2], [0, 0, 0]).thrust_scale == 16 / 15
    assert ground.sample(0, 0.01, [0, 0, -1], [0, 0, 0]).thrust_scale == 16 / 15
    values = [ground.sample(0, 0.01, [0, 0, z], [0, 0, 0]).thrust_scale for z in (0.2, 0.4, 1.0)]
    assert values[0] > values[1] > values[2]


def test_ou_stationarity_mean_variance_and_autocorrelation():
    sigma = np.array([1.0, 2.0, 3.0])
    gust = OUGust(np.zeros(3), sigma, 1.0, seed=7)
    samples = np.array([gust.sample(0, 0.01, [0, 0, 0], [0, 0, 0]).wind_velocity_mps for _ in range(100_000)])
    mean = samples.mean(axis=0)
    variance = samples.var(axis=0)
    a = np.exp(-0.01)
    # Effective sample size is N*(1-a)/(1+a), so use sigma/sqrt(N_eff).
    n_eff = len(samples) * (1 - a) / (1 + a)
    assert np.all(np.abs(mean) <= 3 * sigma / np.sqrt(n_eff))
    np.testing.assert_allclose(variance, sigma**2, rtol=0.03)
    for lag_s in (0.01, 0.5, 1.0, 2.0):
        lag = int(round(lag_s / 0.01))
        # Pool the three independent axes to reduce the finite-sample error.
        centered = samples - mean
        corr = np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered * centered)
        assert abs(corr - np.exp(-lag_s)) < 0.03


def test_ou_sampling_step_independence():
    def collect(step):
        gust = OUGust(np.zeros(3), np.ones(3), 1.0, seed=7)
        return np.array([gust.sample(0, step, [0, 0, 0], [0, 0, 0]).wind_velocity_mps[0] for _ in range(20_000)])

    coarse = collect(0.02)
    fine = collect(0.01)
    assert abs(coarse.var() - 1.0) < 0.15
    assert abs(fine.var() - 1.0) < 0.15
    c1 = np.corrcoef(coarse[:-50], coarse[50:])[0, 1]
    c2 = np.corrcoef(fine[:-100], fine[100:])[0, 1]
    assert abs(c1 - np.exp(-1.0)) < 0.08
    assert abs(c2 - np.exp(-1.0)) < 0.08
