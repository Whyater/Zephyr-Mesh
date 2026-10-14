import numpy as np
from sim.radio.interference import *

def test_timeline_and_noise():
    cfg = InterferenceConfig(3600, 1.0, 30.0)
    ep = interference_timeline(cfg, 10.0, np.random.default_rng(4))
    assert len(ep) > 0
    assert noise_floor_at(ep[0][0], -95, ep, cfg) == -65

def test_detector_hysteresis():
    d = DegradedLinkDetector(4, .5, 5, 2)
    for i in range(4):
        d.update(i, False, np.nan)
    assert d.state == "degraded"
    for i in range(4, 8):
        d.update(i, True, 20)
    assert d.state == "ok"



def test_episode_onsets_are_poisson_with_the_configured_rate():
    import numpy as np
    from sim.radio.interference import InterferenceConfig, interference_timeline
    cfg = InterferenceConfig(onset_rate_per_hour=6.0, mean_duration_s=5.0, noise_rise_db=20.0)
    episodes = interference_timeline(cfg, 1000 * 3600.0, np.random.default_rng(3))
    expected = 6000.0
    assert abs(len(episodes) - expected) < 4 * expected ** 0.5
    durations = np.array([e - s for s, e in episodes])
    assert abs(durations.mean() - 5.0) < 4 * 5.0 / len(durations) ** 0.5


def test_detector_catches_a_strong_episode_and_stays_quiet_on_a_clean_link():
    from sim.radio.channel import ChannelConfig
    from sim.radio.interference import (DegradedLinkDetector, InterferenceConfig, evaluate_detector,
                                        simulate_link_with_interference)
    # 20 m, strong link: SNR margin about 40 dB, so only a large noise rise can hurt it.
    channel = ChannelConfig(tx_power_dbm=19.5, pl0_db=40.2, exponent=2.0)
    strong = InterferenceConfig(onset_rate_per_hour=120.0, mean_duration_s=10.0, noise_rise_db=60.0)
    run = simulate_link_with_interference(channel, 20.0, 20.0, 3600.0, 75, strong, seed=11)
    result = evaluate_detector(run["times"], run["received"], run["snr_db"], run["episodes"],
                               DegradedLinkDetector(20, 0.3, 6.0, 20), grace_s=1.0)
    long_episodes = [e for e in run["episodes"] if e[1] - e[0] > 2.0]
    assert long_episodes and result["missed"] <= len(run["episodes"]) - len(long_episodes)
    assert result["false_triggers"] == 0
    quiet = InterferenceConfig(onset_rate_per_hour=0.0, mean_duration_s=1.0, noise_rise_db=60.0)
    clean = simulate_link_with_interference(channel, 20.0, 20.0, 3600.0, 75, quiet, seed=11)
    assert evaluate_detector(clean["times"], clean["received"], clean["snr_db"], [],
                             DegradedLinkDetector(20, 0.3, 6.0, 20))["false_triggers"] == 0


def test_grace_window_turns_a_late_trigger_into_a_detection():
    from sim.radio.interference import DegradedLinkDetector, evaluate_detector
    # Hand-built: 10 packets lost from t = 1.0 s at 10 Hz, episode (1.0, 1.4) ends before the window fills.
    times = [i / 10 for i in range(40)]
    received = [not (10 <= i < 20) for i in range(40)]
    snr = [30.0 if r else float("nan") for r in received]
    det = lambda: DegradedLinkDetector(10, 0.6, 6.0, 10)
    strict = evaluate_detector(times, received, snr, [(1.0, 1.4)], det())
    graced = evaluate_detector(times, received, snr, [(1.0, 1.4)], det(), grace_s=1.0)
    assert strict["missed"] == 1 and strict["false_triggers"] == 1
    assert graced["missed"] == 0 and graced["false_triggers"] == 0
