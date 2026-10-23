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
    from sim.radio.interference import (
        DegradedLinkDetector,
        InterferenceConfig,
        evaluate_detector,
        simulate_link_with_interference,
    )
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


# Fixed detector: a lost packet counts as low SNR, so a total blackout is caught
# without waiting for the healthy readings to slide out of the window.

def _blackout_trigger_packet(detector, healthy=40, lost=40, snr=30.0):
    """Feed `healthy` good packets at 10 Hz, then `lost` lost ones.
    Return how many lost packets it took to reach "degraded" (None if never)."""
    for i in range(healthy):
        detector.update(i / 10, True, snr)
    assert detector.state == "ok"
    for k in range(1, lost + 1):
        if detector.update((healthy + k - 1) / 10, False, float("nan")) == "degraded":
            return k
    return None


def test_total_blackout_triggers_within_half_the_window():
    import math

    from sim.radio.interference import DegradedLinkDetector
    window, loss_threshold = 20, 0.3
    k = _blackout_trigger_packet(DegradedLinkDetector(window, loss_threshold, 6.0, 3))
    # Bound: the loss rule needs ceil(0.3 x 20) = 6 lost packets and a median of
    # 20 values only falls when half of them are low, so 10 lost packets.
    bound = max(math.ceil(loss_threshold * window), math.ceil(window / 2))
    assert k is not None and k <= bound
    # Hand check: with 9 lost, the 10th and 11th smallest of the 20 values are
    # both 30 dB, median 30 > 6, no trigger.  With 10 lost, the 10th smallest is
    # -inf, median (-inf + 30) / 2 = -inf <= 6, trigger.  Old code: 20.
    assert k == 10


def test_odd_window_blackout_triggers_at_the_hand_computed_packet():
    from sim.radio.interference import DegradedLinkDetector
    # Window 5, loss threshold 0.2: the loss rule is met by 1 lost (an exact
    # tie, 1/5 = 20 %), and the median of 5 is the 3rd smallest, which is -inf
    # once 3 are lost, so the median binds.
    assert _blackout_trigger_packet(DegradedLinkDetector(5, 0.2, 6.0, 3)) == 3
    # Loss threshold 0.8 binds instead: 4 of 5 lost.
    assert _blackout_trigger_packet(DegradedLinkDetector(5, 0.8, 6.0, 3)) == 4


def test_short_blackout_is_caught_after_the_fix():
    from sim.radio.interference import DegradedLinkDetector, evaluate_detector
    # 1.2 s blackout (12 packets) at 10 Hz after 4 s of healthy 30 dB packets.
    times = [i / 10 for i in range(100)]
    received = [not (40 <= i < 52) for i in range(100)]
    snr = [30.0 if r else float("nan") for r in received]
    result = evaluate_detector(times, received, snr, [(4.0, 5.2)], DegradedLinkDetector(20, 0.3, 6.0, 3))
    assert result["missed"] == 0 and result["false_triggers"] == 0
    assert abs(result["trigger_delays_s"][0] - 0.9) < 1e-9  # 10th lost packet, at t = 4.9 s
    # 0.8 s (8 packets) is still too short for a 20-packet median, so this
    # detector misses it.  Blackouts that short are left to the separate
    # lost-link timeout (sim/radio/heartbeat.py).
    received = [not (40 <= i < 48) for i in range(100)]
    snr = [30.0 if r else float("nan") for r in received]
    short = evaluate_detector(times, received, snr, [(4.0, 4.8)], DegradedLinkDetector(20, 0.3, 6.0, 3), grace_s=2.0)
    assert short["missed"] == 1


def test_legacy_mode_keeps_the_old_received_only_median():
    from sim.radio.interference import DegradedLinkDetector
    legacy = DegradedLinkDetector(20, 0.3, 6.0, 3, lost_packets_count_as_low_snr=False)
    assert _blackout_trigger_packet(legacy) == 20


def test_heavy_loss_at_good_snr_now_counts_as_degraded():
    from sim.radio.interference import DegradedLinkDetector
    # Every other packet lost, the ones that arrive at 30 dB.  The fix treats
    # half the window as low SNR, so this triggers; the old rule did not.
    pattern = [i % 2 == 0 for i in range(40)]
    new = DegradedLinkDetector(20, 0.3, 6.0, 3)
    old = DegradedLinkDetector(20, 0.3, 6.0, 3, lost_packets_count_as_low_snr=False)
    for i, r in enumerate(pattern):
        new.update(i / 10, r, 30.0 if r else float("nan"))
        old.update(i / 10, r, 30.0 if r else float("nan"))
    assert new.state == "degraded" and old.state == "ok"


def test_detector_matches_an_independent_reference_on_random_traffic():
    import math
    import statistics
    from fractions import Fraction

    from sim.radio.interference import DegradedLinkDetector
    rng = np.random.default_rng(107)

    def reference(seq, window, lt, st, mn):
        # Written separately from the class: plain lists and statistics.median.
        state, out, items = "ok", [], []
        for r, s in seq:
            items = (items + [(r, s)])[-window:]
            if len(items) >= mn:
                # Exact fractions, so a tie such as 1 lost of 5 at 0.2 is exactly 1/5.
                loss = Fraction(sum(1 for x, _ in items if not x), len(items))
                vals = [(-math.inf if not x else v) for x, v in items if (not x) or math.isfinite(v)]
                low = not vals or statistics.median(vals) <= st
                if state == "ok" and loss >= Fraction(str(lt)) and low:
                    state = "degraded"
                elif state == "degraded" and loss < Fraction(str(lt)) / 2:
                    state = "ok"
            out.append(state)
        return out

    for trial in range(30):
        window = int(rng.integers(3, 25))
        lt = float(rng.choice([0.2, 0.3, 0.5, 0.7]))
        p_loss = float(rng.uniform(0, 0.8))
        seq = [(bool(rng.random() >= p_loss), float(rng.normal(8, 6))) for _ in range(300)]
        det = DegradedLinkDetector(window, lt, 6.0, 3)
        got = [det.update(i / 10, r, s if r else float("nan")) for i, (r, s) in enumerate(seq)]
        assert got == reference(seq, window, lt, 6.0, 3), trial


def test_received_packets_with_nonfinite_snr_are_still_skipped():
    from sim.radio.interference import DegradedLinkDetector
    # A received packet whose SNR is NaN carries no SNR reading, before and after
    # the fix: here 4 healthy, 2 lost and 4 NaN-SNR received packets leave six
    # values (-inf, -inf, 30, 30, 30, 30), median 30: the loss rule (20 % >= 15 %)
    # passes, so only the median blocks the trigger.  Counting the NaN packets
    # as -inf would give a median of -inf and trigger here.
    det = DegradedLinkDetector(10, 0.15, 6.0, 3)
    for i, (r, s) in enumerate([(True, 30.0)] * 4 + [(False, float("nan"))] * 2 + [(True, float("nan"))] * 4):
        det.update(i / 10, r, s)
    assert det.state == "ok"
    # Two more lost packets push out two healthy readings: (-inf x 4, 30, 30), median -inf.
    det.update(1.0, False, float("nan")); det.update(1.1, False, float("nan"))
    assert det.state == "degraded"


def test_loss_exactly_at_the_trigger_threshold_triggers():
    from sim.radio.interference import DegradedLinkDetector
    # 1 lost of 5 is exactly 20 %, so "loss >= 20 %" must pass.  Every received
    # packet reads 0 dB (below the 6 dB SNR threshold), so only the loss rule
    # decides.  The old code computed 1 - 4/5 in floating point: 4/5 is stored
    # as 0.80000000000000004 and 0.2 as 0.20000000000000001, so 1 - 0.8 came
    # out as 0.19999999999999996 and missed the threshold by about 5e-17.
    det = DegradedLinkDetector(5, 0.2, 6.0, 5)
    for i, r in enumerate([True, True, True, True, False]):
        det.update(i / 10, r, 0.0 if r else float("nan"))
    assert det.state == "degraded"


def test_loss_exactly_at_the_clear_threshold_stays_degraded():
    from sim.radio.interference import DegradedLinkDetector
    # Threshold 0.2 clears only below 0.1.  After a full blackout of 10, nine
    # received packets leave 1 lost of 10, exactly 10 %, which is not below
    # 10 %, so the state must stay degraded.  Old code: 1 - 0.9 rounds to
    # 0.09999999999999998 < 0.1 and cleared one packet early.
    det = DegradedLinkDetector(10, 0.2, 6.0, 3)
    for i in range(10):
        det.update(i / 10, False, float("nan"))
    assert det.state == "degraded"
    for i in range(10, 19):
        det.update(i / 10, True, 30.0)
    assert det.state == "degraded"
    det.update(1.9, True, 30.0)  # 0 lost of 10 now: clears
    assert det.state == "ok"
