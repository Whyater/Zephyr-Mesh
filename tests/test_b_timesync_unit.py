"""Unit tests for the sensor clock model and the NTP-style sync (sim/timesync.py)."""
import numpy as np
import pytest

from sim.link import LinkConfig, SimulatedLink
from sim.timesync import (
    ClockModel,
    ntp_exchange,
    offset_from_timestamps,
    round_trip,
    stamp_errors,
    stamp_errors_with_log,
    sync_log,
)

STREAMS = {"clock": 11, "sync_up": 22, "sync_down": 33}


def _links(d_up, d_down):
    return (SimulatedLink(LinkConfig(delay_s=d_up), seed=1),
            SimulatedLink(LinkConfig(delay_s=d_down), seed=2))


def test_offset_hand_example_symmetric():
    # Hand numbers: tracker sends at t1 = 10. Sensor clock is 0.05 s ahead.
    # Request arrives at true 10.01, so t2 = 10.01 + 0.05 = 10.06.
    # Reply leaves 0.001 s later: t3 = 10.061. It lands at true 10.011 + 0.01 = 10.021.
    e0, d, p = 0.05, 0.01, 0.001
    t1, t2, t3, t4 = 10.0, 10.0 + d + e0, 10.0 + d + p + e0, 10.0 + d + p + d
    assert abs(offset_from_timestamps(t1, t2, t3, t4) - 0.05) < 1e-12
    assert abs(round_trip(t1, t2, t3, t4) - 2 * d) < 1e-12


@pytest.mark.parametrize("asym", [0.02, -0.02])
def test_asymmetry_error_is_half_the_difference(asym):
    e0, p, mean = 0.05, 0.001, 0.02
    d_up, d_down = mean + asym / 2, mean - asym / 2
    t1, t2, t3, t4 = 3.0, 3.0 + d_up + e0, 3.0 + d_up + p + e0, 3.0 + d_up + p + d_down
    assert abs(offset_from_timestamps(t1, t2, t3, t4) - (e0 + asym / 2)) < 1e-12
    # Same result through the link-based exchange (zero drift).
    up, down = _links(d_up, d_down)
    sample = ntp_exchange(3.0, ClockModel(offset_s=e0), up, down, 0, processing_s=p)
    assert abs(sample.theta_s - e0 - asym / 2) < 1e-12


def test_unsynced_error_is_offset_plus_drift():
    t = np.linspace(0.0, 60.0, 121)
    params = {"mode": "unsynced", "offset_s": 0.03, "drift_ppm": 150.0}
    err = stamp_errors(t, params, STREAMS)
    assert np.max(np.abs(err - (0.03 + 150.0e-6 * t))) < 1e-12


def test_perfect_mode_is_zero():
    assert np.all(stamp_errors(np.arange(5.0), None, None) == 0.0)


def test_ntp_sawtooth_with_drift():
    # Symmetric, no jitter, no loss. theta from exchange k (sent at T_k = k I) is
    #   e0 + r (T_k + d_up + p/2),
    # so a capture at t that uses exchange k has error r (t - T_k - d_up - p/2).
    # The reply of exchange k lands at A_k = T_k + d_up + p + d_down.
    r, interval, d, p, e0 = 100e-6, 1.0, 0.005, 0.001, 0.02
    params = {"mode": "ntp", "offset_s": e0, "drift_ppm": r * 1e6, "sync_interval_s": interval,
              "sync_delay_s": d, "processing_s": p}
    arrival = lambda k: k * interval + d + p + d
    just_before = np.array([arrival(k) - 1e-9 for k in range(1, 6)])
    just_after = np.array([arrival(k) + 1e-9 for k in range(0, 5)])
    err_before = stamp_errors(just_before, params, STREAMS)
    err_after = stamp_errors(just_after, params, STREAMS)
    # Before reply k lands, exchange k-1 is in use.
    expect_before = r * (just_before - (np.arange(1, 6) - 1) * interval - d - p / 2)
    expect_after = r * (just_after - np.arange(0, 5) * interval - d - p / 2)
    assert np.max(np.abs(err_before - expect_before)) < 1e-11
    assert np.max(np.abs(err_after - expect_after)) < 1e-11
    # In words: just before a sync the error is about r (I + d_down + p/2) = 1.0055e-4 s,
    # right after it is about r (d_down + p/2) = 5.5e-7 s.
    assert np.all(np.abs(err_before - r * (interval + d + p / 2)) < 1e-9)
    assert np.all(np.abs(err_after - r * (d + p / 2)) < 1e-9)
    # Before the first reply arrives the sensor is unsynced: e0 + r t.
    early = np.array([0.0, 0.005])
    assert np.max(np.abs(stamp_errors(early, params, STREAMS) - (e0 + r * early))) < 1e-12


def test_total_sync_loss_equals_unsynced():
    t = np.arange(0.0, 20.0, 0.1)
    base = {"offset_s": 0.04, "drift_ppm": 80.0, "stamp_jitter_s": 0.002}
    lost = stamp_errors(t, {**base, "mode": "ntp", "sync_loss_probability": 1.0}, STREAMS)
    unsynced = stamp_errors(t, {**base, "mode": "unsynced"}, STREAMS)
    assert np.array_equal(lost, unsynced)
    _, samples = stamp_errors_with_log(t, {**base, "mode": "ntp", "sync_loss_probability": 1.0}, STREAMS)
    assert samples and all(s.lost for s in samples)


def test_asymmetry_through_stamp_errors():
    # No drift: after the first sync the stamp error is -a/2 for either sign.
    t = np.array([0.5, 1.5, 2.5])
    for a in (0.02, -0.02):
        params = {"mode": "ntp", "offset_s": 0.05, "sync_delay_s": 0.01, "sync_asymmetry_s": a}
        err = stamp_errors(t, params, STREAMS)
        assert np.max(np.abs(err + a / 2)) < 1e-12


def test_same_seeds_same_arrays_and_different_seeds_differ():
    t = np.arange(0.0, 30.0, 0.05)
    params = {"mode": "ntp", "offset_s": 0.01, "drift_ppm": 50.0, "stamp_jitter_s": 0.001,
              "sync_jitter_s": 0.002, "sync_loss_probability": 0.3}
    a = stamp_errors(t, params, STREAMS)
    b = stamp_errors(t, params, dict(STREAMS))
    c = stamp_errors(t, params, {**STREAMS, "clock": 99, "sync_up": 98})
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sync_log_reports_success_rate():
    params = {"mode": "ntp", "sync_loss_probability": 0.5, "sync_interval_s": 0.1}
    log = sync_log(50.0, params, STREAMS)
    assert len(log) == 501
    lost = np.mean([row["lost"] for row in log])
    assert 0.6 < lost < 0.85  # either of two independent legs can be lost: about 0.75
    ok = [row for row in log if not row["lost"]]
    assert all(abs(row["theta_s"]) < 1e-12 for row in ok)  # no offset, symmetric
    assert all(abs(row["delta_s"] - 0.01) < 1e-12 for row in ok)  # 2 d, processing removed
    assert sync_log(5.0, {"mode": "unsynced"}, STREAMS) == []


def test_validation():
    t = np.arange(3.0)
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "ntp", "bogus": 1.0}, STREAMS)
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "gps"}, STREAMS)
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "ntp", "sync_delay_s": 0.005, "sync_asymmetry_s": 0.02}, STREAMS)
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "ntp", "sync_loss_probability": 1.5}, STREAMS)
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "ntp"}, {"clock": 1})  # missing sync seeds
    with pytest.raises(ValueError):
        stamp_errors(t, {"mode": "unsynced", "stamp_jitter_s": 0.001}, {})  # missing clock seed
    with pytest.raises(ValueError):
        stamp_errors(np.array([-1.0, 0.0]), {"mode": "unsynced"}, STREAMS)


def test_ntp_exchange_none_when_lost():
    up = SimulatedLink(LinkConfig(delay_s=0.01, loss_model="independent", loss_probability=1.0), seed=1)
    down = SimulatedLink(LinkConfig(delay_s=0.01), seed=2)
    assert ntp_exchange(0.0, ClockModel(), up, down, 0) is None


def test_reordered_sync_replies_use_the_newest_exchange_not_the_last_arrival():
    """With 30 ms jitter on 10 ms sync legs and an exchange every 10 ms, replies arrive out of
    order. A capture must use the newest exchange (highest seq) among replies already arrived,
    because that exchange measured the drifting clock most recently. Reproduces the 1000 ppm
    drift case, where the old code picked seq 6 at t = 0.14 instead of seq 13."""
    import numpy as np
    from sim.timesync import ClockModel, stamp_errors, sync_log
    params = {"mode": "ntp", "offset_s": 0.0, "drift_ppm": 1000.0, "sync_interval_s": 0.01,
              "sync_delay_s": 0.01, "sync_jitter_s": 0.03, "sync_loss_probability": 0.0}
    streams = {"clock": 7, "sync_up": 1, "sync_down": 101}
    captures = np.arange(0.0, 0.5, 0.01)
    log = [row for row in sync_log(float(captures.max()), params, streams) if not row["lost"]]
    arrival_order = [row["seq"] for row in sorted(log, key=lambda r: r["arrival_s"])]
    assert arrival_order != sorted(arrival_order), "the case must actually reorder replies"
    got = stamp_errors(captures, params, streams)
    clock = ClockModel(0.0, 1000.0, 0.0)
    for t, e in zip(captures, got):
        arrived = [row for row in log if row["arrival_s"] <= t]
        theta = max(arrived, key=lambda r: r["seq"])["theta_s"] if arrived else 0.0
        assert e == pytest.approx(clock.read(np.array([t]))[0] - theta - t, abs=1e-15)


@pytest.mark.parametrize("horizon", [float("inf"), float("nan"), -1.0])
def test_sync_log_rejects_bad_horizons(horizon):
    """An infinite horizon used to loop forever and NaN returned an empty log."""
    from sim.timesync import sync_log
    with pytest.raises(ValueError):
        sync_log(horizon, {"mode": "ntp", "sync_interval_s": 1.0, "sync_delay_s": 0.01}, {"sync_up": 1, "sync_down": 2})


def test_empty_captures_need_no_random_stream():
    """No captures means no draws, so no stream should be demanded."""
    import numpy as np
    from sim.timesync import stamp_errors
    out = stamp_errors(np.array([]), {"mode": "ntp", "stamp_jitter_s": 0.1, "sync_interval_s": 1.0, "sync_delay_s": 0.01}, {})
    assert out.shape == (0,)
