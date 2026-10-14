import numpy as np
import pytest
from sim.radio.channel import *


def test_path_loss_and_validation():
    c=ChannelConfig()
    assert path_loss_db(10,c)-path_loss_db(1,c)==pytest.approx(20.0,abs=1e-9)
    assert mean_rssi_dbm(10,c)==pytest.approx(-50.05)
    assert path_loss_db(0,c)==pytest.approx(path_loss_db(c.min_distance_m,c))
    with pytest.raises(ValueError): path_loss_db(-1,c)
    with pytest.raises(ValueError): path_loss_db(np.inf,c)


def test_dbpsk_anchor_numbers():
    c=ChannelConfig()
    assert packet_error_rate(-3,1024,c)==pytest.approx(.08,rel=1e-9)
    assert packet_error_rate(-2,1024,c)==pytest.approx(.0050730551,rel=1e-3)
    assert packet_error_rate(-4,1024,c)==pytest.approx(.53654699,rel=1e-3)
    assert packet_error_rate(-3,75,c)==pytest.approx(.00608844,rel=1e-3)


def test_shadow_reciprocity_and_cache():
    c=ChannelConfig(shadowing_sigma_db=3,shadowing_mode='static')
    ch=Channel(c,np.random.default_rng(4))
    a=ch.sample('a','b',2,1,frame_bytes=100); b=ch.sample('b','a',2,1,frame_bytes=100)
    assert a.shadow_db==b.shadow_db
    assert ch.received_power_dbm('a','b',2,1)==pytest.approx(a.rssi_dbm)


def test_ar_and_zero_sigma():
    c=ChannelConfig(shadowing_sigma_db=2,shadowing_mode='time_correlated',shadowing_corr_time_s=1)
    ch=Channel(c,np.random.default_rng(2)); xs=[ch.sample('a','b',1,t,frame_bytes=10).shadow_db for t in range(1000)]
    assert np.std(xs)>0
    with pytest.raises(ValueError): ch.sample('a','b',1,-1,frame_bytes=10)
    z=Channel(ChannelConfig(shadowing_sigma_db=0),np.random.default_rng(1)); assert z.sample('a','b',1,0,frame_bytes=1).shadow_db==0


def test_gilbert_elliott():
    g=GilbertElliott(.02,.2)
    assert g.stationary_bad()==pytest.approx(.09090909)
    assert g.mean_bad_burst()==5
    b,l=g.sample(20000,np.random.default_rng(3),start_state=None)
    assert b.mean()==pytest.approx(g.stationary_bad(),abs=.02)
    assert l.dtype==bool
    x=GilbertElliott.from_loss_and_burst(.1,5)
    assert x.mean_loss()==pytest.approx(.1)


def test_dual_slope_is_continuous_and_steeper_past_breakpoint():
    # Hand: n = 2 up to 50 m, then 10 * 3.5 dB per decade beyond it.
    c = ChannelConfig(exponent=2.0, breakpoint_m=50.0, exponent_far=3.5)
    assert path_loss_db(50.0, c) == pytest.approx(c.pl0_db + 20.0 * np.log10(50.0), abs=1e-9)
    assert path_loss_db(500.0, c) - path_loss_db(50.0, c) == pytest.approx(35.0, abs=1e-9)
    assert path_loss_db(10.0, c) == pytest.approx(path_loss_db(10.0, ChannelConfig(exponent=2.0)), abs=1e-12)
    with pytest.raises(ValueError):
        ChannelConfig(breakpoint_m=0.5)


def test_channel_fast_path_matches_vectorized_functions():
    # Channel.sample caches the mean terms and computes the DBPSK packet error
    # rate with plain math; both must agree with the vectorized module functions.
    from sim.radio.channel import Channel, ChannelConfig, mean_rssi_dbm, packet_error_rate, path_loss_db
    cfg = ChannelConfig(shadowing_sigma_db=0.0, wall_loss_db=4.0, body_loss_db=10.0)
    ch = Channel(cfg, np.random.default_rng(0))
    for d in (0.05, 1.0, 37.5, 900.0, 25_000.0):
        for walls, bodies in ((0, 0), (2, 1)):
            for n_bytes in (14, 75, 1024):
                s = ch.sample("a", "b", d, 0.0, frame_bytes=n_bytes, walls=walls, bodies=bodies)
                assert s.rssi_dbm == float(mean_rssi_dbm(d, cfg, walls=walls, bodies=bodies))
                assert s.path_loss_db == float(path_loss_db(d, cfg))
                assert s.per == pytest.approx(float(packet_error_rate(s.snr_db, n_bytes, cfg)), rel=1e-12, abs=1e-300)
                assert ch.received_power_dbm("a", "b", d, 0.0, walls=walls, bodies=bodies) == s.rssi_dbm
    for snr in np.linspace(-20.0, 20.0, 81):  # across the whole waterfall
        assert ch._per(float(snr), 75) == pytest.approx(float(packet_error_rate(float(snr), 75, cfg)), rel=1e-12, abs=1e-300)
