import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim.radio.channel import Channel
from sim.radio.mac import Frame, MacSimulator, data_airtime_s
from sim.radio.params import channel_preset, default_mac, default_phy
from tools.c_exp014_mac import age_of_information, loss_decomposition, predicted_channel_use


def test_idle_unicast_latency_hand_check():
    phy = default_phy(); positions = {"gs": (0, 10, 1), "d00": (1, 10, 1)}
    sim = MacSimulator(positions, Channel(channel_preset("indoor_gym"), np.random.default_rng(1)), phy,
                       default_mac(unicast_retry_limit=0), seed=2)
    sim.add_periodic_source("d00", "gs", 10, 32, start_s=0, stop_s=3)
    sim.run_until(3.0)
    # The upper-layer delivery is scheduled at data end, before SIFS and ACK.
    expected = phy.difs_s + data_airtime_s(32, phy)
    got = [r.deliver_s - r.created_s for r in sim.rx_records if r.outcome == "delivered"]
    assert got and all(abs(x - expected) < 1e-9 for x in got)


def test_age_hand_sawtooth():
    # One drone receives frames created at 0 and 2 s, at 1 and 3 s. On [2,4),
    # age is 2 then falls to 1 at t=3, giving area 3 s over two seconds.
    value = age_of_information([(1.0, 0.0, "d00"), (3.0, 2.0, "d00")], ["d00"], t0=2, t1=4, dt=0.001)
    assert abs(value - 2.0) < 1e-3


def test_loss_decomposition_hand_records():
    class F:
        def __init__(self, status): self.status = status
    class R:
        def __init__(self, fid, outcome): self.frame_id = fid; self.rx = "gs"; self.outcome = outcome
    frames = {i: F("retry_exhausted") for i in range(5)}; frames[3].status = "queue_drop"
    rx = [R(0, "collision"), R(1, "channel_error"), R(2, "below_sensitivity"), R(4, "delivered")]
    out = loss_decomposition(frames, rx, range(5))
    assert out == {"collision": 1/4, "channel_error": 1/4, "below_sensitivity": 1/4, "queue_drop": 1/4, "retry_exhausted": 0.0}


def test_predicted_u_hand_check():
    assert abs(predicted_channel_use(8, 50, 32, "unicast_r0") - 0.5864) < 1e-6


def test_pilot_byte_identity(tmp_path):
    py = sys.executable
    env = {"PYTHONPATH": ".", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    a, b = tmp_path / "a", tmp_path / "b"
    cmd = [py, "tools/c_exp014_mac.py", "--output", str(a), "--pilot", "--processes", "1"]
    import os
    run_env = {**dict(os.environ), **env}
    subprocess.run(cmd, check=True, env=run_env)
    cmd2 = [py, "tools/c_exp014_mac.py", "--output", str(b), "--pilot", "--processes", "1"]
    subprocess.run(cmd2, check=True, env=run_env)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    from tools.c_common import VOLATILE_MANIFEST_FIELDS
    for name in names:
        if name == "manifest.json":  # compared without runtime, git state and platform
            ma, mb = (json.loads((d / name).read_text()) for d in (a, b))
            for key in (*VOLATILE_MANIFEST_FIELDS, "argv"):  # argv holds each run's own output path
                ma.pop(key, None); mb.pop(key, None)
            assert ma == mb
            continue
        assert hashlib.sha256((a / name).read_bytes()).digest() == hashlib.sha256((b / name).read_bytes()).digest()

