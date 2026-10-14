"""Checks for the EXP-017 runner (tracker through the radio model)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sim import s6_recovery
from sim.link import LinkConfig
from sim.radio.model import NetworkModelConfig
from tools.c_common import REPO_ROOT, VOLATILE_MANIFEST_FIELDS
from tools.c_exp017_tolerance import (build_cells, deliver_radio, headline, nearest_exp006, run, run_one,
                                      scenario_config)


@pytest.mark.parametrize("delay,loss,seed", [(0.0, 0.0, 1), (0.0015, 0.2, 2), (0.03, 0.1, 3)])
def test_abstract_mode_matches_s6_deliver_exactly(delay, loss, seed):
    # The plumbing check: with the abstract link (SimulatedLink unchanged) the
    # runner's delivery path must give EXP-006's deliveries exactly, including
    # the late-packet filter (0.03 s delay pushes the last capture past 12 s).
    cfg = s6_recovery.RecoveryConfig()
    nc = NetworkModelConfig(mode="abstract", link=LinkConfig(delay_s=delay, loss_probability=loss))
    for motion in s6_recovery.MOTIONS:
        ours = deliver_radio(motion, radio_cfg=nc, noise_std_m=0.05, seed=seed, config=cfg)
        theirs = s6_recovery.deliver(motion, delay_s=delay, loss_probability=loss, noise_std_m=0.05, seed=seed,
                                     config=cfg)
        assert ours == theirs


def test_nearest_exp006_uses_the_real_summary():
    # Hand lookup in the committed EXP-006 summary: (0.11 s, 0.28 loss) is
    # nearest to the grid point (0.10 s, 0.30) for the turn motion.
    exp006 = json.loads((REPO_ROOT / "runs/s6-recovery/summary.json").read_text())
    got = nearest_exp006(exp006, delay_s=0.11, loss=0.28, motion="constant_turn", arm="recovery")
    want = next(c for c in exp006["cells"] if c["slice"] == "main" and c["loss_model"] == "independent"
                and c["motion"] == "constant_turn" and c["noise_std_m"] == 0.05
                and c["delay_s"] == 0.1 and c["loss_probability"] == 0.3)
    assert got["cell_id"] == want["cell_id"]
    assert got["label"] == want["arms"]["recovery"]["label"]
    # zero delay, zero loss maps to the lossless cell, which passes
    assert nearest_exp006(exp006, delay_s=0.0, loss=0.0, motion="constant_velocity", arm="naive")["label"] == "passes"


def test_gym_control_delivers_everything_and_never_fails():
    # EXP-013: a 30 m gym link has a margin above 25 dB, so every packet arrives
    # and the tracker behaves as with a perfect link (no failures).
    cell = next(c for c in build_cells() if c["scenario"] == "A" and c["area"] == "gym" and c["mode"] == "unicast"
                and c["processing_s"] == 0.0)
    for seed in s6_recovery.seed_list(3):
        rows = run_one((cell, seed))
        assert len(rows) == 4  # two motions x two arms from one radio run
        assert all(r["received"] == r["sent"] == 121 for r in rows)
        assert not any(r["failed"] for r in rows)


def test_lossy_distance_counts_every_sent_packet():
    # 121 captures (0 to 12 s at 10 Hz) are always sent; at 750 m on the
    # stress corner (EXP-013: d10 = 746 m) many are lost, so received < sent.
    cell = next(c for c in build_cells() if c["scenario"] == "A" and c["area"] == "field" and c["mode"] == "broadcast"
                and c["processing_s"] == 0.0 and c["shadowing"] == "iid" and c["distance_m"] > 700)
    rows = run_one((cell, s6_recovery.seed_list(1)[0]))
    assert all(r["sent"] == 121 for r in rows)
    assert all(0 < r["received"] < 121 for r in rows)


def test_contention_cell_has_the_background_senders():
    cell = next(c for c in build_cells() if c["scenario"] == "B" and c["n"] == 8 and c["rate_hz"] == 50.0)
    radio = scenario_config(cell, seed=5).radio
    assert len(radio.background) == 7
    assert {b.rate_hz for b in radio.background} == {50.0}
    assert len(radio.positions) == 9  # gs, target and 7 others


def test_headline_on_hand_made_cells():
    def cell(d, failed):
        out = {"scenario": "A", "area": "field", "distance_m": d, "mode": "unicast", "processing_s": 0.0,
               "shadowing": "iid", "delivery_pooled": 1.0 - d / 1000, "latency_median_s": 0.001,
               "latency_p95_median_s": 0.002}
        for arm in s6_recovery.ARMS:
            for motion in s6_recovery.MOTIONS:
                out[f"{arm}_{motion}_cell_failed"] = failed
        return out
    h = headline([cell(300.0, False), cell(400.0, False), cell(500.0, True), cell(600.0, True)])
    assert h["recovery_constant_turn"]["scenario_a_smallest_failing_distance_m"] == 500.0
    assert h["recovery_constant_turn"]["scenario_a_largest_passing_distance_below_it_m"] == 400.0
    assert h["max_cell_median_latency_s"] == 0.001


def test_pilot_is_byte_identical(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(a, ["--pilot"], pilot=True, processes=1)
    run(b, ["--pilot"], pilot=True, processes=1)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        if name == "manifest.json":
            ma, mb = (json.loads((d / name).read_text()) for d in (a, b))
            for key in (*VOLATILE_MANIFEST_FIELDS, "argv"):
                ma.pop(key, None); mb.pop(key, None)
            assert ma == mb
        else:
            assert hashlib.sha256((a / name).read_bytes()).digest() == hashlib.sha256((b / name).read_bytes()).digest()
