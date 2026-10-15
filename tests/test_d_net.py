from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import s6_recovery
from sim.coop.net import (
    LinkSpec,
    TrafficSpec,
    default_phases,
    make_deliveries,
    predicted_channel_use,
)
from sim.radio.model import BackgroundSource
from tools.c_exp014_mac import predicted_channel_use as reference_channel_use


def test_single_star_matches_recovery_link(seed_values=(11, 23, 41, 67, 101)):
    for seed in seed_values:
        for delay in (0.0, 0.1):
            for loss in (0.0, 0.3):
                traffic = TrafficSpec("star", ("o1",), rate_hz=10.0, duration_s=12.0)
                table = make_deliveries(traffic, LinkSpec("abstract", delay, loss), seed)
                got = table[("o1", "gs")]
                cfg = s6_recovery.RecoveryConfig()
                expected = s6_recovery.deliver("constant_velocity", delay_s=delay,
                                               loss_probability=loss, noise_std_m=0.0,
                                               seed=seed, config=cfg)
                by_seq = {item.seq: item for item in expected}
                assert len(got) == 121
                for row in got:
                    ref = by_seq.get(row.seq)
                    if ref is None:
                        assert row.receive_time_s is None
                    else:
                        assert row.receive_time_s is not None
                        assert abs(row.receive_time_s - ref.receive_time_s) <= 1e-12


def test_peer_route_losses_are_independent():
    traffic = TrafficSpec("peer", tuple(f"o{i}" for i in range(1, 5)), duration_s=1.0)
    all_lost = 0
    total = 0
    for seed in range(200):
        table = make_deliveries(traffic, LinkSpec("abstract", 0.0, 0.5), seed)
        routes = [table[(f"o{i}", "gs")] for i in range(1, 5)]
        for seq in range(len(routes[0])):
            all_lost += int(all(row[seq].receive_time_s is None for row in routes))
            total += 1
    observed = all_lost / total
    expected = 0.5 ** 4
    se = np.sqrt(expected * (1.0 - expected) / total)
    assert abs(observed - expected) <= 3.0 * se


def test_zero_loss_zero_delay_is_immediate():
    traffic = TrafficSpec("peer", ("o1", "o2"), rate_hz=10.0, duration_s=1.0)
    table = make_deliveries(traffic, LinkSpec("abstract", 0.0, 0.0), 7)
    for rows in table.values():
        assert all(row.receive_time_s == row.send_time_s and row.loss_reason is None for row in rows)


def test_radio_star_meets_delivery_and_airtime_estimate():
    for n in range(1, 7):
        observers = tuple(f"o{i}" for i in range(1, n + 1))
        positions = {"gs": (0.0, 0.0, 1.5)}
        positions.update({name: (3.0, 0.0, 1.5) for name in observers})
        traffic = TrafficSpec("star", observers, rate_hz=10.0, message_bytes=32, duration_s=12.0)
        table = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 1)
        summary = table.summary()
        predicted = predicted_channel_use(n, 10.0, 32, True)
        assert summary["delivery_ratio"] >= 0.99
        assert abs(summary["busy_fraction"] - predicted) / predicted < 0.30


def test_radio_peer_broadcast_uses_one_frame_and_keeps_receiver_rows():
    observers = tuple(f"o{i}" for i in range(1, 5))
    positions = {"gs": (0.0, 0.0, 1.5)}
    positions.update({name: (3.0, 0.0, 1.5) for name in observers})
    traffic = TrafficSpec("peer", observers, rate_hz=50.0, message_bytes=216, duration_s=12.0)
    table = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions,
                                             # Staggered access is intentional here so synchronized
                                             # broadcasts do not all collide at the same instant.
                                             immediate_access=False), 1)
    predicted = predicted_channel_use(4, 50.0, 216, False)
    assert abs(table.summary()["busy_fraction"] - predicted) / predicted < 0.30
    for sender in observers:
        for receiver in (*observers, "gs"):
            if receiver != sender:
                assert (sender, receiver) in table
                assert len(table[(sender, receiver)]) == 601


def test_determinism_and_seed_change():
    traffic = TrafficSpec("peer", ("o1", "o2"), duration_s=2.0)
    link = LinkSpec("abstract", 0.1, 0.3, jitter_s=0.01)
    first = make_deliveries(traffic, link, 15)
    second = make_deliveries(traffic, link, 15)
    third = make_deliveries(traffic, link, 16)
    assert first == second
    assert first != third


def test_predicted_channel_use_matches_run_c():
    for n in (1, 4, 8):
        for unicast in (True, False):
            mode = "unicast_r0" if unicast else "broadcast"
            assert predicted_channel_use(n, 10.0, 32, unicast) == reference_channel_use(n, 10.0, 32, mode)


def test_module_has_no_reserved_terms():
    root = Path(__file__).resolve().parents[1]
    source = (root / "sim/coop/net.py").read_text(encoding="utf-8")
    for word in ("intercept", "interceptor", "kill", "attack", "weapon", "payload", "strike", "targeting"):
        assert word not in source.lower()
    assert "\u2014" not in source


def test_radio_immediate_access_collides_but_dephasing_delivers():
    observers = ("o1", "o2", "o3", "o4")
    positions = {"gs": (0.0, 0.0, 1.5), **{name: (3.0, 0.0, 1.5) for name in observers}}
    sync = TrafficSpec("peer", observers, duration_s=2.0)
    dephased = TrafficSpec("peer", observers, duration_s=2.0,
                           phase_offsets_s=(0.0, 0.03, 0.06, 0.09))
    spec = LinkSpec("radio", 0.0, 0.0, positions=positions)
    assert make_deliveries(sync, spec, 7).summary()["delivery_ratio"] == 0.0
    assert make_deliveries(dephased, spec, 7).summary()["radio_delivery_ratio"] >= 0.99


def test_radio_ratio_excludes_late_end_and_phases_are_fixed():
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5)}
    traffic = TrafficSpec("star", ("o1",), duration_s=0.1)
    summary = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 3).summary()
    assert summary["loss_reasons"] == {"late_end": 1}
    assert summary["radio_delivery_ratio"] == 1.0
    phases = default_phases(("o1", "o2", "o3"), 10.0, 9)
    assert phases == default_phases(("o1", "o2", "o3"), 10.0, 9)
    assert phases[0] == 0.0 and all(0.0 <= p < 0.1 for p in phases)
    reordered = default_phases(("o2", "o1", "o3"), 10.0, 9)
    assert reordered[1] == 0.0


def test_radio_star_latency_grows_with_shared_mac_load():
    positions = {"gs": (0.0, 0.0, 1.5)}
    tables = {}
    for n in (1, 6):
        observers = tuple(f"o{i}" for i in range(1, n + 1))
        positions.update({name: (3.0, 0.0, 1.5) for name in observers})
        traffic = TrafficSpec("star", observers, rate_hz=10.0, message_bytes=32, duration_s=12.0)
        tables[n] = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 1)
    assert tables[6].summary()["mean_latency_s"] > tables[1].summary()["mean_latency_s"]


def test_default_peer_phases_produce_per_receiver_outcomes():
    observers = ("o1", "o2", "o3", "o4")
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5),
                 "o2": (3.0, 0.0, 1.5), "o3": (3.0, 0.0, 1.5), "o4": (6000.0, 0.0, 1.5)}
    traffic = TrafficSpec("peer", observers, rate_hz=10.0, duration_s=3.0,
                          phase_offsets_s=default_phases(observers, 10.0, 3))
    table = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 3)
    counts = tuple(sum(row.receive_time_s is not None for row in table[("o1", receiver)])
                   for receiver in ("o2", "o4", "gs"))
    assert len(set(counts)) > 1


def test_radio_delay_shifts_receives_and_marks_new_late_rows():
    observers = ("o1", "o2", "o3", "o4")
    positions = {"gs": (0.0, 0.0, 1.5), **{name: (3.0, 0.0, 1.5) for name in observers}}
    traffic = TrafficSpec("peer", observers, rate_hz=10.0, message_bytes=32, duration_s=12.0,
                          phase_offsets_s=default_phases(observers, 10.0, 3))
    base = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 3)
    delayed = make_deliveries(traffic, LinkSpec("radio", 0.07, 0.0, positions=positions), 3)
    for key in sorted(base):
        for before, after in zip(base[key], delayed[key]):
            if before.receive_time_s is not None and after.receive_time_s is not None:
                assert abs(after.receive_time_s - before.receive_time_s - 0.07) <= 1e-12
            elif before.receive_time_s is not None:
                assert after.receive_time_s is None and after.loss_reason == "late_end"


def test_radio_extra_loss_matches_seeded_draws():
    observers = ("o1", "o2")
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5), "o2": (3.0, 0.0, 1.5)}
    traffic = TrafficSpec("star", observers, rate_hz=10.0, message_bytes=32, duration_s=3.0)
    link = LinkSpec("radio", 0.0, 0.3, positions=positions)
    table = make_deliveries(traffic, link, 8)
    radio_only = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 8)
    for route_index, key in enumerate(sorted(table)):
        rng = np.random.default_rng(np.random.SeedSequence([8, 2000 + route_index]))
        for row, radio_row in zip(table[key], radio_only[key]):
            draw = float(rng.random())
            if radio_row.receive_time_s is not None and draw < 0.3:
                assert row.receive_time_s is None and row.loss_reason == "extra"
            elif radio_row.receive_time_s is not None:
                assert row.receive_time_s is not None
            else:
                assert row.receive_time_s is None and row.loss_reason == radio_row.loss_reason


def test_radio_background_source_increases_busy_fraction():
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5), "bg": (3.0, 1.0, 1.5)}
    traffic = TrafficSpec("star", ("o1",), rate_hz=10.0, message_bytes=32, duration_s=12.0)
    plain = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions), 1)
    busy = make_deliveries(traffic, LinkSpec("radio", 0.0, 0.0, positions=positions,
        background=(BackgroundSource("bg", "gs", 200.0, payload_bytes=200, stop_s=12.0),)), 1)
    assert busy.summary()["busy_fraction"] - plain.summary()["busy_fraction"] > 0.1


def test_radio_burst_is_seeded_and_seed_sensitive():
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5)}
    traffic = TrafficSpec("star", ("o1",), duration_s=3.0)
    link = LinkSpec("radio", 0.0, 0.3, loss_model="burst", positions=positions)
    assert make_deliveries(traffic, link, 4) == make_deliveries(traffic, link, 4)
    assert make_deliveries(traffic, link, 4) != make_deliveries(traffic, link, 5)


def test_radio_jitter_is_rejected():
    with pytest.raises(ValueError):
        LinkSpec("radio", 0.0, 0.0, jitter_s=0.01,
                 positions={"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5)})


def test_radio_burst_loss_rate_and_run_length():
    positions = {"gs": (0.0, 0.0, 1.5), "o1": (3.0, 0.0, 1.5)}
    traffic = TrafficSpec("star", ("o1",), duration_s=60.0)
    link = LinkSpec("radio", 0.0, 0.3, loss_model="burst", positions=positions)
    losses = []
    runs = []
    for seed in range(60):
        rows = make_deliveries(traffic, link, seed)[("o1", "gs")]
        observed = [row for row in rows if row.loss_reason != "late_end"]
        flags = [row.receive_time_s is None for row in observed]
        losses.extend(flags)
        current = 0
        for lost in flags:
            if lost:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
    assert abs(sum(losses) / len(losses) - 0.3) <= 0.03
    abstract = LinkSpec("abstract", 0.0, 0.3, loss_model="burst")
    abstract_rows = []
    for seed in range(60):
        rows = make_deliveries(traffic, abstract, seed)[("o1", "gs")]
        flags = [row.receive_time_s is None for row in rows if row.loss_reason != "late_end"]
        current = 0
        for lost in flags:
            if lost:
                current += 1
            elif current:
                abstract_rows.append(current)
                current = 0
        if current:
            abstract_rows.append(current)
    assert abs(np.mean(runs) - np.mean(abstract_rows)) / np.mean(abstract_rows) <= 0.10

