import numpy as np
import pytest

from sim.link import LinkConfig, SimulatedLink, lag_error


def signature(link):
    return [e.as_dict() for e in link.events]


def test_zero_loss_fixed_delay_delivers_with_age():
    link = SimulatedLink(LinkConfig(delay_s=0.1, loss_model="none"), seed=4)
    event = link.send("A", "B", 3, 2.0)
    assert event.receive_time == pytest.approx(2.1)
    assert event.packet_age_s == pytest.approx(0.1)
    assert link.advance(2.09) == []
    delivered = link.advance(2.1)
    assert delivered == [event]
    assert not event.duplicate and not event.out_of_order


def test_seeded_link_is_repeatable_and_different_seed_can_differ():
    config = LinkConfig(delay_s=0.1, delay_jitter_s=0.03,
                        loss_probability=0.35)
    def run(seed):
        link = SimulatedLink(config, seed=seed)
        for seq in range(30):
            link.send("A", "B", seq, seq * 0.02)
        link.deliver_all()
        return signature(link)
    assert run(12) == run(12)
    assert run(12) != run(13)


def test_burst_loss_records_burst_reason():
    config = LinkConfig(loss_model="burst", burst_start_probability=1.0,
                        burst_end_probability=0.0)
    link = SimulatedLink(config, seed=1)
    for seq in range(4):
        link.send("A", "B", seq, float(seq))
    assert all(event.loss_reason == "burst_loss" for event in link.events)


def test_duplicate_and_out_of_order_are_distinguished_after_delivery():
    link = SimulatedLink(LinkConfig(delay_s=0.0, loss_model="none"))
    first = link.send("A", "B", 2, 0.0)
    old = link.send("A", "B", 1, 0.1)
    duplicate = link.send("A", "B", 2, 0.2)
    link.deliver_all()
    assert not first.duplicate
    assert old.out_of_order
    assert duplicate.duplicate and not duplicate.out_of_order


def test_serialized_contention_is_explicit_and_deterministic():
    link = SimulatedLink(LinkConfig(contention_mode="serialized",
                                    packet_duration_s=0.05), seed=2)
    a = link.send("A", "B", 0, 0.0)
    b = link.send("A", "B", 1, 0.0)
    assert a.send_time == pytest.approx(0.0)
    assert b.send_time == pytest.approx(0.05)


def test_v_times_l_hand_check():
    assert lag_error(0.7, 0.1) == pytest.approx(0.07)
    with pytest.raises(ValueError):
        lag_error(-1, 0.1)
