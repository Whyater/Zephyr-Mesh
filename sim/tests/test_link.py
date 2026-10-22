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

def test_link_rejects_boolean_and_backwards_requested_time_without_contention():
    link = SimulatedLink(LinkConfig(loss_model="none"))
    with pytest.raises(ValueError):
        link.send("A", "B", True, 0.0)
    link.send("A", "B", 0, 1.0)
    with pytest.raises(ValueError):
        link.send("A", "B", 1, 0.5)


# ---------------------------------------------------------------------------
# delay jitter must not be clipped at zero (mean must match the config)
# ---------------------------------------------------------------------------

def _delays(config, n, seed=11):
    link = SimulatedLink(config, seed=seed)
    for seq in range(n):
        link.send("A", "B", seq, 0.0)  # same send time: age is the pure sampled delay
    return np.array([e.packet_age_s for e in link.events if e.packet_age_s is not None])


def test_jitter_does_not_pile_up_exact_zero_delays_and_keeps_configured_mean():
    # Independent expectation: the old rule max(0, 0.01 + Normal(0, 0.02)) puts
    # P(Normal < -0.5 sigma-units) = Phi(-0.5) = 30.9% of packets at exactly 0 and
    # raises the mean to 0.0140.  A non-negative delay law should give about 0%
    # zeros and a mean of 0.010.
    n = 40_000
    delays = _delays(LinkConfig(delay_s=0.01, delay_jitter_s=0.02), n)
    assert np.count_nonzero(delays <= 0.0) == 0
    # Tolerance: the standard error of the sample mean is jitter / sqrt(n)
    # = 0.02 / 200 = 1e-4.  Allow 5 standard errors (5e-4, 5% of the mean).
    se = 0.02 / np.sqrt(n)
    assert abs(delays.mean() - 0.01) < 5 * se
    # The configured jitter is also the realized spread (loose 5% check).
    assert delays.std() == pytest.approx(0.02, rel=0.1)


def test_small_jitter_still_looks_like_a_bell_around_the_configured_delay():
    n = 40_000
    delays = _delays(LinkConfig(delay_s=0.1, delay_jitter_s=0.005), n)
    se = 0.005 / np.sqrt(n)
    assert abs(delays.mean() - 0.1) < 5 * se
    assert delays.min() > 0.0
    assert delays.std() == pytest.approx(0.005, rel=0.05)


# ---------------------------------------------------------------------------
# loss and delay use separate random streams
# ---------------------------------------------------------------------------

def _run(config, seed=5, n=300):
    link = SimulatedLink(config, seed=seed)
    for seq in range(n):
        link.send("A", "B", seq, seq * 0.01)
    return link.events


def test_changing_jitter_leaves_the_loss_pattern_unchanged():
    base = _run(LinkConfig(delay_s=0.1, loss_probability=0.3, loss_model="independent"))
    jittery = _run(LinkConfig(delay_s=0.1, delay_jitter_s=0.03, loss_probability=0.3,
                              loss_model="independent"))
    assert [e.loss_reason for e in base] == [e.loss_reason for e in jittery]
    assert 0 < sum(e.loss_reason is not None for e in base) < 300


def test_changing_loss_leaves_the_delay_of_each_packet_unchanged():
    clean = _run(LinkConfig(delay_s=0.1, delay_jitter_s=0.03, loss_model="none"))
    lossy = _run(LinkConfig(delay_s=0.1, delay_jitter_s=0.03, loss_probability=0.3,
                            loss_model="independent"))
    compared = 0
    for a, b in zip(clean, lossy):
        if b.packet_age_s is not None:
            assert b.packet_age_s == pytest.approx(a.packet_age_s)
            compared += 1
    assert compared > 100


# ---------------------------------------------------------------------------
# wrapped (uint8) sequence numbers are misread unless unwrapped
# ---------------------------------------------------------------------------

def test_wrapped_sequence_numbers_are_flagged_duplicate_by_the_link_log():
    # Documents the current, honest behavior: the link identifies a packet by
    # (route, seq) only, so a counter that wraps at 256 repeats identities.
    # 600 packets with seq = i % 256 have 256 distinct identities, so 344
    # arrivals are flagged duplicate even though each is a new packet.
    link = SimulatedLink(LinkConfig(loss_model="none"))
    for i in range(600):
        link.send("A", "B", i % 256, i * 0.01)
    link.deliver_all()
    assert sum(e.duplicate for e in link.events) == 600 - 256
    # With a counter that does not wrap, nothing is flagged.
    clean = SimulatedLink(LinkConfig(loss_model="none"))
    for i in range(600):
        clean.send("A", "B", i, i * 0.01)
    clean.deliver_all()
    assert sum(e.duplicate or e.out_of_order for e in clean.events) == 0
