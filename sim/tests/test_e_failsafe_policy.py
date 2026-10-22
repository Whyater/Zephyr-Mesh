import pytest

from sim.failsafe import FailsafeAction, FailsafeConfig, FailsafeController, FailsafeState, SequenceLossWindow


def test_default_boundaries_and_roundtrip():
    c = FailsafeController()
    assert c.observe(0.0, link_age_s=0.15, battery_pct=100).state is FailsafeState.DEGRADED
    c = FailsafeController()
    assert c.observe(0.0, link_age_s=0.30, battery_pct=100).state is FailsafeState.LANDING
    cfg = FailsafeConfig()
    assert FailsafeConfig.from_dict(cfg.as_dict()) == cfg


def test_loss_rule_and_slow_action():
    c = FailsafeController(FailsafeConfig(weak_rule="age_or_loss", degraded_action="slow"))
    decision = c.observe(0.0, link_age_s=0.0, battery_pct=100, recent_loss_fraction=0.3)
    assert decision.state is FailsafeState.DEGRADED
    assert decision.reason == "link_lossy"
    assert decision.action is FailsafeAction.SLOW
    assert decision.command is None
    assert c.observe(0.1, link_age_s=0.0, battery_pct=100, recent_loss_fraction=0.1).state is FailsafeState.NOMINAL


def test_sequence_window_counts_gaps_and_trailing_misses():
    window = SequenceLossWindow(20)
    for seq in range(20):
        if seq not in {1, 3, 5, 7, 9}:
            window.record(seq)
    assert window.loss_fraction() == pytest.approx(0.25)
    assert window.loss_fraction(expected_next_seq=25) == pytest.approx(0.4)

