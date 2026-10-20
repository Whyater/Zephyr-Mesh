"""Survivors must not settle on the same ring slot after a dropout re-spacing.

EXP-026 seed 376990138 (N = 6, two drones landed at 10 s, independent loss 0.3,
heartbeat timeout 1.0 s) is the reproduced case: drones 003 and 004 each kept a
plan that sends both of them to slot 2 from about 10.5 s to the end of the run,
so slot 3 stays empty and the pair sits at the 0.35 m avoidance radius.
"""
from __future__ import annotations

from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator


def _exp026_config(**overrides) -> CoopSwarmConfig:
    values = dict(
        n_agents=6,
        seed=376990138,
        duration_s=14.0,
        control_dt_s=0.02,
        sensor_rate_hz=10.0,
        noise_std_m=0.05,
        estimator="cv",
        fusion="ci",
        decision="onboard",
        target="ct",
        target_params=(("radius", 3.0), ("speed", 0.8)),
        standoff_radius_m=1.5,
        slot_altitude_m=1.5,
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.3, loss_model="independent", share_rate_hz=10.0),
        heartbeat_timeout_s=1.0,
        dropouts=(("drone-005", 10.0, "landed"), ("drone-006", 10.0, "landed")),
        switch_penalty_m=0.3,
        log_every=1,
        ci_mode="fast",
    )
    values.update(overrides)
    return CoopSwarmConfig(**values)


def _late_slot_clashes(steps, start_s: float) -> int:
    """Count logged steps after start_s where two survivors steer to the same slot."""
    clashes = 0
    for step in steps:
        if float(step["time_s"]) < start_s:
            continue
        own = [agent["slot_index"] for agent in step["agents"] if agent["active"]]
        clashes += int(len(own) != len(set(own)))
    return clashes


def test_survivors_do_not_share_a_slot_after_respacing():
    steps = CoopSwarmSimulator(_exp026_config()).run().as_dict()["steps"]
    # Two timeouts (2 s) after the drop every survivor knows the drop happened.
    assert _late_slot_clashes(steps, start_s=12.0) == 0


def test_legacy_private_hysteresis_reproduces_the_clash():
    # The registered Run D outputs were made with private hysteresis. Keep that
    # path so they still reproduce, and document that it clashes here.
    steps = CoopSwarmSimulator(_exp026_config(share_slot_claims=False)).run().as_dict()["steps"]
    assert _late_slot_clashes(steps, start_s=12.0) > 0
