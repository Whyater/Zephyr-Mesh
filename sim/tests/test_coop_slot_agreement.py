"""Survivors must not settle on the same ring slot after a dropout re-spacing.

EXP-026 seed 376990138 (N = 6, two drones landed at 10 s, independent loss 0.3,
heartbeat timeout 1.0 s) is the reproduced case: drones 003 and 004 each kept a
plan that sends both of them to slot 2 from about 10.5 s to the end of the run,
so slot 3 stays empty and the pair sits at the 0.35 m avoidance radius.
"""
from __future__ import annotations

import pytest

from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator

SURVIVORS = ("drone-001", "drone-002", "drone-003", "drone-004")


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
    steps = CoopSwarmSimulator(_exp026_config(share_slot_claims=True)).run().as_dict()["steps"]
    # Two timeouts (2 s) after the drop every survivor knows the drop happened.
    assert _late_slot_clashes(steps, start_s=12.0) == 0


def test_legacy_private_hysteresis_reproduces_the_clash():
    # The registered Run D outputs were made with private hysteresis. Keep that
    # path so they still reproduce, and document that it clashes here.
    steps = CoopSwarmSimulator(_exp026_config(share_slot_claims=False)).run().as_dict()["steps"]
    assert _late_slot_clashes(steps, start_s=12.0) > 0


class _FaultyLinkSimulator(CoopSwarmSimulator):
    """Drop or delay chosen packets after the normal link has decided their fate."""

    def __init__(self, config: CoopSwarmConfig, faults: list[dict]):
        super().__init__(config)
        self.faults = faults

    def _send(self, sender, receiver, t, kind, message):
        super()._send(sender, receiver, t, kind, message)
        event = self._events[-1]
        if event.receive_time is None:
            return
        for fault in self.faults:
            if fault["kind"] != kind or not fault["start_s"] <= t < fault["end_s"]:
                continue
            if fault.get("sender", sender) != sender or fault.get("receiver", receiver) != receiver:
                continue
            link = self._links[(sender, receiver)]
            if fault["delay_s"] is None:
                event.receive_time = None
                event.packet_age_s = None
                event.loss_reason = "test_drop"
                event.outcome = "lost"
                link._pending = [item for item in link._pending if item is not event]
            else:
                event.receive_time += fault["delay_s"]
                event.packet_age_s += fault["delay_s"]
            return


def _pair(kind, first, second, start_s, end_s, delay_s):
    return [
        {"kind": kind, "sender": a, "receiver": b, "start_s": start_s, "end_s": end_s, "delay_s": delay_s}
        for a, b in ((first, second), (second, first))
    ]


FAULTS = {
    # Each survivor of the clashing pair hears nothing from the other for 0.9 s
    # right after the drop (just under the 1 s heartbeat timeout).
    "claims_lost_between_pair": _pair("state", "drone-003", "drone-004", 10.0, 10.9, None),
    # The pair hears each other 0.8 s late for 2 s after the drop.
    "claims_late_between_pair": _pair("state", "drone-003", "drone-004", 10.0, 12.0, 0.8),
    # Survivors disagree on who left: 003 and 004 miss drone-005's leaving
    # message, 001 and 002 miss drone-006's, so two alive lists of the same
    # size but different members exist until the timeouts.
    "leaving_lost_split_views": [
        {"kind": "leaving", "sender": "drone-005", "receiver": r, "start_s": 0.0, "end_s": 99.0, "delay_s": None}
        for r in ("drone-003", "drone-004")
    ] + [
        {"kind": "leaving", "sender": "drone-006", "receiver": r, "start_s": 0.0, "end_s": 99.0, "delay_s": None}
        for r in ("drone-001", "drone-002")
    ],
    # The landed drone's last state packets reach 003 after its leaving message.
    "departed_drone_claims_late": [
        {"kind": "state", "sender": "drone-005", "receiver": "drone-003", "start_s": 9.5, "end_s": 10.0, "delay_s": 0.6},
    ],
}


def _final_slot_problems(steps, start_s: float) -> list[str]:
    """List logged steps after start_s where survivors disagree, clash, or leave a slot empty."""
    problems = []
    for step in steps:
        time_s = float(step["time_s"])
        if time_s < start_s:
            continue
        rows = [row for row in step["agents"] if row["active"] and row["agent_id"] in SURVIVORS]
        views = {tuple(row["believed_alive"]) for row in rows}
        own = sorted(int(row["slot_index"]) for row in rows)
        if views != {SURVIVORS}:
            problems.append(f"{time_s:.2f} views {sorted(views)}")
        elif own != list(range(len(SURVIVORS))):
            problems.append(f"{time_s:.2f} slots {own}")
    return problems


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_lost_or_late_claims_do_not_leave_a_clash_or_an_empty_slot(fault):
    config = _exp026_config(share_slot_claims=True, duration_s=16.0)
    steps = _FaultyLinkSimulator(config, FAULTS[fault]).run().as_dict()["steps"]
    # The faults end by 12 s; two more seconds lets every planner re-agree.
    assert _final_slot_problems(steps, start_s=14.0) == []


def test_ground_mode_with_shared_claims_keeps_distinct_slots():
    # The ground station plans for everyone at once and ignores claims; the
    # drones' own fallback planners still see claims. Check the mix still ends
    # with four distinct slots after a ground-station outage.
    config = CoopSwarmConfig(
        n_agents=4,
        seed=101,
        duration_s=16.0,
        control_dt_s=0.02,
        fusion="central",
        decision="ground",
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.3, loss_model="independent", share_rate_hz=10.0),
        heartbeat_timeout_s=0.5,
        gs_outage=((8.0, 11.0),),
        dropouts=(),
        log_every=1,
        share_slot_claims=True,
    )
    steps = CoopSwarmSimulator(config).run().as_dict()["steps"]
    for step in steps:
        if float(step["time_s"]) >= 14.0:
            own = sorted(int(row["slot_index"]) for row in step["agents"] if row["active"])
            assert own == [0, 1, 2, 3], step["time_s"]
