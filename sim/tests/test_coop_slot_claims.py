from __future__ import annotations

import numpy as np
import pytest

from sim.coop.allocation import reallocate, slots_for_alive
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator


ALIVE = ("drone-001", "drone-002", "drone-003", "drone-004")


def _sim(share_slot_claims: bool = True) -> CoopSwarmSimulator:
    return CoopSwarmSimulator(
        CoopSwarmConfig(
            n_agents=4,
            seed=1,
            duration_s=1.0,
            share_slot_claims=share_slot_claims,
        )
    )


def _claim(slot_index: int, members: tuple[str, ...], sent_at: float = 0.0) -> dict:
    return {
        "slot_index": slot_index,
        "slot_members": members,
        "receive_time": sent_at,
        "sent_at": sent_at,
    }


def test_previous_slots_flag_off_filters_private_slots_only():
    """Guards the legacy path from adopting claims found in peer messages."""
    sim = _sim(False)
    agent = sim.agents["drone-001"]
    agent.previous_slots = {"drone-001": 3, "drone-004": 1, "drone-099": 2}
    agent.latest["drone-002"] = _claim(0, ALIVE)
    assert sim._previous_slots(agent, ALIVE) == {"drone-001": 3, "drone-004": 1}


def test_previous_slots_ignores_same_size_different_membership_claim():
    """Guards against accepting a claim from a different ring with the same size."""
    sim = _sim()
    agent = sim.agents["drone-001"]
    different_members = ("drone-001", "drone-002", "drone-004", "drone-005")
    agent.latest["drone-002"] = _claim(2, different_members)
    assert sim._previous_slots(agent, ALIVE) == {}


def test_previous_slots_ignores_different_ring_size_claim():
    """Guards against accepting claims made before the ring was re-spaced."""
    sim = _sim()
    agent = sim.agents["drone-001"]
    agent.latest["drone-002"] = _claim(2, ALIVE[:3])
    assert sim._previous_slots(agent, ALIVE) == {}


def test_previous_slots_uses_own_slot_only_for_matching_membership():
    """Guards against the planner preserving its own slot across a membership change."""
    sim = _sim()
    agent = sim.agents["drone-001"]
    agent.slot_index = 1
    agent.slot_members = ALIVE[:-1]
    assert sim._previous_slots(agent, ALIVE) == {}
    agent.slot_members = ALIVE
    assert sim._previous_slots(agent, ALIVE) == {"drone-001": 1}


def test_previous_slots_lowest_id_wins_duplicate_claim():
    """Guards against different planners retaining conflicting switch bonuses."""
    sim = _sim()
    agent = sim.agents["drone-001"]
    agent.latest["drone-002"] = _claim(2, ALIVE)
    agent.latest["drone-003"] = _claim(2, ALIVE)
    agent.latest["drone-004"] = _claim(1, ALIVE)
    assert sim._previous_slots(agent, ALIVE) == {
        "drone-002": 2,
        "drone-004": 1,
    }


def test_previous_slots_ignores_matching_claim_from_peer_outside_alive():
    """Guards against departed peers contributing stale switch penalties."""
    sim = _sim()
    agent = sim.agents["drone-001"]
    reduced_alive = ALIVE[:3]
    agent.latest["drone-004"] = _claim(2, reduced_alive)
    assert sim._previous_slots(agent, reduced_alive) == {}


def _planner_geometry() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    center = np.array((0.0, 0.0, 1.5))
    positions = {
        "drone-001": np.array((0.0, 1.5, 1.5)),
        "drone-002": np.array((0.0, -1.5, 1.5)),
        "drone-003": np.array((-1.5, -0.175, 1.5)),
        "drone-004": np.array((-1.5, 0.175, 1.5)),
    }
    centers = []
    for aid in ("drone-003", "drone-004"):
        direction = positions[aid] - center
        centers.append(center + direction / np.linalg.norm(direction) * 0.05)
    return positions, centers[0], centers[1]


def test_shared_claim_tiebreak_makes_shifted_planners_agree():
    """Guards end-to-end reallocation against duplicate slots under shifted ring centres."""
    positions, center_003, center_004 = _planner_geometry()
    claims = {"drone-001": 1, "drone-002": 3, "drone-003": 2, "drone-004": 2}
    sim = _sim()
    assignments = []
    for planner, center in (("drone-003", center_003), ("drone-004", center_004)):
        agent = sim.agents[planner]
        agent.slot_index = claims[planner]
        agent.slot_members = ALIVE
        for peer, slot in claims.items():
            if peer != planner:
                agent.latest[peer] = _claim(slot, ALIVE)
        previous = sim._previous_slots(agent, ALIVE)
        slots = slots_for_alive(center, 4, 1.5, 1.5)
        assignments.append(reallocate(ALIVE, positions, slots, previous, switch_penalty_m=0.3))
    assert assignments[0] == assignments[1]
    assert assignments[0] == {
        "drone-001": 1,
        "drone-002": 3,
        "drone-003": 2,
        "drone-004": 0,
    }


def test_old_duplicate_claim_rule_keeps_both_planners_on_slot_two():
    """Guards that the geometry exposes the duplicate-slot failure without the tie-break."""
    positions, center_003, center_004 = _planner_geometry()
    old_previous = {"drone-001": 1, "drone-002": 3, "drone-003": 2, "drone-004": 2}
    assignments = []
    for center in (center_003, center_004):
        slots = slots_for_alive(center, 4, 1.5, 1.5)
        assignments.append(reallocate(ALIVE, positions, slots, old_previous, switch_penalty_m=0.3))
    assert assignments[0]["drone-003"] == 2
    assert assignments[1]["drone-004"] == 2


@pytest.mark.parametrize("share_slot_claims", [True, False])
def test_delivery_state_guards_depend_on_opt_in(share_slot_claims: bool):
    """Guards opt-in delivery ordering and departure filtering while preserving legacy overwrites."""
    config = CoopSwarmConfig(
        n_agents=4,
        seed=1,
        duration_s=1.0,
        share_slot_claims=share_slot_claims,
        link=CoopLinkConfig(delay_s=0.05, loss_probability=0.0, loss_model="none"),
    )
    sim = CoopSwarmSimulator(config)
    receiver = sim.agents["drone-001"]
    sender = "drone-002"
    sim._send(sender, "drone-001", 0.0, "state", _claim(1, ALIVE, 0.0))
    sim._deliver(0.05)
    sim._send(sender, "drone-001", 0.1, "leaving", {})
    sim._deliver(0.15)
    sim._send(sender, "drone-001", 0.2, "state", _claim(2, ALIVE, 0.2))
    sim._deliver(0.25)
    if share_slot_claims:
        assert sender in receiver.left_peers
        assert sender not in receiver.latest
    else:
        assert receiver.latest[sender]["slot_index"] == 2

    ordering_sim = CoopSwarmSimulator(
        CoopSwarmConfig(
            n_agents=4,
            seed=1,
            duration_s=1.0,
            share_slot_claims=share_slot_claims,
            link=CoopLinkConfig(delay_s=0.05, loss_probability=0.0, loss_model="none"),
        )
    )
    ordering_sim.agents["drone-001"].latest[sender] = _claim(3, ALIVE, 0.3)
    ordering_sim._send(sender, "drone-001", 0.0, "state", _claim(0, ALIVE, 0.0))
    ordering_sim._deliver(0.05)
    if share_slot_claims:
        assert ordering_sim.agents["drone-001"].latest[sender]["slot_index"] == 3
    else:
        assert ordering_sim.agents["drone-001"].latest[sender]["slot_index"] == 0
