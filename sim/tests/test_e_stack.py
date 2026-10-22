import numpy as np

from sim.adapters import SimAdapter
from sim.failsafe import FailsafeConfig, FailsafeState
from sim.missions.comms import LossProcess, MissionLink, OutageWindow
from sim.missions.stack import LinkedDrone


def _stack(outage=False):
    loss = LossProcess("clean", seed=20, duration_s=8.0)
    outages = (OutageWindow(5.0, 8.0, streams=("cmd:a",)),) if outage else ()
    cmd = MissionLink("cmd:a", delay_s=0.0, loss=loss, outages=outages, seed=21)
    tel = MissionLink("tel:a", delay_s=0.0, loss=loss, seed=22)
    adapter = SimAdapter("a", initial_position_m=(0.0, 0.0, 1.0), dt_s=0.01,
                         failsafe_config=FailsafeConfig(degraded_after_s=0.15, land_after_s=0.30))
    return LinkedDrone(adapter, cmd_link=cmd, tel_link=tel), adapter


def test_clean_goal_and_telemetry_age():
    linked, _ = _stack()
    linked.set_goal((2.0, 0.0, 1.0))
    for i in range(300):
        state = linked.tick(i)
    assert np.linalg.norm(state.position - np.array((2.0, 0.0, 1.0))) < 1e-6
    telemetry, age = linked.telemetry_view(2.99)
    assert telemetry is not None
    assert abs(age - (2.99 - telemetry.timestamp_s + 0.0)) < 0.06


def test_outage_age_and_failsafe_thresholds():
    linked, adapter = _stack(outage=True)
    linked.set_goal((2.0, 0.0, 1.0))
    seen = {}
    for i in range(580):
        state = linked.tick(i)
        if i in (514, 529):
            seen[i] = state
    assert seen[514].failsafe_state == FailsafeState.DEGRADED.value
    assert seen[529].failsafe_state == FailsafeState.LANDING.value
    assert seen[514].link_age_s >= 0.15
    assert seen[529].link_age_s >= 0.30

