import numpy as np

from sim.estimation.base import NAIVE, RECOVERY_V1
from sim.estimation.engine import CellSpec, deliver_cell, run_arm
from sim.estimation.targets import make_target
from sim.missions.comms import LossProcess, MissionLink
from sim.missions.estimate_feed import EstimateFeed


def test_engine_seed_seam_matches_online_p95():
    cell = CellSpec(target="cv", delay_s=0.1, loss_probability=0.1)
    bundle = deliver_cell(cell, 475358220)
    times = np.arange(0.0, 12.0 + 1e-9, 0.02)
    for arm, spec in (("recovery", RECOVERY_V1), ("naive", NAIVE)):
        expected = run_arm(bundle, cell, spec)
        feed = EstimateFeed.from_engine_link(arm, cell=cell, seed=475358220, duration_s=12.0)
        errors = []
        for t in times:
            position, _, _ = feed.estimate_at(float(t))
            errors.append(np.linalg.norm(position - bundle.target.position([t])[0]))
        measured = float(np.percentile(np.asarray(errors)[times >= 2.0], 95))
        assert abs(measured - expected.p95_error_m) < 1e-9


def test_age_and_zero_noise_zero_delay_cv():
    target = make_target("cv", {"speed_mps": 1.0}, seed=3)
    link = MissionLink("meas:o", delay_s=0.0, loss=LossProcess("clean", seed=6, duration_s=12), seed=7)
    feed = EstimateFeed("recovery", target=target, sensor_noise_std_m=0.0, rate_hz=10.0,
                        link=link, seed=8, duration_s=12.0)
    # The filter starts at rest with a 1 m^2 position variance, so even with
    # perfect fixes it needs a few updates to learn the 1 m/s velocity.  The
    # error must shrink over time and settle well below the 5 cm noise level
    # the filter assumes.
    position, _, age = feed.estimate_at(2.0)
    assert age == 0.0
    early = np.linalg.norm(position - target.position([2.0])[0])
    assert np.isclose(feed.estimate_at(2.02)[2], 0.02)
    position, _, _ = feed.estimate_at(10.0)
    late = np.linalg.norm(position - target.position([10.0])[0])
    assert late < early
    assert late < 0.01

