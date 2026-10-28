"""EXP-040 (proposal): sweep the recovery tracker's process noise q.

q (``RecoveryConfig.tracker_acceleration_psd``, m^2/s^3) is how much the
filter expects the target's velocity to wander: over t seconds it adds q x t
to the velocity variance on each axis, so the velocity can drift by about
sqrt(q x t) m/s. The registered S6 runs all use q = 0.1, about 0.32 m/s per
square-root second, or 0.1 m/s per 0.1 s sample.

This module only re-runs the recovery arm of ``sim.s6_recovery`` with a
different q. It reuses ``sim.adaptive_r.run_recovery_arm``, which matches the
registered recovery arm bit for bit, and changes q through a replaced
``RecoveryConfig``, so ``sim/tracker.py`` and ``sim/s6_recovery.py`` (hashed
by registered runs) are untouched. Everything is simulated.
"""
from __future__ import annotations

from dataclasses import replace

from .adaptive_r import make_tracker, run_recovery_arm
from .s6_recovery import ArmResult, RecoveryConfig, deliver

# Half-decade steps from 0.01 to 10. Written out so the registered 0.1 is the
# literal 0.1 (the same float the registered runs used), not 10 ** -1.0 computed.
Q_GRID = (0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0)
REGISTERED_Q = 0.1


def simulate_q_condition(motion: str, *, delay_s: float, loss_probability: float, noise_std_m: float,
                         loss_model: str, seed: int, config: RecoveryConfig, q_values=Q_GRID,
                         told_stds: dict[str, float]) -> dict[tuple[float, str], ArmResult]:
    """Run the recovery arm for every (q, told noise) pair on one seed's delivered packets.

    The packets are delivered once and shared by every pair (common random
    numbers), so differences between pairs come only from q and the told noise.
    Returns a dict keyed by (q, told name).
    """
    deliveries = deliver(motion, delay_s=delay_s, loss_probability=loss_probability, noise_std_m=noise_std_m,
                         seed=seed, config=config, loss_model=loss_model)
    out: dict[tuple[float, str], ArmResult] = {}
    for q in q_values:
        q_config = replace(config, tracker_acceleration_psd=float(q))
        for name, std in told_stds.items():
            tracker = make_tracker(motion, q_config, r_mode="told", measurement_std_m=std)
            out[(q, name)] = run_recovery_arm(deliveries, motion, q_config, tracker)
    return out
