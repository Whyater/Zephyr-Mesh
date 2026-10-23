"""Episode-by-episode check of the legacy and fixed degraded-link detectors.

Usage::

    PYTHONPATH=. ./.venv/bin/python tools/c_exp016_episode_check.py --margin 20 --duration 5 --rise 20

It rebuilds one EXP-016 interference cell with the runner's own seed and
distance, runs both detectors on the same packets, and checks per episode
what ``tools/c_exp016_compare.py`` can only see as row averages:

* whenever the legacy detector is degraded, the fixed one is too;
* every episode the legacy detector catches, the fixed one catches no later;
* every episode that ends with the legacy detector degraded also ends with
  the fixed one degraded, and both clear at the same moment afterwards.

It then reports the per-episode reasons each interference.csv column moved.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sim.radio.interference import (
    DegradedLinkDetector,
    InterferenceConfig,
    evaluate_detector,
    simulate_link_with_interference,
)
from sim.radio.mac import frame_bytes
from sim.radio.params import channel_preset, default_phy
from tools.c_exp016_heartbeat import BASE_SEED, HOURS, _clear_delays, _margin_distance


def _per_episode(episodes, transitions, grace_s: float) -> list[dict]:
    """Delay (None if missed), state at the episode end, and clear delay, per episode."""
    def state_at(t):
        state = "ok"
        for when, new_state in transitions:
            if when > t:
                break
            state = new_state
        return state

    starts = [t for t, s in transitions if s == "degraded"]
    out = []
    for start, end in episodes:
        if state_at(start) == "degraded":
            delay = 0.0
        else:
            inside = [t for t in starts if start <= t <= end + grace_s]
            delay = inside[0] - start if inside else None
        clear = _clear_delays([(start, end)], transitions)
        out.append({"delay": delay, "degraded_at_end": state_at(end) == "degraded",
                    "clear": clear[0] if clear else None})
    return out


def check_cell(margin_db: float, mean_duration_s: float, noise_rise_db: float, hours: float = HOURS) -> dict:
    frame_b = frame_bytes(32, default_phy())
    cfg = channel_preset("indoor_gym")
    dist = _margin_distance(cfg, margin_db, frame_b)
    seed = int(np.random.SeedSequence([BASE_SEED, 3, mean_duration_s, noise_rise_db]).generate_state(1)[0])
    run = simulate_link_with_interference(cfg, dist, 10, 3600 * hours, frame_b,
                                          InterferenceConfig(6, mean_duration_s, noise_rise_db), seed=seed)
    old_det = DegradedLinkDetector(20, .3, 6, 3, lost_packets_count_as_low_snr=False)
    new_det = DegradedLinkDetector(20, .3, 6, 3, lost_packets_count_as_low_snr=True)
    states_old, states_new = [], []
    for t, r, s in zip(run["times"], run["received"], run["snr_db"]):
        states_old.append(old_det.update(float(t), bool(r), float(s)))
        states_new.append(new_det.update(float(t), bool(r), float(s)))
    superset = all(n == "degraded" for o, n in zip(states_old, states_new) if o == "degraded")
    eps = run["episodes"]
    old = _per_episode(eps, old_det.transitions, 2.0)
    new = _per_episode(eps, new_det.transitions, 2.0)
    caught_not_later = all(n["delay"] is not None and n["delay"] <= o["delay"] + 1e-12
                           for o, n in zip(old, new) if o["delay"] is not None)
    end_superset = all(n["degraded_at_end"] for o, n in zip(old, new) if o["degraded_at_end"])
    same_clear = all(o["clear"] == n["clear"] for o, n in zip(old, new) if o["clear"] is not None)
    # Cross-check against the scoring function the runner uses.
    ev_old = evaluate_detector(run["times"], run["received"], run["snr_db"], eps,
                               DegradedLinkDetector(20, .3, 6, 3, lost_packets_count_as_low_snr=False), grace_s=2.0)
    ev_new = evaluate_detector(run["times"], run["received"], run["snr_db"], eps,
                               DegradedLinkDetector(20, .3, 6, 3, lost_packets_count_as_low_snr=True), grace_s=2.0)
    agrees = ([o["delay"] for o in old if o["delay"] is not None] == ev_old["trigger_delays_s"]
              and [n["delay"] for n in new if n["delay"] is not None] == ev_new["trigger_delays_s"])
    both = [(o["delay"], n["delay"]) for o, n in zip(old, new) if o["delay"] is not None]
    return {
        "cell": {"margin_db": margin_db, "mean_duration_s": mean_duration_s, "noise_rise_db": noise_rise_db,
                 "hours": hours, "n_episodes": len(eps), "packets": len(states_old)},
        "checks": {"fixed_degraded_whenever_legacy_is": superset,
                   "caught_by_legacy_implies_caught_no_later": caught_not_later,
                   "degraded_at_end_legacy_implies_fixed": end_superset,
                   "same_clear_time_when_both_degraded_at_end": same_clear,
                   "per_episode_delays_match_evaluate_detector": agrees},
        "caught_old": sum(o["delay"] is not None for o in old),
        "caught_new": sum(n["delay"] is not None for n in new),
        "newly_caught": sum(o["delay"] is None and n["delay"] is not None for o, n in zip(old, new)),
        "caught_by_both_mean_delay_old_new": ([float(np.mean([a for a, _ in both])), float(np.mean([b for _, b in both]))]
                                              if both else None),
        "delay_shift_s_when_both_nonzero": sorted({round(a - b, 9) for a, b in both if a > 0.0}),
        "zero_delay_old_new": [sum(o["delay"] == 0.0 for o in old), sum(n["delay"] == 0.0 for n in new)],
        "degraded_at_end_old_new": [sum(o["degraded_at_end"] for o in old), sum(n["degraded_at_end"] for n in new)],
        "clear_counted_old_new": [sum(o["clear"] is not None for o in old), sum(n["clear"] is not None for n in new)],
        "degraded_at_end_new_only_but_caught_by_both": sum(
            (not o["degraded_at_end"]) and n["degraded_at_end"] and o["delay"] is not None for o, n in zip(old, new)),
        "new_false_triggers": ev_new["false_triggers"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--rise", type=float, required=True)
    p.add_argument("--hours", type=float, default=HOURS)
    a = p.parse_args(argv)
    duration = int(a.duration) if a.duration == int(a.duration) else a.duration
    rise = int(a.rise) if a.rise == int(a.rise) else a.rise
    report = check_cell(a.margin, duration, rise, a.hours)
    print(json.dumps(report, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
