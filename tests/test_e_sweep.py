"""EXP-031 checks for the simulated area sweep mission.

The clean, no-dropout run is compared with the independent area-rate hand
check in tests/reference_e/hand_checks.py.  Runs end as soon as coverage is
complete (the mission's own end rule), and the loss comparison only needs the
first 20 s because belief errors are scored at re-plan time (about 12 s).
All scenarios are simulated.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from sim.missions.api import Scenario
from sim.missions.engine import run_mission
from sim.missions.log import validate_log
from sim.missions.sweep import (FIELD_X_M, FIELD_Y_M, HALF_SWATH_M, LOSS_TIMEOUT_S, DROPOUT_TIME_S,
                                N_STRIPS, lane_route, strip_mask)
from sim.s6_recovery import seed_list

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_e import hand_checks as hc  # noqa: E402

SEED = seed_list(1, base_seed=20261031)[0]   # EXP-031 seed base (prereg)
FIELD_AREA_M2 = FIELD_X_M * FIELD_Y_M


def _scenario(*, preset="clean", dropout="none", mode="delta", duration_s=120.0, seed=SEED):
    return Scenario(mission="sweep", seed=seed, duration_s=duration_s, preset=preset,
                    params={"dropout": dropout, "message_mode": mode})


@functools.lru_cache(maxsize=None)
def _run(preset="clean", dropout="none", mode="delta", duration_s=120.0):
    # One logged run per scenario, shared by every test that needs it.
    return run_mission(_scenario(preset=preset, dropout=dropout, mode=mode, duration_s=duration_s), log=True)


def _covered_by_path(route, step_m=0.05):
    """Cells whose centre comes within the half swath of the route polyline.

    Written separately from the mission's marking code: sample the straight
    segments densely and test every cell centre against every sample.
    """
    pts = [np.asarray(route[0][:2], dtype=float)]
    for a, b in zip(route[:-1], route[1:]):
        a, b = np.asarray(a[:2], dtype=float), np.asarray(b[:2], dtype=float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step_m)))
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    pts = np.array(pts)
    cy, cx = np.mgrid[0:FIELD_Y_M, 0:FIELD_X_M] + 0.5
    d2 = (cx[..., None] - pts[:, 0]) ** 2 + (cy[..., None] - pts[:, 1]) ** 2
    return (d2 <= HALF_SWATH_M ** 2).any(axis=-1)


def test_strip_lanes_tile_each_strip_without_spill():
    """Geometry only: the planned lanes see every cell of their strip and none outside it."""
    sizes = []
    for k in range(N_STRIPS):
        mask = strip_mask(k)
        sizes.append(int(mask.sum()))
        route = lane_route(mask, (0.0, 0.0))
        lane_ys = sorted({p[1] for p in route})
        # 20 m deep strip, 4 m lanes, first lane 2 m inside the edge: 2, 6, 10, 14, 18.
        assert lane_ys == [2.0, 6.0, 10.0, 14.0, 18.0]
        seen = _covered_by_path(route)
        assert not np.any(mask & ~seen), f"strip {k}: planned lanes leave cells unseen"
        assert not np.any(seen & ~mask), f"strip {k}: planned lanes reach into another strip"
    # Equal 13.33 m strips on a 1 m grid, cells assigned by centre.
    assert sum(sizes) == FIELD_AREA_M2
    assert max(sizes) - min(sizes) <= FIELD_Y_M


def test_clean_no_dropout_matches_hand_check():
    reference_s = hc.coverage_time_s(FIELD_AREA_M2, 3, 2.0 * HALF_SWATH_M, 2.0)
    m = _run().metrics
    print(f"hand check {reference_s:.2f} s of sweeping; simulated t95 {m['t95_s']} s, t100 {m['t100_s']} s, "
          f"duplicated {m['duplicated_fraction']:.4f}")
    assert m["end_reason"] == "complete"
    assert m["missed_cells"] == 0
    assert m["t100_s"] is not None and m["t95_s"] is not None
    # Prereg hand check: 33.3 s of sweeping plus turns, so 35 to 42 s.  The
    # extra over 33.3 s is the braking and speed ramp at every lane end and
    # the 4 m moves between lanes.
    assert reference_s <= m["t100_s"]
    assert 35.0 <= m["t100_s"] <= 42.0
    assert m["t95_s"] < m["t100_s"]
    # Overlap between drones can only come from the strip borders: two borders,
    # at most two columns either side, 20 rows -> 80 cells = 10 % of the field.
    assert m["duplicated_fraction"] < 80 / FIELD_AREA_M2
    assert m["replans"] == 0 and m["false_loss_declarations"] == 0


def test_clean_dropout_replans_and_completes():
    m = _run(dropout="one", mode="delta").metrics
    base = _run().metrics
    print(f"clean dropout delta: {m}")
    assert m["end_reason"] == "complete" and m["missed_cells"] == 0
    # Both survivors notice the silence and each re-plans once.
    assert m["replans"] == 2
    assert m["false_loss_declarations"] == 0
    # On a clean link every share arrives, so the survivors' beliefs are exact.
    assert m["belief_error_cells"] == 0
    assert m["t100_s"] > base["t100_s"]


def test_clean_dropout_timing_and_log():
    result = _run(dropout="one", mode="delta")
    validate_log(result.log)
    events = result.log["events"]
    dropout = [e for e in events if e["kind"] == "drone_dropout"]
    assert len(dropout) == 1 and dropout[0]["agent"] == "d2"
    assert abs(dropout[0]["time_s"] - DROPOUT_TIME_S) < 1e-6
    replans = [e for e in events if e["kind"] == "replan"]
    assert sorted(e["agent"] for e in replans) == ["d1", "d3"]
    # The last share leaves at 10.0 s and arrives 2 ms later; silence must
    # exceed 2.0 s, so the declaration lands on the next 10 ms step.
    for e in replans:
        assert DROPOUT_TIME_S + LOSS_TIMEOUT_S < e["time_s"] <= DROPOUT_TIME_S + LOSS_TIMEOUT_S + 0.02
        assert e["detail"]["lost_peer"] == "d2"
    # The two survivors split the believed-uncovered cells between them.
    assert replans[0]["detail"]["believed_uncovered"] == replans[1]["detail"]["believed_uncovered"]
    assert sum(e["detail"]["assigned_cells"] for e in replans) == replans[0]["detail"]["believed_uncovered"]
    assert {e["detail"]["assigned_cells"] for e in replans} != {0}


def test_clean_cumulative_replan_matches_delta():
    """On a clean link every message arrives, so both modes give the same belief at re-plan time."""
    def replans(result):
        return sorted((e["agent"], e["time_s"], e["detail"]["believed_uncovered"], e["detail"]["assigned_cells"])
                      for e in result.log["events"] if e["kind"] == "replan")
    delta = _run(dropout="one", mode="delta")
    cumulative = _run(dropout="one", mode="cumulative", duration_s=15.0)
    assert cumulative.metrics["replans"] == 2
    assert cumulative.metrics["belief_error_cells"] == 0
    assert replans(cumulative) == replans(delta)


def test_delta_loses_more_belief_than_cumulative_under_loss():
    """Observed check on one fixed seed: lost delta messages are gone for good."""
    delta = _run(preset="iid_30", dropout="one", mode="delta", duration_s=20.0).metrics
    cumulative = _run(preset="iid_30", dropout="one", mode="cumulative", duration_s=20.0).metrics
    print(f"iid_30 seed {SEED}: belief_error_cells delta {delta['belief_error_cells']}, "
          f"cumulative {cumulative['belief_error_cells']}")
    assert delta["replans"] >= 1 and cumulative["replans"] >= 1
    assert delta["belief_error_cells"] >= cumulative["belief_error_cells"]


def test_same_seed_gives_identical_results():
    # 15 s covers the dropout at 10 s and the re-plans at about 12 s.
    first = _run(preset="iid_30", dropout="one", mode="delta", duration_s=15.0)
    again = run_mission(_scenario(preset="iid_30", dropout="one", mode="delta", duration_s=15.0), log=True)
    assert first.metrics["replans"] >= 1
    assert first.metrics == again.metrics
    assert json.dumps(first.log, sort_keys=True) == json.dumps(again.log, sort_keys=True)


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        run_mission(_scenario(mode="bitmask", duration_s=0.1))
    with pytest.raises(ValueError):
        run_mission(_scenario(dropout="two", duration_s=0.1))
