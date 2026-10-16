"""Checks for the EXP-029 Part B simulated mission runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.e_exp029_part_b as part_b
from sim.missions.presets import get_preset
from tests.reference_e.hand_checks import (
    coarse_grid_chain,
    false_alarms_per_hour_markov,
    heartbeat_false_alarms_per_hour_iid,
    per_slot_chain,
)
from tools.c_common import VOLATILE_MANIFEST_FIELDS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_grid_counts_and_named_cells():
    # Pre-registered: 3 classes x 3 blackout lengths x 9 presets x 15 age policies
    # x 2 degraded actions x 50 seeds = 121,500 jobs.  Run (Addendum C, compute):
    # 3 classes x 3 lengths x 5 presets x 7 policy pairs x 2 actions x 20 seeds = 12,600.
    assert len(part_b.policies()) == 30
    assert len(part_b.build_point_jobs()) == 12600
    pairs = {(j["labels"]["T_w_s"], j["labels"]["T_s"]) for j in part_b.build_point_jobs()}
    assert pairs == set(part_b.RUN_POLICY_PAIRS)
    # Physics correction 5: 3 classes x 2 blackout lengths x 2 timeouts x 20 seeds = 240.
    assert len(part_b.build_physics_jobs()) == 240
    first = part_b.build_point_jobs(seeds=[1])[:1][0]
    assert first["scenario"]["mission"] == "tag"
    assert first["scenario"]["params"]["mode"] == "hold"
    assert first["scenario"]["outages"][0]["streams"] is None
    assert first["labels"]["target_class"] == "turn"
    assert first["labels"]["preset"] == "clean"


def test_outage_draw_is_seeded_and_bounded():
    values = [part_b.draw_outage_start(seed) for seed in (1, 2, 1)]
    assert values[0] == values[2]
    assert all(8.0 <= value <= 18.0 for value in values)
    item = part_b.outage_item(1, None)
    assert item["end_s"] is None
    assert item["streams"] is None


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp029b")
    jobs_a = part_b.build_point_jobs(pilot=True, seeds=[1], duration_s=5.0)[:2]
    jobs_b = part_b.build_point_jobs(pilot=True, seeds=[1], duration_s=5.0)[:2]
    a, b = base / "a", base / "b"
    part_b.run(a, ["--output", str(a)], pilot=True, processes=1, jobs=jobs_a, duration_s=5.0)
    part_b.run(b, ["--output", str(b)], pilot=True, processes=2, jobs=jobs_b, duration_s=5.0)
    return a, b


def test_tiny_run_writes_byte_stable_outputs(tiny_dirs):
    a, b = tiny_dirs
    assert set(part_b.OUTPUTS) | {"manifest.json"} == {path.name for path in a.iterdir()}
    for name in part_b.OUTPUTS:
        assert _sha(a / name) == _sha(b / name), name
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    for key in (*VOLATILE_MANIFEST_FIELDS, "argv"):
        ma.pop(key, None)
        mb.pop(key, None)
    assert ma == mb
    summary = json.loads((a / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "EXP-029 Part B"
    assert "missing_metrics" in summary
    assert "hand_check" in summary


def test_recommendation_constraint_uses_part_a_threshold():
    assert all(
        float(item["smallest_T_s_under_1_per_hour"]) >= 0.20
        for item in part_b._recommendations(
            [], {("clean", 0.15, 0.2): 0.0}
        )
        if item["smallest_T_s_under_1_per_hour"] is not None
    )


def test_hand_check_uses_independent_closed_forms():
    rows = [
        {"preset": "iid_30", "adapter": "point_mass", "blackout_start_s": 10.0,
         "T_w_s": 0.15, "T_s": 0.30, "degraded_action": "hold", "landings_before_outage": 1},
        {"preset": "iid_30", "adapter": "point_mass", "blackout_start_s": 20.0,
         "T_w_s": 0.15, "T_s": 0.30, "degraded_action": "hold", "landings_before_outage": 2},
        {"preset": "burst_10_0.5s", "adapter": "point_mass", "blackout_start_s": 10.0,
         "T_w_s": 0.15, "T_s": 0.30, "degraded_action": "hold", "landings_before_outage": 1},
    ]
    checks = part_b._hand_checks(rows)
    iid = get_preset("iid_30")
    iid_expected = heartbeat_false_alarms_per_hour_iid(iid.loss_probability, 20.0, 0.30)
    assert checks["iid_30"]["expected_rate_per_hour"] == pytest.approx(iid_expected)
    burst = get_preset("burst_10_0.5s")
    q0, r0 = per_slot_chain(burst.loss_probability, burst.burst_mean_s, 100.0)
    q, r = coarse_grid_chain(q0, r0, 5)
    burst_expected = false_alarms_per_hour_markov(q, r, 20.0, 0.30)
    assert checks["burst_10_0.5s"]["expected_rate_per_hour"] == pytest.approx(burst_expected)
