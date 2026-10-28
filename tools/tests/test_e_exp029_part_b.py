"""Checks for the EXP-029 Part B simulated mission runner."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import tools.e_exp029_part_a as part_a
import tools.e_exp029_part_b as part_b
from sim.missions.presets import PRESET_NAMES, get_preset
from sim.tests.reference_e.hand_checks import (
    coarse_grid_chain,
    false_alarms_per_hour_markov,
    heartbeat_false_alarms_per_hour_iid,
    per_slot_chain,
)
from tools.c_common import VOLATILE_MANIFEST_FIELDS
from tools.e_common import make_job, replay_command


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_part_a_fixture(folder: Path, *, hours: float = 10.0, drop: tuple | None = None,
                          summary: dict | None = None) -> Path:
    """A small Part A table in Part A's own CSV format (its write_csv), 20 Hz age rows only.

    Rates are the closed forms, standing in for the measured rates.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for preset in PRESET_NAMES:
        for policy in part_a.policies():
            if policy["weak_rule"] != "age":
                continue
            if drop == (preset, policy["T_w_s"], policy["T_s"]):
                continue
            rate = part_a.closed_form_per_hour(preset, 20.0, policy["T_s"])
            rows.append({"preset": preset, "h_hz": 20.0, "weak_rule": "age", "T_w_s": policy["T_w_s"],
                         "T_s": policy["T_s"], "hours": hours, "unnecessary_landings": int(rate * hours),
                         "landings_per_hour": rate})
    path = folder / "part_a.csv"
    part_a.write_csv(path, rows)
    if summary is not None:
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    return path


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


# ---- Part A dependency -------------------------------------------------------------

def test_part_a_fixture_is_accepted_and_hashed(tmp_path):
    path = _write_part_a_fixture(tmp_path / "a")
    table = part_b.load_part_a(path)
    assert table["sha256"] == _sha(path)
    assert table["rows_used"] == 9 * 15  # 9 presets x 15 age-rule (T_w, T) pairs
    assert table["path"] == "part_a.csv"  # outside the repo: file name only, no absolute path


@pytest.mark.parametrize("case", ["missing_file", "missing_row", "pilot_flag", "pilot_hours"])
def test_part_a_refusal_before_any_job(tmp_path, case):
    if case == "missing_file":
        csv_path = tmp_path / "absent" / "part_a.csv"
        message = "does not exist"
    elif case == "missing_row":
        csv_path = _write_part_a_fixture(tmp_path / "a", drop=("iid_30", 0.15, 0.3))
        message = "lacks 1 of 135"
    elif case == "pilot_flag":
        csv_path = _write_part_a_fixture(tmp_path / "a", summary={"pilot": True})
        message = "pilot output"
    else:
        csv_path = _write_part_a_fixture(tmp_path / "a", hours=1.0)
        message = "simulated hours"
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        part_b.run(out, ["--output", str(out)], processes=1, jobs=[], part_a_csv=csv_path)
    # The refusal happens before the output folder is made, so no job ran.
    assert not out.exists()


# ---- mission-observed unnecessary landings ------------------------------------------

def _row(preset, seed, *, start, first_landing=None, cls="turn", length=2.0, action="hold"):
    landed_before = first_landing is not None and first_landing < start
    return {"preset": preset, "adapter": "point_mass", "T_w_s": 0.15, "T_s": 0.30,
            "degraded_action": action, "seed": seed, "target_class": cls, "outage_length_s": length,
            "blackout_start_s": start, "first_landing_s": first_landing,
            "airborne_at_blackout": not landed_before, "landings_before_outage": int(landed_before)}


def _nine_rows(preset, seed, **kwargs):
    return [_row(preset, seed, cls=cls, length=length, **kwargs)
            for cls in ("turn", "figure8", "jink") for length in (0.5, 2.0, "permanent")]


def test_exposure_rule_and_one_run_per_seed():
    # Hand arithmetic: seed 1 landed at 5 s before a 10 s blackout (at risk 5 s, 1 landing);
    # seed 2 never landed before its 12 s blackout (at risk 12 s); seed 3 landed at 13 s,
    # after its 11 s blackout (at risk 11 s, 0 landings before it).  Each seed has 9 runs
    # (3 classes x 3 lengths) that must count once: 1 landing in 28 s, not 9 in 252 s.
    rows = (_nine_rows("burst_10_0.5s", 1, start=10.0, first_landing=5.0)
            + _nine_rows("burst_10_0.5s", 2, start=12.0)
            + _nine_rows("burst_10_0.5s", 3, start=11.0, first_landing=13.0))
    assert [part_b.pre_blackout_exposure_s(r) for r in rows[::9]] == [5.0, 12.0, 11.0]
    check = part_b.mission_landing_check(rows, preset="burst_10_0.5s", t_w=0.15, timeout=0.30, action="hold")
    assert check["runs_in_cell"] == 27
    assert check["seeds_counted"] == 3
    assert check["count"] == 1
    assert check["exposure_h"] == pytest.approx(28.0 / 3600.0)
    assert check["grounded_before_blackout"] == 1
    assert check["seeds_with_inconsistent_history"] == 0
    # A later run of seed 2 with a different pre-blackout history is flagged, and the
    # representative (the first in grid order) is the one counted.
    tampered = rows + [_row("burst_10_0.5s", 2, start=12.0, first_landing=4.0, cls="jink")]
    check = part_b.mission_landing_check(tampered, preset="burst_10_0.5s", t_w=0.15, timeout=0.30,
                                         action="hold")
    assert check["seeds_with_inconsistent_history"] == 1
    assert check["count"] == 1


def _burst_rate_by_hand(preset_name: str, timeout: float) -> float:
    preset = get_preset(preset_name)
    q0, r0 = per_slot_chain(preset.loss_probability, preset.burst_mean_s, 100.0)
    q, r = coarse_grid_chain(q0, r0, 5)
    return false_alarms_per_hour_markov(q, r, 20.0, timeout)


def test_hand_check_verdicts_from_independent_closed_forms():
    expected_rate = _burst_rate_by_hand("burst_10_0.5s", 0.30)
    # 20 seeds landed at 2 s and 20 stayed airborne to a 10 s blackout: 240 s at risk, 20 landings.
    rows = []
    for seed in range(20):
        rows += _nine_rows("burst_10_0.5s", seed, start=10.0, first_landing=2.0)
    for seed in range(20, 40):
        rows += _nine_rows("burst_10_0.5s", seed, start=10.0)
    checks, missing = part_b._hand_checks(rows)
    burst = checks["burst_10_0.5s"]
    expected = expected_rate * 240.0 / 3600.0
    assert expected >= 10.0
    assert burst["expected_rate_per_hour"] == pytest.approx(expected_rate)
    assert burst["expected_count"] == pytest.approx(expected)
    assert burst["z"] == pytest.approx((20 - expected) / math.sqrt(expected))
    assert burst["verdict"] == ("agrees" if abs(burst["z"]) <= 4.0 else "disagrees")
    assert burst["verdict"] == "agrees"
    # iid_30 was not in this row set: listed under missing_metrics, not silently skipped.
    assert checks["iid_30"]["status"] == "missing_metrics"
    assert any("iid_30" in item for item in missing)
    # No landings in 400 s at risk: far below the burst expectation, so it disagrees.
    quiet = []
    for seed in range(40):
        quiet += _nine_rows("burst_10_0.5s", seed, start=10.0)
    assert part_b._hand_checks(quiet)[0]["burst_10_0.5s"]["verdict"] == "disagrees"
    # iid_30 with 3 seeds expects far fewer than 10 events.
    iid = get_preset("iid_30")
    few = []
    for seed in range(3):
        few += _nine_rows("iid_30", seed, start=10.0)
    iid_check = part_b._hand_checks(few)[0]["iid_30"]
    assert iid_check["expected_rate_per_hour"] == pytest.approx(
        heartbeat_false_alarms_per_hour_iid(iid.loss_probability, 20.0, 0.30))
    assert iid_check["verdict"] == "too few expected events"


# ---- recommendation and predictions ---------------------------------------------------

def test_recommendation_rule_and_not_run_presets(tmp_path):
    rates = part_b.load_part_a(_write_part_a_fixture(tmp_path / "a"))["rates"]
    recs = {item["preset"]: item for item in part_b._recommendations([], rates)}
    # T >= T_w + 0.05 = 0.2 s, so the clean preset (no loss) gets 0.2 s, not 0.15 s.
    assert recs["clean"]["smallest_T_s_under_1_per_hour"] == 0.2
    # burst_10_0.5s: the closed form stays above 1 per hour at every T up to 2 s.
    assert recs["burst_10_0.5s"]["smallest_T_s_under_1_per_hour"] is None
    assert recs["burst_10_0.5s"]["rule_result"] == "none up to 2 s"
    assert recs["iid_1"]["at_recommended_T"] == "not run in Part B (Addendum C)"
    assert recs["iid_1"]["at_planned_0.30"] == "not run in Part B (Addendum C)"


def test_prediction_text_and_comparison_table(tmp_path):
    # Restates the runner's constants (the text itself can only be checked by eye
    # against the pre-registration); kept to guard the stored labels.
    assert part_b.PREDICTION_TEXT.startswith("at 20 Hz the 300 ms rule (m = 6)")
    assert part_b.CORRECTION_1_TEXT.startswith("EXP-029 prediction: by the pre-registered rule")
    assert part_b.PREDICTION_SOURCE == (
        "pre-registered before the runs (EXP-029 section of the private pre-registration)")
    rates = part_b.load_part_a(_write_part_a_fixture(tmp_path / "a"))["rates"]
    table = {item["preset"]: item for item in part_b._prediction_comparison(part_b._recommendations([], rates))}
    assert set(table) == set(PRESET_NAMES)
    assert table["iid_30"]["predicted_T_s"] == [0.5]
    assert table["burst_10_0.2s"]["predicted_T_s"] == [1.0, 2.0]
    # iid_1 was predicted at 0.15 s, which the T_w + 0.05 s constraint rules out.
    assert table["iid_1"]["agrees"] is False
    assert table["iid_1"]["notes"]


# ---- tiny run ------------------------------------------------------------------------------

TINY_DURATION_S = 7.5  # blackout 2.0 to 4.0 s, plus its 3 s late-reaction window, inside the run


def _tiny_jobs() -> list[dict]:
    """Two pilot jobs with the blackout moved to 2.0 s so it falls inside a 7.5 s run."""
    seed = part_b.seeds_for(29, 1)[0]
    chosen = [job for job in part_b.build_point_jobs(pilot=True, seeds=[seed], duration_s=TINY_DURATION_S)
              if job["labels"]["outage_length_s"] == 2.0 and job["labels"]["T_s"] == 0.30
              and job["labels"]["degraded_action"] == "hold" and job["labels"]["target_class"] == "turn"]
    jobs = []
    for job in chosen:
        scenario = copy.deepcopy(job["scenario"])
        scenario["outages"] = ({"start_s": 2.0, "end_s": 4.0, "streams": None},)
        jobs.append(make_job(job["labels"], scenario))
    return jobs


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp029b")
    csv_path = _write_part_a_fixture(base / "part_a")
    jobs = _tiny_jobs()
    assert {job["labels"]["preset"] for job in jobs} == {"clean", "burst_10_0.5s"}
    a, b = base / "a", base / "b"
    argv_a = ["--output", str(a), "--pilot", "--part-a", str(csv_path)]
    argv_b = ["--output", str(b), "--pilot", "--part-a", str(csv_path)]
    part_b.run(a, argv_a, pilot=True, processes=1, jobs=jobs, duration_s=TINY_DURATION_S, part_a_csv=csv_path)
    part_b.run(b, argv_b, pilot=True, processes=2, jobs=jobs, duration_s=TINY_DURATION_S, part_a_csv=csv_path)
    return a, b, argv_a, csv_path


def test_tiny_run_writes_byte_stable_outputs(tiny_dirs):
    a, b, argv_a, csv_path = tiny_dirs
    assert set(part_b.OUTPUTS) | {"manifest.json"} == {path.name for path in a.iterdir()}
    for name in part_b.OUTPUTS:
        assert _sha(a / name) == _sha(b / name), name
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    assert ma["replay_command"] == replay_command("tools/e_exp029_part_b.py", argv_a)
    assert "--pilot" in ma["replay_command"]
    assert ma["config"]["part_a_csv_sha256"] == _sha(csv_path)
    for key in (*VOLATILE_MANIFEST_FIELDS, "argv", "replay_command"):
        ma.pop(key, None)
        mb.pop(key, None)
    assert ma == mb


def test_tiny_run_covers_the_blackout(tiny_dirs):
    a = tiny_dirs[0]
    summary = json.loads((a / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "EXP-029 Part B"
    point = [s for s in summary["summaries"] if s["adapter"] == "point_mass"]
    assert {s["preset"] for s in point} == {"clean", "burst_10_0.5s"}
    clean = next(s for s in point if s["preset"] == "clean")
    # On the clean link the follower is airborne at 2.0 s and its failsafe holds during
    # the blackout: proof the blackout happened inside the run.
    assert clean["grounded_before_blackout"] == 0
    assert clean["outage_hold_link_age_s"]["n"] == 1
    assert clean["outage_hold_link_age_s"]["median"] >= 0.15
    assert summary["hand_check"]["iid_30"]["status"] == "missing_metrics"
    assert any("iid_30" in item for item in summary["missing_metrics"])
    assert summary["hand_check"]["burst_10_0.5s"]["status"] == "computed"
    recs = {item["preset"]: item for item in summary["recommendation"]}
    assert recs["iid_1"]["at_recommended_T"] == "not run in Part B (Addendum C)"
    assert recs["iid_10"]["at_recommended_T"] == "not in this run's job list (pilot or test subset)"
    assert summary["prediction"]["prediction_source"].startswith("pre-registered before the runs")
