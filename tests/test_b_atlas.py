"""Checks for the S6 recovery atlas builder and its strict schema.

Most tests feed hand-made 50-seed fail counts through the real Wilson
interval and compare the atlas against bounds worked out by hand. Label tests
get their counterfactual records from the runner's own builder so the key
format is the real one.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

import tools.run_s6_atlas as runner
from sim.estimation.atlas import build_atlas, validate_atlas
from sim.estimation.base import NAIVE, RECOVERY_V1, EstimatorSettings
from sim.s6_recovery import RecoveryConfig, wilson_interval


DELAYS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
N = 50
TARGETS = {"cv_1": ("cv", {}), "ct_r3": ("ct", {"radius_m": 3.0})}


def _arm(fails: int, p95: float = 1.0) -> dict:
    """Arm summary for ``fails`` failing seeds of 50, using the real Wilson interval."""
    low, high = wilson_interval(fails, N)
    return {"n_seeds": N, "p95_median_m": p95, "fail_rate": fails / N,
            "fail_rate_ci_low": low, "fail_rate_ci_high": high}


def _paired(median: float = 0.1, low: float = 0.05, high: float = 0.15) -> dict:
    return {"n_seeds": N, "p95_diff_median_m": median, "p95_diff_ci_low_m": low, "p95_diff_ci_high_m": high}


def _row(*, delay: float, fails: int, stratum: str = "central", variant: str = "cv_1",
         loss_model: str = "independent", loss: float = 0.0, noise_std: float = 0.05,
         jitter: float = 0.0, arm: str = "recovery", paired: dict | None = None) -> dict:
    target, params = TARGETS[variant]
    cell = {"target": target, "target_params": dict(params), "rate_hz": 10.0, "delay_s": delay,
            "jitter_s": jitter, "loss_probability": loss, "loss_model": loss_model,
            "noise_model": "gaussian", "noise_std_m": noise_std}
    return {"slice": stratum, "stratum": stratum, "variant": variant,
            "key": f"{stratum}/{variant}/{loss_model}:{loss:g}/d{delay:g}", "cell": cell,
            "arms": {"naive": _arm(0), arm: _arm(fails)},
            "paired_vs_naive": {arm: paired or _paired()}, "paired_vs_recovery": {}}


def _series(counts, **kwargs) -> list[dict]:
    assert len(counts) == len(DELAYS)
    return [_row(delay=d, fails=c, jitter=(d / 2 if kwargs.get("stratum") == "jitter" else 0.0), **kwargs)
            for d, c in zip(DELAYS, counts)]


def _stratum_definition(name: str) -> dict:
    return {"rate_hz": 10.0, "noise_model": "gaussian", "noise_std_m": 0.0 if name == "noise0" else 0.05,
            "noise_dof": None, "jitter_rule": "jitter_s = delay_s / 2" if name == "jitter" else "none"}


def _summary(cells: list[dict]) -> dict:
    names = sorted({row["stratum"] for row in cells})
    return {
        "experiment": "s6-atlas", "simulated": True,
        "failure_rule": {"cell_rule": "fail rate >= 0.5", "fail_rate_threshold": 0.5,
                         "cell_rule_status": "proposed, not yet adopted",
                         "per_seed": {"p95_error_m": 0.25, "max_error_m": 0.5, "scored_window_s": [2.0, 12.0]}},
        "design": {"name": "stratified one-factor strata around a central plane",
                   "reason": "a full factorial is too costly", "burst_mean_s": 0.5,
                   "imm_strata": ["central"], "speed_variants_central_only": False,
                   "arms_by_stratum": {name: ["naive", "recovery"] for name in names},
                   "options": {"imm_strata": ["central"], "speed_variants_central_only": False}},
        "design_reason": "a full factorial is too costly",
        "seeds": list(range(100, 100 + N)),
        "arms": [NAIVE.as_dict(), RECOVERY_V1.as_dict()],
        "strata": {name: _stratum_definition(name) for name in names},
        "limits": ["simulated link and sensor only"],
        "cells": cells,
    }


KIND_MEDIANS = {"delay_removed": 0.4, "loss_removed": 0.5, "noise_removed": 0.6, "all_removed": 0.1}


def _runner_counterfactuals(cells: list[dict], medians: dict[str, float] = KIND_MEDIANS) -> dict:
    """The runner's own counterfactuals.json document for hand-made cells."""
    failures = runner._first_failures(cells)
    jobs = runner._counterfactual_jobs(failures, [1], RecoveryConfig(), EstimatorSettings())
    results = [{"arms": {job["arm_names"][0]: {"p95_median_m": medians[job["counterfactual"]]}}}
               for job in jobs]
    return runner._build_counterfactual_doc(failures, jobs, results)


def _build(cells: list[dict], medians: dict[str, float] = KIND_MEDIANS) -> dict:
    return build_atlas(_summary(cells), _runner_counterfactuals(cells, medians))


def _entry(doc: dict, *, stratum: str = "central", variant: str = "cv_1", arm: str = "recovery",
           loss: str = "independent:0") -> dict:
    return next(item for item in doc["entries"]
                if (item["stratum"], item["variant"], item["arm"], item["loss_condition"])
                == (stratum, variant, arm, loss))


# ---------------------------------------------------------------- Wilson anchors

def test_wilson_switch_points_match_hand_computed_values():
    # Hand worked for n = 50, z = 1.95996: the upper bound first reaches 0.5 at
    # 19 failing seeds and the lower bound first reaches 0.5 at 32.
    assert wilson_interval(18, N)[1] == pytest.approx(0.4986, abs=5e-4)
    assert wilson_interval(19, N)[1] == pytest.approx(0.5185, abs=5e-4)
    assert wilson_interval(31, N)[0] == pytest.approx(0.4815, abs=5e-4)
    assert wilson_interval(32, N)[0] == pytest.approx(0.5014, abs=5e-4)
    assert wilson_interval(25, N) == pytest.approx((0.3664, 0.6336), abs=5e-4)


# ------------------------------------------------------- tolerated delay and interval

@pytest.mark.parametrize(
    ("counts", "value", "censored", "fails_at_zero", "interval"),
    [
        # Nothing fails and the upper bound never reaches 0.5: censored, never 0.3.
        ((0, 1, 2, 3, 5, 8, 10), None, ">0.3", False, [">0.3", ">0.3"]),
        # 25/50 first at 0.15 s, so the value is 0.10. The upper bound reaches 0.5
        # at 19 failing seeds (0.10 s, end 0.05); the lower bound at 32 (0.30 s, end 0.20).
        ((0, 5, 18, 19, 25, 31, 32), 0.10, None, False, [0.05, 0.20]),
        # One seed short of the upper-bound switch: 18 never crosses.
        ((0, 5, 10, 15, 18, 18, 18), None, ">0.3", False, [">0.3", ">0.3"]),
        # Point estimate fails at 31/50 (0.62) but the lower bound (0.4815) has not crossed.
        ((0, 5, 18, 19, 25, 31, 31), 0.10, None, False, [0.05, ">0.3"]),
        # Upper bound crosses at 0.10 s but no point estimate reaches 0.5.
        ((0, 0, 0, 19, 20, 24, 24), None, ">0.3", False, [0.05, ">0.3"]),
        # 19/50 at 0 s: "may fail at 0 s" is "<0", never the number 0.0.
        ((19, 10, 10, 10, 10, 10, 10), None, ">0.3", False, ["<0", ">0.3"]),
        # 25/50 at 0 s: point estimate fails at 0 s, lower bound never reaches 0.5.
        ((25, 25, 25, 25, 25, 25, 25), None, None, True, ["<0", ">0.3"]),
        # 32/50 at 0 s: even the lower bound reaches 0.5 at 0 s.
        ((32, 40, 45, 50, 50, 50, 50), None, None, True, ["<0", "<0"]),
        # Tolerates 0 s, fails at 0.05 s: a real 0.025 s, not the marker.
        ((0, 0, 50, 50, 50, 50, 50), 0.025, None, False, [0.025, 0.025]),
        # Tolerates 0 s and fails at 0.025 s: the answer 0.0 is a number here.
        ((0, 50, 50, 50, 50, 50, 50), 0.0, None, False, [0.0, 0.0]),
    ],
)
def test_tolerated_delay_from_fail_counts(counts, value, censored, fails_at_zero, interval):
    tolerated = _entry(_build(_series(counts)))["max_tolerated_delay_s"]
    assert tolerated["value"] == value
    assert tolerated["censored"] == censored
    assert tolerated["fails_at_zero"] is fails_at_zero
    assert tolerated["interval_s"] == interval


def test_censored_label_follows_largest_tested_delay():
    rows = [_row(delay=d, fails=0) for d in (0.0, 0.1)]
    tolerated = _entry(build_atlas(_summary(rows), {}))["max_tolerated_delay_s"]
    assert tolerated["censored"] == ">0.1"
    assert tolerated["interval_s"] == [">0.1", ">0.1"]


def test_interval_never_contradicts_value_on_random_series():
    rng = random.Random(7)

    def position(end):
        return -1.0 if end == "<0" else float("inf") if isinstance(end, str) else float(end)

    for _ in range(30):
        counts = [rng.randint(0, N) for _ in DELAYS]
        tolerated = _entry(_build(_series(counts)))["max_tolerated_delay_s"]
        low, high = (position(end) for end in tolerated["interval_s"])
        assert low <= high, counts
        if tolerated["value"] is not None:
            assert low <= tolerated["value"] <= high, counts
        elif tolerated["censored"] is not None:
            assert high == float("inf"), counts


# ------------------------------------------------------------ missing data and keys

@pytest.mark.parametrize("field", ["fail_rate", "fail_rate_ci_low", "fail_rate_ci_high"])
def test_missing_fail_rate_fields_raise(field):
    rows = _series((50,) * 7)
    del rows[3]["arms"]["recovery"][field]
    with pytest.raises(ValueError, match=field):
        build_atlas(_summary(rows), {})


def test_non_finite_fail_rate_raises():
    rows = _series((0,) * 7)
    rows[2]["arms"]["recovery"]["fail_rate_ci_high"] = float("nan")
    with pytest.raises(ValueError, match="fail_rate_ci_high"):
        build_atlas(_summary(rows), {})


def test_failing_series_without_counterfactual_raises():
    with pytest.raises(ValueError, match="central\\|cv_1\\|independent:0\\|recovery"):
        build_atlas(_summary(_series((0, 0, 0, 0, 30, 40, 50))), {})


def test_counterfactual_under_a_near_miss_key_is_not_used():
    rows = _series((0, 0, 0, 0, 30, 40, 50))
    record = _runner_counterfactuals(rows)["central|cv_1|independent:0|recovery"]
    for key in ("central/cv_1/independent:0/recovery", "central|cv_1|independent:0.1|recovery",
                "central|cv_1|independent|recovery", "jitter|cv_1|independent:0|recovery"):
        with pytest.raises(ValueError):
            build_atlas(_summary(rows), {key: record})


def test_counterfactual_for_another_delay_or_median_is_rejected():
    rows = _series((0, 0, 0, 0, 30, 40, 50))
    key = "central|cv_1|independent:0|recovery"
    good = _runner_counterfactuals(rows)
    for field, bad, message in (("delay_s", 0.2, "first fails at 0.15"),
                                ("original_median_p95_m", 9.0, "does not match the cell median")):
        doc = copy.deepcopy(good)
        doc[key][field] = bad
        with pytest.raises(ValueError, match=message):
            build_atlas(_summary(rows), doc)


def test_counterfactuals_do_not_leak_to_series_that_never_fail():
    failing = _series((0, 0, 0, 0, 30, 40, 50), variant="cv_1")
    clean = _series((0,) * 7, variant="ct_r3")
    doc = build_atlas(_summary(failing + clean), _runner_counterfactuals(failing + clean))
    assert _entry(doc)["counterfactual_medians"] is not None
    assert _entry(doc)["failure_mode"] is not None
    for arm in ("naive", "recovery"):
        leaked = _entry(doc, variant="ct_r3", arm=arm)
        assert leaked["failure_mode"] is None
        assert "counterfactual_medians" not in leaked
    # Same arm, same variant, another loss condition: no record, no medians.
    other_loss = _series((0,) * 7, loss_model="independent", loss=0.1)
    doc = build_atlas(_summary(failing + other_loss), _runner_counterfactuals(failing))
    assert "counterfactual_medians" not in _entry(doc, loss="independent:0.1")


@pytest.mark.parametrize("missing", ["medians", "delay_s", "original_median_p95_m"])
def test_incomplete_counterfactual_record_raises(missing):
    rows = _series((0, 0, 0, 0, 30, 40, 50))
    doc = _runner_counterfactuals(rows)
    del doc["central|cv_1|independent:0|recovery"][missing]
    with pytest.raises(ValueError, match=missing):
        build_atlas(_summary(rows), doc)


def test_missing_all_removed_or_applicable_removal_median_raises():
    rows = _series((0, 0, 0, 0, 30, 40, 50))
    for name in ("all_removed", "delay_removed"):
        doc = _runner_counterfactuals(rows)
        doc["central|cv_1|independent:0|recovery"]["medians"][name] = None
        with pytest.raises(ValueError):
            build_atlas(_summary(rows), doc)


# ---------------------------------------------------------------- failure-mode labels

FAIL_AT_01_CELL = dict(delay=0.10, loss=0.1)


@pytest.mark.parametrize(
    ("label", "medians"),
    [
        # Original median p95 is 1.0, so D_x = 1.0 - median_x.
        ("lag-dominated", {"delay_removed": 0.2, "loss_removed": 0.7, "noise_removed": 0.8, "all_removed": 0.05}),
        ("gap-dominated", {"delay_removed": 0.7, "loss_removed": 0.2, "noise_removed": 0.8, "all_removed": 0.05}),
        ("noise-dominated", {"delay_removed": 0.8, "loss_removed": 0.7, "noise_removed": 0.2, "all_removed": 0.05}),
        # Removing everything is not better than the best single removal (D = 0.4): mismatch.
        ("model-mismatch", {"delay_removed": 0.6, "loss_removed": 0.9, "noise_removed": 0.9, "all_removed": 0.6}),
        # Largest drop D = 0.6 (delay) and E_all = 0.6: equal counts as mismatch (E_all >= D).
        ("model-mismatch", {"delay_removed": 0.4, "loss_removed": 0.9, "noise_removed": 0.9, "all_removed": 0.6}),
        # Tie between delay and loss goes to delay.
        ("lag-dominated", {"delay_removed": 0.5, "loss_removed": 0.5, "noise_removed": 0.9, "all_removed": 0.1}),
    ],
)
def test_failure_label_matches_runner_record(label, medians):
    rows = [_row(delay=0.10, fails=50, loss=0.1), _row(delay=0.0, fails=0, loss=0.1)]
    counterfactuals = _runner_counterfactuals(rows, medians)
    doc = build_atlas(_summary(rows), counterfactuals)
    entry = _entry(doc, loss="independent:0.1")
    record = counterfactuals["central|cv_1|independent:0.1|recovery"]
    assert entry["failure_mode"] == label
    assert record["failure_mode"] == label  # the runner agrees
    assert entry["counterfactual_medians"] == {
        "delay": medians["delay_removed"], "loss": medians["loss_removed"],
        "noise": medians["noise_removed"], "all": medians["all_removed"]}


def test_removals_that_do_not_change_the_cell_are_skipped():
    # Zero delay and zero loss: only noise removal applies. The unused removals
    # carry no medians, so the largest single drop is the noise one.
    rows = [_row(delay=0.0, fails=50)]
    counterfactuals = _runner_counterfactuals(rows)
    entry = _entry(build_atlas(_summary(rows), counterfactuals))
    assert entry["failure_mode"] == "noise-dominated"
    assert entry["counterfactual_medians"] == {"delay": None, "loss": None, "noise": 0.6, "all": 0.1}
    assert counterfactuals["central|cv_1|independent:0|recovery"]["failure_mode"] == "noise-dominated"


def test_jitter_alone_makes_delay_removal_apply():
    rows = _series((0, 0, 0, 0, 50, 50, 50), stratum="jitter")
    entry = _entry(build_atlas(_summary(rows), _runner_counterfactuals(rows, {
        "delay_removed": 0.2, "loss_removed": 0.7, "noise_removed": 0.8, "all_removed": 0.05})),
        stratum="jitter")
    assert entry["failure_mode"] == "lag-dominated"


@pytest.mark.parametrize("fails", [50, 30])
def test_failing_with_nothing_left_to_remove_is_model_mismatch(fails):
    # noise0 stratum, delay 0, loss 0: no removal applies, but the cell fails.
    rows = [_row(delay=0.0, fails=fails, stratum="noise0", noise_std=0.0)]
    counterfactuals = _runner_counterfactuals(rows)
    entry = _entry(build_atlas(_summary(rows), counterfactuals), stratum="noise0")
    record = counterfactuals["noise0|cv_1|independent:0|recovery"]
    assert entry["failure_mode"] == "model-mismatch"
    assert record["failure_mode"] == "model-mismatch"
    assert entry["counterfactual_medians"] == {"delay": None, "loss": None, "noise": None, "all": 0.1}


def test_label_that_disagrees_with_runner_record_is_an_error():
    rows = _series((0, 0, 0, 0, 30, 40, 50))
    doc = _runner_counterfactuals(rows)
    record = doc["central|cv_1|independent:0|recovery"]
    assert record["failure_mode"] == "lag-dominated"
    record["failure_mode"] = "gap-dominated"
    with pytest.raises(ValueError, match="disagrees"):
        build_atlas(_summary(rows), doc)


# ------------------------------------------------------ self-describing document

def test_atlas_describes_its_own_rules_design_and_strata():
    rows = _series((0,) * 7) + _series((0,) * 7, stratum="jitter")
    doc = build_atlas(_summary(rows), {})
    assert doc["schema"] == "1.0"
    assert doc["failure_rule"]["per_seed"] == {"p95_error_m": 0.25, "max_error_m": 0.5, "scored_window_s": [2.0, 12.0]}
    assert doc["failure_rule"]["cell_rule"] == "fail rate >= 0.5"
    assert doc["failure_rule"]["cell_rule_status"] == "proposed, not yet adopted"
    assert doc["design"]["burst_mean_s"] == 0.5
    assert doc["design"]["arms_by_stratum"] == {"central": ["naive", "recovery"], "jitter": ["naive", "recovery"]}
    assert doc["design"]["options"] == {"imm_strata": ["central"], "speed_variants_central_only": False}
    assert doc["strata"]["jitter"]["jitter_rule"] == "jitter_s = delay_s / 2"
    assert doc["strata"]["central"]["jitter_rule"] == "none"
    assert doc["seeds"] == list(range(100, 150))
    assert [arm["name"] for arm in doc["arms"]] == ["naive", "recovery"]
    text = " ".join(doc["limits"])
    assert "max-error rule" in text and "model-mismatch" in text and "<0" in text


@pytest.mark.parametrize("field", ["failure_rule", "design", "design_reason", "seeds", "arms", "strata", "limits"])
def test_missing_summary_fields_raise(field):
    summary = _summary(_series((0,) * 7))
    del summary[field]
    with pytest.raises(ValueError, match=field):
        build_atlas(summary, {})


@pytest.mark.parametrize(
    "path",
    [("failure_rule", "per_seed", "p95_error_m"), ("failure_rule", "per_seed", "scored_window_s"),
     ("failure_rule", "cell_rule_status"), ("design", "burst_mean_s"), ("design", "arms_by_stratum"),
     ("design", "options"), ("design", "imm_strata"), ("strata", "central", "jitter_rule"),
     ("strata", "central", "noise_dof"), ("arms", 0, "model_options")],
)
def test_schema_requires_self_describing_fields(path):
    doc = build_atlas(_summary(_series((0,) * 7)), {})
    parent = doc
    for step in path[:-1]:
        parent = parent[step]
    del parent[path[-1]]
    with pytest.raises(Exception):
        validate_atlas(doc)


@pytest.mark.parametrize(
    "path",
    [(), ("failure_rule",), ("failure_rule", "per_seed"), ("design",), ("design", "options"),
     ("strata", "central"), ("arms", 0), ("entries", 0), ("entries", 0, "max_tolerated_delay_s"),
     ("entries", 1, "improvement_vs_naive", 0)],
)
def test_schema_rejects_extra_keys_at_every_object_level(path):
    doc = build_atlas(_summary(_series((0,) * 7)), {})
    assert validate_atlas(doc)
    target = doc
    for step in path:
        target = target[step]
    target["unexpected"] = 1
    with pytest.raises(Exception):
        validate_atlas(doc)


def test_schema_rejects_wrong_thresholds_and_version():
    doc = build_atlas(_summary(_series((0,) * 7)), {})
    for mutate in (lambda d: d["failure_rule"]["per_seed"].update(p95_error_m=0.3),
                   lambda d: d["failure_rule"]["per_seed"].update(max_error_m=0.4),
                   lambda d: d["failure_rule"].update(cell_rule_status="Adopted"),
                   lambda d: d.update(schema="1.1")):
        broken = copy.deepcopy(doc)
        mutate(broken)
        with pytest.raises(Exception):
            validate_atlas(broken)


@pytest.mark.parametrize(
    "tolerated",
    [
        {"value": 0.3, "censored": ">0.3", "fails_at_zero": False, "interval_s": [0.2, ">0.3"]},
        {"value": 0.1, "censored": None, "fails_at_zero": True, "interval_s": [0.1, 0.1]},
        {"value": None, "censored": ">0.3", "fails_at_zero": True, "interval_s": ["<0", "<0"]},
        {"value": None, "censored": None, "fails_at_zero": False, "interval_s": [0.1, 0.2]},
        {"value": -0.1, "censored": None, "fails_at_zero": False, "interval_s": [0.0, 0.1]},
        {"value": 0.1, "censored": None, "fails_at_zero": False, "interval_s": ["<1", 0.1]},
        {"value": 0.1, "censored": None, "fails_at_zero": False, "interval_s": [0.0, "fails_at_zero"]},
    ],
)
def test_schema_rejects_contradictory_or_malformed_tolerated_delay(tolerated):
    doc = build_atlas(_summary(_series((0,) * 7)), {})
    doc["entries"][0]["max_tolerated_delay_s"] = tolerated
    with pytest.raises(Exception):
        validate_atlas(doc)


@pytest.mark.parametrize(
    "tolerated",
    [
        {"value": 0.2, "censored": None, "fails_at_zero": False, "interval_s": [0.1, 0.3]},
        {"value": 0.0, "censored": None, "fails_at_zero": False, "interval_s": [0.0, 0.0]},
        {"value": None, "censored": ">0.3", "fails_at_zero": False, "interval_s": ["<0", ">0.3"]},
        {"value": None, "censored": None, "fails_at_zero": True, "interval_s": ["<0", 0.2]},
    ],
)
def test_schema_accepts_the_three_states(tolerated):
    doc = build_atlas(_summary(_series((0,) * 7)), {})
    doc["entries"][0]["max_tolerated_delay_s"] = tolerated
    assert validate_atlas(doc)


# ---------------------------------------------------------- improvement vs naive

def test_improvement_vs_naive_keeps_each_delays_paired_difference():
    rows = [_row(delay=d, fails=0, paired=_paired(0.1 * (i + 1), 0.01 * (i + 1), 0.2 * (i + 1)))
            for i, d in enumerate(DELAYS)]
    entry = _entry(build_atlas(_summary(rows), {}))
    assert [item["delay_s"] for item in entry["improvement_vs_naive"]] == list(DELAYS)
    third = entry["improvement_vs_naive"][2]
    assert third["median_p95_diff_m"] == pytest.approx(0.3)
    assert third["interval_m"] == [pytest.approx(0.03), pytest.approx(0.6)]
    assert _entry(build_atlas(_summary(rows), {}), arm="naive")["improvement_vs_naive"] == []


def test_missing_paired_summary_raises():
    rows = _series((0,) * 7)
    del rows[0]["paired_vs_naive"]["recovery"]
    with pytest.raises(ValueError, match="recovery"):
        build_atlas(_summary(rows), {})


def test_repeated_delay_in_one_series_raises():
    rows = _series((0,) * 7)
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="repeated delay"):
        build_atlas(_summary(rows), {})


# ------------------------------------------------- real engine, runner to atlas

def test_failing_cells_through_run_sweep_reach_the_atlas_with_exact_counterfactuals(tmp_path, monkeypatch):
    cells = [c for c in runner.build_cells()
             if c["slice"] == "central" and c["target"] == "cv" and not c["target_params"]
             and c["loss_model"] == "independent" and c["loss_probability"] == 0.5
             and c["delay_s"] in (0.0, 0.3)]
    assert len(cells) == 2
    monkeypatch.setattr(runner, "build_cells", lambda *args, **kwargs: cells)
    runner.run_sweep(tmp_path / "out", seeds_count=2, processes=1, no_checkpoint=True, imm_strata="rate5")
    atlas = json.loads((tmp_path / "out" / "atlas.json").read_text())
    counterfactuals = json.loads((tmp_path / "out" / "counterfactuals.json").read_text())
    validate_atlas(atlas)
    by_arm = {entry["arm"]: entry for entry in atlas["entries"]}
    assert set(by_arm) == {"naive", "recovery", "adaptive"}
    naive = by_arm["naive"]
    assert naive["max_tolerated_delay_s"]["value"] == 0.0  # clean at 0 s, fails at 0.3 s
    failing = [entry for entry in atlas["entries"] if entry["failure_mode"] is not None]
    assert naive in failing
    for entry in atlas["entries"]:
        record = counterfactuals[f"central|cv_1|{entry['loss_condition']}|{entry['arm']}"] \
            if entry["failure_mode"] is not None else None
        if record is None:
            assert "counterfactual_medians" not in entry
            continue
        assert record["delay_s"] == 0.3
        assert entry["failure_mode"] == record["failure_mode"]
        assert entry["counterfactual_medians"] == {
            name: record["medians"][f"{name}_removed"] for name in ("delay", "loss", "noise", "all")}
