"""Checks for the EXP-039 proposal runner (tools/d_run_coop_noise.py). Simulated only."""
import csv
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.coop.observers import run_seed
from sim.tests.platform_compare import EXACT, same_text
from tools import d_run_coop_noise as cn
from tools.d_common import EXPERIMENT_OFFSETS, seeds_for
from tools.d_run_tolerance import _cell

ROOT = Path(__file__).resolve().parents[2]
REGISTERED = ROOT / "sim" / "runs" / "d3-tolerance"
REGISTRY = ROOT / "tools" / "reproduce_registry.json"


@pytest.fixture(scope="module", autouse=True)
def platform_gate():
    """The EXP-024 rows were saved on macOS, where the gate stays character for character.

    Other platforms differ in the last digit or two, so there the gate uses the shared float tolerance.
    """
    with pytest.MonkeyPatch.context() as patch:
        if not EXACT:
            patch.setattr(cn, "_same_text", same_text)
        yield


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_rows(target, loss_model, loss, arm, n, delay):
    rows = {}
    with gzip.open(REGISTERED / "per_seed.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["link"] == "abstract" and row["target"] == target and row["loss_model"] == loss_model
                    and float(row["loss"]) == loss and row["arm"] == arm and int(row["n"]) == n
                    and float(row["delay"]) == delay):
                rows[int(row["seed"])] = row
    return rows


def test_registered_tolerance_outputs_are_untouched():
    # The EXP-024 sources have had opt-in, default-off additions since that run was registered
    # (for example the turn-aware filter option in sim/coop/observers.py), so their hashes no
    # longer match its manifest. What this experiment relies on is that the registered EXP-024
    # outputs are unchanged and that the 5 cm rows still come out the same; the second part is
    # the reproduction gate (test_job_at_five_cm_reproduces_registered_rows and the pilot gate).
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = next(e for e in entries if e["run_dir"] == "sim/runs/d3-tolerance")
    for name, digest in entry["outputs"].items():
        assert _digest(REGISTERED / name) == digest, name


def test_seed_list_starts_with_registered_seeds_and_new_ones_are_disjoint():
    seeds = cn.experiment_seeds(60)
    assert seeds[:20] == seeds_for("tolerance", 20)
    assert len(set(seeds)) == 60
    new = set(seeds[20:])
    for name in EXPERIMENT_OFFSETS:
        assert not new & set(seeds_for(name, 2000)), name


def test_told_five_cm_is_the_same_filter_as_the_registered_default():
    seed = seeds_for("tolerance", 20)[0]
    for n, arm in ((1, "single"), (3, "central")):
        base = _cell({"link": "abstract", "target": "ct_r1", "loss_model": "burst", "loss": 0.3,
                      "n": n, "delay": 0.1, "seed": seed})
        default = run_seed(base, seed, arms=(arm,))[arm]
        explicit = run_seed(replace(base, r_std_m=0.05), seed, arms=(arm,))[arm]
        assert repr(default.p95_error_m) == repr(explicit.p95_error_m)
        assert repr(default.max_error_m) == repr(explicit.max_error_m)
        assert repr(default.mean_nees) == repr(explicit.mean_nees)


@pytest.mark.parametrize("target,loss_model,loss,n,delay", [
    ("ct_r1", "burst", 0.3, 3, 0.2), ("ct_r1", "burst", 0.3, 1, 0.0),
    ("jink", "independent", 0.5, 3, 0.15), ("jink", "independent", 0.3, 1, 0.1),
])
def test_job_at_five_cm_reproduces_registered_rows(target, loss_model, loss, n, delay):
    arm = "single" if n == 1 else "central"
    registered = _registered_rows(target, loss_model, loss, arm, n, delay)
    for seed in seeds_for("tolerance", 20)[:3]:
        job = cn.make_job("main", target, loss_model, loss, n, "random", delay, 0.05, "told5", seed)
        row = cn._job(job)["rows"][0]
        expected = registered[seed]
        for field in cn.GATE_FIELDS:
            assert same_text(cn.csv_text(row[field]), expected[field]), (field, seed)


def test_noise_and_told_value_reach_the_simulation():
    seed = cn.experiment_seeds(60)[25]
    args = ("main", "ct_r1", "independent", 0.3, 3, "random", 0.1)
    five = cn._job(cn.make_job(*args, 0.05, "told5", seed))["rows"][0]
    ten = cn._job(cn.make_job(*args, 0.10, "told5", seed))["rows"][0]
    ten_truth = cn._job(cn.make_job(*args, 0.10, "truth", seed))["rows"][0]
    assert ten["p95"] != five["p95"]
    assert ten_truth["p95"] != ten["p95"]
    assert ten_truth["r_std_m"] == 0.10 and ten["r_std_m"] == 0.05


def test_zero_noise_single_observer_on_a_straight_line_is_almost_exact():
    seed = cn.experiment_seeds(60)[0]
    row = cn._job(cn.make_job("check", "cv_1", "independent", 0.0, 1, "random", 0.0,
                              0.0, "told5", seed))["rows"][0]
    assert row["p95"] < 0.005
    with pytest.raises(ValueError):
        cn.make_job("check", "cv_1", "independent", 0.0, 1, "random", 0.0, 0.0, "truth", seed)


def test_sync_phase_reaches_the_cell():
    seed = cn.experiment_seeds(60)[30]
    random_row = cn._job(cn.make_job("check", "cv_1", "independent", 0.0, 3, "random", 0.0,
                                     0.05, "told5", seed))["rows"][0]
    sync_row = cn._job(cn.make_job("check", "cv_1", "independent", 0.0, 3, "sync", 0.0,
                                   0.05, "told5", seed))["rows"][0]
    assert random_row["p95"] != sync_row["p95"]


@pytest.mark.parametrize("boundary,index", [
    ({"fails_at_zero": True, "censored": None, "value": None}, 0),
    ({"fails_at_zero": False, "censored": None, "value": 0.0}, 1),
    ({"fails_at_zero": False, "censored": None, "value": 0.1}, 4),
    ({"fails_at_zero": False, "censored": None, "value": 0.2}, 6),
    ({"fails_at_zero": False, "censored": ">0.3", "value": None}, 8),
])
def test_step_index(boundary, index):
    assert cn.step_index(boundary) == index


def test_step_index_rejects_off_grid_delays():
    with pytest.raises(ValueError):
        cn.step_index({"fails_at_zero": False, "censored": None, "value": 0.3})


def test_bootstrap_matches_a_plain_loop():
    diffs = np.array([0.0, 1.0, 0.5, 0.25, 0.0, 1.0, 0.75])
    mean, low, high = cn.bootstrap_mean(diffs, resamples=500, seed=39)
    rng = np.random.default_rng(39)
    picks = rng.integers(0, len(diffs), size=(500, len(diffs)))
    loop = sorted(sum(diffs[i] for i in row) / len(row) for row in picks)
    assert mean == pytest.approx(float(np.mean(diffs)))
    assert low == pytest.approx(float(np.percentile(loop, 2.5)))
    assert high == pytest.approx(float(np.percentile(loop, 97.5)))
    assert cn.bootstrap_mean(np.full(9, 0.5), resamples=200, seed=1) == (0.5, 0.5, 0.5)


@pytest.mark.parametrize("holds,label", [
    ({"2|told5": True, "2|truth": True, "10|told5": True, "10|truth": True}, "holds across noise"),
    ({"2|told5": False, "2|truth": True, "10|told5": True, "10|truth": True},
     "holds only when told the truth"),
    ({"2|told5": True, "2|truth": True, "10|told5": False, "10|truth": False}, "breaks at 10 cm"),
    ({"2|told5": True, "2|truth": False, "10|told5": True, "10|truth": True}, "mixed"),
])
def test_verdict_labels(holds, label):
    assert cn.verdict(holds) == label


@pytest.fixture(scope="module")
def pilot_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("coop-noise-pilot")
    one, two = root / "one", root / "two"
    cn.main(["--pilot", "--processes", "2", "--output", str(one)])
    cn.main(["--pilot", "--processes", "1", "--output", str(two)])
    return one, two


def test_pilot_outputs_are_byte_identical(pilot_outputs):
    one, two = pilot_outputs
    manifest = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]
    for name in manifest["outputs"]:
        assert _digest(one / name) == _digest(two / name), name


def test_pilot_gate_passes_and_rows_cover_the_grid(pilot_outputs):
    one, _ = pilot_outputs
    gate = json.loads((one / "reproduction.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True and gate["rows_compared"] > 0
    with gzip.open(one / "per_seed.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(r["slice"], r["target"], r["loss_model"], r["loss"], r["n"], r["phase"], r["delay"],
             r["noise_std_m"], r["told"], r["seed"]) for r in rows}
    assert len(keys) == len(rows) == cn.expected_row_count(pilot=True)


def test_pilot_boundaries_rebuild_from_rows(pilot_outputs):
    one, _ = pilot_outputs
    boundaries = json.loads((one / "boundaries.json").read_text(encoding="utf-8"))
    with gzip.open(one / "per_seed.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["slice"] == "main"]
    seeds = sorted({int(r["seed"]) for r in rows})
    for key, entry in boundaries["all_seeds"].items():
        target, loss, noise, told, n = key.split("|")
        model, probability = loss.split(":")
        counts = []
        for delay in cn.PILOT_DELAYS:
            counts.append(sum(r["failed"] == "true" for r in rows if r["target"] == target
                              and r["loss_model"] == model and float(r["loss"]) == float(probability)
                              and float(r["noise_std_m"]) == float(noise) and r["told"] == told
                              and int(r["n"]) == int(n) and float(r["delay"]) == delay))
        assert counts == entry["fail_counts"], key
        assert sum(counts) <= len(seeds) * len(cn.PILOT_DELAYS)


def _reference():
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "coop_noise_hand.py")],
                          check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


def test_hand_check_passes_on_reference_values_and_fails_when_off():
    ref = _reference()
    check = {}
    for noise, told in cn.NOISE_PAIRS:
        if noise == 0.0:
            continue
        r_told = 0.05 if told == "told5" else noise
        part = ref["part_a"][f"true{noise:.2f}_told{r_told:.2f}"]
        stem = f"cv_1|{noise:g}|{told}"
        check[f"{stem}|1|random"] = {"rmse_median": 1.0, "p95_median": 0.0}
        check[f"{stem}|3|random"] = {"rmse_median": part["ratio_N3stag_over_N1"], "p95_median": 0.0}
        check[f"{stem}|3|sync"] = {"rmse_median": part["ratio_N3sync_over_N1"], "p95_median": 0.0}
    check["cv_1|0|told5|1|random"] = {"rmse_median": 0.0,
                                      "p95_median": ref["part_b"]["straight_line_1mps"]["N1"]["p95_abs"]}
    check["ct_r1|0|told5|1|random"] = {"rmse_median": 0.0,
                                       "p95_median": ref["part_b"]["circle_r1_v1"]["N1"]["p95"]}
    assert cn._hand_check(check)["passed"] is True
    check["cv_1|0.1|told5|3|sync"]["rmse_median"] += 0.05
    assert cn._hand_check(check)["passed"] is False


def test_hand_check_fails_closed_when_a_cell_is_missing():
    assert cn._hand_check({})["passed"] is False


def test_gate_fails_closed_on_missing_or_duplicated_rows(pilot_outputs):
    one, _ = pilot_outputs
    with gzip.open(one / "per_seed.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["loss"], row["delay"], row["noise_std_m"] = (float(row["loss"]), float(row["delay"]),
                                                         float(row["noise_std_m"]))
        row["failed"] = row["failed"] == "true"
        for field in ("p95", "max", "rmse"):
            row[field] = float(row[field])
        row["mean_nees"] = float(row["mean_nees"]) if row["mean_nees"] else None
    registered = set(seeds_for("tolerance", 20)) & set(cn.pilot_seeds())
    assert cn._gate(rows, registered, cn.PILOT_DELAYS, {}, 1)["passed"] is True
    gate_rows = [i for i, r in enumerate(rows) if r["slice"] == "main" and r["noise_std_m"] == 0.05]
    fewer = rows[:gate_rows[0]] + rows[gate_rows[0] + 1:]
    assert cn._gate(fewer, registered, cn.PILOT_DELAYS, {}, 1)["passed"] is False
    assert cn._gate(rows + [rows[gate_rows[0]]], registered, cn.PILOT_DELAYS, {}, 1)["passed"] is False
    assert cn._gate(rows, registered, cn.PILOT_DELAYS, {}, 2)["passed"] is False
