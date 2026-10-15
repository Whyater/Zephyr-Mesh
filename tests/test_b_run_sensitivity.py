"""Tests for the S6 sensitivity runner (simulation only).

Fast tests use a synthetic response function injected by name (`job_function`),
which the runner labels "synthetic_test". The real-engine tests at the bottom
run the actual simulation on a tiny design.
"""
from __future__ import annotations

import gzip
import inspect
import json
import shlex
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from sim.estimation.engine import deliver_cell
from sim.estimation.morris import morris_trajectories
from sim.s6_recovery import RecoveryConfig, seed_list
from tools import run_s6_sensitivity as RUNNER

HERE = "tests.test_b_run_sensitivity"
SYNTH = f"{HERE}:synthetic_job"
SYNTH_FILES = ("summary.json", "oat.json", "oat.csv", "morris.json", "ranking.json", "per_seed.csv.gz")


# ---------------------------------------------------------------------------
# Synthetic response function (tests only). Key order is deliberately not sorted
# so that a resumed run (sorted keys from checkpoints) would expose any
# dependence on dict order.
# ---------------------------------------------------------------------------

def _arm_summary(p95: float) -> dict:
    return {"p95_median_m": p95, "n_seeds": 2, "fail_rate": 0.0, "p95_ci_high_m": p95 + 0.01,
            "p95_ci_low_m": p95 - 0.01}


def synthetic_job(job: dict) -> dict:
    values = dict(zip(RUNNER.INPUTS, job["values"]))
    p95 = 100.0 * values["delay"] + 20.0 * values["loss"]
    if job.get("target_class") == "turn":
        p95 += 3.0 * values["turn_rate"]
    seeds = job["seeds"]
    rows = []
    for index, seed in enumerate(seeds):
        for arm, scale in (("recovery", 1.0), ("imm", 1.5)):
            rows.append({"x_imm_ct_prob_mean": 0.5, "failed": False, "seed": seed, "arm": arm,
                         "cell_key": "synthetic", "seed_index": index,
                         "p95_error_m": p95 * scale, "max_error_m": 2 * p95 * scale})
    return {"per_seed": rows, "arms": {"imm": _arm_summary(1.5 * p95), "recovery": _arm_summary(p95)},
            "slice": job["slice"], "values": job["values"], "case": job.get("case"),
            "input": job.get("input"), "key": "synthetic", "target_class": job.get("target_class")}


def _synthetic_run(out, **kwargs):
    kwargs.setdefault("seeds_count", 2)
    kwargs.setdefault("processes", 1)
    kwargs.setdefault("morris_r", 3)
    kwargs.setdefault("argv", ["--synthetic"])
    return RUNNER.run_experiment(out, job_function=SYNTH, **kwargs)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Mapping and cell construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_class", RUNNER.CLASSES)
def test_unit_mapping_endpoints_and_log_inputs(target_class):
    lows, highs = RUNNER.class_ranges(target_class)
    np.testing.assert_allclose(RUNNER.map_inputs(np.zeros(9), target_class), lows)
    np.testing.assert_allclose(RUNNER.map_inputs(np.ones(9), target_class), highs)
    mid = RUNNER.map_inputs(np.full(9, 0.5), target_class)
    for name in ("sensor_rate", "tuning_error"):
        i = RUNNER.INDEX[name]
        assert mid[i] == pytest.approx(np.sqrt(lows[i] * highs[i]))  # geometric midpoint
    for name in ("delay", "loss", "noise", "speed", "turn_rate", "clock_error"):
        i = RUNNER.INDEX[name]
        assert mid[i] == pytest.approx((lows[i] + highs[i]) / 2)  # linear midpoint


def test_turn_class_turn_rate_range_is_quarter_to_two():
    lows, highs = RUNNER.class_ranges("turn")
    i = RUNNER.INDEX["turn_rate"]
    assert (lows[i], highs[i]) == (0.25, 2.0)
    assert RUNNER.class_ranges("straight")[0][i] == 0.0
    # Registered ranges of the other inputs, written out independently.
    assert list(RUNNER.HIGHS) == [0.3, 0.5, 10.0, 0.10, 2.0, 2.0, 50.0, 0.1, 5.0]
    assert list(RUNNER.LOWS) == [0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 5.0, 0.0, 0.2]


def test_mapping_linear_zero_ranges_emit_no_log_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for target_class in RUNNER.CLASSES:
            RUNNER.map_inputs(np.zeros(9), target_class)
            RUNNER.map_inputs(np.ones(9), target_class)


def test_central_case_is_ten_percent_independent_loss():
    cell = RUNNER._cell(RUNNER.CENTERS)
    assert cell.loss_model == "independent"
    assert cell.loss_probability == 0.10
    assert cell.delay_s == 0.1 and cell.rate_hz == 10.0 and cell.noise_std_m == 0.05
    assert cell.target == "cv" and dict(cell.target_params) == {"speed_mps": 1.0}
    turn = RUNNER._cell(RUNNER.class_center("turn"), "turn")
    assert turn.target == "ct" and dict(turn.target_params) == {"speed_mps": 1.0, "radius_m": 1.0}
    assert turn.loss_model == "independent"


def test_burst_length_one_is_independent_and_longer_is_burst():
    for packets, model in ((1.0, "independent"), (1.5, "burst"), (4.0, "burst"), (10.0, "burst")):
        values = RUNNER.CENTERS.copy()
        values[RUNNER.INDEX["burst_mean_length"]] = packets
        cell = RUNNER._cell(values)
        assert cell.loss_model == model
        assert cell.burst_mean_length_packets == packets


def test_every_oat_point_but_burst_length_is_independent_loss():
    for job in RUNNER.build_oat_jobs():
        cell = RUNNER._cell(job["values"], job["target_class"])
        if job["input"] == "burst_mean_length" and job["level"] > 1.0:
            assert cell.loss_model == "burst", job["slice"]
        else:
            assert cell.loss_model == "independent", job["slice"]


def test_straight_class_ignores_turn_rate_and_turn_class_uses_radius():
    base = RUNNER.CENTERS.copy()
    other = base.copy()
    other[RUNNER.INDEX["turn_rate"]] = 1.7
    assert RUNNER._cell(base, "straight").key() == RUNNER._cell(other, "straight").key()
    values = RUNNER.class_center("turn").copy()
    values[RUNNER.INDEX["speed"]] = 2.0
    values[RUNNER.INDEX["turn_rate"]] = 0.5
    cell = RUNNER._cell(values, "turn")
    assert cell.target == "ct"
    assert dict(cell.target_params)["radius_m"] == pytest.approx(4.0)  # 2 m/s over 0.5 rad/s
    values[RUNNER.INDEX["turn_rate"]] = 0.0
    with pytest.raises(ValueError):
        RUNNER._cell(values, "turn")


def test_clock_error_makes_delivered_stamps_early():
    values = RUNNER.CENTERS.copy()
    values[RUNNER.INDEX["clock_error"]] = 0.1
    cell = RUNNER._cell(values)
    bundle = deliver_cell(cell, seed_list(1)[0], RecoveryConfig())
    assert bundle.deliveries
    lead = np.array([d.capture_time_s - d.stamp_s for d in bundle.deliveries])
    assert np.all(lead > 0), "every stamp must be earlier than the capture time"
    assert lead.mean() == pytest.approx(0.1, abs=0.02)
    assert RUNNER._cell(RUNNER.CENTERS).clock == ()


def test_r_factor_scales_filter_std_but_not_sensor_noise():
    values = RUNNER.CENTERS.copy()
    values[RUNNER.INDEX["tuning_error"]] = 2.0
    arms = RUNNER.build_arms(values)
    assert [a.name for a in arms] == list(RUNNER.ARM_NAMES)
    assert all(a.r_std_m == pytest.approx(0.1) for a in arms)
    assert RUNNER._cell(values).noise_std_m == 0.05
    assert all(a.r_std_m == pytest.approx(0.05) for a in RUNNER.build_arms())


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def test_oat_grid_sizes_levels_and_classes():
    jobs = RUNNER.build_oat_jobs()
    assert len(jobs) == 2 * 9 * 5
    assert len(RUNNER.build_oat_jobs(pilot=True)) == 2 * 9 * 3
    for case, target_class in (("cv", "straight"), ("ct", "turn")):
        lows, highs = RUNNER.class_ranges(target_class)
        for name in RUNNER.INPUTS:
            levels = [j["level"] for j in jobs if j["case"] == case and j["input"] == name]
            i = RUNNER.INDEX[name]
            assert len(levels) == 5
            assert levels[0] == pytest.approx(lows[i]) and levels[-1] == pytest.approx(highs[i])
            assert levels == sorted(levels)
    ct_turn = [j["level"] for j in jobs if j["case"] == "ct" and j["input"] == "turn_rate"]
    assert min(ct_turn) == 0.25
    reduced = RUNNER.build_oat_jobs(inputs=("delay",))
    assert {j["input"] for j in reduced} == {"delay"} and len(reduced) == 10


def test_morris_jobs_cover_both_classes_with_the_registered_design():
    jobs, design, orders = RUNNER.build_morris_jobs(20, seeds=[1, 2])
    assert design.shape == (20, 10, 9) and orders.shape == (20, 9)
    assert len(jobs) == 2 * 200
    reference, _ = morris_trajectories(9, 20, 4, 20261013)
    np.testing.assert_array_equal(design, reference)
    straight = [j for j in jobs if j["target_class"] == "straight"]
    turn = [j for j in jobs if j["target_class"] == "turn"]
    assert len(straight) == len(turn) == 200
    assert len({j["slice"] for j in jobs}) == len(jobs)
    # Same unit design in both classes: only the turn-rate physical value differs.
    t = RUNNER.INDEX["turn_rate"]
    for a, b in zip(straight, turn):
        va, vb = np.array(a["values"]), np.array(b["values"])
        assert np.allclose(np.delete(va, t), np.delete(vb, t))
        assert vb[t] >= 0.25
    assert all(j["seeds"] == [1, 2] for j in jobs)


def test_rank_is_each_inputs_rank_and_order_is_separate():
    mu_star = np.array([0.1, 5.0, 0.0, 3.0, 2.0])
    assert RUNNER.rank_from_mu_star(mu_star) == [4, 1, 5, 2, 3]
    # Ties go to the lower index and ranks stay a permutation.
    assert RUNNER.rank_from_mu_star([0.0, 0.0, 1.0]) == [2, 3, 1]


def test_estimate_full_run_does_not_divide_by_process_count_again():
    # 100 s measured for 94 jobs at 2 seeds; the full run is 490 jobs at 10 seeds.
    assert RUNNER.estimate_full_run(100.0, 94, 490, 2, 10) == pytest.approx(100.0 * 490 / 94 * 5)
    assert "processes" not in inspect.signature(RUNNER.estimate_full_run).parameters


# ---------------------------------------------------------------------------
# Outputs, using the synthetic response (never the engine)
# ---------------------------------------------------------------------------

def test_synthetic_outputs_are_labelled_synthetic_not_simulated(tmp_path):
    _synthetic_run(tmp_path / "out")
    for name in ("summary.json", "oat.json", "morris.json", "manifest.json"):
        doc = _read(tmp_path / "out" / name)
        assert doc.get("synthetic_test") is True, name
        assert "simulated" not in doc, name


def test_cli_cannot_inject_a_response_function(tmp_path):
    with pytest.raises(SystemExit):
        RUNNER.main(["--output", str(tmp_path / "x"), "--job-function", SYNTH])
    assert "job_function" not in inspect.getsource(RUNNER.main)
    assert not (tmp_path / "x").exists()


def test_cli_defaults_to_ten_seeds_and_passes_argv(tmp_path, monkeypatch):
    seen = {}

    def fake(output, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(RUNNER, "run_experiment", fake)
    RUNNER.main(["--output", str(tmp_path / "o"), "--no-checkpoint"])
    assert seen["seeds_count"] == 10 and seen["checkpoint_dir"] is None
    assert seen["argv"] == ["--output", str(tmp_path / "o"), "--no-checkpoint"]
    assert "job_function" not in seen


def test_summary_files_do_not_embed_per_seed_rows(tmp_path):
    _synthetic_run(tmp_path / "out")
    for name in ("summary.json", "oat.json"):
        doc = _read(tmp_path / "out" / name)
        assert doc["cells"] and all("per_seed" not in cell for cell in doc["cells"]), name
        assert "seed_index" not in (tmp_path / "out" / name).read_text(), name
    lines = gzip.decompress((tmp_path / "out" / "per_seed.csv.gz").read_bytes()).decode().splitlines()
    n_jobs = len(RUNNER.build_oat_jobs()) + 2 * 3 * 10
    assert len(lines) == 1 + n_jobs * 2 * 2  # jobs x seeds x arms
    assert lines[0] == ",".join(RUNNER.PER_SEED_COLUMNS)
    assert lines[1].split(",")[0].startswith("oat-cv-delay")


def test_unknown_per_seed_column_is_an_error(tmp_path):
    rows = [{"slice": "s", "result": 1, "per_seed": [{"cell_key": "k", "surprise": 1}]}]
    with pytest.raises(ValueError, match="surprise"):
        RUNNER._per_seed_rows(rows)


def test_morris_json_per_class_per_arm_with_rank_and_order(tmp_path):
    out = tmp_path / "out"
    _synthetic_run(out, morris_r=4)
    doc = _read(out / "morris.json")
    assert set(doc["classes"]) == {"straight", "turn"}
    assert doc["design"]["seed"] == 20261013 and doc["design"]["r"] == 4
    assert len(doc["design"]["unit_design"]) == 4
    for target_class, body in doc["classes"].items():
        assert set(body["effects"]) == {"recovery", "imm"}
        for arm, effects in body["effects"].items():
            mu_star = effects["mu_star"]
            assert sorted(effects["rank"]) == list(range(1, 10))
            # Independent definition: rank = 1 + number of inputs with a strictly larger mu*.
            for i, rank in enumerate(effects["rank"]):
                if mu_star.count(mu_star[i]) == 1:
                    assert rank == 1 + sum(m > mu_star[i] for m in mu_star)
            assert effects["order_names"][0] == "delay" and effects["order_names"][1] == "loss"
            assert effects["rank"][RUNNER.INDEX["delay"]] == 1
            assert effects["order"][0] == RUNNER.INDEX["delay"]
            # Linear synthetic response, imm arm 1.5x the recovery arm. Delay slope is
            # 100 per s over 0.3 s and loss slope 20 per unit over 0.5; a Morris step of
            # 2/3 of the range changes y by 2/3 of that, and mu* divides the step back out.
            scale = 1.5 if arm == "imm" else 1.0
            assert mu_star[RUNNER.INDEX["delay"]] == pytest.approx(30.0 * scale)
            assert mu_star[RUNNER.INDEX["loss"]] == pytest.approx(10.0 * scale)
    ranking = _read(out / "ranking.json")
    assert ranking["by_mu_star"]["straight"]["recovery"][:2] == ["delay", "loss"]
    assert ranking["oat_rank"]["cv"]["recovery"]["delay"] == 1


def test_synthetic_run_is_identical_with_one_or_two_processes(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _synthetic_run(one, processes=1)
    _synthetic_run(two, processes=2)
    for name in SYNTH_FILES:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name


def test_resume_from_half_the_checkpoints_is_byte_identical(tmp_path):
    cache = tmp_path / "cache"
    a, b = tmp_path / "a", tmp_path / "b"
    _synthetic_run(a, checkpoint_dir=cache, processes=2)
    files = sorted(cache.glob("*.json"))
    assert len(files) > 10
    for path in files[::2]:
        path.unlink()
    _synthetic_run(b, checkpoint_dir=cache, processes=2)
    assert len(list(cache.glob("*.json"))) == len(files)
    for name in SYNTH_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # The manifest hashes the data files, so it must agree as well.
    assert _read(a / "manifest.json")["outputs"] == _read(b / "manifest.json")["outputs"]


def test_data_file_columns_and_row_order_are_fixed(tmp_path):
    out = tmp_path / "out"
    _synthetic_run(out)
    lines = (out / "oat.csv").read_text().splitlines()
    assert lines[0] == ",".join(RUNNER.OAT_COLUMNS)
    arms = [line.split(",")[4] for line in lines[1:]]
    assert arms[:4] == ["recovery", "imm", "recovery", "imm"]
    levels = [float(line.split(",")[3]) for line in lines[1:] if line.startswith("oat-cv-delay")]
    assert levels[0::2] == sorted(levels[0::2])  # job order: levels ascend within a sweep


def test_manifest_replay_command_uses_the_real_argv(tmp_path, monkeypatch):
    out = tmp_path / "out"
    argv = ["--output", "runs/s6 sensitivity", "--seeds", "10"]
    _synthetic_run(out, argv=argv)
    manifest = _read(out / "manifest.json")
    assert manifest["argv"] == argv
    assert manifest["replay_command"] == ("PYTHONPATH=. ./.venv/bin/python tools/run_s6_sensitivity.py "
                                          + shlex.join(argv))
    assert "'runs/s6 sensitivity'" in manifest["replay_command"]
    # With argv=None the process arguments are used.
    monkeypatch.setattr(sys, "argv", ["run_s6_sensitivity.py", "--pilot", "--seeds", "3"])
    other = tmp_path / "other"
    RUNNER.run_experiment(other, job_function=SYNTH, seeds_count=2, processes=1, morris_r=2)
    assert _read(other / "manifest.json")["argv"] == ["--pilot", "--seeds", "3"]
    # n_runs counts cell-seed runs, not just jobs.
    assert manifest["n_runs"] == manifest["n_cells"] * 2


def test_non_empty_output_directory_is_refused(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "keep.txt").write_text("keep")
    with pytest.raises(ValueError):
        _synthetic_run(tmp_path / "x")
    assert (tmp_path / "x" / "keep.txt").read_text() == "keep"


# ---------------------------------------------------------------------------
# Checkpoint fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_covers_timesync_and_every_imported_estimation_module():
    repo = Path(RUNNER.__file__).resolve().parents[1]
    files = {p.resolve() for p in RUNNER._fingerprint_files()}
    assert repo / "sim" / "timesync.py" in files
    imported = [m for name, m in sys.modules.items()
                if name.startswith("sim.estimation") and getattr(m, "__file__", None)]
    assert imported
    for module in imported:
        assert Path(module.__file__).resolve() in files, module.__name__
    assert all(p.is_file() for p in files)


def test_fingerprint_works_from_any_directory_and_depends_on_seeds(tmp_path, monkeypatch):
    here = RUNNER._fingerprint([1, 2])
    monkeypatch.chdir(tmp_path)
    assert RUNNER._fingerprint([1, 2]) == here
    assert RUNNER._fingerprint([1, 3]) != here
    assert RUNNER._fingerprint([1, 2], "other:fn") != here


# ---------------------------------------------------------------------------
# Real engine (slow: about 25 s each with four processes)
# ---------------------------------------------------------------------------

def _real_files(out: Path) -> dict[str, bytes]:
    return {name: (out / name).read_bytes() for name in SYNTH_FILES}


def test_real_tiny_run_straight_turn_rate_null_effect_is_exactly_zero(tmp_path):
    out = tmp_path / "real"
    RUNNER.run_experiment(out, pilot=True, seeds_count=2, processes=4, morris_r=2,
                          oat_inputs=("delay", "turn_rate"), argv=["--pilot", "--seeds", "2"])
    morris = _read(out / "morris.json")
    assert morris["simulated"] is True and "synthetic_test" not in morris
    t = RUNNER.INDEX["turn_rate"]
    for arm in RUNNER.ARM_NAMES:
        straight = morris["classes"]["straight"]["effects"][arm]
        turn = morris["classes"]["turn"]["effects"][arm]
        assert straight["mu_star"][t] == 0.0 and straight["mu"][t] == 0.0 and straight["sigma"][t] == 0.0
        # The same input does something real on the turn class, so the zero is not an artefact.
        assert turn["mu_star"][t] > 0.0
        assert all(v > 0 for v in morris["classes"]["straight"]["responses"][arm][0])
    # Engine results are real: per-seed rows exist for every job, seed and arm.
    summary = _read(out / "summary.json")
    assert summary["simulated"] is True and "synthetic_test" not in summary
    n_jobs = len(summary["cells"])
    assert n_jobs == 2 * 2 * 3 + 2 * 2 * 10  # OAT: 2 cases x 2 inputs x 3 levels; Morris: 2 classes x r x 10
    lines = gzip.decompress((out / "per_seed.csv.gz").read_bytes()).decode().splitlines()
    assert len(lines) == 1 + n_jobs * 2 * 2
    assert lines[0] == ",".join(RUNNER.PER_SEED_COLUMNS)
    # Central straight OAT point matches a direct engine call on the same inputs.
    direct = RUNNER.job_cell({"values": RUNNER.CENTERS.tolist(), "seeds": seed_list(2),
                              "target_class": "straight", "slice": "direct"})
    oat = _read(out / "oat.json")["cells"]
    same = [c for c in oat if c["key"] == direct["key"]]
    assert same, "the OAT grid must contain the central case"
    assert same[0]["arms"]["recovery"]["p95_median_m"] == direct["arms"]["recovery"]["p95_median_m"]


def test_real_resume_from_half_the_checkpoints_is_byte_identical(tmp_path):
    cache = tmp_path / "cache"
    a, b = tmp_path / "a", tmp_path / "b"
    kwargs = dict(pilot=True, seeds_count=1, processes=4, morris_r=1, oat_inputs=("delay",),
                  argv=["--pilot"])
    RUNNER.run_experiment(a, checkpoint_dir=cache, **kwargs)
    files = sorted(cache.glob("*.json"))
    for path in files[::2]:
        path.unlink()
    RUNNER.run_experiment(b, checkpoint_dir=cache, **kwargs)
    one, two = _real_files(a), _real_files(b)
    assert one.keys() == two.keys()
    for name in one:
        assert one[name] == two[name], name
    shutil.rmtree(cache)
