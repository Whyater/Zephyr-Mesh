"""Checks for the EXP-038 runner (tools/retry_fades.py) and its hand calculation (tools/retry_fades_hand.py).

Where possible the expected values come from somewhere other than the tool under test:
the registered EXP-017 per-seed file, a Monte Carlo draw written in this file, an AR(1)
correlation formula, and plain arithmetic (1 - p**6, Jensen's inequality).  The synthetic
summaries in the verdict tests are built from the hand calculation, not from the simulator.
All radio parameters are literature-derived; nothing here is a measurement.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import functools
import itertools
import json
import math

import numpy as np
import pytest

from sim import s6_recovery
from sim.radio.channel import Channel, mean_rssi_dbm, packet_error_rate
from sim.tests.platform_compare import EXACT, RTOL, same_csv_row
from tools import c_exp017_tolerance as exp017
from tools import retry_fades as rf
from tools import retry_fades_hand as hand
from tools.c_common import REPO_ROOT

D = rf.DISTANCES_M
D630 = D[10]


@functools.cache
def hand_at(d: float) -> dict:
    return hand.hand_cell(d)


# ---- 1. the scenario_config patch ----------------------------------------------------

def test_patch_is_installed_and_original_is_kept():
    assert exp017.scenario_config is rf._scenario_config
    assert rf._ORIGINAL_SCENARIO_CONFIG is not rf._scenario_config
    assert rf._ORIGINAL_SCENARIO_CONFIG.__module__ == "tools.c_exp017_tolerance"


def test_iid_and_static_cells_get_the_original_config_unchanged():
    cells = rf.build_cells()
    checked = 0
    for cell in cells:
        if cell["fading"] not in ("iid", "static"):
            continue
        assert "corr_time_s" not in cell
        assert exp017.scenario_config(cell, 7) == rf._ORIGINAL_SCENARIO_CONFIG(cell, 7)
        checked += 1
    assert checked == 2 * 2 * len(D)  # 2 fadings x 2 modes x 16 distances


def test_tau_cells_change_only_the_correlation_time():
    taus = {"tau_0.1s": 0.1, "tau_0.5s": 0.5, "tau_2s": 2.0}
    checked = 0
    for cell in rf.build_cells():
        if cell["fading"] not in taus:
            continue
        tau = taus[cell["fading"]]
        assert cell["corr_time_s"] == tau
        orig = rf._ORIGINAL_SCENARIO_CONFIG(cell, 3)
        got = exp017.scenario_config(cell, 3)
        channel = got.radio.channel
        assert channel.shadowing_corr_time_s == tau
        assert channel.shadowing_mode == "time_correlated"
        # the test is only meaningful if the original really has a different correlation time
        assert orig.radio.channel.shadowing_corr_time_s not in (0.1, 0.5, 2.0)
        # everything else equals the original: compare as plain dicts with the one field put back
        want = dataclasses.asdict(orig)
        have = dataclasses.asdict(got)
        have["radio"]["channel"]["shadowing_corr_time_s"] = orig.radio.channel.shadowing_corr_time_s
        assert have == want
        checked += 1
    assert checked == 3 * 2 * len(D)


def test_cell_list_shape():
    cells = rf.build_cells()
    assert len(cells) == 5 * 2 * len(D)
    assert [c["cell"] for c in cells] == list(range(len(cells)))
    assert {c["fading"] for c in cells} == {"iid", "static", "tau_0.1s", "tau_0.5s", "tau_2s"}
    assert all(c["processing_s"] == 0.0 and c["scenario"] == "A" and c["area"] == "field" for c in cells)


# ---- 2. the sanity gate on real data ---------------------------------------------------

@pytest.fixture
def platform_gate():
    """The EXP-017 rows were saved on macOS, where the gate stays text for text.

    Other platforms differ in the last digit or two, so there the gate uses the shared float tolerance
    when it compares with the registered run.
    """
    with pytest.MonkeyPatch.context() as patch:
        if not EXACT:
            patch.setattr(rf, "_same_text", same_csv_row)
        yield


@pytest.fixture(scope="module")
def gate_inputs():
    """Two real cells at 630 m (iid unicast, static broadcast), two seeds each."""
    all_cells = rf.build_cells()
    cells = [next(c for c in all_cells if c["fading"] == "iid" and c["mode"] == "unicast" and c["distance_m"] == D630),
             next(c for c in all_cells if c["fading"] == "static" and c["mode"] == "broadcast"
                  and c["distance_m"] == D630)]
    seeds = s6_recovery.seed_list(2)
    rows = [r for c in cells for s in seeds for r in rf.run_job((c, s))]
    return cells, rows


def test_gate_matches_registered_rows_text_for_text(gate_inputs, platform_gate):
    cells, rows = gate_inputs
    assert len(rows) == 2 * 2 * 2 * 2  # 2 cells x 2 seeds x 2 motions x 2 arms
    gate = rf.sanity_gate(cells, rows, REPO_ROOT / rf.REGISTERED_DIR)
    assert gate["rows_compared"] == 16
    assert gate["rows_mismatched"] == 0
    assert gate["first_mismatch"] is None
    # Hand count of the registered rows left uncompared: 2 modes x 2 shadowing x 16 distances
    # x 50 seeds x 2 motions x 2 arms = 12800, of which 16 were compared.
    assert gate["registered_rows_not_compared"] == 12800 - 16
    assert gate["passed"] is False  # most registered rows were not rerun


def test_gate_reports_a_one_ulp_scale_float_change(gate_inputs, platform_gate):
    cells, rows = gate_inputs
    tampered = copy.deepcopy(rows)
    target = tampered[5]
    assert math.isfinite(target["p95_m"])
    if EXACT:
        target["p95_m"] += 1e-15  # about 70 ulp at this size: changes the printed text
    else:
        target["p95_m"] *= 1 + 100 * RTOL  # well past the cross-platform tolerance
    assert target["p95_m"] != rows[5]["p95_m"]
    gate = rf.sanity_gate(cells, tampered, REPO_ROOT / rf.REGISTERED_DIR)
    assert gate["rows_compared"] == 16
    assert gate["rows_mismatched"] == 1
    assert gate["first_mismatch"] == {"cell": target["cell"], "seed": target["seed"], "motion": target["motion"],
                                      "arm": target["arm"]}
    assert gate["passed"] is False


def test_gate_counts_a_deleted_row_as_not_compared(gate_inputs, platform_gate):
    cells, rows = gate_inputs
    base = rf.sanity_gate(cells, rows, REPO_ROOT / rf.REGISTERED_DIR)
    gate = rf.sanity_gate(cells, rows[:3] + rows[4:], REPO_ROOT / rf.REGISTERED_DIR)
    assert gate["rows_compared"] == 15
    assert gate["rows_mismatched"] == 0
    assert gate["registered_rows_not_compared"] == base["registered_rows_not_compared"] + 1


def test_gate_passes_only_when_every_registered_row_is_rerun(gate_inputs, tmp_path):
    # A small stand-in "registered run" built from the rows themselves, with different cell
    # numbers, so the gate's cell lookup (mode, shadowing, distance) is exercised.
    cells, rows = gate_inputs
    reg_id = {cells[0]["cell"]: 100, cells[1]["cell"]: 101}
    reg_cells = [{"cell": reg_id[c["cell"]], "scenario": "A", "area": "field", "processing_s": "0.0",
                  "mode": c["mode"], "shadowing": c["shadowing"], "distance_m": repr(c["distance_m"])}
                 for c in cells]
    with open(tmp_path / "cells.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reg_cells[0]))
        writer.writeheader()
        writer.writerows(reg_cells)

    def write_registered(rows_out):
        with open(tmp_path / "per_seed.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for r in rows_out:
                writer.writerow({**r, "cell": reg_id[r["cell"]]})

    write_registered(rows)
    gate = rf.sanity_gate(cells, rows, tmp_path)
    assert (gate["rows_compared"], gate["rows_mismatched"], gate["registered_rows_not_compared"]) == (16, 0, 0)
    assert gate["passed"] is True
    # one registered row differs: the gate must fail
    changed = copy.deepcopy(rows)
    changed[0]["rmse_m"] += 1e-12
    write_registered(changed)
    gate = rf.sanity_gate(cells, rows, tmp_path)
    assert gate["rows_mismatched"] == 1 and gate["passed"] is False


# ---- 3. the hand calculation -------------------------------------------------------------

def test_hand_inputs_still_match_the_presets():
    hand.check_inputs()  # asserts against sim.radio.params.PARAMS


@pytest.mark.parametrize("d", D)
def test_hand_delivery_identities(d):
    h = hand_at(d)
    p = h["p_attempt_loss"]
    assert 0.0 <= p <= 1.0
    assert h["unicast_iid_delivery"] == 1.0 - p ** 6
    assert h["broadcast_delivery"] == 1.0 - p
    assert h["crude_frozen_limit"] == 1.0 - p


@pytest.mark.parametrize("d", D)
def test_frozen_fade_delivery_lies_between_the_two_bounds(d):
    # Jensen: PER^6 <= PER, so E[1 - PER^6] >= 1 - p; and E[PER^6] >= (E PER)^6, so E[1 - PER^6] <= 1 - p^6.
    h = hand_at(d)
    p = h["p_attempt_loss"]
    assert 1.0 - p - 1e-12 <= h["unicast_frozen_delivery"] <= 1.0 - p ** 6 + 1e-12


def test_frozen_fade_delivery_is_strictly_inside_where_the_curve_is_not_trivial():
    for d in D[6:13]:  # 435 m to 758 m: p well away from 0 and 1
        h = hand_at(d)
        assert 1.0 - h["p_attempt_loss"] + 1e-3 < h["unicast_frozen_delivery"] < h["unicast_iid_delivery"] - 1e-3


def test_hand_loss_grows_with_distance():
    ps = [hand_at(d)["p_attempt_loss"] for d in D]
    assert all(b > a for a, b in itertools.pairwise(ps))


@pytest.mark.parametrize("d", [1.0, 30.0, hand.BREAKPOINT_M - 1e-6, 250.0, 524.0, D630, 1000.0])
def test_hand_mean_rssi_matches_the_channel_module(d):
    cfg = exp017.stress_corner("iid")
    assert math.isclose(cfg.breakpoint_m, hand.BREAKPOINT_M, rel_tol=1e-12)
    assert abs(hand.mean_rssi_dbm(d) - float(mean_rssi_dbm(d, cfg))) < 1e-9


def test_hand_per_curve_matches_the_channel_module():
    cfg = exp017.stress_corner("iid")
    snr = np.linspace(-20.0, 40.0, 241)
    ours = hand.per_of_snr(snr)
    theirs = np.asarray(packet_error_rate(snr, hand.FRAME_BYTES, cfg), dtype=float)
    assert np.max(np.abs(ours - theirs)) < 1e-12
    # sanity of the curve itself: 8 % at the datasheet sensitivity for 1024 bytes
    snr_ref = hand.SENSITIVITY_DBM - hand.NOISE_FLOOR_DBM
    assert abs(float(packet_error_rate(snr_ref, 1024, cfg)) - 0.08) < 1e-9


def test_hand_cell_matches_a_monte_carlo_draw_at_630_m():
    cfg = exp017.stress_corner("iid")
    assert cfg.noise_floor_dbm == hand.NOISE_FLOOR_DBM
    n = 200_000
    z = np.random.default_rng(20261021).standard_normal(n)
    rssi = float(mean_rssi_dbm(D630, cfg)) - cfg.shadowing_sigma_db * z
    per = np.asarray(packet_error_rate(rssi - cfg.noise_floor_dbm, hand.FRAME_BYTES, cfg), dtype=float)
    h = hand_at(D630)
    for sim_values, hand_value in ((per, h["p_attempt_loss"]),
                                   (1.0 - per, h["broadcast_delivery"]),
                                   (1.0 - per ** 6, h["unicast_frozen_delivery"])):
        se = float(np.std(sim_values, ddof=1)) / math.sqrt(n)
        assert abs(float(np.mean(sim_values)) - hand_value) < 4 * se
    # and the iid unicast figure is 1 - p^6 of the MC estimate of p, within its propagated error
    p_mc, se_p = float(np.mean(per)), float(np.std(per, ddof=1)) / math.sqrt(n)
    assert abs((1.0 - p_mc ** 6) - h["unicast_iid_delivery"]) < 4 * 6 * p_mc ** 5 * se_p + 1e-12


# ---- 4. the channel behaves as the experiment assumes --------------------------------------

def _shadow_series(mode: str, *, tau: float, steps: int, dt: float = 0.01, seed: int = 11) -> np.ndarray:
    cfg = dataclasses.replace(exp017.stress_corner(mode), shadowing_corr_time_s=tau)
    channel = Channel(cfg, np.random.default_rng(seed))
    mean_rssi = float(mean_rssi_dbm(300.0, cfg))
    # shadow = mean RSSI - received power; the pair is sampled at 0, dt, 2 dt, ...
    return np.array([mean_rssi - channel.received_power_dbm("a", "b", 300.0, i * dt) for i in range(steps)])


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def test_time_correlated_shadow_follows_exp_minus_dt_over_tau():
    tau, dt = 0.5, 0.01
    x = _shadow_series("time_correlated", tau=tau, steps=300_000, dt=dt)
    assert abs(_autocorr(x, 1) - math.exp(-dt / tau)) < 0.02
    assert abs(_autocorr(x, 50) - math.exp(-50 * dt / tau)) < 0.05  # one correlation time: 1/e
    # stationary spread stays at sigma = 4 dB (effective sample size is a few thousand here)
    assert abs(float(np.std(x)) - 4.0) < 0.4


def test_iid_shadow_has_no_lag_one_correlation():
    x = _shadow_series("iid", tau=0.5, steps=100_000)
    assert abs(_autocorr(x, 1)) < 0.02
    assert abs(float(np.std(x)) - 4.0) < 0.1


def test_static_shadow_never_changes():
    x = _shadow_series("static", tau=0.5, steps=500)
    assert np.all(x == x[0])


# ---- 5. evaluate() on synthetic summaries ----------------------------------------------------

def make_summary(*, unicast: dict, turn_fails_from: dict) -> list[dict]:
    """A readable fake of ``summarize()`` output.

    ``unicast[fading]`` is "frozen" (hand E[1 - PER^6]), "iid" (hand 1 - p^6) or "between" (halfway).
    Broadcast delivery is the hand 1 - p for every fading.  ``turn_fails_from[fading]`` is the first
    distance (m) at which the recovery arm's 1 m turn fails in at least half the seeds, or None.
    """
    cells = []
    for fading, _, tau in rf.FADINGS:
        for mode in rf.MODES:
            for d in D:
                h = hand_at(d)
                if mode == "broadcast":
                    delivery = h["broadcast_delivery"]
                else:
                    delivery = {"frozen": h["unicast_frozen_delivery"], "iid": h["unicast_iid_delivery"],
                                "between": 0.5 * (h["unicast_frozen_delivery"] + h["unicast_iid_delivery"])}[
                        unicast[fading]]
                start = turn_fails_from[fading]
                failed = start is not None and d >= start
                cells.append({"fading": fading, "corr_time_s": tau, "mode": mode, "distance_m": d,
                              "delivery_pooled": delivery, "delivery_seed_mean": delivery,
                              "delivery_seed_low": delivery - 0.01, "delivery_seed_high": delivery + 0.01,
                              "hand_frozen_unicast": h["unicast_frozen_delivery"],
                              "recovery_constant_turn_cell_failed": failed,
                              "recovery_constant_turn_fail_fraction": 0.9 if failed else 0.1})
    return cells


SUPPORTED_UNICAST = {"iid": "iid", "static": "frozen", "tau_0.1s": "between", "tau_0.5s": "frozen", "tau_2s": "frozen"}
SUPPORTED_TURN = {"iid": D[13], "static": D[9], "tau_0.1s": D[11], "tau_0.5s": D[9], "tau_2s": D[9]}  # 831 / 574 m


def holds(result: dict) -> dict:
    return {k: v["holds"] for k, v in result["predictions"].items()}


def test_evaluate_supported_when_slow_fades_behave_like_the_hand_frozen_values():
    result = rf.evaluate(make_summary(unicast=SUPPORTED_UNICAST, turn_fails_from=SUPPORTED_TURN))
    assert holds(result) == {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True}
    assert result["verdict"] == "SUPPORTED"
    p3 = result["predictions"]["P3"]["by_fading"]
    assert p3["tau_0.5s"]["first_failing_m"] == D[9] and p3["tau_2s"]["first_failing_m"] == D[9]
    assert p3["tau_0.5s"]["last_passing_below_m"] == D[8]
    # retry gains: hand numbers at 524 / 574 / 630 m, tau 2 s gain is a quarter or less of the iid gain
    for row in result["predictions"]["P2"]["by_distance"]:
        h = hand_at(row["distance_m"])
        assert row["gain_iid"] == pytest.approx(h["unicast_iid_delivery"] - h["broadcast_delivery"])
        assert row["gain_tau_2s"] == pytest.approx(h["unicast_frozen_delivery"] - h["broadcast_delivery"])
        assert row["gain_tau_2s"] <= row["gain_iid"] / 3.0
    assert result["predictions"]["P5"]["checked"] == len(D) * 4


def test_evaluate_contradicted_when_slow_fades_behave_like_iid():
    unicast = {**SUPPORTED_UNICAST, "tau_0.1s": "iid", "tau_0.5s": "iid", "tau_2s": "iid"}
    turn = {**SUPPORTED_TURN, "tau_0.1s": D[13], "tau_0.5s": D[13], "tau_2s": D[13]}
    result = rf.evaluate(make_summary(unicast=unicast, turn_fails_from=turn))
    flags = holds(result)
    assert flags["P1"] is False and flags["P2"] is False and flags["P3"] is False and flags["P4"] is False
    assert flags["P5"] is True  # broadcast is the same in every fading
    assert result["verdict"] == "CONTRADICTED"


def test_evaluate_contradicted_when_the_turn_never_fails_at_slow_fades():
    turn = {**SUPPORTED_TURN, "tau_0.5s": None, "tau_2s": None}
    result = rf.evaluate(make_summary(unicast=SUPPORTED_UNICAST, turn_fails_from=turn))
    flags = holds(result)
    assert flags["P1"] and flags["P2"]  # delivery looks like the prediction
    assert flags["P3"] is False
    p3 = result["predictions"]["P3"]["by_fading"]
    assert p3["tau_0.5s"]["first_failing_m"] is None and p3["tau_2s"]["first_failing_m"] is None
    assert p3["tau_2s"]["last_passing_below_m"] == D[-1]
    assert result["verdict"] == "CONTRADICTED"


def test_evaluate_contradicted_boundary_is_758_m():
    # first failing exactly at 758 m for both: CONTRADICTED (>= 758); 691 m for one of them: MIXED
    base = {**SUPPORTED_TURN}
    both = rf.evaluate(make_summary(unicast=SUPPORTED_UNICAST,
                                    turn_fails_from={**base, "tau_0.5s": D[12], "tau_2s": D[12]}))
    assert both["verdict"] == "CONTRADICTED"
    one = rf.evaluate(make_summary(unicast=SUPPORTED_UNICAST,
                                   turn_fails_from={**base, "tau_0.5s": D[12], "tau_2s": D[11]}))
    assert one["verdict"] == "MIXED"


def test_evaluate_mixed_when_p2_fails_but_the_turn_still_fails_early():
    unicast = {**SUPPORTED_UNICAST, "tau_2s": "iid"}  # no suppression of the retry gain at 2 s
    result = rf.evaluate(make_summary(unicast=unicast, turn_fails_from=SUPPORTED_TURN))
    assert holds(result)["P2"] is False and holds(result)["P3"] is True
    assert result["verdict"] == "MIXED"


def test_evaluate_p5_flags_broadcast_that_depends_on_fading_without_changing_the_verdict():
    summary = make_summary(unicast=SUPPORTED_UNICAST, turn_fails_from=SUPPORTED_TURN)
    for c in summary:
        if c["fading"] == "static" and c["mode"] == "broadcast":
            for key in ("delivery_seed_mean", "delivery_seed_low", "delivery_seed_high"):
                c[key] -= 0.2  # shift the whole seed-level interval
    result = rf.evaluate(summary)
    p5 = result["predictions"]["P5"]
    assert p5["holds"] is False
    assert {x["fading"] for x in p5["failures"]} == {"static"}
    assert len(p5["failures"]) == len(D)
    assert result["verdict"] == "SUPPORTED"


# ---- 6. pilot determinism -----------------------------------------------------------------------

def test_pilot_run_is_byte_identical_twice(tmp_path):
    outputs = []
    for name in ("a", "b"):
        out = tmp_path / name
        rf.run(out, [], processes=1, distances=(D[8],), n_seeds=2)
        outputs.append(out)
    for fname in ("cells.csv", "per_seed.csv", "summary.json"):
        assert (outputs[0] / fname).read_bytes() == (outputs[1] / fname).read_bytes(), fname
    summary = json.loads((outputs[0] / "summary.json").read_text())
    assert summary["full_run"] is False and summary["evaluation"] is None
    assert summary["sanity_gate"] == {"passed": None, "note": "pilot seeds"}
    assert len(summary["cells"]) == 10  # 5 fadings x 2 modes at one distance
    with open(outputs[0] / "per_seed.csv", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 10 * 2 * 2 * 2  # cells x seeds x motions x arms
    assert {r["distance_m"] for r in table} == {repr(D[8])}


def test_full_run_with_a_failed_gate_is_void_and_exits(tmp_path, monkeypatch):
    """A failed sanity gate on a full run marks summary.json VOID, withholds the evaluation and exits nonzero."""
    def fake_job(job):
        cell, seed = job
        return [{"cell": cell["cell"], "seed": int(seed), "motion": m, "arm": a, "failed": False, "p95_m": 0.1,
                 "max_m": 0.2, "rmse_m": 0.05, "fused": 120, "dropped": 0, "sent": 121, "received": 121,
                 "received_in_run": 120, "background_senders": 0, "latency_median_s": 0.001,
                 "latency_p95_s": 0.001, "duplicates_suppressed": 0}
                for m in rf.s6_recovery.MOTIONS for a in rf.s6_recovery.ARMS]

    monkeypatch.setattr(rf, "run_job", fake_job)
    monkeypatch.setattr(rf, "sanity_gate", lambda *a, **k: {"passed": False, "rows_mismatched": 1,
                                                             "registered_rows_not_compared": 0})
    out = tmp_path / "void"
    with pytest.raises(SystemExit):
        rf.run(out, [], processes=1)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"].startswith("VOID")
    assert summary["evaluation"] is None and summary["headline_first_failing_distance"] is None
    assert not (out / "exp038_retry_fades.png").exists() and not (out / "manifest.json").exists()
