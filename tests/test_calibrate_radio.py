"""Tests for tools/calibrate_radio_from_trace.py (EXP-018).

Everything fitted here comes from synthetic traces with known parameters.
Nothing in this file is a radio measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.radio.channel import GilbertElliott
from sim.s6_recovery import wilson_interval
from sim.trace import EspNowTrace
from tools.calibrate_radio_from_trace import (
    PositionInput,
    analyze_trace,
    build_trace_dict,
    calibrate,
    calibrate_manifest,
    fit_gilbert,
    fit_path_loss,
    main,
    packets_for_wilson_halfwidth,
    sizing_table,
    synthetic_manifest,
    synthetic_trace_dicts,
)

DISTANCES = [1, 5, 10, 20, 40]
GE = GilbertElliott(0.02, 0.2)


def test_hand_case_four_exact_points():
    d = [1.0, 2.0, 10.0, 100.0]
    y = [-40.0 - 25.0 * math.log10(v) for v in d]  # n = 2.5, A = -40
    fit = fit_path_loss(d, y)
    assert fit["identifiable"]
    assert fit["path_loss_exponent"] == pytest.approx(2.5, abs=1e-9)
    assert fit["a_dbm_at_1m"] == pytest.approx(-40.0, abs=1e-9)
    assert fit["shadowing_sigma_db"] == pytest.approx(0.0, abs=1e-9)
    assert fit["df"] == 2
    lo, hi = fit["path_loss_exponent_ci"]
    assert lo == pytest.approx(2.5, abs=1e-8) and hi == pytest.approx(2.5, abs=1e-8)


def test_hand_case_noisy_three_points_matches_closed_form():
    # k = 3: x = 0, 10, 20 dB (d = 1, 10, 100), y = -40, -66, -80.  Slope by hand:
    # mean x = 10, mean y = -62; Sxy = (-10)(22) + 0 + (10)(-18) = -400; Sxx = 200; slope = -2 -> n = 2.
    fit = fit_path_loss([1, 10, 100], [-40.0, -66.0, -80.0])
    assert fit["path_loss_exponent"] == pytest.approx(2.0, abs=1e-12)
    # intercept = mean y - slope * mean x = -62 + 2 * 10 = -42
    assert fit["a_dbm_at_1m"] == pytest.approx(-42.0, abs=1e-12)
    # residuals -40-(-42)=2, -66-(-62)=-4, -80-(-82)=2; SSR = 24, df = 1, sigma = sqrt(24)
    assert fit["shadowing_sigma_db"] == pytest.approx(math.sqrt(24.0), abs=1e-12)
    assert sum(fit["residuals_db"]) == pytest.approx(0.0, abs=1e-12)


def _recovery_run(n_datasets: int):
    n_cover = r_cover = 0
    errors = []
    for i in range(n_datasets):
        docs = synthetic_trace_dicts(n=2.0, a_dbm=-21.0, sigma_db=3.0, distances_m=DISTANCES, packets_per_distance=1000,
                                     ge=GE, fast_sigma_db=1.0, seed=1000 + i)
        positions = [PositionInput(pid, EspNowTrace.from_dict(doc), dist) for pid, dist, doc in docs]
        result = calibrate(positions)
        pl = result["path_loss"]
        lo, hi = pl["path_loss_exponent_ci"]
        n_cover += lo <= 2.0 <= hi
        errors.append(abs(pl["path_loss_exponent"] - 2.0))
        lo, hi = result["gilbert_pooled"]["p_bad_to_good_ci"]
        r_cover += lo <= 0.2 <= hi
    return n_cover / n_datasets, r_cover / n_datasets, float(np.median(errors))


def test_parameter_recovery_exp018_synthetic():
    """EXP-018 recovery check on synthetic traces: 95 % intervals cover the true values about 95 % of the time.

    Observed with these seeds (1000..1199): n covered 93.0 %, r covered 97.5 %, median |n_hat - 2| = 0.156.
    A separate run of 800 datasets (seeds 5000..5799) gave 95.6 % and 94.3 %.
    """
    n_cov, r_cov, median_err = _recovery_run(200)
    print(f"EXP-018 synthetic recovery: n coverage {n_cov:.3f}, r coverage {r_cov:.3f}, median |n_hat - 2| {median_err:.3f}")
    assert 0.92 <= n_cov <= 0.98
    assert r_cov >= 0.90
    assert median_err < 0.3  # shadowing-limited: roughly sigma / sqrt(Sxx) with 5 positions


def _outcome_trace(outcomes, rssi_value=-60.0, session="s0", trace_id="t", seq_start=0):
    rssi = [rssi_value if o == "received" else None for o in outcomes]
    return EspNowTrace.from_dict(build_trace_dict(trace_id, outcomes, rssi, session_id=session, seq_start=seq_start))


def test_unknown_outcomes_excluded_from_denominator_and_reported():
    outcomes = ["received"] * 6 + ["lost"] * 2 + ["unknown"] * 4 + ["received"] * 2
    a = analyze_trace(_outcome_trace(outcomes))
    dl = a["delivery"]
    assert (dl["received"], dl["lost"], dl["unknown"]) == (8, 2, 4)
    assert dl["denominator"] == 10 and dl["n_packets_total"] == 14
    assert dl["delivery_ratio"] == pytest.approx(0.8)
    assert dl["delivery_ci"] == pytest.approx(list(wilson_interval(8, 10)))
    # the unknown packets break the sequence: no transition is counted across them.
    # [6 rx, 2 lost] gives 5 rx->rx, 1 rx->lost, 1 lost->lost; [2 rx] gives 1 more rx->rx.
    g = a["gilbert"]
    assert g["transitions"] == {"received_to_received": 6, "received_to_lost": 1, "lost_to_received": 0, "lost_to_lost": 1}
    assert g["sequence_breaks"]["unknown_breaks"] == 4
    assert g["n_sequences"] == 2  # [6 rx, 2 lost] and [2 rx]
    assert g["n_packets"] == 10


def test_all_unknown_trace_does_not_crash():
    a = analyze_trace(_outcome_trace(["unknown"] * 5))
    assert a["delivery"]["denominator"] == 0 and a["delivery"]["delivery_ratio"] is None
    assert a["rssi"]["mean_rssi_dbm"] is None
    assert a["gilbert"]["p_good_to_bad"] is None and a["gilbert"]["p_bad_to_good"] is None


def test_gilbert_hand_counts_and_seq_gap_break():
    # received=0, lost=1.  seq 0..7 then a gap (seq 8 missing) then 9..11.
    outcomes = ["received", "lost", "lost", "received", "received", "lost", "received", "received"]
    doc = build_trace_dict("g", outcomes, [-50.0 if o == "received" else None for o in outcomes])
    tail = build_trace_dict("g2", ["lost", "received", "received"], [None, -50.0, -50.0], seq_start=9)
    for pkt in tail["packets"]:
        pkt["packet_id"] = "tail-" + pkt["packet_id"]
    doc["packets"].extend(tail["packets"])
    g = analyze_trace(EspNowTrace.from_dict(doc))["gilbert"]
    assert g["sequence_breaks"]["seq_gap_breaks"] == 1
    # first sequence 0 1 1 0 0 1 0 0: rr=2 (0->0 twice: idx3-4, 6-7), rl=2 (0->1: idx0-1, 4-5), lr=2 (1->0: idx2-3, 5-6), ll=1
    # second sequence 1 0 0: lr=1, rr=1
    t = g["transitions"]
    assert t == {"received_to_received": 3, "received_to_lost": 2, "lost_to_received": 3, "lost_to_lost": 1}
    assert g["p_good_to_bad"] == pytest.approx(2 / 5)
    assert g["p_bad_to_good"] == pytest.approx(3 / 4)
    assert g["p_good_to_bad_ci"] == pytest.approx(list(wilson_interval(2, 5)))


def test_fit_gilbert_recovers_known_chain_and_flags_burstiness():
    rng = np.random.default_rng(7)
    _, lost = GilbertElliott(0.02, 0.2).sample(100_000, rng)
    # One 95 % interval misses the truth 1 time in 20 by design, so a single-trace
    # check uses a 99.99 % interval; 95 % coverage is measured across many traces
    # in the EXP-018 parameter-recovery run.
    g = fit_gilbert([lost.astype(int).tolist()], confidence=0.9999)
    assert g["p_good_to_bad_ci"][0] <= 0.02 <= g["p_good_to_bad_ci"][1]
    assert g["p_bad_to_good_ci"][0] <= 0.2 <= g["p_bad_to_good_ci"][1]
    b = g["burstiness"]
    assert b["ratio_observed_to_independent"] > 3.0  # mean burst 5 vs about 1.1 if independent
    # independent losses at the same rate: ratio near 1
    indep = (rng.random(100_000) < 0.09).astype(int).tolist()
    bi = fit_gilbert([indep])["burstiness"]
    assert bi["ratio_observed_to_independent"] == pytest.approx(1.0, abs=0.1)
    assert abs(bi["approx_z"]) < 4


def test_fewer_than_three_positions_not_identifiable():
    fit = fit_path_loss([1.0, 10.0], [-40.0, -60.0])
    assert fit["identifiable"] is False
    assert fit["path_loss_exponent"] is None and fit["shadowing_sigma_db"] is None
    assert "not identifiable" in fit["reason"]
    # three positions but only two distinct distances: also not identifiable
    assert fit_path_loss([1.0, 10.0, 10.0], [-40.0, -60.0, -61.0])["identifiable"] is False
    # through the whole pipeline, with JSON-safe output
    docs = synthetic_trace_dicts(n=2.0, a_dbm=-21.0, sigma_db=3.0, distances_m=[1, 10], packets_per_distance=200,
                                 ge=GE, fast_sigma_db=1.0, seed=3)
    result = calibrate([PositionInput(p, EspNowTrace.from_dict(doc), d) for p, d, doc in docs])
    assert result["path_loss"]["identifiable"] is False
    json.dumps(result, allow_nan=False)


def test_wall_and_body_entries_are_listed_but_not_fitted():
    docs = synthetic_trace_dicts(n=2.0, a_dbm=-21.0, sigma_db=0.0, distances_m=[1, 5, 10, 20], packets_per_distance=300,
                                 ge=GilbertElliott(0.0, 1.0), fast_sigma_db=0.0, seed=4)
    positions = [PositionInput(p, EspNowTrace.from_dict(doc), d, walls=(1 if d == 20 else 0)) for p, d, doc in docs]
    result = calibrate(positions)
    pl = result["path_loss"]
    assert pl["positions_used"] == ["d0_1m", "d1_5m", "d2_10m"]
    assert [e["position_id"] for e in pl["positions_excluded"]] == ["d3_20m"]
    assert pl["path_loss_exponent"] == pytest.approx(2.0, abs=1e-9)  # exact data, wall entry did not bias it


def test_synthetic_generator_is_deterministic_and_labelled():
    kw = dict(n=2.0, a_dbm=-21.0, sigma_db=3.0, distances_m=[1, 10], packets_per_distance=50, ge=GE, fast_sigma_db=1.0, seed=11)
    a = synthetic_trace_dicts(**kw)
    b = synthetic_trace_dicts(**kw)
    assert a == b
    assert synthetic_trace_dicts(**{**kw, "seed": 12}) != a
    for _, _, doc in a:
        trace = EspNowTrace.from_dict(doc)  # passes the real schema validator
        assert trace.status == "synthetic" and trace.clock_domain == "shared_monotonic"
        assert trace.capture["firmware"] == "simulated" and trace.capture["phy"] == "simulated"
        for pkt in trace.packets:
            assert (pkt.rssi_dbm is None) == (pkt.outcome == "lost")
            if pkt.outcome == "lost":
                assert pkt.loss_reason == "synthetic_burst"


def test_cli_round_trip_and_refuses_to_overwrite_manifest(tmp_path, capsys):
    manifest = synthetic_manifest(tmp_path / "data", n=2.0, a_dbm=-21.0, sigma_db=3.0, distances_m=DISTANCES,
                                  packets_per_distance=400, ge=GE, fast_sigma_db=1.0, seed=21)
    out = tmp_path / "out" / "fit.json"
    assert main([str(manifest), "--output", str(out), "--sizing"]) == 0
    err = capsys.readouterr().err
    assert "Packets needed" in err
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["status"] == "synthetic"
    assert result["path_loss"]["identifiable"] and result["path_loss"]["k_positions"] == 5
    assert result["n_positions"] == 5 and len(result["entries"]) == 5
    assert len(result["sizing_table"]) == 8
    prov = result["provenance"]
    assert prov["tool"] == "calibrate_radio_from_trace"
    assert prov["status"] == "fitted from the traces listed; status of each trace copied"
    assert len(prov["manifest_sha256"]) == 64
    assert [t["status"] for t in prov["trace_sha256"]] == ["synthetic"] * 5
    assert all(len(t["sha256"]) == 64 for t in prov["trace_sha256"])
    # sorted keys, allow_nan False, deterministic
    assert out.read_text(encoding="utf-8") == json.dumps(json.loads(out.read_text(encoding="utf-8")), indent=2, sort_keys=True,
                                                         allow_nan=False) + "\n"
    out2 = tmp_path / "fit2.json"
    assert main([str(manifest), "--output", str(out2)]) == 0
    assert json.loads(out2.read_text(encoding="utf-8"))["path_loss"] == result["path_loss"]
    # stdout mode parses
    assert main([str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["provenance"]["tool"] == "calibrate_radio_from_trace"
    # refuses to overwrite the manifest and a trace file
    before = manifest.read_bytes()
    assert main([str(manifest), "--output", str(manifest)]) == 2
    assert manifest.read_bytes() == before
    trace_file = manifest.parent / "d0_1m.json"
    assert main([str(manifest), "--output", str(trace_file)]) == 2


def test_cli_rejects_invalid_trace_and_bad_manifest(tmp_path, capsys):
    bad_trace = tmp_path / "bad.json"
    bad_trace.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"entries": [{"trace": "bad.json", "distance_m": 5}]}), encoding="utf-8")
    assert main([str(manifest)]) == 2
    assert "schema must be" in capsys.readouterr().err
    manifest.write_text(json.dumps({"entries": [{"trace": "bad.json", "distance_m": -1}]}), encoding="utf-8")
    assert main([str(manifest)]) == 2
    manifest.write_text(json.dumps({"entries": []}), encoding="utf-8")
    assert main([str(manifest)]) == 2


def test_manifest_defaults_position_id_from_path_and_walls_bodies(tmp_path):
    doc = build_trace_dict("only", ["received"] * 5, [-60.0] * 5)
    (tmp_path / "spot_a.json").write_text(json.dumps(doc), encoding="utf-8")
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"entries": [{"trace": "spot_a.json", "distance_m": 3.0}]}), encoding="utf-8")
    result = calibrate_manifest(m)
    entry = result["entries"][0]
    assert entry["position_id"] == "spot_a" and entry["walls"] == 0 and entry["bodies"] == 0
    assert result["path_loss"]["identifiable"] is False


def test_status_never_upgraded_to_measured():
    t_syn = _outcome_trace(["received"] * 5, trace_id="a")
    measured_doc = build_trace_dict("b", ["received"] * 5, [-60.0] * 5, status="measured")
    measured_doc["provenance"] = {
        "instrument": {"kind": "esp32", "model": "xiao-s3", "settings": {}}, "calibration_ref": None,
        "calibration_ref_null_reason": "uncalibrated", "raw_sha256": "0" * 64, "code_revision": "x",
        "captured_at": "2026-10-14T00:00:00Z", "operator": "test", "ground_truth": None,
        "ground_truth_null_reason": "none",
    }
    t_meas = EspNowTrace.from_dict(measured_doc)
    assert calibrate([PositionInput("a", t_syn, 1.0), PositionInput("b", t_meas, 2.0)])["status"] == "synthetic"
    assert calibrate([PositionInput("b", t_meas, 2.0)])["status"] == "measured"


def test_wilson_sizing_matches_direct_wilson_call():
    n = packets_for_wilson_halfwidth(0.95, 0.01)
    print(f"packets_for_wilson_halfwidth(0.95, 0.01) = {n}")
    assert 1750 <= n <= 1850  # close to the ~1800 planning figure
    def half(k, m):
        lo, hi = wilson_interval(k, m)
        return (hi - lo) / 2
    assert half(0.95 * n, n) <= 0.01
    assert half(0.95 * (n - 1), n - 1) > 0.01  # n is the smallest
    # rounding the successes to an integer count moves the half-width only slightly
    assert half(round(0.95 * n), n) == pytest.approx(0.01, abs=2e-4)
    table = {(r["p"], r["halfwidth"]): r["packets"] for r in sizing_table()}
    assert len(table) == 8
    assert table[(0.99, 0.005)] < table[(0.95, 0.005)] < table[(0.5, 0.005)]
    assert table[(0.5, 0.005)] == pytest.approx(4 * table[(0.5, 0.01)], rel=0.01)  # halving the half-width: about 4x packets
    with pytest.raises(ValueError):
        packets_for_wilson_halfwidth(1.0, 0.01)
