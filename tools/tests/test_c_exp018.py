import hashlib
import json
import math
from pathlib import Path

import pytest
import numpy as np

from sim.radio.channel import GilbertElliott
from sim.trace import EspNowTrace
from tools.calibrate_radio_from_trace import PositionInput, calibrate, synthetic_trace_dicts
from tools.c_exp018_recovery import (BURST_AWARE_SIZING_RULE, BURST_P_GOOD_TO_BAD,
                                     BURST_R_BAD_TO_GOOD, _falsification,
                                     _trace_with_provenance, burst_aware_sizing_table,
                                     coverage_stat, run)
from tools.calibrate_radio_from_trace import packets_for_wilson_halfwidth, sizing_table
from tools import c_common


def test_coverage_stat_hand_list():
    result = coverage_stat([1.0, 1.0, 2.0, 2.0], [[0.5, 1.5], [0.8, 1.2], [1.5, 2.5], [1.1, 1.9]], 1.0)
    assert result["coverage"] == 0.5
    assert result["n"] == 4


def test_wilson_sizing_hand_values():
    # Wilson half-width is h = z*sqrt(p(1-p)/n + z^2/(4n^2))/(1 + z^2/n).
    # The Wald values (1825, 1521) explain the pre-registration's "about 1800 / 1500".
    assert packets_for_wilson_halfwidth(0.95, 0.01) == 1838
    assert packets_for_wilson_halfwidth(0.99, 0.005) == 1606


def test_falsification_rssi_rows_name_exp002():
    rows = _falsification()
    rssi_rows = [row for row in rows if "RSSI" in row["measurement"]]
    assert rssi_rows
    assert all("EXP-002" in row["measurement"] for row in rssi_rows)
    channel_parameters = {"a_dbm", "n", "n_outdoor", "sigma_db"}
    assert all("EXP-001" not in row["measurement"]
               for row in rows if row["parameter"] in channel_parameters)


def test_burst_inflation_hand_value():
    rho = 1.0 - 0.02 - 0.2
    inflation_factor = (1.0 + rho) / (1.0 - rho)
    assert inflation_factor == pytest.approx(8.0909, rel=1e-3)


def test_burst_inflation_matches_simulated_chain():
    chain = GilbertElliott(0.02, 0.2)
    n = 2000
    replicates = 4000
    rng = np.random.default_rng(18018)
    delivery_means = np.empty(replicates)
    for i in range(replicates):
        # The default start_state=None samples the stationary distribution.
        _, lost = chain.sample(n, rng, start_state=None)
        delivery_means[i] = np.mean(~lost)
    q = 0.2 / (0.02 + 0.2)
    empirical_inflation = np.var(delivery_means, ddof=1) / (q * (1.0 - q) / n)
    expected_inflation = (1.0 + (1.0 - 0.02 - 0.2)) / (1.0 - (1.0 - 0.02 - 0.2))
    assert empirical_inflation == pytest.approx(expected_inflation, rel=0.10)


def test_burst_aware_sizing_rows_preserve_independent_table():
    independent_rows = sizing_table()
    burst_rows = burst_aware_sizing_table()
    assert sizing_table() == independent_rows
    assert len(burst_rows) == len(independent_rows)
    expected_inflation = (1.0 + (1.0 - BURST_P_GOOD_TO_BAD - BURST_R_BAD_TO_GOOD)) / (
        1.0 - (1.0 - BURST_P_GOOD_TO_BAD - BURST_R_BAD_TO_GOOD))
    expected_fields = {"p", "halfwidth", "packets_independent", "inflation_factor",
                       "packets_bursty", "burst_p", "burst_r"}
    for base, row in zip(independent_rows, burst_rows):
        assert set(row) == expected_fields
        assert row["p"] == base["p"]
        assert row["halfwidth"] == base["halfwidth"]
        assert row["packets_independent"] == base["packets"]
        assert row["inflation_factor"] == pytest.approx(expected_inflation)
        assert row["packets_bursty"] == math.ceil(expected_inflation * base["packets"])
        assert row["burst_p"] == BURST_P_GOOD_TO_BAD
        assert row["burst_r"] == BURST_R_BAD_TO_GOOD
    assert "large-n approximation" in BURST_AWARE_SIZING_RULE
    assert "delivery probability" in BURST_AWARE_SIZING_RULE
    assert "stationary delivery" in BURST_AWARE_SIZING_RULE


def test_coverage_stat_counts_unidentifiable_entries():
    result = coverage_stat([1, None, 1], [[0, 2], None, [3, 4]], 1)
    assert result["n_unidentifiable"] == 1
    assert result["coverage"] == 0.5


def test_wilson_interval_for_five_of_ten():
    result = coverage_stat([1] * 10, [[0, 2]] * 5 + [[2, 3]] * 5, 1)
    assert result["coverage_wilson"] == pytest.approx([0.2366, 0.7634], abs=1e-4)


def test_exact_path_loss_and_zero_loss():
    docs = synthetic_trace_dicts(n=2.0, a_dbm=-20.7, sigma_db=0.0, distances_m=(1., 5., 10., 20., 40.),
                                 packets_per_distance=20, ge=GilbertElliott(0.0, 1.0), fast_sigma_db=0.0, seed=1234)
    positions = [PositionInput(pid, _trace_with_provenance(doc), d) for pid, d, doc in docs]
    result = calibrate(positions)
    assert abs(result["path_loss"]["path_loss_exponent"] - 2.0) < 1e-9
    assert abs(result["path_loss"]["a_dbm_at_1m"] - (-20.7)) < 1e-9
    assert result["path_loss"]["shadowing_sigma_db"] == pytest.approx(0.0, abs=1e-12)
    assert result["gilbert_pooled"]["p_good_to_bad"] == 0.0


def _hash_tree(path: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}


def test_pilot_byte_identical(tmp_path, monkeypatch):
    # Keep Matplotlib's cache in the writable test directory.  This avoids a
    # platform font scan while the pilot is checking output determinism.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("MPL_IGNORE_SYSTEM_FONTS", "1")
    a, b = tmp_path / "a", tmp_path / "b"
    run(a, ["--output", str(a), "--pilot", "--processes", "1"], True, 1)
    run(b, ["--output", str(b), "--pilot", "--processes", "1"], True, 1)
    # Manifest carries runtime and host fields, so compare every reproducible artifact.
    for name in ("summary.json", "recovery.csv", "sweeps.csv", "exp018_coverage.png", "exp018_design.png"):
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == hashlib.sha256((b / name).read_bytes()).hexdigest()

    summary = json.loads((a / "summary.json").read_text())
    assert summary["sizing_table"] == sizing_table()
    assert summary["burst_aware_sizing_table"] == burst_aware_sizing_table()
    assert summary["burst_aware_sizing_rule"] == BURST_AWARE_SIZING_RULE

    manifest_a = json.loads((a / "manifest.json").read_text())
    manifest_b = json.loads((b / "manifest.json").read_text())
    for key in (*c_common.VOLATILE_MANIFEST_FIELDS, "argv"):
        manifest_a.pop(key, None)
        manifest_b.pop(key, None)
    assert manifest_a == manifest_b
