import hashlib
import json
from pathlib import Path

import pytest

from sim.radio.channel import GilbertElliott
from sim.trace import EspNowTrace
from tools.calibrate_radio_from_trace import PositionInput, calibrate, synthetic_trace_dicts
from tools.c_exp018_recovery import coverage_stat, _trace_with_provenance, run
from tools.calibrate_radio_from_trace import packets_for_wilson_halfwidth
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

    manifest_a = json.loads((a / "manifest.json").read_text())
    manifest_b = json.loads((b / "manifest.json").read_text())
    for key in (*c_common.VOLATILE_MANIFEST_FIELDS, "argv"):
        manifest_a.pop(key, None)
        manifest_b.pop(key, None)
    assert manifest_a == manifest_b
