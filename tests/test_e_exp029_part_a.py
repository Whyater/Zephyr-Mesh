"""Checks for the EXP-029 Part A runner (simulated link-only landing counts)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.missions.comms import LossProcess
from sim.missions.presets import get_preset
from tools.c_common import VOLATILE_MANIFEST_FIELDS
import tools.e_exp029_part_a as part_a


@pytest.fixture(scope="module")
def pilot_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp029a")
    a, b = base / "a", base / "b"
    part_a.run(a, ["--pilot", "--processes", "1"], pilot=True, processes=1)
    part_a.run(b, ["--pilot", "--processes", "2"], pilot=True, processes=2)
    return a, b


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pilot_byte_identical_across_process_counts(pilot_dirs):
    a, b = pilot_dirs
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    assert set(part_a.OUTPUTS) | {"manifest.json"} == set(names)
    for name in names:
        if name == "manifest.json":
            ma, mb = (json.loads((d / name).read_text(encoding="utf-8")) for d in (a, b))
            for key in (*VOLATILE_MANIFEST_FIELDS, "argv"):
                ma.pop(key, None)
                mb.pop(key, None)
            assert ma == mb
        else:
            assert _sha(a / name) == _sha(b / name), name
    manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"] == {n: _sha(a / n) for n in part_a.OUTPUTS}
    assert any("synthetic" in s.lower() for s in manifest["limits"])
    assert any("renewal" in s.lower() for s in manifest["limits"])
    assert str(a) not in (a / "summary.json").read_text(encoding="utf-8")


def test_slice_check_controller_equals_vectorised(pilot_dirs):
    a, _ = pilot_dirs
    data = json.loads((a / "slice_check.json").read_text(encoding="utf-8"))
    # 2 cells x 2 timeouts x 2 weak rules, each driven through the real FailsafeController
    assert len(data["checks"]) == 8
    assert data["all_equal"] is True
    for c in data["checks"]:
        assert c["controller"] == c["vectorised"]
    # The pilot is one hour, the same as the slice, so the CSV rows must carry the same counts.
    with open(a / "part_a.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for c in data["checks"]:
        row = next(r for r in rows if r["preset"] == c["preset"] and float(r["h_hz"]) == c["h_hz"]
                   and r["weak_rule"] == c["weak_rule"] and float(r["T_w_s"]) == c["T_w_s"]
                   and float(r["T_s"]) == c["T_s"])
        assert int(row["unnecessary_landings"]) == c["controller"]["landings"]
        assert int(row["degraded_entries"]) == c["controller"]["degraded_entries"]


def test_iid_pilot_cell_agrees_with_hand_formula(pilot_dirs):
    a, _ = pilot_dirs
    with open(a / "part_a.csv", newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["preset"] == "iid_30" and r["weak_rule"] == "age"]
    p, h = 0.30, 20.0
    hours = float(rows[0]["hours"])
    checked = 0
    for r in rows:
        T = float(r["T_s"])
        m = math.floor(T * h + 1e-9)  # misses needed, written here from the formula, not the runner
        expected = 3600.0 * h * (1.0 - p) * p ** m * hours
        if expected < 10:
            continue
        z = (int(r["unnecessary_landings"]) - expected) / math.sqrt(expected)
        assert abs(z) < 4.0, (T, z)
        checked += 1
    assert checked >= 3


def test_effective_misses_match_floor_on_the_grid():
    # Hand derivation: the age just before the next arrival is g/h - 0.002, so a landing needs
    # g >= h (T + 0.002) and the misses are g - 1.  On the grid that is floor(T h).
    for h in part_a.RATES_HZ:
        for T in part_a.T_VALUES_S:
            assert part_a.effective_misses(h, T) == math.floor(T * h + 1e-9)
    assert part_a.effective_misses(50.0, 0.15) == 7  # T h = 7.5 is not whole


def _tiny_arrivals():
    # 20 Hz for 1 s (100 control steps): commands 2..10 lost, so arrivals at 0.002, 0.052, then 0.552.
    n_steps, h = 100, 20.0
    n_cmd = part_a.n_commands_for(n_steps, h)
    assert n_cmd == 20
    lost = np.zeros(n_cmd, dtype=bool)
    lost[2:11] = True
    return part_a.make_arrivals(lost, h, n_steps), n_steps, h


def test_tiny_sequence_known_gap_lands_once():
    arr, n_steps, h = _tiny_arrivals()
    assert arr["proc_step"][:3].tolist() == [1, 6, 56]  # 0.002 -> step 1, 0.052 -> 6, 0.552 -> 56
    times = np.concatenate(([0.0], arr["arrival_s"]))
    proc = np.concatenate(([0], arr["proc_step"]))
    age, lossy = part_a.step_arrays(arr, n_steps, h)
    # The largest age is at step 55 (t = 0.55 s): 0.55 - 0.052 = 0.498 s.
    assert age.max() == pytest.approx(0.498, abs=1e-12)
    for T, want in ((0.3, 1), (0.49, 1), (0.5, 0), (1.0, 0)):
        policy = {"weak_rule": "age", "T_w_s": 0.15, "T_s": T}
        assert part_a.gap_landings(times, proc, n_steps, T) == want
        assert part_a.policy_counts(age, lossy, policy)["landings"] == want
        assert part_a.drive_controller(arr, n_steps, h, policy)["landings"] == want
    # DEGRADED from age 0.15 s: steps 21..35 when T = 0.3 (landing at step 36), steps 21..55 when T = 1.0.
    for T, steps in ((0.3, 15), (1.0, 35)):
        policy = {"weak_rule": "age", "T_w_s": 0.15, "T_s": T}
        want = {"landings": 1 if T < 0.498 else 0, "degraded_entries": 1, "degraded_steps": steps}
        assert part_a.policy_counts(age, lossy, policy) == want
        assert part_a.drive_controller(arr, n_steps, h, policy) == want


def test_tiny_sequence_age_or_loss_matches_controller():
    arr, n_steps, h = _tiny_arrivals()
    age, lossy = part_a.step_arrays(arr, n_steps, h)
    # Hand values for the gap after command 1 (arrived 0.052 s): one full period (0.05 s) has passed
    # from t = 0.102 s, so step 11 (t = 0.11) expects command 2 and sees 1 of 3 missing (0.33 >= 0.3).
    # Step 10 (t = 0.10, 0.048 s after the arrival) still expects nothing new: loss 0.
    assert not lossy[:11].any()
    assert lossy[11:56].all()
    for T in (0.3, 1.0):
        policy = {"weak_rule": "age_or_loss", "T_w_s": 0.15, "T_s": T}
        assert part_a.policy_counts(age, lossy, policy) == part_a.drive_controller(arr, n_steps, h, policy)


def test_loss_free_link_is_never_lossy():
    # A command in flight (sent, not yet arrived) must not count as lost: with no loss at all
    # the loss fraction is 0 at every step, at both rates.
    for h in (20.0, 50.0):
        n_steps = 1000
        lost = np.zeros(part_a.n_commands_for(n_steps, h), dtype=bool)
        arr = part_a.make_arrivals(lost, h, n_steps)
        _, lossy = part_a.step_arrays(arr, n_steps, h)
        assert not lossy.any()
        policy = {"weak_rule": "age_or_loss", "T_w_s": 0.15, "T_s": 0.3}
        assert part_a.drive_controller(arr, n_steps, h, policy) == {"landings": 0, "degraded_entries": 0,
                                                                    "degraded_steps": 0}


@pytest.mark.parametrize("name", ["burst_10_0.5s", "burst_10_0.2s", "iid_10"])
def test_command_losses_equal_is_lost(name):
    h = 20.0
    lp = LossProcess(name, seed=12345, dt_s=0.01, duration_s=300.0)
    n = part_a.n_commands_for(part_a.n_steps_for(300.0 / 3600.0), h)
    fast = part_a.command_losses(lp, h, n)
    fresh = LossProcess(name, seed=12345, dt_s=0.01, duration_s=300.0)
    slow = np.array([fresh.is_lost("cmd:d1", k, k / h) for k in range(n)])
    assert np.array_equal(fast, slow)
    assert 0 < fast.sum() < n


def test_burst_closed_form_limit():
    # Limiting case: with k = 1 (100 Hz) the coarse chain is the fine chain, so the rate must equal
    # 3600 h p r (1 - r)^(m - 1) with r = 0.01 / burst_mean_s (written here by hand).
    preset = get_preset("burst_10_0.5s")
    p, r = preset.loss_probability, 0.01 / preset.burst_mean_s
    for T in (0.3, 1.0):
        m = math.floor(T * 100.0 + 1e-9)
        want = 3600.0 * 100.0 * p * r * (1.0 - r) ** (m - 1)
        assert part_a.closed_form_per_hour("burst_10_0.5s", 100.0, T) == pytest.approx(want, rel=1e-9)
    assert part_a.closed_form_per_hour("clean", 20.0, 0.3) == 0.0


def test_refuses_non_empty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit):
        part_a.run(tmp_path, ["--pilot"], pilot=True, processes=1)
