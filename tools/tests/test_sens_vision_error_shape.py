"""Checks for the vision error shape sensitivity tool.

The statistical checks compare generated errors with numbers worked out by
hand (equal total RMS, sqrt(N) averaging, the Gauss-Markov step size), not
with the code's own formulas.
"""
from __future__ import annotations

from math import exp, sqrt

import numpy as np
import pytest

from sim.s6_recovery import RecoveryConfig, deliver, seed_list
from tools import sens_vision_error_shape as ves

CONFIG = RecoveryConfig()
DT = CONFIG.observation_period_s


def _draw(model, noise, seed, n=121, **kwargs):
    return ves.error_series(model, noise, n, DT, np.random.default_rng(seed), **kwargs)


@pytest.mark.parametrize("model", ves.MODELS)
def test_every_model_has_the_same_total_mean_square(model):
    # Hand value: 3 axes x (0.10 m)^2 = 0.03 m^2 for the 10 cm cell.
    totals = ves.axis_totals(model, 0.10)
    assert float(np.sum(totals**2)) == pytest.approx(0.03, rel=1e-12)


def test_anisotropic_split_by_hand():
    # lateral^2 (1 + 9 + 1) = 3 s^2  ->  lateral = s sqrt(3/11) = 0.0522 m at s = 0.10 m.
    totals = ves.axis_totals(ves.CONST_BIAS, 0.10)
    assert totals[0] == pytest.approx(0.0522233, abs=1e-6)
    assert totals[ves.DEPTH_AXIS] == pytest.approx(3 * 0.0522233, abs=1e-6)


@pytest.mark.parametrize("model", [ves.WHITE_ANISO, ves.CONST_BIAS, ves.OU_BIAS])
def test_ensemble_rms_matches_the_axis_totals(model):
    # Over many seeds and frames the per-axis RMS should land on the target.
    errors = np.concatenate([_draw(model, 0.05, seed) for seed in range(600)])
    rms = np.sqrt(np.mean(errors**2, axis=0))
    assert rms == pytest.approx(ves.axis_totals(model, 0.05), rel=0.06)


def test_position_bias_ensemble_rms_matches():
    truth = np.zeros((121, 3))
    truth[:, 0] = np.arange(121) * DT  # 1 m/s straight line
    errors = np.concatenate([_draw(ves.POS_BIAS, 0.05, s, truth_m=truth) for s in range(800)])
    rms = np.sqrt(np.mean(errors**2, axis=0))
    assert rms == pytest.approx(ves.axis_totals(ves.POS_BIAS, 0.05), rel=0.06)


def test_averaging_shrinks_white_noise_by_sqrt_n_but_not_a_bias():
    # Hand example: average 25 frames. White noise std 0.10 -> 0.10 / 5 = 0.02.
    # A constant bias is the same in all 25 frames, so its average is itself.
    rng = np.random.default_rng(1)
    white = rng.normal(0.0, 0.10, size=(20000, 25))
    assert np.std(white.mean(axis=1)) == pytest.approx(0.02, rel=0.03)
    biased = np.stack([_draw(ves.CONST_BIAS, 0.10, seed, n=25)[:, 0] for seed in range(4000)])
    lateral_total = ves.axis_totals(ves.CONST_BIAS, 0.10)[0]
    assert np.std(biased.mean(axis=1)) == pytest.approx(lateral_total, rel=0.05)


def test_constant_bias_frame_to_frame_change_is_only_jitter():
    errors = _draw(ves.CONST_BIAS, 0.10, 3, n=5000)
    diff_std = np.std(np.diff(errors, axis=0), axis=0)
    # Differencing two frames of white jitter doubles its variance: sqrt(2) x jitter.
    assert diff_std[0] == pytest.approx(sqrt(2) * ves.JITTER_LATERAL_M, rel=0.05)
    assert diff_std[ves.DEPTH_AXIS] == pytest.approx(sqrt(2) * ves.JITTER_DEPTH_M, rel=0.05)


def test_gauss_markov_bias_steps_are_centimetres_per_frame():
    # This is why the pre-registered model c is not camera-shaped: with tau = 3 s at
    # 10 Hz its frame-to-frame change has std sigma_b sqrt(2 (1 - phi)): 0.1567 m x
    # sqrt(2 x 0.0328) = 4.0 cm on the depth axis of the 10 cm cell, far above the 2 mm measured jitter.
    phi = exp(-DT / 3.0)
    _, bias_std = ves.jitter_and_bias_std(ves.OU_BIAS, 0.10)
    expected = bias_std[ves.DEPTH_AXIS] * sqrt(2 * (1 - phi))
    assert expected == pytest.approx(0.0401, abs=0.0005)
    diffs = np.concatenate([np.diff(_draw(ves.OU_BIAS, 0.10, s, n=400)[:, ves.DEPTH_AXIS]) for s in range(200)])
    assert np.std(diffs) == pytest.approx(expected, rel=0.05)


def test_position_bias_is_fixed_for_a_still_target():
    still = np.tile([0.7, -0.3, 0.0], (300, 1))
    errors = _draw(ves.POS_BIAS, 0.10, 5, n=300, truth_m=still)
    diff_std = np.std(np.diff(errors, axis=0), axis=0)
    assert diff_std[0] == pytest.approx(sqrt(2) * ves.JITTER_LATERAL_M, rel=0.15)
    assert abs(np.mean(errors[:, ves.DEPTH_AXIS])) > 10 * ves.JITTER_DEPTH_M / sqrt(300)


@pytest.mark.parametrize("model", [m for m in ves.MODELS if m != ves.WHITE_ISO])
def test_zero_noise_cell_is_exact_truth(model):
    truth = np.zeros((121, 3))
    assert np.array_equal(_draw(model, 0.0, 9, truth_m=truth), truth)


@pytest.mark.parametrize("cell", [("constant_velocity", 0.1, 0.3, 0.05), ("constant_turn", 0.0, 0.0, 0.10)])
def test_white_iso_delivery_is_identical_to_s6(cell):
    motion, delay, loss, noise = cell
    for seed in seed_list(3):
        mine = ves.deliver_shaped(motion, ves.WHITE_ISO, delay_s=delay, loss_probability=loss,
                                  noise_std_m=noise, seed=seed, config=CONFIG)
        theirs = deliver(motion, delay_s=delay, loss_probability=loss, noise_std_m=noise, seed=seed, config=CONFIG)
        assert mine == theirs


def test_models_share_the_same_link_draws():
    kwargs = {"delay_s": 0.1, "loss_probability": 0.3, "noise_std_m": 0.05, "seed": seed_list(1)[0],
              "config": CONFIG}
    base = [(d.seq, d.receive_time_s) for d in ves.deliver_shaped("constant_velocity", ves.WHITE_ISO, **kwargs)]
    for model in ves.MODELS[1:]:
        other = [(d.seq, d.receive_time_s) for d in ves.deliver_shaped("constant_velocity", model, **kwargs)]
        assert other == base


def test_grid_size():
    # 224 S6 main cells for white_iso; every other variant skips the 56 noise-0 cells.
    cells = ves.build_cells(taus=(1.0, 3.0), wavelengths=(2.0,))
    assert len(cells) == 224 + 168 * (len(ves.MODELS) - 2 + 2)
    assert len({c["cell_id"] for c in cells}) == len(cells)


def test_control_check_matches_registered_rows():
    cells = [c for c in ves.build_cells(models=(ves.WHITE_ISO,))
             if c["delay_s"] == 0.05 and c["loss_probability"] == 0.1 and c["noise_std_m"] == 0.05]
    seeds = seed_list(2)
    results = {c["cell_id"]: ves.run_cell(c, seeds) for c in cells}
    report = ves.control_check(cells, results, seeds)
    assert report["values_compared"] == len(cells) * 2 * 2 * len(ves.PER_SEED_KEYS)
    assert report["mismatches"] == 0, report["examples"]


def test_seed_floor_for_a_constant_bias_is_the_bias_length():
    # Hand check: with a constant bias and 1 to 2 mm jitter, a seed's p95 of |error| is its
    # bias length to within a few millimetres, so the share over 0.25 m matches the bias draws.
    seeds = seed_list(5)
    for seed in seeds:
        floor = ves.seed_error_floor(ves.CONST_BIAS, 0.10, "constant_velocity", [seed])
        _, sensor_seed = (int(v) for v in np.random.SeedSequence(seed).generate_state(2))
        _, bias_std = ves.jitter_and_bias_std(ves.CONST_BIAS, 0.10)
        bias = np.random.default_rng(sensor_seed).normal(0.0, 1.0, size=3) * bias_std
        assert floor["median_seed_p95_m"] == pytest.approx(float(np.linalg.norm(bias)), abs=0.005)


def _fake_result(labels_by_arm, fail_rate=0.0):
    return {"arms": {arm: {"label": label, "fail_rate": fail_rate, "p95_median_m": 0.1}
                     for arm, label in labels_by_arm.items()}}


def test_tolerance_rows_on_a_hand_made_pattern():
    # Recovery passes up to 100 ms then fails; naive already fails at 0 ms.
    # A later pass after a failure must not count. Noise 0 is shared by every variant.
    cells = [c for c in ves.build_cells(models=(ves.WHITE_ISO, ves.CONST_BIAS))
             if c["motion"] == "constant_velocity" and c["loss_probability"] == 0.0
             and c["noise_std_m"] in (0.0, 0.05)]
    results = {}
    for c in cells:
        rec = "passes" if c["delay_s"] <= 0.10 or c["delay_s"] == 0.30 else "fails"
        results[c["cell_id"]] = _fake_result({"naive": "fails", "recovery": rec})
    rows = ves.tolerance_rows(cells, results)
    by = {(r["variant"], r["noise_std_m"], r["arm"]): r["tolerated_delay_s"] for r in rows}
    assert by[(ves.WHITE_ISO, 0.05, "recovery")] == 0.10
    assert by[(ves.CONST_BIAS, 0.05, "recovery")] == 0.10
    assert by[(ves.WHITE_ISO, 0.05, "naive")] == ""
    # The noise-0 row exists for the biased variant too, built from the white_iso cells.
    assert by[(ves.CONST_BIAS, 0.0, "recovery")] == 0.10
