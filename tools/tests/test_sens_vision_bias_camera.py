"""Checks for the camera-shaped vision bias tool.

Hand numbers where possible (image coordinates of known points, equal total
RMS, a still marker keeps its bias), plus an exact match of the copied link
code against the merged tool's own delivery function.
"""
from __future__ import annotations

from math import radians, tan

import numpy as np
import pytest

from sim.s6_recovery import RecoveryConfig, seed_list, truth_position
from tools import sens_vision_bias_camera as cam
from tools import sens_vision_error_shape as ves

CONFIG = RecoveryConfig()
TIMES = np.arange(121) * CONFIG.observation_period_s


def test_image_coords_by_hand():
    # Line camera at (6, -6 / tan 35 deg, 0) = (6, -8.569, 0). The path start x = 0 sits
    # 6 m left at 8.569 m depth: u = (6 / 8.569) / tan 35 deg = 1.0 (the image edge).
    assert cam.CAMERA_POSITIONS["constant_velocity"][1] == pytest.approx(-6 / tan(radians(35)), rel=1e-12)
    u, rho = cam.image_coords(np.array([[0.0, 0, 0], [6.0, 0, 0], [12.0, 0, 0]]), "constant_velocity")
    assert u == pytest.approx([-1.0, 0.0, 1.0], abs=1e-12)
    # Range to the edge is depth / cos 35 deg, so rho = 1 / cos 35 deg - 1 = 0.2208.
    assert rho == pytest.approx([0.22077, 0.0, 0.22077], abs=1e-5)
    # Turn camera at (0, -2, 0), circle centre (0, 1, 0) at 3 m: the near point (0, 0, 0)
    # is at 2 m (rho = -1/3), the far point (0, 2, 0) at 4 m (rho = +1/3).
    u, rho = cam.image_coords(np.array([[0.0, 0, 0], [0.0, 2, 0]]), "constant_turn")
    assert u == pytest.approx([0.0, 0.0], abs=1e-12)
    assert rho == pytest.approx([-1 / 3, 1 / 3], abs=1e-12)


@pytest.mark.parametrize("model", cam.CAMERA_MODELS)
@pytest.mark.parametrize("motion", ves.MOTIONS)
def test_ensemble_rms_matches_model_b(model, motion):
    # Same size rule as the merged tool: per-axis total = s sqrt(3/11) lateral, 3x that in depth.
    truth = truth_position(motion, TIMES, CONFIG)
    rng = np.random.default_rng(3)
    errors = np.concatenate([cam.camera_error_series(model, 0.10, truth, motion, rng) for _ in range(3000)])
    rms = np.sqrt(np.mean(errors**2, axis=0))
    assert rms == pytest.approx(ves.axis_totals(ves.CONST_BIAS, 0.10), rel=0.05)
    assert float(np.sum(ves.axis_totals(ves.CONST_BIAS, 0.10) ** 2)) == pytest.approx(0.03, rel=1e-12)


def test_still_marker_keeps_its_bias():
    # A camera's accuracy error depends on where the marker is, not on time.
    still = np.tile([[3.0, 0.0, 0.0]], (121, 1))
    bias = cam.camera_bias(cam.CAMERA, 0.10, still, "constant_velocity", np.random.default_rng(1))
    assert np.ptp(bias, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)


def test_camera_field_moves_by_millimetres_per_frame():
    # The point of the model: on both S6 paths the depth bias steps by under 1 cm per
    # frame at the 10 cm cell, against about 4 cm for the merged run's Gauss-Markov tau 3 s.
    for motion in ves.MOTIONS:
        speed = cam.bias_change_stats(cam.CAMERA, 0.10, motion, draws=800)
        assert speed["frame_step_rms_depth_m"] < 0.010
    ou = cam.bias_change_stats(ves.OU_BIAS, 0.10, "constant_velocity")
    # Hand: 0.1567 m x sqrt(2 (1 - exp(-0.1 / 3))) = 0.0401 m.
    assert ou["frame_step_rms_depth_m"] == pytest.approx(0.0401, abs=2e-4)


def test_the_two_camera_models_share_coefficients():
    truth = truth_position("constant_turn", TIMES, CONFIG)
    full = cam.camera_bias(cam.CAMERA, 0.05, truth, "constant_turn", np.random.default_rng(9))
    part = cam.camera_bias(cam.CAMERA_NC, 0.05, truth, "constant_turn", np.random.default_rng(9))
    phi = cam.basis(*cam.image_coords(truth, "constant_turn"))
    norm_full = np.sqrt(np.mean(np.sum(phi**2, axis=1)))
    phi[:, 0] = 0.0
    norm_part = np.sqrt(np.mean(np.sum(phi**2, axis=1)))
    # Removing the constant term (the same constant c0 x bias_std on every frame) is the only change.
    constant = full * norm_full - part * norm_part
    assert np.ptp(constant, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("model", [ves.CONST_BIAS, ves.WHITE_ANISO])
def test_copied_link_code_matches_the_merged_tool_exactly(model):
    for seed in seed_list(3):
        for delay, loss in ((0.0, 0.0), (0.1, 0.3), (0.3, 0.5)):
            kwargs = {"delay_s": delay, "loss_probability": loss, "seed": seed, "config": CONFIG}
            theirs = ves.deliver_shaped("constant_turn", model, noise_std_m=0.05, **kwargs)
            mine = cam.deliver_with_error(
                "constant_turn",
                lambda truth, times, rng: ves.error_series(model, 0.05, times.size, CONFIG.observation_period_s,
                                                           rng, ves.MAIN_TAU_S, truth, ves.MAIN_WAVELENGTH_M),
                **kwargs)
            assert mine == theirs


def test_grid_has_reference_and_camera_cells():
    cells = cam.build_cells()
    # 3 reference + 2 camera models, 2 motions, 7 delays, 4 losses, 2 noise cells.
    assert len(cells) == 5 * 2 * 7 * 4 * 2
    assert len({c["cell_id"] for c in cells}) == len(cells)
    assert {c["noise_std_m"] for c in cells} == {0.05, 0.10}
