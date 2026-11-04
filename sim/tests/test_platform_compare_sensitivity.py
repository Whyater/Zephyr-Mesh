"""How sensitive is the off-macOS float rule in platform_compare?

Registered numbers were recorded on macOS. Off macOS, platform_compare accepts a
number within a relative error of 1e-9 (absolute 1e-12 near zero). These checks
run that off-macOS rule on every platform (``EXACT`` forced to False) against real
registered numbers, so they pin down two things:

* small real bugs still fail the comparison: a 1e-8 relative change to the
  position gains, a flipped drag sign, gravity 9.80665 instead of 9.81, one
  step too few, a 1e-7 nudge to a registered p95;
* the platform noise actually seen between macOS and Windows still passes: a
  one-ulp change to the mass, the largest Windows difference seen on the
  registered rows (3.7e-13 relative), and a near-zero sideways position whose
  sign differs between the two platforms.

Loosening RTOL to 1e-8 or more, or ATOL to 1e-6 or more, makes the first group
fail. Making the rule exact makes the second group fail.
"""
from __future__ import annotations

import numpy as np
import pytest

import sim.drone
import sim.tests.platform_compare as pc
from sim.scenarios import default_hardware_profile
from sim.vehicle_fleet import ActuatedVehicleFleet


@pytest.fixture(autouse=True)
def off_macos_rule(monkeypatch):
    """Judge with the rule Windows and Linux CI use, on every platform."""
    monkeypatch.setattr(pc, "EXACT", False)


# The registered A4 fleet row (sim/tests/test_a4_vehicle_chain.py,
# test_v1_fleet_default_path_is_bit_identical_after_ten_steps), produced on macOS.
FLEET_ROW = {
    "alpha.position": [-1.0000312078453029e+00, 2.7468810368889150e-20, 1.4642156077432626e+00],
    "alpha.velocity": [-1.8303401970157962e-03, 1.7332712085910140e-18, -5.7471177506341153e-01],
    "alpha.quaternion": [9.999843586810702e-01, -6.209539120597357e-18, -5.5930665299916530e-03,
                         float.fromhex("0x1.49f7c150d1ee6p-59")],
    "alpha.angular_velocity": [-5.0907339472669637e-16, -3.0803294977354573e-01, 1.3066505492540819e-16],
    "beta.position": [1.0000312078451201e+00, 3.7702910384455593e-20, 1.4642156077112434e+00],
    "beta.velocity": [1.8303401813779050e-03, 1.9825103558335440e-18, -5.7471177694580210e-01],
    "beta.quaternion": [float.fromhex("0x1.fffdf32a180ecp-1"), -5.6756084669090474e-18,
                        5.5930665192784178e-03, 1.0834399748626498e-18],
    "beta.angular_velocity": [-3.9370960554297174e-16, 3.0803294867381897e-01, 5.6415899251056072e-17],
}


def _fleet(*, steps=10, gain_factor=1.0, mass_ulps=0):
    fleet = ActuatedVehicleFleet(
        {"beta": default_hardware_profile(1), "alpha": default_hardware_profile(0)},
        initial_positions={"alpha": (-1.0, 0.0, 1.5), "beta": (1.0, 0.0, 1.5)},
    )
    for vehicle in fleet._vehicles.values():
        vehicle.controller.K_p_pos = vehicle.controller.K_p_pos * gain_factor
        for _ in range(mass_ulps):
            vehicle.drone.mass = float(np.nextafter(vehicle.drone.mass, np.inf))
    targets = {"alpha": (-1.5, 0.0, 1.5), "beta": (1.5, 0.0, 1.5)}
    for _ in range(steps):
        fleet.step(targets, 0.01)
    return {f"{name}.{field}": getattr(fleet._vehicles[name].drone, field)
            for name in ("alpha", "beta") for field in ("position", "velocity", "quaternion", "angular_velocity")}


def _matches_registered(state) -> bool:
    return all(pc.same_array(state[key], expected) for key, expected in FLEET_ROW.items())


def _worst_relative_change(state) -> float:
    worst = 0.0
    for key, expected in FLEET_ROW.items():
        got, ref = np.asarray(state[key]), np.asarray(expected)
        big = np.abs(ref) > 1e-12
        worst = max(worst, float(np.max(np.abs(got[big] - ref[big]) / np.abs(ref[big]))))
    return worst


# ------------------------------------------------- the unchanged code passes


def test_unchanged_fleet_matches_the_registered_row():
    assert _matches_registered(_fleet())


# --------------------------------------- planted bugs must fail the comparison


def test_position_gains_off_by_1e8_relative_are_caught():
    state = _fleet(gain_factor=1.0 + 1e-8)
    assert _worst_relative_change(state) > 2e-9, "the planted bug should move the row by about 1e-8"
    assert not _matches_registered(state)


def test_one_step_too_few_is_caught():
    assert not _matches_registered(_fleet(steps=9))


def test_flipped_drag_sign_is_caught(monkeypatch):
    real_drag = sim.drone.calculate_drag
    monkeypatch.setattr(sim.drone, "calculate_drag", lambda *args, **kwargs: -real_drag(*args, **kwargs))
    assert not _matches_registered(_fleet())


def test_gravity_9_80665_instead_of_9_81_is_caught(monkeypatch):
    monkeypatch.setattr(sim.drone, "GRAVITY", 9.80665)
    monkeypatch.setattr("sim.controller.GRAVITY", 9.80665)
    assert not _matches_registered(_fleet())


# Registered numbers (m): an S6 told-arm p95 (sim/runs/s6-recovery/per_seed.csv), an EXP-024
# single-arm p95 (sim/runs/d3-tolerance/per_seed.csv.gz) and the A4 hover z after 500 steps.
REGISTERED_SCALES = [0.3380950599610277, 1.59124671277418, -3.918359815787623e-03]


@pytest.mark.parametrize("value", REGISTERED_SCALES)
@pytest.mark.parametrize("relative", [1e-7, 1e-8])
def test_registered_scale_numbers_reject_small_relative_nudges(value, relative):
    nudged = value * (1.0 + relative)
    assert not pc.same_tree(nudged, value)
    assert not pc.same_array([nudged], [value])
    assert not pc.same_text(repr(nudged), repr(value))
    assert not pc.same_csv_row(f"cell,{nudged!r},true", f"cell,{value!r},true")


# ---------------------------------------- platform noise must still pass


def test_one_ulp_mass_change_still_matches():
    assert _matches_registered(_fleet(mass_ulps=1))


@pytest.mark.parametrize("value", REGISTERED_SCALES)
def test_largest_seen_windows_difference_still_matches(value):
    # The largest relative gap between a Windows rerun and the macOS numbers on the
    # four audited registered rows was 3.7e-13; allow a few times that.
    shifted = value * (1.0 + 1e-12)
    assert pc.same_tree(shifted, value)
    assert pc.same_text(repr(shifted), repr(value))


def test_near_zero_sideways_position_with_the_other_sign_still_matches():
    # Registered macOS value and a Windows rerun of the same fleet row: same code,
    # opposite sign, both far below 1e-12 m. Only the absolute floor accepts this.
    mac, windows = 2.7468810368889150e-20, -1.8268591695227256e-19
    assert pc.same_tree(windows, mac)
    assert pc.same_array([-1.0000312078453029, windows, 1.4642156077432626],
                         [-1.0000312078453029, mac, 1.4642156077432626])
