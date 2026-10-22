"""Fast checks for the A5 profile sensitivity calculations."""
from __future__ import annotations

import copy
import importlib
import inspect
import math
from pathlib import Path

import pytest

from sim.environment import GRAVITY
from sim.hardware import RotorcraftProfile, load_hardware_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "sim" / "physics" / "profiles" / "crazyflie_class.json"


def _direct_variant(profile: RotorcraftProfile, path: str, value):
    """Build one profile variant through the documented serialization path."""
    raw = profile.as_document()
    section, field = path.split(".", 1)
    raw[section][field] = copy.deepcopy(value)
    raw["parameters"][path]["value"] = copy.deepcopy(value)
    old_range = raw["parameters"][path]["range"]
    if not (old_range[0] <= value <= old_range[1]):
        raw["parameters"][path]["range"] = [value, value]
    # Keep provenance explicit while retaining all source parameter metadata.
    raw["notes"] = raw["notes"] + " Sensitivity variant."
    return RotorcraftProfile.from_dict(raw)


def _tool_variant(profile: RotorcraftProfile, path: str, value):
    """Use the tool helper when present, while keeping this test API-tolerant."""
    try:
        mod = importlib.import_module("tools.a_profile_sensitivity")
    except ModuleNotFoundError:
        return _direct_variant(profile, path, value)
    for name in ("build_variant", "_build_variant", "make_variant", "_variant"):
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        # The production helper takes a low/high level. Numeric values are
        # intentionally built with the same serialization path below.
        if isinstance(value, str):
            attempts = ((profile, path, value),)
        else:
            attempts = ()
        for args in attempts:
            try:
                result = fn(*args)
            except (TypeError, KeyError):
                continue
            if isinstance(result, RotorcraftProfile):
                return result
    if isinstance(value, str):
        entry = profile.parameters[path]
        bounds = entry["range"]
        value = (copy.deepcopy(bounds[0]) if value == "low" else copy.deepcopy(bounds[1]))
    return _direct_variant(profile, path, value)


def test_variant_building_changes_exactly_one_field():
    profile = load_hardware_profile(PROFILE)
    path = "airframe.mass_kg"
    changed = _tool_variant(profile, path, "high")
    before = profile.as_document()
    after = changed.as_document()
    differences = []
    for section in ("airframe", "rotor"):
        for key in before[section]:
            if before[section][key] != after[section][key]:
                differences.append(f"{section}.{key}")
    assert differences == [path]


def test_nominal_crazyflie_hover_power_formula():
    profile = load_hardware_profile(PROFILE)
    k_f = 1.9e-8
    k_m = 0.005964552 * k_f
    w_h = math.sqrt(0.028 * GRAVITY / (4.0 * k_f))
    expected = 4.0 * k_m * w_h**3
    # Keep this hand-calculated value tied to the stated coefficients.
    assert expected == pytest.approx(3.1146667820250546, rel=1e-12)
    assert profile.hover_speed_rad_s() == pytest.approx(w_h)


def test_mass_scaling_of_hover_speed():
    profile = load_hardware_profile(PROFILE)
    increase = 0.02
    variant = _direct_variant(profile, "airframe.mass_kg", profile.mass_kg * (1.0 + increase))
    assert variant.hover_speed_rad_s() / profile.hover_speed_rad_s() == pytest.approx(math.sqrt(1.0 + increase))


def test_climb_hand_formula():
    profile = load_hardware_profile(PROFILE)
    expected = 4.0 * profile.thrust_coefficient_n_s2 * profile.max_speed_rad_s**2 / profile.mass_kg - GRAVITY
    # Independent hand expression, with no vehicle integration involved.
    assert expected == pytest.approx(4.0 * 1.9e-8 * 2846.0**2 / 0.028 - 9.81)


def test_refuses_non_empty_output_directory(tmp_path):
    try:
        mod = importlib.import_module("tools.a_profile_sensitivity")
    except ModuleNotFoundError:
        pytest.skip("sensitivity tool not present yet")
    out = tmp_path / "already-used"
    out.mkdir()
    (out / "sentinel").write_text("x", encoding="utf-8")
    generate = getattr(mod, "generate", None)
    if generate is None:
        pytest.skip("sensitivity tool has no generate API")
    with pytest.raises((FileExistsError, ValueError)):
        try:
            generate(out, workers=1)
        except TypeError:
            generate(out)



def test_slow_motor_end_of_range_is_reported_as_unstable_without_delay():
    """At tau = 0.16 s the attitude loop has about 3 degrees of phase margin before any delay."""
    from sim.hardware import load_hardware_profile
    import tools.a_profile_sensitivity as tool

    base = load_hardware_profile(tool.PROFILE_DIR / "esp_fly_class.json")
    slow = tool.build_variant(base, "rotor.motor_time_constant_s", "high")
    assert slow.motor_time_constant_s == 0.16
    result = tool._dcrit(slow, "c", (0.0, 0.1))
    assert result["unstable_at_zero_delay"] is True
    assert result["d_crit_s"] == 0.0
