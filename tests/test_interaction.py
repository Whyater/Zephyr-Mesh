import numpy as np

from sim.interaction import (
    DownwashInteraction,
    downwash_wind_at,
    ellipsoid_separation,
    is_separated,
    min_horizontal_clearance,
    min_vertical_clearance,
)


def test_ellipsoid_keep_out_is_symmetric_and_has_hand_cases():
    assert ellipsoid_separation([0, 0, 0], [0.24, 0, 0]) == 1.0
    assert ellipsoid_separation([0, 0, 0], [0, 0, 0.60]) == 1.0
    assert ellipsoid_separation([0, 0, 0], [0.24, 0, 0.60]) == np.sqrt(2)
    assert ellipsoid_separation([1, 2, 3], [4, 5, 6]) == ellipsoid_separation([4, 5, 6], [1, 2, 3])
    assert is_separated([0, 0, 0], [0.24, 0, 0])
    assert min_horizontal_clearance((0.12, 0.12, 0.30)) == 0.24
    assert min_vertical_clearance((0.12, 0.12, 0.30)) == 0.60


def test_downwash_hand_value_and_column_limits():
    thrust = 0.030 * 9.81
    radius = 0.023
    area = 4 * np.pi * radius**2
    expected_vh = np.sqrt(thrust / (2 * 1.225 * area))
    wind = downwash_wind_at([0, 0, 1], thrust, 4, radius, [0, 0, 0.5])
    np.testing.assert_allclose(wind, [0, 0, -2 * expected_vh])
    assert np.all(downwash_wind_at([0, 0, 1], thrust, 4, radius, [1, 0, 0.5]) == 0)
    assert np.all(downwash_wind_at([0, 0, 1], thrust, 4, radius, [0, 0, 2]) == 0)


def test_downwash_interaction_sums_upper_vehicles():
    positions = {"upper_a": np.array([-0.02, 0, 1.0]), "upper_b": np.array([0.02, 0, 1.0]), "lower": np.array([0, 0, 0.5])}
    interaction = DownwashInteraction(thrust_provider=lambda _: 0.30 * 9.81)
    result = interaction(positions, {name: np.zeros(3) for name in positions})
    one = downwash_wind_at(positions["upper_a"], 0.30 * 9.81, 4, 0.023, positions["lower"])
    np.testing.assert_allclose(result["lower"].wind_velocity_mps, 2 * one)
    np.testing.assert_allclose(result["upper_a"].wind_velocity_mps, 0)
    # Restatement: this only confirms the summed wind feeds the drag formula
    # unchanged; the independent numbers are in test_downwash_hand_numbers.
    cd_area = 0.05 * 1.3
    expected_drag = 0.5 * 1.225 * cd_area * (2 * np.linalg.norm(one)) ** 2
    actual_drag_from_returned_wind = 0.5 * 1.225 * cd_area * np.linalg.norm(result["lower"].wind_velocity_mps) ** 2
    np.testing.assert_allclose(actual_drag_from_returned_wind, expected_drag)


def test_downwash_hand_numbers():
    # Hand arithmetic for a 30 g quad with four 23 mm rotors:
    # T = 0.030 x 9.81 = 0.2943 N; A = 4 x pi x 0.023^2 = 0.006648 m^2;
    # v_h = sqrt(0.2943 / (2 x 1.225 x 0.006648)) = sqrt(18.07) = 4.25 m/s;
    # far wake 2 v_h = 8.50 m/s straight down.
    wind = downwash_wind_at([0, 0, 1], 0.2943, 4, 0.023, [0, 0, 0.6])
    assert abs(wind[2] + 8.50) < 0.01 and wind[0] == 0 and wind[1] == 0
    # Equivalent disk radius sqrt(A/pi) = 2 x 0.023 = 0.046 m; contracted wake
    # radius 0.046 / sqrt(2) = 0.0325 m.  Just inside and just outside it:
    assert downwash_wind_at([0, 0, 1], 0.2943, 4, 0.023, [0.032, 0, 0.6])[2] < 0
    assert downwash_wind_at([0, 0, 1], 0.2943, 4, 0.023, [0.033, 0, 0.6])[2] == 0
    # Beyond the trusted depth (default 1.0 m) the model returns zero.
    assert downwash_wind_at([0, 0, 2.0], 0.2943, 4, 0.023, [0, 0, 0.99])[2] == 0
    assert downwash_wind_at([0, 0, 2.0], 0.2943, 4, 0.023, [0, 0, 1.01])[2] < 0
