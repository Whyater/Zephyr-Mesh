"""Independent geometry checks for the simulated cooperative avoidance rules."""

import numpy as np
import pytest

from sim.coop.avoidance import AvoidanceConfig, closest_approach, extrapolate, orca_halfspaces, safe_velocity


def _capped_goal_velocity(position, goal, maximum):
    delta = np.asarray(goal, dtype=float) - np.asarray(position, dtype=float)
    length = np.linalg.norm(delta)
    if length <= maximum:
        return delta
    return delta * (maximum / length)


def _rollout(starts, goals, config, duration=10.0, dt=0.05):
    positions = np.asarray(starts, dtype=float).copy()
    goals = np.asarray(goals, dtype=float)
    velocities = np.zeros_like(positions)
    history = [positions.copy()]
    for _ in range(int(round(duration / dt))):
        chosen = []
        for i in range(len(positions)):
            others = [(positions[j].copy(), velocities[j].copy()) for j in range(len(positions)) if j != i]
            preferred = _capped_goal_velocity(positions[i], goals[i], config.max_speed_mps)
            velocity, _ = safe_velocity(positions[i], velocities[i], preferred, others, config)
            chosen.append(velocity)
        velocities = np.asarray(chosen, dtype=float)
        positions = positions + dt * velocities
        history.append(positions.copy())
    return np.asarray(history), positions


def _minimum_pair_distance(history):
    values = []
    for state in history:
        for i in range(len(state)):
            for j in range(i):
                values.append(np.linalg.norm(state[i] - state[j]))
    return min(values)


def _rotate_z(points, angle):
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(points) @ matrix.T


def test_closest_approach_and_extrapolate_hand_values():
    t_star, distance = closest_approach(np.array([4.0, 0.0, 0.0]), np.array([-1.0, 0.5, 0.0]))
    assert t_star == pytest.approx(4.0 / 1.25, abs=1e-9)
    assert distance == pytest.approx(2.0 / np.sqrt(1.25), abs=1e-9)
    np.testing.assert_allclose(extrapolate([1.0, -2.0, 0.5], [0.5, 2.0, -1.0], 0.4), [1.2, -1.2, 0.1])


def test_no_neighbor_returns_the_preferred_velocity_after_speed_cap():
    config = AvoidanceConfig(max_speed_mps=2.0)
    preferred = np.array([3.0, 4.0, 0.0])
    result, flags = safe_velocity(np.zeros(3), np.zeros(3), preferred, [], config)
    np.testing.assert_allclose(result, [1.2, 1.6, 0.0], atol=0.0, rtol=0.0)
    assert np.linalg.norm(result) <= config.max_speed_mps + 1e-9
    assert flags == frozenset({"speed_cap"})

    far = [(np.array([100.0, 0.0, 0.0]), np.zeros(3))]
    result_far, flags_far = safe_velocity(np.zeros(3), np.zeros(3), np.array([0.2, 0.3, 0.1]), far, config)
    np.testing.assert_allclose(result_far, [0.2, 0.3, 0.1], atol=0.0, rtol=0.0)
    assert flags_far == frozenset()


def test_emergency_separation_uses_nearest_neighbor_and_right_side_step():
    config = AvoidanceConfig(max_speed_mps=2.0, emergency_distance_m=0.5, sidestep_right=True)
    neighbors = [(np.array([0.2, 0.0, 0.0]), np.zeros(3)), (np.array([0.0, 0.3, 0.0]), np.zeros(3))]
    result, flags = safe_velocity(np.zeros(3), np.zeros(3), np.zeros(3), neighbors, config)
    expected = 2.0 * np.array([-0.8, -0.6, 0.0]) / np.linalg.norm([-0.8, -0.6, 0.0])
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert flags == frozenset({"emergency"})


def test_overlap_rule_separates_stationary_agents_and_uses_configured_margin():
    config = AvoidanceConfig(radius_m=0.35, emergency_distance_m=0.0)
    own = np.array([0.0, 0.0, 1.0])
    other = np.array([0.30, 0.0, 1.0])
    velocity, flags = safe_velocity(own, np.zeros(3), np.zeros(3), [(other, np.zeros(3))], config)
    assert velocity[0] < 0.0
    assert flags == frozenset({"emergency"}) or "orca" in flags
    margin_config = AvoidanceConfig(radius_m=0.35, safety_margin_m=0.20, emergency_distance_m=0.0)
    own = np.array([0.0, 0.0, 1.0])
    neighbor = np.array([0.60, 0.0, 1.0])
    spaces = orca_halfspaces(own, np.zeros(3), [(neighbor, np.zeros(3))], margin_config)
    assert spaces
    no_margin_point, no_margin_normal = orca_halfspaces(own, np.zeros(3), [(neighbor, np.zeros(3))], config)[0]
    margin_point, margin_normal = spaces[0]
    np.testing.assert_allclose(margin_normal, no_margin_normal, atol=1e-12)
    assert not np.allclose(margin_point, no_margin_point)
    no_margin_velocity, _ = safe_velocity(own, np.zeros(3), np.array([1.0, 0.0, 0.0]), [(neighbor, np.zeros(3))], config)
    margin_velocity, _ = safe_velocity(own, np.zeros(3), np.array([1.0, 0.0, 0.0]), [(neighbor, np.zeros(3))], margin_config)
    assert not np.allclose(margin_velocity, no_margin_velocity)


def test_reach_accounts_for_both_maximum_speeds():
    config = AvoidanceConfig(max_speed_mps=2.0, time_horizon_s=2.0, emergency_distance_m=0.0)
    neighbor = (np.array([7.0, 0.0, 1.0]), np.zeros(3))
    assert orca_halfspaces(np.array([0.0, 0.0, 1.0]), np.zeros(3), [neighbor], config)


def test_regression_velocity_pair_is_not_permitted_by_both_halfspaces():
    config = AvoidanceConfig(radius_m=0.35, time_horizon_s=2.0)
    p_a = np.array([0.0, 0.0, 1.0])
    v_a = np.array([0.2, 1.3, 0.0])
    p_b = np.array([0.7, 0.0, 1.0])
    v_b = np.zeros(3)
    candidate_a = np.array([1.57, 0.892, 0.0])
    candidate_b = np.array([-1.239, 0.403, 0.0])
    pa, na = orca_halfspaces(p_a, v_a, [(p_b, v_b)], config)[0]
    pb, nb = orca_halfspaces(p_b, v_b, [(p_a, v_a)], config)[0]
    assert np.dot(candidate_a - pa, na) < 0.0 or np.dot(candidate_b - pb, nb) < 0.0


def test_exact_head_on_breaks_tie_to_each_observers_right():
    config = AvoidanceConfig()
    left, _ = safe_velocity(
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        [(np.array([1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]))],
        config,
    )
    right, _ = safe_velocity(
        np.array([1.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        [(np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))],
        config,
    )
    assert left[1] < -1e-9
    assert right[1] > 1e-9
    assert np.linalg.norm(left) <= config.max_speed_mps + 1e-9
    assert np.linalg.norm(right) <= config.max_speed_mps + 1e-9


def test_head_on_perfect_information_keeps_separation_and_reaches_goals():
    config = AvoidanceConfig()
    starts = np.array([[-3.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    goals = starts[::-1].copy()
    history, final = _rollout(starts, goals, config, duration=12.0)
    assert _minimum_pair_distance(history) >= config.radius_m - 1e-9
    assert np.linalg.norm(final[0] - goals[0]) <= 0.2
    assert np.linalg.norm(final[1] - goals[1]) <= 0.2


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_head_on_speed_variants_reach_swapped_goals(speed):
    config = AvoidanceConfig(max_speed_mps=speed)
    starts = np.array([[-3.0, 0.01, 1.0], [3.0, 0.0, 1.0]])
    goals = starts[::-1].copy()
    history, final = _rollout(starts, goals, config, duration=14.0)
    distances = np.linalg.norm(history[:, 1] - history[:, 0], axis=1)
    assert distances.min() >= config.radius_m - 1e-9
    assert np.linalg.norm(final[0] - goals[0]) <= 0.2
    assert np.linalg.norm(final[1] - goals[1]) <= 0.2
    assert np.max(np.linalg.norm(np.diff(history, axis=0) / 0.05, axis=2)) <= speed + 1e-9


def test_four_meter_rest_swap_reaches_both_goals():
    config = AvoidanceConfig()
    starts = np.array([[-2.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    goals = starts[::-1].copy()
    history, final = _rollout(starts, goals, config, duration=8.0)
    assert _minimum_pair_distance(history) >= config.radius_m - 1e-9
    assert np.linalg.norm(final[0] - goals[0]) <= 0.2
    assert np.linalg.norm(final[1] - goals[1]) <= 0.2


@pytest.mark.parametrize(
    "starts,goals",
    [
        (np.array([[-2.0, 0.01, 0.0], [2.0, 0.0, 0.0]]), np.array([[2.0, 0.01, 0.0], [-2.0, 0.0, 0.0]])),
        (np.array([[-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]), np.array([[0.0, -2.0, 0.0], [-2.0, 0.0, 0.0]])),
    ],
)
def test_offset_head_on_and_right_angle_crossing_remain_separated(starts, goals):
    history, _ = _rollout(starts, goals, AvoidanceConfig(), duration=10.0)
    assert _minimum_pair_distance(history) >= AvoidanceConfig().radius_m - 1e-9


def test_true_ninety_degree_crossing_reaches_goals():
    starts = np.array([[-4.0, 0.0, 1.0], [4.0, 0.0, 1.0], [0.0, -4.0, 1.0], [0.0, 4.0, 1.0]])
    goals = np.array([[4.0, 0.0, 1.0], [-4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, -4.0, 1.0]])
    history, final = _rollout(starts, goals, AvoidanceConfig(), duration=12.0)
    assert _minimum_pair_distance(history) >= AvoidanceConfig().radius_m - 1e-9
    assert np.max(np.linalg.norm(final - goals, axis=1)) <= 0.2


def test_square_and_circle_swaps_remain_separated():
    square = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]])
    square_history, _ = _rollout(square, square[[2, 3, 0, 1]], AvoidanceConfig(), duration=10.0)
    assert _minimum_pair_distance(square_history) >= AvoidanceConfig().radius_m - 1e-9

    angles = np.arange(6, dtype=float) * (2.0 * np.pi / 6.0)
    circle = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles), np.zeros(6)))
    circle_history, _ = _rollout(circle, circle[(np.arange(6) + 3) % 6], AvoidanceConfig(), duration=10.0)
    assert _minimum_pair_distance(circle_history) >= AvoidanceConfig().radius_m - 1e-9


def test_infeasible_epigraph_fallback_is_exercised():
    config = AvoidanceConfig(max_speed_mps=0.2, emergency_distance_m=0.0)
    # Four neighbors closing in from all sides faster than this observer can move.
    neighbors = [
        (np.array([0.4, 0.0, 0.0]), np.array([-0.5, 0.0, 0.0])),
        (np.array([-0.4, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])),
        (np.array([0.0, 0.4, 0.0]), np.array([0.0, -0.5, 0.0])),
        (np.array([0.0, -0.4, 0.0]), np.array([0.0, 0.5, 0.0])),
    ]
    result, flags = safe_velocity(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), neighbors, config)
    assert "infeasible" in flags
    assert np.linalg.norm(result) <= config.max_speed_mps + 1e-9


def test_random_nearest_boundary_reference_matches_orca_correction():
    """Compare against an independent sampled closest safe velocity."""
    rng = np.random.default_rng(np.random.SeedSequence(20261015))
    config = AvoidanceConfig(radius_m=0.35, time_horizon_s=2.0, responsibility=1.0)
    errors = []
    own = np.array([0.0, 0.0, 1.0])
    for _ in range(8):
        distance = float(rng.uniform(1.5, 2.5))
        angle = float(rng.uniform(-1.0, 1.0))
        position = np.array([distance * np.cos(angle), distance * np.sin(angle), 1.0])
        relative_position = position - own
        axis = relative_position / np.linalg.norm(relative_position)
        side = np.array([-axis[1], axis[0], 0.0])
        relative = 1.2 * axis + float(rng.uniform(-0.12, 0.12)) * side
        spaces = orca_halfspaces(own, relative, [(position, np.zeros(3))], config)
        correction = spaces[0][0] - relative

        # The independent reference checks the closed-form closest approach
        # on a dense velocity grid in the plane of the encounter.
        values = np.linspace(-0.8, 0.8, 161)
        best = None
        for first in values:
            for second in values:
                candidate = relative + first * axis + second * side
                t_star = np.dot(relative_position, candidate) / np.dot(candidate, candidate)
                t_star = float(np.clip(t_star, 0.0, config.time_horizon_s))
                distance_at_closest = np.linalg.norm(relative_position - candidate * t_star)
                if distance_at_closest >= config.radius_m - 1e-10:
                    displacement = np.linalg.norm(candidate - relative)
                    if best is None or displacement < best[0]:
                        best = (displacement, candidate)
        assert best is not None
        errors.append(np.linalg.norm(relative + correction - best[1]))
    assert max(errors) < 0.04


def test_pair_symmetry_under_relabel_rotation_and_offset_mirror():
    config = AvoidanceConfig()
    starts = np.array([[-2.0, 0.3, 1.0], [2.0, 0.0, 1.0]])
    goals = starts[::-1].copy()
    paths, _ = _rollout(starts, goals, config, duration=6.0)
    relabeled, _ = _rollout(starts[::-1], goals[::-1], config, duration=6.0)
    np.testing.assert_allclose(paths[:, 0], relabeled[:, 1], atol=1e-6)
    np.testing.assert_allclose(paths[:, 1], relabeled[:, 0], atol=1e-6)

    angle = np.deg2rad(37.0)
    rotated, _ = _rollout(_rotate_z(starts, angle), _rotate_z(goals, angle), config, duration=6.0)
    np.testing.assert_allclose(rotated, _rotate_z(paths, angle), atol=1e-6)

    mirrored_starts = starts.copy()
    mirrored_starts[:, 1] *= -1.0
    mirrored_goals = goals.copy()
    mirrored_goals[:, 1] *= -1.0
    mirrored, _ = _rollout(mirrored_starts, mirrored_goals, config, duration=6.0)
    expected = paths.copy()
    expected[:, :, 1] *= -1.0
    # Here each goal is the other agent's start, so the first step is an exact
    # head-on tie and the right-hand nudge makes the pair chiral by design.
    mirror_error = float(np.max(np.abs(mirrored - expected)))
    assert mirror_error > 1e-6


def test_mirrored_setup_without_an_exact_tie_gives_the_mirrored_path():
    config = AvoidanceConfig()
    starts = np.array([[-2.0, 0.3, 1.0], [2.0, 0.0, 1.0], [0.1, -2.0, 1.0]])
    goals = np.array([[2.0, -0.2, 1.0], [-2.0, 0.4, 1.0], [-0.3, 2.0, 1.0]])
    mirror = np.array([1.0, -1.0, 1.0])
    paths, _ = _rollout(starts, goals, config, duration=6.0)
    mirrored, _ = _rollout(starts * mirror, goals * mirror, config, duration=6.0)
    assert _minimum_pair_distance(paths) >= config.radius_m - 1e-9
    np.testing.assert_allclose(mirrored, paths * mirror, atol=1e-9)


def test_ellipsoid_vertical_stack_uses_scaled_separation():
    radii = (0.12, 0.12, 0.30)
    config = AvoidanceConfig(ellipsoid_radii=radii)
    starts = np.array([[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]])
    history, _ = _rollout(starts, starts[::-1], config, duration=10.0)
    scaled = history[:, 1] - history[:, 0]
    scaled = scaled / np.asarray(radii)
    assert np.min(np.linalg.norm(scaled, axis=1)) >= 2.0 - 1e-6


def test_halfspaces_have_boundary_points_and_reciprocal_shift_is_safe():
    config = AvoidanceConfig(radius_m=0.35, time_horizon_s=2.0, responsibility=0.5)
    own_pos = np.array([-2.0, 0.2, 0.0])
    own_vel = np.array([1.5, 0.0, 0.0])
    neighbor_pos = np.array([2.0, 0.0, 0.0])
    neighbor_vel = np.array([-1.5, 0.0, 0.0])
    own_constraints = orca_halfspaces(own_pos, own_vel, [(neighbor_pos, neighbor_vel)], config)
    other_constraints = orca_halfspaces(neighbor_pos, neighbor_vel, [(own_pos, own_vel)], config)
    assert len(own_constraints) == 1
    assert len(other_constraints) == 1
    point, normal = own_constraints[0]
    other_point, other_normal = other_constraints[0]
    assert np.isfinite(point).all() and np.isfinite(normal).all()
    u = (point - own_vel) / config.responsibility
    other_u = (other_point - neighbor_vel) / config.responsibility
    np.testing.assert_allclose(np.dot(own_vel + config.responsibility * u - point, normal), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(neighbor_vel + config.responsibility * other_u - other_point, other_normal), 0.0, atol=1e-12)

    relative_position = neighbor_pos - own_pos
    relative_velocity = own_vel - neighbor_vel
    boundary_velocity = relative_velocity + u
    boundary_time, boundary_distance = closest_approach(relative_position, -boundary_velocity)
    if boundary_time <= config.time_horizon_s:
        np.testing.assert_allclose(boundary_distance, config.radius_m, atol=1e-9)
    else:
        np.testing.assert_allclose(
            np.linalg.norm(boundary_velocity - relative_position / config.time_horizon_s),
            config.radius_m / config.time_horizon_s,
            atol=1e-9,
        )

    cap_own = np.array([0.0, 0.0, 1.0])
    cap_neighbor = np.array([4.0, 0.0, 1.0])
    cap_velocity = np.array([0.5, 0.05, 0.0])
    cap_point, _ = orca_halfspaces(cap_own, cap_velocity, [(cap_neighbor, np.zeros(3))], config)[0]
    cap_u = (cap_point - cap_velocity) / config.responsibility
    np.testing.assert_allclose(
        np.linalg.norm(cap_velocity + cap_u - (cap_neighbor - cap_own) / config.time_horizon_s),
        config.radius_m / config.time_horizon_s,
        atol=1e-9,
    )

    shifted_own = own_vel + config.responsibility * u
    shifted_neighbor = neighbor_vel + config.responsibility * other_u
    t_star, minimum = closest_approach(neighbor_pos - own_pos, -(shifted_own - shifted_neighbor))
    if t_star <= config.time_horizon_s:
        assert minimum >= config.radius_m - 1e-6
    else:
        np.testing.assert_allclose(
            np.linalg.norm((shifted_own - shifted_neighbor) - (neighbor_pos - own_pos) / config.time_horizon_s),
            config.radius_m / config.time_horizon_s,
            atol=1e-9,
        )


def test_active_set_solver_matches_slsqp_on_random_encounters():
    """The fast exact solver must pick the same velocity as the general optimizer."""
    import numpy as np
    from sim.coop.avoidance import AvoidanceConfig, safe_velocity

    rng = np.random.default_rng(np.random.SeedSequence(20261015))
    config = AvoidanceConfig()
    worst = 0.0
    for _ in range(300):
        own = np.array([0.0, 0.0, 1.5])
        own_vel = rng.uniform(-1.5, 1.5, size=3) * np.array([1.0, 1.0, 0.3])
        preferred = rng.uniform(-2.0, 2.0, size=3) * np.array([1.0, 1.0, 0.3])
        neighbors = []
        for _ in range(int(rng.integers(1, 7))):
            direction = rng.normal(size=3)
            direction[2] *= 0.3
            direction /= np.linalg.norm(direction)
            position = own + direction * rng.uniform(0.5, 4.0)
            velocity = rng.uniform(-1.5, 1.5, size=3) * np.array([1.0, 1.0, 0.3])
            neighbors.append((position, velocity))
        fast, fast_flags = safe_velocity(own, own_vel, preferred, neighbors, config, solver="active_set")
        slow, slow_flags = safe_velocity(own, own_vel, preferred, neighbors, config, solver="slsqp")
        if "infeasible" in fast_flags or "infeasible" in slow_flags or "emergency" in fast_flags:
            continue
        # Same optimum. SLSQP meets constraints only to about 1e-8, so it can sit a hair inside the
        # forbidden side and score a few 1e-9 "better"; allow 1e-6 on the objective.
        assert np.linalg.norm(fast - preferred) <= np.linalg.norm(slow - preferred) + 1e-6
        worst = max(worst, float(np.linalg.norm(fast - slow)))
    assert worst < 1e-5


def _truncated_vo_samples(rel_pos, radius, horizon, rng, count=4000):
    """Sample relative velocities inside the truncated velocity obstacle.

    The obstacle is the union over t in (0, horizon] of balls centred at
    rel_pos / t with radius radius / t: any such relative velocity brings the
    pair closer than radius within the horizon.
    """
    times = rng.uniform(0.05 * horizon, horizon, size=count)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    fractions = rng.uniform(0.0, 0.999, size=count) ** (1.0 / 3.0)
    return rel_pos / times[:, None] + (radius / times)[:, None] * fractions[:, None] * directions


@pytest.mark.parametrize("case", range(40))
def test_full_responsibility_halfspace_supports_the_obstacle(case):
    """With responsibility 1 the plane must leave the whole obstacle on the forbidden side,
    including the tie cases (zero or purely axial relative velocity)."""
    rng = np.random.default_rng(np.random.SeedSequence([20261015, case]))
    config = AvoidanceConfig(radius_m=0.35, time_horizon_s=2.0, responsibility=1.0)
    distance = rng.uniform(0.5, 4.0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rel_pos = distance * axis
    if case % 4 == 0:
        rel_vel = np.zeros(3)
    elif case % 4 == 1:
        rel_vel = rng.uniform(-1.5, 1.5) * axis
    else:
        rel_vel = rng.uniform(-1.5, 1.5, size=3)
    spaces = orca_halfspaces(np.zeros(3), rel_vel, [(rel_pos, np.zeros(3))], config)
    assert len(spaces) == 1
    point, normal = spaces[0]
    samples = _truncated_vo_samples(rel_pos, config.radius_m, config.time_horizon_s, rng)
    assert np.max((samples - point) @ normal) <= 1e-9


def test_rest_pair_one_meter_apart_cannot_choose_a_colliding_pair():
    config = AvoidanceConfig(radius_m=0.35, time_horizon_s=2.0)
    p_a = np.array([0.0, 0.0, 1.0])
    p_b = np.array([1.0, 0.0, 1.0])
    v_a = np.array([0.5, -0.165, 0.0])
    v_b = np.array([-0.5, 0.165, 0.0])
    _, miss = closest_approach(p_b - p_a, v_b - v_a)
    assert miss < config.radius_m
    pa, na = orca_halfspaces(p_a, np.zeros(3), [(p_b, np.zeros(3))], config)[0]
    pb, nb = orca_halfspaces(p_b, np.zeros(3), [(p_a, np.zeros(3))], config)[0]
    assert np.dot(v_a - pa, na) < 0.0 or np.dot(v_b - pb, nb) < 0.0


def test_tie_break_changes_only_the_preference_and_flags_it():
    config = AvoidanceConfig()
    own, other = np.array([-0.5, 0.0, 1.0]), np.array([0.5, 0.0, 1.0])
    preferred = np.array([2.0, 0.0, 0.0])
    velocity, flags = safe_velocity(own, np.zeros(3), preferred, [(other, np.zeros(3))], config)
    assert "tie_break" in flags
    assert velocity[1] < -1e-6
    point, normal = orca_halfspaces(own, np.zeros(3), [(other, np.zeros(3))], config)[0]
    assert np.dot(velocity - point, normal) >= -1e-9
    off, off_flags = safe_velocity(
        own, np.zeros(3), preferred, [(other, np.zeros(3))], AvoidanceConfig(sidestep_right=False)
    )
    assert "tie_break" not in off_flags
    assert abs(off[1]) <= 1e-12


def test_holding_agent_next_to_hovering_neighbor_stays_still():
    config = AvoidanceConfig()
    starts = np.array([[0.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
    history, final = _rollout(starts, starts, config, duration=5.0)
    assert len(history) == 101
    np.testing.assert_allclose(final, starts, atol=1e-12)
