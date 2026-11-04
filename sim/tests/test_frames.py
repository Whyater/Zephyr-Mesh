"""Checks for the ENU/NED and FLU/FRD conversions in ``sim.frames``.

The expected values here are worked out by hand (written next to each test)
or come from the simulator's own body-to-world rotation in ``sim.vehicle``,
not from the functions under test.
"""
import math

import numpy as np
import pytest

from sim import frames
from sim.vehicle import ActuatedQuadrotor

S = math.sqrt(0.5)

# Hand-written axis swaps, kept separate from the module's own constants.
# NED = (north, east, down) = (enu_y, enu_x, -enu_z).
ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# FRD = (forward, right, down) = (flu_x, -flu_y, -flu_z).
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def sim_rotation(q):
    """Body-to-world matrix exactly as the simulator builds it."""
    return ActuatedQuadrotor._rotation_from_quaternion(np.asarray(q, dtype=float))


def same_rotation(q_a, q_b, tol=1e-12):
    """q and -q are the same rotation, so compare up to an overall sign."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    return min(np.max(np.abs(q_a - q_b)), np.max(np.abs(q_a + q_b))) < tol


def random_unit_quaternions(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ---------- positions and velocities ----------

def test_hand_point_north3_east4_up2():
    # North 3 m, east 4 m, up 2 m. ENU lists east first: (4, 3, 2).
    # NED lists north first and measures down: (3, 4, -2).
    enu = np.array([4.0, 3.0, 2.0])
    ned = np.array([3.0, 4.0, -2.0])
    assert np.array_equal(frames.enu_to_ned(enu), ned)
    assert np.array_equal(frames.ned_to_enu(ned), enu)


def test_velocity_uses_the_same_swap():
    # Climbing at 1 m/s while flying north at 2 m/s.
    enu_velocity = np.array([0.0, 2.0, 1.0])
    assert np.array_equal(frames.enu_to_ned(enu_velocity), [2.0, 0.0, -1.0])


def test_vector_conversions_match_hand_matrix_on_a_batch():
    vectors = np.random.default_rng(1).normal(size=(50, 3))
    assert np.allclose(frames.enu_to_ned(vectors), vectors @ ENU_TO_NED.T, atol=0, rtol=0)
    assert np.allclose(frames.flu_to_frd(vectors), vectors @ FLU_TO_FRD.T, atol=0, rtol=0)


def test_vector_round_trips_are_identity():
    vectors = np.random.default_rng(2).normal(scale=100.0, size=(200, 3))
    assert np.max(np.abs(frames.ned_to_enu(frames.enu_to_ned(vectors)) - vectors)) <= 1e-12
    assert np.max(np.abs(frames.frd_to_flu(frames.flu_to_frd(vectors)) - vectors)) <= 1e-12


def test_body_vector_flu_to_frd_by_hand():
    # 1 forward, 2 left, 3 up is 1 forward, -2 right, -3 down.
    assert np.array_equal(frames.flu_to_frd([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])
    # Body rates follow the same rule: roll rate keeps its sign, pitch and yaw rates flip.
    assert np.array_equal(frames.flu_to_frd([0.1, 0.2, 0.3]), [0.1, -0.2, -0.3])


def test_inputs_are_not_modified():
    enu = np.array([4.0, 3.0, 2.0])
    frames.enu_to_ned(enu)
    assert np.array_equal(enu, [4.0, 3.0, 2.0])


@pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, float("nan"), 0.0], [[1.0, 2.0]]])
def test_vector_inputs_are_validated(bad):
    with pytest.raises(ValueError):
        frames.enu_to_ned(bad)


# ---------- quaternions ----------

def test_hand_quaternion_for_facing_north_points_body_x_north_in_the_sim():
    # Sanity check of the hand value used below: a 90 degree turn about +z (up)
    # is (cos 45, 0, 0, sin 45) and should send body x (forward) to world y (north).
    q_north = np.array([S, 0.0, 0.0, S])
    assert np.allclose(sim_rotation(q_north) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)


def test_ninety_degree_yaw_by_hand():
    # Facing north, level. In PX4 (NED world, FRD body), yaw 0 means facing north,
    # so the attitude is the null rotation (1, 0, 0, 0).
    q_north_sim = [S, 0.0, 0.0, S]
    assert same_rotation(frames.sim_quat_to_ned_frd(q_north_sim), [1.0, 0.0, 0.0, 0.0])
    # Facing east, level: the sim null rotation. In NED that is a +90 degree turn
    # about +z (down), which turns north toward east: (cos 45, 0, 0, sin 45).
    q_east_ned = frames.sim_quat_to_ned_frd([1.0, 0.0, 0.0, 0.0])
    assert same_rotation(q_east_ned, [S, 0.0, 0.0, S])
    # And back.
    assert same_rotation(frames.ned_frd_to_sim_quat([1.0, 0.0, 0.0, 0.0]), q_north_sim)


def test_converted_quaternion_matches_hand_matrix_product():
    # R_ned_frd = ENU_TO_NED @ R_enu_flu @ FRD_TO_FLU, with FRD_TO_FLU = FLU_TO_FRD.
    # The same w-first Hamilton matrix formula applies to the PX4 quaternion, so
    # sim_rotation() can build both matrices.
    for q in random_unit_quaternions(200, seed=3):
        expected = ENU_TO_NED @ sim_rotation(q) @ FLU_TO_FRD
        assert np.max(np.abs(sim_rotation(frames.sim_quat_to_ned_frd(q)) - expected)) < 1e-12


def test_world_direction_of_a_body_vector_agrees_in_both_conventions():
    # The physical check that catches sign bugs: rotate a body vector into the
    # world with each convention, then compare after converting the world frame.
    rng = np.random.default_rng(4)
    for q in random_unit_quaternions(200, seed=5):
        v_flu = rng.normal(size=3)
        world_enu = sim_rotation(q) @ v_flu
        world_ned = sim_rotation(frames.sim_quat_to_ned_frd(q)) @ frames.flu_to_frd(v_flu)
        assert np.max(np.abs(frames.enu_to_ned(world_enu) - world_ned)) < 1e-12


def test_thrust_axis_points_up_for_a_level_drone_in_both_frames():
    # Body up (FLU z) is body down negated (FRD z). For a level drone facing any way,
    # thrust along body up must point world up, which is NED (0, 0, -1).
    for yaw in np.linspace(-math.pi, math.pi, 13):
        q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        q_ned = frames.sim_quat_to_ned_frd(q)
        assert np.allclose(sim_rotation(q_ned) @ [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], atol=1e-12)


def test_quaternion_round_trip_is_identity_and_sign_is_canonical():
    qs = random_unit_quaternions(500, seed=6)
    for q in qs:
        back = frames.ned_frd_to_sim_quat(frames.sim_quat_to_ned_frd(q))
        assert same_rotation(back, q)
        # Outputs pick the sign with w >= 0 so logs and comparisons are repeatable.
        assert back[0] >= 0.0
        assert abs(np.linalg.norm(back) - 1.0) < 1e-12


def test_batch_quaternions_match_one_at_a_time():
    qs = random_unit_quaternions(20, seed=7)
    batch = frames.sim_quat_to_ned_frd(qs)
    assert batch.shape == (20, 4)
    for row, q in zip(batch, qs):
        assert np.array_equal(row, frames.sim_quat_to_ned_frd(q))


def test_component_order_helpers_by_hand():
    wxyz = np.array([0.9, 0.1, 0.2, 0.3])
    assert np.array_equal(frames.wxyz_to_xyzw(wxyz), [0.1, 0.2, 0.3, 0.9])
    assert np.array_equal(frames.xyzw_to_wxyz([0.1, 0.2, 0.3, 0.9]), wxyz)
    batch = np.random.default_rng(8).normal(size=(10, 4))
    assert np.array_equal(frames.xyzw_to_wxyz(frames.wxyz_to_xyzw(batch)), batch)


@pytest.mark.parametrize("bad", [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, float("inf"), 0.0, 0.0]])
def test_quaternion_inputs_are_validated(bad):
    with pytest.raises(ValueError):
        frames.sim_quat_to_ned_frd(bad)


def test_non_unit_quaternion_is_normalized():
    assert same_rotation(frames.sim_quat_to_ned_frd([2.0, 0.0, 0.0, 0.0]), [S, 0.0, 0.0, S])


# ---------- yaw angles ----------

@pytest.mark.parametrize(
    ("yaw_enu_deg", "yaw_ned_deg"),
    [(0.0, 90.0), (90.0, 0.0), (180.0, -90.0), (-90.0, 180.0), (45.0, 45.0), (30.0, 60.0)],
)
def test_yaw_by_hand(yaw_enu_deg, yaw_ned_deg):
    # Sim yaw: from east, counter-clockwise seen from above (east 0, north 90).
    # NED yaw: from north, clockwise seen from above (north 0, east 90).
    # So yaw_ned = 90 deg - yaw_enu, wrapped into (-180, 180].
    got = math.degrees(frames.yaw_enu_to_ned(math.radians(yaw_enu_deg)))
    assert got == pytest.approx(yaw_ned_deg, abs=1e-12)
    back = math.degrees(frames.yaw_ned_to_enu(math.radians(yaw_ned_deg)))
    wrapped = (yaw_enu_deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        wrapped = 180.0
    assert back == pytest.approx(wrapped, abs=1e-12)


def test_yaw_helper_agrees_with_quaternion_conversion():
    def yaw_of(q):
        qw, qx, qy, qz = q
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    for yaw in np.linspace(-3.0, 3.0, 25):
        q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        assert yaw_of(frames.sim_quat_to_ned_frd(q)) == pytest.approx(frames.yaw_enu_to_ned(yaw), abs=1e-12)


# ---------- units at the adapter boundary ----------

def test_link_metrics_from_si_by_hand_and_into_an_adapter():
    from sim.adapters import SimAdapter, link_metrics_from_si

    # 0.012 s is 12 ms; a 0.05 loss fraction is 5 percent.
    kwargs = link_metrics_from_si(delay_s=0.012, loss_fraction=0.05, age_s=0.2)
    assert kwargs["delay_ms"] == pytest.approx(12.0, abs=1e-12)
    assert kwargs["loss_pct"] == pytest.approx(5.0, abs=1e-12)
    assert kwargs["age_s"] == 0.2
    adapter = SimAdapter("unit-check")
    adapter.connect()
    adapter.set_link_metrics(**kwargs)
    state = adapter.get_state()
    assert state.link_delay_ms == pytest.approx(12.0, abs=1e-12)
    assert state.link_loss_pct == pytest.approx(5.0, abs=1e-12)
    # A fraction above 1 turns into a percent above 100, which the adapter refuses.
    with pytest.raises(ValueError):
        adapter.set_link_metrics(**link_metrics_from_si(delay_s=0.01, loss_fraction=1.5, age_s=None))
