"""Frame and quaternion conventions for talking to a real flight stack.

The simulator and most flight controllers describe the same physical motion
with different axis labels. Mixing them up gives a drone that flies the
mirror image of the plan, so every conversion lives here, in one place, with
tests that check it by hand.

The simulator's conventions (see ``sim/drone.py`` and ``sim/controller.py``):

* **World frame ENU.** x points east, y points north, z points up, in metres.
* **Body frame FLU.** x points out the nose (forward), y out the left side,
  z out the top (up). Thrust is ``[0, 0, T]`` in body axes.
* **Quaternion order w, x, y, z** (the scalar part first), Hamilton
  convention. The quaternion turns a body-frame vector into the world frame:
  ``world = R(q) @ body``.
* **Yaw** is the compass angle of the nose measured from east, positive
  turning toward north (counter-clockwise seen from above).

PX4 and MAVLink use (PX4 ``VehicleAttitude.msg``, MAVLink ``ATTITUDE_QUATERNION``):

* **World frame NED.** x north, y east, z down.
* **Body frame FRD.** x forward, y out the right side, z out the bottom.
* **Quaternion order w, x, y, z**, Hamilton, body FRD into world NED.
* **Yaw** measured from north, positive turning toward east (clockwise seen
  from above).

So the quaternion layout matches, but both frames differ. The functions below
only relabel axes. They do not move the origin and they do not line up a local
frame's "north" with true north; that alignment is a separate calibration.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "enu_to_ned",
    "ned_to_enu",
    "flu_to_frd",
    "frd_to_flu",
    "sim_quat_to_ned_frd",
    "ned_frd_to_sim_quat",
    "yaw_enu_to_ned",
    "yaw_ned_to_enu",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]

_HALF_SQRT2 = math.sqrt(0.5)
# Half a turn about the axis halfway between east and north. It swaps east and
# north and flips up into down, which is exactly ENU to NED (and back).
_Q_ENU_NED = np.array([0.0, _HALF_SQRT2, _HALF_SQRT2, 0.0])
# Half a turn about the forward axis. It flips left into right and up into
# down, which is exactly FLU to FRD (and back).
_Q_FLU_FRD = np.array([0.0, 1.0, 0.0, 0.0])


def _vectors(value, *, name: str = "vector") -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim not in (1, 2) or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def enu_to_ned(vector) -> np.ndarray:
    """Turn an east-north-up vector into north-east-down.

    Works for positions, velocities, and accelerations alike, one vector of
    shape (3,) or a stack of shape (N, 3). Each row ``[east, north, up]``
    becomes ``[north, east, -up]``: swap the first two numbers and flip the
    sign of the third. Example: 4 m east, 3 m north, 2 m up is
    ``[4, 3, 2]`` in ENU and ``[3, 4, -2]`` in NED.
    """
    v = _vectors(vector)
    return np.stack((v[..., 1], v[..., 0], -v[..., 2]), axis=-1)


def ned_to_enu(vector) -> np.ndarray:
    """Turn a north-east-down vector into east-north-up.

    The same swap as :func:`enu_to_ned`: doing it twice gives back the input.
    """
    return enu_to_ned(vector)


def flu_to_frd(vector) -> np.ndarray:
    """Turn a body forward-left-up vector into forward-right-down.

    ``[forward, left, up]`` becomes ``[forward, -left, -up]``. This also works
    for body angular rates ``[p, q, r]``: the roll rate keeps its sign and the
    pitch and yaw rates flip, because their axes point the other way.
    """
    v = _vectors(vector)
    return np.stack((v[..., 0], -v[..., 1], -v[..., 2]), axis=-1)


def frd_to_flu(vector) -> np.ndarray:
    """Turn a body forward-right-down vector into forward-left-up (same flip)."""
    return flu_to_frd(vector)


def _quaternions(value) -> np.ndarray:
    q = np.array(value, dtype=float)
    if q.ndim not in (1, 2) or q.shape[-1] != 4:
        raise ValueError("quaternion must have shape (4,) or (N, 4)")
    if not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be finite")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("quaternion must have a nonzero length")
    return q / norm


def _hamilton(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for w-first quaternions (broadcasts over rows).

    Rotating by ``a * b`` means: rotate by ``b`` first, then by ``a``.
    """
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=-1)


def _canonical(q: np.ndarray) -> np.ndarray:
    # q and -q describe the same attitude. Pick the one with w >= 0 so the
    # same attitude always prints the same numbers.
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def sim_quat_to_ned_frd(q_sim) -> np.ndarray:
    """Convert a sim attitude into the PX4 / MAVLink attitude.

    Input: w-first quaternion that turns body FLU vectors into world ENU (the
    sim's ``drone.quaternion``). Output: w-first quaternion that turns body
    FRD vectors into world NED. Shape (4,) or (N, 4); the input is normalised
    first and the output has w >= 0.

    In words, the PX4 attitude does three things in a row to a body vector
    written in FRD: relabel it into FLU, apply the sim attitude to get ENU,
    then relabel ENU into NED. Each relabel is a half turn, so the result is
    ``q_enu_ned * q_sim * q_flu_frd``. Example: a level drone facing north is
    ``(0.707, 0, 0, 0.707)`` in the sim (90 degrees from east) and the null
    rotation ``(1, 0, 0, 0)`` in PX4 (0 degrees from north).
    """
    q = _quaternions(q_sim)
    return _canonical(_hamilton(_hamilton(_Q_ENU_NED, q), _Q_FLU_FRD))


def ned_frd_to_sim_quat(q_ned_frd) -> np.ndarray:
    """Convert a PX4 / MAVLink attitude (FRD into NED) into the sim's (FLU into ENU).

    A half turn undone is the same half turn, so this is the same formula as
    :func:`sim_quat_to_ned_frd`, and the two functions undo each other.
    """
    q = _quaternions(q_ned_frd)
    return _canonical(_hamilton(_hamilton(_Q_ENU_NED, q), _Q_FLU_FRD))


def _wrap_pi(angle: float) -> float:
    """Wrap an angle in radians into (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


def yaw_enu_to_ned(yaw_enu_rad: float) -> float:
    """Sim yaw (from east, toward north) to NED yaw (from north, toward east).

    ``yaw_ned = 90 deg - yaw_enu``, wrapped into (-180, 180] degrees (radians in
    and out). Facing east: 0 becomes 90 degrees. Facing north: 90 becomes 0.
    """
    value = float(yaw_enu_rad)
    if not math.isfinite(value):
        raise ValueError("yaw must be finite")
    return _wrap_pi(math.pi / 2.0 - value)


def yaw_ned_to_enu(yaw_ned_rad: float) -> float:
    """NED yaw to sim yaw. The same formula, so it undoes :func:`yaw_enu_to_ned`."""
    return yaw_enu_to_ned(yaw_ned_rad)


def _four(value) -> np.ndarray:
    q = np.array(value, dtype=float)
    if q.ndim not in (1, 2) or q.shape[-1] != 4:
        raise ValueError("quaternion must have shape (4,) or (N, 4)")
    return q


def wxyz_to_xyzw(q) -> np.ndarray:
    """Move w from the front to the back (for libraries that store x, y, z, w).

    This only reorders the numbers. It does not change which frames the
    quaternion connects.
    """
    q = _four(q)
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def xyzw_to_wxyz(q) -> np.ndarray:
    """Move w from the back to the front (the order the sim and PX4 use)."""
    q = _four(q)
    return np.concatenate((q[..., 3:], q[..., :3]), axis=-1)
