"""Reference 6-DOF rigid-body dynamics for the simulator."""
from __future__ import annotations

import numpy as np
from sim.environment import GRAVITY, calculate_drag, relative_air_velocity


class SixDOFInterceptor:
    def __init__(self, init_pos=(0.0, 0.0, 0.0), mass=0.450, *,
                 drag_coefficient=1.3, cross_sectional_area=0.05,
                 wind_velocity=(0.0, 0.0, 0.0)):
        """Create a 13-state rigid body using SI units.

        ``wind_velocity`` is inertial-frame air velocity in m/s. Drag uses
        relative air velocity ``drone_velocity - wind_velocity``. Coefficients
        are transparent scenario parameters, not measured drone coefficients.
        """
        self.mass = float(mass)
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")
        self.I_x, self.I_y, self.I_z = 0.0023, 0.0023, 0.0040
        self.inertia_matrix = np.diag([self.I_x, self.I_y, self.I_z])
        self.inv_inertia = np.linalg.inv(self.inertia_matrix)
        self.drag_coefficient = float(drag_coefficient)
        self.cross_sectional_area = float(cross_sectional_area)
        if not np.isfinite(self.drag_coefficient) or self.drag_coefficient < 0:
            raise ValueError("drag_coefficient must be finite and nonnegative")
        if not np.isfinite(self.cross_sectional_area) or self.cross_sectional_area < 0:
            raise ValueError("cross_sectional_area must be finite and nonnegative")
        self.wind_velocity = np.asarray(wind_velocity, dtype=float).copy()
        if self.wind_velocity.shape != (3,):
            raise ValueError("wind_velocity must have shape (3,)")
        if not np.all(np.isfinite(self.wind_velocity)):
            raise ValueError("wind_velocity must be finite")
        self.position = np.asarray(init_pos, dtype=float).copy()
        if self.position.shape != (3,):
            raise ValueError("init_pos must have shape (3,)")
        if not np.all(np.isfinite(self.position)):
            raise ValueError("init_pos must be finite")
        self.velocity = np.zeros(3, dtype=float)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)

    def _state_vector(self):
        return np.concatenate((self.position, self.velocity, self.quaternion, self.angular_velocity))

    @staticmethod
    def _unpack_state(state):
        state = np.asarray(state, dtype=float)
        if state.shape != (13,):
            raise ValueError("state must have shape (13,)")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must be finite")
        return state[0:3], state[3:6], state[6:10], state[10:13]

    def compute_derivatives(self, total_force_body, total_torque_body, state=None, wind_velocity=None):
        """Return derivatives ``(position_dot, velocity_dot, q_dot, w_dot)``.

        ``total_force_body`` is the commanded body-frame force. Gravity and
        drag are added here as inertial environmental forces. Passing ``state``
        is used internally by RK4; omitting it preserves the original API.
        """
        if state is None:
            _, velocity, quaternion, angular_velocity = (
                self.position, self.velocity, self.quaternion, self.angular_velocity
            )
        else:
            _, velocity, quaternion, angular_velocity = self._unpack_state(state)
        force_body = np.asarray(total_force_body, dtype=float)
        torque_body = np.asarray(total_torque_body, dtype=float)
        if force_body.shape != (3,) or torque_body.shape != (3,):
            raise ValueError("force and torque must have shape (3,)")
        if not np.all(np.isfinite(force_body)) or not np.all(np.isfinite(torque_body)):
            raise ValueError("force and torque must be finite")
        wind = self.wind_velocity if wind_velocity is None else np.asarray(wind_velocity, dtype=float)
        if wind.shape != (3,):
            raise ValueError("wind_velocity must have shape (3,)")
        if not np.all(np.isfinite(wind)):
            raise ValueError("wind_velocity must be finite")

        qw, qx, qy, qz = quaternion
        rotation = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
            [2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)],
            [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)],
        ])
        gravity_force = np.array([0.0, 0.0, -self.mass * GRAVITY])
        relative_velocity = relative_air_velocity(velocity, wind)
        drag_force = calculate_drag(relative_velocity, self.drag_coefficient, self.cross_sectional_area)
        acceleration = (rotation.dot(force_body) + gravity_force + drag_force) / self.mass

        omega_cross_Iw = np.cross(angular_velocity, self.inertia_matrix.dot(angular_velocity))
        angular_acceleration = self.inv_inertia.dot(torque_body - omega_cross_Iw)
        p, q, r = angular_velocity
        quaternion_dot = 0.5 * np.array([
            -qx*p - qy*q - qz*r,
             qw*p + qy*r - qz*q,
             qw*q - qx*r + qz*p,
             qw*r + qx*q - qy*p,
        ])
        return velocity.copy(), acceleration, quaternion_dot, angular_acceleration

    def _derivative_vector(self, state, force_body, torque_body, wind_velocity):
        return np.concatenate(self.compute_derivatives(force_body, torque_body, state, wind_velocity))

    def step_physics(self, force_body, torque_body, dt, wind_velocity=None):
        """Advance one step with classical fourth-order Runge-Kutta (RK4)."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        state = self._state_vector()
        k1 = self._derivative_vector(state, force_body, torque_body, wind_velocity)
        k2 = self._derivative_vector(state + 0.5 * dt * k1, force_body, torque_body, wind_velocity)
        k3 = self._derivative_vector(state + 0.5 * dt * k2, force_body, torque_body, wind_velocity)
        k4 = self._derivative_vector(state + dt * k3, force_body, torque_body, wind_velocity)
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.position = next_state[0:3].copy()
        self.velocity = next_state[3:6].copy()
        self.quaternion = next_state[6:10].copy()
        norm = np.linalg.norm(self.quaternion)
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("quaternion lost a finite, nonzero norm")
        self.quaternion /= norm
        self.angular_velocity = next_state[10:13].copy()
