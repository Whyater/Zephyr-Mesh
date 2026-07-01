import numpy as np
from sim.environment import GRAVITY, calculate_drag

class SixDOFInterceptor:
    def __init__(self, init_pos=[0.0, 0.0, 0.0], mass=0.450):
        """
        Initializes a 6-DOF rigid body model for a high-performance interceptor.
        Mass is in kilograms (450g baseline drone).
        """
        self.mass = mass
        
        # Inertia (kg * m^2) for a standard 5-inch drone frame!!
        self.I_x = 0.0023
        self.I_y = 0.0023
        self.I_z = 0.0040
        self.inertia_matrix = np.diag([self.I_x, self.I_y, self.I_z])
        self.inv_inertia = np.linalg.inv(self.inertia_matrix)
        
        # Core 13-Dimensional State Vector
        self.position = np.array(init_pos, dtype=float)       # [x, y, z] relative to the ground
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float) # [vx, vy, vz] forward/side/up speeds
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) # [qw, qx, qy, qz] rotation state
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float) # [p, q, r] rotation speeds
        
    def compute_derivatives(self, total_force_body, total_torque_body):
        """
        Calculates how the drone's position and rotation are changing right now
        based on the physical forces applied to it.
        """
        # 1. Convert our orientation quaternion into a 3x3 Rotation Matrix
        qw, qx, qy, qz = self.quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
        ])
        
        # 2. Translational Dynamics: Convert body forces to ground perspective and apply gravity
        gravity_force = np.array([0.0, 0.0, -self.mass * GRAVITY])
        force_inertial = R.dot(total_force_body) + gravity_force
        acceleration = force_inertial / self.mass
        
        # 3. Rotational Dynamics: Calculate angular acceleration using Euler's equations
        w = self.angular_velocity
        omega_cross_Iw = np.cross(w, self.inertia_matrix.dot(w))
        angular_acceleration = self.inv_inertia.dot(total_torque_body - omega_cross_Iw)
        
        # 4. Quaternion Derivative: Track the rate of rotation change
        p, q, r = w
        quaternion_dot = 0.5 * np.array([
            -qx*p - qy*q - qz*r,
             qw*p + qy*r - qz*q,
             qw*q - qx*r + qz*p,
             qw*r + qx*q - qy*p
        ])
        
        return self.velocity, acceleration, quaternion_dot, angular_acceleration

    def step_physics(self, force_body, torque_body, dt):
        """
        Advances the drone physics forward in time by a tiny fraction of a second (dt)
        using highly stable 4th-Order Runge-Kutta (RK4) math.
        """
        pos_dot1, vel_dot1, q_dot1, w_dot1 = self.compute_derivatives(force_body, torque_body)
        
        # Temporary mid-point updates for high numerical precision
        pos_mid = self.position + pos_dot1 * (dt / 2)
        vel_mid = self.velocity + vel_dot1 * (dt / 2)
        q_mid = self.quaternion + q_dot1 * (dt / 2)
        w_mid = self.angular_velocity + w_dot1 * (dt / 2)
        
        # Compute trajectory at the mid-point step
        pos_dot2, vel_dot2, q_dot2, w_dot2 = self.compute_derivatives(force_body, torque_body)
        
        # Final state updates
        self.position += pos_dot2 * dt
        self.velocity += vel_dot2 * dt
        self.quaternion += q_dot2 * dt
        self.angular_velocity += w_dot2 * dt
        
        # Normalize the quaternion to clean up tiny rounding errors
        self.quaternion /= np.linalg.norm(self.quaternion)