import numpy as np

class GeometricFlightController:
    def __init__(self):
        """
        Initializes tuning gains for the dual-layer flight control loops.
        P handles responsiveness, D handles dampening (prevents overshooting).
        """
        # Outer Loop: Position Gains (Translating space error to desired forces)
        self.K_p_pos = np.array([2.0, 2.0, 3.0])  # X, Y, Z responsiveness
        self.K_d_pos = np.array([1.5, 1.5, 2.0])  # X, Y, Z dampening

        # Inner Loop: Attitude Gains (Translating orientation error to torque)
        self.K_p_att = np.array([12.0, 12.0, 4.0]) # Roll, Pitch, Yaw responsiveness
        self.K_d_att = np.array([2.5, 2.5, 1.0])   # Roll, Pitch, Yaw dampening

    def update_control(self, drone, target_position, dt):
        """
        Executes the cascaded control loop. Takes current drone state and target,
        outputs the required collective thrust force and 3D body torque vector.
        """
        # =====================================================================
        # LAYER 1: OUTER LOOP (Position Control)
        # =====================================================================
        
        # Calculate distance error and velocity error
        position_error = target_position - drone.position
        velocity_error = np.array([0.0, 0.0, 0.0]) - drone.velocity # Target velocity is 0 (hover)

        # Calculate desired acceleration using proportional and derivative feedback
        desired_accel = (self.K_p_pos * position_error) + (self.K_d_pos * velocity_error)

        # Total inertial force needed (Mass * Accel + Counteracting Gravity)
        gravity_compensation = np.array([0.0, 0.0, drone.mass * 9.81])
        desired_force_inertial = (drone.mass * desired_accel) + gravity_compensation

        # Prevent the drone from trying to fly upside down by capping minimum vertical force
        if desired_force_inertial[2] < 0.1:
            desired_force_inertial[2] = 0.1

        # =====================================================================
        # LAYER 2: INNER LOOP (Attitude & Orientation Control)
        # =====================================================================
        
        # Reconstruct the drone's current orientation matrix from its quaternion state
        qw, qx, qy, qz = drone.quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
        ])
        
        # Extract the current direction the drone's roof is pointing (Current Body Z-axis)
        current_z_body = R[:, 2]

        # Calculate where the drone's roof NEEDS to point to match our force vector
        desired_z_body = desired_force_inertial / np.linalg.norm(desired_force_inertial)

        # Collective Thrust is the amount of force directed along the drone's actual current body Z-axis
        collective_thrust = np.dot(desired_force_inertial, current_z_body)

        # Geometric Tilt Error: Take the cross product of where we are pointing vs where we want to point
        # This gives us a vector showing the exact axis and magnitude of tilt error instantly
        tilt_error = np.cross(current_z_body, desired_z_body)

        # Angular rate error (current spin speeds vs desired spin speeds which are 0)
        rate_error = np.array([0.0, 0.0, 0.0]) - drone.angular_velocity

        # Compute required 3D torque vector to rotate the drone body
        desired_torque = (self.K_p_att * tilt_error) + (self.K_d_att * rate_error)

        # Package outputs for the physics engine
        # Thrust force vector acts strictly upward relative to the drone's own floor [0, 0, Thrust]
        force_body = np.array([0.0, 0.0, collective_thrust])
        
        return force_body, desired_torque