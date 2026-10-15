import numpy as np

from sim.environment import GRAVITY


class GeometricFlightController:
    def __init__(self):
        """
        Initializes tuning gains for the dual-layer flight control loops.
        P handles responsiveness, D handles dampening (prevents overshooting).
        """
        # Outer Loop: Position Gains (Translating space error to desired forces)
        self.K_p_pos = np.array([2.0, 2.0, 3.0])  # X, Y, Z responsiveness
        self.K_d_pos = np.array([1.5, 1.5, 2.0])  # X, Y, Z dampening

        # Inner Loop: Attitude Gains, written per unit of moment of inertia.
        # The torque on each body axis is I * (K_p_att * tilt - K_d_att * rate),
        # so the same law gives the same angular response for a heavy or a
        # light frame.  Natural frequency 10 rad/s and damping ratio 0.9 give
        # K_p_att = 10**2 = 100 per s^2 and K_d_att = 2 * 0.9 * 10 = 18 per s.
        # The controller is sampled: torque is held constant for a whole step
        # (zero-order hold).  The rate term then multiplies the spin rate by
        # (1 - K_d_att * dt) each step, so the loop only settles if
        # K_d_att * dt < 2, which is dt < 0.11 s here.  The old fixed gains
        # (2.5 N m s on I = 0.0023 kg m^2, so 1087 per s) needed dt < 0.0018 s
        # and diverged at the 0.01 s baseline step for any sideways command.
        # Without a target yaw, the yaw entry of K_p_att has no effect: the
        # body-frame tilt error is always perpendicular to body z, so its yaw
        # part is exactly zero, and yaw is rate damping only.  With
        # target_yaw_rad it is the heading gain.
        self.K_p_att = np.array([100.0, 100.0, 100.0])  # Roll, Pitch, Yaw (per s^2)
        self.K_d_att = np.array([18.0, 18.0, 18.0])     # Roll, Pitch, Yaw (per s)

    def update_control(self, drone, target_position, dt, *, target_yaw_rad=None):
        """
        Executes the cascaded control loop. Takes current drone state and target,
        outputs the required collective thrust force and 3D body torque vector.

        ``target_yaw_rad`` optionally sets a heading in radians.  The heading
        error follows the shortest angular path in ``(-pi, pi]`` and adds a
        body-z torque while leaving the roll and pitch tilt control unchanged.
        When it is ``None``, the controller uses the original yaw-rate damping
        path.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        # Necessary condition for the sampled rate loop to settle (see __init__).
        dt_limit = 2.0 / float(np.max(self.K_d_att))
        if dt >= dt_limit:
            raise ValueError(f"dt {dt} s is at or above the sampled attitude-loop limit {dt_limit:.4f} s")
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
            raise ValueError("target_position must be a finite length-3 vector")
        if not np.all(np.isfinite(drone.position)) or not np.all(np.isfinite(drone.velocity)):
            raise ValueError("drone state must be finite")
        if not np.all(np.isfinite(drone.quaternion)) or not np.all(np.isfinite(drone.angular_velocity)):
            raise ValueError("drone state must be finite")
        # =====================================================================
        # LAYER 1: OUTER LOOP (Position Control)
        # =====================================================================
        
        # Calculate distance error and velocity error
        position_error = target_position - drone.position
        velocity_error = np.array([0.0, 0.0, 0.0]) - drone.velocity # Target velocity is 0 (hover)

        # Calculate desired acceleration using proportional and derivative feedback
        desired_accel = (self.K_p_pos * position_error) + (self.K_d_pos * velocity_error)

        # Total inertial force needed (Mass * Accel + Counteracting Gravity)
        gravity_compensation = np.array([0.0, 0.0, drone.mass * GRAVITY])
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
        # A rotor cannot pull downward.  If the vehicle is inverted or tilted
        # past 90 degrees, command zero thrust and let the attitude loop recover.
        collective_thrust = max(0.0, float(np.dot(desired_force_inertial, current_z_body)))

        # Geometric Tilt Error: the cross product of where we point and where we
        # want to point gives the rotation axis and size of the tilt error.  Both
        # vectors are in the ground frame, so the result is too.  Torque is
        # applied in the body frame, so rotate it into body axes with R.T
        # (R maps body to ground; its transpose maps ground to body).  Without
        # this, a drone yawed by 180 degrees tilts the wrong way.
        tilt_error_body = R.T @ np.cross(current_z_body, desired_z_body)

        # Angular rate error (current spin speeds vs desired spin speeds which are 0)
        rate_error = np.array([0.0, 0.0, 0.0]) - drone.angular_velocity

        # Required body torque: per-axis inertia times the desired angular acceleration.
        inertia = np.array([drone.I_x, drone.I_y, drone.I_z])
        desired_torque = inertia * ((self.K_p_att * tilt_error_body) + (self.K_d_att * rate_error))

        if target_yaw_rad is not None:
            target_yaw_rad = float(target_yaw_rad)
            if not np.isfinite(target_yaw_rad):
                raise ValueError("target_yaw_rad must be finite")
            current_yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy**2 + qz**2),
            )
            yaw_error = (target_yaw_rad - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
            if yaw_error <= -np.pi:
                yaw_error += 2.0 * np.pi
            desired_torque[2] = drone.I_z * (
                self.K_p_att[2] * yaw_error - self.K_d_att[2] * drone.angular_velocity[2]
            )

        # Package outputs for the physics engine
        # Thrust force vector acts strictly upward relative to the drone's own floor [0, 0, Thrust]
        force_body = np.array([0.0, 0.0, collective_thrust])
        
        return force_body, desired_torque
