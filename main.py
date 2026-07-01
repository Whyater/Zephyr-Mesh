import numpy as np
import matplotlib.pyplot as plt
from sim.drone import SixDOFInterceptor
from sim.controller import GeometricFlightController

def run_mission():
    # 1. Initialize objects (Spawn the drone at ground level [0, 0, 0])
    drone = SixDOFInterceptor(init_pos=[0.0, 0.0, 0.0])
    controller = GeometricFlightController()

    # Simulation parameters
    dt = 0.01          # 10ms timestep (matches physical 100Hz flight controllers)
    total_time = 7.0   # Run simulation for 7 seconds
    steps = int(total_time / dt)

    # Define the mission objective: Fly up to 3 meters and hold position
    target_position = np.array([0.0, 0.0, 3.0])

    # Telemetry data storage for post-flight plotting
    time_history = []
    z_position_history = []
    z_velocity_history = []

    print("====================================================")
    print("  LAUNCHING PROJECT ZEPHYR-MESH RIGID-BODY SIMULATOR")
    print("====================================================")

    # 2. Real-Time Flight Loop
    for step in range(steps):
        current_time = step * dt

        # Read the current drone state and fetch required forces/torques from the brain
        force_body, torque_body = controller.update_control(drone, target_position, dt)

        # Feed the forces into our high-precision RK4 physics engine
        drone.step_physics(force_body, torque_body, dt)

        # Log active telemetry variables
        time_history.append(current_time)
        z_position_history.append(drone.position[2])
        z_velocity_history.append(drone.velocity[2])

        # Print live telemetry updates every 0.5 seconds (every 50 steps)
        if step % 50 == 0:
            print(f"Time: {current_time:.1f}s | Altitude: {drone.position[2]:.2f}m | Vertical Speed: {drone.velocity[2]:.2f}m/s")

    print("\nSimulation complete. Plotting GNC performance data...")

    # 3. Generate Mathematical Proof-of-Concept Graph
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, z_position_history, label='Actual Interceptor Altitude (Z)', color='#1f77b4', linewidth=2.5)
    plt.axhline(y=target_position[2], color='red', linestyle='--', label='Target Waypoint Target (3.0m)', linewidth=1.5)
    
    plt.title('Zephyr-Mesh Guidance & Control System: Step Response Validation', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('Altitude (meters)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.ylim(-0.2, 4.0)
    
    plt.show()

if __name__ == "__main__":
    run_mission()