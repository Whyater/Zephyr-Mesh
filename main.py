import argparse

import numpy as np

from sim.baseline import BaselineConfig, run_baseline, write_baseline

def run_mission(output_dir=None, plot=True):
    config = BaselineConfig()
    telemetry = run_baseline(config)
    time_history = telemetry["time"]
    z_position_history = telemetry["z"]
    z_velocity_history = telemetry["vz"]

    print("====================================================")
    print("  LAUNCHING PROJECT ZEPHYR-MESH FLIGHT SIMULATOR")
    print("====================================================")

    # 2. Real-Time Flight Loop
    for step in range(0, len(time_history), 50):
        print(f"Time: {time_history[step]:.1f}s | Vehicle altitude: {z_position_history[step]:.2f}m | Vertical speed: {z_velocity_history[step]:.2f}m/s")

    if output_dir is not None:
        print(f"Telemetry written to {write_baseline(output_dir, config, telemetry=telemetry)}")

    if not plot:
        return telemetry

    print("\nSimulation complete. Plotting GNC performance data...")

    # 3. Generate Mathematical Proof-of-Concept Graph
    import matplotlib.pyplot as plt
    target_position = np.asarray(config.target_position)
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, z_position_history, label='Simulated vehicle altitude (Z)', color='#7CC4FF', linewidth=2.5)
    plt.axhline(y=target_position[2], color='#9AA8C0', linestyle='--', label='Target waypoint (3.0 m)', linewidth=1.5)
    
    plt.title('Zephyr-Mesh guidance and control: step response fixture', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('Altitude (meters)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.ylim(-0.2, 4.0)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Zephyr S0 hover baseline")
    parser.add_argument("--output", help="directory for telemetry.csv and manifest.json")
    parser.add_argument("--no-plot", action="store_true", help="skip the Matplotlib window")
    args = parser.parse_args()
    run_mission(output_dir=args.output, plot=not args.no_plot)
