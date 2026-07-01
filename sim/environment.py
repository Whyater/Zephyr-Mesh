import numpy as np

GRAVITY = 9.81  # m/s^2 in the DOWN direction
AIR_DENSITY = 1.225  # kg/m^3 baseline atmospheric density at SEA LEVEL

def calculate_drag(velocity, drag_coefficient=1.3, cross_sectional_area=0.05):
    """
    Computes fluid dynamic drag force vectors acting directly against 
    the velocity vector of a traveling drone.
    """
    vel_magnitude = np.linalg.norm(velocity)
    if vel_magnitude == 0:
        return np.array([0.0, 0.0, 0.0])
    
    # Standard Fluid Drag Equation: Fd = 0.5 * rho * v^2 * Cd * A
    drag_magnitude = 0.5 * AIR_DENSITY * (vel_magnitude ** 2) * drag_coefficient * cross_sectional_area
    drag_direction = -velocity / vel_magnitude
    
    return drag_magnitude * drag_direction