"""Independent geometry references for simulated motion."""
import math
import numpy as np


def min_distance_two_constant_velocity(p1, v1, p2, v2, horizon):
    r = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    u = np.asarray(v1, dtype=float) - np.asarray(v2, dtype=float)
    denom = float(np.dot(u, u))
    t = 0.0 if denom == 0 else float(np.clip(-np.dot(r, u) / denom, 0.0, horizon))
    return float(np.linalg.norm(r + t * u))


def required_buffer_hand(closing_speed, delay_s, reaction_s, sigma_pos, k=2.0):
    return closing_speed * (delay_s + reaction_s) + k * math.sqrt(2.0) * sigma_pos


def dead_reckoning_error(accel, age_s):
    return 0.5 * accel * age_s**2
