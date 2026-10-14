"""Hand estimate of the delay that destabilizes the position loop (EXP-021).

This is written from the linearized equations, not from the simulator, so it
is an independent check on the A6 sweep. Small-angle hover model:

    x'' = g * theta                     (tilt makes horizontal acceleration)
    theta_des = a_des / g               (controller asks for a tilt)
    a_des = Kp * e - Kd * v             (position PD, velocity setpoint is zero)
    theta'' = Kp_att (theta_des - theta) - Kd_att theta'   (attitude PD per unit inertia)

With an optional first-order motor lag tau on the attitude torque, the
attitude loop is theta / theta_des = Kp_att / (tau s^3 + s^2 + Kd_att s + Kp_att).

Open loop of the position loop: L(s) = (Kp + Kd s) / s^2 * attitude(s).
A pure delay d has gain 1 and phase -omega d, so the loop loses stability when
omega_c * d reaches the phase margin: d_crit = PM / omega_c (PM in radians).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def attitude_response(s, kp_att, kd_att, tau):
    return kp_att / (tau * s**3 + s**2 + kd_att * s + kp_att)


def position_open_loop(s, kp, kd, kp_att, kd_att, tau, with_attitude=True):
    loop = (kp + kd * s) / s**2
    if with_attitude:
        loop = loop * attitude_response(s, kp_att, kd_att, tau)
    return loop


def vertical_open_loop(s, kp, kd, tau):
    # Thrust passes straight through the motor lag; z'' = a_des.
    return (kp + kd * s) / s**2 / (tau * s + 1.0)


def attitude_open_loop(s, kp_att, kd_att, tau):
    # Whole-state delay: the attitude loop itself sees stale angle and rate.
    return (kp_att + kd_att * s) / s**2 / (tau * s + 1.0)


def crossover_and_margin(loop_fn, w_lo=1e-2, w_hi=1e3, n=400001):
    """Find the first frequency where |L| falls through 1 and the phase margin there."""
    w = np.logspace(np.log10(w_lo), np.log10(w_hi), n)
    mag = np.abs(loop_fn(1j * w))
    idx = np.where((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    if idx.size == 0:
        raise ValueError("no gain crossover in the search band")
    i = idx[0]
    # Linear interpolation in log magnitude between the two bracketing samples.
    lm0, lm1 = np.log(mag[i]), np.log(mag[i + 1])
    frac = lm0 / (lm0 - lm1)
    w_c = float(np.exp(np.log(w[i]) + frac * (np.log(w[i + 1]) - np.log(w[i]))))
    phase = float(np.angle(loop_fn(1j * w_c)))
    # Phase margin = 180 degrees + phase (phase is between -pi and 0 here,
    # or wraps below -pi; unwrap by taking the value in (-2 pi, 0]).
    if phase > 0:
        phase -= 2 * np.pi
    pm = np.pi + phase
    return w_c, pm


def estimate(kp_xy=2.0, kd_xy=1.5, kp_z=3.0, kd_z=2.0, kp_att=100.0, kd_att=18.0,
             tau=0.0, sample_period=0.02, control_dt=0.01):
    """Return the hand estimates as a plain dict (seconds and radians)."""
    out = {}
    hold = sample_period / 2.0 + control_dt / 2.0  # average extra age from sampling and holds

    w_c, pm = crossover_and_margin(lambda s: position_open_loop(s, kp_xy, kd_xy, kp_att, kd_att, tau, False))
    out["xy_no_attitude"] = {"omega_c": w_c, "pm_rad": pm, "pm_deg": float(np.degrees(pm)), "d_crit_s": pm / w_c}

    w_c, pm = crossover_and_margin(lambda s: position_open_loop(s, kp_xy, kd_xy, kp_att, kd_att, tau, True))
    out["xy"] = {"omega_c": w_c, "pm_rad": pm, "pm_deg": float(np.degrees(pm)), "d_crit_s": pm / w_c,
                 "d_crit_sim_expected_s": pm / w_c - hold}

    w_c, pm = crossover_and_margin(lambda s: vertical_open_loop(s, kp_z, kd_z, tau))
    out["z"] = {"omega_c": w_c, "pm_rad": pm, "pm_deg": float(np.degrees(pm)), "d_crit_s": pm / w_c,
                "d_crit_sim_expected_s": pm / w_c - hold}

    w_c, pm = crossover_and_margin(lambda s: attitude_open_loop(s, kp_att, kd_att, tau))
    out["attitude_whole_state"] = {"omega_c": w_c, "pm_rad": pm, "pm_deg": float(np.degrees(pm)),
                                   "d_crit_s": pm / w_c, "d_crit_sim_expected_s": pm / w_c - control_dt / 2.0}
    out["inputs"] = {"kp_xy": kp_xy, "kd_xy": kd_xy, "kp_z": kp_z, "kd_z": kd_z, "kp_att": kp_att,
                     "kd_att": kd_att, "tau": tau, "sample_period": sample_period, "control_dt": control_dt,
                     "hold_age_s": hold}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tau", type=float, action="append", help="motor time constant(s) in seconds")
    args = parser.parse_args(argv)
    taus = args.tau or [0.0, 0.03, 0.05, 0.072]
    result = {f"tau={t:g}": estimate(tau=t) for t in taus}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
