"""Write the two literature-derived v2 hardware profiles (sources checked 2026-10-14).

Usage: python tools/a_build_profiles.py OUTPUT_DIR

Every value in the written profiles carries its source, URL, access date, plausible
range and status. Derived values (for example the Crazyflie-class top rotor speed)
are computed here so the arithmetic can be checked. Nothing here is measured.
"""
import json
import math
import sys
from pathlib import Path

ACC = "2026-10-14"
FORSTER = "https://www.research-collection.ethz.ch/handle/20.500.11850/214143"
ESCHMANN = "https://arxiv.org/html/2404.07837v2"
LANDING = "https://arxiv.org/pdf/2403.00128"
LUIS = "https://arxiv.org/pdf/1608.05786"
BITCRAZE_PWM = "https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/"
CF_FW = "https://raw.githubusercontent.com/bitcraze/crazyflie-firmware/master/src/platform/interface/platform_defaults_cf2.h"
LANDRY = "http://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf"
SEEED = "https://raw.githubusercontent.com/Seeed-Projects/Co-Create_ESP-FLY/main/README.md"


def p(value, rng, source, url, status, note=None):
    entry = {"value": value, "range": rng, "source": source, "url": url, "accessed": ACC, "status": status}
    if note:
        entry["note"] = note
    return entry


def drag_off(parameters):
    parameters["airframe.drag_coefficient"] = p(
        0.0, [0.0, 0.0], "no body-drag source found for this class", "none", "EXCLUDED",
        "Body drag is UNCONFIRMED (no sourced value), so it is switched off by default. Sweep it separately if needed.")
    parameters["airframe.cross_sectional_area_m2"] = p(
        0.001, [0.001, 0.001], "placeholder, unused while drag_coefficient is 0", "none", "EXCLUDED",
        "Irrelevant while drag is off; kept only because the schema requires the field.")


def common_rotor(parameters, source, url):
    parameters["rotor.count"] = p(4, [4, 4], source, url, "CONFIRMED")
    parameters["rotor.spin_directions"] = p([1, -1, 1, -1], [[1, 1], [-1, -1], [1, 1], [-1, -1]],
                                            "standard X quad, diagonal pairs spin the same way", url, "CONFIRMED",
                                            "Order front-left, front-right, rear-right, rear-left; +1 is counterclockwise seen from above.")


def crazyflie():
    k_f = 1.9e-8
    t_max = 0.1539
    w_max = round(math.sqrt(t_max / k_f), 1)
    doc = {
        "schema": "zephyr-hardware-profile-2",
        "profile_id": "crazyflie-class",
        "status": "literature-derived",
        "notes": ("Bitcraze Crazyflie 2.x class micro-quad, stock 7 mm coreless motors and 45 mm props. "
                  "Literature-derived from published system identification (mainly Forster 2015, ETH Zurich). "
                  "Not measured by this project. Battery sag is off by default because internal resistance is UNCONFIRMED. "
                  "Known source errors: one paper's table prints the mass 10x too large (0.27 kg); some widely copied "
                  "simulators use a torque-to-thrust ratio of 0.0251 m, 4 to 6 times the identified value."),
        "airframe": {"mass_kg": 0.028,
                     "inertia_kg_m2": [1.657171e-05, 1.6655602e-05, 2.9261652e-05],
                     "arm_length_m": 0.046, "drag_coefficient": 0.0, "cross_sectional_area_m2": 0.001},
        "rotor": {"count": 4, "layout": "x", "thrust_coefficient_n_s2": k_f, "torque_to_thrust_m": 0.005964552,
                  "max_speed_rad_s": w_max, "motor_time_constant_s": 0.072, "prop_radius_m": 0.0225,
                  "spin_directions": [1, -1, 1, -1]},
        "battery": None,
        "parameters": {},
    }
    P = doc["parameters"]
    P["airframe.mass_kg"] = p(0.028, [0.027, 0.029], "Forster 2015 Table 2.2 p.17; Bitcraze CF2.1 datasheet 29 g; firmware CF_MASS 0.029", FORSTER, "CROSS-CHECKED")
    P["airframe.inertia_kg_m2"] = p(doc["airframe"]["inertia_kg_m2"],
                                    [[1.029e-05, 2.3951e-05], [1.105e-05, 2.3951e-05], [1.955e-05, 3.2347e-05]],
                                    "Forster 2015 Appendix A p.57; ranges span Eschmann, Luis and Le Ny, Landry", FORSTER, "CROSS-CHECKED")
    P["airframe.arm_length_m"] = p(0.046, [0.0397, 0.046], "Bitcraze firmware ARM_LENGTH 0.046; Landry 92 mm motor to motor; Luis and Le Ny 39.73 mm (competing definition)", CF_FW, "CROSS-CHECKED")
    drag_off(P)
    common_rotor(P, "Crazyflie 2.x is a 4-rotor X frame", FORSTER)
    P["rotor.thrust_coefficient_n_s2"] = p(k_f, [1.71e-08, 2.88e-08], "derived from Forster eq 3.3 (thrust map) and eq 3.4 (speed map) near hover; Luis and Le Ny theory 2.88e-8; Bitcraze stand 2.28e-8 at max", FORSTER, "CONFIRMED",
                                           "Derived, not printed: thrust per rotor at hover divided by hover speed squared from Forster's two fitted maps.")
    P["rotor.torque_to_thrust_m"] = p(0.005964552, [0.004548, 0.007350862856566459], "Forster eq 3.5 p.29; Eschmann Sec IV-A; Bitcraze firmware THRUST2TORQUE", FORSTER, "CROSS-CHECKED",
                                      "The 0.0251 m value in some simulators is an outlier and is excluded.")
    P["rotor.max_speed_rad_s"] = p(w_max, [2270.0, 3052.0], "set so k_f w_max^2 equals the cross-checked max thrust 0.1539 N (Forster p.27, eq 3.3 at cmd 64000)", FORSTER, "SCALED",
                                   f"w_max = sqrt(0.1539 N / {k_f:g}) = {w_max} rad/s. Forster's speed map gives 3052 rad/s at full command, but rotor speed is the least reliable quantity, so thrust is matched instead.")
    P["rotor.motor_time_constant_s"] = p(0.072, [0.065, 0.0775], "Eschmann Sec IV-A 0.072 s; Bitcraze step response about 0.073 s; Forster H2(z) gives 0.0647 s", ESCHMANN, "CROSS-CHECKED")
    P["rotor.prop_radius_m"] = p(0.0225, [0.0225, 0.0235], "Landry p.26 45 mm prop; Luis and Le Ny 23.1348 mm", LANDRY, "CROSS-CHECKED")
    return doc


def espfly():
    doc = {
        "schema": "zephyr-hardware-profile-2",
        "profile_id": "esp-fly-class",
        "status": "literature-derived",
        "notes": ("Seeed ESP-FLY kit as sold (XIAO ESP32-S3, 615 coreless motors, 30 mm tri-blade props, 1S 250 mAh). "
                  "Mass and prop size are vendor numbers; inertia, thrust and torque coefficients and motor lag are SCALED "
                  "from Crazyflie identification data with the formula given per field. Not measured by this project. "
                  "Battery sag is off by default because the cell's internal resistance is UNCONFIRMED."),
        "airframe": {"mass_kg": 0.025, "inertia_kg_m2": [4.4e-06, 5.2e-06, 8.4e-06],
                     "arm_length_m": 0.026, "drag_coefficient": 0.0, "cross_sectional_area_m2": 0.001},
        "rotor": {"count": 4, "layout": "x", "thrust_coefficient_n_s2": 4.0e-09, "torque_to_thrust_m": 0.0034,
                  "max_speed_rad_s": 6450.0, "motor_time_constant_s": 0.07, "prop_radius_m": 0.015,
                  "spin_directions": [1, -1, 1, -1]},
        "battery": None,
        "parameters": {},
    }
    P = doc["parameters"]
    P["airframe.mass_kg"] = p(0.025, [0.025, 0.028], "Seeed ESP-FLY README specifications: about 25 g with LiPo, 28 g with FPV camera", SEEED, "CONFIRMED")
    P["airframe.inertia_kg_m2"] = p(doc["airframe"]["inertia_kg_m2"], [[3.3e-06, 6.1e-06], [4.0e-06, 7.3e-06], [7.0e-06, 1.2e-05]],
                                    "scaled from the measured Crazyflie 2.1 inertia (landing paper Table II, bifilar pendulum, 30.0 g)", LANDING, "SCALED",
                                    "I = I_cf (m / 0.030) (L / 0.0397)^2 with I_cf = (12.19, 14.55, 23.55)e-6, m = 0.025 kg, L = 0.026 m.")
    P["airframe.arm_length_m"] = p(0.026, [0.025, 0.028], "Seeed README dimensions 67 x 67 mm with 30 mm props", SEEED, "SCALED",
                                   "Motor axis spacing (67 - 30) mm = 37 mm; L = 37 / sqrt(2) = 26.2 mm.")
    drag_off(P)
    common_rotor(P, "Seeed ESP-FLY has 4 motors in an X frame", SEEED)
    P["rotor.thrust_coefficient_n_s2"] = p(4.0e-09, [3.1e-09, 5.1e-09], "scaled from Bitcraze thrust-stand data by prop diameter", BITCRAZE_PWM, "SCALED",
                                           "k_f = 2.27e-8 (D / 0.04627 m)^4 with D = 0.030 m (T = k_T rho n^2 D^4, Luis and Le Ny eq 2.3.1).")
    P["rotor.torque_to_thrust_m"] = p(0.0034, [0.0024, 0.016], "scaled from identified Crazyflie values (Eschmann Sec IV-A)", ESCHMANN, "SCALED",
                                      "Ratio scales with diameter: 5.25e-3 m (median Crazyflie value) x 30 / 46.27 = 3.4e-3 m.")
    P["rotor.max_speed_rad_s"] = p(6450.0, [5720.0, 7330.0], "vendor 17 g thrust per motor (Seeed README) with the scaled k_f; vendor 70,000 rpm read as a no-load ceiling", SEEED, "SCALED",
                                   "w_max = sqrt(0.1668 N / 4.0e-9) = 6450 rad/s, so k_f w_max^2 matches the vendor max thrust.")
    P["rotor.motor_time_constant_s"] = p(0.07, [0.05, 0.16], "Crazyflie brushed-motor identification (no 615-motor source): Eschmann fitted 0.072 s; landing paper tau_up 0.05, tau_down 0.16", ESCHMANN, "SCALED",
                                         "Borrowed from the Crazyflie class; no ESP-FLY specific value exists.")
    P["rotor.prop_radius_m"] = p(0.015, [0.015, 0.0155], "Seeed README parts list: 30 mm tri-blade props", SEEED, "CONFIRMED")
    return doc


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("crazyflie_class.json", crazyflie()), ("esp_fly_class.json", espfly())):
        (out / name).write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        print(out / name)
