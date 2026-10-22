"""Tiny smoke checks for every reference implementation."""

import numpy as np

from .ref_cv import run_cv, predict as cv_predict
from .ref_ca import run_ca, predict as ca_predict
from .ref_ct import run_ct, predict as ct_predict
from .ref_imm import run_imm, predict_combined
from .ref_retro import run_retro
from .ref_nis import nis, chi_square_window_band, adaptive_scale
from .ref_ntp import offset_round_trip, simulate_exchange
from .ref_quant_event import quantize_int16, event_trigger
from .ref_morris import elementary_effects, morris_statistics


def main():
    """Run tiny examples covering every estimator, clock, sender and screening rule."""
    z = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t = np.array([0.0, 1.0])
    print("cv", run_cv(z, t, [0.0, 0.0, 0.0])[-1][0])
    cv_hist = run_cv(z, t, [0.0, 0.0, 0.0])
    print("cv_predict", cv_predict(cv_hist[-1][0], cv_hist[-1][1], 1.0, 1.5)[0])
    print("ca", run_ca(z, t, [0.0, 0.0, 0.0])[-1][0])
    ca_hist = run_ca(z, t, [0.0, 0.0, 0.0])
    print("ca_predict", ca_predict(ca_hist[-1][0], ca_hist[-1][1], 1.0, 1.5)[0])
    print("ct", run_ct(z, t, [0.0, 0.0, 0.0])[-1][0])
    ct_hist = run_ct(z, t, [0.0, 0.0, 0.0])
    print("ct_predict", ct_predict(ct_hist[-1][0], ct_hist[-1][1], 1.0, 1.5)[0])
    imm_hist = run_imm(z, t, [0.0, 0.0, 0.0])
    print("imm", imm_hist[-1][0], imm_hist[-1][2])
    modes = [(np.zeros(7), np.eye(7)), (np.zeros(7), np.eye(7))]
    print("imm_predict", predict_combined(modes, [0.5, 0.5], 0.0, 0.1)[0])
    packets = [(0.0, z[0], 0.0), (1.0, z[1], 1.0), (0.5, [0.5, 0.0, 0.0], 1.1)]
    print("retro", len(run_retro(packets, 1.0)))
    print("nis", nis([1.0, 2.0, 3.0], np.eye(3)))
    print("nis_band", chi_square_window_band(2))
    print("adaptive", adaptive_scale([3.0] * 10 + [6.0]))
    print("ntp_offset", offset_round_trip(0.0, 1.1, 1.2, 2.0))
    print("ntp_exchange", simulate_exchange(0.1, 10.0, 0.2, 0.2, 0.05, 1.0))
    print("quant", quantize_int16([0.04, 0.06, 100000.0], 0.1))
    z3 = np.vstack([z, [2.0, 0.0, 0.0]])
    print("events", event_trigger([0.0, 1.0, 2.0], z3, 0.1))
    design = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.5], [0.5, 0.5]])
    outputs = design.sum(axis=1)
    print("morris_ee", elementary_effects(design, outputs, p=4))
    print("morris_stats", morris_statistics(design, outputs, p=4)[:2])


if __name__ == "__main__":
    main()
