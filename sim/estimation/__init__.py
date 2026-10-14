"""Estimation package: more recovery arms beside the S6 naive and recovery-v1 arms.

Everything here is simulation. The modules add estimators (constant
acceleration, constant turn, a two-mode IMM), retrodiction for late packets,
consistency checks, sender policies and the sweep and atlas tooling. The
original S6 code in ``sim/s6_recovery.py`` is left as it was so the S6 recovery sweep stays reproducible.
"""
