"""Reference innovation, gating, and adaptive scale functions."""
import numpy as np
from scipy.stats import chi2

def nis(nu, S):
    """Return nu' S^-1 nu."""
    nu=np.asarray(nu,float); S=np.asarray(S,float)
    return float(nu @ np.linalg.solve(S,nu))

def chi_square_window_band(W):
    """Return lower and upper mean-NIS band for W."""
    W=int(W)
    if W<=0: raise ValueError("W must be positive")
    return (float(chi2.ppf(.025,3*W)/W), float(chi2.ppf(.975,3*W)/W))

def gate_threshold(g):
    """Return chi-square gate threshold for probability g."""
    return float(chi2.ppf(g,3))

def adaptive_scale(nis_values, mode="q", initial=1.0, W=10):
    """Return scale after each NIS."""
    if mode not in ("q","r"): raise ValueError("mode must be q or r")
    scale=float(initial); buf=[]; hist=[]
    for value in nis_values:
        buf.append(float(value))
        if len(buf)==W:
            factor=np.clip(np.mean(buf)/3.0,.5,2.0)
            scale=float(np.clip(scale*factor,.01,100.0)); buf=[]
        hist.append(scale)
    return hist

# convenient aliases
window_band=chi_square_window_band
adaptive_scale_history=adaptive_scale
adaptive_scales = adaptive_scale
