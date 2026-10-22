"""Reference Morris elementary effects, written from the textbook definition."""
import numpy as np

def elementary_effects(design, outputs, p=4):
    """Compute elementary effects from trajectory design and outputs."""
    X=np.asarray(design,float); y=np.asarray(outputs,float).reshape(-1)
    if X.ndim!=2 or len(y)!=len(X): raise ValueError("design and outputs shape mismatch")
    k=X.shape[1]; Delta=float(p)/(2*(p-1)); ees=[]
    # each trajectory has k+1 consecutive points
    if len(X)%(k+1): raise ValueError("design length must be multiple of k+1")
    for b in range(0,len(X),k+1):
        for j in range(k):
            d=X[b+j+1]-X[b+j]; idx=np.flatnonzero(np.abs(d)>1e-12)
            if len(idx)!=1: raise ValueError("trajectory must change one coordinate at a time")
            i=int(idx[0]); ees.append((i,(y[b+j+1]-y[b+j])/Delta))
    return np.asarray(ees,float)

def morris_statistics(design, outputs, p=4):
    """Return (mu_star, sigma, effects) for a trajectory design and its outputs."""
    ee=elementary_effects(design,outputs,p); k=np.asarray(design).shape[1]
    vals=[ee[ee[:,0]==i,1] for i in range(k)]
    mu=np.array([np.mean(np.abs(v)) for v in vals]); sigma=np.array([np.std(v,ddof=1) if len(v)>1 else np.nan for v in vals])
    return mu,sigma,ee

def elementary_effect_summary(design, outputs, p=4):
    """Return Morris mu-star and sigma from a design and its outputs."""
    return morris_statistics(design,outputs,p)[:2]

def morris(design, outputs, p=4):
    """Return mu-star and sigma from a design and its outputs."""
    return morris_statistics(design, outputs, p)[:2]

morris_stats = morris_statistics
