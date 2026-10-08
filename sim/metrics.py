"""Pure, scenario-agnostic metrics for S1 simulation telemetry."""
from __future__ import annotations
import math
import numpy as np

def _xy(t, y):
    t, y = np.asarray(t,float), np.asarray(y,float)
    if t.ndim != 1 or y.ndim != 1 or len(t)!=len(y): raise ValueError("time and values must be 1-D and equal length")
    if len(t) and np.any(~np.isfinite(t)): raise ValueError("time must be finite")
    if len(t)>1 and np.any(np.diff(t)<=0): raise ValueError("time must be strictly increasing")
    return t,y

def step_response(t, y, goal, *, tolerance=0.05, rise_fraction=0.1):
    t,y=_xy(t,y); out={"rise_time_s":None,"overshoot_pct":None,"settling_time_s":None,"censored":{},"count":len(y)}
    out["definitions"]={"rise":"first sample at rise_fraction of step amplitude","overshoot":"peak beyond goal divided by step amplitude","settling":"first sample after which all samples stay within tolerance of goal"}
    for k in ("rise_time_s", "overshoot_pct", "settling_time_s"): out["censored"][k]=False
    if np.any(~np.isfinite(y)) or not math.isfinite(float(goal)):
        raise ValueError("response and goal must be finite")
    if not len(y):
        out["censored"]={k:True for k in out["censored"]}
        return out
    initial=float(y[0]); amp=float(goal-initial)
    if amp==0: out["censored"]={"rise_time_s":True,"overshoot_pct":True,"settling_time_s":True}; return out
    level=initial+rise_fraction*amp
    idx=np.flatnonzero(y>=level if amp>0 else y<=level)
    if len(idx): out["rise_time_s"]=float(t[idx[0]]-t[0])
    else: out["censored"]["rise_time_s"]=True
    peak=float(np.max(y) if amp>0 else np.min(y)); excess=max(0., (peak-goal)/abs(amp) if amp>0 else (goal-peak)/abs(amp)); out["overshoot_pct"]=100*excess
    tol=abs(amp)*tolerance; ok=np.abs(y-goal)<=tol
    starts=np.flatnonzero(np.all(ok[np.newaxis,:] if False else ok[None,:],axis=0))
    settle=None
    for i in starts:
        if np.all(ok[i:]): settle=float(t[i]-t[0]); break
    if settle is None: out["censored"]["settling_time_s"]=True
    else: out["settling_time_s"]=settle
    return out

def tracking_error(truth, estimate):
    a,b=np.asarray(truth,float),np.asarray(estimate,float)
    if a.shape!=b.shape or a.size==0: return {"count":0,"empty":True,"rms_m":None,"median_m":None,"p90_m":None,"p95_m":None,"max_m":None,"per_axis":None}
    if a.ndim==1: a,b=a[:,None],b[:,None]
    if a.ndim!=2 or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)): raise ValueError("truth and estimate must be finite arrays")
    e=np.linalg.norm(b-a,axis=1); return {"count":len(e),"empty":False,"rms_m":float(np.sqrt(np.mean(e*e))),"median_m":float(np.percentile(e,50)),"p90_m":float(np.percentile(e,90)),"p95_m":float(np.percentile(e,95)),"max_m":float(np.max(e)),"per_axis":{"rms_m":np.sqrt(np.mean((b-a)**2,axis=0)).tolist()}}

def packet_delivery(records, *, sent_denominator=None):
    rows=list(records); sent=set(); received=set(); duplicates=0; out_of_order=0; previous={}
    for r in rows:
        seq=r.get("seq"); sender=r.get("sender_id"); key=(sender,seq)
        if r.get("sent",False): sent.add(key)
        if r.get("received",False):
            if key in received: duplicates+=1
            received.add(key)
            if sender in previous and seq is not None and seq < previous[sender]: out_of_order+=1
            if seq is not None: previous[sender]=seq
    explicit = sent_denominator is not None
    if explicit:
        if not isinstance(sent_denominator, (int, np.integer)) or isinstance(sent_denominator, bool):
            raise ValueError("sent denominator must be a nonnegative integer")
        denom=int(sent_denominator)
        if denom<0: raise ValueError("sent denominator must be nonnegative")
    else:
        denom=len(sent) if sent else None
    if denom is not None and len(received)>denom:
        raise ValueError("received unique packets exceed declared sent denominator")
    seqs=sorted(k[1] for k in received if isinstance(k[1],(int,np.integer)) and not isinstance(k[1],bool))
    known_seq = sorted(k[1] for k in (sent | received) if isinstance(k[1],(int,np.integer)) and not isinstance(k[1],bool))
    if denom is None or not known_seq:
        missing_sequences=None
    else:
        origin=min(known_seq) if explicit or known_seq else 0
        expected=set(range(origin, origin+denom))
        missing_sequences=sorted(expected-set(seqs))
    if denom is None:
        missing=loss_rate=None
    else:
        missing=denom-len(received); loss_rate=None if denom==0 else missing/denom
    return {"sent_unique":denom,"received_unique":len(received),"lost_unique":missing,"missing_sequences":missing_sequences,"loss_rate":loss_rate,"duplicates":duplicates,"out_of_order":out_of_order}

def packet_age(now, observation_time, *, same_clock=True):
    if not same_clock: raise ValueError("packet age requires a shared clock domain")
    age=float(now)-float(observation_time)
    if not math.isfinite(age) or age<0: raise ValueError("packet age must be finite and nonnegative")
    return age

def reacquisition_time(t, available, *, quality=None, consecutive=1):
    if isinstance(consecutive, bool) or not isinstance(consecutive,(int,np.integer)) or consecutive<1: raise ValueError("consecutive must be a positive integer")
    t,a=_xy(t,np.asarray(available,float))
    if np.any(~np.isfinite(a)) or np.any((a!=0)&(a!=1)): raise ValueError("available must contain only finite 0/1 values")
    a=a.astype(bool)
    if quality is not None:
        q=np.asarray(quality,float)
        if q.shape!=a.shape or np.any(~np.isfinite(q)) or np.any((q!=0)&(q!=1)): raise ValueError("quality must contain finite 0/1 values")
        a &= q.astype(bool)
    if not len(a): return {"time_s":None,"censored":True}
    miss=np.flatnonzero(~a)
    if not len(miss): return {"time_s":0.0,"censored":False}
    start=miss[0]
    for i in range(start+1,len(a)-consecutive+1):
        if np.all(a[i:i+consecutive]): return {"time_s":float(t[i]-t[start]),"censored":False}
    return {"time_s":None,"censored":True}

def controller_recovery_time(t, settled):
    t,s=_xy(t,np.asarray(settled,float))
    if np.any(~np.isfinite(s)) or np.any((s!=0)&(s!=1)): raise ValueError("settled must contain finite 0/1 values")
    idx=np.flatnonzero(s.astype(bool)); return {"time_s":None if not len(idx) else float(t[idx[0]]-t[0]),"censored":not bool(len(idx))}

def miss_distance_time(t, distance, *, capture_radius=0.0):
    t,d=_xy(t,distance)
    if np.any(~np.isfinite(d)) or np.any(d<0) or not math.isfinite(float(capture_radius)) or capture_radius<0: raise ValueError("distance and capture radius must be finite and nonnegative")
    i=np.flatnonzero(d<=capture_radius)
    return {"miss_distance_m":None if not len(d) else float(np.min(d)),"time_to_capture_s":None if not len(i) else float(t[i[0]]-t[0]),"censored":not bool(len(d)) or not bool(len(i))}

def failure_label(*, tracking_error_m=None, max_error_m=None, packet_loss=None, max_loss=None, scenario="unspecified"):
    failed=(tracking_error_m is not None and max_error_m is not None and tracking_error_m>max_error_m) or (packet_loss is not None and max_loss is not None and packet_loss>max_loss)
    return {"scenario":scenario,"status":"failure" if failed else "pass","criterion":"declared scenario thresholds"}
