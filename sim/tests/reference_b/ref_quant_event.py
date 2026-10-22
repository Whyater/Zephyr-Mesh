"""Reference quantization and event trigger."""
import numpy as np

def quantize_int16(z,q):
    """Quantize and dequantize values using int16 codes."""
    z=np.asarray(z,float)
    code=np.clip(np.round(z/float(q)),-32768,32767).astype(np.int16)
    return code.astype(float)*float(q)

def event_trigger(capture_times,zs,delta,q_acc=0.1,r_std=0.05,heartbeat=None,quantization=None):
    """Return indices sent by event trigger with a replica CV filter."""
    from .ref_cv import run_cv, predict
    ts=np.asarray(capture_times,float); Z=np.asarray(zs,float)
    if len(ts)!=len(Z): raise ValueError("capture_times and zs length mismatch")
    if len(ts)==0:return []
    sent=[]; sent_t=[]; sent_z=[]; state=P=None; last_t=None
    for i,(t,z) in enumerate(zip(ts,Z)):
        zsend=quantize_int16(z,quantization) if quantization is not None else np.asarray(z,float)
        trigger=False
        if i==0: trigger=True
        else:
            # predict replica posterior to current capture time
            pred_state,pred_P=predict(state,P,sent_t[-1],float(t),q_acc)
            err=np.linalg.norm(np.asarray(z)-pred_state[:3])
            trigger=err>=float(delta)
            if heartbeat is not None and (float(t)-float(last_t)>=float(heartbeat)-1e-12): trigger=True
        if trigger:
            sent.append(i); sent_t.append(float(t)); sent_z.append(np.asarray(zsend,float)); last_t=float(t)
            hist=run_cv(np.asarray(sent_z),np.asarray(sent_t),start_position=np.asarray(sent_z[0]),q_acc=q_acc,r_std=r_std)
            state,P=hist[-1]
    return sent
