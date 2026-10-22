"""Reference two-mode IMM."""
import numpy as np
from .ref_ct import predict as ct_predict, _step, _Q

def _cv_predict(x,P,dt,q_acc,q_turn):
    """CV mode prediction in the seven-dimensional state."""
    F=np.eye(7); F[0,3]=F[1,4]=F[2,5]=dt; Q=_Q(dt,q_acc,q_turn)
    return F@x, (F@P@F.T+Q + (F@P@F.T+Q).T)/2

def _update(x,P,z,R):
    """Joseph position measurement update."""
    H=np.zeros((3,7)); H[:,:3]=np.eye(3); nu=z-x[:3]; S=P[:3,:3]+R; K=P[:,:3]@np.linalg.inv(S); x=x+K@nu
    I=np.eye(7); P=(I-K@H)@P@(I-K@H).T+K@R@K.T; return x,(P+P.T)/2,nu,S

def predict_combined(modes,mu,t_from,t_to,q_acc=0.1,q_turn=0.1):
    """predict_at rule: propagate current modes without mixing, combine with c_j."""
    dt=float(t_to)-float(t_from); mu=np.asarray(mu,float); p=(1.0-np.exp(-2.0*dt/2.0))/2.0; Pi=np.array([[1-p,p],[p,1-p]]); c=Pi.T@mu
    xs=[]; Ps=[]
    for j,(x,P) in enumerate(modes): xs.append(_cv_predict(x,P,dt,q_acc,q_turn)[0] if j==0 else ct_predict(x,P,0,dt,q_acc,q_turn)[0]); Ps.append(_cv_predict(x,P,dt,q_acc,q_turn)[1] if j==0 else ct_predict(x,P,0,dt,q_acc,q_turn)[1])
    xc=sum(c[j]*xs[j] for j in range(2)); Pc=sum(c[j]*(Ps[j]+np.outer(xs[j]-xc,xs[j]-xc)) for j in range(2)); return xc,Pc

def run_imm(measurements,stamps,start_position,q_acc=0.1,r_std=0.05,p0_pos=1.0,p0_vel=1.0,query=None,known_turn_rate=None,q_turn=0.1):
    """IMM updates; returns list of (combined state, covariance, mode probabilities)."""
    x0=np.zeros(7); x0[:3]=start_position; x0[6]=0 if known_turn_rate is None else known_turn_rate
    P0=np.diag([p0_pos]*3+[p0_vel]*3+[0 if known_turn_rate is not None else 1.0]); modes=[(x0.copy(),P0.copy()),(x0.copy(),P0.copy())]; mu=np.array([.5,.5]); tcur=float(stamps[0]) if len(stamps) else 0.; R=r_std**2*np.eye(3); out=[]
    for z,t in zip(measurements,stamps):
        dt=float(t)-tcur; p=(1.0-np.exp(-2.0*dt/2.0))/2.0; Pi=np.array([[1-p,p],[p,1-p]]); c=Pi.T@mu; pred=[]
        for j in range(2):
            weights=Pi[:,j]*mu/c[j]; xm=sum(weights[i]*modes[i][0] for i in range(2)); Pm=sum(weights[i]*(modes[i][1]+np.outer(modes[i][0]-xm,modes[i][0]-xm)) for i in range(2)); pred.append(_cv_predict(xm,Pm,dt,q_acc,0 if known_turn_rate is not None else q_turn) if j==0 else ct_predict(xm,Pm,0,dt,q_acc,0 if known_turn_rate is not None else q_turn))
        logs=[]; upd=[]
        for j in range(2):
            xx,PP=pred[j]; zz=np.asarray(z,float); nu=zz-xx[:3]; S=PP[:3,:3]+R; sign,ld=np.linalg.slogdet(S); logs.append(-.5*nu@np.linalg.solve(S,nu)-.5*ld-1.5*np.log(2*np.pi)); upd.append(_update(xx,PP,zz,R)[:2])
        lw=np.log(c)+logs; lw-=np.max(lw); mu=np.exp(lw); mu/=mu.sum(); modes=upd; tcur=float(t)
        xc=sum(mu[j]*modes[j][0] for j in range(2)); Pc=sum(mu[j]*(modes[j][1]+np.outer(modes[j][0]-xc,modes[j][0]-xc)) for j in range(2)); out.append((xc.copy(),Pc.copy(),mu.copy()))
    if query is not None:
        query_out=[]
        query_q_turn=0.0 if known_turn_rate is not None else q_turn
        for tq in query:
            xc,Pc=predict_combined(modes,mu,tcur,float(tq),q_acc,query_q_turn)
            query_out.append((xc,Pc,mu.copy()))
        return query_out
    return out
