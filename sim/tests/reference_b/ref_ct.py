"""Reference constant-turn EKF implementation."""
import numpy as np

def _step(x, dt):
    """Nonlinear CT transition and Jacobian."""
    x=np.asarray(x,float); w=float(x[6]); u=w*dt
    if abs(u)<1e-4:
        s=np.sin(u); c=np.cos(u)
        A=dt*(1-u**2/6+u**4/120); B=dt*(u/2-u**3/24+u**5/720)
        dA=dt**2*(-u/3+u**3/30); dB=dt**2*(.5-u**2/8+u**4/144)
    else:
        s=np.sin(u); c=np.cos(u); A=s/w; B=(1-c)/w
        dA=(dt*c-A)/w; dB=(dt*s-B)/w
    y=x.copy(); px,py,pz,vx,vy,vz=x[:6]
    y[0]=px+A*vx-B*vy; y[1]=py+B*vx+A*vy; y[2]=pz+vz*dt
    y[3]=c*vx-s*vy; y[4]=s*vx+c*vy; y[5]=vz; y[6]=w
    F=np.eye(7); F[0,3]=A; F[0,4]=-B; F[0,6]=dA*vx-dB*vy
    F[1,3]=B; F[1,4]=A; F[1,6]=dB*vx+dA*vy; F[2,5]=dt
    F[3,3]=c; F[3,4]=-s; F[3,6]=dt*(-s*vx-c*vy)
    F[4,3]=s; F[4,4]=c; F[4,6]=dt*(c*vx-s*vy)
    return y,F

def _Q(dt,q_acc,q_turn):
    """Continuous white noise acceleration and turn-rate covariance."""
    Q=np.zeros((7,7)); b=q_acc*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
    for p,v in ((0,3),(1,4),(2,5)): Q[np.ix_([p,v],[p,v])]=b
    Q[6,6]=q_turn*dt; return Q

def predict(state,P,t_from,t_to,q_acc=0.1,q_turn=0.1):
    """EKF prediction with nonlinear CT transition and Jacobian."""
    dt=float(t_to)-float(t_from)
    if dt<0: raise ValueError('t_to must be >= t_from')
    x,F=_step(np.asarray(state,float),dt); Pn=F@np.asarray(P,float)@F.T+_Q(dt,q_acc,q_turn)
    return x,(Pn+Pn.T)/2

def run_ct(measurements,stamps,start_position,q_acc=0.1,r_std=0.05,p0_pos=1.0,p0_vel=1.0,query=None,known_turn_rate=None,q_turn=0.1):
    """CT EKF updates; returns (state,P) after each update."""
    x=np.zeros(7); x[:3]=np.asarray(start_position,float)
    if known_turn_rate is not None: x[6]=float(known_turn_rate); q_turn=0.0
    P=np.diag([p0_pos]*3+[p0_vel]*3+[0.0 if known_turn_rate is not None else 1.0]); tcur=float(stamps[0]) if len(stamps) else 0.0
    out=[]; R=(r_std**2)*np.eye(3)
    for z,t in zip(measurements,stamps):
        x,P=predict(x,P,tcur,float(t),q_acc,q_turn); tcur=float(t)
        z=np.asarray(z,float); nu=z-x[:3]; S=P[:3,:3]+R; K=P[:,:3]@np.linalg.inv(S); x=x+K@nu
        I=np.eye(7); H=np.zeros((3,7)); H[:,:3]=np.eye(3); P=(I-K@H)@P@(I-K@H).T+K@R@K.T; P=(P+P.T)/2
        out.append((x.copy(),P.copy()))
    if query is not None:
        return [predict(x, P, tcur, float(tq), q_acc, q_turn) for tq in query]
    return out
