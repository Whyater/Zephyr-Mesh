"""Reference NTP clock exchange calculations."""
from collections import namedtuple

ExchangeResult = namedtuple("ExchangeResult", "t1 t2 t3 t4 theta")

def offset_round_trip(t1,t2,t3,t4):
    """Return (theta, delta) from four timestamps."""
    theta=((t2-t1)+(t3-t4))/2.0
    delta=(t4-t1)-(t3-t2)
    return float(theta),float(delta)

def simulate_exchange(e0, drift_ppm, d_up, d_down, processing, t_send):
    """Return t1,t2,t3,t4,theta for one exchange."""
    r=float(drift_ppm)*1e-6
    t1=float(t_send)
    t2_true=t1+float(d_up); t2=t2_true+float(e0)+r*t2_true
    t3_true=t2_true+float(processing); t3=t3_true+float(e0)+r*t3_true
    t4=float(t3_true+float(d_down))
    theta,delta=offset_round_trip(t1,t2,t3,t4)
    return ExchangeResult(t1,t2,t3,t4,theta)

def ntp_exchange(*args,**kwargs):
    """Alias for the exchange simulation."""
    return simulate_exchange(*args,**kwargs)
offset_and_round_trip = offset_round_trip
simulate_ntp = simulate_exchange
