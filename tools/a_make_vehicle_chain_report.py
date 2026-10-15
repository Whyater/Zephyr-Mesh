"""Generate a deterministic simulated EXP-020 vehicle-chain report."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sim.environment import GRAVITY
from sim.hardware import load_hardware_profile, RotorcraftProfile
from sim.actuator import RotorcraftActuator
from sim.vehicle import ActuatedQuadrotor
from sim.physics.run_manifest import write_manifest
from tools import horizon_mpl

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "sim" / "physics" / "profiles"
PROFILE_NAMES = ("crazyflie-class", "esp-fly-class")
PASS = {"relative": 0.01, "absolute": 1e-12, "position": 1e-3}


def _profile(name):
    return load_hardware_profile(PROFILE_DIR / f"{name.replace('-', '_')}.json")


def _entry(sim, hand, threshold, mode="relative"):
    sim, hand = float(sim), float(hand)
    error = abs(sim - hand) if mode == "absolute" else abs(sim - hand) / max(abs(hand), 1e-30)
    return {"sim_value": sim, "hand_value": hand, "error": error, "threshold": float(threshold), "pass": bool(error < threshold)}


def check_hover(profile, duration=0.1):
    vehicle = ActuatedQuadrotor(profile)
    speed = profile.hover_speed_rad_s()
    vehicle.actuator.speed_rad_s[:] = speed
    state = vehicle.actuator._state(False, 1.0)
    per = float(state.thrust_n[0])
    hand_thrust = profile.airframe.mass_kg * GRAVITY / 4.0
    hand_speed = math.sqrt(profile.airframe.mass_kg * GRAVITY / (4.0 * profile.rotor.thrust_coefficient_n_s2))
    return {"per_rotor_thrust_n": _entry(per, hand_thrust, PASS["relative"]), "hover_speed_rad_s": _entry(speed, hand_speed, PASS["relative"])}


def check_motor_step(profile, duration=None, dt=0.001):
    tau = profile.rotor.motor_time_constant_s
    if duration is None: duration = 5.0 * tau
    act = RotorcraftActuator(profile); w0 = profile.hover_speed_rad_s(); w1 = w0 + 0.20 * (profile.max_speed_rad_s - w0)
    act.speed_rad_s[:] = w0
    duty = w1 / profile.max_speed_rad_s
    rows=[]; t=0.0; n=max(1, int(round(duration/dt)))
    hover_duty = w0 / profile.max_speed_rad_s
    for _ in range(n):
        state=act.step((duty, hover_duty, hover_duty, hover_duty), dt); t += dt
        hand=w0+(w1-w0)*(1.0-math.exp(-t/tau)); rows.append((t,float(state.speed_rad_s[0]),hand))
    max_dev=max(abs(s-h) for _,s,h in rows)/abs(w1-w0)
    at_tau=min(rows,key=lambda x:abs(x[0]-tau)); at_3=min(rows,key=lambda x:abs(x[0]-3*tau))
    deviation = {"sim_value": float(max_dev), "hand_value": 0.0, "error": float(max_dev), "threshold": PASS["relative"], "pass": bool(max_dev < PASS["relative"])}
    return {"command": "one rotor: hover + 20% of (w_max - hover)", "max_first_order_deviation": deviation, "speed_fraction_at_tau": _entry((at_tau[1]-w0)/(w1-w0),1-math.exp(-1),PASS["relative"]), "speed_fraction_at_3tau": _entry((at_3[1]-w0)/(w1-w0),1-math.exp(-3),PASS["relative"]), "trace": rows}


def check_max_climb(profile, settle=None, window=0.2, dt=0.01):
    """Fly the full chain at full throttle and measure the climb acceleration.

    The goal is 50 m straight up, so the controller asks for far more than the
    rotors can give and the mixer clips every rotor at its maximum. After ten
    motor time constants the rotors are at full speed, and the vertical
    acceleration is the change in vertical speed over a short window.
    """
    tau = profile.rotor.motor_time_constant_s
    if settle is None: settle = 10.0 * tau
    vehicle = ActuatedQuadrotor(profile, initial_position=(0.0, 0.0, 1.0), substeps=1)
    vehicle.actuator.speed_rad_s[:] = profile.hover_speed_rad_s()
    goal = (0.0, 0.0, 50.0)
    for _ in range(int(round(settle / dt))): vehicle.step(goal, dt)
    v0 = float(vehicle.drone.velocity[2]); saturated = True
    for _ in range(int(round(window / dt))): saturated = saturated and vehicle.step(goal, dt).mixer.saturated
    sim = (float(vehicle.drone.velocity[2]) - v0) / window
    hand = 4*profile.rotor.thrust_coefficient_n_s2*profile.max_speed_rad_s**2/profile.mass_kg-GRAVITY
    return {"vertical_acceleration_mps2": _entry(sim,hand,PASS["relative"]), "all_window_steps_saturated": bool(saturated),
            "settle_s": float(settle), "window_s": float(window)}


def _battery_profile(profile):
    raw=profile.as_document(); raw["status"]="fixture"; raw["battery"]={"nominal_voltage_v":4.0,"cutoff_voltage_v":3.0,"capacity_mah":5.0,"internal_resistance_ohm":0.15,"efficiency":0.8}
    return RotorcraftProfile.from_dict(raw)


def check_battery_sag(profile, duration=2.0, dt=0.01):
    p=_battery_profile(profile); vehicle=ActuatedQuadrotor(p, initial_position=(0.0, 0.0, 1.0), substeps=1); speed=p.hover_speed_rad_s(); vehicle.actuator.speed_rad_s[:]=speed
    rows=[]; states=5; sample_steps=np.linspace(1, max(1,int(duration/dt)), states,dtype=int)
    for i in range(1,int(round(duration/dt))+1):
        st=vehicle.step((0.0, 0.0, 1.0), dt).actuator
        if i in set(sample_steps):
            soc=float(st.battery_charge_ah/(p.battery.capacity_mah / 1000.0)); current=float(st.mean_power_w/(p.battery.efficiency*max(st.battery_voltage_v,1e-9)))
            voc=p.battery.cutoff_voltage_v+(p.battery.nominal_voltage_v-p.battery.cutoff_voltage_v)*soc
            hand_v=voc-current*p.battery.internal_resistance_ohm
            rows.append({"soc":soc,"terminal_voltage":_entry(st.battery_voltage_v,hand_v,PASS["relative"]),"duty":float(np.mean(st.duty)),"speed_rad_s":float(np.mean(st.speed_rad_s))})
    speeds=[r["speed_rad_s"] for r in rows]; duties=[r["duty"] for r in rows]
    return {"airframe":profile.profile_id,"battery_label":"illustrative test battery, not a profile value","hand_current":"mean mechanical power / (efficiency x terminal voltage)","samples":rows,"hover_speed_constant":max(speeds)-min(speeds) < 0.01*speed,"hover_duty_monotonic":all(b>=a for a,b in zip(duties,duties[1:]))}


def check_yaw_trim(profile, duration=10.0, dt=0.01):
    act=RotorcraftActuator(profile); act.speed_rad_s[:]=profile.hover_speed_rad_s(); st=act._state(False,1.0)
    vehicle=ActuatedQuadrotor(profile,initial_position=(0,0,1),substeps=1); vehicle.actuator.speed_rad_s[:]=profile.hover_speed_rad_s()
    for _ in range(int(round(duration/dt))): vehicle.step((0,0,1),dt)
    return {"reaction_yaw_torque_nm": _entry(st.body_torque_nm[2],0.0,PASS["absolute"],"absolute"),"yaw_rate_after_s":float(vehicle.drone.angular_velocity[2])}


def check_hover_hold(profile, durations=(30.0,), dts=(0.01,0.005,0.002)):
    out={}
    for dt in dts:
        duration=durations[0] if len(durations)==1 else durations[dts.index(dt)]
        v=ActuatedQuadrotor(profile,initial_position=(0,0,1),substeps=1); v.actuator.speed_rad_s[:]=profile.hover_speed_rad_s(); initial=v.drone.position.copy(); md=0.0
        for _ in range(int(round(duration/dt))): v.step((0,0,1),dt); md=max(md,float(np.linalg.norm(v.drone.position-initial)))
        out[str(dt)]={"max_position_drift_m":_entry(md,0.0,PASS["position"],"absolute")}
    return out


def check_substeps(profile, duration=10.0, dt=0.01):
    def run(n):
        v=ActuatedQuadrotor(profile,initial_position=(0,0,1),substeps=n); v.actuator.speed_rad_s[:]=profile.hover_speed_rad_s(); arr=[]
        for _ in range(int(round(duration/dt))): arr.append(v.step((1,0,1),dt).position_m)
        return np.asarray(arr)
    ref=run(50); vals={}
    for n in (1,2,5):
        rms=float(np.sqrt(np.mean(np.sum((run(n)-ref)**2,axis=1)))); vals[str(n)]={"rms_position_error_m":_entry(rms,0.0,PASS["position"],"absolute")}
    passing=[int(n) for n,e in vals.items() if e["rms_position_error_m"]["pass"]]
    return {"reference_substeps":50,"reference_step_s":0.0002,"substeps":vals,"smallest_passing":min(passing) if passing else None}


def check_profile(profile, duration_scale=1.0):
    """Run the report checks for one profile, scaling long integrations for tests."""
    scale=float(duration_scale)
    return {"hover":check_hover(profile),
            "motor_step":check_motor_step(profile, duration=5.0*profile.rotor.motor_time_constant_s*max(1.0, scale)),
            "max_climb":check_max_climb(profile),
            "battery_sag":check_battery_sag(profile, duration=2.0*scale),
            "yaw_trim":check_yaw_trim(profile, duration=10.0*scale),
            "hover_hold":check_hover_hold(profile, durations=(30.0*scale,)),
            "substeps":check_substeps(profile, duration=10.0*scale)}


def _round(v):
    if isinstance(v,float): return float(f"{v:.9g}")
    if isinstance(v,dict): return {k:_round(x) for k,x in v.items() if k != "trace"}
    if isinstance(v,list): return [_round(x) for x in v]
    return v


def _figure(data,path):
    horizon_mpl.apply_style("light"); pal=horizon_mpl.palette("light"); fig,ax=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for name, d in data.items():
        tr=d["motor_step"]["trace"]; ax[0].plot([x[0] for x in tr],[x[1] for x in tr],".",ms=2,label=f"{name} sim",color=pal["accent"] if name.startswith("crazy") else pal["info"]); ax[0].plot([x[0] for x in tr],[x[2] for x in tr],color=pal["text"])
        xs=sorted(int(k) for k in d["substeps"]["substeps"]); ys=[d["substeps"]["substeps"][str(x)]["rms_position_error_m"]["sim_value"] for x in xs]; ax[1].plot(xs,ys,"o-",label=name,color=pal["accent"] if name.startswith("crazy") else pal["info"])
    ax[0].set(title="Motor step (simulated)",xlabel="time (s)",ylabel="speed (rad/s)"); ax[0].legend(fontsize=7); ax[1].set(title="Substep RMS error (simulated)",xlabel="substeps",ylabel="RMS position error (m)",yscale="log"); ax[1].axhline(1e-3,color=pal["alert"],ls="--",label="1 mm threshold"); ax[1].legend(fontsize=7)
    fig.savefig(path,dpi=150,metadata={"Software":None}); plt.close(fig)


def generate(output_dir:Path, *, durations=None):
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(f"refusing to write into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); data={}
    for name in PROFILE_NAMES:
        p=_profile(name); data[name]={"hover":check_hover(p),"motor_step":check_motor_step(p,duration=(durations or {}).get("motor_step")),"max_climb":check_max_climb(p),"yaw_trim":check_yaw_trim(p),"hover_hold":check_hover_hold(p,durations=((durations or {}).get("hover_hold",30.0),)),"substeps":check_substeps(p,duration=(durations or {}).get("substeps",10.0))}
    data["battery_sag"]=check_battery_sag(_profile("crazyflie-class"),duration=(durations or {}).get("battery",2.0))
    report=_round({"schema":"zephyr-exp-020-vehicle-chain-report-1","profiles":data,"notes":["simulated only","literature-derived profiles","illustrative battery","50 substeps of a 10 ms control step gives a 0.2 ms reference, finer than the preregistered 1 ms reference"]})
    (output_dir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _figure({k:data[k] for k in PROFILE_NAMES}, output_dir/"a4_vehicle_chain.png")
    write_manifest(output_dir/"manifest.json",schema="zephyr-exp-020-vehicle-chain-manifest-1",argv=sys.argv[1:],config={"profiles":PROFILE_NAMES},seeds=[],limits=["simulated only","literature-derived profiles","illustrative battery"],outputs=[output_dir/"report.json",output_dir/"a4_vehicle_chain.png"],output_dir=output_dir,runtime_s=time.perf_counter()-started)
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",required=True,type=Path); args=ap.parse_args(argv); generate(args.output_dir); return 0
if __name__=="__main__": raise SystemExit(main())
