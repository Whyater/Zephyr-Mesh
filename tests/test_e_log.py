import json, math
from sim.missions.log import (MissionLogRecorder, build_log, load_log, scenario_sha256,
                              validate_log, write_log)

def base():
    n=3; agents=["a0","a1"]
    ser={a:{"position_m":[[0,0,1]]*n,"velocity_mps":[[0,0,0]]*n,
      "failsafe_state":["nominal"]*n,"authority":["mission"]*n,
      "link_age_s":[0.0]*n,"estimate_position_m":None,"estimate_age_s":[None]*n} for a in agents}
    return build_log(schema="zephyr-mission-log-1",simulated=True,mission="tag",scenario={"x":1},code_revision="x",adapter="point_mass",preset="clean",seed=1,policy={},limits=[],dt_s=.01,log_every=1,agents=agents,frames={"time_s":[0,.1,.2],"target_position_m":None,"agents":ser},events=[],link_stats={},metrics={})

def test_example_valid(): validate_log(json.load(open("sim/missions/mission_log_example.json")))
def test_write_repeat_and_roundtrip(tmp_path):
    x=base(); p=tmp_path/"a.json"; q=tmp_path/"b.json"; write_log(p,x); write_log(q,x); assert p.read_bytes()==q.read_bytes(); assert load_log(p)==x
def test_extra_and_lengths_rejected():
    x=base(); x["frames"]["agents"]["a0"]["oops"]=1
    try: validate_log(x); assert False
    except ValueError: pass
    x=base(); x["frames"]["agents"]["a0"]["authority"]=[]
    try: validate_log(x); assert False
    except ValueError: pass
def test_unknown_event_kind_rejected():
    x=base(); x["events"]=[{"time_s":0,"kind":"unknown","agent":None,"detail":{}}]
    try: validate_log(x); assert False
    except ValueError: pass

def test_rounding_and_hash():
    x=base(); x["metrics"]={"nan":float("nan"),"neg":-0.0}; x=build_log(**x); assert x["metrics"]=={"nan":None,"neg":0.0}
    assert scenario_sha256({"a":1,"b":2})==scenario_sha256({"b":2,"a":1}); assert scenario_sha256({"a":1})!=scenario_sha256({"a":2})
def test_recorder_frames():
    r=MissionLogRecorder(["b","a"],.01,5)
    d={a:{"position_m":[0,0,0],"velocity_mps":[0,0,0],"failsafe_state":"nominal","authority":"mission","link_age_s":0,"estimate_position_m":None,"estimate_age_s":None} for a in ["a","b"]}
    for i in range(11): r.frame(i,i*.01,[0,0,0],d)
    assert r.to_log(schema="zephyr-mission-log-1",simulated=True,mission="tag",scenario={},code_revision="x",adapter="point_mass",preset="clean",seed=1,policy={},limits=[],events=[],link_stats={},metrics={})["frames"]["time_s"]==[0,.05,.1]
