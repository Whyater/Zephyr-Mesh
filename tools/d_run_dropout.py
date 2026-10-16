from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, CoopSwarmSimulator
from sim.estimation.sweep import run_jobs
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import MOVING_AGENT_LIMITS, argv_or_default, default_processes, prepare_output, seeds_for, write_json, write_manifest
THRESHOLD_PATH=Path('runs/c5-heartbeat/thresholds.json')

def _thresholds():
    data=json.loads(THRESHOLD_PATH.read_text(encoding='utf-8')); out={}
    for row in data['records']:
        c=row.get('loss_condition','')
        if row.get('h_hz')==10 and c in {'independent_p_0.01','independent_p_0.1','independent_p_0.3'}: out[float(c.rsplit('_',1)[1])]=float(row['false_alarm_below_1_per_hour']['T_s'])
    return out

def _gap_deg(positions,target):
    if not positions:return 360.0
    a=sorted(float(np.arctan2(p[1]-target[1],p[0]-target[0]))%(2*np.pi) for p in positions)
    return float(np.degrees(max([a[i+1]-a[i] for i in range(len(a)-1)]+[2*np.pi+a[0]-a[-1]])))

def _job(job):
    n,k,mode,loss,seed,timeout=int(job['n']),int(job['k']),job['mode'],float(job['link_loss']),int(job['seed']),float(job['timeout_s'])
    dropped=tuple((f'drone-{i:03d}',10.0,mode) for i in range(n-k+1,n+1))
    cfg=CoopSwarmConfig(n_agents=n,seed=seed,duration_s=25.0,control_dt_s=0.02,sensor_rate_hz=10.0,estimator='cv',fusion='ci',decision='onboard',target='ct',target_params=(('radius',3.0),('speed',0.8)),standoff_radius_m=1.5,link=CoopLinkConfig(delay_s=0.05,loss_probability=loss,loss_model='independent',share_rate_hz=10.0),heartbeat_timeout_s=timeout,dropouts=dropped,log_every=1,metric_windows=((5.0,10.0),(10.0,15.0)),ci_mode='fast')
    steps=CoopSwarmSimulator(cfg).run().as_dict()['steps']; survivors=tuple(f'drone-{i:03d}' for i in range(1,n-k+1)); dropped_ids={x[0] for x in dropped}
    detection=None
    for s in steps:
        if s['time_s']<10: continue
        rows={a['agent_id']:a for a in s['agents']}
        if all(dropped_ids.isdisjoint(rows[a]['believed_alive']) for a in survivors): detection=float(s['time_s']-10); break
    good=[]; threshold=360/len(survivors)+15
    timeline=[]
    for s in steps:
        t=float(s['time_s']); rows={a['agent_id']:a for a in s['agents']}
        if 10<=t<15:
            target=np.asarray(s['target_truth_position_m']); gap=_gap_deg([np.asarray(rows[a]['position_m']) for a in survivors],target)
            good.append((t,gap<=threshold and all(float(rows[a]['standoff_error_m'])<=.3 for a in survivors),gap)); timeline.append((t,gap))
    recovery=None
    for i,(t,ok,_) in enumerate(good):
        if ok and all(flag for ts,flag,_ in good[i:] if ts<t+1-1e-9): recovery=t-10; break
    def stats(lo,hi):
        v=[float(s['gs_group_error_m']) for s in steps if lo<=s['time_s']<hi and s['gs_group_error_m'] is not None]
        return (float(np.percentile(v,95)),float(np.max(v))) if v else (None,None)
    b95,bmx=stats(5,10); g95,gmx=stats(10,15); prev={}; role=plan=0; minsep=float('inf'); viol=0
    for s in steps:
        t=float(s['time_s']); rows={a['agent_id']:a for a in s['agents']}; pos={a:np.asarray(rows[a]['position_m']) for a in survivors}; ids=sorted(pos)
        if 10<=t<15:
            ass={a:rows[a]['believed_assignment'] for a in survivors}; role+=sum(1 for a in survivors if a in prev and ass[a]!=prev[a]); prev=ass; plan+=int(bool(s.get('plan_disagreement',False)))
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                d=float(np.linalg.norm(pos[a]-pos[b])); minsep=min(minsep,d); viol+=int(d<.35)
    return {'n':n,'k':k,'mode':mode,'link_loss':loss,'seed':seed,'detection_time_s':detection,'recoverage_time_s':recovery,'never_detected':detection is None,'gs_before_p95_m':b95,'gs_before_max_m':bmx,'gs_gap_p95_m':g95,'gs_gap_max_m':gmx,'failed_in_gap':bool(g95 is not None and (g95>.25 or gmx>.5)),'role_changes_after_drop':role,'plan_disagreement_steps':plan,'min_pairwise_true_separation_m':None if not np.isfinite(minsep) else minsep,'separation_violations':viol,'violation':bool(viol),'hand_detection_s':timeout+0.1*loss/(1-loss),'timeline':timeline}

CSV_COLUMNS=['n','k','mode','link_loss','seed','detection_time_s','recoverage_time_s','never_detected','gs_before_p95_m','gs_before_max_m','gs_gap_p95_m','gs_gap_max_m','failed_in_gap','role_changes_after_drop','plan_disagreement_steps','min_pairwise_true_separation_m','separation_violations','violation','hand_detection_s']
def _write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=CSV_COLUMNS,lineterminator='\n'); w.writeheader(); w.writerows({k:r[k] for k in CSV_COLUMNS} for r in rows)

def run(output,argv,pilot=False,processes=None):
    start=time.perf_counter(); output=prepare_output(output); th=_thresholds(); losses=(0.0,0.3) if pilot else (0.0,0.1,0.3)
    cells=[(4,1,'silent',x) for x in losses] if pilot else [(n,k,m,l) for n in (3,4,6) for k in (1,2) if k<=n-2 for m in ('silent','landed') for l in (0.0,0.1,0.3)]
    seeds=seeds_for('dropout',3 if pilot else 50); jobs=[{'n':n,'k':k,'mode':m,'link_loss':l,'seed':s,'timeout_s':th[0.01 if l==0.0 else l]} for n,k,m,l in cells for s in seeds]; nproc=processes or default_processes()
    rows=run_jobs(jobs,'tools.d_run_dropout:_job',processes=nproc,progress=False)
    if len(rows)!=len(jobs) or any(r is None for r in rows): raise RuntimeError('unexpected result row count')
    _write_csv(output/'per_seed.csv',rows); summary=[]
    for key in sorted({(r['n'],r['k'],r['mode'],r['link_loss']) for r in rows}):
        g=[r for r in rows if tuple(r[x] for x in ('n','k','mode','link_loss'))==key]
        def med(name):
            v=[r[name] for r in g if r[name] is not None]; return list(median_interval(v)) if v else None
        summary.append({'n':key[0],'k':key[1],'mode':key[2],'link_loss':key[3],'n_seeds':len(g),'detection_time_s':med('detection_time_s'),'recoverage_time_s':med('recoverage_time_s'),'gs_before_p95_m':med('gs_before_p95_m'),'gs_gap_p95_m':med('gs_gap_p95_m'),'never_detected_rate':list(wilson_interval(sum(r['never_detected'] for r in g),len(g))),'failed_in_gap_rate':list(wilson_interval(sum(r['failed_in_gap'] for r in g),len(g))),'violation_rate':list(wilson_interval(sum(r['violation'] for r in g),len(g)))})
    write_json(output/'summary.json',summary)
    from tools.d_make_dropout_figure import make_figures
    make_figures(output,rows,summary)
    config={'grid':cells,'heartbeat_rate_hz':10,'heartbeat_thresholds_s':th,'thresholds_sha256':hashlib.sha256(THRESHOLD_PATH.read_bytes()).hexdigest(),'control_dt_s':0.02,'log_every':1,'hand_detection_formula':'T + delay_s - mean_phase_s + period * p/(1-p)','seeds_count':len(seeds)}
    return write_manifest(output,schema='zephyr-d-dropout-manifest-1',argv=argv,command='PYTHONPATH=. python tools/d_run_dropout.py',config=config,seeds=seeds,outputs=['per_seed.csv','summary.json','dropout_times.png','dropout_timeline.png'],limits=list(MOVING_AGENT_LIMITS)+['Heartbeat loss, timeout, and dropout modes are simulated; geometry is kinematic.'],runtime_s=time.perf_counter()-start,n_processes=nproc,hashed_sources=['tools/d_run_dropout.py','tools/d_make_dropout_figure.py','sim/coop/simulator.py','sim/s7_coop_run_schema.json','runs/c5-heartbeat/thresholds.json'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--pilot',action='store_true'); p.add_argument('--processes',type=int); a=p.parse_args(); run(Path(a.output),argv_or_default(None),a.pilot,a.processes)
