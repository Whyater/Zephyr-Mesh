"""Run the simulated ground versus onboard decision experiment."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from sim.coop.simulator import CoopLinkConfig, CoopSwarmConfig, run as sim_run
from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import argv_or_default, default_processes, prepare_output, seeds_for, write_json, write_manifest
MODES=("ground","onboard"); LOSSES=(0.0,0.1,0.3); OUTAGES=("none","10_15"); PILOT_LOSSES=(0.0,0.3)
def _config(job):
    return CoopSwarmConfig(n_agents=4,seed=int(job['seed']),duration_s=20.0,control_dt_s=0.02,sensor_rate_hz=10.0,
      estimator='cv',fusion='central' if job['mode']=='ground' else 'ci',decision=job['mode'],
      link=CoopLinkConfig(delay_s=0.05,loss_probability=float(job['loss']),loss_model='independent',share_rate_hz=10.0),
      standoff_radius_m=1.5,heartbeat_timeout_s=0.5,log_every=1,gs_outage=() if job['outage']=='none' else ((10.0,15.0),),dropouts=(),ci_mode='fast')
def _worker(job):
    raw=sim_run(_config(job)).as_dict(); steps=raw['steps']; scored=[s for s in steps if float(s['time_s'])>=2.0-1e-12]
    if job['mode']=='ground':
        decision=[float(s['gs_fuser_error_m']) for s in scored if s.get('gs_fuser_error_m') is not None]
    else:
        decision=[float(np.median([a['truth_error_m'] for a in s['agents'] if a.get('active') and a.get('truth_error_m') is not None])) for s in scored if any(a.get('active') for a in s['agents'])]
    if not decision: raise RuntimeError('no scored decision estimates')
    def vals(lo,hi,key='standoff_error_m'):
        return [float(a[key]) for s in steps if lo<=float(s['time_s'])<hi for a in s['agents'] if a.get('active') and a.get(key) is not None]
    allv=vals(0,20); v10=vals(10,15); v15=vals(15,20); cov=vals(10,15,'coverage_largest_gap_deg')
    active_steps=sum(1 for s in steps for a in s['agents'] if a.get('active')); timeout=sum('command_timeout' in a.get('flags',[]) for s in steps for a in s['agents'] if a.get('active'))
    traffic=raw['summary']['traffic']; kinds=('raw','track','command') if job['mode']=='ground' else ('state','leaving')
    msgs=sum(int(traffic[k]['messages']) for k in kinds); byt=sum(int(traffic[k]['bytes']) for k in kinds)
    timeline=[]
    for s in steps:
        x=[float(a['standoff_error_m']) for a in s['agents'] if a.get('active') and a.get('standoff_error_m') is not None]
        if x: timeline.append({'mode':job['mode'],'loss':float(job['loss']),'outage':job['outage'],'seed':int(job['seed']),'time_s':float(s['time_s']),'standoff_error_m':float(np.median(x))})
    return {'mode':job['mode'],'loss':float(job['loss']),'outage':job['outage'],'seed':int(job['seed']),
      'decision_estimate_error_p95_m':float(np.percentile(decision,95)),'gs_group_estimate_p95_m':raw['summary']['gs_group_error_p95_m'],
      'standoff_error_m':float(np.median(allv)),'standoff_error_10_15_m':float(np.median(v10)),'standoff_error_15_20_m':float(np.median(v15)),
      'standoff_error_p95_10_15_m':float(np.percentile(v10,95)),'coverage_largest_gap_median_deg':float(np.median(cov)),
      'min_true_separation_m':raw['summary']['min_pairwise_separation_m'],'violation_count':int(raw['summary']['spherical_violations_count']),
      'plan_disagreement_fraction':float(raw['summary']['plan_disagreement_fraction']),'command_timeout_fraction':float(timeout/max(1,active_steps)),
      'messages_per_second':float(msgs/20.0),'bytes_per_second':float(byt/20.0),'timeline':timeline}
COLS=['mode','loss','outage','seed','decision_estimate_error_p95_m','gs_group_estimate_p95_m','standoff_error_m','standoff_error_10_15_m','standoff_error_15_20_m','standoff_error_p95_10_15_m','coverage_largest_gap_median_deg','min_true_separation_m','violation_count','plan_disagreement_fraction','command_timeout_fraction','messages_per_second','bytes_per_second']
def _summary(rows):
    cells=[]
    for mode in MODES:
      for loss in sorted({r['loss'] for r in rows}):
       for outage in OUTAGES:
        rr=[r for r in rows if r['mode']==mode and r['loss']==loss and r['outage']==outage]
        if not rr: continue
        c={'mode':mode,'loss':loss,'outage':outage,'n_seeds':len(rr)}
        for key in COLS[4:]:
            vv=[float(r[key]) for r in rr]; c[key]=median_interval(vv)
            if key in ('violation_count','plan_disagreement_fraction','command_timeout_fraction'): c[key+'_wilson']=wilson_interval(sum(v>0 for v in vv),len(vv))
        cells.append(c)
    paired=[]
    for loss in sorted({r['loss'] for r in rows}):
      for outage in OUTAGES:
        g={r['seed']:r['standoff_error_10_15_m'] for r in rows if r['mode']=='ground' and r['loss']==loss and r['outage']==outage}; o={r['seed']:r['standoff_error_10_15_m'] for r in rows if r['mode']=='onboard' and r['loss']==loss and r['outage']==outage}; d=[g[s]-o[s] for s in sorted(set(g)&set(o))]
        if d: paired.append({'loss':loss,'outage':outage,'median_ground_minus_onboard_standoff_10_15_m':median_interval(d)})
    return {'cells':cells,'paired_differences':paired}
def run(output:Path,*,pilot=False,processes=None,argv=None):
    start=time.perf_counter(); cpu_start=time.process_time(); output=prepare_output(output); n=3 if pilot else 50; seeds=seeds_for('decisions',n); losses=PILOT_LOSSES if pilot else LOSSES; nproc=min(4, int(processes or default_processes()));
    if nproc < 1: raise ValueError('processes must be positive')
    jobs=[{'mode':m,'loss':l,'outage':o,'seed':s} for m in MODES for l in losses for o in OUTAGES for s in seeds]; rows=run_jobs(jobs,'tools.d_run_decisions:_worker',processes=nproc,progress=False)
    if len(rows)!=len(jobs) or any(not r for r in rows): raise RuntimeError('decision grid returned incomplete data')
    write_csv(output/'per_seed.csv',rows,COLS); timeline=[]
    for key in sorted({(r['mode'],r['loss'],r['outage']) for r in rows}):
      mode,loss,outage=key; times=sorted({x['time_s'] for r in rows if (r['mode'],r['loss'],r['outage'])==key for x in r['timeline']})
      for t in times:
        vv=[x['standoff_error_m'] for r in rows if (r['mode'],r['loss'],r['outage'])==key for x in r['timeline'] if x['time_s']==t]; timeline.append({'mode':mode,'loss':loss,'outage':outage,'time_s':t,'standoff_error_m':float(np.median(vv))})
    write_csv(output/'timeline.csv',timeline,['mode','loss','outage','time_s','standoff_error_m']); write_json(output/'summary.json',_summary(rows))
    from tools.d_make_decisions_figure import make_figures
    make_figures(output)
    outputs=['per_seed.csv','timeline.csv','summary.json','decisions_outage.png','decisions_traffic.png']; runtime=time.perf_counter()-start; cpu_runtime=time.process_time()-cpu_start
    write_manifest(output,schema='zephyr-d-decisions-manifest-1',argv=argv_or_default(argv),command='PYTHONPATH=. ./.venv/bin/python tools/d_run_decisions.py --output {out}',config={'pilot':pilot,'n_seeds':n,'n_agents':4,'modes':list(MODES),'losses':list(losses),'outages':list(OUTAGES),'delay_s':0.05,'rate_hz':10.0,'duration_s':20.0,'control_dt_s':0.02,'heartbeat_timeout_s':0.5,'failure_queries_dt_s':0.02},seeds=seeds,outputs=outputs,limits=['Agents are simulated kinematic point masses; target motion, Gaussian sensor noise, and abstract links are simulated.','Link timing and message sizes are literature-derived inputs, and no values are measured.','This experiment compares decision authority during a simulated ground-station outage.'],runtime_s=runtime,n_processes=nproc,hashed_sources=['tools/d_run_decisions.py','tools/d_make_decisions_figure.py','sim/coop/simulator.py','sim/coop/fusion.py','sim/coop/avoidance.py','sim/coop/observers.py'])
    print(f'pilot={pilot} seeds={n} runtime_s={runtime:.3f} cpu_s={cpu_runtime:.3f} full_run_estimate_cpu_s={(cpu_runtime*(50/n))/4.0:.3f}')
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path); p.add_argument('--pilot',action='store_true'); p.add_argument('--processes',type=int); a=p.parse_args(argv); run(a.output,pilot=a.pilot,processes=a.processes,argv=argv_or_default(argv))
if __name__=='__main__': main()
