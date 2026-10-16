"""Figures for the simulated decision-authority experiment."""
from pathlib import Path
import csv
import numpy as np

def make_figures(directory: Path) -> None:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    p=apply_style('light'); directory=Path(directory)
    with (directory/'timeline.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    available=sorted({float(r['loss']) for r in rows if r['outage']=='10_15'})
    loss=0.1 if 0.1 in available else (available[-1] if available else 0.0)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for mode,color in (('ground',p['accent']),('onboard',p['ok'])):
        rr=[r for r in rows if r['mode']==mode and float(r['loss'])==loss and r['outage']=='10_15']; rr.sort(key=lambda r:float(r['time_s']))
        if rr: ax.plot([float(r['time_s']) for r in rr],[float(r['standoff_error_m']) for r in rr],label=mode,color=color)
    ax.axvspan(10,15,color='lightgray',alpha=.6); ax.set(xlabel='time (s)',ylabel='median standoff error (m)',title='When the ground link dies, commands stop and the ring drifts'); ax.text(.02,.03,'Onboard decisions keep following',transform=ax.transAxes); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(directory/'decisions_outage.png',dpi=150,metadata={'Software':None}); plt.close(fig)
    with (directory/'per_seed.csv').open(encoding='utf-8',newline='') as f: data=list(csv.DictReader(f))
    vals=[float(np.median([float(r['bytes_per_second']) for r in data if r['mode']==mode])) for mode in ('ground','onboard')]
    fig,ax=plt.subplots(figsize=(6,4.4)); ax.bar(['ground','onboard'],vals,color=[p['accent'],p['ok']]); ax.set(ylabel='bytes per second',title='Simulated traffic by decision mode\nmessage sizes: raw 56 B, track 216 B, command 24 B, state 240 B'); fig.tight_layout(); fig.savefig(directory/'decisions_traffic.png',dpi=150,metadata={'Software':None}); plt.close(fig)
