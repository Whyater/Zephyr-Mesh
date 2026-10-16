from __future__ import annotations
import os, tempfile
from pathlib import Path
import numpy as np

def make_figures(output: Path, rows: list[dict], summary: list[dict]):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'zephyr-mplconfig'))
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    style=apply_style('light')
    fig,ax=plt.subplots(figsize=(8,5))
    for key in sorted({(r['n'],r['k']) for r in rows if r['mode']=='silent'}):
        for metric,marker,label in (('detection_time_s','o','detection'),('recoverage_time_s','s','re-coverage')):
            xs=[]; ys=[]; lo=[]; hi=[]
            for loss in sorted({r['link_loss'] for r in rows}):
                g=[r[metric] for r in rows if r['mode']=='silent' and (r['n'],r['k'])==key and r['link_loss']==loss and r[metric] is not None]
                if not g: continue
                xs.append(loss); ys.append(float(np.median(g))); lo.append(float(np.percentile(g,25))); hi.append(float(np.percentile(g,75)))
            ax.errorbar(xs,ys,yerr=[np.asarray(ys)-lo,np.asarray(hi)-ys],marker=marker,capsize=3,label=f'N={key[0]}, k={key[1]} {label}')
            if metric=='detection_time_s':
                pred=[next((r['hand_detection_s'] for r in rows if r['mode']=='silent' and (r['n'],r['k'])==key and r['link_loss']==x),np.nan) for x in xs]
                ax.plot(xs,pred,'x',ms=5,color=style['line'],label=f'N={key[0]}, k={key[1]} hand prediction')
    ax.set_xlabel('neighbor link loss probability'); ax.set_ylabel('time after drop (s)'); ax.set_title('Simulated dropout detection and re-coverage'); ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig(output/'dropout_times.png',dpi=150,metadata={'Software':None}); plt.close(fig)
    reps=[r for r in rows if r['n']==4 and r['k']==1 and r['mode']=='silent' and abs(r['link_loss']-.1)<1e-12]
    rep=min(reps,key=lambda r:r['seed']) if reps else rows[0]
    fig,ax=plt.subplots(figsize=(8,4.5)); tl=rep.get('timeline',[])
    if tl:
        t,g=zip(*tl); ax.plot(t,g,color=style['accent'],label='largest survivor gap')
    det=rep.get('detection_time_s'); rec=rep.get('recoverage_time_s')
    if det is not None: ax.axvline(10+det,color='tab:red',lw=1,label='detection')
    if rec is not None: ax.axvline(10+rec,color='tab:orange',lw=1,label='re-coverage')
    ax.axhline(120,color=style['line'],lw=.8,label='ideal 120° gap'); ax.set_xlabel('time (s)'); ax.set_ylabel('largest angular gap (degrees)'); ax.set_title('Dropout geometry timeline'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'dropout_timeline.png',dpi=150,metadata={'Software':None}); plt.close(fig)
