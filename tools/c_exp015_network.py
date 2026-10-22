"""EXP-015 network topology simulation.

Usage: ``PYTHONPATH=. ./.venv/bin/python tools/c_exp015_network.py --output sim/runs/c-exp015-network``
"""
from __future__ import annotations
import argparse, csv, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sim.radio.channel import Channel, mean_rssi_dbm, packet_error_rate
from sim.radio.network import Network, NetworkConfig
from sim.radio.params import PARAMS, channel_preset, default_mac, default_phy
from sim.s6_recovery import median_interval, wilson_interval
from tools.c_common import prepare_output, seeds_from, write_json, write_manifest, default_processes, argv_or_default

BASE_SEED = 20261014 + 15
TOPOLOGIES = ("star", "relay", "flood", "gossip", "broadcast")
PRE_REGISTERED_TOPOLOGIES = ("star", "relay", "flood", "gossip")

def stress_channel():
    wavelength = 299792458.0 / 2.437e9
    return channel_preset("outdoor_open", shadowing_sigma_db=PARAMS["sigma_outdoor_open_db"].high,
        breakpoint_m=4*1.5*1.5/wavelength, exponent_far=4.0,
        orientation_loss_db=PARAMS["orientation_loss_db"].high)

def positions_for(seed: int, area: str, n: int) -> dict[str, tuple[float,float,float]]:
    ss = np.random.SeedSequence(seed).spawn(2)[0]; rng = np.random.default_rng(ss)
    size = (30.0,20.0) if area == "gym" else (600.0,600.0)
    xy = rng.uniform([0,0], size, (n,2)); z = rng.uniform(1,3,n)
    out = {"gs": (0.0,0.0,1.5)}
    out.update({f"d{i:02d}": (float(xy[i,0]),float(xy[i,1]),float(z[i])) for i in range(n)})
    return out

def direct_range_fraction(positions, cfg, payload_bytes=32, *, draws=2048, seed=1):
    """Fraction of drones whose shadowing-averaged direct delivery is at least 0.9."""
    drones = [k for k in positions if k.startswith("d")]
    if not drones: return 0.0
    vals=[]
    for d in drones:
        p1=np.asarray(positions[d]); p2=np.asarray(positions["gs"]); dist=float(np.linalg.norm(p1-p2))
        vals.append(float(direct_link_in_range(dist, cfg, payload_bytes, draws=draws, seed=seed)))
    return float(np.mean(vals))

def direct_link_in_range(distance_m, cfg, payload_bytes=32, *, draws=4096, seed=1):
    """Return whether one link's shadowing-averaged delivery reaches 0.9."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    shadow = rng.standard_normal(draws) * cfg.shadowing_sigma_db
    mean = mean_rssi_dbm(float(distance_m), cfg)
    per = packet_error_rate(mean + shadow - cfg.noise_floor_dbm,
                            payload_bytes + default_phy().mac_overhead_bytes, cfg)
    return bool(float(np.mean(1.0 - np.asarray(per))) >= 0.9)

def simulate_cell(cell):
    area,n,rate,topo,seed = cell["area"],int(cell["n"]),float(cell["rate"]),cell["topology"],int(cell["seed"])
    pos=positions_for(seed,area,n)
    if topo=="relay": pos["relay"]=(30/2,20/2,1.5) if area=="gym" else (300.0,300.0,1.5)
    cfg=channel_preset("indoor_gym" if area=="gym" else "outdoor_open") if area=="gym" else stress_channel()
    sim_seed=int(np.random.SeedSequence(seed).spawn(3)[2].generate_state(1)[0])
    duration = float(cell.get("duration_s", 30.0))
    net=Network(pos,Channel(cfg,np.random.default_rng(np.random.SeedSequence(sim_seed))),default_phy(),
                default_mac(unicast_retry_limit=5), NetworkConfig(topo,relay="relay" if topo=="relay" else None,
                ttl=4,gossip_g=0.5,forward_delay_s=0.0, gateway_forwards=False), seed=sim_seed)
    for d in [f"d{i:02d}" for i in range(n)]: net.add_periodic_origin(d,rate,32,start_s=0.0,stop_s=duration + 1.0)
    net.run_until(duration + 1.5); s=net.summary(1.0,duration + 1.0)
    lat=s["latencies_s"]
    return {"area":area,"n":n,"rate":rate,"topology":topo,"seed":seed,"delivery_ratio":s["delivery_ratio"],
            "n_messages":s["n_messages"],"n_delivered":s["n_delivered"],"latency_median_s":float(np.median(lat)) if lat else float("nan"),
            "latency_p95_s":float(np.percentile(lat,95)) if lat else float("nan"),"mean_hops":s["mean_hops"],
            "transmissions_per_message":s["transmissions_per_message"],"busy_fraction":s["busy_fraction"],
            "direct_range_fraction":direct_range_fraction(pos,cfg,seed=seed)}

def aggregate(rows):
    groups={}
    for r in rows: groups.setdefault((r["area"],r["n"],r["rate"],r["topology"]),[]).append(r)
    cells=[]
    for key,rs in sorted(groups.items()):
        cell = aggregate_cell(rs)
        succ=sum(x["n_delivered"] for x in rs); total=sum(x["n_messages"] for x in rs); wl,wh=wilson_interval(succ,total)
        cell.update({"area":key[0],"n":key[1],"rate":key[2],"delivery_ci_low":cell.pop("delivery_median_lo"),"delivery_ci_high":cell.pop("delivery_median_hi"),
                     "delivery_pooled":succ/total,"delivery_wilson_low":wl,"delivery_wilson_high":wh})
        cells.append(cell)
    return cells

def aggregate_cell(rows):
    """Aggregate one topology's per-seed rows for small checks and callers."""
    if not rows:
        raise ValueError("rows must be nonempty")
    def intervals(name):
        # Median over seeds with a finite value (a seed with no delivered message has no latency or hops).
        vals = [float(r[name]) for r in rows if r[name] is not None and np.isfinite(float(r[name]))]
        return median_interval(vals) if vals else (None, None, None)
    total = sum(int(r["n_messages"]) for r in rows)
    delivered = sum(int(r["n_delivered"]) for r in rows)
    dmed, dlo, dhi = intervals("delivery_ratio")
    out = {"topology": rows[0]["topology"], "delivery_median": dmed, "delivery_median_lo": dlo, "delivery_median_hi": dhi,
           "delivery_pooled": delivered / total, "latency_median_s": intervals("latency_median_s")[0],
           "latency_p95_s": intervals("latency_p95_s")[0], "mean_hops": intervals("mean_hops")[0],
           "transmissions_per_message": intervals("transmissions_per_message")[0], "busy_fraction": intervals("busy_fraction")[0],
           "direct_range_fraction": intervals("direct_range_fraction")[0] if "direct_range_fraction" in rows[0] else None}
    for name in ("latency_median_s", "latency_p95_s", "mean_hops", "transmissions_per_message", "busy_fraction", "direct_range_fraction"):
        if name in rows[0]:
            _, out[f"{name}_lo"], out[f"{name}_hi"] = intervals(name)
    return out

def answer_fields(rows, cells):
    """Return pre-registered comparisons, computed from paired seed rows."""
    out = []
    keys = sorted({(r["area"], int(r["n"]), float(r["rate"])) for r in rows})
    for area, n, rate in keys:
        subset = [r for r in rows if r["area"] == area and int(r["n"]) == n and float(r["rate"]) == rate]
        by = {t: sorted((r for r in subset if r["topology"] == t), key=lambda r: int(r["seed"])) for t in PRE_REGISTERED_TOPOLOGIES}
        pooled = {t: next(c for c in cells if c["area"] == area and c["n"] == n and c["rate"] == rate and c["topology"] == t) for t in TOPOLOGIES}
        winner = max(PRE_REGISTERED_TOPOLOGIES, key=lambda t: pooled[t]["delivery_pooled"])
        star_lo, star_hi = pooled["star"]["delivery_wilson_low"], pooled["star"]["delivery_wilson_high"]
        w = pooled[winner]
        def gain(t):
            diffs = [a["delivery_ratio"] - b["delivery_ratio"] for a, b in zip(by[t], by["star"])]
            med, lo, hi = median_interval(diffs)
            return {"median": med, "ci_low": lo, "ci_high": hi}
        out.append({"area": area, "n": n, "rate": rate, "highest_delivery_topology": winner,
                    "highest_interval_excludes_star": bool(w["delivery_wilson_low"] > star_hi or w["delivery_wilson_high"] < star_lo),
                    "transmissions_per_message": {t: pooled[t]["transmissions_per_message"] for t in PRE_REGISTERED_TOPOLOGIES},
                    "relay_gain": gain("relay"), "flood_gain": gain("flood")})
    return out

def run(output:Path, argv, pilot=False, processes=1, n_seeds=50):
    import os
    started = time.perf_counter()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/zephyr-mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.horizon_mpl import apply_style
    output=prepare_output(output); seeds=seeds_from(BASE_SEED,2 if pilot else n_seeds); ns=(4,) if pilot else (4,8,12); rates=(10,) if pilot else (10,20); areas=("gym","field")
    jobs=[{"area":a,"n":n,"rate":r,"topology":t,"seed":s,"duration_s":5.0 if pilot else 30.0} for a in areas for n in ns for r in rates for t in TOPOLOGIES for s in seeds]
    rows=list(ProcessPoolExecutor(max_workers=processes).map(simulate_cell,jobs)) if processes>1 else [simulate_cell(j) for j in jobs]
    cells=aggregate(rows)
    for a in areas:
        for n in ns:
            for r in rates:
                print(f"completed cell area={a} N={n} rate={r:g} Hz", flush=True)
    answers = answer_fields(rows, cells)
    write_json(output/"summary.json",{"experiment":"EXP-015","status":"simulated; literature-derived parameters","seeds":seeds,"n_seeds":len(seeds),"pre_registered_n_seeds":50,"topologies":list(TOPOLOGIES),"additions":["broadcast (single-hop baseline, not pre-registered)"],"pooled_wilson_note":"The pooled Wilson interval treats messages as independent.","excludes_star_note":"excludes star means the 95% intervals do not overlap.","retry_note":"star sends each message by unicast with ACK and up to 5 retries; in the relay topology the drone's first leg is a single broadcast with no ACK or retry, and only the relay-to-gs leg retries, so relay_gain partly reflects that retry difference.","cells":cells,"answers":answers})
    with open(output/"per_seed.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(output/"cells.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    colors=apply_style("light"); fig,axs=plt.subplots(2,2,figsize=(10,7),sharey=True)
    # Horizon colours per topology (red and amber stay reserved for alerts); broadcast, the
    # unregistered baseline, is dashed.
    tcolor={"star":colors["accent"],"relay":colors["ok"],"flood":colors["keepOut"],"gossip":colors["target"],"broadcast":colors["textSecondary"]}
    for ax,(a,r) in zip(axs.flat,[(a,r) for a in areas for r in rates]):
        for t in TOPOLOGIES:
            z=[x for x in cells if x["area"]==a and x["rate"]==r and x["topology"]==t]; z=sorted(z,key=lambda x:x["n"]); line,=ax.plot([x["n"] for x in z],[x["delivery_pooled"] for x in z],marker="o",label=t,color=tcolor[t],ls="--" if t=="broadcast" else "-")
            lo=[x["delivery_wilson_low"] for x in z]; hi=[x["delivery_wilson_high"] for x in z]
            ys = np.asarray([x["delivery_pooled"] for x in z], dtype=float)
            ax.errorbar([x["n"] for x in z], ys,
                        yerr=[np.maximum(0.0, ys - np.asarray(lo)), np.maximum(0.0, np.asarray(hi) - ys)],
                        fmt="none", capsize=2, ecolor=line.get_color())
        ax.set_title(f"{a}, {r:g} Hz (simulated)"); ax.set_xlabel("drones"); ax.set_ylabel("pooled delivery ratio"); ax.set_ylim(0,1.05)
    axs[0,0].legend(fontsize=7,frameon=False); fig.tight_layout(); fig.savefig(output/"exp015_delivery.png",dpi=150,metadata={"Software":None}); plt.close(fig)
    fig,axs=plt.subplots(2,len(areas),figsize=(12,7),sharex=False)
    for i,metric in enumerate(("transmissions_per_message","busy_fraction")):
        for j,a in enumerate(areas):
            ax=axs[i,j]
            subset=[x for x in cells if x["area"]==a]
            labels=sorted({(x["n"],x["rate"]) for x in subset})
            xpos=np.arange(len(labels)); width=0.8/len(TOPOLOGIES)
            for k,t in enumerate(TOPOLOGIES):
                vals=[next(x[metric] for x in subset if x["topology"]==t and (x["n"],x["rate"])==label) for label in labels]
                ax.bar(xpos+(k-(len(TOPOLOGIES)-1)/2)*width, vals, width=width, label=t, color=tcolor[t])
            ax.set_title(f"{a} (simulated)"); ax.set_xticks(xpos, [f"N={n}, {rate:g} Hz" for n,rate in labels], rotation=35, ha="right")
            ax.set_ylabel(metric.replace("_"," "))
    axs[0,0].legend(fontsize=7,frameon=False); fig.suptitle("Simulated network airtime cost by topology"); fig.tight_layout(); fig.savefig(output/"exp015_cost.png",dpi=150,metadata={"Software":None}); plt.close(fig)
    stable_argv = (["--pilot"] if pilot else ["--seeds", str(n_seeds)]) + ["--processes", str(processes)]
    write_manifest(output,schema="zephyr-run-c-manifest-1",argv=stable_argv,command="PYTHONPATH=. ./.venv/bin/python tools/c_exp015_network.py --output {out} " + " ".join(stable_argv),config={"pilot":pilot,"payload_bytes":32,"areas":areas,"n":ns,"rates":rates,"additions":["broadcast (single-hop baseline, not pre-registered)"],"n_seeds":len(seeds),"pre_registered_n_seeds":50},seeds=seeds,outputs=["summary.json","per_seed.csv","cells.csv","exp015_delivery.png","exp015_cost.png"],limits=["Field busy fraction is the union over the whole network medium.","Star retries by unicast (ACK, up to 5 retries); the relay topology's first leg is one unacknowledged broadcast.","Fewer seeds than pre-registered when --seeds is below 50 (see summary.json n_seeds)."],runtime_s=time.perf_counter()-started,n_processes=processes,hashed_sources=["sim/radio/network.py","sim/radio/mac.py","sim/radio/channel.py","sim/radio/params.py","tools/c_exp015_network.py"])

def main(argv=None):
    argv=argv_or_default(argv); p=argparse.ArgumentParser(description=__doc__.splitlines()[0]); p.add_argument("--output",type=Path,required=True); p.add_argument("--pilot",action="store_true"); p.add_argument("--seeds",type=int,default=50); p.add_argument("--processes",type=int,default=default_processes()); a=p.parse_args(argv); run(a.output,argv,a.pilot,a.processes,a.seeds); return 0
if __name__=="__main__": raise SystemExit(main())
