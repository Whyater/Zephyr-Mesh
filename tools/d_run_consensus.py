"""Run the simulated cooperative average-consensus experiment."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from sim.coop.consensus import (algebraic_connectivity, complete, eigenvalues, line,
                                laplacian, max_degree, measured_rate, random_geometric,
                                ring, run_consensus, spectral_rate, star,
                                steps_to_agreement)
from sim.estimation.sweep import run_jobs, write_csv
from sim.s6_recovery import median_interval, wilson_interval
from tools.d_common import (argv_or_default, default_processes, prepare_output,
                            seeds_for, write_json, write_manifest)

GRAPH_NAMES = ("ring", "line", "star", "complete", "random_geometric")
LOSS = (0.0, 0.1, 0.3, 0.5, 0.7)
DELAYS = (0, 1, 2, 5)


def _check(condition: bool, message: str) -> None:
    """Row-count checks that still run under python -O."""
    if not condition:
        raise ValueError(message)


def _graph(name: str, seed: int | None = None):
    if name == "ring": return ring(6), None
    if name == "line": return line(5), None
    if name == "star": return star(5), None
    if name == "complete": return complete(4), None
    if name != "random_geometric":
        raise ValueError(f"unknown graph {name!r}")
    used = int(seed or 0)
    while True:
        a, pos = random_geometric(6, 2.0, seed=used)
        if algebraic_connectivity(laplacian(a)) > 1e-12:
            return a, used
        used += 1


def _graph_table(seed: int):
    out = {}
    for name in GRAPH_NAMES:
        a, used = _graph(name, seed if name == "random_geometric" else None)
        out[name] = {"adjacency": a.tolist(), "random_seed": used}
    return out


def _loss_job(job: dict):
    a = np.asarray(job["adjacency"], dtype=float)
    initial = np.asarray(job["initial"], dtype=float)
    result = run_consensus(initial, a, float(job["eps"]), int(job["steps"]),
                           loss_probability=float(job["p"]), delay_steps=int(job["delay"]),
                           symmetric_loss=bool(job["symmetric"]), seed=int(job["seed"]))
    agree = steps_to_agreement(result, 0.01)
    spread = float(np.max(np.ptp(initial, axis=0)))
    drift = float(np.linalg.norm(result.average[-1] - result.average[0]) / max(spread, 1e-15))
    return {"graph": job["graph"], "random_seed": job.get("random_seed"), "p": float(job["p"]), "delay_steps": int(job["delay"]),
            "symmetric_loss": bool(job["symmetric"]), "seed": int(job["seed"]),
            "steps_to_agreement": agree, "did_not_agree": agree is None,
            "final_average_drift": drift, "messages_lost_fraction": result.messages_lost / max(result.messages_sent, 1)}


def _acting_job(job: dict):
    a = np.asarray(job["adjacency"], dtype=float)
    truth = np.asarray([0.0, 0.0, 1.5])
    streams = np.random.SeedSequence([int(job["seed"]), 2, 41]).spawn(a.shape[0])
    initial = np.asarray([truth + np.random.default_rng(stream).normal(0.0, 0.05, size=3)
                          for stream in streams])
    result = run_consensus(initial, a, float(job["eps"]), int(job["k"]),
                           loss_probability=float(job["p"]), delay_steps=int(job["delay"]),
                           symmetric_loss=True, seed=int(job["seed"]) + 991)
    x = result.states[-1]
    largest = float(np.max(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)))
    return {"graph": job["graph"], "p": float(job["p"]), "delay_steps": int(job["delay"]),
            "updates": int(job["k"]), "seed": int(job["seed"]), "largest_pairwise_distance_m": largest,
            "still_disagreeing": largest > 0.10}


def _spectral(seed: int):
    rows = []
    for name in GRAPH_NAMES:
        a, used = _graph(name, seed if name == "random_geometric" else None)
        L = laplacian(a); vals = eigenvalues(L); dmax = max_degree(a); eps = 1.0 / (dmax + 1.0)
        rng = np.random.default_rng(np.random.SeedSequence([seed, GRAPH_NAMES.index(name), 7]))
        initial = rng.uniform(-1.0, 1.0, size=(a.shape[0], 3))
        # 80 steps: the slowest mode is read from steps 20 to 60, after the faster modes have died
        # out, so the geometric-mean ratio converges to rho instead of underestimating it.
        result = run_consensus(initial, a, eps, 80, seed=seed + 17)
        rho = spectral_rate(L, eps)
        row = {"graph": name, "random_seed": used, "eigenvalues": vals.tolist(),
               "lambda2": algebraic_connectivity(L), "lambdaN": float(vals[-1]), "d_max": dmax,
               "eps": eps, "predicted_rho": rho}
        if name == "complete":
            row.update({"measured_steps_to_agreement": steps_to_agreement(result, 1e-9),
                        "rate_check": "not applicable (rho = 0)"})
        else:
            start = 20
            positive = np.flatnonzero(result.disagreement > 1e-12 * result.disagreement[0])
            stop = min(60, int(positive[-1])) if positive.size else start
            if stop <= start:
                raise ValueError(f"{name}: disagreement vanished before step {start}; cannot measure the rate")
            measured = measured_rate(result.disagreement, start, stop)
            relative_error = abs(measured - rho) / rho
            row.update({"measured_rate": measured, "measurement_start": start, "measurement_stop": stop,
                        "relative_error": relative_error,
                        "rate_check": "ok" if relative_error < 1e-6 else "mismatch"})
        rows.append(row)
    return rows


def _summarize(rows):
    out = []
    keys = sorted({(r["graph"], r["p"], r["delay_steps"], r["symmetric_loss"]) for r in rows})
    for graph, p, delay, sym in keys:
        rr = [r for r in rows if (r["graph"], r["p"], r["delay_steps"], r["symmetric_loss"]) == (graph, p, delay, sym)]
        cap = 400 + 1
        steps = [cap if r["steps_to_agreement"] is None else r["steps_to_agreement"] for r in rr]
        med0, lo0, hi0 = median_interval(steps)
        med, lo, hi = (None if value >= cap else value for value in (med0, lo0, hi0))
        # With an even seed count the median averages the two middle values; if the
        # upper one never agreed, the median is not a real step count.
        if sorted(steps)[len(steps) // 2] >= cap:
            med = None
        fails = sum(r["did_not_agree"] for r in rr); fl, fh = wilson_interval(fails, len(rr))
        dm, dl, dh = median_interval([r["final_average_drift"] for r in rr])
        a, _ = _graph(graph, next((r.get("random_seed") for r in rr), None)); eps = 1 / (max_degree(a) + 1)
        # Mean-field prediction (expected Laplacian (1 - p) L) only covers the no-delay case.
        rate = spectral_rate(laplacian(a), (1-p) * eps)
        if delay > 0 or rate >= 1: pred = None
        elif rate <= 0: pred = 1
        else: pred = int(math.ceil(math.log(0.01) / math.log(rate)))
        out.append({"graph": graph, "p": p, "delay_steps": delay, "symmetric_loss": sym, "n_seeds": len(rr),
                    "median_steps": med, "median_steps_low": lo, "median_steps_high": hi,
                    "did_not_agree_rate": fails/len(rr), "did_not_agree_low": fl, "did_not_agree_high": fh,
                    "median_drift": dm, "median_drift_low": dl, "median_drift_high": dh,
                    "cell_failed": len(rr) - fails < len(rr) / 2,
                    "mean_field_predicted_steps_zero_delay": pred})
    return out


def run(output: Path, *, seeds_n: int | None = None, pilot: bool = False,
        processes: int | None = None, argv: list[str] | None = None):
    start = time.perf_counter(); output = prepare_output(output)
    n = (5 if pilot else 50) if seeds_n is None else int(seeds_n)
    if n <= 0:
        raise ValueError("seeds must be positive")
    seeds = seeds_for("consensus", n); base_seed = seeds[0]
    spectral = _spectral(base_seed); write_json(output / "spectral_check.json", spectral)
    # Neighbors start on opposite sides (+1, -1, +1, -1 around the ring, on every axis): this is the
    # mode with eigenvalue 4 = 2 x max degree, which eps = 1/max degree flips forever.
    cinit = np.array([[1.0] * 3, [-1.0] * 3, [1.0] * 3, [-1.0] * 3]); ca = ring(4)
    cr = run_consensus(cinit, ca, 0.5, 50, seed=base_seed)
    write_json(output / "counterexample.json", {"graph": "ring(4)", "eps": 0.5, "rho": spectral_rate(laplacian(ca), .5),
                                                "initial_disagreement": float(cr.disagreement[0]),
                                                "disagreement_after_50": float(cr.disagreement[-1])})
    graphs = {name: _graph(name, base_seed if name == "random_geometric" else None)[0] for name in GRAPH_NAMES}
    jobs = []
    for gi, name in enumerate(GRAPH_NAMES):
        a = graphs[name]; eps = 1/(max_degree(a)+1); random_seed = base_seed if name == "random_geometric" else None
        for p in (0, .5) if pilot else LOSS:
            for delay in (0, 2) if pilot else DELAYS:
                for sym in (True, False):
                    for si, seed in enumerate(seeds):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, gi, 31]))
                        jobs.append({"graph":name,"random_seed":random_seed,"adjacency":a.tolist(),"eps":eps,"p":p,"delay":delay,"symmetric":sym,
                                     "seed":seed,"steps":400,"initial":rng.uniform(-1,1,size=(a.shape[0],3)).tolist()})
    results = run_jobs(jobs, "tools.d_run_consensus:_loss_job", processes=processes or default_processes(), progress=False)
    expected_loss_cells = len(GRAPH_NAMES) * (2 if pilot else len(LOSS)) * (2 if pilot else len(DELAYS)) * 2
    _check(len(results) == len(jobs), "loss jobs and results differ in count")
    _check(len(results) == expected_loss_cells * len(seeds), "unexpected number of loss rows")
    _check(all(sum(1 for r in results if (r["graph"], r["p"], r["delay_steps"], r["symmetric_loss"]) == key) == len(seeds)
               for key in {(r["graph"], r["p"], r["delay_steps"], r["symmetric_loss"]) for r in results}),
           "a loss cell is missing seeds")
    columns = ["graph","p","delay_steps","symmetric_loss","seed","steps_to_agreement","did_not_agree","final_average_drift","messages_lost_fraction"]
    write_csv(output / "loss_delay.csv", results, columns); write_json(output / "loss_delay_summary.json", _summarize(results))
    ajobs=[]
    for name in ("complete", "line"):
        a = complete(4) if name == "complete" else line(4); eps=1/(max_degree(a)+1)
        for p in (0,.5) if pilot else (0,.1,.3,.5):
            for delay in (0,1):
                for k in (0,1,2,5,10):
                    for seed in seeds: ajobs.append({"graph":name,"adjacency":a.tolist(),"eps":eps,"p":p,"delay":delay,"k":k,"seed":seed})
    acting = run_jobs(ajobs, "tools.d_run_consensus:_acting_job", processes=processes or default_processes(), progress=False)
    _check(len(acting) == len(ajobs), "acting jobs and results differ in count")
    expected_acting_cells = 2 * (2 if pilot else 4) * 2 * 5
    _check(len(acting) == expected_acting_cells * len(seeds), "unexpected number of acting rows")
    write_csv(output / "acting.csv", acting, ["graph","p","delay_steps","updates","seed","largest_pairwise_distance_m","still_disagreeing"])
    asum=[]
    for key in sorted({(r["graph"],r["p"],r["delay_steps"],r["updates"]) for r in acting}):
        rr=[r for r in acting if tuple(r[k] for k in ("graph","p","delay_steps","updates"))==key]; succ=sum(r["still_disagreeing"] for r in rr); lo,hi=wilson_interval(succ,len(rr)); med,ml,mh=median_interval([r["largest_pairwise_distance_m"] for r in rr])
        asum.append({"graph":key[0],"p":key[1],"delay_steps":key[2],"updates":key[3],"n_seeds":len(rr),"fraction":succ/len(rr),"fraction_low":lo,"fraction_high":hi,"median_distance_m":med,"median_distance_low_m":ml,"median_distance_high_m":mh,"cell_failed":len(rr)-succ < len(rr) / 2})
    _check(len(asum) == expected_acting_cells, "unexpected number of acting cells")
    _check(all(cell["n_seeds"] == len(seeds) for cell in asum), "an acting cell is missing seeds")
    write_json(output / "acting_summary.json", asum)
    import tools.d_make_consensus_figure as figmod
    figmod.make_figures(output)
    outputs=["spectral_check.json","counterexample.json","loss_delay.csv","loss_delay_summary.json","acting.csv","acting_summary.json","consensus_loss.png","consensus_acting.png"]
    command = "PYTHONPATH=. python tools/d_run_consensus.py --output {out}"
    if seeds_n is not None:
        command += f" --seeds {n}"
    if pilot:
        command += " --pilot"
    config = {
        "pilot": pilot,
        "seeds": n,
        "graphs": list(GRAPH_NAMES),
        "eps_rule": "1 / (max_degree(adjacency) + 1)",
        "loss_probabilities": [0.0, 0.5] if pilot else list(LOSS),
        "delay_steps": [0, 2] if pilot else list(DELAYS),
        "symmetric_loss": [True, False],
        "consensus_steps": 400,
        "agreement_tolerance": 0.01,
        "spectral_steps": 80,
        "spectral_measurement_steps": [20, 60],
        "acting_graphs": ["complete", "line"],
        "acting_loss_probabilities": [0.0, 0.5] if pilot else [0.0, 0.1, 0.3, 0.5],
        "acting_delay_steps": [0, 1],
        "acting_updates": [0, 1, 2, 5, 10],
        "acting_truth": [0.0, 0.0, 1.5],
        "acting_noise_std": 0.05,
        "acting_seed_recipe": "SeedSequence([seed, 2, 41]).spawn(n_observers)",
        "acting_symmetric_loss": True,
        "part_c": {
            "graphs": ["complete", "line"],
            "loss": [0.0, 0.5] if pilot else [0.0, 0.1, 0.3, 0.5],
            "delays": [0, 1],
            "updates": [0, 1, 2, 5, 10],
            "truth": [0.0, 0.0, 1.5],
            "noise_std": 0.05,
        },
        "part_c_note": "Part C was added after pre-registration; exploratory",
    }
    write_manifest(output, schema="zephyr-d-consensus-manifest-1", argv=argv_or_default(argv), command=command, config=config, seeds=seeds, outputs=outputs,
                   limits=["Agents are nodes holding a value with no motion.",
                           "Links are abstract with independent per-message loss, no bursts, and fixed whole-step delay.",
                           "Part C uses simulated Gaussian noise of 0.05 m per axis around a fixed truth.",
                           "Five small fixed graphs are used.",
                           "Disagreement is the root-sum-square distance of agents' values from their mean."],
                   runtime_s=time.perf_counter()-start, n_processes=processes or default_processes(),
                   hashed_sources=["tools/d_run_consensus.py","tools/d_make_consensus_figure.py","sim/coop/consensus.py"])


def main(argv: list[str] | None = None):
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True,type=Path); p.add_argument("--seeds",type=int); p.add_argument("--pilot",action="store_true"); p.add_argument("--processes",type=int); a=p.parse_args(argv); run(a.output,seeds_n=a.seeds,pilot=a.pilot,processes=a.processes,argv=argv_or_default(argv))
if __name__ == "__main__": main()
