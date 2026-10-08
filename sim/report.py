"""Read-only S1 report generation for retained S0 telemetry."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from .metrics import step_response, tracking_error

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_baseline_report(run_dir):
    root=Path(run_dir); telemetry=root/"telemetry.csv"; manifest=root/"manifest.json"
    if not telemetry.exists(): raise FileNotFoundError(telemetry)
    data=np.genfromtxt(telemetry,delimiter=",",names=True)
    names=data.dtype.names or (); t=np.atleast_1d(data["time"]).astype(float); z=np.atleast_1d(data["z"]).astype(float)
    metadata=json.loads(manifest.read_text()) if manifest.exists() else {}
    goal=float(metadata.get("config", {}).get("target_position", [0, 0, 3])[2])
    # S0 labels time before each post-step state. Preserve labels and state the convention.
    response=step_response(t,z,goal)
    x = np.atleast_1d(data["x"]).astype(float)
    y = np.atleast_1d(data["y"]).astype(float)
    err=tracking_error(np.column_stack([np.zeros(len(z)),np.zeros(len(z)),np.full(len(z),goal)]),np.column_stack([x,y,z]))
    # Keep provenance portable: absolute TemporaryDirectory paths are not durable evidence.
    source={"telemetry_source":"runs/s0-baseline/telemetry.csv","telemetry_sha256":_sha(telemetry),"manifest_sha256":_sha(manifest) if manifest.exists() else None,"columns":list(names)}
    return {"schema":"zephyr-s1-report-1","status":"retained baseline measurement; source revision is recorded separately; not flight performance","source":source,"provenance":{"artifact_status":metadata.get("artifact_status", "run manifest"),"source_git_revision":metadata.get("git_revision"),"source_git_tree_state":metadata.get("git_tree_state"),"current_worktree_match":"not claimed","python":platform.python_version(),"numpy":np.__version__,"command":"python -m sim.report runs/s0-baseline --output <report.json>","runtime_s":"unavailable: report builder does not time simulation","output_sha256":None,"output_hash_status":"not claimed before serialization","hash_parity":{"telemetry_sha256":"sha256 of retained telemetry.csv bytes","manifest_sha256":"sha256 of retained manifest.json bytes if present"}},"sampling":{"time_column":"time","time_semantics":"S0 pre-step label for the state recorded after that step; no relabeling performed","sample_count":len(t),"duration_s":None if len(t)<2 else float(t[-1]-t[0])},"step_response":{"goal_m":goal,**response},"tracking_error":err,"unavailable_metrics":{"packet_loss":"No packet events in S0 telemetry","packet_age":"No shared-clock observation timestamps in S0 telemetry","detection_reacquisition":"No detection availability events in S0 telemetry","controller_recovery":"No outage and settled-window events in S0 telemetry","miss_distance":"No target trajectory/capture event in S0 telemetry"},"known_issues": []}

def write_report(run_dir, output_path):
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); payload=json.dumps(build_baseline_report(run_dir),indent=2)+"\n"; out.write_text(payload); return out

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("run_dir"); p.add_argument("--output",required=True); a=p.parse_args(argv); print(write_report(a.run_dir,a.output))
if __name__=="__main__": main()
