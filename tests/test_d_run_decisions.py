"""Checks for the simulated decision-authority runner."""
from pathlib import Path
import json
import subprocess
import sys



def test_pilot_is_byte_stable_and_figures_are_png(tmp_path):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        subprocess.run([sys.executable, "tools/d_run_decisions.py", "--pilot", "--processes", "4", "--output", str(out)], check=True)
        outs.append(out)
    names = ["per_seed.csv", "timeline.csv", "summary.json", "decisions_outage.png", "decisions_traffic.png"]
    for name in names:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()
    m0 = json.loads((outs[0] / "manifest.json").read_text(encoding="utf-8"))
    m1 = json.loads((outs[1] / "manifest.json").read_text(encoding="utf-8"))
    for m in (m0, m1):
        for key in ("runtime_s", "git_revision", "git_tree_state", "platform", "n_processes", "argv"):
            m.pop(key, None)
    assert m0 == m1
    for name in ("decisions_outage.png", "decisions_traffic.png"):
        data = (outs[0] / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000


def test_worker_matches_simulator_directly():
    from tools.d_run_decisions import _worker
    from sim.coop.simulator import run
    job = {"mode": "onboard", "loss": 0.0, "outage": "none", "seed": 20261015}
    row = _worker(job)
    raw = run(__import__("tools.d_run_decisions", fromlist=["_config"])._config(job)).as_dict()
    assert row["seed"] == raw["seed"]
    assert row["violation_count"] == raw["summary"]["spherical_violations_count"]
    vals = [a["standoff_error_m"] for step in raw["steps"] for a in step["agents"] if a.get("active") and a.get("standoff_error_m") is not None]
    assert abs(row["standoff_error_m"] - float(__import__('numpy').median(vals))) < 1e-12
