"""Time EXP-016 at several --workers values and record memory (proposal benchmark).

Usage: ``PYTHONPATH=. ./.venv/bin/python tools/bench_exp016_workers.py --output /tmp/bench --workers 1 2 4``

Each run is a fresh process. Every 0.5 s the script reads ``ps`` and keeps the
largest resident memory seen for the parent and for each pool worker, plus the
largest total over the whole process tree. A process that lives less than one
sampling interval can be missed, so use this on runs of a minute or more. Wall
times depend on what else the machine is doing, so they are measurements of
this machine at that moment, not results. The script also checks that every run's outputs hash the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUTS = ("summary.json", "false_alarms.csv", "detection_delay.csv", "interference.csv",
           "thresholds.json", "exp016_false_alarms.png", "exp016_rule_300ms.png")


def _tree_rss_kib(root: int) -> dict[int, int]:
    table = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, check=True).stdout
    rows = [tuple(int(x) for x in line.split()) for line in table.splitlines() if line.strip()]
    children: dict[int, list[int]] = {}
    rss = {}
    for pid, ppid, kib in rows:
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kib
    found, stack = {}, [root]
    while stack:
        pid = stack.pop()
        if pid in rss:
            found[pid] = rss[pid]
        stack.extend(children.get(pid, []))
    return found


def run_once(output: Path, workers: int, pilot: bool) -> dict:
    command = [sys.executable, "tools/c_exp016_heartbeat.py", "--output", str(output), "--workers", str(workers)]
    if pilot:
        command.append("--pilot")
    start = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak_by_pid: dict[int, int] = {}
    peak_total = 0
    while proc.poll() is None:
        sample = _tree_rss_kib(proc.pid)
        for pid, kib in sample.items():
            peak_by_pid[pid] = max(peak_by_pid.get(pid, 0), kib)
        peak_total = max(peak_total, sum(sample.values()))
        time.sleep(0.5)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"run with {workers} workers failed: exit {proc.returncode}")
    worker_peaks = sorted((kib for pid, kib in peak_by_pid.items() if pid != proc.pid), reverse=True)
    return {"workers": workers, "wall_s": round(wall, 1),
            "parent_peak_mib": round(peak_by_pid.get(proc.pid, 0) / 1024, 1),
            "worker_peak_mib": [round(k / 1024, 1) for k in worker_peaks],
            "tree_peak_mib": round(peak_total / 1024, 1),
            "hashes": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in OUTPUTS}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    for w in args.workers:
        runs.append(run_once(args.output / f"w{w}", w, args.pilot))
        print(json.dumps({k: v for k, v in runs[-1].items() if k != "hashes"}), flush=True)
    report = {"runs": runs, "identical_outputs": len({json.dumps(r["hashes"], sort_keys=True) for r in runs}) == 1}
    base = runs[0]["wall_s"]
    for r in runs:
        r["speedup_vs_first"] = round(base / r["wall_s"], 2)
    (args.output / "bench.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in runs:
        print(f"workers={r['workers']} wall_s={r['wall_s']} speedup={r['speedup_vs_first']} "
              f"parent_mib={r['parent_peak_mib']} worker_mib={r['worker_peak_mib']} tree_mib={r['tree_peak_mib']}")
    print(f"identical_outputs={report['identical_outputs']}")
    return 0 if report["identical_outputs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
