"""Compare two EXP-016 output folders made with different degraded-link detectors.

Usage::

    PYTHONPATH=. ./.venv/bin/python tools/c_exp016_compare.py OLD_DIR NEW_DIR [--json report.json]

OLD_DIR is a run with the legacy detector (for example sim/runs/c5-heartbeat) and
NEW_DIR a run with ``--detector lost-as-low-snr``.  The detector only feeds the
interference cells, so the script checks that:

* the heartbeat files (false alarms, detection delay, both figures) are byte
  identical;
* ``thresholds.json`` and ``summary.json`` differ only in the text fields that
  name the detector;
* every row of ``interference.csv`` keeps its cell keys and episode count, the
  never-triggered cells stay never-triggered, no false triggers appear, and
  the fixed detector catches at least as many episodes as the old one.

It prints a per-row table of the interference changes and exits with status 1
if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BYTE_IDENTICAL = ("false_alarms.csv", "detection_delay.csv", "exp016_false_alarms.png", "exp016_rule_300ms.png")
TEXT_ONLY = {"thresholds.json": {"generating_command"}, "summary.json": {"interference_method"}}
KEY_COLUMNS = ("margin_db", "distance_m", "mean_duration_s", "noise_rise_db", "n_episodes")
CHANGED_COLUMNS = ("mean_trigger_s", "p95_trigger_s", "fraction_detected", "missed_episodes",
                   "false_triggers_per_hour", "mean_clear_after_episode_s")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_diff_paths(a, b, prefix: str = "") -> list[str]:
    """Return the key paths where two parsed JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(path)
            else:
                out += json_diff_paths(a[key], b[key], path)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [prefix]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff_paths(x, y, f"{prefix}[{i}]")
        return out
    return [] if a == b and type(a) is type(b) else [prefix]


def _num(text: str) -> float | None:
    return None if text == "" else float(text)


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def compare(old: Path, new: Path) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    for name in BYTE_IDENTICAL:
        a, b = sha256(old / name), sha256(new / name)
        check(f"{name} byte identical", a == b, f"old {a[:12]} new {b[:12]}")

    for name, allowed in TEXT_ONLY.items():
        a = json.loads((old / name).read_text(encoding="utf-8"))
        b = json.loads((new / name).read_text(encoding="utf-8"))
        paths = json_diff_paths(a, b)
        check(f"{name} differs only in {sorted(allowed)}", set(paths) <= allowed, f"differing keys: {paths}")

    old_cols, old_rows = read_rows(old / "interference.csv")
    new_cols, new_rows = read_rows(new / "interference.csv")
    check("interference.csv has the same columns", old_cols == new_cols, f"{old_cols} vs {new_cols}")
    check("interference.csv has the same number of rows", len(old_rows) == len(new_rows),
          f"{len(old_rows)} vs {len(new_rows)}")
    rows = []
    for i, (o, n) in enumerate(zip(old_rows, new_rows)):
        same_keys = all(o[c] == n[c] for c in KEY_COLUMNS)
        check(f"row {i}: cell keys and episode count unchanged", same_keys,
              ", ".join(f"{c} {o[c]}/{n[c]}" for c in KEY_COLUMNS if o[c] != n[c]))
        old_missed, new_missed = int(o["missed_episodes"]), int(n["missed_episodes"])
        episodes = int(o["n_episodes"])
        check(f"row {i}: missed episodes did not grow", new_missed <= old_missed, f"{old_missed} -> {new_missed}")
        check(f"row {i}: fraction detected did not drop",
              float(n["fraction_detected"]) >= float(o["fraction_detected"]),
              f"{o['fraction_detected']} -> {n['fraction_detected']}")
        check(f"row {i}: no false triggers", _num(o["false_triggers_per_hour"]) == 0.0
              and _num(n["false_triggers_per_hour"]) == 0.0,
              f"old {o['false_triggers_per_hour']} new {n['false_triggers_per_hour']}")
        if old_missed == episodes:
            check(f"row {i}: never-triggered cell stays never-triggered", new_missed == episodes,
                  f"missed {old_missed} -> {new_missed} of {episodes}")
        changed = [c for c in CHANGED_COLUMNS if o[c] != n[c]]
        rows.append({"row": i, **{c: o[c] for c in ("margin_db", "mean_duration_s", "noise_rise_db", "n_episodes")},
                     **{f"old_{c}": _num(o[c]) for c in CHANGED_COLUMNS},
                     **{f"new_{c}": _num(n[c]) for c in CHANGED_COLUMNS},
                     "changed_columns": changed})
    return {"old": str(old), "new": str(new), "ok": all(c["ok"] for c in checks), "checks": checks, "rows": rows}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    failed = [c for c in report["checks"] if not c["ok"]]
    print(f"old: {report['old']}\nnew: {report['new']}")
    print(f"checks: {len(report['checks']) - len(failed)} passed, {len(failed)} failed")
    for c in report["checks"]:
        if not c["ok"] or "byte identical" in c["check"] or "differs only" in c["check"]:
            print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}  ({c['detail']})")
    print("\ninterference.csv (old -> new)")
    print("margin rise mean_dur | caught (of n)  | mean trigger s  | p95 trigger s   | mean clear s")
    for r in report["rows"]:
        n = int(r["n_episodes"])
        caught = f"{n - int(r['old_missed_episodes'])} -> {n - int(r['new_missed_episodes'])} ({n})"
        print(f"{r['margin_db']:>6} {r['noise_rise_db']:>4} {r['mean_duration_s']:>8} | {caught:<14} | "
              f"{_fmt(r['old_mean_trigger_s'])} -> {_fmt(r['new_mean_trigger_s'])} | "
              f"{_fmt(r['old_p95_trigger_s'])} -> {_fmt(r['new_p95_trigger_s'])} | "
              f"{_fmt(r['old_mean_clear_after_episode_s'])} -> {_fmt(r['new_mean_clear_after_episode_s'])}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--json", type=Path, help="also write the full report as JSON")
    a = p.parse_args(argv)
    report = compare(a.old, a.new)
    print_report(report)
    if a.json:
        a.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
