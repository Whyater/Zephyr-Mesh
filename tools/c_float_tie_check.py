"""Exhaustive check that the exact-integer loss comparison changed no detector decision.

Usage::

    PYTHONPATH=. ./.venv/bin/python tools/c_float_tie_check.py [--json report.json]

numpy's mean of a bool array is an exact integer count divided by n, correctly
rounded. So the old loss value (1 - mean) depends only on (lost, n), the number
of lost packets and the packets in the window. The rest of update() did not
change. So if the old and new predicates agree on every (lost, n) a
configuration can reach, the detector's state sequence is identical for every
possible packet sequence, and every output of a run that only uses those
configurations is unchanged.

The script walks sim/ and tools/ for every literal DegradedLinkDetector( call (aliases
and subclasses are not scanned; none exist today),
requires each one to be a checked configuration, and enumerates every
(n, lost) with n from min_packets to window_packets. The old predicates are
the old expression verbatim; the new ones are the real detector methods; the
exact answer uses fractions.Fraction. The sanity case (threshold 0.2, n=5,
lost=1) must show the old and exact answers differing.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from sim.radio.interference import DegradedLinkDetector

ROOT = Path(__file__).resolve().parents[1]
CHECKED = [("runner and episode check", 20, 0.3, 3), ("evaluate_detector default", 10, 0.5, 3)]
TEST_CONFIGS = ([("test", 4, 0.5, 2), ("test", 20, 0.3, 3), ("test", 10, 0.6, 3), ("test", 5, 0.2, 3),
                 ("test", 5, 0.8, 3), ("test", 10, 0.15, 3), ("test", 10, 0.2, 3)]
                + [("test sweep", w, t, 3) for t in (0.2, 0.3, 0.5, 0.7) for w in range(3, 25)])
_CALL = re.compile(r"DegradedLinkDetector\(")


def old_predicates(lost: int, n: int, thr: float) -> tuple[bool, bool]:
    """Old trigger and clear tests, with the old expression verbatim."""
    rec = np.array([True] * (n - lost) + [False] * lost, dtype=bool)
    loss_rate = 1.0 - float(np.mean(rec))
    return bool(loss_rate >= thr), bool(loss_rate < thr / 2.0)


def exact_predicates(lost: int, n: int, thr: float) -> tuple[bool, bool]:
    t = Fraction(str(thr))
    return Fraction(lost, n) >= t, Fraction(lost, n) < t / 2


def enumerate_config(window: int, thr: float, min_packets: int) -> dict:
    """Every (kind, n, lost) where old and new disagree, and where new and exact disagree."""
    det = DegradedLinkDetector(window, thr, 6.0, min_packets)
    old_new, new_exact = [], []
    for n in range(min_packets, window + 1):
        for lost in range(n + 1):
            old, new = old_predicates(lost, n, thr), (det.loss_meets_trigger(lost, n), det.loss_below_clear(lost, n))
            exact = exact_predicates(lost, n, thr)
            for i, kind in enumerate(("trigger", "clear")):
                if old[i] != new[i]:
                    old_new.append((kind, n, lost))
                if new[i] != exact[i]:
                    new_exact.append((kind, n, lost))
    return {"window": window, "loss_threshold": thr, "min_packets": min_packets,
            "old_vs_new": old_new, "new_vs_exact": new_exact}


def find_constructions() -> list[dict]:
    """Every DegradedLinkDetector( call in sim/ and tools/, with parsed (window, threshold, min_packets)."""
    found = []
    for path in sorted(p for d in ("sim", "tools") for p in (ROOT / d).rglob("*.py")
                       if p != Path(__file__).resolve() and "tests" not in p.parts):
        text = path.read_text(encoding="utf-8")
        for m in _CALL.finditer(text):
            depth, i = 1, m.end()
            while i < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            entry = {"where": f"{path.relative_to(ROOT).as_posix()}:{text.count(chr(10), 0, m.start()) + 1}", "config": None}
            try:
                args = ast.parse("f(" + text[m.end():i - 1] + ")", mode="eval").body.args
                entry["config"] = tuple(ast.literal_eval(args[k]) for k in (0, 1, 3))
            except (SyntaxError, ValueError, IndexError):
                pass
            found.append(entry)
    return found


def run() -> dict:
    checked = {(w, t, m) for _, w, t, m in CHECKED}
    constructions = find_constructions()
    unknown = [c for c in constructions if c["config"] is None or tuple(c["config"]) not in checked]
    checked_res = [dict(enumerate_config(w, t, m), label=label) for label, w, t, m in CHECKED]
    test_res = [dict(enumerate_config(w, t, m), label=label) for label, w, t, m in TEST_CONFIGS]
    sanity = enumerate_config(5, 0.2, 3)
    sanity_ok = (("trigger", 5, 1) in sanity["old_vs_new"] and old_predicates(1, 5, 0.2)[0] is False
                 and exact_predicates(1, 5, 0.2)[0] is True)
    ok = (not unknown and bool(constructions) and sanity_ok
          and not any(r["old_vs_new"] for r in checked_res)
          and not any(r["new_vs_exact"] for r in checked_res + test_res + [sanity]))
    return {"ok": ok, "constructions": constructions, "unknown_constructions": unknown, "checked": checked_res,
            "test_configs": test_res, "sanity_ok": sanity_ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = ap.parse_args()
    rep = run()
    for c in rep["constructions"]:
        print(f"construction {c['where']}: (window, threshold, min_packets) = {c['config']}")
    print(f"constructions outside the checked list or unparsable: {len(rep['unknown_constructions'])}")
    for r in rep["checked"]:
        print(f"checked {r['label']} ({r['window']}, {r['loss_threshold']}, min {r['min_packets']}): "
              f"old vs new {len(r['old_vs_new'])}, new vs exact {len(r['new_vs_exact'])}")
    for r in rep["test_configs"]:
        if r["old_vs_new"] or r["new_vs_exact"]:
            print(f"info test config ({r['window']}, {r['loss_threshold']}, min {r['min_packets']}): "
                  f"old vs new {r['old_vs_new']}, new vs exact {r['new_vs_exact']}")
    print(f"sanity (0.2, n=5, lost=1: old False, exact True) observed: {rep['sanity_ok']}")
    print("RESULT:", "PASS" if rep["ok"] else "FAIL")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
