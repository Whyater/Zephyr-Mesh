#!/usr/bin/env python3
"""Explain why reproduced run outputs do not match their registered sha256.

reproduce_all.py only says whether each output file is byte-identical. When a run
is rebuilt on another operating system, a mismatch can come from harmless causes
(Windows line endings, last-digit float differences) or from a real change. This
tool compares an expected run folder (for example the committed sim/runs/<name>)
with a reproduced one (for example a reproduce_all.py --keep-temp tree) and prints
one JSON line per file with one of these classes:

  byte-identical      same bytes
  line-endings-only   same bytes after turning CRLF into LF
  gzip-wrapper-only   .gz files whose decompressed bytes are the same
  formatting-only     JSON files that parse to exactly the same values
  numeric-within-tol  equal under the float rule in sim/tests/platform_compare.py
                      (relative 1e-9, absolute 1e-12), applied here on every OS
  pixels-identical    PNG files with the same decoded pixels
  real-difference     anything else, with where it first differs and the largest gap

Usage:
  PYTHONPATH=. python tools/compare_reproduce_tolerance.py EXPECTED_DIR ACTUAL_DIR [FILE ...]

With no FILE arguments every file in EXPECTED_DIR is compared. Exit code 0 when no
file is a real difference, 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from sim.tests import platform_compare


@contextlib.contextmanager
def tolerance_rule() -> Iterator[None]:
    """Use the off-macOS tolerance rule of platform_compare, even on macOS."""
    saved = platform_compare.EXACT
    platform_compare.EXACT = False
    try:
        yield
    finally:
        platform_compare.EXACT = saved


def relative_gap(a: float, b: float) -> float:
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b))


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def json_leaf_differences(a: Any, b: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    """Every leaf where two parsed JSON trees are not exactly equal, as (path, a, b)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append((f"{path}.{key}", a.get(key, "<missing>"), b.get(key, "<missing>")))
            else:
                out.extend(json_leaf_differences(a[key], b[key], f"{path}.{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for index, (x, y) in enumerate(zip(a, b)):
            out.extend(json_leaf_differences(x, y, f"{path}[{index}]"))
        return out
    if a == b and type(a) is type(b):
        return []
    return [(path, a, b)]


def _compare_json(expected: str, actual: str) -> tuple[str, dict[str, Any]]:
    e, a = json.loads(expected), json.loads(actual)
    leaves = json_leaf_differences(e, a)
    if not leaves:
        return "formatting-only", {}
    gaps = [relative_gap(float(x), float(y)) for _, x, y in leaves if _is_number(x) and _is_number(y)]
    other = [(p, x, y) for p, x, y in leaves if not (_is_number(x) and _is_number(y))]
    info = {
        "leaves_differing": len(leaves),
        "max_relative_gap": max(gaps) if gaps else 0.0,
        "first": [str(v)[:120] for v in leaves[0]],
        "first_non_number": [str(v)[:120] for v in other[0]] if other else None,
    }
    with tolerance_rule():
        same = platform_compare.same_tree(a, e)
    return ("numeric-within-tol" if same else "real-difference"), info


def _compare_lines(expected: str, actual: str, is_csv: bool) -> tuple[str, dict[str, Any]]:
    e_lines, a_lines = expected.split("\n"), actual.split("\n")
    if len(e_lines) != len(a_lines):
        return "real-difference", {"reason": f"{len(e_lines)} lines expected, {len(a_lines)} found"}
    differing, worst, first, first_other = 0, 0.0, None, None
    with tolerance_rule():
        same = True
        for number, (x, y) in enumerate(zip(e_lines, a_lines), start=1):
            if x == y:
                continue
            differing += 1
            if is_csv:
                same = same and platform_compare.same_csv_row(y, x)
                xs = next(csv.reader(io.StringIO(x)), [])
                ys = next(csv.reader(io.StringIO(y)), [])
            else:
                xs, ys = x.split(), y.split()
                same = same and len(xs) == len(ys) and all(platform_compare.same_text(q, p) for p, q in zip(xs, ys))
            if len(xs) != len(ys):
                same = False
                first_other = first_other or {"line": number, "expected": x[:160], "found": y[:160]}
                continue
            for cx, cy in zip(xs, ys):
                if cx == cy:
                    continue
                nx, ny = _as_number(cx), _as_number(cy)
                if nx is None or ny is None:
                    first_other = first_other or {"line": number, "expected": cx[:80], "found": cy[:80]}
                    continue
                worst = max(worst, relative_gap(nx, ny))
                first = first or {"line": number, "expected": cx, "found": cy}
    info = {"lines_differing": differing, "max_relative_gap": worst, "first": first, "first_non_number": first_other}
    return ("numeric-within-tol" if same else "real-difference"), info


def _compare_png(expected: bytes, actual: bytes) -> tuple[str, dict[str, Any]]:
    import matplotlib.image as mpimg

    e = mpimg.imread(io.BytesIO(expected), format="png")
    a = mpimg.imread(io.BytesIO(actual), format="png")
    if e.shape != a.shape:
        return "real-difference", {"reason": f"image shape {e.shape} expected, {a.shape} found"}
    if np.array_equal(e, a):
        return "pixels-identical", {}
    changed = np.any(e != a, axis=-1) if e.ndim == 3 else (e != a)
    return "real-difference", {
        "pixels_differing": int(changed.sum()),
        "pixels_total": int(changed.size),
        "max_channel_gap": float(np.max(np.abs(e.astype(float) - a.astype(float)))),
    }


def classify(expected_path: Path, actual_path: Path) -> tuple[str, dict[str, Any]]:
    """Return (class, details) for one expected/actual file pair."""
    if not actual_path.is_file():
        return "real-difference", {"reason": "missing from the reproduced folder"}
    eb, ab = expected_path.read_bytes(), actual_path.read_bytes()
    if eb == ab:
        return "byte-identical", {}
    if eb.replace(b"\r\n", b"\n") == ab.replace(b"\r\n", b"\n"):
        return "line-endings-only", {"crlf_expected": eb.count(b"\r\n"), "crlf_found": ab.count(b"\r\n")}
    name = expected_path.name
    if name.endswith(".png"):
        return _compare_png(eb, ab)
    if name.endswith(".gz"):
        eb, ab = gzip.decompress(eb), gzip.decompress(ab)
        if eb == ab:
            return "gzip-wrapper-only", {}
        name = name[:-3]
    extra: dict[str, Any] = {}
    if (b"\r\n" in eb) != (b"\r\n" in ab):
        extra["line_endings_also_differ"] = True
        eb, ab = eb.replace(b"\r\n", b"\n"), ab.replace(b"\r\n", b"\n")
    try:
        et, at = eb.decode("utf-8"), ab.decode("utf-8")
    except UnicodeDecodeError:
        return "real-difference", {"reason": "binary file with different bytes", **extra}
    if name.endswith(".json"):
        try:
            cls, info = _compare_json(et, at)
        except json.JSONDecodeError:
            cls, info = _compare_lines(et, at, is_csv=False)
    else:
        cls, info = _compare_lines(et, at, is_csv=name.endswith(".csv"))
    return cls, {**info, **extra}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify why reproduced outputs differ from the expected ones.")
    parser.add_argument("expected_dir", type=Path)
    parser.add_argument("actual_dir", type=Path)
    parser.add_argument("files", nargs="*", help="file names relative to both folders (default: every file in EXPECTED_DIR)")
    args = parser.parse_args(argv)
    names = args.files or sorted(
        p.relative_to(args.expected_dir).as_posix() for p in args.expected_dir.rglob("*") if p.is_file()
    )
    real = False
    for name in names:
        cls, info = classify(args.expected_dir / name, args.actual_dir / name)
        real = real or cls == "real-difference"
        print(json.dumps({"file": name, "class": cls, **info}, default=str))
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
