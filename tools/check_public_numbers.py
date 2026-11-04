"""Check the numbers in the public text against the run outputs they come from.

The project rule is that every published performance number comes from a run
that a script can regenerate.  This tool reads the public pages (README.md,
docs/RESULTS.md, docs/LIMITATIONS.md), finds every number in them, and looks
each one up in ``tools/public_number_sources.json``.

The mapping file lists short quotes from the public pages.  Each quote names,
in order, every number inside it and says what kind of number it is:

* ``performance``: a result.  It must name a source: a registered run output
  (a file listed under ``outputs`` in ``tools/reproduce_registry.json``) and a
  selector that picks the value out of it.
* ``setting``: an input chosen for a study (a speed, a rate, a seed count, the
  0.25 m research line).  A source is optional; when one is given it is checked
  the same way.
* ``not_a_claim``: version numbers, step numbers in commands, counts of files,
  words like "one quadcopter".

Each number ends up in one of these states:

* ``matched``: the source value, after ``scale``, rounds to the public value at
  the precision the public text shows (0.36 matches 0.3591, 0.34 matches
  0.338).  A wider ``tolerance`` is allowed only with a ``rounding_note``
  saying where the text states the rounding (for example "about").  A bound
  in the text ("under 1 cm") uses ``"compare": "below"`` (or ``"above"``).
* ``mismatched``: the source says something else.
* ``stale_quote``: a quote in the mapping is no longer in the public text (the
  text was edited); its numbers then also show as unmatched.
* ``unmatched``: a performance number with no source, a source file that is
  not a registered output, a source file whose bytes differ from the sha256 the
  registry records for it, or a number in the public text that no quote covers.

Known drift is listed in the mapping with ``"known": {"status": ..., "reason": ...}``.
A known item in the state it was listed with is reported as ``known`` and
does not fail.  A known item that now matches fails as ``stale_known`` so the
list gets cleaned up, the same way the privacy test treats its known hits.

Selectors are JSON lists so values with dots never need escaping: a string is
a key, an integer is a list index, and an object filters a list of rows (keys
may be dotted paths into each row; numbers compare within 1e-9).  An optional
``op`` (``min``, ``max`` or ``count``) folds the selected rows, using ``field``
(a selector relative to each row) for ``min`` and ``max``; ``field_minus``
makes each row's value a difference of two fields.  ``divide_by`` (a selector
in the same file) divides the result, ``transform`` (only ``log2``) is applied
next, then ``scale``.  CSV outputs load as a list of rows, with numeric cells
turned into floats.

Limits: the scanner reads digits and a short list of number words.  It does
not read signs, thousands separators, or claims made only in words ("every
packet arrives", "no failing cell").

Usage::

    python tools/check_public_numbers.py            # report, exit 1 on new drift
    python tools/check_public_numbers.py --json     # machine-readable report
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "tools" / "public_number_sources.json"

KINDS = {"performance", "setting", "not_a_claim"}
OK_STATES = {"matched", "known", "not_checked"}
FILTER_TOL = 1e-9

# A number that is not part of a name like s6-recovery, x86_64 or v0.2.4.
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\d)")
# Number words that can carry a claim.  "two thirds" is matched before "two".
WORD_VALUES = {
    "two thirds": 2 / 3, "half": 0.5, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "twice": 2, "once": 1,
}
NUMBER_WORD = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(WORD_VALUES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


class MappingError(ValueError):
    """The mapping file is malformed or no longer fits the public text."""


@dataclass(frozen=True)
class Token:
    file: str
    line: int
    start: int  # offset in the whole file
    end: int
    text: str


@dataclass
class Result:
    file: str
    line: int
    text: str
    kind: str
    state: str
    public_value: float | None = None
    source_values: list[float | None] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    detail: str = ""
    quote: str = ""


def find_tokens(rel: str, text: str) -> list[Token]:
    """Every digit number and number word in ``text``, in file order."""
    found: dict[int, Token] = {}
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line_of(pos: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    for pattern in (NUMBER, NUMBER_WORD):
        for m in pattern.finditer(text):
            if any(t.start <= m.start() < t.end for t in found.values()):
                continue
            found[m.start()] = Token(rel, line_of(m.start()), m.start(), m.end(), m.group())
    return [found[k] for k in sorted(found)]


def decimals_of(text: str) -> int:
    if "." in text and not re.search(r"[eE]", text):
        return len(text.split(".", 1)[1])
    return 0


def public_value(text: str) -> float:
    word = WORD_VALUES.get(text.lower())
    if word is not None:
        return float(word)
    return float(text)


def _num(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def _get(row: Any, dotted: str) -> Any:
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


def _row_matches(row: Any, flt: dict[str, Any]) -> bool:
    for key, want in flt.items():
        got = _get(row, key)
        if got is _MISSING:
            return False
        got, want = _num(got), _num(want)
        if isinstance(got, bool) or isinstance(want, bool):
            if got is not want and got != want:
                return False
        elif isinstance(got, (int, float)) and isinstance(want, (int, float)):
            if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=FILTER_TOL):
                return False
        elif got != want:
            return False
    return True


def select(data: Any, path: list[Any]) -> Any:
    """Walk ``path``: string keys, integer indexes, object row filters."""
    cur = data
    for step in path:
        if isinstance(step, dict):
            if not isinstance(cur, list):
                raise MappingError(f"filter {step} applied to a non-list")
            cur = [row for row in cur if _row_matches(row, step)]
        elif isinstance(step, bool):
            raise MappingError(f"bad selector step {step!r}")
        elif isinstance(step, int):
            if not isinstance(cur, list) or not -len(cur) <= step < len(cur):
                raise MappingError(f"index {step} out of range")
            cur = cur[step]
        elif isinstance(step, str):
            if isinstance(cur, list):
                if len(cur) != 1:
                    raise MappingError(f"key {step!r} needs exactly one row, found {len(cur)}")
                cur = cur[0]
            if not isinstance(cur, dict) or step not in cur:
                raise MappingError(f"missing key {step!r}")
            cur = cur[step]
        else:
            raise MappingError(f"bad selector step {step!r}")
    return cur


def _flatten(values: Any, skip_empty: bool = False) -> list[float]:
    """Numbers in ``values``.  ``skip_empty`` drops blank CSV cells and nulls (for min and max)."""
    out: list[float] = []
    for v in values if isinstance(values, list) else [values]:
        if isinstance(v, list):
            out.extend(_flatten(v, skip_empty))
        elif skip_empty and (v is None or v == ""):
            continue
        else:
            v = _num(v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            else:
                raise MappingError(f"value {v!r} is not a number")
    return out


def _one(x: Any) -> float:
    vals = _flatten(x)
    if len(vals) != 1:
        raise MappingError(f"expected one number, found {len(vals)}")
    return vals[0]


def evaluate_source(src: dict[str, Any], root: Path, cache: dict[str, Any]) -> float:
    rel = src["path"]
    if rel not in cache:
        path = root / rel
        if not path.is_file():
            raise MappingError(f"source file {rel} not found")
        if rel.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
                cache[rel] = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        elif rel.endswith(".csv"):
            with path.open(encoding="utf-8", newline="") as fh:
                cache[rel] = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        else:
            cache[rel] = json.loads(path.read_text(encoding="utf-8"))
    picked = select(cache[rel], list(src.get("select", [])))
    op = src.get("op")
    if op is None:
        if isinstance(picked, list):
            if len(picked) != 1:
                raise MappingError(f"selector picked {len(picked)} rows; add a filter or an op")
            picked = picked[0]
        if "field" in src:
            picked = select(picked, src["field"])
        values = _flatten(picked)
        if len(values) != 1:
            raise MappingError(f"selector picked {len(values)} values")
        value = values[0]
    elif op == "count":
        if not isinstance(picked, list):
            raise MappingError("count needs a list of rows")
        value = float(len(picked))
    elif op in ("min", "max"):
        rows = picked if isinstance(picked, list) else [picked]
        if not rows:
            raise MappingError(f"{op} over zero rows")
        if "field_minus" in src:  # a per-row difference, such as one arm's error minus another's
            vals = [_one(select(r, src["field"])) - _one(select(r, src["field_minus"])) for r in rows]
        else:
            vals = _flatten([select(r, src["field"]) if "field" in src else r for r in rows], skip_empty=True)
        if not vals:
            raise MappingError(f"{op} over zero values")
        value = min(vals) if op == "min" else max(vals)
    else:
        raise MappingError(f"unknown op {op!r}")
    if "divide_by" in src:  # a share, such as a gap over a published value, from the same file
        value /= _one(select(cache[rel], src["divide_by"]))
    transform = src.get("transform")
    if transform == "log2":
        value = math.log2(value)
    elif transform is not None:
        raise MappingError(f"unknown transform {transform!r}")
    return value * float(src.get("scale", 1.0))


def registered_outputs(registry: list[dict[str, Any]]) -> dict[str, str]:
    """Map each registered output path to its recorded sha256."""
    return {f"{entry['run_dir']}/{name}": digest
            for entry in registry for name, digest in entry.get("outputs", {}).items()}


def _hash_matches(rel: str, digest: str, root: Path, hashes: dict[str, bool]) -> bool:
    if rel not in hashes:
        path = root / rel
        hashes[rel] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return hashes[rel]


def _compare(public: float, text: str, actual: float, item: dict[str, Any]) -> bool:
    if "tolerance" in item:
        if not item.get("rounding_note"):
            raise MappingError(f"tolerance on {text!r} needs a rounding_note")
        tol = float(item["tolerance"])
    else:
        tol = 0.5 * 10 ** (-decimals_of(text))
    slack = 1e-9 * max(1.0, abs(public))
    compare = item.get("compare", "equal")
    if compare == "equal":
        return abs(actual - public) <= tol + slack
    if compare == "below":  # "under 1 cm": strictly less than the public bound
        return actual < public
    if compare == "above":  # "over an hour", "more than 2"
        return actual > public
    raise MappingError(f"unknown compare {compare!r} on {text!r}")


def check(mapping: dict[str, Any], root: Path = REPO_ROOT,
          registry: list[dict[str, Any]] | None = None) -> list[Result]:
    """Return one Result per number in the mapped public files."""
    if registry is None:
        registry = json.loads((root / "tools" / "reproduce_registry.json").read_text(encoding="utf-8"))
    registered = registered_outputs(registry)
    cache: dict[str, Any] = {}
    files = list(mapping["files"])
    texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in files}
    tokens = {rel: find_tokens(rel, texts[rel]) for rel in files}
    covered: set[tuple[str, int]] = set()
    results: list[Result] = []

    for entry in mapping["quotes"]:
        rel, quote = entry["file"], entry["quote"]
        if rel not in texts:
            raise MappingError(f"quote names {rel}, which is not in files")
        count = texts[rel].count(quote)
        if count == 0:
            # The public text changed under this quote: report it and keep checking the rest.
            results.append(Result(rel, 0, quote, "unknown", "stale_quote", quote=quote,
                                  detail="quote not found; the public text changed, so update the mapping"))
            continue
        if count != 1:
            raise MappingError(f"{rel}: quote found {count} times, needs exactly 1: {quote!r}")
        start = texts[rel].index(quote)
        inside = [t for t in tokens[rel] if start <= t.start and t.end <= start + len(quote)]
        items = [n if isinstance(n, dict) else {"text": n} for n in entry["numbers"]]
        if [t.text for t in inside] != [n["text"] for n in items]:
            raise MappingError(
                f"{rel}: quote numbers {[t.text for t in inside]} do not match mapping "
                f"{[n['text'] for n in items]}: {quote!r}")
        for tok, item in zip(inside, items):
            covered.add((rel, tok.start))
            kind = item.get("kind", entry.get("kind"))
            if kind not in KINDS:
                raise MappingError(f"{rel}:{tok.line}: {tok.text!r} needs a kind from {sorted(KINDS)}")
            results.append(_check_item(tok, item, kind, quote, root, registered, cache))

    for rel in files:
        for tok in tokens[rel]:
            if (rel, tok.start) not in covered:
                results.append(Result(rel, tok.line, tok.text, "unknown", "unmatched",
                                      detail="no quote in the mapping covers this number"))
    results.sort(key=lambda r: (files.index(r.file), r.line))
    return results


def _check_item(tok: Token, item: dict[str, Any], kind: str, quote: str, root: Path,
                registered: dict[str, str], cache: dict[str, Any]) -> Result:
    res = Result(tok.file, tok.line, tok.text, kind, "not_checked", quote=quote)
    srcs = item.get("sources") or ([item["source"]] if "source" in item else [])
    try:
        res.public_value = float(item["value"]) if "value" in item else public_value(tok.text)
    except ValueError as exc:
        raise MappingError(f"{tok.file}:{tok.line}: cannot read {tok.text!r}") from exc
    if kind == "not_a_claim":
        if srcs:
            raise MappingError(f"{tok.file}:{tok.line}: not_a_claim {tok.text!r} has a source")
        return res
    if not srcs:
        if kind == "performance":
            res.state, res.detail = "unmatched", "performance number with no registered source"
        return _apply_known(res, item)
    states = []
    for src in srcs:
        res.sources.append(src["path"])
        if src["path"] not in registered:
            res.source_values.append(None)
            states.append("unmatched")
            res.detail = f"{src['path']} is not a registered run output"
            continue
        if not _hash_matches(src["path"], registered[src["path"]], root, cache.setdefault("#hashes", {})):
            res.source_values.append(None)
            states.append("unmatched")
            res.detail = f"{src['path']} differs from its sha256 in the run registry"
            continue
        try:
            actual = evaluate_source(src, root, cache)
        except MappingError as exc:
            raise MappingError(f"{tok.file}:{tok.line}: {tok.text!r}: {exc}") from exc
        res.source_values.append(actual)
        states.append("matched" if _compare(res.public_value, tok.text, actual, item) else "mismatched")
    if "mismatched" in states:
        res.state = "mismatched"
        res.detail = res.detail or "source value does not round to the public value"
    elif "unmatched" in states:
        res.state = "unmatched"
    else:
        res.state = "matched"
    return _apply_known(res, item)


def _apply_known(res: Result, item: dict[str, Any]) -> Result:
    known = item.get("known")
    if not known:
        return res
    if not known.get("reason"):
        raise MappingError(f"{res.file}:{res.line}: known item {res.text!r} needs a reason")
    if known.get("status") == res.state:
        res.detail = f"known: {known['reason']}"
        res.state = "known"
    else:
        res.detail = f"listed as known {known.get('status')!r} but is now {res.state!r}: update the mapping"
        res.state = "stale_known"
    return res


def failures(results: list[Result]) -> list[Result]:
    return [r for r in results if r.state not in OK_STATES]


def format_report(results: list[Result]) -> str:
    lines = []
    counts: dict[str, int] = {}
    for r in results:
        counts[r.state] = counts.get(r.state, 0) + 1
    lines.append("states: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in results:
        if r.kind == "not_a_claim" or (r.state == "not_checked" and r.kind == "setting"):
            continue
        vals = ", ".join("none" if v is None else f"{v:.6g}" for v in r.source_values)
        src = "; ".join(r.sources)
        lines.append(f"{r.state:11} {r.file}:{r.line}  {r.text!r} ({r.kind})"
                     + (f"  source {src} = {vals}" if src else "")
                     + (f"  [{r.detail}]" if r.detail else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    results = check(mapping, args.root)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=1))
    else:
        print(format_report(results))
    bad = failures(results)
    if bad:
        print(f"\n{len(bad)} number(s) need attention", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
