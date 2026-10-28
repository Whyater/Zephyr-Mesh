"""Compare a rerun's outputs with a registered run folder.

Usage (repo root):
    PYTHONPATH=. python tools/e_compare_rerun.py sim/runs/e-exp030-tag /tmp/rerun/e-exp030-tag \
        --identical tag.csv,tag_success_delay.png --additive summary.json --report /tmp/compare.json

Every file named on --identical must match byte for byte.  Every JSON file
named on --additive may only gain keys: every value present in the registered
copy must be unchanged in the rerun.  Files named on --changed must differ
(for example a redrawn figure); an unchanged copy fails.  The exit code is
0 when every expectation holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_differences(old: Any, new: Any, path: str = "") -> dict[str, list[str]]:
    """Paths added, removed and changed between two JSON trees.

    Lists are compared item by item; a longer new list counts as added items.
    """
    found: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}
    if isinstance(old, dict) and isinstance(new, dict):
        for key in old:
            child = f"{path}/{key}"
            if key not in new:
                found["removed"].append(child)
            else:
                _merge(found, json_differences(old[key], new[key], child))
        found["added"].extend(f"{path}/{key}" for key in new if key not in old)
    elif isinstance(old, list) and isinstance(new, list):
        for index, (left, right) in enumerate(zip(old, new)):
            _merge(found, json_differences(left, right, f"{path}[{index}]"))
        found["removed"].extend(f"{path}[{index}]" for index in range(len(new), len(old)))
        found["added"].extend(f"{path}[{index}]" for index in range(len(old), len(new)))
    elif type(old) is not type(new) or old != new:
        found["changed"].append(path or "/")
    return found


def _merge(into: dict[str, list[str]], extra: dict[str, list[str]]) -> None:
    for key, values in extra.items():
        into[key].extend(values)


def compare(registered: Path, rerun: Path, *, identical: list[str], additive: list[str],
            changed: list[str]) -> dict:
    files = []
    ok = True
    for name in identical + additive + changed:
        left, right = registered / name, rerun / name
        entry: dict[str, Any] = {"file": name}
        if not left.is_file() or not right.is_file():
            entry["missing"] = [str(p) for p in (left, right) if not p.is_file()]
            entry["ok"] = False
            ok = False
            files.append(entry)
            continue
        entry["registered_sha256"] = sha256_file(left)
        entry["rerun_sha256"] = sha256_file(right)
        entry["byte_identical"] = entry["registered_sha256"] == entry["rerun_sha256"]
        if name in identical:
            entry["expected"] = "identical"
            entry["ok"] = entry["byte_identical"]
        elif name in additive:
            entry["expected"] = "additive"
            diff = json_differences(json.loads(left.read_text(encoding="utf-8")),
                                    json.loads(right.read_text(encoding="utf-8")))
            entry["added"] = diff["added"]
            entry["removed"] = diff["removed"]
            entry["changed"] = diff["changed"]
            entry["ok"] = not diff["removed"] and not diff["changed"]
        else:
            entry["expected"] = "changed"
            entry["ok"] = not entry["byte_identical"]
        ok = ok and entry["ok"]
        files.append(entry)
    return {"registered": str(registered), "rerun": str(rerun), "ok": ok, "files": files}


def _names(value: str | None) -> list[str]:
    return [item for item in (value or "").split(",") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registered", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--identical", default="")
    parser.add_argument("--additive", default="")
    parser.add_argument("--changed", default="")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    result = compare(args.registered, args.rerun, identical=_names(args.identical),
                     additive=_names(args.additive), changed=_names(args.changed))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8", newline="\n")
    for entry in result["files"]:
        status = "ok" if entry["ok"] else "FAILED"
        detail = "byte identical" if entry.get("byte_identical") else "differs"
        if entry.get("expected") == "additive":
            detail += (f", {len(entry['added'])} added, {len(entry['removed'])} removed, "
                       f"{len(entry['changed'])} changed")
        if "missing" in entry:
            detail = "missing " + ", ".join(entry["missing"])
        print(f"{status}: {entry['file']} ({entry.get('expected', '?')}): {detail}")
    print("all expectations hold" if result["ok"] else "some expectations failed")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
