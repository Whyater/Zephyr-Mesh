"""Check that the opt-in NEES divergence wiring leaves registered runs unchanged.

The wiring adds a divergence rule that is off unless a caller asks for it. This
script gathers evidence that, with the rule off, every registered runner still
produces the same output as the base commit.

Part A (static): list the files that changed against the base, follow the import
graph of every registered command, and report which registered runs can reach a
changed file at all. Runs that cannot reach one are unchanged by construction.
It also checks that the registry file itself is unchanged and looks for source
hashes inside registered outputs.

Part B (dynamic): for each registered runner that can reach a changed file, build
the runner's real job list, pick a sample that covers every value of every grid
axis, run the runner's real job function in the base tree and in the working
tree (separate subprocesses, one core), and compare a hash of each result.

Part C (optional, slow): rerun a whole registered entry from the working tree with
tools/reproduce_all.py and compare it with the registry hashes.

Exit code 0 means Part A and Part B passed (and Part C, if it was asked for).
"""
from __future__ import annotations

import argparse
import ast
import functools
import gzip
import hashlib
import io
import json
import os
import random
import re
import resource
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

BASE_REV = "0750eca"
REGISTRY = "tools/reproduce_registry.json"
SEED = 20261020

# Files that are listed but never counted for reachability.
IGNORED_PREFIXES = ("tests/", "tools/d_divergence_")

# Runner scripts that have a dynamic check in Part B.
RUNNERS = {
    "tools/d_run_fusion.py": "fusion",
    "tools/d_run_tolerance.py": "tolerance",
}

# Grid axes that must each be covered at least once (job key names).
AXES = {
    "fusion": ("n_observers", "noise_std_m", "target_label", "loss_probability", "ci"),
    "tolerance": ("n", "link", "delay", "loss_model", "loss", "target"),
}

# Rough CPU seconds per job, from timing single jobs on the base tree. They only
# steer the sample towards cheap jobs; they are never used as results.
def estimated_cost(runner: str, job: dict) -> float:
    if runner == "fusion":
        n = int(job["n_observers"])
        if job["ci"] == "exact":
            return {2: 2.2, 3: 6.6, 6: 43.0}.get(n, 20.0)
        return 0.36 * n
    n = int(job["n"])
    per_n = {"abstract": 0.27, "radio_gym": 0.38, "radio_stress": 2.1}[str(job["link"])]
    return 0.1 + per_n * n


# ---------------------------------------------------------------------------
# git helpers


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=False)
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def git_bytes(repo: Path, *args: str) -> bytes | None:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          check=False)
    return proc.stdout if proc.returncode == 0 else None


def changed_files(repo: Path, base: str) -> dict:
    """Files that differ from the base: tracked changes plus untracked .py in sim/ and tools/."""
    tracked = [p for p in git(repo, "diff", "--name-only", "-z", base).split("\0") if p]
    untracked = []
    entries = git(repo, "status", "--porcelain", "-z", "-uall").split("\0")
    skip_next = False
    for entry in entries:
        if skip_next:
            skip_next = False
            continue
        if len(entry) < 4:
            continue
        if entry[0] in "RC" or entry[1] in "RC":
            skip_next = True  # the next entry is the old name of a rename
        if entry.startswith("??"):
            untracked.append(entry[3:])
    kept_untracked = [p for p in untracked
                      if p.endswith(".py") and p.startswith(("sim/", "tools/"))]
    paths = sorted(set(tracked) | set(kept_untracked))
    ignored = [p for p in paths if p.startswith(IGNORED_PREFIXES)]
    relevant = [p for p in paths if p not in ignored]
    return {
        "tracked_changed": sorted(tracked),
        "untracked_considered": sorted(kept_untracked),
        "untracked_not_considered": len(untracked) - len(kept_untracked),
        "ignored": ignored,
        "relevant": relevant,
        "relevant_python": [p for p in relevant if p.endswith(".py")],
        "relevant_other": [p for p in relevant if not p.endswith(".py")],
    }


# ---------------------------------------------------------------------------
# Part A: import closure


def module_file(search_roots: list[Path], root: Path, dotted: str) -> str | None:
    """Repo-relative path of the file a dotted module name points at, or None."""
    parts = dotted.split(".")
    for base in search_roots:
        for candidate in (base.joinpath(*parts).with_suffix(".py"),
                          base.joinpath(*parts, "__init__.py")):
            if candidate.is_file():
                try:
                    return candidate.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    continue
    return None


def _file_package(rel: str) -> list[str]:
    """Package parts of a repo file: its directory, for modules and __init__.py alike."""
    return list(Path(rel).parent.parts)


_STRING_REF = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_STRING_PATH = re.compile(r"^[\w./-]+\.py$")


def _candidate_modules(tree: ast.AST, rel: str) -> tuple[set[str], set[str]]:
    """Dotted module names a parsed file may import, and string paths it names."""
    package = _file_package(rel)
    modules: set[str] = set()
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                if keep < 0:
                    continue
                prefix = package[:keep]
                base = ".".join(prefix + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                modules.add(base)
            for alias in node.names:
                if alias.name != "*":
                    modules.add(f"{base}.{alias.name}" if base else alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _STRING_REF.match(node.value):
                modules.add(node.value.split(":")[0])
            elif _STRING_PATH.match(node.value):
                paths.add(node.value)
        elif isinstance(node, ast.Call):
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if (name in ("import_module", "__import__") and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                modules.add(node.args[0].value)
    return modules, paths


@functools.cache
def _parsed(root: str, rel: str) -> tuple[frozenset[str], frozenset[str]]:
    """Candidate module names and string paths of one file (cached; raises on a bad file)."""
    text = (Path(root) / rel).read_text(encoding="utf-8")
    modules, paths = _candidate_modules(ast.parse(text, filename=rel), rel)
    return frozenset(modules), frozenset(paths)


def import_closure(root: Path, entries: list[str]) -> tuple[dict[str, str | None], list[str]]:
    """Repo files reachable from the entry files through imports.

    Returns ({file: file that first pulled it in, or None for an entry}, errors).
    Imports inside functions count. For ``import a.b.c`` every parent package
    ``__init__.py`` counts. For ``from a.b import c`` the name ``c`` counts as a
    submodule if such a file exists. Module names also resolve against the folder
    of each entry script, because Python puts that folder on sys.path. Strings of
    the form ``module:function`` and literal ``import_module`` arguments count.
    """
    root = root.resolve()
    reached: dict[str, str | None] = {}
    errors: list[str] = []
    queue: list[str] = []
    entry_dirs: list[Path] = []
    for entry in entries:
        if (root / entry).is_file():
            reached.setdefault(entry, None)
            queue.append(entry)
            entry_dirs.append((root / entry).parent)
        else:
            errors.append(f"entry file not found: {entry}")
    search_roots = [root] + [d for d in entry_dirs if d != root]

    def add(rel: str | None, parent: str) -> None:
        if rel is not None and rel not in reached:
            reached[rel] = parent
            queue.append(rel)

    while queue:
        rel = queue.pop()
        try:
            modules, paths = _parsed(str(root), rel)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"cannot parse {rel}: {exc}")
            continue
        for dotted in modules:
            pieces = dotted.split(".")
            for end in range(1, len(pieces) + 1):  # every parent package is imported too
                add(module_file(search_roots, root, ".".join(pieces[:end])), rel)
        for path in paths:
            if (root / path).is_file():
                add(path, rel)
    return reached, errors


def chain_to(reached: dict[str, str | None], target: str) -> list[str]:
    chain = [target]
    while reached.get(chain[-1]) is not None:
        chain.append(reached[chain[-1]])
        if len(chain) > 100:
            break
    return list(reversed(chain))


def command_scripts(command: str) -> list[str]:
    """Every ``*.py`` token in a registry command."""
    found = []
    for token in re.findall(r"[\w./-]+\.py\b", command):
        if token not in found:
            found.append(token)
    return found


_HASH_WORDS = re.compile(
    r"_source_paths|hashed_sources|code_sha256|sha256|hashlib|_checkpoint_fingerprint"
    r"|write_manifest|fingerprint")
_HEX64 = re.compile(rb"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")


@functools.cache
def base_source_hashes(repo: Path, base: str) -> frozenset[bytes]:
    """sha256 hex digests of every .py file at the base commit (one git call)."""
    proc = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", base],
                          capture_output=True, check=True)
    digests = set()
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                digests.add(hashlib.sha256(tar.extractfile(member).read()).hexdigest().encode())
    return frozenset(digests)


def scan_source_hashes(repo: Path, base: str, entry: dict, reached: dict[str, str | None]) -> dict:
    """Find where a runner writes hashes of source files, and whether a registered output holds one."""
    root = repo.resolve()
    hits: dict[str, list[str]] = {}
    for rel in sorted(reached):
        if not rel.startswith("tools/") and rel != "sim/estimation/sweep.py":
            continue
        lines = []
        text = (root / rel).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(text, 1):
            if _HASH_WORDS.search(line):
                lines.append(f"{number}: {line.strip()[:110]}")
        if lines:
            hits[rel] = lines
    outputs = list(entry["outputs"])
    has_manifest_call = any("write_manifest(" in line for lines in hits.values() for line in lines)
    uses_checkpoint = any("_checkpoint_fingerprint" in line for lines in hits.values()
                          for line in lines)
    conclusion = []
    if has_manifest_call:
        registered = "manifest.json" in outputs
        conclusion.append(
            "hashes of imported source files go to manifest.json (code_sha256); "
            f"manifest.json is {'REGISTERED' if registered else 'not registered'}")
    else:
        registered = False
    if uses_checkpoint:
        used = "--checkpoint" in entry["command"]
        conclusion.append(
            "the checkpoint fingerprint hashes source files and is stored only in "
            f"checkpoint files; the registered command {'uses' if used else 'does not use'} "
            "--checkpoint")
    # Empirical check: look for any source-file hash inside the registered outputs.
    out_dir = root / entry["run_dir"]
    source_hashes = set(base_source_hashes(repo, base))
    for rel in sorted(reached):
        if (root / rel).is_file():
            source_hashes.add(hashlib.sha256((root / rel).read_bytes()).hexdigest().encode())
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():  # hashes the committed run recorded for its own sources
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("code_sha256", {})
        source_hashes.update(str(v).encode() for v in recorded.values())
    empirical = {}
    for name in outputs:
        path = out_dir / name
        if not path.is_file():
            empirical[name] = "file not present in this checkout"
            continue
        data = path.read_bytes()
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        tokens = set(_HEX64.findall(data))
        empirical[name] = {"hex64_tokens": len(tokens),
                           "matching_source_hashes": len(tokens & source_hashes)}
    clean = all(isinstance(v, dict) and v["matching_source_hashes"] == 0
                for v in empirical.values())
    return {"hits": hits, "conclusion": conclusion, "manifest_registered": registered,
            "uses_checkpoint": uses_checkpoint, "registered_outputs_scan": empirical,
            "no_source_hash_in_registered_outputs": clean}


def part_a(repo: Path, base: str) -> dict:
    changes = changed_files(repo, base)
    registry_text = (repo / REGISTRY).read_bytes()
    base_registry = git_bytes(repo, "show", f"{base}:{REGISTRY}")
    registry_unchanged = base_registry is not None and base_registry == registry_text
    entries = json.loads(registry_text.decode("utf-8"))
    changed_py = set(changes["relevant_python"])
    reach = {}
    scans = {}
    problems = []
    if changes["relevant_other"]:
        problems.append("changed non-Python files outside tests/ and tools/d_divergence_*: "
                        + ", ".join(changes["relevant_other"]))
    if not registry_unchanged:
        problems.append(f"{REGISTRY} differs from the base")
    for entry in entries:
        scripts = command_scripts(entry["command"])
        reached, errors = import_closure(repo, scripts)
        per_script = {}
        for script in scripts:
            closure, _ = import_closure(repo, [script])
            touched = sorted(changed_py & set(closure))
            per_script[script] = {
                "reaches": touched,
                "chains": {path: chain_to(closure, path) for path in touched},
            }
        touched_all = sorted(changed_py & set(reached))
        info = {"command_scripts": scripts, "closure_size": len(reached),
                "reaches_changed": touched_all, "per_script": per_script, "errors": errors}
        if errors:
            problems.append(f"{entry['run_dir']}: " + "; ".join(errors))
        if touched_all:
            runner_scripts = [s for s, v in per_script.items() if v["reaches"]]
            info["runner_scripts_reaching"] = runner_scripts
            if not any(s in RUNNERS for s in runner_scripts):
                problems.append(f"{entry['run_dir']} reaches changed files but has no dynamic "
                                "check in Part B")
            scans[entry["run_dir"]] = scan_source_hashes(repo, base, entry, reached)
        reach[entry["run_dir"]] = info
    reaching = sorted(k for k, v in reach.items() if v["reaches_changed"])
    return {
        "base": base, "changes": changes, "registry_unchanged": registry_unchanged,
        "entries_total": len(entries), "entries_reaching_changed_files": reaching,
        "reach": reach, "source_hash_scan": scans, "problems": problems,
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Part B: dynamic equivalence

WORKER_SOURCE = r'''
import contextlib, hashlib, importlib, io, json, re, sys, tempfile, time
from pathlib import Path

MODULES = {"fusion": "tools.d_run_fusion", "tolerance": "tools.d_run_tolerance"}
VOLATILE = re.compile(r"(cpu|runtime|elapsed|wall)", re.I)


def strip(obj, found):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if VOLATILE.search(str(key)):
                found.add(str(key))
                continue
            out[key] = strip(value, found)
        return out
    if isinstance(obj, (list, tuple)):
        return [strip(v, found) for v in obj]
    return obj


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture(runner, argv):
    """Call the runner's own main() with run_jobs replaced by a recorder."""
    module = importlib.import_module(MODULES[runner])
    calls = []

    def fake(jobs, fn, **kwargs):
        calls.append({"fn": fn, "jobs": [dict(j) for j in jobs]})
        return []

    module.run_jobs = fake
    ended = "completed"
    with tempfile.TemporaryDirectory() as tmp:
        argv = [a.replace("{out}", tmp + "/out") for a in argv]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(argv)
        except (Exception, SystemExit) as exc:
            ended = type(exc).__name__
    return {"calls": calls, "ended": ended,
            "row_columns": list(getattr(module, "ROW_COLUMNS", []))}


def aggregate(runner, module, results, jobs):
    """Run the runner's pure aggregation helpers on the sampled rows."""
    rows = [row for result in results for row in result["rows"]]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if runner == "fusion":
                fast = [r for r in rows if r["ci"] == "fast"]
                exact = [r for r in rows if r["ci"] != "fast"]
                csv_path = Path(tmp) / "per_seed.csv"
                module.write_csv(csv_path, fast + exact, module.ROW_COLUMNS)
                summary = {"cells": module._summarize(fast),
                           "exact": module._exact_subset(fast, exact)}
                return {"csv": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        "summary": sha(json.dumps(summary, sort_keys=True, default=repr))}
            grouped = module._group_rows(rows)
            seeds = sorted({int(j["seed"]) for j in jobs})
            module._write_cells(Path(tmp), grouped, seeds)
            return {
                "cells_csv": hashlib.sha256((Path(tmp) / "cells.csv").read_bytes()).hexdigest(),
                "gaps": sha(json.dumps(module._build_gaps(grouped), sort_keys=True, default=repr)),
                "channel_use": sha(json.dumps(module._build_channel_use(grouped),
                                              sort_keys=True, default=repr)),
            }
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)}


def run(runner, fn_name, jobs):
    module_name, func_name = fn_name.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    found = set()
    items = []
    raw = []
    for job in jobs:
        start = time.process_time()
        result = fn(job)
        cpu = time.process_time() - start
        text = json.dumps(strip(result, found), sort_keys=True, default=repr)
        items.append({"sha256": sha(text), "cpu_s": cpu, "bytes": len(text)})
        raw.append(result)
    return {"items": items, "stripped_keys": sorted(found),
            "aggregate": aggregate(runner, module, raw, jobs)}


request = json.load(open(sys.argv[1]))
if request["mode"] == "capture":
    answer = capture(request["runner"], request["argv"])
else:
    answer = run(request["runner"], request["fn"], request["jobs"])
json.dump(answer, open(sys.argv[2], "w"))
'''


def run_worker(python: str, tree: Path, request: dict, workdir: Path) -> tuple[dict, float]:
    """Run the worker in ``tree`` (cwd and PYTHONPATH) and return its answer and child CPU seconds."""
    req_path = workdir / "request.json"
    out_path = workdir / "answer.json"
    req_path.write_text(json.dumps(request, default=repr), encoding="utf-8")
    if out_path.exists():
        out_path.unlink()
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tree), "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1"})
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run([python, "-c", WORKER_SOURCE, str(req_path), str(out_path)],
                          cwd=tree, env=env, capture_output=True, text=True, check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    if proc.returncode or not out_path.is_file():
        raise RuntimeError(f"worker failed in {tree} (exit {proc.returncode}):\n"
                           + proc.stderr[-2000:])
    return json.loads(out_path.read_text(encoding="utf-8")), cpu


def runner_argv(command: str, script: str) -> list[str]:
    """Arguments the registry command gives the runner script."""
    for chunk in command.split("&&"):
        parts = shlex.split(chunk, posix=True)
        for index, part in enumerate(parts):
            if part.endswith(script):
                return parts[index + 1:]
    raise ValueError(f"{script} not found in {command!r}")


def stratified_sample(jobs: list[dict], axes: tuple[str, ...], cost, must: list,
                      seed: int, extra: int = 0, extra_cost_cap: float = 3.0) -> list[int]:
    """Indices of a sample that covers every value of every axis at least once.

    ``must`` is a list of (predicate, count) pairs; those jobs are taken first
    (distinct seeds where possible). The rest is a greedy cover that prefers jobs
    covering many new values per CPU second. ``extra`` random cheap jobs are added.
    """
    rng = random.Random(seed)
    wanted = {(axis, repr(job[axis])) for job in jobs for axis in axes}
    chosen: list[int] = []
    covered: set = set()

    def take(index: int) -> None:
        if index in chosen:
            return
        chosen.append(index)
        covered.update((axis, repr(jobs[index][axis])) for axis in axes)

    for predicate, count in must:
        pool = [i for i, job in enumerate(jobs) if predicate(job) and i not in chosen]
        rng.shuffle(pool)
        used_seeds: set = set()
        taken = 0
        for index in pool:
            if jobs[index]["seed"] in used_seeds and len(pool) > count:
                continue
            take(index)
            used_seeds.add(jobs[index]["seed"])
            taken += 1
            if taken == count:
                break
    order = list(range(len(jobs)))
    rng.shuffle(order)
    while covered != wanted:
        best, best_score = None, 0.0
        for index in order:
            new = len({(axis, repr(jobs[index][axis])) for axis in axes} - covered)
            if not new:
                continue
            score = new / (cost(jobs[index]) + 0.05)
            if score > best_score:
                best, best_score = index, score
        if best is None:
            break
        take(best)
    cheap = [i for i in order if i not in chosen and cost(jobs[i]) <= extra_cost_cap]
    for index in cheap[:extra]:
        take(index)
    return chosen


def build_sample(runner: str, jobs: list[dict], seed: int) -> tuple[list[int], dict]:
    axes = AXES[runner]
    if runner == "fusion":
        must = [
            # peer_naive is the arm that fails in every N=6 run; N=6 fast jobs run it.
            (lambda j: j["ci"] == "fast" and j["n_observers"] == 6
             and j["target_label"] == "turn", 3),
            (lambda j: j["ci"] == "fast" and j["n_observers"] == 6
             and j["target_label"] != "turn" and j["loss_probability"] > 0, 2),
            # one exact-CI job at N=6, the heaviest registered computation
            (lambda j: j["ci"] == "exact" and j["n_observers"] == 6, 1),
            (lambda j: j["ci"] == "exact" and j["n_observers"] == 3, 1),
        ]
        extra = 8
    else:
        must = [
            (lambda j: j["link"] == "abstract" and j["n"] == 6, 2),
            (lambda j: j["link"] == "radio_gym" and j["n"] == 6, 1),
            (lambda j: j["link"] == "radio_stress" and j["n"] == 6, 1),
            (lambda j: j["loss_model"] == "burst" and j["n"] == 4, 1),
        ]
        extra = 8
    def cost(job: dict) -> float:
        return estimated_cost(runner, job)

    indices = stratified_sample(jobs, axes, cost, must, seed, extra=extra)
    coverage = {axis: sorted({repr(jobs[i][axis]) for i in indices}) for axis in axes}
    universe = {axis: sorted({repr(job[axis]) for job in jobs}) for axis in axes}
    return indices, {"axes": axes, "covered": coverage,
                     "all_values_covered": all(coverage[a] == universe[a] for a in axes),
                     "universe": universe}


def part_b(repo: Path, base: str, python: str, runners: list[str], registry: list[dict],
           seed: int, workdir: Path) -> dict:
    """Compare sampled job results between the base export and the working tree."""
    base_tree = workdir / "base"
    base_tree.mkdir()
    archive = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", base],
                               stdout=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(base_tree)], stdin=archive.stdout, check=False)
    archive.stdout.close()
    if archive.wait() or untar.returncode:
        raise RuntimeError("could not export the base tree")
    result = {"base_tree_export": "git archive, plain directory", "runners": {}, "passed": True}
    for script in runners:
        runner = RUNNERS[script]
        entry = next(e for e in registry if script in command_scripts(e["command"]))
        argv = runner_argv(entry["command"], script)
        info: dict = {"script": script, "registry_run": entry["run_dir"], "argv": argv,
                      "failures": []}
        scratch = workdir / runner
        scratch.mkdir()
        captured = {}
        for label, tree in (("base", base_tree), ("new", repo)):
            answer, _ = run_worker(python, tree, {"mode": "capture", "runner": runner,
                                                  "argv": argv}, scratch)
            captured[label] = answer
        base_calls = captured["base"]["calls"]
        if not base_calls or not any(c["jobs"] for c in base_calls):
            raise RuntimeError(f"no jobs captured from {script}")
        info["jobs_identical_base_vs_new"] = (captured["base"]["calls"] == captured["new"]["calls"]
                                              and captured["base"]["row_columns"]
                                              == captured["new"]["row_columns"])
        if not info["jobs_identical_base_vs_new"]:
            info["failures"].append("job list or ROW_COLUMNS differs between base and new")
        fn_names = {c["fn"] for c in base_calls if c["jobs"]}
        if len(fn_names) != 1:
            raise RuntimeError(f"expected one job function, got {fn_names}")
        fn_name = fn_names.pop()
        jobs = [dict(j) for c in base_calls for j in c["jobs"]]
        indices, coverage = build_sample(runner, jobs, seed)
        sample = [jobs[i] for i in indices]
        info.update({"job_function": fn_name, "total_jobs": len(jobs), "sampled_jobs": len(sample),
                     "calls_captured": [len(c["jobs"]) for c in base_calls],
                     "coverage": coverage,
                     "estimated_cpu_s": round(sum(estimated_cost(runner, j) for j in sample), 1)})
        if runner == "fusion":
            info["peer_naive_n6_jobs"] = sum(1 for j in sample if j["ci"] == "fast"
                                             and j["n_observers"] == 6)
        if not coverage["all_values_covered"]:
            info["failures"].append("sample does not cover every grid axis value")
        request = {"mode": "run", "runner": runner, "fn": fn_name, "jobs": sample}
        started = time.perf_counter()
        base_answer, base_cpu = run_worker(python, base_tree, request, scratch)
        new_answer, new_cpu = run_worker(python, repo, request, scratch)
        info["wall_s_both_trees"] = round(time.perf_counter() - started, 1)
        info["cpu_s"] = {"base": round(base_cpu, 1), "new": round(new_cpu, 1)}
        info["stripped_keys"] = sorted(set(base_answer["stripped_keys"])
                                       | set(new_answer["stripped_keys"]))
        mismatches = []
        for position, (a, b) in enumerate(zip(base_answer["items"], new_answer["items"])):
            if a["sha256"] != b["sha256"]:
                mismatches.append({"position": position, "job": sample[position],
                                   "base_sha256": a["sha256"], "new_sha256": b["sha256"]})
        if len(base_answer["items"]) != len(new_answer["items"]):
            info["failures"].append("different number of results")
        info["mismatched_jobs"] = mismatches
        info["aggregate_base"] = base_answer["aggregate"]
        info["aggregate_new"] = new_answer["aggregate"]
        info["aggregate_identical"] = base_answer["aggregate"] == new_answer["aggregate"]
        info["aggregate_errored"] = "error" in base_answer["aggregate"]
        if mismatches:
            info["failures"].append(f"{len(mismatches)} job result(s) differ")
        if not info["aggregate_identical"]:
            info["failures"].append("aggregation output differs")
        # Control: run the base tree again on the three cheapest sampled jobs. If this
        # does not repeat exactly, a hash match between trees would prove nothing.
        order = sorted(range(len(sample)), key=lambda i: estimated_cost(runner, sample[i]))[:3]
        control_jobs = [sample[i] for i in order]
        control, _ = run_worker(python, base_tree, {**request, "jobs": control_jobs}, scratch)
        info["control_repeat_identical"] = all(
            control["items"][k]["sha256"] == base_answer["items"][i]["sha256"]
            for k, i in enumerate(order))
        if not info["control_repeat_identical"]:
            info["failures"].append("base tree is not deterministic on a repeat run")
        if runner == "fusion":
            # Negative control: with the opt-in flag on, the new tree must NOT match the base.
            # If it did, this comparison would be unable to see the wiring at all.
            fast = [i for i in order if sample[i]["ci"] == "fast"]
            if fast:
                flagged = {**sample[fast[0]], "nees_divergence": True}
                on, _ = run_worker(python, repo, {**request, "jobs": [flagged]}, scratch)
                info["negative_control_flag_on_differs"] = (
                    on["items"][0]["sha256"] != base_answer["items"][fast[0]]["sha256"])
                if not info["negative_control_flag_on_differs"]:
                    info["failures"].append("flag-on result equals the base, so the check is blind")
        info["passed"] = not info["failures"]
        result["passed"] = result["passed"] and info["passed"]
        result["runners"][runner] = info
    return result


# ---------------------------------------------------------------------------
# Part C: full rerun


def part_c(repo: Path, name: str, python: str, json_path: Path) -> dict:
    command = [python, "tools/reproduce_all.py", "--only", name, "--source-dir", str(repo),
               "--workers", "1", "--python", python, "--json", str(json_path)]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    report = {}
    if json_path.is_file():
        report = json.loads(json_path.read_text(encoding="utf-8"))
    statuses = {r["run_dir"]: r.get("status") for r in report.get("results", [])}
    return {"command": command, "exit_code": proc.returncode, "statuses": statuses,
            "stdout_tail": proc.stdout.splitlines()[-10:], "wall_s": round(time.perf_counter() - started, 1),
            "passed": proc.returncode == 0 and bool(statuses)
            and all(s == "byte-identical" for s in statuses.values())}


# ---------------------------------------------------------------------------
# report


def print_report(report: dict) -> None:
    a = report["part_a"]
    print(f"Base {a['base']}. Changed files used for reachability: {len(a['changes']['relevant'])}")
    for path in a["changes"]["relevant"]:
        print(f"  changed: {path}")
    print(f"  listed but not counted ({len(a['changes']['ignored'])}): "
          + ", ".join(a["changes"]["ignored"]))
    print(f"Part A: registry unchanged: {a['registry_unchanged']}")
    print(f"  {len(a['entries_reaching_changed_files'])} of {a['entries_total']} registered "
          "entries can reach a changed file:")
    for run_dir in a["entries_reaching_changed_files"]:
        info = a["reach"][run_dir]
        print(f"    {run_dir}: " + ", ".join(info["reaches_changed"]))
        for data in info["per_script"].values():
            for path, chain in data["chains"].items():
                note = " -> ".join(chain) if len(chain) > 1 else f"{path} (the runner itself)"
                print(f"      {note}")
    for run_dir, scan in a["source_hash_scan"].items():
        print(f"  source-hash scan {run_dir}:")
        for line in scan["conclusion"]:
            print(f"    {line}")
        for name, value in scan["registered_outputs_scan"].items():
            print(f"    registered output {name}: {value}")
    for problem in a["problems"]:
        print(f"  PROBLEM: {problem}")
    print(f"Part A: {'PASS' if a['passed'] else 'FAIL'}")
    b = report.get("part_b")
    if b is None:
        print("Part B: not run")
    else:
        for runner, info in b["runners"].items():
            print(f"Part B {runner}: {info['sampled_jobs']} of {info['total_jobs']} jobs, "
                  f"cpu base {info['cpu_s']['base']} s, new {info['cpu_s']['new']} s, "
                  f"mismatched {len(info['mismatched_jobs'])}, "
                  f"aggregates identical {info['aggregate_identical']}, "
                  f"control repeat {info['control_repeat_identical']}, "
                  f"flag-on differs {info.get('negative_control_flag_on_differs', 'n/a')}")
            print(f"  axes covered: {info['coverage']['all_values_covered']} "
                  + str({k: len(v) for k, v in info["coverage"]["covered"].items()}))
            for mismatch in info["mismatched_jobs"]:
                print(f"  MISMATCH job {mismatch['job']}")
            for failure in info["failures"]:
                print(f"  FAIL: {failure}")
        print(f"Part B: {'PASS' if b['passed'] else 'FAIL'}")
    c = report.get("part_c")
    if c is not None:
        print(f"Part C: {c['statuses']} -> {'PASS' if c['passed'] else 'FAIL'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base", default=BASE_REV)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--seed", type=int, default=SEED, help="sampling seed")
    parser.add_argument("--skip-b", action="store_true", help="run Part A only")
    parser.add_argument("--full", metavar="NAME",
                        help="also rerun one registered entry in full (Part C), for example d2-fusion")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    report: dict = {"base": args.base, "repo": str(repo)}
    report["part_a"] = part_a(repo, args.base)
    ok = report["part_a"]["passed"]
    if not args.skip_b:
        registry = json.loads((repo / REGISTRY).read_text(encoding="utf-8"))
        reaching = report["part_a"]["entries_reaching_changed_files"]
        scripts = []
        for run_dir in reaching:
            for script in report["part_a"]["reach"][run_dir].get("runner_scripts_reaching", []):
                if script in RUNNERS and script not in scripts:
                    scripts.append(script)
        with tempfile.TemporaryDirectory(prefix="divergence-identity-", dir="/tmp") as tmp:
            report["part_b"] = part_b(repo, args.base, args.python, scripts, registry,
                                      args.seed, Path(tmp))
        ok = ok and report["part_b"]["passed"]
    if args.full:
        json_path = (args.json.with_suffix(".full.json") if args.json
                     else Path(tempfile.gettempdir()) / "divergence-identity-full.json")
        report["part_c"] = part_c(repo, args.full, args.python, json_path)
        ok = ok and report["part_c"]["passed"]
    print_report(report)
    report["passed"] = ok
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True, default=repr) + "\n",
                             encoding="utf-8")
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
