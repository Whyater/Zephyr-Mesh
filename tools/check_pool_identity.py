"""Check that runner output is identical across serial and four worker runs.

This proves that the selected runners give byte-identical output when run
serially and with four workers through sim/ordered_pool.py. It also reads each
variant's manifest.json and fails if a variant asked for more than one worker
but really ran on one process (for example on a two-core machine the cap can
turn 4 into 2, which still counts, but never into 1).

Full runs are slow. The registered manifests (runs/c5-heartbeat and
runs/c-exp018-recovery-r2) record runtime_s 2324 s and 154 s, both with
n_processes 2, so together the variants here are well over 10 CPU minutes: run
full mode under the heavy lock. This script prints and saves the wall and CPU
seconds of every variant it runs.

Examples:
    PYTHONPATH=. python tools/check_pool_identity.py --pilot
    PYTHONPATH=. python tools/check_pool_identity.py --legacy --json report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - only used on Windows
    resource = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tools" / "reproduce_registry.json"
RUNNERS: dict[str, dict[str, str]] = {
    "exp016": {
        "script": "tools/c_exp016_heartbeat.py",
        "run_dir": "runs/c5-heartbeat",
    },
    "exp018": {
        "script": "tools/c_exp018_recovery.py",
        "run_dir": "runs/c-exp018-recovery-r2",
    },
}


class CheckerError(RuntimeError):
    """An error that should produce a concise command line failure."""


def registry_outputs(registry: Any, runner: str) -> dict[str, str]:
    """The registered output files of a runner and their sha256 hashes.

    registry is the parsed tools/reproduce_registry.json (a list of entries) or a path to it.
    """
    if runner not in RUNNERS:
        raise CheckerError(f"unknown runner: {runner}")
    if isinstance(registry, (str, os.PathLike)):
        registry = json.loads(Path(registry).read_text(encoding="utf-8"))
    if not isinstance(registry, list):
        raise CheckerError("registry must be a list of entries")
    run_dir = RUNNERS[runner]["run_dir"]
    for entry in registry:
        if isinstance(entry, dict) and entry.get("run_dir") == run_dir:
            outputs = entry.get("outputs")
            if not isinstance(outputs, dict) or not outputs:
                raise CheckerError(f"registry entry for {runner} has no outputs")
            return dict(outputs)
    raise CheckerError(f"runner {runner} is missing from the registry")


def sha256_file(path: str | os.PathLike[str] | Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_verdict(hashes: dict[str, dict[str, str | None]]) -> tuple[bool, dict[str, str]]:
    """Check that every variant has the same hash for every file."""
    variants = list(hashes)
    files = sorted({name for variant in hashes.values() for name in variant})
    detail: dict[str, str] = {}
    identical = True
    for name in files:
        values = [hashes[variant].get(name) for variant in variants]
        if any(value is None for value in values):
            detail[name] = "missing file"
            identical = False
        elif len(set(values)) != 1:
            detail[name] = "hash differs"
            identical = False
    return identical, detail


def registry_match(
    actual: dict[str, str | None], expected: dict[str, str]
) -> tuple[bool, dict[str, str]]:
    """Compare actual hashes with registry hashes and describe differences."""
    detail: dict[str, str] = {}
    for name, expected_hash in expected.items():
        value = actual.get(name)
        if value is None:
            detail[name] = "missing file"
        elif value != expected_hash:
            detail[name] = "hash differs"
    for name in actual:
        if name not in expected:
            detail[name] = "unexpected file"
    return not detail, detail


def _child_cpu_seconds() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run_variant(
    runner: str,
    variant: str,
    output_dir: Path,
    pilot: bool,
    workers: int | None,
) -> tuple[float, float | None]:
    """Run one runner variant and return wall and child CPU seconds."""
    command = [
        sys.executable,
        str(ROOT / RUNNERS[runner]["script"]),
        "--output",
        str(output_dir),
    ]
    if pilot:
        command.append("--pilot")
    if workers is not None:
        command.extend(["--workers", str(workers)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    before_cpu = _child_cpu_seconds()
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    wall = time.perf_counter() - started
    after_cpu = _child_cpu_seconds()
    cpu = None if before_cpu is None or after_cpu is None else after_cpu - before_cpu
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-40:])
        raise CheckerError(
            f"{runner} variant {variant} failed with exit code {result.returncode}\n{tail}"
        )
    return wall, cpu


def _manifest_processes(output_dir: Path) -> int | None:
    """The process count the runner wrote into its manifest (None if absent)."""
    try:
        value = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")).get("n_processes")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, int) else None


def _git_value(args: list[str]) -> str | bool | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if args[:1] == ["status"]:
        return bool(result.stdout.strip())
    return result.stdout.strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=["exp016", "exp018", "all"], default="all")
    parser.add_argument("--pilot", action="store_true", help="run the small pilot grids")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--legacy", action="store_true", help="also run the registered no-workers command")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--keep", type=Path, help="keep outputs in this directory")
    return parser.parse_args(argv)


def _print_runner(
    runner: str,
    outputs: list[str],
    hashes: dict[str, dict[str, str | None]],
    wall: dict[str, float],
    cpu: dict[str, float | None],
    used: dict[str, int | None],
) -> None:
    variants = list(hashes)
    print(runner)
    header = "file".ljust(34) + "  " + "  ".join(name.ljust(14) for name in variants) + "  same?"
    print(header)
    for name in outputs:
        values = [hashes[variant].get(name) for variant in variants]
        labels = [(value[:12] if value else "MISSING").ljust(14) for value in values]
        same = "yes" if values and all(value is not None for value in values) and len(set(values)) == 1 else "no"
        print(name.ljust(34) + "  " + "  ".join(labels) + f"  {same}")
    for variant in variants:
        cpu_text = "null" if cpu[variant] is None else f"{cpu[variant]:.3f}"
        print(f"{variant}: wall {wall[variant]:.3f}s, CPU {cpu_text}s, processes {used[variant]}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runners = list(RUNNERS) if args.runner == "all" else [args.runner]
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        runner_expected = {name: registry_outputs(registry, name) for name in runners}
        runner_outputs = {name: list(expected) for name, expected in runner_expected.items()}
        variants: list[tuple[str, int | None]] = []
        for workers in args.workers:
            if workers < 1:
                raise CheckerError("--workers values must be at least 1")
            variants.append((f"workers{workers}", workers))
        if args.legacy:
            variants.append(("legacy", None))
        if not variants:
            raise CheckerError("at least one worker variant is required")

        report: dict[str, Any] = {
            "runners": runners,
            "variants": {name: [variant for variant, _ in variants] for name in runners},
            "files": {},
            "identical": {},
            "registry_match": {},
            "wall_s": {},
            "cpu_s": {},
            "pilot": args.pilot,
            "python_version": platform.python_version(),
            "numpy_version": None,
            "platform": platform.platform(),
            "os_cpu_count": os.cpu_count(),
            "git_revision": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": _git_value(["status", "--porcelain"]),
        }
        try:
            import numpy

            report["numpy_version"] = numpy.__version__
        except ImportError:
            pass

        temporary: tempfile.TemporaryDirectory[str] | None = None
        if args.keep is None:
            temporary = tempfile.TemporaryDirectory(prefix="pool-identity-")
            base = Path(temporary.name)
        else:
            base = args.keep
            base.mkdir(parents=True, exist_ok=True)
        try:
            failed: list[str] = []
            for runner in runners:
                hashes: dict[str, dict[str, str | None]] = {}
                wall: dict[str, float] = {}
                cpu: dict[str, float | None] = {}
                used: dict[str, int | None] = {}
                for variant, workers in variants:
                    output_dir = base / runner / variant
                    output_dir.mkdir(parents=True, exist_ok=True)
                    wall[variant], cpu[variant] = run_variant(
                        runner, variant, output_dir, args.pilot, workers
                    )
                    used[variant] = _manifest_processes(output_dir)
                    if workers is not None and workers > 1 and (used[variant] or 0) < 2:
                        failed.append(f"{runner} {variant}: asked for {workers} workers, "
                                      f"manifest says {used[variant]} process")
                    hashes[variant] = {
                        name: sha256_file(output_dir / name) if (output_dir / name).is_file() else None
                        for name in runner_outputs[runner]
                    }
                identical, identity_detail = identity_verdict(hashes)
                report["files"][runner] = {
                    name: {variant: hashes[variant].get(name) for variant, _ in variants}
                    for name in runner_outputs[runner]
                }
                report["wall_s"][runner] = wall
                report["cpu_s"][runner] = cpu
                report.setdefault("n_processes", {})[runner] = used
                report["identical"][runner] = identical
                if args.pilot:
                    report["registry_match"][runner] = None
                else:
                    first = next(iter(hashes.values()))
                    matched, registry_detail = registry_match(first, runner_expected[runner])
                    report["registry_match"][runner] = matched
                    report.setdefault("registry_detail", {})[runner] = registry_detail
                _print_runner(runner, runner_outputs[runner], hashes, wall, cpu, used)
                if identity_detail:
                    failed.extend(f"{runner} {name}: {reason}" for name, reason in identity_detail.items())
                if not args.pilot and report["registry_match"][runner] is False:
                    failed.extend(
                        f"{runner} {name}: {reason}"
                        for name, reason in report["registry_detail"][runner].items()
                    )
            if failed:
                print(f"RESULT: FAILED ({'; '.join(failed)})")
            else:
                print("RESULT: all runners identical" + ("" if args.pilot else " and match the registry"))
            if args.json_path:
                args.json_path.parent.mkdir(parents=True, exist_ok=True)
                args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 1 if failed else 0
        finally:
            if temporary is not None:
                temporary.cleanup()
    except CheckerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
