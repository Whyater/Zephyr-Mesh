#!/usr/bin/env python3
"""Reproduce registered run outputs from a clean source tree."""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen[str], int | None]] = {}
_ACTIVE_LOCK = threading.Lock()
_STOPPING = threading.Event()


class RegistryError(ValueError):
    pass


def load_registry(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RegistryError("registry must be a JSON array")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"run_dir", "command", "outputs"}
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RegistryError(f"entry {index} must be an object")
        if set(entry) != required:
            raise RegistryError(
                f"entry {index} must have exactly keys run_dir, command, outputs"
            )
        run_dir = entry["run_dir"]
        command = entry["command"]
        outputs = entry["outputs"]
        if (not isinstance(run_dir, str) or not run_dir or Path(run_dir).is_absolute()
                or not run_dir.startswith("sim/runs/") or run_dir.endswith("/")):
            raise RegistryError(f"entry {index} run_dir must be a sim/runs/<name> relative path")
        if any(part == ".." for part in Path(run_dir).parts):
            raise RegistryError(f"entry {index} run_dir may not contain '..'")
        if run_dir in seen:
            raise RegistryError(f"duplicate run_dir: {run_dir}")
        seen.add(run_dir)
        if not isinstance(command, str) or "{out}" not in command:
            raise RegistryError(f"entry {index} command must contain {{out}}")
        if not isinstance(outputs, dict):
            raise RegistryError(f"entry {index} outputs must be an object")
        for name, digest in outputs.items():
            if not isinstance(name, str) or not name or Path(name).is_absolute() or any(
                part == ".." for part in Path(name).parts
            ):
                raise RegistryError(f"entry {index} has invalid output path")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise RegistryError(f"entry {index} output {name!r} must be a sha256 hex digest")
        entries.append(entry)
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_archive(archive: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=__import__("io").BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath((str(destination.resolve()), str(target))) != str(destination.resolve()):
                raise RuntimeError(f"unsafe path in git archive: {member.name}")
        tar.extractall(destination)


def make_clean_copy(repo_root: Path, destination: Path, revision: str, source_dir: Path | None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        source_dir = source_dir.resolve()
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
        return
    proc = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        message = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git archive failed ({proc.returncode}): {message}")
    safe_extract_archive(proc.stdout, destination)


def clear_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def shell_command(command: str) -> list[str]:
    """Run a registry command (POSIX shell syntax) with a POSIX shell on every OS.

    cmd.exe cannot read ``VAR=value cmd`` or single quotes, so Windows uses the
    bash that ships with Git for Windows.
    """
    if os.name != "nt":
        return ["/bin/sh", "-c", command]
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("reproduce_all.py needs bash on Windows (install Git for Windows)")
    return [bash, "-c", command]


def _terminate_process_group(proc: subprocess.Popen[str], process_group: int | None) -> None:
    """Stop a runner and every process it started in its private group."""
    if process_group is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _unregister_process(proc: subprocess.Popen[str]) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.pop(id(proc), None)


def _terminate_active_processes() -> None:
    with _ACTIVE_LOCK:
        active = list(_ACTIVE_PROCESSES.values())
    for proc, process_group in active:
        _terminate_process_group(proc, process_group)


def _install_signal_cleanup() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    _STOPPING.clear()

    def handle(signum: int, _frame: Any) -> None:
        _STOPPING.set()
        _terminate_active_processes()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


def _restore_signal_cleanup(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


atexit.register(_terminate_active_processes)


def run_entry(
    entry: dict[str, Any], repo_root: Path, revision: str, source_dir: Path | None,
    python_executable: str, timeout: float, keep_temp: bool,
) -> dict[str, Any]:
    run_dir = entry["run_dir"]
    command = entry["command"].replace("{python}", shlex.quote(python_executable))
    if _STOPPING.is_set():
        return {"run_dir": run_dir, "command": entry["command"], "status": "cancelled"}
    temp = Path(tempfile.mkdtemp(prefix="reproduce-"))
    result: dict[str, Any] = {"run_dir": run_dir, "command": entry["command"]}
    proc: subprocess.Popen[str] | None = None
    process_group: int | None = None
    try:
        clean = temp / "tree"
        make_clean_copy(repo_root, clean, revision, source_dir)
        out = clean / run_dir
        clear_directory(out)
        command = command.replace("{out}", shlex.quote(str(out.resolve())))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(clean)
        popen_kwargs: dict[str, Any] = {
            "cwd": clean,
            "env": env,
            "text": True,
            # Runner stdout is intentionally discarded. Keeping it out of a
            # pipe lets us notice the runner's exit even if a leftover child
            # inherited its file descriptors.
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        with _ACTIVE_LOCK:
            if _STOPPING.is_set():
                return {"run_dir": run_dir, "command": entry["command"], "status": "cancelled"}
            proc = subprocess.Popen(shell_command(command), **popen_kwargs)
            if os.name != "nt":
                process_group = os.getpgid(proc.pid)
            _ACTIVE_PROCESSES[id(proc)] = (proc, process_group)
        timed_out = False
        try:
            _stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc, process_group)
            _stdout, stderr = proc.communicate()
        else:
            _terminate_process_group(proc, process_group)
        if timed_out:
            result.update({"status": "failed", "error": f"timeout after {timeout:g} seconds", "stderr_tail": stderr.splitlines()[-20:]})
            return result
        if proc.returncode != 0:
            result.update({"status": "failed", "exit_code": proc.returncode, "stderr_tail": stderr.splitlines()[-20:]})
            return result
        differences: dict[str, dict[str, str | None]] = {}
        missing: list[str] = []
        for name, expected in entry["outputs"].items():
            path = out / name
            if not path.is_file():
                missing.append(name)
                continue
            actual = sha256_file(path)
            if actual.lower() != expected.lower():
                differences[name] = {"expected": expected, "got": actual}
        if missing:
            result.update({"status": "missing-output", "missing": missing, "differences": differences})
        elif differences:
            result.update({"status": "differs", "differences": differences})
        else:
            result["status"] = "byte-identical"
        return result
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
        return result
    finally:
        if proc is not None:
            _terminate_process_group(proc, process_group)
            _unregister_process(proc)
        if not keep_temp:
            shutil.rmtree(temp, ignore_errors=True)
        else:
            result["temp_dir"] = str(temp)


def parse_names(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def print_result(result: dict[str, Any]) -> None:
    status = result["status"]
    line = f"{result['run_dir']}: {status}"
    if status == "differs":
        details = [
            f"{name}: expected {item['expected']}, got {item['got']}"
            for name, item in result["differences"].items()
        ]
        line += " (" + "; ".join(details) + ")"
    elif status == "missing-output":
        line += " (" + ", ".join(result["missing"]) + ")"
    elif status == "failed":
        if "exit_code" in result:
            line += f" (exit code {result['exit_code']})"
        elif "error" in result:
            line += f" ({result['error']})"
        tail = result.get("stderr_tail", [])
        if tail:
            line += "\n  stderr: " + " | ".join(tail)
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce registered runs from a clean tree. Uncommitted changes are never included."
    )
    parser.add_argument("--registry", default="tools/reproduce_registry.json")
    parser.add_argument("--rev", default="HEAD", help="git revision for the clean archive")
    parser.add_argument("--source-dir", type=Path, help="copy this plain directory instead of using git archive")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 3))
    parser.add_argument("--only", help="comma-separated run directory basenames")
    parser.add_argument("--skip", help="comma-separated run directory basenames")
    parser.add_argument("--json", type=Path, help="write machine-readable report")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--hash", nargs="+", metavar="ARG", help="print sha256 for files under RUN_DIR")
    args = parser.parse_args(argv)
    repo_root = Path.cwd().resolve()
    if args.hash:
        if len(args.hash) < 2:
            parser.error("--hash requires RUN_DIR and at least one FILE")
        run_path = Path(args.hash[0])
        for name in args.hash[1:]:
            path = run_path / name
            try:
                print(f"{name}: {sha256_file(path)}")
            except OSError as exc:
                print(f"{name}: error: {exc}", file=sys.stderr)
                return 1
        return 0
    try:
        entries = load_registry((repo_root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry))
    except RegistryError as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 2
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    only, skip = parse_names(args.only), parse_names(args.skip)
    selected = [e for e in entries if (only is None or Path(e["run_dir"]).name in only) and (skip is None or Path(e["run_dir"]).name not in skip)]
    source_dir = args.source_dir.resolve() if args.source_dir else None
    results: list[dict[str, Any]] = []
    previous_signals = _install_signal_cleanup()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(run_entry, e, repo_root, args.rev, source_dir, args.python, args.timeout, args.keep_temp) for e in selected]
        for future in futures:
            results.append(future.result())
    except BaseException:
        _STOPPING.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    finally:
        _restore_signal_cleanup(previous_signals)
    for result in results:
        print_result(result)
    report = {"results": results, "selected": len(selected), "workers": args.workers}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r.get("status") == "byte-identical" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
