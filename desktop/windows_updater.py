"""Detached helper for the packaged Windows preview.

The preview starts this executable and exits. The helper waits for the old
process to release its files, swaps the verified staged application directory,
and launches the replacement. It contains no network client and therefore
cannot install an artifact that was not staged and digest-checked first.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time

from tools.release_updater import apply_staged_update


def _pid_is_listed(output: str, pid: int) -> bool:
    """Parse tasklist CSV so PID 12 cannot match PID 112."""
    for row in csv.reader(output.splitlines()):
        if len(row) >= 2 and row[1].strip() == str(pid):
            return True
    return False


def wait_for_process(pid: int, timeout_s: float = 45.0) -> None:
    """Wait for a Windows PID to exit, raising on timeout."""
    if pid <= 0:
        raise ValueError("pid must be positive")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not _pid_is_listed(result.stdout, pid):
            return
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for process {pid}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a verified Zephyr Mesh Windows update")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staged", type=Path, required=True, help="staged application directory")
    parser.add_argument("--install-dir", type=Path, required=True, help="current application directory")
    parser.add_argument("--launch", type=Path, required=True, help="replacement executable")
    args = parser.parse_args(argv)
    wait_for_process(args.pid)
    apply_staged_update(args.staged, args.install_dir)
    subprocess.Popen([str(args.launch)], close_fds=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
