"""Detached helper for the packaged Windows preview.

The preview starts this executable and exits. The helper waits for the old
process to release its files, checks the staged folder again, runs the new
build's ``--self-test``, swaps the folders (the old one goes to a backup in
``%LOCALAPPDATA%``), starts the replacement and watches it briefly, then
deletes the backup. It contains no network client and therefore cannot
install an artifact that was not staged and digest-checked first. The steps
live in ``tools.release_updater.run_update_helper`` so they are tested once.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.release_updater import pid_listed_in_tasklist, run_update_helper, wait_for_exit

# Kept for callers and tests that use the old names.
_pid_is_listed = pid_listed_in_tasklist


def wait_for_process(pid: int, timeout_s: float = 45.0) -> None:
    """Wait for a Windows PID to exit, raising TimeoutError on timeout."""
    if pid <= 0:
        raise ValueError("pid must be positive")
    wait_for_exit(pid, timeout_s=timeout_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a verified Zephyr Mesh Windows update")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staged", type=Path, required=True, help="staged application directory")
    parser.add_argument("--install-dir", type=Path, required=True, help="current application directory")
    parser.add_argument("--launch", type=Path, required=True, help="replacement executable")
    args = parser.parse_args(argv)
    return run_update_helper(args.pid, args.staged, args.install_dir, args.launch, wait=lambda pid: wait_for_process(pid))


if __name__ == "__main__":
    raise SystemExit(main())
