"""Wait for a desktop process to exit, then apply a staged update."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

from tools.release_updater import apply_staged_update


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    return subprocess.run(["kill", "-0", str(pid)], check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a staged Zephyr Mesh app update after process exit")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--launch", type=Path)
    args = parser.parse_args()
    deadline = time.monotonic() + 30.0
    while _process_exists(args.pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _process_exists(args.pid):
        raise SystemExit("timed out waiting for the desktop process")
    apply_staged_update(args.staged, args.install_dir)
    if args.launch:
        subprocess.Popen([str(args.launch)], close_fds=True)


if __name__ == "__main__":
    main()
