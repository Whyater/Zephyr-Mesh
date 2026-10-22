"""Wait for a desktop process to exit, then apply a staged update.

Same steps as the packaged Windows helper (``tools/desktop/windows_updater.py``):
``tools.release_updater.run_update_helper`` re-checks the staged folder, runs
the new build's ``--self-test``, swaps, starts the new app, and puts the old
one back on any failure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.release_updater import run_update_helper, wait_for_exit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a staged Zephyr Mesh app update after process exit")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True, help="executable inside the app folder to self-test and start")
    args = parser.parse_args(argv)
    return run_update_helper(args.pid, args.staged, args.install_dir, args.launch,
                             wait=lambda pid: wait_for_exit(pid, timeout_s=30.0))


if __name__ == "__main__":
    raise SystemExit(main())
