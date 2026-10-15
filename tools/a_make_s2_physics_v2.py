"""Generate the S2 convergence table, plot, and reproducibility manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.physics.run_manifest import write_manifest
from tools import horizon_mpl
from tools.make_s2_convergence import build_convergence


def _figure(document: dict, path: Path) -> None:
    horizon_mpl.apply_style("light")
    pal = horizon_mpl.palette("light")
    fig, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    rows = document["results"]
    dt = np.asarray([row["dt_s"] for row in rows], dtype=float)
    error = np.asarray([row["absolute_error"] for row in rows], dtype=float)
    order = np.argsort(dt)
    axis.loglog(dt[order], error[order], marker="o", color=pal["accent"], linewidth=1.4, label="measured error")
    anchor_dt, anchor_error = dt[order][0], error[order][0]
    reference = anchor_error * (dt / anchor_dt) ** 4
    axis.loglog(dt[order], reference[order], linestyle="--", color=pal["lineStrong"], linewidth=1.0, label="slope 4")
    axis.set_xlabel("step size, dt (s)")
    axis.set_ylabel("final-state error")
    axis.set_title("S2 integrator convergence")
    axis.legend(loc="best")
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def generate(output_dir: Path) -> list[Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    document = build_convergence()
    convergence_path = output_dir / "convergence.json"
    convergence_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    figure_path = output_dir / "a_s2_convergence.png"
    _figure(document, figure_path)
    runtime = time.perf_counter() - started
    written = [convergence_path, figure_path]
    write_manifest(
        output_dir / "manifest.json",
        schema="zephyr-s2-physics-v2-manifest-1",
        argv=sys.argv[1:],
        config={"generator": "tools/a_make_s2_physics_v2.py", "convergence": document},
        seeds=[],
        limits=["The reference is a finer numerical integration, not an analytic solution."],
        outputs=written,
        output_dir=output_dir,
        replay_command="PYTHONPATH=. " + " ".join(
            shlex.quote(str(argument)) for argument in [sys.executable, *sys.argv]
        ),
        runtime_s=runtime,
    )
    return written + [output_dir / "manifest.json"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
