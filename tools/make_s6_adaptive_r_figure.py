"""Figure for EXP-037 (proposal): told 5 cm vs told the truth vs adaptive noise estimate.

Reads sim/runs/s6-adaptive-r/summary.json and writes s6_adaptive_r.png next to it:

    PYTHONPATH=. ./.venv/bin/python tools/make_s6_adaptive_r_figure.py --run sim/runs/s6-adaptive-r
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from tools import horizon_mpl

ARM_LABELS = {"told_5cm": "told 5 cm (current setting)", "told_truth": "told the true noise (oracle)",
              "adaptive": "adaptive estimate"}


def make_figure(run: Path) -> Path:
    summary = json.loads((run / "summary.json").read_text())
    colors = horizon_mpl.apply_style("light")
    arm_colors = {"told_5cm": colors["lineStrong"], "told_truth": colors["accent"], "adaptive": colors["keepOut"]}
    cells = {c["condition_id"]: c for c in summary["cells"]}
    noises = sorted({c["noise_std_m"] for c in summary["cells"]})
    delays = sorted({c["delay_s"] for c in summary["cells"]})
    groups = [(n, d) for n in noises for d in delays]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, loss in enumerate(("none", "burst30")):
        for col, motion in enumerate(("constant_velocity", "constant_turn")):
            ax = axes[row][col]
            for k, arm in enumerate(ARM_LABELS):
                xs, ys, lows, highs = [], [], [], []
                for g, (noise, delay) in enumerate(groups):
                    a = cells[f"{motion}/n{noise:.2f}/d{delay:.1f}/{loss}"]["arms"][arm]
                    xs.append(g + (k - 1) * 0.22)
                    ys.append(a["p95_median_m"])
                    lows.append(a["p95_median_m"] - a["p95_ci_low_m"])
                    highs.append(a["p95_ci_high_m"] - a["p95_median_m"])
                ax.errorbar(xs, ys, yerr=[lows, highs], fmt="o", ms=4, capsize=2, color=arm_colors[arm],
                            label=ARM_LABELS[arm])
            ax.axhline(summary["config"]["failure_p95_m"], color=colors["alert"], lw=1, ls="--")
            if loss == "burst30":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            for boundary in (2.5, 5.5):
                ax.axvline(boundary, color=colors["line"], lw=0.8)
            ax.set_title(f"{'straight line' if motion == 'constant_velocity' else '1 m turn'}, "
                         f"{'no loss' if loss == 'none' else '30 % loss in bursts'}")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([f"{n * 100:.0f} cm\n{d * 1000:.0f} ms" for n, d in groups], fontsize=8)
            if col == 0:
                ax.set_ylabel("median p95 error (m)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("EXP-037 proposal (simulated): can the recovery tracker learn its own camera noise? "
                 "Dashed line: 0.25 m failure threshold", fontsize=10)
    fig.tight_layout()
    out = run / "s6_adaptive_r.png"
    fig.savefig(out, dpi=130, metadata={"Software": None})
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    print(make_figure(args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
