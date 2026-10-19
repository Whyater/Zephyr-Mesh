"""Make the simulated tolerance-versus-observer-count figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tools import horizon_mpl

DISPLAY_ARMS = ("central", "star_avg", "peer_ci")
LINK_STYLES = {"abstract": "-", "radio_gym": "--", "radio_stress": ":"}
FULL_LOSSES = (("independent", 0.1), ("independent", 0.5), ("burst", 0.3))
ARM_STYLE_NAMES = {"central": "recovery", "star_avg": "ct", "peer_ci": "imm"}


def _key(link: str, target: str, model: str, probability: float, arm: str, n: int) -> str:
    return f"{link}|{target}|{model}:{probability:g}|{arm}|{n}"


def _available(data: dict) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...], tuple[int, ...]]:
    parsed = []
    for key in data:
        parts = key.split("|")
        if len(parts) != 5:
            continue
        model, probability = parts[2].split(":", 1)
        parsed.append((parts[0], parts[1], model, float(probability), int(parts[4])))
    order = ("abstract", "radio_gym", "radio_stress")
    links = tuple(name for name in order if any(row[0] == name for row in parsed))
    losses = tuple((model, probability) for model, probability in FULL_LOSSES
                   if any(row[2] == model and row[3] == probability for row in parsed))
    ns = tuple(sorted({row[4] for row in parsed}))
    return links, losses, ns


def make_figure(boundaries_path: str | Path, output_path: str | Path, *, pilot: bool = False) -> None:
    """Render simulated boundaries, including explicit link and arm legends."""
    data = json.loads(Path(boundaries_path).read_text(encoding="utf-8"))
    links, losses, ns = _available(data)
    if not links or not losses or not ns:
        raise ValueError("boundary data has no complete figure cells")
    if not pilot and (set(links) != set(LINK_STYLES) or losses != FULL_LOSSES
                      or ns != (1, 2, 3, 4, 6)):
        raise ValueError("full figure requested but a registered cell is missing")
    if pilot and ("abstract" not in links or "radio_stress" not in links):
        raise ValueError("pilot figure is missing a registered link")
    target_names = ("ct_r1", "jink")
    targets = tuple(target for target in target_names
                    if any(key.split("|")[1] == target for key in data))
    if not targets:
        raise ValueError("figure data has no target panels")
    apply_palette = horizon_mpl.apply_style("light")
    styles = {arm: horizon_mpl.arm_style(ARM_STYLE_NAMES[arm], apply_palette)
              for arm in DISPLAY_ARMS}
    fig, axes = plt.subplots(len(targets), len(losses), figsize=(4.3 * len(losses), 3.5 * len(targets)),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, target in enumerate(targets):
        for column_index, (model, probability) in enumerate(losses):
            axis = axes[row_index][column_index]
            for arm in DISPLAY_ARMS:
                style = styles[arm]
                for link in links:
                    values = []
                    states = []
                    for n in ns:
                        source_arm = "single" if n == 1 else arm
                        key = _key(link, target, model, probability, source_arm, n)
                        if key not in data:
                            raise ValueError(f"requested figure cell is missing: {key}")
                        boundary = data[key]
                        states.append(boundary)
                        if boundary["fails_at_zero"]:
                            values.append(-8.0)
                        elif boundary["censored"]:
                            values.append(300.0)
                        elif boundary["value"] is None:
                            raise ValueError(f"boundary has no state: {key}")
                        else:
                            values.append(float(boundary["value"]) * 1000.0)
                    axis.plot(ns, values, color=style["color"], linestyle=LINK_STYLES[link],
                              marker=style["marker"], markerfacecolor="white", markeredgecolor=style["color"],
                              linewidth=1.2, markersize=4.5)
                    for n, value, boundary in zip(ns, values, states):
                        if boundary["fails_at_zero"] or boundary["censored"]:
                            marker = "o" if boundary["fails_at_zero"] else "^"
                            axis.plot(n, value, marker=marker, markersize=7, markerfacecolor="none",
                                      markeredgecolor=style["color"], linestyle="None", zorder=5)
            panel = "turn r = 1 m" if target == "ct_r1" else "jinking"
            axis.set_title(panel + f", {model} {probability:g}")
            axis.set_xticks(ns)
            axis.set_ylim(-15.0, 300.0)
            if row_index == len(targets) - 1:
                axis.set_xlabel("number of observers, N")
            if column_index == 0:
                axis.set_ylabel("maximum tolerated delay (ms)")
    arm_handles = [Line2D([0], [0], color=styles[arm]["color"], marker=styles[arm]["marker"],
                          markerfacecolor="white", linestyle="-", label=arm) for arm in DISPLAY_ARMS]
    link_handles = [Line2D([0], [0], color="0.25", linestyle=LINK_STYLES[link], label=link)
                    for link in links]
    fig.legend(handles=arm_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(arm_handles), title="arm")
    fig.legend(handles=link_handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=len(link_handles), title="link")
    fig.suptitle("Simulated tolerance to delay and loss, noise 0.05 m, 10/50 Hz", y=0.995)
    fig.text(0.99, 0.01, "Hollow circles: failure at zero delay. Hollow triangles: above tested range.",
             ha="right", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    fig.savefig(output_path, dpi=150, metadata={"Software": None})
    plt.close(fig)
