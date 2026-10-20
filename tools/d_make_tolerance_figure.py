"""Make the simulated tolerance-versus-observer-count figure."""
from __future__ import annotations

import json
import textwrap
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
ZERO_BAND = (-30.0, -10.0)
CENSORED_BAND = (300.0, 335.0)
FOOTER_INCHES = 1.45
MIN_WIDTH_INCHES = 9.0
ZERO_Y = -20.0
CENSORED_Y = 320.0


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


def _link_label(link: str, rates_hz: dict | None, link_notes: dict | None) -> str:
    """Legend text for a link: name, capture rate, and an optional note such as background load."""
    details = []
    if rates_hz and link in rates_hz:
        details.append(f"{float(rates_hz[link]):g} Hz")
    if link_notes and link_notes.get(link):
        details.append(str(link_notes[link]))
    return f"{link} ({', '.join(details)})" if details else link


def _band_offsets(links: tuple[str, ...]) -> dict[tuple[str, str], float]:
    """Sideways shift in N units so every arm and link marker in a band stays visible."""
    count = len(DISPLAY_ARMS) * len(links)
    step = 0.09
    offsets = {}
    for arm_index, arm in enumerate(DISPLAY_ARMS):
        for link_index, link in enumerate(links):
            slot = arm_index * len(links) + link_index
            offsets[(arm, link)] = (slot - (count - 1) / 2.0) * step
    return offsets


def _band_marker(axis, x: float, y: float, stem_from: float, *, marker: str, color: str,
                 linestyle: str) -> None:
    """Hollow marker in a band, with a short stem in the link line style."""
    axis.plot([x, x], [stem_from, y], color=color, linestyle=linestyle, linewidth=1.0, zorder=4)
    axis.plot(x, y, marker=marker, markersize=6, markerfacecolor="white", markeredgecolor=color,
              linestyle="None", zorder=5)


def make_figure(boundaries_path: str | Path, output_path: str | Path, *, pilot: bool = False,
                rates_hz: dict | None = None, link_notes: dict | None = None) -> None:
    """Render simulated boundaries, including explicit link and arm legends.

    rates_hz gives each link's capture rate and link_notes any extra legend text per link.
    """
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
    offsets = _band_offsets(links)
    width = max(4.3 * len(losses), MIN_WIDTH_INCHES)
    height = 3.5 * len(targets) + FOOTER_INCHES
    fig, axes = plt.subplots(len(targets), len(losses), figsize=(width, height),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, target in enumerate(targets):
        for column_index, (model, probability) in enumerate(losses):
            axis = axes[row_index][column_index]
            for arm in DISPLAY_ARMS:
                style = styles[arm]
                for link in links:
                    line_values = []
                    states = []
                    for n in ns:
                        source_arm = "single" if n == 1 else arm
                        key = _key(link, target, model, probability, source_arm, n)
                        if key not in data:
                            raise ValueError(f"requested figure cell is missing: {key}")
                        boundary = data[key]
                        states.append(boundary)
                        if boundary["fails_at_zero"]:
                            line_values.append(float("nan"))
                        elif boundary["censored"]:
                            line_values.append(float("nan"))
                        elif boundary["value"] is None:
                            raise ValueError(f"boundary has no state: {key}")
                        else:
                            line_values.append(float(boundary["value"]) * 1000.0)
                    axis.plot(ns, line_values, color=style["color"], linestyle=LINK_STYLES[link],
                              marker=style["marker"], markerfacecolor="white", markeredgecolor=style["color"],
                              linewidth=1.2, markersize=4.5)
                    for n, boundary in zip(ns, states):
                        x = n + offsets[(arm, link)]
                        if boundary["fails_at_zero"]:
                            _band_marker(axis, x, ZERO_Y, ZERO_BAND[1], marker="o",
                                         color=style["color"], linestyle=LINK_STYLES[link])
                        elif boundary["censored"]:
                            _band_marker(axis, x, CENSORED_Y, CENSORED_BAND[0], marker="^",
                                         color=style["color"], linestyle=LINK_STYLES[link])
            axis.axhspan(*ZERO_BAND, color="0.92", zorder=0)
            axis.axhspan(*CENSORED_BAND, color="0.92", zorder=0)
            panel = "turn r = 1 m" if target == "ct_r1" else "jinking"
            axis.set_title(panel + f", {model} {probability:g}")
            axis.set_xticks(ns)
            axis.set_xlim(min(ns) - 0.55, max(ns) + 0.55)
            axis.set_ylim(-35.0, 335.0)
            axis.set_yticks([ZERO_Y, 0.0, 100.0, 200.0, CENSORED_Y])
            axis.set_yticklabels(["fails at zero", "0", "100", "200", "censored"])
            if row_index == len(targets) - 1:
                axis.set_xlabel("number of observers, N")
            if column_index == 0:
                axis.set_ylabel("maximum tolerated delay (ms)")
    arm_handles = [Line2D([0], [0], color=styles[arm]["color"], marker=styles[arm]["marker"],
                          markerfacecolor="white", linestyle="-", label=arm) for arm in DISPLAY_ARMS]
    link_handles = [Line2D([0], [0], color="0.25", linestyle=LINK_STYLES[link],
                           label=_link_label(link, rates_hz, link_notes)) for link in links]
    fig.legend(handles=arm_handles, loc="lower left", bbox_to_anchor=(0.06, 0.055 + 0.2 / height),
               ncol=1, title="arm", frameon=False, alignment="left")
    fig.legend(handles=link_handles, loc="lower left", bbox_to_anchor=(0.30, 0.055 + 0.2 / height),
               ncol=1, title="link (capture rate)", frameon=False, alignment="left")
    fig.suptitle("Simulated tolerance to delay and loss, noise 0.05 m, capture rate per link in legend",
                 y=0.995)
    caption = ("Lower band, circles: fails at zero. Upper band, triangles: censored above the tested "
               "range. Band markers are spread sideways by arm and link and keep the link line style.")
    fig.text(0.5, 0.008, textwrap.fill(caption, width=int(width * 12)), ha="center", va="bottom",
             fontsize=8)
    fig.tight_layout(rect=(0, FOOTER_INCHES / height, 1, 0.95))
    fig.savefig(output_path, dpi=150, metadata={"Software": None})
    plt.close(fig)
