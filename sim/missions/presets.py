"""Named synthetic packet-loss presets for mission simulations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinkPreset:
    name: str
    loss_model: str
    loss_probability: float
    burst_mean_s: float | None
    description: str


_ROWS = (
    ("clean", "none", 0.0, None),
    ("iid_1", "independent", 0.01, None),
    ("iid_5", "independent", 0.05, None),
    ("iid_10", "independent", 0.10, None),
    ("iid_30", "independent", 0.30, None),
    ("burst_1_0.5s", "burst", 0.01, 0.5),
    ("burst_10_0.2s", "burst", 0.10, 0.2),
    ("burst_10_0.5s", "burst", 0.10, 0.5),
    ("burst_30_0.5s", "burst", 0.30, 0.5),
)
PRESET_NAMES = tuple(row[0] for row in _ROWS)
_PRESETS = {
    name: LinkPreset(name, model, p, mean, "synthetic loss pattern (simulated)")
    for name, model, p, mean in _ROWS
}


def get_preset(name: str) -> LinkPreset:
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown link preset: {name!r}") from exc

