"""Cross-platform desktop tokens for the Windows preview.

The preview intentionally uses Windows system typography and a quiet neutral
surface. State color is reserved for warnings and selection, not decoration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DesktopTokens:
    background: str = "#f3f3f3"
    surface: str = "#ffffff"
    canvas: str = "#e8edf2"
    text: str = "#1f1f1f"
    muted: str = "#616161"
    grid: str = "#d5dce3"
    accent: str = "#276ea5"
    warning: str = "#c77b30"
    control_background: str = "#ffffff"
    control_track: str = "#d5dce3"
    control_active: str = "#276ea5"
    window_min_width: int = 1180
    window_min_height: int = 760
    fleet_row_height: int = 28
    inspector_width: int = 270
    inspector_text_width: int = 246
    button_pad_x: int = 10
    button_pad_y: int = 6
    panel_gap: int = 10
    timeline_gap: int = 14
    tight_padding: int = 9
    button_gap: int = 3
    update_gap: int = 10
    disclosure_pad: int = 3
    header_vertical: int = 13
    table_top_gap: int = 8
    telemetry_top_gap: int = 8
    telemetry_bottom_gap: int = 16
    content_pad: int = 4
    canvas_margin: int = 20
    canvas_extent: int = 80
    target_radius: int = 8
    target_cross: int = 13
    selection_outline: int = 5
    label_offset: int = 12
    canvas_label_x: int = 18
    canvas_label_y: int = 16
    selection_outline_color: str = "#eef2f7"
    spacing_xs: int = 8
    spacing_sm: int = 12
    spacing_md: int = 16
    spacing_lg: int = 18
    ui_font: str = "Segoe UI"
    mono_font: str = "Consolas"
    title_size: int = 15
    body_size: int = 10
    small_size: int = 9
    fleet_column_vehicle: int = 112
    fleet_column_state: int = 74
    fleet_column_metric: int = 62


TOKENS = DesktopTokens()
