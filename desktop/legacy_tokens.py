"""TEMPORARY compatibility tokens. Deleted by the Windows rebuild step."""
from __future__ import annotations
from types import SimpleNamespace
try:
    from .design_tokens import DARK, LAYOUT, SPACE
except ImportError:
    from design_tokens import DARK, LAYOUT, SPACE

TOKENS = SimpleNamespace(
    background=DARK.background, surface=DARK.surface, canvas=DARK.background,
    text=DARK.text, muted=DARK.textSecondary, grid=DARK.line,
    accent=DARK.accent, warning=DARK.caution, control_background=DARK.surface,
    control_track=DARK.line, control_active=DARK.accent,
    window_min_width=LAYOUT["windowMinWidth"], window_min_height=LAYOUT["windowMinHeight"],
    fleet_row_height=LAYOUT["rowHeight"], inspector_width=LAYOUT["inspectorWidth"],
    inspector_text_width=LAYOUT["inspectorWidth"] - 54,
    button_pad_x=10, button_pad_y=6, panel_gap=10, timeline_gap=14,
    tight_padding=SPACE["xs"], button_gap=3, update_gap=10, disclosure_pad=3,
    header_vertical=13, table_top_gap=SPACE["sm"], telemetry_top_gap=SPACE["sm"],
    telemetry_bottom_gap=SPACE["lg"], content_pad=SPACE["xxs"], canvas_margin=20,
    canvas_extent=80, target_radius=8, target_cross=13, selection_outline=5,
    label_offset=12, canvas_label_x=18, canvas_label_y=16,
    selection_outline_color=DARK.text, spacing_xs=SPACE["xs"], spacing_sm=SPACE["sm"],
    spacing_md=SPACE["md"], spacing_lg=SPACE["lg"], ui_font="Segoe UI Variable Text",
    mono_font="Cascadia Mono", title_size=20, body_size=13, small_size=12,
    fleet_column_vehicle=112, fleet_column_state=74, fleet_column_metric=62,
)
