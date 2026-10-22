"""The Results window for the Windows replay: the S6 delay-and-loss result and the tolerance atlas.

Three columns: a two-row sidebar, the selected view in a scroll area, and
an inspector for the selected chart point or atlas cell. Every number comes
from ``results_fixture.json`` through ``results_data``; this module only
lays out widgets. The window has one honesty badge, collapsed Limits and
Source sections under each view, and nothing that runs on a timer.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .design_tokens import LAYOUT, SPACE, UI_TEXT, WIN_LAYOUT
from . import results_data as rd
from . import ui_format as fmt
from .ui_icons import photo_data, render_png
from .ui_results_atlas import AtlasGrid, rounded_points
from .ui_results_chart import DelayChart
from .ui_theme import RoundedElements, control_padding
from .ui_widgets import Disclosure, KeyValueGrid, ScrollFrame, Tooltip


TEXT = rd.TEXT
ATLAS_TEXT = rd.ATLAS_TEXT
VIEWS = ("delay", "atlas")
VIEW_TITLES = {"delay": TEXT["sidebarDelay"], "atlas": TEXT["sidebarAtlas"]}
HEADINGS = {"delay": TEXT["delayTitle"], "atlas": TEXT["atlasTitle"]}
# Sidebar glyphs in unit-square coordinates (the shared rasterizer): a small line chart, and a 3 x 3 grid.
SIDEBAR_GLYPHS = {
    "delay": (("line", 0.14, 0.12, 0.14, 0.86, 0.09), ("line", 0.14, 0.86, 0.88, 0.86, 0.09),
              ("line", 0.26, 0.66, 0.46, 0.44, 0.1), ("line", 0.46, 0.44, 0.6, 0.56, 0.1), ("line", 0.6, 0.56, 0.86, 0.22, 0.1)),
    "atlas": tuple(("poly", ((x, y), (x + 0.22, y), (x + 0.22, y + 0.22), (x, y + 0.22)))
                   for y in (0.12, 0.39, 0.66) for x in (0.12, 0.39, 0.66)),
}
SEGMENT_MIN_CHARS = 7  # the narrowest segment, in characters: "Random" and "10 %" fill the same width
LEGEND_DELAY = (("recovery", TEXT["armRecovery"]), ("naive", TEXT["armNaive"]), ("band", None), ("failure", TEXT["legendFailure"]))
LEGEND_ATLAS = (("stop1", ATLAS_TEXT["bin1"]), ("stop2", ATLAS_TEXT["bin2"]), ("stop3", ATLAS_TEXT["bin3"]),
                ("stop4", ATLAS_TEXT["censoredLong"]), ("fails", ATLAS_TEXT["failsAtZeroLong"]), ("winner", ATLAS_TEXT["clearWinner"]))


# ---- small shared pieces ------------------------------------------------------------------------
class Wrap:
    """Calls each registered function with a new content width when a scroll area is resized."""

    def __init__(self, scroll: ScrollFrame, padding: int) -> None:
        self.padding = padding
        self.width = 0
        self.callbacks: list[Callable[[int], None]] = []
        scroll.canvas.bind("<Configure>", self._resized, add="+")

    def add(self, callback: Callable[[int], None]) -> None:
        self.callbacks.append(callback)
        if self.width:
            callback(self.width)

    def _resized(self, event: Any) -> None:
        width = max(1, int(event.width) - self.padding)
        if width != self.width:
            self.width = width
            for callback in self.callbacks:
                callback(width)


def segments(ctx: Any, parent: Any, choices: Sequence[tuple[Any, str]], variable: Any, command: Callable[[], None]) -> Any:
    """Equal-width segmented choice, the same control as the replay window's view and speed switches."""
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Surface.TFrame")
    width = max(SEGMENT_MIN_CHARS, max(len(text) for _value, text in choices))  # one minimum width, sized to the longest label
    for value, text in choices:
        ttk.Radiobutton(frame, text=text, value=value, variable=variable, style="Segment.Toolbutton", width=width,
                        command=command, takefocus=True).pack(side="left", padx=0)  # butted, so a group reads as one control
    return frame


def labelled(ctx: Any, parent: Any, label: str, control_factory: Callable[[Any], Any]) -> Any:
    """A caption-style label to the left of a control, both centred on the control height."""
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Surface.TFrame")
    ttk.Label(frame, text=label, style="Surface.Secondary.TLabel").pack(side="left", padx=(0, ctx.px(SPACE["sm"])))
    control_factory(frame).pack(side="left")
    return frame


class Legend:
    """Legend row on a canvas: a drawn sample of each mark, then its words; wraps onto more lines when narrow."""

    def __init__(self, ctx: Any, parent: Any, items: Sequence[tuple[str, str]]) -> None:
        import tkinter as tk

        self.ctx = ctx
        self.items = list(items)
        self.line_h = max(int(ctx.fonts["caption"].metrics("linespace")), ctx.px(WIN_LAYOUT["legendSwatchHeight"])) + ctx.px(SPACE["xs"])
        self.canvas = tk.Canvas(parent, height=self.line_h, highlightthickness=0, borderwidth=0, takefocus=0)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self._key: tuple[Any, ...] | None = None

    def apply_theme(self, _theme: Any = None) -> None:
        self._key = None
        self.draw()

    def draw(self) -> None:
        ctx, canvas = self.ctx, self.canvas
        width = float(canvas.winfo_width())
        theme = ctx.theme()
        if width <= 1 or self._key == (width, theme.name):
            return
        self._key = (width, theme.name)
        p = theme.p
        canvas.configure(background=p.surface)
        canvas.delete("all")
        caption = ctx.fonts["caption"]
        sw, sh = ctx.px(WIN_LAYOUT["legendSwatchWidth"]), ctx.px(WIN_LAYOUT["legendSwatchHeight"])
        gap, item_gap = ctx.px(SPACE["xs"]) + ctx.px(SPACE["xxs"]), ctx.px(SPACE["lg"])
        x, line = 0.0, 0
        for kind, text in self.items:
            item_w = sw + gap + caption.measure(text)
            if x > 0 and x + item_w > width:
                x, line = 0.0, line + 1
            middle = line * self.line_h + self.line_h / 2.0
            self._swatch(kind, x, middle, sw, sh, p)
            canvas.create_text(x + sw + gap, middle, text=text, anchor="w", fill=p.text, font=caption)
            x += item_w + item_gap
        height = (line + 1) * self.line_h
        if int(canvas.cget("height")) != height:
            canvas.configure(height=height)

    def _swatch(self, kind: str, x: float, middle: float, sw: int, sh: int, p: Any) -> None:
        ctx, canvas = self.ctx, self.canvas
        line_w = max(1.0, LAYOUT["chartLineWidth"] * ctx.scale)
        dot = ctx.px(WIN_LAYOUT["chartMarkerSize"]) / 2.0
        center = x + sw / 2.0
        top, bottom = middle - sh / 2.0, middle + sh / 2.0
        hairline = max(1, ctx.px(WIN_LAYOUT["strokeHairline"]))
        if kind == "recovery":
            canvas.create_line(x, middle, x + sw, middle, fill=p.accent, width=line_w)
            canvas.create_oval(center - dot, middle - dot, center + dot, middle + dot, fill=p.accent, outline=p.accent)
        elif kind == "naive":
            dash = tuple(max(1, int(round(value * ctx.scale))) for value in WIN_LAYOUT["dashPattern"])
            canvas.create_line(x, middle, x + sw, middle, fill=p.lineStrong, width=line_w, dash=dash)
            canvas.create_oval(center - dot, middle - dot, center + dot, middle + dot, fill=p.surface, outline=p.lineStrong,
                               width=line_w)
        elif kind == "band":
            canvas.create_rectangle(x, top, x + sw, bottom, fill=p.bandFill, outline="")
        elif kind == "failure":
            canvas.create_line(x, middle, x + sw, middle, fill=p.alert, width=max(1.0, LAYOUT["failureLineWidth"] * ctx.scale))
        elif kind.startswith("stop"):
            fill = getattr(p, f"atlasStop{kind[-1]}")
            canvas.create_rectangle(x, top, x + sw, bottom, fill=fill, outline=p.line, width=hairline)
        elif kind == "fails":
            size = ctx.px(WIN_LAYOUT["statusMarkerSize"])
            canvas.create_rectangle(center - size / 2.0, middle - size / 2.0, center + size / 2.0, middle + size / 2.0,
                                    fill=p.alert, outline="")
        elif kind == "winner":
            # Square corners like the other swatches: a surface box with the accent bar along its bottom edge.
            bar = max(1, ctx.px(LAYOUT["atlasWinnerBar"]))
            canvas.create_rectangle(x, top, x + sw, bottom, fill=p.surface, outline=p.line, width=hairline)
            canvas.create_rectangle(x + hairline, bottom - bar, x + sw - hairline + 1, bottom - hairline + 1, fill=p.accent,
                                    outline="")


def bullet_list(ctx: Any, parent: Any, lines: Sequence[str], wrap: Wrap) -> None:
    from tkinter import ttk

    indent = ctx.fonts["body"].measure("•") + ctx.px(SPACE["sm"])
    limit = ctx.px(LAYOUT["viewerContentMaxWidth"])
    for text in lines:
        row = ttk.Frame(parent, style="Surface.TFrame")
        row.pack(fill="x", pady=(0, ctx.px(SPACE["xxs"])))
        ttk.Label(row, text="•", style="Surface.Body.TLabel").pack(side="left", anchor="n", padx=(0, ctx.px(SPACE["sm"])))
        label = ttk.Label(row, text=text, style="Surface.Body.TLabel", justify="left")
        label.pack(side="left", anchor="w")
        wrap.add(lambda width, label=label: label.configure(wraplength=max(1, min(width, limit) - indent)))


def limits_section(ctx: Any, body: Any, limits: Sequence[str], rules: Sequence[str], wrap: Wrap) -> Disclosure:
    """Collapsed "Limits": the run's limits word for word, then how a cell is judged."""
    ctx.hairline(body).pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["xs"])))
    disclosure = Disclosure(body, TEXT["limitsTitle"], ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x")
    bullet_list(ctx, disclosure.body, list(limits) + list(rules), wrap)
    return disclosure


def source_section(ctx: Any, body: Any, source: Mapping[str, Any]) -> Disclosure:
    """Collapsed "Source": where the numbers came from. Hashes show 12 characters; the tooltip has all of them."""
    from tkinter import ttk

    disclosure = Disclosure(body, TEXT["sourceTitle"], ctx.images, pad=ctx.px(SPACE["xs"]))
    disclosure.frame.pack(fill="x", pady=(ctx.px(SPACE["xs"]), 0))
    grid = ttk.Frame(disclosure.body, style="Surface.TFrame")
    grid.pack(fill="x")
    grid.columnconfigure(0, minsize=ctx.label_width)
    rows = [
        (TEXT["sourceRunFolder"], source.get("run_dir"), False),
        (TEXT["sourceFile"], source.get("file"), False),
        (fmt.field_label("sha256"), source.get("sha256"), True),
        (TEXT["sourceManifest"], source.get("manifest_sha256"), True),
        (fmt.field_label("schema"), source.get("schema"), False),
        (TEXT["sourceRevision"], source.get("git_revision"), True),
        (TEXT["sourceTreeState"], source.get("git_tree_state"), False),
    ]
    for index, (label, value, short) in enumerate(rows):
        ttk.Label(grid, text=label, style="Surface.Secondary.TLabel").grid(row=index, column=0, sticky="nw",
                                                                           pady=(0, ctx.px(SPACE["xs"])))
        shown = fmt.NOT_RECORDED if value is None else (str(value)[:12] if short else str(value))
        widget = ttk.Label(grid, text=shown, style="Surface.Secondary.TLabel" if value is None else "Surface.Mono.TLabel")
        widget.grid(row=index, column=1, sticky="nw", pady=(0, ctx.px(SPACE["xs"])))
        if short and value is not None:
            tip = Tooltip(widget, str(value), ctx.theme, ctx.fonts["mono"], scale=ctx.scale)
            # A hint still waiting for its delay when the window closes must not fire into a destroyed widget.
            widget.bind("<Destroy>", lambda _event, tip=tip: tip._hide(), add="+")
    return disclosure


class FindingBlock:
    """The lead sentence quoted from the write-up with its source caption ("From README.md"), and any other sentences in
    a collapsed "More from the write-up" section. Backticks from the markdown are removed for display only."""

    def __init__(self, ctx: Any, parent: Any, wrap: Wrap, findings: Sequence[Mapping[str, Any]], lead: int) -> None:
        from tkinter import ttk

        self.frame = ttk.Frame(parent, style="Surface.TFrame")
        first, more = rd.split_findings(findings, lead)
        limit = ctx.px(LAYOUT["viewerContentMaxWidth"])
        self.text: str | None = None
        self.more_texts: list[str] = []
        self.disclosure: Disclosure | None = None
        if first is not None:
            self.text = rd.display_text(first["text"])
            label = ttk.Label(self.frame, text=self.text, style="Surface.Body.TLabel", justify="left")
            label.pack(anchor="w", fill="x")
            wrap.add(lambda width, label=label: label.configure(wraplength=min(width, limit)))
            self.caption = ttk.Label(self.frame, text=TEXT["sourcePrefix"].format(source=first["source"]),
                                     style="Surface.Caption.TLabel")
            self.caption.pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
        if more:
            disclosure = Disclosure(self.frame, TEXT["moreFindings"], ctx.images, pad=ctx.px(SPACE["xs"]))
            disclosure.frame.pack(fill="x", pady=(ctx.px(SPACE["xs"]), 0))
            indent = ctx.px(SPACE["lg"]) + ctx.px(SPACE["xs"])
            for finding in more:
                text = rd.display_text(finding["text"])
                self.more_texts.append(text)
                label = ttk.Label(disclosure.body, text=text, style="Surface.Body.TLabel", justify="left")
                label.pack(anchor="w", fill="x", padx=(indent, 0), pady=(0, ctx.px(SPACE["sm"])))
                wrap.add(lambda width, label=label: label.configure(wraplength=min(width - indent, limit)))
            ttk.Label(disclosure.body, text=TEXT["sourcePrefix"].format(source=rd.sources_text(more)),
                      style="Surface.Caption.TLabel").pack(anchor="w", padx=(indent, 0))
            self.disclosure = disclosure


class ResultTile:
    """A bordered tile with a caption and one value. Unlike the replay's MetricTile, a long value such as
    "Below 0 to above 300 ms" wraps onto a second line instead of being cut off; a missing value reads
    Not recorded in the secondary color."""

    def __init__(self, ctx: Any, parent: Any, caption: str, wrap: int) -> None:
        from tkinter import ttk

        pad = ctx.px(SPACE["sm"])
        self.frame = ttk.Frame(parent, style="Tile.TFrame", padding=(pad, pad, pad, pad))
        ttk.Label(self.frame, text=caption, style="Tile.Caption.TLabel", wraplength=wrap, justify="left").pack(anchor="w")
        self.value = ttk.Label(self.frame, style="Tile.Value.TLabel", wraplength=wrap, justify="left", anchor="w")
        self.value.pack(anchor="w", fill="x", pady=(ctx.px(SPACE["xxs"]), 0))
        self.text: str | None = None

    def set(self, text: str | None) -> None:
        self.text = text
        if text is None:
            self.value.configure(text=TEXT["notComputed"], style="Tile.Missing.TLabel")
        else:
            # A unit never wraps alone onto a line: "0.298 to 0.332 m" breaks after "to", not before "m".
            self.value.configure(text=re.sub(r"(\d) (ms|m)\b", "\\1\u00a0\\2", text), style="Tile.Value.TLabel")


class InspectorPage:
    """One view's inspector: the empty state, or a header, tiles, sections, and a collapsed Details."""

    def __init__(self, ctx: Any, parent: Any, content_width: int, empty_hint: str) -> None:
        from tkinter import ttk

        self.ctx = ctx
        self.width = content_width
        self.frame = ttk.Frame(parent, style="Surface.TFrame")
        self.empty = ttk.Frame(self.frame, style="Surface.TFrame")
        ttk.Label(self.empty, text=TEXT["inspectorEmptyTitle"], style="Surface.Headline.TLabel").pack(anchor="w")
        ttk.Label(self.empty, text=empty_hint, style="Surface.Secondary.TLabel", wraplength=content_width,
                  justify="left").pack(anchor="w", pady=(ctx.px(SPACE["xs"]), 0))
        self.content = ttk.Frame(self.frame, style="Surface.TFrame")
        self.title = ttk.Label(self.content, style="Surface.Headline.TLabel", wraplength=content_width, justify="left")
        self.title.pack(anchor="w")
        self.subtitle = ttk.Label(self.content, style="Surface.Secondary.TLabel", wraplength=content_width, justify="left")
        self.subtitle.pack(anchor="w", pady=(ctx.px(SPACE["xxs"]), 0))
        self.detail = ttk.Label(self.content, style="Surface.Secondary.TLabel", wraplength=content_width, justify="left")
        self.show(False)

    def set_detail(self, text: str | None) -> None:
        """An optional last header line, such as the atlas conditions, kept whole on its own line."""
        if text:
            self.detail.configure(text=text)
            self.detail.pack(anchor="w", after=self.subtitle)
        else:
            self.detail.pack_forget()

    def tiles(self, captions: Sequence[str]) -> list[ResultTile]:
        from tkinter import ttk

        holder = ttk.Frame(self.content, style="Surface.TFrame")
        holder.pack(fill="x", pady=(self.ctx.px(SPACE["md"]), 0))
        holder.columnconfigure(0, weight=1, uniform="tiles")
        holder.columnconfigure(1, weight=1, uniform="tiles")
        gap = self.ctx.px(SPACE["sm"])
        border = 2 * self.ctx.px(WIN_LAYOUT["hairline"])
        wrap = max(1, (self.width - gap) // 2 - 2 * self.ctx.px(SPACE["sm"]) - border)
        tiles = []
        for index, caption in enumerate(captions):
            tile = ResultTile(self.ctx, holder, caption, wrap)
            tile.frame.grid(row=index // 2, column=index % 2, sticky="nsew", padx=(0, gap) if index % 2 == 0 else (0, 0),
                            pady=(0, gap))
            tiles.append(tile)
        return tiles

    def heading(self, text: str) -> Any:
        from tkinter import ttk

        label = ttk.Label(self.content, text=text, style="Surface.Strong.TLabel")
        label.pack(anchor="w", pady=(self.ctx.px(SPACE["md"]), self.ctx.px(SPACE["xs"])))
        return label

    def body_line(self, style: str = "Surface.Body.TLabel") -> Any:
        from tkinter import ttk

        label = ttk.Label(self.content, style=style, wraplength=self.width, justify="left")
        label.pack(anchor="w", fill="x", pady=(0, self.ctx.px(SPACE["xxs"])))
        return label

    def details(self) -> Disclosure:
        self.details_line = self.ctx.hairline(self.content)
        self._line_pad = (self.ctx.px(SPACE["md"]), self.ctx.px(SPACE["xs"]))
        self.details_line.pack(fill="x", pady=self._line_pad)
        self.disclosure = Disclosure(self.content, TEXT["detailsTitle"], self.ctx.images, pad=self.ctx.px(SPACE["xs"]))
        self.disclosure.frame.pack(fill="x")
        return self.disclosure

    def show_details(self, visible: bool, before: Any) -> None:
        """Hide the Details hairline and disclosure when there is nothing to put in them."""
        if visible:
            self.details_line.pack(fill="x", pady=self._line_pad, before=before)
            self.disclosure.frame.pack(fill="x", before=before)
        else:
            self.details_line.pack_forget()
            self.disclosure.frame.pack_forget()

    def show(self, filled: bool) -> None:
        self.filled = filled
        if filled:
            self.empty.pack_forget()
            self.content.pack(fill="x")
        else:
            self.content.pack_forget()
            self.empty.pack(fill="x")


# ---- delay and loss --------------------------------------------------------------------------------
class DelayView:
    def __init__(self, owner: "ResultsWindow", body: Any, inspector_parent: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        ctx, data = owner.ctx, owner.data
        self.ctx, self.data, self.owner = ctx, data, owner
        self.frame = ttk.Frame(body, style="Surface.TFrame")
        wrap = owner.wrap
        self.finding = FindingBlock(ctx, self.frame, wrap, data.s6["findings"], rd.LEAD_FINDING["delay"])
        self.finding.frame.pack(fill="x")
        self.path_var = tk.StringVar(master=owner.window, value="constant_velocity")
        self.loss_var = tk.StringVar(master=owner.window, value="independent")
        controls = ttk.Frame(self.frame, style="Surface.TFrame")
        controls.pack(fill="x", pady=(ctx.px(SPACE["lg"]), ctx.px(SPACE["md"])))
        labelled(ctx, controls, TEXT["pathLabel"], lambda parent: segments(
            ctx, parent, [(motion, rd.PATH_TITLES[motion]) for motion in data.s6["motions"]], self.path_var,
            self._changed)).pack(side="left")
        labelled(ctx, controls, TEXT["lossTypeLabel"], lambda parent: segments(
            ctx, parent, [(model, rd.LOSS_TYPE_WORDS[model]) for model in data.s6["loss_models"]], self.loss_var,
            self._changed)).pack(side="left", padx=(ctx.px(SPACE["xl"]), 0))
        # The four panels share one error axis per path and loss type, so say where it ends.
        self.scale_note = ttk.Label(controls, style="Surface.Caption.TLabel")
        self.scale_note.pack(side="left", padx=(ctx.px(SPACE["xl"]), 0))
        self._render_scale()
        # The legend sits above the charts, so it is on screen without scrolling.
        legend = [(kind, text if text else TEXT["legendInterval"].format(n=data.s6_seeds)) for kind, text in LEGEND_DELAY]
        self.legend = Legend(ctx, self.frame, legend)
        self.legend.canvas.pack(fill="x", pady=(0, ctx.px(SPACE["md"])))
        self.chart = DelayChart(ctx, self.frame, data, self._selected)
        self.chart.canvas.pack(fill="x")
        s6 = data.s6
        rules = [rd.sentence(s6["status"]) + ".", rd.sentence(s6["criterion"]) + ".", rd.rule_text(data)]
        self.limits = limits_section(ctx, self.frame, s6["limits"], rules, wrap)
        self.source = source_section(ctx, self.frame, s6["source"])
        self._build_inspector(inspector_parent, owner.inspector_width)

    def _build_inspector(self, parent: Any, width: int) -> None:
        page = InspectorPage(self.ctx, parent, width, TEXT["inspectorEmptyHintDelay"])
        self.page = page
        self.tiles = page.tiles((TEXT["tileMedian"], TEXT["tileInterval"], TEXT["tileSeedsFailing"], TEXT["tileCellResult"]))
        page.heading(TEXT["compareTitle"])
        self.compare = [page.body_line() for _ in range(3)]
        disclosure = page.details()
        self.details_disclosure = disclosure
        self.details = KeyValueGrid(disclosure.body, wrap=width, scale=self.ctx.scale, stacked=True)
        self.details.frame.pack(fill="x")

    def _render_scale(self) -> None:
        top = self.data.error_scale(self.path_var.get(), self.loss_var.get()).top
        self.scale_note.configure(text=rd.scale_caption(top))

    def _changed(self) -> None:
        self.chart.set_view(self.path_var.get(), self.loss_var.get())
        self._render_scale()

    def focus_target(self) -> Any:
        return self.chart.canvas

    def apply_select(self, spec: Mapping[str, str]) -> None:
        motion, loss_type, point = rd.delay_selection(self.data, spec)
        self.path_var.set(motion)
        self.loss_var.set(loss_type)
        self._changed()
        if point is not None:
            self.chart.select(point)

    def apply_theme(self, theme: Any) -> None:
        self.chart.apply_theme(theme)
        self.legend.apply_theme(theme)

    def _selected(self, point: rd.Point | None) -> None:
        page = self.page
        if point is None:
            page.show(False)
            return
        data = self.data
        series = data.series(point.motion, point.loss_model, point.loss_probability, point.arm)
        page.title.configure(text=rd.arm_name(point.arm))
        page.subtitle.configure(text=rd.point_subtitle(data, point))
        if series is None:
            for tile in self.tiles:
                tile.set(None)
        else:
            i = point.index
            self.tiles[0].set(rd.metres_text(series["p95_median_m"][i]))
            self.tiles[1].set(rd.range_text(series["p95_ci_low_m"][i], series["p95_ci_high_m"][i]))
            self.tiles[2].set(rd.count_text(series["fail_count"][i], series["n_seeds"][i]))
            fails = series["label"][i] == "fails"
            self.tiles[3].set(TEXT["cellFails"] if fails else TEXT["cellPasses"])
            image = self.ctx.images.status("alert", WIN_LAYOUT["statusMarkerSize"]) if fails else ""
            self.tiles[3].value.configure(image=image, compound="left")
        lines = rd.compare_lines(data, point)
        for index, label in enumerate(self.compare):
            label.pack_forget()
            if index < len(lines):
                label.configure(text=lines[index], style="Surface.Body.TLabel" if index == 0 else "Surface.Secondary.TLabel")
                label.pack(anchor="w", fill="x", pady=(0, self.ctx.px(SPACE["xxs"])), before=page.details_line)
        s6 = data.s6
        cell_id = series["cell_ids"][point.index] if series is not None else None
        self.details.set_rows([
            (TEXT["detailCellID"], cell_id, "mono"),
            (TEXT["detailRule"], f"{rd.sentence(s6['criterion'])}. {rd.sentence(s6['cell_label_rule'])}."),
            (TEXT["detailIntervals"], rd.sentence(s6["interval_note"]) + "."),
        ])
        page.show(True)


# ---- tolerance atlas --------------------------------------------------------------------------------
class AtlasView:
    def __init__(self, owner: "ResultsWindow", body: Any, inspector_parent: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        ctx, data = owner.ctx, owner.data
        self.ctx, self.data, self.owner = ctx, data, owner
        self.frame = ttk.Frame(body, style="Surface.TFrame")
        wrap = owner.wrap
        self.finding = FindingBlock(ctx, self.frame, wrap, data.atlas["findings"], rd.LEAD_FINDING["atlas"])
        self.finding.frame.pack(fill="x")
        self.stratum_var = tk.StringVar(master=owner.window, value=data.stratum_order[0])
        self.loss_var = tk.StringVar(master=owner.window, value="independent")
        self.level_var = tk.DoubleVar(master=owner.window, value=0.0)
        first = ttk.Frame(self.frame, style="Surface.TFrame")
        first.pack(fill="x", pady=(ctx.px(SPACE["lg"]), 0))
        self.menu = tk.Menu(owner.window, tearoff=False)
        for stratum in data.stratum_order:
            self.menu.add_radiobutton(label=data.strata[stratum]["title"], value=stratum, variable=self.stratum_var,
                                      command=self._changed)
        self.conditions = None

        def conditions(parent: Any) -> Any:
            self.conditions = ttk.Menubutton(parent, text=self._stratum_title(), menu=self.menu, style="Results.TMenubutton",
                                             compound="right", takefocus=True,
                                             image=ctx.images.icon("chevron_down", "textSecondary", WIN_LAYOUT["chevronSize"]))
            return self.conditions

        labelled(ctx, first, ATLAS_TEXT["conditionsLabel"], conditions).pack(side="left")
        # What the conditions are (rate, noise) is a hint on the control, as in the macOS app, not a line of text beside it.
        self.conditions_tip = Tooltip(self.conditions, data.strata[self.stratum_var.get()]["description"], ctx.theme,
                                      ctx.fonts["caption"], scale=ctx.scale)
        self.conditions.bind("<Destroy>", lambda _event: self.conditions_tip._hide(), add="+")
        second = ttk.Frame(self.frame, style="Surface.TFrame")
        second.pack(fill="x", pady=(ctx.px(SPACE["sm"]), 0))
        labelled(ctx, second, TEXT["lossTypeLabel"], lambda parent: segments(
            ctx, parent, [(model, rd.LOSS_TYPE_WORDS[model]) for model in rd.LOSS_TYPES], self.loss_var, self._changed)).pack(side="left")
        levels = [(level, TEXT["lossNone"] if level == 0.0 else ATLAS_TEXT["levelPercent"].format(percent=rd.percent(level)))
                  for level in rd.LOSS_LEVELS]
        labelled(ctx, second, TEXT["lossLevelLabel"], lambda parent: segments(
            ctx, parent, levels, self.level_var, self._changed)).pack(side="left", padx=(ctx.px(SPACE["xl"]), 0))
        self.summary = ttk.Label(self.frame, style="Surface.Secondary.TLabel", justify="left")
        self.summary.pack(anchor="w", pady=(ctx.px(SPACE["md"]), ctx.px(SPACE["sm"])))
        wrap.add(lambda width: self.summary.configure(wraplength=width))
        # The legend sits above the grid, between the summary line and the column headers, as in the delay view.
        self.legend = Legend(ctx, self.frame, LEGEND_ATLAS)
        self.legend.canvas.pack(fill="x", pady=(0, ctx.px(SPACE["sm"])))
        self.grid = AtlasGrid(ctx, self.frame, data, self._selected)
        self.grid.frame.pack(fill="x")
        self.limits = limits_section(ctx, self.frame, data.atlas["limits"], [rd.rule_text(data)], wrap)
        self.source = source_section(ctx, self.frame, data.atlas["source"])
        self._build_inspector(inspector_parent, owner.inspector_width)
        self._render_controls()

    def _stratum_title(self) -> str:
        return " " + self.data.strata[self.stratum_var.get()]["title"]  # the space keeps the text off the left edge of the image

    def _build_inspector(self, parent: Any, width: int) -> None:
        from tkinter import ttk

        page = InspectorPage(self.ctx, parent, width, TEXT["inspectorEmptyHintAtlas"])
        self.page = page
        self.help = page.body_line("Surface.Caption.TLabel")
        self.help.pack_configure(pady=(self.ctx.px(SPACE["xs"]), self.ctx.px(SPACE["xxs"])))
        self.tiles = page.tiles((ATLAS_TEXT["tileMostDelay"], ATLAS_TEXT["tileInterval"], ATLAS_TEXT["tileCause"],
                                 ATLAS_TEXT["tileSeeds"]))
        self.interval_note = ttk.Label(page.content, text=ATLAS_TEXT["intervalCaption"], style="Surface.Caption.TLabel",
                                       wraplength=width, justify="left")
        self.interval_note.pack(anchor="w")
        page.heading(ATLAS_TEXT["acrossLossTitle"])
        self.loss_rows = ttk.Frame(page.content, style="Surface.TFrame")
        self.loss_rows.pack(fill="x")
        self.loss_rows.columnconfigure(0, weight=1)
        self._row_widgets: list[tuple[Any, Any, Any]] = []
        self._conditions: list[str] = []
        disclosure = page.details()
        self.details_disclosure = disclosure
        ttk.Label(disclosure.body, text=ATLAS_TEXT["diffTitle"], style="Surface.Strong.TLabel").pack(anchor="w")
        self.diff_table = ttk.Frame(disclosure.body, style="Surface.TFrame")
        self.diff_table.pack(fill="x", pady=(self.ctx.px(SPACE["xs"]), 0))
        ttk.Label(disclosure.body, text=ATLAS_TEXT["diffCaption"], style="Surface.Caption.TLabel", wraplength=width,
                  justify="left").pack(anchor="w", pady=(self.ctx.px(SPACE["xs"]), 0))
        self.rule = ttk.Label(page.content, style="Surface.Caption.TLabel", wraplength=width, justify="left")
        self.rule.pack(anchor="w", pady=(self.ctx.px(SPACE["md"]), 0))

    def _render_controls(self) -> None:
        stratum = self.data.strata[self.stratum_var.get()]
        if self.conditions is not None:
            self.conditions.configure(text=self._stratum_title())
        self.conditions_tip.set_text(stratum["description"])
        self.summary.configure(text=rd.summary_text(stratum))

    def pick_loss(self, index: int) -> None:
        """Show the loss condition of an "Across loss levels" row in the atlas (the no-loss row sets level None)."""
        if index >= len(self._conditions):
            return
        model, _, level = self._conditions[index].partition(":")
        if float(level) > 0:
            self.loss_var.set(model)
        self.level_var.set(float(level))
        self._changed()

    def loss_key(self) -> str:
        return rd.level_loss_key(self.loss_var.get(), self.level_var.get())

    def _changed(self) -> None:
        self._render_controls()
        self.grid.set_view(self.stratum_var.get(), self.loss_key())

    def focus_target(self) -> Any:
        return self.grid.canvas

    def apply_select(self, spec: Mapping[str, str]) -> None:
        stratum, loss_type, level, variant, arm = rd.atlas_selection(self.data, spec)
        self.stratum_var.set(stratum)
        self.loss_var.set(loss_type)
        self.level_var.set(level)
        self._changed()
        if variant is not None and arm is not None:
            self.grid.select((variant, arm))

    def apply_theme(self, theme: Any) -> None:
        self.grid.apply_theme(theme)
        self.legend.apply_theme(theme)

    def _selected(self, selection: tuple[str, str] | None) -> None:
        from tkinter import ttk

        page, data, ctx = self.page, self.data, self.ctx
        if selection is None:
            page.show(False)
            return
        variant, arm = selection
        stratum_id = self.stratum_var.get()
        stratum = data.strata[stratum_id]
        loss = self.loss_key()
        entry = data.entry(stratum_id, variant, loss, arm)
        page.title.configure(text=data.arm_label(arm))
        page.subtitle.configure(text=rd.cell_subtitle(data.variant_title(variant), data.loss_title(loss)))
        page.set_detail(stratum["title"])
        self.interval_note.configure(text=rd.interval_caption(entry, data.atlas["n_seeds"]))
        self.help.configure(text=rd.arm_help(arm) or "")
        self.tiles[0].set(rd.tolerated_text(entry))
        fails = entry is not None and entry["kind"] == "fails_at_zero"
        self.tiles[0].value.configure(image=ctx.images.status("alert", WIN_LAYOUT["statusMarkerSize"]) if fails else "",
                                      compound="left")
        self.tiles[1].set(rd.interval_text(entry["interval_s"]) if entry else None)
        self.tiles[2].set(rd.cause_text(entry["failure_mode"]) if entry else None)
        self.tiles[3].set(ATLAS_TEXT["seedsPerCell"].format(n=data.atlas["n_seeds"]))
        conditions = list(stratum["loss_conditions"])
        self._conditions = conditions
        while len(self._row_widgets) < len(conditions):
            row = ttk.Frame(self.loss_rows, style="Results.Row.TFrame", padding=(ctx.px(SPACE["xs"]), ctx.px(SPACE["xxs"])))
            row.grid(row=len(self._row_widgets), column=0, sticky="ew")
            name = ttk.Label(row, style="Results.Row.TLabel")
            name.pack(side="left")
            value = ttk.Label(row, style="Results.Row.TLabel")
            value.pack(side="right")
            index = len(self._row_widgets)
            for widget in (row, name, value):  # a click shows that loss level in the atlas
                widget.bind("<Button-1>", lambda _event, index=index: self.pick_loss(index))
                widget.configure(cursor="hand2")
            self._row_widgets.append((row, name, value))
        for index, (row, name, value) in enumerate(self._row_widgets):
            if index >= len(conditions):
                row.grid_remove()
                continue
            row.grid()
            other = data.entry(stratum_id, variant, conditions[index], arm)
            chosen = conditions[index] == loss
            suffix = "Selected" if chosen else ""
            row.configure(style=f"Results.Row{suffix}.TFrame")
            name.configure(text=data.loss_title(conditions[index]), style=f"Results.Row{suffix}.TLabel")
            value.configure(text=rd.entry_cell_text(other), style=f"Results.Row{suffix}.TLabel")
        for child in self.diff_table.winfo_children():
            child.destroy()
        rows = rd.diff_rows(entry)
        page.show_details(bool(rows), self.rule)
        if rows:
            headers = (ATLAS_TEXT["diffColumnDelay"], ATLAS_TEXT["diffColumnMedian"], ATLAS_TEXT["diffColumnInterval"])
            for column, header in enumerate(headers):
                ttk.Label(self.diff_table, text=header, style="Surface.Caption.TLabel").grid(
                    row=0, column=column, sticky="w", padx=(0, ctx.px(SPACE["md"])), pady=(0, ctx.px(SPACE["xxs"])))
            for row_index, values in enumerate(rows, start=1):
                for column, value in enumerate(values):
                    ttk.Label(self.diff_table, text=value, style="Surface.Mono.TLabel").grid(
                        row=row_index, column=column, sticky="w", padx=(0, ctx.px(SPACE["md"])))
        self.rule.configure(text=rd.rule_text(data))
        page.show(True)


# ---- the window ---------------------------------------------------------------------------------------
class ResultsWindow:
    """One Results window per app. ``data`` is the loaded fixture, or ``None`` with ``error`` set."""

    def __init__(self, ctx: Any, data: rd.ResultsData | None, *, error: str | None = None, view: str = "delay") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.ctx = ctx
        self.data = data
        self.error = error
        self.views: dict[str, Any] = {}
        self.view = ""
        self._selecting = False
        size = WIN_LAYOUT["resultsWindow"]
        window = tk.Toplevel(ctx.root)
        self.window = window
        window.title(TEXT["windowTitle"])
        window.geometry(f"{ctx.px(size['width'])}x{ctx.px(size['height'])}")
        window.minsize(ctx.px(size["minWidth"]), ctx.px(size["minHeight"]))
        window.transient(ctx.root)
        window.bind("<Escape>", lambda _event: self.destroy())
        window.rowconfigure(0, weight=1)
        window.columnconfigure(2, weight=1)
        self._configure_styles(ctx.theme())
        self._build_sidebar(window).grid(row=0, column=0, sticky="nsw")
        self._vline(window).grid(row=0, column=1, sticky="ns")
        self._build_content(window).grid(row=0, column=2, sticky="nsew")
        self._vline(window).grid(row=0, column=3, sticky="ns")
        self._build_inspector(window).grid(row=0, column=4, sticky="nse")
        window.bind("<MouseWheel>", self._wheel, add="+")
        try:
            window.bind("<TouchpadScroll>", self._touchpad, add="+")
        except tk.TclError:
            pass
        self.apply_theme(ctx.theme(), redraw=False)
        ctx.add_theme_hook(self.apply_theme)
        self.show_view(view if view in VIEWS else "delay")
        self.sidebar.focus_set()

    # ---- construction ------------------------------------------------------------------------
    def _vline(self, parent: Any) -> Any:
        from tkinter import ttk

        return ttk.Frame(parent, style="Line.TFrame", width=self.ctx.px(WIN_LAYOUT["hairline"]))

    def _build_sidebar(self, parent: Any) -> Any:
        import tkinter as tk
        from tkinter import ttk

        ctx = self.ctx
        width = ctx.px(WIN_LAYOUT["resultsSidebarWidth"])
        pad = ctx.px(SPACE["sm"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width, padding=(pad, ctx.px(SPACE["md"]), pad, pad))
        frame.pack_propagate(False)
        self.icons = {name: tk.PhotoImage(master=self.window) for name in VIEWS}
        tree = ttk.Treeview(frame, show="tree", selectmode="browse", style="Events.Treeview", takefocus=True, height=len(VIEWS))
        tree.column("#0", width=width - 2 * pad, stretch=True)
        for name in VIEWS:
            tree.insert("", "end", iid=name, text=f" {VIEW_TITLES[name]}", image=self.icons[name], tags=("row",))
        tree.pack(fill="x")
        tree.bind("<<TreeviewSelect>>", self._on_sidebar)
        self.sidebar = tree
        return frame

    def _build_content(self, parent: Any) -> Any:
        from tkinter import ttk

        ctx = self.ctx
        frame = ttk.Frame(parent, style="Surface.TFrame")
        inset = ctx.px(LAYOUT["contentInset"])
        header = ttk.Frame(frame, style="Surface.TFrame", padding=(inset, ctx.px(SPACE["md"]), inset, ctx.px(SPACE["md"])))
        header.pack(fill="x")
        self.heading = ttk.Label(header, style="Surface.Title.TLabel")
        self.heading.pack(side="left")
        ttk.Label(header, text=TEXT["badge"], style="Badge.TLabel").pack(side="right")
        ctx.hairline(frame).pack(fill="x")
        scroll = ScrollFrame(frame, scale=ctx.scale)
        scroll.outer.pack(fill="both", expand=True)
        self.content_scroll = scroll
        right = max(0, inset - scroll.reserve)
        self.body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(inset, ctx.px(SPACE["md"]), right, ctx.px(SPACE["xl"])))
        self.body.pack(fill="both", expand=True)
        self.wrap = Wrap(scroll, inset + right)
        if self.data is None:
            self._build_failure(self.body)
        return frame

    def _build_failure(self, body: Any) -> None:
        from tkinter import ttk

        ctx = self.ctx
        row = ttk.Frame(body, style="Surface.TFrame")
        row.pack(anchor="w", pady=(ctx.px(SPACE["sm"]), 0))
        ttk.Label(row, image=ctx.images.status("alert", WIN_LAYOUT["statusMarkerSize"]), style="Surface.Body.TLabel").pack(
            side="left", padx=(0, ctx.px(SPACE["sm"])))
        ttk.Label(row, text=TEXT["loadFailed"], style="Surface.Strong.TLabel").pack(side="left")
        hint = ttk.Label(body, text=TEXT["loadFailedHint"], style="Surface.Secondary.TLabel", justify="left")
        hint.pack(anchor="w", pady=(ctx.px(SPACE["xs"]), 0))
        self.wrap.add(lambda width: hint.configure(wraplength=width))
        if self.error:
            reason = ttk.Label(body, text=self.error, style="Surface.Caption.TLabel", justify="left")
            reason.pack(anchor="w", pady=(ctx.px(SPACE["xs"]), 0))
            self.wrap.add(lambda width: reason.configure(wraplength=width))

    def _build_inspector(self, parent: Any) -> Any:
        from tkinter import ttk

        ctx = self.ctx
        width = ctx.px(LAYOUT["inspectorWidth"])
        frame = ttk.Frame(parent, style="Surface.TFrame", width=width)
        frame.pack_propagate(False)
        scroll = ScrollFrame(frame, scale=ctx.scale)
        scroll.outer.pack(fill="both", expand=True)
        self.inspector_scroll = scroll
        pad = ctx.px(SPACE["lg"])
        self.inspector_body = ttk.Frame(scroll.inner, style="Surface.TFrame", padding=(pad, ctx.px(SPACE["md"]),
                                                                                      max(0, pad - scroll.reserve), pad))
        self.inspector_body.pack(fill="both", expand=True)
        self.inspector_width = width - pad - scroll.reserve - max(0, pad - scroll.reserve)
        return frame

    def _configure_styles(self, theme: Any) -> None:
        """Styles only this window uses: the Conditions menu button and the "Across loss levels" rows."""
        from tkinter import ttk

        ctx, p = self.ctx, theme.p
        style = ttk.Style(self.window)
        pads = control_padding(ctx.fonts, ctx.scale)
        style.layout("Results.TMenubutton", RoundedElements.layout("Quiet", "Menubutton", "Menubutton.label"))
        style.configure("Results.TMenubutton", background=p.surface, foreground=p.text, padding=pads["text"], font=ctx.fonts["body"],
                        anchor="w")
        style.map("Results.TMenubutton", background=[(state, p.surface) for state in ("active", "pressed", "focus", "disabled")],
                  foreground=[("disabled", p.textSecondary)])
        style.configure("Results.Row.TFrame", background=p.surface)
        style.configure("Results.RowSelected.TFrame", background=p.selectionFill)
        style.configure("Results.Row.TLabel", background=p.surface, foreground=p.text, font=ctx.fonts["body"])
        style.configure("Results.RowSelected.TLabel", background=p.selectionFill, foreground=p.text, font=ctx.fonts["body"])

    # ---- behaviour ---------------------------------------------------------------------------------
    def _on_sidebar(self, _event: Any = None) -> None:
        if self._selecting:
            return
        selection = self.sidebar.selection()
        if selection and selection[0] != self.view:
            self.show_view(selection[0])

    def show_view(self, name: str) -> None:
        """Switch the detail and inspector to one view, building it the first time it is shown."""
        if name not in VIEWS:
            raise ValueError(f"unknown Results view {name!r}")
        if tuple(self.sidebar.selection()) != (name,):
            self._selecting = True
            try:
                self.sidebar.selection_set(name)
                self.sidebar.focus(name)
            finally:
                self._selecting = False
        self.heading.configure(text=HEADINGS[name])
        if name == self.view or self.data is None:
            self.view = name
            return
        if self.view in self.views:
            self.views[self.view].frame.pack_forget()
            self.views[self.view].page.frame.pack_forget()
        if name not in self.views:
            factory = DelayView if name == "delay" else AtlasView
            self.views[name] = factory(self, self.body, self.inspector_body)
            self.views[name].apply_theme(self.ctx.theme())
        view = self.views[name]
        view.frame.pack(fill="x")
        view.page.frame.pack(fill="x")
        self.view = name
        self.content_scroll.canvas.yview_moveto(0)
        self.inspector_scroll.canvas.yview_moveto(0)

    def apply_select(self, spec: Mapping[str, str]) -> None:
        """Populate the current view from a ``--results-select`` spec (screenshots) and focus its chart or grid."""
        if self.data is None or self.view not in self.views:
            return
        view = self.views[self.view]
        view.apply_select(spec)
        view.focus_target().focus_set()

    def stand_alone(self) -> None:
        """For screenshots only (``--open-results``): drop the transient link to the main window, so a window capture shows
        this window alone instead of the pair AppKit draws as parent and child."""
        try:
            self.window.transient("")
            self.window.lift()
        except Exception:
            pass

    def focus(self) -> None:
        try:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_set()
        except Exception:
            pass

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def destroy(self) -> None:
        try:
            self.window.destroy()
        except Exception:
            pass

    def _route(self, widget: Any, delta: float) -> None:
        for scroll in (self.content_scroll, self.inspector_scroll):
            if widget is not None and scroll.contains(widget):
                scroll.scroll(delta)
                return

    def _wheel(self, event: Any) -> None:
        if event.state & 0x0001:  # Shift: the atlas scrolls sideways itself
            return
        self._route(self.window.winfo_containing(event.x_root, event.y_root), event.delta)

    def _touchpad(self, event: Any) -> None:
        try:
            delta = int(event.delta)
            dy = (delta & 0xFFFF) - (0x10000 if delta & 0x8000 else 0)
        except (TypeError, ValueError):
            return
        self._route(self.window.winfo_containing(event.x_root, event.y_root), dy * 40)

    def apply_theme(self, theme: Any, redraw: bool = True) -> None:
        """Recolor in place. Raises TclError once the window is closed, which drops this hook."""
        import tkinter as tk

        if not self.exists():
            raise tk.TclError("the Results window is closed")
        p = theme.p
        self.window.configure(background=p.background)
        self._configure_styles(theme)
        for name, image in self.icons.items():
            image.configure(data=photo_data(render_png(SIDEBAR_GLYPHS[name], p.textSecondary, self.ctx.px(WIN_LAYOUT["iconSize"]))))
        self.sidebar.tag_configure("row", foreground=p.text)
        for scroll in (self.content_scroll, self.inspector_scroll):
            scroll.apply_theme(theme)
        if redraw:
            for view in self.views.values():
                view.apply_theme(theme)
        self.ctx.apply_title_bar(self.window, theme.dark)


def open_results(ctx: Any, view: str = "delay", loader: Callable[[], rd.ResultsData] = rd.load_fixture) -> ResultsWindow:
    """Build the window; a broken fixture shows a plain failure state instead of any numbers."""
    try:
        data, error = loader(), None
    except rd.FixtureError as exc:
        data, error = None, str(exc)
    return ResultsWindow(ctx, data, error=error, view=view)


__all__ = ["AtlasView", "DelayView", "Legend", "ResultsWindow", "VIEWS", "open_results"]
