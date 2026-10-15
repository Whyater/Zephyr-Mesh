"""Pure data and chart logic for the Results window (no Tk, no NumPy).

The window reads one generated file, ``results_fixture.json`` (schema
``zephyr-results-fixture-1``). This module loads it strictly, then provides
everything the Tk views need that is not drawing: number formatting, the
shared chart scale, band polygons, hit testing, atlas color bins, and
keyboard selection movement. Nothing here recomputes a statistic; values are
only formatted, binned, or looked up.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .design_tokens import UI_TEXT


SCHEMA = "zephyr-results-fixture-1"
FIXTURE_NAME = "results_fixture.json"
TEXT = UI_TEXT["results"]
ATLAS_TEXT = UI_TEXT["atlas"]

BELOW_ZERO = "<0"
ABOVE_LAST = ">0.3"
LABELS = ("passes", "fails")
ENTRY_KINDS = ("value", "censored", "fails_at_zero")
CELL_KINDS = ("winner", "tie", "none")
FAILURE_MODES = {
    "lag-dominated": "causeLag",
    "gap-dominated": "causeGap",
    "noise-dominated": "causeNoise",
    "model-mismatch": "causeModel",
}
PATHS = {"straight": "constant_velocity", "turn": "constant_turn"}
PATH_TITLES = {"constant_velocity": TEXT["pathStraight"], "constant_turn": TEXT["pathTurn"]}
ARM_NAMES = {"naive": TEXT["armNaive"], "recovery": TEXT["armRecovery"]}
LOSS_TYPES = ("independent", "burst")
LOSS_TYPE_WORDS = {"independent": TEXT["lossRandom"], "burst": TEXT["lossBursty"]}
LOSS_LEVELS = (0.0, 0.1, 0.3, 0.5)
X_TICKS_MS = (0, 100, 200, 300)
TIE_LISTED_MAX = 1  # ties are summarised as a count ("Tie: 2 trackers"), so the column stays narrow; the cells show which
SERIES_ARRAYS = (
    "p95_median_m", "p95_ci_low_m", "p95_ci_high_m", "fail_count", "fail_rate", "n_seeds", "label",
    "paired_diff_median_m", "paired_diff_ci_low_m", "paired_diff_ci_high_m",
    "recovery_better_count", "recovery_worse_count", "tie_count",
)
COUNT_ARRAYS = ("fail_count", "n_seeds", "recovery_better_count", "recovery_worse_count", "tie_count")


class FixtureError(ValueError):
    """The results file is missing, malformed, or does not match its schema."""


# ---- loading -------------------------------------------------------------------------------
def resource_roots() -> tuple[Path, ...]:
    """Where bundled files live: the PyInstaller folder (and its ``_internal``), else the checkout."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return (root / "_internal", root)


def fixture_path() -> Path:
    """``desktop/results_fixture.json`` in the first resource root that has it."""
    for root in resource_roots():
        candidate = root / "desktop" / FIXTURE_NAME
        if candidate.is_file():
            return candidate
    return resource_roots()[-1] / "desktop" / FIXTURE_NAME


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FixtureError(f"{where} must be a finite number")
    return float(value)


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FixtureError(f"{where} must be a whole number")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureError(f"{where} must be a non-empty string")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise FixtureError(f"{where} must be a list")
    return value


def _require(obj: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise FixtureError(f"{where} is missing {', '.join(missing)}")


def interval_end(value: Any, where: str = "interval end") -> float | str:
    """One atlas interval end: a finite number of seconds, or the marker ``"<0"`` or ``">0.3"``."""
    if value in (BELOW_ZERO, ABOVE_LAST):
        return value
    return _number(value, where)


def _source(source: Any, where: str) -> Mapping[str, Any]:
    source = _mapping(source, where)
    _require(source, ("run_dir", "file", "sha256", "schema", "git_revision"), where)
    for key in ("run_dir", "file", "sha256", "schema", "git_revision"):
        _text(source[key], f"{where}.{key}")
    if len(source["sha256"]) != 64:
        raise FixtureError(f"{where}.sha256 must be 64 hex characters")
    return source


def _findings(value: Any, where: str) -> list[Mapping[str, Any]]:
    items = _list(value, where)
    if not items:
        raise FixtureError(f"{where} is empty")
    for index, item in enumerate(items):
        item = _mapping(item, f"{where}[{index}]")
        _text(item.get("text"), f"{where}[{index}].text")
        _text(item.get("source"), f"{where}[{index}].source")
    return items


def _limits(value: Any, where: str) -> list[str]:
    items = _list(value, where)
    return [_text(item, f"{where}[{index}]") for index, item in enumerate(items)]


def _check_s6(s6: Mapping[str, Any]) -> None:
    _require(s6, ("source", "status", "criterion", "cell_label_rule", "interval_note", "failure_p95_m", "delays_s",
                  "motions", "loss_models", "loss_levels", "arms", "limits", "findings", "series"), "s6")
    _source(s6["source"], "s6.source")
    for key in ("status", "criterion", "cell_label_rule", "interval_note"):
        _text(s6[key], f"s6.{key}")
    _number(s6["failure_p95_m"], "s6.failure_p95_m")
    delays = [_number(value, "s6.delays_s") for value in _list(s6["delays_s"], "s6.delays_s")]
    if not delays or delays != sorted(set(delays)):
        raise FixtureError("s6.delays_s must be sorted and unique")
    for key in ("motions", "loss_models", "arms"):
        for value in _list(s6[key], f"s6.{key}"):
            _text(value, f"s6.{key}")
    if set(s6["arms"]) != set(ARM_NAMES) or set(s6["motions"]) != set(PATH_TITLES):
        raise FixtureError("s6 arms and motions do not match the Results window")
    if set(s6["loss_models"]) != set(LOSS_TYPES):
        raise FixtureError("s6.loss_models do not match the Results window")
    _limits(s6["limits"], "s6.limits")
    _findings(s6["findings"], "s6.findings")
    seen = set()
    for index, series in enumerate(_list(s6["series"], "s6.series")):
        where = f"s6.series[{index}]"
        series = _mapping(series, where)
        _require(series, ("motion", "loss_model", "loss_probability", "arm", "cell_ids") + SERIES_ARRAYS, where)
        key = (series["motion"], series["loss_model"], _number(series["loss_probability"], where), series["arm"])
        if key in seen:
            raise FixtureError(f"{where} repeats {key}")
        seen.add(key)
        if series["arm"] not in ARM_NAMES or series["motion"] not in PATH_TITLES or series["loss_model"] not in LOSS_TYPES:
            raise FixtureError(f"{where} has an unknown arm, motion, or loss model")
        for name in ("cell_ids",) + SERIES_ARRAYS:
            values = _list(series[name], f"{where}.{name}")
            if len(values) != len(delays):
                raise FixtureError(f"{where}.{name} has {len(values)} values for {len(delays)} delays")
            for value in values:
                if name == "cell_ids":
                    _text(value, f"{where}.cell_ids")
                elif name == "label":
                    if value not in LABELS:
                        raise FixtureError(f"{where}.label has {value!r}")
                elif name in COUNT_ARRAYS:
                    _count(value, f"{where}.{name}")
                else:
                    _number(value, f"{where}.{name}")


def _check_atlas(atlas: Mapping[str, Any]) -> None:
    _require(atlas, ("source", "n_seeds", "failure_rule", "limits", "delays_s", "findings", "strata", "variant_titles",
                     "arm_labels", "loss_titles", "entries", "cells"), "atlas")
    _source(atlas["source"], "atlas.source")
    _count(atlas["n_seeds"], "atlas.n_seeds")
    rule = _mapping(atlas["failure_rule"], "atlas.failure_rule")
    _text(rule.get("cell_rule_status"), "atlas.failure_rule.cell_rule_status")
    _limits(atlas["limits"], "atlas.limits")
    _findings(atlas["findings"], "atlas.findings")
    titles = {name: _mapping(atlas[name], f"atlas.{name}") for name in ("variant_titles", "arm_labels", "loss_titles")}
    for name, table in titles.items():
        for key, value in table.items():
            _text(value, f"atlas.{name}.{key}")
    strata = set()
    for index, stratum in enumerate(_list(atlas["strata"], "atlas.strata")):
        where = f"atlas.strata[{index}]"
        stratum = _mapping(stratum, where)
        _require(stratum, ("id", "title", "description", "variants", "arms", "loss_conditions", "counts"), where)
        strata.add(_text(stratum["id"], f"{where}.id"))
        _text(stratum["title"], f"{where}.title")
        for key, table in (("variants", "variant_titles"), ("arms", "arm_labels"), ("loss_conditions", "loss_titles")):
            values = _list(stratum[key], f"{where}.{key}")
            unknown = [value for value in values if value not in titles[table]]
            if not values or unknown:
                raise FixtureError(f"{where}.{key} has no title for {unknown or 'an empty list'}")
        counts = _mapping(stratum["counts"], f"{where}.counts")
        for key in ("cells", "none", "tie", "winner"):
            _count(counts.get(key), f"{where}.counts.{key}")
    if not strata:
        raise FixtureError("atlas.strata is empty")
    seen = set()
    for index, entry in enumerate(_list(atlas["entries"], "atlas.entries")):
        where = f"atlas.entries[{index}]"
        entry = _mapping(entry, where)
        _require(entry, ("stratum", "variant", "loss_condition", "arm", "kind", "value_s", "censored", "interval_s",
                         "failure_mode", "improvement_vs_naive"), where)
        key = (entry["stratum"], entry["variant"], entry["loss_condition"], entry["arm"])
        if key in seen:
            raise FixtureError(f"{where} repeats {key}")
        seen.add(key)
        if entry["stratum"] not in strata:
            raise FixtureError(f"{where} names an unknown stratum")
        if entry["kind"] not in ENTRY_KINDS:
            raise FixtureError(f"{where}.kind is {entry['kind']!r}")
        if entry["kind"] == "value":
            _number(entry["value_s"], f"{where}.value_s")
        elif entry["value_s"] is not None:
            raise FixtureError(f"{where}.value_s must be null for kind {entry['kind']}")
        if entry["kind"] == "censored" and entry["censored"] != ABOVE_LAST:
            raise FixtureError(f"{where}.censored must be {ABOVE_LAST!r}")
        interval = _list(entry["interval_s"], f"{where}.interval_s")
        if len(interval) != 2:
            raise FixtureError(f"{where}.interval_s must have two ends")
        for end in interval:
            interval_end(end, f"{where}.interval_s")
        if entry["failure_mode"] is not None and entry["failure_mode"] not in FAILURE_MODES:
            raise FixtureError(f"{where}.failure_mode is {entry['failure_mode']!r}")
        for row_index, row in enumerate(_list(entry["improvement_vs_naive"], f"{where}.improvement_vs_naive")):
            row_where = f"{where}.improvement_vs_naive[{row_index}]"
            row = _mapping(row, row_where)
            _number(row.get("delay_s"), f"{row_where}.delay_s")
            _number(row.get("median_p95_diff_m"), f"{row_where}.median_p95_diff_m")
            ends = _list(row.get("interval_m"), f"{row_where}.interval_m")
            if len(ends) != 2:
                raise FixtureError(f"{row_where}.interval_m must have two ends")
            for end in ends:
                _number(end, f"{row_where}.interval_m")
    for index, cell in enumerate(_list(atlas["cells"], "atlas.cells")):
        where = f"atlas.cells[{index}]"
        cell = _mapping(cell, where)
        _require(cell, ("stratum", "variant", "loss_condition", "kind", "arms", "delay_text"), where)
        if cell["kind"] not in CELL_KINDS:
            raise FixtureError(f"{where}.kind is {cell['kind']!r}")
        arms = _list(cell["arms"], f"{where}.arms")
        if (cell["kind"] == "none") != (not arms) or (cell["kind"] == "winner" and len(arms) != 1):
            raise FixtureError(f"{where}.arms does not match its kind")


def parse_fixture(obj: Any) -> "ResultsData":
    """Validate a decoded fixture and index it. Raises ``FixtureError`` on any mismatch."""
    obj = _mapping(obj, "fixture")
    _require(obj, ("schema", "simulated", "s6", "atlas"), "fixture")
    if obj["schema"] != SCHEMA:
        raise FixtureError(f"schema is {obj['schema']!r}, expected {SCHEMA!r}")
    if obj["simulated"] is not True:
        raise FixtureError("the fixture must declare simulated: true")
    s6 = _mapping(obj["s6"], "s6")
    atlas = _mapping(obj["atlas"], "atlas")
    _check_s6(s6)
    _check_atlas(atlas)
    return ResultsData(s6, atlas)


def load_fixture(path: Path | None = None) -> "ResultsData":
    """Read and validate the results file; every failure is a ``FixtureError`` with a plain reason."""
    path = Path(path) if path is not None else fixture_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FixtureError(f"{path.name} was not found") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise FixtureError(f"{path.name} could not be read") from exc

    def no_constants(name: str) -> Any:
        raise FixtureError(f"{name} is not allowed in the results file")

    try:
        obj = json.loads(raw, parse_constant=no_constants)
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{path.name} is not valid JSON (line {exc.lineno})") from exc
    return parse_fixture(obj)


# ---- indexed data --------------------------------------------------------------------------
def loss_key(model: str, probability: float) -> str:
    """The fixture's loss condition id, for example ``independent:0`` or ``burst:0.3``."""
    return f"{model}:{float(probability):g}"


@dataclass(frozen=True)
class Point:
    """One chart point: a series and a delay index."""

    motion: str
    loss_model: str
    loss_probability: float
    arm: str
    index: int


class ResultsData:
    """The validated fixture plus lookups. Values are copied out unchanged."""

    def __init__(self, s6: Mapping[str, Any], atlas: Mapping[str, Any]) -> None:
        self.s6 = s6
        self.atlas = atlas
        self.delays_s: list[float] = [float(value) for value in s6["delays_s"]]
        self._series = {(item["motion"], item["loss_model"], float(item["loss_probability"]), item["arm"]): item
                        for item in s6["series"]}
        self.strata = {item["id"]: item for item in atlas["strata"]}
        self.stratum_order = [item["id"] for item in atlas["strata"]]
        self._entries = {(item["stratum"], item["variant"], item["loss_condition"], item["arm"]): item for item in atlas["entries"]}
        self._cells = {(item["stratum"], item["variant"], item["loss_condition"]): item for item in atlas["cells"]}
        self.y_max = nice_ceiling(max((value for item in s6["series"] for value in item["p95_ci_high_m"]), default=0.0))

    # S6
    def series(self, motion: str, loss_model: str, probability: float, arm: str) -> Mapping[str, Any] | None:
        return self._series.get((motion, loss_model, float(probability), arm))

    def value(self, point: Point, name: str) -> Any:
        series = self.series(point.motion, point.loss_model, point.loss_probability, point.arm)
        return None if series is None else series[name][point.index]

    @property
    def s6_seeds(self) -> int:
        counts = {value for item in self.s6["series"] for value in item["n_seeds"]}
        return max(counts) if counts else 0

    # Atlas
    def entry(self, stratum: str, variant: str, loss: str, arm: str) -> Mapping[str, Any] | None:
        return self._entries.get((stratum, variant, loss, arm))

    def cell(self, stratum: str, variant: str, loss: str) -> Mapping[str, Any] | None:
        return self._cells.get((stratum, variant, loss))

    def loss_title(self, key: str) -> str:
        return str(self.atlas["loss_titles"].get(key, key))

    def arm_label(self, arm: str) -> str:
        return str(self.atlas["arm_labels"].get(arm, arm))

    def variant_title(self, variant: str) -> str:
        return str(self.atlas["variant_titles"].get(variant, variant))


# ---- formatting ----------------------------------------------------------------------------
def _fixed(value: float, digits: int) -> str:
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text  # never "-0.000"


def metres(value: float) -> str:
    """Three decimals, no unit: ``0.158``. Negative zero prints as ``0.000``."""
    return _fixed(value, 3)


def metres_text(value: float | None) -> str | None:
    """``0.158 m``, or ``None`` when the value is missing (the caller shows Not recorded)."""
    return None if value is None else f"{metres(value)} m"


def range_text(low: float, high: float) -> str:
    """``0.149 to 0.170 m``."""
    return TEXT["rangeTemplate"].format(low=metres(low), high=metres(high))


def delay_ms(seconds: float) -> int:
    """Whole milliseconds, rounded the same way in both apps."""
    return int(round(float(seconds) * 1000.0))


def delay_text(seconds: float) -> str:
    return f"{delay_ms(seconds)} ms"


def count_text(count: int, total: int) -> str:
    return TEXT["seedsOf"].format(count=count, total=total)


def percent(probability: float) -> int:
    """Percent for titles, ``int(p * 100)`` exactly as the macOS app does it."""
    return int(float(probability) * 100)


def panel_title(probability: float) -> str:
    return TEXT["panelNoLoss"] if float(probability) == 0.0 else TEXT["panelLoss"].format(percent=percent(probability))


def sentence(text: str) -> str:
    """A raw data string with a leading capital; the words themselves are unchanged."""
    return text[:1].upper() + text[1:]


# ---- chart scale -----------------------------------------------------------------------------
def nice_step(span: float, target: int = 4) -> float:
    """A round step (1, 2, or 5 times a power of ten) giving about ``target`` intervals."""
    raw = max(float(span), 1e-9) / target
    power = 10 ** math.floor(math.log10(raw))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= raw - 1e-12)


def nice_ceiling(value: float) -> float:
    """The smallest multiple of the round step that holds ``value`` (1.48 becomes 1.5)."""
    if value <= 0:
        return 1.0
    step = nice_step(value)
    return round(step * math.ceil(value / step - 1e-9), 10)


def y_ticks(y_max: float) -> list[float]:
    step = nice_step(y_max)
    count = int(round(y_max / step))
    return [round(step * index, 10) for index in range(count + 1)]


def tick_text(value: float, ticks: Sequence[float]) -> str:
    """Tick labels share one number of decimals (``0.0``, ``0.5``, ``1.0``)."""
    digits = max((len(f"{tick:g}".split(".")[1]) if "." in f"{tick:g}" else 0) for tick in ticks) if ticks else 0
    return _fixed(value, digits)


@dataclass(frozen=True)
class Scale:
    """Linear map from data to pixels; y grows upward in data and downward on screen."""

    x0: float
    x1: float
    y0: float  # pixel y of the data minimum (bottom)
    y1: float  # pixel y of the data maximum (top)
    x_max_ms: float
    y_max: float

    def x(self, ms: float) -> float:
        return self.x0 + (self.x1 - self.x0) * (float(ms) / self.x_max_ms)

    def y(self, value: float) -> float:
        return self.y0 - (self.y0 - self.y1) * (float(value) / self.y_max)


def band_polygon(scale: Scale, delays_s: Sequence[float], low: Sequence[float], high: Sequence[float]) -> list[float]:
    """Flat coordinates of the interval band: along the upper edge, then back along the lower edge."""
    upper = [(scale.x(delay_ms(delay)), scale.y(value)) for delay, value in zip(delays_s, high)]
    lower = [(scale.x(delay_ms(delay)), scale.y(value)) for delay, value in zip(delays_s, low)]
    return [coordinate for point in upper + lower[::-1] for coordinate in point]


def nearest(points: Iterable[tuple[float, float, Any]], x: float, y: float, radius: float) -> Any:
    """The key of the point closest to (x, y) within ``radius`` pixels, or ``None``. Ties keep the first point."""
    best, best_distance = None, float(radius)
    for px, py, key in points:
        distance = math.hypot(px - x, py - y)
        if distance <= best_distance and (best is None or distance < best_distance):
            best, best_distance = key, distance
    return best


def rects_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def free_slot(candidates: Sequence[Sequence[float]], obstacles: Iterable[Sequence[float]]) -> int | None:
    """Index of the first candidate box that touches no obstacle box, or ``None`` when every candidate collides."""
    obstacles = list(obstacles)
    for index, box in enumerate(candidates):
        if not any(rects_overlap(box, obstacle) for obstacle in obstacles):
            return index
    return None


# ---- delay-and-loss view -------------------------------------------------------------------
def panels(loss_type: str) -> list[tuple[str, float]]:
    """(loss model, probability) for the four panels; the no-loss panel is always the independent:0 run."""
    return [("independent", 0.0)] + [(loss_type, level) for level in LOSS_LEVELS[1:]]


def hover_text(data: ResultsData, point: Point) -> str:
    series = data.series(point.motion, point.loss_model, point.loss_probability, point.arm)
    if series is None:
        return TEXT["notComputed"]
    i = point.index
    return TEXT["hoverTemplate"].format(
        arm=ARM_NAMES[point.arm], delay=delay_ms(data.delays_s[i]), median=metres(series["p95_median_m"][i]),
        low=metres(series["p95_ci_low_m"][i]), high=metres(series["p95_ci_high_m"][i]),
        failing=series["fail_count"][i], total=series["n_seeds"][i])


def point_subtitle(data: ResultsData, point: Point) -> str:
    return TEXT["pointSubtitle"].format(path=PATH_TITLES[point.motion],
                                        loss=data.loss_title(loss_key(point.loss_model, point.loss_probability)),
                                        delay=delay_ms(data.delays_s[point.index]))


def compare_lines(data: ResultsData, point: Point) -> list[str]:
    """The "Compared with the other tracker" lines. Paired differences are basic minus recovery,
    so a positive median means recovery had the lower error."""
    series = data.series(point.motion, point.loss_model, point.loss_probability, point.arm)
    if series is None:
        return [TEXT["notComputed"]]
    i = point.index
    total, tied = series["n_seeds"][i], series["tie_count"][i]
    if tied == total:
        identical = point.loss_probability == 0.0 and data.delays_s[i] == 0.0
        return [TEXT["identicalNoDelay"] if identical else TEXT["tiedEverySeed"]]
    median = float(series["paired_diff_median_m"][i])
    if median > 0:
        first = TEXT["recoveryLower"].format(value=metres(median))
    elif median < 0:
        first = TEXT["recoveryHigher"].format(value=metres(-median))
    else:
        first = TEXT["recoveryNoDifference"]
    return [
        first,
        TEXT["pairedInterval"].format(low=metres(series["paired_diff_ci_low_m"][i]), high=metres(series["paired_diff_ci_high_m"][i])),
        TEXT["pairedCounts"].format(better=series["recovery_better_count"][i], worse=series["recovery_worse_count"][i],
                                    tied=tied, total=total),
    ]


def finding_for_path(data: ResultsData, motion: str) -> Mapping[str, Any]:
    """The README finding shown above the charts: the headline for the straight line, the loss finding for the turn."""
    findings = data.s6["findings"]
    return findings[-1] if motion == "constant_turn" else findings[0]


def move_point(point: Point, key: str, delay_count: int, arms: Sequence[str]) -> Point:
    """Arrow keys inside one panel: Left and Right step the delay, Up and Down switch tracker. Edges clamp, never wrap."""
    index, arm = point.index, point.arm
    if key == "Left":
        index = max(0, index - 1)
    elif key == "Right":
        index = min(delay_count - 1, index + 1)
    elif key in ("Up", "Down"):
        order = list(arms)
        position = order.index(arm) if arm in order else 0
        arm = order[max(0, position - 1) if key == "Up" else min(len(order) - 1, position + 1)]
    return Point(point.motion, point.loss_model, point.loss_probability, arm, index)


def step_panel(current: int, delta: int, count: int) -> int | None:
    """Next panel index for Tab or ] (delta +1) and Shift+Tab or [ (delta -1); ``None`` past either end."""
    target = current + delta
    return target if 0 <= target < count else None


# ---- atlas ---------------------------------------------------------------------------------
def entry_stop(entry: Mapping[str, Any] | None) -> int:
    """Ramp stop for a matrix cell: 0 none (surface), 1 for 0-25 ms, 2 for 50-100 ms, 3 for 150-200 ms, 4 for censored."""
    if entry is None or entry["kind"] == "fails_at_zero":
        return 0
    if entry["kind"] == "censored":
        return 4
    value = float(entry["value_s"])
    if value <= 0.025 + 1e-9:
        return 1
    if value <= 0.1 + 1e-9:
        return 2
    return 3


def entry_cell_text(entry: Mapping[str, Any] | None) -> str:
    """Short text inside a matrix cell."""
    if entry is None:
        return TEXT["notComputed"]
    if entry["kind"] == "censored":
        return ATLAS_TEXT["censored"]
    if entry["kind"] == "fails_at_zero":
        return ATLAS_TEXT["failsAtZero"]
    return delay_text(entry["value_s"])


def tolerated_text(entry: Mapping[str, Any] | None) -> str | None:
    """The inspector's "Most delay tolerated" value."""
    if entry is None:
        return None
    if entry["kind"] == "fails_at_zero":
        return ATLAS_TEXT["failsAtZeroLong"]
    return entry_cell_text(entry)


def _end_words(end: float | str) -> str:
    if end == BELOW_ZERO:
        return ATLAS_TEXT["intervalBelowZero"]
    if end == ABOVE_LAST:
        return ATLAS_TEXT["intervalAbove"]
    return str(delay_ms(float(end)))


def interval_text(interval: Sequence[Any] | None) -> str | None:
    """Atlas interval in ms: ``100 to 200 ms``, ``Below 0 to 100 ms``, ``Above 300 ms`` (both ends above)."""
    if not interval or len(interval) != 2:
        return None
    low, high = interval_end(interval[0]), interval_end(interval[1])
    if low == high:
        return sentence(f"{_end_words(low)} ms")
    return sentence(ATLAS_TEXT["intervalRange"].format(low=_end_words(low), high=_end_words(high)))


def cause_text(mode: str | None) -> str:
    return ATLAS_TEXT[FAILURE_MODES[mode]] if mode in FAILURE_MODES else ATLAS_TEXT["causeNone"]


def arm_help(arm: str) -> str | None:
    """``atlas.armHelp<Arm>`` from UI_TEXT (``recovery_gated`` reads ``armHelpRecoveryGated``)."""
    key = "armHelp" + "".join(part[:1].upper() + part[1:] for part in arm.split("_"))
    return ATLAS_TEXT.get(key)


def most_delay_text(data: ResultsData, cell: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """(classification, delay text) for the Most delay column."""
    if cell is None:
        return TEXT["notComputed"], None
    if cell["kind"] == "none":
        return ATLAS_TEXT["nothingPasses"], None
    arms = list(cell["arms"])
    if cell["kind"] == "winner":
        return data.arm_label(arms[0]), cell["delay_text"]
    if len(arms) > TIE_LISTED_MAX:
        return ATLAS_TEXT["tieCount"].format(count=len(arms)), cell["delay_text"]
    return ATLAS_TEXT["tieListed"].format(arms=", ".join(data.arm_label(arm) for arm in arms)), cell["delay_text"]


def summary_text(stratum: Mapping[str, Any]) -> str:
    counts = stratum["counts"]
    return ATLAS_TEXT["summaryLine"].format(tie=counts["tie"], winner=counts["winner"], none=counts["none"], cells=counts["cells"])


def rule_text(data: ResultsData) -> str:
    return ATLAS_TEXT["cellRule"].format(status=data.atlas["failure_rule"]["cell_rule_status"])


def level_loss_key(loss_type: str, level: float) -> str:
    """Loss condition for the atlas controls: level 0 ("None") is independent:0 for both loss types."""
    return loss_key("independent", 0.0) if float(level) == 0.0 else loss_key(loss_type, level)


def move_cell(row: int, column: int, key: str, rows: int, columns: int) -> tuple[int, int]:
    """Arrow-key movement in the matrix; edges clamp."""
    if key == "Up":
        row -= 1
    elif key == "Down":
        row += 1
    elif key == "Left":
        column -= 1
    elif key == "Right":
        column += 1
    elif key == "Home":
        column = 0
    elif key == "End":
        column = columns - 1
    return max(0, min(rows - 1, row)), max(0, min(columns - 1, column))


def diff_rows(entry: Mapping[str, Any] | None) -> list[tuple[str, str, str]]:
    """(delay, median, interval) rows of the "Difference from basic tracker" table, in metres."""
    if not entry:
        return []
    rows = []
    for row in entry["improvement_vs_naive"]:
        low, high = row["interval_m"]
        rows.append((str(delay_ms(row["delay_s"])), metres(row["median_p95_diff_m"]), f"{metres(low)} to {metres(high)}"))
    return rows


# ---- screenshot selection spec ---------------------------------------------------------------
def parse_select_spec(text: str) -> dict[str, str]:
    """``key=value`` pairs separated by commas, for the ``--results-select`` screenshot flag."""
    result: dict[str, str] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value, got {part!r}")
        key, value = (piece.strip() for piece in part.split("=", 1))
        if not key or not value:
            raise ValueError(f"expected key=value, got {part!r}")
        result[key] = value
    return result


def _probability(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"loss level must be a number, got {value!r}") from exc
    if number not in LOSS_LEVELS:
        raise ValueError(f"loss level must be one of {', '.join(f'{level:g}' for level in LOSS_LEVELS)}")
    return number


def _loss_type(value: str) -> str:
    aliases = {"independent": "independent", "random": "independent", "burst": "burst", "bursty": "burst"}
    if value not in aliases:
        raise ValueError(f"loss must be independent or burst, got {value!r}")
    return aliases[value]


def delay_selection(data: ResultsData, spec: Mapping[str, str]) -> tuple[str, str, Point | None]:
    """(motion, loss type, point) for the delay view from a parsed select spec."""
    motion = PATHS.get(spec.get("path", "straight"), spec.get("path", ""))
    if motion not in PATH_TITLES:
        raise ValueError("path must be straight or turn")
    loss_type = _loss_type(spec.get("loss", "independent"))
    if "arm" not in spec:
        return motion, loss_type, None
    arm = spec["arm"]
    if arm not in ARM_NAMES:
        raise ValueError("arm must be naive or recovery")
    level = _probability(spec.get("level", "0"))
    model = "independent" if level == 0.0 else loss_type
    seconds = float(spec.get("delay", "0"))
    matches = [index for index, delay in enumerate(data.delays_s) if delay_ms(delay) == delay_ms(seconds)]
    if not matches:
        raise ValueError(f"delay {seconds:g} s was not tested")
    return motion, loss_type, Point(motion, model, level, arm, matches[0])


def atlas_selection(data: ResultsData, spec: Mapping[str, str]) -> tuple[str, str, float, str | None, str | None]:
    """(stratum, loss type, level, variant, arm) for the atlas from a parsed select spec."""
    stratum = spec.get("stratum", data.stratum_order[0])
    if stratum not in data.strata:
        raise ValueError(f"unknown stratum {stratum!r}")
    loss_type = _loss_type(spec.get("loss", "independent"))
    level = _probability(spec.get("level", "0"))
    variant, arm = spec.get("variant"), spec.get("arm")
    if variant is not None and variant not in data.strata[stratum]["variants"]:
        raise ValueError(f"variant {variant!r} is not in {stratum}")
    if arm is not None and arm not in data.strata[stratum]["arms"]:
        raise ValueError(f"arm {arm!r} is not in {stratum}")
    return stratum, loss_type, level, variant, arm


__all__ = [
    "ARM_NAMES", "FixtureError", "LOSS_LEVELS", "LOSS_TYPES", "PATHS", "PATH_TITLES", "Point", "ResultsData", "Scale",
    "X_TICKS_MS", "arm_help", "atlas_selection", "band_polygon", "cause_text", "compare_lines", "count_text",
    "delay_ms", "delay_selection", "delay_text", "diff_rows", "entry_cell_text", "entry_stop", "finding_for_path",
    "fixture_path", "free_slot", "hover_text", "interval_end", "interval_text", "level_loss_key", "load_fixture",
    "loss_key", "metres", "metres_text", "most_delay_text", "move_cell", "move_point", "nearest", "nice_ceiling",
    "nice_step", "panel_title", "panels", "parse_fixture", "parse_select_spec", "percent", "point_subtitle",
    "range_text", "rects_overlap", "rule_text", "sentence", "step_panel", "summary_text", "tick_text",
    "tolerated_text", "y_ticks",
]
