"""Pure checks for desktop/results_data.py (no Tk).

Expected values come from the raw fixture JSON read here directly, from hand
calculations, or from the formatting rules both apps share, not from the
module under test, so a wrong lookup or formatter fails instead of agreeing
with itself.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from desktop import results_data as rd
from desktop.design_tokens import UI_TEXT


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "desktop" / "results_fixture.json"
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
DATA = rd.load_fixture(FIXTURE)
TEXT, ATLAS = UI_TEXT["results"], UI_TEXT["atlas"]


def _raw_series(motion, model, probability, arm):
    return next(s for s in RAW["s6"]["series"] if (s["motion"], s["loss_model"], s["loss_probability"], s["arm"])
                == (motion, model, probability, arm))


def _broken(mutate):
    obj = copy.deepcopy(RAW)
    mutate(obj)
    return obj


# ---- loader strictness -------------------------------------------------------------------------
def test_real_fixture_loads_and_indexes_every_series_and_entry():
    assert DATA.delays_s == [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    assert len(DATA._series) == len(RAW["s6"]["series"]) == 28
    assert len(DATA._entries) == len(RAW["atlas"]["entries"])
    assert DATA.stratum_order[0] == "central"


@pytest.mark.parametrize("mutate, words", [
    (lambda o: o.update(schema="zephyr-results-fixture-0"), "schema"),
    (lambda o: o.update(simulated=False), "simulated"),
    (lambda o: o["s6"]["series"][0]["p95_median_m"].pop(), "values for 7 delays"),
    (lambda o: o["s6"]["series"][0]["fail_count"].__setitem__(0, True), "whole number"),
    (lambda o: o["s6"]["series"][0]["p95_ci_low_m"].__setitem__(0, "0.1"), "finite number"),
    (lambda o: o["s6"]["series"][0]["label"].__setitem__(0, "pass"), "label"),
    (lambda o: o["atlas"]["entries"][0]["interval_s"].__setitem__(0, "<1"), "finite number"),
    (lambda o: o["atlas"]["entries"][0].update(kind="maybe"), "kind"),
    (lambda o: o["atlas"]["entries"].append(copy.deepcopy(o["atlas"]["entries"][0])), "repeats"),
    (lambda o: o["atlas"]["cells"][0].update(kind="winner"), "arms does not match"),
    (lambda o: o["atlas"]["loss_titles"].pop("burst:0.5"), "no title"),
    (lambda o: o["s6"].pop("limits"), "missing limits"),
    (lambda o: o["s6"]["source"].update(sha256="abc"), "64 hex"),
])
def test_loader_rejects_each_contract_break(mutate, words):
    with pytest.raises(rd.FixtureError, match=words):
        rd.parse_fixture(_broken(mutate))


def test_loader_rejects_nan_and_reports_missing_or_bad_files(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace('"failure_p95_m": 0.25', '"failure_p95_m": NaN', 1)
    path = tmp_path / "results_fixture.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(rd.FixtureError, match="NaN"):
        rd.load_fixture(path)
    with pytest.raises(rd.FixtureError, match="was not found"):
        rd.load_fixture(tmp_path / "missing.json")
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(rd.FixtureError, match="not valid JSON"):
        rd.load_fixture(path)


def test_fixture_path_prefers_the_frozen_bundle(tmp_path, monkeypatch):
    bundled = tmp_path / "_internal" / "desktop" / "results_fixture.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rd.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert rd.fixture_path() == bundled
    monkeypatch.delattr(rd.sys, "_MEIPASS")
    assert rd.fixture_path() == FIXTURE


def test_interval_ends_are_numbers_or_the_two_markers():
    assert rd.interval_end("<0") == "<0" and rd.interval_end(">0.3") == ">0.3"
    assert rd.interval_end(0.15) == 0.15 and rd.interval_end(0) == 0.0
    for bad in ("<1", None, True, float("nan")):
        with pytest.raises(rd.FixtureError):
            rd.interval_end(bad)


# ---- formatting ---------------------------------------------------------------------------------
def test_metres_use_three_decimals_and_never_print_negative_zero():
    assert rd.metres(0.15835) == "0.158"
    assert rd.metres(-0.0) == "0.000" and rd.metres(-0.0004) == "0.000"
    assert rd.metres(-0.024) == "-0.024"
    assert rd.metres_text(0.2) == "0.200 m" and rd.metres_text(None) is None
    assert rd.range_text(0.149, 0.17) == "0.149 to 0.170 m"


def test_delays_are_whole_milliseconds_and_counts_read_x_of_n():
    assert [rd.delay_text(value) for value in (0.0, 0.025, 0.3)] == ["0 ms", "25 ms", "300 ms"]
    assert rd.delay_ms(0.15) == 150  # 0.15 * 1000 is 149.99999999999997 in floating point
    assert rd.count_text(0, 50) == "0 of 50"


def test_panel_titles_use_int_percent():
    assert rd.panel_title(0.0) == "No loss"
    assert [rd.panel_title(level) for level in (0.1, 0.3, 0.5)] == ["10 % loss", "30 % loss", "50 % loss"]
    assert rd.percent(0.29) == 28, "int(p * 100) truncates, the same rule as the macOS app"


# ---- chart scale and geometry ------------------------------------------------------------------
@pytest.mark.parametrize("value, top, step, ticks", [
    (0.389, 0.4, 0.1, [0.0, 0.1, 0.2, 0.3, 0.4]),   # the cases written in the macOS NiceScale comment
    (0.606, 0.7, 0.1, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    (1.484, 1.5, 0.5, [0.0, 0.5, 1.0, 1.5]),
    (0.2, 0.2, 0.05, [0.0, 0.05, 0.1, 0.15, 0.2]),
    (0.0, 1.0, 1.0, [0.0, 1.0]),                      # nothing to scale: a plain 0 to 1 axis
])
def test_nice_scale_by_hand(value, top, step, ticks):
    scale = rd.NiceScale(value)
    assert math.isclose(scale.top, top) and math.isclose(scale.step, step)
    assert scale.ticks == pytest.approx(ticks)
    assert all(tick <= scale.top + 1e-9 for tick in scale.ticks) and scale.top >= value


def test_nice_scale_labels_share_the_decimals_that_show_every_tick():
    assert [rd.NiceScale(1.484).label(t) for t in rd.NiceScale(1.484).ticks] == ["0.0", "0.5", "1.0", "1.5"]
    assert [rd.NiceScale(0.2).label(t) for t in rd.NiceScale(0.2).ticks] == ["0.00", "0.05", "0.10", "0.15", "0.20"]
    assert rd.NiceScale(0.4).label(-0.0) == "0.0"


@pytest.mark.parametrize("value, ceiling", [(1.48, 1.5), (0.36, 0.4), (0.5, 0.5), (0.0, 1.0), (7.2, 7.5)])
def test_nice_ceiling(value, ceiling):
    assert math.isclose(rd.nice_ceiling(value), ceiling)


@pytest.mark.parametrize("motion, loss, top", [
    ("constant_velocity", "independent", 0.4),
    ("constant_velocity", "burst", 0.5),
    ("constant_turn", "independent", 0.7),
    ("constant_turn", "burst", 1.5),
])
def test_y_scale_follows_the_path_and_loss_type_on_screen(motion, loss, top):
    """The ceiling is chosen from the raw interval tops of the four panels shown, not from the worst case in the fixture."""
    shown = [(model, level) for model, level in [("independent", 0.0)] + [(loss, p) for p in (0.1, 0.3, 0.5)]]
    highs = [max(series["p95_ci_high_m"]) for series in RAW["s6"]["series"]
             if series["motion"] == motion and (series["loss_model"], series["loss_probability"]) in shown]
    assert len(highs) == 8  # four panels, two trackers each
    scale = DATA.error_scale(motion, loss)
    assert scale.top >= max(highs) and math.isclose(scale.top, top), (max(highs), scale.top)
    assert scale.top < 2 * max(highs), "no more headroom than the nice ceiling needs"


def test_the_straight_path_is_not_squashed_by_the_turn_scale():
    straight, turn = DATA.error_scale("constant_velocity", "independent"), DATA.error_scale("constant_turn", "burst")
    assert straight.top < turn.top / 3


def test_scale_and_band_polygon_by_hand():
    scale = rd.Scale(x0=10, x1=310, y0=200, y1=0, x_max_ms=300, y_max=1.0)
    assert scale.x(150) == 160 and scale.y(0.5) == 100 and scale.y(0.0) == 200
    band = rd.band_polygon(scale, [0.0, 0.3], [0.1, 0.2], [0.3, 0.4])
    # Upper edge left to right, then the lower edge right to left.
    assert band == pytest.approx([10, 140, 310, 120, 310, 160, 10, 180])


def test_nearest_point_within_the_hit_radius():
    points = [(0, 0, "a"), (10, 0, "b"), (30, 0, "c")]
    assert rd.nearest(points, 4, 0, 14) == "a"
    assert rd.nearest(points, 9, 1, 14) == "b"
    assert rd.nearest(points, 60, 0, 14) is None
    assert rd.nearest(points, 5, 0, 14) == "a", "equal distance keeps the first point"


def test_free_slot_skips_blocked_boxes_and_reports_none_when_all_collide():
    candidates = [(0, 0, 10, 10), (20, 0, 30, 10)]
    assert rd.free_slot(candidates, [(5, 5, 6, 6)]) == 1
    assert rd.free_slot(candidates, []) == 0
    assert rd.free_slot(candidates, [(0, 0, 40, 40)]) is None


def test_selection_moves_clamp_at_the_edges():
    point = rd.Point("constant_velocity", "independent", 0.0, "naive", 0)
    assert rd.move_point(point, "Left", 7, ("naive", "recovery")).index == 0
    assert rd.move_point(point, "Up", 7, ("naive", "recovery")).arm == "naive"
    assert rd.move_point(point, "Down", 7, ("naive", "recovery")).arm == "recovery"
    end = rd.Point("constant_velocity", "independent", 0.0, "recovery", 6)
    assert rd.move_point(end, "Right", 7, ("naive", "recovery")).index == 6
    assert rd.step_panel(0, -1, 4) is None and rd.step_panel(3, 1, 4) is None and rd.step_panel(1, 1, 4) == 2
    assert rd.move_cell(0, 0, "Up", 9, 5) == (0, 0)
    assert rd.move_cell(8, 4, "Right", 9, 5) == (8, 4)
    assert rd.move_cell(3, 2, "Home", 9, 5) == (3, 0) and rd.move_cell(3, 2, "End", 9, 5) == (3, 4)


def test_bursty_panels_reuse_the_no_loss_run():
    assert rd.panels("burst") == [("independent", 0.0), ("burst", 0.1), ("burst", 0.3), ("burst", 0.5)]
    assert rd.level_loss_key("burst", 0.0) == "independent:0" and rd.level_loss_key("burst", 0.3) == "burst:0.3"


# ---- delay-and-loss text ---------------------------------------------------------------------
def test_hover_and_subtitle_text_for_a_known_point():
    raw = _raw_series("constant_turn", "independent", 0.3, "recovery")
    point = rd.Point("constant_turn", "independent", 0.3, "recovery", 5)
    expected = (f"Recovery (capture time), 200 ms: {raw['p95_median_m'][5]:.3f} m ({raw['p95_ci_low_m'][5]:.3f} to "
                f"{raw['p95_ci_high_m'][5]:.3f}), {raw['fail_count'][5]} of {raw['n_seeds'][5]} fail")
    assert rd.hover_text(DATA, point) == expected
    assert rd.point_subtitle(DATA, point) == "1 m turn, 30 % random, 200 ms"


def test_compare_wording_follows_the_sign_of_basic_minus_recovery():
    seen = set()
    for raw in RAW["s6"]["series"]:
        for i, median in enumerate(raw["paired_diff_median_m"]):
            point = rd.Point(raw["motion"], raw["loss_model"], raw["loss_probability"], raw["arm"], i)
            first = rd.compare_lines(DATA, point)[0]
            if raw["tie_count"][i] == raw["n_seeds"][i]:
                seen.add("tie")
                assert first in (TEXT["identicalNoDelay"], TEXT["tiedEverySeed"])
            elif median > 0:
                seen.add("lower")
                assert first == TEXT["recoveryLower"].format(value=f"{median:.3f}")
            elif median < 0:
                seen.add("higher")
                assert first == TEXT["recoveryHigher"].format(value=f"{-median:.3f}")
    assert {"tie", "lower", "higher"} <= seen
    no_delay = rd.Point("constant_velocity", "independent", 0.0, "naive", 0)
    assert rd.compare_lines(DATA, no_delay) == [TEXT["identicalNoDelay"]]


def test_lead_finding_is_the_second_sentence_for_delay_and_the_first_for_the_atlas():
    findings = RAW["s6"]["findings"]
    assert rd.LEAD_FINDING == {"delay": 1, "atlas": 0}
    lead, more = rd.split_findings(findings, rd.LEAD_FINDING["delay"])
    assert lead == findings[1] and more == [findings[0], findings[2]]
    lead, more = rd.split_findings(RAW["atlas"]["findings"], rd.LEAD_FINDING["atlas"])
    assert lead == RAW["atlas"]["findings"][0] and more == []


def test_split_findings_edges():
    items = [{"text": "a", "source": "README.md"}, {"text": "b", "source": "docs/RESULTS.md"}]
    assert rd.split_findings([], 1) == (None, [])
    assert rd.split_findings(items, 7) == (items[0], [items[1]]), "a lead past the end falls back to the first"
    assert rd.split_findings(items, -1)[0] == items[0]
    assert rd.split_findings(items, 1) == (items[1], [items[0]])
    assert rd.sources_text(items + items) == "README.md, docs/RESULTS.md"


def test_display_text_strips_backticks_but_the_fixture_keeps_them():
    raw = RAW["s6"]["findings"][0]["text"]
    assert "`runs/s6-recovery`" in raw
    assert rd.display_text(raw) == raw.replace("`", "") and "`" not in rd.display_text(raw)
    assert "(runs/s6-recovery)" in rd.display_text(raw)
    assert rd.display_text("no code here") == "no code here"


# ---- atlas --------------------------------------------------------------------------------------
@pytest.mark.parametrize("entry, stop, text", [
    ({"kind": "value", "value_s": 0.0}, 1, "0 ms"),
    ({"kind": "value", "value_s": 0.025}, 1, "25 ms"),
    ({"kind": "value", "value_s": 0.05}, 2, "50 ms"),
    ({"kind": "value", "value_s": 0.1}, 2, "100 ms"),
    ({"kind": "value", "value_s": 0.15}, 3, "150 ms"),
    ({"kind": "value", "value_s": 0.2}, 3, "200 ms"),
    ({"kind": "censored", "value_s": None}, 4, ">300 ms"),
    ({"kind": "fails_at_zero", "value_s": None}, 0, "Fails at 0"),
    (None, 0, "Not computed"),
])
def test_atlas_binning_and_cell_text_per_kind(entry, stop, text):
    assert rd.entry_stop(entry) == stop
    assert rd.entry_cell_text(entry) == text


def test_tolerated_text_never_shows_zero_for_missing():
    assert rd.tolerated_text(None) is None
    assert rd.tolerated_text({"kind": "fails_at_zero", "value_s": None}) == "Fails at 0 ms"
    assert rd.tolerated_text({"kind": "censored", "value_s": None}) == ">300 ms"
    assert rd.tolerated_text({"kind": "value", "value_s": 0.2}) == "200 ms"


@pytest.mark.parametrize("interval, text", [
    ([0.1, 0.2], "100 ms to 200 ms"),
    (["<0", 0.2], "Below 0 ms to 200 ms"),
    (["<0", 0.1], "Below 0 ms to 100 ms"),
    ([0.2, ">0.3"], "200 ms to above 300 ms"),
    ([">0.3", ">0.3"], "Above 300 ms"),
    (["<0", "<0"], "Below 0 ms"),
    ([0.0, 0.025], "0 ms to 25 ms"),
])
def test_interval_text(interval, text):
    assert rd.interval_text(interval) == text


def test_failure_causes_and_arm_help_cover_every_value_in_the_fixture():
    modes = {entry["failure_mode"] for entry in RAW["atlas"]["entries"]}
    assert {rd.cause_text(mode) for mode in modes} <= {ATLAS[key] for key in ("causeLag", "causeGap", "causeNoise", "causeModel",
                                                                               "causeNone")}
    assert rd.cause_text(None) == "No failure seen" and rd.cause_text("gap-dominated") == "Lost packets"
    for arm in RAW["atlas"]["arm_labels"]:
        assert rd.arm_help(arm), arm


def test_loss_titles_come_from_the_fixture():
    for key, title in RAW["atlas"]["loss_titles"].items():
        assert DATA.loss_title(key) == title
    assert DATA.loss_title(rd.loss_key("independent", 0.0)) == "No loss"


def test_most_delay_text_for_winner_tie_and_none():
    cells = RAW["atlas"]["cells"]
    winner = next(c for c in cells if c["kind"] == "winner")
    none = next(c for c in cells if c["kind"] == "none")
    assert rd.most_delay_text(DATA, winner) == (RAW["atlas"]["arm_labels"][winner["arms"][0]], winner["delay_text"])
    assert rd.most_delay_text(DATA, none) == (ATLAS["nothingPasses"], None)
    assert rd.most_delay_text(DATA, None)[0] == TEXT["notComputed"]


def test_tie_limit_matches_the_atlas_figure():
    figure = (ROOT / "tools" / "make_s6_atlas_figure.py").read_text(encoding="utf-8")
    assert f"TIE_LISTED_MAX = {rd.TIE_LISTED_MAX}" in figure and rd.TIE_LISTED_MAX == 3


@pytest.mark.parametrize("count", [2, 3, 4])
def test_ties_list_up_to_three_trackers_and_count_more(count):
    """Pick a real tie with that many trackers; the fixture has ties of 2 to 7."""
    labels = RAW["atlas"]["arm_labels"]
    tie = next(c for c in RAW["atlas"]["cells"] if c["kind"] == "tie" and len(c["arms"]) == count)
    label, delay = rd.most_delay_text(DATA, tie)
    assert delay == tie["delay_text"]
    if count <= 3:
        assert label == "Tie: " + ", ".join(labels[arm] for arm in tie["arms"])
    else:
        assert label == f"Tie: {count} trackers"


def test_a_tie_of_two_names_both_trackers():
    tie = next(c for c in RAW["atlas"]["cells"] if c["kind"] == "tie" and len(c["arms"]) == 2)
    first, second = (RAW["atlas"]["arm_labels"][arm] for arm in tie["arms"])
    assert rd.most_delay_text(DATA, tie)[0] == f"Tie: {first}, {second}"


def test_failure_line_label_uses_three_decimals():
    assert RAW["s6"]["failure_p95_m"] == 0.25
    assert rd.failure_label(RAW["s6"]["failure_p95_m"]) == "Failure line 0.250 m"
    assert rd.failure_label(0.3) == "Failure line 0.300 m" and rd.failure_label(-0.0) == "Failure line 0.000 m"


def test_wrap_lines_breaks_at_spaces_and_never_drops_a_word():
    measure = lambda text: 10.0 * len(text)  # noqa: E731 - ten pixels per character
    assert rd.wrap_lines("Tie: Recovery gated, CT gated, IMM gated", 200, measure) == ["Tie: Recovery gated,", "CT gated, IMM gated"]
    assert rd.wrap_lines("Nothing passes", 400, measure) == ["Nothing passes"]
    assert rd.wrap_lines("Extraordinarily", 50, measure) == ["Extraordinarily"], "a long word stays whole, never cut"
    assert rd.wrap_lines("", 100, measure) == []
    text = "Tie: Adaptive, Recovery gated, Retro"
    assert " ".join(rd.wrap_lines(text, 120, measure)) == text


def test_winner_bar_follows_the_rounded_corners():
    points = rd.winner_bar_points(0.0, 100.0, 36.0, 4.0, 3.0)
    xs, ys = points[0::2], points[1::2]
    assert max(ys) == 36.0 and min(ys) == 33.0, "the bar is 3 tall and sits on the bottom edge"
    assert min(xs) >= 0.0 and max(xs) <= 100.0
    # At the bottom edge a radius-4 corner has pulled in by 4 px; at the top of the bar by 4 - sqrt(16 - 1) = 0.127 px.
    bottom_xs = sorted(x for x, y in zip(xs, ys) if y == 36.0)
    top_xs = sorted(x for x, y in zip(xs, ys) if y == 33.0)
    assert bottom_xs[0] == pytest.approx(4.0) and bottom_xs[-1] == pytest.approx(96.0)
    assert top_xs[0] == pytest.approx(4.0 - math.sqrt(16.0 - 1.0)) and top_xs[-1] == pytest.approx(100.0 - 4.0 + math.sqrt(15.0))
    square = rd.winner_bar_points(0.0, 100.0, 36.0, 0.0, 3.0)
    assert min(square[0::2]) == 0.0 and max(square[0::2]) == 100.0


def test_scale_caption_names_the_top_of_the_axis_in_metres():
    assert rd.scale_caption(0.7) == "Error axis 0 to 0.700 m for this path and loss type"
    assert rd.scale_caption(DATA.error_scale("constant_turn", "burst").top) == "Error axis 0 to 1.500 m for this path and loss type"


def test_interval_caption_explains_a_fails_at_zero_cell_with_a_wide_interval():
    cells = [e for e in RAW["atlas"]["entries"] if e["kind"] == "fails_at_zero"]
    wide = next(e for e in cells if e["interval_s"][1] not in ("<0", None))
    note = rd.interval_caption(wide, 20)
    assert note.startswith("Fails at 0 ms because at least half its seeds fail, but with 20 seeds per cell")
    assert note.endswith(f"as high as {rd._end_text(wide['interval_s'][1])}.")
    assert rd.interval_caption({"kind": "fails_at_zero", "interval_s": ["<0", "<0"]}, 20) == ATLAS["intervalCaption"]
    assert rd.interval_caption({"kind": "value", "interval_s": [0.1, 0.2]}, 20) == ATLAS["intervalCaption"]
    assert rd.interval_caption(None, 20) == ATLAS["intervalCaption"]


def test_cell_subtitle_keeps_the_conditions_for_their_own_line():
    assert rd.cell_subtitle("Figure eight", "30 % bursty") == "Figure eight, 30 % bursty"


def test_overlap_area_by_hand():
    assert rd.overlap_area((0, 0, 10, 10), (5, 5, 20, 20)) == 25
    assert rd.overlap_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0, "boxes that only touch share no area"
    assert rd.overlap_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


PLOT = (0.0, 0.0, 100.0, 50.0)
SLOTS = [(60, 5, 100, 15), (60, 20, 100, 30), (0, 5, 40, 15), (0, 20, 40, 30)]  # right above, right below, left above, left below


def test_label_slot_takes_the_first_clear_spot_in_order():
    assert rd.label_slot(SLOTS, [], PLOT) == 0
    assert rd.label_slot(SLOTS, [(70, 8, 80, 12)], PLOT) == 1, "above is blocked, so below at the right end"
    assert rd.label_slot(SLOTS, [(70, 8, 80, 12), (70, 22, 80, 26)], PLOT) == 2, "both right spots blocked: left end above"
    assert rd.label_slot(SLOTS, [(70, 8, 80, 12), (70, 22, 80, 26), (10, 8, 20, 12)], PLOT) == 3


def test_label_slot_skips_spots_outside_the_plot():
    high = [(60, -3, 100, 7), (60, 20, 100, 30), (0, -3, 40, 7), (0, 20, 40, 30)]
    assert rd.label_slot(high, [], PLOT) == 1, "the box above hangs off the top of the plot"


def test_label_slot_with_everything_blocked_picks_the_least_covered_spot_and_is_stable():
    blocks = [(60, 5, 100, 15), (60, 20, 90, 30), (0, 5, 40, 15), (0, 20, 30, 30)]  # covers 400, 300, 400, 300 square units
    assert rd.label_slot(SLOTS, blocks, PLOT) == 1, "the first of the two least-covered spots wins the tie"
    assert rd.label_slot(SLOTS, blocks, PLOT) == rd.label_slot(SLOTS, list(reversed(blocks)), PLOT)
    assert rd.label_slot([], [], PLOT) == 0


def test_summary_rule_and_difference_rows():
    counts = RAW["atlas"]["strata"][0]["counts"]
    assert rd.summary_text(DATA.strata["central"]) == (
        f"{counts['tie']} ties, {counts['winner']} clear winners, {counts['none']} cells where nothing passes, of {counts['cells']} cells")
    assert rd.rule_text(DATA).endswith(f"({RAW['atlas']['failure_rule']['cell_rule_status']}).")
    entry = DATA.entry("central", "cv_0.5", "burst:0.3", "ct")
    raw = entry["improvement_vs_naive"][0]
    assert rd.diff_rows(entry)[0] == ("0", f"{raw['median_p95_diff_m']:.3f}",
                                      f"{raw['interval_m'][0]:.3f} to {raw['interval_m'][1]:.3f} m")
    assert rd.diff_rows(DATA.entry("central", "cv_0.5", "burst:0.3", "naive")) == []


# ---- screenshot select specs ---------------------------------------------------------------------
def test_select_specs_resolve_to_real_points_and_cells():
    spec = rd.parse_select_spec("path=turn,loss=independent,level=0.3,arm=recovery,delay=0.2")
    motion, loss_type, point = rd.delay_selection(DATA, spec)
    assert (motion, loss_type) == ("constant_turn", "independent")
    assert point == rd.Point("constant_turn", "independent", 0.3, "recovery", 5)
    spec = rd.parse_select_spec("stratum=central,loss=burst,level=0.3,variant=figure8,arm=ct")
    assert rd.atlas_selection(DATA, spec) == ("central", "burst", 0.3, "figure8", "ct")


@pytest.mark.parametrize("text", ["path", "path=", "=turn"])
def test_select_spec_syntax_errors(text):
    with pytest.raises(ValueError):
        rd.parse_select_spec(text)


@pytest.mark.parametrize("spec", [
    {"path": "loop"}, {"arm": "imm"}, {"arm": "naive", "delay": "0.07"}, {"arm": "naive", "level": "0.2"},
])
def test_delay_select_spec_rejects_untested_values(spec):
    with pytest.raises(ValueError):
        rd.delay_selection(DATA, spec)


def test_atlas_select_spec_rejects_paths_and_trackers_outside_the_stratum():
    with pytest.raises(ValueError):
        rd.atlas_selection(DATA, {"stratum": "noise0", "arm": "imm"})
    with pytest.raises(ValueError):
        rd.atlas_selection(DATA, {"stratum": "noise0", "variant": "cv_2"})
    with pytest.raises(ValueError):
        rd.atlas_selection(DATA, {"stratum": "moon"})


def test_cli_rejects_a_malformed_select_spec_before_opening_any_window(capsys):
    from desktop.windows_preview import main

    with pytest.raises(SystemExit) as caught:
        main(["--open-results", "delay", "--results-select", "path"])
    assert caught.value.code == 2
    assert "--results-select" in capsys.readouterr().err


def test_results_modules_never_import_tk_numpy_or_matplotlib_at_import_time():
    for name in ("results_data", "ui_results", "ui_results_chart", "ui_results_atlas"):
        source = (ROOT / "desktop" / f"{name}.py").read_text(encoding="utf-8")
        assert "numpy" not in source and "matplotlib" not in source, name
        top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not [line for line in top if "tkinter" in line], name


# ---- strict loading: exact keys, field types, kind rules, references ---------------------------------
def _entry_index(**match):
    return next(i for i, e in enumerate(RAW["atlas"]["entries"]) if all(e[k] == v for k, v in match.items()))


@pytest.mark.parametrize("mutate, words", [
    (lambda o: o.update(extra=1), "fixture has unknown extra"),
    (lambda o: o["s6"].update(extra=1), "s6 has unknown extra"),
    (lambda o: o["s6"]["source"].update(extra="x"), "s6.source has unknown extra"),
    (lambda o: o["s6"]["findings"][0].update(extra="x"), r"s6.findings\[0\] has unknown extra"),
    (lambda o: o["s6"]["series"][0].update(extra=[]), r"s6.series\[0\] has unknown extra"),
    (lambda o: o["atlas"].update(extra=1), "atlas has unknown extra"),
    (lambda o: o["atlas"]["failure_rule"].update(extra=1), "failure_rule has unknown extra"),
    (lambda o: o["atlas"]["failure_rule"]["per_seed"].update(extra=1), "per_seed has unknown extra"),
    (lambda o: o["atlas"]["strata"][0].update(extra=1), r"atlas.strata\[0\] has unknown extra"),
    (lambda o: o["atlas"]["strata"][0]["counts"].update(extra=1), "counts has unknown extra"),
    (lambda o: o["atlas"]["entries"][0].update(extra=1), r"atlas.entries\[0\] has unknown extra"),
    (lambda o: o["atlas"]["entries"][_entry_index(arm="recovery", kind="value")]["improvement_vs_naive"][0].update(extra=1),
     "improvement_vs_naive.0. has unknown extra"),
    (lambda o: o["atlas"]["cells"][0].update(extra=1), r"atlas.cells\[0\] has unknown extra"),
    (lambda o: o.pop("generator"), "fixture is missing generator"),
    (lambda o: o["atlas"]["strata"][0]["counts"].pop("survive"), "counts is missing survive"),
    (lambda o: o["atlas"]["entries"][0].pop("loss_model"), "is missing loss_model"),
])
def test_loader_rejects_unknown_and_missing_keys_at_every_level(mutate, words):
    with pytest.raises(rd.FixtureError, match=words):
        rd.parse_fixture(_broken(mutate))


def _first(kind):
    return _entry_index(kind=kind)


@pytest.mark.parametrize("mutate, words", [
    (lambda o: o["atlas"]["entries"][_first("value")].update(value_s=None), "finite number"),
    (lambda o: o["atlas"]["entries"][_first("value")].update(value_s=-0.1), "kind value needs"),
    (lambda o: o["atlas"]["entries"][_first("value")].update(censored=">0.3"), "kind value needs"),
    (lambda o: o["atlas"]["entries"][_first("censored")].update(value_s=0.3), "value_s must be null"),
    (lambda o: o["atlas"]["entries"][_first("censored")].update(censored=None), "censored must be"),
    (lambda o: o["atlas"]["entries"][_first("fails_at_zero")].update(value_s=0.0), "value_s must be null"),
    (lambda o: o["atlas"]["entries"][_first("fails_at_zero")].update(censored=">0.3"), "censored must be null"),
    (lambda o: o["atlas"]["entries"][0].update(arm=3), "non-empty string"),
    (lambda o: o["atlas"]["entries"][0].update(loss_probability="0.1"), "finite number"),
    (lambda o: o["atlas"]["entries"][0].update(interval_s=[0.1]), "two ends"),
    (lambda o: o["atlas"]["entries"][0].update(failure_mode="mystery"), "failure_mode"),
    (lambda o: o["atlas"]["cells"][0].update(delay_text=None), "delay_text is missing"),
    (lambda o: o["atlas"]["cells"][0].update(arms=[]), "arms does not match"),
    (lambda o: o["atlas"]["cells"][0].update(arms="recovery"), "must be a list"),
])
def test_loader_rejects_malformed_atlas_entries_and_cells(mutate, words):
    with pytest.raises(rd.FixtureError, match=words):
        rd.parse_fixture(_broken(mutate))


@pytest.mark.parametrize("mutate, words", [
    (lambda o: o["atlas"]["cells"][0].update(stratum="nowhere"), "unknown stratum"),
    (lambda o: o["atlas"]["cells"][0].update(variant="not_a_path"), "variant 'not_a_path' is not in stratum"),
    (lambda o: o["atlas"]["cells"][0].update(loss_condition="burst:0.9"), "loss_condition 'burst:0.9' is not in stratum"),
    (lambda o: o["atlas"]["entries"][0].update(stratum="nowhere"), "unknown stratum"),
    (lambda o: o["atlas"]["entries"][0].update(arm="imm_gated"), "arm 'imm_gated' is not in stratum"),
    (lambda o: o["atlas"]["entries"][0].update(variant="not_a_path"), "is not in stratum"),
    (lambda o: o["atlas"]["cells"].append(copy.deepcopy(o["atlas"]["cells"][0])), "repeats"),
    (lambda o: o["atlas"]["strata"].append(copy.deepcopy(o["atlas"]["strata"][0])), "repeats the stratum"),
])
def test_loader_rejects_references_that_do_not_exist(mutate, words):
    with pytest.raises(rd.FixtureError, match=words):
        rd.parse_fixture(_broken(mutate))


def test_a_cell_naming_a_tracker_outside_its_stratum_is_rejected():
    index = next(i for i, c in enumerate(RAW["atlas"]["cells"]) if c["stratum"] == "noise0" and c["kind"] == "winner")

    def mutate(obj):
        obj["atlas"]["cells"][index]["arms"] = ["imm"]  # IMM is not run in the noise0 stratum

    with pytest.raises(rd.FixtureError, match="outside stratum noise0"):
        rd.parse_fixture(_broken(mutate))


@pytest.mark.parametrize("digest, ok", [
    ("a" * 64, True), ("A" * 64, False), ("0123456789abcdef" + "0" * 48, True), ("ABCDEF" + "0" * 58, False),
    ("a" * 63, False), ("a" * 65, False), ("g" * 64, False), ("a" * 63 + " ", False), ("", False),
])
def test_source_digests_are_exactly_64_hex_characters(digest, ok):
    for key in ("sha256", "manifest_sha256"):
        obj = _broken(lambda o: o["s6"]["source"].update({key: digest}))
        if ok:
            assert rd.parse_fixture(obj) is not None
        else:
            with pytest.raises(rd.FixtureError, match=f"{key} must be 64 hex|non-empty"):
                rd.parse_fixture(obj)


def test_the_real_fixture_has_exactly_the_documented_keys():
    assert set(RAW) == {"atlas", "generator", "s6", "schema", "simulated"}
    assert set(RAW["s6"]) == set(rd.S6_KEYS) and set(RAW["atlas"]) == set(rd.ATLAS_KEYS)
    assert all(set(entry) == set(rd.ENTRY_KEYS) for entry in RAW["atlas"]["entries"])


# ---- atlas binning on whole milliseconds -------------------------------------------------------------
@pytest.mark.parametrize("seconds, stop", [
    (0.0, 1), (0.025, 1), (0.026, 2), (0.05, 2), (0.1, 2), (0.101, 3), (0.15, 3), (0.2, 3), (0.3, 3),
    (0.025 + 4e-4, 1),   # 25.4 ms rounds to 25
    (0.1 + 4e-4, 2),     # 100.4 ms rounds to 100
    (0.07500000000000001, 2), (0.15000000000000002, 3), (0.19999999999999998, 3),
])
def test_binning_uses_whole_milliseconds(seconds, stop):
    assert rd.entry_stop({"kind": "value", "value_s": seconds}) == stop


@pytest.mark.parametrize("ms, stop", [(0, 1), (25, 1), (50, 2), (100, 2), (150, 3), (200, 3)])
def test_binning_boundaries_for_every_delay_the_atlas_tests(ms, stop):
    assert rd.entry_stop({"kind": "value", "value_s": ms / 1000.0}) == stop
    assert rd.entry_cell_text({"kind": "value", "value_s": ms / 1000.0}) == f"{ms} ms"


# ---- string parity with ResultsFormat.swift: inputs and expected strings written by hand from the Swift --------
def test_formatters_match_the_swift_strings():
    """Each row is read off ResultsFormat.swift (or the shared UI text it fills), not produced by the Python under test."""
    table = [
        (rd.metres(0.158), "0.158"), (rd.metres(-0.0), "0.000"), (rd.metres(-0.0004), "0.000"),
        (rd.metres_text(0.158), "0.158 m"), (rd.range_text(0.149, 0.17), "0.149 to 0.170 m"),
        (rd.delay_text(0.025), "25 ms"), (rd.delay_text(0.0), "0 ms"), (rd.delay_text(0.3), "300 ms"),
        (rd.count_text(0, 50), "0 of 50"), (rd.panel_title(0), "No loss"), (rd.panel_title(0.3), "30 % loss"),
        (rd.panel_title(0.29), "28 % loss"), (rd.loss_key("burst", 0.3), "burst:0.3"), (rd.loss_key("independent", 0), "independent:0"),
        (rd.scale_caption(0.7), "Error axis 0 to 0.700 m for this path and loss type"),
        (rd.failure_label(0.25), "Failure line 0.250 m"),
        (rd.display_text("see `runs/s6`"), "see runs/s6"), (rd.sentence("a cell fails"), "A cell fails"), (rd.sentence(""), ""),
        (rd.entry_cell_text({"kind": "censored", "value_s": None}), ">300 ms"),
        (rd.entry_cell_text({"kind": "fails_at_zero", "value_s": None}), "Fails at 0"),
        (rd.entry_cell_text(None), "Not computed"),
        (rd.tolerated_text({"kind": "fails_at_zero", "value_s": None}), "Fails at 0 ms"),
        (rd.tolerated_text({"kind": "value", "value_s": 0.2}), "200 ms"),
        (rd.interval_text([0.1, 0.2]), "100 ms to 200 ms"), (rd.interval_text(["<0", 0.2]), "Below 0 ms to 200 ms"),
        (rd.interval_text([0.1, ">0.3"]), "100 ms to above 300 ms"), (rd.interval_text([">0.3", ">0.3"]), "Above 300 ms"),
        (rd.interval_text([0.1]), "Not computed"), (rd.interval_text(None), "Not computed"),
        (rd.cause_text(None), "No failure seen"), (rd.cause_text("lag-dominated"), "Delay"),
        (rd.cause_text("gap-dominated"), "Lost packets"), (rd.cause_text("noise-dominated"), "Sensor noise"),
        (rd.cause_text("model-mismatch"), "Path model"), (rd.cause_text("slow-start"), "Slow start"),
        (rd.arm_name("naive"), "Naive tracker"), (rd.arm_name("recovery"), "Recovery (capture time)"),
        (rd.arm_name("some_arm"), "Some arm"), (rd.path_name("constant_velocity"), "Straight line"),
        (rd.path_name("constant_turn"), "1 m turn"), (rd.path_name("figure_eight"), "Figure eight"),
        (rd.label_text("some_key_name"), "Some key name"),
        (rd.cell_subtitle("Figure eight", "30 % bursty"), "Figure eight, 30 % bursty"),
        (DATA.loss_title("burst:0.3"), "30 % bursty"), (DATA.loss_title("burst:0.9"), "Not computed"),
        (DATA.arm_label("ct"), "Turn-rate (CT)"), (DATA.arm_label("new_arm"), "New arm"),
        (DATA.variant_title("figure8"), "Figure eight"), (DATA.variant_title("odd_shape"), "Odd shape"),
        (rd.interval_caption(None, 20), "The interval shows how far the tolerated delay could move with more seeds."),
    ]
    for got, want in table:
        assert got == want


# ---- the bundle validator checks the bundled results file --------------------------------------------------
def _fake_bundle(tmp_path, *, fixture="real"):
    """A minimal frozen-bundle layout: the files validate() looks for, copied from the checkout."""
    import shutil

    dist = tmp_path / "dist"
    bundle = dist / "ZephyrMeshWindows"
    (bundle / "_internal" / "desktop").mkdir(parents=True)
    (bundle / "_internal" / "runs" / "s7-swarm").mkdir(parents=True)
    (bundle / "ZephyrMeshWindows.exe").write_bytes(b"")
    (dist / "ZephyrMeshUpdater.exe").write_bytes(b"")
    (bundle / "VERSION.txt").write_text("0.0.0\n", encoding="utf-8")
    (bundle / "README-Windows.txt").write_text("Extract the complete ZIP ... startup.log\n", encoding="utf-8")
    shutil.copy(ROOT / "runs" / "s7-swarm" / "run.json", bundle / "_internal" / "runs" / "s7-swarm" / "run.json")
    for name in ("evidence_report_example.json", "investigation_report_example.json", "s7_report_example.json"):
        shutil.copy(ROOT / "desktop" / name, bundle / "_internal" / "desktop" / name)
    target = bundle / "_internal" / "desktop" / "results_fixture.json"
    if fixture == "real":
        shutil.copy(FIXTURE, target)
    elif fixture == "broken":
        target.write_text(json.dumps(_broken(lambda o: o["s6"].update(extra=1))), encoding="utf-8")
    return bundle


def test_bundle_validator_loads_the_bundled_results_file(tmp_path):
    from tools.validate_windows_bundle import validate

    result = validate(_fake_bundle(tmp_path))
    assert "results_fixture" in result and "strata" in result["results_fixture"]


@pytest.mark.parametrize("fixture, error, words", [
    ("missing", FileNotFoundError, "results_fixture.json is missing"),
    ("broken", RuntimeError, "results_fixture.json is invalid: s6 has unknown extra"),
])
def test_bundle_validator_fails_on_a_missing_or_invalid_results_file(tmp_path, fixture, error, words):
    from tools.validate_windows_bundle import validate

    with pytest.raises(error, match=words):
        validate(_fake_bundle(tmp_path, fixture=fixture))


def test_the_build_script_bundles_the_results_file_in_short_lines():
    script = (ROOT / "desktop" / "build_windows.ps1").read_text(encoding="utf-8")
    assert 'desktop\\results_fixture.json;desktop' in script
    for module in ("results_data", "ui_results_chart", "ui_results_atlas", "ui_results"):
        assert f"--hidden-import desktop.{module}" in script
    start = script.index("--name ZephyrMeshWindows")
    pyinstaller = script[start:script.index("if ($LASTEXITCODE", start)]
    lines = pyinstaller.splitlines()
    assert all(len(line) <= 140 for line in lines)
    assert all(line.endswith(" `") for line in lines[:-1]), "every line but the last continues with a backtick"
    assert "desktop\\windows_preview.py" in lines[-1] and lines[-1].lstrip().startswith("--distpath")
