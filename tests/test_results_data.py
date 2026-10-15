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
def test_shared_y_scale_is_the_nice_ceiling_of_the_widest_interval():
    top = max(value for series in RAW["s6"]["series"] for value in series["p95_ci_high_m"])
    assert 1.4 < top < 1.5  # 1.484 in the checked-in run
    assert DATA.y_max == 1.5
    assert rd.y_ticks(DATA.y_max) == [0.0, 0.5, 1.0, 1.5]
    assert [rd.tick_text(t, [0.0, 0.5, 1.0, 1.5]) for t in (0.0, 1.0)] == ["0.0", "1.0"]


@pytest.mark.parametrize("value, ceiling", [(1.48, 1.5), (0.36, 0.4), (1.0, 1.0), (0.0, 1.0), (7.2, 8.0)])
def test_nice_ceiling(value, ceiling):
    assert math.isclose(rd.nice_ceiling(value), ceiling)


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


def test_findings_are_quoted_from_the_fixture():
    assert rd.finding_for_path(DATA, "constant_velocity")["text"] == RAW["s6"]["findings"][0]["text"]
    assert rd.finding_for_path(DATA, "constant_turn")["text"] == RAW["s6"]["findings"][-1]["text"]


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
    assert rd.tolerated_text({"kind": "fails_at_zero", "value_s": None}) == "Fails even at 0 ms"


@pytest.mark.parametrize("interval, text", [
    ([0.1, 0.2], "100 to 200 ms"),
    (["<0", 0.1], "Below 0 to 100 ms"),
    ([0.2, ">0.3"], "200 to above 300 ms"),
    ([">0.3", ">0.3"], "Above 300 ms"),
    (["<0", "<0"], "Below 0 ms"),
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
    tie = next(c for c in cells if c["kind"] == "tie")
    none = next(c for c in cells if c["kind"] == "none")
    assert rd.most_delay_text(DATA, winner) == (RAW["atlas"]["arm_labels"][winner["arms"][0]], winner["delay_text"])
    assert rd.most_delay_text(DATA, tie) == (ATLAS["tieCount"].format(count=len(tie["arms"])), tie["delay_text"])
    assert rd.most_delay_text(DATA, none) == (ATLAS["nothingPasses"], None)
    assert rd.most_delay_text(DATA, None)[0] == TEXT["notComputed"]


def test_summary_rule_and_difference_rows():
    counts = RAW["atlas"]["strata"][0]["counts"]
    assert rd.summary_text(DATA.strata["central"]) == (
        f"{counts['tie']} ties, {counts['winner']} clear winners, {counts['none']} cells where nothing passes, of {counts['cells']} cells")
    assert rd.rule_text(DATA).endswith(f"({RAW['atlas']['failure_rule']['cell_rule_status']}).")
    entry = DATA.entry("central", "cv_0.5", "burst:0.3", "ct")
    raw = entry["improvement_vs_naive"][0]
    assert rd.diff_rows(entry)[0] == ("0", f"{raw['median_p95_diff_m']:.3f}",
                                      f"{raw['interval_m'][0]:.3f} to {raw['interval_m'][1]:.3f}")
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
