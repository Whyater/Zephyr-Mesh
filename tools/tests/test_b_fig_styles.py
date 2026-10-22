from __future__ import annotations

import pytest

from tools import horizon_mpl
from tools import make_s6_atlas_figure as atlas
from tools import make_s6_clocks_figure as clocks
from tools import make_s6_retro_figure as retro
from tools import make_s6_sending_figure as sending
from tools import make_s6_sensitivity_figure as sensitivity
from tools import make_s6_tuning_figure as tuning
from tools import make_s6_turn_figure as turn

TUNING_EXAMPLES = ("cv_r0.2", "cv_r5", "ca_q10", "ca_q25", "ct_q0.04", "ct_r0.2",
                   "imm_adaptq_r0.2", "imm_q25")

RUNNER_ARMS = {
    "retro": retro.ARMS,
    "turn": turn.ARMS,
    "clocks": clocks.ARMS,
    "atlas": atlas.ARM_ORDER,
    "sensitivity": sensitivity.ARMS,
    "sending": (sending.ARM,),
    "tuning models": tuning.MODELS,
    "tuning examples": TUNING_EXAMPLES,
}


@pytest.fixture(params=horizon_mpl.THEMES)
def pal(request):
    return horizon_mpl.palette(request.param)


@pytest.mark.parametrize("runner", sorted(RUNNER_ARMS))
def test_every_runner_arm_has_a_style_without_status_colors(runner, pal):
    for arm in RUNNER_ARMS[runner]:
        style = horizon_mpl.arm_style(arm, pal)
        assert set(style) == {"color", "linestyle", "marker"}
        assert style["color"].upper() not in (pal["alert"].upper(), pal["caution"].upper()), arm
        assert style["linestyle"] and style["marker"]


def test_fixed_mapping(pal):
    mix, style = horizon_mpl.mix, horizon_mpl.arm_style
    surface = pal["surface"]
    assert style("naive", pal) == {"color": pal["lineStrong"], "linestyle": "--", "marker": "o"}
    for name in ("recovery", "recovery-v1", "cv"):
        assert style(name, pal) == {"color": pal["accent"], "linestyle": "-", "marker": "o"}
    assert style("retro", pal) == {"color": mix(pal["accent"], pal["ok"], 0.5),
                                   "linestyle": "-", "marker": "D"}
    assert style("ca", pal) == {"color": pal["target"], "linestyle": "-", "marker": "^"}
    assert style("ca_q10", pal) == {"color": mix(pal["target"], surface, 0.4),
                                    "linestyle": ":", "marker": "^"}
    assert style("ct", pal) == {"color": pal["ok"], "linestyle": "-", "marker": "s"}
    assert style("ct_q10", pal) == {"color": mix(pal["ok"], surface, 0.4),
                                    "linestyle": ":", "marker": "s"}
    assert style("imm", pal) == {"color": pal["keepOut"], "linestyle": "-", "marker": "v"}
    assert style("imm_gated", pal) == {"color": mix(pal["keepOut"], surface, 0.4),
                                       "linestyle": "-.", "marker": "v"}
    assert style("adaptive", pal) == {"color": mix(pal["target"], pal["keepOut"], 0.5),
                                      "linestyle": "-", "marker": "P"}
    assert style("recovery_gated", pal) == {"color": mix(pal["accent"], surface, 0.4),
                                            "linestyle": "-.", "marker": "o"}


def test_tuning_names_follow_their_model_prefix(pal):
    style = horizon_mpl.arm_style
    assert style("cv_r0.2", pal)["color"] == pal["accent"]
    assert style("ct_q25", pal)["color"] == pal["ok"]
    assert style("ca_q25", pal)["color"] == pal["target"]
    assert style("imm_adaptq_r0.2", pal)["color"] == pal["keepOut"]


def test_arms_are_distinguishable(pal):
    names = ("naive", "recovery", "retro", "ca", "ct", "imm", "adaptive")
    colors = [horizon_mpl.arm_style(n, pal)["color"] for n in names]
    assert len(set(colors)) == len(names)


@pytest.mark.parametrize("name", ["", "bogus", "xyz_q10", "cvx", "caution", "alert"])
def test_unknown_names_raise(name, pal):
    with pytest.raises(ValueError):
        horizon_mpl.arm_style(name, pal)
