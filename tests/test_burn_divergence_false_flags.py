"""Small fast tests for the divergence false-flag study helpers."""
from sim.s6_recovery import wilson_interval
from tools import burn_divergence_false_flags as study


def test_zero_upper_closed_form_matches_wilson_interval():
    for n in (10, 100, 1234, 5000):
        assert abs(study.wilson_zero_upper(n) - wilson_interval(0, n)[1]) < 1e-12


def test_zero_upper_hand_value():
    # n = 100: 1.959964^2 / (100 + 1.959964^2) = 3.841455 / 103.841455
    assert abs(study.wilson_zero_upper(100) - 0.036992) < 1e-5
    assert abs(study.Z95 - 1.959964) < 1e-6


def test_rule_of_three_is_close_to_wilson_for_large_n():
    n = 5000
    ratio = study.wilson_zero_upper(n) / study.rule_of_three(n)
    assert 1.0 < ratio < 1.5  # 3.84 / 3 for large n


def test_zero_check_reports_k_nonzero_without_asserting():
    assert "note" in study.zero_check(2, 100)
    assert "closed_form_upper" in study.zero_check(0, 100)


def test_strata_counts():
    groups = [s["group"] for s in study.build_strata()]
    assert groups.count("matched") == 6
    assert groups.count("mismodelled") == 6
    assert groups.count("overconfident") == 4


def test_smoke_run_one_stratum_keys_and_counts():
    stratum = next(s for s in study.build_strata() if s["group"] == "matched")
    out = study.run_stratum(stratum, 3)
    assert out["n"] == 3
    for key in ("diverged", "inconsistent", "old_rule_failed", "new_rule_failed"):
        assert set(out[key]) == {"k", "rate", "wilson95_low", "wilson95_high"}
        assert 0 <= out[key]["k"] <= 3
    assert "median_mean_nees" in out and "old_pass_but_diverged_count" in out


def test_strata_seed_blocks_do_not_overlap():
    n = len(study.build_strata())
    blocks = [set(study.stratum_seeds(i, 500)) for i in range(n)]
    assert sum(len(b) for b in blocks) == len(set().union(*blocks)) == 500 * n
