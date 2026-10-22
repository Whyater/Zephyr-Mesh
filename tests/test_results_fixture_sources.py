"""Independent comparer: every number in results_fixture.json must equal its source run file.

Written only from the fixture contract and from the source files (sim/runs/s6-recovery, sim/runs/s6-atlas,
README.md, docs/RESULTS.md). It does not import or read the fixture builder. A missing fixture is a
failure with a clear message, not a skip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATHS = [
    ROOT / "tools" / "desktop" / "results_fixture.json",
    ROOT / "tools" / "macos" / "ZephyrMeshApp" / "Sources" / "ZephyrMeshApp" / "Resources" / "results_fixture.json",
]
S6_DIR = ROOT / "sim" / "runs" / "s6-recovery"
ATLAS_DIR = ROOT / "sim" / "runs" / "s6-atlas"

FAILS_SHOWN = 12


class Checker:
    """Collects mismatches so one run reports every disagreement, and counts comparisons."""

    def __init__(self, name: str):
        self.name = name
        self.bad: list[str] = []
        self.n = 0

    def eq(self, got, want, what: str) -> None:
        self.n += 1
        same = got == want and isinstance(got, bool) == isinstance(want, bool)
        if not same:
            self.bad.append(f"{what}: fixture={got!r} source={want!r}")

    def true(self, cond: bool, what: str) -> None:
        self.n += 1
        if not cond:
            self.bad.append(what)

    def done(self) -> None:
        print(f"\n[{self.name}] comparisons: {self.n}")
        if self.bad:
            shown = "\n  ".join(self.bad[:FAILS_SHOWN])
            more = f"\n  ... and {len(self.bad) - FAILS_SHOWN} more" if len(self.bad) > FAILS_SHOWN else ""
            pytest.fail(f"{self.name}: {len(self.bad)} disagreement(s)\n  {shown}{more}", pytrace=False)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixture(path: Path) -> dict:
    if not path.is_file():
        pytest.fail(f"fixture missing: {path.relative_to(ROOT)} (run tools/build_results_fixture.py)", pytrace=False)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fx() -> dict:
    return load_fixture(FIXTURE_PATHS[0])


@pytest.fixture(scope="module")
def s6_src() -> dict:
    return read_json(S6_DIR / "summary.json")


@pytest.fixture(scope="module")
def atlas_src() -> dict:
    return read_json(ATLAS_DIR / "atlas.json")


# ---------------------------------------------------------------- 5. file level

def test_both_fixture_copies_identical_and_formatted():
    c = Checker("fixture files")
    datas = []
    for p in FIXTURE_PATHS:
        if not p.is_file():
            pytest.fail(f"fixture missing: {p.relative_to(ROOT)} (run tools/build_results_fixture.py)", pytrace=False)
        datas.append(p.read_bytes())
    c.true(datas[0] == datas[1], "desktop and macos fixture copies differ byte for byte")
    raw = datas[0]
    obj = json.loads(raw.decode("utf-8"))
    want = (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    c.true(raw == want, "fixture bytes are not json.dumps(indent=1, sort_keys, ensure_ascii=False) + LF")
    c.true(b"\r" not in raw, "fixture contains CR")
    c.true(len(raw) < 3_000_000, f"fixture is {len(raw)} bytes, target is under 3 MB")
    c.eq(obj.get("schema"), "zephyr-results-fixture-1", "schema")
    c.eq(obj.get("generator"), "tools/build_results_fixture.py", "generator")
    c.eq(obj.get("simulated"), True, "simulated")
    c.eq(sorted(obj.keys()), ["atlas", "generator", "s6", "schema", "simulated"], "top level keys")
    c.done()


# ---------------------------------------------------------------- 1. hashes and revisions

def test_source_hashes_and_revisions(fx, s6_src, atlas_src):
    c = Checker("sources")
    for key, folder, fname in (("s6", S6_DIR, "summary.json"), ("atlas", ATLAS_DIR, "atlas.json")):
        src = fx[key]["source"]
        manifest = read_json(folder / "manifest.json")
        c.eq(src.get("run_dir"), "sim/runs/" + folder.name, f"{key}.source.run_dir")
        c.eq(src.get("file"), fname, f"{key}.source.file")
        c.eq(src.get("sha256"), sha(folder / fname), f"{key}.source.sha256")
        c.eq(src.get("manifest_sha256"), sha(folder / "manifest.json"), f"{key}.source.manifest_sha256")
        c.eq(src.get("git_revision"), manifest["git_revision"], f"{key}.source.git_revision")
        c.eq(src.get("git_tree_state"), manifest["git_tree_state"], f"{key}.source.git_tree_state")
    c.eq(fx["s6"]["source"].get("schema"), s6_src["schema"], "s6.source.schema")
    c.eq(fx["atlas"]["source"].get("schema"), atlas_src["schema"], "atlas.source.schema")
    c.done()


# ---------------------------------------------------------------- 2. s6

MOTIONS = ["constant_velocity", "constant_turn"]
LOSS_MODELS = ["independent", "burst"]
ARMS = ["naive", "recovery"]
ARM_KEYS = ["p95_median_m", "p95_ci_low_m", "p95_ci_high_m", "fail_count", "fail_rate", "n_seeds", "label"]
PAIRED_MAP = {
    "paired_diff_median_m": "p95_diff_median_m",
    "paired_diff_ci_low_m": "p95_diff_ci_low_m",
    "paired_diff_ci_high_m": "p95_diff_ci_high_m",
    "recovery_better_count": "recovery_better_count",
    "recovery_worse_count": "recovery_worse_count",
    "tie_count": "tie_count",
}


def selected_cells(s6_src: dict) -> list[dict]:
    out = []
    for cell in s6_src["cells"]:
        if cell["noise_std_m"] != 0.05:
            continue
        if cell["slice"] == "burst" or (cell["slice"] == "main" and cell["loss_model"] == "independent"):
            out.append(cell)
    return out


def test_s6_header_fields(fx, s6_src):
    c = Checker("s6 header")
    s6 = fx["s6"]
    sel = selected_cells(s6_src)
    c.eq(len(sel), 98, "source selected cell count (56 main + 42 burst)")
    c.eq(s6["status"], s6_src["status"], "status")
    c.eq(s6["criterion"], s6_src["design"]["criterion"], "criterion")
    c.eq(s6["cell_label_rule"], s6_src["design"]["cell_label_rule"], "cell_label_rule")
    c.eq(s6["interval_note"], s6_src["design"]["interval"], "interval_note")
    for key in ("failure_p95_m", "failure_max_m", "target_speed_mps", "turn_radius_m"):
        c.eq(s6[key], s6_src["config"][key], key)
    c.eq(s6["noise_std_m"], 0.05, "noise_std_m")
    c.eq(s6["delays_s"], sorted({cell["delay_s"] for cell in sel}), "delays_s")
    c.eq(s6["delays_s"], [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3], "delays_s literal list")
    c.eq(s6["motions"], MOTIONS, "motions")
    c.eq(s6["loss_models"], LOSS_MODELS, "loss_models")
    for model in LOSS_MODELS:
        levels = sorted({cell["loss_probability"] for cell in sel if cell["loss_model"] == model})
        c.eq(s6["loss_levels"][model], levels, f"loss_levels[{model}]")
    c.eq(s6["loss_levels"], {"independent": [0.0, 0.1, 0.3, 0.5], "burst": [0.1, 0.3, 0.5]}, "loss_levels literal")
    c.eq(s6["arms"], ARMS, "arms")
    c.eq(s6["limits"], s6_src["limits"], "limits verbatim")
    c.done()


def test_s6_series_values(fx, s6_src):
    c = Checker("s6 series")
    s6 = fx["s6"]
    by_id = {cell["cell_id"]: cell for cell in s6_src["cells"]}
    c.eq(len(by_id), len(s6_src["cells"]), "source cell_ids unique")
    delays = s6["delays_s"]
    c.eq(len(s6["series"]), 2 * 7 * 2, "series count (2 motions x 7 loss conditions x 2 arms)")
    order = []
    for s in s6["series"]:
        tag = f"{s['motion']}/{s['loss_model']}/{s['loss_probability']}/{s['arm']}"
        order.append((MOTIONS.index(s["motion"]), LOSS_MODELS.index(s["loss_model"]), s["loss_probability"],
                      ARMS.index(s["arm"])))
        for key in ("cell_ids",) + tuple(ARM_KEYS) + tuple(PAIRED_MAP):
            c.eq(len(s[key]), 7, f"{tag} len({key})")
        for i, cid in enumerate(s["cell_ids"]):
            cell = by_id.get(cid)
            c.true(cell is not None, f"{tag}[{i}] cell_id {cid!r} not in source")
            if cell is None:
                continue
            where = f"{tag}[{i}] {cid}"
            c.eq(delays[i], cell["delay_s"], f"{where} delay_s vs delays_s[{i}]")
            c.eq(cell["motion"], s["motion"], f"{where} motion")
            c.eq(cell["loss_model"], s["loss_model"], f"{where} loss_model")
            c.eq(cell["loss_probability"], s["loss_probability"], f"{where} loss_probability")
            c.eq(cell["noise_std_m"], 0.05, f"{where} source noise_std_m")
            c.true(cell["slice"] == "burst" or (cell["slice"] == "main" and cell["loss_model"] == "independent"),
                   f"{where} source slice {cell['slice']!r} not selectable")
            arm = cell["arms"][s["arm"]]
            for key in ARM_KEYS:
                c.eq(s[key][i], arm[key], f"{where} {key}")
            for fkey, skey in PAIRED_MAP.items():
                c.eq(s[fkey][i], cell["paired"][skey], f"{where} {fkey}")
    expected = [(m, lm, p, a) for m in range(2) for lm, ps in ((0, [0.0, 0.1, 0.3, 0.5]), (1, [0.1, 0.3, 0.5]))
                for p in ps for a in range(2)]
    c.eq(order, expected, "series order (motion, loss model, loss probability ascending, arm)")
    c.done()


def test_s6_cell_coverage(fx, s6_src):
    c = Checker("s6 coverage")
    want = {cell["cell_id"] for cell in selected_cells(s6_src)}
    seen: dict[tuple[str, str], int] = {}
    for s in fx["s6"]["series"]:
        for cid in s["cell_ids"]:
            seen[(cid, s["arm"])] = seen.get((cid, s["arm"]), 0) + 1
    got = {cid for cid, _ in seen}
    c.eq(sorted(got - want), [], "fixture cell_ids not selectable in source")
    c.eq(sorted(want - got), [], "selectable source cells missing from fixture")
    c.eq(len(got), 98, "distinct fixture cell_ids")
    for cid in sorted(want):
        for arm in ARMS:
            c.eq(seen.get((cid, arm), 0), 1, f"{cid} appears once for arm {arm}")
    c.done()


# ---------------------------------------------------------------- 3. atlas

STRATUM_ORDER = ["central", "noise0", "noise2", "noise10", "rate5", "rate20", "rate50", "jitter", "student_t"]
VARIANT_ORDER = ["cv_0.5", "cv_1", "cv_2", "ct_r0.5", "ct_r1", "ct_r3", "jink", "figure8", "start_stop"]
ARM_ORDER = ["naive", "recovery", "imm", "ct", "adaptive", "retro", "recovery_gated", "imm_gated", "ct_gated"]
STRATUM_TITLES = {"central": "Central: 10 Hz, 5 cm noise", "noise0": "No sensor noise", "noise2": "2 cm noise",
                  "noise10": "10 cm noise", "rate5": "5 Hz sensor", "rate20": "20 Hz sensor", "rate50": "50 Hz sensor",
                  "jitter": "Delay jitter", "student_t": "Heavy-tail noise"}
# What each title implies about the atlas.json stratum parameters.
STRATUM_PARAMS = {
    "central": {"noise_std_m": 0.05, "rate_hz": 10.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "noise0": {"noise_std_m": 0.0, "rate_hz": 10.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "noise2": {"noise_std_m": 0.02, "rate_hz": 10.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "noise10": {"noise_std_m": 0.1, "rate_hz": 10.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "rate5": {"noise_std_m": 0.05, "rate_hz": 5.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "rate20": {"noise_std_m": 0.05, "rate_hz": 20.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "rate50": {"noise_std_m": 0.05, "rate_hz": 50.0, "noise_model": "gaussian", "noise_dof": None, "jitter_rule": "none"},
    "jitter": {"noise_std_m": 0.05, "rate_hz": 10.0, "noise_model": "gaussian", "noise_dof": None,
               "jitter_rule": "jitter_s = delay_s / 2"},
    "student_t": {"noise_std_m": 0.05, "rate_hz": 10.0, "noise_model": "student_t", "noise_dof": 3.0, "jitter_rule": "none"},
}
ARM_TITLES = {"naive": "Naive", "recovery": "Recovery", "imm": "IMM", "ct": "Turn-rate (CT)", "adaptive": "Adaptive",
              "retro": "Retro", "recovery_gated": "Recovery gated", "imm_gated": "IMM gated", "ct_gated": "CT gated"}
FIXED_VARIANT_TITLES = {"jink": "Random jinking", "figure8": "Figure eight", "start_stop": "Start and stop"}


def variant_name(v: str) -> str:
    if v.startswith("cv_"):
        return f"Straight line, {v[3:]} m/s"
    if v.startswith("ct_r"):
        return f"Circle, {v[4:]} m radius"
    return FIXED_VARIANT_TITLES[v]


def loss_sort(cond: str) -> tuple[int, float]:
    model, prob = cond.split(":")
    return (1 if model == "burst" else 0, float(prob))


def entry_sort_key(e: dict) -> tuple:
    return (STRATUM_ORDER.index(e["stratum"]), VARIANT_ORDER.index(e["variant"]), loss_sort(e["loss_condition"]),
            ARM_ORDER.index(e["arm"]))


def ordered(values, order) -> list:
    return [v for v in order if v in set(values)]


def rank(tol: dict) -> float:
    if tol["fails_at_zero"]:
        return -1.0
    if tol["censored"] is not None:
        assert tol["censored"] == ">0.3"
        return 0.3 + 1e-9
    return float(tol["value"])


def derive_cells(atlas_src: dict) -> dict[tuple, tuple[str, set]]:
    groups: dict[tuple, dict[str, dict]] = {}
    for e in atlas_src["entries"]:
        groups.setdefault((e["stratum"], e["variant"], e["loss_condition"]), {})[e["arm"]] = e
    out = {}
    for key, arms in groups.items():
        ranked = {a: rank(e["max_tolerated_delay_s"]) for a, e in arms.items()}
        ranked = {a: r for a, r in ranked.items() if r >= 0}
        if not ranked:
            out[key] = ("none", set())
            continue
        best = max(ranked.values())
        top = {a for a, r in ranked.items() if r == best}
        out[key] = ("winner" if len(top) == 1 else "tie", top)
    return out


def delay_string(entry: dict) -> str:
    tol = entry["max_tolerated_delay_s"]
    if tol["censored"] is not None:
        return ">300 ms"
    return f"{round(tol['value'] * 1000, 6):g} ms"


def test_atlas_entries(fx, atlas_src):
    c = Checker("atlas entries")
    at = fx["atlas"]
    src = {}
    for e in atlas_src["entries"]:
        key = (e["stratum"], e["variant"], e["loss_condition"], e["arm"])
        assert key not in src, f"duplicate source key {key}"
        src[key] = e
    got = {}
    for f in at["entries"]:
        key = (f["stratum"], f["variant"], f["loss_condition"], f["arm"])
        c.true(key not in got, f"duplicate fixture entry {key}")
        got[key] = f
    c.eq(sorted(set(got) - set(src)), [], "fixture entry keys not in source")
    c.eq(sorted(set(src) - set(got)), [], "source entry keys missing in fixture")
    c.eq(len(at["entries"]), len(atlas_src["entries"]), "entry count")
    c.eq([entry_sort_key(f) for f in at["entries"]], sorted(entry_sort_key(e) for e in atlas_src["entries"]),
         "entries order (stratum, variant, loss random-then-burst by probability, arm)")
    for key, e in src.items():
        f = got.get(key)
        if f is None:
            continue
        where = "/".join(str(k) for k in key)
        tol = e["max_tolerated_delay_s"]
        kind = "fails_at_zero" if tol["fails_at_zero"] else ("censored" if tol["censored"] is not None else "value")
        c.eq(f["kind"], kind, f"{where} kind")
        c.eq(f["value_s"], tol["value"], f"{where} value_s")
        c.eq(f["censored"], tol["censored"], f"{where} censored")
        c.eq(f["interval_s"], tol["interval_s"], f"{where} interval_s")
        c.eq(f["failure_mode"], e["failure_mode"], f"{where} failure_mode")
        c.eq(f["loss_model"], e["loss_model"], f"{where} loss_model")
        c.eq(f["loss_probability"], e["loss_probability"], f"{where} loss_probability")
        imp = f["improvement_vs_naive"]
        want_imp = e["improvement_vs_naive"]
        c.eq(len(imp), len(want_imp), f"{where} improvement count")
        if key[3] == "naive":
            c.eq(imp, [], f"{where} naive improvement empty")
        for j, (a, b) in enumerate(zip(imp, want_imp)):
            c.eq(a["delay_s"], b["delay_s"], f"{where} imp[{j}].delay_s")
            c.eq(a["median_p95_diff_m"], b["median_p95_diff_m"], f"{where} imp[{j}].median_p95_diff_m")
            c.eq(a["interval_m"], b["interval_m"], f"{where} imp[{j}].interval_m")
    c.done()


def test_atlas_header_fields(fx, atlas_src):
    c = Checker("atlas header")
    at = fx["atlas"]
    c.eq(at["schema_of_source"], atlas_src["schema"], "schema_of_source")
    c.eq(at["n_seeds"], len(atlas_src["seeds"]), "n_seeds")
    c.eq(at["failure_rule"], atlas_src["failure_rule"], "failure_rule verbatim")
    c.eq(at["limits"], atlas_src["limits"], "limits verbatim")
    delays = sorted({d["delay_s"] for e in atlas_src["entries"] if e["arm"] != "naive" for d in e["improvement_vs_naive"]})
    c.eq(at["delays_s"], delays, "delays_s")
    c.eq(len(at["delays_s"]), 7, "7 tested delays")
    c.eq([s["id"] for s in at["strata"]], STRATUM_ORDER, "strata order")
    by_stratum: dict[str, list[dict]] = {}
    for e in atlas_src["entries"]:
        by_stratum.setdefault(e["stratum"], []).append(e)
    for s in at["strata"]:
        sid = s["id"]
        c.eq(s["title"], STRATUM_TITLES[sid], f"{sid} title")
        c.eq(atlas_src["strata"][sid], STRATUM_PARAMS[sid], f"{sid} source parameters match what the title says")
        ents = by_stratum[sid]
        c.eq(s["variants"], ordered({e["variant"] for e in ents}, VARIANT_ORDER), f"{sid} variants (ordered)")
        c.eq(s["arms"], ordered({e["arm"] for e in ents}, ARM_ORDER), f"{sid} arms (ordered)")
        c.eq(s["loss_conditions"], sorted({e["loss_condition"] for e in ents}, key=loss_sort), f"{sid} loss_conditions (ordered)")
        c.true(isinstance(s["description"], str) and s["description"] != "", f"{sid} description present")
    all_variants = {e["variant"] for e in atlas_src["entries"]}
    all_arms = {e["arm"] for e in atlas_src["entries"]}
    c.eq(sorted(at["variant_titles"]), sorted(all_variants), "variant_titles keys")
    c.eq(sorted(at["arm_labels"]), sorted(all_arms), "arm_labels keys")
    c.eq(at["variant_titles"], {v: variant_name(v) for v in all_variants}, "variant_titles values")
    c.eq(at["arm_labels"], {a: ARM_TITLES[a] for a in all_arms}, "arm_labels values")
    c.eq(sorted(at["loss_titles"]), sorted({e["loss_condition"] for e in atlas_src["entries"]}), "loss_titles keys")
    # Written out by hand from the README loss levels, so a wrong title formula in the builder cannot pass here.
    expected_titles = {
        "independent:0": "No loss", "independent:0.1": "10 % random", "independent:0.3": "30 % random",
        "independent:0.5": "50 % random", "burst:0.1": "10 % bursty", "burst:0.3": "30 % bursty", "burst:0.5": "50 % bursty",
    }
    for key, title in at["loss_titles"].items():
        c.eq(title, expected_titles.get(key, f"<no hand-written title for {key}>"), f"loss_titles[{key}]")
    c.done()


def test_atlas_cells_and_counts(fx, atlas_src):
    c = Checker("atlas cells")
    at = fx["atlas"]
    derived = derive_cells(atlas_src)
    entries = {(e["stratum"], e["variant"], e["loss_condition"], e["arm"]): e for e in atlas_src["entries"]}
    got = {}
    for cell in at["cells"]:
        key = (cell["stratum"], cell["variant"], cell["loss_condition"])
        c.true(key not in got, f"duplicate fixture cell {key}")
        got[key] = cell
    c.eq(sorted(set(got) - set(derived)), [], "fixture cells not in source groups")
    c.eq(sorted(set(derived) - set(got)), [], "source groups missing from fixture cells")
    c.eq(len(at["cells"]), len(derived), "cell count")
    cell_key = lambda k: (STRATUM_ORDER.index(k[0]), VARIANT_ORDER.index(k[1]), loss_sort(k[2]))  # noqa: E731
    c.eq([cell_key((x["stratum"], x["variant"], x["loss_condition"])) for x in at["cells"]],
         sorted(cell_key(k) for k in derived), "cells order (stratum, variant, loss)")
    for key, (kind, arms) in derived.items():
        cell = got.get(key)
        if cell is None:
            continue
        where = "/".join(key)
        c.eq(cell["kind"], kind, f"{where} kind")
        c.eq(cell["arms"], ordered(arms, ARM_ORDER), f"{where} arms (ordered list in arm order)")
        if kind == "none":
            c.eq(cell["delay_text"], None, f"{where} delay_text")
        else:
            top = sorted(arms)[0]
            c.eq(cell["delay_text"], delay_string(entries[key + (top,)]), f"{where} delay_text")
    # per stratum counts
    for s in at["strata"]:
        kinds = [k for (st, _, _), (k, _) in derived.items() if st == s["id"]]
        want = {"cells": len(kinds), "none": kinds.count("none"), "winner": kinds.count("winner"),
                "tie": kinds.count("tie"), "survive": kinds.count("winner") + kinds.count("tie")}
        c.eq(s["counts"], want, f"{s['id']} counts")
        c.eq(len(kinds), len(s["variants"]) * len(s["loss_conditions"]), f"{s['id']} cells = variants x losses")
    central = next(s for s in at["strata"] if s["id"] == "central")["counts"]
    c.eq(central["cells"], 63, "central cells")
    c.eq(central["tie"], 39, "central ties")
    c.eq(central["winner"], 16, "central winners")
    c.eq(central["none"], 8, "central none")
    ckinds = [cell["kind"] for cell in at["cells"] if cell["stratum"] == "central"]
    c.eq((len(ckinds), ckinds.count("tie"), ckinds.count("winner"), ckinds.count("none")), (63, 39, 16, 8),
         "central totals from cells list")
    c.done()


# ---------------------------------------------------------------- 4. findings

def test_findings_verbatim_in_sources(fx):
    c = Checker("findings")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    texts = {"README.md": readme, "docs/RESULTS.md": results}
    s6_find = fx["s6"]["findings"]
    c.eq(len(s6_find), 3, "s6 findings count")
    lines = readme.splitlines()
    for i, f in enumerate(s6_find):
        c.eq(f["source"], "README.md", f"s6 finding[{i}] source")
        c.true(f["text"] in texts[f["source"]], f"s6 finding[{i}] text not verbatim in {f['source']}: {f['text'][:60]!r}")
        # contract: README lines 37, 38, 39
        c.true(f["text"] in lines[36 + i], f"s6 finding[{i}] not within README.md line {37 + i}")
    at_find = fx["atlas"]["findings"]
    c.true(len(at_find) >= 1, "atlas has a finding")
    for i, f in enumerate(at_find):
        c.eq(f["source"], "docs/RESULTS.md", f"atlas finding[{i}] source")
        c.true(f["text"] in texts[f["source"]], f"atlas finding[{i}] text not verbatim in {f['source']}: {f['text'][:60]!r}")
    c.true(at_find[0]["text"].startswith("The map: most cells are ties"), "atlas finding[0] starts with the map bullet")
    c.true(not at_find[0]["text"].startswith("- "), "atlas finding[0] drops the bullet marker")
    c.done()
