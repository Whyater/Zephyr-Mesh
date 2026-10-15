#!/usr/bin/env python3
"""Build the shared S6 results and tolerance atlas fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_s6_atlas_figure import (  # noqa: E402
    ARM_ORDER,
    VARIANT_ORDER,
    arm_label,
    arm_sort_key,
    cell_counts,
    classify_cell,
    delay_text,
    group_entries,
    loss_key,
    stratum_description,
    tie_text,
    variant_sort_key,
    variant_title,
)

OUT_MACOS = ROOT / "macos/ZephyrMeshApp/Sources/ZephyrMeshApp/Resources/results_fixture.json"
OUT_DESKTOP = ROOT / "desktop/results_fixture.json"
S6_DELAYS = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
STRATUM_ORDER = ["central", "noise0", "noise2", "noise10", "rate5", "rate20", "rate50", "jitter", "student_t"]
STRATUM_TITLES = {
    "central": "Central: 10 Hz, 5 cm noise", "noise0": "No sensor noise", "noise2": "2 cm noise",
    "noise10": "10 cm noise", "rate5": "5 Hz sensor", "rate20": "20 Hz sensor",
    "rate50": "50 Hz sensor", "jitter": "Delay jitter", "student_t": "Heavy-tail noise",
}
README_FINDINGS = [line[2:] for line in (
    "- A basic tracker's error grows by about speed x delay. Its p95 error (the error it stays under 95 % of the time) "
    "crosses the 0.25 m line at 200 ms of delay with no loss, for a 1 m/s target, 10 Hz measurements, 5 cm noise and "
    "50 seeds (`runs/s6-recovery`).",
    "- Stamping each measurement with the time it was taken, then predicting forward, cuts the 300 ms error with no loss "
    "from 0.36 m to 0.16 m on a straight line, and from 0.38 m to 0.29 m on a 1 m radius turn (`runs/s6-recovery`).",
    "- On the turn, lost packets matter more than delay: at 30 % random loss the recovery tracker fails from 150 ms just "
    "like the basic one, and at 50 % both fail at every delay (`runs/s6-recovery`).",
)]


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def source_block(run_dir: str, file_name: str, raw: bytes, manifest: dict[str, Any], schema: str) -> dict[str, Any]:
    return {"run_dir": run_dir, "file": file_name, "sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_sha256": hashlib.sha256((ROOT / run_dir / "manifest.json").read_bytes()).hexdigest(),
            "schema": schema, "git_revision": manifest["git_revision"], "git_tree_state": manifest["git_tree_state"]}


def build_s6(summary: dict[str, Any], summary_raw: bytes, manifest: dict[str, Any], readme: str) -> dict[str, Any]:
    for finding in README_FINDINGS:
        if finding not in readme:
            raise ValueError(f"README finding missing: {finding}")
    selected = [c for c in summary["cells"] if (
        c["slice"] == "main" and c["noise_std_m"] == 0.05 and c["loss_model"] == "independent"
    ) or c["slice"] == "burst"]
    for c in selected:
        if c["slice"] == "burst" and c["noise_std_m"] != 0.05:
            raise ValueError(f"burst cell {c['cell_id']} is not noise 0.05")
        if set(c["arms"]) != {"naive", "recovery"}:
            raise ValueError(f"unexpected arm set in {c['cell_id']}: {sorted(c['arms'])}")
        if not finite(c):
            raise ValueError(f"non-finite number in {c['cell_id']}")
    motions = ["constant_velocity", "constant_turn"]
    loss_models = ["independent", "burst"]
    levels = {"independent": [0.0, 0.1, 0.3, 0.5], "burst": [0.1, 0.3, 0.5]}
    delays = sorted({c["delay_s"] for c in selected})
    if delays != S6_DELAYS:
        raise ValueError(f"selected delays are not the required seven: {delays}")
    by_key = {(c["motion"], c["loss_model"], c["loss_probability"], c["delay_s"]): c for c in selected}
    series = []
    for motion in motions:
        for model in loss_models:
            for probability in levels[model]:
                cells = [by_key.get((motion, model, probability, delay)) for delay in delays]
                if any(c is None for c in cells):
                    raise ValueError(f"series missing one of seven delays: {motion}/{model}/{probability}")
                for arm in ("naive", "recovery"):
                    vals = [c["arms"][arm] for c in cells]
                    paired = [c["paired"] for c in cells]
                    series.append({"motion": motion, "loss_model": model, "loss_probability": probability, "arm": arm,
                        "cell_ids": [c["cell_id"] for c in cells],
                        "p95_median_m": [v["p95_median_m"] for v in vals], "p95_ci_low_m": [v["p95_ci_low_m"] for v in vals],
                        "p95_ci_high_m": [v["p95_ci_high_m"] for v in vals], "fail_count": [v["fail_count"] for v in vals],
                        "fail_rate": [v["fail_rate"] for v in vals], "n_seeds": [v["n_seeds"] for v in vals],
                        "label": [v["label"] for v in vals], "paired_diff_median_m": [p["p95_diff_median_m"] for p in paired],
                        "paired_diff_ci_low_m": [p["p95_diff_ci_low_m"] for p in paired],
                        "paired_diff_ci_high_m": [p["p95_diff_ci_high_m"] for p in paired],
                        "recovery_better_count": [p["recovery_better_count"] for p in paired],
                        "recovery_worse_count": [p["recovery_worse_count"] for p in paired], "tie_count": [p["tie_count"] for p in paired]})
    return {"source": source_block("runs/s6-recovery", "summary.json", summary_raw, manifest, summary["schema"]),
        "status": summary["status"], "criterion": summary["design"]["criterion"], "cell_label_rule": summary["design"]["cell_label_rule"],
        "interval_note": summary["design"]["interval"], "failure_p95_m": summary["config"]["failure_p95_m"],
        "failure_max_m": summary["config"]["failure_max_m"], "target_speed_mps": summary["config"]["target_speed_mps"],
        "turn_radius_m": summary["config"]["turn_radius_m"], "noise_std_m": 0.05, "delays_s": delays,
        "motions": motions, "loss_models": loss_models, "loss_levels": levels, "arms": ["naive", "recovery"],
        "limits": summary["limits"], "findings": [{"text": f, "source": "README.md"} for f in README_FINDINGS], "series": series}


def atlas_entry(entry: dict[str, Any]) -> dict[str, Any]:
    tol = entry["max_tolerated_delay_s"]
    if tol["fails_at_zero"]:
        kind = "fails_at_zero"
    elif tol["censored"] is not None:
        kind = "censored"
    else:
        kind = "value"
    return {"stratum": entry["stratum"], "variant": entry["variant"], "loss_condition": entry["loss_condition"],
        "loss_model": entry["loss_model"], "loss_probability": entry["loss_probability"], "arm": entry["arm"], "kind": kind,
        "value_s": tol["value"], "censored": tol["censored"], "interval_s": tol["interval_s"],
        "failure_mode": entry["failure_mode"], "improvement_vs_naive": entry["improvement_vs_naive"]}


def build_atlas(atlas: dict[str, Any], atlas_raw: bytes, manifest: dict[str, Any], results_text: str) -> dict[str, Any]:
    matches = [line[2:] for line in results_text.splitlines() if line.startswith("- The map: most cells are ties")]
    if len(matches) != 1:
        raise ValueError(f"expected one atlas finding, found {len(matches)}")
    groups = group_entries(atlas)
    loss_conditions = sorted(
        {e["loss_condition"] for e in atlas["entries"]},
        key=lambda x: (0 if x.startswith("independent:") else 1, float(x.split(":")[1])),
    )
    loss_titles = {}
    for condition in loss_conditions:
        model, prob = condition.split(":")
        loss_titles[condition] = (
            "No loss" if condition == "independent:0"
            else f"{int(float(prob) * 100)} % {'random' if model == 'independent' else 'bursty'}"
        )
    for condition in loss_conditions:
        if condition not in loss_titles:
            raise ValueError(f"missing loss title for {condition}")
    delays = None
    for entry in atlas["entries"]:
        if entry["arm"] != "naive":
            current = [x["delay_s"] for x in entry["improvement_vs_naive"]]
            if delays is None:
                delays = current
            elif current != delays:
                raise ValueError("atlas improvement delay lists differ")
    strata = []
    cells = []
    ordered_entries = []
    for stratum in STRATUM_ORDER:
        info = atlas["strata"][stratum]
        title = STRATUM_TITLES[stratum]
        checks = {"central": info["rate_hz"] == 10 and info["noise_std_m"] == 0.05 and info["noise_model"] == "gaussian",
                  "noise0": info["noise_std_m"] == 0.0, "noise2": info["noise_std_m"] == 0.02, "noise10": info["noise_std_m"] == 0.1,
                  "rate5": info["rate_hz"] == 5, "rate20": info["rate_hz"] == 20, "rate50": info["rate_hz"] == 50,
                  "jitter": info["jitter_rule"] != "none", "student_t": info["noise_model"] == "student_t" and info["noise_dof"] == 3}
        if not checks[stratum]:
            raise ValueError(f"stratum parameters do not match {stratum}")
        sg = groups[stratum]
        variants = sorted({v for v, _ in sg}, key=variant_sort_key)
        arms = sorted({a for slot in sg.values() for a in slot}, key=arm_sort_key)
        sloss = sorted({lk for _, lk in sg}, key=lambda x: (x[0], x[1]))
        classified = []
        for variant in variants:
            for lk in sloss:
                slot = sg[(variant, lk)]
                kind, winning_arms, representative = classify_cell(slot)
                classified.append((kind, winning_arms, representative))
                condition = f"{'burst' if lk[0] else 'independent'}:{lk[1]:g}"
                cells.append({"stratum": stratum, "variant": variant, "loss_condition": condition,
                    "kind": kind, "arms": winning_arms, "delay_text": delay_text(representative) if representative else None})
        strata.append({"id": stratum, "title": title, "description": stratum_description(atlas, stratum),
            "variants": variants, "arms": arms,
            "loss_conditions": [f"{'burst' if lk[0] else 'independent'}:{lk[1]:g}" for lk in sloss],
            "counts": cell_counts(classified)})
        for variant in variants:
            for lk in sloss:
                for arm in sorted(sg[(variant, lk)], key=arm_sort_key):
                    ordered_entries.append(atlas_entry(sg[(variant, lk)][arm]))
    finding = {"text": matches[0], "source": "docs/RESULTS.md"}
    return {"source": source_block("runs/s6-atlas", "atlas.json", atlas_raw, manifest, atlas["schema"]),
        "schema_of_source": atlas["schema"],
        "n_seeds": len(atlas["seeds"]), "failure_rule": atlas["failure_rule"], "limits": atlas["limits"], "delays_s": delays or [],
        "findings": [finding], "strata": strata, "variant_titles": {v: variant_title(v) for v in VARIANT_ORDER},
        "arm_labels": {a: arm_label(a) for a in ARM_ORDER}, "loss_titles": loss_titles, "entries": ordered_entries, "cells": cells}


def build_fixture() -> dict[str, Any]:
    summary, summary_raw = read_json(ROOT / "runs/s6-recovery/summary.json")
    s6_manifest, _ = read_json(ROOT / "runs/s6-recovery/manifest.json")
    atlas, atlas_raw = read_json(ROOT / "runs/s6-atlas/atlas.json")
    atlas_manifest, _ = read_json(ROOT / "runs/s6-atlas/manifest.json")
    fixture = {"schema": "zephyr-results-fixture-1", "generator": "tools/build_results_fixture.py", "simulated": True,
        "s6": build_s6(summary, summary_raw, s6_manifest, (ROOT / "README.md").read_text(encoding="utf-8")),
        "atlas": build_atlas(atlas, atlas_raw, atlas_manifest, (ROOT / "docs/RESULTS.md").read_text(encoding="utf-8"))}
    if not finite(fixture):
        raise ValueError("fixture contains a non-finite number")
    return fixture


def encoded(fixture: dict[str, Any]) -> bytes:
    return (json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = encoded(build_fixture())
    if args.check:
        for path in (OUT_MACOS, OUT_DESKTOP):
            if not path.is_file() or path.read_bytes() != data:
                print(f"results fixture differs: {path}", file=sys.stderr)
                return 1
        return 0
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results_fixture.json").write_bytes(data)
    else:
        for path in (OUT_MACOS, OUT_DESKTOP):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
