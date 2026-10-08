import json
from sim.report import build_baseline_report, write_report

def test_report_contract_is_json_and_read_only(tmp_path):
    src="runs/s0-baseline"
    r=build_baseline_report(src)
    assert r["schema"]=="zephyr-s1-report-1"
    json.dumps(r)
    out=write_report(src,tmp_path/"report.json")
    assert out.exists() and json.loads(out.read_text())["source"]["telemetry_sha256"]
    assert r["sampling"]["time_semantics"].startswith("S0 pre-step")
    assert r["source"]["telemetry_source"] == "runs/s0-baseline/telemetry.csv"
    assert "/" not in r["source"].get("telemetry_path", "")
    assert r["provenance"]["output_sha256"] is None
    assert r["provenance"]["hash_parity"]["telemetry_sha256"].startswith("sha256")
    assert r["provenance"]["artifact_status"] == "frozen historical fixture"
    assert r["provenance"]["current_worktree_match"] == "not claimed"
    manifest=json.load(open(src+"/manifest.json"))
    assert r["provenance"]["source_git_revision"] == manifest["git_revision"]
    assert r["provenance"]["source_git_tree_state"] == "dirty"
