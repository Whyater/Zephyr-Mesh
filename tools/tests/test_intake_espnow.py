"""Tests for tools/intake_espnow.py (one command from echo logs to a summary).

Every input here is a synthetic echo log pair made by tools/make_espnow_echo_fixture.py, a generator written
separately from the fit, so the settings it was made from are known. The checks compare the intake's numbers
with those settings. Nothing in this file is a radio measurement, and the "measured" runs below only test
that the status label travels through the stages: their logs are still synthetic.

The shared runs use few bootstrap resamples and the pilot slice so the file stays quick.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from tools import intake_espnow as tool
from tools.make_espnow_echo_fixture import EchoParams, write_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SEED = 20261020
N_BOOT = 200
MEASURED_ARGS = ["--operator", "bench-test", "--captured-at", "2026-11-02T10:00:00-06:00",
                 "--code-revision", "abc1234", "--instrument-model", "esp32-devkit-pair",
                 "--source", "bench test, nothing real"]
PROVENANCE_FLAGS = ["--operator", "--captured-at", "--code-revision", "--instrument-model", "--source"]

# A log whose burstiness z is between the fit's cutoff (1.645) and the bridge's (2.0): 600 packets, no return
# loss, this seed. Chosen once so the two tools disagree on a whole fixture; the test asserts the z is in the band.
BAND_PARAMS = EchoParams(n_packets=600, burst_start=0.02, burst_end=0.7, return_loss=0.0)
BAND_SEED = 18


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intake(*args) -> int:
    return tool.main([str(a) for a in args])


def _logs(root: Path, params: EchoParams | None = None, seed: int = FIXTURE_SEED) -> tuple[Path, Path, dict]:
    folder = root / "logs"
    write_fixture(folder, params or EchoParams(), seed)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    return folder / "sender.csv", folder / "echoer.csv", truth


def _run(root: Path, out_name: str, status: str = "synthetic", *extra, params: EchoParams | None = None,
         seed: int = FIXTURE_SEED, with_pilot: bool = True) -> Path:
    sender, echoer, _ = _logs(root, params, seed) if not (root / "logs").exists() else (
        root / "logs" / "sender.csv", root / "logs" / "echoer.csv", None)
    out = root / out_name
    args = ["--sender", sender, "--echoer", echoer, "--status", status, "--output", out,
            "--n-boot", N_BOOT, "--seeds", 1, *extra]
    if with_pilot:
        args.append("--pilot")
    assert _intake(*args) == 0
    return out


def _row(md: str, label: str) -> tuple[float, float, float]:
    """(estimate, low, high) of one parameter row in the fit table of INTAKE.md."""
    match = re.search(rf"^\| {re.escape(label)} \| (\S+) \| (\S+) to (\S+) \|$", md, re.MULTILINE)
    assert match, f"no row {label!r} in INTAKE.md"
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _inputs_table(md: str) -> dict[str, str]:
    block = md.split("## Inputs")[1].split("##")[0]
    return dict(re.findall(r"^\| (\S+) \| ([0-9a-f]{64}) \|", block, re.MULTILINE))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    sender, echoer, truth = _logs(root)
    before = (_sha(sender), _sha(echoer))
    out = _run(root, "out")
    return {"root": root, "out": out, "truth": truth, "sender": sender, "echoer": echoer, "before": before,
            "md": (out / "INTAKE.md").read_text(encoding="utf-8")}


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("measured")
    out = _run(root, "out", "measured", *MEASURED_ARGS)
    return {"out": out, "md": (out / "INTAKE.md").read_text(encoding="utf-8")}


# ---------------------------------------------------------------- the summary against the known settings

def test_every_stage_ran_and_left_its_output(synthetic):
    out = synthetic["out"]
    for name in ("trace.json", "fit.json", "burst_bridge.json", "burst_bridge.txt", "INTAKE.md",
                 "s6/summary.json", "s6/per_seed.csv", "s6/manifest.json"):
        assert (out / name).is_file(), name


def test_summary_values_sit_inside_the_fits_intervals_and_match_the_known_settings(synthetic):
    """Delay, jitter and the burst chain, read back from INTAKE.md itself, against truth.json.

    The fixture's truth: one-way delay 2.5 ms, jitter 0.8 ms, burst start 0.02, burst end 0.3, so the
    stationary loss is 0.0637 and the mean loss run is 3.40 packets. Each 95% interval misses its truth about
    5 times in 100, so one seed can miss one row; the seed is fixed, so this is a fixed check, not a flaky one.
    """
    expected = synthetic["truth"]["expected"]
    md = synthetic["md"]
    checks = [("one-way delay (ms)", 1000 * expected["echo_visible_delay_s"]),
              ("one-way delay jitter (ms)", 1000 * expected["echo_visible_jitter_s"]),
              ("burst start probability", expected["outbound_p_received_to_lost"]),
              ("burst end probability", 0.3),
              ("stationary loss rate", expected["outbound_stationary_loss"]),
              ("mean loss run (packets)", expected["outbound_mean_loss_run_packets"])]
    for label, truth in checks:
        estimate, low, high = _row(md, label)
        assert low <= truth <= high, f"{label}: truth {truth} outside {low} to {high} (estimate {estimate})"
        assert low <= estimate <= high


def test_summary_estimates_are_the_fits_own_numbers(synthetic):
    fit = json.loads((synthetic["out"] / "fit.json").read_text(encoding="utf-8"))
    estimate, _, _ = _row(synthetic["md"], "one-way delay (ms)")
    assert estimate == pytest.approx(1000 * fit["delay"]["one_way_estimate"]["delay_s"]["value"], abs=6e-5)
    start, _, _ = _row(synthetic["md"], "burst start probability")
    assert start == pytest.approx(fit["loss"]["burst_chain"]["burst_start_probability"]["value"], abs=6e-5)


def test_the_rerun_uses_the_fitted_link(synthetic):
    fit = json.loads((synthetic["out"] / "fit.json").read_text(encoding="utf-8"))
    summary = json.loads((synthetic["out"] / "s6" / "summary.json").read_text(encoding="utf-8"))
    link = summary["link"]
    assert link["delay_s"] == pytest.approx(fit["s6_deliver"]["delay_s"])
    assert link["loss_model"] == fit["s6_deliver"]["loss_model"] == "burst"
    assert summary["design"]["pilot"] is True and summary["design"]["n_seeds"] == 1
    assert len(summary["cells"]) == 4


def test_inputs_are_hashed_and_never_modified(synthetic):
    table = _inputs_table(synthetic["md"])
    assert table["sender.csv"] == synthetic["before"][0]
    assert table["echoer.csv"] == synthetic["before"][1]
    assert (_sha(synthetic["sender"]), _sha(synthetic["echoer"])) == synthetic["before"]


def test_output_hashes_match_the_files(synthetic):
    block = synthetic["md"].split("## Output files")[1]
    rows = dict(re.findall(r"^\| (\S+) \| ([0-9a-f]{64}) \|$", block, re.MULTILINE))
    assert "fit.json" in rows and "s6/summary.json" in rows and "trace.json" in rows
    for name, digest in rows.items():
        assert _sha(synthetic["out"] / name) == digest
    assert "INTAKE.md" not in rows


def test_two_runs_give_the_same_files_and_the_same_summary(synthetic):
    again = _run(synthetic["root"], "again")
    first, second = synthetic["out"], again
    names = sorted(p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(second).as_posix() for p in second.rglob("*") if p.is_file())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


# ---------------------------------------------------------------- status

def test_a_synthetic_run_says_synthetic_everywhere(synthetic):
    out, md = synthetic["out"], synthetic["md"]
    assert "Overall status: synthetic" in md
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "s6" / "summary.json").read_text(encoding="utf-8"))
    assert trace["status"] == fit["source_trace"]["status"] == summary["provenance"]["input_status"] == "synthetic"
    assert summary["status"].startswith("synthetic investigation")


def test_a_fixture_run_is_labelled_fixture_not_measured(tmp_path):
    out = _run(tmp_path, "out", "fixture")
    md = (out / "INTAKE.md").read_text(encoding="utf-8")
    fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "s6" / "summary.json").read_text(encoding="utf-8"))
    assert "Overall status: fixture" in md
    assert fit["source_trace"]["status"] == summary["provenance"]["input_status"] == "fixture"
    assert summary["status"].startswith("synthetic investigation")  # a fixture is never upgraded


def test_a_measured_run_carries_its_status_from_the_trace_to_the_rerun(measured):
    out, md = measured["out"], measured["md"]
    assert "Overall status: measured" in md
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "s6" / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "s6" / "manifest.json").read_text(encoding="utf-8"))
    assert trace["status"] == "measured"
    assert fit["source_trace"]["status"] == "measured"
    assert summary["provenance"]["input_status"] == manifest["provenance"]["input_status"] == "measured"
    assert summary["status"] == "investigation: link from a measured fit, sensor simulated"
    prov = trace["provenance"]
    assert (prov["operator"], prov["code_revision"], prov["captured_at"]) == (
        "bench-test", "abc1234", "2026-11-02T10:00:00-06:00")


def test_the_summary_says_the_sensor_is_simulated_and_the_numbers_are_a_pilot(measured):
    md = measured["md"]
    assert "sensor simulated" in md
    assert "not a result" in md.lower()


def test_a_stage_that_changes_the_status_stops_the_run(tmp_path, monkeypatch, capsys):
    sender, echoer, _ = _logs(tmp_path)
    real = tool._import_trace

    def upgraded(*args, **kwargs):
        trace = real(*args, **kwargs)
        trace["status"] = "measured"  # a stage that quietly upgrades the label
        return trace

    monkeypatch.setattr(tool, "_import_trace", upgraded)
    out = tmp_path / "out"
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out,
                   "--n-boot", N_BOOT, "--seeds", 1, "--pilot")
    assert code == 2
    assert "status changed" in capsys.readouterr().err
    assert not out.exists()


# ---------------------------------------------------------------- measured needs provenance, before any stage

@pytest.mark.parametrize("missing", PROVENANCE_FLAGS)
def test_a_measured_run_without_one_provenance_field_is_refused_before_any_stage(tmp_path, monkeypatch, capsys,
                                                                                  missing):
    sender, echoer, _ = _logs(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("a stage ran")

    monkeypatch.setattr(tool, "_run_stages", boom)
    args = list(MEASURED_ARGS)
    index = args.index(missing)
    del args[index:index + 2]
    out = tmp_path / "out"
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "measured", "--output", out, *args)
    assert code != 0
    assert missing in capsys.readouterr().err
    assert not out.exists()


def test_a_measured_run_with_no_provenance_at_all_names_every_missing_field(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "measured", "--output", tmp_path / "out")
    err = capsys.readouterr().err
    assert code == 2
    for flag in PROVENANCE_FLAGS:
        assert flag in err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)
def test_a_blank_provenance_value_counts_as_missing(tmp_path, capsys, flag):
    sender, echoer, _ = _logs(tmp_path)
    args = list(MEASURED_ARGS)
    args[args.index(flag) + 1] = "   "
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "measured", "--output", tmp_path / "out",
                   *args)
    assert code == 2 and flag in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_provenance_flags_on_a_synthetic_run_are_not_needed(tmp_path):
    _run(tmp_path, "out", "synthetic")  # no provenance flags, and it runs


# ---------------------------------------------------------------- the two burst verdicts side by side

def test_both_burst_verdicts_are_reported_and_neither_is_chosen(synthetic):
    md = synthetic["md"]
    section = md.split("## Burst verdicts")[1].split("\n## ")[0]
    assert "tools/fit_espnow_log.py" in section and "tools/burst_bridge.py" in section
    assert "burst if z > 1.645" in section and "bursty if z >= 2.0" in section
    assert "burst model" in section
    assert "bursty" in section
    assert "agree" in section.lower()
    assert "not combined" in md.lower() or "neither" in section.lower()


def test_the_two_verdicts_are_the_tools_own_outputs(synthetic):
    fit = json.loads((synthetic["out"] / "fit.json").read_text(encoding="utf-8"))
    bridge = json.loads((synthetic["out"] / "burst_bridge.json").read_text(encoding="utf-8"))
    z_fit = fit["loss"]["independence_check"]["approx_z"]
    z_bridge = bridge["gilbert"]["burstiness"]["approx_z"]
    assert z_fit == pytest.approx(z_bridge)  # the same statistic, only the cutoffs differ
    assert f"{z_fit:.3f}" in synthetic["md"]


def test_a_log_between_the_two_cutoffs_is_reported_as_a_disagreement(tmp_path):
    out = _run(tmp_path, "out", params=BAND_PARAMS, seed=BAND_SEED)
    fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    bridge = json.loads((out / "burst_bridge.json").read_text(encoding="utf-8"))
    z = bridge["gilbert"]["burstiness"]["approx_z"]
    assert 1.645 < z < 2.0  # the fixture really is in the band
    assert fit["loss"]["model_choice"]["loss_model"] == "burst"
    assert bridge["verdict"].startswith("consistent with independent loss")
    md = (out / "INTAKE.md").read_text(encoding="utf-8")
    section = md.split("## Burst verdicts")[1].split("\n## ")[0]
    assert "disagree" in section.lower()
    assert "the two rules agree" not in section.lower()
    # Both verdicts are still shown; the report does not pick one.
    assert "burst model" in section and "consistent with independent loss" in section


@pytest.mark.parametrize(("fit_burst", "bridge_text", "agree"), [
    (True, "bursty (loss runs longer than independent loss would give)", True),
    (False, "consistent with independent loss", True),
    (True, "consistent with independent loss", False),
    (False, "bursty (loss runs longer than independent loss would give)", False),
    (False, "anti-bursty (loss runs shorter than independent loss would give)", True),
    (True, "not enough complete loss runs to tell", False),
])
def test_agreement_rule(fit_burst, bridge_text, agree):
    assert tool.verdicts_agree(fit_burst, bridge_text) is agree


# ---------------------------------------------------------------- raw captures go through the extractor

def _raw(path: Path, clean: Path, banner: bytes = b"ets Jun  8 2016 00:22:57\r\nrst:0x1 (POWERON_RESET)\r\n") -> None:
    text = clean.read_bytes().decode("utf-8")
    path.write_bytes(banner + text.replace("\n", "\r\n").encode("utf-8"))


def test_raw_captures_give_the_same_fit_as_the_clean_logs(tmp_path):
    sender, echoer, _ = _logs(tmp_path)
    clean = _run(tmp_path, "clean")
    _raw(tmp_path / "sender_raw.txt", sender)
    _raw(tmp_path / "echoer_raw.txt", echoer)
    out = tmp_path / "from_raw"
    code = _intake("--sender-raw", tmp_path / "sender_raw.txt", "--echoer-raw", tmp_path / "echoer_raw.txt",
                   "--status", "synthetic", "--output", out, "--n-boot", N_BOOT, "--seeds", 1, "--pilot")
    assert code == 0
    assert (out / "fit.json").read_bytes() == (clean / "fit.json").read_bytes()
    assert (out / "logs" / "sender.csv").read_bytes() == sender.read_bytes()
    md = (out / "INTAKE.md").read_text(encoding="utf-8")
    assert _inputs_table(md)["sender_raw.txt"] == _sha(tmp_path / "sender_raw.txt")
    assert (out / "logs" / "sender.audit.json").is_file()
    assert "boot" in md.lower() or "left out" in md.lower()


def test_a_raw_capture_with_two_blocks_is_refused_and_points_at_split(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    text = sender.read_bytes()
    (tmp_path / "two.txt").write_bytes(text + b"boot again\n" + text)
    _raw(tmp_path / "echoer_raw.txt", echoer)
    code = _intake("--sender-raw", tmp_path / "two.txt", "--echoer-raw", tmp_path / "echoer_raw.txt",
                   "--status", "synthetic", "--output", tmp_path / "out")
    assert code == 2
    err = capsys.readouterr().err
    assert "second format line" in err and "--split" in err
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- bad input and bad output

def test_a_damaged_echoer_log_stops_the_run_and_leaves_no_folder(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    lines = echoer.read_text(encoding="utf-8").splitlines()
    lines[12] = "not,a,row"
    echoer.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out)
    assert code == 2
    assert "echoer.csv" in capsys.readouterr().err
    assert not out.exists()


def test_a_failure_in_an_existing_empty_output_folder_empties_it_again(tmp_path):
    sender, echoer, _ = _logs(tmp_path)
    echoer.write_text("garbage\n", encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    out.mkdir()
    assert _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out) == 2
    assert out.is_dir() and not any(out.iterdir())


def test_a_non_empty_output_folder_is_refused_and_left_alone(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine\n", encoding="utf-8")
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out)
    assert code == 2 and "not empty" in capsys.readouterr().err
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine\n"


def test_output_inside_sim_runs_is_refused(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    out = REPO_ROOT / "sim" / "runs" / "intake-espnow-test-should-not-exist"
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out)
    assert code == 2 and "sim/runs" in capsys.readouterr().err
    assert not out.exists()


def test_an_input_inside_the_output_folder_is_refused(tmp_path, capsys):
    sender, echoer, _ = _logs(tmp_path)
    code = _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", sender.parent)
    assert code == 2
    assert "output folder" in capsys.readouterr().err
    assert sender.is_file() and echoer.is_file()


def test_a_missing_input_file_is_refused(tmp_path, capsys):
    _, echoer, _ = _logs(tmp_path)
    code = _intake("--sender", tmp_path / "nope.csv", "--echoer", echoer, "--status", "synthetic",
                   "--output", tmp_path / "out")
    assert code == 2 and "nope.csv" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("args", [
    ["--sender", "a.csv"],  # no echoer
    ["--echoer", "b.csv"],  # no sender
    ["--sender", "a.csv", "--echoer", "b.csv", "--sender-raw", "c.txt", "--echoer-raw", "d.txt"],  # both kinds
    ["--sender", "a.csv", "--echoer-raw", "d.txt"],  # mixed
])
def test_exactly_one_matching_pair_of_inputs_is_needed(tmp_path, capsys, args):
    code = _intake(*args, "--status", "synthetic", "--output", tmp_path / "out")
    assert code == 2
    assert "sender" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_trace_id_has_a_fixed_default_and_can_be_given(tmp_path):
    sender, echoer, _ = _logs(tmp_path)
    out = tmp_path / "out"
    assert _intake("--sender", sender, "--echoer", echoer, "--status", "synthetic", "--output", out,
                   "--trace-id", "bench-a", "--n-boot", N_BOOT, "--seeds", 1, "--pilot") == 0
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    assert trace["trace_id"] == "bench-a"
    other = _run(tmp_path, "default")
    assert json.loads((other / "trace.json").read_text(encoding="utf-8"))["trace_id"] == tool.DEFAULT_TRACE_ID


def test_no_local_paths_in_the_summary_or_the_outputs(synthetic):
    for path in synthetic["out"].rglob("*"):
        if path.is_file() and path.suffix in {".md", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8")
            assert str(synthetic["root"]) not in text, path.name
            assert "/Users/" not in text, path.name
