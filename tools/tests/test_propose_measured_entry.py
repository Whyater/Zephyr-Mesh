"""Tests for tools/propose_measured_entry.py.

The fake intake below is a separate, tiny program written for these tests: it hashes its one input into
result.txt and writes an INTAKE.md in the intake tools' table format. The checks name the truth they
recover: the proposal's hashes must equal hashes computed here from the files, and the proposed entry
must pass the real loader in tools/reproduce_all.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from tools import propose_measured_entry as pme
from tools import reproduce_all

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REPO_ROOT / "tools" / "reproduce_registry.json"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None and sys.platform == "win32",
                                reason="the replay runs registry commands with bash")

FAKE_INTAKE = r'''
import argparse, hashlib, pathlib
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--extra", action="store_true")
a = p.parse_args()
src, out = pathlib.Path(a.input), pathlib.Path(a.output)
out.mkdir(parents=True)
digest = hashlib.sha256(src.read_bytes()).hexdigest()
(out / "result.txt").write_text("input sha256 " + digest + "\n", encoding="utf-8")
if a.extra:
    (out / "extra.txt").write_text("made only with --extra\n", encoding="utf-8")
res = hashlib.sha256((out / "result.txt").read_bytes()).hexdigest()
(out / "INTAKE.md").write_text(
    "# Fake intake\n\nOverall status: synthetic\n\n## Inputs\n\n| File | sha256 | Role |\n|---|---|---|\n"
    "| " + src.name + " | " + digest + " | log |\n\n## Output files\n\n| File | sha256 |\n|---|---|\n"
    "| result.txt | " + res + " |\n", encoding="utf-8")
'''

COMMAND = "PYTHONPATH=. {python} tools/fake_intake.py --input data/bench/log.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A plain source folder with the fake intake and one input, replayed with --source-dir."""
    root = tmp_path / "tree"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "fake_intake.py").write_text(FAKE_INTAKE, encoding="utf-8")
    (root / "data" / "bench").mkdir(parents=True)
    (root / "data" / "bench" / "log.csv").write_text("t,rtt\n0,0.004\n1,0.005\n", encoding="utf-8")
    (root / "sim" / "runs").mkdir(parents=True)
    return root


def _intake(tree: Path, out: Path, *extra: str) -> Path:
    import subprocess
    subprocess.run([sys.executable, str(tree / "tools" / "fake_intake.py"), "--input",
                    str(tree / "data" / "bench" / "log.csv"), "--output", str(out), *extra], check=True)
    return out


def _propose(intake: Path, tree: Path, registry: Path = REAL_REGISTRY, *, command: str = COMMAND,
             run_dir: str = "sim/runs/zz-proposed-test") -> Path:
    return pme.propose(intake, run_dir=run_dir, intake_command=command, registry=registry,
                       repo_root=tree, source_dir=tree, python=sys.executable, timeout=120)


def test_proposal_passes_real_loader_and_registry_is_untouched(tree: Path, tmp_path: Path) -> None:
    before = REAL_REGISTRY.read_bytes()
    intake = _intake(tree, tmp_path / "intake")
    path = _propose(intake, tree)
    assert REAL_REGISTRY.read_bytes() == before
    proposal = json.loads(path.read_text(encoding="utf-8"))
    entry = proposal["entry"]
    assert set(entry) == {"run_dir", "command", "outputs"}
    assert entry["outputs"] == {"result.txt": _sha(intake / "result.txt")}
    assert proposal["inputs"] == [{"label": "log.csv", "path": "data/bench/log.csv",
                                   "sha256": _sha(tree / "data" / "bench" / "log.csv")}]
    assert proposal["volatile_not_hashed"] == ["INTAKE.md"]
    assert proposal["intake_status"] == "synthetic" and proposal["proposal_only"] is True
    # The real loader accepts the registry with the entry appended.
    trial = tmp_path / "registry.json"
    trial.write_text(json.dumps([*json.loads(before), entry]), encoding="utf-8")
    assert reproduce_all.load_registry(trial)[-1] == entry


def test_reproduce_all_regenerates_the_proposed_entry(tree: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    intake = _intake(tree, tmp_path / "intake")
    entry = json.loads(_propose(intake, tree).read_text(encoding="utf-8"))["entry"]
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([entry]), encoding="utf-8")
    monkeypatch.chdir(tree)
    code = reproduce_all.main(["--registry", str(registry), "--source-dir", str(tree), "--only",
                               "zz-proposed-test", "--workers", "1"])
    assert code == 0
    assert "zz-proposed-test: byte-identical" in capsys.readouterr().out


def test_refuses_folder_without_provenance(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    (intake / "INTAKE.md").unlink()
    with pytest.raises(pme.ProposalError, match="no provenance"):
        _propose(intake, tree)
    assert not (intake / pme.PROPOSAL_NAME).exists()


def test_refuses_intake_md_without_input_hashes(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    text = (intake / "INTAKE.md").read_text(encoding="utf-8")
    head, _, tail = text.partition("## Inputs")
    (intake / "INTAKE.md").write_text(head + "## Output files" + tail.split("## Output files", 1)[1],
                                      encoding="utf-8")
    with pytest.raises(pme.ProposalError, match="no provenance: INTAKE.md lists no inputs"):
        _propose(intake, tree)


def test_refuses_run_dir_already_in_registry(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    taken = reproduce_all.load_registry(REAL_REGISTRY)[0]["run_dir"]
    with pytest.raises(pme.ProposalError, match="already in the registry"):
        _propose(intake, tree, run_dir=taken)


def test_refuses_output_the_command_does_not_reproduce(tree: Path, tmp_path: Path) -> None:
    # Edit result.txt and its INTAKE.md hash together, so the folder agrees with itself but not with
    # what the command makes.
    intake = _intake(tree, tmp_path / "intake")
    old = _sha(intake / "result.txt")
    (intake / "result.txt").write_text("hand edited\n", encoding="utf-8")
    md = intake / "INTAKE.md"
    md.write_text(md.read_text(encoding="utf-8").replace(old, _sha(intake / "result.txt")), encoding="utf-8")
    with pytest.raises(pme.ProposalError, match="did not reproduce these outputs byte for byte: result.txt"):
        _propose(intake, tree)


def test_refuses_file_in_folder_the_command_does_not_make(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake", "--extra")
    with pytest.raises(pme.ProposalError, match="in the folder but not made by the command: extra.txt"):
        _propose(intake, tree)


def test_refuses_output_changed_since_intake(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    (intake / "result.txt").write_text("changed later\n", encoding="utf-8")
    with pytest.raises(pme.ProposalError, match="changed since the intake"):
        _propose(intake, tree)


def test_refuses_command_that_names_a_different_input(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    (tree / "data" / "bench" / "log.csv").write_text("t,rtt\n0,0.009\n", encoding="utf-8")
    with pytest.raises(pme.ProposalError, match="lists input log.csv"):
        _propose(intake, tree)


@pytest.mark.parametrize("command, message", [
    ("PYTHONPATH=. python tools/fake_intake.py --input data/bench/log.csv", r"\{python\}"),
    (COMMAND + " --output somewhere", "--output"),
    (COMMAND + " --output={out}", r"\{out\}"),
    ("PYTHONPATH=. {python} tools/fake_intake.py --input /abs/log.csv", "repo-relative"),
    ("PYTHONPATH=. {python} tools/fake_intake.py --input=C:/abs/log.csv", "repo-relative"),
    ("PYTHONPATH=. {python} tools/fake_intake.py --input ../log.csv", "repo-relative"),
])
def test_refuses_unsafe_commands(tree: Path, tmp_path: Path, command: str, message: str) -> None:
    intake = _intake(tree, tmp_path / "intake")
    with pytest.raises(pme.ProposalError, match=message):
        _propose(intake, tree, command=command)


def test_cli_refusal_exit_code(tree: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    intake = tmp_path / "empty"
    intake.mkdir()
    monkeypatch.chdir(REPO_ROOT)
    code = pme.main([str(intake), "--run-dir", "sim/runs/zz-x", "--intake-command", COMMAND,
                     "--source-dir", str(tree)])
    assert code == 2 and "no provenance" in capsys.readouterr().err


def test_refuses_file_the_command_makes_that_the_folder_lacks(tree: Path, tmp_path: Path) -> None:
    # The folder was made without --extra; the command adds it, so the replay makes extra.txt too.
    intake = _intake(tree, tmp_path / "intake")
    with pytest.raises(pme.ProposalError, match="made by the command but not in the folder: extra.txt"):
        _propose(intake, tree, command=COMMAND + " --extra")
    assert not (intake / pme.PROPOSAL_NAME).exists()


@pytest.mark.parametrize("run_dir", [
    "sim/runs/x;touch $ZM_MARK;#",
    "sim/runs/$(touch $ZM_MARK)",
    "sim/runs/x`touch $ZM_MARK`",
    "sim/runs/my run",
    "sim/runs/a/b",
    "sim/runs/Upper",
    "sim/runs/.hidden",
    "sim/runs/a..b",
    "sim/runs/",
    "runs/x",
])
def test_refuses_bad_run_dir_before_running_anything(tree: Path, tmp_path: Path, monkeypatch, request,
                                                     run_dir: str) -> None:
    # A harmless marker file shows whether any shell text in run_dir was executed. It sits in the
    # system temp folder, whose path has no spaces, so an unquoted $ZM_MARK would really be touched.
    marker_dir = Path(tempfile.mkdtemp(prefix="zm-mark-"))
    marker = marker_dir / "INJECTED"
    monkeypatch.setenv("ZM_MARK", marker.as_posix())
    request.addfinalizer(lambda: shutil.rmtree(marker_dir, ignore_errors=True))
    intake = _intake(tree, tmp_path / "intake")
    with pytest.raises(pme.ProposalError, match="run_dir must be exactly sim/runs/<name>"):
        _propose(intake, tree, run_dir=run_dir)
    assert not marker.exists()
    assert not (intake / pme.PROPOSAL_NAME).exists()


def test_shell_text_in_the_intake_command_is_not_executed(tree: Path, tmp_path: Path, monkeypatch) -> None:
    # Shell operators in the intake command become plain arguments, so nothing extra runs.
    marker = tmp_path / "INJECTED"
    monkeypatch.setenv("ZM_MARK", str(marker))
    intake = _intake(tree, tmp_path / "intake")
    with pytest.raises(pme.ProposalError):
        _propose(intake, tree, command=COMMAND + " ';' touch '$ZM_MARK'")
    assert not marker.exists()


def test_proposed_command_has_no_rm_and_quotes_the_stage(tree: Path, tmp_path: Path) -> None:
    intake = _intake(tree, tmp_path / "intake")
    command = json.loads(_propose(intake, tree).read_text(encoding="utf-8"))["entry"]["command"]
    assert "rm " not in command
    assert command.endswith("--output .intake-staging-zz-proposed-test && "
                            "cp -R .intake-staging-zz-proposed-test/. {out}/")


@pytest.mark.parametrize("where", ["tree", "elsewhere"])
def test_refuses_intake_folder_under_sim_runs(tree: Path, tmp_path: Path, where: str) -> None:
    base = tree if where == "tree" else tmp_path / "other-checkout"
    intake = _intake(tree, base / "sim" / "runs" / "zz-inplace")
    with pytest.raises(pme.ProposalError, match="outside sim/runs"):
        _propose(intake, tree)
    assert not (intake / pme.PROPOSAL_NAME).exists()
