from pathlib import Path

from tools.desktop.windows_updater import _pid_is_listed
from tools.desktop.windows_preview import updater_launch_path


def test_windows_updater_matches_exact_pid_in_tasklist_csv():
    output = '"alpha.exe","112","Console","1","10 K"\n'
    assert _pid_is_listed(output, 112)
    assert not _pid_is_listed(output, 12)


def test_embedded_updater_is_copied_outside_directory_before_swap(tmp_path, monkeypatch):
    install = tmp_path / "ZephyrMeshWindows"
    install.mkdir()
    helper = install / "ZephyrMeshUpdater.exe"
    helper.write_bytes(b"helper")
    monkeypatch.setattr("tools.desktop.windows_preview.tempfile.gettempdir", lambda: str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()

    launched = updater_launch_path(helper, install, pid=123)

    assert launched.parent == tmp_path / "temp"
    assert launched.read_bytes() == b"helper"
    assert helper.exists()


def test_sibling_updater_does_not_need_a_copy(tmp_path):
    install = tmp_path / "ZephyrMeshWindows"
    install.mkdir()
    helper = tmp_path / "ZephyrMeshUpdater.exe"
    helper.write_bytes(b"helper")

    assert updater_launch_path(helper, install, pid=123) == helper.resolve()


# ---- the detached helper, run for real on temporary folders ---------------
# Only three things are faked: the old app's PID wait, the self-test process
# (a .exe cannot run on every test machine), and Popen for the relaunch.


class _FakeProcess:
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


def _stage(tmp_path, marker="new"):
    """The layout download_and_stage writes: zephyr-update-*/payload/<app> plus staged.json."""
    import json
    from tools.release_updater import STAGED_RECORD, payload_digest

    root = tmp_path / "temp" / "zephyr-update-test"
    payload = root / "payload" / "ZephyrMeshWindows"
    (payload / "_internal").mkdir(parents=True)
    (payload / "ZephyrMeshWindows.exe").write_text("exe " + marker, encoding="utf-8")
    (payload / "_internal" / "marker.txt").write_text(marker, encoding="utf-8")
    record = {"payload": payload.name, "tree_sha256": payload_digest(payload), "archive_sha256": "0" * 64, "asset": "x.zip"}
    (root / STAGED_RECORD).write_text(json.dumps(record), encoding="utf-8")
    return payload


def _install(tmp_path):
    install = tmp_path / "apps" / "ZephyrMeshWindows"
    (install / "_internal").mkdir(parents=True)
    (install / "ZephyrMeshWindows.exe").write_text("exe old", encoding="utf-8")
    (install / "_internal" / "marker.txt").write_text("old", encoding="utf-8")
    return install


def _run_helper(monkeypatch, tmp_path, *, wait=lambda pid: None, self_test=lambda exe: None, exit_codes=(None,), staged=None):
    import tools.desktop.windows_updater as helper
    import tools.release_updater as updater

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(updater, "default_backup_dir", lambda install_dir: tmp_path / "appdata" / "ZephyrMesh" / "update-backup" / "ZephyrMeshWindows")
    monkeypatch.setattr(updater, "run_self_test", self_test)
    monkeypatch.setattr(updater, "STARTUP_GRACE_S", 0.3)
    monkeypatch.setattr(helper, "wait_for_process", lambda pid, timeout_s=45.0: wait(pid))
    launched = []
    codes = list(exit_codes)

    def popen(command, **kwargs):
        launched.append(command)
        return _FakeProcess(codes.pop(0) if codes else None)

    monkeypatch.setattr(updater.subprocess, "Popen", popen)
    staged = staged or _stage(tmp_path)
    install = _install(tmp_path)
    code = helper.main(["--pid", "4242", "--staged", str(staged), "--install-dir", str(install), "--launch", str(install / "ZephyrMeshWindows.exe")])
    return code, launched, install, staged


def _marker(install):
    return (install / "_internal" / "marker.txt").read_text(encoding="utf-8")


def test_successful_update_installs_deletes_the_backup_and_the_staging_folder(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    tested = []
    code, launched, install, staged = _run_helper(monkeypatch, tmp_path, self_test=tested.append)

    assert code == 0
    assert _marker(install) == "new"
    assert tested == [staged.resolve() / "ZephyrMeshWindows.exe"], "the new build is self-tested before the swap"
    assert launched == [[str(install / "ZephyrMeshWindows.exe")]]
    assert not (tmp_path / "appdata" / "ZephyrMesh" / "update-backup" / "ZephyrMeshWindows").exists()
    assert sorted(path.name for path in install.parent.iterdir()) == ["ZephyrMeshWindows"], "nothing is left beside the app"
    assert not staged.parent.parent.exists(), "the staging folder is removed"
    assert take_update_result() is None


def test_payload_whose_self_test_fails_is_never_installed(monkeypatch, tmp_path):
    from tools.release_updater import UpdateError, take_update_result

    def failing(executable):
        raise UpdateError("self-test failed: The bundled replay file is missing")

    code, launched, install, staged = _run_helper(monkeypatch, tmp_path, self_test=failing)

    assert code == 1
    assert _marker(install) == "old"
    assert launched == [[str(install / "ZephyrMeshWindows.exe")]], "the old app is opened again"
    message = take_update_result()
    assert "current version was kept" in message and "self-test failed" in message
    assert not staged.parent.parent.exists()


def test_new_app_that_crashes_at_launch_is_rolled_back(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    code, launched, install, _ = _run_helper(monkeypatch, tmp_path, exit_codes=(3, None))

    assert code == 1
    assert _marker(install) == "old"
    assert len(launched) == 2, "the new app, then the restored old one"
    assert "quit right after it started" in take_update_result()


def test_payload_changed_after_staging_is_refused(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    staged = _stage(tmp_path)
    (staged / "_internal" / "marker.txt").write_text("tampered", encoding="utf-8")
    code, _launched, install, _ = _run_helper(monkeypatch, tmp_path, staged=staged)

    assert code == 1
    assert _marker(install) == "old"
    assert "changed after it was verified" in take_update_result()


def test_staged_folder_outside_the_staging_layout_is_refused(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    loose = tmp_path / "somewhere" / "ZephyrMeshWindows"
    loose.mkdir(parents=True)
    (loose / "ZephyrMeshWindows.exe").write_text("exe", encoding="utf-8")
    code, _launched, install, _ = _run_helper(monkeypatch, tmp_path, staged=loose)

    assert code == 1
    assert _marker(install) == "old"
    assert "not in a verified staging folder" in take_update_result()
    assert loose.exists(), "cleanup never deletes a folder it did not create"


def test_move_that_fails_halfway_across_disks_puts_the_old_app_back(monkeypatch, tmp_path):
    import errno
    import os
    import shutil
    import tools.release_updater as updater
    from tools.release_updater import take_update_result

    real_replace, real_copytree = os.replace, shutil.copytree

    def replace(source, destination):
        if "payload" in Path(source).parts:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(source, destination)

    def copytree(source, destination, **kwargs):
        Path(destination, "_internal").mkdir(parents=True)
        Path(destination, "_internal", "marker.txt").write_text("partial", encoding="utf-8")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(updater.os, "replace", replace)
    monkeypatch.setattr(updater.shutil, "copytree", copytree)
    try:
        code, launched, install, _ = _run_helper(monkeypatch, tmp_path)
    finally:
        monkeypatch.setattr(updater.os, "replace", real_replace)
        monkeypatch.setattr(updater.shutil, "copytree", real_copytree)

    assert code == 1
    assert _marker(install) == "old"
    assert launched == [[str(install / "ZephyrMeshWindows.exe")]]
    assert "current version was kept" in take_update_result()
    assert not (tmp_path / "appdata" / "ZephyrMesh" / "update-backup" / "ZephyrMeshWindows").exists()


def test_timeout_waiting_for_old_app_does_not_start_a_second_copy(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    def never_exits(pid):
        raise TimeoutError("timed out waiting for process 4242")

    code, launched, install, _ = _run_helper(monkeypatch, tmp_path, wait=never_exits)

    assert code == 1
    assert launched == []
    assert _marker(install) == "old"
    assert "timed out" in take_update_result()


def test_unanswerable_process_check_does_not_start_a_second_copy(monkeypatch, tmp_path):
    from tools.release_updater import ProcessCheckError, take_update_result

    def broken_tasklist(pid):
        raise ProcessCheckError("tasklist did not answer within 10s")

    code, launched, install, _ = _run_helper(monkeypatch, tmp_path, wait=broken_tasklist)

    assert code == 1
    assert launched == []
    assert "tasklist did not answer" in take_update_result()


def test_process_check_uses_exact_pids_and_reports_tasklist_failures(monkeypatch):
    import subprocess
    import pytest
    import tools.release_updater as updater

    class Done:
        def __init__(self, code, out=""):
            self.returncode, self.stdout = code, out

    replies = {"ok": Done(0, '"alpha.exe","112","Console","1","10 K"\n'), "fail": Done(1)}
    mode = {"value": "ok"}

    def run(command, **kwargs):
        assert kwargs.get("timeout"), "tasklist must have a timeout"
        if mode["value"] == "hang":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return replies[mode["value"]]

    monkeypatch.setattr(updater.subprocess, "run", run)
    assert updater.process_running(112, windows=True)
    assert not updater.process_running(12, windows=True)
    for broken in ("fail", "hang"):
        mode["value"] = broken
        with pytest.raises(updater.ProcessCheckError):
            updater.process_running(12, windows=True)


def test_app_reads_and_clears_the_previous_update_failure(monkeypatch, tmp_path):
    from tools.desktop.windows_preview import _take_previous_update_failure
    from tools.release_updater import write_update_result

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert _take_previous_update_failure() is None
    write_update_result("The last update did not install: OSError: disk full")
    assert "disk full" in _take_previous_update_failure()
    assert _take_previous_update_failure() is None
