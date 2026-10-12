from pathlib import Path

from desktop.windows_updater import _pid_is_listed
from desktop.windows_preview import updater_launch_path


def test_windows_updater_matches_exact_pid_in_tasklist_csv():
    output = '"alpha.exe","112","Console","1","10 K"\n'
    assert _pid_is_listed(output, 112)
    assert not _pid_is_listed(output, 12)


def test_embedded_updater_is_copied_outside_directory_before_swap(tmp_path, monkeypatch):
    install = tmp_path / "ZephyrMeshWindows"
    install.mkdir()
    helper = install / "ZephyrMeshUpdater.exe"
    helper.write_bytes(b"helper")
    monkeypatch.setattr("desktop.windows_preview.tempfile.gettempdir", lambda: str(tmp_path / "temp"))
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


# A failed swap must leave a readable trace and put the old app back on screen.
def _run_helper(monkeypatch, tmp_path, *, apply, wait=lambda pid: None):
    import desktop.windows_updater as helper

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    launched = []
    monkeypatch.setattr(helper, "wait_for_process", lambda pid, timeout_s=45.0: wait(pid))
    monkeypatch.setattr(helper, "apply_staged_update", apply)
    monkeypatch.setattr(helper.subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    code = helper.main(["--pid", "4242", "--staged", str(tmp_path / "staged"), "--install-dir", str(tmp_path / "install"), "--launch", str(tmp_path / "install" / "ZephyrMeshWindows.exe")])
    return code, launched


def test_failed_swap_writes_a_result_file_and_relaunches_the_old_copy(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    def broken_swap(staged, install_dir):
        raise OSError("cannot move across volumes")

    code, launched = _run_helper(monkeypatch, tmp_path, apply=broken_swap)

    assert code == 1
    assert launched == [[str(tmp_path / "install" / "ZephyrMeshWindows.exe")]]
    message = take_update_result()
    assert message is not None and "cannot move across volumes" in message
    assert take_update_result() is None  # read once, then cleared


def test_successful_swap_relaunches_and_leaves_no_failure_record(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    code, launched = _run_helper(monkeypatch, tmp_path, apply=lambda staged, install_dir: tmp_path / "backup")

    assert code == 0
    assert len(launched) == 1
    assert take_update_result() is None


def test_timeout_waiting_for_old_app_does_not_start_a_second_copy(monkeypatch, tmp_path):
    from tools.release_updater import take_update_result

    def never_exits(pid):
        raise TimeoutError("timed out waiting for process 4242")

    code, launched = _run_helper(monkeypatch, tmp_path, apply=lambda staged, install_dir: None, wait=never_exits)

    assert code == 1
    assert launched == []
    assert "timed out" in take_update_result()


def test_app_reads_and_clears_the_previous_update_failure(monkeypatch, tmp_path):
    from desktop.windows_preview import _take_previous_update_failure
    from tools.release_updater import write_update_result

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert _take_previous_update_failure() is None
    write_update_result("The last update did not install: OSError: disk full")
    assert "disk full" in _take_previous_update_failure()
    assert _take_previous_update_failure() is None
