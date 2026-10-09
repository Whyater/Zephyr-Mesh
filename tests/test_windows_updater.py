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
