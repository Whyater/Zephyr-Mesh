"""The Windows update check and smoke error paths, without building a real window."""
from pathlib import Path
import threading
import time
import sys
from types import ModuleType, SimpleNamespace

import pytest

from desktop.windows_preview import WindowsReplayApp


class _Label:
    def __init__(self):
        self.text = None

    def configure(self, text):
        self.text = text


class _Root:
    """Records after() callbacks and which thread scheduled them."""

    def __init__(self):
        self.pending = []
        self.threads = set()
        self.destroyed = False

    def after(self, ms, callback=None):
        self.threads.add(threading.get_ident())
        self.pending.append(callback)
        return "after#1"

    def destroy(self):
        self.destroyed = True


def _drive(app, dialogs, *, count=None, timeout_s=5.0):
    """Run scheduled callbacks on this (the 'Tk') thread until `count` dialogs exist."""
    count = len(dialogs) + 1 if count is None else count
    deadline = time.monotonic() + timeout_s
    while len(dialogs) < count and time.monotonic() < deadline:
        if app.root.pending:
            callback = app.root.pending.pop(0)
            if callback is not None:
                callback()
        else:
            time.sleep(0.01)
    return dialogs


def _fake_messagebox(monkeypatch):
    """A stand-in tkinter.messagebox, so these tests run without a Tk build or a display."""
    messagebox = ModuleType("tkinter.messagebox")
    tkinter = ModuleType("tkinter")
    tkinter.messagebox = messagebox
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", messagebox)
    return messagebox


@pytest.fixture
def harness(monkeypatch):
    import tools.release_updater as updater

    messagebox = _fake_messagebox(monkeypatch)
    dialogs = []
    messagebox.showinfo = lambda title, message, **kw: dialogs.append(("info", message))
    messagebox.showwarning = lambda title, message, **kw: dialogs.append(("warning", message))
    calls = []

    def install(client_check):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def platform_name():
                return "windows-x86_64"

            def check(self, current, platform=None):
                calls.append(current)
                return client_check()

        monkeypatch.setattr(updater, "GitHubReleaseClient", FakeClient)

    app = SimpleNamespace(notice=_Label(), root=_Root())
    return app, dialogs, install, calls


def test_result_reaches_the_tk_thread_and_the_worker_never_calls_tk(harness):
    app, dialogs, install, _calls = harness
    install(lambda: (SimpleNamespace(tag="v0.2.2"), object(), False))
    WindowsReplayApp._check_update(app)
    assert _drive(app, dialogs) == [("info", "Up to date: v0.2.2.")]
    assert app.root.threads == {threading.get_ident()}, "only the Tk thread may call root.after"
    assert app.notice.text == ""


def test_second_click_while_checking_does_not_start_another_check(harness):
    app, dialogs, install, calls = harness
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return SimpleNamespace(tag="v0.2.2"), object(), False

    install(slow)
    WindowsReplayApp._check_update(app)
    WindowsReplayApp._check_update(app)
    gate.set()
    _drive(app, dialogs)
    _drive(app, dialogs, count=2, timeout_s=0.5)  # give a second check time to report, if one started
    assert len(dialogs) == 1
    assert len(calls) == 1
    WindowsReplayApp._check_update(app)  # finished, so a new click works again
    _drive(app, dialogs)
    assert len(dialogs) == 2


def test_network_failure_shows_a_short_plain_reason(harness):
    from urllib.error import URLError
    import socket
    from tools.release_updater import UpdateError, describe_update_error

    app, dialogs, install, _calls = harness

    def offline():
        raise UpdateError(describe_update_error(URLError(socket.gaierror(8, "nodename nor servname provided"))))

    install(offline)
    WindowsReplayApp._check_update(app)
    (kind, message), = _drive(app, dialogs)
    assert kind == "warning"
    assert message == "Update check failed: Could not reach GitHub. Check the internet connection."


def test_unexpected_error_text_is_not_raw_python(harness):
    app, dialogs, install, _calls = harness

    def broken():
        raise KeyError("tag_name")

    install(broken)
    WindowsReplayApp._check_update(app)
    (_kind, message), = _drive(app, dialogs)
    assert "'tag_name'" not in message
    assert message == "Update check failed: unexpected KeyError"


def _viewer(smoke):
    app = SimpleNamespace(_smoke_mode=smoke, root=None)
    app._evidence_loader = WindowsReplayApp._evidence_loader
    app._plain_open_error = WindowsReplayApp._plain_open_error
    app._report_open_failure = lambda title, message: WindowsReplayApp._report_open_failure(app, title, message)
    return app


def test_smoke_run_prints_open_errors_instead_of_showing_a_dialog(tmp_path, capsys, monkeypatch):
    messagebox = _fake_messagebox(monkeypatch)
    messagebox.showerror = lambda *a, **k: pytest.fail("a dialog would hang a --smoke run")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert WindowsReplayApp._show_evidence_report(_viewer(True), bad) is False
    assert WindowsReplayApp._show_evidence_report(_viewer(True), tmp_path / "missing.json") is False
    err = capsys.readouterr().err
    assert "Evidence report: The file bad.json is not valid JSON" in err
    assert "Evidence report: The file missing.json was not found." in err
    assert "Errno" not in err


def test_open_errors_read_the_same_for_both_viewers(tmp_path):
    missing = tmp_path / "missing.json"
    from desktop.scenario_run import ScenarioRunFormatError

    assert WindowsReplayApp._plain_open_error(FileNotFoundError(2, "No such file"), missing, "evidence report") == "The file missing.json was not found."
    assert WindowsReplayApp._plain_open_error(ScenarioRunFormatError("scenario run must be a regular file"), missing, "scenario run") == "The file missing.json was not found."
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    assert "not UTF-8 text" in WindowsReplayApp._plain_open_error(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), binary, "scenario run")


def test_headless_missing_scenario_is_a_message_not_a_traceback(tmp_path, capsys):
    from desktop.windows_preview import main

    assert main(["--headless", "--scenario-run", str(tmp_path / "missing.json")]) == 2
    err = capsys.readouterr().err
    assert "missing.json was not found" in err and "Traceback" not in err
