from __future__ import annotations

import sys
import types

from tools.desktop.ui_settings import load_appearance, save_appearance, system_prefers_dark


def test_missing_corrupt_and_round_trip_settings(tmp_path):
    assert load_appearance(tmp_path) == "dark"
    path = tmp_path / ".zephyr-mesh" / "settings.json"
    path.parent.mkdir()
    path.write_text("not json")
    assert load_appearance(tmp_path) == "dark"
    save_appearance("light", tmp_path)
    assert load_appearance(tmp_path) == "light"
    save_appearance("unknown", tmp_path)
    assert load_appearance(tmp_path) == "dark"


def test_fake_windows_registry(monkeypatch):
    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = types.SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        OpenKey=lambda *_args: Key(),
        QueryValueEx=lambda *_args: (0, 0),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert system_prefers_dark() is True
    fake.QueryValueEx = lambda *_args: (1, 0)
    assert system_prefers_dark() is False
