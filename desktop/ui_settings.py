"""Small, fault-tolerant persistence for the Windows desktop appearance choice."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


APPEARANCES = ("system", "dark", "light")
DEFAULT_APPEARANCE = "dark"
SETTINGS_FILENAME = "settings.json"
_THEME_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def settings_path(home: str | os.PathLike[str] | None = None) -> Path:
    """Return the per-user settings file path."""
    base = Path(home) if home is not None else Path.home()
    return base / ".zephyr-mesh" / SETTINGS_FILENAME


def _valid(value: Any) -> str:
    return value if isinstance(value, str) and value in APPEARANCES else DEFAULT_APPEARANCE


def load_appearance(home: str | os.PathLike[str] | None = None) -> str:
    """Load the saved appearance, using dark when the file is absent or invalid."""
    try:
        document = json.loads(settings_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return DEFAULT_APPEARANCE
    if not isinstance(document, Mapping):
        return DEFAULT_APPEARANCE
    return _valid(document.get("appearance"))


def save_appearance(appearance: str, home: str | os.PathLike[str] | None = None) -> Path:
    """Persist an appearance choice and return the written path."""
    value = _valid(appearance)
    path = settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"appearance": value}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_settings(home: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Return the settings document's supported values."""
    return {"appearance": load_appearance(home)}


def save_settings(settings: Mapping[str, Any], home: str | os.PathLike[str] | None = None) -> Path:
    """Persist supported settings from a mapping."""
    return save_appearance(settings.get("appearance", DEFAULT_APPEARANCE), home)


def system_prefers_dark() -> bool | None:
    """Read Windows' ``AppsUseLightTheme`` preference when the registry is available."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _THEME_KEY) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
    except (OSError, AttributeError, TypeError, ValueError):
        return None
    if value in (0, "0"):
        return True
    if value in (1, "1"):
        return False
    return None


__all__ = [
    "APPEARANCES",
    "DEFAULT_APPEARANCE",
    "load_appearance",
    "load_settings",
    "save_appearance",
    "save_settings",
    "settings_path",
    "system_prefers_dark",
]
