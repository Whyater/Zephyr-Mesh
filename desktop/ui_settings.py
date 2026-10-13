"""Small, fault-tolerant persistence for the Windows desktop appearance choice."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


APPEARANCES = ("system", "dark", "light")
DEFAULT_APPEARANCE = "dark"
SETTINGS_FILENAME = "settings.json"
# The real file is about 30 bytes. Anything far larger is not ours; refuse it before parsing.
MAX_SETTINGS_BYTES = 64 * 1024
_THEME_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def settings_path(home: str | os.PathLike[str] | None = None) -> Path:
    """Return the per-user settings file path."""
    base = Path(home) if home is not None else Path.home()
    return base / ".zephyr-mesh" / SETTINGS_FILENAME


def _valid(value: Any) -> str:
    return value if isinstance(value, str) and value in APPEARANCES else DEFAULT_APPEARANCE


def load_appearance(home: str | os.PathLike[str] | None = None) -> str:
    """Load the saved appearance, using dark when the file is absent or invalid."""
    path = settings_path(home)
    try:
        if path.stat().st_size > MAX_SETTINGS_BYTES:
            return DEFAULT_APPEARANCE
        with path.open("rb") as handle:
            raw = handle.read(MAX_SETTINGS_BYTES + 1)
        if len(raw) > MAX_SETTINGS_BYTES:
            return DEFAULT_APPEARANCE
        document = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, TypeError, RecursionError):
        # Missing, unreadable, not UTF-8, not JSON, or nested deep enough to exhaust the parser: use the default.
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


def resolve_theme(appearance: str, system_dark: bool | None = None) -> str:
    """Map an appearance choice to the theme to draw: System follows Windows, else Dark."""
    choice = _valid(appearance)
    if choice == "system":
        return "light" if system_dark is False else "dark"
    return choice


def system_reduced_motion() -> bool:
    """True when Windows has client-area animations turned off (Settings > Accessibility)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    except (ImportError, AttributeError, OSError):
        return False
    try:
        enabled = ctypes.c_int(1)
        get_client_area_animation = 0x1042
        if not user32.SystemParametersInfoW(get_client_area_animation, 0, ctypes.byref(enabled), 0):
            return False
        return enabled.value == 0
    except (AttributeError, OSError, TypeError, ValueError):
        return False


__all__ = [
    "APPEARANCES",
    "DEFAULT_APPEARANCE",
    "load_appearance",
    "load_settings",
    "resolve_theme",
    "save_appearance",
    "save_settings",
    "settings_path",
    "system_prefers_dark",
    "system_reduced_motion",
]
