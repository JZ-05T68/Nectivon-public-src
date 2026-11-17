"""Offline presentation preferences, separate from knowledge and study data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

LOGGER = logging.getLogger(__name__)
DEFAULT_THEME = "清朗"
THEMES = ("清朗", "深空", "纸页")
_PALETTES = {
    "清朗": ("light", "#f6f7fb", "#ffffff", "#fafbfe", "#eef0f7", "#e0e4ee",
           "#bdc5d7", "#202638", "#626d83", "#5161c7", "#4353ad", "#35458f",
           "#edf0ff", "#d6dcfa"),
    "深空": ("dark", "#121721", "#1b2330", "#18202c", "#253142", "#364457",
           "#4e6076", "#e5ebf4", "#acbbce", "#a8b5ff", "#c1cbff", "#d3daff",
           "#2b3453", "#46527e"),
    "纸页": ("light", "#f3eee4", "#fffbf2", "#faf5e9", "#eee5d5", "#ded2bb",
           "#c4b69c", "#3d352b", "#756653", "#866132", "#6e4e26", "#583c1b",
           "#f0e5ce", "#dfcba3"),
}


def read_theme(data_dir: Path) -> str:
    """Read without creating directories, schemas or files; fail to the default."""
    path = data_dir / "appearance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("theme") if isinstance(payload, dict) else None
        return value if isinstance(value, str) and value in THEMES else DEFAULT_THEME
    except FileNotFoundError:
        return DEFAULT_THEME
    except (OSError, ValueError) as exc:
        LOGGER.warning("外观偏好无法读取，已使用清朗皮肤（%s）", type(exc).__name__)
        return DEFAULT_THEME


def write_theme(data_dir: Path, theme: str) -> None:
    """Atomically save only appearance; never touch content or reading size."""
    if theme not in THEMES:
        raise ValueError("请选择清朗、深空或纸页皮肤。")
    path = data_dir / "appearance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".appearance.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"theme": theme}, temporary, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def active_theme() -> str:
    """Resolve the actual instance's preference without initializing business data."""
    from src.config import runtime_settings

    return read_theme(runtime_settings().data_dir)


def theme_tokens(theme: str = DEFAULT_THEME) -> dict[str, str]:
    """Return a fresh palette for the shared shell and local iframe canvases."""
    keys = ("scheme", "background", "surface", "subtle", "raised", "border", "edge",
            "text", "muted", "accent", "hover", "active", "soft", "accent_border")
    palette = dict(zip(keys, _PALETTES.get(theme, _PALETTES[DEFAULT_THEME]), strict=True))
    # Annotation semantics and stored markup stay unchanged; dark reading needs lighter ink.
    palette.update(note_red="#ff9292" if palette["scheme"] == "dark" else "#D62828",
                   note_blue="#8bbcff" if palette["scheme"] == "dark" else "#2563EB")
    return palette


def theme_css(theme: str) -> str:
    """Project a whitelisted palette to shared CSS; no global Streamlit mutation."""
    palette = theme_tokens(theme)
    aliases = {
        "gray-0": "surface", "gray-25": "subtle", "gray-50": "background",
        "gray-100": "raised", "gray-200": "border", "gray-300": "edge",
        "gray-500": "muted", "gray-700": "text", "gray-900": "text",
        "ink": "text", "muted": "muted", "line": "border", "surface": "surface",
        "green": "accent", "green-hover": "hover", "green-active": "active",
        "green-soft": "soft", "green-line": "accent_border",
        "note-red": "note_red", "note-blue": "note_blue",
    }
    variables = ";".join(f"--ekb-{name}:{palette[key]}" for name, key in aliases.items())
    primary_ink = "#171d37" if palette["scheme"] == "dark" else "#ffffff"
    return (f"<style>:root{{color-scheme:{palette['scheme']};{variables};"
            f"--ekb-primary-ink:{primary_ink};--ekb-reading-ink:{palette['text']};}}"
            "</style>")
