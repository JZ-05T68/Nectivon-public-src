"""Skin changes preserve content and remain available offline across restarts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.appearance import DEFAULT_THEME, THEMES, read_theme, theme_tokens, write_theme
from src.font_size_preferences import read_font_level, write_font_level


def test_preferences_are_independent_and_do_not_initialize_data(tmp_path: Path) -> None:
    root = tmp_path / "new"
    assert read_theme(root) == DEFAULT_THEME
    assert not root.exists()
    root.mkdir()
    content = root / "review.md"
    content.write_text("人工复盘 $p=F/A$", encoding="utf-8")
    before = content.read_bytes()
    for theme in THEMES:
        write_theme(root, theme)
        write_font_level(root, 5)
        assert read_theme(root) == theme
        assert read_font_level(root) == 5
        assert content.read_bytes() == before
    write_theme(root, DEFAULT_THEME)
    assert read_theme(root) == DEFAULT_THEME
    assert set(path.name for path in root.iterdir()) == {
        "appearance.json", "ui_preferences.json", "review.md",
    }


@pytest.mark.parametrize("payload", ["[]", "{", '{"theme":"unknown"}', '{"theme":[]}'])
def test_invalid_local_preference_falls_back(tmp_path: Path, payload: str) -> None:
    (tmp_path / "appearance.json").write_text(payload, encoding="utf-8")
    assert read_theme(tmp_path) == DEFAULT_THEME
    with pytest.raises(ValueError):
        write_theme(tmp_path, "unknown")


def test_failed_atomic_save_keeps_previous_choice(tmp_path: Path, monkeypatch) -> None:
    import src.appearance as appearance

    write_theme(tmp_path, "深空")

    def fail(*args) -> None:
        raise PermissionError("test")

    monkeypatch.setattr(appearance.os, "replace", fail)
    with pytest.raises(PermissionError):
        write_theme(tmp_path, "纸页")
    assert read_theme(tmp_path) == "深空"
    assert not list(tmp_path.glob("*.tmp"))


def _luminance(value: str) -> float:
    values = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
    return sum(a * b for a, b in zip(channels, (.2126, .7152, .0722), strict=True))


@pytest.mark.parametrize("theme", THEMES)
def test_text_and_accent_contrast(theme: str) -> None:
    palette = theme_tokens(theme)
    for ink in ("text", "muted", "accent", "note_red", "note_blue"):
        a, b = sorted((_luminance(palette[ink]), _luminance(palette["surface"])))
        assert (b + .05) / (a + .05) >= 4.5


def test_annotation_palette_changes_only_dark_presentation() -> None:
    from src.appearance import theme_css

    for theme in ("清朗", "纸页"):
        assert theme_tokens(theme)["note_red"] == "#D62828"
        assert theme_tokens(theme)["note_blue"] == "#2563EB"
    assert "--ekb-note-red:#ff9292" in theme_css("深空")
    assert "--ekb-note-blue:#8bbcff" in theme_css("深空")


def test_save_reload_and_restore_controls(tmp_path: Path) -> None:
    source = ("from pathlib import Path\n"
              "from src.appearance_ui import render_appearance\n"
              f"render_appearance(Path({str(tmp_path)!r}))")
    app = AppTest.from_string(source).run()
    app.radio[0].set_value("深空")
    app.button[0].click().run()
    assert not app.exception
    assert read_theme(tmp_path) == "深空"
    reloaded = AppTest.from_string(source).run()
    assert reloaded.radio[0].value == "深空"
    reloaded.button[1].click().run()
    assert read_theme(tmp_path) == DEFAULT_THEME
    assert json.loads((tmp_path / "appearance.json").read_text(encoding="utf-8")) == {
        "theme": DEFAULT_THEME,
    }


def test_broken_config_does_not_make_appearance_a_startup_dependency(monkeypatch) -> None:
    import src.workspace_ui as workspace

    def fail():
        raise ValueError("synthetic configuration failure")

    monkeypatch.setattr(workspace.config, "runtime_settings", fail)
    monkeypatch.setattr(workspace, "render_voice_input", lambda: None)
    monkeypatch.setattr(workspace, "_render_links", lambda *args: None)
    monkeypatch.setattr(workspace, "_render_home_topbar", lambda *args: None)
    app = AppTest.from_string(
        "from src.workspace_ui import render_workspace\nrender_workspace('app.py')"
    ).run()
    assert not app.exception
    assert any("--ekb-primary-ink:#ffffff" in item.value for item in app.markdown)
