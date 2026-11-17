"""The independent study route keeps the shared navigation without opening data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src import workspace_ui
from src.study_loop import ui as study_ui

ROOT = Path(__file__).resolve().parents[1]


def test_study_sidebar_remains_available_without_opening_either_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Render the actual route and shell without initializing either library."""
    monkeypatch.setattr(
        workspace_ui.config, "runtime_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setattr(study_ui, "hold_reason", lambda: "验收占位：未打开学习整理库。")

    def forbid_storage(*args: object, **kwargs: object) -> None:
        raise AssertionError("仅显示导航时不得初始化学习整理库")

    monkeypatch.setattr(study_ui, "StudyStore", forbid_storage)
    app = AppTest.from_file(ROOT / "pages" / "18_学习整理.py").run()

    assert not app.exception
    links = app.sidebar.get("page_link")
    labels = {link.proto.label for link in links}
    assert {"首页", "知识", "对话", "文件", "设置", "学习整理", "知识串联"} <= labels
    current = next(link for link in links if link.proto.label == "学习整理")
    assert current.proto.disabled
    assert app.title[0].value == "学习整理与主动复盘"
    assert app.info[0].value == "验收占位：未打开学习整理库。"
    assert not list(tmp_path.iterdir())
