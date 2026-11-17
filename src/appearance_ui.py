"""Shared appearance controls with explicit local save and reset."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.appearance import DEFAULT_THEME, THEMES, read_theme, theme_tokens, write_theme


def render_appearance(data_dir: Path) -> None:
    """Offer the same three visual skins without changing any business data."""
    current = read_theme(data_dir)
    st.subheader("外观")
    st.caption("选择适合当前工作的色彩。只保存本机外观偏好，所有学习内容共用原有组件。")
    previews = []
    for name, detail in zip(THEMES, ("简洁浅色，轻松开始", "柔和深色，保持专注",
                                   "暖色纸面，慢慢读懂"), strict=True):
        palette = theme_tokens(name)
        previews.append(
            f'<div class="ekb-skin-preview" style="--preview-bg:{palette["background"]};'
            f'--preview-panel:{palette["surface"]};--preview-accent:{palette["accent"]};'
            f'--preview-border:{palette["border"]}"><div class="ekb-skin-mini">'
            '<aside><i></i><i></i><i></i></aside><main><i></i><i></i><b></b></main></div>'
            f'<strong>{name}</strong><small>{detail}</small></div>'
        )
    st.markdown('<div class="ekb-skin-previews">' + "".join(previews) + '</div>',
                unsafe_allow_html=True)
    with st.form("appearance_preferences", border=True):
        selected = st.radio("界面皮肤", THEMES, index=THEMES.index(current), horizontal=True)
        save, reset = st.columns(2)
        submitted = save.form_submit_button("保存外观", type="primary", width="stretch")
        restored = reset.form_submit_button("恢复默认皮肤", width="stretch")
    if submitted or restored:
        try:
            write_theme(data_dir, DEFAULT_THEME if restored else selected)
        except OSError as exc:
            st.error(f"外观未保存，请检查本地目录的写入权限：{exc}")
        else:
            st.session_state["appearance_saved_notice"] = True
            st.rerun()
    if st.session_state.pop("appearance_saved_notice", False):
        st.success(f"已保存：{current}。重新打开应用后仍会保留。")
