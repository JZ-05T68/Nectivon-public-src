"""Small read-only views inside the existing learning page; no background work."""

from __future__ import annotations

import re

import streamlit as st

from src.time_display import format_beijing_time

from .display import render_saved_text
from .import_ui import render_source
from .insights import (
    ErrorEvidence,
    ReviewEvent,
    _instant,
    distribution,
    error_evidence,
    evidence_counts,
    question_nodes,
    recent_reviews,
)
from .models import OUTCOME_LABELS, Draft, StudyError
from .storage import StudyStore

ORIGINS = {"user_attempt": "本人确认的作答错误", "manual_marker": "用户标记错题"}


def _time(value: str | None) -> str:
    return format_beijing_time(value) + "（北京时间）" if _instant(value) else "时间未知"


def _title(item: Draft) -> str:
    return f"{item.subject or '科目未分类'} · {item.question.number or '未编号'} · " + (
        item.question.content[:70]
    )


def _literal_label(value: str) -> str:
    """Keep imported labels literal in markdown-capable buttons, including image links."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|>\-])", r"\\\1", value)


def _render_material(store: StudyStore, item: Draft, unit_id: str | None, prefix: str) -> None:
    """Resolve only the exact saved version and source IDs, including missing references."""
    render_saved_text(item.question.content)
    if unit_id:
        unit = next((q for q in question_nodes(item.question) if q.id == unit_id), None)
        if unit is None:
            st.warning("原题目 / 小问引用已不可用；未跳转到同名内容。")
        elif unit.id != item.question.id:
            render_saved_text(unit.content)
    with st.expander("高级设置", expanded=False):
        try:
            pages = {p.id for p in store.pages(item.source_id)}
            if not set(item.page_ids) <= pages:
                st.warning("部分原始来源段已不可用；保留原引用，未按标题或相似文字替换。")
            render_source(store, item.source_id, item.page_ids, prefix=prefix)
        except (StudyError, OSError):
            st.warning("原始来源缺失、已变化或无法读取；已保存记录保留，未猜测其他来源。")


def _render_error_record(store: StudyStore, record: ErrorEvidence, items: dict[str, Draft]) -> None:
    st.markdown("**错题依据与记录**")
    st.caption(f"{ORIGINS[record.origin]} · {_time(record.created_at)}")
    if record.answer:
        st.markdown("**本人实际作答**")
        render_saved_text(record.answer)
    if record.note:
        st.markdown("**已记录说明 / 依据**")
        render_saved_text(record.note)
    if record.origin == "manual_marker":
        st.info("依据为用户确认保存的题目属性；没有作答事件时不会计入答错次数。标记时间未记录。")
    st.text("已记录错因：" + ("、".join(record.tags) or "未记录"))
    item = items.get(record.item_id)
    if item is None:
        st.warning("原材料版本已不可用，未跳转到同名材料。")
        return
    _render_material(store, item, record.unit_id, "insight_error_" + record.id)


def render_error_profile(store: StudyStore) -> None:
    """Display selectable counts with their exact evidence, never model-generated scores."""
    st.subheader("错题画像")
    items = store.items()
    attempts = store.attempt_records()
    records = error_evidence(items, attempts)
    eligible_ids = {record.id for record in records if record.origin == "user_attempt"}
    unresolved = sum(
        attempt.get("source") == "user_attempt" and attempt.get("error_confirmed") is True
        and attempt.get("id") not in eligible_ids for attempt in attempts
    )
    if unresolved:
        st.warning(
            f"有 {unresolved} 条错误确认记录的身份、题目类型或作答证据无法核对，未计入统计；"
            "原记录保留，未猜测关联。"
        )
    origin = st.selectbox(
        "错题证据来源", ["", *ORIGINS],
        format_func=lambda value: ORIGINS.get(value, "全部来源（题目去重）"),
        key="sl_insight_origin",
    )
    scoped = [record for record in records if not origin or record.origin == origin]
    subject = st.selectbox(
        "画像科目", ["", *sorted({r.subject for r in scoped if r.subject})],
        format_func=lambda value: value or "全部（含未分类）", key="sl_insight_subject",
    )
    scoped = [record for record in scoped if not subject or record.subject == subject]
    counts = evidence_counts(scoped)
    columns = st.columns(3)
    columns[0].metric("错题道数", counts["questions"])
    columns[1].metric("答错次数（本人确认）", counts["attempts"])
    columns[2].metric("用户标记错题道数", counts["markers"])
    if not scoped:
        st.info(
            "暂无符合条件的已记录错题。可在已有待确认材料中明确标记错题后确认保存，"
            "或在主动复盘记录真实作答时明确确认本次错误；未确认的历史记录保持未知。"
        )
        return
    dimension_labels = {"knowledge": "知识点", "tags": "已记录错因"}
    dimension = st.radio(
        "分布维度", list(dimension_labels), format_func=dimension_labels.get,
        horizontal=True, key="sl_insight_dimension",
    )
    grouped = distribution(scoped, dimension)
    # Keep chart and selectable entries derived from the same group membership.
    st.bar_chart(
        {"分布项": list(grouped), "错题道数": [
            len({r.question_key for r in values}) for values in grouped.values()
        ]}, x="分布项", y="错题道数", horizontal=True,
    )
    context = (origin, subject, dimension)
    if st.session_state.get("sl_insight_context") != context:
        st.session_state["sl_insight_context"] = context
        st.session_state["sl_insight_group"] = ""
        st.session_state.pop("sl_insight_record", None)
    if st.button("显示全部对应记录", key="sl_insight_all"):
        st.session_state["sl_insight_group"] = ""
    for index, (name, values) in enumerate(grouped.items()):
        count = len({record.question_key for record in values})
        if st.button(_literal_label(f"{name} · {count} 道"), key=f"sl_insight_group_{index}"):
            st.session_state["sl_insight_group"] = name
            st.session_state.pop("sl_insight_record", None)
    selected_group = st.session_state.get("sl_insight_group", "")
    shown = grouped.get(selected_group, scoped)
    st.text(f"对应记录：{selected_group or '全部'} · {len(shown)} 条依据")
    by_id = {item.id: item for item in items}
    for record in shown:
        title = _title(by_id[record.item_id])
        st.text(f"{title} · {ORIGINS[record.origin]} · {_time(record.created_at)}")
        if st.button("查看错题记录", key="sl_insight_open_" + record.id):
            st.session_state["sl_insight_record"] = record.id
    selected = next((r for r in shown if r.id == st.session_state.get("sl_insight_record")), None)
    if selected:
        _render_error_record(store, selected, by_id)


def _render_session(store: StudyStore, event: ReviewEvent, items: dict[str, Draft]) -> None:
    st.markdown("**复盘记录详情**")
    st.caption(f"开始：{_time(event.started_at)}")
    if event.status == "completed":
        st.caption("结束：" + _time(event.ended_at))
    if event.invalid_reference or event.unavailable_items:
        st.warning("部分材料引用已不可用；保留原记录 ID，未跳转到同名材料。")
    if not event.attempts:
        st.info("本轮尚无实际作答记录；开始或结束复盘不代表已经掌握。")
    for identity in event.item_ids:
        item = items.get(identity)
        if item is None:
            st.caption("材料已删除或不可用。")
            continue
        st.text(_title(item))
        _render_material(store, item, None, f"insight_session_{event.id}_{identity}")
        units = {unit.id for unit in item.question.units()}
        for attempt in event.attempts:
            if attempt["item_id"] != identity:
                continue
            st.caption(
                f"{_time(attempt.get('created_at'))} · "
                f"{OUTCOME_LABELS.get(attempt.get('outcome'), '结果未知')}（本人自评）"
            )
            if attempt["unit_id"] not in units:
                st.warning("此作答的小问引用已不可用，保留历史作答，不猜测替代题目。")
            render_saved_text(str(attempt.get("answer") or attempt.get("note") or "未记录正文"))


def render_recent_reviews(store: StudyStore) -> None:
    """Read the existing sessions; viewing and filtering never create or resume one."""
    st.subheader("近期复盘")
    from .ai_review_ui import render_recent_ai_reviews

    render_recent_ai_reviews(store)
    items = {item.id: item for item in store.items()}
    all_events = recent_reviews(
        list(items.values()), store.sessions(), store.attempt_records(), limit=None,
    )
    show_all = st.checkbox("查看全部已有复盘历史", key="sl_recent_all")
    events = all_events if show_all else all_events[:10]
    st.caption(f"按题练习／主动回忆 {len(all_events)} 轮；当前展示 {len(events)} 轮。")
    if not events:
        if not store.sessions():
            st.caption("暂无按题练习／主动回忆记录。")
        return
    for event in events:
        status = "已结束" if event.status == "completed" else "进行中"
        st.text(f"{_time(event.started_at)} · 主动复盘 · {status}")
        st.text("；".join(
            _title(items[identity]) if identity in items else "材料已删除或不可用"
            for identity in event.item_ids
        ) or "材料引用不可用")
        st.caption(f"实际作答 {len(event.attempts)} 次")
        if st.button("查看复盘记录", key="sl_recent_open_" + event.id):
            st.session_state["sl_recent_record"] = event.id
    selected = next((e for e in events if e.id == st.session_state.get("sl_recent_record")), None)
    if selected:
        _render_session(store, selected, items)
