"""Overnight-round tests: display labels, reason-tag guard, no internal ids."""

from __future__ import annotations

from src.display_labels import (
    BANNED_FINAL_REASON_TAGS,
    QUESTION_KIND_LABELS,
    normalize_reason_tags,
    question_display_title,
    stem_digest,
)


def test_banned_reason_tags_are_replaced_by_root_cause_ask() -> None:
    """粗心/马虎/计算错误 may be a phenomenon, never the final error cause."""

    normalized = normalize_reason_tags(["计算错误", "粗心", "条件遗漏"])
    assert "条件遗漏" in normalized
    for banned in BANNED_FINAL_REASON_TAGS:
        assert banned not in normalized
    assert any(tag.startswith("根因待确认") for tag in normalized)


def test_normalize_reason_tags_keeps_concrete_roots_and_dedupes() -> None:
    normalized = normalize_reason_tags(["概念混淆", "概念混淆", "", "方法触发失败"])
    assert normalized == ["概念混淆", "方法触发失败"]


def test_question_display_title_uses_printed_number_or_digest() -> None:
    class _Q:  # minimal duck-typed stand-in for QuestionItem
        id = 12
        question_kind = "error"
        question_number = "4"
        stem_text = "已知函数 f(x)=x²lnx，求极值。"

    title = question_display_title(_Q())
    assert "#12" not in title
    assert "第4题" in title
    assert QUESTION_KIND_LABELS["error"] in title

    class _QNoNumber(_Q):
        question_number = ""
        stem_text = "抛体运动：炮弹从 A 处发射，求最大射角与最大射程。"

    digest_title = question_display_title(_QNoNumber())
    assert "#12" not in digest_title
    assert stem_digest(_QNoNumber.stem_text) in digest_title


def test_current_review_label_has_readable_context_without_internal_id(tmp_path) -> None:
    from src.study_loop.review_ui import label
    from tests.study_migration_support import panel, text_case

    store, source, draft = text_case(tmp_path / "study")
    title = label(draft)
    assert draft.id not in title
    assert draft.source_id not in title
    assert "TEST_COURSE" in title
    assert draft.question.number in title
    assert draft.question.content[:36] in title
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    app = panel(store, "review")
    assert not app.exception
    assert app.selectbox(key=f"sl_active_{session}").options == [source.name]
    assert app.selectbox(key=f"sl_unit_{session}_{draft.id}").options == ["1", "2"]
