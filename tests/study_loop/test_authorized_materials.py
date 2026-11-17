"""Opt-in source contracts; expected content comes from an authorized external manifest."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import pytest

from src.study_loop.components import import_source
from src.study_loop.review import unit_states


def test_original_bytes_and_duplicate(imported):
    store, source, case = imported
    original = case["path"].read_bytes()
    assert store.read_source(source) == original
    assert hashlib.sha256(original).hexdigest() == case["sha256"]
    duplicate = store.receive(source.name, original, source.kind, transcript=source.transcript)
    assert duplicate == source


def test_idempotent_segments_and_drafts(imported):
    store, source, _ = imported
    before = ([asdict(p) for p in store.pages(source.id)], [d.to_dict() for d in store.drafts()])
    import_source(store, source.id)
    after = ([asdict(p) for p in store.pages(source.id)], [d.to_dict() for d in store.drafts()])
    assert after == before


def test_import_never_confirms_answers_or_mastery(imported):
    store, _, _ = imported
    assert store.items() == []
    assert store.attempts() == []
    assert store.sessions() == []
    for draft in store.drafts():
        for unit in draft.question.units():
            assert unit.answer_status == "unconfirmed"
            assert unit.reference_answer == ""
            assert unit.reference_analysis == ""
    assert set(unit_states(store.drafts(), []).values()) <= {"unknown"}


def test_source_segments_match_user_annotations(imported):
    store, source, case = imported
    expected = case.get("expected_segments")
    if expected is None:
        pytest.skip("用户尚未提供该材料的来源段标注，不能自编预期内容。")
    pages = store.pages(source.id)
    assert len(pages) == len(expected)
    for page, annotation in zip(pages, expected, strict=True):
        for key in ("number", "text"):
            if key in annotation:
                assert getattr(page, key) == annotation[key]
        for key, value in annotation.get("metadata", {}).items():
            assert page.metadata[key] == value


def test_question_boundaries_match_user_annotations(imported):
    store, _, case = imported
    expected = case.get("expected_questions")
    if expected is None:
        pytest.skip("用户尚未提供题目和小问标注，不能编造标准答案。")
    actual = [
        {
            "number": d.question.number,
            "content": d.question.content,
            "children": [
                {"number": q.number, "content": q.content, "observed_answer": q.observed_answer}
                for q in d.question.children
            ],
        }
        for d in store.drafts()
    ]
    assert actual == expected


def test_review_uses_only_authorized_actual_work(imported):
    store, _, case = imported
    attempts = case.get("actual_attempts")
    if not attempts:
        pytest.skip("没有用户授权的实际作答记录；不生成模拟作答。")
    draft = store.drafts()[case["review_draft_index"]]
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    for index, attempt in enumerate(attempts):
        unit = draft.question.units()[attempt["unit_index"]]
        kwargs = {
            name: attempt[name]
            for name in ("outcome", "answer", "note", "independent", "attempted", "error_tags")
        }
        store.record_attempt(session, draft.id, unit.id, submission_id=str(index), **kwargs)
        store.record_attempt(session, draft.id, unit.id, submission_id=str(index), **kwargs)
        assert len(store.attempts()) == index + 1
        assert unit_states([draft], store.attempts())[(draft.id, unit.id)] == attempt["outcome"]
    store.end_review(session)
