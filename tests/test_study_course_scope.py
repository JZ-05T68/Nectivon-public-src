"""Synthetic course-wide scope routing; no real material, model call or history rewrite."""

from __future__ import annotations

import copy
import json

import pytest

from src.ai.conversation import decode_conversation
from src.ai.provider import CompletionResult
from src.study_loop import chat
from src.study_loop.chat_store import ChatStore
from src.study_loop.components import import_source
from src.study_loop.storage import StudyStore


@pytest.fixture
def course_case(tmp_path):
    study = StudyStore(tmp_path / "study", create=True)
    sources = []
    for index in range(4):
        source = study.receive(
            f"NON_REAL_SOURCE_{index}.txt",
            f"1. NON_REAL_{index}_FIRST\n2. NON_REAL_{index}_SECOND".encode(), "txt",
        )
        sources.append(source)
        import_source(study, source.id)
    chats = ChatStore(study)
    return study, sources, chats


def _context(chats: ChatStore, identity: str) -> tuple[dict, list[dict]]:
    prompt, _ = chat.prepare(chats, identity, "NON_REAL_CURRENT_QUESTION")
    messages = decode_conversation(prompt)
    return json.loads(messages[1]["content"].split("\n", 1)[1]), messages


def test_latest_scope_reaches_all_questions_across_the_courses_sources(course_case):
    study, sources, chats = course_case
    chats.set_course("COURSE_A", [sources[0].id, sources[1].id], "USER_SCOPE_A_V1")
    chats.set_course("COURSE_B", [sources[2].id], "USER_SCOPE_B")
    a_questions = [draft for draft in study.drafts()
                   if draft.source_id in {sources[0].id, sources[1].id}]
    assert len(a_questions) == 4
    identities = [chats.open_question(draft.id) for draft in a_questions]
    originals = {identity: chats.snapshot(identity) for identity in identities}
    for identity in identities:
        context, _ = _context(chats, identity)
        assert context["course"] == "COURSE_A"
        assert context["user_provided_exam_scope"] == "USER_SCOPE_A_V1"

    chats.set_course("COURSE_A", [sources[0].id, sources[1].id], "USER_SCOPE_A_V2")
    for identity in identities:
        context, messages = _context(chats, identity)
        assert context["user_provided_exam_scope"] == "USER_SCOPE_A_V2"
        assert "USER_SCOPE_B" not in messages[1]["content"]
        assert chats.snapshot(identity) == originals[identity]
    b_question = next(draft for draft in study.drafts() if draft.source_id == sources[2].id)
    context, _ = _context(chats, chats.open_question(b_question.id))
    assert context["user_provided_exam_scope"] == "USER_SCOPE_B"


def test_existing_question_without_a_course_uses_later_unique_course_scope(course_case):
    study, sources, chats = course_case
    draft = next(d for d in study.drafts() if d.source_id == sources[0].id)
    identity = chats.open_question(draft.id)
    chats.begin(identity, "old", "NON_REAL_OLD_USER", {})
    chats.finish(identity, "old", "NON_REAL_OLD_ASSISTANT", {})
    old = copy.deepcopy(chats.snapshot(identity))
    assert old["conversation"]["subject"] == ""
    chats.set_course("COURSE_A", [sources[0].id], "USER_LATER_SCOPE")
    context, messages = _context(chats, identity)
    assert context["course"] == "COURSE_A"
    assert context["user_provided_exam_scope"] == "USER_LATER_SCOPE"
    assert messages[2:4] == [
        {"role": "user", "content": "NON_REAL_OLD_USER"},
        {"role": "assistant", "content": "NON_REAL_OLD_ASSISTANT"},
    ]
    assert chats.snapshot(identity) == old


def test_explicit_question_subject_wins_over_other_source_memberships(course_case):
    study, sources, chats = course_case
    draft = next(d for d in study.drafts() if d.source_id == sources[0].id)
    draft.subject = "COURSE_A"
    study.confirm(draft, user_confirmed=True)
    chats.set_course("COURSE_A", [sources[1].id], "USER_SCOPE_A")
    chats.set_course("COURSE_B", [sources[0].id], "USER_SCOPE_B")
    identity = chats.open_question(draft.id)
    context, _ = _context(chats, identity)
    assert context["course"] == "COURSE_A" and context["user_provided_exam_scope"] == "USER_SCOPE_A"
    assert chats.conversation(identity)["source_ids"] == [sources[0].id]
    assert {ref["source_id"] for ref in context["evidence"]} == {sources[0].id}


def test_explicit_unconfigured_subject_does_not_adopt_another_course_scope(course_case):
    study, sources, chats = course_case
    draft = study.drafts()[0]
    draft.subject = "COURSE_WITHOUT_SETTINGS"
    study.confirm(draft, user_confirmed=True)
    chats.set_course("UNRELATED_COURSE", [draft.source_id], "USER_UNRELATED_SCOPE")
    context, _ = _context(chats, chats.open_question(draft.id))
    assert context["course"] == "COURSE_WITHOUT_SETTINGS"
    assert context["user_provided_exam_scope"] == ""


def test_shared_material_does_not_merge_courses_even_for_an_old_inferred_conversation(course_case):
    study, sources, chats = course_case
    draft = next(d for d in study.drafts() if d.source_id == sources[3].id)
    chats.set_course("COURSE_A", [sources[3].id], "USER_SCOPE_A")
    identity = chats.open_question(draft.id)
    old = chats.snapshot(identity)
    assert old["conversation"]["subject"] == "COURSE_A"
    chats.set_course("COURSE_B", [sources[3].id], "USER_SCOPE_B")
    context, _ = _context(chats, identity)
    assert context["course"] == "" and context["user_provided_exam_scope"] == ""
    assert chats.snapshot(identity) == old
    another = next(d for d in study.drafts()
                   if d.source_id == sources[3].id and d.id != draft.id)
    context, _ = _context(chats, chats.open_question(another.id))
    assert context["user_provided_exam_scope"] == ""


def test_free_course_uses_latest_scope_without_expanding_its_material_selection(course_case):
    _, sources, chats = course_case
    chats.set_course("COURSE_A", [sources[0].id], "USER_SCOPE_A_V1")
    identity = chats.open_course("COURSE_A")
    old = chats.snapshot(identity)
    chats.set_course("COURSE_A", [sources[0].id, sources[1].id], "USER_SCOPE_A_V2")
    context, _ = _context(chats, identity)
    assert context["user_provided_exam_scope"] == "USER_SCOPE_A_V2"
    assert context["retrieval_scope"]["selected_sources"] == 1
    assert {ref["source_id"] for ref in context["evidence"]} == {sources[0].id}
    assert chats.snapshot(identity) == old


@pytest.mark.parametrize("remove", ["clear", "delete"])
def test_removed_course_scope_never_falls_back_to_old_saved_exam_notes(course_case, remove):
    _, sources, chats = course_case
    chats.set_course("COURSE_A", [sources[0].id], "USER_SCOPE_OLD")
    identity = chats.open_course("COURSE_A")
    old = chats.snapshot(identity)
    if remove == "clear":
        chats.set_course("COURSE_A", [sources[0].id], "")
    else:
        chats.delete("course", "COURSE_A", confirmed=True)
    context, _ = _context(chats, identity)
    assert context["user_provided_exam_scope"] == ""
    assert chats.snapshot(identity) == old


def test_scope_remains_a_user_constraint_and_new_sends_keep_older_messages_intact(course_case):
    study, sources, chats = course_case
    chats.set_course("COURSE_A", [sources[0].id], "USER_SCOPE_A_V1")
    draft = next(d for d in study.drafts() if d.source_id == sources[0].id)
    identity = chats.open_question(draft.id)

    class RecordingProvider:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return CompletionResult("NON_REAL_ANSWER", "NON_REAL_MODEL")

    provider = RecordingProvider()
    chat.send(chats, identity, "NON_REAL_FIRST", "first", provider, allow_remote=True)
    old_messages = chats.messages(identity)
    old_conversation = chats.conversation(identity)
    chats.set_course("COURSE_A", [sources[0].id], "USER_SCOPE_A_V2")
    chat.send(chats, identity, "NON_REAL_SECOND", "second", provider, allow_remote=True)
    assert chats.messages(identity)[:2] == old_messages
    assert chats.conversation(identity) == old_conversation
    contexts = [json.loads(decode_conversation(prompt)[1]["content"].split("\n", 1)[1])
                for prompt in provider.prompts]
    assert [context["user_provided_exam_scope"] for context in contexts] == [
        "USER_SCOPE_A_V1", "USER_SCOPE_A_V2",
    ]
    policy = decode_conversation(provider.prompts[-1])[0]["content"]
    assert "本课程全部题目" in policy and "范围约束" in policy and "不是已核实的考试事实" in policy
