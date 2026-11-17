"""Synthetic confirmed-memory editing, collision and historical prompt projection regressions."""

from __future__ import annotations

import copy
import hashlib

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.ai.conversation import decode_conversation
from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.chat_store import ChatStore, digest
from src.study_loop.knowledge_memory import KnowledgeMemoryService
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore

COURSE = "NON_REAL_COURSE_A"


@pytest.fixture
def case(tmp_path):
    return review_case(tmp_path / "NON_REAL_MEMORY_REVISIONS")


def confirmed(case, content: str, subject: str = COURSE):
    conversation = case.chats.open_course(subject, new=True)
    case.chats.begin(conversation, "NON_REAL_ORIGINAL", "NON_REAL_USER", {})
    case.chats.finish(conversation, "NON_REAL_ORIGINAL", "NON_REAL_AI", {})
    archive = case.chats.archive(conversation)
    candidate = case.chats.add_candidate(archive, "NON_REAL_CANDIDATE")
    identity = case.chats.confirm(candidate, content, confirmed=True)
    return identity, conversation, archive, candidate


def test_stable_id_edit_history_and_restore_leave_original_evidence_unchanged(case):
    identity, conversation, archive, candidate = confirmed(case, "NON_REAL_ORIGINAL_MEMORY")
    messages = copy.deepcopy(case.chats.messages(conversation))
    snapshot = copy.deepcopy(case.chats.archived(archive))
    original = case.chats.memory(identity)
    assert original["revision"] == 1
    updated = case.chats.update_memory(
        identity, "NON_REAL_EDITED_MEMORY", expected_revision=1,
    )
    assert updated["id"] == identity and updated["revision"] == 2
    assert updated["content"] == "NON_REAL_EDITED_MEMORY"
    assert updated["links"] == original["links"]
    assert updated["sources"] == original["sources"]
    assert case.chats.messages(conversation) == messages
    assert case.chats.archived(archive) == snapshot
    assert case.chats.candidates(conversation)[0]["id"] == candidate
    assert case.chats.candidates(conversation)[0]["content"] == "NON_REAL_CANDIDATE"
    reopened = ChatStore(StudyStore(case.store.root))
    assert reopened.memory(identity)["content"] == "NON_REAL_EDITED_MEMORY"
    assert not reopened.memories(COURSE, "ORIGINAL_MEMORY")
    assert reopened.memories(COURSE, "EDITED_MEMORY")[0]["id"] == identity
    restored = reopened.restore_memory_revision(identity, 1, expected_revision=2)
    assert restored["id"] == identity and restored["revision"] == 3
    assert restored["content"] == original["content"]
    history = reopened.memory_revisions(identity)
    assert [row["revision"] for row in history] == [3, 2, 1]
    assert [row["content"] for row in history] == [
        "NON_REAL_ORIGINAL_MEMORY", "NON_REAL_EDITED_MEMORY", "NON_REAL_ORIGINAL_MEMORY",
    ]
    assert history[0]["action"] == "restored" and history[0]["restored_from_revision"] == 1
    assert not case.store.attempts() and not case.store.sessions()


def test_conflict_rejection_preserves_both_current_content_and_history(case):
    identity, *_ = confirmed(case, "NON_REAL_BASELINE")
    case.chats.update_memory(identity, "NON_REAL_WINNING_EDIT", expected_revision=1)
    before = case.chats.memory_revisions(identity)
    with pytest.raises(StudyError, match="保留当前输入"):
        case.chats.update_memory(identity, "NON_REAL_STALE_INPUT", expected_revision=1)
    with pytest.raises(StudyError, match="保留当前输入"):
        case.chats.restore_memory_revision(identity, 1, expected_revision=1)
    assert case.chats.memory(identity)["content"] == "NON_REAL_WINNING_EDIT"
    assert case.chats.memory_revisions(identity) == before


def test_edit_and_history_restore_preserve_indentation_and_terminal_newlines_exactly(case):
    identity, *_ = confirmed(case, "NON_REAL_INITIAL_BODY")
    body = "    NON_REAL_INDENTED_CODE = 17\n    print(NON_REAL_INDENTED_CODE)\n\n"
    edited = case.chats.update_memory(identity, body, expected_revision=1)
    assert edited["content"] == body
    assert case.chats.memory_revisions(identity)[0]["content"] == body
    unchanged = case.chats.update_memory(identity, body, expected_revision=2)
    assert unchanged["revision"] == 2
    original = case.chats.restore_memory_revision(identity, 1, expected_revision=2)
    assert original["revision"] == 3
    restored = case.chats.restore_memory_revision(identity, 2, expected_revision=3)
    assert restored["revision"] == 4 and restored["content"] == body
    reopened = ChatStore(StudyStore(case.store.root))
    assert reopened.memory(identity)["content"] == body
    assert reopened.memory_revisions(identity)[0]["content"] == body


def test_edit_and_restore_reject_active_duplicate_without_silent_merge(case):
    first, *_ = confirmed(case, "NON_REAL_FIRST")
    second, *_ = confirmed(case, "NON_REAL_SECOND")
    with pytest.raises(StudyError, match="相同正文"):
        case.chats.update_memory(second, "NON_REAL_FIRST", expected_revision=1)
    assert case.chats.memory(second)["content"] == "NON_REAL_SECOND"
    assert len(case.chats.memory_revisions(second)) == 1
    case.chats.update_memory(first, "NON_REAL_FIRST_CHANGED", expected_revision=1)
    case.chats.update_memory(second, "NON_REAL_FIRST", expected_revision=1)
    with pytest.raises(StudyError, match="相同正文"):
        case.chats.restore_memory_revision(first, 1, expected_revision=2)
    assert case.chats.memory(first)["content"] == "NON_REAL_FIRST_CHANGED"


def test_confirmation_uses_current_content_and_never_reuses_an_occupied_old_hash_id(case):
    first, _, _, original_candidate = confirmed(case, "NON_REAL_HASH_ORIGINAL")
    assert first == digest([COURSE, "NON_REAL_HASH_ORIGINAL"])
    case.chats.update_memory(first, "NON_REAL_HASH_EDITED", expected_revision=1)
    second = case.chats.confirm(original_candidate, "NON_REAL_HASH_ORIGINAL", confirmed=True)
    assert second != first and len(second) == 32
    assert case.chats.memory(first)["content"] == "NON_REAL_HASH_EDITED"
    assert case.chats.memory(second)["content"] == "NON_REAL_HASH_ORIGINAL"
    same, _, _, _ = confirmed(case, "NON_REAL_HASH_EDITED")
    assert same == first
    assert len(case.chats.memories(COURSE)) == 2
    assert len(case.chats.memory(first)["links"]) == 2
    assert case.chats.memory_revisions(first)[-1]["content"] == "NON_REAL_HASH_ORIGINAL"


def test_inactive_duplicate_is_also_rejected_so_later_restore_cannot_create_two_copies(case):
    first, *_ = confirmed(case, "NON_REAL_ACTIVE_BODY")
    second, *_ = confirmed(case, "NON_REAL_DELETED_BODY")
    case.chats.delete("memory", second, confirmed=True)
    with pytest.raises(StudyError, match="相同正文"):
        case.chats.update_memory(first, "NON_REAL_DELETED_BODY", expected_revision=1)
    assert case.chats.memory(first)["content"] == "NON_REAL_ACTIVE_BODY"
    assert len(case.chats.memory_revisions(first)) == 1


def test_deleted_memory_can_be_edited_and_restored_as_revision_without_becoming_retrievable(case):
    identity, *_ = confirmed(case, "NON_REAL_DELETED_BASE")
    case.chats.delete("memory", identity, confirmed=True)
    changed = case.chats.update_memory(identity, "NON_REAL_DELETED_EDIT", expected_revision=1)
    assert changed["deleted"] and not changed["source_status"]["active"]
    assert case.chats.memories(COURSE) == []
    assert case.chats.memories(COURSE, include_deleted=True)[0]["id"] == identity
    restored_body = case.chats.restore_memory_revision(identity, 1, expected_revision=2)
    assert restored_body["revision"] == 3 and restored_body["deleted"]
    assert case.chats.memories(COURSE) == []


def test_stale_typed_memory_edits_preserve_original_candidate_and_confirmation(case):
    reviews = AIReviewService(case.store)
    document = reviews.generate([case.selected.id], RecordingProvider())
    reviews.update_working(document["id"], "NON_REAL_SAVED_REVIEW")
    reviews.save_version(document["id"])
    extraction_service = KnowledgeMemoryService(case.store)
    extraction = extraction_service.extract_review(
        document["id"], RecordingProvider("NON_REAL_EXTRACTED_CANDIDATE"),
    )
    identity = extraction_service.confirm(
        extraction["id"], "NON_REAL_CONFIRMED_TYPED", confirmed=True,
    )
    original_extraction = extraction_service.extraction(extraction["id"])
    source = copy.deepcopy(case.chats.memory(identity)["sources"])
    reviews.update_working(document["id"], "NON_REAL_NEW_REVIEW_VERSION")
    reviews.save_version(document["id"])
    changed = case.chats.update_memory(identity, "NON_REAL_STALE_EDIT", expected_revision=1)
    assert changed["stale"] and not changed["source_status"]["active"]
    assert changed["sources"] == source
    assert extraction_service.extraction(extraction["id"]) == original_extraction
    assert not case.chats.memories(COURSE)
    assert case.chats.memories(COURSE, include_stale=True)[0]["content"] == "NON_REAL_STALE_EDIT"
    reviews.delete_document(document["id"], confirmed=True)
    unavailable = case.chats.update_memory(
        identity, "NON_REAL_UNAVAILABLE_EDIT", expected_revision=2,
    )
    assert unavailable["source_status"]["unavailable"]
    assert not case.chats.memories(COURSE)
    restored_body = case.chats.restore_memory_revision(identity, 1, expected_revision=3)
    assert restored_body["source_status"]["unavailable"] and restored_body["revision"] == 4
    assert extraction_service.extraction(extraction["id"]) == original_extraction


def test_history_memory_projection_refreshes_latest_body_without_rewriting_stored_messages(case):
    import json

    identity, _, archive, _ = confirmed(case, "NON_REAL_OLD_MEMORY_KEY")
    conversation = case.chats.open_course(COURSE, new=True)
    refs = chat.retrieve(
        case.chats, case.chats.conversation(conversation), "NON_REAL_OLD_MEMORY_KEY",
    )
    old_ref = next(ref for ref in refs if ref["label"].startswith("M"))
    case.chats.begin(conversation, "NON_REAL_MEMORY_QUERY", "NON_REAL_OLD_MEMORY_KEY",
                     {"evidence": refs})
    case.chats.finish(conversation, "NON_REAL_MEMORY_QUERY", "NON_REAL_OLD_AI_REPLY", {})
    before = copy.deepcopy(case.chats.messages(conversation))
    old_archive = case.chats.archived(archive)
    case.chats.update_memory(identity, "NON_REAL_LATEST_CHANGED_BODY", expected_revision=1)
    prompt, evidence = chat.prepare(case.chats, conversation, "请继续解释")
    context = json.loads(decode_conversation(prompt)[1]["content"].split("\n", 1)[1])
    current = next(ref for ref in context["evidence"] if ref["label"] == old_ref["label"])
    assert current["text"] == "NON_REAL_LATEST_CHANGED_BODY"
    assert current["memory_id"] == identity and current["memory_revision"] == 2
    assert current["sha256"] == hashlib.sha256(current["text"].encode()).hexdigest()
    assert current["sha256"] != identity
    assert any(ref == current for ref in evidence)
    assert case.chats.messages(conversation) == before
    assert case.chats.archived(archive) == old_archive
    assert before[0]["metadata"]["evidence"] == refs
    case.chats.delete("memory", identity, confirmed=True)
    with pytest.raises(StudyError, match="新建对话"):
        chat.prepare(case.chats, conversation, "再继续")


def test_legacy_memory_baseline_is_backfilled_without_altering_original_rows(case):
    identity, conversation, archive, _ = confirmed(case, "NON_REAL_LEGACY_BASE")
    original_memory = case.chats.memory(identity)
    snapshot = case.chats.archived(archive)
    with case.chats.connect() as db:
        db.execute("DROP TABLE memory_revisions")
    reopened = ChatStore(StudyStore(case.store.root))
    assert reopened.memory(identity)["content"] == original_memory["content"]
    assert reopened.memory(identity)["links"] == original_memory["links"]
    assert reopened.memory_revisions(identity)[0]["content"] == "NON_REAL_LEGACY_BASE"
    assert reopened.archived(archive) == snapshot
    assert reopened.messages(conversation) == case.chats.messages(conversation)


def test_subjects_remain_separate_and_cross_course_history_memory_is_blocked(case):
    first, *_ = confirmed(case, "NON_REAL_SAME_CONTENT", COURSE)
    other, *_ = confirmed(case, "NON_REAL_SAME_CONTENT", "NON_REAL_COURSE_B")
    assert first != other
    case.chats.update_memory(first, "NON_REAL_ONLY_COURSE_A", expected_revision=1)
    assert case.chats.memory(other)["content"] == "NON_REAL_SAME_CONTENT"
    foreign_ref = chat.retrieve(
        case.chats, case.chats.conversation(case.chats.open_course("NON_REAL_COURSE_B")),
        "NON_REAL_SAME_CONTENT",
    )
    foreign_ref = [ref for ref in foreign_ref if ref["label"].startswith("M")]
    conversation = case.chats.open_course(COURSE, new=True)
    case.chats.begin(conversation, "NON_REAL_FOREIGN_REFERENCE", "NON_REAL_USER",
                     {"evidence": foreign_ref})
    case.chats.finish(conversation, "NON_REAL_FOREIGN_REFERENCE", "NON_REAL_AI", {})
    with pytest.raises(StudyError, match="当前课程"):
        chat.prepare(case.chats, conversation, "继续")
