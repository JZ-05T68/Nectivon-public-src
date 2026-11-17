"""Synthetic review-to-course and confirmed-memory integration; no network or user data."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.ai.conversation import decode_conversation
from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.course_reviews import CourseReviewIndex
from src.study_loop.models import StudyError

COURSE_A = "NON_REAL_COURSE_A"
COURSE_B = "NON_REAL_COURSE_B"
SAVED_A = r"SAVED_REVIEW_A_SENTINEL：条件 $b>0$，关系 $q=\frac{a}{b}$。"


@pytest.fixture
def case(tmp_path):
    result = review_case(tmp_path / "NON_REAL_COURSE_MEMORY")
    return SimpleNamespace(
        **vars(result), service=AIReviewService(result.store),
        index=CourseReviewIndex(result.store),
        provider=RecordingProvider("RAW_REVIEW_MESSAGE_MUST_NOT_BECOME_COURSE_TURN"),
    )


def _saved(case, *, item=None, content=SAVED_A):
    selected = item or case.selected
    document = case.service.generate([selected.id], case.provider)
    case.service.update_working(document["id"], content)
    version = case.service.save_version(document["id"])
    return case.service.document(document["id"]), version


def _context(case, conversation_id, query="SAVED_REVIEW_A_SENTINEL"):
    prompt, evidence = chat.prepare(case.chats, conversation_id, query)
    messages = decode_conversation(prompt)
    return json.loads(messages[1]["content"].split("\n", 1)[1]), messages, evidence


def test_saved_review_immediately_enters_only_its_course_and_unsaved_text_stays_out(case):
    before = case.store.db.read_bytes()
    document = case.service.generate([case.selected.id], case.provider)
    assert case.index.entries(COURSE_A) == []
    case.service.update_working(document["id"], SAVED_A)
    assert case.index.entries(COURSE_A) == []
    first = case.service.save_version(document["id"])
    rows = case.index.entries(COURSE_A)
    assert len(rows) == 1 and rows[0]["document_id"] == document["id"]
    row = rows[0]
    assert row["version_id"] == first["id"] and row["version_number"] == first["number"]
    assert row["content"] == SAVED_A and row["status"] == "saved_review"
    assert row["source_ids"] == [case.source.id] and row["filenames"] == [case.source.name]
    assert row["questions"][0]["title"] == case.selected.question.content
    assert row["updated_at"] == first["created_at"]
    case.service.update_working(document["id"], "UNSAVED_WORKING_NOT_IN_COURSE_INDEX")
    assert case.index.entry(document["id"], COURSE_A) == row
    assert not case.index.entries(COURSE_A, "UNSAVED_WORKING_NOT_IN_COURSE_INDEX")
    _, foreign = _saved(case, item=case.foreign, content="OTHER_COURSE_SAVED_SENTINEL")
    assert [entry["version_id"] for entry in case.index.entries(COURSE_B)] == [foreign["id"]]
    assert [entry["version_id"] for entry in case.index.entries(COURSE_A)] == [first["id"]]
    assert case.store.db.read_bytes() == before
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert case.chats.memories(None) == []


def test_course_index_reads_latest_saved_revision_without_replacing_old_versions(case):
    document, first = _saved(case)
    first_bytes = copy.deepcopy(case.service.version(first["id"]))
    case.service.update_working(document["id"], "LATEST_SAVED_REVIEW_A_SENTINEL")
    second = case.service.save_version(document["id"])
    case.service.update_working(document["id"], "NEWER_UNSAVED_MUST_NOT_REPLACE_LATEST")
    row = case.index.entry(document["id"])
    assert row["version_id"] == second["id"] and row["content"] == second["content"]
    assert len(case.index.entries(COURSE_A)) == 1
    assert case.service.version(first["id"]) == first_bytes
    refs = case.index.retrieve(COURSE_A, "LATEST_SAVED_REVIEW_A_SENTINEL", [case.source.id])
    assert len(refs) == 1
    assert refs[0]["kind"] == "saved_review" and refs[0]["label"].startswith("R")
    assert refs[0]["version_id"] == second["id"] and refs[0]["text"] == second["content"]
    assert refs[0]["sha256"] == hashlib.sha256(second["content"].encode()).hexdigest()
    assert case.index.retrieve(COURSE_A, "LATEST_SAVED_REVIEW_A_SENTINEL", []) == []
    assert case.index.retrieve(COURSE_B, "LATEST_SAVED_REVIEW_A_SENTINEL") == []


def test_review_document_source_and_course_delete_restore_keep_index_dynamic(case):
    document, version = _saved(case)
    original = case.store.read_source(case.source)
    with pytest.raises(StudyError):
        case.service.delete_document(document["id"], confirmed=False)
    case.service.delete_document(document["id"], confirmed=True)
    assert not case.index.entries(COURSE_A) and not case.service.documents()
    assert case.service.documents(include_deleted=True)[0]["id"] == document["id"]
    assert case.service.version(version["id"]) == version
    case.service.restore_document(document["id"])
    assert case.index.entry(document["id"])["version_id"] == version["id"]
    case.store.delete_source(case.source.id, user_confirmed=True)
    assert not case.index.entries(COURSE_A)
    case.store.restore_source(case.source.id, user_confirmed=True)
    assert case.index.entry(document["id"])["version_id"] == version["id"]
    case.chats.delete("course", COURSE_A, confirmed=True)
    assert not case.index.entries(COURSE_A)
    case.chats.restore("course", COURSE_A)
    assert case.index.entry(document["id"])["version_id"] == version["id"]
    assert case.store.read_source(case.source) == original


def test_course_index_follows_current_saved_subject_without_rewriting_review_snapshot(case):
    document, version = _saved(case)
    original_conversation = copy.deepcopy(case.chats.snapshot(document["conversation_id"]))
    changed = copy.deepcopy(case.selected)
    changed.subject = COURSE_B
    case.store.confirm(changed, user_confirmed=True)
    assert not case.index.entries(COURSE_A)
    assert case.index.entry(document["id"], COURSE_B)["version_id"] == version["id"]
    assert case.chats.snapshot(document["conversation_id"]) == original_conversation
    assert case.service.document(document["id"])["subject"] == COURSE_A


def test_course_comparison_accepts_only_selected_live_documents_and_preserves_order(case):
    first, _ = _saved(case)
    second, _ = _saved(case, item=case.sibling, content="SAVED_SIBLING_REVIEW_SENTINEL")
    foreign, _ = _saved(case, item=case.foreign, content="SAVED_FOREIGN_REVIEW_SENTINEL")
    rows = case.index.compare(COURSE_A, [second["id"], first["id"]])
    assert [row["document_id"] for row in rows] == [second["id"], first["id"]]
    with pytest.raises(StudyError):
        case.index.compare(COURSE_A, [first["id"], foreign["id"]])
    with pytest.raises(StudyError):
        case.index.compare(COURSE_A, [first["id"], first["id"]])
    case.service.delete_document(second["id"], confirmed=True)
    with pytest.raises(StudyError):
        case.index.compare(COURSE_A, [second["id"], first["id"]])


def test_new_course_sessions_share_saved_index_but_not_each_others_or_review_messages(case):
    document, version = _saved(case)
    review_before = copy.deepcopy(case.chats.snapshot(document["conversation_id"]))
    first = case.chats.open_course(COURSE_A)
    context, messages, evidence = _context(case, first)
    assert any(ref.get("version_id") == version["id"] for ref in evidence)
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert context["course"] == COURSE_A
    assert "RAW_REVIEW_MESSAGE_MUST_NOT_BECOME_COURSE_TURN" not in json.dumps(messages)
    case.provider.content = "FIRST_COURSE_REPLY_MUST_NOT_CROSS_SESSIONS"
    chat.send(case.chats, first, "FIRST_COURSE_USER_ONLY", "FIRST_COURSE_REQUEST",
              case.provider, allow_remote=True)
    second = case.chats.open_course(COURSE_A, new=True)
    assert second != first and case.chats.messages(second) == []
    _, messages, evidence = _context(case, second)
    assert any(ref.get("version_id") == version["id"] for ref in evidence)
    assert "FIRST_COURSE_USER_ONLY" not in json.dumps(messages)
    assert "FIRST_COURSE_REPLY_MUST_NOT_CROSS_SESSIONS" not in json.dumps(messages)
    assert case.chats.snapshot(document["conversation_id"]) == review_before
    assert {conversation["id"] for conversation in case.chats.conversations("course")} == {
        first, second,
    }


@pytest.mark.parametrize("change", [
    "new_version", "delete_document", "delete_source", "delete_course",
])
def test_historical_saved_review_evidence_cannot_bypass_current_index_guards(case, change):
    document, version = _saved(case)
    refs = case.index.retrieve(COURSE_A, "SAVED_REVIEW_A_SENTINEL")
    case.index.validate_evidence(COURSE_A, refs)
    identity = case.chats.open_course(COURSE_A)
    case.provider.content = "NON_REAL_COURSE_USES_SAVED_REVIEW"
    chat.send(case.chats, identity, "SAVED_REVIEW_A_SENTINEL", "CITE_SAVED_VERSION",
              case.provider, allow_remote=True)
    snapshot = copy.deepcopy(case.chats.snapshot(identity))
    count = len(case.provider.prompts)
    if change == "new_version":
        case.service.update_working(document["id"], "REPLACEMENT_SAVED_REVIEW")
        case.service.save_version(document["id"])
    elif change == "delete_document":
        case.service.delete_document(document["id"], confirmed=True)
    elif change == "delete_course":
        case.chats.delete("course", COURSE_A, confirmed=True)
    else:
        case.store.delete_source(case.source.id, user_confirmed=True)
    with pytest.raises(StudyError):
        case.index.validate_evidence(COURSE_A, refs)
    with pytest.raises(StudyError):
        chat.send(case.chats, identity, "FOLLOW_UP_MUST_NOT_SEND_STALE_REVIEW", "NO_DISPATCH",
                  case.provider, allow_remote=True)
    assert len(case.provider.prompts) == count
    assert case.chats.snapshot(identity) == snapshot
    assert case.service.version(version["id"]) == version
    if change != "new_version":
        if change == "delete_document":
            case.service.restore_document(document["id"])
        elif change == "delete_course":
            case.chats.restore("course", COURSE_A)
        else:
            case.store.restore_source(case.source.id, user_confirmed=True)
        case.index.validate_evidence(COURSE_A, refs)
        chat.prepare(case.chats, identity, "RESTORED_SAME_VERSION_IS_VALID")


@pytest.mark.parametrize("field,value", [
    ("version_id", "UNVERIFIED_VERSION"), ("sha256", "0" * 64),
    ("source_ids", []), ("text", "REPLACED_REVIEW_EXCERPT"),
])
def test_saved_review_evidence_detects_tampered_version_source_hash_or_excerpt(case, field, value):
    _saved(case)
    refs = case.index.retrieve(COURSE_A, "SAVED_REVIEW_A_SENTINEL")
    forged = copy.deepcopy(refs)
    forged[0][field] = value
    with pytest.raises(StudyError):
        case.index.validate_evidence(COURSE_A, forged)
