"""Explicit knowledge extraction, immutable provenance and retrieval status regressions."""

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
from src.study_loop.chat_store import ChatStore, digest
from src.study_loop.course_reviews import CourseReviewIndex
from src.study_loop.knowledge_memory import KnowledgeMemoryService
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore

COURSE = "NON_REAL_COURSE_A"
SAVED = r"NON_REAL_SAVED_KNOWLEDGE_BASIS $q=\frac{a}{b}$，限定 $b>0$。"


@pytest.fixture
def case(tmp_path):
    source = review_case(tmp_path / "NON_REAL_MEMORY_PROVENANCE")
    return SimpleNamespace(
        **vars(source), reviews=AIReviewService(source.store),
        memories=KnowledgeMemoryService(source.store),
        review_provider=RecordingProvider("RAW_AI_MUST_NOT_BE_EXTRACTED_INSTEAD_OF_SAVED"),
    )


def _saved(case):
    document = case.reviews.generate([case.selected.id], case.review_provider)
    case.reviews.update_working(document["id"], SAVED)
    version = case.reviews.save_version(document["id"])
    return document, version


def _course(case):
    identity = case.chats.open_course(COURSE, new=True)
    provider = RecordingProvider("NON_REAL_COURSE_AI_RECORD")
    chat.send(case.chats, identity, "NON_REAL_COURSE_USER_RECORD", "NON_REAL_ORIGIN_TURN",
              provider, allow_remote=True)
    return identity, provider


def test_review_extraction_uses_saved_content_and_manual_confirmation_with_exact_provenance(case):
    document, version = _saved(case)
    case.reviews.update_working(document["id"], "UNSAVED_WORKING_MUST_NOT_BE_EXTRACTED")
    document_before = copy.deepcopy(case.reviews.document(document["id"]))
    original = case.store.db.read_bytes()
    provider = RecordingProvider("PENDING_KNOWLEDGE_SENTINEL")
    extraction = case.memories.extract_review(document["id"], provider)
    messages = decode_conversation(provider.prompts[0])
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert payload["saved_review_content"] == SAVED
    assert "RAW_AI_MUST_NOT_BE_EXTRACTED" not in provider.prompts[0]
    assert "UNSAVED_WORKING_MUST_NOT_BE_EXTRACTED" not in provider.prompts[0]
    assert "UNSELECTED_5_1_SENTINEL" not in provider.prompts[0]
    assert extraction["status"] == "ready" and len(provider.prompts) == 1
    source = extraction["source"]
    assert source["type"] == "review_memory" and source["version_id"] == version["id"]
    assert source["document_id"] == document["id"] and source["item_ids"] == [case.selected.id]
    assert source["source_ids"] == [case.source.id] and source["subject"] == COURSE
    assert source["content_sha256"] == hashlib.sha256(SAVED.encode()).hexdigest()
    assert case.chats.memories(COURSE) == []
    conversation = case.chats.open_course(COURSE, new=True)
    prompt, _ = chat.prepare(case.chats, conversation, "PENDING_KNOWLEDGE_SENTINEL")
    assert "PENDING_KNOWLEDGE_SENTINEL" not in decode_conversation(prompt)[1]["content"]
    edited = r"CONFIRMED_KNOWLEDGE_SENTINEL $q=\frac{a}{b}$，人工核对条件。"
    revision = extraction["working_revision"]
    updated = case.memories.update_candidate(extraction["id"], edited, expected_revision=revision)
    with pytest.raises(StudyError):
        case.memories.update_candidate(extraction["id"], "STALE_EDIT", expected_revision=revision)
    assert updated["raw_ai"] == "PENDING_KNOWLEDGE_SENTINEL"
    assert case.chats.candidates(document["conversation_id"])[0]["content"] == edited
    with pytest.raises(StudyError):
        case.memories.confirm(extraction["id"], edited, confirmed=False)
    memory = case.memories.confirm(extraction["id"], edited, confirmed=True)
    row = case.chats.memories(COURSE)[0]
    assert row["id"] == memory and row["content"] == edited and row["sources"] == [source]
    assert row["source_status"]["active"] and not row["stale"]
    refs = chat.retrieve(case.chats, case.chats.conversation(conversation), edited)
    assert any(ref.get("text") == edited and ref["label"].startswith("M") for ref in refs)
    reopened = KnowledgeMemoryService(StudyStore(case.store.root))
    assert reopened.extraction(extraction["id"])["content"] == edited
    assert reopened.extraction(extraction["id"])["raw_ai"] == "PENDING_KNOWLEDGE_SENTINEL"
    assert case.reviews.document(document["id"]) == document_before
    assert case.store.db.read_bytes() == original
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert len(provider.prompts) == 1


def test_course_extraction_preserves_selected_real_turns_and_remains_valid_after_new_turns(case):
    identity, course_provider = _course(case)
    original = copy.deepcopy(case.chats.messages(identity))
    provider = RecordingProvider("COURSE_KNOWLEDGE_SENTINEL")
    extraction = case.memories.extract_course(identity, provider)
    payload = json.loads(decode_conversation(provider.prompts[0])[1]["content"].split("\n", 1)[1])
    assert payload["course_dialogue"] == [
        {"sequence": index, "role": message["role"], "content": message["content"]}
        for index, message in enumerate(original, 1)
    ]
    assert "UNSELECTED_5_1_SENTINEL" not in provider.prompts[0]
    source = extraction["source"]
    assert source["type"] == "course_dialogue_memory"
    assert source["conversation_id"] == identity
    assert source["message_ids"] == [message["id"] for message in original]
    assert source["content_sha256"] == digest([ChatStore.memory_message(m) for m in original])
    archive = case.chats.archived(extraction["archive_id"])
    assert archive["messages"] == original and archive["knowledge_memory_source"] == source
    assert case.chats.memories(COURSE) == []
    memory = case.memories.confirm(extraction["id"], "COURSE_KNOWLEDGE_SENTINEL", confirmed=True)
    chat.send(case.chats, identity, "COURSE_KNOWLEDGE_SENTINEL 新问题", "NEW_COURSE_TURN",
              course_provider, allow_remote=True)
    assert case.chats.messages(identity)[:2] == original
    assert case.chats.memory_status(memory)["active"]
    assert not case.chats.memory_status(memory)["stale"]
    assert case.chats.memories(COURSE)[0]["sources"] == [source]
    assert len(provider.prompts) == 1 and len(course_provider.prompts) == 2
    assert not case.store.sessions() and not case.store.attempts()


def test_review_memory_delete_restore_and_new_version_status_never_rewrite_confirmed_content(case):
    document, version = _saved(case)
    extraction = case.memories.extract_review(
        document["id"], RecordingProvider("ORIGINAL_CANDIDATE")
    )
    content = "IMMUTABLE_CONFIRMED_MEMORY_SENTINEL"
    memory = case.memories.confirm(extraction["id"], content, confirmed=True)
    original_source = copy.deepcopy(case.chats.memories(COURSE)[0]["sources"])
    for delete, restore in (
        (lambda: case.reviews.delete_document(document["id"], confirmed=True),
         lambda: case.reviews.restore_document(document["id"])),
        (lambda: case.store.delete_source(case.source.id, user_confirmed=True),
         lambda: case.store.restore_source(case.source.id, user_confirmed=True)),
        (lambda: case.chats.delete("course", COURSE, confirmed=True),
         lambda: case.chats.restore("course", COURSE)),
    ):
        delete()
        assert case.chats.memory_status(memory)["unavailable"]
        assert case.chats.memories(COURSE, include_stale=True) == []
        restore()
        assert case.chats.memory_status(memory)["active"]
        assert case.chats.memories(COURSE)[0]["content"] == content
    # Repeated saves are idempotent and must not invalidate an unchanged origin.
    versions = copy.deepcopy(case.reviews.versions(document["id"]))
    case.reviews.update_working(document["id"], SAVED)
    unchanged = case.reviews.save_version(document["id"])
    assert unchanged["id"] == version["id"]
    assert case.reviews.versions(document["id"]) == versions
    assert case.chats.memory_status(memory)["active"]
    assert not case.chats.memory_status(memory)["stale"]
    assert case.chats.memories(COURSE)[0]["content"] == content
    assert case.chats.memories(COURSE)[0]["sources"] == original_source
    # A genuinely revised saved origin still makes its prior memory stale.
    case.reviews.update_working(document["id"], SAVED + "\nNON_REAL_REVISED_SAVED_ORIGIN")
    newer = case.reviews.save_version(document["id"])
    assert newer["id"] != version["id"]
    state = case.chats.memory_status(memory)
    assert state["stale"] and not state["active"] and not state["unavailable"] and state["reasons"]
    assert case.chats.memories(COURSE) == []
    stale = case.chats.memories(COURSE, include_stale=True)[0]
    assert stale["content"] == content and stale["sources"] == original_source
    case.reviews.delete_document(document["id"], confirmed=True)
    case.reviews.restore_document(document["id"])
    assert case.chats.memory_status(memory)["stale"]  # Restore preserves genuine version staleness.
    assert case.chats.memories(COURSE) == []
    assert case.reviews.version(version["id"]) == version


def test_stale_extraction_cannot_be_confirmed_through_service_or_legacy_candidate_api(case):
    document, _ = _saved(case)
    provider = RecordingProvider("STALE_PENDING_KNOWLEDGE")
    extraction = case.memories.extract_review(document["id"], provider)
    case.reviews.update_working(document["id"], "NEW_SAVED_SOURCE_AFTER_EXTRACTION")
    case.reviews.save_version(document["id"])
    assert case.memories.status(extraction["id"])["stale"]
    with pytest.raises(StudyError):
        case.memories.confirm(extraction["id"], extraction["content"], confirmed=True)
    with pytest.raises(StudyError):
        case.chats.confirm(extraction["candidate_id"], extraction["content"], confirmed=True)
    assert case.chats.memories(COURSE, include_stale=True) == []
    assert case.memories.extraction(extraction["id"])["raw_ai"] == "STALE_PENDING_KNOWLEDGE"
    assert len(provider.prompts) == 1


def test_two_memory_types_deduplicate_without_losing_independent_provenance(case):
    # Selected messages predate the review, so contain no saved-review dependency.
    origin, _ = _course(case)
    document, _ = _saved(case)
    provider = RecordingProvider("SHARED_CONFIRMED_KNOWLEDGE")
    review = case.memories.extract_review(document["id"], provider)
    first = case.memories.confirm(review["id"], provider.content, confirmed=True)
    course = case.memories.extract_course(origin, provider)
    second = case.memories.confirm(course["id"], provider.content, confirmed=True)
    assert first == second
    row = case.chats.memories(COURSE)[0]
    assert {source["type"] for source in row["sources"]} == {
        "review_memory", "course_dialogue_memory",
    }
    assert len(row["links"]) == 2
    case.reviews.update_working(document["id"], "NEW_REVIEW_REVISION")
    case.reviews.save_version(document["id"])
    assert case.chats.memory_status(first)["active"]  # Independent course origin still supports it.
    assert not case.chats.memory_status(first)["stale"]
    case.chats.delete("conversation", origin, confirmed=True)
    assert not case.chats.memory_status(first)["active"]
    assert case.chats.memories(COURSE) == []
    case.chats.restore("conversation", origin)
    assert case.chats.memory_status(first)["active"]
    assert case.chats.memories(COURSE)[0]["content"] == "SHARED_CONFIRMED_KNOWLEDGE"
    assert len(provider.prompts) == 2


def test_invalid_memory_provenance_cannot_reenter_through_prior_course_evidence(case):
    origin, _ = _course(case)
    extraction = case.memories.extract_course(
        origin, RecordingProvider("MEMORY_SOURCE_GUARD_SENTINEL")
    )
    memory = case.memories.confirm(extraction["id"], "MEMORY_SOURCE_GUARD_SENTINEL", confirmed=True)
    consumer = case.chats.open_course(COURSE, new=True)
    provider = RecordingProvider("NON_REAL_CONSUMER_REPLY")
    chat.send(case.chats, consumer, "MEMORY_SOURCE_GUARD_SENTINEL", "USE_MEMORY",
              provider, allow_remote=True)
    snapshot = copy.deepcopy(case.chats.snapshot(consumer))
    assert any(ref["label"].startswith("M") for message in snapshot["messages"]
               for ref in message["metadata"].get("evidence", []))
    case.chats.delete("conversation", origin, confirmed=True)
    assert case.chats.memory_status(memory)["unavailable"]
    with pytest.raises(StudyError):
        chat.send(case.chats, consumer, "DO_NOT_REUSE_INVALID_MEMORY", "NO_SEND",
                  provider, allow_remote=True)
    assert len(provider.prompts) == 1 and case.chats.snapshot(consumer) == snapshot
    case.chats.restore("conversation", origin)
    assert case.chats.memory_status(memory)["active"]
    chat.prepare(case.chats, consumer, "RESTORED_ORIGINAL_MEMORY_SOURCE")


def test_changed_selected_course_message_is_stale_but_its_archived_source_is_immutable(case):
    origin, _ = _course(case)
    extraction = case.memories.extract_course(
        origin, RecordingProvider("COURSE_SOURCE_HASH_SENTINEL")
    )
    memory = case.memories.confirm(extraction["id"], "COURSE_SOURCE_HASH_SENTINEL", confirmed=True)
    archive = copy.deepcopy(case.chats.archived(extraction["archive_id"]))
    with case.chats.connect() as database:
        database.execute("UPDATE messages SET content=? WHERE id=?", (
            "NON_REAL_TAMPERED_CURRENT_MESSAGE", extraction["source"]["message_ids"][0],
        ))
    assert case.chats.memory_status(memory)["stale"]
    assert case.chats.memories(COURSE) == []
    assert case.chats.memories(COURSE, include_stale=True)[0]["content"] == (
        "COURSE_SOURCE_HASH_SENTINEL"
    )
    assert case.chats.archived(extraction["archive_id"]) == archive


def test_course_memory_tracks_the_saved_review_evidence_used_by_its_selected_messages(case):
    document, _ = _saved(case)
    origin, _ = _course(case)
    assert any(ref.get("kind") == "saved_review" for message in case.chats.messages(origin)
               for ref in message["metadata"].get("evidence", []))
    extraction = case.memories.extract_course(origin, RecordingProvider("R_DEPENDENT_KNOWLEDGE"))
    memory = case.memories.confirm(extraction["id"], "R_DEPENDENT_KNOWLEDGE", confirmed=True)
    assert case.chats.memory_status(memory)["active"]
    case.reviews.update_working(document["id"], "CHANGED_REVIEW_USED_BY_OLD_DIALOGUE")
    case.reviews.save_version(document["id"])
    assert case.chats.memory_status(memory)["stale"]
    assert case.chats.memories(COURSE) == []
    assert case.chats.memories(COURSE, include_stale=True)[0]["content"] == "R_DEPENDENT_KNOWLEDGE"


def test_legacy_memory_remains_usable_without_inventing_typed_provenance(case):
    origin, _ = _course(case)
    archive = case.chats.archive(origin)
    candidate = case.chats.add_candidate(archive, "NON_REAL_LEGACY_KNOWLEDGE")
    memory = case.chats.confirm(candidate, "NON_REAL_LEGACY_KNOWLEDGE", confirmed=True)
    row = case.chats.memories(COURSE)[0]
    assert row["id"] == memory and row["sources"] == []
    assert case.chats.memory_status(memory)["active"]
    assert row["links"][0]["archive_id"] == archive
    assert not case.store.sessions() and not case.store.attempts()


def test_legacy_candidate_prompt_preserves_exact_source_versions_and_cannot_confirm_memory(case):
    """The legacy candidate path gets the same provenance/learner boundary as newer extraction."""
    document, version = _saved(case)
    memory_text = "NON_REAL_CONFIRMED_MEMORY_SENTINEL"
    extracted = case.memories.extract_review(document["id"], RecordingProvider(memory_text))
    case.memories.confirm(extracted["id"], memory_text, confirmed=True)
    identity = case.chats.open_course(COURSE, new=True)
    chat.send(case.chats, identity, memory_text, "NON_REAL_SOURCE_TURN",
              RecordingProvider("NON_REAL_COURSE_REPLY"), allow_remote=True)
    before = copy.deepcopy(case.chats.messages(identity))
    original = case.store.db.read_bytes()
    memories = copy.deepcopy(case.chats.memories(COURSE))
    provider = RecordingProvider("NON_REAL_PENDING_NEXT_CANDIDATE")
    candidate = chat.propose(case.chats, identity, provider, allow_remote=True)
    prompt = provider.prompts[0]
    instructions, payload = prompt[:prompt.index("{")], json.loads(prompt[prompt.index("{"):])
    assert "只在真实作答或明确反馈有证据时保留个人障碍" in instructions
    assert "不推断已经作答、答错或掌握" in instructions
    assert "原标签及版本" in instructions and "不把历史正文拼成当前版本" in instructions
    assert "待人工审核的候选" in instructions and "不自动确认记忆" in instructions
    references = {ref["label"]: ref for ref in payload["evidence"]}
    assert {label[0] for label in references} == {"S", "R", "M"}
    original_refs = {ref["label"]: ref for message in before
                     for ref in message["metadata"].get("evidence", [])}
    assert references == original_refs
    assert next(ref for ref in references.values() if ref["label"].startswith("R"))[
        "version_id"
    ] == version["id"]
    assert "FOREIGN_COURSE_CONTENT_SENTINEL" not in prompt
    assert len(provider.prompts) == 1
    assert case.chats.messages(identity) == before
    assert case.chats.memories(COURSE) == memories
    assert any(value["id"] == candidate for value in case.chats.candidates(identity))
    assert case.store.db.read_bytes() == original
    assert not case.store.sessions() and not case.store.attempts()


def test_review_memory_tracks_current_course_without_changing_original_dialogue(case):
    document, version = _saved(case)
    original = copy.deepcopy(case.chats.snapshot(document["conversation_id"]))
    provider = RecordingProvider("ORIGINAL_COURSE_A_KNOWLEDGE")
    first = case.memories.extract_review(document["id"], provider)
    old_memory = case.memories.confirm(first["id"], provider.content, confirmed=True)
    case.chats.set_course(
        "NON_REAL_COURSE_B", [case.foreign.source_id, case.source.id], "NON_REAL_CURRENT_B_SCOPE"
    )
    current = copy.deepcopy(case.selected)
    current.subject = "NON_REAL_COURSE_B"
    case.store.confirm(current, user_confirmed=True)
    assert case.chats.memories(COURSE) == []
    assert not case.chats.memory_status(old_memory)["active"]
    managed = case.chats.memories(COURSE, include_stale=True, include_unavailable=True)
    assert managed[0]["content"] == "ORIGINAL_COURSE_A_KNOWLEDGE"
    provider.content = "CURRENT_COURSE_B_KNOWLEDGE"
    second = case.memories.extract_review(document["id"], provider)
    assert second["source"]["subject"] == "NON_REAL_COURSE_B"
    assert second["source"]["version_id"] == version["id"]
    memory = case.memories.confirm(second["id"], provider.content, confirmed=True)
    assert case.chats.memories("NON_REAL_COURSE_B")[0]["id"] == memory
    assert case.chats.memories("NON_REAL_COURSE_B")[0]["content"] == "CURRENT_COURSE_B_KNOWLEDGE"
    assert case.chats.memories(COURSE) == []
    assert case.chats.snapshot(document["conversation_id"]) == original
    assert len(provider.prompts) == 2
    assert not case.store.sessions() and not case.store.attempts()


def test_explicit_item_course_remains_authoritative_outside_course_source_configuration(case):
    case.chats.set_course(COURSE, [case.foreign.source_id], "NON_REAL_EXPLICIT_ITEM_SCOPE")
    document, version = _saved(case)
    entry = CourseReviewIndex(case.store).entry(document["id"], COURSE)
    assert entry["version_id"] == version["id"]
    study_before = case.store.db.read_bytes()
    provider = RecordingProvider("EXPLICIT_ITEM_COURSE_KNOWLEDGE")
    extraction = case.memories.extract_review(document["id"], provider)
    assert extraction["source"]["subject"] == COURSE
    assert extraction["source"]["source_ids"] == [case.source.id]
    memory = case.memories.confirm(extraction["id"], provider.content, confirmed=True)
    assert case.chats.memory_status(memory)["active"]
    assert case.chats.memories(COURSE)[0]["content"] == provider.content
    assert case.chats.memories("NON_REAL_COURSE_B") == []
    assert case.store.db.read_bytes() == study_before
    assert not case.store.sessions() and not case.store.attempts()


def test_extraction_preconditions_make_no_call_or_candidate_and_failure_never_retries_on_read(case):
    provider = RecordingProvider()
    document = case.reviews.generate([case.selected.id], case.review_provider)
    empty_course = case.chats.open_course(COURSE)
    for operation in (
        lambda: case.memories.extract_review(document["id"], provider),
        lambda: case.memories.extract_course(document["conversation_id"], provider),
        lambda: case.memories.extract_course(empty_course, provider),
    ):
        with pytest.raises(StudyError):
            operation()
    assert provider.prompts == [] and case.memories.extractions() == []
    case.reviews.save_version(document["id"])

    class Broken(RecordingProvider):
        def complete(self, prompt, **kwargs):
            self.prompts.append(prompt)
            raise RuntimeError("NON_REAL_PRIVATE_EXCEPTION_MUST_NOT_LEAK")

    broken = Broken()
    failed = case.memories.extract_review(document["id"], broken)
    assert failed["status"] == "failed" and failed["candidate_id"] is None
    assert "NON_REAL_PRIVATE_EXCEPTION_MUST_NOT_LEAK" not in json.dumps(failed)
    reopened = KnowledgeMemoryService(StudyStore(case.store.root))
    assert reopened.extraction(failed["id"]) == failed
    assert reopened.extractions(document["conversation_id"]) == [failed]
    reopened.status(failed["id"])
    with pytest.raises(StudyError):
        reopened.confirm(failed["id"], "NO_CANDIDATE", confirmed=True)
    assert len(broken.prompts) == 1
    assert case.chats.memories(None) == []
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
