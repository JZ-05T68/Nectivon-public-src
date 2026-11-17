"""Existing course chats see new saved reviews while retaining their raw-source snapshots."""

from __future__ import annotations

import copy
import json

from test_ai_review_support import RecordingProvider, review_case

from src.ai.conversation import decode_conversation
from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.models import Draft, Page, Question


def test_existing_course_session_retrieves_new_same_course_saved_review(tmp_path):
    case = review_case(tmp_path / "NON_REAL_COURSE_DYNAMIC_SYNC")
    identity = case.chats.open_course(case.selected.subject)
    snapshot = copy.deepcopy(case.chats.conversation(identity))
    assert snapshot["source_ids"] == [case.source.id]
    source = case.store.receive("NON_REAL_NEW_COURSE_MATERIAL.txt", b"NON_REAL_NEW_PROBLEM", "txt")
    page = Page("NON_REAL_NEW_PAGE", source.id, None, "NON_REAL_NEW_PROBLEM")
    case.store.put_page(page)
    item = Draft(
        "NON_REAL_NEW_COURSE_ITEM", source.id, [page.id],
        Question(content="NON_REAL_NEW_PROBLEM"), subject=case.selected.subject,
    )
    case.store.add_draft(item)
    case.chats.set_course(case.selected.subject, [case.source.id, source.id], "USER_SCOPE_NEW")
    service, provider = AIReviewService(case.store), RecordingProvider("NON_REAL_RAW_CANDIDATE")
    document = service.generate([item.id], provider)
    saved_content = "NON_REAL_NEW_SAVED_REVIEW $a=b+c$"
    service.update_working(document["id"], saved_content)
    version = service.save_version(document["id"])
    service.update_working(document["id"], "NON_REAL_UNSAVED_MUST_NOT_SYNC")
    study_before, chat_before = case.store.db.read_bytes(), case.chats.db.read_bytes()

    prompt, evidence = chat.prepare(case.chats, identity, "NON_REAL_NEW_SAVED_REVIEW")
    reviews = [ref for ref in evidence if ref.get("kind") == "saved_review"]
    assert len(reviews) == 1 and reviews[0]["version_id"] == version["id"]
    assert reviews[0]["source_ids"] == [source.id] and reviews[0]["text"] == saved_content
    assert not any(ref.get("source_id") == source.id for ref in evidence)
    messages = decode_conversation(prompt)
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    context = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert "当前课程的保存版视图" in context["saved_review_guidance"]
    assert "NON_REAL_UNSAVED_MUST_NOT_SYNC" not in prompt
    assert "NON_REAL_RAW_CANDIDATE" not in prompt
    assert case.chats.conversation(identity) == snapshot
    assert case.chats.messages(identity) == []
    assert case.store.db.read_bytes() == study_before
    assert case.chats.db.read_bytes() == chat_before
    assert len(provider.prompts) == 1
