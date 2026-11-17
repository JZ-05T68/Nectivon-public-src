"""Synthetic course constraints reach each review stage; no model/network calls."""

from __future__ import annotations

import json
from uuid import uuid4

from src.study_loop import model_review
from src.study_loop.chat_store import ChatStore
from tests.study_migration_support import text_case


def test_course_scope_is_frozen_per_review_and_reaches_all_stages(tmp_path):
    store, source, draft = text_case(tmp_path / "study")
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    chats = ChatStore(store)
    chats.set_course(draft.subject, [source.id], "NON_REAL_TEST_DATA OLD_SCOPE")
    payload = {
        "item_id": draft.id, "unit_id": draft.question.units()[0].id,
        "session_id": session, "run_id": uuid4().hex,
        "input_mode": "text", "text_complete": True,
        "allow_remote": True, "real_material_confirmed": True, "max_extra_attempts": 0,
        "selections": [{"provider": "deepseek", "model": "TEST_ONLY"},
                       {"provider": "qwen", "model": "TEST_ONLY"}],
    }
    old = model_review.prepare(store, payload)
    notes = "NON_REAL_TEST_DATA NEW_COURSE_SCOPE"
    chats.set_course(draft.subject, [source.id], notes)
    prompts = []

    class Context:
        def step(self, name, operation, **kwargs):
            return operation("NON_REAL_TEST_DATA")

        def check_control(self):
            return None

    class Provider:
        def __init__(self, provider):
            self.provider = provider

        def review(self, prompt, *, image=None):
            prompts.append(prompt)
            if prompt.startswith(model_review.SUMMARY_PROMPT):
                answer = {k: "NON_REAL_TEST_DATA" for k in model_review.SUMMARY_TEXT}
                answer.update({k: [] for k in model_review.SUMMARY_LISTS})
            else:
                answer = {k: "NON_REAL_TEST_DATA " + self.provider
                          for k in model_review.ANALYSIS_TEXT}
                answer.update({k: [] for k in model_review.ANALYSIS_LISTS})
            return {
                "status": "ok", "text": json.dumps(answer), "provider": self.provider,
                "requested_model": "TEST_ONLY", "reported_model": "TEST_ONLY",
            }

    model_review.run_review(store, Context(), payload, [Provider("deepseek"), Provider("qwen")])
    assert len(prompts) == 5  # Two analyses, two disagreement checks, one summary.
    assert all(notes in prompt for prompt in prompts)
    assert all("user_provided_exam_scope" in prompt for prompt in prompts)
    assert old["course_scope"]["exam_notes"] == "NON_REAL_TEST_DATA OLD_SCOPE"
    assert store.item(draft.id).question == draft.question
