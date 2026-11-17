"""Synthetic AI-review fixtures; no user materials, runtime models or network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ai.provider import CompletionResult
from src.study_loop.chat_store import ChatStore
from src.study_loop.models import Draft, Page, Question, Source
from src.study_loop.storage import StudyStore

FORMULA_ANSWER = (
    "NON_REAL_TEST_DATA 复盘讲义\n\n"
    + r"$$F=\frac{P}{v}$$"
    + "\n\n|变量|含义|\n|---|---|\n|$F$|NON_REAL_FORCE|\n\n"
    "没有实际作答证据，个人错因留空。"
)


class RecordingProvider:
    """Record exact completion envelopes while producing visibly synthetic content."""

    def __init__(self, content: str = FORMULA_ANSWER) -> None:
        self.content = content
        self.prompts: list[str] = []
        self.options: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        self.prompts.append(prompt)
        self.options.append(kwargs)
        return CompletionResult(
            text=self.content, model="NON_REAL_REVIEW_MODEL", finish_reason="stop"
        )


@dataclass
class ReviewCase:
    """Two questions share a page; a third material belongs to another course."""

    store: StudyStore
    chats: ChatStore
    source: Source
    selected: Draft
    sibling: Draft
    foreign: Draft


def review_case(root: Path) -> ReviewCase:
    """Create real current stores, preserving a deliberate same-page isolation trap."""
    store = StudyStore(root, create=True)
    raw = (
        "NON_REAL_TEST_DATA\n"
        "4-2 SELECTED_4_2_QUESTION: $F=P/v$，请推导关系。\n"
        "5-1 UNSELECTED_5_1_SENTINEL：OTHER_ONLY_SPEED_8412，OTHER_ONLY_R_0_987。"
    ).encode()
    source = store.receive("NON_REAL_SHARED_MATERIAL.txt", raw, "txt")
    page = Page("NON_REAL_SHARED_PAGE", source.id, None, raw.decode())
    store.put_page(page)
    selected = Draft(
        "NON_REAL_SELECTED_4_2", source.id, [page.id],
        Question(
            id="NON_REAL_SELECTED_UNIT", number="4-2",
            content=r"SELECTED_4_2_QUESTION：由 $F=\frac{P}{v}$ 推导关系。",
            observed_answer="IMPORTED_REFERENCE_CLAIM_NOT_PERSONAL_ANSWER",
            observed_analysis="HISTORICAL_AI_EXPLANATION_NOT_PERSONAL_ATTEMPT",
            reference_answer="USER_ANNOTATION_NOT_PERSONAL_ATTEMPT",
            answer_status="user_note",
            obstacle="IMPORTED_OBSTACLE_IS_NOT_AN_ATTEMPT",
            obstacle_evidence="SOURCE_TEXT_IS_NOT_ACTUAL_USER_WORK",
        ), subject="NON_REAL_COURSE_A",
    )
    sibling = Draft(
        "NON_REAL_UNSELECTED_5_1", source.id, [page.id],
        Question(
            id="NON_REAL_SIBLING_UNIT", number="5-1",
            content="UNSELECTED_5_1_SENTINEL OTHER_ONLY_SPEED_8412 OTHER_ONLY_R_0_987",
            observed_analysis="UNSELECTED_IMPORTED_AI_SENTINEL",
        ), subject="NON_REAL_COURSE_A",
    )
    foreign_source = store.receive(
        "NON_REAL_OTHER_COURSE.txt", b"FOREIGN_COURSE_CONTENT_SENTINEL", "txt"
    )
    foreign_page = Page(
        "NON_REAL_FOREIGN_PAGE", foreign_source.id, None, "FOREIGN_COURSE_CONTENT_SENTINEL"
    )
    store.put_page(foreign_page)
    foreign = Draft(
        "NON_REAL_FOREIGN_ITEM", foreign_source.id, [foreign_page.id],
        Question(id="NON_REAL_FOREIGN_UNIT", content="FOREIGN_COURSE_CONTENT_SENTINEL"),
        subject="NON_REAL_COURSE_B",
    )
    for draft in (selected, sibling, foreign):
        store.add_draft(draft)
    chats = ChatStore(store)
    chats.set_course("NON_REAL_COURSE_A", [source.id], "USER_SCOPE_A_ONLY_NON_REAL")
    chats.set_course(
        "NON_REAL_COURSE_B", [foreign_source.id], "USER_SCOPE_B_FORBIDDEN_NON_REAL"
    )
    return ReviewCase(store, chats, source, selected, sibling, foreign)


def record_actual_attempt(case: ReviewCase, item: Draft, marker: str) -> None:
    """Append an explicit synthetic personal attempt through the real storage service."""
    case.store.confirm(item, user_confirmed=True)
    session = case.store.start_review([item.id], user_started=True)
    case.store.record_attempt(
        session, item.id, item.question.units()[0].id,
        outcome="partial", answer=marker, note="NON_REAL_EXPLICIT_USER_OBSTACLE",
        independent=True, attempted=True, error_tags=["NON_REAL_USER_CONFIRMED_TAG"],
        submission_id="NON_REAL_ATTEMPT_" + item.id, error_confirmed=True,
    )
