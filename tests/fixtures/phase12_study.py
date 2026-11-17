"""NON_REAL_TEST_DATA: infrastructure identifiers only; never seed a production view."""

from __future__ import annotations

import json
from pathlib import Path

from src.study_loop.models import Draft, Page, Question
from src.study_loop.storage import StudyStore, encode


def seed_study_root(root: Path) -> dict[str, str]:
    """Explicit helper for the caller's isolated NON_REAL_TEST_DATA browser instance."""
    return seed_study(StudyStore(root, create=True))


def seed_study(store: StudyStore) -> dict[str, str]:
    """Seed an explicitly supplied empty isolated test store, never on app startup."""
    if store.sources() or store.items() or store.sessions():
        raise ValueError("NON_REAL_TEST_DATA requires an empty test store")
    source = store.receive(
        "NON_REAL_TEST_DATA.txt", b"NON_REAL_TEST_DATA TEST_SOURCE_01", "txt",
    )
    page = Page("TEST_PAGE_01", source.id, None, text="NON_REAL_TEST_DATA TEST_SOURCE_01")
    store.put_page(page)
    for number in ("01", "02", "03", "04"):
        item = Draft(
            id=f"TEST_ITEM_{number}", source_id=source.id, page_ids=[page.id],
            question=Question(
                id=f"TEST_UNIT_{number}", content=f"NON_REAL_TEST_DATA TEST_ITEM_{number}",
            ),
        )
        if number == "01":
            item.subject, item.chapter = "TEST_SUBJECT", "TEST_CHAPTER"
            item.question.attribute = "error"
            item.question.knowledge = ["TEST_POINT_A", "TEST_POINT_B"]
            item.question.obstacle = "TEST_REASON_A"
            item.question.obstacle_evidence = "NON_REAL_TEST_DATA TEST_MARKER_EVIDENCE"
        if number == "03":
            item.verification_status, item.verification_note = "error", "NON_REAL_TEST_DATA"
        if number == "04":
            item.question.attribute = "error"
        store.add_draft(item)
        if number != "04":
            store.confirm(item, user_confirmed=True)
    first = store.start_review(["TEST_ITEM_01"], user_started=True)
    with store.connect(write=True) as connection:
        connection.execute(
            "UPDATE sessions SET id=?,started_at=? WHERE id=?",
            ("TEST_SESSION_01", "2026-01-01T02:00:00+00:00", first),
        )
    for suffix, tags in (("01", ["TEST_REASON_A", "TEST_REASON_B"]), ("02", ["TEST_REASON_A"])):
        store.record_attempt(
            "TEST_SESSION_01", "TEST_ITEM_01", "TEST_UNIT_01", outcome="partial",
            answer="NON_REAL_TEST_DATA TEST_RESPONSE_" + suffix, note="NON_REAL_TEST_DATA",
            independent=True, attempted=True, error_tags=tags,
            submission_id="TEST_ATTEMPT_" + suffix, error_confirmed=True,
        )
    store.end_review("TEST_SESSION_01")
    second = store.start_review(["TEST_ITEM_02", "TEST_ITEM_03"], user_started=True)
    with store.connect(write=True) as connection:
        connection.execute(
            "UPDATE sessions SET id=?,started_at=? WHERE id=?",
            ("TEST_SESSION_02", "2026-01-02T02:00:00+00:00", second),
        )
        connection.execute(
            "UPDATE sessions SET ended_at=? WHERE id='TEST_SESSION_01'",
            ("2026-01-01T03:00:00+00:00",),
        )
    store.record_attempt(
        "TEST_SESSION_02", "TEST_ITEM_02", "TEST_UNIT_02", outcome="not_yet",
        answer="NON_REAL_TEST_DATA TEST_RESPONSE_03", note="NON_REAL_TEST_DATA",
        independent=True, attempted=True, error_tags=[],
        submission_id="TEST_ATTEMPT_03", error_confirmed=True,
    )
    store.record_attempt(
        "TEST_SESSION_02", "TEST_ITEM_03", "TEST_UNIT_03", outcome="not_yet",
        answer="", note="NON_REAL_TEST_DATA TEST_INCOMPLETE", independent=False,
        attempted=True, error_tags=["TEST_SUGGESTION"], submission_id="TEST_ATTEMPT_04",
    )
    with store.connect(write=True) as connection:
        for row in connection.execute("SELECT id,payload FROM attempts").fetchall():
            payload = json.loads(row["payload"])
            payload["created_at"] = (
                "2026-01-01T02:10:00+00:00" if row["id"] < "TEST_ATTEMPT_03"
                else "2026-01-02T02:10:00+00:00"
            )
            connection.execute(
                "UPDATE attempts SET payload=? WHERE id=?", (encode(payload), row["id"]),
            )
    return {"source_id": source.id, "page_id": page.id}
