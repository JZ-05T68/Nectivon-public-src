"""NON_REAL_TEST_DATA: exact-count, identity and read-only learning view contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from fixtures.phase12_study import seed_study
from streamlit.testing.v1 import AppTest

from src.study_loop.insights import distribution, error_evidence, evidence_counts, recent_reviews
from src.study_loop.insights_ui import _literal_label
from src.study_loop.models import Page, StudyError
from src.study_loop.storage import StudyStore, encode


@pytest.fixture
def store(tmp_path):
    value = StudyStore(tmp_path / "NON_REAL_TEST_DATA", create=True)
    seed_study(value)
    return value


def records(store):
    return error_evidence(store.items(), store.attempt_records())


def test_exact_counts_and_underlying_group_membership(store):
    evidence = records(store)
    assert evidence_counts(evidence) == {
        "questions": 2, "attempts": 3, "markers": 1, "unknown_time": 1,
        "unclassified": 1, "untagged": 1, "tag_assignments": 4,
    }
    grouped = distribution(evidence, "tags")
    assert set(grouped) == {"TEST_REASON_A", "TEST_REASON_B", "未记录错因"}
    assert {r.id for r in grouped["TEST_REASON_A"]} == {
        "marker:TEST_ITEM_01:TEST_UNIT_01", "TEST_ATTEMPT_01", "TEST_ATTEMPT_02",
    }
    assert {r.question_key for r in grouped["未记录错因"]} == {("TEST_ITEM_02", "TEST_UNIT_02")}
    assert len(distribution(evidence, "knowledge")) == 3
    assert len(distribution(evidence, "chapter")) == 2


@pytest.mark.parametrize("patch", [
    {"source": "model_suggestion"}, {"error_confirmed": None}, {"error_confirmed": False},
    {"outcome": "correct"}, {"outcome": "skipped"}, {"answer": ""},
    {"identity_valid": False}, {"item_id": "TEST_MISSING"}, {"unit_id": "TEST_MISSING"},
])
def test_non_errors_and_unreliable_links_are_excluded(store, patch):
    attempt = {**store.attempt_records()[0], **patch}
    found = error_evidence(store.items(), [attempt])
    assert all(record.origin == "manual_marker" for record in found)


def test_material_error_draft_note_and_import_do_not_count(store):
    assert all(r.item_id not in {"TEST_ITEM_03", "TEST_ITEM_04"} for r in records(store))
    item = store.item("TEST_ITEM_01")
    item.material_kind = "note"
    assert error_evidence([item], store.attempt_records()) == []
    assert error_evidence([], []) == []


def test_manual_marker_unknown_time_and_confirmed_obstacle_only(store):
    item = store.item("TEST_ITEM_01")
    item.question.obstacle_evidence = ""
    marker = error_evidence([item], [])[0]
    assert marker.created_at is None
    assert marker.tags == ()


def test_repeat_event_deduplicates_but_revision_identity_does_not(store):
    attempts = store.attempt_records()
    assert records(store) == error_evidence(store.items(), attempts + attempts)
    original = store.item("TEST_ITEM_01")
    revision = deepcopy(original)
    revision.id, revision.revision_of = "TEST_REVISION_01", original.id
    evidence = error_evidence([original, revision], attempts)
    assert evidence_counts(evidence)["questions"] == 2


def test_parent_marker_does_not_mark_every_child(store):
    item = store.item("TEST_ITEM_01")
    first = deepcopy(item.question)
    first.id, first.attribute = "TEST_CHILD_01", "normal"
    second = deepcopy(first)
    second.id = "TEST_CHILD_02"
    item.question.children = [first, second]
    assert [r.unit_id for r in error_evidence([item], [])] == ["TEST_UNIT_01"]


def test_same_label_different_course_has_separate_distribution(store):
    first = records(store)[0]
    second = replace(first, id="TEST_OTHER", item_id="TEST_OTHER", subject="TEST_SUBJECT_02")
    assert len(distribution([first, second], "knowledge")) == 4


def test_storage_exposes_authoritative_id_and_detects_payload_mismatch(store):
    with store.connect(write=True) as connection:
        row = connection.execute(
            "SELECT payload FROM attempts WHERE id='TEST_ATTEMPT_01'",
        ).fetchone()
        payload = json.loads(row[0])
        payload["item_id"] = "TEST_ITEM_02"
        connection.execute(
            "UPDATE attempts SET payload=? WHERE id='TEST_ATTEMPT_01'", (encode(payload),),
        )
    attempt = store.attempt_records()[0]
    assert attempt["item_id"] == "TEST_ITEM_01"
    assert attempt["identity_valid"] is False
    assert evidence_counts(records(store))["attempts"] == 2


def test_error_confirmation_requires_actual_answer_and_is_compatible(store):
    kwargs = dict(
        outcome="not_yet", answer="", note="NON_REAL_TEST_DATA", independent=False,
        attempted=True, error_tags=[], submission_id="TEST_NEW", error_confirmed=True,
    )
    with pytest.raises(StudyError, match="实际作答文字"):
        store.record_attempt("TEST_SESSION_02", "TEST_ITEM_02", "TEST_UNIT_02", **kwargs)
    kwargs.update(answer="NON_REAL_TEST_DATA", outcome="correct", independent=True)
    with pytest.raises(StudyError, match="不能同时"):
        store.record_attempt("TEST_SESSION_02", "TEST_ITEM_02", "TEST_UNIT_02", **kwargs)
    kwargs.update(outcome="partial")
    store.record_attempt("TEST_SESSION_02", "TEST_ITEM_02", "TEST_UNIT_02", **kwargs)
    store.record_attempt("TEST_SESSION_02", "TEST_ITEM_02", "TEST_UNIT_02", **kwargs)
    assert sum(a["id"] == "TEST_NEW" for a in store.attempt_records()) == 1
    assert "error_confirmed" not in store.attempts()[3]


def test_recent_sort_status_and_rounds_are_exact_and_stable(store):
    events = recent_reviews(store.items(), store.sessions(), store.attempt_records())
    assert [e.id for e in events] == ["TEST_SESSION_02", "TEST_SESSION_01"]
    assert [e.status for e in events] == ["in_progress", "completed"]
    assert [len(e.attempts) for e in events] == [2, 2]
    sessions = [dict(store.sessions()[0], id=f"TEST_SESSION_{n:02d}") for n in range(20)]
    sessions[3]["started_at"] = None
    sessions[4]["started_at"] = "invalid"
    limited = recent_reviews(store.items(), sessions, [])
    assert len(limited) == 10
    assert [e.id for e in limited] == sorted(e.id for e in limited)
    assert all(e.started_at not in (None, "invalid") for e in limited)
    assert len(recent_reviews(store.items(), sessions, [], limit=None)) == 20
    assert recent_reviews([], [], []) == []


def test_missing_reference_and_retry_do_not_guess_or_duplicate(store):
    session = dict(store.sessions()[0], item_ids=encode(["TEST_MISSING"]))
    event = recent_reviews(store.items(), [session, session], store.attempt_records())[0]
    assert event.unavailable_items == ("TEST_MISSING",)
    assert event.attempts == ()
    event = recent_reviews([], [dict(session, item_ids="broken")], [])[0]
    assert event.invalid_reference is True


def test_views_and_selections_preserve_database_bytes_and_sources(store):
    before = hashlib.sha256(store.db.read_bytes()).hexdigest()
    app = AppTest.from_string(f'''
from pathlib import Path
from src.study_loop.storage import StudyStore
from src.study_loop.insights_ui import render_error_profile, render_recent_reviews
store = StudyStore(Path({str(store.root)!r}))
render_error_profile(store)
render_recent_reviews(store)
''').run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["2", "3", "1"]
    app.radio(key="sl_insight_dimension").set_value("tags").run()
    app.button(key="sl_insight_group_0").click().run()
    app.button(key="sl_insight_open_TEST_ATTEMPT_01").click().run()
    assert not app.exception
    assert app.session_state["sl_insight_record"] == "TEST_ATTEMPT_01"
    assert any("NON_REAL_TEST_DATA TEST_RESPONSE_01" in m.value for m in app.markdown)
    assert all("记录 ID：" not in c.value for c in app.caption)
    app.button(key="sl_recent_open_TEST_SESSION_01").click().run()
    assert app.session_state["sl_recent_record"] == "TEST_SESSION_01"
    assert any("开始：" in c.value for c in app.caption)
    assert all("复盘 ID：" not in c.value for c in app.caption)
    app.run()
    assert hashlib.sha256(store.db.read_bytes()).hexdigest() == before
    assert store.read_source(store.sources()[0]) == b"NON_REAL_TEST_DATA TEST_SOURCE_01"


def test_missing_source_and_safe_text_keep_record_available(store):
    # Only an isolated test original is moved; the production code never mutates sources.
    source = store.sources()[0]
    store.path(source.blob).rename(store.path(source.blob + ".fixture-hidden"))
    app = AppTest.from_string(f'''
from pathlib import Path
from src.study_loop.storage import StudyStore
from src.study_loop.insights_ui import render_error_profile
render_error_profile(StudyStore(Path({str(store.root)!r})))
''').run()
    app.button(key="sl_insight_open_TEST_ATTEMPT_01").click().run()
    assert not app.exception
    assert any("原始来源缺失" in w.value for w in app.warning)


def test_imported_distribution_label_cannot_activate_markdown_images():
    label = _literal_label("TEST_TAG ![image](https://example.invalid/x.png)")
    assert r"\!\[image\]\(https://example\.invalid/x\.png\)" in label


def test_source_captions_neutralize_images_without_changing_originals(tmp_path):
    """NON_REAL_TEST_DATA: short original images and metadata stay passive on source lookup."""
    store = StudyStore(tmp_path / "NON_REAL_TEST_DATA_SOURCE", create=True)
    image = "![x](https://x.test/a)"
    raw_text = image + "\nNON_REAL_TEST_DATA"
    original = raw_text.encode("utf-8")
    source = store.receive("NON_REAL_TEST_DATA.txt", original, "txt")
    pages = [
        Page("TEST_SHORT_IMAGE", source.id, None, text=raw_text, metadata={"warning": image}),
        Page("TEST_METADATA_IMAGE", source.id, None, text=raw_text, metadata={
            "conversation": {"title": image}, "pointer": image, "warning": image,
            "created_at": image, "text_basis": image, "character_start": 0,
            "character_end": 10, "character_basis": image,
        }),
    ]
    for page in pages:
        store.put_page(page)
    before = hashlib.sha256(store.db.read_bytes()).hexdigest()
    app = AppTest.from_string(f'''
from pathlib import Path
from src.study_loop.storage import StudyStore
from src.study_loop.import_ui import render_source
render_source(StudyStore(Path({str(store.root)!r})), {source.id!r},
              ["TEST_SHORT_IMAGE", "TEST_METADATA_IMAGE"], prefix="safety_fixture")
''').run()
    assert not app.exception
    assert not app.caption
    assert not app.text_area
    assert not app.get("imgs")
    assert all(image not in element.value for element in app.markdown)
    app.run()
    assert not app.exception
    assert hashlib.sha256(store.db.read_bytes()).hexdigest() == before
    assert store.read_source(source) == original
    assert store.pages(source.id) == pages
