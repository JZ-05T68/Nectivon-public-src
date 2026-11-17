"""Committed blue note bytes must agree across independent SQLite reads and immutable history."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.models import StudyError

BLUE = '<span style="color:#2563EB">NON_REAL 蓝色条件 $b\\ne0$</span>\n'


@pytest.fixture
def case(tmp_path):
    return review_case(tmp_path / "NON_REAL_SAVE_READBACK")


def _memory(case):
    conversation = case.chats.open_course("NON_REAL_COURSE_A", new=True)
    case.chats.begin(conversation, "NON_REAL_MEMORY", "NON_REAL_USER", {})
    case.chats.finish(conversation, "NON_REAL_MEMORY", "NON_REAL_SOURCE", {})
    archive = case.chats.archive(conversation)
    candidate = case.chats.add_candidate(archive, "NON_REAL_CANDIDATE")
    return case.chats.confirm(candidate, "NON_REAL_ORIGINAL", confirmed=True)


def _review(case):
    service = AIReviewService(case.store)
    document = service.generate([case.selected.id], RecordingProvider("NON_REAL_AI_ORIGINAL"))
    return service, document


def test_blue_memory_update_is_exactly_committed_once_and_independently_readable(case):
    identity = _memory(case)
    result = case.chats.update_memory(identity, BLUE, expected_revision=1)
    assert result["revision"] == 2 and result["content"] == BLUE
    same = case.chats.update_memory(identity, BLUE, expected_revision=2)
    assert same["revision"] == 2
    with sqlite3.connect(case.chats.db.as_uri() + "?mode=ro", uri=True) as check:
        assert check.execute("SELECT content FROM memories WHERE id=?", (identity,)).fetchone() == (
            BLUE,
        )
        assert check.execute(
            "SELECT revision,content FROM memory_revisions WHERE memory_id=? ORDER BY revision",
            (identity,),
        ).fetchall() == [(1, "NON_REAL_ORIGINAL"), (2, BLUE)]


def test_memory_cannot_claim_success_when_current_row_disagrees_after_commit(case):
    identity = _memory(case)
    with case.chats.connect() as db:
        db.execute("""CREATE TRIGGER corrupt_saved_memory AFTER UPDATE ON memories
            BEGIN UPDATE memories SET content='NON_REAL_MISMATCH' WHERE id=NEW.id; END""")
    with pytest.raises(StudyError, match="正文或修订号不一致"):
        case.chats.update_memory(identity, BLUE, expected_revision=1)
    assert case.chats.memory_revisions(identity)[0]["content"] == BLUE
    assert case.chats.memory_revisions(identity)[-1]["content"] == "NON_REAL_ORIGINAL"
    with pytest.raises(StudyError, match="其他页面更新"):
        case.chats.update_memory(identity, BLUE, expected_revision=1)
    assert len(case.chats.memory_revisions(identity)) == 2


def test_memory_cannot_claim_success_when_immutable_revision_disagrees(case):
    identity = _memory(case)
    with case.chats.connect() as db:
        db.execute("""CREATE TRIGGER corrupt_new_memory_revision AFTER INSERT ON memory_revisions
            WHEN NEW.revision>1 BEGIN UPDATE memory_revisions SET content='NON_REAL_MISMATCH'
            WHERE memory_id=NEW.memory_id AND revision=NEW.revision; END""")
    with pytest.raises(StudyError, match="正文或修订号不一致"):
        case.chats.update_memory(identity, BLUE, expected_revision=1)
    assert case.chats.memory(identity)["content"] == BLUE
    assert case.chats.memory_revisions(identity)[-1]["content"] == "NON_REAL_ORIGINAL"


def test_review_save_rejects_stale_snapshot_and_same_ack_is_idempotent(case):
    service, document = _review(case)
    working = service.update_working(document["id"], BLUE, expected_revision=0)
    with pytest.raises(StudyError, match="正文已变化"):
        service.save_version(document["id"], expected_revision=0, expected_content=BLUE)
    with pytest.raises(StudyError, match="正文已变化"):
        service.save_version(document["id"], expected_revision=1,
                             expected_content="NON_REAL_OLD_BODY")
    assert service.versions(document["id"]) == []
    first = service.save_version(document["id"], expected_revision=1, expected_content=BLUE)
    same_working = service.update_working(document["id"], BLUE, expected_revision=1)
    second = service.save_version(document["id"], expected_revision=1, expected_content=BLUE)
    assert same_working["working_revision"] == working["working_revision"] == 1
    assert first == second and len(service.versions(document["id"])) == 1
    with sqlite3.connect(case.chats.db.as_uri() + "?mode=ro", uri=True) as check:
        saved = check.execute("SELECT content FROM ai_review_versions WHERE id=?",
                              (first["id"],)).fetchone()[0]
        current = json.loads(check.execute(
            "SELECT payload FROM ai_review_documents WHERE id=?", (document["id"],),
        ).fetchone()[0])
    assert saved == current["working"] == BLUE
    assert current["last_saved_version_id"] == first["id"]
    assert current["raw_ai"] == "NON_REAL_AI_ORIGINAL"


def test_review_cannot_claim_success_when_immutable_version_disagrees_after_commit(case):
    service, document = _review(case)
    baseline = service.save_version(document["id"])
    service.update_working(document["id"], BLUE)
    with case.chats.connect() as db:
        db.execute("""CREATE TRIGGER corrupt_new_review_version AFTER INSERT ON ai_review_versions
            BEGIN UPDATE ai_review_versions SET content='NON_REAL_MISMATCH' WHERE id=NEW.id; END""")
    with pytest.raises(StudyError, match="正文或版本不一致"):
        service.save_version(document["id"], expected_revision=1, expected_content=BLUE)
    assert service.version(baseline["id"]) == baseline
    assert service.document(document["id"])["working"] == BLUE


def test_review_ui_missing_ack_receipt_never_saves_or_reports_success(case, monkeypatch):
    from src.study_loop import ai_review_ui as ui
    from src.study_loop import color_editor as editor

    service, document = _review(case)
    key = ui.PREFIX + "body_" + document["id"]
    state = {ui.PREFIX + "document_id": document["id"], key: "NON_REAL_UNACKNOWLEDGED"}
    fake = SimpleNamespace(session_state=state)
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(editor, "st", fake)
    monkeypatch.setattr(ui, "review_service", lambda store: service)
    ui._save(case.store, flushed=True)
    assert service.versions(document["id"]) == []
    assert service.document(document["id"])["working"] == "NON_REAL_AI_ORIGINAL"
    assert state[key] == "NON_REAL_UNACKNOWLEDGED"
    assert "最新输入尚未确认" in state[ui.PREFIX + "error"]
    assert ui.PREFIX + "flash" not in state


def test_review_ui_saves_exact_receipt_not_replayed_canonical_text(case, monkeypatch):
    from src.study_loop import ai_review_ui as ui
    from src.study_loop import color_editor as editor

    service, document = _review(case)
    key = ui.PREFIX + "body_" + document["id"]
    state = {
        ui.PREFIX + "document_id": document["id"], key: "NON_REAL_AI_ORIGINAL",
        ui.PREFIX + "revision_" + document["id"]: 0,
        ui.PREFIX + "loaded_" + document["id"]: "NON_REAL_AI_ORIGINAL",
        key + "_color_ack": 7, key + "_color_external": 2,
        key + "_color_flushed": {"text": BLUE, "serial": 7, "external_revision": 2, "token": 3},
    }
    fake = SimpleNamespace(session_state=state)
    monkeypatch.setattr(ui, "st", fake)
    monkeypatch.setattr(editor, "st", fake)
    monkeypatch.setattr(ui, "review_service", lambda store: service)
    ui._save(case.store, flushed=True)
    assert service.document(document["id"])["working"] == state[key] == BLUE
    assert [version["content"] for version in service.versions(document["id"])] == [BLUE]
    assert state[ui.PREFIX + "flash"] == "复盘已保存。"
    assert ui.PREFIX + "error" not in state
