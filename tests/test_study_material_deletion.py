"""NON_REAL_TEST_DATA: reversible material removal preserves source bytes and study history."""

from __future__ import annotations

import hashlib

import pytest
from fixtures.phase12_study import seed_study

from src.study_loop.material_management_ui import visible_session_items
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore


def test_remove_requires_confirmation_and_restore_recovers_all_entries(tmp_path):
    store = StudyStore(tmp_path / "NON_REAL_TEST_DATA", create=True)
    ids = seed_study(store)
    source = store.source(ids["source_id"])
    originals = {
        path.relative_to(store.root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.rglob("*") if path.is_file() and path != store.db
    }
    items, drafts, attempts = store.items(), store.drafts(), store.attempts()
    with pytest.raises(StudyError, match="确认删除"):
        store.delete_source(source.id, user_confirmed=False)
    assert store.sources() == [source]
    store.delete_source(source.id, user_confirmed=True)
    assert store.sources() == store.items() == store.drafts() == []
    assert store.deleted_sources() == [source]
    assert all(not visible_session_items(store, session) for session in store.sessions())
    assert store.attempts() == attempts
    assert originals == {
        path.relative_to(store.root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.rglob("*") if path.is_file() and path != store.db
    }
    with pytest.raises(StudyError, match="最近删除"):
        store.receive(source.name, store.read_source(source), source.kind)
    with pytest.raises(StudyError, match="已删除"):
        store.create_revision(items[0].id, user_requested=True)
    with pytest.raises(StudyError, match="已删除"):
        store.confirm(drafts[0], user_confirmed=True)
    with pytest.raises(StudyError, match="确认恢复"):
        store.restore_source(source.id, user_confirmed=False)
    store.restore_source(source.id, user_confirmed=True)
    assert store.sources() == [source]
    assert store.items() == items
    assert store.drafts() == drafts
    assert store.attempts() == attempts
    assert store.deleted_sources() == []


def test_opening_old_library_does_not_write_trash_schema(tmp_path):
    store = StudyStore(tmp_path / "NON_REAL_TEST_DATA", create=True)
    before = store.db.read_bytes()
    assert store.deleted_sources() == []
    assert store.sources() == store.items() == store.drafts() == []
    assert store.db.read_bytes() == before
