"""NON_REAL_TEST_DATA: seed retained SQL rows for current graph/deletion tests.

This fixture does not restore the retired learning services or simulate their
business behavior. The application under test remains the current implementation.
Every caller supplies a newly initialized pytest temporary database.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.database import Database, _tokenize_for_fts

STAMP = "2026-01-01T00:00:00+00:00"
TABLES = frozenset({"question_items", "question_families", "question_family_members",
                   "mastery_records", "question_groups", "question_group_materials",
                   "question_item_evidence"})


def seed_row(database: Database, table: str, **values: Any) -> int:
    """Insert an explicit synthetic row using the real current schema/constraints."""
    assert table in TABLES
    with database._connection() as connection:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        assert values.keys() <= columns
        for name in ("created_at", "updated_at"):
            if name in columns:
                values.setdefault(name, STAMP)
        if table == "mastery_records":
            values.setdefault("practiced_at", STAMP)
        encoded = {key: json.dumps(value, ensure_ascii=False)
                   if isinstance(value, (dict, list)) else value for key, value in values.items()}
        cursor = connection.execute(
            f"INSERT INTO {table} ({','.join(encoded)}) VALUES ({','.join('?' for _ in encoded)})",
            tuple(encoded.values()),
        )
        return int(cursor.lastrowid)


def seed_question(database: Database, **values: Any) -> SimpleNamespace:
    """Seed historical question content plus the source snapshot a migrated row retains."""
    values.setdefault("search_stem_text", _tokenize_for_fts(values.get("stem_text", "")))
    if values.get("document_id"):
        document = database.get_document(values["document_id"])
        values.setdefault("source_document_title_snapshot", document.title)
    if values.get("page_id"):
        page = database.get_page(values["page_id"])
        values.setdefault("source_page_label_snapshot", f"第 {page.page_number} 页")
    identifier = seed_row(database, "question_items", **values)
    return read_question(database, identifier)


def read_question(database: Database, identifier: int) -> SimpleNamespace:
    """Read actual SQL contents after the real service has run; no synthesized state."""
    with database._connection() as connection:
        row = connection.execute(
            "SELECT * FROM question_items WHERE id=?", (identifier,),
        ).fetchone()
    assert row is not None
    values = dict(row)
    if values["ai_draft"] is not None:
        values["ai_draft"] = json.loads(values["ai_draft"])
    return SimpleNamespace(**values)


def content_revisions(database: Database, identifier: int) -> list[dict[str, Any]]:
    """Read persisted revisions, without creating a replacement revision service."""
    with database._connection() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM question_content_revisions WHERE question_id=? ORDER BY id",
            (identifier,),
        )]
