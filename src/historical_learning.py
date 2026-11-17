"""Read-only source projection for retained historical assets, not a learning engine."""

import sqlite3
from pathlib import Path


def question_source(database_path: Path, question_id: int) -> dict | None:
    """Read an old item's anchor without creating a DB or running migrations."""
    connection = sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT document_id,page_id FROM question_items WHERE id=?", (question_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()
