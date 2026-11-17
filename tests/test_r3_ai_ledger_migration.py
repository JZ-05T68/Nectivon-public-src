"""Ledger CHECK widening preserves existing values, sequences, indexes and triggers."""

import sqlite3

import pytest

from src.migrations import MigrationError, _apply_version_forty


def _old_ledger():
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE ai_calls(
            id INTEGER PRIMARY KEY AUTOINCREMENT, call_uuid TEXT NOT NULL UNIQUE,
            model TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'qwen'
                CHECK (provider IN ('deepseek', 'qwen', 'kimi', 'hunyuan')),
            resolved_model TEXT
        );
        CREATE INDEX synthetic_ledger_idx ON ai_calls(provider, model);
        CREATE TABLE counter(value INTEGER NOT NULL);
        INSERT INTO counter VALUES (0);
        CREATE TRIGGER synthetic_ledger_trigger AFTER INSERT ON ai_calls
            BEGIN UPDATE counter SET value=value+1; END;
        INSERT INTO ai_calls(id,call_uuid,model,provider,resolved_model)
            VALUES (1,'historical-1','kimi-k3','kimi','actual-historical');
        INSERT INTO ai_calls(id,call_uuid,model,provider)
            VALUES (99,'synthetic-deleted-gap','qwen3.8-max','qwen');
        DELETE FROM ai_calls WHERE id=99;
    """)
    connection.commit()
    return connection


def test_v40_ledger_migration_preserves_values_and_allows_honest_new_provider():
    connection = _old_ledger()
    before = connection.execute("SELECT * FROM ai_calls ORDER BY id").fetchall()
    _apply_version_forty(connection)
    assert connection.execute("SELECT * FROM ai_calls ORDER BY id").fetchall() == before
    assert connection.execute("SELECT value FROM counter").fetchone()[0] == 2
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE name='synthetic_ledger_idx'"
    ).fetchone()
    connection.execute(
        "INSERT INTO ai_calls(call_uuid,model,provider) "
        "VALUES ('new-openai','gpt-6.1-sol','openai')"
    )
    assert (
        connection.execute("SELECT id FROM ai_calls WHERE call_uuid='new-openai'").fetchone()[0]
        == 100
    )
    assert connection.execute("SELECT value FROM counter").fetchone()[0] == 3
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_v40_ledger_migration_failure_rolls_back_even_after_table_replacement(monkeypatch):
    import src.migrations as migrations

    connection = _old_ledger()
    before = connection.execute("SELECT * FROM ai_calls").fetchall()
    monkeypatch.setattr(
        migrations,
        "_validate_version_sixteen_schema",
        lambda _: (_ for _ in ()).throw(MigrationError("synthetic injection")),
    )
    with pytest.raises(MigrationError, match="synthetic injection"):
        _apply_version_forty(connection)
    assert connection.execute("SELECT * FROM ai_calls").fetchall() == before
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE name='synthetic_ledger_idx'"
    ).fetchone()
    assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] is None
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO ai_calls(call_uuid,model,provider) "
            "VALUES ('blocked','gpt-6.1-sol','openai')"
        )
