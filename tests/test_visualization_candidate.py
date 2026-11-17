"""NON_REAL_TEST_DATA: candidate storage must never adopt existing directories."""

from __future__ import annotations

import json

import pytest

from scripts.visualization_candidate import MARKER, prepare_environment


def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").write_text("gitdir: TEST_WORKTREE", encoding="utf-8")
    return source


def test_candidate_uses_isolated_host_and_keeps_credentials_out(tmp_path, monkeypatch):
    source = workspace(tmp_path)
    root = tmp_path / "candidate"
    monkeypatch.setenv("EKB_STUDY_LOOP_ROOT", "TEST_FORBIDDEN_FORMAL_ROOT")
    monkeypatch.setenv("OPENAI_API_KEY", "NON_REAL_TEST_DATA")
    monkeypatch.setenv("EKB_AI_API_KEY", "NON_REAL_TEST_DATA")
    env = prepare_environment(root, 8515, source)
    assert "OPENAI_API_KEY" not in env and "EKB_AI_API_KEY" not in env
    assert env["EKB_STUDY_LOOP_ROOT"] == str(root / "study")
    assert env["LOCALAPPDATA"] == str(root / "host/local")
    assert env["EKB_STAGING_ROOT"] == str(root / "platform")
    assert not (root / "study").exists()  # Explicit UI action creates the library.
    assert prepare_environment(root, 8515, source) == env


def test_candidate_refuses_existing_non_candidate_and_foreign_marker(tmp_path):
    source = workspace(tmp_path)
    root = tmp_path / "existing"
    root.mkdir()
    data = root / "TEST_EXISTING"
    data.write_bytes(b"NON_REAL_TEST_DATA")
    with pytest.raises(ValueError, match="专用目录"):
        prepare_environment(root, 8515, source)
    assert data.read_bytes() == b"NON_REAL_TEST_DATA"
    assert not (root / MARKER).exists()
    (root / MARKER).write_text(json.dumps({"source": "OTHER"}), encoding="utf-8")
    with pytest.raises(ValueError, match="专用目录"):
        prepare_environment(root, 8515, source)


@pytest.mark.parametrize("port", [8501, 8511, 8514, 80, 65536])
def test_reserved_or_invalid_ports_rejected_before_writes(tmp_path, port):
    source = workspace(tmp_path)
    root = tmp_path / "candidate"
    with pytest.raises(ValueError, match="端口"):
        prepare_environment(root, port, source)
    assert not root.exists()


def test_candidate_refuses_source_storage_and_dotenv(tmp_path):
    source = workspace(tmp_path)
    with pytest.raises(ValueError, match="源码目录之外"):
        prepare_environment(source / "data", 8515, source)
    (source / ".env").write_text("NON_REAL_TEST_DATA", encoding="utf-8")
    with pytest.raises(ValueError, match=".env"):
        prepare_environment(tmp_path / "candidate", 8515, source)
    assert not (tmp_path / "candidate").exists()
