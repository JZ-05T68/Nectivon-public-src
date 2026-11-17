"""Post-exam contract tests using only explicitly supplied, hash-pinned materials.

No example questions, generated PDFs, fake dialogue or audio fixtures are supplied.
The manifest and source materials must remain outside the repository.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest


def pytest_generate_tests(metafunc) -> None:
    """Load cases only after human authorization and the development window."""
    if "material_case" not in metafunc.fixturenames:
        return
    opted_in = metafunc.config.getoption("--study-postexam-confirmed")
    manifest = metafunc.config.getoption("--study-authorized-materials")
    if not opted_in and not manifest:
        metafunc.parametrize(
            "material_case",
            [pytest.param(None, marks=pytest.mark.skip(reason="未选择真实材料验收清单。"))],
        )
        return
    if not metafunc.config.getoption("--study-postexam-confirmed") or datetime.now(UTC) < datetime(
        2026, 11, 11, 12, 10, tzinfo=UTC
    ):
        raise pytest.UsageError("考前不收集或执行本组测试；考后须由用户明确确认。")
    selected = metafunc.config.getoption("--study-authorized-materials")
    if not selected:
        raise pytest.UsageError("必须指定用户授权的材料清单；本组不提供自编材料。")
    manifest = Path(selected).resolve()
    if any((parent / ".git").exists() for parent in (manifest.parent, *manifest.parents)):
        raise pytest.UsageError("材料清单必须放在 Git 仓库之外。")
    raw = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if raw.get("authorized_by_user") is not True or not raw.get("cases"):
        raise pytest.UsageError("清单必须明确用户授权并包含材料条目。")
    cases = []
    for case in raw["cases"]:
        path = Path(case["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if any((parent / ".git").exists() for parent in (path.parent, *path.parents)):
            raise pytest.UsageError("用户材料必须位于 Git 仓库之外。")
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise pytest.UsageError("授权材料指纹不匹配，停止测试。")
        cases.append({**case, "path": path})
    metafunc.parametrize("material_case", cases, ids=[f"authorized-{i}" for i in range(len(cases))])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """An imported link or attachment must never cause network access."""

    def deny(*args, **kwargs):
        raise AssertionError("离线验收禁止网络访问")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


@pytest.fixture
def imported(material_case, tmp_path):
    from src.study_loop.components import import_source
    from src.study_loop.storage import StudyStore

    store = StudyStore(tmp_path / "study", create=True)
    source = store.receive(
        material_case["path"].name,
        material_case["path"].read_bytes(),
        material_case["kind"],
        transcript=material_case.get("transcript", False),
    )
    import_source(store, source.id)
    return store, source, material_case
