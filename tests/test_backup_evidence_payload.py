"""Backup payload for linkage/reading evidence (v0.8.8 round-1 P1-2).

data/question-candidates/ (candidates, figures-index.json,
figures_page_*.json) and data/agent-readings/ are the image-text linkage
and AI reading evidence chain. They live outside SQLite, so format v2 adds
them to the backup payload with symmetric restore. Format v1 backups keep
validating and restoring, and a v1 restore PRESERVES the current evidence
directories instead of destroying them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.backup_service import BackupService, validate_backup
from src.database import Database
from src.models import PageStatus


def _service(root: Path, *, version: str = "0.0.8") -> BackupService:
    data = root / "data"
    return BackupService(
        app_version=version,
        data_dir=data,
        raw_dir=data / "raw",
        pages_dir=data / "pages",
        markdown_dir=data / "markdown",
        database_path=data / "database" / "knowledge.db",
        backups_dir=root / "backups",
    )


def _library(
    root: Path, *, marker: bytes, version: str = "0.0.8", with_evidence: bool = False
) -> BackupService:
    service = _service(root, version=version)
    for directory in (
        service.raw_dir,
        service.pages_dir,
        service.markdown_dir,
        service.database_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    pdf = service.raw_dir / "manual.pdf"
    pdf.write_bytes(marker)
    image = service.pages_dir / "1" / "page_0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png:" + marker)
    database = Database(service.database_path)
    document = database.create_document(
        title="控制手册",
        filename="manual.pdf",
        source_path=pdf,
        sha256=hashlib.sha256(marker).hexdigest(),
    )
    database.create_page(
        document_id=document.id,
        page_number=1,
        image_path=image,
        extracted_text="闭环控制原文",
        status=PageStatus.READY,
    )
    database.update_document_page_count(document.id, 1)
    if with_evidence:
        candidates = service.question_candidates_dir
        candidates.mkdir(parents=True, exist_ok=True)
        (candidates / "page_1.json").write_text(
            json.dumps({"format_version": 2, "page_id": 1, "candidates": []}),
            encoding="utf-8",
        )
        (candidates / "figures-index.json").write_text(
            json.dumps({"figures": {"图6-33": [{"page_id": 1}]}}), encoding="utf-8"
        )
        readings = service.agent_readings_dir / "pages"
        readings.mkdir(parents=True, exist_ok=True)
        (readings / "page_1.json").write_text(
            json.dumps({"page_id": 1, "marker": marker.decode()}), encoding="utf-8"
        )
    return service


def _downgrade_to_v1(backup: Path) -> None:
    """Rewrite a v2 backup into a genuine v1-era backup (pre-change format)."""

    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backup_format_version"] = 1
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record["kind"] not in ("question_candidates", "agent_readings")
    ]
    for key in ("question_candidates", "agent_readings"):
        manifest["directories"].pop(key, None)
        manifest["key_paths"].pop(key, None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(backup / "data" / "question-candidates", ignore_errors=True)
    shutil.rmtree(backup / "data" / "agent-readings", ignore_errors=True)


def test_backup_captures_evidence_directories_with_hashes(tmp_path: Path) -> None:
    service = _library(tmp_path / "source", marker=b"src", with_evidence=True)

    result = service.create_backup()
    validation = validate_backup(result.backup_path, expected_app_version="0.0.8")

    assert validation.valid, validation.errors
    assert result.manifest["backup_format_version"] == 2
    assert result.manifest["directories"]["question_candidates"] == {
        "files": 2,
        "bytes": (
            (service.question_candidates_dir / "page_1.json").stat().st_size
            + (service.question_candidates_dir / "figures-index.json").stat().st_size
        ),
    }
    assert result.manifest["directories"]["agent_readings"]["files"] == 1
    assert result.manifest["key_paths"]["question_candidates"] == (
        "data/question-candidates"
    )
    kinds = {record["kind"] for record in result.manifest["files"]}
    assert {"question_candidates", "agent_readings"} <= kinds
    assert (result.backup_path / "data" / "question-candidates" / "figures-index.json").is_file()
    assert (result.backup_path / "data" / "agent-readings" / "pages" / "page_1.json").is_file()


def test_restore_roundtrip_returns_evidence_files(tmp_path: Path) -> None:
    source = _library(tmp_path / "source", marker=b"src", with_evidence=True)
    backup = source.create_backup().backup_path
    target = _library(tmp_path / "target", marker=b"tgt")

    target.restore_backup(backup, service_is_running=lambda: False)

    restored_index = target.question_candidates_dir / "figures-index.json"
    assert restored_index.is_file()
    assert json.loads(restored_index.read_text(encoding="utf-8"))["figures"]["图6-33"] == [
        {"page_id": 1}
    ]
    restored_reading = target.agent_readings_dir / "pages" / "page_1.json"
    assert json.loads(restored_reading.read_text(encoding="utf-8"))["marker"] == "src"


def test_v1_backup_still_validates_restores_and_preserves_current_evidence(
    tmp_path: Path,
) -> None:
    source = _library(tmp_path / "source", marker=b"src", with_evidence=True)
    backup = source.create_backup().backup_path
    _downgrade_to_v1(backup)
    validation = validate_backup(backup, expected_app_version="0.0.8")
    assert validation.valid, validation.errors

    target = _library(tmp_path / "target", marker=b"tgt", with_evidence=True)
    (target.question_candidates_dir / "figures-index.json").write_text(
        json.dumps({"figures": {"图9-9": [{"page_id": 1}]}}), encoding="utf-8"
    )
    target.restore_backup(backup, service_is_running=lambda: False)

    # The v1 backup never contained evidence: the CURRENT evidence chain
    # survives the restore instead of being destroyed with the old data dir.
    kept_index = target.question_candidates_dir / "figures-index.json"
    assert kept_index.is_file()
    assert "图9-9" in json.loads(kept_index.read_text(encoding="utf-8"))["figures"]
    assert (target.agent_readings_dir / "pages" / "page_1.json").is_file()
    # Core payload still restored symmetrically.
    assert (target.raw_dir / "manual.pdf").read_bytes() == b"src"


def test_v1_manifest_with_evidence_keys_is_rejected(tmp_path: Path) -> None:
    source = _library(tmp_path / "source", marker=b"src", with_evidence=True)
    backup = source.create_backup().backup_path
    _downgrade_to_v1(backup)
    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["directories"]["agent_readings"] = {"files": 0, "bytes": 0}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_backup(backup, expected_app_version="0.0.8")
    assert not validation.valid
    assert any("白名单" in error for error in validation.errors)


def test_future_format_version_is_rejected(tmp_path: Path) -> None:
    service = _library(tmp_path / "source", marker=b"src")
    backup = service.create_backup().backup_path
    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backup_format_version"] = 3
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    validation = validate_backup(backup, expected_app_version="0.0.8")
    assert not validation.valid
    assert any("格式版本" in error for error in validation.errors)
