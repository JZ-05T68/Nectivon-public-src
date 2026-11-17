"""Explicit, isolated phase-three import and server entry. Never target the formal library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
MARKER = ".nectivon-phase3-test.json"


def environment(root: Path, port: int) -> dict[str, str]:
    """Require an empty or marker-owned root, separated from all source and platform data."""
    from src.storage_boundaries import local_directory

    root = local_directory(root)
    if (
        root == Path(root.anchor)
        or root.is_relative_to(SOURCE_ROOT)
        or SOURCE_ROOT.is_relative_to(root)
    ):
        raise ValueError("测试目录必须与源码分离。")
    for parent in (root, *root.parents):
        if (parent / ".git").exists() or (parent / ".nectivon-study-loop").exists():
            raise ValueError("测试实例不能位于源码或已有学习库内。")
    if port in {8501, 8511, 8514, 8515, 8516} or not 1024 <= port <= 65535:
        raise ValueError("请选择本轮专用端口，例如 8517。")
    marker = root / MARKER
    identity = {"version": 1, "purpose": "phase3-test", "port": port}
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("目标非空且不是本轮测试实例，拒绝操作。")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps(identity), encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("EKB_", "DASHSCOPE_", "OPENAI_", "NECTIVON_"))
        and not any(token in k.upper() for token in ("API_KEY", "ACCESS_TOKEN", "SECRET"))
    }
    for variable, folder in {
        "LOCALAPPDATA": "local",
        "APPDATA": "roaming",
        "TEMP": "tmp",
        "TMP": "tmp",
        "XDG_DATA_HOME": "xdg",
    }.items():
        location = root / "host" / folder
        location.mkdir(parents=True, exist_ok=True)
        env[variable] = str(location)
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(SOURCE_ROOT),
            "NECTIVON_PYTHON": sys.executable,
            "EKB_STAGING_INSTANCE": "1",
            "EKB_STAGING_ROOT": str(root / "platform"),
            "EKB_STAGING_PORT": str(port),
            "EKB_CONFIG_DIR": str(root / "platform/config"),
            "EKB_AI_MODE": "manual",
            "EKB_AI_API_KEY": "",
            "EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED": "yes",
            "EKB_STUDY_LOOP_ROOT": str(root / "study"),
            "EKB_STUDY_AUTO_OPEN": "yes",
            "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    return env


def import_files(root: Path, files: list[Path], subject: str, exam: str) -> dict:
    """Import originals through the same UI service; record timings per actual file."""
    from src.study_loop.chat_store import ChatStore
    from src.study_loop.components import import_source
    from src.study_loop.storage import StudyStore

    study_root = root / "study"
    study = StudyStore(study_root, create=not study_root.exists())
    rows = []
    for path in files:
        start = time.monotonic()
        row = {
            "filename": path.name,
            "path": str(path),
            "type": path.suffix.lower(),
            "method": "StudyStore.receive → Import Core/Office 渲染 → 公式文字 → 标题草稿",
        }
        try:
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            row.update(bytes=len(data), sha256=sha)
            kind = path.suffix.lstrip(".").lower()
            existing = {s.id for s in study.sources()}
            source = study.receive(path.name, data, kind)
            result = import_source(study, source.id)
            pages = study.pages(source.id)
            drafts = [d for d in study.drafts() if d.source_id == source.id]
            row.update(
                status="duplicate" if source.id in existing else "success",
                source_id=source.id,
                draft_count=result["draft_count"],
                rendered_pages=sum(bool(p.image) for p in pages),
                segments=len(pages),
                observed_answers=sum(bool(d.question.observed_answer) for d in drafts),
                observed_analyses=sum(bool(d.question.observed_analysis) for d in drafts),
                followup_mentions=sum(
                    p.text.count("追问")
                    for p in pages
                    if p.metadata.get("format") == "word_math_v1"
                ),
                live_conversations_imported=0,
                original_verified=study.read_source(source) == data,
                original_unchanged=hashlib.sha256(path.read_bytes()).hexdigest() == sha,
            )
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__)
        row["seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    chats = ChatStore(study)
    prior = next((c for c in chats.courses() if c["subject"] == subject), {})
    selected = list(
        dict.fromkeys(
            [
                *prior.get("source_ids", []),
                *[r["source_id"] for r in rows if r["status"] in {"success", "duplicate"}],
            ]
        )
    )
    chats.set_course(subject, selected, exam or prior.get("exam_notes", ""))
    report = {
        "files": rows,
        "subject": subject,
        "study_root": str(study.root),
        "timing_scope": "每文件：读取、SHA-256、接收、Office 渲染/校验、公式提取及草稿写入",
        "human_acceptance": "pending",
        "sessions": len(chats.conversations("course")),
    }
    output = root / "evidence"
    output.mkdir(exist_ok=True)
    target = output / ("import-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("导入报告：" + str(target))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["import", "start", "status", "stop"])
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--port", default=8517, type=int)
    parser.add_argument("--file", action="append", type=Path, default=[])
    parser.add_argument("--subject", default="")
    parser.add_argument("--exam-notes", default="")
    parser.add_argument(
        "--use-configured-qwen",
        action="store_true",
        help="仅本次进程内读取已有 Qwen 凭据；不复制凭据文件、不改正式设置",
    )
    args = parser.parse_args()
    try:
        env = environment(args.state_root, args.port)
        if args.use_configured_qwen and args.action == "start":
            from src.ai.credential_store import build_default_credential_store
            from src.config import Settings, settings_env_path

            credential = build_default_credential_store().get("qwen")
            secret = (
                credential.reveal()
                if credential
                else Settings(_env_file=settings_env_path()).ai_api_key.get_secret_value()
            )
            if not secret:
                raise ValueError("没有已配置的 Qwen 密钥；可不带该选项离线启动后在页面配置。")
            env.update(EKB_AI_MODE="api", EKB_AI_API_KEY=secret)
        if args.action == "import":
            if not args.file or not args.subject.strip():
                raise ValueError("导入必须显式提供 --file 和 --subject。")
            os.environ.clear()
            os.environ.update(env)
            result = import_files(
                args.state_root.resolve(), args.file, args.subject, args.exam_notes
            )
            return int(any(r["status"] == "failed" for r in result["files"]))
        command = [
            sys.executable,
            str(SOURCE_ROOT / "scripts/service_manager.py"),
            args.action,
            "--staging",
            "--staging-root",
            env["EKB_STAGING_ROOT"],
            "--staging-port",
            str(args.port),
        ]
        if args.action == "start":
            command.append("--no-browser")
        return subprocess.run(command, cwd=SOURCE_ROOT, env=env, check=False).returncode
    except (ValueError, OSError) as exc:
        print("操作未执行：" + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
