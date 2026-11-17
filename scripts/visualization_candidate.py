"""Launch the phase 1/2 candidate with private, credential-free host storage.

This wrapper delegates process ownership and lifecycle to the existing service
manager. It never copies production settings or initializes a learning library.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MARKER = ".nectivon-visualization-candidate.json"


def prepare_environment(root: Path, port: int, source: Path = SOURCE_ROOT) -> dict[str, str]:
    """Validate a dedicated external root before creating any writable locations."""
    if not root.is_absolute() or str(root).startswith(("\\\\", "//")):
        raise ValueError("候选目录必须为本机绝对路径。")
    for parent in (root, *root.parents):
        if parent.is_symlink() or (
            parent.exists() and getattr(parent.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("候选目录不能通过符号链接或目录联接指向其他位置。")
    root, source = root.resolve(), source.resolve()
    if root == Path(root.anchor) or root == source or root.is_relative_to(source):
        raise ValueError("候选数据必须放在源码目录之外。")
    if source.is_relative_to(root):
        raise ValueError("候选目录不能包含源码。")
    if not (source / ".git").is_file():
        raise ValueError("此入口仅允许在隔离 Git 工作区运行。")
    if (source / ".env").exists():
        raise ValueError("隔离工作区存在 .env；请保留现场并核对，不加载可能的正式配置。")
    if not 1024 <= port <= 65535 or port in {8501, 8511, 8514}:
        raise ValueError("请选择独立候选端口；不能使用正式或已知历史候选端口。")
    marker = root / MARKER
    identity = {"version": 1, "source": str(source), "port": port}
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("此目录不是本候选的专用目录，未写入任何数据。")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("EKB_", "DASHSCOPE_", "OPENAI_", "ANTHROPIC_", "NECTIVON_"))
        and not any(part in key.upper() for part in ("API_KEY", "ACCESS_TOKEN", "SECRET"))
    }
    for variable, folder in {
        "LOCALAPPDATA": "host/local", "APPDATA": "host/roaming",
        "USERPROFILE": "host/home", "TEMP": "host/tmp", "TMP": "host/tmp",
        "XDG_DATA_HOME": "host/xdg",
    }.items():
        location = root / folder
        location.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(location)
    environment.update({
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(source),
        "NECTIVON_PYTHON": sys.executable,
        "EKB_STAGING_INSTANCE": "1",
        "EKB_STAGING_ROOT": str(root / "platform"),
        "EKB_STAGING_PORT": str(port),
        "EKB_CONFIG_DIR": str(root / "platform/config"),
        "EKB_AI_MODE": "manual",
        "EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED": "yes",
        "EKB_STUDY_LOOP_ROOT": str(root / "study"),
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    })
    return environment


def main() -> int:
    """Start, inspect or stop only this explicitly isolated candidate instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8515)
    args = parser.parse_args()
    try:
        environment = prepare_environment(args.state_root, args.port)
    except (ValueError, OSError) as exc:
        print(f"候选入口未执行：{exc}", file=sys.stderr)
        return 2
    command = [sys.executable, str(SOURCE_ROOT / "scripts/service_manager.py"), args.action]
    if args.action == "start":
        command.append("--no-browser")
    command += ["--staging", "--staging-root", environment["EKB_STAGING_ROOT"],
                "--staging-port", str(args.port)]
    return subprocess.run(command, cwd=SOURCE_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
