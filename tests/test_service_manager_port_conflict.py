"""R5 closure: explicit, non-silent port-conflict handling on start.

Covers the 2026-09-27 R4 finding: a bash-resident Streamlit holding the
staging port made ``start`` ambiguous - the user could believe a new
instance was started while the browser silently reached an old process.
The manager must now distinguish:

A) a healthy Streamlit already on the port -> report "already running",
   open the existing instance, warn the code version cannot be verified;
B) a foreign/unknown listener -> precise refusal with the owner pid,
   an explicit "will not kill" statement, exit code 2;

and must never spawn a second instance while the port stays occupied.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.service_manager as manager


def make_settings(tmp_path: Path, port: int) -> SimpleNamespace:
    """Return the small settings surface used by lifecycle tests."""

    runtime_dir = tmp_path / "runtime"
    logs_dir = tmp_path / "logs"

    def ensure_directories() -> None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        port=port,
        runtime_dir=runtime_dir,
        pid_path=runtime_dir / "engineering-kb.pid.json",
        logs_dir=logs_dir,
        ensure_directories=ensure_directories,
    )


def _port_occupied(monkeypatch, settings: SimpleNamespace) -> None:
    monkeypatch.setattr(manager, "get_settings", lambda: settings)
    monkeypatch.setattr(
        manager,
        "detect_state",
        lambda **kwargs: manager.ServiceState("port_occupied", "端口被占用"),
    )


def _forbid_spawn(monkeypatch) -> None:
    def no_popen(*args, **kwargs):
        raise AssertionError("start must not spawn while the port is occupied")

    monkeypatch.setattr(manager.subprocess, "Popen", no_popen)


# ------------------------------------------------------------------- case A
def test_start_reports_existing_streamlit_without_respawning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = make_settings(tmp_path, 8591)
    _port_occupied(monkeypatch, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(
        manager,
        "_describe_port_owner",
        lambda port: (4321, r"D:\somewhere\python.exe"),
    )
    _forbid_spawn(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(
        manager.webbrowser, "open", lambda url: opened.append(url) or True
    )

    assert manager.start_service(open_browser=True) == 0

    out = capsys.readouterr().out
    assert "已在运行" in out
    assert "未重复启动" in out
    assert "4321" in out
    assert "可能不是最新代码" in out
    assert opened == ["http://127.0.0.1:8591"]


def test_start_existing_streamlit_without_browser_still_reports(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = make_settings(tmp_path, 8591)
    _port_occupied(monkeypatch, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_describe_port_owner", lambda port: (None, ""))
    _forbid_spawn(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr(
        manager.webbrowser, "open", lambda url: opened.append(url) or True
    )

    assert manager.start_service(open_browser=False) == 0

    out = capsys.readouterr().out
    assert "已在运行" in out
    assert "未能识别" in out
    assert opened == []


# ------------------------------------------------------------------- case B
def test_start_refuses_unknown_listener_without_killing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = make_settings(tmp_path, 8592)
    _port_occupied(monkeypatch, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: False)
    monkeypatch.setattr(
        manager,
        "_describe_port_owner",
        lambda port: (4322, r"C:\Tools\other.exe"),
    )
    _forbid_spawn(monkeypatch)

    assert manager.start_service(open_browser=False) == 2

    out = capsys.readouterr().out
    assert "已被其他进程占用" in out
    assert "未启动" in out
    assert "4322" in out
    assert "不会自动结束" in out


def test_real_dummy_listener_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    """Integration: a real listening socket yields exit 2 and no spawn."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        settings = make_settings(tmp_path, port)
        _port_occupied(monkeypatch, settings)
        monkeypatch.setattr(manager, "is_healthy", lambda p, timeout=1.0: False)
        monkeypatch.setattr(manager, "_describe_port_owner", lambda p: (None, ""))
        _forbid_spawn(monkeypatch)

        assert manager.start_service(open_browser=False) == 2

    assert "已被其他进程占用" in capsys.readouterr().out


# ------------------------------------------------------- owner identification
def test_describe_port_owner_uses_native_listener_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        manager, "listener_pids", lambda port: {8593: [4323], 8594: [4324]}.get(port, [])
    )
    monkeypatch.setattr(
        manager, "process_executable", lambda pid: Path(r"D:\x\python.exe")
    )

    assert manager._describe_port_owner(8593) == (4323, str(Path(r"D:\x\python.exe")))
    assert manager._describe_port_owner(8594) == (4324, str(Path(r"D:\x\python.exe")))
    assert manager._describe_port_owner(9999) == (None, "")


def test_describe_port_owner_survives_native_query_failure(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("Windows listener query unavailable")

    monkeypatch.setattr(manager, "listener_pids", boom)
    assert manager._describe_port_owner(8595) == (None, "")


# ------------------------------------------------------------------ status text
def test_unmanaged_healthy_streamlit_status_is_explained(
    tmp_path: Path, monkeypatch
) -> None:
    settings = make_settings(tmp_path, 8597)
    monkeypatch.setattr(manager, "get_settings", lambda: settings)
    monkeypatch.setattr(manager, "is_port_open", lambda port: True)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)

    state = manager.detect_state()

    assert state.code == "port_occupied"
    assert "无本实例 PID 记录" in state.detail
    assert "Streamlit" in state.detail


# ------------------------------------------------- spawned-tree ownership gate
def _spawnable(monkeypatch, tmp_path: Path, settings: SimpleNamespace) -> None:
    monkeypatch.setattr(manager, "get_settings", lambda: settings)
    monkeypatch.setattr(
        manager,
        "detect_state",
        lambda **kwargs: manager.ServiceState("stopped", "服务未运行"),
    )
    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"")
    monkeypatch.setattr(manager, "expected_python", lambda **kwargs: fake_python)
    monkeypatch.setattr(manager, "write_pid_record", lambda pid: None)
    monkeypatch.setattr(manager, "_rotate_console_log", lambda path: None)

    class FakeProcess:
        pid = 4321

        def poll(self) -> None:
            return None

    monkeypatch.setattr(manager.subprocess, "Popen", lambda *a, **k: FakeProcess())


def test_start_fails_when_ghost_answers_health(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A ghost holding the port must never make start report success (R5 HIGH)."""

    settings = make_settings(tmp_path, 8598)
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(manager, "START_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: [9999])
    monkeypatch.setattr(manager, "_parent_pid", lambda pid: None)

    assert manager.start_service(open_browser=False) == 8

    out = capsys.readouterr().out
    assert "其他进程" in out
    assert "9999" in out
    assert "启动成功" not in out


@pytest.mark.parametrize("owners", [[], [9999]])
def test_unverified_start_cleans_only_its_fresh_child(
    tmp_path: Path, monkeypatch, capsys, owners: list[int]
) -> None:
    """Unknown/foreign responders must not leave our new child orphaned."""
    settings = make_settings(tmp_path, 8598)
    write_record = manager.write_pid_record
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(manager, "write_pid_record", write_record)
    monkeypatch.setattr(manager, "START_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: owners)
    monkeypatch.setattr(manager, "_parent_pid", lambda pid: None)
    monkeypatch.setattr(manager, "record_matches_process", lambda record: True)
    stopped = []
    monkeypatch.setattr(
        manager, "_terminate_process_tree", lambda pid: stopped.append(pid) or True
    )

    assert manager.start_service(open_browser=False) == 8
    assert stopped == [4321]
    assert 9999 not in stopped
    assert not settings.pid_path.exists()
    assert "已停止本次启动" in capsys.readouterr().out


@pytest.mark.skipif(manager.os.name != "nt", reason="Windows taskkill only")
def test_process_tree_timeout_is_safe_failure(monkeypatch, caplog) -> None:
    """All lifecycle callers receive a failure without losing their PID record."""
    commands = []

    def timeout(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == manager.STOP_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(manager.subprocess, "run", timeout)
    monkeypatch.setattr(
        manager.os, "kill", lambda *args: pytest.fail("must not broaden timed-out cleanup")
    )
    assert manager._terminate_process_tree(4321) is False
    assert commands == [["taskkill", "/PID", "4321", "/T", "/F"]]
    assert "保留 PID 记录" in caplog.text


@pytest.mark.parametrize(
    "failure", ["record_changed", "identity_unknown", "stop_failed", "timeout"]
)
def test_unverified_start_preserves_record_when_cleanup_is_not_safe(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    """A cleanup failure must retain recovery information and never widen its target."""
    settings = make_settings(tmp_path, 8598)
    write_record = manager.write_pid_record
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(
        manager, "write_pid_record",
        lambda pid: write_record(9999 if failure == "record_changed" else pid),
    )
    monkeypatch.setattr(manager, "START_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: [])
    monkeypatch.setattr(
        manager, "record_matches_process", lambda record: failure != "identity_unknown"
    )
    stopped = []

    def stop(pid: int) -> bool:
        stopped.append(pid)
        if failure == "timeout":
            raise subprocess.TimeoutExpired("taskkill", manager.STOP_TIMEOUT_SECONDS)
        return False

    monkeypatch.setattr(manager, "_terminate_process_tree", stop)
    assert manager.start_service(open_browser=False) == 8
    record = manager.read_pid_record()
    assert record is not None
    assert record["pid"] == (9999 if failure == "record_changed" else 4321)
    assert stopped == ([] if failure in {"record_changed", "identity_unknown"} else [4321])


@pytest.mark.parametrize("corrupted", [b'{"pid":', b'\xff\xfe'])
def test_unverified_cleanup_preserves_corrupt_record_without_killing(
    tmp_path: Path, monkeypatch, corrupted: bytes
) -> None:
    """Malformed JSON or encoding cannot justify discarding evidence or stopping a PID."""
    settings = make_settings(tmp_path, 8598)
    settings.ensure_directories()
    settings.pid_path.write_bytes(corrupted)
    monkeypatch.setattr(manager, "get_settings", lambda: settings)
    monkeypatch.setattr(
        manager, "_terminate_process_tree", lambda pid: pytest.fail("uncertain PID must survive")
    )

    process = SimpleNamespace(pid=4321, poll=lambda: None)
    assert manager._cleanup_unverified_start(settings, process) is False
    assert settings.pid_path.read_bytes() == corrupted


def test_unverified_cleanup_preserves_record_corrupted_during_stop(
    tmp_path: Path, monkeypatch
) -> None:
    """The second read must also preserve a record changed while awaiting our child's exit."""
    settings = make_settings(tmp_path, 8598)
    settings.ensure_directories()
    monkeypatch.setattr(manager, "get_settings", lambda: settings)
    manager.write_pid_record(4321)
    monkeypatch.setattr(manager, "record_matches_process", lambda record: True)
    stopped = []
    corrupted = b'{"pid":'

    def stop(pid: int) -> bool:
        stopped.append(pid)
        settings.pid_path.write_bytes(corrupted)
        return True

    monkeypatch.setattr(manager, "_terminate_process_tree", stop)
    process = SimpleNamespace(pid=4321, poll=lambda: None)
    assert manager._cleanup_unverified_start(settings, process) is True
    assert stopped == [4321]
    assert settings.pid_path.read_bytes() == corrupted


def test_start_succeeds_when_owner_is_spawned_pid(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = make_settings(tmp_path, 8598)
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: [4321])

    assert manager.start_service(open_browser=False) == 0
    assert "启动成功" in capsys.readouterr().out


def test_start_succeeds_when_owner_is_child_of_spawned(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = make_settings(tmp_path, 8598)
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: [5555])
    monkeypatch.setattr(manager, "_parent_pid", lambda pid: 4321 if pid == 5555 else None)

    assert manager.start_service(open_browser=False) == 0
    assert "启动成功" in capsys.readouterr().out


def test_start_refuses_when_one_of_dual_bind_owners_is_foreign(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A second foreign listener makes the responder ambiguous; refuse success."""

    settings = make_settings(tmp_path, 8598)
    _spawnable(monkeypatch, tmp_path, settings)
    monkeypatch.setattr(manager, "is_healthy", lambda port, timeout=1.0: True)
    monkeypatch.setattr(manager, "_port_owner_pids", lambda port: [9999, 5555])
    monkeypatch.setattr(manager, "_parent_pid", lambda pid: 4321 if pid == 5555 else None)

    monkeypatch.setattr(manager, "START_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(manager, "_cleanup_unverified_start", lambda *args: True)
    assert manager.start_service(open_browser=False) == 8
    assert "启动失败" in capsys.readouterr().out


def test_multiple_listeners_in_our_tree_are_allowed(monkeypatch) -> None:
    monkeypatch.setattr(manager, "_parent_pid", lambda pid: 4321 if pid == 5555 else None)
    assert manager._owns_serving_process(4321, [4321, 5555])


def test_pid_in_tree_walks_ancestors() -> None:
    chain = {5556: 5555, 5555: 4321}
    monkey_target = {}
    import scripts.service_manager as manager_module

    original = manager_module._parent_pid
    manager_module._parent_pid = lambda pid: chain.get(pid)
    try:
        assert manager_module._pid_in_tree(4321, 4321) is True
        assert manager_module._pid_in_tree(4321, 5555) is True
        assert manager_module._pid_in_tree(4321, 5556) is True
        assert manager_module._pid_in_tree(4321, 9999) is False
    finally:
        manager_module._parent_pid = original
    monkey_target.clear()
