"""Exercise the real background spawn through Windows' venv redirector."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from scripts.service_manager import spawn_background_process

_REPORT_PROCESS = """
import json, os, sys
report = {
    "pid": os.getpid(), "cwd": os.getcwd(), "stdin_eof": sys.stdin.read() == "",
    "stdout_is_terminal": sys.stdout.isatty(),
    "marker": os.environ.get("NECTIVON_BACKGROUND_TEST_MARKER"),
}
if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleWindow.restype = wintypes.HWND
    window = kernel.GetConsoleWindow()
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    report.update(console_window=window, visible_console=bool(user.IsWindowVisible(window)))
print(json.dumps(report))
"""


@pytest.mark.parametrize("interpreter", ["venv", "base"])
def test_background_process_has_no_console_window(
    tmp_path: Path, interpreter: str
) -> None:
    """A detached venv must not create an inner console that users can close."""

    python = sys.executable if interpreter == "venv" else sys._base_executable
    output_path = tmp_path / "background.json"
    environment = {
        **os.environ,
        "NECTIVON_BACKGROUND_TEST_MARKER": "NON_REAL_TEST_DATA",
    }
    with output_path.open("w", encoding="utf-8") as output:
        process = spawn_background_process(
            [python, "-c", _REPORT_PROCESS],
            cwd=tmp_path,
            environment=environment,
            output=output,
        )
        assert process.wait(timeout=15) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert Path(report["cwd"]) == tmp_path
    assert report["stdin_eof"] is True
    assert report["stdout_is_terminal"] is False
    assert report["marker"] == "NON_REAL_TEST_DATA"
    if os.name == "nt":
        assert report["visible_console"] is False, report
        assert report["console_window"] in (None, 0), report
