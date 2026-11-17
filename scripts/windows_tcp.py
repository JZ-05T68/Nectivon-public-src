"""Read Windows listener ownership without localized command-line state labels."""

from __future__ import annotations

import ctypes
import socket

_DWORD = ctypes.c_uint32
_OWNER_PID_LISTENER = 3
_INSUFFICIENT_BUFFER = 122
_MAX_TABLE_BYTES = 16 * 1024 * 1024


class _IPv4Row(ctypes.Structure):
    _fields_ = [(name, _DWORD) for name in (
        "state", "local_addr", "local_port", "remote_addr", "remote_port", "pid"
    )]


class _IPv6Row(ctypes.Structure):
    _fields_ = [
        ("local_addr", ctypes.c_ubyte * 16),
        ("local_scope", _DWORD),
        ("local_port", _DWORD),
        ("remote_addr", ctypes.c_ubyte * 16),
        ("remote_scope", _DWORD),
        ("remote_port", _DWORD),
        ("state", _DWORD),
        ("pid", _DWORD),
    ]


def _listener_table(family: int) -> bytes:
    """Ask IP Helper for listeners only; bound allocation and table-growth retries.

    The table class, not the rendered state field, selects listening sockets.
    Some Windows environments expose an empty/inconsistent state through netstat
    and CIM even though the listener-only API correctly reports socket ownership.
    https://learn.microsoft.com/windows/win32/api/iphlpapi/nf-iphlpapi-getextendedtcptable
    """
    api = ctypes.WinDLL("iphlpapi").GetExtendedTcpTable
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DWORD), ctypes.c_int,
                    _DWORD, ctypes.c_int, _DWORD]
    api.restype = _DWORD
    size = _DWORD(0)
    buffer = None
    for _ in range(4):
        result = api(buffer, ctypes.byref(size), False, family, _OWNER_PID_LISTENER, 0)
        if result == 0:
            if buffer is None or not 4 <= size.value <= len(buffer):
                raise OSError("Windows listener table has an invalid size")
            return buffer.raw[:size.value]
        if result != _INSUFFICIENT_BUFFER:
            raise OSError(result, "Windows listener ownership query failed")
        if not 4 <= size.value <= _MAX_TABLE_BYTES:
            raise OSError("Windows listener table exceeds the bounded buffer size")
        buffer = ctypes.create_string_buffer(size.value)
    raise OSError("Windows listener table kept changing during ownership query")


def _table_pids(data: bytes, family: int, port: int) -> set[int]:
    """Validate one listener table and return every owner of the exact local port."""
    row_type = _IPv4Row if family == socket.AF_INET else _IPv6Row
    if len(data) < 4:
        raise OSError("Truncated Windows listener table header")
    count = _DWORD.from_buffer_copy(data).value
    row_size = ctypes.sizeof(row_type)
    if count > (len(data) - 4) // row_size:
        raise OSError("Truncated Windows listener table rows")
    pids = set()
    for index in range(count):
        row = row_type.from_buffer_copy(data, 4 + index * row_size)
        if socket.ntohs(row.local_port & 0xFFFF) == port:
            if row.pid == 0:
                raise OSError("Windows listener owner could not be identified")
            pids.add(int(row.pid))
    return pids


def listener_pids(port: int) -> list[int]:
    """Return the complete IPv4/IPv6 listener owner set, or fail closed."""
    pids: set[int] = set()
    for family in (socket.AF_INET, socket.AF_INET6):
        pids.update(_table_pids(_listener_table(family), family, port))
    return sorted(pids)
