"""Native listener ownership: binary contracts, bounded failure and real sockets."""

from __future__ import annotations

import ctypes
import os
import socket
import struct
from types import SimpleNamespace

import pytest

from scripts import windows_tcp as tcp


def table(family: int, *entries: tuple[int, int]) -> bytes:
    """Build documented fixed-width listener rows (state intentionally unreliable)."""
    rows = []
    for port, pid in entries:
        row = (tcp._IPv4Row if family == socket.AF_INET else tcp._IPv6Row)()
        row.state = 1
        row.local_port = socket.htons(port)
        row.pid = pid
        rows.append(bytes(row))
    return struct.pack("<I", len(rows)) + b"".join(rows)


def test_exact_port_all_owners_and_both_address_families(monkeypatch) -> None:
    tables = {
        socket.AF_INET: table(socket.AF_INET, (8512, 10), (18512, 99), (8512, 11)),
        socket.AF_INET6: table(socket.AF_INET6, (8512, 11), (8512, 12)),
    }
    monkeypatch.setattr(tcp, "_listener_table", tables.__getitem__)
    assert tcp.listener_pids(8512) == [10, 11, 12]
    assert tcp.listener_pids(18512) == [99]
    assert tcp.listener_pids(8513) == []


@pytest.mark.parametrize("payload", [b"", b"\1\0\0\0", struct.pack("<I", 0xFFFFFFFF)])
def test_incomplete_tables_fail_closed(payload) -> None:
    with pytest.raises(OSError, match="Truncated"):
        tcp._table_pids(payload, socket.AF_INET, 8501)


def test_unidentified_matching_owner_is_not_silently_ignored() -> None:
    with pytest.raises(OSError, match="could not be identified"):
        tcp._table_pids(table(socket.AF_INET, (8501, 10), (8501, 0)), socket.AF_INET, 8501)


def test_ipv6_failure_does_not_return_partial_ipv4_success(monkeypatch) -> None:
    def read(family):
        if family == socket.AF_INET6:
            raise OSError("unavailable")
        return table(family, (8501, 10))

    monkeypatch.setattr(tcp, "_listener_table", read)
    with pytest.raises(OSError):
        tcp.listener_pids(8501)


def install_fake_api(monkeypatch, implementation):
    """Replace only the system call; exercise the real sizing/retry/parser code."""
    class API:
        def __call__(self, buffer, size_ptr, ordered, family, kind, reserved):
            assert kind == 3  # TCP_TABLE_OWNER_PID_LISTENER; never ALL/CONNECTIONS.
            assert reserved == 0
            return implementation(buffer, size_ptr._obj, family)

    api = API()
    monkeypatch.setattr(tcp.ctypes, "WinDLL", lambda name: SimpleNamespace(GetExtendedTcpTable=api),
                        raising=False)


def test_query_handles_table_growth_with_bounded_retry(monkeypatch) -> None:
    data = table(socket.AF_INET, (8501, 10), (8501, 11))
    calls = []

    def query(buffer, size, family):
        calls.append(family)
        if len(calls) < 3:
            size.value = 4 if len(calls) == 1 else len(data)
            return 122
        ctypes.memmove(buffer, data, len(data))
        size.value = len(data)
        return 0

    install_fake_api(monkeypatch, query)
    assert tcp._listener_table(socket.AF_INET) == data
    assert len(calls) == 3


@pytest.mark.parametrize("result,size", [(5, 0), (122, 0), (122, 32 * 1024 * 1024), (0, 0)])
def test_query_errors_and_invalid_allocations_fail_closed(monkeypatch, result, size) -> None:
    def query(buffer, target, family):
        target.value = size
        return result

    install_fake_api(monkeypatch, query)
    with pytest.raises(OSError):
        tcp._listener_table(socket.AF_INET)


def test_continuously_growing_table_has_finite_attempts(monkeypatch) -> None:
    calls = []

    def query(buffer, size, family):
        calls.append(family)
        size.value += 4
        return 122

    install_fake_api(monkeypatch, query)
    with pytest.raises(OSError, match="kept changing"):
        tcp._listener_table(socket.AF_INET)
    assert len(calls) == 4


@pytest.mark.skipif(os.name != "nt", reason="Windows IP Helper integration")
@pytest.mark.parametrize(
    "family,address", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
)
def test_real_listener_found_but_established_client_excluded(family, address) -> None:
    # These sockets are private ephemeral fixtures, never a production port.
    with socket.socket(family) as listener, socket.socket(family) as client:
        listener.bind((address, 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert tcp.listener_pids(port) == [os.getpid()]
        client.connect((address, port))
        with listener.accept()[0]:
            assert tcp.listener_pids(port) == [os.getpid()]
            assert tcp.listener_pids(client.getsockname()[1]) == []
