"""Lifecycle fault-injection tests for DurableJsonlSink.

Pins the fd-release contract: every failure after os.open (fstat, regular
file / mode checks, directory fsync) must release the destination fd
exactly once while preserving the original exception; a failing close
during cleanup only warns.  close() must still release the fd when the
final fsync fails, and the fsync error must propagate untouched.  The
open flags/mode, line format, 0600 mode, append semantics and fsync
cadence stay unchanged.

POSIX-only: the constructor uses O_NOFOLLOW open flags; hosted Linux CI
is the acceptance gate, Windows skips are not acceptance.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import types
from pathlib import Path
from typing import NoReturn

import pytest

from sakuramoon.telemetry.metrics import DurableJsonlSink

LINUX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="DurableJsonlSink uses POSIX open flags (O_NOFOLLOW)",
)

_real_close = os.close
_real_fsync = os.fsync
_real_open = os.open
# Sentinel fd that is never passed to a delegating close, so no real
# descriptor can be closed by accident.
_SENTINEL_FD = 1_000_007


class _CloseRecorder:
    """Records os.close attempts; optionally delegates to the real close."""

    def __init__(self, delegate: bool, close_error: BaseException | None) -> None:
        self.delegate = delegate
        self.close_error = close_error
        self.attempted: list[int] = []

    def __call__(self, fd: int) -> None:
        self.attempted.append(fd)
        if self.delegate:
            _real_close(fd)
        if self.close_error is not None:
            raise self.close_error


def _open_recorder(calls: list[tuple[tuple[object, ...], int]]) -> object:
    def open_recorder(*args: object, **kwargs: object) -> int:
        fd = _real_open(*args, **kwargs)  # type: ignore[arg-type]
        calls.append((args, fd))
        return fd

    return open_recorder


def _sentinel_open(calls: list[tuple[tuple[object, ...], int]]) -> object:
    def open_stub(*args: object, **kwargs: object) -> int:
        calls.append((args, _SENTINEL_FD))
        return _SENTINEL_FD

    return open_stub


def _raising(captured: list[BaseException], error: BaseException) -> object:
    def raise_injection(*_args: object, **_kwargs: object) -> NoReturn:
        captured.append(error)
        raise error

    return raise_injection


def _good_regular_stat(mode: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(st_mode=stat.S_IFREG | mode)


def _stat_stub(st_mode: int) -> object:
    def stat_stub(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        return _good_regular_stat(st_mode)

    return stat_stub


def _no_fsync(*_args: object, **_kwargs: object) -> None:
    return None


@LINUX_ONLY
def test_init_fstat_failure_closes_fd_once_and_preserves_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    recorder = _CloseRecorder(delegate=False, close_error=None)
    fstat_error = OSError(5, "injected fstat failure")
    monkeypatch.setattr(os, "open", _sentinel_open([]))
    monkeypatch.setattr(os, "fstat", _raising([], fstat_error))
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(OSError) as excinfo:
        DurableJsonlSink(target, fsync_every_records=1)
    assert excinfo.value is fstat_error
    assert recorder.attempted == [_SENTINEL_FD]


@LINUX_ONLY
def test_init_failure_releases_real_destination_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls: list[tuple[tuple[object, ...], int]] = []
    original = OSError(5, "injected fstat failure")
    monkeypatch.setattr(os, "open", _open_recorder(open_calls))
    monkeypatch.setattr(os, "fstat", _raising([], original))
    with pytest.raises(OSError) as caught:
        DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    assert caught.value is original
    assert len(open_calls) == 1
    monkeypatch.undo()
    with pytest.raises(OSError) as closed:
        os.fstat(open_calls[0][1])
    assert closed.value.errno == 9  # EBADF: real kernel fd was released


@LINUX_ONLY
@pytest.mark.parametrize("cleanup_error", [OSError(9, "EBADF"), KeyboardInterrupt()])
def test_init_fstat_failure_with_failing_close_keeps_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cleanup_error: BaseException,
) -> None:
    target = tmp_path / "metrics.jsonl"
    recorder = _CloseRecorder(delegate=False, close_error=cleanup_error)
    fstat_error = OSError(5, "injected fstat failure")
    monkeypatch.setattr(os, "open", _sentinel_open([]))
    monkeypatch.setattr(os, "fstat", _raising([], fstat_error))
    monkeypatch.setattr(os, "close", recorder)
    with (
        caplog.at_level(logging.WARNING, logger="sakuramoon.telemetry.metrics"),
        pytest.raises(OSError) as excinfo,
    ):
        DurableJsonlSink(target, fsync_every_records=1)
    assert excinfo.value is fstat_error
    assert recorder.attempted == [_SENTINEL_FD]
    assert any("Failed to close" in record.message for record in caplog.records)


@LINUX_ONLY
@pytest.mark.parametrize(
    ("st_mode", "expected"),
    [
        (stat.S_IFIFO, ValueError),
        (stat.S_IFREG | 0o644, PermissionError),
    ],
    ids=["non-regular-file", "wrong-mode"],
)
def test_init_fd_checks_close_fd_and_raise_original_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    st_mode: int,
    expected: type[Exception],
) -> None:
    target = tmp_path / "metrics.jsonl"
    recorder = _CloseRecorder(delegate=False, close_error=None)
    monkeypatch.setattr(os, "open", _sentinel_open([]))
    monkeypatch.setattr(os, "fstat", _stat_stub(st_mode))
    monkeypatch.setattr(os, "fsync", _no_fsync)
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(expected):
        DurableJsonlSink(target, fsync_every_records=1)
    assert recorder.attempted == [_SENTINEL_FD]


@LINUX_ONLY
def test_init_keeps_open_flags_mode_and_directory_fsync_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    open_calls: list[tuple[tuple[object, ...], int]] = []
    recorder = _CloseRecorder(delegate=False, close_error=None)
    monkeypatch.setattr(os, "open", _sentinel_open(open_calls))
    monkeypatch.setattr(os, "fstat", _stat_stub(0o600))
    monkeypatch.setattr(os, "fsync", _no_fsync)
    monkeypatch.setattr(os, "close", recorder)
    sink = DurableJsonlSink(target, fsync_every_records=1)
    # Mirror guard_publish._optional_flags: getattr keeps the expectation
    # typed on platforms where a flag constant is absent from typeshed.
    expected_flags = 0
    for name in ("O_APPEND", "O_CREAT", "O_WRONLY", "O_CLOEXEC", "O_NOFOLLOW"):
        expected_flags |= getattr(os, name, 0)
    expected_dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    assert open_calls == [
        ((target, expected_flags, 0o600), _SENTINEL_FD),
        ((target.parent, expected_dir_flags), _SENTINEL_FD),
    ]
    assert sink._fd == _SENTINEL_FD  # pyright: ignore[reportPrivateUsage]
    # The single directory close of the success path is recorded.
    assert recorder.attempted == [_SENTINEL_FD]


@LINUX_ONLY
def test_init_directory_fsync_failure_closes_both_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    open_calls: list[tuple[tuple[object, ...], int]] = []
    # The directory close fails as well: the original fsync error must
    # still be the one that propagates, never the close error.
    recorder = _CloseRecorder(
        delegate=True, close_error=OSError(9, "injected close failure")
    )
    fsync_error = OSError(28, "injected directory fsync failure")
    monkeypatch.setattr(os, "open", _open_recorder(open_calls))
    monkeypatch.setattr(os, "fsync", _raising([], fsync_error))
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(OSError) as excinfo:
        DurableJsonlSink(target, fsync_every_records=1)
    assert excinfo.value is fsync_error
    opened_fds = [fd for _, fd in open_calls]
    assert len(opened_fds) == 2  # destination fd + directory fd
    assert sorted(recorder.attempted) == sorted(opened_fds)
    assert len(recorder.attempted) == 2


@LINUX_ONLY
def test_init_directory_close_failure_is_primary_and_closes_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    open_calls: list[tuple[tuple[object, ...], int]] = []
    close_error = OSError(9, "injected directory close failure")
    recorder = _CloseRecorder(delegate=True, close_error=close_error)
    monkeypatch.setattr(os, "open", _open_recorder(open_calls))
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(OSError) as excinfo:
        DurableJsonlSink(target, fsync_every_records=1)
    # Original interface: a failing directory close after a successful
    # directory fsync is the primary constructor error.
    assert excinfo.value is close_error
    opened_fds = [fd for _, fd in open_calls]
    assert len(opened_fds) == 2
    assert sorted(recorder.attempted) == sorted(opened_fds)
    assert len(recorder.attempted) == 2


@LINUX_ONLY
def test_close_fsync_failure_still_closes_fd_and_preserves_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    sink.write({"a": 1})
    fsync_error = OSError(5, "injected close fsync failure")
    recorder = _CloseRecorder(delegate=True, close_error=None)
    monkeypatch.setattr(os, "fsync", _raising([], fsync_error))
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(OSError) as excinfo:
        sink.close()
    assert excinfo.value is fsync_error
    assert recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]
    # The record itself is unaffected by the failing final fsync.
    assert (tmp_path / "metrics.jsonl").read_bytes() == b'{"a":1}\n'


@LINUX_ONLY
def test_close_failure_is_primary_error_when_fsync_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    sink.write({"a": 1})
    close_error = OSError(9, "injected close failure")
    recorder = _CloseRecorder(delegate=True, close_error=close_error)
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(OSError) as excinfo:
        sink.close()
    assert excinfo.value is close_error
    assert recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    # Marked closed so a retry cannot act on a possibly recycled fd.
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]


@LINUX_ONLY
def test_close_fsync_and_close_failure_warns_only_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    sink.write({"a": 1})
    fsync_error = OSError(5, "injected close fsync failure")
    close_error = OSError(9, "injected close failure")
    recorder = _CloseRecorder(delegate=True, close_error=close_error)
    monkeypatch.setattr(os, "fsync", _raising([], fsync_error))
    monkeypatch.setattr(os, "close", recorder)
    with (
        caplog.at_level(logging.WARNING, logger="sakuramoon.telemetry.metrics"),
        pytest.raises(OSError) as excinfo,
    ):
        sink.close()
    assert excinfo.value is fsync_error
    assert recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]
    assert any("Failed to close" in record.message for record in caplog.records)


@LINUX_ONLY
@pytest.mark.parametrize("fsync_error", [OSError(5, "fsync failure"), MemoryError("cleanup failure"), KeyboardInterrupt()])
def test_exit_preserves_original_exception_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fsync_error: BaseException,
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    sink.write({"a": 1})
    recorder = _CloseRecorder(delegate=True, close_error=None)
    monkeypatch.setattr(os, "fsync", _raising([], fsync_error))
    monkeypatch.setattr(os, "close", recorder)
    original = RuntimeError("original failure")
    with pytest.raises(RuntimeError) as caught, sink:
        raise original
    assert caught.value is original
    assert recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]


@LINUX_ONLY
def test_broken_logging_handler_never_masks_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sakuramoon.telemetry import metrics as metrics_module

    target = tmp_path / "metrics.jsonl"
    recorder = _CloseRecorder(delegate=False, close_error=OSError(9, "EBADF"))
    fstat_error = OSError(5, "injected fstat failure")

    def broken_warning(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("broken logging handler")

    monkeypatch.setattr(os, "open", _sentinel_open([]))
    monkeypatch.setattr(os, "fstat", _raising([], fstat_error))
    monkeypatch.setattr(os, "close", recorder)
    monkeypatch.setattr(metrics_module.logger, "warning", broken_warning)
    with pytest.raises(OSError) as excinfo:
        DurableJsonlSink(target, fsync_every_records=1)
    assert excinfo.value is fstat_error
    assert recorder.attempted == [_SENTINEL_FD]


@LINUX_ONLY
def test_close_base_exception_still_closes_and_marks_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    sink.write({"a": 1})
    memory_error = MemoryError("injected close unwind")
    recorder = _CloseRecorder(delegate=True, close_error=None)
    monkeypatch.setattr(os, "fsync", _raising([], memory_error))
    monkeypatch.setattr(os, "close", recorder)
    with pytest.raises(MemoryError) as excinfo:
        sink.close()
    assert excinfo.value is memory_error
    assert recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]
    # A second close() must be a no-op: no fsync, no close on the fd.
    second_recorder = _CloseRecorder(delegate=False, close_error=None)
    monkeypatch.setattr(os, "close", second_recorder)
    sink.close()
    assert second_recorder.attempted == []


@LINUX_ONLY
def test_write_fsync_failure_exit_closes_fd_and_propagates_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    fsync_error = OSError(5, "injected write fsync failure")
    close_recorder = _CloseRecorder(delegate=True, close_error=None)
    monkeypatch.setattr(os, "fsync", _raising([], fsync_error))
    monkeypatch.setattr(os, "close", close_recorder)
    with pytest.raises(OSError) as excinfo, sink:
        sink.write({"a": 1})
    # The write's own fsync error is the exception that propagates, and the
    # with-exit still releases the fd exactly once.
    assert excinfo.value is fsync_error
    assert close_recorder.attempted == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    assert sink._closed is True  # pyright: ignore[reportPrivateUsage]


@LINUX_ONLY
def test_lock_failure_before_open_leaks_no_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sakuramoon.telemetry import metrics as metrics_module

    class BrokenLock:
        def __init__(self) -> None:
            raise MemoryError("injected lock allocation failure")

    open_calls: list[tuple[tuple[object, ...], int]] = []
    monkeypatch.setattr(metrics_module.threading, "Lock", BrokenLock)
    monkeypatch.setattr(os, "open", _sentinel_open(open_calls))
    with pytest.raises(MemoryError):
        DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    assert open_calls == []


@LINUX_ONLY
def test_ctor_validation_failure_opens_no_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sakuramoon.telemetry import metrics as metrics_module

    def failing_validation(*_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("injected validation failure")

    open_calls: list[tuple[tuple[object, ...], int]] = []
    monkeypatch.setattr(metrics_module, "_nonnegative_int", failing_validation)
    monkeypatch.setattr(os, "open", _sentinel_open(open_calls))
    with pytest.raises(TypeError):
        DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=0)
    assert open_calls == []


@LINUX_ONLY
def test_write_close_success_keeps_format_mode_and_fsync_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    sink = DurableJsonlSink(target, fsync_every_records=2)
    fsyncs: list[int] = []

    def counting_fsync(fd: int) -> None:
        fsyncs.append(fd)
        _real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    sink.write({"a": 1, "b": [2, 3]})
    sink.write({"c": "d"})
    assert fsyncs == [sink._fd]  # pyright: ignore[reportPrivateUsage]
    sink.close()
    assert fsyncs == [sink._fd, sink._fd]  # pyright: ignore[reportPrivateUsage]
    expected = (
        json.dumps(
            {"a": 1, "b": [2, 3]}, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
        + json.dumps({"c": "d"}, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    assert target.read_bytes() == expected
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


@LINUX_ONLY
def test_second_instance_appends_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.jsonl"
    with DurableJsonlSink(target, fsync_every_records=1) as first:
        first.write({"n": 1})
    with DurableJsonlSink(target, fsync_every_records=1) as second:
        second.write({"n": 2})
    assert target.read_bytes() == b'{"n":1}\n{"n":2}\n'

    sink = DurableJsonlSink(target, fsync_every_records=1)
    sink.close()
    with pytest.raises(RuntimeError, match="closed"):
        sink.write({"n": 3})
    fsyncs: list[int] = []

    def counting_fsync(fd: int) -> None:
        fsyncs.append(fd)
        _real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    sink.close()
    assert fsyncs == []
    assert target.read_bytes() == b'{"n":1}\n{"n":2}\n'
