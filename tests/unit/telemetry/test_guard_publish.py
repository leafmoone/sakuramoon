"""Unit tests for the atomic no-clobber publication primitive.

Covers the D4 commit-point protocol: precommit fsync failure -> no formal;
dual publisher -> at most one winner and the winner is complete;
postcommit temp-unlink failure -> still success + warning.  Also pins the
warning contract: a failing warning must never mask the primary exception
or flip a successful publication.

POSIX-specific cases are LINUX_ONLY (hosted Linux CI is the acceptance
gate; Windows skips are not acceptance).
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import threading
from pathlib import Path

import pytest

from sakuramoon.telemetry.guard_publish import PublicationStatus, publish_atomically

LINUX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX publication semantics (O_NOFOLLOW open flags)",
)


def _line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _writer(line: bytes):
    def write(temp: Path) -> None:
        temp.write_bytes(line)

    return write


def test_publish_success_content_and_temp_cleanup(tmp_path: Path) -> None:
    target = tmp_path / "rank-0" / "w1.jsonl"
    target.parent.mkdir()
    result = publish_atomically(target, tmp_path / ".tmp", _writer(_line({"a": 1})))
    assert result.status is PublicationStatus.PUBLISHED
    assert result.warnings == ()
    assert target.read_bytes() == _line({"a": 1})
    assert not list((tmp_path / ".tmp").iterdir())


def test_target_exists_keeps_competitor_and_cleans_temp(tmp_path: Path) -> None:
    target = tmp_path / "w1.jsonl"
    original = b'{"winner":1}\n'
    target.write_bytes(original)
    temps: list[Path] = []

    def prepare(temp: Path) -> None:
        temps.append(temp)
        temp.write_bytes(b'{"loser":2}\n')

    result = publish_atomically(target, tmp_path / ".tmp", prepare)
    assert result.status is PublicationStatus.TARGET_EXISTS
    assert target.read_bytes() == original  # competitor byte-identical
    assert temps and not temps[0].exists()


def test_precommit_failure_raises_original_and_no_formal(tmp_path: Path) -> None:
    target = tmp_path / "w1.jsonl"

    def prepare(temp: Path) -> None:
        temp.write_bytes(b"partial")  # short write, then fail
        raise OSError(28, "injected write failure")

    with pytest.raises(OSError, match="injected write failure"):
        publish_atomically(target, tmp_path / ".tmp", prepare)
    assert not target.exists()
    assert not list((tmp_path / ".tmp").iterdir())


def test_precommit_original_identity_preserved_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed temp cleanup must not mask the original exception (identity)."""

    class _OriginalError(RuntimeError):
        pass

    original = _OriginalError("the real precommit error")

    def prepare(temp: Path) -> None:
        temp.write_bytes(b"x")
        raise original

    def broken_unlink(path: Path) -> None:
        raise OSError(16, "cleanup fails")

    monkeypatch.setattr(Path, "unlink", broken_unlink)
    with pytest.raises(_OriginalError) as excinfo:
        publish_atomically(tmp_path / "w1.jsonl", tmp_path / ".tmp", prepare)
    assert excinfo.value is original  # same identity, never wrapped
    assert not (tmp_path / "w1.jsonl").exists()


def test_postcommit_unlink_failure_still_success_with_stderr_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "rank-0" / "w1.jsonl"
    target.parent.mkdir()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)

    def unlink(path: Path) -> None:
        if path.name.startswith("w1.jsonl.tmp-"):
            raise OSError(16, "injected unlink failure")
        path.unlink()

    monkeypatch.setattr(Path, "unlink", unlink)
    result = publish_atomically(target, tmp_path / ".tmp", _writer(_line({"a": 1})))
    assert result.status is PublicationStatus.PUBLISHED
    assert result.warnings and "unlink" in result.warnings[0]
    assert target.read_bytes() == _line({"a": 1})  # published target intact
    assert "WARNING" in stderr.getvalue() and "unlink" in stderr.getvalue()
    leftovers = list((tmp_path / ".tmp").iterdir())
    assert len(leftovers) == 1 and leftovers[0].name.startswith("w1.jsonl.tmp-")


def test_failing_warning_cannot_mask_precommit_or_flip_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warning whose print fails must not mask the primary exception
    (precommit) nor flip a successful publication (postcommit)."""

    class _BrokenStderr:
        def write(self, *_: object) -> int:
            raise OSError(5, "stderr broken")

    monkeypatch.setattr("sys.stderr", _BrokenStderr())

    class _OriginalError(RuntimeError):
        pass

    original = _OriginalError("primary precommit error")

    def prepare_fail(temp: Path) -> None:
        temp.write_bytes(b"x")
        raise original

    with pytest.raises(_OriginalError) as excinfo:
        publish_atomically(tmp_path / "w1.jsonl", tmp_path / ".tmp", prepare_fail)
    assert excinfo.value is original
    assert not (tmp_path / "w1.jsonl").exists()

    # postcommit: unlink fails AND stderr is broken -> still PUBLISHED.
    def broken_unlink2(path: Path) -> None:
        raise OSError(16, "unlink fails")

    monkeypatch.setattr(Path, "unlink", broken_unlink2)
    target2 = tmp_path / "rank-0" / "w2.jsonl"
    target2.parent.mkdir(exist_ok=True)
    result = publish_atomically(target2, tmp_path / ".tmp", _writer(_line({"b": 2})))
    assert result.status is PublicationStatus.PUBLISHED
    assert result.warnings  # the cleanup warning is still reported in-band


def test_dual_publisher_at_most_one_winner_and_winner_complete(
    tmp_path: Path,
) -> None:
    target = tmp_path / "rank-0" / "race.jsonl"
    target.parent.mkdir()
    barrier = threading.Barrier(2)
    results: list[PublicationStatus] = []
    lock = threading.Lock()

    def attempt(payload: dict[str, object]) -> None:
        barrier.wait()
        result = publish_atomically(
            target,
            tmp_path / ".tmp",
            _writer(_line(payload)),
        )
        with lock:
            results.append(result.status)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, {"p": i}) for i in (0, 1)]
        for future in futures:
            future.result()

    assert sorted(r is PublicationStatus.PUBLISHED for r in results) == [False, True]
    body = json.loads(target.read_text())
    assert body["p"] in (0, 1)  # exactly one complete payload, no interleaving
    assert not list((tmp_path / ".tmp").iterdir())


@LINUX_ONLY
def test_link_never_attempted_after_prepare_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poison latch at the primitive level: a prepare failure must not reach
    os.link even if later cleanup succeeds (no SEALED, no formal)."""
    linked: list[bool] = []
    real_link = os.link

    def spy_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        linked.append(True)
        real_link(src, dst)

    monkeypatch.setattr(os, "link", spy_link)

    def prepare(temp: Path) -> None:
        temp.write_bytes(b"x")
        raise OSError(28, "short write")

    target = tmp_path / "rank-0" / "w1.jsonl"
    target.parent.mkdir()
    with pytest.raises(OSError, match="short write"):
        publish_atomically(target, tmp_path / ".tmp", prepare)
    assert linked == []
    assert not target.exists()
    assert not list((tmp_path / ".tmp").iterdir())


@LINUX_ONLY
def test_temp_created_0600_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat as stat_module

    target = tmp_path / "w1.jsonl"
    seen: list[tuple[int, bool, int]] = []
    real_open = os.open

    def spy_open(
        file: str | os.PathLike[str],
        flags: int,
        mode: int | None = None,
        *rest: object,
    ) -> int:
        fd = real_open(file, flags, mode if mode is not None else 0o666, *rest)  # type: ignore[arg-type]
        # fstat while the fd is still open (a later fstat would EBADF).
        metadata = os.fstat(fd)
        seen.append(
            (
                fd,
                stat_module.S_ISREG(metadata.st_mode),
                stat_module.S_IMODE(metadata.st_mode),
            )
        )
        return fd

    monkeypatch.setattr(os, "open", spy_open)
    publish_atomically(target, tmp_path / ".tmp", _writer(_line({"z": 0})))
    assert seen, "expected at least one os.open call"
    # The temp file is opened with 0600 and must be a regular file; the dir
    # fsync open (if any) is O_RDONLY on a directory and is not asserted.
    temp_opens = [entry for entry in seen if entry[2] == 0o600]
    assert temp_opens, "expected the temp file open with mode 0600"
    for _fd, is_reg, _mode in temp_opens:
        assert is_reg


@LINUX_ONLY
def test_fstat_and_close_dual_fault_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WSL-reproduced dual fault: fstat raises the PRIMARY exception and the
    subsequent cleanup close (which first closes for real) raises a SECONDARY
    one.  The top-level exception must keep the primary identity, and the
    temp must still be removed."""
    import sakuramoon.telemetry.guard_publish as gp

    target = tmp_path / "w1.jsonl"
    real_open, real_close, real_fstat = gp.os.open, gp.os.close, gp.os.fstat
    captured: dict[int, None] = {}

    def fake_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o666, *rest: object
    ) -> int:
        fd = real_open(path, flags, mode, *rest)
        captured[fd] = None
        return fd

    def faulty_fstat(fd: int) -> object:
        if fd in captured:
            raise RuntimeError("PRIMARY_FSTAT")
        return real_fstat(fd)

    def faulty_close(fd: int) -> None:
        real_close(fd)
        if fd in captured:
            raise OSError(5, "SECONDARY_CLOSE")

    monkeypatch.setattr(gp.os, "open", fake_open)
    monkeypatch.setattr(gp.os, "fstat", faulty_fstat)
    monkeypatch.setattr(gp.os, "close", faulty_close)
    with pytest.raises(RuntimeError) as excinfo:
        gp._make_temp(tmp_path / ".tmp", target)  # pyright: ignore[reportPrivateUsage]
    assert excinfo.value.args == ("PRIMARY_FSTAT",)  # primary identity preserved
    assert not list((tmp_path / ".tmp").iterdir())  # temp removed despite close fault


@LINUX_ONLY
def test_success_path_close_fault_cleans_temp_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fstat OK but the final close fails: the module's own temp is removed
    and the close failure (the primary here) propagates; the already-closed
    FD is never closed twice."""
    import sakuramoon.telemetry.guard_publish as gp

    target = tmp_path / "w1.jsonl"
    real_open, real_close = gp.os.open, gp.os.close
    captured: dict[int, list[int]] = {}

    def fake_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o666, *rest: object
    ) -> int:
        fd = real_open(path, flags, mode, *rest)
        captured[fd] = []
        return fd

    def faulty_close(fd: int) -> None:
        calls = captured.setdefault(fd, [])
        calls.append(fd)
        if len(calls) == 1:
            real_close(fd)
            raise OSError(5, "close fault")
        # Second attempt on the already-closed fd: emulate the kernel's
        # EBADF without touching the kernel twice.
        raise OSError(9, "Bad file descriptor (simulated second close)")

    monkeypatch.setattr(gp.os, "open", fake_open)
    monkeypatch.setattr(gp.os, "close", faulty_close)
    with pytest.raises(OSError, match="close fault"):
        gp._make_temp(tmp_path / ".tmp", target)  # pyright: ignore[reportPrivateUsage]
    for calls in captured.values():
        assert len(calls) <= 2  # the retry is the second and final attempt
    assert not list((tmp_path / ".tmp").iterdir())
