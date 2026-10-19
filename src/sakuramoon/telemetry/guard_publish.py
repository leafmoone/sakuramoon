"""Atomic no-clobber single-record publication (guard window telemetry).

Pure standard library: no torch, no project imports.  Loadable by file path
for independent POSIX verification.

Commit point
------------
``os.link(temp, target)`` is the LAST fallible step.  Everything the caller
must durably prepare (serialize, write, fsync, close) happens BEFORE the
link, inside ``prepare``.  Semantics:

- any precommit failure (temp open, ``prepare``, link other than EEXIST):
  the ORIGINAL exception propagates unchanged; our temp is removed
  best-effort (a failed cleanup, and a failed warning, never mask it); no
  formal target is created and an existing competitor target is never
  touched.
- ``os.link`` raising ``FileExistsError``: the competitor target is kept
  byte-identical, our temp is removed best-effort, and the call returns
  ``PublicationStatus.TARGET_EXISTS`` (a normal outcome, not an error).
- link success: the target is atomically visible with complete content and
  the file is already fsynced by ``prepare``.  We do NOT promise crash
  durability of the new directory entry, and there is no post-link fsync
  failure path.  The temp is then unlinked best-effort: a cleanup failure
  reports a warning (stderr, best-effort) but the publication still
  reports success, and we never unlink the published target.

Warnings go to stderr through :func:`report_warning`, which cannot raise:
a failing warning must never mask the primary exception or flip a
successful publication.

Known shared-sink gaps (documented, NOT fixed here; the baseline
``DurableJsonlSink`` is intentionally unchanged):

- if its constructor raises after ``os.open`` (e.g. the directory fsync
  fails for a new file), the half-initialized object is unreachable from
  the caller (the attribute assignment never completes) and its FD leaks
  until process exit.  The guard stream treats any such failure as
  session-disable (poison latched), so the leak is bounded to one FD per
  failed session, not unbounded.
- if ``close()`` raises from its fsync, ``os.close`` is not reached and the
  FD stays open; same session-disable bound applies.
"""

from __future__ import annotations

import os
import stat
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PublicationStatus(Enum):
    """Outcome of an atomic no-clobber publication."""

    PUBLISHED = "published"
    TARGET_EXISTS = "target_exists"


@dataclass(frozen=True)
class PublicationResult:
    """Result of :func:`publish_atomically`.

    ``warnings`` carries post-commit cleanup problems (e.g. the temp file
    could not be removed after a successful link); a non-empty ``warnings``
    does not change ``status``.
    """

    status: PublicationStatus
    target: Path
    temp: Path | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def report_warning(message: str) -> None:
    """Best-effort stderr warning that cannot raise or mask anything."""
    try:
        print(f"[guard_publish] WARNING: {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001,S110 - a warning must never change the outcome
        pass


def _optional_flags() -> int:
    flags = 0
    for name in ("O_NOFOLLOW", "O_CLOEXEC"):
        flags |= getattr(os, name, 0)
    return flags


def _best_effort_unlink(path: Path) -> str | None:
    """Remove ``path``; return a warning message on failure, else None."""
    try:
        path.unlink()
    except FileNotFoundError:
        return None
    except OSError as error:
        warning = f"could not remove temp file {path}: {error}"
        report_warning(warning)
        return warning


def _make_temp(temp_dir: Path, target: Path) -> Path:
    """Create a unique empty temp file (0600, regular) in ``temp_dir``.

    ``temp_dir`` must be on the same filesystem as ``target``'s parent for
    the link to be atomic; the caller is responsible for that.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp = temp_dir / f"{target.name}.tmp-{uuid.uuid4().hex[:12]}"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _optional_flags(), 0o600)
    closed = False
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"temp file {temp} is not a regular file")
        os.close(fd)
        closed = True
    except BaseException:
        # Exception path: release the fd we still own (never close twice),
        # then best-effort remove the temp.  A failing close must WARN, not
        # become the new top-level exception: the ORIGINAL exception must
        # propagate unchanged (primary identity preserved).
        if not closed:
            try:
                os.close(fd)
            except Exception:  # noqa: BLE001 - warning only, never masks
                report_warning(f"could not close temp file descriptor {fd}")
        _best_effort_unlink(temp)
        raise
    return temp


def publish_atomically(
    target: Path,
    temp_dir: Path,
    prepare: Callable[[Path], None],
) -> PublicationResult:
    """Publish ``target`` atomically without ever clobbering an existing one.

    ``prepare(temp_path)`` must fully write, fsync, and close ``temp_path``
    (or raise).  The commit is ``os.link(temp, target)``; see the module
    docstring for the precommit/post-commit contract.

    Raises the ORIGINAL ``prepare``/I/O exception on precommit failure
    (never wrapped, never masked by a cleanup or warning error).
    """
    temp = _make_temp(temp_dir, target)
    try:
        prepare(temp)
    except BaseException:
        _best_effort_unlink(temp)
        raise
    warnings: list[str] = []
    try:
        # COMMIT POINT: last fallible step.
        os.link(temp, target)
    except FileExistsError:
        warning = _best_effort_unlink(temp)
        if warning is not None:
            warnings.append(warning)
        return PublicationResult(
            status=PublicationStatus.TARGET_EXISTS,
            target=target,
            temp=temp,
            warnings=tuple(warnings),
        )
    except OSError:
        _best_effort_unlink(temp)
        raise
    # post-commit: best-effort temp cleanup; failure is a warning only and
    # never touches the published target.
    warning = _best_effort_unlink(temp)
    if warning is not None:
        warnings.append(warning)
    return PublicationResult(
        status=PublicationStatus.PUBLISHED,
        target=target,
        temp=temp,
        warnings=tuple(warnings),
    )


__all__ = [
    "PublicationResult",
    "PublicationStatus",
    "publish_atomically",
    "report_warning",
]
