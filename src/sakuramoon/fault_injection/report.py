"""No-clobber publication for fault-matrix evidence."""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

from sakuramoon.fault_injection.schema import FaultMatrixReport


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_fault_matrix(path: Path, report: FaultMatrixReport) -> None:
    """Publish canonical JSON without replacing any existing filesystem entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("fault matrix evidence already exists")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    body = (
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            _fsync_file(handle.fileno())
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError("fault matrix temporary entry is not a regular file")
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                raise FileExistsError("fault matrix evidence already exists") from None
            published = True
            _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise


__all__ = ["write_fault_matrix"]
