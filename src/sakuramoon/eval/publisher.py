"""Atomic no-clobber publication for one complete evaluator run."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath

import torch
from PIL import Image

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class EvaluationPublicationError(RuntimeError):
    """The evaluator output tree cannot be published durably."""


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise EvaluationPublicationError("artifact relative path is invalid")
    return Path(*path.parts)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    directories: list[Path] = []
    for root, child_directories, _files in os.walk(path, topdown=False):
        current = Path(root)
        if current.is_symlink():
            raise EvaluationPublicationError("evaluator staging tree contains a symlink")
        for child in child_directories:
            if (current / child).is_symlink():
                raise EvaluationPublicationError(
                    "evaluator staging tree contains a symlink"
                )
        directories.append(current)
    for directory in directories:
        _fsync_directory(directory)


def _remove_staging_payload(path: Path) -> None:
    """Remove every staged entry except the root COMPLETE commit source."""

    for root, child_directories, files in os.walk(path, topdown=False):
        current = Path(root)
        for file_name in files:
            if current == path and file_name == "COMPLETE":
                continue
            (current / file_name).unlink()
        for child in child_directories:
            (current / child).rmdir()


def _publish_tree_noreplace(staging: Path, destination: Path) -> None:
    """Reserve the final path and publish COMPLETE last using NFS-safe links."""

    destination.mkdir(mode=0o700)
    _fsync_directory(destination.parent)
    for root, child_directories, files in os.walk(staging):
        source_directory = Path(root)
        relative = source_directory.relative_to(staging)
        destination_directory = destination / relative
        for child in sorted(child_directories):
            (destination_directory / child).mkdir(mode=0o700)
        for file_name in sorted(files):
            if relative == Path() and file_name == "COMPLETE":
                continue
            os.link(
                source_directory / file_name,
                destination_directory / file_name,
                follow_symlinks=False,
            )
    _fsync_tree(destination)

    # Remove the staging payload before exposing COMPLETE. The final hard links retain
    # the already-fsynced inodes, while any cleanup failure leaves no commit marker.
    _remove_staging_payload(staging)
    os.link(
        staging / "COMPLETE",
        destination / "COMPLETE",
        follow_symlinks=False,
    )
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    with contextlib.suppress(OSError):
        (staging / "COMPLETE").unlink()
        staging.rmdir()
        _fsync_directory(destination.parent)


class AtomicEvaluationPublisher:
    """Stage outputs, reserve the final directory, and link COMPLETE only at commit."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        if (
            not output_root.is_absolute()
            or ".." in output_root.parts
            or _RUN_ID.fullmatch(run_id) is None
        ):
            raise EvaluationPublicationError("evaluator publication target is invalid")
        if _has_symlink_component(output_root):
            raise EvaluationPublicationError(
                "evaluator output path contains a symbolic link"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(output_root) or not output_root.is_dir():
            raise EvaluationPublicationError("evaluator output root is invalid")
        self.output_root = output_root
        self.final_path = output_root / run_id
        self.staging_path = output_root / f".{run_id}.incomplete"
        if (
            self.final_path.exists()
            or self.final_path.is_symlink()
            or self.staging_path.exists()
            or self.staging_path.is_symlink()
        ):
            raise FileExistsError("evaluator output run already exists")
        self.staging_path.mkdir(mode=0o700)
        _fsync_directory(output_root)
        self._committed = False

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        if self._committed or type(payload) is not bytes:
            raise EvaluationPublicationError("evaluator publisher is not writable")
        relative = _relative_path(relative_path)
        destination = self.staging_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("evaluator artifact already exists")
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError("evaluator artifact temporary path exists")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def write_json(self, relative_path: str, payload: dict[str, object]) -> Path:
        body = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        return self.write_bytes(relative_path, body)

    def write_png(self, relative_path: str, image: torch.Tensor) -> tuple[Path, bytes]:
        if image.dtype != torch.uint8 or image.device.type != "cpu" or image.ndim != 3:
            raise EvaluationPublicationError("PNG image must be CPU uint8 [3,H,W]")
        if image.shape[0] != 3:
            raise EvaluationPublicationError("PNG image must have three RGB channels")
        array = image.permute(1, 2, 0).contiguous().numpy()
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        payload = buffer.getvalue()
        return self.write_bytes(relative_path, payload), payload

    def commit(self, summary: dict[str, object]) -> Path:
        if self._committed:
            raise EvaluationPublicationError("evaluator run is already committed")
        self.write_json("summary.json", summary)
        self.write_bytes("COMPLETE", b"complete\n")
        _fsync_tree(self.staging_path)
        _publish_tree_noreplace(self.staging_path, self.final_path)
        self._committed = True
        return self.final_path


__all__ = [
    "AtomicEvaluationPublisher",
    "EvaluationPublicationError",
]
