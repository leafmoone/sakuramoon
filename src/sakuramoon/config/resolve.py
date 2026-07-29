"""Deterministic resolved-config serialization and hashing."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import tomli_w

from sakuramoon.config.redact import redact_mapping
from sakuramoon.config.schema import RuntimeConfig


def resolved_config_bytes(config: RuntimeConfig) -> bytes:
    """Serialize a validated config with stable schema order and no secret values."""

    payload = config.model_dump(mode="python", by_alias=True)
    redacted = redact_mapping(payload)
    return tomli_w.dumps(redacted).encode("utf-8")


def resolved_config_sha256(config: RuntimeConfig) -> str:
    return hashlib.sha256(resolved_config_bytes(config)).hexdigest()


def write_resolved_config(config: RuntimeConfig, destination: Path) -> str:
    """Atomically write resolved TOML and return its SHA-256."""

    lexical_destination = destination if destination.is_absolute() else Path.cwd() / destination
    if lexical_destination.is_symlink():
        raise ValueError(f"refusing to replace symlink: {destination}")
    current = Path(lexical_destination.anchor)
    for part in lexical_destination.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"resolved-config parent may not be a symlink: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"resolved-config parent is not a directory: {current}")
        else:
            current.mkdir()
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"could not create safe resolved-config parent: {current}")
    payload = resolved_config_bytes(config)
    temporary = lexical_destination.with_name(
        f".{lexical_destination.name}.{os.getpid()}.tmp"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary resolved-config path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, lexical_destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()
