"""Redaction helpers that never expose credential material."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import SecretBytes, SecretStr

REDACTED = "<redacted>"
_SECRET_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:api_?key|password|secret|token|credential)(?:_|$)"
)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith("_env"):
        return False
    return _SECRET_KEY_PATTERN.search(lowered) is not None


def redact_value(value: object, *, key: str = "") -> object:
    """Return a recursively redacted, serialization-safe value."""

    if isinstance(value, (SecretStr, SecretBytes)) or (key and _is_secret_key(key)):
        return REDACTED
    if isinstance(value, Mapping):
        table = cast(Mapping[object, object], value)
        return {
            str(child_key): redact_value(child, key=str(child_key))
            for child_key, child in table.items()
        }
    if isinstance(value, tuple):
        return [redact_value(child) for child in cast(tuple[object, ...], value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(child) for child in cast(Sequence[object], value)]
    return value


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    redacted = redact_value(value)
    if not isinstance(redacted, dict):
        raise TypeError("redaction root must remain a mapping")
    return cast(dict[str, Any], redacted)
