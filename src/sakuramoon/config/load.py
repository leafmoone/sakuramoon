"""Root-confined TOML loading, deterministic merge, and strict validation."""

from __future__ import annotations

import copy
import hashlib
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal, cast

from pydantic import SecretStr, ValidationError

from sakuramoon.config.resolve import resolved_config_bytes
from sakuramoon.config.schema import (
    RuntimeConfig,
    looks_like_unresolved_sentinel,
    secret_environment_names,
)


class ConfigurationError(ValueError):
    """A safe-to-log configuration failure without input values."""

    unresolved_bindings: tuple[UnresolvedConfigBinding, ...]

    def __init__(
        self,
        message: str,
        *,
        unresolved_bindings: tuple[UnresolvedConfigBinding, ...] = (),
    ) -> None:
        super().__init__(message)
        self.unresolved_bindings = unresolved_bindings


UnresolvedBindingKind = Literal["benchmark", "decision", "required"]


@dataclass(frozen=True, order=True, slots=True)
class UnresolvedConfigBinding:
    """One safe, structured production input that has not been governed yet."""

    path: str
    sentinel: str
    kind: UnresolvedBindingKind

    def __post_init__(self) -> None:
        prefix = self.sentinel.partition("_")[0].lower()
        if (
            not self.path
            or re.fullmatch(r"(?:BENCHMARK|DECISION|REQUIRED)_[A-Z0-9_]+", self.sentinel)
            is None
            or prefix != self.kind
        ):
            raise ValueError("unresolved config binding is invalid")


@dataclass(frozen=True)
class InputFileDigest:
    path: str
    sha256: str


@dataclass(frozen=True)
class LoadedConfig:
    config: RuntimeConfig
    inputs: tuple[InputFileDigest, ...]
    resolved_toml: str
    resolved_sha256: str


def _safe_validation_error(exc: ValidationError) -> ConfigurationError:
    lines: list[str] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"{location}: {error['msg']} [{error['type']}]")
    return ConfigurationError("configuration validation failed:\n" + "\n".join(lines))


def _find_unresolved_bindings(
    value: object, prefix: str = ""
) -> list[UnresolvedConfigBinding]:
    bindings: list[UnresolvedConfigBinding] = []
    if isinstance(value, Mapping):
        table = cast(Mapping[object, object], value)
        for key, child in table.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            bindings.extend(_find_unresolved_bindings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(cast(list[object], value)):
            bindings.extend(_find_unresolved_bindings(child, f"{prefix}[{index}]"))
    elif looks_like_unresolved_sentinel(value):
        sentinel = cast(str, value)
        kind = cast(UnresolvedBindingKind, sentinel.partition("_")[0].lower())
        bindings.append(UnresolvedConfigBinding(prefix, sentinel, kind))
    return bindings


def _validate_path_components(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigurationError("config paths must be root-relative without traversal")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ConfigurationError(f"config symlink is forbidden: {relative.as_posix()}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(
            f"config file does not exist: {relative.as_posix()}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError("config path escapes the configured root") from exc
    if not resolved.is_file() or resolved.suffix != ".toml":
        raise ConfigurationError("config input must be a regular .toml file")
    return resolved


def _requested_relative(root: Path, requested: Path) -> Path:
    if requested.is_absolute():
        try:
            return requested.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError("config path escapes the configured root") from exc
    return requested


def _deep_merge(
    base: Mapping[str, Any], overlay: Mapping[str, Any], *, location: str = ""
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, overlay_value in overlay.items():
        child_location = f"{location}.{key}" if location else key
        if key not in merged:
            merged[key] = copy.deepcopy(overlay_value)
            continue
        base_value = merged[key]
        base_is_table = isinstance(base_value, Mapping)
        overlay_is_table = isinstance(overlay_value, Mapping)
        if base_is_table != overlay_is_table:
            raise ConfigurationError(
                f"table/scalar merge conflict at {child_location}"
            )
        if base_is_table:
            merged[key] = _deep_merge(
                cast(Mapping[str, Any], base_value),
                cast(Mapping[str, Any], overlay_value),
                location=child_location,
            )
        else:
            merged[key] = copy.deepcopy(overlay_value)
    return merged


class _Loader:
    def __init__(self, root: Path) -> None:
        lexical_root = root if root.is_absolute() else Path.cwd() / root
        current = Path(lexical_root.anchor)
        for part in lexical_root.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ConfigurationError("config root may not contain symlink components")
        try:
            self.root = lexical_root.resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError("config root does not exist") from exc
        if not self.root.is_dir():
            raise ConfigurationError("config root must be a non-symlink directory")
        self._active: list[Path] = []
        self._seen: set[Path] = set()
        self.inputs: list[InputFileDigest] = []

    def load(self, requested: Path, *, including: Path | None = None) -> dict[str, Any]:
        relative = _requested_relative(self.root, requested)
        if including is not None:
            relative = including.parent.relative_to(self.root) / relative
        path = _validate_path_components(self.root, relative)
        if path in self._active:
            chain = [item.relative_to(self.root).as_posix() for item in (*self._active, path)]
            raise ConfigurationError(f"extends cycle: {' -> '.join(chain)}")
        if path in self._seen:
            raise ConfigurationError(
                f"config included more than once: {path.relative_to(self.root).as_posix()}"
            )
        self._active.append(path)
        self._seen.add(path)
        try:
            raw_bytes = path.read_bytes()
            try:
                payload = tomllib.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise ConfigurationError(
                    f"invalid TOML in {path.relative_to(self.root).as_posix()}"
                ) from exc
            raw_includes = payload.pop("extends", [])
            if type(raw_includes) is not list or any(
                type(item) is not str for item in cast(list[object], raw_includes)
            ):
                raise ConfigurationError("extends must be an array of relative TOML paths")
            includes = cast(list[str], raw_includes)
            if len(includes) != len(set(includes)):
                raise ConfigurationError("extends contains duplicate paths")
            merged: dict[str, Any] = {}
            for include in includes:
                include_path = Path(include)
                if include_path.is_absolute() or ".." in PurePath(include).parts:
                    raise ConfigurationError(
                        "extends entries must be relative and may not traverse"
                    )
                merged = _deep_merge(merged, self.load(include_path, including=path))
            merged = _deep_merge(merged, payload)
            self.inputs.append(
                InputFileDigest(
                    path=path.relative_to(self.root).as_posix(),
                    sha256=hashlib.sha256(raw_bytes).hexdigest(),
                )
            )
            return merged
        finally:
            self._active.pop()


def _validate_secret_environment(
    config: RuntimeConfig, environment: Mapping[str, str]
) -> None:
    missing = [
        name
        for name in secret_environment_names(config)
        if name not in environment or not environment[name]
    ]
    if missing:
        raise ConfigurationError(
            "required secret environment variables are missing or empty: "
            + ", ".join(missing)
        )


def resolve_secret(name: str, environment: Mapping[str, str] | None = None) -> SecretStr:
    """Resolve one named variable without persisting or rendering its value."""

    selected = os.environ if environment is None else environment
    value = selected.get(name)
    if not value:
        raise ConfigurationError(f"required secret environment variable is missing: {name}")
    return SecretStr(value)


def unresolved_config_bindings(
    config_path: Path,
    *,
    config_root: Path,
) -> tuple[UnresolvedConfigBinding, ...]:
    """Inspect merged TOML bindings without resolving secrets or validating fallbacks."""

    loader = _Loader(config_root)
    payload = loader.load(config_path)
    return tuple(sorted(_find_unresolved_bindings(payload)))


def load_config(
    config_path: Path,
    *,
    config_root: Path,
    environment: Mapping[str, str] | None = None,
) -> LoadedConfig:
    """Load, merge, strictly validate, redact, and hash a runtime config."""

    loader = _Loader(config_root)
    payload = loader.load(config_path)
    unresolved = tuple(sorted(_find_unresolved_bindings(payload)))
    if unresolved:
        rendered = ", ".join(
            f"{binding.path}={binding.sentinel}" for binding in unresolved
        )
        raise ConfigurationError(
            "unresolved decision/benchmark placeholders at: " + rendered,
            unresolved_bindings=unresolved,
        )
    try:
        config = RuntimeConfig.model_validate(payload)
    except ValidationError as exc:
        raise _safe_validation_error(exc) from None
    _validate_secret_environment(config, os.environ if environment is None else environment)
    resolved = resolved_config_bytes(config)
    return LoadedConfig(
        config=config,
        inputs=tuple(loader.inputs),
        resolved_toml=resolved.decode("utf-8"),
        resolved_sha256=hashlib.sha256(resolved).hexdigest(),
    )
