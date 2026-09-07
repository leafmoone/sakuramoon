"""TOML loading, deterministic merge, legacy normalization, and validation.

Path rules: config inputs are trusted.  A top-level config path may be
absolute (used as given) or relative to the deployment root; ``extends``
entries resolve against the containing file.  Symlink components are
allowed.  Cycles in the extends graph are still rejected (with the cycle
chain), and the same file may be extended through multiple parents
(diamonds) — the merge is deterministic (left-to-right).
"""

from __future__ import annotations

import copy
import os
import re
import sys
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
            or re.fullmatch(
                r"(?:BENCHMARK|DECISION|REQUIRED)_[A-Z0-9_]+", self.sentinel
            )
            is None
            or prefix != self.kind
        ):
            raise ValueError("unresolved config binding is invalid")


@dataclass(frozen=True)
class InputFile:
    path: str


@dataclass(frozen=True)
class LoadedConfig:
    config: RuntimeConfig
    inputs: tuple[InputFile, ...]
    resolved_toml: str


def _safe_validation_error(exc: ValidationError) -> ConfigurationError:
    lines: list[str] = []
    for error in exc.errors(
        include_url=False, include_context=False, include_input=False
    ):
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


def _validate_config_file(path: Path) -> Path:
    """A config input is a trusted path: it must exist as a regular .toml file."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"config file does not exist: {path.as_posix()}") from exc
    if not resolved.is_file() or resolved.suffix != ".toml":
        raise ConfigurationError(f"config input must be a regular .toml file: {path.as_posix()}")
    return resolved


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
            raise ConfigurationError(f"table/scalar merge conflict at {child_location}")
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
        try:
            self.root = lexical_root.resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError("config root does not exist") from exc
        if not self.root.is_dir():
            raise ConfigurationError("config root must be a directory")
        self._active: list[Path] = []
        self.inputs: list[InputFile] = []

    def _record(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def load(self, requested: Path, *, including: Path | None = None) -> dict[str, Any]:
        if including is None:
            path = (
                requested
                if requested.is_absolute()
                else self.root / requested
            )
        else:
            entry = PurePath(requested.as_posix())
            if entry.is_absolute():
                raise ConfigurationError(
                    "extends entries must be relative to the including file"
                )
            path = (including.parent / entry)
        path = _validate_config_file(path)
        if path in self._active:
            chain = [self._record(item) for item in (*self._active, path)]
            raise ConfigurationError(f"extends cycle: {' -> '.join(chain)}")
        self._active.append(path)
        try:
            raw_bytes = path.read_bytes()
            try:
                payload = tomllib.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise ConfigurationError(
                    f"invalid TOML in {self._record(path)}"
                ) from exc
            raw_includes = payload.pop("extends", [])
            if type(raw_includes) is not list or any(
                type(item) is not str for item in cast(list[object], raw_includes)
            ):
                raise ConfigurationError(
                    "extends must be an array of relative TOML paths"
                )
            includes = cast(list[str], raw_includes)
            if len(includes) != len(set(includes)):
                raise ConfigurationError("extends contains duplicate paths")
            merged: dict[str, Any] = {}
            for include in includes:
                merged = _deep_merge(merged, self.load(Path(include), including=path))
            merged = _deep_merge(merged, payload)
            self.inputs.append(InputFile(path=self._record(path)))
            return merged
        finally:
            self._active.pop()


# ---------------------------------------------------------------------------
# One-time legacy normalization (the only legacy translation in the codebase).
#
# Historical [stage] vocabulary is translated to the current interface at the
# config-read boundary: stage.{resolution,local_batch,accumulation,
# planned_updates,activation_checkpoint_mode,global_batch} -> [train],
# stage.world_size -> [distributed], stage.depth -> [model.dit].depth,
# stage.name/run.stage -> run.label (display-only).  Removed capability
# fields (storage, failure, pinned shapes, timing phase table, ...) are
# dropped with a one-time notice.  A new/old pair that carries the same
# meaning but different values is a configuration conflict error.
# ---------------------------------------------------------------------------

_STAGE_TO_TRAIN: tuple[tuple[str, str], ...] = (
    ("resolution", "resolution"),
    ("local_batch", "local_batch"),
    ("accumulation", "accumulation"),
    ("planned_updates", "max_updates"),
    ("activation_checkpoint_mode", "activation_checkpoint_mode"),
    ("global_batch", "global_batch"),
)
# (table, key) -> reason; value must match the enforced invariant or error.
_DROP_WITH_VALUE_CHECK: tuple[tuple[str, str, str, object, str], ...] = (
    ("gradient", "global_sample_mean", "true", True, "sample-mean gradient normalization is an unconditional invariant"),
    ("distributed", "frozen_encoders_outside_wrapper", "true", True, "frozen encoders outside the wrapper is an unconditional invariant"),
    ("distributed", "automatic_backend_fallback", "false", False, "automatic backend fallback is not supported"),
)
# (table, key) pairs dropped with a notice only (no value constraint).
_DROP_NOTICE_ONLY: tuple[tuple[str, str], ...] = (
    ("data.buckets", "shape_count"),
    ("model.dit", "stable_slot_count"),
    ("model.dit", "patch_size"),
    ("model.packing", "cross_sample_attention"),
    ("scheduler", "after_warmup"),
    ("cfg", "full_interval"),
    ("cfg", "rescale"),
    ("growth", "alpha_fraction"),
    ("growth", "min_updates"),
    ("growth", "max_updates"),
    ("growth", "random_new_slots"),
    ("growth", "copy_old_slots"),
    ("timing", "cuda_events"),
    ("timing", "force_synchronize_each_phase"),
    ("timing", "phases"),
)
_STAGE_DROPPED_KEYS = frozenset(
    {"name", "enabled", "predecessor", "manual_finalize", "automatic_transition"}
)


def _assign_or_conflict(
    payload: dict[str, Any],
    table_path: str,
    key: str,
    value: Any,
    *,
    notices: list[str],
) -> None:
    table = payload
    parts = table_path.split(".")
    for part in parts[:-1]:
        child = table.get(part)
        if type(child) is not dict:
            table[part] = {}
            table = table[part]
        else:
            table = child
    if key in table and table[key] != value:
        raise ConfigurationError(
            f"config conflict: legacy stage value for {table_path}.{key} "
            "disagrees with the current config field"
        )
    table[key] = value
    notices.append(
        f"config: legacy key stage.{key} mapped to {table_path}.{key} "
        "(one-time migration notice)"
    )


def _remove_with_notice(
    payload: dict[str, Any],
    location: str,
    *,
    notices: list[str],
    expected: object | None = None,
    invariant: str | None = None,
) -> None:
    parts = location.split(".")
    table = payload
    for part in parts[:-1]:
        child = table.get(part)
        if type(child) is not dict:
            return
        table = child
    if parts[-1] not in table:
        return
    if expected is not None and table[parts[-1]] is not expected:
        raise ConfigurationError(
            f"config conflict: {location} must be {expected!r} "
            f"({invariant or 'enforced invariant'})"
        )
    del table[parts[-1]]
    notices.append(f"config: deprecated key {location} ignored (one-time notice)")


def normalize_legacy_config(payload: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Translate the historical stage vocabulary once, at the read boundary.

    Returns the normalized payload and the human-readable notices that were
    emitted.  This is the single legacy-normalization function; runtime code
    never sees stage vocabulary.
    """

    notices: list[str] = []
    data = copy.deepcopy(payload)

    # Whole removed tables.
    if "storage" in data:
        del data["storage"]
        notices.append(
            "config: removed table [storage] ignored (deployment facts are "
            "decided at runtime, one-time notice)"
        )
    if "failure" in data:
        del data["failure"]
        notices.append(
            "config: removed table [failure] ignored (failure behavior is "
            "unconditional, one-time notice)"
        )

    stage = data.get("stage")
    if stage is not None:
        if type(stage) is not dict:
            raise ConfigurationError("stage must be a table")
        stage = cast(dict[str, Any], stage)
        run = data.get("run")
        if not isinstance(run, dict):
            run = {}
            data["run"] = run
        depth_target: Any = None
        for source_key, target_key in _STAGE_TO_TRAIN:
            if source_key in stage:
                _assign_or_conflict(
                    data, "train", target_key, stage[source_key], notices=notices
                )
        if "world_size" in stage:
            _assign_or_conflict(
                data, "distributed", "world_size", stage["world_size"], notices=notices
            )
        if "depth" in stage:
            dit = data.get("model")
            if not isinstance(dit, dict):
                data["model"] = {}
                dit = data["model"]
            dit_table = dit.get("dit")
            if not isinstance(dit_table, dict):
                dit["dit"] = {}
                dit_table = dit["dit"]
            depth_target = stage["depth"]
            if "depth" in dit_table and dit_table["depth"] != depth_target:
                raise ConfigurationError(
                    "config conflict: legacy stage.depth disagrees with "
                    "model.dit.depth"
                )
            dit_table["depth"] = depth_target
            notices.append(
                "config: legacy key stage.depth mapped to model.dit.depth "
                "(one-time migration notice)"
            )
        if "name" in stage and isinstance(run.get("label"), (str, type(None))):
            if run.get("label") is not None and run["label"] != stage["name"]:
                raise ConfigurationError(
                    "config conflict: legacy stage.name disagrees with run.label"
                )
            run["label"] = stage["name"]
            notices.append(
                "config: legacy key stage.name mapped to run.label "
                "(display-only, one-time migration notice)"
            )
        for key in stage:
            if key in _STAGE_DROPPED_KEYS:
                notices.append(
                    f"config: deprecated key stage.{key} ignored (one-time notice)"
                )
        del data["stage"]
        if "stage" in run:
            if run.get("label") is not None and run["label"] != run["stage"]:
                raise ConfigurationError(
                    "config conflict: run.stage disagrees with run.label"
                )
            run["label"] = run["stage"]
            notices.append(
                "config: deprecated key run.stage mapped to run.label "
                "(display-only, one-time notice)"
            )
            del run["stage"]

    for table_path, key, _expected_text, expected, invariant in _DROP_WITH_VALUE_CHECK:
        _remove_with_notice(
            data,
            f"{table_path}.{key}",
            notices=notices,
            expected=expected,
            invariant=invariant,
        )
    for table_path, key in _DROP_NOTICE_ONLY:
        _remove_with_notice(data, f"{table_path}.{key}", notices=notices)

    return data, tuple(notices)


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


def resolve_secret(
    name: str, environment: Mapping[str, str] | None = None
) -> SecretStr:
    """Resolve one named variable without persisting or rendering its value."""

    selected = os.environ if environment is None else environment
    value = selected.get(name)
    if not value:
        raise ConfigurationError(
            f"required secret environment variable is missing: {name}"
        )
    return SecretStr(value)


def unresolved_config_bindings(
    config_path: Path,
    *,
    config_root: Path,
) -> tuple[UnresolvedConfigBinding, ...]:
    """Inspect merged TOML bindings without resolving secrets or validating fallbacks."""

    loader = _Loader(config_root)
    payload, _ = normalize_legacy_config(loader.load(config_path))
    return tuple(sorted(_find_unresolved_bindings(payload)))


def load_config(
    config_path: Path,
    *,
    config_root: Path,
    environment: Mapping[str, str] | None = None,
    validate_secrets: bool = True,
) -> LoadedConfig:
    """Load, merge, normalize legacy keys, validate, and redact a runtime config."""

    if type(validate_secrets) is not bool:
        raise TypeError("validate_secrets must be a bool")

    loader = _Loader(config_root)
    payload, notices = normalize_legacy_config(loader.load(config_path))
    for notice in notices:
        print(notice, file=sys.stderr)
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
    except (TypeError, ValueError) as exc:
        # A before-validator may raise a plain type error; keep the loader
        # contract: every load failure is a safe, value-free error.
        raise ConfigurationError(f"configuration validation failed: {exc}") from None
    if validate_secrets:
        _validate_secret_environment(
            config,
            os.environ if environment is None else environment,
        )
    resolved = resolved_config_bytes(config)
    return LoadedConfig(
        config=config,
        inputs=tuple(loader.inputs),
        resolved_toml=resolved.decode("utf-8"),
    )
