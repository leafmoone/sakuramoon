"""Whole-shard validation selection, preparation, and strict prompt loading."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from PIL import Image

from sakuramoon.data.caption import CaptionFields
from sakuramoon.data.manifest import DatasetManifest, DatasetManifestError, ShardRecord
from sakuramoon.data.modelscope import (
    FetchedShard,
    ModelScopeDatasetTransport,
    fetch_dataset_shard,
)

VALIDATION_SELECTION_SEED = 44
VALIDATION_SHARD_PATHS = (
    "data/2_2026.1/shard-000509.tar",
    "data/2_2026.1/shard-000060.tar",
)
VALIDATION_SHARD_COUNT = len(VALIDATION_SHARD_PATHS)
_MAX_SELECTION_BYTES = 64 * 1024
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_IMAGE_SUFFIXES = frozenset({"jpg", "jpeg", "png", "webp"})


class ValidationSelectionError(ValueError):
    """The whole-shard validation contract cannot be established."""


class ValidationSelectionExistsError(ValidationSelectionError):
    """A no-clobber selection destination already exists."""


class ValidationPromptError(ValueError):
    """A selected validation shard is not a strict image/JSON prompt source."""


CaptionFieldsParser = Callable[[Mapping[str, object]], CaptionFields]


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    selection_id: str
    manifest_id: str
    seed: int
    shards: tuple[ShardRecord, ...]

    def __post_init__(self) -> None:
        paths = tuple(item.path for item in self.shards)
        if (
            not _is_sha256(self.selection_id)
            or not _is_sha256(self.manifest_id)
            or type(self.seed) is not int
            or self.seed != VALIDATION_SELECTION_SEED
            or type(self.shards) is not tuple
            or len(self.shards) != VALIDATION_SHARD_COUNT
            or any(type(item) is not ShardRecord for item in self.shards)
            or paths != VALIDATION_SHARD_PATHS
            or self.selection_id != _selection_id(self.manifest_id, self.seed, self.shards)
        ):
            raise ValidationSelectionError("validation shard selection is invalid")

    @property
    def shard_paths(self) -> tuple[str, str]:
        return cast(tuple[str, str], tuple(item.path for item in self.shards))


@dataclass(frozen=True, slots=True)
class PreparedValidationShards:
    selection: ValidationSelection
    root: Path
    paths: tuple[Path, Path]


@dataclass(frozen=True, slots=True)
class ValidationPromptSample:
    prompt_id: str
    sample_id: int
    source_shard: str
    member_key: str
    prompt: str | None
    seed: int
    height: int
    width: int
    caption_fields: CaptionFields | None = None

    def __post_init__(self) -> None:
        if (
            type(self.prompt_id) is not str
            or not self.prompt_id.startswith("validation-")
            or len(self.prompt_id) != 43
            or type(self.sample_id) is not int
            or self.sample_id <= 0
            or type(self.source_shard) is not str
            or not self.source_shard
            or type(self.member_key) is not str
            or not self.member_key
            or (
                self.prompt is not None
                and (
                    type(self.prompt) is not str
                    or not self.prompt
                    or self.prompt != self.prompt.strip()
                    or "<think>" in self.prompt
                    or "</think>" in self.prompt
                )
            )
            or (
                self.caption_fields is not None
                and type(cast(object, self.caption_fields)) is not CaptionFields
            )
            or (self.prompt is None and self.caption_fields is None)
            or type(self.seed) is not int
            or self.seed < 0
            or any(
                type(value) is not int or value <= 0
                for value in (self.height, self.width)
            )
        ):
            raise ValidationPromptError("validation prompt sample is invalid")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _record_mapping(record: ShardRecord) -> dict[str, object]:
    return {
        "bytes": record.bytes,
        "path": record.path,
        "upstream_sha256": record.upstream_sha256,
    }


def _selection_identity_bytes(
    manifest_id: str, seed: int, shards: tuple[ShardRecord, ...]
) -> bytes:
    return (
        json.dumps(
            {
                "manifest_id": manifest_id,
                "schema_version": 1,
                "seed": seed,
                "shards": [_record_mapping(item) for item in shards],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _selection_id(
    manifest_id: str, seed: int, shards: tuple[ShardRecord, ...]
) -> str:
    return hashlib.sha256(
        _selection_identity_bytes(manifest_id, seed, shards)
    ).hexdigest()


def canonical_validation_selection_bytes(selection: ValidationSelection) -> bytes:
    return (
        json.dumps(
            {
                "manifest_id": selection.manifest_id,
                "schema_version": 1,
                "seed": selection.seed,
                "selection_id": selection.selection_id,
                "shards": [_record_mapping(item) for item in selection.shards],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationSelectionError("validation JSON contains duplicate keys")
        result[key] = value
    return result


def parse_validation_selection(payload: bytes) -> ValidationSelection:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_SELECTION_BYTES:
        raise ValidationSelectionError("validation selection bytes are invalid")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
        if type(raw) is not dict:
            raise TypeError
        document = cast(dict[str, object], raw)
        if set(document) != {
            "manifest_id",
            "schema_version",
            "seed",
            "selection_id",
            "shards",
        } or document["schema_version"] != 1:
            raise ValueError
        raw_shards = document["shards"]
        if type(raw_shards) is not list:
            raise TypeError
        shards: list[ShardRecord] = []
        for raw_shard in cast(list[object], raw_shards):
            if type(raw_shard) is not dict:
                raise TypeError
            shard = cast(dict[str, object], raw_shard)
            if set(shard) != {"bytes", "path", "upstream_sha256"}:
                raise ValueError
            shards.append(
                ShardRecord(
                    path=cast(str, shard["path"]),
                    bytes=cast(int, shard["bytes"]),
                    upstream_sha256=cast(str, shard["upstream_sha256"]),
                )
            )
        selection = ValidationSelection(
            selection_id=cast(str, document["selection_id"]),
            manifest_id=cast(str, document["manifest_id"]),
            seed=cast(int, document["seed"]),
            shards=tuple(shards),
        )
    except ValidationSelectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ValidationSelectionError("validation selection is invalid") from None
    if canonical_validation_selection_bytes(selection) != payload:
        raise ValidationSelectionError("validation selection is not canonical")
    return selection


def select_validation_shards(
    manifest: DatasetManifest,
    *,
    seed: int = VALIDATION_SELECTION_SEED,
) -> ValidationSelection:
    if type(seed) is not int or seed != VALIDATION_SELECTION_SEED:
        raise ValidationSelectionError("validation selection seed must equal run seed 44")
    if len(manifest.shards) <= VALIDATION_SHARD_COUNT:
        raise ValidationSelectionError("validation selection would leave no training shard")

    try:
        selected = tuple(manifest.shard(path) for path in VALIDATION_SHARD_PATHS)
    except DatasetManifestError as error:
        raise ValidationSelectionError(
            f"fixed validation shard is absent from the operational manifest: {error}"
        ) from None
    return ValidationSelection(
        selection_id=_selection_id(manifest.manifest_id, seed, selected),
        manifest_id=manifest.manifest_id,
        seed=seed,
        shards=selected,
    )


def validate_selection_manifest(
    selection: ValidationSelection, manifest: DatasetManifest
) -> None:
    if selection.manifest_id != manifest.manifest_id:
        raise ValidationSelectionError("validation selection manifest_id differs")
    for selected in selection.shards:
        if manifest.shard(selected.path) != selected:
            raise ValidationSelectionError(
                "validation selection shard differs from the operational manifest"
            )
    expected = select_validation_shards(manifest, seed=selection.seed)
    if selection != expected:
        raise ValidationSelectionError(
            "validation selection differs from the stable seed-44 selection"
        )


def load_validation_selection(
    path: Path, manifest: DatasetManifest
) -> ValidationSelection:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        payload = path.read_bytes()
    except OSError:
        raise ValidationSelectionError("validation selection could not be read") from None
    selection = parse_validation_selection(payload)
    validate_selection_manifest(selection, manifest)
    return selection


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_validation_selection(
    selection: ValidationSelection, destination: Path
) -> None:
    payload = canonical_validation_selection_bytes(selection)
    temporary: Path | None = None
    published = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
            published = True
        except FileExistsError:
            raise ValidationSelectionExistsError(
                "validation selection destination already exists"
            ) from None
        temporary.unlink()
        _fsync_directory(destination.parent)
    except ValidationSelectionExistsError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if published:
            try:
                destination.unlink(missing_ok=True)
                _fsync_directory(destination.parent)
            except OSError:
                pass
        raise ValidationSelectionError(
            "validation selection could not be published"
        ) from None


def ensure_validation_selection(
    manifest: DatasetManifest,
    path: Path,
    *,
    seed: int = VALIDATION_SELECTION_SEED,
) -> ValidationSelection:
    if path.exists() or path.is_symlink():
        return load_validation_selection(path, manifest)
    selected = select_validation_shards(manifest, seed=seed)
    try:
        write_validation_selection(selected, path)
    except ValidationSelectionExistsError:
        return load_validation_selection(path, manifest)
    return selected


def prepare_validation_shards(
    transport: ModelScopeDatasetTransport,
    manifest: DatasetManifest,
    selection: ValidationSelection,
    root: Path,
) -> PreparedValidationShards:
    validate_selection_manifest(selection, manifest)
    if not root.is_absolute() or root.is_symlink():
        raise ValidationSelectionError("validation shard root is invalid")
    fetched: list[FetchedShard] = []
    for selected in selection.shards:
        item = fetch_dataset_shard(transport, manifest, selected.path, root)
        if (
            item.relative_path != selected.path
            or item.bytes != selected.bytes
            or item.sha256 != selected.upstream_sha256
        ):
            raise ValidationSelectionError("prepared validation shard identity differs")
        fetched.append(item)
    return PreparedValidationShards(
        selection=selection,
        root=root,
        paths=cast(tuple[Path, Path], tuple(item.path for item in fetched)),
    )


def _verified_shard_path(root: Path, record: ShardRecord) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise ValidationPromptError("validation shard root is invalid")
    path = root / record.path
    current = root
    for part in Path(record.path).parts:
        current /= part
        if current.is_symlink():
            raise ValidationPromptError("validation shard path contains a symlink")
    try:
        if not path.is_file() or path.stat().st_size != record.bytes:
            raise OSError
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        raise ValidationPromptError("validation shard could not be verified") from None
    if digest.hexdigest() != record.upstream_sha256:
        raise ValidationPromptError("validation shard digest differs from selection")
    return path


def _safe_member_name(name: str) -> tuple[str, str]:
    posix = PurePosixPath(name)
    if (
        not name
        or name != name.strip()
        or "\\" in name
        or posix.is_absolute()
        or posix.as_posix() != name
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "." not in posix.name
    ):
        raise ValidationPromptError("validation tar member path is invalid")
    base, suffix = name.rsplit(".", 1)
    if not base or not suffix:
        raise ValidationPromptError("validation tar member name is invalid")
    return base, suffix.casefold()


def _member_bytes(
    archive: tarfile.TarFile, member: tarfile.TarInfo, *, maximum: int | None = None
) -> bytes:
    if member.size <= 0 or (maximum is not None and member.size > maximum):
        raise ValidationPromptError("validation tar member size is invalid")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValidationPromptError("validation tar member could not be read")
    payload = handle.read(member.size + 1)
    if len(payload) != member.size:
        raise ValidationPromptError("validation tar member size differs")
    return payload


def _metadata_document(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationSelectionError):
        raise ValidationPromptError("validation metadata JSON is invalid") from None
    if type(value) is not dict:
        raise ValidationPromptError("validation metadata must be an object")
    return cast(dict[str, object], value)


def _mapping(document: dict[str, object], key: str) -> dict[str, object]:
    value = document.get(key)
    if type(value) is not dict:
        raise ValidationPromptError(f"validation metadata {key} must be an object")
    mapping = cast(dict[object, object], value)
    if any(type(item) is not str for item in mapping):
        raise ValidationPromptError(f"validation metadata {key} keys are invalid")
    return cast(dict[str, object], mapping)


def _prompt_text(
    document: dict[str, object], *, allow_empty: bool = False
) -> str | None:
    for group_name in ("captions", "multicaptions"):
        group = _mapping(document, group_name)
        for key in sorted(group):
            value = group[key]
            if value is None or value == "":
                continue
            if type(value) is not str:
                raise ValidationPromptError(
                    f"validation metadata {group_name}.{key} must be text or null"
                )
            prompt = value.strip()
            if prompt:
                if "<think>" in prompt or "</think>" in prompt:
                    raise ValidationPromptError("validation prompt contains thinking tags")
                return prompt
    if allow_empty:
        return None
    raise ValidationPromptError("validation metadata has no non-empty caption text")


def _prompt_dimensions(document: dict[str, object]) -> tuple[int, int]:
    image = _mapping(document, "image")
    width = image.get("width")
    height = image.get("height")
    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ValidationPromptError("validation metadata image dimensions are invalid")
    return height, width


def _prompt_seed(run_seed: int, identity: str) -> int:
    if type(run_seed) is not int or run_seed != VALIDATION_SELECTION_SEED:
        raise ValidationPromptError("validation prompt run seed must equal 44")
    digest = hashlib.sha256(f"{run_seed}\0{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def _load_shard_prompts(
    path: Path,
    record: ShardRecord,
    *,
    run_seed: int,
    caption_fields_parser: CaptionFieldsParser | None,
) -> tuple[ValidationPromptSample, ...]:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            samples = _load_archive_prompts(
                archive,
                record,
                run_seed=run_seed,
                caption_fields_parser=caption_fields_parser,
            )
    except (OSError, tarfile.TarError):
        raise ValidationPromptError("validation shard is not a readable tar") from None
    if not samples:
        raise ValidationPromptError("validation shard contains no image/JSON pairs")
    return tuple(samples)


def _load_archive_prompts(
    archive: tarfile.TarFile,
    record: ShardRecord,
    *,
    run_seed: int,
    caption_fields_parser: CaptionFieldsParser | None,
) -> list[ValidationPromptSample]:
    names: set[str] = set()
    grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in archive:
        if not member.isfile():
            raise ValidationPromptError("validation tar members must be regular files")
        if member.name in names:
            raise ValidationPromptError("validation tar contains duplicate members")
        names.add(member.name)
        base, suffix = _safe_member_name(member.name)
        suffixes = grouped.setdefault(base, {})
        if suffix in suffixes:
            raise ValidationPromptError(
                "validation tar contains duplicate member suffixes"
            )
        suffixes[suffix] = member

    samples: list[ValidationPromptSample] = []
    for member_key in sorted(grouped):
        members = grouped[member_key]
        image_members = tuple(
            members[suffix] for suffix in sorted(_IMAGE_SUFFIXES & members.keys())
        )
        metadata_member = members.get("json")
        if len(image_members) != 1 or metadata_member is None:
            raise ValidationPromptError(
                "validation sample is missing exactly one image or JSON member"
            )
        metadata = _metadata_document(
            _member_bytes(archive, metadata_member, maximum=_MAX_METADATA_BYTES)
        )
        height, width = _prompt_dimensions(metadata)
        image_payload = _member_bytes(archive, image_members[0])
        try:
            with Image.open(io.BytesIO(image_payload)) as image:
                image.load()
                decoded_width, decoded_height = image.size
        except (OSError, Image.DecompressionBombError):
            raise ValidationPromptError("validation image is invalid") from None
        if (decoded_height, decoded_width) != (height, width):
            raise ValidationPromptError(
                "validation metadata image dimensions differ from decoded image"
            )
        sample_id = metadata.get("id")
        if type(sample_id) is not int or sample_id <= 0:
            raise ValidationPromptError("validation metadata id is invalid")
        caption_fields: CaptionFields | None = None
        if caption_fields_parser is not None:
            try:
                parsed_fields = cast(object, caption_fields_parser(metadata))
            except (TypeError, ValueError):
                raise ValidationPromptError(
                    "validation metadata caption fields are invalid"
                ) from None
            if not isinstance(parsed_fields, CaptionFields):
                raise ValidationPromptError(
                    "validation caption parser returned an invalid value"
                )
            caption_fields = parsed_fields
        prompt = _prompt_text(metadata, allow_empty=caption_fields is not None)
        identity = f"{record.path}\0{member_key}\0{sample_id}"
        prompt_digest = hashlib.sha256(identity.encode()).hexdigest()
        samples.append(
            ValidationPromptSample(
                prompt_id=f"validation-{prompt_digest[:32]}",
                sample_id=sample_id,
                source_shard=record.path,
                member_key=member_key,
                prompt=prompt,
                seed=_prompt_seed(run_seed, identity),
                height=height,
                width=width,
                caption_fields=caption_fields,
            )
        )
    return samples


def load_validation_prompt_samples(
    selection: ValidationSelection,
    root: Path,
    *,
    run_seed: int,
    caption_fields_parser: CaptionFieldsParser | None = None,
) -> tuple[ValidationPromptSample, ...]:
    if caption_fields_parser is not None and not callable(caption_fields_parser):
        raise ValidationPromptError("validation caption parser is invalid")
    samples: list[ValidationPromptSample] = []
    for record in selection.shards:
        path = _verified_shard_path(root, record)
        samples.extend(
            _load_shard_prompts(
                path,
                record,
                run_seed=run_seed,
                caption_fields_parser=caption_fields_parser,
            )
        )
    identifiers = tuple(item.sample_id for item in samples)
    prompt_ids = tuple(item.prompt_id for item in samples)
    if len(set(identifiers)) != len(identifiers):
        raise ValidationPromptError("validation metadata IDs are not globally unique")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValidationPromptError("validation prompt identities are not unique")
    return tuple(
        sorted(
            samples,
            key=lambda item: (
                item.height,
                item.width,
                item.source_shard,
                item.member_key,
            ),
        )
    )


__all__ = [
    "VALIDATION_SELECTION_SEED",
    "VALIDATION_SHARD_COUNT",
    "VALIDATION_SHARD_PATHS",
    "CaptionFieldsParser",
    "PreparedValidationShards",
    "ValidationPromptError",
    "ValidationPromptSample",
    "ValidationSelection",
    "ValidationSelectionError",
    "ValidationSelectionExistsError",
    "canonical_validation_selection_bytes",
    "ensure_validation_selection",
    "load_validation_prompt_samples",
    "load_validation_selection",
    "parse_validation_selection",
    "prepare_validation_shards",
    "select_validation_shards",
    "validate_selection_manifest",
    "write_validation_selection",
]
