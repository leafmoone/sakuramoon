"""Deterministic validation selection and training exclusion."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from sakuramoon.data.metadata import MetadataRecord, scan_duplicate_ids

VALIDATION_SAMPLE_COUNT = 2000
AspectBucketResolver = Callable[[int, int], str]


class ValidationSelectionError(ValueError):
    """The metadata cannot produce the locked validation selection."""


class ValidationPublicationError(RuntimeError):
    """The independent validation bundle could not be published."""


class ValidationBundleExistsError(ValidationPublicationError):
    """The requested validation bundle path already exists."""


@dataclass(frozen=True, order=True)
class ValidationStratum:
    release: str
    aspect_bucket: str
    caption_available: bool


@dataclass(frozen=True)
class ValidationEntry:
    id: int
    stratum: ValidationStratum


@dataclass(frozen=True)
class ValidationSelection:
    entries: tuple[ValidationEntry, ...]
    seed: int

    def __post_init__(self) -> None:
        ids = tuple(entry.id for entry in self.entries)
        if type(self.seed) is not int:
            raise ValidationSelectionError("validation seed must be an integer")
        if len(ids) != VALIDATION_SAMPLE_COUNT or len(set(ids)) != len(ids):
            raise ValidationSelectionError(
                "validation selection is not exactly 2,000 unique IDs"
            )
        if ids != tuple(sorted(ids)) or any(
            type(sample_id) is not int or sample_id <= 0 for sample_id in ids
        ):
            raise ValidationSelectionError(
                "validation selection IDs must be sorted positive integers"
            )

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(entry.id for entry in self.entries)

    def stratum_counts(self) -> dict[ValidationStratum, int]:
        counts: dict[ValidationStratum, int] = {}
        for entry in self.entries:
            counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
        return counts


@dataclass(frozen=True)
class TrainingExclusionReport:
    input_records: int
    output_records: int
    excluded_records: int
    encountered_validation_ids: frozenset[int]


@dataclass(frozen=True)
class ValidationShardMember:
    suffix: str
    payload: bytes

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", self.suffix) is None:
            raise ValidationSelectionError("validation member suffix is invalid")
        if type(self.payload) is not bytes:
            raise ValidationSelectionError("validation member payload must be bytes")


@dataclass(frozen=True)
class ValidationShardSample:
    id: int
    members: tuple[ValidationShardMember, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not int or self.id <= 0:
            raise ValidationSelectionError("validation shard id must be positive")
        suffixes = tuple(member.suffix for member in self.members)
        if not suffixes:
            raise ValidationSelectionError("validation shard sample must have members")
        if suffixes != tuple(sorted(suffixes)) or len(suffixes) != len(set(suffixes)):
            raise ValidationSelectionError(
                "validation shard member suffixes must be sorted and unique"
            )


@dataclass(frozen=True)
class ValidationBundle:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    shard_path: Path
    shard_sha256: str
    shard_bytes: int


def _stable_rank(seed: int, stratum: ValidationStratum, sample_id: int) -> bytes:
    payload = (
        f"{seed}\0{stratum.release}\0{stratum.aspect_bucket}\0"
        f"{int(stratum.caption_available)}\0{sample_id}"
    ).encode()
    return hashlib.sha256(payload).digest()


def _validated_bucket_key(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValidationSelectionError("aspect bucket resolver returned an invalid key")
    return value


def _allocate_counts(
    sizes: dict[ValidationStratum, int], total: int
) -> dict[ValidationStratum, int]:
    """Give every stratum one sample, then allocate remaining slots by capacity."""

    strata = tuple(sorted(sizes))
    if len(strata) > total:
        raise ValidationSelectionError(
            "validation sample count is smaller than the number of strata"
        )
    allocation = {stratum: 1 for stratum in strata}
    remaining = total - len(strata)
    capacities = {stratum: sizes[stratum] - 1 for stratum in strata}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValidationSelectionError("not enough unique metadata IDs for validation")
    if remaining == 0:
        return allocation

    used = 0
    remainders: list[tuple[int, ValidationStratum]] = []
    for stratum in strata:
        numerator = remaining * capacities[stratum]
        extra, remainder = divmod(numerator, capacity_total)
        allocation[stratum] += extra
        used += extra
        remainders.append((remainder, stratum))
    for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : remaining - used
    ]:
        allocation[stratum] += 1
    return allocation


def select_validation_records(
    records: tuple[MetadataRecord, ...],
    *,
    seed: int,
    aspect_bucket: AspectBucketResolver,
) -> ValidationSelection:
    """Select exactly 2,000 unique IDs across the locked three-part strata."""

    duplicate_report = scan_duplicate_ids(records)
    if duplicate_report.has_duplicates:
        raise ValidationSelectionError("metadata contains duplicate logical IDs")
    if len(records) < VALIDATION_SAMPLE_COUNT:
        raise ValidationSelectionError("not enough unique metadata IDs for validation")

    grouped: dict[ValidationStratum, list[MetadataRecord]] = {}
    for record in records:
        bucket = _validated_bucket_key(aspect_bucket(record.width, record.height))
        stratum = ValidationStratum(
            release=record.release,
            aspect_bucket=bucket,
            caption_available=record.caption_available,
        )
        grouped.setdefault(stratum, []).append(record)

    allocation = _allocate_counts(
        {stratum: len(candidates) for stratum, candidates in grouped.items()},
        VALIDATION_SAMPLE_COUNT,
    )
    selected: list[ValidationEntry] = []
    for stratum in sorted(grouped):
        candidates = sorted(
            grouped[stratum],
            key=lambda record: (_stable_rank(seed, stratum, record.id), record.id),
        )
        selected.extend(
            ValidationEntry(record.id, stratum)
            for record in candidates[: allocation[stratum]]
        )
    selected.sort(key=lambda entry: entry.id)
    result = ValidationSelection(entries=tuple(selected), seed=seed)
    if len(result.entries) != VALIDATION_SAMPLE_COUNT or len(result.ids) != len(
        result.entries
    ):
        raise ValidationSelectionError("validation selection is not exactly 2,000 unique IDs")
    return result


def exclude_validation_records(
    records: Iterable[MetadataRecord], validation_ids: frozenset[int]
) -> tuple[tuple[MetadataRecord, ...], TrainingExclusionReport]:
    """Remove validation IDs before downstream shuffle or sample processing."""

    training: list[MetadataRecord] = []
    encountered: set[int] = set()
    total = 0
    for record in records:
        total += 1
        if record.id in validation_ids:
            encountered.add(record.id)
        else:
            training.append(record)
    report = TrainingExclusionReport(
        input_records=total,
        output_records=len(training),
        excluded_records=total - len(training),
        encountered_validation_ids=frozenset(encountered),
    )
    return tuple(training), report


def validation_manifest_bytes(selection: ValidationSelection) -> bytes:
    """Serialize the deterministic validation selection as canonical JSONL."""

    lines: list[str] = []
    for entry in selection.entries:
        payload = {
            "aspect_bucket": entry.stratum.aspect_bucket,
            "caption_available": entry.stratum.caption_available,
            "id": entry.id,
            "release": entry.stratum.release,
        }
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


class _HashingWriter:
    def __init__(self, handle: io.BufferedWriter) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        written = self._handle.write(payload)
        if written != len(payload):
            raise OSError("short validation shard write")
        self._digest.update(payload)
        self.bytes_written += written
        return written

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_validation_shard(
    path: Path,
    selection: ValidationSelection,
    samples: Iterable[ValidationShardSample],
) -> tuple[str, int]:
    iterator = iter(samples)
    with path.open("xb") as handle:
        writer = _HashingWriter(handle)
        with tarfile.open(
            fileobj=cast(BinaryIO, writer),
            mode="w|",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for entry in selection.entries:
                try:
                    sample = next(iterator)
                except StopIteration:
                    raise ValidationSelectionError(
                        "validation shard is missing selected IDs"
                    ) from None
                if sample.id != entry.id:
                    raise ValidationSelectionError(
                        "validation shard IDs must exactly match manifest order"
                    )
                for member in sample.members:
                    archive.addfile(
                        _tar_info(f"{sample.id}.{member.suffix}", len(member.payload)),
                        io.BytesIO(member.payload),
                    )
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValidationSelectionError(
                    "validation shard contains IDs outside the manifest"
                )
        handle.flush()
        os.fsync(handle.fileno())
    return writer.sha256, writer.bytes_written


def write_validation_bundle(
    selection: ValidationSelection,
    samples: Iterable[ValidationShardSample],
    destination: Path,
) -> ValidationBundle:
    """Publish canonical JSONL and one deterministic validation tar as a directory."""

    manifest_payload = validation_manifest_bytes(selection)
    manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
    temporary: Path | None = None
    manifest_path: Path | None = None
    shard_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ValidationBundleExistsError(
                "validation bundle destination already exists"
            )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
        )
        manifest_path = temporary / "validation_manifest.jsonl"
        with manifest_path.open("xb") as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        shard_path = temporary / "validation.tar"
        shard_digest, shard_bytes = _write_validation_shard(
            shard_path, selection, samples
        )
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists() or destination.is_symlink():
            raise ValidationBundleExistsError(
                "validation bundle destination already exists"
            )
        os.rename(temporary, destination)
        temporary = None
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except (ValidationSelectionError, ValidationPublicationError):
        raise
    except OSError:
        raise ValidationPublicationError(
            "validation bundle could not be published"
        ) from None
    finally:
        if temporary is not None:
            if shard_path is not None:
                shard_path.unlink(missing_ok=True)
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)
            try:
                temporary.rmdir()
            except OSError:
                pass

    published_manifest = destination / "validation_manifest.jsonl"
    published_shard = destination / "validation.tar"
    return ValidationBundle(
        root=destination,
        manifest_path=published_manifest,
        manifest_sha256=manifest_digest,
        shard_path=published_shard,
        shard_sha256=shard_digest,
        shard_bytes=shard_bytes,
    )
