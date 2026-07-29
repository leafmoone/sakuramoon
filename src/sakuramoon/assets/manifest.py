"""Strict schema and loader for repository-local asset declarations."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
BlockerId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
def _list_to_tuple[T](value: T) -> T:
    if type(value) is list:
        return cast(T, tuple(cast(list[object], value)))
    return value


class ManifestError(ValueError):
    """Raised when an asset manifest is missing or structurally invalid."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _is_safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


class SourceIdentity(StrictModel):
    repo_id: NonEmpty | None = None
    revision: Commit | None = None
    license_id: NonEmpty | None = None
    access_terms: NonEmpty | None = None


class DatabaseSourceIdentity(StrictModel):
    origin_kind: Literal["upstream_repo", "user_derived"]
    repo_id: NonEmpty | None = None
    revision: Commit | None = None
    derived_from: NonEmpty | None = None
    license_id: NonEmpty
    access_terms: NonEmpty

    @model_validator(mode="after")
    def validate_origin(self) -> DatabaseSourceIdentity:
        if self.origin_kind == "upstream_repo":
            if self.repo_id is None or self.revision is None or self.derived_from is not None:
                raise ValueError("upstream database source requires repo/revision only")
        elif self.derived_from is None or self.repo_id is not None or self.revision is not None:
            raise ValueError("user-derived database source requires derived_from only")
        return self


class FileLock(StrictModel):
    path: NonEmpty
    kind: Literal["config", "tokenizer", "weights", "license", "database"]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_path(self) -> FileLock:
        if not _is_safe_relative(self.path):
            raise ValueError("file path must be a normalized repository-relative POSIX path")
        return self


class QwenSummary(StrictModel):
    config_sha256: Sha256
    tokenizer_sha256: Sha256
    layers: Literal[24]
    hidden_size: Literal[2048]
    dtype: Literal["bfloat16"]
    frozen: Literal[True]
    use_cache: Literal[False]
    visual_path_enabled: Literal[False]


class VaeSummary(StrictModel):
    config_sha256: Sha256
    latent_channels: Literal[128]
    downsample_factor: Literal[16]
    sample_posterior: Literal[False]
    posterior_mean_required: Literal[True]
    dtype: Literal["bfloat16"]
    frozen: Literal[True]


class ModelAssetBase(StrictModel):
    asset_id: NonEmpty
    local_path: NonEmpty
    lock_state: Literal["ready", "blocked"]
    blockers: Annotated[tuple[BlockerId, ...], BeforeValidator(_list_to_tuple)]
    source: SourceIdentity
    files: Annotated[tuple[FileLock, ...], BeforeValidator(_list_to_tuple)]

    @model_validator(mode="after")
    def validate_lock(self) -> ModelAssetBase:
        if not _is_safe_relative(self.local_path):
            raise ValueError("local_path must be a normalized repository-relative POSIX path")
        if not self.files:
            raise ValueError("model asset must declare at least one file")
        if len({item.path for item in self.files}) != len(self.files):
            raise ValueError("model asset file paths must be unique")
        complete_source = all(
            value is not None
            for value in (
                self.source.repo_id,
                self.source.revision,
                self.source.license_id,
                self.source.access_terms,
            )
        )
        complete_files = all(item.sha256 is not None for item in self.files)
        if self.lock_state == "ready":
            if self.blockers or not complete_source or not complete_files:
                raise ValueError("ready model asset requires complete source and file locks")
        elif not self.blockers:
            raise ValueError("blocked model asset requires at least one blocker")
        return self


class QwenAsset(ModelAssetBase):
    kind: Literal["qwen"]
    summary: QwenSummary

    @model_validator(mode="after")
    def validate_qwen_source(self) -> QwenAsset:
        expected = "spawner/Qwen3_5_2b_claude_heretic_spawner"
        if self.source.repo_id != expected:
            raise ValueError(f"Qwen source must be {expected}")
        configs = [item for item in self.files if item.kind == "config"]
        tokenizers = [item for item in self.files if item.kind == "tokenizer"]
        weights = [item for item in self.files if item.kind == "weights"]
        licenses = [item for item in self.files if item.kind == "license"]
        if len(configs) != 1 or len(tokenizers) != 1 or not weights or len(licenses) != 1:
            raise ValueError("Qwen requires one config, one tokenizer, weights, and one license file")
        if configs[0].sha256 != self.summary.config_sha256:
            raise ValueError("Qwen config summary hash must match its file lock")
        if tokenizers[0].sha256 != self.summary.tokenizer_sha256:
            raise ValueError("Qwen tokenizer summary hash must match its file lock")
        return self


class VaeAsset(ModelAssetBase):
    kind: Literal["vae"]
    summary: VaeSummary

    @model_validator(mode="after")
    def validate_official_source(self) -> VaeAsset:
        configs = [item for item in self.files if item.kind == "config"]
        weights = [item for item in self.files if item.kind == "weights"]
        if len(configs) != 1 or not weights:
            raise ValueError("Mage-VAE requires one config and at least one weights file")
        if configs[0].sha256 != self.summary.config_sha256:
            raise ValueError("Mage-VAE config summary hash must match its file lock")
        if self.lock_state == "ready":
            repo_id = self.source.repo_id
            if repo_id is None or not repo_id.casefold().startswith("microsoft/"):
                raise ValueError("ready Mage-VAE must use a Microsoft repository")
        return self


ModelAsset = Annotated[QwenAsset | VaeAsset, Field(discriminator="kind")]


class DatabaseAsset(StrictModel):
    asset_id: NonEmpty
    local_path: NonEmpty
    lock_state: Literal["ready", "blocked"]
    blockers: Annotated[tuple[BlockerId, ...], BeforeValidator(_list_to_tuple)]
    required_for_runtime: bool
    source: DatabaseSourceIdentity
    schema_version: NonEmpty | None = None
    files: Annotated[tuple[FileLock, ...], BeforeValidator(_list_to_tuple)]
    allowed_aggregate_statistics: Annotated[
        tuple[NonEmpty, ...], BeforeValidator(_list_to_tuple)
    ]

    @model_validator(mode="after")
    def validate_lock(self) -> DatabaseAsset:
        if not _is_safe_relative(self.local_path):
            raise ValueError("local_path must be a normalized repository-relative POSIX path")
        if not self.files or any(item.kind != "database" for item in self.files):
            raise ValueError("database asset files must use kind=database")
        if len({item.path for item in self.files}) != len(self.files):
            raise ValueError("database asset file paths must be unique")
        complete_files = all(item.sha256 is not None for item in self.files)
        if self.lock_state == "ready":
            if self.blockers or not complete_files or self.schema_version is None:
                raise ValueError("ready database requires complete source, schema, and file locks")
        elif not self.blockers:
            raise ValueError("blocked database requires at least one blocker")
        return self


class LicenseLock(StrictModel):
    name: NonEmpty
    scope: NonEmpty
    path: NonEmpty
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> LicenseLock:
        if not _is_safe_relative(self.path):
            raise ValueError("license path must be repository-relative")
        return self


class ReferenceAsset(StrictModel):
    asset_id: NonEmpty
    local_path: NonEmpty
    origin_url: NonEmpty
    commit: Commit
    licenses: Annotated[tuple[LicenseLock, ...], BeforeValidator(_list_to_tuple)]
    tracked_worktree_required_clean: Literal[True]

    @model_validator(mode="after")
    def validate_reference(self) -> ReferenceAsset:
        if not _is_safe_relative(self.local_path):
            raise ValueError("local_path must be a normalized repository-relative POSIX path")
        if not self.origin_url.startswith("https://") or any(
            marker in self.origin_url for marker in ("?", "#", "@")
        ):
            raise ValueError("origin_url must be a credential-free immutable HTTPS remote")
        if not self.licenses:
            raise ValueError("reference repository must declare its licenses")
        return self


class AssetManifest(StrictModel):
    schema_version: Literal[1]
    manifest_revision: Annotated[int, Field(ge=1)]
    models: Annotated[tuple[ModelAsset, ...], BeforeValidator(_list_to_tuple)]
    databases: Annotated[tuple[DatabaseAsset, ...], BeforeValidator(_list_to_tuple)]
    references: Annotated[tuple[ReferenceAsset, ...], BeforeValidator(_list_to_tuple)]

    @model_validator(mode="after")
    def validate_inventory(self) -> AssetManifest:
        all_ids = [
            *(asset.asset_id for asset in self.models),
            *(asset.asset_id for asset in self.databases),
            *(asset.asset_id for asset in self.references),
        ]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("asset IDs must be globally unique")
        if {asset.kind for asset in self.models} != {"qwen", "vae"}:
            raise ValueError("manifest must declare exactly the Qwen and Mage-VAE model kinds")
        if len(self.models) != 2:
            raise ValueError("manifest must contain exactly two model assets")
        expected_references = {"reference_hdm", "reference_jlt", "reference_krea2"}
        if {asset.asset_id for asset in self.references} != expected_references:
            raise ValueError("manifest must lock HDM, JLT, and krea-2 references")
        if not self.databases:
            raise ValueError("manifest must declare at least one local database asset")
        return self


def load_manifest(path: Path) -> AssetManifest:
    """Read and strictly validate a TOML manifest without touching its assets."""

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        return AssetManifest.model_validate(payload, strict=True)
    except OSError as exc:
        raise ManifestError(f"cannot read asset manifest: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError("invalid asset manifest TOML") from exc
    except ValidationError as exc:
        locations = sorted(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        summary = ", ".join(locations[:8])
        raise ManifestError(f"invalid asset manifest fields: {summary}") from exc


def is_sha256(value: str) -> bool:
    """Return whether a string is a lowercase SHA-256 digest."""

    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
